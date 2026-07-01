"""
train_db.py

Training pipeline for the CNN-based RFI knee-index classifier using dB-normalized eigenvalues.

This script is identical to train.py except that eigenvalues are normalized in dB scale
rather than linear scale. The hypothesis is that dB normalization will make the "knee"
feature more prominent by expanding the dynamic range in the noise floor region.

Label convention
----------------
Keras requires non-negative class indices, so the 17 classes are encoded as:

    label 0       -> no RFI (clean sentinel)
    label 1..16   -> knee at position 1..16 (1-based indexing, matches plot visualization)

The knee label equals the number of distinct pulse rows occupied by RFI in
this block, because each distinct row contributes rank-1 to the SCM and
therefore produces one elevated eigenvalue in the descending spectrum.

In 1-based indexing (matching eigenvalue profile plots):
    n_distinct_rows == 1 -> label 1  (knee at position 1, largest eigenvalue)
    n_distinct_rows == 2 -> label 2  (knee at position 2, 2nd largest eigenvalue)
    ...
    n_distinct_rows == 6 -> label 6  (knee at position 6, 6th largest eigenvalue)
    clean (no RFI)       -> label 0

In 0-based array indexing (for accessing eigvals_normalized[idx]):
    label 1 -> eigvals_normalized[0]  (knee after 1st eigenvalue)
    label 2 -> eigvals_normalized[1]  (knee after 2nd eigenvalue)
    ...
    label k -> eigvals_normalized[k-1] (knee after k-th eigenvalue)

Distinct rows are determined from the 'pulse_positions' fields stored in the
'rfi_bands' JSON attribute on each HDF5 dataset. Two bands sharing the
same pulse position still occupy one row and produce one elevated eigenvalue.

Model output has 17 classes (labels 0..16).

Feature extraction (per CPI block)
-----------------------------------
eigen_input  shape (M, 2):
    channel 0 -- eigenvalues in dB scale, normalized to [0, 1]
                 Conversion: eigvals_db = 10 * log10(eigvals + eps)
                 Then normalized by: (eigvals_db - min_db) / (max_db - min_db)
    channel 1 -- finite differences of dB-normalized eigenvalues (slopes), length M,
                 zero-padded at index M-1 so the tensor stays (M, 2)

global_input shape (5,):
    0  condition_number -- lambda_max / lambda_min (ratio, scale-invariant)
    1  sigma_min        -- std of bottom-half SCM diagonal (in dB, normalized)
    2  sigma_max        -- std of top-half SCM diagonal (in dB, normalized)
    3  mu_min           -- mean of bottom-half SCM diagonal (in dB, normalized)
    4  f_factor         -- sigma_max / (sigma_min + eps) (ratio, scale-invariant)

Directory layout expected on disk
-----------------------------------
    data/
        multi_band/
            clean/          image_*.h5 (clean samples, knee=0)
            contaminated/   image_*.h5 (RFI samples, knee=1-16)

Outputs
-------
Model saved to:
    models/multi_band_db/best_model.keras

Evaluation PNGs saved to models/multi_band_db/:
    training_curves.png   -- loss and accuracy vs epoch
    confusion_matrix.png  -- knee confusion matrix (normalised by row)
    metrics.png           -- bar chart of scalar evaluation metrics

JSON metrics saved to:
    models/multi_band_db/eval_results.json
    models/summary_db.json
"""

import os
import sys
import json
import numpy as np
import h5py
import tensorflow as tf
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_synthetic_data import BLOCK_HEIGHT
from model import build_model

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

DATA_ROOT   = 'data'
MODELS_ROOT = 'models'
M           = BLOCK_HEIGHT      # pulses per CPI = number of eigenvalues
N_CLASSES   = M + 1             # labels 0..M (0=clean, 1-16=RFI knee position)
N_GLOBAL    = 5                 # global features: cond_number, sigma_min, sigma_max, mu_min, f_factor

EPOCHS      = 50
BATCH_SIZE  = 64
LR          = 3e-4
VAL_FRAC    = 0.1
TEST_FRAC   = 0.1

# dB conversion parameters
EPS_DB      = 1e-12  # floor for log10 to avoid -inf
DB_FLOOR    = -80    # clip dB values below this threshold


# ---------------------------------------------------------------------------
# FEATURE EXTRACTION (dB-normalized)
# ---------------------------------------------------------------------------

def _scm_from_cpi(cpi):
    """
    Compute sample covariance matrix from a complex CPI tile.

    Args:
        cpi (np.ndarray): Complex array shape (M, K).

    Returns:
        R                  (np.ndarray): Complex (M, M) SCM = CPI @ CPI^H / K.
        eigvals_db_normalized (np.ndarray): Eigenvalues in dB scale, normalized to [0,1], shape (M,).
    """
    M, K = cpi.shape
    R = (cpi @ cpi.conj().T) / K
    eigvals = np.linalg.eigvalsh(R)             # ascending
    eigvals = np.sort(np.real(eigvals))[::-1]   # descending

    # Convert to dB scale
    eigvals_db = 10 * np.log10(eigvals + EPS_DB)
    eigvals_db = np.maximum(eigvals_db, DB_FLOOR)  # floor at -80 dB

    # Normalize dB values to [0, 1]
    min_db = np.min(eigvals_db)
    max_db = np.max(eigvals_db)
    if max_db > min_db:
        eigvals_db_normalized = (eigvals_db - min_db) / (max_db - min_db)
    else:
        eigvals_db_normalized = np.zeros_like(eigvals_db)

    return R, eigvals_db_normalized


def extract_features(cpi):
    """
    Extract eigen_input and global_input feature vectors from one CPI tile.
    Uses dB-normalized eigenvalues for enhanced dynamic range in the noise floor.

    Args:
        cpi (np.ndarray): Complex array shape (M, K).

    Returns:
        eigen   (np.ndarray): shape (M, 2)  -- [eigvals_db_normalized, slopes_db_normalized]
        global_ (np.ndarray): shape (5,)    -- scalar context features
    """
    M, K = cpi.shape
    R, eigvals_db_normalized = _scm_from_cpi(cpi)

    # --- Eigenvalue branch features ----------------------------------------
    # Compute slopes from dB-normalized eigenvalues
    slopes        = np.diff(eigvals_db_normalized)     # length M-1
    slopes_padded = np.append(slopes, 0.0)             # length M, zero at end
    eigen = np.stack([eigvals_db_normalized, slopes_padded], axis=-1).astype(np.float32)

    # --- Global branch features --------------------------------------------
    # For global features, we use dB-normalized diagonal statistics
    # First get the diagonal in linear scale
    diag = np.real(np.diag(R))

    # Convert to dB and normalize
    diag_db = 10 * np.log10(diag + EPS_DB)
    diag_db = np.maximum(diag_db, DB_FLOOR)
    min_diag_db = np.min(diag_db)
    max_diag_db = np.max(diag_db)
    if max_diag_db > min_diag_db:
        diag_db_normalized = (diag_db - min_diag_db) / (max_diag_db - min_diag_db)
    else:
        diag_db_normalized = np.zeros_like(diag_db)

    half     = max(M // 2, 1)
    sigma_max = float(np.std(diag_db_normalized[:half]))
    sigma_min = float(np.std(diag_db_normalized[half:]))
    mu_min    = float(np.mean(diag_db_normalized[half:]))

    # Condition number (ratio, scale-invariant) - still use normalized eigenvalues
    cond_number = eigvals_db_normalized[0] / max(eigvals_db_normalized[-1], 1e-12)

    # F-factor (ratio, scale-invariant)
    eps       = 1e-6
    f_factor  = sigma_max / (sigma_min + eps)

    global_ = np.array(
        [cond_number, sigma_min, sigma_max, mu_min, f_factor],
        dtype=np.float32,
    )
    return eigen, global_


# ---------------------------------------------------------------------------
# LABEL DERIVATION
# ---------------------------------------------------------------------------

def label_from_rfi_bands(rfi_bands_json):
    """
    Derive the knee label from the 'rfi_bands' JSON attribute of an HDF5 tile.

    The label equals the number of distinct pulse positions occupied by RFI.
    This directly corresponds to the knee position in 1-based indexing used in plots.

    Examples (1-based indexing):
        - 0 distinct RFI pulses → label 0 (clean)
        - 1 distinct RFI pulse  → label 1 (knee at position 1)
        - 2 distinct RFI pulses → label 2 (knee at position 2)
        - etc.

    Args:
        rfi_bands_json (str): JSON string with keys:
            pulse_positions (list[int]): 1-based pulse positions of RFI bands
            knee (int): number of RFI bands (may include duplicates)
            jnr_db_list (list[int]): JNR in dB for each band

    Returns:
        label (int): Number of distinct RFI pulse positions.
                     0 for clean, [1, MAX_BANDS] for RFI.
                     Matches 1-based knee position in eigenvalue plots.
    """
    payload = json.loads(rfi_bands_json)

    # Count distinct pulse positions
    # Multiple RFI bands on the same pulse row count as 1 elevated eigenvalue
    if payload['knee'] == 0:
        return 0  # No RFI
    else:
        pulse_positions = payload['pulse_positions']
        n_distinct = len(set(pulse_positions))
        return n_distinct


# ---------------------------------------------------------------------------
# DATASET LOADING
# ---------------------------------------------------------------------------

def load_dataset_from_folder(folder, expected_is_clean=None):
    """
    Load all CPI samples from all HDF5 files in a folder.

    The knee label is derived per-tile from the 'rfi_bands' attribute
    (see label_from_rfi_bands).

    Args:
        folder (str): Path to directory containing image_*.h5 files.
        expected_is_clean (bool|None): If provided, verify that loaded files match this flag.

    Returns:
        eigen   (np.ndarray): shape (N, M, 2)
        global_ (np.ndarray): shape (N, 5)
        labels  (np.ndarray): shape (N,)  int32 knee indices
    """
    eigen_list, global_list, label_list = [], [], []

    h5_files = sorted(f for f in os.listdir(folder) if f.endswith('.h5'))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {folder}")

    for fname in h5_files:
        fpath = os.path.join(folder, fname)
        with h5py.File(fpath, 'r') as f:
            # Verify is_clean flag if expected
            if expected_is_clean is not None:
                file_is_clean = f.attrs.get('is_clean', False)
                if file_is_clean != expected_is_clean:
                    print(f"Warning: {fname} has is_clean={file_is_clean}, expected {expected_is_clean}")

            for key in f.keys():
                # Skip eigenvalue and diagonal datasets, only load CPI data
                if key.endswith('_eigenvalues') or key.endswith('_diagonal'):
                    continue
                cpi           = f[key][:]
                eigen, glob   = extract_features(cpi)
                label         = label_from_rfi_bands(str(f[key].attrs['rfi_bands']))
                eigen_list.append(eigen)
                global_list.append(glob)
                label_list.append(label)

    return (
        np.stack(eigen_list).astype(np.float32),
        np.stack(global_list).astype(np.float32),
        np.array(label_list, dtype=np.int32),
    )




def split_dataset(eigen, global_, labels):
    """
    Split into train / val / test sets (80 / 10 / 10), stratified on labels.

    Args:
        eigen   (np.ndarray): shape (N, M, 2)
        global_ (np.ndarray): shape (N, 6)
        labels  (np.ndarray): shape (N,)

    Returns:
        e_train, e_val, e_test,
        g_train, g_val, g_test,
        y_train, y_val, y_test
    """
    e_tv, e_test, g_tv, g_test, y_tv, y_test = train_test_split(
        eigen, global_, labels,
        test_size=TEST_FRAC, random_state=42, stratify=labels,
    )
    val_frac_of_tv = VAL_FRAC / (1.0 - TEST_FRAC)
    e_train, e_val, g_train, g_val, y_train, y_val = train_test_split(
        e_tv, g_tv, y_tv,
        test_size=val_frac_of_tv, random_state=42, stratify=y_tv,
    )
    return e_train, e_val, e_test, g_train, g_val, g_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# PLOT HELPERS
# ---------------------------------------------------------------------------

def save_training_curves_png(history, out_dir):
    """
    Save loss and accuracy training curves to out_dir/training_curves.png.

    Two-panel figure: left panel shows train/val loss; right panel shows
    train/val sparse categorical accuracy.

    Args:
        history  : Keras History object returned by model.fit.
        out_dir  (str): Directory where the PNG is written.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = range(1, len(history.history['loss']) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(epochs, history.history['loss'],     label='Train loss')
    ax_loss.plot(epochs, history.history['val_loss'], label='Val loss')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Training & Validation Loss')
    ax_loss.legend()
    ax_loss.grid(True, linestyle='--', alpha=0.5)

    acc_key     = 'acc'     if 'acc'     in history.history else 'sparse_categorical_accuracy'
    val_acc_key = 'val_acc' if 'val_acc' in history.history else 'val_sparse_categorical_accuracy'
    ax_acc.plot(epochs, history.history[acc_key],     label='Train acc')
    ax_acc.plot(epochs, history.history[val_acc_key], label='Val acc')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_title('Training & Validation Accuracy')
    ax_acc.legend()
    ax_acc.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    path = os.path.join(out_dir, 'training_curves.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def save_confusion_matrix_png(y_true, y_pred, class_labels, out_dir):
    """
    Save a normalised knee confusion matrix heatmap to out_dir/confusion_matrix.png.

    Only classes present in y_true or y_pred are shown.  Each row is
    normalised by its true-class count so cell values are recall fractions.
    The raw count is annotated inside each cell.

    Args:
        y_true       (np.ndarray): Ground-truth integer labels, shape (N,).
        y_pred       (np.ndarray): Predicted integer labels, shape (N,).
        class_labels (list[str]): Name for each class index (index 0 = clean).
        out_dir      (str): Directory where the PNG is written.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    n       = len(present)
    idx_map = {cls: i for i, cls in enumerate(present)}
    labels  = [class_labels[c] for c in present]

    matrix  = np.zeros((n, n), dtype=np.int32)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        matrix[idx_map[t], idx_map[p]] += 1

    row_sums = matrix.sum(axis=1, keepdims=True).clip(min=1)
    normed   = matrix / row_sums

    fig, ax = plt.subplots(figsize=(max(6, n * 0.7), max(5, n * 0.6)))
    im = ax.imshow(normed, vmin=0.0, vmax=1.0, cmap='Blues')
    fig.colorbar(im, ax=ax, label='Recall (row-normalised)')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Knee Confusion Matrix (dB-normalized)')

    thresh = 0.5
    for ri in range(n):
        for ci in range(n):
            color = 'white' if normed[ri, ci] > thresh else 'black'
            ax.text(ci, ri, str(matrix[ri, ci]),
                    ha='center', va='center', fontsize=7, color=color)

    fig.tight_layout()
    path = os.path.join(out_dir, 'confusion_matrix.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def save_metrics_png(results, out_dir):
    """
    Save a horizontal bar chart of scalar evaluation metrics to out_dir/metrics.png.

    Args:
        results (dict): Output of evaluate() containing accuracy metrics.
        out_dir (str):  Directory where the PNG is written.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    metric_names = [
        'Exact accuracy',
        'Tol-1 accuracy',
        'RFI exact',
        'RFI tol-1',
        'No-RFI exact',
    ]
    metric_keys = [
        'exact_acc', 'tol1_acc', 'exact_rfi', 'tol1_rfi', 'exact_no_rfi',
    ]
    values = [results[k] for k in metric_keys]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(metric_names, values, color='steelblue')
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Value')
    ax.set_title(f"Evaluation Metrics (dB-normalized)  --  {results['run']}\n"
                 f"N test = {results['n_test']}")
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    for bar, val in zip(bars, values):
        ax.text(min(val + 0.01, 1.0), bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=9)

    fig.tight_layout()
    path = os.path.join(out_dir, 'metrics.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def evaluate(model, eigen_test, global_test, y_test, run_name, out_dir):
    """
    Evaluate model on a held-out test set and save result PNGs + JSON.

    Saves:
        out_dir/confusion_matrix.png  -- row-normalised heatmap
        out_dir/metrics.png           -- bar chart of scalar metrics
        out_dir/eval_results.json     -- all metrics as JSON

    Args:
        model       : Trained Keras model.
        eigen_test  (np.ndarray): shape (N, M, 2)
        global_test (np.ndarray): shape (N, 5)
        y_test      (np.ndarray): shape (N,) integer labels
        run_name    (str): Label used in figure titles.
        out_dir     (str): Directory where outputs are written.

    Returns:
        results (dict): Computed metrics.
    """
    probs  = model.predict([eigen_test, global_test], verbose=0)
    y_pred = np.argmax(probs, axis=-1)

    exact  = float(np.mean(y_pred == y_test))
    tol1   = float(np.mean(np.abs(y_pred.astype(int) - y_test.astype(int)) <= 1))

    rfi_mask   = y_test > 0
    norfi_mask = y_test == 0

    exact_rfi   = float(np.mean(y_pred[rfi_mask]   == y_test[rfi_mask]))   \
                  if rfi_mask.any()   else float('nan')
    tol1_rfi    = float(np.mean(
                      np.abs(y_pred[rfi_mask].astype(int)
                             - y_test[rfi_mask].astype(int)) <= 1
                  )) if rfi_mask.any() else float('nan')
    exact_norfi = float(np.mean(y_pred[norfi_mask] == y_test[norfi_mask])) \
                  if norfi_mask.any() else float('nan')

    results = {
        'run'          : run_name,
        'n_test'       : int(len(y_test)),
        'exact_acc'    : exact,
        'tol1_acc'     : tol1,
        'exact_rfi'    : exact_rfi,
        'tol1_rfi'     : tol1_rfi,
        'exact_no_rfi' : exact_norfi,
    }

    print(f"\n=== {run_name} ===")
    print(f"  N test  : {results['n_test']}")
    print(f"  Exact   : {exact:.4f}")
    print(f"  Tol-1   : {tol1:.4f}")
    print(f"  RFI exact / tol-1 : {exact_rfi:.4f} / {tol1_rfi:.4f}")
    print(f"  No-RFI exact      : {exact_norfi:.4f}")
    print("  Saving evaluation plots ...")

    # class_labels: label 0 = clean, label k = knee at position k (1-based)
    # Example: label 1 = "knee@1", label 2 = "knee@2", etc.
    class_labels = ["clean"] + [f"knee@{k}" for k in range(1, M + 1)]
    save_confusion_matrix_png(y_test, y_pred, class_labels, out_dir)
    save_metrics_png(results, out_dir)

    with open(os.path.join(out_dir, 'eval_results.json'), 'w') as fh:
        json.dump(results, fh, indent=2)

    return results


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def train_one_run(run_name, eigen_train, eigen_val, eigen_test,
                  global_train, global_val, global_test,
                  y_train, y_val, y_test):
    """
    Build, train, evaluate and save a model for one dataset configuration.

    Best validation-loss checkpoint saved to:
        models/<run_name>/best_model.keras

    Args:
        run_name   (str): Unique name for this run (used as folder name).
        eigen_*    (np.ndarray): shape (N, M, 2)
        global_*   (np.ndarray): shape (N, 5)
        y_*        (np.ndarray): shape (N,) integer labels

    Returns:
        results (dict): Evaluation metrics on the test set.
    """
    out_dir    = os.path.join(MODELS_ROOT, run_name)
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'best_model.keras')

    print(f"\n{'='*60}")
    print(f"Run : {run_name}")
    print(f"  train={len(y_train)}  val={len(y_val)}  test={len(y_test)}")
    print(f"  classes present: {np.unique(y_train).tolist()}")
    print(f"{'='*60}")

    model = build_model(
        cpi_size          = M,
        n_global_features = N_GLOBAL,
        n_knee_classes    = N_CLASSES,
        dropout_rate      = 0.5,
        learning_rate     = LR,
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath             = model_path,
            monitor              = 'val_loss',
            save_best_only       = True,
            verbose              = 1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor              = 'val_loss',
            patience             = 10,
            restore_best_weights = True,
            verbose              = 1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor              = 'val_loss',
            factor               = 0.5,
            patience             = 5,
            min_lr               = 1e-6,
            verbose              = 1,
        ),
    ]

    model.fit(
        x               = [eigen_train, global_train],
        y               = y_train,
        validation_data = ([eigen_val, global_val], y_val),
        epochs          = EPOCHS,
        batch_size      = BATCH_SIZE,
        callbacks       = callbacks,
        verbose         = 2,
    )

    save_training_curves_png(model.history, out_dir)
    results = evaluate(model, eigen_test, global_test, y_test, run_name, out_dir)

    with open(os.path.join(out_dir, 'eval_results.json'), 'w') as fh:
        json.dump(results, fh, indent=2)

    return results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    """
    Full training pipeline using dB-normalized eigenvalues.

    Step 1: Load clean samples from data/multi_band/clean/ (label = 0).
    Step 2: Load contaminated samples from data/multi_band/contaminated/ (label = 1-16).
    Step 3: Merge, split 80/10/10, train, and evaluate a single combined model.
    Step 4: Print summary metrics and knee confusion matrix.
    """
    os.makedirs(MODELS_ROOT, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load clean dataset
    # ------------------------------------------------------------------
    clean_folder = os.path.join(DATA_ROOT, 'multi_band', 'clean')
    print(f"\nLoading CLEAN dataset from {clean_folder} ...")
    e_clean, g_clean, y_clean = load_dataset_from_folder(clean_folder, expected_is_clean=True)
    print(f"  Clean samples: {len(y_clean)}")
    unique, counts = np.unique(y_clean, return_counts=True)
    for u, c in zip(unique.tolist(), counts.tolist()):
        print(f"    label {u} (clean): {c}")

    # ------------------------------------------------------------------
    # Step 2: Load contaminated (RFI) dataset
    # ------------------------------------------------------------------
    rfi_folder = os.path.join(DATA_ROOT, 'multi_band', 'contaminated')
    print(f"\nLoading CONTAMINATED dataset from {rfi_folder} ...")
    e_rfi, g_rfi, y_rfi = load_dataset_from_folder(rfi_folder, expected_is_clean=False)
    print(f"  RFI samples  : {len(y_rfi)}")
    unique, counts = np.unique(y_rfi, return_counts=True)
    for u, c in zip(unique.tolist(), counts.tolist()):
        if u == 0:
            print(f"    label {u} (clean - unexpected in contaminated folder): {c}")
        else:
            print(f"    label {u} (knee at position {u}): {c}")

    # ------------------------------------------------------------------
    # Step 3: Merge + split + train
    # ------------------------------------------------------------------
    eigen  = np.concatenate([e_clean, e_rfi],  axis=0)
    global_ = np.concatenate([g_clean, g_rfi],  axis=0)
    labels  = np.concatenate([y_clean, y_rfi],  axis=0)
    print(f"\nTotal samples: {len(labels)}")
    print(f"  Clean (label 0): {np.sum(labels == 0)}")
    print(f"  RFI (label 1-{M}): {np.sum(labels > 0)}")

    (e_tr, e_va, e_te,
     g_tr, g_va, g_te,
     y_tr, y_va, y_te) = split_dataset(eigen, global_, labels)

    results = train_one_run(
        'multi_band_db',
        e_tr, e_va, e_te,
        g_tr, g_va, g_te,
        y_tr, y_va, y_te,
    )

    # ------------------------------------------------------------------
    # Step 4: Summary
    # ------------------------------------------------------------------
    summary_path = os.path.join(MODELS_ROOT, 'summary_db.json')
    with open(summary_path, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSummary saved to {summary_path}")
    print(f"Training plots : models/multi_band_db/training_curves.png")
    print(f"Confusion matrix: models/multi_band_db/confusion_matrix.png")
    print(f"Metrics bar chart: models/multi_band_db/metrics.png")


if __name__ == '__main__':
    main()
