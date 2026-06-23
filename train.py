"""
train.py

Training pipeline for the CNN-based RFI knee-index classifier (model.py).

Label convention
----------------
Keras requires non-negative class indices, so the 17 classes are encoded as:

    label 0       -> no RFI (clean sentinel)
    label 1..16   -> knee at eigenvalue index 0..15

The knee index equals the number of distinct pulse rows occupied by RFI in
this block, because each distinct row contributes rank-1 to the SCM and
therefore produces one elevated eigenvalue in the descending spectrum:

    n_distinct_rows == 1 -> label 1  (knee at index 0)
    n_distinct_rows == 2 -> label 2  (knee at index 1)
    ...
    n_distinct_rows == 6 -> label 6  (knee at index 5)
    clean                -> label 0

Distinct rows are determined from the 'local_idx' fields stored in the
'rfi_bands' JSON attribute on each HDF5 dataset.  Two bands sharing the
same local_idx still occupy one row and produce one elevated eigenvalue.

Model output has 17 classes (labels 0..16).

Feature extraction (per CPI block)
-----------------------------------
eigen_input  shape (M, 2):
    channel 0 -- eigenvalues in dB, descending, from SCM = CPI @ CPI^H / K
    channel 1 -- finite differences of eigenvalues (slopes), length M,
                 zero-padded at index M-1 so the tensor stays (M, 2)

global_input shape (6,):
    0  trace_db         -- 10*log10(trace(R))
    1  condition_number -- lambda_max / lambda_min (dB: ev_db[0] - ev_db[-1])
    2  sigma_min        -- std of bottom-half SCM diagonal rows
    3  sigma_max        -- std of top-half SCM diagonal rows
    4  mu_min           -- mean of bottom-half SCM diagonal rows
    5  f_factor         -- sigma_max / (sigma_min + eps)

Directory layout expected on disk
-----------------------------------
    data/
        multi_band/    image_0.h5 .. image_9.h5

    Clean baseline is synthesised on-the-fly from generate_nisar_image.py
    using seeds 0..N_SEEDS-1.

Outputs
-------
Model saved to:
    models/multi_band/best_model.keras

Evaluation PNGs saved to models/multi_band/:
    training_curves.png   -- loss and accuracy vs epoch
    confusion_matrix.png  -- knee confusion matrix (normalised by row)
    metrics.png           -- bar chart of scalar evaluation metrics

JSON metrics saved to:
    models/multi_band/eval_results.json
    models/summary.json
"""

import os
import sys
import json
import numpy as np
import h5py
import tensorflow as tf
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_nisar_image import (
    generate_clean_image,
    BLOCK_HEIGHT, N_BLOCKS, RANGE_BINS, BLOCK_WIDTH,
)
from model import build_model

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

DATA_ROOT   = 'data'
MODELS_ROOT = 'models'
M           = BLOCK_HEIGHT      # pulses per CPI = number of eigenvalues
N_CLASSES   = M + 1             # labels 0..M-1 are RFI knee indices; M = no RFI
N_GLOBAL    = 6
N_SEEDS     = 10                # image_0.h5 .. image_9.h5

EPOCHS      = 50
BATCH_SIZE  = 64
LR          = 3e-4
VAL_FRAC    = 0.1
TEST_FRAC   = 0.1


# ---------------------------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------------------------

def _scm_from_cpi(cpi):
    """
    Compute sample covariance matrix from a complex CPI tile.

    Args:
        cpi (np.ndarray): Complex array shape (M, K).

    Returns:
        R          (np.ndarray): Complex (M, M) SCM = CPI @ CPI^H / K.
        eigvals_db (np.ndarray): Real eigenvalues in dB, descending, shape (M,).
    """
    M, K = cpi.shape
    R = (cpi @ cpi.conj().T) / K
    eigvals = np.linalg.eigvalsh(R)             # ascending
    eigvals = np.sort(np.real(eigvals))[::-1]   # descending
    eigvals_db = 10.0 * np.log10(np.maximum(eigvals, 1e-12))
    return R, eigvals_db


def extract_features(cpi):
    """
    Extract eigen_input and global_input feature vectors from one CPI tile.

    Args:
        cpi (np.ndarray): Complex array shape (M, K).

    Returns:
        eigen   (np.ndarray): shape (M, 2)  -- [eigvals_db, slopes_padded]
        global_ (np.ndarray): shape (6,)    -- scalar context features
    """
    R, eigvals_db = _scm_from_cpi(cpi)

    # --- Eigenvalue branch features ----------------------------------------
    slopes        = np.diff(eigvals_db)             # length M-1
    slopes_padded = np.append(slopes, 0.0)          # length M, zero at end
    eigen = np.stack([eigvals_db, slopes_padded], axis=-1).astype(np.float32)

    # --- Global branch features --------------------------------------------
    # Diagonal of SCM gives per-row power estimates.
    diag_db  = 10.0 * np.log10(np.maximum(np.real(np.diag(R)), 1e-12))
    half     = max(M // 2, 1)
    sigma_max = float(np.std(diag_db[:half]))
    sigma_min = float(np.std(diag_db[half:]))
    mu_min    = float(np.mean(diag_db[half:]))
    trace_db  = 10.0 * np.log10(float(np.maximum(np.real(np.trace(R)), 1e-12)))
    cond_db   = float(eigvals_db[0] - eigvals_db[-1])
    eps       = 1e-6
    f_factor  = sigma_max / (sigma_min + eps)

    global_ = np.array(
        [trace_db, cond_db, sigma_min, sigma_max, mu_min, f_factor],
        dtype=np.float32,
    )
    return eigen, global_


# ---------------------------------------------------------------------------
# LABEL DERIVATION
# ---------------------------------------------------------------------------

def label_from_rfi_bands(rfi_bands_json):
    """
    Derive the knee label from the 'rfi_bands' JSON attribute of an HDF5 tile.

    Label equals the number of distinct local pulse indices occupied by RFI.
    Each distinct row contributes rank-1 to the SCM, producing one elevated
    eigenvalue. The knee sits at eigenvalue index (n_distinct - 1), and the
    encoded label is n_distinct so that label 0 is reserved for the clean
    sentinel (no RFI).

    Args:
        rfi_bands_json (str): JSON string with keys:
            n_bands (int): total bands injected
            bands (list): dicts each containing 'local_idx', 'row', 'jnr_db'

    Returns:
        label (int): n_distinct_rows in [1, MAX_BANDS].
    """
    payload       = json.loads(rfi_bands_json)
    local_indices = {b['local_idx'] for b in payload['bands']}
    return len(local_indices)


# ---------------------------------------------------------------------------
# DATASET LOADING
# ---------------------------------------------------------------------------

def load_rfi_dataset(folder):
    """
    Load all CPI samples from all HDF5 files in the multi_band folder.

    The knee label is derived per-tile from the 'rfi_bands' attribute
    (see label_from_rfi_bands).

    Args:
        folder (str): Path to directory containing image_*.h5 files.

    Returns:
        eigen   (np.ndarray): shape (N, M, 2)
        global_ (np.ndarray): shape (N, 6)
        labels  (np.ndarray): shape (N,)  int32 knee indices
    """
    eigen_list, global_list, label_list = [], [], []

    h5_files = sorted(f for f in os.listdir(folder) if f.endswith('.h5'))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {folder}")

    for fname in h5_files:
        fpath = os.path.join(folder, fname)
        with h5py.File(fpath, 'r') as f:
            for key in f.keys():
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


def load_clean_dataset():
    """
    Generate clean CPI samples on-the-fly (no RFI, label = 0 = clean sentinel).

    Uses seeds 0..N_SEEDS-1 and BLOCK_WIDTH range tiles per block to match
    the tile geometry used by generate_nisar_image.py.

    Returns:
        eigen   (np.ndarray): shape (N, M, 2)
        global_ (np.ndarray): shape (N, 6)
        labels  (np.ndarray): shape (N,)  all equal to 0
    """
    eigen_list, global_list, label_list = [], [], []
    n_range_tiles = RANGE_BINS // BLOCK_WIDTH

    for seed in range(N_SEEDS):
        clean_image = generate_clean_image(seed=seed)   # (TOTAL_PULSES, RANGE_BINS)

        for b in range(N_BLOCKS):
            pulse_start = b * M
            for j in range(n_range_tiles):
                col_start   = j * BLOCK_WIDTH
                cpi         = clean_image[pulse_start:pulse_start + M,
                                          col_start:col_start + BLOCK_WIDTH]
                eigen, glob = extract_features(cpi)
                eigen_list.append(eigen)
                global_list.append(glob)
                label_list.append(0)    # clean sentinel

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
    ax.set_title('Knee Confusion Matrix')

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
    ax.set_title(f"Evaluation Metrics  --  {results['run']}\n"
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
        global_test (np.ndarray): shape (N, 6)
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

    # class_labels: index 0 = clean, indices 1..16 = knee-0..knee-15
    class_labels = ["clean"] + [f"knee-{k}" for k in range(M)]
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
        global_*   (np.ndarray): shape (N, 6)
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
    Full training pipeline.

    Step 1: Load the multi_band RFI dataset and derive per-tile knee labels.
    Step 2: Generate the clean baseline dataset (label = M).
    Step 3: Merge, split 80/10/10, train, and evaluate a single combined model.
    Step 4: Print summary metrics and knee confusion matrix.
    """
    os.makedirs(MODELS_ROOT, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: RFI dataset
    # ------------------------------------------------------------------
    rfi_folder = os.path.join(DATA_ROOT, 'multi_band')
    print(f"\nLoading RFI dataset from {rfi_folder} ...")
    e_rfi, g_rfi, y_rfi = load_rfi_dataset(rfi_folder)
    print(f"  RFI samples  : {len(y_rfi)}")
    unique, counts = np.unique(y_rfi, return_counts=True)
    for u, c in zip(unique.tolist(), counts.tolist()):
        knee_idx = u - 1  # label u -> knee at eigenvalue index u-1
        print(f"    label {u} (knee-{knee_idx}): {c}")

    # ------------------------------------------------------------------
    # Step 2: Clean baseline
    # ------------------------------------------------------------------
    print("\nGenerating clean baseline dataset ...")
    e_clean, g_clean, y_clean = load_clean_dataset()
    print(f"  Clean samples: {len(y_clean)}")

    # ------------------------------------------------------------------
    # Step 3: Merge + split + train
    # ------------------------------------------------------------------
    eigen  = np.concatenate([e_rfi,  e_clean],  axis=0)
    global_ = np.concatenate([g_rfi,  g_clean],  axis=0)
    labels  = np.concatenate([y_rfi,  y_clean],  axis=0)
    print(f"\nTotal samples: {len(labels)}")

    (e_tr, e_va, e_te,
     g_tr, g_va, g_te,
     y_tr, y_va, y_te) = split_dataset(eigen, global_, labels)

    results = train_one_run(
        'multi_band',
        e_tr, e_va, e_te,
        g_tr, g_va, g_te,
        y_tr, y_va, y_te,
    )

    # ------------------------------------------------------------------
    # Step 4: Summary
    # ------------------------------------------------------------------
    summary_path = os.path.join(MODELS_ROOT, 'summary.json')
    with open(summary_path, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSummary saved to {summary_path}")
    print(f"Training plots : models/multi_band/training_curves.png")
    print(f"Confusion matrix: models/multi_band/confusion_matrix.png")
    print(f"Metrics bar chart: models/multi_band/metrics.png")


if __name__ == '__main__':
    main()