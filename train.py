"""
train.py

Training pipeline for the CNN-based RFI knee-index classifier (model.py).

Label convention
----------------
The knee label is the 0-based sorted eigenvalue index of the last RFI
eigenvalue in the descending SCM spectrum. It equals (rank - 1) of the
RFI contribution to the SCM, which is fixed by RFI type regardless of JNR:

    single_tone : label = 0  (rank-1 -> one elevated eigenvalue)
    wideband    : label = 3  (rank-4 -> four elevated eigenvalues)
    clean       : label = M  (= BLOCK_SIZE = 16, the no-RFI sentinel class)

Model output has M+1 classes (0..M). Class M means "no RFI present".

Feature extraction (per CPI block)
-----------------------------------
eigen_input  shape (M, 2):
    channel 0 -- eigenvalues in dB, descending, from SCM = CPI @ CPI^H / K
    channel 1 -- finite differences of eigenvalues (slopes), length M,
                 zero-padded at index M-1 so the tensor stays (M, 2)

global_input shape (6,):
    0  trace_db         -- 10*log10(trace(R))
    1  condition_number -- lambda_max / lambda_min (dB: ev_db[0] - ev_db[-1])
    2  sigma_min        -- std of lambda_min across blocks in the same range
                          tile (proxy for ST-EST sigma_min; approximated here
                          as std of the last eigenvalue across the 16 rows of
                          this block's SCM diagonal)
    3  sigma_max        -- std of lambda_max across blocks (approximated as
                          std of the first eigenvalue)
    4  mu_min           -- mean of the smallest eigenvalue (proxy for mu_min)
    5  f_factor         -- sigma_max / sigma_min  (ST-EST F indicator)

    Note: the true ST-EST sigma/mu statistics aggregate lambda_max and
    lambda_min *across CPIs within a TB*. Since each HDF5 CPI tile is
    independent here, we approximate using the diagonal of the SCM (the M
    per-row power estimates) as a within-block substitute. This gives the
    model the same qualitative signal with the data available per tile.

Directory layout expected on disk
-----------------------------------
    data/
        low_power/single_tone/    image_0.h5 .. image_9.h5
        low_power/wideband/       image_0.h5 .. image_9.h5
        high_power/single_tone/   image_0.h5 .. image_9.h5
        high_power/wideband/      image_0.h5 .. image_9.h5

    Clean baseline is synthesised on-the-fly from generate_nisar_image.py
    using the same seeds (0..9) so the underlying clean image is identical
    to what was injected into for the RFI runs.

Outputs
-------
Per-subfolder models saved to:
    models/<power_level>_power_<rfi_type>/best_model.keras

Combined model saved to:
    models/combined/best_model.keras

Per-run evaluation printed to stdout and saved as:
    models/<run_name>/eval_results.txt
"""

import os
import sys
import json
import numpy as np
import h5py
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Ensure generate_nisar_image is importable (same directory as train.py or
# adjust this path to wherever it lives in your project layout)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_nisar_image import (
    generate_clean_image,
    BLOCK_SIZE, N_BLOCKS, RANGE_BINS,
)
from model import build_model

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

DATA_ROOT   = 'data'
MODELS_ROOT = 'models'
M           = BLOCK_SIZE        # number of pulses per CPI = number of eigenvalues
N_CLASSES   = M + 1             # 0..M-1 are RFI knee indices; M = no RFI
N_GLOBAL    = 6                 # number of scalar global features
N_SEEDS     = 10                # image_0.h5 .. image_9.h5

EPOCHS      = 50
BATCH_SIZE  = 64
LR          = 3e-4
VAL_FRAC    = 0.1               # 10% of training set used for validation
TEST_FRAC   = 0.1               # 10% held out as test set

SUBFOLDER_CONFIGS = [
    ('low_power',  'single_tone'),
    ('low_power',  'wideband'),
    ('high_power', 'single_tone'),
    ('high_power', 'wideband'),
]


# ---------------------------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------------------------

def _scm_from_cpi(cpi):
    """
    Compute sample covariance matrix from a complex CPI tile.

    Args:
        cpi (np.ndarray): Complex array shape (M, K).

    Returns:
        R (np.ndarray): Real symmetric (M, M) SCM = Re(CPI @ CPI^H / K).
        eigvals_db (np.ndarray): Eigenvalues in dB, descending, shape (M,).
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
        eigen  (np.ndarray): shape (M, 2)  -- [eigvals_db, slopes_padded]
        global_ (np.ndarray): shape (6,)   -- scalar context features
    """
    R, eigvals_db = _scm_from_cpi(cpi)

    # --- Eigenvalue branch features ----------------------------------------
    slopes = np.diff(eigvals_db)                    # length M-1
    slopes_padded = np.append(slopes, 0.0)          # length M, zero at end
    eigen = np.stack([eigvals_db, slopes_padded], axis=-1).astype(np.float32)

    # --- Global branch features --------------------------------------------
    # Diagonal of SCM gives per-row power estimates (proxy for across-CPI stats)
    diag_db  = 10.0 * np.log10(np.maximum(np.real(np.diag(R)), 1e-12))
    sigma_max = float(np.std(diag_db))
    sigma_min = float(np.std(diag_db))             # same approximation here;
    # a more faithful approximation: use max/min rows as proxies
    max_row_power = float(np.max(diag_db))
    min_row_power = float(np.min(diag_db))
    # Re-derive sigma_max from top-half rows, sigma_min from bottom-half
    half = max(M // 2, 1)
    sigma_max = float(np.std(diag_db[:half]))       # top eigenvalue rows
    sigma_min = float(np.std(diag_db[half:]))       # bottom eigenvalue rows
    mu_min    = float(np.mean(diag_db[half:]))

    trace_db  = 10.0 * np.log10(float(np.maximum(np.real(np.trace(R)), 1e-12)))
    cond_db   = float(eigvals_db[0] - eigvals_db[-1])   # lambda_max - lambda_min in dB

    eps = 1e-6
    f_factor = sigma_max / (sigma_min + eps)

    global_ = np.array(
        [trace_db, cond_db, sigma_min, sigma_max, mu_min, f_factor],
        dtype=np.float32,
    )
    return eigen, global_


# ---------------------------------------------------------------------------
# LABEL DERIVATION
# ---------------------------------------------------------------------------

# Fixed knee labels derived from the rank of the RFI contribution to the SCM.
# Single-tone injects one row -> rank-1 -> one elevated eigenvalue -> knee at index 0.
# Wideband injects four rows  -> rank-4 -> four elevated eigenvalues -> knee at index 3.
# Power variation per block (JNR) changes eigenvalue magnitude but not count,
# so the knee index is constant for a given RFI type regardless of JNR.
KNEE_LABEL = {
    'single_tone': 0,
    'wideband'   : 3,
}


def label_from_rfi_type(rfi_type):
    """
    Return the fixed ground-truth knee label for a given RFI type.

    The knee index is the 0-based sorted eigenvalue index of the last RFI
    eigenvalue in the descending spectrum. It equals (rank - 1) of the RFI
    contribution to the SCM.

    Args:
        rfi_type (str): 'single_tone' or 'wideband'.

    Returns:
        label (int): 0 for single_tone, 3 for wideband.
    """
    if rfi_type not in KNEE_LABEL:
        raise ValueError(f"Unknown rfi_type: '{rfi_type}'. Expected one of {list(KNEE_LABEL)}")
    return KNEE_LABEL[rfi_type]


# ---------------------------------------------------------------------------
# DATASET LOADING
# ---------------------------------------------------------------------------

def load_dataset_from_folder(folder, rfi_type):
    """
    Load all CPI samples from all HDF5 files in a subfolder.

    Reads every dataset from every image_*.h5 file and extracts features.
    The label is constant for all samples in a folder: it equals the fixed
    knee index for the given rfi_type (see KNEE_LABEL).

    Args:
        folder   (str): Path to subfolder containing image_*.h5 files.
        rfi_type (str): 'single_tone' or 'wideband'.

    Returns:
        eigen   (np.ndarray): shape (N, M, 2)
        global_ (np.ndarray): shape (N, 6)
        labels  (np.ndarray): shape (N,)  all equal to KNEE_LABEL[rfi_type]
    """
    eigen_list, global_list = [], []
    label = label_from_rfi_type(rfi_type)

    h5_files = sorted(
        f for f in os.listdir(folder) if f.endswith('.h5')
    )
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {folder}")

    for fname in h5_files:
        fpath = os.path.join(folder, fname)
        with h5py.File(fpath, 'r') as f:
            for key in f.keys():
                cpi         = f[key][:]
                eigen, glob = extract_features(cpi)
                eigen_list.append(eigen)
                global_list.append(glob)

    n = len(eigen_list)
    return (
        np.stack(eigen_list).astype(np.float32),
        np.stack(global_list).astype(np.float32),
        np.full(n, label, dtype=np.int32),
    )


def load_clean_dataset():
    """
    Generate clean CPI samples on-the-fly (no RFI, label = M = no-RFI class).

    Uses the same seeds (0..N_SEEDS-1) as the RFI pipeline so the underlying
    scene statistics match what the model sees in RFI runs.

    Returns:
        eigen   (np.ndarray): shape (N, M, 2)
        global_ (np.ndarray): shape (N, 6)
        labels  (np.ndarray): shape (N,)  all equal to M (no-RFI class)
    """
    eigen_list, global_list, label_list = [], [], []
    n_range_tiles = RANGE_BINS // 1000  # matches BLOCK_WIDTH=1000 in generator

    for seed in range(N_SEEDS):
        clean_image = generate_clean_image(seed=seed)   # (TOTAL_PULSES, RANGE_BINS)

        for b in range(N_BLOCKS):
            pulse_start = b * M
            for j in range(n_range_tiles):
                col_start = j * 1000
                cpi       = clean_image[pulse_start:pulse_start + M,
                                        col_start:col_start + 1000]
                eigen, glob = extract_features(cpi)
                eigen_list.append(eigen)
                global_list.append(glob)
                label_list.append(M)   # no-RFI class

    return (
        np.stack(eigen_list).astype(np.float32),
        np.stack(global_list).astype(np.float32),
        np.array(label_list, dtype=np.int32),
    )


def split_dataset(eigen, global_, labels):
    """
    Split into train / val / test sets (80 / 10 / 10).

    Stratified on labels to preserve class distribution across splits.

    Args:
        eigen   (np.ndarray): shape (N, M, 2)
        global_ (np.ndarray): shape (N, 6)
        labels  (np.ndarray): shape (N,)

    Returns:
        Six arrays: eigen_train, eigen_val, eigen_test,
                    global_train, global_val, global_test,
                    y_train, y_val, y_test
    """
    # First cut: 80% train+val, 20% test -> then split train+val 89/11 ~ 80/10
    e_tv, e_test, g_tv, g_test, y_tv, y_test = train_test_split(
        eigen, global_, labels,
        test_size=TEST_FRAC, random_state=42, stratify=labels,
    )
    val_frac_of_tv = VAL_FRAC / (1.0 - TEST_FRAC)   # ~0.111 to get 10% overall
    e_train, e_val, g_train, g_val, y_train, y_val = train_test_split(
        e_tv, g_tv, y_tv,
        test_size=val_frac_of_tv, random_state=42, stratify=y_tv,
    )
    return e_train, e_val, e_test, g_train, g_val, g_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------

def evaluate(model, eigen_test, global_test, y_test, run_name):
    """
    Evaluate model on a held-out test set and print / return results.

    Metrics reported:
        exact accuracy    -- argmax prediction == label
        tol-1 accuracy    -- |argmax - label| <= 1
        per-class breakdown for 'no-RFI' (label==M) vs 'RFI present' (label<M)

    Args:
        model       : Trained Keras model.
        eigen_test  (np.ndarray): shape (N, M, 2)
        global_test (np.ndarray): shape (N, 6)
        y_test      (np.ndarray): shape (N,) integer labels
        run_name    (str): Label used in printout.

    Returns:
        results (dict): All computed metrics.
    """
    probs      = model.predict([eigen_test, global_test], verbose=0)
    y_pred     = np.argmax(probs, axis=-1)

    exact      = float(np.mean(y_pred == y_test))
    tol1       = float(np.mean(np.abs(y_pred.astype(int) - y_test.astype(int)) <= 1))

    # No-RFI class (label == M) vs RFI class (label < M)
    rfi_mask   = y_test < M
    norfi_mask = y_test == M

    exact_rfi   = float(np.mean(y_pred[rfi_mask]   == y_test[rfi_mask]))   \
                  if rfi_mask.any() else float('nan')
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

    lines = [
        f"\n=== {run_name} ===",
        f"  N test samples : {results['n_test']}",
        f"  Exact accuracy : {exact:.4f}",
        f"  Tol-1 accuracy : {tol1:.4f}  (|pred - label| <= 1)",
        f"  --- RFI samples (label < M) ---",
        f"  Exact           : {exact_rfi:.4f}",
        f"  Tol-1           : {tol1_rfi:.4f}",
        f"  --- No-RFI samples (label == M) ---",
        f"  Exact           : {exact_norfi:.4f}",
    ]
    report = '\n'.join(lines)
    print(report)
    return results, report


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def train_one_run(run_name, eigen_train, eigen_val, eigen_test,
                  global_train, global_val, global_test,
                  y_train, y_val, y_test):
    """
    Build, train, evaluate and save a model for one dataset configuration.

    Saves the best validation-loss checkpoint to:
        models/<run_name>/best_model.keras

    Args:
        run_name   (str): Unique name for this run (used as folder name).
        eigen_*    (np.ndarray): Eigenvalue + slope features, shape (N, M, 2).
        global_*   (np.ndarray): Scalar context features, shape (N, 6).
        y_*        (np.ndarray): Integer labels, shape (N,).

    Returns:
        results (dict): Evaluation metrics on the test set.
    """
    out_dir = os.path.join(MODELS_ROOT, run_name)
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'best_model.keras')

    print(f"\n{'='*60}")
    print(f"Run : {run_name}")
    print(f"  train={len(y_train)}  val={len(y_val)}  test={len(y_test)}")
    print(f"  classes: {np.unique(y_train)}")
    print(f"{'='*60}")

    model = build_model(
        cpi_size         = M,
        n_global_features= N_GLOBAL,
        n_knee_classes   = N_CLASSES,
        dropout_rate     = 0.5,
        learning_rate    = LR,
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath          = model_path,
            monitor           = 'val_loss',
            save_best_only    = True,
            verbose           = 1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor           = 'val_loss',
            patience          = 10,
            restore_best_weights = True,
            verbose           = 1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor           = 'val_loss',
            factor            = 0.5,
            patience          = 5,
            min_lr            = 1e-6,
            verbose           = 1,
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

    results, report = evaluate(model, eigen_test, global_test, y_test, run_name)

    # Save eval report alongside model
    with open(os.path.join(out_dir, 'eval_results.txt'), 'w') as f:
        f.write(report + '\n')
    with open(os.path.join(out_dir, 'eval_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    """
    Full training pipeline.

    Step 1: Load clean baseline dataset (no RFI, label = M).
    Step 2: For each of the four RFI subfolders, load that subfolder's data,
            merge with the clean baseline, split 80/10/10, and train a model.
            Best model saved under models/<power>_power_<rfi_type>/.
    Step 3: Pool all four RFI datasets together with the clean baseline,
            split 80/10/10, and train a combined model.
            Best model saved under models/combined/.
    Step 4: Print a summary table of all eval results.
    """
    os.makedirs(MODELS_ROOT, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Clean baseline
    # ------------------------------------------------------------------
    print("\nLoading clean baseline dataset...")
    e_clean, g_clean, y_clean = load_clean_dataset()
    print(f"  Clean samples: {len(y_clean)}")

    # ------------------------------------------------------------------
    # Step 2: Per-subfolder models
    # ------------------------------------------------------------------
    all_results   = []
    # Accumulate all RFI data for the combined run
    all_eigen_rfi  = []
    all_global_rfi = []
    all_labels_rfi = []

    for power_level, rfi_type in SUBFOLDER_CONFIGS:
        folder   = os.path.join(DATA_ROOT, power_level, rfi_type)
        run_name = f"{power_level}_{rfi_type}"

        print(f"\nLoading {run_name} from {folder} ...")
        e_rfi, g_rfi, y_rfi = load_dataset_from_folder(folder, rfi_type)
        print(f"  RFI samples: {len(y_rfi)}")

        all_eigen_rfi.append(e_rfi)
        all_global_rfi.append(g_rfi)
        all_labels_rfi.append(y_rfi)

        # Merge RFI data with clean baseline
        eigen   = np.concatenate([e_rfi,  e_clean],  axis=0)
        global_ = np.concatenate([g_rfi,  g_clean],  axis=0)
        labels  = np.concatenate([y_rfi,  y_clean],  axis=0)

        (e_tr, e_va, e_te,
         g_tr, g_va, g_te,
         y_tr, y_va, y_te) = split_dataset(eigen, global_, labels)

        results = train_one_run(
            run_name,
            e_tr, e_va, e_te,
            g_tr, g_va, g_te,
            y_tr, y_va, y_te,
        )
        all_results.append(results)

    # ------------------------------------------------------------------
    # Step 3: Combined model (all four RFI types + clean)
    # ------------------------------------------------------------------
    print("\n\nBuilding combined dataset (all RFI types + clean)...")

    e_all_rfi  = np.concatenate(all_eigen_rfi,  axis=0)
    g_all_rfi  = np.concatenate(all_global_rfi, axis=0)
    y_all_rfi  = np.concatenate(all_labels_rfi, axis=0)

    eigen_comb   = np.concatenate([e_all_rfi,  e_clean],  axis=0)
    global_comb  = np.concatenate([g_all_rfi,  g_clean],  axis=0)
    labels_comb  = np.concatenate([y_all_rfi,  y_clean],  axis=0)
    print(f"  Combined samples: {len(labels_comb)}")

    (e_tr, e_va, e_te,
     g_tr, g_va, g_te,
     y_tr, y_va, y_te) = split_dataset(eigen_comb, global_comb, labels_comb)

    results_comb = train_one_run(
        'combined',
        e_tr, e_va, e_te,
        g_tr, g_va, g_te,
        y_tr, y_va, y_te,
    )
    all_results.append(results_comb)

    # ------------------------------------------------------------------
    # Step 4: Summary table
    # ------------------------------------------------------------------
    print("\n\n" + "="*70)
    print(f"{'Run':<35} {'Exact':>8} {'Tol-1':>8} {'RFI-Exact':>10} {'RFI-Tol1':>9}")
    print("="*70)
    for r in all_results:
        print(
            f"{r['run']:<35} "
            f"{r['exact_acc']:>8.4f} "
            f"{r['tol1_acc']:>8.4f} "
            f"{r['exact_rfi']:>10.4f} "
            f"{r['tol1_rfi']:>9.4f}"
        )
    print("="*70)

    # Save overall summary
    summary_path = os.path.join(MODELS_ROOT, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == '__main__':
    main()