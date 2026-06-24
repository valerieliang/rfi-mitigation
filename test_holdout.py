"""
test_holdout.py

Evaluate the saved best_model against freshly generated data that was never
seen during training.  Two data sources are combined:

  1. RFI tiles from  data/low_power_multi_band/  (image_*.h5)
  2. Clean baseline tiles synthesised from generate_nisar_image.py
     using seeds HOLDOUT_SEED_START .. HOLDOUT_SEED_START + N_HOLDOUT_SEEDS - 1

Seeds used for training clean data were 0 .. N_SEEDS-1 (defined in train.py
as N_SEEDS = 10), so holdout seeds start at 100 to guarantee no overlap.

Outputs written to  models/multi_band/holdout/:
    holdout_confusion_matrix.png
    holdout_metrics.png
    holdout_eval.json

Usage
-----
    python test_holdout.py
    python test_holdout.py --model  models/my_run/best_model.keras
    python test_holdout.py --data   data/low_power_multi_band
    python test_holdout.py --seeds  100 20          # start count
"""

import os
import sys
import json
import argparse
import numpy as np
import h5py
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_nisar_image import (
    generate_clean_image,
    BLOCK_HEIGHT, N_BLOCKS, RANGE_BINS, BLOCK_WIDTH,
)
from train import (
    extract_features,
    label_from_rfi_bands,
    save_confusion_matrix_png,
    save_metrics_png,
)

# ---------------------------------------------------------------------------
# DEFAULTS
# ---------------------------------------------------------------------------

DEFAULT_MODEL  = os.path.join('models', 'multi_band', 'best_model.keras')
DEFAULT_DATA   = os.path.join('data', 'low_power_multi_band')
DEFAULT_OUT    = os.path.join('models', 'multi_band', 'holdout')

M              = BLOCK_HEIGHT       # pulses per CPI
N_CLASSES      = M + 1              # labels 0 .. M

# Holdout clean seeds must not overlap with training seeds 0..9
HOLDOUT_SEED_START  = 100
N_HOLDOUT_SEEDS     = 10


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_rfi_holdout(folder):
    """
    Load all RFI CPI tiles from an HDF5 folder not used during training.

    Args:
        folder (str): Path to directory containing image_*.h5 files.

    Returns:
        eigen   (np.ndarray): shape (N, M, 2)
        global_ (np.ndarray): shape (N, 6)
        labels  (np.ndarray): shape (N,)  int32 knee labels
    """
    eigen_list, global_list, label_list = [], [], []

    h5_files = sorted(f for f in os.listdir(folder) if f.endswith('.h5'))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in: {folder}")

    for fname in h5_files:
        fpath = os.path.join(folder, fname)
        with h5py.File(fpath, 'r') as f:
            for key in f.keys():
                cpi         = f[key][:]
                eigen, glob = extract_features(cpi)
                label       = label_from_rfi_bands(str(f[key].attrs['rfi_bands']))
                eigen_list.append(eigen)
                global_list.append(glob)
                label_list.append(label)

    print(f"  Loaded {len(label_list)} RFI tiles from {len(h5_files)} file(s)")
    return (
        np.stack(eigen_list).astype(np.float32),
        np.stack(global_list).astype(np.float32),
        np.array(label_list, dtype=np.int32),
    )


def load_clean_holdout(seed_start, n_seeds):
    """
    Generate clean CPI tiles on-the-fly from seeds not used during training.

    Args:
        seed_start (int): First RNG seed (must be >= N_SEEDS used in train.py).
        n_seeds    (int): Number of seeds to generate.

    Returns:
        eigen   (np.ndarray): shape (N, M, 2)
        global_ (np.ndarray): shape (N, 6)
        labels  (np.ndarray): shape (N,)  all zeros (clean sentinel)
    """
    eigen_list, global_list, label_list = [], [], []
    n_range_tiles = RANGE_BINS // BLOCK_WIDTH

    for seed in range(seed_start, seed_start + n_seeds):
        clean = generate_clean_image(seed=seed)   # (TOTAL_PULSES, RANGE_BINS)
        for b in range(N_BLOCKS):
            pulse_start = b * M
            for j in range(n_range_tiles):
                col_start   = j * BLOCK_WIDTH
                cpi         = clean[pulse_start:pulse_start + M,
                                    col_start:col_start + BLOCK_WIDTH]
                eigen, glob = extract_features(cpi)
                eigen_list.append(eigen)
                global_list.append(glob)
                label_list.append(0)

    print(f"  Generated {len(label_list)} clean tiles "
          f"(seeds {seed_start}..{seed_start + n_seeds - 1})")
    return (
        np.stack(eigen_list).astype(np.float32),
        np.stack(global_list).astype(np.float32),
        np.array(label_list, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

def evaluate_holdout(model, eigen, global_, y_true, out_dir):
    """
    Run inference, compute metrics, and save diagnostic PNGs + JSON.

    Metrics reported:
        exact_acc    -- fraction of samples where predicted label == true label
        tol1_acc     -- fraction where |pred - true| <= 1
        exact_rfi    -- exact_acc restricted to RFI samples (label > 0)
        tol1_rfi     -- tol1_acc  restricted to RFI samples
        exact_no_rfi -- exact_acc restricted to clean samples (label == 0)
        binary_f1    -- F1 on the binary RFI-present / clean detection task

    Args:
        model    : Loaded Keras model.
        eigen    (np.ndarray): shape (N, M, 2)
        global_  (np.ndarray): shape (N, 6)
        y_true   (np.ndarray): shape (N,) integer labels
        out_dir  (str): Directory where outputs are written.

    Returns:
        results (dict): All computed metrics.
    """
    probs  = model.predict([eigen, global_], verbose=0, batch_size=256)
    y_pred = np.argmax(probs, axis=-1).astype(np.int32)
    y_true = y_true.astype(np.int32)

    n = len(y_true)

    # --- Per-class metrics -------------------------------------------------
    exact      = float(np.mean(y_pred == y_true))
    tol1       = float(np.mean(np.abs(y_pred - y_true) <= 1))

    rfi_mask   = y_true > 0
    norfi_mask = y_true == 0

    exact_rfi   = float(np.mean(y_pred[rfi_mask]   == y_true[rfi_mask]))   \
                  if rfi_mask.any()   else float('nan')
    tol1_rfi    = float(np.mean(np.abs(
                      y_pred[rfi_mask].astype(int) - y_true[rfi_mask].astype(int)
                  ) <= 1))                                                   \
                  if rfi_mask.any()   else float('nan')
    exact_norfi = float(np.mean(y_pred[norfi_mask] == y_true[norfi_mask])) \
                  if norfi_mask.any() else float('nan')

    # --- Binary RFI detection F1 ------------------------------------------
    # Positive = RFI present (label > 0)
    tp = int(np.sum((y_pred > 0) & (y_true > 0)))
    fp = int(np.sum((y_pred > 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true > 0)))
    prec      = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    rec       = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    binary_f1 = (2 * prec * rec / (prec + rec)
                 if (not any(map(lambda x: x != x, [prec, rec]))
                     and (prec + rec) > 0)
                 else float('nan'))

    # --- Confidence / entropy diagnostics ----------------------------------
    eps     = 1e-12
    entropy = -np.sum(probs * np.log(probs + eps), axis=-1)   # shape (N,)
    conf    = np.max(probs, axis=-1)

    results = {
        'run'              : 'holdout',
        'n_test'           : n,
        'n_rfi'            : int(rfi_mask.sum()),
        'n_clean'          : int(norfi_mask.sum()),
        'exact_acc'        : exact,
        'tol1_acc'         : tol1,
        'exact_rfi'        : exact_rfi,
        'tol1_rfi'         : tol1_rfi,
        'exact_no_rfi'     : exact_norfi,
        'binary_f1'        : binary_f1,
        'binary_precision' : prec,
        'binary_recall'    : rec,
        'mean_confidence'  : float(conf.mean()),
        'mean_entropy'     : float(entropy.mean()),
    }

    # --- Console summary ---------------------------------------------------
    print("\n=== Holdout Evaluation ===")
    print(f"  N total   : {n}  (RFI={results['n_rfi']}  clean={results['n_clean']})")
    print(f"  Exact acc : {exact:.4f}")
    print(f"  Tol-1 acc : {tol1:.4f}")
    print(f"  RFI exact / tol-1 : {exact_rfi:.4f} / {tol1_rfi:.4f}")
    print(f"  No-RFI exact      : {exact_norfi:.4f}")
    print(f"  Binary F1         : {binary_f1:.4f}  "
          f"(prec={prec:.4f}  rec={rec:.4f})")
    print(f"  Mean confidence   : {conf.mean():.4f}")
    print(f"  Mean entropy      : {entropy.mean():.4f}")

    # --- Save per-class error table ----------------------------------------
    labels_present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    print("\n  Per-class breakdown (label : n_true  exact  tol-1):")
    for lbl in labels_present:
        mask_l = y_true == lbl
        if not mask_l.any():
            continue
        ex_l  = float(np.mean(y_pred[mask_l] == lbl))
        t1_l  = float(np.mean(np.abs(y_pred[mask_l] - lbl) <= 1))
        tag   = 'clean' if lbl == 0 else f'knee-{lbl - 1}'
        print(f"    label {lbl:2d} ({tag:<8s})  "
              f"n={mask_l.sum():5d}  exact={ex_l:.3f}  tol-1={t1_l:.3f}")

    # --- Plots -------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)

    class_labels = ['clean'] + [f'knee-{k}' for k in range(M)]
    save_confusion_matrix_png(y_true, y_pred, class_labels, out_dir)

    # save_metrics_png expects specific keys; build a compatible dict
    metrics_for_plot = {
        'run'          : 'holdout',
        'n_test'       : n,
        'exact_acc'    : exact,
        'tol1_acc'     : tol1,
        'exact_rfi'    : exact_rfi,
        'tol1_rfi'     : tol1_rfi,
        'exact_no_rfi' : exact_norfi,
    }
    save_metrics_png(metrics_for_plot, out_dir)

    # Rename outputs to avoid collision with training-set plots
    for stem in ('confusion_matrix', 'metrics'):
        src = os.path.join(out_dir, f'{stem}.png')
        dst = os.path.join(out_dir, f'holdout_{stem}.png')
        if os.path.exists(src):
            os.replace(src, dst)
            print(f"  Renamed  {src}  ->  {dst}")

    # --- JSON --------------------------------------------------------------
    json_path = os.path.join(out_dir, 'holdout_eval.json')
    with open(json_path, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f"\n  JSON saved to {json_path}")

    return results


# ---------------------------------------------------------------------------
# OVERFITTING DIAGNOSTICS
# ---------------------------------------------------------------------------

def compare_to_training_results(holdout_results, train_json_path):
    """
    Load the training-set eval_results.json and print a side-by-side
    comparison to flag potential overfitting.

    A difference of > 0.05 on exact_acc or tol1_acc between training eval
    and holdout is flagged as a potential overfit indicator.

    Args:
        holdout_results (dict): Output of evaluate_holdout().
        train_json_path (str): Path to models/multi_band/eval_results.json.
    """
    if not os.path.exists(train_json_path):
        print(f"\n  [INFO] No training eval JSON found at {train_json_path}; "
              f"skipping comparison.")
        return

    with open(train_json_path) as fh:
        train_res = json.load(fh)

    print("\n=== Overfit Check (training test-set vs. holdout) ===")
    pairs = [
        ('exact_acc', 'Exact acc'),
        ('tol1_acc',  'Tol-1 acc'),
        ('exact_rfi', 'RFI exact'),
        ('tol1_rfi',  'RFI tol-1'),
    ]
    WARN_THRESHOLD = 0.05
    any_flag = False
    for key, label in pairs:
        tr_val = train_res.get(key, float('nan'))
        ho_val = holdout_results.get(key, float('nan'))
        delta  = tr_val - ho_val
        flag   = '  <<< WARN' if abs(delta) > WARN_THRESHOLD else ''
        if flag:
            any_flag = True
        print(f"  {label:<12s}  train={tr_val:.4f}  holdout={ho_val:.4f}  "
              f"delta={delta:+.4f}{flag}")

    if any_flag:
        print("\n  WARNING: metric drop > 0.05 detected -- possible overfitting.")
    else:
        print("\n  No significant overfit signal detected.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate best_model on holdout data from low_power_multi_band.'
    )
    parser.add_argument(
        '--model', default=DEFAULT_MODEL,
        help=f'Path to saved .keras model  (default: {DEFAULT_MODEL})',
    )
    parser.add_argument(
        '--data', default=DEFAULT_DATA,
        help=f'HDF5 folder for RFI holdout tiles  (default: {DEFAULT_DATA})',
    )
    parser.add_argument(
        '--out', default=DEFAULT_OUT,
        help=f'Output directory for PNGs and JSON  (default: {DEFAULT_OUT})',
    )
    parser.add_argument(
        '--seeds', nargs=2, type=int,
        default=[HOLDOUT_SEED_START, N_HOLDOUT_SEEDS],
        metavar=('START', 'COUNT'),
        help=(f'Start seed and count for clean holdout generation '
              f'(default: {HOLDOUT_SEED_START} {N_HOLDOUT_SEEDS})'),
    )
    args = parser.parse_args()

    seed_start, n_seeds = args.seeds

    # Sanity-check: warn if clean holdout seeds overlap with training seeds
    # (train.py uses seeds 0 .. N_SEEDS-1; N_SEEDS defaults to 10)
    TRAIN_SEED_END = 10
    if seed_start < TRAIN_SEED_END:
        print(f"WARNING: holdout seed start ({seed_start}) overlaps with "
              f"training seeds 0..{TRAIN_SEED_END - 1}. "
              f"Clean holdout tiles will not be fully independent.")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    print(f"\nLoading model from {args.model} ...")
    model = tf.keras.models.load_model(args.model)
    model.summary(line_length=80)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"\nLoading RFI holdout data from {args.data} ...")
    e_rfi, g_rfi, y_rfi = load_rfi_holdout(args.data)

    print(f"\nGenerating clean holdout data ...")
    e_clean, g_clean, y_clean = load_clean_holdout(seed_start, n_seeds)

    eigen   = np.concatenate([e_rfi,   e_clean],  axis=0)
    global_ = np.concatenate([g_rfi,   g_clean],  axis=0)
    labels  = np.concatenate([y_rfi,   y_clean],  axis=0)

    print(f"\nTotal holdout samples: {len(labels)}")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique.tolist(), counts.tolist()):
        tag = 'clean' if u == 0 else f'knee-{u - 1}'
        print(f"  label {u:2d} ({tag}): {c}")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    results = evaluate_holdout(model, eigen, global_, labels, args.out)

    # ------------------------------------------------------------------
    # Compare against training-set eval to surface overfit
    # ------------------------------------------------------------------
    train_json = os.path.join('models', 'multi_band', 'eval_results.json')
    compare_to_training_results(results, train_json)


if __name__ == '__main__':
    main()
