"""
test_nisar_holdout.py

Evaluate the saved best_model against:

  1. Real NISAR L0B CPI blocks extracted to nisar_data/processed/*.h5
     (no ground-truth labels -- inference only, confidence/entropy reported)

  2. Clean baseline tiles synthesised from generate_nisar_image.py
     using seeds HOLDOUT_SEED_START .. HOLDOUT_SEED_START + N_HOLDOUT_SEEDS - 1
     (label 0 = clean sentinel, ground truth known)

Each processed block HDF5 has a single 'data' dataset of shape
(n_pulses, n_range_samples) complex64, with attrs:
    pulse_start, pulse_end, range_start, range_end, source_file, source_path

The block is sliced into non-overlapping CPI tiles of M pulses x BLOCK_WIDTH
range samples.  Because real data has no knee-index ground truth, the real-data
path skips accuracy metrics and only reports prediction distributions,
confidence, and entropy.

Outputs written to  models/multi_band/holdout/:
    holdout_confusion_matrix.png   (clean holdout only)
    holdout_metrics.png            (clean holdout only)
    holdout_predictions.png        (real data -- predicted knee distribution)
    holdout_eval.json

Usage
-----
    python test_holdout.py
    python test_holdout.py --model  models/my_run/best_model.keras
    python test_holdout.py --blocks nisar_data/processed
    python test_holdout.py --seeds  100 20
"""

import os
import sys
import json
import argparse
import numpy as np
import h5py
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_nisar_image import (
    generate_clean_image,
    BLOCK_HEIGHT, N_BLOCKS, RANGE_BINS, BLOCK_WIDTH,
)
from train import (
    extract_features,
    save_confusion_matrix_png,
    save_metrics_png,
)

# ---------------------------------------------------------------------------
# DEFAULTS
# ---------------------------------------------------------------------------

DEFAULT_MODEL  = os.path.join('models', 'multi_band', 'best_model.keras')
DEFAULT_BLOCKS = os.path.join('nisar_data', 'processed')
DEFAULT_OUT    = os.path.join('models', 'multi_band', 'holdout')

M         = BLOCK_HEIGHT    # pulses per CPI
N_CLASSES = M + 1           # labels 0 .. M

HOLDOUT_SEED_START = 100
N_HOLDOUT_SEEDS    = 10


# ---------------------------------------------------------------------------
# REAL DATA LOADING (no ground-truth labels)
# ---------------------------------------------------------------------------

def load_real_blocks(folder):
    """
    Slice all processed block HDF5 files into CPI tiles and extract features.

    Each file must have a 'data' dataset of shape (n_pulses, n_range) complex64
    and attrs: pulse_start, range_start (true file coordinates).

    The block is divided into non-overlapping tiles of shape (M, BLOCK_WIDTH).
    Tiles that do not fill a full M x BLOCK_WIDTH window are discarded.

    Args:
        folder (str): Path to nisar_data/processed/ directory.

    Returns:
        eigen    (np.ndarray): shape (N, M, 2)
        global_  (np.ndarray): shape (N, 6)
        origins  (list[dict]): true-coordinate origin of each tile,
                               keys: file, pulse_start, range_start
    """
    eigen_list, global_list, origin_list = [], [], []

    h5_files = sorted(f for f in os.listdir(folder) if f.endswith('.h5'))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in: {folder}")

    for fname in h5_files:
        fpath = os.path.join(folder, fname)
        with h5py.File(fpath, 'r') as f:
            ds           = f['data']
            block        = ds[:]                           # complex64 (P, R)
            blk_pulse0   = int(ds.attrs['pulse_start'])
            blk_range0   = int(ds.attrs['range_start'])

        n_pulses, n_range = block.shape
        n_cpi_tiles   = n_pulses // M
        n_range_tiles = n_range  // BLOCK_WIDTH

        tile_count = 0
        for ci in range(n_cpi_tiles):
            p0 = ci * M
            for ri in range(n_range_tiles):
                r0  = ri * BLOCK_WIDTH
                cpi = block[p0:p0 + M, r0:r0 + BLOCK_WIDTH]

                eigen, glob = extract_features(cpi)
                eigen_list.append(eigen)
                global_list.append(glob)
                origin_list.append({
                    'file'        : fname,
                    'pulse_start' : blk_pulse0 + p0,
                    'range_start' : blk_range0 + r0,
                })
                tile_count += 1

        print(f"  {fname}: {n_cpi_tiles} x {n_range_tiles} = {tile_count} tiles "
              f"(block shape {n_pulses}x{n_range})")

    print(f"  Total real tiles: {len(eigen_list)}")
    return (
        np.stack(eigen_list).astype(np.float32),
        np.stack(global_list).astype(np.float32),
        origin_list,
    )


# ---------------------------------------------------------------------------
# CLEAN HOLDOUT LOADING (ground truth label = 0)
# ---------------------------------------------------------------------------

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
        clean = generate_clean_image(seed=seed)
        for b in range(N_BLOCKS):
            pulse_start = b * M
            for j in range(n_range_tiles):
                col_start = j * BLOCK_WIDTH
                cpi       = clean[pulse_start:pulse_start + M,
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
# INFERENCE ON REAL DATA (no labels)
# ---------------------------------------------------------------------------

def infer_real_blocks(model, eigen, global_, origins, out_dir):
    """
    Run inference on real NISAR blocks and report prediction distribution,
    confidence, and entropy.  No accuracy metrics (no ground truth).

    Args:
        model   : Loaded Keras model.
        eigen   (np.ndarray): shape (N, M, 2)
        global_ (np.ndarray): shape (N, 6)
        origins (list[dict]): true-coordinate origin per tile
        out_dir (str): Directory where the distribution plot is saved.

    Returns:
        preds (np.ndarray): predicted knee indices, shape (N,)
    """
    probs  = model.predict([eigen, global_], verbose=0, batch_size=256)
    preds  = np.argmax(probs, axis=-1).astype(np.int32)
    conf   = np.max(probs, axis=-1)
    eps    = 1e-12
    entropy = -np.sum(probs * np.log(probs + eps), axis=-1)

    rfi_frac = float(np.mean(preds > 0))

    print("\n=== Real NISAR Block Inference ===")
    print(f"  Tiles inferred : {len(preds)}")
    print(f"  Predicted RFI  : {int((preds > 0).sum())}  ({rfi_frac:.3f})")
    print(f"  Predicted clean: {int((preds == 0).sum())}")
    print(f"  Mean confidence: {conf.mean():.4f}")
    print(f"  Mean entropy   : {entropy.mean():.4f}")

    unique, counts = np.unique(preds, return_counts=True)
    print("  Predicted knee distribution:")
    for u, c in zip(unique.tolist(), counts.tolist()):
        tag = 'clean' if u == 0 else f'knee-{u - 1}'
        print(f"    label {u:2d} ({tag:<8s}): {c:5d}  ({100*c/len(preds):.1f}%)")

    # --- Prediction distribution bar chart --------------------------------
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    bins   = np.arange(N_CLASSES)
    ax.bar(bins, np.bincount(preds, minlength=N_CLASSES), color='steelblue')
    ax.set_xlabel('Predicted Knee Index (0 = clean)')
    ax.set_ylabel('Tile Count')
    ax.set_title('Real NISAR Block -- Predicted Knee Distribution')
    ax.set_xticks(bins[::2])
    fig.tight_layout()
    plot_path = os.path.join(out_dir, 'holdout_predictions.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {plot_path}")

    return preds, conf, entropy


# ---------------------------------------------------------------------------
# EVALUATION ON CLEAN HOLDOUT (labels known)
# ---------------------------------------------------------------------------

def evaluate_clean_holdout(model, eigen, global_, y_true, out_dir):
    """
    Run inference on synthetic clean tiles, compute accuracy metrics,
    and save diagnostic PNGs + JSON.

    Args:
        model    : Loaded Keras model.
        eigen    (np.ndarray): shape (N, M, 2)
        global_  (np.ndarray): shape (N, 6)
        y_true   (np.ndarray): shape (N,) all zeros for clean tiles
        out_dir  (str): Directory where outputs are written.

    Returns:
        results (dict): Computed metrics.
    """
    probs  = model.predict([eigen, global_], verbose=0, batch_size=256)
    y_pred = np.argmax(probs, axis=-1).astype(np.int32)
    y_true = y_true.astype(np.int32)

    n           = len(y_true)
    exact       = float(np.mean(y_pred == y_true))
    tol1        = float(np.mean(np.abs(y_pred - y_true) <= 1))
    norfi_mask  = y_true == 0
    exact_norfi = float(np.mean(y_pred[norfi_mask] == y_true[norfi_mask])) \
                  if norfi_mask.any() else float('nan')

    # Binary F1: positive = predicted RFI on clean tiles = false alarms
    fp        = int(np.sum(y_pred > 0))
    fn        = 0   # all true labels are 0; no missed RFI
    tp        = 0
    prec      = float('nan')
    rec       = float('nan')
    binary_f1 = float('nan')
    false_alarm_rate = fp / n if n > 0 else float('nan')

    eps     = 1e-12
    entropy = -np.sum(probs * np.log(probs + eps), axis=-1)
    conf    = np.max(probs, axis=-1)

    results = {
        'run'               : 'holdout_clean',
        'n_test'            : n,
        'exact_acc'         : exact,
        'tol1_acc'          : tol1,
        'exact_rfi'         : float('nan'),
        'tol1_rfi'          : float('nan'),
        'exact_no_rfi'      : exact_norfi,
        'false_alarm_rate'  : false_alarm_rate,
        'mean_confidence'   : float(conf.mean()),
        'mean_entropy'      : float(entropy.mean()),
    }

    print("\n=== Clean Holdout Evaluation ===")
    print(f"  N clean tiles  : {n}")
    print(f"  Exact acc      : {exact:.4f}")
    print(f"  Tol-1 acc      : {tol1:.4f}")
    print(f"  False alarm rate (pred RFI on clean): {false_alarm_rate:.4f}  ({fp}/{n})")
    print(f"  Mean confidence: {conf.mean():.4f}")
    print(f"  Mean entropy   : {entropy.mean():.4f}")

    os.makedirs(out_dir, exist_ok=True)
    class_labels = ['clean'] + [f'knee-{k}' for k in range(M)]
    save_confusion_matrix_png(y_true, y_pred, class_labels, out_dir)
    metrics_for_plot = {
        'run'          : 'holdout_clean',
        'n_test'       : n,
        'exact_acc'    : exact,
        'tol1_acc'     : tol1,
        'exact_rfi'    : float('nan'),
        'tol1_rfi'     : float('nan'),
        'exact_no_rfi' : exact_norfi,
    }
    save_metrics_png(metrics_for_plot, out_dir)

    for stem in ('confusion_matrix', 'metrics'):
        src = os.path.join(out_dir, f'{stem}.png')
        dst = os.path.join(out_dir, f'holdout_{stem}.png')
        if os.path.exists(src):
            os.replace(src, dst)

    return results


# ---------------------------------------------------------------------------
# OVERFIT CHECK
# ---------------------------------------------------------------------------

def compare_to_training_results(holdout_results, train_json_path):
    """
    Compare holdout metrics against training-set eval_results.json.

    Args:
        holdout_results (dict): Output of evaluate_clean_holdout().
        train_json_path (str): Path to models/multi_band/eval_results.json.
    """
    if not os.path.exists(train_json_path):
        print(f"\n  [INFO] No training eval JSON at {train_json_path}; "
              f"skipping comparison.")
        return

    with open(train_json_path) as fh:
        train_res = json.load(fh)

    print("\n=== Overfit Check (training test-set vs. clean holdout) ===")
    pairs = [
        ('exact_acc',   'Exact acc'),
        ('tol1_acc',    'Tol-1 acc'),
        ('exact_no_rfi','No-RFI exact'),
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
        print(f"  {label:<14s}  train={tr_val:.4f}  holdout={ho_val:.4f}  "
              f"delta={delta:+.4f}{flag}")

    if any_flag:
        print("\n  WARNING: metric drop > 0.05 -- possible overfitting.")
    else:
        print("\n  No significant overfit signal detected.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate best_model on real NISAR processed blocks '
                    'and synthetic clean holdout tiles.'
    )
    parser.add_argument(
        '--model', default=DEFAULT_MODEL,
        help=f'Path to saved .keras model  (default: {DEFAULT_MODEL})',
    )
    parser.add_argument(
        '--blocks', default=DEFAULT_BLOCKS,
        help=f'Directory of processed block HDF5 files  (default: {DEFAULT_BLOCKS})',
    )
    parser.add_argument(
        '--out', default=DEFAULT_OUT,
        help=f'Output directory for PNGs and JSON  (default: {DEFAULT_OUT})',
    )
    parser.add_argument(
        '--seeds', nargs=2, type=int,
        default=[HOLDOUT_SEED_START, N_HOLDOUT_SEEDS],
        metavar=('START', 'COUNT'),
        help=f'Start seed and count for clean holdout generation '
             f'(default: {HOLDOUT_SEED_START} {N_HOLDOUT_SEEDS})',
    )
    parser.add_argument(
        '--no-clean', action='store_true',
        help='Skip synthetic clean holdout generation',
    )
    args = parser.parse_args()

    seed_start, n_seeds = args.seeds

    TRAIN_SEED_END = 10
    if seed_start < TRAIN_SEED_END:
        print(f"WARNING: holdout seed start ({seed_start}) overlaps with "
              f"training seeds 0..{TRAIN_SEED_END - 1}.")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    print(f"\nLoading model from {args.model} ...")
    model = tf.keras.models.load_model(args.model)

    # ------------------------------------------------------------------
    # Real NISAR blocks -- inference only
    # ------------------------------------------------------------------
    print(f"\nLoading real NISAR blocks from {args.blocks} ...")
    e_real, g_real, origins = load_real_blocks(args.blocks)

    real_preds, real_conf, real_entropy = infer_real_blocks(
        model, e_real, g_real, origins, args.out
    )

    # ------------------------------------------------------------------
    # Clean holdout -- accuracy metrics
    # ------------------------------------------------------------------
    clean_results = {}
    if not args.no_clean:
        print(f"\nGenerating clean holdout tiles ...")
        e_clean, g_clean, y_clean = load_clean_holdout(seed_start, n_seeds)
        clean_results = evaluate_clean_holdout(
            model, e_clean, g_clean, y_clean, args.out
        )

    # ------------------------------------------------------------------
    # Save combined JSON
    # ------------------------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    combined = {
        'real_blocks': {
            'n_tiles'         : len(real_preds),
            'rfi_fraction'    : float(np.mean(real_preds > 0)),
            'mean_confidence' : float(real_conf.mean()),
            'mean_entropy'    : float(real_entropy.mean()),
            'knee_counts'     : {
                str(k): int(c)
                for k, c in zip(*np.unique(real_preds, return_counts=True))
            },
        },
        'clean_holdout': clean_results,
    }
    json_path = os.path.join(args.out, 'holdout_eval.json')
    with open(json_path, 'w') as fh:
        json.dump(combined, fh, indent=2)
    print(f"\nJSON saved to {json_path}")

    # ------------------------------------------------------------------
    # Overfit check against training eval
    # ------------------------------------------------------------------
    if clean_results:
        train_json = os.path.join('models', 'multi_band', 'eval_results.json')
        compare_to_training_results(clean_results, train_json)


if __name__ == '__main__':
    main()