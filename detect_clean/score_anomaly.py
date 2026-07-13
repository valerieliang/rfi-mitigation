"""
score_anomaly.py

Inference-only script: scores a region of NEW L0B data (not used in training)
against an already-trained anomaly autoencoder, using the exact normalization
stats saved at training time.

This does NOT retrain or fine-tune the model in any way -- it only reads
best_model.keras and norm_stats.npz and runs forward passes.

Usage
-----
    python score_anomaly.py granule.h5 --freq A --pol HV \
        --pulse-start 888222 --pulse-end 896222 \
        --range-start 2000 --range-end 25000 \
        --model-dir models/anomaly_v1 \
        --output-dir eval/next_8000

By default, --model-dir is expected to contain:
    best_model.keras
    norm_stats.npz
and, if present, run_summary.json is used to auto-populate the recommended
p95/p99 anomaly thresholds for reporting (override with --threshold-p95 /
--threshold-p99).

Outputs (in --output-dir)
--------------------------
    anomaly_scores.npz     -- per-tile eigen_err, global_err, total_score,
                               tile_pulse_idx, tile_range_idx (absolute,
                               original file coordinates)
    anomaly_score_hist.png -- histogram of total_score with threshold lines
    anomaly_score_map.png  -- 2D heatmap of total_score over the
                               (pulse_tile x range_tile) grid, so spatial
                               clustering of anomalies is visible at a glance
    top_anomalies.json     -- the --n-top highest-scoring tiles, with their
                               absolute pulse_start/range_start so you can
                               feed them directly into plot_random_cpi.py
                               via a --region-manifest for visual inspection
    score_summary.json     -- percentile summary and threshold pass rates
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from anomaly_features import tile_and_extract_features
from model_anomaly import compute_anomaly_scores

from nisar.products.readers.Raw import Raw

EIGEN_LOSS_WEIGHT_DEFAULT = 1.0
GLOBAL_LOSS_WEIGHT_DEFAULT = 0.5
N_TOP_DEFAULT = 50


# ---------------------------------------------------------------------------
# ISCE3 RAW DATA / SUBSWATH MASK HELPERS (self-contained, matches
# train_anomaly.py / plot_random_cpi.py)
# ---------------------------------------------------------------------------

def read_raw_data_batch(raw: Raw, freq: str, pol: str, pulse_slice: slice, range_slice: slice):
    dataset = raw.getRawDataset(freq, pol)

    pulse_start = pulse_slice.start if pulse_slice.start is not None else 0
    pulse_stop = pulse_slice.stop if pulse_slice.stop is not None else dataset.shape[0]
    range_start = range_slice.start if range_slice.start is not None else 0
    range_stop = range_slice.stop if range_slice.stop is not None else dataset.shape[1]

    return dataset[pulse_start:pulse_stop, range_start:range_stop]


def get_subswath_mask(raw: Raw, freq: str, pol: str, pulse_indices: np.ndarray, range_indices: np.ndarray) -> np.ndarray:
    tx_pol = pol[0]
    subswaths = raw.getSubSwaths(freq, tx_pol)
    swaths = subswaths[:, pulse_indices, :]

    num_pulses = len(pulse_indices)
    num_range_samples = len(range_indices)
    mask = np.zeros((num_pulses, num_range_samples), dtype=bool)

    r_offset = int(range_indices[0])

    if swaths is not None:
        for i in range(num_pulses):
            for start, end in swaths[:, i, :]:
                s = max(int(start) - r_offset, 0)
                e = min(int(end) - r_offset, num_range_samples)
                if e > s:
                    mask[i, s:e] = True

    return mask


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------

def save_score_hist_png(total_score, threshold_p95, threshold_p99, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(total_score, bins=80, alpha=0.8, color='tab:blue')
    if threshold_p95 is not None:
        ax.axvline(threshold_p95, color='orange', linestyle='--', label=f'Train p95 = {threshold_p95:.4f}')
    if threshold_p99 is not None:
        ax.axvline(threshold_p99, color='red', linestyle='--', label=f'Train p99 = {threshold_p99:.4f}')
    ax.set_xlabel('Total anomaly score (weighted reconstruction MSE)')
    ax.set_ylabel('Tile count')
    ax.set_title('Anomaly Score Distribution (New Data)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    path = os.path.join(out_dir, 'anomaly_score_hist.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def save_score_map_png(total_score, n_pulse_tiles, n_range_tiles, out_dir):
    score_grid = total_score.reshape(n_pulse_tiles, n_range_tiles)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(score_grid, aspect='auto', origin='upper', cmap='inferno')
    ax.set_xlabel('Range Tile Index')
    ax.set_ylabel('Pulse Tile Index (CPI index)')
    ax.set_title('Anomaly Score Map (New Data)')
    fig.colorbar(im, ax=ax, label='Total anomaly score')

    fig.tight_layout()
    path = os.path.join(out_dir, 'anomaly_score_map.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Score new L0B data against an already-trained anomaly autoencoder (inference only)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('l0b_file', help='Input NISAR L0B HDF5 file')
    parser.add_argument('--freq', choices=['A', 'B'], default='A')
    parser.add_argument('--pol', choices=['HH', 'HV', 'VH', 'VV'], default='HV')

    parser.add_argument('--pulse-start', type=int, default=None)
    parser.add_argument('--pulse-end', type=int, default=None)
    parser.add_argument('--range-start', type=int, default=None)
    parser.add_argument('--range-end', type=int, default=None)

    parser.add_argument('--model-dir', type=str, required=True,
                        help='Directory containing best_model.keras and norm_stats.npz from training')
    parser.add_argument('--output-dir', type=str, default='eval/scored', help='Output directory for this scoring run')

    parser.add_argument('--eigen-loss-weight', type=float, default=None,
                        help=f'Overrides the eigen reconstruction weight (default: {EIGEN_LOSS_WEIGHT_DEFAULT}, '
                             f'should match the weight used at training time)')
    parser.add_argument('--global-loss-weight', type=float, default=None,
                        help=f'Overrides the global reconstruction weight (default: {GLOBAL_LOSS_WEIGHT_DEFAULT}, '
                             f'should match the weight used at training time)')

    parser.add_argument('--threshold-p95', type=float, default=None,
                        help='Anomaly score threshold for reporting (default: auto-read from run_summary.json '
                             'in --model-dir if present)')
    parser.add_argument('--threshold-p99', type=float, default=None,
                        help='Anomaly score threshold for reporting (default: auto-read from run_summary.json '
                             'in --model-dir if present)')

    parser.add_argument('--n-top', type=int, default=N_TOP_DEFAULT,
                        help=f'Number of highest-scoring tiles to save for follow-up inspection (default: {N_TOP_DEFAULT})')

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load trained model and normalization stats (no training here)
    # ------------------------------------------------------------------
    model_path = os.path.join(args.model_dir, 'best_model.keras')
    norm_stats_path = os.path.join(args.model_dir, 'norm_stats.npz')

    print(f"Loading model from {model_path} ...")
    model = tf.keras.models.load_model(model_path)

    print(f"Loading normalization stats from {norm_stats_path} ...")
    stats = np.load(norm_stats_path)
    eigen_mean, eigen_std = stats['eigen_mean'], stats['eigen_std']
    global_mean, global_std = stats['global_mean'], stats['global_std']
    n_keep = int(stats['n_keep'])
    cpi_len = int(stats['cpi_len'])
    cpi_width = int(stats['cpi_width'])
    off_diag_overlap_ratio = float(stats['off_diag_overlap_ratio'])
    diag_valid_ratio = float(stats['diag_valid_ratio'])

    print(f"  n_keep={n_keep}, cpi_len={cpi_len}, cpi_width={cpi_width}, "
          f"off_diag_overlap_ratio={off_diag_overlap_ratio}, diag_valid_ratio={diag_valid_ratio}")

    eigen_loss_weight = args.eigen_loss_weight if args.eigen_loss_weight is not None else EIGEN_LOSS_WEIGHT_DEFAULT
    global_loss_weight = args.global_loss_weight if args.global_loss_weight is not None else GLOBAL_LOSS_WEIGHT_DEFAULT

    threshold_p95 = args.threshold_p95
    threshold_p99 = args.threshold_p99
    run_summary_path = os.path.join(args.model_dir, 'run_summary.json')
    if (threshold_p95 is None or threshold_p99 is None) and os.path.exists(run_summary_path):
        with open(run_summary_path, 'r') as fh:
            run_summary = json.load(fh)
        if threshold_p95 is None:
            threshold_p95 = run_summary.get('recommended_threshold_p95')
        if threshold_p99 is None:
            threshold_p99 = run_summary.get('recommended_threshold_p99')
        print(f"  Auto-loaded thresholds from run_summary.json: p95={threshold_p95}, p99={threshold_p99}")

    # ------------------------------------------------------------------
    # Step 2: Read the requested region from the NEW data (inference only,
    # nothing here is added to any training corpus)
    # ------------------------------------------------------------------
    print(f"\nOpening {args.l0b_file} ...")
    raw = Raw(hdf5file=str(args.l0b_file))
    raw.parsePolarizations()

    dataset = raw.getRawDataset(args.freq, args.pol)
    total_pulses, total_range = dataset.shape

    p_start = args.pulse_start if args.pulse_start is not None else 0
    p_end = args.pulse_end if args.pulse_end is not None else total_pulses
    r_start = args.range_start if args.range_start is not None else 0
    r_end = args.range_end if args.range_end is not None else total_range

    n_pulse_tiles = (p_end - p_start) // cpi_len
    n_range_tiles = (r_end - r_start) // cpi_width
    p_end = p_start + n_pulse_tiles * cpi_len
    r_end = r_start + n_range_tiles * cpi_width

    print(f"  Region: pulses [{p_start}:{p_end}], range [{r_start}:{r_end}]")
    print(f"  Grid: {n_pulse_tiles} pulse tiles x {n_range_tiles} range tiles "
          f"= {n_pulse_tiles * n_range_tiles} CPI tiles")

    raw_data = read_raw_data_batch(raw, args.freq, args.pol, slice(p_start, p_end), slice(r_start, r_end))
    pulse_indices = np.arange(p_start, p_end)
    range_indices = np.arange(r_start, r_end)
    mask_valid = get_subswath_mask(raw, args.freq, args.pol, pulse_indices, range_indices)

    valid_pct = 100.0 * mask_valid.sum() / mask_valid.size
    print(f"  Subswath valid fraction: {valid_pct:.1f}%")

    # ------------------------------------------------------------------
    # Step 3: Extract features with the SAME extraction settings used
    # at training time (loaded from norm_stats.npz above)
    # ------------------------------------------------------------------
    eigen_feats, global_feats, diag_valid_fracs, tile_pulse_idx, tile_range_idx = tile_and_extract_features(
        raw_data,
        mask_valid=mask_valid,
        cpi_len=cpi_len,
        cpi_width=cpi_width,
        n_keep=n_keep,
        off_diag_overlap_ratio=off_diag_overlap_ratio,
        diag_valid_ratio=diag_valid_ratio,
        min_tile_valid_frac=0.0,
    )
    n_tiles = eigen_feats.shape[0]
    print(f"  Extracted {n_tiles} tiles")

    if n_tiles != n_pulse_tiles * n_range_tiles:
        print(f"  NOTE: tile count ({n_tiles}) != full grid ({n_pulse_tiles * n_range_tiles}); "
              f"the score map below assumes no tiles were dropped (min_tile_valid_frac=0.0 keeps them all).")

    # Convert tile start indices to absolute file coordinates
    tile_pulse_idx_abs = tile_pulse_idx + p_start
    tile_range_idx_abs = tile_range_idx + r_start

    # ------------------------------------------------------------------
    # Step 4: Normalize using the TRAINING stats (never refit here) and score
    # ------------------------------------------------------------------
    eigen_norm = ((eigen_feats - eigen_mean) / eigen_std).astype(np.float32)
    global_norm = ((global_feats - global_mean) / global_std).astype(np.float32)

    eigen_err, global_err, total_score = compute_anomaly_scores(
        model, eigen_norm, global_norm,
        eigen_weight=eigen_loss_weight, global_weight=global_loss_weight,
    )

    # ------------------------------------------------------------------
    # Step 5: Report percentiles and threshold pass rates
    # ------------------------------------------------------------------
    percentiles = [50, 75, 90, 95, 99, 99.9]
    pct_vals = np.percentile(total_score, percentiles)
    print("\nAnomaly score percentiles (new data):")
    for p, v in zip(percentiles, pct_vals):
        print(f"  {p}th percentile: {v:.6f}")

    n_above_p95 = int(np.sum(total_score > threshold_p95)) if threshold_p95 is not None else None
    n_above_p99 = int(np.sum(total_score > threshold_p99)) if threshold_p99 is not None else None

    if threshold_p95 is not None:
        print(f"\nTiles above train p95 threshold ({threshold_p95:.6f}): "
              f"{n_above_p95} / {n_tiles} ({100.0 * n_above_p95 / n_tiles:.2f}%)")
    if threshold_p99 is not None:
        print(f"Tiles above train p99 threshold ({threshold_p99:.6f}): "
              f"{n_above_p99} / {n_tiles} ({100.0 * n_above_p99 / n_tiles:.2f}%)")

    # ------------------------------------------------------------------
    # Step 6: Save plots
    # ------------------------------------------------------------------
    save_score_hist_png(total_score, threshold_p95, threshold_p99, args.output_dir)

    if n_tiles == n_pulse_tiles * n_range_tiles:
        save_score_map_png(total_score, n_pulse_tiles, n_range_tiles, args.output_dir)
    else:
        print("  Skipping score map (tile count does not match full grid)")

    # ------------------------------------------------------------------
    # Step 7: Save raw scores and top-N anomalies for follow-up inspection
    # ------------------------------------------------------------------
    scores_path = os.path.join(args.output_dir, 'anomaly_scores.npz')
    np.savez(
        scores_path,
        eigen_err=eigen_err,
        global_err=global_err,
        total_score=total_score,
        tile_pulse_idx=tile_pulse_idx_abs,
        tile_range_idx=tile_range_idx_abs,
        diag_valid_frac=diag_valid_fracs,
    )
    print(f"\nSaved per-tile scores to {scores_path}")

    top_idx = np.argsort(total_score)[::-1][:args.n_top]
    top_anomalies = [
        {
            'rank': int(rank + 1),
            'pulse_start': int(tile_pulse_idx_abs[i]),
            'range_start': int(tile_range_idx_abs[i]),
            'total_score': float(total_score[i]),
            'eigen_err': float(eigen_err[i]),
            'global_err': float(global_err[i]),
            'diag_valid_frac': float(diag_valid_fracs[i]),
        }
        for rank, i in enumerate(top_idx)
    ]
    top_path = os.path.join(args.output_dir, 'top_anomalies.json')
    with open(top_path, 'w') as fh:
        json.dump(top_anomalies, fh, indent=2)
    print(f"Saved top {args.n_top} anomalies to {top_path}")
    print("  (feed pulse_start/range_start values into plot_random_cpi.py via a "
          "--region-manifest to visually inspect the highest-scoring tiles)")

    # ------------------------------------------------------------------
    # Step 8: Summary
    # ------------------------------------------------------------------
    summary = {
        'l0b_file': args.l0b_file,
        'freq': args.freq,
        'pol': args.pol,
        'pulse_start': p_start,
        'pulse_end': p_end,
        'range_start': r_start,
        'range_end': r_end,
        'n_tiles': int(n_tiles),
        'model_dir': args.model_dir,
        'eigen_loss_weight': eigen_loss_weight,
        'global_loss_weight': global_loss_weight,
        'threshold_p95': threshold_p95,
        'threshold_p99': threshold_p99,
        'n_above_p95': n_above_p95,
        'n_above_p99': n_above_p99,
        'percentiles': {str(p): float(v) for p, v in zip(percentiles, pct_vals)},
    }
    summary_path = os.path.join(args.output_dir, 'score_summary.json')
    with open(summary_path, 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved score summary to {summary_path}")


if __name__ == '__main__':
    main()
