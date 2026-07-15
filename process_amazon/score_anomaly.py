"""
score_anomaly.py

Inference-only script: scores a region of NEW L0B data (not used in training)
against an already-trained anomaly autoencoder, using the exact normalization
stats saved at training time.

This does NOT retrain or fine-tune the model in any way -- it only reads
best_model.keras and norm_stats.npz and runs forward passes.

Polarization handling (default: HH and HV, two separate maps)
------------------------------------------------------------------
By default this script scores BOTH HH and HV (--pol HH HV) using the same
shared model and normalization stats (the model was trained on both pooled
together, see train_anomaly.py). Each polarization gets its OWN set of
output files (histogram, score map, top-anomalies list) so an anomaly
visible in only one receive chain is not washed out by pooling -- RFI
couples independently per polarization, so HH and HV are reported
separately even though a single shared model scores both.

Plot conventions (read this before interpreting the outputs)
------------------------------------------------------------------
- LOW anomaly score = clean-like (reconstructs well, close to the training
  distribution). HIGH anomaly score = anomalous (reconstructs poorly,
  unlike anything in the clean training corpus). This holds for both the
  histogram (right tail = anomalous) and the score map (bright/high-value
  cells = anomalous, using the 'inferno' colormap where dark = low score).
- The score map is PER-CPI-TILE, not per-pixel: each cell in
  anomaly_score_map_<pol>.png represents one whole 16x250 CPI tile, not a
  single range/azimuth sample.
- Pulse index 0 is at the TOP of every 2D plot (matches the convention used
  throughout this project), so "pulse tile index 0" (top row of the score
  map) corresponds to the first CPI in the requested pulse range.

Usage
-----
    # Default: score both HH and HV, two separate maps
    python score_anomaly.py granule.h5 --freq A \
        --pulse-start 888222 --pulse-end 896222 \
        --range-start 2000 --range-end 25000 \
        --model-dir models/anomaly_v1 \
        --output-dir eval/next_8000

    # Single polarization only
    python score_anomaly.py granule.h5 --freq A --pol HV \
        --pulse-start 888222 --pulse-end 896222 \
        --range-start 2000 --range-end 25000 \
        --model-dir models/anomaly_v1 \
        --output-dir eval/next_8000_hv_only

By default, --model-dir is expected to contain:
    best_model.keras
    norm_stats.npz
and, if present, run_summary.json is used to auto-populate the recommended
p95/p99 anomaly thresholds for reporting (override with --threshold-p95 /
--threshold-p99).

Outputs (in --output-dir), one set per polarization scored
------------------------------------------------------------
    anomaly_scores_<pol>.npz     -- per-tile eigen_err, global_err,
                                     total_score, tile_pulse_idx,
                                     tile_range_idx (absolute file
                                     coordinates)
    anomaly_score_hist_<pol>.png -- histogram of total_score, low=clean,
                                     high=anomalous, with threshold lines
    anomaly_score_map_<pol>.png  -- 2D heatmap of total_score over the
                                     (pulse_tile x range_tile) grid, one
                                     cell per CPI tile, pulse 0 at top
    top_anomalies_<pol>.json     -- the --n-top highest-scoring tiles for
                                     this polarization, with absolute
                                     pulse_start/range_start for follow-up
                                     inspection via plot_random_cpi.py
    score_summary.json           -- percentile summary and threshold pass
                                     rates for every polarization scored
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

def save_score_hist_png(total_score, threshold_p95, threshold_p99, pol, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(total_score, bins=80, alpha=0.8, color='tab:blue')
    if threshold_p95 is not None:
        ax.axvline(threshold_p95, color='orange', linestyle='--', label=f'Train p95 = {threshold_p95:.4f}')
    if threshold_p99 is not None:
        ax.axvline(threshold_p99, color='red', linestyle='--', label=f'Train p99 = {threshold_p99:.4f}')
    ax.set_xlabel('Total anomaly score  (low = clean-like  -->  high = anomalous)')
    ax.set_ylabel('Tile count (one 16x250 CPI tile per count)')
    ax.set_title(f'Anomaly Score Distribution - {pol} (New Data)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    path = os.path.join(out_dir, f'anomaly_score_hist_{pol}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def save_score_map_png(total_score, n_pulse_tiles, n_range_tiles, pol, out_dir):
    """
    Plot the per-CPI-tile anomaly score grid.

    Each cell is one whole 16x250 CPI tile, not a single range/azimuth
    pixel. Row 0 (top of the image) is pulse tile index 0, i.e. the first
    CPI in the requested pulse range (pulse 0 at top, matching the
    convention used throughout this project). Colormap is 'inferno': dark
    cells = low score = clean-like, bright cells = high score = anomalous.
    """
    score_grid = total_score.reshape(n_pulse_tiles, n_range_tiles)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(score_grid, aspect='auto', origin='upper', cmap='inferno')
    ax.set_xlabel('Range Tile Index (one cell = one 250-sample-wide CPI tile)')
    ax.set_ylabel('Pulse Tile Index (one cell = one 16-pulse CPI; pulse 0 at top)')
    ax.set_title(f'Anomaly Score Map - {pol} (per CPI tile, dark=clean-like, bright=anomalous)')
    fig.colorbar(im, ax=ax, label='Anomaly score (low=clean-like, high=anomalous)')

    fig.tight_layout()
    path = os.path.join(out_dir, f'anomaly_score_map_{pol}.png')
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
    parser.add_argument('--pol', choices=['HH', 'HV', 'VH', 'VV'], nargs='+', default=['HH', 'HV'],
                        help='Polarization(s) to score (default: HH HV, each gets its own set of output '
                             'files). Pass a single value (e.g. --pol HV) to score one polarization only.')

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


def score_one_polarization(
    raw: Raw,
    l0b_file: str,
    freq: str,
    pol: str,
    model,
    norm_stats: dict,
    p_start: int,
    p_end: int,
    r_start: int,
    r_end: int,
    eigen_loss_weight: float,
    global_loss_weight: float,
    threshold_p95,
    threshold_p99,
    n_top: int,
    output_dir: str,
):
    """
    Extract, normalize, and score one polarization's tiles over the
    requested region, saving that polarization's own set of output files.

    Returns
    -------
    summary : dict
        Per-polarization percentile/threshold summary, used to build the
        combined score_summary.json across all polarizations scored.
    """
    cpi_len = norm_stats['cpi_len']
    cpi_width = norm_stats['cpi_width']
    n_keep = norm_stats['n_keep']
    off_diag_overlap_ratio = norm_stats['off_diag_overlap_ratio']
    diag_valid_ratio = norm_stats['diag_valid_ratio']
    eigen_mean, eigen_std = norm_stats['eigen_mean'], norm_stats['eigen_std']
    global_mean, global_std = norm_stats['global_mean'], norm_stats['global_std']

    n_pulse_tiles = (p_end - p_start) // cpi_len
    n_range_tiles = (r_end - r_start) // cpi_width

    print(f"\n--- Scoring {freq}-{pol} ---")
    print(f"  Region: pulses [{p_start}:{p_end}], range [{r_start}:{r_end}]")
    print(f"  Grid: {n_pulse_tiles} pulse tiles x {n_range_tiles} range tiles "
          f"= {n_pulse_tiles * n_range_tiles} CPI tiles")

    raw_data = read_raw_data_batch(raw, freq, pol, slice(p_start, p_end), slice(r_start, r_end))
    pulse_indices = np.arange(p_start, p_end)
    range_indices = np.arange(r_start, r_end)
    mask_valid = get_subswath_mask(raw, freq, pol, pulse_indices, range_indices)

    valid_pct = 100.0 * mask_valid.sum() / mask_valid.size
    print(f"  Subswath valid fraction: {valid_pct:.1f}%")

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
              f"the score map will be skipped for this polarization.")

    tile_pulse_idx_abs = tile_pulse_idx + p_start
    tile_range_idx_abs = tile_range_idx + r_start

    eigen_norm = ((eigen_feats - eigen_mean) / eigen_std).astype(np.float32)
    global_norm = ((global_feats - global_mean) / global_std).astype(np.float32)

    eigen_err, global_err, total_score = compute_anomaly_scores(
        model, eigen_norm, global_norm,
        eigen_weight=eigen_loss_weight, global_weight=global_loss_weight,
    )

    percentiles = [50, 75, 90, 95, 99, 99.9]
    pct_vals = np.percentile(total_score, percentiles)
    print(f"  Anomaly score percentiles ({pol}):")
    for p, v in zip(percentiles, pct_vals):
        print(f"    {p}th percentile: {v:.6f}")

    n_above_p95 = int(np.sum(total_score > threshold_p95)) if threshold_p95 is not None else None
    n_above_p99 = int(np.sum(total_score > threshold_p99)) if threshold_p99 is not None else None

    if threshold_p95 is not None:
        print(f"  Tiles above train p95 threshold ({threshold_p95:.6f}): "
              f"{n_above_p95} / {n_tiles} ({100.0 * n_above_p95 / n_tiles:.2f}%)")
    if threshold_p99 is not None:
        print(f"  Tiles above train p99 threshold ({threshold_p99:.6f}): "
              f"{n_above_p99} / {n_tiles} ({100.0 * n_above_p99 / n_tiles:.2f}%)")

    save_score_hist_png(total_score, threshold_p95, threshold_p99, pol, output_dir)

    if n_tiles == n_pulse_tiles * n_range_tiles:
        save_score_map_png(total_score, n_pulse_tiles, n_range_tiles, pol, output_dir)
    else:
        print("  Skipping score map (tile count does not match full grid)")

    scores_path = os.path.join(output_dir, f'anomaly_scores_{pol}.npz')
    np.savez(
        scores_path,
        eigen_err=eigen_err,
        global_err=global_err,
        total_score=total_score,
        tile_pulse_idx=tile_pulse_idx_abs,
        tile_range_idx=tile_range_idx_abs,
        diag_valid_frac=diag_valid_fracs,
    )
    print(f"  Saved per-tile scores to {scores_path}")

    top_idx = np.argsort(total_score)[::-1][:n_top]
    top_anomalies = [
        {
            'rank': int(rank + 1),
            'pol': pol,
            'pulse_start': int(tile_pulse_idx_abs[i]),
            'range_start': int(tile_range_idx_abs[i]),
            'total_score': float(total_score[i]),
            'eigen_err': float(eigen_err[i]),
            'global_err': float(global_err[i]),
            'diag_valid_frac': float(diag_valid_fracs[i]),
        }
        for rank, i in enumerate(top_idx)
    ]
    top_path = os.path.join(output_dir, f'top_anomalies_{pol}.json')
    with open(top_path, 'w') as fh:
        json.dump(top_anomalies, fh, indent=2)
    print(f"  Saved top {n_top} anomalies to {top_path}")

    return {
        'pol': pol,
        'pulse_start': p_start,
        'pulse_end': p_end,
        'range_start': r_start,
        'range_end': r_end,
        'n_tiles': int(n_tiles),
        'threshold_p95': threshold_p95,
        'threshold_p99': threshold_p99,
        'n_above_p95': n_above_p95,
        'n_above_p99': n_above_p99,
        'percentiles': {str(p): float(v) for p, v in zip(percentiles, pct_vals)},
    }


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
    raw_stats = np.load(norm_stats_path)
    norm_stats = {
        'eigen_mean': raw_stats['eigen_mean'],
        'eigen_std': raw_stats['eigen_std'],
        'global_mean': raw_stats['global_mean'],
        'global_std': raw_stats['global_std'],
        'n_keep': int(raw_stats['n_keep']),
        'cpi_len': int(raw_stats['cpi_len']),
        'cpi_width': int(raw_stats['cpi_width']),
        'off_diag_overlap_ratio': float(raw_stats['off_diag_overlap_ratio']),
        'diag_valid_ratio': float(raw_stats['diag_valid_ratio']),
    }

    print(f"  n_keep={norm_stats['n_keep']}, cpi_len={norm_stats['cpi_len']}, "
          f"cpi_width={norm_stats['cpi_width']}, "
          f"off_diag_overlap_ratio={norm_stats['off_diag_overlap_ratio']}, "
          f"diag_valid_ratio={norm_stats['diag_valid_ratio']}")

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

    print(f"\nPolarizations to score (each gets its own map): {args.pol}")

    # ------------------------------------------------------------------
    # Step 2: Open the file once and score each requested polarization
    # over the same region (inference only, nothing here is added to any
    # training corpus)
    # ------------------------------------------------------------------
    print(f"\nOpening {args.l0b_file} ...")
    raw = Raw(hdf5file=str(args.l0b_file))
    raw.parsePolarizations()

    # Resolve the region once using the first polarization's dataset shape
    # as the fallback for "full extent" (HH/HV share the same pulse/range
    # indexing for a given file).
    dataset0 = raw.getRawDataset(args.freq, args.pol[0])
    total_pulses, total_range = dataset0.shape

    p_start = args.pulse_start if args.pulse_start is not None else 0
    p_end = args.pulse_end if args.pulse_end is not None else total_pulses
    r_start = args.range_start if args.range_start is not None else 0
    r_end = args.range_end if args.range_end is not None else total_range

    cpi_len = norm_stats['cpi_len']
    cpi_width = norm_stats['cpi_width']
    n_pulse_tiles = (p_end - p_start) // cpi_len
    n_range_tiles = (r_end - r_start) // cpi_width
    p_end = p_start + n_pulse_tiles * cpi_len
    r_end = r_start + n_range_tiles * cpi_width

    per_pol_summaries = {}
    for pol in args.pol:
        per_pol_summaries[pol] = score_one_polarization(
            raw, args.l0b_file, args.freq, pol, model, norm_stats,
            p_start, p_end, r_start, r_end,
            eigen_loss_weight, global_loss_weight,
            threshold_p95, threshold_p99, args.n_top, args.output_dir,
        )

    # ------------------------------------------------------------------
    # Step 3: Combined summary across all polarizations scored
    # ------------------------------------------------------------------
    summary = {
        'l0b_file': args.l0b_file,
        'freq': args.freq,
        'pol': args.pol,
        'model_dir': args.model_dir,
        'eigen_loss_weight': eigen_loss_weight,
        'global_loss_weight': global_loss_weight,
        'plot_conventions': {
            'score_direction': 'low = clean-like, high = anomalous',
            'score_map_resolution': 'per CPI tile (16 pulses x 250 range samples), not per pixel',
            'score_map_orientation': 'pulse index 0 at top',
        },
        'per_polarization': per_pol_summaries,
    }
    summary_path = os.path.join(args.output_dir, 'score_summary.json')
    with open(summary_path, 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved combined score summary to {summary_path}")
    print("  (feed pulse_start/range_start values from top_anomalies_<pol>.json into "
          "plot_random_cpi.py via a --region-manifest to visually inspect the highest-scoring tiles)")


if __name__ == '__main__':
    main()