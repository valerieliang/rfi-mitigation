"""
score_anomaly_jsr_sweep.py

Empirically calibrates the anomaly autoencoder's score-to-severity mapping
using REAL background CPI tiles (not idealized synthetic spectra), by
overlaying synthetic RFI bands directly on top of real L0B data and sweeping
the jammer-to-signal ratio (JSR).

Why JSR instead of JNR
------------------------
generate_synthetic_data.py builds fully synthetic images (synthetic noise +
synthetic signal), so RFI strength there is naturally expressed as JNR
(jammer power relative to a fixed synthetic noise floor). Here the
background is REAL L0B data -- there is no separate synthetic noise
component to reference. The only ratio that makes sense is jammer power
relative to the real tile's own observed power, i.e. JSR (jammer-to-signal
ratio), where "signal" means the real tile's own average power over its
valid samples. A JSR of 0 dB means the injected RFI has the same average
power as the real background it's being added to.

Per project convention, JSR is swept starting at a minimum of 3 dB (weaker
injections are considered below the threshold of practical interest for
this study, though the script does not hard-fail if you pass lower values).

RFI injection model (adapted from generate_synthetic_data.py)
------------------------------------------------------------------
For each background CPI tile and each requested band count n_bands:
  1. Estimate the tile's own signal power from its valid (mask=True) samples.
  2. Compute the total injected RFI power as signal_power * 10^(JSR_db/10).
  3. Choose n_bands random pulse rows within the tile (with replacement, so
     two bands may land on the same row and sum incoherently, matching the
     original generator's convention).
  4. Each band gets an independent complex Gaussian range-coefficient vector
     scaled to the per-band power, modulated by a random Doppler phase
     across the tile's local pulse index.
  5. Add the resulting RFI matrix directly onto the real complex CPI tile.

This produces a REAL-background, REAL-noise-floor contaminated tile whose
only synthetic component is the injected interference itself -- as close as
this project's synthetic injection approach can get to a true controlled
experiment on real data.

Comparison produced
------------------------------------------------------------------
For a set of randomly sampled real background tiles from the requested
region:
  - "baseline" scores: the same tiles, scored as-is (no injection). This is
    a proxy for "real clean-like data", not verified ground truth -- see
    caveat below.
  - "contaminated" scores: the SAME tiles, with RFI injected at each
    (JSR, n_bands) combination in the sweep, scored again. Because it's the
    same underlying background, this isolates the injected RFI's
    contribution to the score rather than confounding it with tile-to-tile
    scene variation.

CAVEAT: the background tiles are drawn from a region assumed mostly clean
(the same assumption train_anomaly.py's clean-corpus filter relies on), not
independently verified. If a handful of background tiles already contain
real RFI, the "baseline" distribution will be slightly contaminated by that,
which would make the calibration slightly conservative (baseline scores
would be a bit higher than a truly clean baseline).

Usage
-----
    python score_anomaly_jsr_sweep.py granule.h5 --freq A \
        --pulse-start 888222 --pulse-end 896222 \
        --range-start 2000 --range-end 25000 \
        --model-dir models/anomaly_v1 \
        --output-dir eval/jsr_sweep \
        --jsr-db-list 3 6 10 15 20 25 30 \
        --n-bands-list 1 2

Outputs (in --output-dir)
--------------------------
    anomaly_score_map_<pol>.png -- full-region spatial anomaly score map
                                   AFTER synthetic RFI has been injected
                                   into every tile at a single severity
                                   (--map-jsr-db / --map-n-bands, default:
                                   the weakest values in the sweep). Same
                                   visual convention as score_anomaly.py's
                                   map: one cell per CPI tile, pulse 0 at
                                   top, dark=clean-like, bright=anomalous.
    jsr_sweep_scores.npz       -- raw per-tile scores: baseline and every
                                   (pol, jsr_db, n_bands) combination
    jsr_sweep_curve_<pol>.png  -- score vs JSR, one line per n_bands, with
                                   IQR shading and baseline reference lines
    jsr_sweep_hist_<pol>.png   -- baseline vs weakest-JSR contaminated score
                                   histogram (the hardest detection case)
    jsr_sweep_summary.json     -- percentile tables and threshold
                                   recommendations (see below)
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from anomaly_features import extract_anomaly_features, tile_and_extract_features
from model_anomaly import compute_anomaly_scores

from nisar.products.readers.Raw import Raw

EIGEN_LOSS_WEIGHT_DEFAULT = 1.0
GLOBAL_LOSS_WEIGHT_DEFAULT = 0.5

JSR_DB_LIST_DEFAULT = [3, 6, 10, 15, 20, 25, 30]
JSR_MIN_DB = 3.0
N_BANDS_LIST_DEFAULT = [1]
N_BACKGROUND_TILES_DEFAULT = 300

EPS = 1e-12


# ---------------------------------------------------------------------------
# ISCE3 RAW DATA / SUBSWATH MASK HELPERS (self-contained, matches
# train_anomaly.py / score_anomaly.py / plot_random_cpi.py)
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
# RFI INJECTION (JSR-parameterized, adapted from generate_synthetic_data.py)
# ---------------------------------------------------------------------------

def inject_rfi_bands(cpi_data: np.ndarray, cpi_mask: np.ndarray, jsr_db: float, n_bands: int, rng: np.random.Generator):
    """
    Overlay n_bands synthetic RFI bands onto a real CPI tile at the
    requested jammer-to-signal ratio.

    Parameters
    ----------
    cpi_data : (M, K) complex array
        Real CPI tile (the "signal" whose own power defines JSR).
    cpi_mask : (M, K) bool array
        Valid-sample mask; only valid samples are used to estimate the
        tile's own signal power (so gap/dropout regions don't bias it).
    jsr_db : float
        Jammer-to-signal ratio in dB: injected RFI power relative to the
        tile's own average power.
    n_bands : int
        Number of independent RFI bands to inject (pulse rows chosen with
        replacement, so bands may coincide and sum incoherently).
    rng : np.random.Generator

    Returns
    -------
    contaminated : (M, K) complex64 array
        cpi_data with the RFI matrix added on top.
    affected_rows : list[int]
        Local pulse row indices that received at least one band (for
        diagnostics; duplicates collapsed).
    """
    M, K = cpi_data.shape

    valid = cpi_mask if cpi_mask is not None and cpi_mask.any() else np.ones_like(cpi_data, dtype=bool)
    signal_power_linear = float(np.mean(np.abs(cpi_data[valid]) ** 2))
    signal_power_linear = max(signal_power_linear, EPS)

    rfi_power_linear = signal_power_linear * (10.0 ** (jsr_db / 10.0))
    sigma = np.sqrt(rfi_power_linear / 2.0)

    rfi_matrix = np.zeros((M, K), dtype=np.complex64)
    local_indices = rng.integers(0, M, size=n_bands)

    for local_idx in local_indices:
        doppler_freq = rng.uniform(-0.5, 0.5)
        range_coeff = (
            rng.standard_normal(K) + 1j * rng.standard_normal(K)
        ) * sigma
        phase = np.exp(1j * 2.0 * np.pi * doppler_freq * local_idx)
        rfi_matrix[local_idx, :] += (phase * range_coeff).astype(np.complex64)

    contaminated = (cpi_data + rfi_matrix).astype(np.complex64)
    affected_rows = sorted(set(int(i) for i in local_indices))
    return contaminated, affected_rows


def inject_rfi_into_full_grid(raw_data, mask_valid, cpi_len, cpi_width, jsr_db, n_bands, rng):
    """
    Inject synthetic RFI into every non-overlapping CPI tile of a full
    (pulses, range) array, at a single (jsr_db, n_bands) severity, so the
    resulting array can be scored and mapped the same way as real data.

    Parameters
    ----------
    raw_data : (n_pulses, n_range) complex array
    mask_valid : (n_pulses, n_range) bool array
    cpi_len, cpi_width : int
    jsr_db : float
    n_bands : int
    rng : np.random.Generator

    Returns
    -------
    contaminated : (n_pulses, n_range) complex64 array
        Copy of raw_data with RFI injected into every tile.
    """
    n_pulses, n_range = raw_data.shape
    n_pulse_tiles = n_pulses // cpi_len
    n_range_tiles = n_range // cpi_width

    contaminated = raw_data.copy()

    for pt in range(n_pulse_tiles):
        p0, p1 = pt * cpi_len, pt * cpi_len + cpi_len
        for rt in range(n_range_tiles):
            r0, r1 = rt * cpi_width, rt * cpi_width + cpi_width
            cpi_data = contaminated[p0:p1, r0:r1]
            cpi_mask = mask_valid[p0:p1, r0:r1]
            contaminated_cpi, _ = inject_rfi_bands(cpi_data, cpi_mask, jsr_db, n_bands, rng)
            contaminated[p0:p1, r0:r1] = contaminated_cpi

    return contaminated


# ---------------------------------------------------------------------------
# BACKGROUND TILE SAMPLING
# ---------------------------------------------------------------------------

def sample_background_tiles(raw_data, mask_valid, cpi_len, cpi_width, n_tiles, rng):
    """
    Randomly sample n_tiles distinct CPI-grid positions from a (pulses,
    range) array and return their raw complex tiles and masks. Memory-cheap
    even for large regions: only the sampled tiles themselves are extracted,
    not the whole tile grid.

    Returns
    -------
    tiles : list[(cpi_data, cpi_mask, tile_pulse_start, tile_range_start)]
    """
    n_pulses, n_range = raw_data.shape
    n_pulse_tiles = n_pulses // cpi_len
    n_range_tiles = n_range // cpi_width
    n_available = n_pulse_tiles * n_range_tiles

    n_pick = min(n_tiles, n_available)
    flat_choices = rng.choice(n_available, size=n_pick, replace=False)

    tiles = []
    for flat_idx in flat_choices:
        pt = int(flat_idx // n_range_tiles)
        rt = int(flat_idx % n_range_tiles)
        p0, p1 = pt * cpi_len, pt * cpi_len + cpi_len
        r0, r1 = rt * cpi_width, rt * cpi_width + cpi_width

        cpi_data = raw_data[p0:p1, r0:r1]
        cpi_mask = mask_valid[p0:p1, r0:r1]
        tiles.append((cpi_data, cpi_mask, p0, r0))

    return tiles


def score_tiles(tiles, model, norm_stats, eigen_loss_weight, global_loss_weight):
    """
    Extract features and score a list of (cpi_data, cpi_mask, ...) tuples.

    Returns
    -------
    scores : (N,) float32 array
    """
    n_keep = norm_stats['n_keep']
    off_diag_overlap_ratio = norm_stats['off_diag_overlap_ratio']
    diag_valid_ratio = norm_stats['diag_valid_ratio']
    eigen_mean, eigen_std = norm_stats['eigen_mean'], norm_stats['eigen_std']
    global_mean, global_std = norm_stats['global_mean'], norm_stats['global_std']

    eigen_list, global_list = [], []
    for cpi_data, cpi_mask, *_ in tiles:
        eigen_feat, global_feat, _, _ = extract_anomaly_features(
            cpi_data,
            mask_valid_cpi=cpi_mask,
            n_keep=n_keep,
            off_diag_overlap_ratio=off_diag_overlap_ratio,
            diag_valid_ratio=diag_valid_ratio,
        )
        eigen_list.append(eigen_feat)
        global_list.append(global_feat)

    eigen_arr = np.stack(eigen_list).astype(np.float32)
    global_arr = np.stack(global_list).astype(np.float32)

    eigen_norm = ((eigen_arr - eigen_mean) / eigen_std).astype(np.float32)
    global_norm = ((global_arr - global_mean) / global_std).astype(np.float32)

    _, _, total_score = compute_anomaly_scores(
        model, eigen_norm, global_norm,
        eigen_weight=eigen_loss_weight, global_weight=global_loss_weight,
    )
    return total_score


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------

def save_score_map_png(total_score, n_pulse_tiles, n_range_tiles, pol, jsr_db, n_bands, out_dir):
    """
    Full-grid anomaly score map AFTER synthetic RFI has been injected into
    every tile at the given (jsr_db, n_bands) severity -- same visual
    convention as score_anomaly.py's map so it can be read the same way:
    each cell = one 16x250 CPI tile (not a pixel), pulse index 0 at top,
    dark (inferno colormap) = low score = clean-like, bright = high score
    = anomalous.
    """
    score_grid = total_score.reshape(n_pulse_tiles, n_range_tiles)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(score_grid, aspect='auto', origin='upper', cmap='inferno')
    ax.set_xlabel('Range Tile Index (one cell = one 250-sample-wide CPI tile)')
    ax.set_ylabel('Pulse Tile Index (one cell = one 16-pulse CPI; pulse 0 at top)')
    ax.set_title(
        f'Anomaly Score Map - {pol} (AFTER injecting JSR={jsr_db} dB, n_bands={n_bands} '
        f'into every tile; dark=clean-like, bright=anomalous)'
    )
    fig.colorbar(im, ax=ax, label='Anomaly score (low=clean-like, high=anomalous)')

    fig.tight_layout()
    path = os.path.join(out_dir, f'anomaly_score_map_{pol}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def save_jsr_curve_png(jsr_list, n_bands_list, contaminated_scores, baseline_scores, threshold_p95, threshold_p99, pol, out_dir):
    """
    Plot median contaminated score vs JSR (one line per n_bands), with
    25th-75th percentile shading, against baseline (no-injection) reference
    lines. Log scale on the score axis given the wide dynamic range.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    baseline_median = float(np.median(baseline_scores))
    ax.axhline(baseline_median, color='gray', linestyle='-', alpha=0.7, label=f'Baseline median = {baseline_median:.4f}')
    if threshold_p95 is not None:
        ax.axhline(threshold_p95, color='orange', linestyle='--', alpha=0.7, label=f'Train p95 = {threshold_p95:.4f}')
    if threshold_p99 is not None:
        ax.axhline(threshold_p99, color='red', linestyle='--', alpha=0.7, label=f'Train p99 = {threshold_p99:.4f}')

    for n_bands in n_bands_list:
        medians, p25s, p75s = [], [], []
        for jsr_db in jsr_list:
            scores = contaminated_scores[(n_bands, jsr_db)]
            medians.append(np.median(scores))
            p25s.append(np.percentile(scores, 25))
            p75s.append(np.percentile(scores, 75))
        medians, p25s, p75s = np.array(medians), np.array(p25s), np.array(p75s)

        ax.plot(jsr_list, medians, marker='o', label=f'n_bands={n_bands} (median)')
        ax.fill_between(jsr_list, p25s, p75s, alpha=0.2)

    ax.set_yscale('log')
    ax.set_xlabel('JSR (dB) -- injected RFI power relative to real tile signal power')
    ax.set_ylabel('Anomaly score (log scale, low=clean-like, high=anomalous)')
    ax.set_title(f'Anomaly Score vs JSR - {pol} (real background tiles, synthetic RFI overlay)')
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.4)

    fig.tight_layout()
    path = os.path.join(out_dir, f'jsr_sweep_curve_{pol}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def save_weakest_jsr_hist_png(baseline_scores, weakest_scores, weakest_jsr_db, weakest_n_bands, pol, out_dir):
    """
    Overlaid histogram: baseline (no injection) vs the weakest contaminated
    case in the sweep -- the hardest detection scenario, so this shows the
    worst-case separation you can expect.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    all_scores = np.concatenate([baseline_scores, weakest_scores])
    bins = np.logspace(np.log10(max(all_scores.min(), 1e-6)), np.log10(all_scores.max() + 1e-6), 60)

    ax.hist(baseline_scores, bins=bins, alpha=0.6, label='Baseline (real, no injection)', color='tab:blue')
    ax.hist(weakest_scores, bins=bins, alpha=0.6,
            label=f'Contaminated (JSR={weakest_jsr_db} dB, n_bands={weakest_n_bands})', color='tab:red')
    ax.set_xscale('log')
    ax.set_xlabel('Anomaly score (log scale, low=clean-like, high=anomalous)')
    ax.set_ylabel('Tile count')
    ax.set_title(f'Baseline vs Weakest-JSR Contaminated - {pol} (hardest detection case)')
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.4)

    fig.tight_layout()
    path = os.path.join(out_dir, f'jsr_sweep_hist_{pol}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# THRESHOLD ANALYSIS
# ---------------------------------------------------------------------------

def threshold_tradeoff(baseline_scores, contaminated_scores, candidate_thresholds):
    """
    For each candidate threshold, compute the false-positive rate on the
    baseline (clean) distribution and the detection rate on the
    contaminated distribution.

    Returns
    -------
    rows : list[dict]
    """
    rows = []
    for t in candidate_thresholds:
        fpr = float(np.mean(baseline_scores > t))
        tpr = float(np.mean(contaminated_scores > t))
        rows.append({'threshold': float(t), 'false_positive_rate': fpr, 'detection_rate': tpr})
    return rows


def threshold_for_detection_rate(contaminated_scores, target_rate):
    """
    Smallest threshold such that at least target_rate fraction of
    contaminated_scores exceed it (i.e. the (1 - target_rate) percentile).
    """
    pct = (1.0 - target_rate) * 100.0
    return float(np.percentile(contaminated_scores, pct))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Calibrate the anomaly autoencoder threshold using real backgrounds with synthetic RFI overlaid at swept JSR levels',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('l0b_file', help='Input NISAR L0B HDF5 file')
    parser.add_argument('--freq', choices=['A', 'B'], default='A')
    parser.add_argument('--pol', choices=['HH', 'HV', 'VH', 'VV'], nargs='+', default=['HH', 'HV'],
                        help='Polarization(s) to run the sweep for (default: HH HV)')

    parser.add_argument('--pulse-start', type=int, default=None)
    parser.add_argument('--pulse-end', type=int, default=None)
    parser.add_argument('--range-start', type=int, default=None)
    parser.add_argument('--range-end', type=int, default=None)

    parser.add_argument('--model-dir', type=str, required=True,
                        help='Directory containing best_model.keras and norm_stats.npz from training')
    parser.add_argument('--output-dir', type=str, default='eval/jsr_sweep', help='Output directory')

    parser.add_argument('--jsr-db-list', type=float, nargs='+', default=JSR_DB_LIST_DEFAULT,
                        help=f'JSR sweep values in dB (default: {JSR_DB_LIST_DEFAULT}). Values below '
                             f'{JSR_MIN_DB} dB are allowed but are below this study\'s minimum JSR of interest.')
    parser.add_argument('--n-bands-list', type=int, nargs='+', default=N_BANDS_LIST_DEFAULT,
                        help=f'Number of independent RFI bands to inject per tile (default: {N_BANDS_LIST_DEFAULT})')
    parser.add_argument('--n-background-tiles', type=int, default=N_BACKGROUND_TILES_DEFAULT,
                        help=f'Number of real background tiles to sample for the sweep (default: {N_BACKGROUND_TILES_DEFAULT})')

    parser.add_argument('--map-jsr-db', type=float, default=None,
                        help='JSR (dB) used to inject RFI into every tile for the full-grid score map '
                             '(default: the weakest/minimum value in --jsr-db-list)')
    parser.add_argument('--map-n-bands', type=int, default=None,
                        help='Number of RFI bands used to inject into every tile for the full-grid score map '
                             '(default: the weakest/minimum value in --n-bands-list)')

    parser.add_argument('--eigen-loss-weight', type=float, default=None)
    parser.add_argument('--global-loss-weight', type=float, default=None)
    parser.add_argument('--threshold-p95', type=float, default=None,
                        help='Overrides auto-loaded p95 threshold from run_summary.json')
    parser.add_argument('--threshold-p99', type=float, default=None,
                        help='Overrides auto-loaded p99 threshold from run_summary.json')

    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    if min(args.jsr_db_list) < JSR_MIN_DB:
        print(f"WARNING: --jsr-db-list contains values below the study minimum of {JSR_MIN_DB} dB; "
              f"these represent RFI weaker than this project's threshold of interest.")

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
    cpi_len = norm_stats['cpi_len']
    cpi_width = norm_stats['cpi_width']

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

    print(f"\nJSR sweep: {args.jsr_db_list} dB")
    print(f"n_bands sweep: {args.n_bands_list}")
    print(f"Background tiles per polarization: {args.n_background_tiles}")

    # ------------------------------------------------------------------
    # Step 2: Open the file, resolve region
    # ------------------------------------------------------------------
    print(f"\nOpening {args.l0b_file} ...")
    raw = Raw(hdf5file=str(args.l0b_file))
    raw.parsePolarizations()

    dataset0 = raw.getRawDataset(args.freq, args.pol[0])
    total_pulses, total_range = dataset0.shape

    p_start = args.pulse_start if args.pulse_start is not None else 0
    p_end = args.pulse_end if args.pulse_end is not None else total_pulses
    r_start = args.range_start if args.range_start is not None else 0
    r_end = args.range_end if args.range_end is not None else total_range

    n_pulse_tiles = (p_end - p_start) // cpi_len
    n_range_tiles = (r_end - r_start) // cpi_width
    p_end = p_start + n_pulse_tiles * cpi_len
    r_end = r_start + n_range_tiles * cpi_width

    print(f"Region: pulses [{p_start}:{p_end}], range [{r_start}:{r_end}]")

    all_summaries = {}
    all_raw_scores = {}

    for pol in args.pol:
        print(f"\n--- Polarization {pol} ---")
        raw_data = read_raw_data_batch(raw, args.freq, pol, slice(p_start, p_end), slice(r_start, r_end))
        pulse_indices = np.arange(p_start, p_end)
        range_indices = np.arange(r_start, r_end)
        mask_valid = get_subswath_mask(raw, args.freq, pol, pulse_indices, range_indices)

        # ------------------------------------------------------------------
        # Step 3a: Full-grid scoring AFTER injecting synthetic RFI into
        # every tile at the map's chosen severity, to produce a spatial map
        # showing what widespread contamination at this severity would look
        # like -- gives spatial context alongside the JSR calibration
        # curves below.
        # ------------------------------------------------------------------
        map_jsr_db = args.map_jsr_db if args.map_jsr_db is not None else min(args.jsr_db_list)
        map_n_bands = args.map_n_bands if args.map_n_bands is not None else min(args.n_bands_list)
        print(f"  Injecting RFI into every tile for the map (JSR={map_jsr_db} dB, n_bands={map_n_bands}) ...")

        contaminated_raw_data = inject_rfi_into_full_grid(
            raw_data, mask_valid, cpi_len, cpi_width, map_jsr_db, map_n_bands, rng,
        )

        full_eigen, full_global, _, _, _ = tile_and_extract_features(
            contaminated_raw_data,
            mask_valid=mask_valid,
            cpi_len=cpi_len,
            cpi_width=cpi_width,
            n_keep=norm_stats['n_keep'],
            off_diag_overlap_ratio=norm_stats['off_diag_overlap_ratio'],
            diag_valid_ratio=norm_stats['diag_valid_ratio'],
            min_tile_valid_frac=0.0,
        )
        n_full_tiles = full_eigen.shape[0]

        if n_full_tiles == n_pulse_tiles * n_range_tiles:
            full_eigen_norm = ((full_eigen - norm_stats['eigen_mean']) / norm_stats['eigen_std']).astype(np.float32)
            full_global_norm = ((full_global - norm_stats['global_mean']) / norm_stats['global_std']).astype(np.float32)
            _, _, full_grid_scores = compute_anomaly_scores(
                model, full_eigen_norm, full_global_norm,
                eigen_weight=eigen_loss_weight, global_weight=global_loss_weight,
            )
            save_score_map_png(full_grid_scores, n_pulse_tiles, n_range_tiles, pol, map_jsr_db, map_n_bands, args.output_dir)
        else:
            print(f"  NOTE: full-grid tile count ({n_full_tiles}) != expected grid "
                  f"({n_pulse_tiles * n_range_tiles}); skipping score map for {pol}")

        # ------------------------------------------------------------------
        # Step 3b: Sample real background tiles for the JSR sweep (same set
        # reused for every JSR/n_bands combination, for a fair paired
        # comparison)
        # ------------------------------------------------------------------
        background_tiles = sample_background_tiles(
            raw_data, mask_valid, cpi_len, cpi_width, args.n_background_tiles, rng,
        )
        n_bg = len(background_tiles)
        print(f"  Sampled {n_bg} background tiles")

        # ------------------------------------------------------------------
        # Step 4: Baseline scores (no injection)
        # ------------------------------------------------------------------
        baseline_scores = score_tiles(background_tiles, model, norm_stats, eigen_loss_weight, global_loss_weight)
        print(f"  Baseline scores: median={np.median(baseline_scores):.5f}, "
              f"p95={np.percentile(baseline_scores, 95):.5f}, p99={np.percentile(baseline_scores, 99):.5f}")

        # ------------------------------------------------------------------
        # Step 5: Inject RFI at every (JSR, n_bands) combination and re-score
        # the SAME background tiles
        # ------------------------------------------------------------------
        contaminated_scores = {}
        for n_bands in args.n_bands_list:
            for jsr_db in args.jsr_db_list:
                contaminated_tiles = []
                for cpi_data, cpi_mask, p0, r0 in background_tiles:
                    contaminated_cpi, _ = inject_rfi_bands(cpi_data, cpi_mask, jsr_db, n_bands, rng)
                    contaminated_tiles.append((contaminated_cpi, cpi_mask, p0, r0))

                scores = score_tiles(contaminated_tiles, model, norm_stats, eigen_loss_weight, global_loss_weight)
                contaminated_scores[(n_bands, jsr_db)] = scores
                print(f"  n_bands={n_bands}, JSR={jsr_db:>5.1f} dB: "
                      f"median={np.median(scores):.5f}, p25={np.percentile(scores,25):.5f}, "
                      f"p75={np.percentile(scores,75):.5f}")

        # ------------------------------------------------------------------
        # Step 6: Plots
        # ------------------------------------------------------------------
        save_jsr_curve_png(
            args.jsr_db_list, args.n_bands_list, contaminated_scores, baseline_scores,
            threshold_p95, threshold_p99, pol, args.output_dir,
        )

        weakest_jsr = min(args.jsr_db_list)
        weakest_n_bands = min(args.n_bands_list)
        weakest_scores = contaminated_scores[(weakest_n_bands, weakest_jsr)]
        save_weakest_jsr_hist_png(baseline_scores, weakest_scores, weakest_jsr, weakest_n_bands, pol, args.output_dir)

        # ------------------------------------------------------------------
        # Step 7: Threshold tradeoff analysis at the weakest (hardest) case
        # ------------------------------------------------------------------
        candidate_thresholds = sorted(set(
            [float(np.percentile(baseline_scores, p)) for p in [90, 95, 99, 99.9]]
            + ([threshold_p95] if threshold_p95 is not None else [])
            + ([threshold_p99] if threshold_p99 is not None else [])
        ))
        tradeoff_rows = threshold_tradeoff(baseline_scores, weakest_scores, candidate_thresholds)

        threshold_for_90pct = threshold_for_detection_rate(weakest_scores, 0.90)
        threshold_for_95pct = threshold_for_detection_rate(weakest_scores, 0.95)
        fpr_at_90pct = float(np.mean(baseline_scores > threshold_for_90pct))
        fpr_at_95pct = float(np.mean(baseline_scores > threshold_for_95pct))

        print(f"\n  Threshold tradeoff at weakest case (JSR={weakest_jsr} dB, n_bands={weakest_n_bands}):")
        for row in tradeoff_rows:
            print(f"    threshold={row['threshold']:.5f}: FPR(baseline)={row['false_positive_rate']:.3f}, "
                  f"detection_rate(weakest)={row['detection_rate']:.3f}")
        print(f"  Threshold for 90% detection of weakest case: {threshold_for_90pct:.5f} "
              f"(FPR on baseline = {fpr_at_90pct:.3f})")
        print(f"  Threshold for 95% detection of weakest case: {threshold_for_95pct:.5f} "
              f"(FPR on baseline = {fpr_at_95pct:.3f})")

        all_raw_scores[f'baseline_{pol}'] = baseline_scores
        for (n_bands, jsr_db), scores in contaminated_scores.items():
            all_raw_scores[f'contaminated_{pol}_nbands{n_bands}_jsr{jsr_db}'] = scores

        all_summaries[pol] = {
            'n_background_tiles': n_bg,
            'baseline_percentiles': {
                str(p): float(np.percentile(baseline_scores, p)) for p in [50, 75, 90, 95, 99, 99.9]
            },
            'jsr_sweep': {
                f'nbands{n_bands}_jsr{jsr_db}': {
                    'jsr_db': jsr_db,
                    'n_bands': n_bands,
                    'median': float(np.median(scores)),
                    'p25': float(np.percentile(scores, 25)),
                    'p75': float(np.percentile(scores, 75)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                }
                for (n_bands, jsr_db), scores in contaminated_scores.items()
            },
            'weakest_case': {'jsr_db': weakest_jsr, 'n_bands': weakest_n_bands},
            'threshold_tradeoff_at_weakest_case': tradeoff_rows,
            'threshold_for_90pct_detection_of_weakest_case': threshold_for_90pct,
            'fpr_at_90pct_threshold': fpr_at_90pct,
            'threshold_for_95pct_detection_of_weakest_case': threshold_for_95pct,
            'fpr_at_95pct_threshold': fpr_at_95pct,
        }

    # ------------------------------------------------------------------
    # Step 8: Save raw scores and combined summary
    # ------------------------------------------------------------------
    scores_path = os.path.join(args.output_dir, 'jsr_sweep_scores.npz')
    np.savez(scores_path, **all_raw_scores)
    print(f"\nSaved raw per-tile scores to {scores_path}")

    summary = {
        'l0b_file': args.l0b_file,
        'freq': args.freq,
        'pol': args.pol,
        'pulse_start': p_start,
        'pulse_end': p_end,
        'range_start': r_start,
        'range_end': r_end,
        'model_dir': args.model_dir,
        'jsr_db_list': args.jsr_db_list,
        'n_bands_list': args.n_bands_list,
        'jsr_min_db_of_interest': JSR_MIN_DB,
        'threshold_p95_from_training': threshold_p95,
        'threshold_p99_from_training': threshold_p99,
        'per_polarization': all_summaries,
    }
    summary_path = os.path.join(args.output_dir, 'jsr_sweep_summary.json')
    with open(summary_path, 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved sweep summary to {summary_path}")


if __name__ == '__main__':
    main()