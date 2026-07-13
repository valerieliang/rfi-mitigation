"""
train_anomaly.py

Training pipeline for the CNN-autoencoder RFI anomaly detector, trained on
real NISAR L0B data (no synthetic RFI injection required for the training
corpus itself).

Standard CPI size: 16 pulses x 250 range samples.

Why train on real data directly
---------------------------------
The previous classifier was trained on synthetic clean/RFI data and failed
to transfer to real L0B data (softmax overconfidence on out-of-distribution
inputs, noise floor mismatch between synthetic and real clean regimes). An
autoencoder trained ONLY on real clean tiles sidesteps both problems: there
is no synthetic distribution to transfer from, and there is no classifier
decision boundary to be overconfident about.

Clean corpus self-filtering (important caveat)
-------------------------------------------------
Real L0B data has no ground-truth RFI labels. This script assumes the BULK
of tiles in a typical granule are RFI-free, and self-filters obvious outliers
before training using a condition-number-in-dB threshold
(--cond-db-filter, default 25.0 dB). This is a heuristic, not ground truth:
- Too low a threshold discards legitimately clean but naturally sharp-kneed
  tiles (false exclusion).
- Too high a threshold lets contaminated tiles leak into the "clean"
  training corpus (fewer exclusions, dirtier corpus).
Inspect the printed condition-number percentiles and the saved histogram
plot, and adjust --cond-db-filter for your data before committing to a run.

Usage
-----
    python train_anomaly.py file1.h5 file2.h5 --freq A --pol HV \
        --output-dir models/anomaly_v1 --cond-db-filter 25.0

Outputs (in --output-dir)
--------------------------
    best_model.keras       -- trained autoencoder (ModelCheckpoint best val_loss)
    norm_stats.npz         -- per-feature mean/std used to normalize eigen/global
    training_curves.png    -- loss vs epoch (train/val)
    recon_error_hist.png   -- train/val reconstruction error histograms
    cond_number_hist.png   -- condition number distribution used for filtering
    run_summary.json       -- tile counts, filter threshold, recommended anomaly
                               threshold (95th/99th percentile of val error)
"""

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf

from anomaly_features import (
    tile_and_extract_features,
    CPI_LEN_DEFAULT,
    CPI_WIDTH_DEFAULT,
    N_KEEP_DEFAULT,
    OFF_DIAG_OVERLAP_RATIO_DEFAULT,
    DIAG_VALID_RATIO_DEFAULT,
)
from model_anomaly import build_anomaly_autoencoder, compute_anomaly_scores

from nisar.products.readers.Raw import Raw

EPOCHS_DEFAULT = 100
BATCH_SIZE_DEFAULT = 128
LR_DEFAULT = 1e-3
VAL_FRAC_DEFAULT = 0.1
COND_DB_FILTER_DEFAULT = 25.0


# ---------------------------------------------------------------------------
# ISCE3 RAW DATA / SUBSWATH MASK HELPERS (self-contained, see
# read_nisar_swaths_isce3.py for the original gap-exclusion-mask derivation)
# ---------------------------------------------------------------------------

def read_raw_data_batch(raw: Raw, freq: str, pol: str, pulse_slice: slice, range_slice: slice):
    """
    Read a batch of raw data using ISCE3's efficient reader.

    Parameters
    ----------
    raw : Raw
    freq : str
    pol : str
    pulse_slice, range_slice : slice

    Returns
    -------
    data : (n_pulses, n_range) complex64 array
    """
    dataset = raw.getRawDataset(freq, pol)

    pulse_start = pulse_slice.start if pulse_slice.start is not None else 0
    pulse_stop = pulse_slice.stop if pulse_slice.stop is not None else dataset.shape[0]
    range_start = range_slice.start if range_slice.start is not None else 0
    range_stop = range_slice.stop if range_slice.stop is not None else dataset.shape[1]

    return dataset[pulse_start:pulse_stop, range_start:range_stop]


def get_subswath_mask(raw: Raw, freq: str, pol: str, pulse_indices: np.ndarray, range_indices: np.ndarray) -> np.ndarray:
    """
    Build a boolean valid-data mask from ISCE3 subswath boundaries.

    getSubSwaths() returns ABSOLUTE range sample indices, while the mask
    array being built is window-relative, so boundaries are shifted by the
    window's first range index before painting (this matches the fix
    already applied in read_nisar_swaths_isce3.py).

    Parameters
    ----------
    raw : Raw
    freq : str
    pol : str
    pulse_indices : (n_pulses,) int array
    range_indices : (n_range,) int array

    Returns
    -------
    mask : (n_pulses, n_range) bool array
    """
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


def extract_tiles_from_file(
    l0b_path: str,
    freq: str,
    pol: str,
    cpi_len: int,
    cpi_width: int,
    n_keep: int,
    off_diag_overlap_ratio: float,
    diag_valid_ratio: float,
    pulse_start: int = None,
    pulse_end: int = None,
    range_start: int = None,
    range_end: int = None,
):
    """
    Open one L0B file, read the requested freq/pol slab, build the subswath
    mask, and extract per-tile anomaly features for every 16x250 CPI block.

    Returns
    -------
    eigen_feats : (N, n_keep, 2) float32 array
    global_feats : (N, 2) float32 array
    diag_valid_fracs : (N,) float32 array
    """
    print(f"\nOpening {l0b_path} ...")
    raw = Raw(hdf5file=str(l0b_path))
    raw.parsePolarizations()

    dataset = raw.getRawDataset(freq, pol)
    total_pulses, total_range = dataset.shape

    p_start = pulse_start if pulse_start is not None else 0
    p_end = pulse_end if pulse_end is not None else total_pulses
    r_start = range_start if range_start is not None else 0
    r_end = range_end if range_end is not None else total_range

    # Align to CPI boundary
    n_pulses = p_end - p_start
    num_cpi_tiles = n_pulses // cpi_len
    p_end = p_start + num_cpi_tiles * cpi_len

    n_range = r_end - r_start
    num_range_tiles = n_range // cpi_width
    r_end = r_start + num_range_tiles * cpi_width

    print(f"  {freq}-{pol}: reading [{p_start}:{p_end}, {r_start}:{r_end}]")
    raw_data = read_raw_data_batch(raw, freq, pol, slice(p_start, p_end), slice(r_start, r_end))

    pulse_indices = np.arange(p_start, p_end)
    range_indices = np.arange(r_start, r_end)
    mask_valid = get_subswath_mask(raw, freq, pol, pulse_indices, range_indices)

    valid_pct = 100.0 * mask_valid.sum() / mask_valid.size
    print(f"  Subswath valid fraction: {valid_pct:.1f}%")

    eigen_feats, global_feats, diag_valid_fracs, _, _ = tile_and_extract_features(
        raw_data,
        mask_valid=mask_valid,
        cpi_len=cpi_len,
        cpi_width=cpi_width,
        n_keep=n_keep,
        off_diag_overlap_ratio=off_diag_overlap_ratio,
        diag_valid_ratio=diag_valid_ratio,
        min_tile_valid_frac=0.0,  # keep dithered tiles; filtering happens later
    )
    print(f"  Extracted {eigen_feats.shape[0]} tiles")

    return eigen_feats, global_feats, diag_valid_fracs


# ---------------------------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------------------------

def fit_normalization_stats(eigen_train: np.ndarray, global_train: np.ndarray):
    """
    Compute per-channel z-score stats from the training split only.

    Parameters
    ----------
    eigen_train : (N, n_keep, 2)
    global_train : (N, 2)

    Returns
    -------
    stats : dict of numpy arrays
        eigen_mean, eigen_std : (2,) -- per-channel (db, slope) stats,
            pooled across the n_keep sequence positions.
        global_mean, global_std : (2,) -- per-feature stats.
    """
    eigen_mean = eigen_train.reshape(-1, 2).mean(axis=0).astype(np.float32)
    eigen_std = eigen_train.reshape(-1, 2).std(axis=0).astype(np.float32)
    eigen_std = np.maximum(eigen_std, 1e-6)

    global_mean = global_train.mean(axis=0).astype(np.float32)
    global_std = global_train.std(axis=0).astype(np.float32)
    global_std = np.maximum(global_std, 1e-6)

    return {
        'eigen_mean': eigen_mean,
        'eigen_std': eigen_std,
        'global_mean': global_mean,
        'global_std': global_std,
    }


def apply_normalization(eigen, global_, stats):
    eigen_norm = (eigen - stats['eigen_mean']) / stats['eigen_std']
    global_norm = (global_ - stats['global_mean']) / stats['global_std']
    return eigen_norm.astype(np.float32), global_norm.astype(np.float32)


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------

def save_training_curves_png(history, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = range(1, len(history.history['loss']) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history.history['loss'], label='Train loss')
    ax.plot(epochs, history.history['val_loss'], label='Val loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (weighted MSE)')
    ax.set_title('Autoencoder Training & Validation Loss')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    path = os.path.join(out_dir, 'training_curves.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def save_recon_error_hist_png(train_scores, val_scores, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(train_scores, bins=60, alpha=0.6, label='Train (clean corpus)', density=True)
    ax.hist(val_scores, bins=60, alpha=0.6, label='Val (clean corpus)', density=True)
    ax.set_xlabel('Total anomaly score (weighted reconstruction MSE)')
    ax.set_ylabel('Density')
    ax.set_title('Reconstruction Error Distribution')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    path = os.path.join(out_dir, 'recon_error_hist.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def save_cond_number_hist_png(cond_db_all, threshold, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(cond_db_all, bins=80, alpha=0.8)
    ax.axvline(threshold, color='red', linestyle='--', label=f'Filter threshold = {threshold:.1f} dB')
    ax.set_xlabel('Condition number (dB), max - 12th eigenvalue')
    ax.set_ylabel('Tile count')
    ax.set_title('Condition Number Distribution (Clean-Corpus Self-Filter)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    path = os.path.join(out_dir, 'cond_number_hist.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# PER-FILE REGION RESOLUTION
# ---------------------------------------------------------------------------

def load_region_manifest(manifest_path: str) -> dict:
    """
    Load a JSON manifest mapping input file paths (or basenames) to a
    per-file pulse/range subset.

    Manifest format
    ---------------
    {
        "GRANULE_1.h5": {"pulse_start": 0,     "pulse_end": 50000,
                          "range_start": 0,     "range_end": 20000},
        "GRANULE_2.h5": {"pulse_start": 10000, "pulse_end": 90000}
    }

    Any of the four keys may be omitted for a given file; omitted keys fall
    back to the global --pulse-start/--pulse-end/--range-start/--range-end
    CLI arguments (which themselves default to the full extent of the file).

    Keys may be given as either the exact path passed on the command line or
    just the basename, so the manifest does not need to be rewritten if you
    move files around; exact-path entries take precedence over basename
    entries when both are present.

    Parameters
    ----------
    manifest_path : str
        Path to a JSON file with the format above.

    Returns
    -------
    manifest : dict
        Parsed JSON content, keyed by whatever strings were used in the file.
    """
    with open(manifest_path, 'r') as fh:
        manifest = json.load(fh)
    return manifest


def resolve_region_for_file(l0b_path: str, manifest: dict, cli_args) -> dict:
    """
    Resolve the effective pulse/range window for one input file.

    Lookup order: exact CLI path match in the manifest, then basename match
    in the manifest, then the global --pulse-start/--pulse-end/--range-start/
    --range-end CLI arguments as the final fallback. Any key still unset
    after that resolves to None (full extent of the file).

    Parameters
    ----------
    l0b_path : str
        Path exactly as given on the command line for this file.
    manifest : dict
        Parsed region manifest (possibly empty dict if none was provided).
    cli_args : argparse.Namespace
        Parsed CLI arguments, used for the global fallback values.

    Returns
    -------
    region : dict
        Keys: pulse_start, pulse_end, range_start, range_end (each int or None).
    """
    entry = manifest.get(l0b_path)
    if entry is None:
        entry = manifest.get(os.path.basename(l0b_path), {})

    region = {
        'pulse_start': entry.get('pulse_start', cli_args.pulse_start),
        'pulse_end': entry.get('pulse_end', cli_args.pulse_end),
        'range_start': entry.get('range_start', cli_args.range_start),
        'range_end': entry.get('range_end', cli_args.range_end),
    }
    return region


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train the CNN autoencoder RFI anomaly detector on real NISAR L0B data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('l0b_files', nargs='+', help='One or more input NISAR L0B HDF5 files')
    parser.add_argument('--freq', choices=['A', 'B'], default='A', help='Frequency to process (default: A)')
    parser.add_argument('--pol', choices=['HH', 'HV', 'VH', 'VV'], default='HV',
                        help='Polarization to process (default: HV, RFI couples independently per receive chain)')
    parser.add_argument('--output-dir', type=str, default='models/anomaly_v1', help='Output directory')

    parser.add_argument('--pulse-start', type=int, default=None,
                        help='Global pulse start index, used as fallback for any file not covered '
                             'by --region-manifest (default: 0)')
    parser.add_argument('--pulse-end', type=int, default=None,
                        help='Global pulse end index, used as fallback for any file not covered '
                             'by --region-manifest (default: full extent)')
    parser.add_argument('--range-start', type=int, default=None,
                        help='Global range start index, used as fallback for any file not covered '
                             'by --region-manifest (default: 0)')
    parser.add_argument('--range-end', type=int, default=None,
                        help='Global range end index, used as fallback for any file not covered '
                             'by --region-manifest (default: full extent)')
    parser.add_argument('--region-manifest', type=str, default=None,
                        help='Path to a JSON file specifying a per-file pulse/range subset, so each '
                             'input granule can use a different window. Keys are the input file paths '
                             '(or basenames); values are objects with any of pulse_start, pulse_end, '
                             'range_start, range_end. Files/keys not covered fall back to the global '
                             '--pulse-start/--pulse-end/--range-start/--range-end flags.')


    parser.add_argument('--cpi-len', type=int, default=CPI_LEN_DEFAULT, help='CPI length in pulses (default: 16)')
    parser.add_argument('--cpi-width', type=int, default=CPI_WIDTH_DEFAULT, help='CPI width in range samples (default: 250)')
    parser.add_argument('--n-keep', type=int, default=N_KEEP_DEFAULT, help='Number of leading eigenvalues kept (default: 12)')
    parser.add_argument('--off-diag-overlap-ratio', type=float, default=OFF_DIAG_OVERLAP_RATIO_DEFAULT,
                        help='Gap-exclusion off-diagonal overlap ratio (default: 0.03)')
    parser.add_argument('--diag-valid-ratio', type=float, default=DIAG_VALID_RATIO_DEFAULT,
                        help='Gap-exclusion diagonal valid ratio (default: 0.02)')

    parser.add_argument('--cond-db-filter', type=float, default=COND_DB_FILTER_DEFAULT,
                        help='Heuristic clean-corpus filter: exclude tiles with condition number '
                             '(dB, max - 12th eigenvalue) above this value (default: 25.0)')

    parser.add_argument('--val-frac', type=float, default=VAL_FRAC_DEFAULT, help='Validation split fraction (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=EPOCHS_DEFAULT)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument('--learning-rate', type=float, default=LR_DEFAULT)
    parser.add_argument('--latent-dim', type=int, default=8)
    parser.add_argument('--dropout-rate', type=float, default=0.1)
    parser.add_argument('--eigen-loss-weight', type=float, default=1.0)
    parser.add_argument('--global-loss-weight', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("RFI Anomaly Autoencoder Training")
    print("=" * 70)
    print(f"Files: {args.l0b_files}")
    print(f"Freq/Pol: {args.freq}/{args.pol}")
    print(f"CPI size: {args.cpi_len}x{args.cpi_width}, n_keep={args.n_keep}")
    print(f"Gap-exclusion ratios: off_diag={args.off_diag_overlap_ratio}, diag={args.diag_valid_ratio}")
    print(f"Output dir: {args.output_dir}")

    # ------------------------------------------------------------------
    # Step 1: Extract tiles from every input file, using a per-file
    # pulse/range window if a region manifest was provided
    # ------------------------------------------------------------------
    region_manifest = {}
    if args.region_manifest is not None:
        region_manifest = load_region_manifest(args.region_manifest)
        print(f"\nLoaded region manifest from {args.region_manifest} "
              f"({len(region_manifest)} entries)")

    eigen_all, global_all, frac_all = [], [], []

    for l0b_path in args.l0b_files:
        region = resolve_region_for_file(l0b_path, region_manifest, args)
        print(f"\nRegion for {l0b_path}: "
              f"pulses [{region['pulse_start']}:{region['pulse_end']}], "
              f"range [{region['range_start']}:{region['range_end']}]")

        eigen_feats, global_feats, diag_valid_fracs = extract_tiles_from_file(
            l0b_path,
            freq=args.freq,
            pol=args.pol,
            cpi_len=args.cpi_len,
            cpi_width=args.cpi_width,
            n_keep=args.n_keep,
            off_diag_overlap_ratio=args.off_diag_overlap_ratio,
            diag_valid_ratio=args.diag_valid_ratio,
            pulse_start=region['pulse_start'],
            pulse_end=region['pulse_end'],
            range_start=region['range_start'],
            range_end=region['range_end'],
        )
        eigen_all.append(eigen_feats)
        global_all.append(global_feats)
        frac_all.append(diag_valid_fracs)

    eigen_all = np.concatenate(eigen_all, axis=0)
    global_all = np.concatenate(global_all, axis=0)
    frac_all = np.concatenate(frac_all, axis=0)

    n_total = eigen_all.shape[0]
    print(f"\nTotal tiles extracted across all files: {n_total}")
    if n_total == 0:
        print("ERROR: No tiles extracted. Check input files and freq/pol arguments.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: Clean-corpus self-filter on condition number (heuristic)
    # ------------------------------------------------------------------
    cond_db_all = global_all[:, 0]
    percentiles = [50, 75, 90, 95, 99]
    pct_vals = np.percentile(cond_db_all, percentiles)
    print("\nCondition number (dB) percentiles across all extracted tiles:")
    for p, v in zip(percentiles, pct_vals):
        print(f"  {p}th percentile: {v:.2f} dB")

    save_cond_number_hist_png(cond_db_all, args.cond_db_filter, args.output_dir)

    clean_mask = cond_db_all <= args.cond_db_filter
    n_clean = int(clean_mask.sum())
    n_excluded = n_total - n_clean
    print(f"\nClean-corpus self-filter: keeping {n_clean}/{n_total} tiles "
          f"(cond_db <= {args.cond_db_filter} dB), excluding {n_excluded}")

    if n_clean < 100:
        print("WARNING: Very few tiles survived the clean-corpus filter. "
              "Consider raising --cond-db-filter or providing more input files.")

    eigen_clean = eigen_all[clean_mask]
    global_clean = global_all[clean_mask]

    # ------------------------------------------------------------------
    # Step 3: Train / val split
    # ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    n_clean_total = eigen_clean.shape[0]
    perm = rng.permutation(n_clean_total)
    n_val = max(1, int(round(args.val_frac * n_clean_total)))

    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    eigen_train_raw = eigen_clean[train_idx]
    global_train_raw = global_clean[train_idx]
    eigen_val_raw = eigen_clean[val_idx]
    global_val_raw = global_clean[val_idx]

    print(f"\nTrain tiles: {len(train_idx)}, Val tiles: {len(val_idx)}")

    # ------------------------------------------------------------------
    # Step 4: Normalization (fit on train split only)
    # ------------------------------------------------------------------
    norm_stats = fit_normalization_stats(eigen_train_raw, global_train_raw)
    eigen_train, global_train = apply_normalization(eigen_train_raw, global_train_raw, norm_stats)
    eigen_val, global_val = apply_normalization(eigen_val_raw, global_val_raw, norm_stats)

    norm_stats_path = os.path.join(args.output_dir, 'norm_stats.npz')
    np.savez(
        norm_stats_path,
        eigen_mean=norm_stats['eigen_mean'],
        eigen_std=norm_stats['eigen_std'],
        global_mean=norm_stats['global_mean'],
        global_std=norm_stats['global_std'],
        n_keep=args.n_keep,
        cpi_len=args.cpi_len,
        cpi_width=args.cpi_width,
        off_diag_overlap_ratio=args.off_diag_overlap_ratio,
        diag_valid_ratio=args.diag_valid_ratio,
        cond_db_filter=args.cond_db_filter,
    )
    print(f"Saved normalization stats to {norm_stats_path}")

    # ------------------------------------------------------------------
    # Step 5: Build and train model
    # ------------------------------------------------------------------
    model = build_anomaly_autoencoder(
        cpi_size=args.n_keep,
        n_global_features=2,
        latent_dim=args.latent_dim,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        eigen_loss_weight=args.eigen_loss_weight,
        global_loss_weight=args.global_loss_weight,
    )
    model.summary()

    model_path = os.path.join(args.output_dir, 'best_model.keras')
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path, monitor='val_loss', save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1,
        ),
    ]

    history = model.fit(
        x=[eigen_train, global_train],
        y=[eigen_train, global_train],
        validation_data=([eigen_val, global_val], [eigen_val, global_val]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    save_training_curves_png(history, args.output_dir)

    # ------------------------------------------------------------------
    # Step 6: Reconstruction-error diagnostics and recommended threshold
    # ------------------------------------------------------------------
    _, _, train_scores = compute_anomaly_scores(
        model, eigen_train, global_train,
        eigen_weight=args.eigen_loss_weight, global_weight=args.global_loss_weight,
    )
    _, _, val_scores = compute_anomaly_scores(
        model, eigen_val, global_val,
        eigen_weight=args.eigen_loss_weight, global_weight=args.global_loss_weight,
    )

    save_recon_error_hist_png(train_scores, val_scores, args.output_dir)

    recommended_threshold_95 = float(np.percentile(val_scores, 95))
    recommended_threshold_99 = float(np.percentile(val_scores, 99))

    print(f"\nRecommended anomaly score thresholds (from val clean-corpus distribution):")
    print(f"  95th percentile: {recommended_threshold_95:.6f}")
    print(f"  99th percentile: {recommended_threshold_99:.6f}")

    # ------------------------------------------------------------------
    # Step 7: Summary
    # ------------------------------------------------------------------
    summary = {
        'l0b_files': args.l0b_files,
        'freq': args.freq,
        'pol': args.pol,
        'cpi_len': args.cpi_len,
        'cpi_width': args.cpi_width,
        'n_keep': args.n_keep,
        'off_diag_overlap_ratio': args.off_diag_overlap_ratio,
        'diag_valid_ratio': args.diag_valid_ratio,
        'n_tiles_total': int(n_total),
        'cond_db_filter': args.cond_db_filter,
        'n_tiles_clean_corpus': int(n_clean),
        'n_tiles_excluded': int(n_excluded),
        'n_train': int(len(train_idx)),
        'n_val': int(len(val_idx)),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'recommended_threshold_p95': recommended_threshold_95,
        'recommended_threshold_p99': recommended_threshold_99,
    }

    summary_path = os.path.join(args.output_dir, 'run_summary.json')
    with open(summary_path, 'w') as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nSaved run summary to {summary_path}")
    print(f"Model saved to {model_path}")
    print(f"Normalization stats saved to {norm_stats_path}")


if __name__ == '__main__':
    main()