"""
plot_random_cpi.py

Diagnostic/inspection script: randomly samples 16x250 CPI tiles from real
L0B data and plots, for each one:

    1. Raw data magnitude (dB)
    2. Raw data magnitude with the subswath validity mask overlaid
       (invalid/gap samples highlighted)
    3. SCM covariance matrix magnitude (dB), origin at top-left (pulse 0 at
       the top, matching the convention used elsewhere in this project)
    4. Eigenvalue profile: all 16 RAW (non-normalized) eigenvalues in dB,
       fixed y-axis 0 to 60 dB

Each plot also reports (printed to console, saved as a companion JSON, and
annotated on the figure):
    - the max-normalized dB eigenvalue vector actually fed to the model
      (the first n_keep=12 values, same computation as anomaly_features.py)
    - condition number (dB), max - 12th eigenvalue
    - effective rank of the top 12 eigenvalues

This script does NOT train anything. It shares its feature-extraction
convention with train_anomaly.py (via anomaly_features.py) purely so that
what you see plotted here matches exactly what the model would be trained
on, but it duplicates the L0B reading / subswath-mask / region-manifest
logic locally rather than importing train_anomaly.py, consistent with the
project's convention of keeping per-script data-reading code self-contained.

Usage
-----
    python plot_random_cpi.py granule.h5 --freq A --pol HV \
        --pulse-start 813924 --pulse-end 888222 \
        --range-start 2000 --range-end 25000 \
        --n-samples 8 --output-dir plots/random_cpi
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from anomaly_features import (
    compute_gap_exclusion_scm,
    eigen_decompose_descending,
    normalize_eigvals_db,
    extract_anomaly_features,
    CPI_LEN_DEFAULT,
    CPI_WIDTH_DEFAULT,
    N_KEEP_DEFAULT,
    OFF_DIAG_OVERLAP_RATIO_DEFAULT,
    DIAG_VALID_RATIO_DEFAULT,
    EPS,
)

from nisar.products.readers.Raw import Raw

EIGEN_PROFILE_YLIM = (0.0, 60.0)


# ---------------------------------------------------------------------------
# ISCE3 RAW DATA / SUBSWATH MASK HELPERS (self-contained, matches
# train_anomaly.py; see read_nisar_swaths_isce3.py for the original
# gap-exclusion-mask derivation)
# ---------------------------------------------------------------------------

def read_raw_data_batch(raw: Raw, freq: str, pol: str, pulse_slice: slice, range_slice: slice):
    """
    Read a batch of raw data using ISCE3's efficient reader.

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
    window's first range index before painting.

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


# ---------------------------------------------------------------------------
# PER-FILE REGION RESOLUTION (matches train_anomaly.py)
# ---------------------------------------------------------------------------

def load_region_manifest(manifest_path: str) -> dict:
    """Load a JSON manifest mapping input file paths (or basenames) to a per-file pulse/range subset."""
    with open(manifest_path, 'r') as fh:
        return json.load(fh)


def resolve_region_for_file(l0b_path: str, manifest: dict, cli_args) -> dict:
    """Resolve the effective pulse/range window for one input file (see train_anomaly.py for details)."""
    entry = manifest.get(l0b_path)
    if entry is None:
        entry = manifest.get(os.path.basename(l0b_path), {})

    return {
        'pulse_start': entry.get('pulse_start', cli_args.pulse_start),
        'pulse_end': entry.get('pulse_end', cli_args.pulse_end),
        'range_start': entry.get('range_start', cli_args.range_start),
        'range_end': entry.get('range_end', cli_args.range_end),
    }


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def plot_one_cpi_sample(
    cpi_data: np.ndarray,
    cpi_mask: np.ndarray,
    scm: np.ndarray,
    eigvals_raw_db: np.ndarray,
    eigen_feat: np.ndarray,
    global_feat: np.ndarray,
    title_info: dict,
    out_path: str,
):
    """
    Build and save the 4-panel diagnostic figure for one CPI tile.

    Parameters
    ----------
    cpi_data : (M, K) complex array
        Raw slow-time CPI block.
    cpi_mask : (M, K) bool array
        Subswath validity mask for this tile (True = valid).
    scm : (M, M) complex array
        Gap-excluded sample covariance matrix for this tile.
    eigvals_raw_db : (M,) float array
        RAW (non-normalized) eigenvalues in dB, descending order.
    eigen_feat : (n_keep, 2) float array
        [normalized_db, slope_db] feature vector actually used by the model.
    global_feat : (2,) float array
        [condition_number_db, effective_rank].
    title_info : dict
        Metadata for the figure title (file, freq, pol, pulse/range start).
    out_path : str
        PNG output path.
    """
    M, K = cpi_data.shape
    n_keep = eigen_feat.shape[0]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax_raw, ax_mask, ax_scm, ax_eig = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # ---- Panel 1: raw data magnitude (dB), pulse 0 at top ----
    raw_mag_db = 20.0 * np.log10(np.abs(cpi_data) + EPS)
    im0 = ax_raw.imshow(raw_mag_db, aspect='auto', origin='upper', cmap='viridis')
    ax_raw.set_title('Raw Data Magnitude (dB)')
    ax_raw.set_xlabel('Range Sample Index')
    ax_raw.set_ylabel('Pulse Index')
    fig.colorbar(im0, ax=ax_raw, label='Magnitude (dB)')

    # ---- Panel 2: raw data magnitude with mask overlay, pulse 0 at top ----
    im1 = ax_mask.imshow(raw_mag_db, aspect='auto', origin='upper', cmap='gray')
    invalid_overlay = np.ma.masked_where(cpi_mask, np.ones_like(raw_mag_db))
    ax_mask.imshow(invalid_overlay, aspect='auto', origin='upper', cmap='Reds', alpha=0.6, vmin=0, vmax=1)
    valid_pct = 100.0 * cpi_mask.sum() / cpi_mask.size
    ax_mask.set_title(f'Subswath Mask Overlay (red = invalid, valid={valid_pct:.1f}%)')
    ax_mask.set_xlabel('Range Sample Index')
    ax_mask.set_ylabel('Pulse Index')

    # ---- Panel 3: SCM covariance matrix magnitude (dB), pulse 0 at top-left ----
    scm_mag_db = 20.0 * np.log10(np.abs(scm) + EPS)
    im2 = ax_scm.imshow(scm_mag_db, aspect='auto', origin='upper', cmap='viridis')
    ax_scm.set_title('SCM Magnitude (dB)')
    ax_scm.set_xlabel('Pulse Index')
    ax_scm.set_ylabel('Pulse Index')
    fig.colorbar(im2, ax=ax_scm, label='Magnitude (dB)')

    # ---- Panel 4: eigenvalue profile, all M raw (non-normalized) eigenvalues ----
    eig_idx = np.arange(M)
    ax_eig.plot(eig_idx, eigvals_raw_db, marker='o', color='tab:blue')
    ax_eig.axvline(n_keep - 1, color='tab:red', linestyle='--', alpha=0.6, label=f'n_keep={n_keep} cutoff')
    ax_eig.set_ylim(*EIGEN_PROFILE_YLIM)
    ax_eig.set_xlabel('Eigenvalue Index')
    ax_eig.set_ylabel('Eigenvalue (dB)')
    ax_eig.set_title('Eigenvalue Profile (raw, all 16)')
    ax_eig.grid(True, linestyle='--', alpha=0.5)
    ax_eig.legend(loc='upper right')

    # ---- Figure title and text report ----
    fig.suptitle(
        f"{title_info['file']}  |  {title_info['freq']}-{title_info['pol']}  |  "
        f"pulse [{title_info['pulse_start']}:{title_info['pulse_start'] + M}]  "
        f"range [{title_info['range_start']}:{title_info['range_start'] + K}]",
        fontsize=11,
    )

    normalized_vec_str = ", ".join(f"{v:.2f}" for v in eigen_feat[:, 0])
    report_text = (
        f"Normalized eigenvalue vector fed to model (dB, max=0dB, n_keep={n_keep}):\n"
        f"[{normalized_vec_str}]\n"
        f"Condition number (dB, max - {n_keep}th): {global_feat[0]:.2f}\n"
        f"Effective rank (of top {n_keep}): {global_feat[1]:.2f}"
    )
    fig.text(0.5, 0.01, report_text, ha='center', va='bottom', fontsize=9, family='monospace')

    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Randomly sample and plot CPI eigenvalue profiles from real NISAR L0B data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('l0b_files', nargs='+', help='One or more input NISAR L0B HDF5 files')
    parser.add_argument('--freq', choices=['A', 'B'], default='A', help='Frequency to process (default: A)')
    parser.add_argument('--pol', choices=['HH', 'HV', 'VH', 'VV'], default='HV', help='Polarization (default: HV)')
    parser.add_argument('--output-dir', type=str, default='plots/random_cpi', help='Output directory for figures')

    parser.add_argument('--pulse-start', type=int, default=None,
                        help='Global pulse start index, fallback for files not covered by --region-manifest')
    parser.add_argument('--pulse-end', type=int, default=None,
                        help='Global pulse end index, fallback for files not covered by --region-manifest')
    parser.add_argument('--range-start', type=int, default=None,
                        help='Global range start index, fallback for files not covered by --region-manifest')
    parser.add_argument('--range-end', type=int, default=None,
                        help='Global range end index, fallback for files not covered by --region-manifest')
    parser.add_argument('--region-manifest', type=str, default=None,
                        help='JSON file specifying a per-file pulse/range subset (see train_anomaly.py)')

    parser.add_argument('--cpi-len', type=int, default=CPI_LEN_DEFAULT, help='CPI length in pulses (default: 16)')
    parser.add_argument('--cpi-width', type=int, default=CPI_WIDTH_DEFAULT, help='CPI width in range samples (default: 250)')
    parser.add_argument('--n-keep', type=int, default=N_KEEP_DEFAULT, help='Number of leading eigenvalues kept (default: 12)')
    parser.add_argument('--off-diag-overlap-ratio', type=float, default=OFF_DIAG_OVERLAP_RATIO_DEFAULT,
                        help='Gap-exclusion off-diagonal overlap ratio (default: 0.03)')
    parser.add_argument('--diag-valid-ratio', type=float, default=DIAG_VALID_RATIO_DEFAULT,
                        help='Gap-exclusion diagonal valid ratio (default: 0.02)')

    parser.add_argument('--n-samples', type=int, default=6, help='Number of random CPI tiles to plot (default: 6)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (default: unseeded, different each run)')

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    region_manifest = {}
    if args.region_manifest is not None:
        region_manifest = load_region_manifest(args.region_manifest)
        print(f"Loaded region manifest from {args.region_manifest} ({len(region_manifest)} entries)")

    sample_count = 0

    for l0b_path in args.l0b_files:
        region = resolve_region_for_file(l0b_path, region_manifest, args)
        print(f"\nOpening {l0b_path} ...")
        print(f"  Region: pulses [{region['pulse_start']}:{region['pulse_end']}], "
              f"range [{region['range_start']}:{region['range_end']}]")

        raw = Raw(hdf5file=str(l0b_path))
        raw.parsePolarizations()

        dataset = raw.getRawDataset(args.freq, args.pol)
        total_pulses, total_range = dataset.shape

        p_start = region['pulse_start'] if region['pulse_start'] is not None else 0
        p_end = region['pulse_end'] if region['pulse_end'] is not None else total_pulses
        r_start = region['range_start'] if region['range_start'] is not None else 0
        r_end = region['range_end'] if region['range_end'] is not None else total_range

        n_pulse_tiles = (p_end - p_start) // args.cpi_len
        n_range_tiles = (r_end - r_start) // args.cpi_width
        p_end = p_start + n_pulse_tiles * args.cpi_len
        r_end = r_start + n_range_tiles * args.cpi_width

        if n_pulse_tiles == 0 or n_range_tiles == 0:
            print(f"  WARNING: region too small for CPI size {args.cpi_len}x{args.cpi_width}, skipping file")
            continue

        print(f"  Reading [{p_start}:{p_end}, {r_start}:{r_end}] "
              f"({n_pulse_tiles} pulse tiles x {n_range_tiles} range tiles available)")

        raw_data = read_raw_data_batch(raw, args.freq, args.pol, slice(p_start, p_end), slice(r_start, r_end))
        pulse_indices = np.arange(p_start, p_end)
        range_indices = np.arange(r_start, r_end)
        mask_valid = get_subswath_mask(raw, args.freq, args.pol, pulse_indices, range_indices)

        # Randomly choose (pulse_tile, range_tile) grid positions for this file
        n_available = n_pulse_tiles * n_range_tiles
        n_pick = min(args.n_samples - sample_count, n_available)
        if n_pick <= 0:
            break

        flat_choices = rng.choice(n_available, size=n_pick, replace=False)

        for flat_idx in flat_choices:
            pt = int(flat_idx // n_range_tiles)
            rt = int(flat_idx % n_range_tiles)

            tile_p_start = pt * args.cpi_len
            tile_r_start = rt * args.cpi_width
            tile_p_end = tile_p_start + args.cpi_len
            tile_r_end = tile_r_start + args.cpi_width

            cpi_data = raw_data[tile_p_start:tile_p_end, tile_r_start:tile_r_end]
            cpi_mask = mask_valid[tile_p_start:tile_p_end, tile_r_start:tile_r_end]

            # Model-convention features (normalized dB vector, condition number, effective rank)
            eigen_feat, global_feat, diag_valid_frac, _ = extract_anomaly_features(
                cpi_data,
                mask_valid_cpi=cpi_mask,
                n_keep=args.n_keep,
                off_diag_overlap_ratio=args.off_diag_overlap_ratio,
                diag_valid_ratio=args.diag_valid_ratio,
            )

            # Raw (non-normalized) eigenvalues in dB, for the 0-60 dB profile plot
            scm, _, _ = compute_gap_exclusion_scm(
                cpi_data,
                mask_valid_cpi=cpi_mask,
                off_diag_overlap_ratio=args.off_diag_overlap_ratio,
                diag_valid_ratio=args.diag_valid_ratio,
            )
            eigvals_linear = eigen_decompose_descending(scm)
            eigvals_raw_db = (10.0 * np.log10(np.clip(eigvals_linear, EPS, None))).astype(np.float32)

            sample_count += 1
            out_png = os.path.join(args.output_dir, f'sample_{sample_count:03d}.png')
            out_json = os.path.join(args.output_dir, f'sample_{sample_count:03d}.json')

            title_info = {
                'file': os.path.basename(l0b_path),
                'freq': args.freq,
                'pol': args.pol,
                'pulse_start': p_start + tile_p_start,
                'range_start': r_start + tile_r_start,
            }

            plot_one_cpi_sample(
                cpi_data, cpi_mask, scm, eigvals_raw_db, eigen_feat, global_feat, title_info, out_png,
            )

            report = {
                'file': l0b_path,
                'freq': args.freq,
                'pol': args.pol,
                'pulse_start': int(p_start + tile_p_start),
                'range_start': int(r_start + tile_r_start),
                'diag_valid_frac': float(diag_valid_frac),
                'eigvals_raw_db': eigvals_raw_db.tolist(),
                'normalized_eigvals_db_used': eigen_feat[:, 0].tolist(),
                'slopes_db_used': eigen_feat[:, 1].tolist(),
                'condition_number_db': float(global_feat[0]),
                'effective_rank': float(global_feat[1]),
            }
            with open(out_json, 'w') as fh:
                json.dump(report, fh, indent=2)

            print(f"\n  Sample {sample_count}: {title_info['file']} "
                  f"pulse_start={title_info['pulse_start']} range_start={title_info['range_start']}")
            print(f"    diag_valid_frac       : {diag_valid_frac:.3f}")
            print(f"    normalized eigvals(db): {np.round(eigen_feat[:, 0], 2).tolist()}")
            print(f"    condition_number (dB) : {global_feat[0]:.2f}")
            print(f"    effective_rank        : {global_feat[1]:.2f}")

        if sample_count >= args.n_samples:
            break

    print(f"\nDone. Plotted {sample_count} random CPI samples to {args.output_dir}")


if __name__ == '__main__':
    main()
