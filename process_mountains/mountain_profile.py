#!/usr/bin/env python
"""
mountain_profile.py

Per-CPI eigenvalue "mountain" profiler for a NISAR L0B scene.

Reads a targeted pulse range and range width from an L0B granule, tiles it into
standard 16 x 250 CPI blocks, builds a gap-exclusion sample covariance matrix
(SCM) per CPI (same pipeline as anomaly_features.py), and produces detailed
diagnostic plots of the per-CPI eigenvalue profiles:

  1. average shape   - the mean eigenvalue profile with a +/-1 std band and a
                       min/max envelope (the "typical" mountain shape)
  2. all overlaid    - every CPI profile drawn on the same axes
  3. grid of shapes  - a grid of small multiples, one CPI profile per cell

It also plots the SCM diagonal values per CPI, and stores every profile and
diagonal to HDF5. All plots show only the first 12 values (largest 12
eigenvalues / first 12 diagonal entries), consistent with N_KEEP in
anomaly_features.py.

There is no model and no classification: this is a pure profiling / inspection
tool driven only by the selected pulse and range window.

NOTE on data reading: open_raw()/read_scene_block() below are the L0B reader
integration points. They use the ISCE3 / nisar Raw + getRawDataset interface
for BFPQLUT decoding. Align them with read_nisar_swaths_isce3.py if the local
API differs.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py

# ---------------------------------------------------------------------------
# CONSTANTS (copied from anomaly_features.py to make this script standalone)
# ---------------------------------------------------------------------------

CPI_LEN_DEFAULT = 16
CPI_WIDTH_DEFAULT = 250

# Permissive gap-exclusion ratios: preserve as much dithered data as possible
OFF_DIAG_OVERLAP_RATIO_DEFAULT = 0.03
DIAG_VALID_RATIO_DEFAULT = 0.02

# Number of leading (largest) eigenvalues to keep as features.
N_KEEP_DEFAULT = 12

# Number of values to show in plots (all 16)
N_SHOW = 16

EPS = 1e-12


# ---------------------------------------------------------------------------
# GAP-EXCLUSION SCM (standalone - copied from anomaly_features.py)
# ---------------------------------------------------------------------------

def compute_gap_exclusion_scm(
    data: np.ndarray,
    *,
    mask_valid_cpi: np.ndarray = None,
    off_diag_overlap_ratio: float = OFF_DIAG_OVERLAP_RATIO_DEFAULT,
    diag_valid_ratio: float = DIAG_VALID_RATIO_DEFAULT,
):
    """
    Compute a gap-excluded slow-time sample covariance matrix (SCM).

    Parameters
    ----------
    data : (num_pulses, num_rng_samples) complex array
        Slow-time CPI block: M pulses x K range samples.
    mask_valid_cpi : (num_pulses, num_rng_samples) bool array, optional
        True indicates a valid sample. If None, all samples are valid.
    off_diag_overlap_ratio : float, default 0.03
        Minimum fraction of overlapping valid range samples required to
        compute an off-diagonal SCM entry R_ij.
    diag_valid_ratio : float, default 0.02
        Minimum fraction of valid samples required to compute a diagonal
        SCM entry R_ii.

    Returns
    -------
    scm : (num_pulses, num_pulses) complex64
        Gap-excluded sample covariance matrix.
    diag_valid_idx : (num_pulses,) bool array
        True where the diagonal term had enough valid samples to be trusted.
    diag_valid_frac : float
        Fraction of diagonal entries that were valid (0 to 1).
    """
    num_pulses, num_rng_samples = data.shape

    if mask_valid_cpi is None:
        mask_valid_cpi = np.ones(data.shape, dtype=bool)
    else:
        mask_valid_cpi = mask_valid_cpi.astype(bool, copy=False)

    if mask_valid_cpi.shape != data.shape:
        raise ValueError(f"CPI mask shape {mask_valid_cpi.shape} != CPI data shape {data.shape}")

    if not (0.0 < off_diag_overlap_ratio <= 1.0):
        raise ValueError("off_diag_overlap_ratio must be between 0 and 1.")
    if not (0.0 < diag_valid_ratio <= 1.0):
        raise ValueError("diag_valid_ratio must be between 0 and 1.")

    min_valid_off_diag = max(1, int(np.ceil(off_diag_overlap_ratio * num_rng_samples)))
    min_valid_diag = max(1, int(np.ceil(diag_valid_ratio * num_rng_samples)))

    # Zero-out invalid samples
    x_valid = data * mask_valid_cpi

    # Overlap counts per SCM entry
    mask_int = mask_valid_cpi.astype(np.int32)
    overlap_counts = mask_int @ mask_int.T

    # Unnormalized conjugate products
    scm_sum = x_valid @ x_valid.conj().T

    scm = np.zeros((num_pulses, num_pulses), dtype=np.complex64)

    diag_idx = np.diag_indices(num_pulses)
    diag_counts = overlap_counts[diag_idx]
    diag_sum = scm_sum[diag_idx]

    diag_valid_idx = diag_counts >= min_valid_diag
    diag_vals = np.zeros(num_pulses, dtype=np.complex64)
    diag_vals[diag_valid_idx] = diag_sum[diag_valid_idx] / diag_counts[diag_valid_idx]
    scm[diag_idx] = diag_vals

    off_diag_valid = overlap_counts >= min_valid_off_diag
    np.fill_diagonal(off_diag_valid, False)
    scm[off_diag_valid] = scm_sum[off_diag_valid] / overlap_counts[off_diag_valid]

    # Ensure Hermitian numerically
    scm = (0.5 * (scm + scm.conj().T)).astype(np.complex64)

    diag_valid_frac = float(np.mean(diag_valid_idx))

    return scm, diag_valid_idx, diag_valid_frac


def eigen_decompose_descending(scm: np.ndarray) -> np.ndarray:
    """
    Eigenvalue decomposition of a Hermitian SCM, returned in descending order.

    Parameters
    ----------
    scm : (M, M) complex array

    Returns
    -------
    eigvals : (M,) float64 array, descending order
    """
    eigvals = np.linalg.eigvalsh(scm)
    return np.sort(np.real(eigvals))[::-1]


def eigvals_to_db(eigvals: np.ndarray) -> np.ndarray:
    """
    Convert eigenvalues to dB (un-normalized power in dB).

    Parameters
    ----------
    eigvals : (M,) float array, descending order, linear scale

    Returns
    -------
    eigvals_db : (M,) float32 array
    """
    eigvals_db = 10.0 * np.log10(np.clip(eigvals, EPS, None))
    return eigvals_db.astype(np.float32)

# Default pulse read chunk (keeps memory bounded for large pulse ranges).
PULSE_CHUNK_DEFAULT = 8192


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Profile per-CPI eigenvalue shapes over a targeted pulse and '
                    'range window of a NISAR L0B scene. Produces average-shape, '
                    'all-overlaid, and grid-of-shapes eigenvalue plots plus an '
                    'SCM-diagonal plot. No model, no classification.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('l0b_file', help='Input NISAR L0B HDF5 granule')

    parser.add_argument('--freq', choices=['A', 'B'], default=None,
                        help='Default: every frequency in the granule.')
    parser.add_argument('--pol', default=None,
                        help='Default: every polarization in the granule.')

    parser.add_argument('--pulse-start', type=int, required=True)
    parser.add_argument('--pulse-end', type=int, required=True)
    parser.add_argument('--range-start', type=int, default=None,
                        help='Default: 0 (start of the swath).')
    parser.add_argument('--range-end', type=int, default=None,
                        help='Default: the full range extent of the granule.')

    parser.add_argument('--cpi-len', type=int, default=CPI_LEN_DEFAULT)
    parser.add_argument('--cpi-width', type=int, default=CPI_WIDTH_DEFAULT)
    parser.add_argument('--pulse-chunk', type=int, default=PULSE_CHUNK_DEFAULT)

    parser.add_argument('--compute-subswath-mask', action='store_true',
                        help='Gap-exclusion SCM via subswath / gap masking.')
    parser.add_argument('--off-diag-overlap-ratio', type=float,
                        default=OFF_DIAG_OVERLAP_RATIO_DEFAULT)
    parser.add_argument('--diag-valid-ratio', type=float,
                        default=DIAG_VALID_RATIO_DEFAULT)

    # Plot controls
    parser.add_argument('--max-grid', type=int, default=48,
                        help='Max number of individual shapes drawn in the grid '
                             'plot (evenly sampled across the scene if exceeded).')
    parser.add_argument('--grid-cols', type=int, default=6,
                        help='Number of columns in the grid-of-shapes plot.')
    parser.add_argument('--output-dir', default='results/scene')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# L0B READING (integration point)
# ---------------------------------------------------------------------------

def open_raw(l0b_file):
    """Open a NISAR L0B granule with the ISCE3 / nisar Raw reader."""
    try:
        from nisar.products.readers.Raw import Raw
    except Exception as exc:
        raise RuntimeError(
            "Could not import the nisar Raw reader. Align open_raw() with the "
            "L0B reader used by read_nisar_swaths_isce3.py in this project. "
            "Underlying import error: {}".format(exc)
        )
    return Raw(hdf5file=l0b_file)


def resolve_freq_pols(raw, freq_arg, pol_arg):
    """Return the list of (freq, pol) pairs to process."""
    try:
        pol_map = dict(raw.polarizations)
    except Exception as exc:
        raise RuntimeError(
            "Could not read frequency/polarization map from the granule: {}".format(exc)
        )

    freqs = [freq_arg] if freq_arg is not None else sorted(pol_map.keys())
    pairs = []
    for freq in freqs:
        if freq not in pol_map:
            print("[warn] frequency {} not in granule; skipping".format(freq))
            continue
        pols = list(pol_map[freq])
        if pol_arg is not None:
            if pol_arg in pols:
                pols = [pol_arg]
            else:
                print("[warn] pol {} not in frequency {}; skipping".format(pol_arg, freq))
                continue
        for pol in pols:
            pairs.append((freq, pol))
    if not pairs:
        raise RuntimeError("No matching frequency/polarization to process.")
    return pairs


def read_scene_block(ds, p0, p1, r0, r1, pulse_chunk):
    """Read and BFPQLUT-decode a [p0:p1, r0:r1] window in pulse chunks."""
    n_p = p1 - p0
    n_r = r1 - r0
    out = np.empty((n_p, n_r), dtype=np.complex64)
    for cs in range(p0, p1, pulse_chunk):
        ce = min(cs + pulse_chunk, p1)
        out[cs - p0:ce - p0, :] = np.asarray(ds[cs:ce, r0:r1], dtype=np.complex64)
    return out


# ---------------------------------------------------------------------------
# VALIDITY MASK
# ---------------------------------------------------------------------------

def amplitude_gap_mask(data_block, gap_frac=0.10):
    """
    Self-contained validity mask based on ADC fill level.

    Inter-subswath transmission gaps show up as near-zero (not exactly zero)
    ADC fill. Mark range samples whose mean magnitude across pulses falls below
    gap_frac of the peak mean magnitude as invalid. Returns a bool mask the same
    shape as data_block (True = valid).
    """
    mag = np.abs(data_block)
    col_mean = mag.mean(axis=0)
    peak = float(col_mean.max())
    if peak <= 0.0:
        return np.ones(data_block.shape, dtype=bool)
    valid_cols = col_mean >= (gap_frac * peak)
    return np.broadcast_to(valid_cols[None, :], data_block.shape).copy()


def build_valid_mask(data_block, raw, freq, pol, use_subswath):
    """
    Build a per-sample validity mask for gap-exclusion SCM.

    Best effort: probe subswath geometry first (to match training), then fall
    back to the amplitude-based gap detection documented for this project.
    Returns None when masking is disabled. For exact parity with training-set
    masking, wire the subswath-to-mask logic from read_nisar_swaths_isce3.py in
    place of the amplitude fallback below.
    """
    if not use_subswath:
        return None
    try:
        _ = raw.getSubSwaths(freq, pol[0])
        print("[info] subswath info available; using amplitude gap mask for this "
              "window (wire read_nisar_swaths_isce3.py for exact parity).")
    except Exception:
        print("[info] subswath info unavailable; using amplitude gap mask.")
    return amplitude_gap_mask(data_block)


# ---------------------------------------------------------------------------
# PER-CPI EIGENVALUE PROFILES AND SCM DIAGONALS
# ---------------------------------------------------------------------------

def process_freq_pol(data_block, mask_block, p0, r0, cpi_len, cpi_width,
                     off_diag_ratio, diag_ratio):
    """
    Tile data_block into non-overlapping cpi_len x cpi_width CPIs and extract,
    per CPI: descending eigenvalues (power in dB) and the SCM diagonal (power in dB).
    SCM is computed as M^H*M/250. Returns a dict of stacked arrays, or None if no
    full tile fits.
    """
    n_p, n_r = data_block.shape
    n_pt = n_p // cpi_len
    n_rt = n_r // cpi_width

    eig_lin_list = []       # (16,) linear eigenvalues, descending
    eig_db_list = []        # (16,) eigenvalues in dB (un-normalized power)
    diag_lin_list = []      # (16,) per-pulse SCM diagonal power (linear)
    diag_db_list = []       # (16,) per-pulse SCM diagonal power (dB)
    diag_valid_list = []
    pulse_idx_list = []
    range_idx_list = []

    for pt in range(n_pt):
        ps = pt * cpi_len
        pe = ps + cpi_len
        for rt in range(n_rt):
            rs = rt * cpi_width
            re = rs + cpi_width

            cpi = data_block[ps:pe, rs:re]
            cpi_mask = None if mask_block is None else mask_block[ps:pe, rs:re]

            scm, _diag_valid_idx, diag_valid_frac = compute_gap_exclusion_scm(
                cpi,
                mask_valid_cpi=cpi_mask,
                off_diag_overlap_ratio=off_diag_ratio,
                diag_valid_ratio=diag_ratio,
            )

            # Normalize SCM by number of range samples (M^H*M/250)
            scm = scm / cpi_width

            eigvals = eigen_decompose_descending(scm)      # (16,) linear
            eig_db = eigvals_to_db(eigvals)                # power in dB
            diag_lin = np.real(np.diag(scm)).astype(np.float64)
            diag_db = 10.0 * np.log10(np.clip(diag_lin, EPS, None))

            eig_lin_list.append(eigvals.astype(np.float64))
            eig_db_list.append(eig_db.astype(np.float32))
            diag_lin_list.append(diag_lin)
            diag_db_list.append(diag_db.astype(np.float32))
            diag_valid_list.append(diag_valid_frac)
            pulse_idx_list.append(p0 + ps)
            range_idx_list.append(r0 + rs)

    if not eig_lin_list:
        return None

    eig_lin = np.stack(eig_lin_list)             # (N, 16)
    eig_db = np.stack(eig_db_list)               # (N, 16)
    diag_lin = np.stack(diag_lin_list)           # (N, 16)
    diag_db = np.stack(diag_db_list)             # (N, 16)

    return {
        "eig_lin": eig_lin,
        "eig_db": eig_db,
        "diag_lin": diag_lin.astype(np.float32),
        "diag_db": diag_db.astype(np.float32),
        "diag_valid_frac": np.array(diag_valid_list, dtype=np.float32),
        "pulse_idx": np.array(pulse_idx_list, dtype=np.int64),
        "range_idx": np.array(range_idx_list, dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# PLOTTING (first N_SHOW values only)
# ---------------------------------------------------------------------------

def plot_average_shape(eig_db, out_path, freq, pol, n_show):
    """
    Average eigenvalue shape: mean profile with a +/-1 std band and a faint
    min/max envelope, plus the median.
    """
    eig = eig_db[:, :n_show]
    x = np.arange(n_show)

    mean = eig.mean(axis=0)
    std = eig.std(axis=0)
    med = np.median(eig, axis=0)
    lo = eig.min(axis=0)
    hi = eig.max(axis=0)

    # Dynamic y-limits based on 12 largest eigenvalues (descending order)
    eig_12 = eig_db[:, :12]  # Top 12 eigenvalues
    y_max = np.ceil(np.max(eig_12))
    y_min = np.floor(np.min(eig_12))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill_between(x, lo, hi, color="C0", alpha=0.12, label="min / max envelope")
    ax.fill_between(x, mean - std, mean + std, color="C0", alpha=0.30,
                    label="mean +/- 1 std")
    ax.plot(x, mean, color="C0", lw=2.2, label="mean")
    ax.plot(x, med, color="C3", lw=1.4, ls="--", label="median")

    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue (dB, power)")
    ax.set_title("Average eigenvalue shape: freq {} pol {} ({} CPIs, all {})"
                 .format(freq, pol, eig.shape[0], n_show))
    ax.set_xticks(x)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_overlaid(eig_db, out_path, freq, pol, n_show):
    """All CPI eigenvalue profiles overlaid on a single axes."""
    eig = eig_db[:, :n_show]
    x = np.arange(n_show)

    # Alpha thins out as the number of overlaid lines grows so the density of
    # the "mountain" remains readable.
    n = eig.shape[0]
    alpha = float(np.clip(30.0 / max(n, 1), 0.03, 0.5))

    # Dynamic y-limits based on 12 largest eigenvalues (descending order)
    eig_12 = eig_db[:, :12]  # Top 12 eigenvalues
    y_max = np.ceil(np.max(eig_12))
    y_min = np.floor(np.min(eig_12))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(n):
        ax.plot(x, eig[i], color="C0", alpha=alpha, lw=0.8)
    ax.plot(x, eig.mean(axis=0), color="k", lw=1.6, label="mean")

    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue (dB, power)")
    ax.set_title("All eigenvalue profiles overlaid: freq {} pol {} ({} CPIs, all {})"
                 .format(freq, pol, n, n_show))
    ax.set_xticks(x)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_grid(eig_db, diag_db, cpi_pulse_idx, cpi_range_idx, out_path, freq, pol,
              n_show, max_grid, grid_cols):
    """
    Grid of small multiples: one CPI eigenvalue profile next to SCM diagonal
    scatter plot per row. If there are more CPIs than max_grid, the shown CPIs
    are sampled evenly across the scene so the grid stays representative.

    Splits into multiple images with at most max_per_image CPIs per image to avoid
    excessively long plots.
    """
    eig = eig_db[:, :n_show]
    diag = diag_db[:, :n_show]
    x = np.arange(n_show)
    N = eig.shape[0]

    if N <= max_grid:
        sel = np.arange(N)
    else:
        sel = np.unique(np.linspace(0, N - 1, max_grid).round().astype(int))

    n_total = len(sel)

    # Split into multiple images with at most 5 CPIs per image
    max_per_image = 5
    n_images = int(np.ceil(n_total / max_per_image))

    # Generate base filename without extension
    out_base = os.path.splitext(out_path)[0]
    out_ext = os.path.splitext(out_path)[1]

    output_files = []

    for img_idx in range(n_images):
        start_idx = img_idx * max_per_image
        end_idx = min(start_idx + max_per_image, n_total)
        sel_chunk = sel[start_idx:end_idx]
        n_in_chunk = len(sel_chunk)

        # Compute dynamic y-limits for this chunk based on 12 largest eigenvalues
        eig_chunk_12 = eig_db[sel_chunk, :12]
        diag_chunk_12 = np.sort(diag_db[sel_chunk, :])[:, ::-1][:, :12]  # Sort each CPI descending, take top 12
        combined_12 = np.concatenate([eig_chunk_12.flatten(), diag_chunk_12.flatten()])
        y_max = np.ceil(np.max(combined_12))
        y_min = np.floor(np.min(combined_12))

        n_cols = 2  # Two columns: eigenvalue plot and SCM diagonal scatter plot

        fig, axes = plt.subplots(n_in_chunk, n_cols,
                                 figsize=(4.8, 1.9 * n_in_chunk),
                                 squeeze=False, sharex=True, sharey=True)

        for k, cpi_i in enumerate(sel_chunk):
            # Left column: eigenvalue plot
            ax_eig = axes[k][0]
            ax_eig.plot(x, eig[cpi_i], color="C0", lw=1.1)
            ax_eig.set_ylim(y_min, y_max)
            ax_eig.set_title("CPI {} Eigenvalues (p{}, r{})".format(
                cpi_i, cpi_pulse_idx[cpi_i], cpi_range_idx[cpi_i]), fontsize=7)
            ax_eig.tick_params(labelsize=6)
            ax_eig.grid(True, alpha=0.3)

            # Right column: SCM diagonal scatter plot (in matrix diagonal order, not sorted)
            ax_diag = axes[k][1]
            ax_diag.scatter(x, diag[cpi_i], color="C3", s=15, alpha=0.7)
            ax_diag.set_ylim(y_min, y_max)
            ax_diag.set_title("CPI {} SCM Diagonal (p{}, r{})".format(
                cpi_i, cpi_pulse_idx[cpi_i], cpi_range_idx[cpi_i]), fontsize=7)
            ax_diag.tick_params(labelsize=6)
            ax_diag.grid(True, alpha=0.3)

        subtitle = "" if N <= max_grid else " (showing {}/{}, evenly sampled)".format(n_total, N)
        part_info = " - Part {}/{}".format(img_idx + 1, n_images) if n_images > 1 else ""
        fig.suptitle("Eigenvalue shapes & SCM diagonal grid: freq {} pol {}{}{}  [all {}]"
                     .format(freq, pol, subtitle, part_info, n_show), fontsize=11)
        fig.text(0.5, 0.01, "Index", ha="center", fontsize=9)
        fig.text(0.01, 0.5, "Power (dB)", va="center", rotation="vertical", fontsize=9)
        fig.tight_layout(rect=[0.02, 0.02, 1, 0.97])

        # Save with part number if multiple images
        if n_images > 1:
            out_file = "{}_part{:02d}{}".format(out_base, img_idx + 1, out_ext)
        else:
            out_file = out_path

        fig.savefig(out_file, dpi=130)
        plt.close(fig)
        output_files.append(out_file)

    return output_files


def plot_scm_diag(diag_db, out_path, freq, pol, n_show):
    """SCM diagonal values per CPI, all n_show diagonal positions."""
    d = diag_db[:, :n_show]
    x = np.arange(n_show)
    n = d.shape[0]
    alpha = float(np.clip(30.0 / max(n, 1), 0.03, 0.5))

    # Dynamic y-limits based on 12 largest values (sorted descending per CPI)
    d_sorted_12 = np.sort(d, axis=1)[:, ::-1][:, :12]  # Sort each CPI descending, take top 12
    y_max = np.ceil(np.max(d_sorted_12))
    y_min = np.floor(np.min(d_sorted_12))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(n):
        ax.plot(x, d[i], color="C3", alpha=alpha, lw=0.8)
    ax.plot(x, d.mean(axis=0), color="k", lw=1.6, label="mean")

    ax.set_xlabel("Pulse index (SCM diagonal position)")
    ax.set_ylabel("SCM diagonal power (dB)")
    ax.set_title("SCM diagonal per CPI: freq {} pol {} ({} CPIs, all {})"
                 .format(freq, pol, n, n_show))
    ax.set_xticks(x)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# STORAGE
# ---------------------------------------------------------------------------

def store_profiles(out_h5, res, freq, pol, cpi_len, cpi_width):
    """Store per-CPI eigenvalue profiles and SCM diagonals to HDF5."""
    with h5py.File(out_h5, "w") as h:
        h.attrs["freq"] = freq
        h.attrs["pol"] = pol
        h.attrs["cpi_len"] = cpi_len
        h.attrs["cpi_width"] = cpi_width
        h.attrs["n_keep"] = N_KEEP_DEFAULT
        h.attrs["n_show"] = N_SHOW
        h.attrs["scm_normalization"] = "M^H*M/cpi_width"
        h.create_dataset("eigvals_linear", data=res["eig_lin"].astype(np.float32))
        h.create_dataset("eigvals_db", data=res["eig_db"])
        h.create_dataset("scm_diag_linear", data=res["diag_lin"])
        h.create_dataset("scm_diag_db", data=res["diag_db"])
        h.create_dataset("diag_valid_frac", data=res["diag_valid_frac"])
        h.create_dataset("cpi_pulse_idx", data=res["pulse_idx"])
        h.create_dataset("cpi_range_idx", data=res["range_idx"])


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    raw = open_raw(args.l0b_file)
    pairs = resolve_freq_pols(raw, args.freq, args.pol)

    for freq, pol in pairs:
        print("[info] processing freq {} pol {}".format(freq, pol))
        ds = raw.getRawDataset(freq, pol)
        n_p_total, n_r_total = ds.shape

        p0 = max(0, args.pulse_start)
        p1 = min(n_p_total, args.pulse_end)
        r0 = 0 if args.range_start is None else max(0, args.range_start)
        r1 = n_r_total if args.range_end is None else min(n_r_total, args.range_end)

        if p1 - p0 < args.cpi_len or r1 - r0 < args.cpi_width:
            print("[warn] window {}x{} too small for CPI {}x{}; skipping."
                  .format(p1 - p0, r1 - r0, args.cpi_len, args.cpi_width))
            continue

        data_block = read_scene_block(ds, p0, p1, r0, r1, args.pulse_chunk)
        mask_block = build_valid_mask(data_block, raw, freq, pol,
                                      args.compute_subswath_mask)

        res = process_freq_pol(
            data_block, mask_block, p0, r0,
            args.cpi_len, args.cpi_width,
            args.off_diag_overlap_ratio, args.diag_valid_ratio,
        )
        if res is None:
            print("[warn] no CPIs extracted for freq {} pol {}.".format(freq, pol))
            continue

        tag = "{}_{}".format(freq, pol)
        p_avg = os.path.join(args.output_dir, "eig_average_{}.png".format(tag))
        p_over = os.path.join(args.output_dir, "eig_overlaid_{}.png".format(tag))
        p_grid = os.path.join(args.output_dir, "eig_grid_{}.png".format(tag))
        p_diag = os.path.join(args.output_dir, "scm_diagonal_{}.png".format(tag))
        p_h5 = os.path.join(args.output_dir, "mountain_profiles_{}.h5".format(tag))

        plot_average_shape(res["eig_db"], p_avg, freq, pol, N_SHOW)
        plot_overlaid(res["eig_db"], p_over, freq, pol, N_SHOW)

        grid_files = plot_grid(res["eig_db"], res["diag_db"], res["pulse_idx"], res["range_idx"],
                               p_grid, freq, pol, N_SHOW, args.max_grid, args.grid_cols)

        plot_scm_diag(res["diag_db"], p_diag, freq, pol, N_SHOW)

        store_profiles(p_h5, res, freq, pol, args.cpi_len, args.cpi_width)

        n_cpi = res["eig_lin"].shape[0]
        print("[info]   {} CPIs -> {}".format(n_cpi, p_h5))
        grid_str = " | ".join(grid_files) if len(grid_files) > 1 else grid_files[0]
        print("[info]   plots: {} | {} | {} | {}"
              .format(p_avg, p_over, grid_str, p_diag))


if __name__ == "__main__":
    main()