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

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py

from anomaly_features import (
    CPI_LEN_DEFAULT,
    CPI_WIDTH_DEFAULT,
    OFF_DIAG_OVERLAP_RATIO_DEFAULT,
    DIAG_VALID_RATIO_DEFAULT,
    N_KEEP_DEFAULT,
    EPS,
    compute_gap_exclusion_scm,
    eigen_decompose_descending,
    normalize_eigvals_db,
)

# Number of leading values shown in every plot (largest 12 eigenvalues and the
# first 12 SCM diagonal entries). Tied to N_KEEP so the plots share the same
# feature basis as anomaly_features.py.
N_SHOW = N_KEEP_DEFAULT

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
    parser.add_argument('--diag-raw', action='store_true',
                        help='Plot the SCM diagonal as raw power in dB instead of '
                             'normalizing each CPI to its top eigenvalue (0 dB).')

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
    per CPI: descending eigenvalues, the (max eigenvalue -> 0 dB) normalized
    eigenvalue profile, and the SCM diagonal. Returns a dict of stacked arrays,
    or None if no full tile fits.
    """
    n_p, n_r = data_block.shape
    n_pt = n_p // cpi_len
    n_rt = n_r // cpi_width

    eig_lin_list = []       # (16,) linear eigenvalues, descending
    eig_db_plot_list = []   # (16,) max eigenvalue -> 0 dB (anomaly-style)
    diag_lin_list = []      # (16,) per-pulse SCM diagonal power
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

            eigvals = eigen_decompose_descending(scm)      # (16,) linear
            eig_db_plot = normalize_eigvals_db(eigvals)    # max -> 0 dB
            diag_lin = np.real(np.diag(scm)).astype(np.float64)

            eig_lin_list.append(eigvals.astype(np.float64))
            eig_db_plot_list.append(eig_db_plot.astype(np.float32))
            diag_lin_list.append(diag_lin)
            diag_valid_list.append(diag_valid_frac)
            pulse_idx_list.append(p0 + ps)
            range_idx_list.append(r0 + rs)

    if not eig_lin_list:
        return None

    eig_lin = np.stack(eig_lin_list)             # (N, 16)
    eig_db_plot = np.stack(eig_db_plot_list)     # (N, 16)
    diag_lin = np.stack(diag_lin_list)           # (N, 16)

    # SCM diagonal in dB. For a Hermitian PSD SCM every diagonal entry is <= the
    # top eigenvalue, so normalizing to it keeps the diagonal plot on the same
    # 0 dB reference as the eigenvalue plots. Raw dB is kept too for absolute
    # power inspection.
    lam_max = np.clip(eig_lin[:, :1], EPS, None)                    # (N, 1)
    diag_db_norm = 10.0 * np.log10(np.clip(diag_lin / lam_max, EPS, None))
    diag_db_raw = 10.0 * np.log10(np.clip(diag_lin, EPS, None))

    return {
        "eig_lin": eig_lin,
        "eig_db_plot": eig_db_plot,
        "diag_lin": diag_lin.astype(np.float32),
        "diag_db_norm": diag_db_norm.astype(np.float32),
        "diag_db_raw": diag_db_raw.astype(np.float32),
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

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill_between(x, lo, hi, color="C0", alpha=0.12, label="min / max envelope")
    ax.fill_between(x, mean - std, mean + std, color="C0", alpha=0.30,
                    label="mean +/- 1 std")
    ax.plot(x, mean, color="C0", lw=2.2, label="mean")
    ax.plot(x, med, color="C3", lw=1.4, ls="--", label="median")

    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue (dB, max -> 0 dB)")
    ax.set_title("Average eigenvalue shape: freq {} pol {} ({} CPIs, first {})"
                 .format(freq, pol, eig.shape[0], n_show))
    ax.set_xticks(x)
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

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(n):
        ax.plot(x, eig[i], color="C0", alpha=alpha, lw=0.8)
    ax.plot(x, eig.mean(axis=0), color="k", lw=1.6, label="mean")

    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue (dB, max -> 0 dB)")
    ax.set_title("All eigenvalue profiles overlaid: freq {} pol {} ({} CPIs, first {})"
                 .format(freq, pol, n, n_show))
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_grid(eig_db, cpi_pulse_idx, cpi_range_idx, out_path, freq, pol,
              n_show, max_grid, grid_cols):
    """
    Grid of small multiples: one CPI eigenvalue profile per cell. If there are
    more CPIs than max_grid, the shown CPIs are sampled evenly across the scene
    so the grid stays representative. All cells share y-limits for comparison.
    """
    eig = eig_db[:, :n_show]
    x = np.arange(n_show)
    N = eig.shape[0]

    if N <= max_grid:
        sel = np.arange(N)
    else:
        sel = np.unique(np.linspace(0, N - 1, max_grid).round().astype(int))

    n = len(sel)
    n_cols = int(min(grid_cols, n))
    n_rows = int(np.ceil(n / float(n_cols)))

    y_lo = float(eig[sel].min())
    y_hi = float(eig[sel].max())
    pad = 0.05 * (y_hi - y_lo + 1e-6)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.4 * n_cols, 1.9 * n_rows),
                             squeeze=False, sharex=True, sharey=True)

    for k, cpi_i in enumerate(sel):
        r, c = divmod(k, n_cols)
        ax = axes[r][c]
        ax.plot(x, eig[cpi_i], color="C0", lw=1.1)
        ax.set_ylim(y_lo - pad, y_hi + pad)
        ax.set_title("CPI {} (p{}, r{})".format(
            cpi_i, cpi_pulse_idx[cpi_i], cpi_range_idx[cpi_i]), fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

    for k in range(n, n_rows * n_cols):
        r, c = divmod(k, n_cols)
        axes[r][c].axis("off")

    subtitle = "" if N <= max_grid else " (showing {}/{}, evenly sampled)".format(n, N)
    fig.suptitle("Eigenvalue shapes grid: freq {} pol {}{}  [first {}]"
                 .format(freq, pol, subtitle, n_show), fontsize=11)
    fig.text(0.5, 0.01, "Eigenvalue index", ha="center", fontsize=9)
    fig.text(0.01, 0.5, "Eigenvalue (dB, max -> 0 dB)", va="center", rotation="vertical", fontsize=9)
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_scm_diag(diag_db, out_path, freq, pol, n_show, ylabel):
    """SCM diagonal values per CPI, first n_show diagonal positions."""
    d = diag_db[:, :n_show]
    x = np.arange(n_show)
    n = d.shape[0]
    alpha = float(np.clip(30.0 / max(n, 1), 0.03, 0.5))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(n):
        ax.plot(x, d[i], color="C3", alpha=alpha, lw=0.8)
    ax.plot(x, d.mean(axis=0), color="k", lw=1.6, label="mean")

    ax.set_xlabel("Pulse index (SCM diagonal position)")
    ax.set_ylabel(ylabel)
    ax.set_title("SCM diagonal per CPI: freq {} pol {} ({} CPIs, first {})"
                 .format(freq, pol, n, n_show))
    ax.set_xticks(x)
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
        h.attrs["eig_norm"] = "max_eigenvalue_to_0dB"
        h.create_dataset("eigvals_linear", data=res["eig_lin"].astype(np.float32))
        h.create_dataset("eigvals_db", data=res["eig_db_plot"])
        h.create_dataset("scm_diag_linear", data=res["diag_lin"])
        h.create_dataset("scm_diag_db_norm", data=res["diag_db_norm"])
        h.create_dataset("scm_diag_db_raw", data=res["diag_db_raw"])
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

        plot_average_shape(res["eig_db_plot"], p_avg, freq, pol, N_SHOW)
        plot_overlaid(res["eig_db_plot"], p_over, freq, pol, N_SHOW)
        plot_grid(res["eig_db_plot"], res["pulse_idx"], res["range_idx"],
                  p_grid, freq, pol, N_SHOW, args.max_grid, args.grid_cols)

        if args.diag_raw:
            plot_scm_diag(res["diag_db_raw"], p_diag, freq, pol, N_SHOW,
                          "SCM diagonal power (dB)")
        else:
            plot_scm_diag(res["diag_db_norm"], p_diag, freq, pol, N_SHOW,
                          "SCM diagonal (dB, max eigenvalue -> 0 dB)")

        store_profiles(p_h5, res, freq, pol, args.cpi_len, args.cpi_width)

        n_cpi = res["eig_lin"].shape[0]
        print("[info]   {} CPIs -> {}".format(n_cpi, p_h5))
        print("[info]   plots: {} | {} | {} | {}"
              .format(p_avg, p_over, p_grid, p_diag))


if __name__ == "__main__":
    main()