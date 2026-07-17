#!/usr/bin/env python
"""
plot_condition_number_colormap.py

Render per-CPI condition number (dB) as a 2D colormap: CPI index (slow time)
on the x-axis, range block index on the y-axis -- the same layout as the
"Detection Thresholds" heatmaps in the IGARSS 2023 slides, but built from
the project's condition_number_db metric (max / 12th-kept-eigenvalue, in dB)
rather than the ST-EST alpha threshold.

Why this exists
---------------
Summary statistics (mean/median/p05/p95) can show that two classes overlap
in feature space, but they cannot show WHERE that overlap comes from. If a
"clean"-labeled region's high-condition-number tiles are scattered randomly
across a scene, that is more consistent with real terrain/clutter texture.
If they cluster in a compact spatial region, that is more consistent with a
real (even if weak) interference source that slipped past the clean-tile
selection. This script exists to make that spatial pattern visible directly,
rather than inferring it from a distribution shape.

Two input modes, matching the project's other tools:
  --mode nisar         : raw NISAR L0B HDF5 granule; tiles CPIs on the fly
                         over the requested pulse/range window (16 x 250,
                         per project convention). Produces a dense grid
                         heatmap (imshow), since every tile in the window is
                         computed, not just a pre-selected subset.
  --mode preprocessed  : an existing per-CPI file (clean_mountains*.h5 or
                         rfi_data_<freq>_<pol>.h5). Produces a SCATTER plot
                         at each tile's actual (pulse_tile, range_tile)
                         location, since these files often hold only a
                         sparse subset of tiles (e.g. clean-only selections),
                         not a complete grid. Sparsity itself is informative
                         here: gaps show where no tile passed selection.

If the preprocessed file carries a "labels" dataset (generate_amazon_data.py
/ generate_mountain_data.py output), tiles with an injected RFI label
(--rfi-label-min..--rfi-label-max) get an extra black outline, so injected
RFI tiles and their surrounding clean neighborhood can be compared directly
in the same figure.

--highlight-above draws a black outline around any tile (regardless of mode
or label) whose condition number exceeds the given dB value, e.g. to flag
tiles sitting in the overlap zone identified from compare_global_metrics.py.

Usage
-----
  Raw NISAR L0B, dense grid over a scene window:
    python plot_condition_number_colormap.py --mode nisar \\
        --input NISAR_L0_..._h5 --freq A --pol HH \\
        --pulse-start 813924 --pulse-end 888222 \\
        --range-start 2000 --range-end 25000 \\
        --output amazon_hh_train_condnum.png

  Preprocessed clean-tile selection, sparse scatter, flagging overlap tiles:
    python plot_condition_number_colormap.py --mode preprocessed \\
        --input clean_mountains_filtered.h5 --freq A --pol HH \\
        --highlight-above 12.0 \\
        --output mountains_czech_hh_condnum.png

  Preprocessed RFI training file, outlining injected-RFI tiles:
    python plot_condition_number_colormap.py --mode preprocessed \\
        --input rfi_data_A_HH.h5 --freq A --pol HH \\
        --output amazon_hh_rfi_condnum.png
"""

import argparse
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


CPI_PULSES = 16
CPI_RANGE = 250

DEFAULT_N_KEEP = 12
DEFAULT_DIAG_VALID_FRAC_THRESH = 0.8
DEFAULT_GAP_MAG_FRAC = 0.10

DEFAULT_RFI_LABEL_MIN = 1
DEFAULT_RFI_LABEL_MAX = 6

EPS = 1e-12


# ---------------------------------------------------------------------------
# Metric core (matches compare_global_metrics.py's definition)
# ---------------------------------------------------------------------------

def compute_condition_number_db(eig_kept):
    """
    eig_kept : (n_cpi, n_keep) linear-scale eigenvalues, descending along axis 1.
    Returns  : (n_cpi,) condition number in dB (max / smallest-kept eigenvalue).
    """
    eig_max = eig_kept[:, 0]
    eig_min_kept = eig_kept[:, -1]
    with np.errstate(divide="ignore", invalid="ignore"):
        cond_db = 10.0 * np.log10(eig_max / eig_min_kept)
    return cond_db


# ---------------------------------------------------------------------------
# Raw NISAR L0B loader (dense grid over the requested window)
# ---------------------------------------------------------------------------

def compute_gap_exclusion_cov_simple(pulse_block, gap_mag_frac=DEFAULT_GAP_MAG_FRAC):
    """
    Simplified gap-exclusion SCM (per-pulse validity, not per-element).
    See compare_global_metrics.py's docstring for the caveat versus the
    project's real compute_gap_exclusion_cov (3 percent off-diag / 2 percent
    diag element-wise ratios). Swap in the real function for exact parity.
    """
    n_pulses, n_range = pulse_block.shape
    mean_mag = np.abs(pulse_block).mean(axis=1)
    max_mag = mean_mag.max() if mean_mag.size else 0.0
    gap_thresh = max_mag * gap_mag_frac
    valid_pulse = mean_mag >= gap_thresh

    diag_valid_frac = float(valid_pulse.mean()) if n_pulses else 0.0

    cov = np.full((n_pulses, n_pulses), np.nan, dtype=complex)
    valid_idx = np.where(valid_pulse)[0]
    if valid_idx.size > 0:
        s_valid = pulse_block[valid_idx, :]
        r_valid = (s_valid @ s_valid.conj().T) / n_range
        cov[np.ix_(valid_idx, valid_idx)] = r_valid

    return cov, diag_valid_frac


def load_nisar_grid(path, freq, pol, diag_valid_frac_thresh, n_keep,
                     pulse_start, pulse_end, range_start, range_end,
                     gap_mag_frac):
    """
    Tile a raw NISAR L0B granule into a DENSE (n_cpi_pulse, n_cpi_range) grid
    of condition_number_db values (NaN where a tile fails the validity check).

    Requires isce3 to decode the BFPQLUT-compressed raw data; run inside the
    project's 'isce3' conda environment (py-isce3) on nisar-adt-dev-5.
    """
    try:
        from nisar.products.readers.Raw import Raw
    except ImportError as exc:
        raise RuntimeError(
            "nisar.products.readers.Raw is required for --mode nisar. "
            "Run inside the project's 'isce3' conda environment (py-isce3)."
        ) from exc

    raw = Raw(hdf5file=path)
    raw.parsePolarizations()
    raw_dataset = raw.getRawDataset(freq, pol)
    n_pulses_total, n_range_total = raw_dataset.shape

    p_start = 0 if pulse_start is None else pulse_start
    p_end = n_pulses_total if pulse_end is None else min(pulse_end, n_pulses_total)
    r_start = 0 if range_start is None else range_start
    r_end = n_range_total if range_end is None else min(range_end, n_range_total)

    n_cpi_pulse = (p_end - p_start) // CPI_PULSES
    n_cpi_range = (r_end - r_start) // CPI_RANGE

    if n_cpi_pulse <= 0 or n_cpi_range <= 0:
        raise ValueError("Requested window is smaller than one CPI tile "
                          f"({CPI_PULSES} x {CPI_RANGE}).")

    cond_grid = np.full((n_cpi_range, n_cpi_pulse), np.nan)

    CHUNK_PULSES = 1000
    chunk_cpi = max(1, CHUNK_PULSES // CPI_PULSES)

    print(f"Tiling {n_cpi_pulse} x {n_cpi_range} CPI grid "
          f"({n_cpi_pulse * n_cpi_range} tiles total) ...", file=sys.stderr)

    for chunk_idx in range(0, n_cpi_pulse, chunk_cpi):
        chunk_n = min(chunk_cpi, n_cpi_pulse - chunk_idx)
        chunk_p0 = p_start + chunk_idx * CPI_PULSES
        chunk_p1 = chunk_p0 + chunk_n * CPI_PULSES

        raw_chunk = raw_dataset[chunk_p0:chunk_p1, r_start:r_end]

        for ip_local in range(chunk_n):
            ip = chunk_idx + ip_local
            lp0 = ip_local * CPI_PULSES
            lp1 = lp0 + CPI_PULSES

            for ir in range(n_cpi_range):
                lr0 = ir * CPI_RANGE
                lr1 = lr0 + CPI_RANGE
                block = raw_chunk[lp0:lp1, lr0:lr1]

                cov, dvf = compute_gap_exclusion_cov_simple(block, gap_mag_frac)
                if dvf < diag_valid_frac_thresh:
                    continue

                valid_idx = np.where(~np.isnan(np.diag(cov)))[0]
                if valid_idx.size < n_keep:
                    continue

                sub_cov = cov[np.ix_(valid_idx, valid_idx)]
                eigvals = np.linalg.eigvalsh(sub_cov)
                eigvals = np.sort(np.real(eigvals))[::-1][:n_keep]
                eigvals = np.clip(eigvals, EPS, None)
                cond_db = 10.0 * np.log10(eigvals[0] / eigvals[-1])
                cond_grid[ir, ip] = cond_db

        done = min(chunk_idx + chunk_n, n_cpi_pulse)
        print(f"  {done}/{n_cpi_pulse} CPI columns processed", file=sys.stderr)

    return cond_grid, None, None, None


# ---------------------------------------------------------------------------
# Preprocessed-file loader (sparse scatter; reuses either file layout)
# ---------------------------------------------------------------------------

def _read_preprocessed_arrays(path, freq, pol):
    """Same two layouts compare_global_metrics.py supports."""
    group_name = f"freq_{freq}_pol_{pol}"
    with h5py.File(path, "r") as f:
        if group_name in f:
            g = f[group_name]
            eigenvalues = g["eigenvalues"][:]
            diag_valid_frac = g["diag_valid_frac"][:]
            pulse_idx = g["pulse_idx"][:]
            range_idx = g["range_idx"][:]
            labels = g["labels"][:] if "labels" in g else None
        elif "eigenvalues" in f:
            eigenvalues = f["eigenvalues"][:]
            diag_valid_idx = f["diag_valid_idx"][:]
            diag_valid_frac = diag_valid_idx.mean(axis=1)
            pulse_idx = f["tile_pulse"][:]
            range_idx = f["tile_range"][:]
            labels = f["labels"][:] if "labels" in f else None
        else:
            available = list(f.keys())
            raise KeyError(
                f"Neither group '{group_name}' nor a root-level 'eigenvalues' "
                f"dataset found in {path}. Available top-level keys: {available}"
            )
    return eigenvalues, diag_valid_frac, pulse_idx, range_idx, labels


def load_preprocessed_points(path, freq, pol, diag_valid_frac_thresh, n_keep,
                              rfi_label_min, rfi_label_max):
    """
    Returns per-tile arrays for a scatter plot:
        cpi_idx     : (n_valid,) CPI index = pulse_idx // CPI_PULSES
        range_block : (n_valid,) range block index = range_idx // CPI_RANGE
        cond_db     : (n_valid,) condition number in dB
        is_rfi      : (n_valid,) bool, True if label in [rfi_label_min, rfi_label_max]
                      (all False if the file has no "labels" dataset)
    """
    eigenvalues, diag_valid_frac, pulse_idx, range_idx, labels = \
        _read_preprocessed_arrays(path, freq, pol)

    valid_mask = diag_valid_frac >= diag_valid_frac_thresh
    eig_valid = eigenvalues[valid_mask][:, :n_keep]
    eig_valid = np.clip(eig_valid, EPS, None)
    cond_db = compute_condition_number_db(eig_valid)

    cpi_idx = pulse_idx[valid_mask] // CPI_PULSES
    range_block = range_idx[valid_mask] // CPI_RANGE

    if labels is not None:
        labels_valid = labels[valid_mask]
        is_rfi = (labels_valid >= rfi_label_min) & (labels_valid <= rfi_label_max)
    else:
        is_rfi = np.zeros(cond_db.shape, dtype=bool)

    return cpi_idx, range_block, cond_db, is_rfi


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_dense_grid(cond_grid, output_path, title, highlight_above, vmin, vmax, cmap):
    fig_w = max(6.0, min(18.0, cond_grid.shape[1] / 40.0))
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))

    masked = np.ma.masked_invalid(cond_grid)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#dddddd")

    norm = Normalize(vmin=vmin, vmax=vmax) if (vmin is not None or vmax is not None) else None
    im = ax.imshow(masked, aspect="auto", origin="lower", cmap=cmap_obj, norm=norm,
                    interpolation="nearest")

    if highlight_above is not None:
        hi_mask = np.where(np.nan_to_num(cond_grid, nan=-np.inf) > highlight_above)
        ax.scatter(hi_mask[1], hi_mask[0], facecolors="none", edgecolors="black",
                   linewidths=0.6, s=12, label=f"> {highlight_above} dB")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

    ax.set_xlabel("CPI index")
    ax.set_ylabel("Range block")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Condition number (dB)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sparse_scatter(cpi_idx, range_block, cond_db, is_rfi, output_path, title,
                         highlight_above, vmin, vmax, cmap, marker_size):
    fig_w = max(6.0, min(18.0, (cpi_idx.max() - cpi_idx.min() + 1) / 40.0)) if cpi_idx.size else 8.0
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))

    norm = Normalize(vmin=vmin, vmax=vmax) if (vmin is not None or vmax is not None) else None
    sc = ax.scatter(cpi_idx, range_block, c=cond_db, cmap=cmap, norm=norm,
                     marker="s", s=marker_size, edgecolors="none")

    if is_rfi.any():
        ax.scatter(cpi_idx[is_rfi], range_block[is_rfi], facecolors="none",
                   edgecolors="black", linewidths=0.8, marker="s", s=marker_size * 1.6,
                   label="labeled RFI tile")

    if highlight_above is not None:
        hi = cond_db > highlight_above
        if hi.any():
            ax.scatter(cpi_idx[hi], range_block[hi], facecolors="none",
                       edgecolors="red", linewidths=0.8, marker="s",
                       s=marker_size * 2.2, label=f"> {highlight_above} dB")

    if is_rfi.any() or (highlight_above is not None and (cond_db > highlight_above).any()):
        ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

    ax.set_xlabel("CPI index")
    ax.set_ylabel("Range block")
    ax.set_title(title)
    ax.grid(alpha=0.15)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Condition number (dB)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["nisar", "preprocessed"], required=True)
    parser.add_argument("--input", required=True, help="Path to the HDF5 file.")
    parser.add_argument("--freq", default="A")
    parser.add_argument("--pol", default="HH")

    parser.add_argument("--pulse-start", type=int, default=None,
                        help="[nisar mode] scene start pulse index")
    parser.add_argument("--pulse-end", type=int, default=None,
                        help="[nisar mode] scene end pulse index, exclusive")
    parser.add_argument("--range-start", type=int, default=None,
                        help="[nisar mode] scene start range index")
    parser.add_argument("--range-end", type=int, default=None,
                        help="[nisar mode] scene end range index, exclusive")
    parser.add_argument("--gap-mag-frac", type=float, default=DEFAULT_GAP_MAG_FRAC,
                        help="[nisar mode] transmission-gap detection threshold")

    parser.add_argument("--n-keep", type=int, default=DEFAULT_N_KEEP)
    parser.add_argument("--diag-valid-frac-thresh", type=float,
                        default=DEFAULT_DIAG_VALID_FRAC_THRESH)

    parser.add_argument("--rfi-label-min", type=int, default=DEFAULT_RFI_LABEL_MIN,
                        help="[preprocessed mode] outline tiles with label >= this")
    parser.add_argument("--rfi-label-max", type=int, default=DEFAULT_RFI_LABEL_MAX,
                        help="[preprocessed mode] outline tiles with label <= this")

    parser.add_argument("--highlight-above", type=float, default=None,
                        help="Outline any tile whose condition number (dB) exceeds "
                             "this value, e.g. to flag a suspected overlap zone.")

    parser.add_argument("--vmin", type=float, default=None,
                        help="Colorbar lower bound in dB (default: auto)")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Colorbar upper bound in dB (default: auto)")
    parser.add_argument("--cmap", default="inferno",
                        help="Matplotlib colormap name (default: inferno)")
    parser.add_argument("--marker-size", type=float, default=10.0,
                        help="[preprocessed mode] scatter marker size (default: 10)")

    parser.add_argument("--title", default=None, help="Plot title (default: auto)")
    parser.add_argument("--output", required=True, help="Output PNG path")

    args = parser.parse_args()

    default_title = f"Condition number (dB) -- {args.input} [{args.freq}/{args.pol}]"
    title = args.title if args.title else default_title

    if args.mode == "nisar":
        cond_grid, _, _, _ = load_nisar_grid(
            args.input, args.freq, args.pol,
            args.diag_valid_frac_thresh, args.n_keep,
            args.pulse_start, args.pulse_end,
            args.range_start, args.range_end,
            args.gap_mag_frac,
        )
        n_finite = int(np.isfinite(cond_grid).sum())
        print(f"{n_finite} / {cond_grid.size} tiles passed the validity check.")
        plot_dense_grid(cond_grid, args.output, title, args.highlight_above,
                         args.vmin, args.vmax, args.cmap)
    else:
        cpi_idx, range_block, cond_db, is_rfi = load_preprocessed_points(
            args.input, args.freq, args.pol,
            args.diag_valid_frac_thresh, args.n_keep,
            args.rfi_label_min, args.rfi_label_max,
        )
        print(f"{cond_db.size} valid tiles loaded "
              f"({int(is_rfi.sum())} labeled RFI).")
        if cond_db.size == 0:
            print("No valid tiles to plot.", file=sys.stderr)
            sys.exit(1)
        plot_sparse_scatter(cpi_idx, range_block, cond_db, is_rfi, args.output, title,
                             args.highlight_above, args.vmin, args.vmax, args.cmap,
                             args.marker_size)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
