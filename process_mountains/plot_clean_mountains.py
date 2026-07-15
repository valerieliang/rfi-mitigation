#!/usr/bin/env python
"""
plot_clean_mountains.py

Plot eigenvalue profiles and SSCM matrices from clean_mountains.h5 files
produced by select_clean_mountain.py. Similar to mountain_profile.py but
reads pre-selected clean tiles from HDF5 instead of processing raw L0B.

Produces:
  1. Average eigenvalue shape plot
  2. All eigenvalue profiles overlaid
  3. Grid of individual eigenvalue profiles and SCM diagonal plots
  4. SCM diagonal plot (all CPIs overlaid)

Uses the same y-axis rule of thumb as mountain_profile.py: dynamic y-limits
computed from the top 12 eigenvalues.
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
# CONSTANTS
# ---------------------------------------------------------------------------

N_SHOW = 16  # Show all 16 values in plots
EPS = 1e-12

# ---------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------------------------

def eigvals_to_db(eigvals: np.ndarray) -> np.ndarray:
    """Convert eigenvalues to dB (un-normalized power in dB)."""
    eigvals_db = 10.0 * np.log10(np.clip(eigvals, EPS, None))
    return eigvals_db.astype(np.float32)

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
    y_min = 0

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill_between(x, lo, hi, color="C0", alpha=0.12, label="min / max envelope")
    ax.fill_between(x, mean - std, mean + std, color="C0", alpha=0.30,
                    label="mean +/- 1 std")
    ax.plot(x, mean, color="C0", lw=2.2, label="mean")
    ax.plot(x, med, color="C3", lw=1.4, ls="--", label="median")

    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue (dB, power)")
    ax.set_title("Average eigenvalue shape (clean tiles): freq {} pol {} ({} CPIs, all {})"
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
    y_min = 0

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(n):
        ax.plot(x, eig[i], color="C0", alpha=alpha, lw=0.8)
    ax.plot(x, eig.mean(axis=0), color="k", lw=1.6, label="mean")

    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue (dB, power)")
    ax.set_title("All eigenvalue profiles overlaid (clean tiles): freq {} pol {} ({} CPIs, all {})"
                 .format(freq, pol, n, n_show))
    ax.set_xticks(x)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_grid(eig_db, diag_db, cpi_pulse_idx, cpi_range_idx, out_path, freq, pol,
              n_show, max_grid):
    """
    Grid of small multiples: eigenvalue profile and SCM diagonal scatter plot per CPI.
    Each row shows one CPI with 2 plots.

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
        y_min = 0

        n_cols = 2  # Two columns: eigenvalue plot and SCM diagonal scatter plot

        fig, axes = plt.subplots(n_in_chunk, n_cols,
                                 figsize=(9.6, 1.9 * n_in_chunk),
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
        fig.suptitle("Eigenvalue & SCM diagonal grid (clean tiles): freq {} pol {}{}{}  [all {}]"
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
    y_min = 0

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(n):
        ax.plot(x, d[i], color="C3", alpha=alpha, lw=0.8)
    ax.plot(x, d.mean(axis=0), color="k", lw=1.6, label="mean")

    ax.set_xlabel("Pulse index (SCM diagonal position)")
    ax.set_ylabel("SCM diagonal power (dB)")
    ax.set_title("SCM diagonal per CPI (clean tiles): freq {} pol {} ({} CPIs, all {})"
                 .format(freq, pol, n, n_show))
    ax.set_xticks(x)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('input_h5', help='Input clean_mountains.h5 file')
    parser.add_argument('--max-grid', type=int, default=48,
                        help='Max number of individual shapes drawn in the grid '
                             'plot (evenly sampled across the scene if exceeded).')
    parser.add_argument('--output-dir', default='results/clean_mountains',
                        help='Output directory for plots.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("[+] Reading clean mountain profiles from: {}".format(args.input_h5))
    h5_in = h5py.File(args.input_h5, 'r')

    # Iterate over all groups (freq_X_pol_Y)
    for grp_name in h5_in.keys():
        print("\n[+] Processing group: {}".format(grp_name))
        grp = h5_in[grp_name]

        # Read attributes
        freq = grp.attrs['frequency']
        pol = grp.attrs['polarization']
        n_clean = grp.attrs['n_clean_tiles']
        print("    freq={}, pol={}, n_clean_tiles={}".format(freq, pol, n_clean))

        # Read datasets
        eig_lin = grp['eigenvalues'][:]        # (N, 16) linear eigenvalues
        diag_lin = grp['diagonal'][:]          # (N, 16) linear diagonal
        pulse_idx = grp['pulse_idx'][:]        # (N,)
        range_idx = grp['range_idx'][:]        # (N,)

        # Convert to dB
        eig_db = eigvals_to_db(eig_lin)
        diag_db = 10.0 * np.log10(np.clip(diag_lin, EPS, None))

        # Generate plots
        tag = "{}_{}".format(freq, pol)

        p_avg = os.path.join(args.output_dir, "clean_avg_shape_{}.png".format(tag))
        plot_average_shape(eig_db, p_avg, freq, pol, N_SHOW)
        print("    [saved] {}".format(p_avg))

        p_overlaid = os.path.join(args.output_dir, "clean_overlaid_{}.png".format(tag))
        plot_overlaid(eig_db, p_overlaid, freq, pol, N_SHOW)
        print("    [saved] {}".format(p_overlaid))

        p_grid = os.path.join(args.output_dir, "clean_grid_{}.png".format(tag))
        grid_files = plot_grid(eig_db, diag_db, pulse_idx, range_idx,
                               p_grid, freq, pol, N_SHOW, args.max_grid)
        grid_str = " | ".join(grid_files) if len(grid_files) > 1 else grid_files[0]
        print("    [saved] {}".format(grid_str))

        p_diag = os.path.join(args.output_dir, "clean_scm_diag_{}.png".format(tag))
        plot_scm_diag(diag_db, p_diag, freq, pol, N_SHOW)
        print("    [saved] {}".format(p_diag))

    h5_in.close()
    print("\n[+] Done.")


if __name__ == '__main__':
    main()
