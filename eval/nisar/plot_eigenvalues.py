"""
plot_eigenvalues.py  --  Stage 2: Eigenvalue Profile Plots

Reads cpi_blocks.h5 and produces targeted diagnostic plots for two
pulse regions of interest (RFI-contaminated and clean urban), sampling
the middle range column of each CPI row in the region.

For each region, two PNGs are saved:

  1. stacked_{region}.png
       All eigenvalue profiles overlaid on one axes, color-coded by
       the max eigenvalue (dB) of each profile.  Gives a population
       view of the spread and RFI lift.

  2. grid_{region}.png
       Each profile in its own subplot on a shared y-axis, arranged
       in a grid.  Every subplot has the same dB scale so profiles
       are directly comparable.

Global pulse -> CPI row conversion:
    ci = (global_pulse - PULSE_OFFSET) // M
    where PULSE_OFFSET is the pulse index of the first pulse in the file
    (stored as root attr pulse_start, or defaulting to 0 if absent).

Usage
-----
    python plot_eigenvalues.py
    python plot_eigenvalues.py --h5 nisar_data/processed/cpi_blocks.h5
    python plot_eigenvalues.py --out nisar_data/processed
    python plot_eigenvalues.py --rfi-start 63000 --rfi-stop 87000
    python plot_eigenvalues.py --clean-start 106000 --clean-stop 122000
"""

import os
import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

DEFAULT_H5          = os.path.join('nisar_data', 'processed', 'cpi_blocks.h5')
DEFAULT_OUT         = os.path.join('nisar_data', 'processed')
DEFAULT_RFI_START   = 63000
DEFAULT_RFI_STOP    = 87000
DEFAULT_CLEAN_START = 106000
DEFAULT_CLEAN_STOP  = 122000
# First global pulse in the L0B file (matches tile_cpi.py source)
DEFAULT_FILE_PULSE_OFFSET = 46528


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def load_meta(h5_path):
    with h5py.File(h5_path, 'r') as f:
        return {k: f.attrs[k] for k in f.attrs}


def pulse_to_ci(global_pulse, pulse_offset, M):
    """Convert a global pulse index to a CPI row index."""
    return (global_pulse - pulse_offset) // M


def load_region(h5_path, ci_start, ci_stop, ri_mid, M):
    """
    Load eigenvalue profiles for every CPI row in [ci_start, ci_stop)
    at range column ri_mid.

    Returns list of (ci, ev_db) for valid tiles only.
    """
    profiles = []
    with h5py.File(h5_path, 'r') as f:
        for ci in range(ci_start, ci_stop):
            key = f'cpi_{ci}_{ri_mid}'
            if key not in f:
                continue
            grp = f[key]
            if not bool(grp['valid'][()]):
                continue
            ev_db = grp['eigen_input'][:, 0].astype(np.float32)
            profiles.append((ci, ev_db))
    return profiles


# ---------------------------------------------------------------------------
# PLOT 1: STACKED (all profiles overlaid, color = max eigenvalue)
# ---------------------------------------------------------------------------

def plot_stacked(profiles, region_label, out_dir, M,
                 pulse_offset, ci_start, ci_stop):
    if not profiles:
        print(f'  [{region_label}] no valid profiles -- skipping stacked plot')
        return

    ev_index = np.arange(M)
    max_vals = np.array([p[1][0] for p in profiles])   # first EV = max in dB
    vmin, vmax = max_vals.min(), max_vals.max()
    norm  = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap  = cm.plasma

    fig, ax = plt.subplots(figsize=(9, 5))
    for ci, ev_db in profiles:
        color = cmap(norm(ev_db[0]))
        ax.plot(ev_index, ev_db, color=color, linewidth=0.6, alpha=0.7)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02)
    cb.set_label('Max eigenvalue (dB)', fontsize=9)

    g_start = pulse_offset + ci_start * M
    g_stop  = pulse_offset + ci_stop  * M
    ax.set_title(
        f'{region_label}  --  stacked eigenvalue profiles\n'
        f'global pulses {g_start}--{g_stop}  '
        f'({len(profiles)} valid CPI rows  ri=mid)',
        fontsize=10,
    )
    ax.set_xlabel('Eigenvalue index', fontsize=9)
    ax.set_ylabel('Eigenvalue (dB)', fontsize=9)
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)
    fig.tight_layout()

    out_path = os.path.join(out_dir, f'stacked_{region_label}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {os.path.basename(out_path)}')


# ---------------------------------------------------------------------------
# PLOT 2: GRID (individual subplots, shared y-axis scale)
# ---------------------------------------------------------------------------

def plot_grid(profiles, region_label, out_dir, M,
              pulse_offset, ci_start, ci_stop, n_cols=6):
    if not profiles:
        print(f'  [{region_label}] no valid profiles -- skipping grid plot')
        return

    n       = len(profiles)
    n_rows  = int(np.ceil(n / n_cols))
    ev_index = np.arange(M)

    # Shared y-axis limits across all profiles
    all_ev = np.stack([p[1] for p in profiles])
    y_min  = float(all_ev.min()) - 1.0
    y_max  = float(all_ev.max()) + 1.0

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.4, n_rows * 2.0),
        sharey=True,
    )

    g_start = pulse_offset + ci_start * M
    g_stop  = pulse_offset + ci_stop  * M
    fig.suptitle(
        f'{region_label}  --  individual eigenvalue profiles  '
        f'(shared y-axis)\n'
        f'global pulses {g_start}--{g_stop}  ri=mid',
        fontsize=10,
    )

    for ax, (ci, ev_db) in zip(axes.flat, profiles):
        span = float(ev_db[0] - ev_db[-1])
        g    = pulse_offset + ci * M
        ax.plot(ev_index, ev_db, linewidth=0.9, color='steelblue')
        ax.set_title(f'g={g}\nspan={span:.1f}dB', fontsize=6)
        ax.set_ylim(y_min, y_max)
        ax.tick_params(labelsize=5)
        ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.5)

    for ax in axes.flat[n:]:
        ax.axis('off')

    fig.tight_layout()
    out_path = os.path.join(out_dir, f'grid_{region_label}.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {os.path.basename(out_path)}')


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Stage 2: targeted eigenvalue profile plots for RFI '
                    'and clean regions.'
    )
    parser.add_argument('--h5',           default=DEFAULT_H5)
    parser.add_argument('--out',          default=DEFAULT_OUT)
    parser.add_argument('--rfi-start',    type=int, default=DEFAULT_RFI_START,
                        help='Global pulse start of RFI region (default 63000).')
    parser.add_argument('--rfi-stop',     type=int, default=DEFAULT_RFI_STOP,
                        help='Global pulse stop  of RFI region (default 87000).')
    parser.add_argument('--clean-start',  type=int, default=DEFAULT_CLEAN_START,
                        help='Global pulse start of clean region (default 106000).')
    parser.add_argument('--clean-stop',   type=int, default=DEFAULT_CLEAN_STOP,
                        help='Global pulse stop  of clean region (default 122000).')
    parser.add_argument('--pulse-offset', type=int, default=DEFAULT_FILE_PULSE_OFFSET,
                        help='Global pulse index of the first pulse in the file '
                             '(default 46528).')
    parser.add_argument('--cols',         type=int, default=6,
                        help='Columns in the grid plot (default 6).')
    args = parser.parse_args()

    if not os.path.exists(args.h5):
        raise FileNotFoundError(f'cpi_blocks.h5 not found: {args.h5}')

    plots_dir = os.path.join(args.out, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    meta         = load_meta(args.h5)
    M            = int(meta['M'])
    n_cpi_rows   = int(meta['n_cpi_rows'])
    n_range_cols = int(meta['n_range_cols'])
    ri_mid       = n_range_cols // 2

    print(f'Source       : {args.h5}')
    print(f'M            : {M}')
    print(f'Grid         : {n_cpi_rows} CPI rows x {n_range_cols} range cols')
    print(f'Middle ri    : {ri_mid}')
    print(f'Pulse offset : {args.pulse_offset}')
    print()

    regions = [
        ('rfi',   args.rfi_start,   args.rfi_stop),
        ('clean', args.clean_start, args.clean_stop),
    ]

    for label, g_start, g_stop in regions:
        ci_start = pulse_to_ci(g_start, args.pulse_offset, M)
        ci_stop  = pulse_to_ci(g_stop,  args.pulse_offset, M)
        ci_start = max(0, ci_start)
        ci_stop  = min(n_cpi_rows, ci_stop)

        print(f'--- {label}  g=[{g_start}, {g_stop})  '
              f'ci=[{ci_start}, {ci_stop})  '
              f'({ci_stop - ci_start} rows) ---')

        profiles = load_region(args.h5, ci_start, ci_stop, ri_mid, M)
        print(f'  Valid profiles loaded : {len(profiles)}')

        plot_stacked(profiles, label, plots_dir, M,
                     args.pulse_offset, ci_start, ci_stop)
        plot_grid(profiles, label, plots_dir, M,
                  args.pulse_offset, ci_start, ci_stop, n_cols=args.cols)
        print()

    print(f'Done.  Plots in {plots_dir}')


if __name__ == '__main__':
    main()