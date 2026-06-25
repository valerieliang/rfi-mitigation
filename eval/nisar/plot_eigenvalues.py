"""
plot_eigenvalues.py  --  Stage 2: Eigenvalue Profile Plots

Read cpi_blocks.h5 produced by tile_cpi.py and generate eigenvalue profile
plots for every CPI block (or a sampled subset).  Each plot is a small
grid of subplots -- one subplot per block -- saved as a PNG named by the
block index range it covers.

Output layout
-------------
<out_dir>/plots/
    eigenvalues_ci0000-0031_ri0000-0009.png   (one PNG per grid page)
    eigenvalues_ci0032-0063_ri0000-0009.png
    ...

Each subplot title shows ci, ri, the dB span of the profile, and whether
the tile was flagged as invalid.

Usage
-----
    python plot_eigenvalues.py
    python plot_eigenvalues.py --h5   nisar_data/processed/cpi_blocks.h5
    python plot_eigenvalues.py --out  nisar_data/processed
    python plot_eigenvalues.py --step 8    (plot every 8th CPI row)
    python plot_eigenvalues.py --ri   0    (only range tile 0)
    python plot_eigenvalues.py --cols 4 --rows 4
"""

import os
import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEFAULT_H5  = os.path.join('nisar_data', 'processed', 'cpi_blocks.h5')
DEFAULT_OUT = os.path.join('nisar_data', 'processed')


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _load_meta(h5_path):
    """Return root-level metadata dict from cpi_blocks.h5."""
    with h5py.File(h5_path, 'r') as f:
        return {k: f.attrs[k] for k in f.attrs}


def _load_tile(f, ci, ri):
    """
    Load eigen_input, valid flag from an open HDF5 file handle.

    Returns:
        ev_db  : float32 array (M,)   -- eigenvalues in dB, descending
        valid  : bool
    """
    key = f'cpi_{ci}_{ri}'
    if key not in f:
        return None, False
    grp   = f[key]
    valid = bool(grp['valid'][()])
    if not valid:
        return None, False
    ev_db = grp['eigen_input'][:, 0]   # column 0 = eigenvalues in dB
    return ev_db, True


# ---------------------------------------------------------------------------
# PLOT ONE PAGE
# ---------------------------------------------------------------------------

def _plot_page(tiles, page_label, out_dir, M, n_cols, n_rows):
    """
    Render a grid of eigenvalue profiles for a list of (ci, ri, ev_db, valid)
    tuples.  Saves one PNG per call.

    Args:
        tiles      : list of (ci, ri, ev_db_or_None, is_valid)
        page_label : str used in filename and suptitle
        out_dir    : destination directory (already exists)
        M          : CPI size (x-axis length)
        n_cols     : subplot grid columns
        n_rows     : subplot grid rows
    """
    ev_index = np.arange(M)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.5, n_rows * 2.8),
                             sharey=False)
    fig.suptitle(f'Eigenvalue Profiles  --  {page_label}', fontsize=10)

    for ax, (ci, ri, ev_db, is_valid) in zip(axes.flat, tiles):
        if not is_valid or ev_db is None:
            ax.set_title(f'ci={ci} ri={ri}\nINVALID', fontsize=7, color='red')
            ax.axis('off')
            continue
        span = float(ev_db[0] - ev_db[-1])
        ax.plot(ev_index, ev_db, linewidth=1.0, color='steelblue')
        ax.set_title(f'ci={ci} ri={ri}  span={span:.1f}dB', fontsize=7)
        ax.set_xlabel('EV index', fontsize=6)
        ax.set_ylabel('dB', fontsize=6)
        ax.tick_params(labelsize=5)
        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5)

    # Hide unused subplots
    for ax in axes.flat[len(tiles):]:
        ax.axis('off')

    fig.tight_layout()

    fname    = f'eigenvalues_{page_label}.png'
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Stage 2: plot eigenvalue profiles from cpi_blocks.h5.'
    )
    parser.add_argument('--h5',   default=DEFAULT_H5,
                        help='Path to cpi_blocks.h5 (output of tile_cpi.py).')
    parser.add_argument('--out',  default=DEFAULT_OUT,
                        help='Root output directory.  A plots/ subfolder '
                             'is created automatically.')
    parser.add_argument('--step', type=int, default=1,
                        help='Plot every Nth CPI row (default 1 = all rows).')
    parser.add_argument('--ri',   type=int, default=None,
                        help='Restrict to a single range tile index.  '
                             'Default: all range tiles.')
    parser.add_argument('--cols', type=int, default=5,
                        help='Subplot grid columns per page (default 5).')
    parser.add_argument('--rows', type=int, default=4,
                        help='Subplot grid rows per page (default 4).')
    args = parser.parse_args()

    if not os.path.exists(args.h5):
        raise FileNotFoundError(f'cpi_blocks.h5 not found: {args.h5}')

    plots_dir = os.path.join(args.out, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    meta         = _load_meta(args.h5)
    M            = int(meta['M'])
    n_cpi_rows   = int(meta['n_cpi_rows'])
    n_range_cols = int(meta['n_range_cols'])

    ci_list = list(range(0, n_cpi_rows, args.step))
    ri_list = [args.ri] if args.ri is not None else list(range(n_range_cols))

    per_page = args.cols * args.rows
    total    = len(ci_list) * len(ri_list)
    print(f'Source      : {args.h5}')
    print(f'M           : {M}')
    print(f'Grid        : {n_cpi_rows} CPI rows x {n_range_cols} range cols')
    print(f'Selected    : {len(ci_list)} CPI rows x {len(ri_list)} range cols '
          f'= {total} tiles')
    print(f'Per page    : {per_page}  ({args.rows} rows x {args.cols} cols)')
    print(f'Plots dir   : {plots_dir}')
    print()

    page_tiles  = []
    page_ci_min = page_ri_min = None
    page_ci_max = page_ri_max = None
    n_pages     = 0

    def _flush_page(tiles, ci_min, ci_max, ri_min, ri_max):
        nonlocal n_pages
        label    = (f'ci{ci_min:04d}-{ci_max:04d}'
                    f'_ri{ri_min:04d}-{ri_max:04d}')
        out_path = _plot_page(tiles, label, plots_dir, M,
                              args.cols, args.rows)
        n_pages += 1
        print(f'  Saved {os.path.basename(out_path)}  ({len(tiles)} tiles)')

    with h5py.File(args.h5, 'r') as f:
        for ci in ci_list:
            for ri in ri_list:
                ev_db, valid = _load_tile(f, ci, ri)
                page_tiles.append((ci, ri, ev_db, valid))

                if page_ci_min is None:
                    page_ci_min = page_ci_max = ci
                    page_ri_min = page_ri_max = ri
                else:
                    page_ci_min = min(page_ci_min, ci)
                    page_ci_max = max(page_ci_max, ci)
                    page_ri_min = min(page_ri_min, ri)
                    page_ri_max = max(page_ri_max, ri)

                if len(page_tiles) == per_page:
                    _flush_page(page_tiles,
                                page_ci_min, page_ci_max,
                                page_ri_min, page_ri_max)
                    page_tiles  = []
                    page_ci_min = page_ri_min = None
                    page_ci_max = page_ri_max = None

        # Flush remaining tiles
        if page_tiles:
            _flush_page(page_tiles,
                        page_ci_min, page_ci_max,
                        page_ri_min, page_ri_max)

    print()
    print(f'Done.  {n_pages} PNG(s) written to {plots_dir}')


if __name__ == '__main__':
    main()
