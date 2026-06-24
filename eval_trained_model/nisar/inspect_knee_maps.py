"""
inspect_knee_maps.py

Inspect model outputs on real NISAR data.

Reads the per-CPI knee-index results from knee_maps.h5 (produced by
infer_knee_map.py) and generates colormapped 2D plots of:
    - Knee index map     (500 x 211)  -- azimuth x range tiles
    - Confidence map     (500 x 211)
    - Entropy map        (500 x 211)

Invalid tiles (satellite gap zones, stored with valid=False) are set to
NaN and rendered in black so they are visually distinct from model outputs.

Output PNGs are saved to the same directory as knee_maps.h5 by default.

Usage
-----
    python inspect_knee_maps.py
    python inspect_knee_maps.py --h5  nisar_data/processed/knee_maps.h5
    python inspect_knee_maps.py --out nisar_data/plots/
    python inspect_knee_maps.py --blocks block_mid block_bottom
"""

import os
import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch


# ---------------------------------------------------------------------------
# DEFAULTS
# ---------------------------------------------------------------------------

DEFAULT_H5  = os.path.join('nisar_data', 'processed', 'knee_maps.h5')
DEFAULT_OUT = None   # defaults to same directory as --h5


# ---------------------------------------------------------------------------
# MAP RECONSTRUCTION
# ---------------------------------------------------------------------------

def read_block_maps(h5_path, block_name):
    """
    Reconstruct 2D knee, confidence, entropy, and valid maps from the
    per-CPI group structure in knee_maps.h5.

    Args:
        h5_path    : Path to knee_maps.h5.
        block_name : Group name, e.g. 'block_mid'.

    Returns:
        knee_map : float32, shape (n_cpi_rows, n_range_cols)  NaN = invalid
        conf_map : float32, shape (n_cpi_rows, n_range_cols)  NaN = invalid
        ent_map  : float32, shape (n_cpi_rows, n_range_cols)  NaN = invalid
        valid    : bool,    shape (n_cpi_rows, n_range_cols)
        attrs    : dict of block-level metadata
    """
    with h5py.File(h5_path, 'r') as f:
        blk          = f[block_name]
        n_cpi_rows   = int(blk.attrs['n_cpi_rows'])
        n_range_cols = int(blk.attrs['n_range_cols'])
        attrs        = {k: blk.attrs[k] for k in blk.attrs}

        knee_map = np.full((n_cpi_rows, n_range_cols), np.nan, dtype=np.float32)
        conf_map = np.full((n_cpi_rows, n_range_cols), np.nan, dtype=np.float32)
        ent_map  = np.full((n_cpi_rows, n_range_cols), np.nan, dtype=np.float32)
        valid    = np.zeros((n_cpi_rows, n_range_cols), dtype=bool)

        for ci in range(n_cpi_rows):
            for ri in range(n_range_cols):
                grp      = blk[f'cpi_{ci}_{ri}']
                is_valid = bool(grp['valid'][()])
                valid[ci, ri] = is_valid
                if is_valid:
                    knee_map[ci, ri] = float(grp['knee_index'][()])
                    conf_map[ci, ri] = float(grp['confidence'][()])
                    ent_map[ci, ri]  = float(grp['entropy'][()])
                # invalid tiles stay NaN

    return knee_map, conf_map, ent_map, valid, attrs


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def _make_masked_cmap(base_cmap_name):
    """
    Return a colormap that renders NaN as black.

    h5py masked/NaN values in imshow are handled by set_bad on the colormap.
    """
    cmap = matplotlib.colormaps[base_cmap_name].copy()
    cmap.set_bad(color='black')
    return cmap


def plot_block(knee_map, conf_map, ent_map, valid, attrs, block_name,
               out_dir, m_pulses):
    """
    Generate and save the three-panel inspection figure for one block.

    Layout:
        Row 1 -- Knee index map     (plasma colormap, 0..M, NaN=black)
        Row 2 -- Confidence map     (viridis,          0..1, NaN=black)
        Row 3 -- Entropy map        (magma reversed,   0..max, NaN=black)

    Args:
        knee_map   : float32 (n_cpi_rows, n_range_cols), NaN = invalid
        conf_map   : float32 (n_cpi_rows, n_range_cols), NaN = invalid
        ent_map    : float32 (n_cpi_rows, n_range_cols), NaN = invalid
        valid      : bool    (n_cpi_rows, n_range_cols)
        attrs      : dict of block-level HDF5 attrs
        block_name : str, e.g. 'block_mid'
        out_dir    : directory to save PNG
        m_pulses   : int, CPI size M (used for knee colormap max)
    """
    n_invalid = int((~valid).sum())
    n_total   = valid.size
    pct_gap   = 100.0 * n_invalid / n_total

    pulse_start = int(attrs.get('pulse_start', 0))
    pulse_end   = int(attrs.get('pulse_end',   0))

    cmap_knee = _make_masked_cmap('plasma')
    cmap_conf = _make_masked_cmap('viridis')
    cmap_ent  = _make_masked_cmap('magma_r')

    fig, axes = plt.subplots(3, 1, figsize=(14, 13))
    fig.suptitle(
        f'Knee-Index Model Outputs on Real NISAR Data\n'
        f'Block: {block_name}   Pulses [{pulse_start}:{pulse_end}]   '
        f'Shape: {knee_map.shape[0]} CPI rows x {knee_map.shape[1]} range tiles\n'
        f'Black = invalid / satellite gap  ({n_invalid}/{n_total} tiles, '
        f'{pct_gap:.1f}%)',
        fontsize=11,
    )

    aspect = 'auto'

    # --- Panel 1: Knee index ---
    ax = axes[0]
    im = ax.imshow(
        knee_map,
        aspect=aspect,
        origin='upper',
        cmap=cmap_knee,
        vmin=0,
        vmax=m_pulses,
        interpolation='nearest',
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cb.set_label('Knee Index', fontsize=9)
    ax.set_title('Predicted Knee Index', fontsize=10)
    ax.set_xlabel('Range Tile', fontsize=9)
    ax.set_ylabel('CPI Row (azimuth)', fontsize=9)
    ax.tick_params(labelsize=8)
    _add_gap_legend(ax)

    # --- Panel 2: Confidence ---
    ax = axes[1]
    im = ax.imshow(
        conf_map,
        aspect=aspect,
        origin='upper',
        cmap=cmap_conf,
        vmin=0.0,
        vmax=1.0,
        interpolation='nearest',
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cb.set_label('Confidence', fontsize=9)
    ax.set_title('Model Confidence (max softmax probability)', fontsize=10)
    ax.set_xlabel('Range Tile', fontsize=9)
    ax.set_ylabel('CPI Row (azimuth)', fontsize=9)
    ax.tick_params(labelsize=8)
    _add_gap_legend(ax)

    # --- Panel 3: Entropy ---
    valid_ent   = ent_map[valid]
    ent_max     = float(np.nanmax(ent_map)) if valid_ent.size > 0 else 1.0
    ax = axes[2]
    im = ax.imshow(
        ent_map,
        aspect=aspect,
        origin='upper',
        cmap=cmap_ent,
        vmin=0.0,
        vmax=ent_max,
        interpolation='nearest',
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cb.set_label('Shannon Entropy (nats)', fontsize=9)
    ax.set_title('Prediction Entropy (low = confident, high = uncertain)',
                 fontsize=10)
    ax.set_xlabel('Range Tile', fontsize=9)
    ax.set_ylabel('CPI Row (azimuth)', fontsize=9)
    ax.tick_params(labelsize=8)
    _add_gap_legend(ax)

    fig.tight_layout()

    out_path = os.path.join(out_dir, f'{block_name}_knee_maps.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {out_path}')

    # --- Summary stats ---
    valid_knee = knee_map[valid]
    valid_conf = conf_map[valid]
    if valid_knee.size > 0:
        print(f'  knee  : min={int(np.nanmin(valid_knee))}  '
              f'max={int(np.nanmax(valid_knee))}  '
              f'mean={np.nanmean(valid_knee):.3f}  '
              f'median={np.nanmedian(valid_knee):.1f}')
        unique, counts = np.unique(valid_knee.astype(int), return_counts=True)
        top5 = sorted(zip(counts, unique), reverse=True)[:5]
        top5_str = '  '.join(f'knee={k}:{c}' for c, k in top5)
        print(f'  top-5 knee values: {top5_str}')
        print(f'  conf  : mean={np.nanmean(valid_conf):.4f}  '
              f'min={np.nanmin(valid_conf):.4f}  '
              f'max={np.nanmax(valid_conf):.4f}')
        low_conf = int(np.sum(valid_conf < 0.5))
        print(f'  low-confidence tiles (<0.5): {low_conf} '
              f'({100*low_conf/valid_conf.size:.1f}%)')


def _add_gap_legend(ax):
    """Add a small legend indicating black = satellite gap."""
    legend_elem = Patch(facecolor='black', edgecolor='grey',
                        label='Satellite gap (invalid)')
    ax.legend(handles=[legend_elem], loc='upper right', fontsize=7,
              framealpha=0.7)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Inspect knee-index model outputs on real NISAR data.'
    )
    parser.add_argument('--h5',     default=DEFAULT_H5,
                        help='Path to knee_maps.h5 output file.')
    parser.add_argument('--out',    default=DEFAULT_OUT,
                        help='Output directory for PNGs '
                             '(default: same dir as --h5).')
    parser.add_argument('--blocks', nargs='+', default=None,
                        help='Block names to inspect (default: all in file).')
    args = parser.parse_args()

    if not os.path.exists(args.h5):
        raise FileNotFoundError(f'knee_maps.h5 not found: {args.h5}')

    out_dir = args.out if args.out else os.path.dirname(os.path.abspath(args.h5))
    os.makedirs(out_dir, exist_ok=True)

    # Discover blocks
    with h5py.File(args.h5, 'r') as f:
        all_blocks = list(f.keys())
        m_pulses   = int(f[all_blocks[0]].attrs.get('M', 16))

    blocks = args.blocks if args.blocks else all_blocks
    unknown = [b for b in blocks if b not in all_blocks]
    if unknown:
        raise ValueError(f'Unknown blocks: {unknown}. Available: {all_blocks}')

    print(f'Source : {args.h5}')
    print(f'Blocks : {blocks}')
    print(f'Out    : {out_dir}')
    print(f'M      : {m_pulses}')
    print()

    for block_name in blocks:
        print(f'--- {block_name} ---')
        print(f'  Reading maps ...')
        knee_map, conf_map, ent_map, valid, attrs = read_block_maps(
            args.h5, block_name
        )
        n_invalid = int((~valid).sum())
        print(f'  Shape        : {knee_map.shape}')
        print(f'  Invalid tiles: {n_invalid} / {valid.size}')
        print(f'  Plotting ...')
        plot_block(knee_map, conf_map, ent_map, valid, attrs,
                   block_name, out_dir, m_pulses)
        print()

    print('Done.')


if __name__ == '__main__':
    main()