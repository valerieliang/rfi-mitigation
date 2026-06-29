"""
plot_cpi_eigenvalues.py

Extract and plot CPI eigenvalues for a specific CPI tile given its (row, col) index.
Generates two plots:
  1. Raw eigenvalues in dB (as stored in the HDF5 file)
  2. Normalized eigenvalue profile (normalized to [0,1])

Usage:
    python plot_cpi_eigenvalues.py --h5 nisar_data/processed/cpi_blocks.h5 --row 1000 --col 50
    python plot_cpi_eigenvalues.py --h5 nisar_data/processed/cpi_blocks.h5 --row 1000 --col 50 --out plots
"""

import os
import sys
import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DEFAULT_H5 = os.path.join('nisar_data', 'processed', 'cpi_blocks.h5')
DEFAULT_OUT = 'plots'


def load_cpi_eigenvalues(h5_path, ci, ri):
    """
    Load eigenvalue data for a specific CPI tile.

    Args:
        h5_path (str): Path to cpi_blocks.h5 file
        ci (int): CPI row index (pulse dimension)
        ri (int): Range column index

    Returns:
        eigen_input (np.ndarray): Shape (M, 2) with [ev_db, slope_db]
        global_input (np.ndarray): Shape (6,) with global features
        valid (bool): Whether the CPI is valid (not a gap/fill region)
        M (int): Number of pulses per CPI
        metadata (dict): Additional metadata
    """
    key = f'cpi_{ci}_{ri}'

    with h5py.File(h5_path, 'r') as f:
        if key not in f:
            raise KeyError(f"CPI tile '{key}' not found in {h5_path}")

        grp = f[key]
        eigen_input = grp['eigen_input'][:]
        global_input = grp['global_input'][:]
        valid = bool(grp['valid'][()])

        # Get metadata from root attributes
        M = int(f.attrs['M'])
        BLOCK_WIDTH = int(f.attrs['BLOCK_WIDTH'])
        n_cpi_rows = int(f.attrs['n_cpi_rows'])
        n_range_cols = int(f.attrs['n_range_cols'])

        # Get CPI-specific attributes
        pulse_start = grp.attrs['pulse_start']
        pulse_end = grp.attrs['pulse_end']
        range_start = grp.attrs['range_start']
        range_end = grp.attrs['range_end']

        metadata = {
            'M': M,
            'BLOCK_WIDTH': BLOCK_WIDTH,
            'n_cpi_rows': n_cpi_rows,
            'n_range_cols': n_range_cols,
            'pulse_start': pulse_start,
            'pulse_end': pulse_end,
            'range_start': range_start,
            'range_end': range_end,
            'global_features': {
                'F_factor': float(global_input[0]),
                'sigma_min': float(global_input[1]),
                'sigma_max': float(global_input[2]),
                'mu_min': float(global_input[3]),
                'trace_db': float(global_input[4]),
                'cond_number': float(global_input[5]),
            }
        }

    return eigen_input, global_input, valid, M, metadata


def plot_eigenvalues(eigen_input, ci, ri, valid, metadata, output_dir):
    """
    Generate two plots: raw eigenvalues (dB) and normalized eigenvalues.

    Args:
        eigen_input (np.ndarray): Shape (M, 2) with [ev_db, slope_db]
        ci (int): CPI row index
        ri (int): Range column index
        valid (bool): Whether CPI is valid
        metadata (dict): Metadata dictionary
        output_dir (str): Output directory for plots
    """
    M = metadata['M']
    ev_db = eigen_input[:, 0]

    # Compute normalized eigenvalues (convert from dB back to linear, then normalize)
    ev_linear = 10.0 ** (ev_db / 10.0)
    ev_normalized = ev_linear / np.max(ev_linear)

    ev_index = np.arange(M)

    # Global features for annotation
    gf = metadata['global_features']

    # --- Plot 1: Raw eigenvalues in dB ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(ev_index, ev_db, linewidth=2.0, color='steelblue', marker='o',
             markersize=5, markerfacecolor='white', markeredgewidth=1.5)

    ax1.set_xlabel('Eigenvalue Index', fontsize=12)
    ax1.set_ylabel('Eigenvalue (dB)', fontsize=12)
    ax1.set_title(
        f'CPI Eigenvalue Profile (Raw dB) - CPI ({ci}, {ri})\n'
        f'Pulses: {metadata["pulse_start"]}-{metadata["pulse_end"]} | '
        f'Range: {metadata["range_start"]}-{metadata["range_end"]} | '
        f'Valid: {valid}',
        fontsize=11
    )
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax1.set_xlim(-0.5, M - 0.5)

    # Add text box with global features
    textstr = '\n'.join([
        f'F-factor: {gf["F_factor"]:.2f}',
        f'Trace: {gf["trace_db"]:.2f} dB',
        f'Cond #: {gf["cond_number"]:.2f}',
        f'σ_max: {gf["sigma_max"]:.2e}',
        f'σ_min: {gf["sigma_min"]:.2e}',
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=props)

    fig1.tight_layout()
    out_path_db = os.path.join(output_dir, f'cpi_{ci}_{ri}_eigenvalues_db.png')
    fig1.savefig(out_path_db, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f'Saved: {out_path_db}')

    # --- Plot 2: Normalized eigenvalues [0,1] ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    ax2.plot(ev_index, ev_normalized, linewidth=2.0, color='darkgreen', marker='s',
             markersize=5, markerfacecolor='white', markeredgewidth=1.5)

    ax2.set_xlabel('Eigenvalue Index', fontsize=12)
    ax2.set_ylabel('Normalized Eigenvalue', fontsize=12)
    ax2.set_title(
        f'CPI Eigenvalue Profile (Normalized) - CPI ({ci}, {ri})\n'
        f'Pulses: {metadata["pulse_start"]}-{metadata["pulse_end"]} | '
        f'Range: {metadata["range_start"]}-{metadata["range_end"]} | '
        f'Valid: {valid}',
        fontsize=11
    )
    ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax2.set_xlim(-0.5, M - 0.5)
    ax2.set_ylim(-0.05, 1.05)

    # Add horizontal line at 0.5 for reference
    ax2.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.0, alpha=0.5, label='0.5 reference')
    ax2.legend(loc='upper right', fontsize=9)

    # Add text box with eigenvalue span
    span_db = float(ev_db[0] - ev_db[-1])
    max_ev_db = float(ev_db[0])
    min_ev_db = float(ev_db[-1])
    textstr = '\n'.join([
        f'Max EV: {max_ev_db:.2f} dB',
        f'Min EV: {min_ev_db:.2f} dB',
        f'Span: {span_db:.2f} dB',
        f'Normalized range: [{ev_normalized[-1]:.3f}, 1.000]',
    ])
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=props)

    fig2.tight_layout()
    out_path_norm = os.path.join(output_dir, f'cpi_{ci}_{ri}_eigenvalues_normalized.png')
    fig2.savefig(out_path_norm, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f'Saved: {out_path_norm}')


def main():
    parser = argparse.ArgumentParser(
        description='Extract and plot eigenvalues for a specific CPI tile'
    )
    parser.add_argument('--h5', default=DEFAULT_H5,
                        help='Path to cpi_blocks.h5 file')
    parser.add_argument('--row', type=int, required=True,
                        help='CPI row index (ci, pulse dimension)')
    parser.add_argument('--col', type=int, required=True,
                        help='CPI column index (ri, range dimension)')
    parser.add_argument('--out', default=DEFAULT_OUT,
                        help='Output directory for plots')

    args = parser.parse_args()

    if not os.path.exists(args.h5):
        print(f"ERROR: HDF5 file not found: {args.h5}")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    print("="*70)
    print("CPI Eigenvalue Extraction and Plotting")
    print("="*70)
    print(f"HDF5 file: {args.h5}")
    print(f"CPI index: ({args.row}, {args.col})")
    print(f"Output dir: {args.out}")
    print()

    # Load eigenvalue data
    print("Loading CPI data...")
    try:
        eigen_input, global_input, valid, M, metadata = load_cpi_eigenvalues(
            args.h5, args.row, args.col
        )
    except KeyError as e:
        print(f"ERROR: {e}")
        print(f"\nAvailable CPI indices:")
        print(f"  Rows (ci): 0 to {metadata.get('n_cpi_rows', '?') - 1}")
        print(f"  Cols (ri): 0 to {metadata.get('n_range_cols', '?') - 1}")
        sys.exit(1)

    print(f"  M (pulses per CPI): {M}")
    print(f"  Valid: {valid}")
    print(f"  Pulse range: {metadata['pulse_start']} to {metadata['pulse_end']}")
    print(f"  Range bins: {metadata['range_start']} to {metadata['range_end']}")
    print(f"\n  Global Features:")
    gf = metadata['global_features']
    for key, val in gf.items():
        if key in ['sigma_min', 'sigma_max', 'mu_min']:
            print(f"    {key}: {val:.6e}")
        else:
            print(f"    {key}: {val:.4f}")
    print()

    # Generate plots
    print("Generating plots...")
    plot_eigenvalues(eigen_input, args.row, args.col, valid, metadata, args.out)

    print()
    print("="*70)
    print("Done!")
    print("="*70)


if __name__ == '__main__':
    main()
