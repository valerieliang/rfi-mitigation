"""
plot_cpi_eigenvalues.py

Extract and plot CPI eigenvalues for a specific CPI tile given its (row, col) index.
Generates two plots:
  1. Raw eigenvalues in dB (as stored in the HDF5 file)
  2. Normalized eigenvalue profile (normalized to [0,1])

Supports two HDF5 formats:
  - tile_cpi.py format (cpi_blocks.h5): groups with eigen_input/global_input/valid
  - process_nisar_to_cpi.py format: datasets with eigenvalues/eigenvalues_normalized

Usage:
    # List available CPIs first
    python plot_cpi_eigenvalues.py --h5 <file.h5> --list

    # tile_cpi.py format (use tile indices)
    python plot_cpi_eigenvalues.py --h5 nisar_data/processed/cpi_blocks.h5 --row 1000 --col 50

    # process_nisar_to_cpi.py format (use pulse/range indices from CPI keys)
    python plot_cpi_eigenvalues.py --h5 nisar_data/processed/nisar_hv_cpi_tiles.h5 --row 64000 --col 12500 --out plots
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


def list_available_cpis(h5_path, max_show=10):
    """
    List available CPI tiles in the HDF5 file.

    Args:
        h5_path (str): Path to HDF5 file
        max_show (int): Maximum number of CPIs to show

    Returns:
        list: List of (ci, ri) tuples or (pulse_start, range_start) tuples
    """
    with h5py.File(h5_path, 'r') as f:
        cpi_keys = sorted([k for k in f.keys() if k.startswith('cpi_') and
                          '_eigenvalues' not in k and '_diagonal' not in k])

        if not cpi_keys:
            return []

        # Parse keys to get indices
        indices = []
        for key in cpi_keys[:max_show]:
            parts = key.replace('cpi_', '').split('_')
            if len(parts) == 2:
                indices.append((int(parts[0]), int(parts[1])))

        return indices, len(cpi_keys)


def detect_hdf5_format(h5_path):
    """
    Detect which HDF5 format is being used.

    Returns:
        str: 'tile_cpi' or 'process_nisar'
    """
    with h5py.File(h5_path, 'r') as f:
        # Check first CPI dataset
        cpi_keys = [k for k in f.keys() if k.startswith('cpi_') and '_eigenvalues' not in k and '_diagonal' not in k]
        if not cpi_keys:
            raise ValueError(f"No CPI datasets found in {h5_path}")

        first_cpi = cpi_keys[0]

        # tile_cpi.py format has groups with 'eigen_input' dataset
        if isinstance(f[first_cpi], h5py.Group) and 'eigen_input' in f[first_cpi]:
            return 'tile_cpi'
        # process_nisar_to_cpi.py format has direct datasets
        elif f'{first_cpi}_eigenvalues' in f or f'{first_cpi}_eigenvalues_normalized' in f:
            return 'process_nisar'
        else:
            raise ValueError(f"Unknown HDF5 format in {h5_path}")


def load_cpi_eigenvalues(h5_path, ci, ri):
    """
    Load eigenvalue data for a specific CPI tile.
    Supports both tile_cpi.py and process_nisar_to_cpi.py formats.

    Args:
        h5_path (str): Path to HDF5 file
        ci (int): CPI row index (pulse dimension)
        ri (int): Range column index

    Returns:
        eigen_input (np.ndarray): Shape (M, 2) with [ev_db, slope_db]
        global_input (np.ndarray|None): Shape (6,) with global features (None for process_nisar format)
        valid (bool): Whether the CPI is valid (True for process_nisar format)
        M (int): Number of pulses per CPI
        metadata (dict): Additional metadata
    """
    fmt = detect_hdf5_format(h5_path)

    with h5py.File(h5_path, 'r') as f:
        # Get root metadata
        if 'M' in f.attrs:
            M = int(f.attrs['M'])
            cpi_height = M
        elif 'cpi_height' in f.attrs:
            cpi_height = int(f.attrs['cpi_height'])
            M = cpi_height
        else:
            # Try to infer from first CPI dataset
            cpi_keys = [k for k in f.keys() if k.startswith('cpi_') and '_eigenvalues' not in k and '_diagonal' not in k]
            if cpi_keys:
                first_cpi = f[cpi_keys[0]]
                if isinstance(first_cpi, h5py.Group) and 'eigen_input' in first_cpi:
                    M = len(first_cpi['eigen_input'][:])
                else:
                    M = first_cpi.shape[0]
            else:
                M = 16  # Default

        if fmt == 'tile_cpi':
            key = f'cpi_{ci}_{ri}'
            if key not in f:
                n_cpi_rows = int(f.attrs.get('n_cpi_rows', '?'))
                n_range_cols = int(f.attrs.get('n_range_cols', '?'))
                raise KeyError(f"CPI tile '{key}' not found", n_cpi_rows, n_range_cols)

            grp = f[key]
            eigen_input = grp['eigen_input'][:]
            global_input = grp['global_input'][:]
            valid = bool(grp['valid'][()])

            BLOCK_WIDTH = int(f.attrs.get('BLOCK_WIDTH', grp.attrs.get('range_end', 0) - grp.attrs.get('range_start', 0)))
            n_cpi_rows = int(f.attrs.get('n_cpi_rows', 0))
            n_range_cols = int(f.attrs.get('n_range_cols', 0))
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
                'format': 'tile_cpi',
                'global_features': {
                    'F_factor': float(global_input[0]),
                    'sigma_min': float(global_input[1]),
                    'sigma_max': float(global_input[2]),
                    'mu_min': float(global_input[3]),
                    'trace_db': float(global_input[4]),
                    'cond_number': float(global_input[5]),
                }
            }

        else:  # process_nisar format
            # Calculate pulse and range indices from ci, ri
            cpi_width = int(f.attrs.get('cpi_width', 250))
            pulse_start = ci * M
            range_start = ri * cpi_width

            key = f'cpi_{pulse_start}_{range_start}'
            if key not in f:
                n_pulse_tiles = int(f.attrs.get('n_pulse_tiles', '?'))
                n_range_tiles = int(f.attrs.get('n_range_tiles', '?'))
                raise KeyError(f"CPI tile '{key}' not found (try indices based on pulse/range, not tile count)", n_pulse_tiles, n_range_tiles)

            # Load eigenvalues
            if f'{key}_eigenvalues' in f:
                ev_linear = f[f'{key}_eigenvalues'][:]
                ev_db = 10.0 * np.log10(np.maximum(ev_linear, 1e-12))
            elif f'{key}_eigenvalues_normalized' in f:
                ev_normalized = f[f'{key}_eigenvalues_normalized'][:]
                # Try to get max eigenvalue from attributes
                max_eigval_db = f[key].attrs.get('max_eigval_db', None)
                if max_eigval_db is not None:
                    # Reconstruct dB values from normalized
                    ev_db = max_eigval_db + 10.0 * np.log10(np.maximum(ev_normalized, 1e-12))
                else:
                    # Just use normalized values scaled by their max
                    ev_linear = ev_normalized / np.max(ev_normalized)
                    ev_db = 10.0 * np.log10(np.maximum(ev_linear, 1e-12))
            else:
                raise ValueError(f"No eigenvalue data found for {key}")

            # Create slopes (dummy values since not available)
            slopes = np.diff(ev_db)
            slopes = np.append(slopes, slopes[-1])
            eigen_input = np.stack([ev_db, slopes], axis=1).astype(np.float32)

            global_input = None
            valid = True

            n_pulse_tiles = int(f.attrs.get('n_pulse_tiles', 0))
            n_range_tiles = int(f.attrs.get('n_range_tiles', 0))

            metadata = {
                'M': M,
                'BLOCK_WIDTH': cpi_width,
                'n_cpi_rows': n_pulse_tiles,
                'n_range_cols': n_range_tiles,
                'pulse_start': pulse_start,
                'pulse_end': pulse_start + M,
                'range_start': range_start,
                'range_end': range_start + cpi_width,
                'format': 'process_nisar',
                'global_features': None
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

    # Global features for annotation (may be None for process_nisar format)
    gf = metadata.get('global_features')

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

    # Add text box with global features (if available)
    if gf is not None:
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
                        help='Path to HDF5 file')
    parser.add_argument('--row', type=int,
                        help='CPI row index: tile_cpi format uses ci (0, 1, 2, ...), '
                             'process_nisar format uses pulse_start (0, 16, 32, ...). '
                             'Use --list to see available indices.')
    parser.add_argument('--col', type=int,
                        help='CPI column index: tile_cpi format uses ri (0, 1, 2, ...), '
                             'process_nisar format uses range_start (0, 250, 500, ...). '
                             'Use --list to see available indices.')
    parser.add_argument('--out', default=DEFAULT_OUT,
                        help='Output directory for plots')
    parser.add_argument('--list', action='store_true',
                        help='List available CPI tiles and exit')

    args = parser.parse_args()

    if not os.path.exists(args.h5):
        print(f"ERROR: HDF5 file not found: {args.h5}")
        sys.exit(1)

    # List mode (--row and --col not required for list)
    if args.list:
        print("="*70)
        print("Available CPI Tiles")
        print("="*70)
        print(f"HDF5 file: {args.h5}")
        print()

        indices, total = list_available_cpis(args.h5, max_show=20)
        fmt = detect_hdf5_format(args.h5)

        print(f"Format: {fmt}")
        print(f"Total CPI tiles: {total}")
        print()
        print("First 20 CPI tiles:")

        if fmt == 'process_nisar':
            print("  (pulse_start, range_start)")
            for pulse, range_idx in indices:
                print(f"    ({pulse}, {range_idx})")
            print()
            print("To plot, use: --row <pulse_start> --col <range_start>")
            print(f"Example: python {sys.argv[0]} --h5 {args.h5} --row {indices[0][0]} --col {indices[0][1]}")
        else:
            print("  (ci, ri)")
            for ci, ri in indices:
                print(f"    ({ci}, {ri})")
            print()
            print("To plot, use: --row <ci> --col <ri>")
            print(f"Example: python {sys.argv[0]} --h5 {args.h5} --row {indices[0][0]} --col {indices[0][1]}")

        sys.exit(0)

    # Check that row and col are provided when not in list mode
    if args.row is None or args.col is None:
        print("ERROR: --row and --col are required (or use --list to see available indices)")
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
        print(f"ERROR: {e.args[0]}")
        if len(e.args) > 2:
            n_cpi_rows = e.args[1]
            n_range_cols = e.args[2]
            print(f"\nAvailable CPI indices:")
            print(f"  Rows (ci): 0 to {n_cpi_rows - 1 if isinstance(n_cpi_rows, int) else '?'}")
            print(f"  Cols (ri): 0 to {n_range_cols - 1 if isinstance(n_range_cols, int) else '?'}")
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"  Format: {metadata.get('format', 'unknown')}")
    print(f"  M (pulses per CPI): {M}")
    print(f"  Valid: {valid}")
    print(f"  Pulse range: {metadata['pulse_start']} to {metadata['pulse_end']}")
    print(f"  Range bins: {metadata['range_start']} to {metadata['range_end']}")

    gf = metadata.get('global_features')
    if gf is not None:
        print(f"\n  Global Features:")
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
