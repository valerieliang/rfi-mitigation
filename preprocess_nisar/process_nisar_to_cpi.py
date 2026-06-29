"""
process_nisar_to_cpi.py

Process NISAR data (HH or HV polarization) into CPI blocks matching the synthetic data format.

Automatically detects polarization from the dataset path.

Supports two input modes:
  1. Decoded .npy file (memory-mapped complex64 array)
  2. Raw NISAR HDF5 file with dataset path (auto-decodes using BFPQLUT)

Divides the input into non-overlapping CPI tiles of:
  - 16 pulses (rows) x 250 range samples (columns)

For each CPI tile, computes and stores:
  - Raw complex data
  - Eigenvalues (sorted descending, normalized to [0,1])
  - SCM diagonal values
  - Maximum eigenvalue in dB

Output format matches generate_synthetic_data.py HDF5 structure for consistency
with the trained model pipeline.

Usage:
  # From raw NISAR HDF5 (HV polarization):
  python process_nisar_to_cpi.py nisar.h5 output_hv.h5 \\
    --dataset /science/LSAR/RRSD/swaths/frequencyA/txH/rxV/HV

  # From raw NISAR HDF5 (HH polarization):
  python process_nisar_to_cpi.py nisar.h5 output_hh.h5 \\
    --dataset /science/LSAR/RRSD/swaths/frequencyA/txH/rxH/HH

  # From decoded .npy file:
  python process_nisar_to_cpi.py hv_decoded.npy output_hv.h5 \\
    --shape 182760 52866 --polarization HV
"""

import os
import sys
import numpy as np
import h5py
from pathlib import Path
import argparse
import re


# CPI tile dimensions (must match training data)
CPI_HEIGHT = 16   # pulses per CPI block
CPI_WIDTH = 250   # range samples per CPI tile


def detect_polarization(dataset_path=None, polarization_arg=None):
    """
    Detect polarization from dataset path or explicit argument.

    Args:
        dataset_path (str|None): HDF5 dataset path (e.g., '/science/.../HV')
        polarization_arg (str|None): Explicit polarization ('HH', 'HV', etc.)

    Returns:
        str: Polarization string ('HH', 'HV', 'VH', 'VV')
    """
    if polarization_arg:
        pol = polarization_arg.upper()
        if pol not in ['HH', 'HV', 'VH', 'VV']:
            raise ValueError(f"Invalid polarization: {pol}. Must be HH, HV, VH, or VV")
        return pol

    if dataset_path:
        # Extract from path like: /science/LSAR/RRSD/swaths/frequencyA/txH/rxV/HV
        match = re.search(r'/(HH|HV|VH|VV)$', dataset_path)
        if match:
            return match.group(1)

    raise ValueError("Could not detect polarization. Provide --polarization or use dataset path ending in HH/HV/VH/VV")


def compute_eigenvalues_normalized(cpi):
    """
    Compute sorted descending eigenvalues normalized to [0, 1] from a complex CPI tile.

    Forms the sample covariance matrix SCM = M @ M^H / K where K is the
    number of range bins (columns), then returns eigenvalues sorted descending
    normalized by the largest eigenvalue.

    Args:
        cpi (np.ndarray): Complex array of shape (M, K) where M = CPI_HEIGHT, K = CPI_WIDTH.

    Returns:
        eigvals_normalized (np.ndarray): Shape (M,), eigenvalues normalized to [0,1].
        max_eigval_db (float): Largest eigenvalue in dB (10*log10).
    """
    M, K = cpi.shape
    SCM = (cpi @ cpi.conj().T) / K            # (M, M) sample covariance
    eigvals = np.linalg.eigvalsh(SCM)         # ascending real eigenvalues
    eigvals = np.sort(eigvals)[::-1]          # descending
    max_eigval = eigvals[0]
    max_eigval_db = 10.0 * np.log10(max(max_eigval, 1e-12))
    eigvals_normalized = eigvals / max(max_eigval, 1e-12)
    return eigvals_normalized, max_eigval_db


def load_data_source(input_path, dataset_path=None, h5_shape=None):
    """
    Load data from either a decoded .npy file or raw NISAR HDF5.

    Args:
        input_path (str): Path to input file (.npy or .h5)
        dataset_path (str|None): HDF5 dataset path (e.g., '/science/.../HV')
        h5_shape (tuple|None): Expected shape for .npy memmap, or None to auto-detect from HDF5

    Returns:
        data_accessor: Object with .shape and indexing support (memmap or H5DataAccessor)
        source_type (str): 'npy' or 'hdf5'
    """
    input_path = Path(input_path)

    if input_path.suffix == '.npy':
        # Decoded numpy array
        if h5_shape is None:
            raise ValueError("Must provide h5_shape for .npy files")

        print(f"  Source: Decoded .npy file")
        data = np.memmap(str(input_path), dtype='complex64', mode='r', shape=h5_shape)
        return data, 'npy'

    elif input_path.suffix in ['.h5', '.hdf5']:
        # Raw NISAR HDF5 - needs decoding
        if dataset_path is None:
            raise ValueError("Must provide dataset_path for HDF5 files")

        print(f"  Source: Raw NISAR HDF5")
        print(f"  Dataset: {dataset_path}")

        # Create accessor that decodes on-the-fly
        accessor = H5DataAccessor(str(input_path), dataset_path)
        return accessor, 'hdf5'

    else:
        raise ValueError(f"Unsupported file type: {input_path.suffix}. Use .npy or .h5/.hdf5")


class H5DataAccessor:
    """
    Provides array-like access to NISAR HDF5 data with on-the-fly BFPQLUT decoding.

    Acts like a numpy array but decodes chunks as they're accessed.
    """

    def __init__(self, h5_path, dataset_path):
        self.h5_path = h5_path
        self.dataset_path = dataset_path

        # Open file and get metadata
        with h5py.File(h5_path, 'r') as f:
            dataset = f[dataset_path]
            self.shape = dataset.shape
            self.dtype = np.complex64

            # Extract BFPQLUT from the same parent path
            # Dataset path like: /science/LSAR/RRSD/swaths/frequencyA/txH/rxV/HV
            # BFPQLUT at: /science/LSAR/RRSD/swaths/frequencyA/txH/rxV/BFPQLUT
            parent_path = '/'.join(dataset_path.split('/')[:-1])
            bfpqlut_path = f"{parent_path}/BFPQLUT"

            if bfpqlut_path not in f:
                raise ValueError(f"BFPQLUT not found at {bfpqlut_path}")

            self.bfpqlut = f[bfpqlut_path][:]

        print(f"  Shape: {self.shape}")
        print(f"  BFPQLUT loaded: {len(self.bfpqlut)} entries")

    def __getitem__(self, key):
        """Access data with on-the-fly decoding."""
        with h5py.File(self.h5_path, 'r') as f:
            dataset = f[self.dataset_path]

            # Read quantized data
            quantized = dataset[key]

            # Decode using BFPQLUT
            real_decoded = self.bfpqlut[quantized['r']]
            imag_decoded = self.bfpqlut[quantized['i']]

            return (real_decoded + 1j * imag_decoded).astype(np.complex64)

    def copy(self):
        """For compatibility with code that calls .copy()"""
        return self


def process_nisar_to_cpi(
    input_path,
    output_h5_path,
    dataset_path=None,
    input_shape=None,
    polarization=None,
    cpi_height=CPI_HEIGHT,
    cpi_width=CPI_WIDTH,
    max_range_lines=None,
    range_start=None,
    range_end=None
):
    """
    Process NISAR data into CPI tiles and save as HDF5.

    Args:
        input_path (str): Path to input file (.npy or .h5)
        output_h5_path (str): Output HDF5 file path
        dataset_path (str|None): HDF5 dataset path (required for .h5 input)
        input_shape (tuple|None): Shape for .npy memmap (required for .npy input)
        polarization (str|None): Polarization ('HH', 'HV', etc.) - auto-detected from dataset_path if not provided
        cpi_height (int): Pulses per CPI tile (default 16)
        cpi_width (int): Range samples per CPI tile (default 250)
        max_range_lines (int|None): Limit processing to first N range lines (for testing)
        range_start (int|None): Start at this range line (must be multiple of cpi_height)
        range_end (int|None): End at this range line (must be multiple of cpi_height)
    """

    # Detect polarization
    pol = detect_polarization(dataset_path, polarization)

    print("="*70)
    print(f"Processing NISAR {pol} Data to CPI Tiles")
    print("="*70)

    # Load data source
    print(f"\nLoading: {input_path}")
    data, _ = load_data_source(input_path, dataset_path, input_shape)

    total_pulses = data.shape[0]
    total_range = data.shape[1]

    # Handle range selection
    pulse_start = 0
    pulse_end = total_pulses

    if range_start is not None or range_end is not None:
        if range_start is not None:
            if range_start % cpi_height != 0:
                raise ValueError(f"range_start ({range_start}) must be multiple of cpi_height ({cpi_height})")
            pulse_start = range_start
        if range_end is not None:
            if range_end % cpi_height != 0:
                raise ValueError(f"range_end ({range_end}) must be multiple of cpi_height ({cpi_height})")
            pulse_end = range_end

        total_pulses = pulse_end - pulse_start
        print(f"  Processing range: lines {pulse_start} to {pulse_end} ({total_pulses} lines)")

    elif max_range_lines is not None:
        total_pulses = min(max_range_lines, total_pulses)
        pulse_end = pulse_start + total_pulses
        print(f"  Limiting to first {total_pulses} range lines")

    print(f"  Full shape: {data.shape}")
    print(f"  Processing: ({total_pulses}, {total_range})")
    print(f"  Dtype: {data.dtype}")
    print(f"  Polarization: {pol}")

    # Calculate number of tiles
    n_pulse_tiles = total_pulses // cpi_height
    n_range_tiles = total_range // cpi_width

    # Truncate to fit tiles evenly
    total_pulses_used = n_pulse_tiles * cpi_height
    total_range_used = n_range_tiles * cpi_width

    print(f"\nCPI Tile Configuration:")
    print(f"  CPI size: {cpi_height} pulses × {cpi_width} samples")
    print(f"  Pulse tiles: {n_pulse_tiles}")
    print(f"  Range tiles: {n_range_tiles}")
    print(f"  Total CPI tiles: {n_pulse_tiles * n_range_tiles:,}")
    print(f"  Coverage: {total_pulses_used}/{total_pulses} pulses, {total_range_used}/{total_range} samples")

    # Create output directory
    os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)

    # Create HDF5 file
    print(f"\nCreating: {output_h5_path}")

    with h5py.File(output_h5_path, 'w') as f:

        # Root attributes
        f.attrs['source'] = 'NISAR_L0_PR_RRSD'
        f.attrs['polarization'] = pol
        f.attrs['total_pulses'] = total_pulses_used
        f.attrs['range_bins'] = total_range_used
        f.attrs['cpi_height'] = cpi_height
        f.attrs['cpi_width'] = cpi_width
        f.attrs['n_pulse_tiles'] = n_pulse_tiles
        f.attrs['n_range_tiles'] = n_range_tiles
        f.attrs['n_cpi_tiles'] = n_pulse_tiles * n_range_tiles

        # Process each CPI tile
        total_tiles = n_pulse_tiles * n_range_tiles
        tile_count = 0

        print(f"\nProcessing CPI tiles...")

        for i in range(pulse_start, pulse_start + total_pulses_used, cpi_height):
            for j in range(0, total_range_used, cpi_width):

                # Extract CPI tile
                cpi = data[i:i+cpi_height, j:j+cpi_width].copy()  # Copy to get actual data

                # Compute SCM: M * M^H / cpi_width
                M = cpi
                SCM = (M @ M.conj().T) / cpi_width

                # Extract diagonal and eigenvalues
                diagonal = np.diag(SCM).real
                eigvals = np.linalg.eigvalsh(SCM)
                eigvals_sorted = np.sort(eigvals)[::-1]  # Largest to smallest

                # Normalized eigenvalues
                eigvals_normalized, max_eigval_db = compute_eigenvalues_normalized(cpi)

                # Save CPI data
                dset = f.create_dataset(f"cpi_{i}_{j}", data=cpi)

                # Save eigenvalue analysis
                f.create_dataset(f"cpi_{i}_{j}_eigenvalues", data=eigvals_sorted)
                f.create_dataset(f"cpi_{i}_{j}_eigenvalues_normalized", data=eigvals_normalized)
                f.create_dataset(f"cpi_{i}_{j}_diagonal", data=diagonal)

                # Store max eigenvalue as attribute
                dset.attrs['max_eigval_db'] = max_eigval_db

                # Progress
                tile_count += 1
                if tile_count % 1000 == 0 or tile_count == total_tiles:
                    percent = 100 * tile_count / total_tiles
                    print(f"  [{percent:5.1f}%] Processed {tile_count:,}/{total_tiles:,} tiles")

    print(f"\n{'='*70}")
    print("Processing Complete!")
    print(f"{'='*70}")
    print(f"Output: {output_h5_path}")
    print(f"Polarization: {pol}")
    print(f"Size: {Path(output_h5_path).stat().st_size / (1024**3):.2f} GB")
    print(f"Total CPI tiles: {tile_count:,}")


def verify_h5_file(h5_path, n_samples=5):
    """Verify the HDF5 file structure and sample some tiles."""

    print(f"\n{'='*70}")
    print("Verifying HDF5 File")
    print(f"{'='*70}")

    with h5py.File(h5_path, 'r') as f:

        print(f"\nRoot Attributes:")
        for key in f.attrs:
            print(f"  {key}: {f.attrs[key]}")

        # Count datasets
        cpi_datasets = [k for k in f.keys() if k.startswith('cpi_') and '_eigenvalues' not in k and '_diagonal' not in k]
        print(f"\nDatasets:")
        print(f"  CPI tiles: {len(cpi_datasets)}")

        # Sample a few tiles
        print(f"\nSampling {n_samples} random tiles:")
        np.random.seed(42)
        sample_keys = np.random.choice(cpi_datasets, size=min(n_samples, len(cpi_datasets)), replace=False)

        for key in sample_keys:
            dset = f[key]
            cpi_data = dset[:]

            # Check for associated datasets
            has_eigvals = f"{key}_eigenvalues" in f
            has_eigvals_norm = f"{key}_eigenvalues_normalized" in f
            has_diagonal = f"{key}_diagonal" in f

            max_eigval_db = dset.attrs.get('max_eigval_db', None)

            print(f"\n  {key}:")
            print(f"    Shape: {cpi_data.shape}")
            print(f"    Dtype: {cpi_data.dtype}")
            print(f"    Value range: [{cpi_data.real.min():.2f}, {cpi_data.real.max():.2f}] (real)")
            print(f"    Max eigenvalue: {max_eigval_db:.2f} dB" if max_eigval_db else "    Max eigenvalue: N/A")
            print(f"    Has eigenvalues: {has_eigvals}")
            print(f"    Has normalized eigenvalues: {has_eigvals_norm}")
            print(f"    Has diagonal: {has_diagonal}")

            if has_eigvals_norm:
                eigvals_norm = f[f"{key}_eigenvalues_normalized"][:]
                print(f"    Normalized eigenvalues: {eigvals_norm[:5]} ...")

    print(f"\n{'='*70}")
    print("Verification Complete")
    print(f"{'='*70}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process NISAR data (any polarization) into CPI tiles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From raw NISAR HDF5 (polarization auto-detected from dataset path):
  python process_nisar_to_cpi.py nisar.h5 output_hv.h5 \\
    --dataset /science/LSAR/RRSD/swaths/frequencyA/txH/rxV/HV

  python process_nisar_to_cpi.py nisar.h5 output_hh.h5 \\
    --dataset /science/LSAR/RRSD/swaths/frequencyA/txH/rxH/HH

  # From decoded .npy file (must specify shape and polarization):
  python process_nisar_to_cpi.py hv_decoded.npy output_hv.h5 \\
    --shape 182760 52866 --polarization HV

  # With range line limit for testing:
  python process_nisar_to_cpi.py nisar.h5 output_hv.h5 \\
    --dataset /science/LSAR/RRSD/swaths/frequencyA/txH/rxV/HV \\
    --range-start 0 --range-end 1600
        """
    )

    parser.add_argument('input', help='Input file (.npy or .h5/.hdf5)')
    parser.add_argument('output', help='Output HDF5 file')
    parser.add_argument('--dataset', help='HDF5 dataset path (required for .h5 input)')
    parser.add_argument('--shape', nargs=2, type=int, metavar=('ROWS', 'COLS'),
                        help='Shape for .npy memmap (required for .npy input)')
    parser.add_argument('--polarization', choices=['HH', 'HV', 'VH', 'VV'],
                        help='Polarization (auto-detected from dataset path if not provided)')
    parser.add_argument('--cpi-height', type=int, default=16,
                        help='CPI height in pulses (default: 16)')
    parser.add_argument('--cpi-width', type=int, default=250,
                        help='CPI width in samples (default: 250)')
    parser.add_argument('--max-range-lines', type=int, default=None,
                        help='Limit processing to first N range lines (for testing)')
    parser.add_argument('--range-start', type=int, default=None,
                        help='Start at this range line index (must be multiple of cpi_height)')
    parser.add_argument('--range-end', type=int, default=None,
                        help='End at this range line index (must be multiple of cpi_height)')

    args = parser.parse_args()

    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    # Determine input type and validate required args
    if input_path.suffix == '.npy':
        if args.shape is None:
            print(f"ERROR: --shape required for .npy input")
            print(f"Usage: python process_nisar_to_cpi.py {args.input} {args.output} --shape ROWS COLS --polarization HV")
            sys.exit(1)
        input_shape = tuple(args.shape)
        dataset_path = None
    elif input_path.suffix in ['.h5', '.hdf5']:
        if args.dataset is None:
            print(f"ERROR: --dataset required for HDF5 input")
            print(f"Usage: python process_nisar_to_cpi.py {args.input} {args.output} --dataset /path/to/dataset")
            sys.exit(1)
        input_shape = None
        dataset_path = args.dataset
    else:
        print(f"ERROR: Unsupported file type: {input_path.suffix}")
        print(f"Supported: .npy, .h5, .hdf5")
        sys.exit(1)

    # Validate range arguments
    if args.range_start is not None or args.range_end is not None:
        if args.max_range_lines is not None:
            print("ERROR: Cannot use --max-range-lines with --range-start/--range-end")
            sys.exit(1)
        if args.range_start is not None and args.range_start % args.cpi_height != 0:
            print(f"ERROR: --range-start ({args.range_start}) must be multiple of --cpi-height ({args.cpi_height})")
            sys.exit(1)
        if args.range_end is not None and args.range_end % args.cpi_height != 0:
            print(f"ERROR: --range-end ({args.range_end}) must be multiple of --cpi-height ({args.cpi_height})")
            sys.exit(1)

    # Process
    process_nisar_to_cpi(
        input_path=args.input,
        output_h5_path=args.output,
        dataset_path=dataset_path,
        input_shape=input_shape,
        polarization=args.polarization,
        cpi_height=args.cpi_height,
        cpi_width=args.cpi_width,
        max_range_lines=args.max_range_lines,
        range_start=args.range_start,
        range_end=args.range_end
    )

    # Verify
    verify_h5_file(args.output, n_samples=5)
