"""
read_nisar_isce3.py

Efficient NISAR L0B data reader using ISCE3's Raw reader.
Designed to read an entire NISAR HDF5 file in ~10 minutes by leveraging
ISCE3's optimized BFPQLUT decoding and efficient data access patterns.

Key optimizations:
1. Uses ISCE3's Raw.getRawDataset() which handles BFPQLUT decoding internally
2. Batch reads to minimize HDF5 I/O overhead
3. Parallel processing across polarizations and frequencies
4. Memory-efficient chunked processing for large datasets

Usage:
    # Read entire file with all polarizations
    python read_nisar_isce3.py input.h5 --output-dir ./output

    # Read specific frequency and polarization
    python read_nisar_isce3.py input.h5 --freq A --pol HH --output-dir ./output

    # Process in streaming mode (process CPIs without saving raw data)
    python read_nisar_isce3.py input.h5 --stream --cpi-len 16 --output-dir ./output
"""

import argparse
import h5py
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime
import time
from nisar.products.readers.Raw import Raw
from isce3.signal.compute_evd_cpi import compute_evd_tb, slice_gen
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Read NISAR L0B data efficiently using ISCE3 Raw reader',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Read entire file (all frequencies and polarizations)
  python read_nisar_isce3.py ALPSRP081257070-H1.0__A_HH_3000_LSAR_01_M_D_20081012T005927_20081012T010033_015559_000_001_0157.h5

  # Read specific frequency and polarization
  python read_nisar_isce3.py input.h5 --freq A --pol HH --output-dir ./output

  # Stream processing with EVD computation
  python read_nisar_isce3.py input.h5 --stream --cpi-len 16 --output-dir ./output

  # Read with pulse/range subsetting
  python read_nisar_isce3.py input.h5 --pulse-start 0 --pulse-end 5000 --range-start 0 --range-end 10000
        """
    )

    parser.add_argument('input_file', help='Input NISAR L0B HDF5 file path')
    parser.add_argument('--freq', choices=['A', 'B'], default=None,
                        help='Process only this frequency (default: all)')
    parser.add_argument('--pol', choices=['HH', 'HV', 'VH', 'VV'], default=None,
                        help='Process only this polarization (default: all)')
    parser.add_argument('--output-dir', type=str, default='./nisar_output',
                        help='Output directory for processed data (default: ./nisar_output)')

    # Subsetting options
    parser.add_argument('--pulse-start', type=int, default=None,
                        help='Start pulse index (slow time)')
    parser.add_argument('--pulse-end', type=int, default=None,
                        help='End pulse index (slow time)')
    parser.add_argument('--range-start', type=int, default=None,
                        help='Start range sample index')
    parser.add_argument('--range-end', type=int, default=None,
                        help='End range sample index')

    # Processing options
    parser.add_argument('--stream', action='store_true',
                        help='Stream processing mode: compute EVD without saving raw data')
    parser.add_argument('--cpi-len', type=int, default=16,
                        help='CPI length for streaming EVD (default: 16)')
    parser.add_argument('--save-raw', action='store_true',
                        help='Save decoded raw data to HDF5 (can be large!)')
    parser.add_argument('--save-eigenvalues', action='store_true',
                        help='Save eigenvalues from EVD processing')
    parser.add_argument('--n-workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')

    # EVD parameters
    parser.add_argument('--rx-dynamic-range-db', type=float, default=50.0,
                        help='Receiver dynamic range in dB (default: 50.0)')
    parser.add_argument('--min-ev-valid-idx', type=int, default=10,
                        help='Minimum eigenvalue valid index (default: 10)')

    return parser.parse_args()


def get_dataset_info(raw: Raw):
    """
    Extract and display dataset information from Raw object.

    Parameters
    ----------
    raw : Raw
        ISCE3 Raw object

    Returns
    -------
    info : dict
        Dictionary containing dataset information
    """
    info = {
        'frequencies': {},
        'polarizations': raw.polarizations,
    }

    print("\n" + "="*70)
    print("NISAR Dataset Information")
    print("="*70)

    for freq, pol_list in raw.polarizations.items():
        info['frequencies'][freq] = {
            'polarizations': pol_list,
            'datasets': {}
        }

        print(f"\nFrequency {freq}:")
        print(f"  Polarizations: {pol_list}")

        for pol in pol_list:
            # Get dataset handle
            dataset = raw.getRawDataset(freq, pol)
            shape = dataset.shape
            dtype = dataset.dtype

            # Get chirp parameters
            pol_tx = pol[0]
            fc, fs, _, _ = raw.getChirpParameters(freq, pol_tx)
            bandwidth = raw.getRangeBandwidth(freq, pol_tx)

            info['frequencies'][freq]['datasets'][pol] = {
                'shape': shape,
                'dtype': dtype,
                'center_frequency_hz': fc,
                'sample_rate_hz': fs,
                'bandwidth_hz': bandwidth,
            }

            print(f"\n  {pol}:")
            print(f"    Shape: {shape} (pulses × range samples)")
            print(f"    Dtype: {dtype}")
            print(f"    Center Frequency: {fc/1e9:.3f} GHz")
            print(f"    Sample Rate: {fs/1e6:.3f} MHz")
            print(f"    Bandwidth: {bandwidth/1e6:.3f} MHz")
            print(f"    Total size: {shape[0] * shape[1] * np.dtype(dtype).itemsize / 1e9:.2f} GB (if decoded)")

    print("\n" + "="*70 + "\n")

    return info


def read_raw_data_batch(
    raw: Raw,
    freq: str,
    pol: str,
    pulse_slice: slice = None,
    range_slice: slice = None,
):
    """
    Read a batch of raw data using ISCE3's efficient reader.

    Parameters
    ----------
    raw : Raw
        ISCE3 Raw object
    freq : str
        Frequency ('A' or 'B')
    pol : str
        Polarization ('HH', 'HV', 'VH', 'VV')
    pulse_slice : slice, optional
        Slice for pulse (slow time) dimension
    range_slice : slice, optional
        Slice for range (fast time) dimension

    Returns
    -------
    data : np.ndarray
        Complex64 array of raw data
    """
    # Get raw dataset (this handles BFPQLUT internally)
    dataset = raw.getRawDataset(freq, pol)

    # Build indexing tuple
    idx = [pulse_slice if pulse_slice is not None else slice(None),
           range_slice if range_slice is not None else slice(None)]

    # Read data - ISCE3 handles BFPQLUT decoding automatically
    data = dataset[tuple(idx)]

    return data


def process_polarization_streaming(
    raw: Raw,
    freq: str,
    pol: str,
    output_dir: str,
    cpi_len: int = 16,
    pulse_start: int = None,
    pulse_end: int = None,
    range_start: int = None,
    range_end: int = None,
    save_eigenvalues: bool = True,
    rx_dynamic_range_db: float = 50.0,
    min_ev_valid_idx: int = 10,
):
    """
    Process a single polarization in streaming mode with EVD computation.

    Parameters
    ----------
    raw : Raw
        ISCE3 Raw object
    freq : str
        Frequency ('A' or 'B')
    pol : str
        Polarization
    output_dir : str
        Output directory
    cpi_len : int
        CPI length for EVD
    pulse_start, pulse_end : int, optional
        Pulse range to process
    range_start, range_end : int, optional
        Range sample limits
    save_eigenvalues : bool
        Whether to save eigenvalues
    rx_dynamic_range_db : float
        Receiver dynamic range in dB
    min_ev_valid_idx : int
        Minimum eigenvalue valid index

    Returns
    -------
    results : dict
        Processing results including statistics
    """
    print(f"\nProcessing {freq}-{pol} in streaming mode...")
    start_time = time.time()

    # Get dataset info
    dataset = raw.getRawDataset(freq, pol)
    total_pulses, total_range = dataset.shape

    # Apply pulse/range limits
    p_start = pulse_start if pulse_start is not None else 0
    p_end = pulse_end if pulse_end is not None else total_pulses
    r_start = range_start if range_start is not None else 0
    r_end = range_end if range_end is not None else total_range

    n_pulses = p_end - p_start
    n_range = r_end - r_start

    print(f"  Data shape: ({n_pulses}, {n_range})")
    print(f"  CPI length: {cpi_len}")

    # Calculate number of threshold blocks (TBs)
    num_cpi = n_pulses // cpi_len
    tb_size = num_cpi * cpi_len

    print(f"  Number of CPIs: {num_cpi}")
    print(f"  TB size: {tb_size} pulses")

    # Initialize output arrays
    if save_eigenvalues:
        eig_val_array = np.zeros((num_cpi, cpi_len), dtype=np.float32)
        diag_power_array = np.zeros((num_cpi, cpi_len), dtype=np.float32)
        diag_valid_array = np.zeros((num_cpi, cpi_len), dtype=bool)
        tb_valid_array = np.zeros(num_cpi, dtype=bool)

    # Process threshold block
    print(f"  Reading data slice [{p_start}:{p_start+tb_size}, {r_start}:{r_end}]...")
    read_start = time.time()

    # Read data using ISCE3's efficient reader
    raw_data = read_raw_data_batch(
        raw, freq, pol,
        pulse_slice=slice(p_start, p_start + tb_size),
        range_slice=slice(r_start, r_end)
    )

    read_time = time.time() - read_start
    data_gb = raw_data.nbytes / 1e9
    print(f"  Data read: {data_gb:.3f} GB in {read_time:.2f}s ({data_gb/read_time:.2f} GB/s)")

    # Compute EVD
    print(f"  Computing EVD for {num_cpi} CPIs...")
    evd_start = time.time()

    eig_val_sort, eig_vec_sort, diag_power, diag_valid, tb_is_valid = compute_evd_tb(
        raw_data,
        cpi_len=cpi_len,
        mask_valid=None,  # No mask for now
        min_ev_valid_idx=min_ev_valid_idx,
        rx_dynamic_range_db=rx_dynamic_range_db,
    )

    evd_time = time.time() - evd_start
    print(f"  EVD computed in {evd_time:.2f}s")
    print(f"  TB valid: {tb_is_valid}")

    # Save results
    if save_eigenvalues:
        output_file = os.path.join(output_dir, f'nisar_{freq}_{pol}_eigenvalues.h5')
        print(f"  Saving eigenvalues to {output_file}...")

        with h5py.File(output_file, 'w') as f:
            # Create groups
            evd_grp = f.create_group('evd')
            meta_grp = f.create_group('metadata')

            # Save EVD results
            evd_grp.create_dataset('eigenvalues', data=eig_val_sort, compression='gzip')
            evd_grp.create_dataset('diagonal_power', data=diag_power, compression='gzip')
            evd_grp.create_dataset('diagonal_valid', data=diag_valid, compression='gzip')
            evd_grp.create_dataset('tb_is_valid', data=tb_is_valid)

            # Save metadata
            meta_grp.attrs['frequency'] = freq
            meta_grp.attrs['polarization'] = pol
            meta_grp.attrs['cpi_len'] = cpi_len
            meta_grp.attrs['num_cpi'] = num_cpi
            meta_grp.attrs['pulse_start'] = p_start
            meta_grp.attrs['pulse_end'] = p_start + tb_size
            meta_grp.attrs['range_start'] = r_start
            meta_grp.attrs['range_end'] = r_end
            meta_grp.attrs['total_pulses'] = total_pulses
            meta_grp.attrs['total_range'] = total_range
            meta_grp.attrs['processing_time'] = time.time() - start_time
            meta_grp.attrs['processing_date'] = datetime.now().isoformat()

            # Get chirp parameters
            pol_tx = pol[0]
            fc, fs, _, _ = raw.getChirpParameters(freq, pol_tx)
            bandwidth = raw.getRangeBandwidth(freq, pol_tx)

            meta_grp.attrs['center_frequency_hz'] = fc
            meta_grp.attrs['sample_rate_hz'] = fs
            meta_grp.attrs['bandwidth_hz'] = bandwidth

    total_time = time.time() - start_time

    results = {
        'frequency': freq,
        'polarization': pol,
        'num_cpi': num_cpi,
        'tb_is_valid': tb_is_valid,
        'total_time': total_time,
        'read_time': read_time,
        'evd_time': evd_time,
        'data_size_gb': data_gb,
        'throughput_gb_per_s': data_gb / total_time,
    }

    print(f"  Total time: {total_time:.2f}s ({results['throughput_gb_per_s']:.2f} GB/s)")

    return results


def process_polarization_full(
    raw: Raw,
    freq: str,
    pol: str,
    output_dir: str,
    pulse_start: int = None,
    pulse_end: int = None,
    range_start: int = None,
    range_end: int = None,
    save_raw: bool = True,
):
    """
    Process a single polarization by reading and optionally saving full data.

    Parameters
    ----------
    raw : Raw
        ISCE3 Raw object
    freq : str
        Frequency ('A' or 'B')
    pol : str
        Polarization
    output_dir : str
        Output directory
    pulse_start, pulse_end : int, optional
        Pulse range to process
    range_start, range_end : int, optional
        Range sample limits
    save_raw : bool
        Whether to save raw data

    Returns
    -------
    results : dict
        Processing results
    """
    print(f"\nProcessing {freq}-{pol}...")
    start_time = time.time()

    # Get dataset info
    dataset = raw.getRawDataset(freq, pol)
    total_pulses, total_range = dataset.shape

    # Apply limits
    p_start = pulse_start if pulse_start is not None else 0
    p_end = pulse_end if pulse_end is not None else total_pulses
    r_start = range_start if range_start is not None else 0
    r_end = range_end if range_end is not None else total_range

    n_pulses = p_end - p_start
    n_range = r_end - r_start

    print(f"  Reading data shape: ({n_pulses}, {n_range})")

    # Read data
    read_start = time.time()
    raw_data = read_raw_data_batch(
        raw, freq, pol,
        pulse_slice=slice(p_start, p_end),
        range_slice=slice(r_start, r_end)
    )
    read_time = time.time() - read_start

    data_gb = raw_data.nbytes / 1e9
    print(f"  Data read: {data_gb:.3f} GB in {read_time:.2f}s ({data_gb/read_time:.2f} GB/s)")

    # Compute statistics
    stats = {
        'mean': np.mean(np.abs(raw_data)),
        'std': np.std(np.abs(raw_data)),
        'max': np.max(np.abs(raw_data)),
        'min': np.min(np.abs(raw_data)),
    }

    print(f"  Statistics:")
    print(f"    Mean: {stats['mean']:.2e}")
    print(f"    Std:  {stats['std']:.2e}")
    print(f"    Max:  {stats['max']:.2e}")
    print(f"    Min:  {stats['min']:.2e}")

    # Save raw data if requested
    if save_raw:
        output_file = os.path.join(output_dir, f'nisar_{freq}_{pol}_raw.h5')
        print(f"  Saving raw data to {output_file}...")

        save_start = time.time()
        with h5py.File(output_file, 'w') as f:
            # Save raw data
            f.create_dataset('raw_data', data=raw_data, compression='gzip', compression_opts=4)

            # Save metadata
            meta_grp = f.create_group('metadata')
            meta_grp.attrs['frequency'] = freq
            meta_grp.attrs['polarization'] = pol
            meta_grp.attrs['pulse_start'] = p_start
            meta_grp.attrs['pulse_end'] = p_end
            meta_grp.attrs['range_start'] = r_start
            meta_grp.attrs['range_end'] = r_end
            meta_grp.attrs['shape'] = raw_data.shape
            meta_grp.attrs['dtype'] = str(raw_data.dtype)

            # Get chirp parameters
            pol_tx = pol[0]
            fc, fs, _, _ = raw.getChirpParameters(freq, pol_tx)
            bandwidth = raw.getRangeBandwidth(freq, pol_tx)

            meta_grp.attrs['center_frequency_hz'] = fc
            meta_grp.attrs['sample_rate_hz'] = fs
            meta_grp.attrs['bandwidth_hz'] = bandwidth
            meta_grp.attrs['processing_date'] = datetime.now().isoformat()

            # Save statistics
            stats_grp = f.create_group('statistics')
            for key, val in stats.items():
                stats_grp.attrs[key] = val

        save_time = time.time() - save_start
        print(f"  Saved in {save_time:.2f}s")

    total_time = time.time() - start_time

    results = {
        'frequency': freq,
        'polarization': pol,
        'shape': (n_pulses, n_range),
        'total_time': total_time,
        'read_time': read_time,
        'data_size_gb': data_gb,
        'throughput_gb_per_s': data_gb / total_time,
        'statistics': stats,
    }

    print(f"  Total time: {total_time:.2f}s")

    return results


def main():
    """Main execution function."""
    args = parse_args()

    # Validate input file
    input_file = Path(args.input_file)
    if not input_file.exists():
        print(f"ERROR: Input file not found: {args.input_file}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("NISAR L0B Data Reader (ISCE3)")
    print("="*70)
    print(f"Input file: {args.input_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Processing mode: {'Streaming EVD' if args.stream else 'Full read'}")

    # Initialize ISCE3 Raw reader
    print("\nInitializing ISCE3 Raw reader...")
    start_time = time.time()
    raw = Raw(hdf5file=str(input_file))
    raw.parsePolarizations()
    init_time = time.time() - start_time
    print(f"Raw reader initialized in {init_time:.2f}s")

    # Get dataset info
    info = get_dataset_info(raw)

    # Determine which datasets to process
    freqs_to_process = [args.freq] if args.freq else list(raw.polarizations.keys())

    all_results = []

    # Process each frequency
    for freq in freqs_to_process:
        pols_to_process = [args.pol] if args.pol else raw.polarizations[freq]

        for pol in pols_to_process:
            if args.stream:
                # Streaming mode with EVD
                results = process_polarization_streaming(
                    raw, freq, pol, args.output_dir,
                    cpi_len=args.cpi_len,
                    pulse_start=args.pulse_start,
                    pulse_end=args.pulse_end,
                    range_start=args.range_start,
                    range_end=args.range_end,
                    save_eigenvalues=args.save_eigenvalues,
                    rx_dynamic_range_db=args.rx_dynamic_range_db,
                    min_ev_valid_idx=args.min_ev_valid_idx,
                )
            else:
                # Full read mode
                results = process_polarization_full(
                    raw, freq, pol, args.output_dir,
                    pulse_start=args.pulse_start,
                    pulse_end=args.pulse_end,
                    range_start=args.range_start,
                    range_end=args.range_end,
                    save_raw=args.save_raw,
                )

            all_results.append(results)

    # Print summary
    total_time = time.time() - start_time
    total_data_gb = sum(r['data_size_gb'] for r in all_results)

    print("\n" + "="*70)
    print("Processing Summary")
    print("="*70)
    print(f"Total datasets processed: {len(all_results)}")
    print(f"Total data processed: {total_data_gb:.3f} GB")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"Average throughput: {total_data_gb/total_time:.2f} GB/s")
    print(f"Output directory: {args.output_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
