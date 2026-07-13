"""
read_nisar_swaths_padded_isce3.py

NISAR L0B data reader using ISCE3's Raw reader, extended with an optional
padding step that fills invalid raw data samples (as identified by the
subswath mask) with valid data taken from directly beneath them in the
pulse (slow time) dimension.

This script's flags mirror read_nisar_swaths_isce3.py's behavior for
raw/mask handling: nothing is computed or saved unless you ask for it.
Two additional flags control padding:

    --pad-invalid   Fill invalid samples using the value directly beneath
                     them (requires the subswath mask, which is computed
                     automatically if not already requested)
    --save-padded    Save the padded data to nisar_padded_<freq>_<pol>.h5
                     (requires --pad-invalid)

For every invalid sample at pulse index i and range index j, the fill
value is the nearest valid sample at pulse index k >= i in the same range
column (i.e. the sample "directly beneath" it, since increasing pulse
index is further down the pulse dimension). If no valid sample exists
below a given position (for example near the very last pulses of a
column), the nearest valid sample above the position is used instead as a
fallback.

Usage:
    # Match the original script: just save raw data, no mask, no padding
    python read_nisar_swaths_padded_isce3.py input.h5 --save-raw --output-dir ./output

    # Compute and save the subswath mask alongside raw data
    python read_nisar_swaths_padded_isce3.py input.h5 --save-raw \\
        --compute-subswath-mask --save-subswath-mask --output-dir ./output

    # Pad invalid samples and save the padded result
    python read_nisar_swaths_padded_isce3.py input.h5 \\
        --pad-invalid --save-padded --output-dir ./output

    # Process a specific frequency/polarization with range subsetting
    python read_nisar_swaths_padded_isce3.py input.h5 --freq A --pol HV \\
        --range-start 0 --range-end 25000 --pad-invalid --save-padded \\
        --output-dir ./output
"""

import argparse
import h5py
import numpy as np
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from nisar.products.readers.Raw import Raw


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Read NISAR L0B data, optionally padding invalid samples using the subswath mask',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Save raw data only (no mask, no padding)
  python read_nisar_swaths_padded_isce3.py input.h5 --save-raw

  # Compute and save subswath mask alongside raw data
  python read_nisar_swaths_padded_isce3.py input.h5 --save-raw \\
      --compute-subswath-mask --save-subswath-mask

  # Pad invalid samples and save the padded result
  python read_nisar_swaths_padded_isce3.py input.h5 --pad-invalid --save-padded

  # Process a specific frequency/pol with range subsetting
  python read_nisar_swaths_padded_isce3.py input.h5 --freq A --pol HV \\
      --range-start 0 --range-end 25000 --pad-invalid --save-padded
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
                        help='Start range sample index (default: 0)')
    parser.add_argument('--range-end', type=int, default=None,
                        help='End range sample index (default: all)')

    # Processing options - what to compute
    parser.add_argument('--compute-subswath-mask', action='store_true',
                        help='Generate subswath mask to identify valid data regions')
    parser.add_argument('--pad-invalid', action='store_true',
                        help='Pad invalid samples using the value directly beneath them '
                             '(automatically enables --compute-subswath-mask)')

    # Saving options - what to save to disk
    parser.add_argument('--save-raw', action='store_true',
                        help='Save unpadded raw data to nisar_<freq>_<pol>_raw.h5')
    parser.add_argument('--save-subswath-mask', action='store_true',
                        help='Save subswath mask (requires --compute-subswath-mask)')
    parser.add_argument('--save-padded', action='store_true',
                        help='Save padded data to nisar_padded_<freq>_<pol>.h5 (requires --pad-invalid)')

    return parser.parse_args()


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

    # Convert slices to actual indices
    pulse_start = pulse_slice.start if pulse_slice and pulse_slice.start else 0
    pulse_stop = pulse_slice.stop if pulse_slice and pulse_slice.stop else dataset.shape[0]
    range_start = range_slice.start if range_slice and range_slice.start else 0
    range_stop = range_slice.stop if range_slice and range_slice.stop else dataset.shape[1]

    # Read data - ISCE3 handles BFPQLUT decoding automatically
    data = dataset[pulse_start:pulse_stop, range_start:range_stop]

    return data


def get_subswath_mask(
    raw: Raw,
    freq: str,
    pol: str,
    pulse_indices: np.ndarray,
    range_indices: np.ndarray,
) -> np.ndarray:
    """
    Generate a boolean mask indicating valid data regions based on subswath boundaries.

    NISAR data has gaps between subswaths. This function uses ISCE3's getSubSwaths()
    to identify valid data regions and creates a mask.

    Parameters
    ----------
    raw : Raw
        ISCE3 Raw object
    freq : str
        Frequency ('A' or 'B')
    pol : str
        Polarization ('HH', 'HV', 'VH', 'VV')
    pulse_indices : np.ndarray
        Array of pulse indices to generate mask for
    range_indices : np.ndarray
        Array of range indices to generate mask for

    Returns
    -------
    mask : np.ndarray
        Boolean mask of shape (len(pulse_indices), num_range_samples)
        True indicates valid data within subswath boundaries
    """
    # Get transmit polarization from first character (e.g., 'H' from 'HH')
    tx_pol = pol[0]

    # Get subswath boundaries: shape (n_subswaths, n_total_pulses, 2)
    subswaths = raw.getSubSwaths(freq, tx_pol)
    swaths = subswaths[:, pulse_indices, :]

    num_pulses = len(pulse_indices)
    num_range_samples = len(range_indices)

    mask = np.zeros((num_pulses, num_range_samples), dtype=bool)

    # Subswath boundaries are absolute range sample indices into the full
    # swath. The mask array is window-relative, so shift boundaries by the
    # window's first range index and clip to the window extent.
    r_offset = int(range_indices[0])

    if swaths is not None:
        for i in range(num_pulses):
            for start, end in swaths[:, i, :]:
                s = max(int(start) - r_offset, 0)
                e = min(int(end) - r_offset, num_range_samples)
                if e > s:
                    mask[i, s:e] = True

    return mask


def fill_invalid_with_valid_below(data: np.ndarray, mask: np.ndarray):
    """
    Fill invalid raw data samples using the valid sample directly beneath
    them (the next pulse at the same range sample).

    For a given invalid sample at pulse index i and range index j, the
    replacement value is taken from the nearest valid sample at pulse index
    k >= i (same range index j), i.e. the sample directly beneath it in the
    pulse (slow time) dimension. If no valid sample exists below a given
    position (for example near the last pulses in a column), the nearest
    valid sample above the position is used instead as a fallback.

    Parameters
    ----------
    data : (num_pulses, num_range) complex ndarray
        Raw data to pad.
    mask : (num_pulses, num_range) bool ndarray
        True indicates a valid sample. False indicates an invalid sample
        that should be replaced.

    Returns
    -------
    padded : (num_pulses, num_range) complex ndarray
        Data with invalid samples replaced.
    num_filled : int
        Number of samples that were replaced.
    num_unfilled : int
        Number of samples that could not be filled because no valid sample
        exists anywhere in that range column.
    """
    if data.shape != mask.shape:
        raise ValueError(f"data shape {data.shape} != mask shape {mask.shape}")

    num_pulses, num_range = data.shape
    valid = mask.astype(bool, copy=False)

    row_idx = np.broadcast_to(
        np.arange(num_pulses, dtype=np.int64)[:, None], (num_pulses, num_range)
    )
    col_idx = np.broadcast_to(
        np.arange(num_range, dtype=np.int64)[None, :], (num_pulses, num_range)
    )

    # Nearest valid row at or below each position (search downward, toward
    # increasing pulse index), per range column.
    sentinel_below = num_pulses  # larger than any valid row index
    rows_or_sentinel_below = np.where(valid, row_idx, sentinel_below)
    next_valid_row = np.minimum.accumulate(rows_or_sentinel_below[::-1, :], axis=0)[::-1, :]

    # Nearest valid row at or above each position (search upward, toward
    # decreasing pulse index), used as a fallback when nothing valid exists
    # below.
    sentinel_above = -1
    rows_or_sentinel_above = np.where(valid, row_idx, sentinel_above)
    prev_valid_row = np.maximum.accumulate(rows_or_sentinel_above, axis=0)

    needs_fill = ~valid
    has_below = next_valid_row < sentinel_below
    has_above = prev_valid_row > sentinel_above

    use_below = needs_fill & has_below
    use_above = needs_fill & (~has_below) & has_above
    unfilled = needs_fill & (~has_below) & (~has_above)

    padded = data.copy()

    if np.any(use_below):
        src_rows = next_valid_row[use_below]
        src_cols = col_idx[use_below]
        padded[use_below] = data[src_rows, src_cols]

    if np.any(use_above):
        src_rows = prev_valid_row[use_above]
        src_cols = col_idx[use_above]
        padded[use_above] = data[src_rows, src_cols]

    num_filled = int(np.sum(use_below) + np.sum(use_above))
    num_unfilled = int(np.sum(unfilled))

    return padded, num_filled, num_unfilled


def process_polarization(
    raw: Raw,
    freq: str,
    pol: str,
    output_dir: str,
    pulse_start: int = None,
    pulse_end: int = None,
    range_start: int = None,
    range_end: int = None,
    # What to compute
    compute_subswath_mask: bool = False,
    pad_invalid: bool = False,
    # What to save
    save_raw: bool = False,
    save_subswath_mask: bool = False,
    save_padded: bool = False,
):
    """
    Unified processing function for a single polarization.

    Parameters
    ----------
    raw : Raw
        ISCE3 Raw object
    freq : str
        Frequency ('A' or 'B')
    pol : str
        Polarization ('HH', 'HV', 'VH', 'VV')
    output_dir : str
        Output directory
    pulse_start, pulse_end : int, optional
        Pulse range to process
    range_start, range_end : int, optional
        Range sample limits
    compute_subswath_mask : bool
        Whether to generate the subswath mask
    pad_invalid : bool
        Whether to pad invalid samples using the subswath mask (requires
        compute_subswath_mask=True)
    save_raw : bool
        Whether to save unpadded raw data to nisar_<freq>_<pol>_raw.h5
    save_subswath_mask : bool
        Whether to save the subswath mask (requires compute_subswath_mask=True)
    save_padded : bool
        Whether to save padded data to nisar_padded_<freq>_<pol>.h5 (requires
        pad_invalid=True)

    Returns
    -------
    results : dict
        Processing results including statistics
    """
    print(f"\nProcessing {freq}-{pol}...")
    start_time = time.time()

    dataset = raw.getRawDataset(freq, pol)
    total_pulses, total_range = dataset.shape

    p_start = pulse_start if pulse_start is not None else 0
    p_end = pulse_end if pulse_end is not None else total_pulses
    r_start = range_start if range_start is not None else 0
    r_end = range_end if range_end is not None else total_range

    n_pulses = p_end - p_start
    n_range = r_end - r_start
    print(f"  Data shape: ({n_pulses}, {n_range})")

    print(f"  Reading data slice [{p_start}:{p_end}, {r_start}:{r_end}]...")
    read_start = time.time()
    raw_data = read_raw_data_batch(
        raw, freq, pol,
        pulse_slice=slice(p_start, p_end),
        range_slice=slice(r_start, r_end)
    )
    read_time = time.time() - read_start
    data_gb = raw_data.nbytes / 1e9
    print(f"  Data read: {data_gb:.3f} GB in {read_time:.2f}s ({data_gb / max(read_time, 1e-9):.2f} GB/s)")

    if raw_data.size == 0:
        raise ValueError(f"No data read! Check range limits. Dataset shape: {dataset.shape}")

    # Generate subswath mask if requested
    subswath_mask = None
    if compute_subswath_mask:
        print("  Generating subswath mask...")
        mask_start = time.time()
        pulse_indices = np.arange(p_start, p_end)
        range_indices = np.arange(r_start, r_end)
        subswath_mask = get_subswath_mask(raw, freq, pol, pulse_indices, range_indices)
        mask_time = time.time() - mask_start

        valid_pct = 100 * subswath_mask.sum() / subswath_mask.size
        print(f"  Subswath mask generated in {mask_time:.2f}s")
        print(f"  Valid data within subswaths: {valid_pct:.1f}%")

    # Pad invalid samples if requested
    padded_data = None
    num_filled = None
    num_unfilled = None
    if pad_invalid:
        print("  Padding invalid samples using data directly beneath...")
        pad_start = time.time()
        padded_data, num_filled, num_unfilled = fill_invalid_with_valid_below(raw_data, subswath_mask)
        pad_time = time.time() - pad_start
        print(f"  Padding complete in {pad_time:.2f}s")
        print(f"  Samples filled: {num_filled}")
        if num_unfilled > 0:
            print(f"  WARNING: {num_unfilled} samples could not be filled (no valid sample in column)")

    # Get chirp parameters for metadata
    pol_tx = pol[0]
    fc, fs, _, _ = raw.getChirpParameters(freq, pol_tx)
    bandwidth = raw.getRangeBandwidth(freq, pol_tx)

    # Save raw (unpadded) data if requested
    if save_raw:
        output_file = os.path.join(output_dir, f'nisar_{freq}_{pol}_raw.h5')
        print(f"  Saving raw data to {output_file}...")

        save_start = time.time()
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('raw_data', data=raw_data, compression='gzip', compression_opts=4)

            meta_grp = f.create_group('metadata')
            meta_grp.attrs['frequency'] = freq
            meta_grp.attrs['polarization'] = pol
            meta_grp.attrs['pulse_start'] = p_start
            meta_grp.attrs['pulse_end'] = p_end
            meta_grp.attrs['range_start'] = r_start
            meta_grp.attrs['range_end'] = r_end
            meta_grp.attrs['shape'] = raw_data.shape
            meta_grp.attrs['dtype'] = str(raw_data.dtype)
            meta_grp.attrs['center_frequency_hz'] = fc
            meta_grp.attrs['sample_rate_hz'] = fs
            meta_grp.attrs['bandwidth_hz'] = bandwidth
            meta_grp.attrs['processing_date'] = datetime.now().isoformat()

            stats = {
                'mean': np.mean(np.abs(raw_data)),
                'std': np.std(np.abs(raw_data)),
                'max': np.max(np.abs(raw_data)),
                'min': np.min(np.abs(raw_data)),
            }
            stats_grp = f.create_group('statistics')
            for key, val in stats.items():
                stats_grp.attrs[key] = val

            if save_subswath_mask and subswath_mask is not None:
                mask_grp = f.create_group('subswath_mask')
                mask_grp.create_dataset('mask', data=subswath_mask, compression='gzip')
                mask_grp.attrs['description'] = 'Boolean mask indicating valid data within subswath boundaries'
                mask_grp.attrs['shape'] = subswath_mask.shape
                mask_grp.attrs['valid_fraction'] = float(subswath_mask.sum() / subswath_mask.size)

        save_time = time.time() - save_start
        print(f"  Raw data saved in {save_time:.2f}s")

    # Save padded data if requested
    if save_padded and padded_data is not None:
        output_file = os.path.join(output_dir, f'nisar_padded_{freq}_{pol}.h5')
        print(f"  Saving padded data to {output_file}...")

        save_start = time.time()
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('raw_data', data=padded_data, compression='gzip', compression_opts=4)

            # Save the original (pre-padding) subswath mask for reference
            mask_grp = f.create_group('subswath_mask')
            mask_grp.create_dataset('mask', data=subswath_mask, compression='gzip')
            mask_grp.attrs['description'] = (
                'Original boolean mask indicating valid data within subswath '
                'boundaries, prior to padding'
            )
            mask_grp.attrs['shape'] = subswath_mask.shape
            mask_grp.attrs['valid_fraction'] = float(subswath_mask.sum() / subswath_mask.size)

            meta_grp = f.create_group('metadata')
            meta_grp.attrs['frequency'] = freq
            meta_grp.attrs['polarization'] = pol
            meta_grp.attrs['pulse_start'] = p_start
            meta_grp.attrs['pulse_end'] = p_end
            meta_grp.attrs['range_start'] = r_start
            meta_grp.attrs['range_end'] = r_end
            meta_grp.attrs['shape'] = padded_data.shape
            meta_grp.attrs['dtype'] = str(padded_data.dtype)
            meta_grp.attrs['center_frequency_hz'] = fc
            meta_grp.attrs['sample_rate_hz'] = fs
            meta_grp.attrs['bandwidth_hz'] = bandwidth
            meta_grp.attrs['fill_method'] = 'nearest_valid_pulse_below_with_above_fallback'
            meta_grp.attrs['num_samples_filled'] = num_filled
            meta_grp.attrs['num_samples_unfilled'] = num_unfilled
            meta_grp.attrs['processing_date'] = datetime.now().isoformat()

            stats = {
                'mean': np.mean(np.abs(padded_data)),
                'std': np.std(np.abs(padded_data)),
                'max': np.max(np.abs(padded_data)),
                'min': np.min(np.abs(padded_data)),
            }
            stats_grp = f.create_group('statistics')
            for key, val in stats.items():
                stats_grp.attrs[key] = val

        save_time = time.time() - save_start
        print(f"  Padded data saved in {save_time:.2f}s")

    total_time = time.time() - start_time
    print(f"  Total time: {total_time:.2f}s")

    return {
        'frequency': freq,
        'polarization': pol,
        'shape': raw_data.shape,
        'num_filled': num_filled if num_filled is not None else 0,
        'num_unfilled': num_unfilled if num_unfilled is not None else 0,
        'total_time': total_time,
        'data_size_gb': data_gb,
    }


def main():
    """Main execution function."""
    args = parse_args()

    input_file = Path(args.input_file)
    if not input_file.exists():
        print(f"ERROR: Input file not found: {args.input_file}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Handle dependent flags, same pattern as read_nisar_swaths_isce3.py
    compute_subswath_mask = args.compute_subswath_mask
    pad_invalid = args.pad_invalid

    if args.pad_invalid and not compute_subswath_mask:
        print("NOTE: --pad-invalid requires the subswath mask. Enabling --compute-subswath-mask.")
        compute_subswath_mask = True

    if args.save_subswath_mask and not compute_subswath_mask:
        print("WARNING: --save-subswath-mask requires --compute-subswath-mask. Enabling mask computation.")
        compute_subswath_mask = True

    if args.save_padded and not pad_invalid:
        print("WARNING: --save-padded requires --pad-invalid. Enabling padding.")
        pad_invalid = True
        if not compute_subswath_mask:
            compute_subswath_mask = True

    print("\n" + "=" * 70)
    print("NISAR L0B Data Reader (ISCE3) with Optional Padding")
    print("=" * 70)
    print(f"Input file: {args.input_file}")
    print(f"Output directory: {args.output_dir}")
    print("\nProcessing configuration:")
    print(f"  Compute subswath mask: {compute_subswath_mask}")
    print(f"  Pad invalid samples: {pad_invalid}")
    print("\nSaving configuration:")
    print(f"  Save raw data: {args.save_raw}")
    print(f"  Save subswath mask: {args.save_subswath_mask}")
    print(f"  Save padded data: {args.save_padded}")

    print("\nInitializing ISCE3 Raw reader...")
    start_time = time.time()
    raw = Raw(hdf5file=str(input_file))
    raw.parsePolarizations()
    init_time = time.time() - start_time
    print(f"Raw reader initialized in {init_time:.2f}s")

    freqs_to_process = [args.freq] if args.freq else list(raw.polarizations.keys())

    all_results = []

    for freq in freqs_to_process:
        pols_to_process = [args.pol] if args.pol else raw.polarizations[freq]

        for pol in pols_to_process:
            results = process_polarization(
                raw, freq, pol, args.output_dir,
                pulse_start=args.pulse_start,
                pulse_end=args.pulse_end,
                range_start=args.range_start,
                range_end=args.range_end,
                compute_subswath_mask=compute_subswath_mask,
                pad_invalid=pad_invalid,
                save_raw=args.save_raw,
                save_subswath_mask=args.save_subswath_mask,
                save_padded=args.save_padded,
            )
            all_results.append(results)

    total_time = time.time() - start_time
    total_data_gb = sum(r['data_size_gb'] for r in all_results)
    total_filled = sum(r['num_filled'] for r in all_results)
    total_unfilled = sum(r['num_unfilled'] for r in all_results)

    print("\n" + "=" * 70)
    print("Processing Summary")
    print("=" * 70)
    print(f"Total datasets processed: {len(all_results)}")
    print(f"Total data processed: {total_data_gb:.3f} GB")
    if pad_invalid:
        print(f"Total samples filled: {total_filled}")
        if total_unfilled > 0:
            print(f"Total samples unfilled: {total_unfilled}")
    print(f"Total time: {total_time:.2f}s ({total_time / 60:.2f} min)")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()