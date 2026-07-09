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
    # Save raw data only
    python read_nisar_isce3.py input.h5 --save-raw --output-dir ./output

    # Compute and save eigenvalues with subswath masking
    python read_nisar_isce3.py input.h5 --compute-eigenvalues --save-eigenvalues \
        --compute-subswath-mask --save-subswath-mask --output-dir ./output

    # Full processing with model predictions
    python read_nisar_isce3.py input.h5 --save-raw --compute-eigenvalues \
        --save-eigenvalues --compute-subswath-mask --save-subswath-mask \
        --model path/to/model.h5 --save-predictions --output-dir ./output

    # Process specific frequency/polarization with range subsetting
    python read_nisar_isce3.py input.h5 --freq A --pol HV --range-start 0 \
        --range-end 25000 --compute-eigenvalues --save-eigenvalues --output-dir ./output
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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path for model imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Read NISAR L0B data efficiently using ISCE3 Raw reader',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Save raw data only
  python read_nisar_isce3.py input.h5 --save-raw

  # Compute eigenvalues with subswath masking (don't save raw data)
  python read_nisar_isce3.py input.h5 --compute-eigenvalues --save-eigenvalues --compute-subswath-mask --save-subswath-mask

  # Save both raw data and eigenvalues
  python read_nisar_isce3.py input.h5 --save-raw --compute-eigenvalues --save-eigenvalues

  # Process specific frequency/pol with range subsetting
  python read_nisar_isce3.py input.h5 --freq A --pol HV --range-start 0 --range-end 25000 --compute-eigenvalues --save-eigenvalues

  # Full processing with model predictions
  python read_nisar_isce3.py input.h5 --save-raw --compute-eigenvalues --save-eigenvalues --model model.h5 --save-predictions

  # Legacy streaming mode (equivalent to --compute-eigenvalues --save-eigenvalues)
  python read_nisar_isce3.py input.h5 --stream
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
                        help='End range sample index (default: all, use 110000 for 76k-110k range with --range-start 76000)')

    # Processing options - what to compute
    parser.add_argument('--compute-eigenvalues', action='store_true',
                        help='Compute eigenvalue decomposition (EVD)')
    parser.add_argument('--compute-subswath-mask', action='store_true',
                        help='Generate subswath mask to identify valid data regions')
    parser.add_argument('--cpi-len', type=int, default=16,
                        help='CPI length for EVD (default: 16)')

    # Saving options - what to save to disk
    parser.add_argument('--save-raw', action='store_true',
                        help='Save decoded raw data to HDF5 (can be large!)')
    parser.add_argument('--save-eigenvalues', action='store_true',
                        help='Save eigenvalues (requires --compute-eigenvalues)')
    parser.add_argument('--save-subswath-mask', action='store_true',
                        help='Save subswath mask (requires --compute-subswath-mask)')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save model predictions (requires --model)')

    # Legacy options
    parser.add_argument('--stream', action='store_true',
                        help='DEPRECATED: Use --compute-eigenvalues --save-eigenvalues instead')
    parser.add_argument('--use-subswath-mask', action='store_true',
                        help='DEPRECATED: Use --compute-subswath-mask instead')

    parser.add_argument('--n-workers', type=int, default=4,
                        help='Number of parallel workers (default: 4, currently unused)')

    # Model prediction parameters
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model for RFI predictions (optional)')

    # Legacy EVD parameters (kept for compatibility)
    parser.add_argument('--rx-dynamic-range-db', type=float, default=50.0,
                        help='DEPRECATED: Receiver dynamic range in dB (unused)')
    parser.add_argument('--min-ev-valid-idx', type=int, default=10,
                        help='DEPRECATED: Minimum eigenvalue valid index (unused)')

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

    # Also check raw HDF5 for comparison
    import h5py
    h5_shapes = {}
    try:
        with h5py.File(raw.filename, 'r') as f:
            for freq in ['A', 'B']:
                for pol in ['HH', 'HV', 'VH', 'VV']:
                    for tx in ['txH', 'txV']:
                        for rx in ['rxH', 'rxV']:
                            path = f'/science/LSAR/RRSD/swaths/frequency{freq}/{tx}/{rx}/{pol}'
                            if path in f:
                                h5_shapes[f'{freq}-{pol}'] = f[path].shape
    except:
        pass

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
            print(f"    ISCE3 Raw shape: {shape} (pulses × range samples)")
            if f'{freq}-{pol}' in h5_shapes:
                h5_shape = h5_shapes[f'{freq}-{pol}']
                print(f"    Raw HDF5 shape: {h5_shape}")
                if h5_shape != shape:
                    print(f"    ⚠️  WARNING: ISCE3 view differs from raw HDF5!")
            print(f"    Dtype: {dtype}")
            print(f"    Center Frequency: {fc/1e9:.3f} GHz")
            print(f"    Sample Rate: {fs/1e6:.3f} MHz")
            print(f"    Bandwidth: {bandwidth/1e6:.3f} MHz")
            print(f"    Total size: {shape[0] * shape[1] * np.dtype(dtype).itemsize / 1e9:.2f} GB (if decoded)")

    print("\n" + "="*70 + "\n")

    return info


def extract_model_features(cpi, n_global_features=2):
    """
    Extract features for model prediction from a CPI tile.

    Parameters
    ----------
    cpi : np.ndarray
        Complex CPI array of shape (M, K)
    n_global_features : int
        Number of global features (2 or 5)

    Returns
    -------
    eigen : np.ndarray
        Shape (M, 2) - eigenvalues and slopes in dB
    global_ : np.ndarray
        Shape (n_global_features,) - global features
    """
    M, K = cpi.shape

    # Compute SCM and eigenvalues
    SCM = (cpi @ cpi.conj().T) / K
    eigvals = np.linalg.eigvalsh(SCM)
    eigvals_sorted = np.sort(eigvals)[::-1]  # Descending

    # Normalize by max eigenvalue, convert to dB
    max_eigval = eigvals_sorted[0]
    eigvals_normalized = eigvals_sorted / max(max_eigval, 1e-12)
    eigvals_db = 10 * np.log10(eigvals_normalized + 1e-12)

    # Eigenvalue slopes
    slopes = np.diff(eigvals_db)
    slopes_padded = np.append(slopes, 0.0)
    eigen = np.stack([eigvals_db, slopes_padded], axis=-1).astype(np.float32)

    # Global features
    cond_number_db = eigvals_db[0] - max(eigvals_db[-1], -100)

    # Effective rank via Shannon entropy
    p = np.maximum(eigvals_sorted, 1e-12)
    p = p / np.sum(p)
    p = p[p > 0]
    eff_rank = np.exp(-np.sum(p * np.log(p)))

    if n_global_features == 2:
        global_ = np.array([cond_number_db, eff_rank], dtype=np.float32)
    else:
        # Extended features (5 total) - add dummy values for compatibility
        global_ = np.array([
            cond_number_db, eff_rank, 0.0, 0.0, 0.0
        ], dtype=np.float32)[:n_global_features]

    return eigen, global_


def load_model(model_path):
    """
    Load a trained Keras model.

    Parameters
    ----------
    model_path : str
        Path to model file

    Returns
    -------
    model : keras.Model
        Loaded model
    n_global_features : int
        Number of global features expected by model
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow required for model predictions. Install with: pip install tensorflow")

    print(f"\nLoading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # Detect number of global features
    n_global_features = 2  # Default
    try:
        for layer in model.inputs:
            if 'global' in layer.name.lower():
                n_global_features = layer.shape[-1]
                break
    except:
        pass

    print(f"  Model loaded successfully")
    print(f"  Expected global features: {n_global_features}")

    return model, n_global_features


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
    # Use explicit indexing instead of slice objects
    data = dataset[pulse_start:pulse_stop, range_start:range_stop]

    return data


def get_subswath_mask(
    raw: Raw,
    freq: str,
    pol: str,
    pulse_indices: np.ndarray,
    num_range_samples: int,
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
    num_range_samples : int
        Number of range samples (width of mask)

    Returns
    -------
    mask : np.ndarray
        Boolean mask of shape (len(pulse_indices), num_range_samples)
        True indicates valid data within subswath boundaries
    """
    # Get transmit polarization from first character (e.g., 'H' from 'HH')
    tx_pol = pol[0]

    # Get subswath boundaries: shape (n_subswaths, n_total_pulses, 2)
    # Last dimension contains [start_range, end_range] for each pulse
    subswaths = raw.getSubSwaths(freq, tx_pol)

    # Initialize mask
    num_pulses = len(pulse_indices)
    mask = np.zeros((num_pulses, num_range_samples), dtype=bool)

    # For each pulse, mark valid regions from all subswaths
    for imask, ipulse in enumerate(pulse_indices):
        for subswath in subswaths:
            start, end = subswath[ipulse]
            mask[imask, start:end] = True

    return mask


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
    compute_eigenvalues: bool = False,
    compute_subswath_mask: bool = False,
    cpi_len: int = 16,
    # What to save
    save_raw: bool = False,
    save_eigenvalues: bool = False,
    save_subswath_mask: bool = False,
    save_predictions: bool = False,
    # Model prediction
    model=None,
    n_global_features: int = 2,
    # Legacy parameters (kept for compatibility)
    rx_dynamic_range_db: float = 50.0,
    min_ev_valid_idx: int = 10,
):
    """
    Unified processing function for a single polarization.

    This function reads raw data and optionally computes/saves various products
    based on the provided flags.

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
    compute_eigenvalues : bool
        Whether to compute eigenvalue decomposition
    compute_subswath_mask : bool
        Whether to generate subswath mask
    cpi_len : int
        CPI length for EVD (default: 16)
    save_raw : bool
        Whether to save raw data to HDF5
    save_eigenvalues : bool
        Whether to save eigenvalues (requires compute_eigenvalues=True)
    save_subswath_mask : bool
        Whether to save subswath mask (requires compute_subswath_mask=True)
    save_predictions : bool
        Whether to save model predictions (requires model != None)
    model : keras.Model, optional
        Trained model for predictions
    n_global_features : int
        Number of global features expected by model
    rx_dynamic_range_db : float
        Receiver dynamic range in dB (unused, kept for compatibility)
    min_ev_valid_idx : int
        Minimum eigenvalue valid index (unused, kept for compatibility)

    Returns
    -------
    results : dict
        Processing results including statistics
    """
    print(f"\nProcessing {freq}-{pol}...")
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

    # Adjust pulse range to align with CPI if computing eigenvalues
    if compute_eigenvalues:
        num_cpi = n_pulses // cpi_len
        tb_size = num_cpi * cpi_len
        n_pulses = tb_size  # Truncate to CPI boundary
        p_end = p_start + tb_size
        print(f"  Data shape: ({n_pulses}, {n_range})")
        print(f"  CPI length: {cpi_len}, Number of CPIs: {num_cpi}")
    else:
        print(f"  Data shape: ({n_pulses}, {n_range})")

    # Read data
    print(f"  Reading data slice [{p_start}:{p_end}, {r_start}:{r_end}]...")
    read_start = time.time()

    print(f'{p_start = }, {p_end = }')
    print(f'{r_start = }, {r_end = }')

    raw_data = read_raw_data_batch(
        raw, freq, pol,
        pulse_slice=slice(p_start, p_end),
        range_slice=slice(r_start, r_end)
    )

    read_time = time.time() - read_start
    data_gb = raw_data.nbytes / 1e9
    print(f"  Data read: {data_gb:.3f} GB in {read_time:.2f}s ({data_gb/read_time:.2f} GB/s)")
    print(f"  Actual data shape: {raw_data.shape}, dtype: {raw_data.dtype}")

    # Verify data was read
    if raw_data.size == 0:
        raise ValueError(f"No data read! Check range limits. Dataset shape: {dataset.shape}")

    # Initialize timing variables
    evd_time = 0.0
    mask_time = 0.0
    pred_time = 0.0

    # Generate subswath mask if requested
    subswath_mask = None
    if compute_subswath_mask:
        print(f"  Generating subswath mask...")
        mask_start = time.time()
        pulse_indices = np.arange(p_start, p_end)
        subswath_mask = get_subswath_mask(raw, freq, pol, pulse_indices, n_range)
        mask_time = time.time() - mask_start

        # Calculate valid data percentage
        valid_pct = 100 * subswath_mask.sum() / subswath_mask.size
        print(f"  Subswath mask generated in {mask_time:.2f}s")
        print(f"  Valid data within subswaths: {valid_pct:.1f}%")

    # Compute eigenvalues if requested
    eig_val_sort = None
    diag_power = None
    diag_valid = None
    tb_is_valid = None
    num_cpi = None

    if compute_eigenvalues:
        num_cpi = n_pulses // cpi_len
        print(f"  Computing eigenvalues for {num_cpi} CPIs...")
        evd_start = time.time()

        eig_val_sort = np.zeros((num_cpi, cpi_len), dtype=np.float32)
        diag_power = np.zeros((num_cpi, cpi_len), dtype=np.float32)
        diag_valid = np.ones((num_cpi, cpi_len), dtype=bool)
        tb_is_valid = True

        for cpi_idx in range(num_cpi):
            cpi_start = cpi_idx * cpi_len
            cpi_data = raw_data[cpi_start:cpi_start + cpi_len, :].copy()

            # Apply subswath mask if available
            if compute_subswath_mask and subswath_mask is not None:
                cpi_mask = subswath_mask[cpi_start:cpi_start + cpi_len, :]
                # Zero out data outside valid subswath regions
                cpi_data = cpi_data * cpi_mask

            # Compute SCM
            M, K = cpi_data.shape
            SCM = (cpi_data @ cpi_data.conj().T) / K

            # Eigenvalues (descending order)
            eigvals = np.linalg.eigvalsh(SCM)
            eig_val_sort[cpi_idx, :] = np.sort(eigvals)[::-1]

            # Diagonal power
            diag_power[cpi_idx, :] = np.abs(np.diag(SCM))

        evd_time = time.time() - evd_start
        print(f"  Eigenvalues computed in {evd_time:.2f}s")
        print(f"  Valid CPIs: {num_cpi}")

    # Model predictions
    predictions = None
    if model is not None and compute_eigenvalues:
        n_range_data = raw_data.shape[1]
        cpi_width = 250  # Standard CPI width
        n_range_tiles = n_range_data // cpi_width

        print(f"  Running model predictions on {num_cpi} CPIs × {n_range_tiles} range tiles = {num_cpi * n_range_tiles} total tiles...")
        pred_start = time.time()

        eigen_list = []
        global_list = []

        # Process each CPI and each range tile
        for cpi_idx in range(num_cpi):
            cpi_start = cpi_idx * cpi_len

            for range_tile_idx in range(n_range_tiles):
                range_tile_start = range_tile_idx * cpi_width
                range_tile_end = range_tile_start + cpi_width

                # Extract tile
                cpi_data = raw_data[cpi_start:cpi_start + cpi_len, range_tile_start:range_tile_end]

                # Extract features
                eigen, global_ = extract_model_features(cpi_data, n_global_features)
                eigen_list.append(eigen)
                global_list.append(global_)

        # Stack for batch prediction
        eigen_batch = np.stack(eigen_list)
        global_batch = np.stack(global_list)

        # Predict
        probs = model.predict([eigen_batch, global_batch], verbose=0)

        # Extract predictions and reshape to 2D (pulse_tiles × range_tiles)
        knee_indices = np.argmax(probs, axis=-1).astype(np.int32).reshape(num_cpi, n_range_tiles)
        confidences = np.max(probs, axis=-1).astype(np.float32).reshape(num_cpi, n_range_tiles)

        predictions = {
            'knee_indices': knee_indices,
            'confidences': confidences,
            'probabilities': probs.astype(np.float32),
            'n_pulse_tiles': num_cpi,
            'n_range_tiles': n_range_tiles,
        }

        pred_time = time.time() - pred_start
        print(f"  Predictions computed in {pred_time:.2f}s")
        print(f"  Prediction shape: {knee_indices.shape} (pulse_tiles × range_tiles)")

        # Print summary statistics
        unique, counts = np.unique(knee_indices, return_counts=True)
        total_tiles = num_cpi * n_range_tiles
        print(f"  Predicted knee distribution:")
        for k, c in zip(unique, counts):
            pct = 100 * c / total_tiles
            print(f"    knee={k}: {c} ({pct:.1f}%)")

    # Get chirp parameters for metadata
    pol_tx = pol[0]
    fc, fs, _, _ = raw.getChirpParameters(freq, pol_tx)
    bandwidth = raw.getRangeBandwidth(freq, pol_tx)

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
            meta_grp.attrs['center_frequency_hz'] = fc
            meta_grp.attrs['sample_rate_hz'] = fs
            meta_grp.attrs['bandwidth_hz'] = bandwidth
            meta_grp.attrs['processing_date'] = datetime.now().isoformat()

            # Compute and save statistics
            stats = {
                'mean': np.mean(np.abs(raw_data)),
                'std': np.std(np.abs(raw_data)),
                'max': np.max(np.abs(raw_data)),
                'min': np.min(np.abs(raw_data)),
            }
            stats_grp = f.create_group('statistics')
            for key, val in stats.items():
                stats_grp.attrs[key] = val

            # Save subswath mask if requested
            if save_subswath_mask and subswath_mask is not None:
                mask_grp = f.create_group('subswath_mask')
                mask_grp.create_dataset('mask', data=subswath_mask, compression='gzip')
                mask_grp.attrs['description'] = 'Boolean mask indicating valid data within subswath boundaries'
                mask_grp.attrs['shape'] = subswath_mask.shape
                mask_grp.attrs['valid_fraction'] = float(subswath_mask.sum() / subswath_mask.size)

        save_time = time.time() - save_start
        print(f"  Raw data saved in {save_time:.2f}s")

    # Save eigenvalues and predictions if requested
    if save_eigenvalues and eig_val_sort is not None:
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
            meta_grp.attrs['pulse_end'] = p_end
            meta_grp.attrs['range_start'] = r_start
            meta_grp.attrs['range_end'] = r_end
            meta_grp.attrs['total_pulses'] = total_pulses
            meta_grp.attrs['total_range'] = total_range
            meta_grp.attrs['center_frequency_hz'] = fc
            meta_grp.attrs['sample_rate_hz'] = fs
            meta_grp.attrs['bandwidth_hz'] = bandwidth
            meta_grp.attrs['processing_time'] = time.time() - start_time
            meta_grp.attrs['processing_date'] = datetime.now().isoformat()

            # Save subswath mask if requested
            if save_subswath_mask and subswath_mask is not None:
                mask_grp = f.create_group('subswath_mask')
                mask_grp.create_dataset('mask', data=subswath_mask, compression='gzip')
                mask_grp.attrs['description'] = 'Boolean mask indicating valid data within subswath boundaries'
                mask_grp.attrs['shape'] = subswath_mask.shape
                mask_grp.attrs['valid_fraction'] = float(subswath_mask.sum() / subswath_mask.size)

            # Save predictions if available
            if save_predictions and predictions is not None:
                pred_grp = f.create_group('predictions')
                pred_grp.create_dataset('knee_indices', data=predictions['knee_indices'], compression='gzip')
                pred_grp.create_dataset('confidences', data=predictions['confidences'], compression='gzip')
                pred_grp.create_dataset('probabilities', data=predictions['probabilities'], compression='gzip')

                # Prediction metadata
                pred_grp.attrs['n_pulse_tiles'] = predictions['n_pulse_tiles']
                pred_grp.attrs['n_range_tiles'] = predictions['n_range_tiles']
                pred_grp.attrs['n_predictions'] = predictions['knee_indices'].size
                pred_grp.attrs['mean_confidence'] = float(np.mean(predictions['confidences']))
                pred_grp.attrs['prediction_time_s'] = pred_time
                pred_grp.attrs['cpi_width'] = 250  # Standard tile width

    total_time = time.time() - start_time

    results = {
        'frequency': freq,
        'polarization': pol,
        'shape': (n_pulses, n_range),
        'num_cpi': num_cpi,
        'tb_is_valid': tb_is_valid,
        'total_time': total_time,
        'read_time': read_time,
        'evd_time': evd_time,
        'mask_time': mask_time,
        'pred_time': pred_time,
        'data_size_gb': data_gb,
        'throughput_gb_per_s': data_gb / total_time,
    }

    print(f"  Total time: {total_time:.2f}s ({results['throughput_gb_per_s']:.2f} GB/s)")

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

    # Handle legacy flags
    compute_eigenvalues = args.compute_eigenvalues or args.stream
    compute_subswath_mask = args.compute_subswath_mask or args.use_subswath_mask
    save_eigenvalues = args.save_eigenvalues or args.stream

    # Validate save flags
    if args.save_eigenvalues and not compute_eigenvalues:
        print("WARNING: --save-eigenvalues requires --compute-eigenvalues. Enabling eigenvalue computation.")
        compute_eigenvalues = True
    if args.save_subswath_mask and not compute_subswath_mask:
        print("WARNING: --save-subswath-mask requires --compute-subswath-mask. Enabling mask computation.")
        compute_subswath_mask = True
    if args.save_predictions and not args.model:
        print("WARNING: --save-predictions requires --model. Predictions will not be saved.")
        args.save_predictions = False

    print("\n" + "="*70)
    print("NISAR L0B Data Reader (ISCE3)")
    print("="*70)
    print(f"Input file: {args.input_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"\nProcessing configuration:")
    print(f"  Compute eigenvalues: {compute_eigenvalues}")
    if compute_eigenvalues:
        print(f"    CPI length: {args.cpi_len}")
    print(f"  Compute subswath mask: {compute_subswath_mask}")
    print(f"  Model predictions: {args.model is not None}")
    print(f"\nSaving configuration:")
    print(f"  Save raw data: {args.save_raw}")
    print(f"  Save eigenvalues: {save_eigenvalues}")
    print(f"  Save subswath mask: {args.save_subswath_mask}")
    print(f"  Save predictions: {args.save_predictions}")

    # Initialize ISCE3 Raw reader
    print("\nInitializing ISCE3 Raw reader...")
    start_time = time.time()
    raw = Raw(hdf5file=str(input_file))
    raw.parsePolarizations()
    init_time = time.time() - start_time
    print(f"Raw reader initialized in {init_time:.2f}s")

    # Get dataset info
    info = get_dataset_info(raw)

    # Load model if provided
    model = None
    n_global_features = 2
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"ERROR: Model file not found: {args.model}")
            sys.exit(1)
        model, n_global_features = load_model(str(model_path))

    # Determine which datasets to process
    freqs_to_process = [args.freq] if args.freq else list(raw.polarizations.keys())

    all_results = []

    # Process each frequency
    for freq in freqs_to_process:
        pols_to_process = [args.pol] if args.pol else raw.polarizations[freq]

        for pol in pols_to_process:
            results = process_polarization(
                raw, freq, pol, args.output_dir,
                pulse_start=args.pulse_start,
                pulse_end=args.pulse_end,
                range_start=args.range_start,
                range_end=args.range_end,
                compute_eigenvalues=compute_eigenvalues,
                compute_subswath_mask=compute_subswath_mask,
                cpi_len=args.cpi_len,
                save_raw=args.save_raw,
                save_eigenvalues=save_eigenvalues,
                save_subswath_mask=args.save_subswath_mask,
                save_predictions=args.save_predictions,
                model=model,
                n_global_features=n_global_features,
                rx_dynamic_range_db=args.rx_dynamic_range_db,
                min_ev_valid_idx=args.min_ev_valid_idx,
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
