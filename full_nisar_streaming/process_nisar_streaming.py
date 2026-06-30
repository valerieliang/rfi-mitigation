"""
process_nisar_streaming.py

Stream-process NISAR data from HDF5 file into CPI blocks and compute:
  - Eigenvalues (sorted descending, normalized)
  - SCM matrices
  - Model predictions (if model provided)
  - Log output for each CPI

This script iterates through a NISAR dataset without loading the entire file
into memory, processing CPIs one at a time or in batches.

Usage:
    # Process and save all metrics to HDF5:
    python process_nisar_streaming.py nisar.h5 output.h5 \\
        --dataset /science/LSAR/RRSD/swaths/frequencyA/txH/rxV/HV \\
        --model models/multi_band/best_model.keras

    # Process specific range with no model:
    python process_nisar_streaming.py nisar.h5 output.h5 \\
        --dataset /science/LSAR/RRSD/swaths/frequencyA/txH/rxH/HH \\
        --range-start 0 --range-end 1600

    # Process and log to console without saving:
    python process_nisar_streaming.py nisar.h5 output.h5 \\
        --dataset /science/LSAR/RRSD/swaths/frequencyA/txH/rxV/HV \\
        --log-only
"""

import os
import sys
import numpy as np
import h5py
from pathlib import Path
import argparse
import re
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# CPI tile dimensions
CPI_HEIGHT = 16   # pulses per CPI block
CPI_WIDTH = 250   # range samples per CPI tile


def detect_polarization(dataset_path=None, polarization_arg=None):
    """Detect polarization from dataset path or explicit argument."""
    if polarization_arg:
        pol = polarization_arg.upper()
        if pol not in ['HH', 'HV', 'VH', 'VV']:
            raise ValueError(f"Invalid polarization: {pol}")
        return pol

    if dataset_path:
        match = re.search(r'/(HH|HV|VH|VV)$', dataset_path)
        if match:
            return match.group(1)

    raise ValueError("Could not detect polarization")


class H5DataAccessor:
    """
    Provides array-like access to NISAR HDF5 data with on-the-fly BFPQLUT decoding.
    """
    def __init__(self, h5_path, dataset_path):
        self.h5_path = h5_path
        self.dataset_path = dataset_path

        with h5py.File(h5_path, 'r') as f:
            dataset = f[dataset_path]
            self.shape = dataset.shape
            self.dtype = np.complex64

            # Extract BFPQLUT
            parent_path = '/'.join(dataset_path.split('/')[:-1])
            bfpqlut_path = f"{parent_path}/BFPQLUT"

            if bfpqlut_path not in f:
                raise ValueError(f"BFPQLUT not found at {bfpqlut_path}")

            self.bfpqlut = f[bfpqlut_path][:]

        print(f"  Loaded BFPQLUT: {len(self.bfpqlut)} entries")

    def __getitem__(self, key):
        """Access data with on-the-fly decoding."""
        with h5py.File(self.h5_path, 'r') as f:
            dataset = f[self.dataset_path]
            quantized = dataset[key]
            real_decoded = self.bfpqlut[quantized['r']]
            imag_decoded = self.bfpqlut[quantized['i']]
            return (real_decoded + 1j * imag_decoded).astype(np.complex64)


def compute_scm_and_eigenvalues(cpi):
    """
    Compute SCM and eigenvalues from a complex CPI tile.

    Args:
        cpi (np.ndarray): Complex array of shape (M, K)

    Returns:
        SCM (np.ndarray): Sample covariance matrix (M, M)
        eigvals_sorted (np.ndarray): Eigenvalues sorted descending
        eigvals_normalized (np.ndarray): Eigenvalues normalized to [0,1]
        max_eigval_db (float): Maximum eigenvalue in dB
    """
    M, K = cpi.shape

    # Compute SCM: M @ M^H / K
    SCM = (cpi @ cpi.conj().T) / K

    # Compute eigenvalues
    eigvals = np.linalg.eigvalsh(SCM)  # Ascending
    eigvals_sorted = np.sort(eigvals)[::-1]  # Descending

    # Normalize eigenvalues
    max_eigval = eigvals_sorted[0]
    max_eigval_db = 10.0 * np.log10(max(max_eigval, 1e-12))
    eigvals_normalized = eigvals_sorted / max(max_eigval, 1e-12)

    return SCM, eigvals_sorted, eigvals_normalized, max_eigval_db


def extract_model_features(cpi, eigvals_normalized):
    """
    Extract features for model prediction (same as train.py).

    Args:
        cpi (np.ndarray): Complex CPI array (M, K)
        eigvals_normalized (np.ndarray): Normalized eigenvalues (M,)

    Returns:
        eigen (np.ndarray): Shape (M, 2) - [eigenvalues, slopes]
        global_ (np.ndarray): Shape (5,) - global features
    """
    M, K = cpi.shape
    SCM = (cpi @ cpi.conj().T) / K

    # Eigenvalue branch: normalized eigenvalues + slopes
    slopes = np.diff(eigvals_normalized)
    slopes_padded = np.append(slopes, 0.0)
    eigen = np.stack([eigvals_normalized, slopes_padded], axis=-1).astype(np.float32)

    # Global branch: SCM diagonal statistics
    diag = np.real(np.diag(SCM))
    eigvals_original = np.linalg.eigvalsh(SCM)
    max_eigval = np.max(eigvals_original)
    diag_normalized = diag / max(max_eigval, 1e-12)

    half = max(M // 2, 1)
    sigma_max = float(np.std(diag_normalized[:half]))
    sigma_min = float(np.std(diag_normalized[half:]))
    mu_min = float(np.mean(diag_normalized[half:]))

    cond_number = eigvals_normalized[0] / max(eigvals_normalized[-1], 1e-12)
    eps = 1e-6
    f_factor = sigma_max / (sigma_min + eps)

    global_ = np.array(
        [cond_number, sigma_min, sigma_max, mu_min, f_factor],
        dtype=np.float32
    )

    return eigen, global_


def process_cpi_tile(cpi, pulse_idx, range_idx, model=None):
    """
    Process a single CPI tile and compute all metrics.

    Args:
        cpi (np.ndarray): Complex CPI array (M, K)
        pulse_idx (int): Starting pulse index
        range_idx (int): Starting range index
        model: Optional trained Keras model

    Returns:
        dict: CPI metrics including eigenvalues, SCM, predictions, etc.
    """
    # Compute SCM and eigenvalues
    SCM, eigvals_sorted, eigvals_normalized, max_eigval_db = compute_scm_and_eigenvalues(cpi)

    # Extract SCM diagonal
    diagonal = np.diag(SCM).real

    # Prepare result dictionary
    result = {
        'pulse_idx': pulse_idx,
        'range_idx': range_idx,
        'cpi_shape': cpi.shape,
        'max_eigval_db': max_eigval_db,
        'eigenvalues': eigvals_sorted,
        'eigenvalues_normalized': eigvals_normalized,
        'scm_diagonal': diagonal,
        'scm_matrix': SCM,  # Full SCM if needed
    }

    # Model prediction if available
    if model is not None:
        eigen, global_ = extract_model_features(cpi, eigvals_normalized)

        # Reshape for batch prediction
        eigen_batch = eigen[np.newaxis, ...]
        global_batch = global_[np.newaxis, ...]

        # Predict
        probs = model.predict([eigen_batch, global_batch], verbose=0)
        pred_knee = np.argmax(probs[0])
        pred_confidence = np.max(probs[0])

        # Compute entropy as confidence measure
        eps = 1e-12
        entropy = -np.sum(probs[0] * np.log(probs[0] + eps))

        result['prediction'] = {
            'knee_index': int(pred_knee),
            'confidence': float(pred_confidence),
            'entropy': float(entropy),
            'probabilities': probs[0],
        }

    return result


def process_row_of_cpis(data, pulse_idx, range_indices, cpi_height, cpi_width, model):
    """
    Process a full row of CPI tiles in parallel.

    Args:
        data: H5DataAccessor for NISAR data
        pulse_idx (int): Starting pulse index for this row
        range_indices (list): List of starting range indices
        cpi_height (int): CPI height
        cpi_width (int): CPI width
        model: Optional trained model

    Returns:
        list: List of (range_idx, result) tuples
    """
    results = []

    for range_idx in range_indices:
        # Extract CPI tile
        cpi = data[pulse_idx:pulse_idx+cpi_height, range_idx:range_idx+cpi_width]

        # Process tile
        result = process_cpi_tile(cpi, pulse_idx, range_idx, model=model)
        results.append((range_idx, result))

    return pulse_idx, results


def stream_process_nisar(
    input_path,
    output_h5_path,
    dataset_path,
    polarization=None,
    model=None,
    cpi_height=CPI_HEIGHT,
    cpi_width=CPI_WIDTH,
    range_start=None,
    range_end=None,
    log_only=False,
    log_interval=100,
    n_workers=4,
):
    """
    Stream-process NISAR data into CPI tiles with all computations.
    Processes different pulse rows in parallel for improved performance.

    Args:
        input_path (str): Path to NISAR HDF5 file
        output_h5_path (str): Output HDF5 file path
        dataset_path (str): HDF5 dataset path
        polarization (str|None): Polarization
        model: Optional trained Keras model
        cpi_height (int): CPI height in pulses
        cpi_width (int): CPI width in samples
        range_start (int|None): Start pulse index
        range_end (int|None): End pulse index
        log_only (bool): If True, only log to console without saving
        log_interval (int): Log progress every N tiles
        n_workers (int): Number of parallel workers for processing rows
    """

    # Detect polarization
    pol = detect_polarization(dataset_path, polarization)

    print("="*70)
    print(f"Streaming NISAR {pol} Data Processing")
    print("="*70)
    print(f"Input: {input_path}")
    print(f"Dataset: {dataset_path}")
    if model is not None:
        print(f"Model: Loaded for predictions")
    print(f"Output: {output_h5_path if not log_only else 'Log only (no save)'}")

    # Load data accessor
    data = H5DataAccessor(input_path, dataset_path)
    total_pulses = data.shape[0]
    total_range = data.shape[1]

    print(f"\nData shape: {data.shape}")
    print(f"Dtype: {data.dtype}")

    # Handle range selection
    pulse_start = range_start if range_start is not None else 0
    pulse_end = range_end if range_end is not None else total_pulses

    # Calculate complete tiles (drop remainder)
    n_pulses = pulse_end - pulse_start
    n_pulse_tiles = n_pulses // cpi_height
    n_range_tiles = total_range // cpi_width

    # Truncate to complete tiles
    n_pulses_used = n_pulse_tiles * cpi_height
    n_range_used = n_range_tiles * cpi_width
    pulse_end = pulse_start + n_pulses_used

    # Log dropped data if any
    dropped_pulses = n_pulses - n_pulses_used
    dropped_range = total_range - n_range_used
    if dropped_pulses > 0 or dropped_range > 0:
        print(f"\nNote: Dropping incomplete tiles:")
        if dropped_pulses > 0:
            print(f"  Dropped {dropped_pulses} pulses ({dropped_pulses/n_pulses*100:.2f}%)")
        if dropped_range > 0:
            print(f"  Dropped {dropped_range} range samples ({dropped_range/total_range*100:.2f}%)")

    total_tiles = n_pulse_tiles * n_range_tiles

    print(f"\nProcessing configuration:")
    print(f"  Pulse range: {pulse_start} to {pulse_end} ({n_pulses_used} pulses)")
    print(f"  CPI size: {cpi_height} × {cpi_width}")
    print(f"  Pulse tiles: {n_pulse_tiles}")
    print(f"  Range tiles: {n_range_tiles}")
    print(f"  Total CPI tiles: {total_tiles:,}")

    # Create output directory
    if not log_only:
        output_dir = os.path.dirname(output_h5_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    # Initialize output file and arrays
    output_file = None
    predictions_array = None
    probabilities_array = None
    confidence_array = None

    if not log_only:
        output_file = h5py.File(output_h5_path, 'w')

        # Store minimal HDF5 attributes for self-documentation
        output_file.attrs['n_pulse_tiles'] = n_pulse_tiles
        output_file.attrs['n_range_tiles'] = n_range_tiles
        output_file.attrs['cpi_height'] = cpi_height
        output_file.attrs['cpi_width'] = cpi_width

        # Pre-allocate arrays for predictions (indexed by pulse_tile, range_tile)
        predictions_array = output_file.create_dataset(
            'predictions',
            shape=(n_pulse_tiles, n_range_tiles),
            dtype=np.int8,
            chunks=True,
            compression='gzip'
        )
        confidence_array = output_file.create_dataset(
            'confidence',
            shape=(n_pulse_tiles, n_range_tiles),
            dtype=np.float32,
            chunks=True,
            compression='gzip'
        )
        probabilities_array = output_file.create_dataset(
            'probabilities',
            shape=(n_pulse_tiles, n_range_tiles, 17),
            dtype=np.float32,
            chunks=True,
            compression='gzip'
        )

    print(f"\n{'='*70}")
    print("Processing CPI tiles...")
    print(f"{'='*70}\n")

    # Generate all pulse and range indices
    pulse_indices = list(range(pulse_start, pulse_start + n_pulse_tiles * cpi_height, cpi_height))
    range_indices = list(range(0, n_range_tiles * cpi_width, cpi_width))

    # Process tiles with parallel rows
    tile_count = 0
    rfi_count = 0  # Track RFI detections if model available

    try:
        # Use ThreadPoolExecutor to process rows in parallel
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Submit all rows for parallel processing
            future_to_pulse = {
                executor.submit(
                    process_row_of_cpis,
                    data,
                    pulse_idx,
                    range_indices,
                    cpi_height,
                    cpi_width,
                    model
                ): pulse_idx
                for pulse_idx in pulse_indices
            }

            # Process completed rows as they finish
            for future in as_completed(future_to_pulse):
                pulse_idx, row_results = future.result()

                # Process each tile in the row
                for range_idx, result in row_results:
                    # Log to console
                    if tile_count % log_interval == 0 or tile_count == total_tiles - 1:
                        log_msg = f"[{tile_count+1:6d}/{total_tiles:6d}] "
                        log_msg += f"CPI [{pulse_idx:6d}, {range_idx:6d}] "
                        log_msg += f"Max λ: {result['max_eigval_db']:7.2f} dB"

                        if 'prediction' in result:
                            pred = result['prediction']
                            knee = pred['knee_index']
                            conf = pred['confidence']
                            log_msg += f" | Pred: knee={knee:2d} conf={conf:.3f}"

                        print(log_msg)

                    # Track RFI count
                    if 'prediction' in result and result['prediction']['knee_index'] > 0:
                        rfi_count += 1

                    # Save predictions to arrays (no lock needed for different indices)
                    if not log_only and 'prediction' in result:
                        pred = result['prediction']
                        pulse_tile_idx = (pulse_idx - pulse_start) // cpi_height
                        range_tile_idx = range_idx // cpi_width

                        predictions_array[pulse_tile_idx, range_tile_idx] = pred['knee_index']
                        confidence_array[pulse_tile_idx, range_tile_idx] = pred['confidence']
                        probabilities_array[pulse_tile_idx, range_tile_idx, :] = pred['probabilities']

                    tile_count += 1

        # Final statistics
        print(f"\n{'='*70}")
        print("Processing Complete!")
        print(f"{'='*70}")
        print(f"Total tiles processed: {tile_count:,}")

        rfi_rate = rfi_count / tile_count if tile_count > 0 else 0
        print(f"RFI detections: {rfi_count:,} ({100*rfi_rate:.2f}%)")
        print(f"Clean CPIs: {tile_count - rfi_count:,} ({100*(1-rfi_rate):.2f}%)")

        if not log_only:
            print(f"\nOutput saved to: {output_h5_path}")
            file_size = Path(output_h5_path).stat().st_size / (1024**3)
            print(f"File size: {file_size:.2f} GB")

            # Save comprehensive metadata as JSON
            metadata_path = output_h5_path.replace('.h5', '.json')
            metadata = {
                'processing': {
                    'date': datetime.now().isoformat(),
                    'script': 'process_nisar_streaming.py',
                    'n_workers': int(n_workers),
                },
                'input': {
                    'source_file': str(input_path),
                    'dataset_path': dataset_path,
                    'polarization': pol,
                },
                'dimensions': {
                    'total_pulses': int(total_pulses),
                    'total_range': int(total_range),
                    'cpi_height': int(cpi_height),
                    'cpi_width': int(cpi_width),
                },
                'region': {
                    'pulse_start': int(pulse_start),
                    'pulse_end': int(pulse_end),
                    'n_pulses_used': int(n_pulses_used),
                    'n_pulse_tiles': int(n_pulse_tiles),
                    'n_range_tiles': int(n_range_tiles),
                    'n_range_used': int(n_range_used),
                    'total_tiles': int(total_tiles),
                },
                'coverage': {
                    'pulses_used': int(n_pulses_used),
                    'pulses_total': int(total_pulses),
                    'pulses_percent': float(n_pulses_used/total_pulses*100) if total_pulses > 0 else 0,
                    'range_used': int(n_range_used),
                    'range_total': int(total_range),
                    'range_percent': float(n_range_used/total_range*100) if total_range > 0 else 0,
                    'dropped_pulses': int(dropped_pulses),
                    'dropped_range': int(dropped_range),
                },
                'coordinate_mapping': {
                    'description': 'Array index [i, j] maps to pulse and range coordinates',
                    'pulse': {
                        'start': int(pulse_start),
                        'step': int(cpi_height),
                        'formula': f'{pulse_start} + i*{cpi_height}',
                    },
                    'range': {
                        'start': 0,
                        'step': int(cpi_width),
                        'formula': f'j*{cpi_width}',
                    }
                },
                'results': {
                    'total_cpis': int(tile_count),
                    'rfi_detections': int(rfi_count),
                    'clean_cpis': int(tile_count - rfi_count),
                    'rfi_rate': float(rfi_rate),
                    'rfi_percent': float(rfi_rate * 100),
                },
                'output': {
                    'hdf5_file': str(output_h5_path),
                    'file_size_gb': float(file_size),
                    'arrays': {
                        'predictions': {
                            'shape': [int(n_pulse_tiles), int(n_range_tiles)],
                            'dtype': 'int8',
                            'description': 'Predicted knee index (0-15) for each CPI'
                        },
                        'confidence': {
                            'shape': [int(n_pulse_tiles), int(n_range_tiles)],
                            'dtype': 'float32',
                            'description': 'Prediction confidence (max probability) for each CPI'
                        },
                        'probabilities': {
                            'shape': [int(n_pulse_tiles), int(n_range_tiles), 17],
                            'dtype': 'float32',
                            'description': 'Full probability distribution across 17 classes for each CPI (0=clean, 1-16=RFI knee positions)'
                        }
                    }
                }
            }

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"Metadata saved to: {metadata_path}")

    finally:
        if output_file is not None:
            output_file.close()


def main():
    parser = argparse.ArgumentParser(
        description='Stream-process NISAR data with all computations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with model predictions:
  python process_nisar_streaming.py nisar.h5 output.h5 \\
    --dataset /science/LSAR/RRSD/swaths/frequencyA/txH/rxV/HV \\
    --model models/multi_band/best_model.keras

  # Process specific range:
  python process_nisar_streaming.py nisar.h5 output.h5 \\
    --dataset /science/LSAR/RRSD/swaths/frequencyA/txH/rxH/HH \\
    --range-start 0 --range-end 1600

  # Log only without saving:
  python process_nisar_streaming.py nisar.h5 output.h5 \\
    --dataset /science/LSAR/RRSD/swaths/frequencyA/txH/rxV/HV \\
    --log-only
        """
    )

    parser.add_argument('input', help='Input NISAR HDF5 file')
    parser.add_argument('output', help='Output HDF5 file (ignored if --log-only)')
    parser.add_argument('--dataset', required=True, help='HDF5 dataset path')
    parser.add_argument('--polarization', choices=['HH', 'HV', 'VH', 'VV'],
                        help='Polarization (auto-detected if not provided)')
    parser.add_argument('--model', required=True,
                        help='Path to trained model for predictions')
    parser.add_argument('--cpi-height', type=int, default=16,
                        help='CPI height in pulses (default: 16)')
    parser.add_argument('--cpi-width', type=int, default=250,
                        help='CPI width in samples (default: 250)')
    parser.add_argument('--range-start', type=int, default=None,
                        help='Start at this pulse index')
    parser.add_argument('--range-end', type=int, default=None,
                        help='End at this pulse index')
    parser.add_argument('--log-only', action='store_true',
                        help='Only log to console, do not save output')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Log progress every N tiles (default: 100)')
    parser.add_argument('--n-workers', type=int, default=4,
                        help='Number of parallel workers for row processing (default: 4)')

    args = parser.parse_args()

    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    # Validate range arguments
    if args.range_start is not None or args.range_end is not None:
        range_start = args.range_start if args.range_start is not None else 0
        range_end = args.range_end
        if range_end is not None:
            range_length = range_end - range_start
            if range_length % args.cpi_height != 0:
                print(f"ERROR: Range length ({range_length}) must be divisible by --cpi-height ({args.cpi_height})")
                sys.exit(1)

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)

    print(f"Loading model: {args.model}")
    import tensorflow as tf
    model = tf.keras.models.load_model(args.model)
    print(f"  Model loaded successfully\n")

    # Process
    stream_process_nisar(
        input_path=args.input,
        output_h5_path=args.output,
        dataset_path=args.dataset,
        polarization=args.polarization,
        model=model,
        cpi_height=args.cpi_height,
        cpi_width=args.cpi_width,
        range_start=args.range_start,
        range_end=args.range_end,
        log_only=args.log_only,
        log_interval=args.log_interval,
        n_workers=args.n_workers,
    )


if __name__ == '__main__':
    main()
