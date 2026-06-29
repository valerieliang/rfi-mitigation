"""
process_nisar_streaming_batched.py

Optimized streaming processor with batched model inference for maximum performance.

Differences from process_nisar_streaming.py:
  - Batches multiple CPIs together for model inference (much faster)
  - Processes rows in parallel with configurable batch sizes
  - Decouples data extraction from model prediction for better GPU utilization

Usage:
    # Process with batched predictions (batch size 32):
    python process_nisar_streaming_batched.py nisar.h5 output.h5 \\
        --dataset /science/LSAR/RRSD/swaths/frequencyA/txH/rxV/HV \\
        --model models/multi_band/best_model.keras \\
        --batch-size 32

    # Large batch for maximum GPU utilization:
    python process_nisar_streaming_batched.py nisar.h5 output.h5 \\
        --dataset /science/LSAR/RRSD/swaths/frequencyA/txH/rxH/HH \\
        --model models/multi_band/best_model.keras \\
        --batch-size 512 --n-workers 8
"""

import os
import sys
import numpy as np
import h5py
from pathlib import Path
import argparse
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import deque

# Import from process_nisar_streaming
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from process_nisar_streaming import (
    CPI_HEIGHT, CPI_WIDTH,
    detect_polarization,
    H5DataAccessor,
    compute_scm_and_eigenvalues,
    extract_model_features
)


def process_row_cpis_batched(data, pulse_idx, range_indices, cpi_height, cpi_width):
    """
    Extract CPIs from a row and compute features (no model prediction yet).

    Args:
        data: H5DataAccessor
        pulse_idx (int): Starting pulse index
        range_indices (list): List of range indices
        cpi_height (int): CPI height
        cpi_width (int): CPI width

    Returns:
        list: List of (pulse_idx, range_idx, cpi, features, scm_results) tuples
    """
    results = []

    for range_idx in range_indices:
        # Extract CPI
        cpi = data[pulse_idx:pulse_idx+cpi_height, range_idx:range_idx+cpi_width]

        # Compute SCM and eigenvalues
        SCM, eigvals_sorted, eigvals_normalized, max_eigval_db = compute_scm_and_eigenvalues(cpi)
        diagonal = np.diag(SCM).real

        # Extract model features
        eigen, global_ = extract_model_features(cpi, eigvals_normalized)

        scm_results = {
            'eigenvalues': eigvals_sorted,
            'eigenvalues_normalized': eigvals_normalized,
            'scm_diagonal': diagonal,
            'max_eigval_db': max_eigval_db,
        }

        results.append((pulse_idx, range_idx, cpi, eigen, global_, scm_results))

    return results


def batch_predict(model, batch_data, batch_size=32):
    """
    Run batched model inference on multiple CPIs.

    Args:
        model: Trained Keras model
        batch_data (list): List of (eigen, global_) tuples
        batch_size (int): Prediction batch size

    Returns:
        list: List of prediction dictionaries
    """
    if not batch_data:
        return []

    # Stack features
    eigen_batch = np.stack([item[0] for item in batch_data])
    global_batch = np.stack([item[1] for item in batch_data])

    # Predict in batches
    predictions = []
    n_samples = len(batch_data)

    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)

        probs = model.predict(
            [eigen_batch[i:end_idx], global_batch[i:end_idx]],
            verbose=0
        )

        for prob in probs:
            pred_knee = np.argmax(prob)
            pred_confidence = np.max(prob)
            eps = 1e-12
            entropy = -np.sum(prob * np.log(prob + eps))

            predictions.append({
                'knee_index': int(pred_knee),
                'confidence': float(pred_confidence),
                'entropy': float(entropy),
                'probabilities': prob,
            })

    return predictions


def stream_process_nisar_batched(
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
    batch_size=32,
):
    """
    Stream-process NISAR data with batched model inference.

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
        log_only (bool): If True, only log to console
        log_interval (int): Log progress every N tiles
        n_workers (int): Number of parallel workers
        batch_size (int): Model prediction batch size
    """

    # Detect polarization
    pol = detect_polarization(dataset_path, polarization)

    print("="*70)
    print(f"Batched Streaming NISAR {pol} Data Processing")
    print("="*70)
    print(f"Input: {input_path}")
    print(f"Dataset: {dataset_path}")
    if model is not None:
        print(f"Model: Loaded for predictions (batch size: {batch_size})")
    print(f"Workers: {n_workers}")
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

    # Initialize output file
    output_file = None
    if not log_only:
        output_file = h5py.File(output_h5_path, 'w')

        # Store metadata
        output_file.attrs['source'] = str(input_path)
        output_file.attrs['dataset_path'] = dataset_path
        output_file.attrs['polarization'] = pol
        output_file.attrs['cpi_height'] = cpi_height
        output_file.attrs['cpi_width'] = cpi_width
        output_file.attrs['n_pulse_tiles'] = n_pulse_tiles
        output_file.attrs['n_range_tiles'] = n_range_tiles
        output_file.attrs['n_cpi_tiles'] = total_tiles
        output_file.attrs['pulse_start'] = pulse_start
        output_file.attrs['pulse_end'] = pulse_end
        output_file.attrs['processing_date'] = datetime.now().isoformat()
        output_file.attrs['has_predictions'] = model is not None

    print(f"\n{'='*70}")
    print("Processing CPI tiles...")
    print(f"{'='*70}\n")

    # Generate indices
    pulse_indices = list(range(pulse_start, pulse_start + n_pulse_tiles * cpi_height, cpi_height))
    range_indices = list(range(0, n_range_tiles * cpi_width, cpi_width))

    tile_count = 0
    rfi_count = 0
    write_lock = Lock()

    # Accumulator for batched predictions
    batch_queue = []

    try:
        # Process rows in parallel
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Submit all rows
            future_to_pulse = {
                executor.submit(
                    process_row_cpis_batched,
                    data,
                    pulse_idx,
                    range_indices,
                    cpi_height,
                    cpi_width
                ): pulse_idx
                for pulse_idx in pulse_indices
            }

            # Process completed rows
            for future in as_completed(future_to_pulse):
                row_results = future.result()

                # Add to batch queue
                batch_queue.extend(row_results)

                # Process batch if full or last row
                if len(batch_queue) >= batch_size or tile_count + len(row_results) >= total_tiles:
                    # Extract features for batch prediction
                    if model is not None:
                        feature_batch = [(item[3], item[4]) for item in batch_queue]
                        predictions = batch_predict(model, feature_batch, batch_size)
                    else:
                        predictions = [None] * len(batch_queue)

                    # Save results
                    for idx, (pulse_idx, range_idx, cpi, eigen, global_, scm_results) in enumerate(batch_queue):
                        prediction = predictions[idx]

                        # Log
                        if tile_count % log_interval == 0 or tile_count == total_tiles - 1:
                            log_msg = f"[{tile_count+1:6d}/{total_tiles:6d}] "
                            log_msg += f"CPI [{pulse_idx:6d}, {range_idx:6d}] "
                            log_msg += f"Max λ: {scm_results['max_eigval_db']:7.2f} dB"

                            if prediction:
                                knee = prediction['knee_index']
                                conf = prediction['confidence']
                                log_msg += f" | Pred: knee={knee:2d} conf={conf:.3f}"

                            print(log_msg)

                        # Track RFI
                        if prediction and prediction['knee_index'] > 0:
                            rfi_count += 1

                        # Save to HDF5
                        if not log_only and output_file is not None:
                            with write_lock:
                                tile_key = f"cpi_{pulse_idx}_{range_idx}"

                                # Save CPI data
                                dset = output_file.create_dataset(tile_key, data=cpi)
                                dset.attrs['max_eigval_db'] = scm_results['max_eigval_db']
                                dset.attrs['pulse_idx'] = pulse_idx
                                dset.attrs['range_idx'] = range_idx

                                # Save eigenvalues
                                output_file.create_dataset(
                                    f"{tile_key}_eigenvalues",
                                    data=scm_results['eigenvalues']
                                )
                                output_file.create_dataset(
                                    f"{tile_key}_eigenvalues_normalized",
                                    data=scm_results['eigenvalues_normalized']
                                )
                                output_file.create_dataset(
                                    f"{tile_key}_diagonal",
                                    data=scm_results['scm_diagonal']
                                )

                                # Save predictions
                                if prediction:
                                    dset.attrs['predicted_knee'] = prediction['knee_index']
                                    dset.attrs['prediction_confidence'] = prediction['confidence']
                                    dset.attrs['prediction_entropy'] = prediction['entropy']

                                    output_file.create_dataset(
                                        f"{tile_key}_probabilities",
                                        data=prediction['probabilities']
                                    )

                        tile_count += 1

                    # Clear batch queue
                    batch_queue = []

        # Final statistics
        print(f"\n{'='*70}")
        print("Processing Complete!")
        print(f"{'='*70}")
        print(f"Total tiles processed: {tile_count:,}")

        if model is not None:
            rfi_rate = rfi_count / tile_count if tile_count > 0 else 0
            print(f"RFI detections: {rfi_count:,} ({100*rfi_rate:.2f}%)")
            print(f"Clean CPIs: {tile_count - rfi_count:,} ({100*(1-rfi_rate):.2f}%)")

        if not log_only:
            print(f"\nOutput saved to: {output_h5_path}")
            file_size = Path(output_h5_path).stat().st_size / (1024**3)
            print(f"File size: {file_size:.2f} GB")

            if model is not None:
                output_file.attrs['rfi_detections'] = rfi_count
                output_file.attrs['clean_cpis'] = tile_count - rfi_count
                output_file.attrs['rfi_rate'] = rfi_rate

    finally:
        if output_file is not None:
            output_file.close()


def main():
    parser = argparse.ArgumentParser(
        description='Batched streaming NISAR processor (optimized for model inference)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('input', help='Input NISAR HDF5 file')
    parser.add_argument('output', help='Output HDF5 file')
    parser.add_argument('--dataset', required=True, help='HDF5 dataset path')
    parser.add_argument('--polarization', choices=['HH', 'HV', 'VH', 'VV'])
    parser.add_argument('--model', default=None, help='Path to trained model')
    parser.add_argument('--cpi-height', type=int, default=16)
    parser.add_argument('--cpi-width', type=int, default=250)
    parser.add_argument('--range-start', type=int, default=None)
    parser.add_argument('--range-end', type=int, default=None)
    parser.add_argument('--log-only', action='store_true')
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--n-workers', type=int, default=4,
                        help='Parallel workers for data extraction')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Model prediction batch size (default: 32)')

    args = parser.parse_args()

    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    # Load model
    model = None
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"ERROR: Model not found: {args.model}")
            sys.exit(1)

        print(f"Loading model: {args.model}")
        import tensorflow as tf
        model = tf.keras.models.load_model(args.model)
        print(f"  Model loaded successfully\n")

    # Process
    stream_process_nisar_batched(
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
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()
