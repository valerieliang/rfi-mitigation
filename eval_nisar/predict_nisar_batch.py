"""
predict_nisar_batch.py

Optimized batch processing for multiple NISAR files with efficient GPU utilization.

This script processes multiple NISAR HDF5 files sequentially with optimized memory
management and large batch sizes for maximum GPU throughput. Better than parallel
processing when GPU memory is the bottleneck.

Features:
    - Sequential processing with large batches for optimal GPU utilization
    - Automatic memory management and batch size optimization
    - Progress tracking with ETA
    - Configurable memory limits

Usage:
    # Process both HH and HV files with automatic batch sizing:
    python predict_nisar_batch.py \
        --nisar-h5 nisar_data/processed/nisar_hh_cpi_tiles_64k_86k.h5 \
        --nisar-h5 nisar_data/processed/nisar_hv_cpi_tiles_64k_86k.h5 \
        --model models/multi_band/best_model.keras

    # With custom batch size:
    python predict_nisar_batch.py \
        --nisar-h5 nisar_data/processed/nisar_hh_cpi_tiles_64k_86k.h5 \
        --nisar-h5 nisar_data/processed/nisar_hv_cpi_tiles_64k_86k.h5 \
        --model models/multi_band/best_model.keras \
        --batch-size 2048
"""

import os
import sys
import json
import time
import numpy as np
import h5py
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_nisar.predict_nisar import (
    load_nisar_dataset,
    analyze_predictions,
    save_predictions_h5
)


def predict_nisar_optimized(model, eigen, global_, batch_size=512, use_tqdm=True):
    """
    Optimized model inference with progress tracking.

    Args:
        model: Trained Keras model
        eigen (np.ndarray): shape (N, M, 2)
        global_ (np.ndarray): shape (N, 5)
        batch_size (int): Batch size for inference
        use_tqdm (bool): Show progress bar

    Returns:
        predictions (np.ndarray): shape (N,) - predicted knee indices
        probabilities (np.ndarray): shape (N, N_CLASSES) - class probabilities
    """

    n_samples = len(eigen)
    n_batches = (n_samples + batch_size - 1) // batch_size

    print(f"  Running inference on {n_samples:,} CPIs in {n_batches} batches...")

    probs_list = []

    if use_tqdm:
        pbar = tqdm(total=n_samples, desc="  Inference", unit="CPI")

    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)

        batch_probs = model.predict(
            [eigen[i:end_idx], global_[i:end_idx]],
            verbose=0
        )

        probs_list.append(batch_probs)

        if use_tqdm:
            pbar.update(end_idx - i)

    if use_tqdm:
        pbar.close()

    probs = np.vstack(probs_list)
    predictions = np.argmax(probs, axis=-1)

    return predictions, probs


def process_single_file(
    h5_path,
    model,
    output_dir,
    max_tiles,
    batch_size,
    save_to_h5,
    file_idx,
    total_files
):
    """
    Process a single NISAR file.

    Args:
        h5_path (str): Path to NISAR HDF5 file
        model: Loaded Keras model
        output_dir (str|None): Output directory
        max_tiles (int|None): Limit to first N tiles
        batch_size (int): Batch size for inference
        save_to_h5 (bool): Save predictions back to HDF5
        file_idx (int): File index (for progress)
        total_files (int): Total number of files

    Returns:
        dict: Processing results and statistics
    """

    print(f"\n{'='*80}")
    print(f"Processing file {file_idx+1}/{total_files}: {Path(h5_path).name}")
    print(f"{'='*80}")

    start_time = time.time()

    # Load NISAR data
    print(f"\nLoading NISAR data...")
    eigen, global_, tile_names, tile_indices, metadata = load_nisar_dataset(
        h5_path,
        max_tiles=max_tiles
    )

    load_time = time.time() - start_time
    print(f"  Loaded {len(eigen):,} CPIs in {load_time:.2f}s")

    # Auto-set output directory if not provided
    if output_dir is None:
        polarization = metadata['polarization']
        output_dir = f'results/nisar_eval_{polarization}'
        print(f"  Output directory: {output_dir}")

    # Run inference
    print(f"\nRunning inference (batch_size={batch_size})...")
    pred_start = time.time()

    predictions, probabilities = predict_nisar_optimized(
        model, eigen, global_,
        batch_size=batch_size,
        use_tqdm=True
    )

    pred_time = time.time() - pred_start
    throughput = len(eigen) / pred_time

    print(f"  Prediction time: {pred_time:.2f}s")
    print(f"  Throughput: {throughput:.1f} CPIs/sec")

    # Analyze results
    print(f"\nAnalyzing predictions...")
    stats, pred_map = analyze_predictions(predictions, tile_indices, metadata)

    # Save results
    print(f"\nSaving results to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    # Save statistics JSON
    stats_path = os.path.join(output_dir, 'nisar_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    # Save predictions as numpy arrays
    np.save(os.path.join(output_dir, 'predictions.npy'), predictions)
    np.save(os.path.join(output_dir, 'probabilities.npy'), probabilities)
    np.save(os.path.join(output_dir, 'prediction_map.npy'), pred_map)

    # Save metadata
    metadata_full = {
        **metadata,
        'tile_indices': tile_indices,
        'n_cpis': len(predictions),
        'processing_time_seconds': time.time() - start_time,
        'throughput_cpis_per_sec': throughput,
    }
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata_full, f, indent=2)

    # Optionally save back to HDF5
    if save_to_h5:
        print(f"  Saving predictions to HDF5...")
        save_predictions_h5(h5_path, tile_names, predictions, probabilities)

    total_time = time.time() - start_time

    # Print summary
    print(f"\n{'─'*80}")
    print(f"File {file_idx+1}/{total_files} Complete!")
    print(f"{'─'*80}")
    print(f"  Polarization: {metadata['polarization']}")
    print(f"  Total CPIs: {stats['total_cpis']:,}")
    print(f"  RFI detected: {stats['rfi_cpis']:,} ({stats['rfi_rate']*100:.2f}%)")
    print(f"  Clean CPIs: {stats['clean_cpis']:,}")
    print(f"  Processing time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"  Output: {output_dir}")

    return {
        'file': h5_path,
        'output_dir': output_dir,
        'polarization': metadata['polarization'],
        'stats': stats,
        'processing_time': total_time,
        'throughput': throughput,
    }


def main():
    """
    Main batch prediction pipeline.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Optimized batch processing for multiple NISAR files',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--nisar-h5', action='append', required=True,
                        help='Path to processed NISAR HDF5 file (can specify multiple times)')
    parser.add_argument('--model', default='models/multi_band/best_model.keras',
                        help='Path to trained model')
    parser.add_argument('--output-dir', action='append', default=None,
                        help='Output directories (one per file, or auto-detect from polarization)')
    parser.add_argument('--max-tiles', type=int, default=None,
                        help='Limit to first N CPIs per file (for testing)')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='Batch size for inference (default: 1024)')
    parser.add_argument('--save-to-h5', action='store_true',
                        help='Save predictions back to HDF5 files as attributes')

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.model).exists():
        print(f"ERROR: Model not found: {args.model}")
        print(f"Train a model first: python train.py")
        sys.exit(1)

    # Check all HDF5 files exist
    for h5_path in args.nisar_h5:
        if not Path(h5_path).exists():
            print(f"ERROR: NISAR HDF5 not found: {h5_path}")
            sys.exit(1)

    n_files = len(args.nisar_h5)

    # Handle output directories
    if args.output_dir is None:
        output_dirs = [None] * n_files  # Auto-detect
    elif len(args.output_dir) == n_files:
        output_dirs = args.output_dir
    else:
        print(f"ERROR: Number of --output-dir ({len(args.output_dir)}) must match --nisar-h5 ({n_files})")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"Batch NISAR Prediction Pipeline")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Files to process: {n_files}")
    print(f"Batch size: {args.batch_size}")
    if args.max_tiles:
        print(f"Max tiles per file: {args.max_tiles:,}")
    print()

    for i, h5_path in enumerate(args.nisar_h5):
        print(f"  {i+1}. {h5_path}")

    # Load model once
    print(f"\n{'='*80}")
    print(f"Loading model: {args.model}")
    print(f"{'='*80}")

    model_load_start = time.time()
    model = tf.keras.models.load_model(args.model)
    model_load_time = time.time() - model_load_start

    print(f"  Model loaded in {model_load_time:.2f}s")

    # Process files sequentially
    pipeline_start = time.time()
    results = []

    for i, (h5_path, output_dir) in enumerate(zip(args.nisar_h5, output_dirs)):
        result = process_single_file(
            h5_path=h5_path,
            model=model,
            output_dir=output_dir,
            max_tiles=args.max_tiles,
            batch_size=args.batch_size,
            save_to_h5=args.save_to_h5,
            file_idx=i,
            total_files=n_files
        )
        results.append(result)

    total_time = time.time() - pipeline_start

    # Final summary
    print(f"\n{'='*80}")
    print(f"Pipeline Complete!")
    print(f"{'='*80}")

    total_cpis = sum(r['stats']['total_cpis'] for r in results)
    total_rfi = sum(r['stats']['rfi_cpis'] for r in results)
    avg_throughput = np.mean([r['throughput'] for r in results])

    print(f"\nOverall Statistics:")
    print(f"  Total files processed: {n_files}")
    print(f"  Total CPIs: {total_cpis:,}")
    print(f"  Total RFI detected: {total_rfi:,} ({100*total_rfi/total_cpis:.2f}%)")
    print(f"  Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"  Average throughput: {avg_throughput:.1f} CPIs/sec")

    print(f"\nPer-file results:")
    for i, result in enumerate(results):
        pol = result['polarization']
        cpis = result['stats']['total_cpis']
        rfi = result['stats']['rfi_cpis']
        time_min = result['processing_time'] / 60
        print(f"  {i+1}. {pol}: {cpis:,} CPIs, {rfi:,} RFI ({100*rfi/cpis:.2f}%), {time_min:.2f} min")

    print(f"\nNext step: Generate plots with plot_nisar_predictions.py")


if __name__ == '__main__':
    main()
