"""
predict_nisar_parallel.py

Run trained RFI knee classifier on multiple NISAR CPI files in parallel.

This script enables efficient parallel processing of multiple polarization files
using multiprocessing. Each file is processed independently with its own process,
allowing for maximum CPU/GPU utilization.

Features:
    - Process multiple NISAR HDF5 files simultaneously
    - Automatic output directory detection based on polarization
    - Progress tracking across all files
    - Configurable batch sizes and max tiles

Usage:
    # Process both HH and HV files in parallel:
    python predict_nisar_parallel.py \
        --nisar-h5 nisar_data/processed/nisar_hh_cpi_tiles_64k_86k.h5 \
        --nisar-h5 nisar_data/processed/nisar_hv_cpi_tiles_64k_86k.h5 \
        --model models/multi_band/best_model.keras

    # With custom settings:
    python predict_nisar_parallel.py \
        --nisar-h5 nisar_data/processed/nisar_hh_cpi_tiles_64k_86k.h5 \
        --nisar-h5 nisar_data/processed/nisar_hv_cpi_tiles_64k_86k.h5 \
        --model models/multi_band/best_model.keras \
        --batch-size 1024 \
        --max-tiles 10000 \
        --n-jobs 2
"""

import os
import sys
import json
import time
import numpy as np
import h5py
import tensorflow as tf
from pathlib import Path
from multiprocessing import Process, Queue, Manager
from datetime import datetime

# Add parent directory to path to import predict_nisar
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_nisar.predict_nisar import (
    load_nisar_dataset,
    predict_nisar,
    analyze_predictions,
    save_predictions_h5
)


def worker_process_file(
    h5_path,
    model_path,
    output_dir,
    max_tiles,
    batch_size,
    save_to_h5,
    progress_queue,
    worker_id
):
    """
    Worker function to process a single NISAR file.

    Args:
        h5_path (str): Path to NISAR HDF5 file
        model_path (str): Path to trained model
        output_dir (str|None): Output directory (auto-detected if None)
        max_tiles (int|None): Limit to first N tiles
        batch_size (int): Batch size for inference
        save_to_h5 (bool): Save predictions back to HDF5
        progress_queue (Queue): Queue for progress updates
        worker_id (int): Worker identifier
    """

    try:
        progress_queue.put({
            'worker_id': worker_id,
            'file': h5_path,
            'status': 'started',
            'timestamp': datetime.now().isoformat()
        })

        # Load model
        progress_queue.put({
            'worker_id': worker_id,
            'file': h5_path,
            'status': 'loading_model',
            'message': f'Loading model: {model_path}'
        })

        model = tf.keras.models.load_model(model_path)

        # Load NISAR data
        progress_queue.put({
            'worker_id': worker_id,
            'file': h5_path,
            'status': 'loading_data',
            'message': 'Loading NISAR data...'
        })

        eigen, global_, tile_names, tile_indices, metadata = load_nisar_dataset(
            h5_path,
            max_tiles=max_tiles
        )

        # Auto-set output directory if not provided
        if output_dir is None:
            polarization = metadata['polarization']
            output_dir = f'results/nisar_eval_{polarization}'

        progress_queue.put({
            'worker_id': worker_id,
            'file': h5_path,
            'status': 'predicting',
            'message': f'Running inference on {len(eigen):,} CPIs...',
            'n_tiles': len(eigen),
            'polarization': metadata['polarization']
        })

        # Run inference
        predictions, probabilities = predict_nisar(
            model, eigen, global_,
            batch_size=batch_size
        )

        # Analyze results
        progress_queue.put({
            'worker_id': worker_id,
            'file': h5_path,
            'status': 'analyzing',
            'message': 'Analyzing predictions...'
        })

        stats, pred_map = analyze_predictions(predictions, tile_indices, metadata)

        # Save results
        progress_queue.put({
            'worker_id': worker_id,
            'file': h5_path,
            'status': 'saving',
            'message': f'Saving results to {output_dir}...'
        })

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
        }
        metadata_path = os.path.join(output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata_full, f, indent=2)

        # Optionally save back to HDF5
        if save_to_h5:
            save_predictions_h5(h5_path, tile_names, predictions, probabilities)

        # Report completion
        progress_queue.put({
            'worker_id': worker_id,
            'file': h5_path,
            'status': 'completed',
            'output_dir': output_dir,
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        progress_queue.put({
            'worker_id': worker_id,
            'file': h5_path,
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })
        raise


def progress_monitor(progress_queue, n_workers):
    """
    Monitor and display progress from all workers.

    Args:
        progress_queue (Queue): Queue receiving progress updates
        n_workers (int): Number of worker processes
    """

    workers_status = {}
    completed = 0

    print(f"\n{'='*80}")
    print(f"Processing {n_workers} files in parallel...")
    print(f"{'='*80}\n")

    while completed < n_workers:
        try:
            update = progress_queue.get(timeout=1)

            worker_id = update['worker_id']
            status = update['status']

            workers_status[worker_id] = update

            # Display update
            timestamp = datetime.now().strftime('%H:%M:%S')
            file_name = Path(update['file']).name

            if status == 'started':
                print(f"[{timestamp}] Worker {worker_id}: Started processing {file_name}")

            elif status == 'loading_model':
                print(f"[{timestamp}] Worker {worker_id}: Loading model...")

            elif status == 'loading_data':
                print(f"[{timestamp}] Worker {worker_id}: Loading NISAR data...")

            elif status == 'predicting':
                pol = update.get('polarization', 'UNKNOWN')
                n_tiles = update.get('n_tiles', 0)
                print(f"[{timestamp}] Worker {worker_id}: Predicting {pol} - {n_tiles:,} CPIs")

            elif status == 'analyzing':
                print(f"[{timestamp}] Worker {worker_id}: Analyzing predictions...")

            elif status == 'saving':
                print(f"[{timestamp}] Worker {worker_id}: Saving results...")

            elif status == 'completed':
                completed += 1
                stats = update['stats']
                rfi_rate = stats['rfi_rate'] * 100
                print(f"\n[{timestamp}] Worker {worker_id}: COMPLETED ✓")
                print(f"  File: {file_name}")
                print(f"  Output: {update['output_dir']}")
                print(f"  Total CPIs: {stats['total_cpis']:,}")
                print(f"  RFI detected: {stats['rfi_cpis']:,} ({rfi_rate:.2f}%)")
                print(f"  Clean CPIs: {stats['clean_cpis']:,}\n")

            elif status == 'error':
                completed += 1
                print(f"\n[{timestamp}] Worker {worker_id}: ERROR ✗")
                print(f"  File: {file_name}")
                print(f"  Error: {update['error']}\n")

        except:
            # Timeout - continue waiting
            continue

    print(f"{'='*80}")
    print(f"All {n_workers} files processed!")
    print(f"{'='*80}\n")


def main():
    """
    Main parallel prediction pipeline.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Run trained model on multiple NISAR files in parallel',
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
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size for inference')
    parser.add_argument('--save-to-h5', action='store_true',
                        help='Save predictions back to HDF5 files as attributes')
    parser.add_argument('--n-jobs', type=int, default=None,
                        help='Number of parallel jobs (default: number of files)')

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
    n_jobs = args.n_jobs if args.n_jobs else n_files

    # Handle output directories
    if args.output_dir is None:
        output_dirs = [None] * n_files  # Auto-detect
    elif len(args.output_dir) == n_files:
        output_dirs = args.output_dir
    else:
        print(f"ERROR: Number of --output-dir ({len(args.output_dir)}) must match --nisar-h5 ({n_files})")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"Parallel NISAR Prediction Pipeline")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Files to process: {n_files}")
    print(f"Parallel jobs: {n_jobs}")
    print(f"Batch size: {args.batch_size}")
    if args.max_tiles:
        print(f"Max tiles per file: {args.max_tiles:,}")
    print()

    for i, h5_path in enumerate(args.nisar_h5):
        print(f"  {i+1}. {h5_path}")

    # Create progress queue
    manager = Manager()
    progress_queue = manager.Queue()

    # Start progress monitor in separate process
    monitor = Process(target=progress_monitor, args=(progress_queue, n_files))
    monitor.start()

    # Start worker processes
    start_time = time.time()
    workers = []

    for i, (h5_path, output_dir) in enumerate(zip(args.nisar_h5, output_dirs)):
        worker = Process(
            target=worker_process_file,
            args=(
                h5_path,
                args.model,
                output_dir,
                args.max_tiles,
                args.batch_size,
                args.save_to_h5,
                progress_queue,
                i
            )
        )
        worker.start()
        workers.append(worker)

    # Wait for all workers to complete
    for worker in workers:
        worker.join()

    # Stop progress monitor
    monitor.join()

    elapsed = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"Pipeline Complete!")
    print(f"{'='*80}")
    print(f"Total time: {elapsed/60:.2f} minutes")
    print(f"Average time per file: {elapsed/n_files:.2f} seconds")
    print(f"\nNext step: Generate plots with plot_nisar_predictions.py")


if __name__ == '__main__':
    main()
