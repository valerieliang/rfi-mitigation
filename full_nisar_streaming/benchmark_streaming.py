"""
benchmark_streaming.py

Benchmark different streaming processing approaches on a sample of NISAR data.

Compares:
  1. process_nisar_streaming.py (parallel rows, per-CPI prediction)
  2. process_nisar_streaming_batched.py (parallel rows, batched prediction)

Usage:
    python benchmark_streaming.py nisar.h5 \\
        --dataset /science/LSAR/RRSD/swaths/frequencyA/txH/rxV/HV \\
        --model models/multi_band/best_model.keras \\
        --n-samples 1000
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from process_nisar_streaming import (
    H5DataAccessor,
    process_cpi_tile,
    detect_polarization,
    CPI_HEIGHT,
    CPI_WIDTH
)
from process_nisar_streaming_batched import (
    process_row_cpis_batched,
    batch_predict
)


def benchmark_per_cpi(data, pulse_indices, range_indices, cpi_height, cpi_width, model):
    """Benchmark per-CPI processing (no parallelism)."""
    print("\n" + "="*70)
    print("BENCHMARK 1: Per-CPI Processing (Sequential)")
    print("="*70)

    start_time = time.time()
    results = []

    for pulse_idx in pulse_indices:
        for range_idx in range_indices:
            cpi = data[pulse_idx:pulse_idx+cpi_height, range_idx:range_idx+cpi_width]
            result = process_cpi_tile(cpi, pulse_idx, range_idx, model=model)
            results.append(result)

    elapsed = time.time() - start_time
    n_cpis = len(results)

    print(f"\nResults:")
    print(f"  CPIs processed: {n_cpis}")
    print(f"  Total time: {elapsed:.2f} seconds")
    print(f"  Time per CPI: {elapsed/n_cpis*1000:.2f} ms")
    print(f"  Throughput: {n_cpis/elapsed:.2f} CPIs/sec")

    return elapsed, results


def benchmark_batched(data, pulse_indices, range_indices, cpi_height, cpi_width, model, batch_size=32):
    """Benchmark batched processing."""
    print("\n" + "="*70)
    print(f"BENCHMARK 2: Batched Processing (batch_size={batch_size})")
    print("="*70)

    start_time = time.time()

    # Extract all CPIs and features
    all_data = []
    for pulse_idx in pulse_indices:
        row_results = process_row_cpis_batched(
            data, pulse_idx, range_indices, cpi_height, cpi_width
        )
        all_data.extend(row_results)

    # Batch predict
    feature_batch = [(item[3], item[4]) for item in all_data]
    predictions = batch_predict(model, feature_batch, batch_size)

    elapsed = time.time() - start_time
    n_cpis = len(all_data)

    print(f"\nResults:")
    print(f"  CPIs processed: {n_cpis}")
    print(f"  Total time: {elapsed:.2f} seconds")
    print(f"  Time per CPI: {elapsed/n_cpis*1000:.2f} ms")
    print(f"  Throughput: {n_cpis/elapsed:.2f} CPIs/sec")

    return elapsed, predictions


def main():
    parser = argparse.ArgumentParser(description='Benchmark streaming processing approaches')
    parser.add_argument('input', help='Input NISAR HDF5 file')
    parser.add_argument('--dataset', required=True, help='HDF5 dataset path')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--n-samples', type=int, default=1000,
                        help='Number of CPIs to benchmark (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for batched approach (default: 32)')
    parser.add_argument('--cpi-height', type=int, default=16)
    parser.add_argument('--cpi-width', type=int, default=250)

    args = parser.parse_args()

    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)

    print("="*70)
    print("NISAR STREAMING PROCESSING BENCHMARK")
    print("="*70)
    print(f"Input: {args.input}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Sample size: {args.n_samples} CPIs")

    # Load model
    print("\nLoading model...")
    import tensorflow as tf
    model = tf.keras.models.load_model(args.model)
    print("  Model loaded")

    # Load data accessor
    print("\nLoading data accessor...")
    data = H5DataAccessor(args.input, args.dataset)
    pol = detect_polarization(args.dataset)
    print(f"  Polarization: {pol}")
    print(f"  Shape: {data.shape}")

    # Determine sample range
    n_range_tiles = data.shape[1] // args.cpi_width
    n_pulse_tiles = data.shape[0] // args.cpi_height

    # Calculate how many rows we need for n_samples CPIs
    cpis_per_row = n_range_tiles
    n_rows_needed = (args.n_samples + cpis_per_row - 1) // cpis_per_row
    n_rows_needed = min(n_rows_needed, n_pulse_tiles)

    pulse_indices = [i * args.cpi_height for i in range(n_rows_needed)]
    range_indices = [j * args.cpi_width for j in range(n_range_tiles)]

    actual_n_samples = n_rows_needed * n_range_tiles

    print(f"\nBenchmark configuration:")
    print(f"  Rows to process: {n_rows_needed}")
    print(f"  CPIs per row: {cpis_per_row}")
    print(f"  Actual sample size: {actual_n_samples} CPIs")

    # Run benchmarks
    time1, results1 = benchmark_per_cpi(
        data, pulse_indices, range_indices,
        args.cpi_height, args.cpi_width, model
    )

    time2, results2 = benchmark_batched(
        data, pulse_indices, range_indices,
        args.cpi_height, args.cpi_width, model,
        batch_size=args.batch_size
    )

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nPer-CPI approach:")
    print(f"  Time: {time1:.2f} seconds")
    print(f"  Throughput: {actual_n_samples/time1:.2f} CPIs/sec")

    print(f"\nBatched approach (batch_size={args.batch_size}):")
    print(f"  Time: {time2:.2f} seconds")
    print(f"  Throughput: {actual_n_samples/time2:.2f} CPIs/sec")

    speedup = time1 / time2
    print(f"\nSpeedup: {speedup:.2f}x faster with batched approach")

    if speedup > 1.5:
        print(f"\n✓ Recommendation: Use process_nisar_streaming_batched.py")
        print(f"  Expected time for full dataset: ~{time2/actual_n_samples * (n_pulse_tiles * n_range_tiles) / 60:.1f} minutes")
    else:
        print(f"\n✓ Recommendation: Either approach is fine")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
