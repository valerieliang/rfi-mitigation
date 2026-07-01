"""
plot_nisar_predictions.py

Generate visualization plots for eigenvalue knee predictions from streaming batch processing.

This script reads the HDF5 output from process_nisar_streaming_batched.py and generates:
1. Histogram of knee predictions before and after bounding (two-bar comparison)
2. Spatial comparison map (original vs bounded side-by-side)
3. Individual spatial maps (optional)

Convention:
- Clean: knee index = 0
- Contaminated: knee indices 1-16 (number of pulses contaminated by RFI)
- Bounding: Any knee > knee_bound is set to 0 (treated as clean/recoverable)

Usage:
    python plot_nisar_predictions.py hv_outputs.h5 --knee-bound 4
    python plot_nisar_predictions.py hv_outputs.h5 --knee-bound 4 --output-dir plots/
    python plot_nisar_predictions.py hv_outputs.h5 --knee-bound 6 --no-individual
"""

import os
import sys
import json
import numpy as np
import argparse
import h5py
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm


def load_predictions(h5_path):
    """
    Load predictions and metadata from HDF5 file.

    Args:
        h5_path (str): Path to HDF5 file from process_nisar_streaming_batched.py

    Returns:
        predictions (np.ndarray): 2D array of predicted knee indices (n_pulse_tiles, n_range_tiles)
        confidence (np.ndarray): 2D array of prediction confidence
        metadata (dict): Metadata from JSON file
    """
    print(f"Loading predictions from: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        predictions = f['predictions'][:]
        confidence = f['confidence'][:]

        print(f"  Predictions shape: {predictions.shape}")
        print(f"  Dtype: {predictions.dtype}")

    # Load metadata
    json_path = str(h5_path).replace('.h5', '.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            metadata = json.load(f)
    else:
        print(f"  Warning: Metadata file not found: {json_path}")
        metadata = {}

    return predictions, confidence, metadata


def compute_bounded_map(pred_map, knee_bound):
    """
    Create bounded map where knee values > knee_bound are set to 0.

    Args:
        pred_map (np.ndarray): Original predictions
        knee_bound (int): Maximum knee value to preserve

    Returns:
        bounded_map (np.ndarray): Bounded predictions
        stats (dict): Bounding statistics
    """
    bounded_map = pred_map.copy()
    bounded_map[bounded_map > knee_bound] = 0

    total_cpis = pred_map.size
    n_original_rfi = np.sum(pred_map > 0)
    n_recovered = np.sum((pred_map > knee_bound) & (pred_map > 0))
    n_remaining_rfi = np.sum(bounded_map > 0)

    stats = {
        'total_cpis': total_cpis,
        'original_rfi': n_original_rfi,
        'recovered': n_recovered,
        'remaining_rfi': n_remaining_rfi,
        'recovery_rate': n_recovered / total_cpis if total_cpis > 0 else 0,
        'original_rfi_rate': n_original_rfi / total_cpis if total_cpis > 0 else 0,
        'bounded_rfi_rate': n_remaining_rfi / total_cpis if total_cpis > 0 else 0,
    }

    return bounded_map, stats


def plot_histogram_comparison(predictions, knee_bound, output_path, metadata=None):
    """
    Plot histogram comparing original vs bounded knee predictions (two-bar comparison).

    Args:
        predictions (np.ndarray): Original prediction map
        knee_bound (int): Maximum knee value to preserve
        output_path (str): Output file path
        metadata (dict): Optional metadata
    """
    print(f"\nGenerating histogram comparison: {output_path}")

    # Flatten predictions
    pred_flat = predictions.flatten()
    bounded_flat = predictions.copy()
    bounded_flat[bounded_flat > knee_bound] = 0
    bounded_flat = bounded_flat.flatten()

    # Get unique knee values
    unique_orig = np.unique(pred_flat)
    unique_bounded = np.unique(bounded_flat)
    all_unique = sorted(set(unique_orig) | set(unique_bounded))

    # Count occurrences
    orig_counts = {k: np.sum(pred_flat == k) for k in all_unique}
    bounded_counts = {k: np.sum(bounded_flat == k) for k in all_unique}

    # Prepare data for plotting
    x = np.arange(len(all_unique))
    orig_values = [orig_counts[k] for k in all_unique]
    bounded_values = [bounded_counts[k] for k in all_unique]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    width = 0.35
    x_pos = np.arange(len(all_unique))

    # Plot bars
    bars1 = ax.bar(x_pos - width/2, orig_values, width,
                   label='Original', alpha=0.8, color='#D62728', edgecolor='black')
    bars2 = ax.bar(x_pos + width/2, bounded_values, width,
                   label=f'Bounded (≤{knee_bound})', alpha=0.8, color='#2CA02C', edgecolor='black')

    # Styling
    ax.set_xlabel('Knee Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of CPIs', fontsize=14, fontweight='bold')

    title = f'Eigenvalue Knee Distribution: Original vs Bounded (knee_bound={knee_bound})'
    if metadata and 'results' in metadata:
        total = metadata['results']['total_cpis']
        orig_rfi = metadata['results']['rfi_detections']
        title += f'\nTotal CPIs: {total:,} | Original RFI: {orig_rfi:,} ({100*orig_rfi/total:.2f}%)'
    ax.set_title(title, fontsize=15, fontweight='bold')

    # X-axis labels
    labels = []
    for k in all_unique:
        if k == 0:
            labels.append('Clean\n(0)')
        else:
            labels.append(f'{k}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11)

    ax.legend(fontsize=13, loc='upper right')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')

    # Add count labels on bars (only for significant differences)
    for i, (bar1, bar2, orig_val, bound_val) in enumerate(zip(bars1, bars2, orig_values, bounded_values)):
        if orig_val > 0:
            height1 = bar1.get_height()
            ax.text(bar1.get_x() + bar1.get_width()/2., height1,
                   f'{orig_val:,}', ha='center', va='bottom', fontsize=8)
        if bound_val > 0 and bound_val != orig_val:
            height2 = bar2.get_height()
            ax.text(bar2.get_x() + bar2.get_width()/2., height2,
                   f'{bound_val:,}', ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_comparison_maps(pred_map, knee_bound, output_path, metadata=None):
    """
    Plot side-by-side comparison of original vs bounded spatial maps.

    Args:
        pred_map (np.ndarray): Original prediction map
        knee_bound (int): Maximum knee value to preserve
        output_path (str): Output file path
        metadata (dict): Optional metadata
    """
    print(f"\nGenerating comparison spatial maps: {output_path}")

    bounded_map, stats = compute_bounded_map(pred_map, knee_bound)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))

    # Original map
    vmax_orig = int(np.max(pred_map))
    im1 = ax1.imshow(pred_map, aspect='auto', cmap='turbo',
                     vmin=0, vmax=vmax_orig, interpolation='nearest')
    cbar1 = fig.colorbar(im1, ax=ax1, label='Knee Index', shrink=0.8)
    cbar1.ax.tick_params(labelsize=11)

    ax1.set_xlabel('Range Tile Index', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Pulse Tile Index', fontsize=14, fontweight='bold')
    ax1.set_title(f'Original Predictions\n'
                  f'RFI CPIs: {stats["original_rfi"]:,} ({100*stats["original_rfi_rate"]:.2f}%)',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, which='both', color='white', linewidth=0.3, alpha=0.3)

    # Bounded map
    im2 = ax2.imshow(bounded_map, aspect='auto', cmap='turbo',
                     vmin=0, vmax=knee_bound, interpolation='nearest')
    cbar2 = fig.colorbar(im2, ax=ax2, label=f'Bounded Knee Index (0-{knee_bound})', shrink=0.8)
    cbar2.ax.tick_params(labelsize=11)

    ax2.set_xlabel('Range Tile Index', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Pulse Tile Index', fontsize=14, fontweight='bold')
    ax2.set_title(f'Bounded Map (knee_bound={knee_bound})\n'
                  f'Recovered: {stats["recovered"]:,} | Remaining RFI: {stats["remaining_rfi"]:,} ({100*stats["bounded_rfi_rate"]:.2f}%)',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, which='both', color='white', linewidth=0.3, alpha=0.3)

    # Main title
    suptitle = f'Eigenvalue Knee Predictions - Spatial Comparison\n'
    suptitle += f'Total CPIs: {stats["total_cpis"]:,} | Recovery Rate: {100*stats["recovery_rate"]:.2f}%'
    fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.98)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")

    # Print statistics
    print(f"\n  Bounding Statistics (knee_bound={knee_bound}):")
    print(f"    Total CPIs:           {stats['total_cpis']:,}")
    print(f"    Original RFI CPIs:    {stats['original_rfi']:,} ({100*stats['original_rfi_rate']:.2f}%)")
    print(f"    Recovered CPIs:       {stats['recovered']:,} ({100*stats['recovery_rate']:.2f}%)")
    print(f"    Remaining RFI CPIs:   {stats['remaining_rfi']:,} ({100*stats['bounded_rfi_rate']:.2f}%)")


def plot_individual_map(pred_map, output_path, title, knee_bound=None, metadata=None):
    """
    Plot individual spatial map.

    Args:
        pred_map (np.ndarray): Prediction map
        output_path (str): Output file path
        title (str): Plot title
        knee_bound (int|None): If provided, use as vmax
        metadata (dict): Optional metadata
    """
    print(f"\nGenerating individual map: {output_path}")

    vmax = knee_bound if knee_bound is not None else int(np.max(pred_map))

    fig, ax = plt.subplots(figsize=(16, 10))

    im = ax.imshow(pred_map, aspect='auto', cmap='turbo',
                   vmin=0, vmax=vmax, interpolation='nearest')

    cbar = fig.colorbar(im, ax=ax, label='Knee Index', shrink=0.8)
    cbar.ax.tick_params(labelsize=11)

    ax.set_xlabel('Range Tile Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('Pulse Tile Index', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.grid(True, which='both', color='white', linewidth=0.3, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot eigenvalue knee predictions with bounding comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('h5_file', help='Input HDF5 file from process_nisar_streaming_batched.py')
    parser.add_argument('--knee-bound', type=int, default=4,
                        help='Maximum knee value to preserve (default: 4). '
                             'Any knee > knee_bound is set to 0 (recovered as clean)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for plots (default: same as input file)')
    parser.add_argument('--no-individual', action='store_true',
                        help='Skip individual spatial map plots (only generate comparison)')
    parser.add_argument('--prefix', default='knee',
                        help='Filename prefix for output plots (default: knee)')

    args = parser.parse_args()

    # Validate input
    h5_path = Path(args.h5_file)
    if not h5_path.exists():
        print(f"ERROR: HDF5 file not found: {args.h5_file}")
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = h5_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Eigenvalue Knee Plotting")
    print("="*70)
    print(f"Input:       {h5_path}")
    print(f"Output dir:  {output_dir}")
    print(f"Knee bound:  {args.knee_bound}")
    print(f"Prefix:      {args.prefix}")

    # Load predictions
    predictions, confidence, metadata = load_predictions(h5_path)

    print(f"\n{'='*70}")
    print("Generating Plots")
    print(f"{'='*70}")

    # Generate histogram comparison (always)
    histogram_path = output_dir / f"{args.prefix}_histogram_comparison.png"
    plot_histogram_comparison(predictions, args.knee_bound, histogram_path, metadata)

    # Generate spatial comparison (always)
    comparison_path = output_dir / f"{args.prefix}_spatial_comparison.png"
    plot_comparison_maps(predictions, args.knee_bound, comparison_path, metadata)

    # Generate individual maps (optional)
    if not args.no_individual:
        # Original map
        original_path = output_dir / f"{args.prefix}_original.png"
        plot_individual_map(
            predictions,
            original_path,
            f'Original Eigenvalue Knee Predictions\n'
            f'Total CPIs: {metadata.get("results", {}).get("total_cpis", predictions.size):,}',
            knee_bound=None,
            metadata=metadata
        )

        # Bounded map
        bounded_map, stats = compute_bounded_map(predictions, args.knee_bound)
        bounded_path = output_dir / f"{args.prefix}_bounded.png"
        plot_individual_map(
            bounded_map,
            bounded_path,
            f'Bounded Eigenvalue Knee Predictions (knee_bound={args.knee_bound})\n'
            f'Remaining RFI: {stats["remaining_rfi"]:,} ({100*stats["bounded_rfi_rate"]:.2f}%)',
            knee_bound=args.knee_bound,
            metadata=metadata
        )

    print(f"\n{'='*70}")
    print("Plotting Complete!")
    print(f"{'='*70}")
    print(f"Plots saved to: {output_dir}")
    print(f"  - {args.prefix}_histogram_comparison.png")
    print(f"  - {args.prefix}_spatial_comparison.png")
    if not args.no_individual:
        print(f"  - {args.prefix}_original.png")
        print(f"  - {args.prefix}_bounded.png")


if __name__ == '__main__':
    main()
