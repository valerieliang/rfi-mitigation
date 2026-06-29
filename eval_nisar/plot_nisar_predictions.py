"""
plot_nisar_predictions.py

Generate visualization plots for NISAR knee predictions.

This script reads the predictions saved by predict_nisar.py and generates:
1. Histogram of knee predictions (per CPI)
2. Spatial map of knee positions (sequential colormap)
3. Bounded spatial map (with max_knee threshold for data preservation)

Usage:
    python plot_nisar_predictions.py --results-dir results/nisar_eval_HV
    python plot_nisar_predictions.py --results-dir results/nisar_eval_HV --max-knee 4
"""

import os
import sys
import json
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


def load_predictions(results_dir):
    """
    Load predictions and metadata from results directory.

    Args:
        results_dir (str): Path to results directory

    Returns:
        predictions (np.ndarray): Predicted knee indices (N,)
        pred_map (np.ndarray): 2D spatial map
        stats (dict): Statistics
        metadata (dict): Metadata
    """
    print(f"Loading predictions from: {results_dir}")

    predictions = np.load(os.path.join(results_dir, 'predictions.npy'))
    pred_map = np.load(os.path.join(results_dir, 'prediction_map.npy'))

    with open(os.path.join(results_dir, 'nisar_stats.json'), 'r') as f:
        stats = json.load(f)

    with open(os.path.join(results_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    print(f"  Loaded {len(predictions):,} CPI predictions")
    print(f"  Spatial map shape: {pred_map.shape}")

    return predictions, pred_map, stats, metadata


def plot_histogram(predictions, stats, output_path):
    """
    Plot histogram of knee predictions.

    Args:
        predictions (np.ndarray): Predicted knee indices
        stats (dict): Statistics dictionary
        output_path (str): Output file path
    """
    print(f"\nGenerating histogram: {output_path}")

    unique = sorted(stats['distribution'].keys())
    counts = [stats['distribution'][k] for k in unique]
    colors = ['green' if k == 0 else 'red' for k in unique]
    labels = ['Clean' if k == 0 else f'Knee@{k}' for k in unique]

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(unique, counts, color=colors, alpha=0.7, edgecolor='black', width=0.8)
    ax.set_xlabel('Predicted Knee Index', fontsize=14)
    ax.set_ylabel('Number of CPIs', fontsize=14)
    ax.set_title(f'NISAR RFI Knee Prediction Distribution\n'
                 f'RFI Rate: {100*stats["rfi_rate"]:.2f}% ({stats["rfi_cpis"]:,}/{stats["total_cpis"]:,} CPIs)',
                 fontsize=15)
    ax.set_xticks(unique)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}',
                ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_spatial_map(pred_map, stats, output_path, cmap_name='viridis', vmax=None):
    """
    Plot spatial map of knee predictions with sequential colormap.

    Args:
        pred_map (np.ndarray): 2D spatial map
        stats (dict): Statistics dictionary
        output_path (str): Output file path
        cmap_name (str): Matplotlib colormap name (sequential)
        vmax (int|None): Maximum value for colormap (default: max knee value)
    """
    print(f"\nGenerating spatial map: {output_path}")

    # Determine colormap range
    if vmax is None:
        vmax = int(np.max(pred_map))

    fig, ax = plt.subplots(figsize=(16, 10))

    # Use sequential colormap that increases with knee value
    im = ax.imshow(pred_map, aspect='auto', cmap=cmap_name,
                   vmin=0, vmax=vmax, interpolation='nearest')

    cbar = fig.colorbar(im, ax=ax, label='Predicted Knee Index', shrink=0.8)
    cbar.ax.tick_params(labelsize=11)

    ax.set_xlabel('Range CPI Index', fontsize=14)
    ax.set_ylabel('Pulse CPI Index', fontsize=14)
    ax.set_title(f'NISAR RFI Knee Predictions - Spatial Distribution\n'
                 f'RFI Rate: {100*stats["rfi_rate"]:.2f}% | Total CPIs: {stats["total_cpis"]:,}',
                 fontsize=15)

    # Add grid for better readability
    ax.grid(True, which='both', color='white', linewidth=0.3, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_bounded_spatial_map(pred_map, stats, output_path, max_knee=4, cmap_name='turbo'):
    """
    Plot bounded spatial map where knee values > max_knee are set to 0.

    This "recovers" CPIs with weak RFI by treating them as clean,
    preserving the data even if contaminated.

    Args:
        pred_map (np.ndarray): 2D spatial map
        stats (dict): Statistics dictionary
        output_path (str): Output file path
        max_knee (int): Maximum knee value to preserve (default: 4)
        cmap_name (str): Matplotlib colormap name (default: turbo)
    """
    print(f"\nGenerating bounded spatial map (max_knee={max_knee}): {output_path}")

    # Create bounded version: knee > max_knee → 0
    bounded_map = pred_map.copy()
    bounded_map[bounded_map > max_knee] = 0

    # Calculate statistics for bounded map
    n_recovered = np.sum((pred_map > max_knee) & (pred_map > 0))
    n_rfi_bounded = np.sum(bounded_map > 0)
    recovery_rate = n_recovered / stats['total_cpis'] if stats['total_cpis'] > 0 else 0

    # Create discrete colormap using BoundaryNorm
    cmap = plt.get_cmap(cmap_name)
    norm = BoundaryNorm(
        np.arange(-0.5, max_knee + 1.5, 1),
        cmap.N
    )

    fig, ax = plt.subplots(figsize=(16, 10))

    im = ax.imshow(bounded_map, aspect='auto', cmap=cmap, norm=norm,
                   interpolation='nearest')

    cbar = fig.colorbar(im, ax=ax, label=f'Bounded Knee Index (0-{max_knee})',
                        shrink=0.8, ticks=np.arange(0, max_knee + 1))
    cbar.ax.tick_params(labelsize=11)

    ax.set_xlabel('Range CPI Index', fontsize=14)
    ax.set_ylabel('Pulse CPI Index', fontsize=14)
    ax.set_title(f'NISAR RFI Knee Predictions - Bounded Map (max_knee={max_knee})\n'
                 f'Original RFI: {stats["rfi_cpis"]:,} CPIs | '
                 f'Recovered: {n_recovered:,} CPIs ({100*recovery_rate:.2f}%) | '
                 f'Remaining RFI: {n_rfi_bounded:,} CPIs',
                 fontsize=15)

    ax.grid(True, which='both', color='white', linewidth=0.3, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")

    # Print recovery statistics
    print(f"\n  Bounded Map Statistics (max_knee={max_knee}):")
    print(f"    Original RFI CPIs: {stats['rfi_cpis']:,}")
    print(f"    Recovered CPIs (knee > {max_knee}): {n_recovered:,} ({100*recovery_rate:.2f}%)")
    print(f"    Remaining RFI CPIs: {n_rfi_bounded:,}")
    print(f"    Effective RFI rate: {100*n_rfi_bounded/stats['total_cpis']:.2f}%")


def plot_comparison_maps(pred_map, stats, output_path, max_knee=4, cmap_name='viridis'):
    """
    Plot side-by-side comparison of original vs bounded maps.

    Args:
        pred_map (np.ndarray): 2D spatial map
        stats (dict): Statistics dictionary
        output_path (str): Output file path
        max_knee (int): Maximum knee value to preserve
        cmap_name (str): Matplotlib colormap name
    """
    print(f"\nGenerating comparison maps: {output_path}")

    bounded_map = pred_map.copy()
    bounded_map[bounded_map > max_knee] = 0

    n_recovered = np.sum((pred_map > max_knee) & (pred_map > 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

    # Original map
    vmax_orig = int(np.max(pred_map))
    im1 = ax1.imshow(pred_map, aspect='auto', cmap=cmap_name,
                     vmin=0, vmax=vmax_orig, interpolation='nearest')
    cbar1 = fig.colorbar(im1, ax=ax1, label='Knee Index', shrink=0.8)
    ax1.set_xlabel('Range CPI Index', fontsize=13)
    ax1.set_ylabel('Pulse CPI Index', fontsize=13)
    ax1.set_title(f'Original Predictions\nRFI CPIs: {stats["rfi_cpis"]:,} ({100*stats["rfi_rate"]:.2f}%)',
                  fontsize=14)
    ax1.grid(True, which='both', color='white', linewidth=0.3, alpha=0.3)

    # Bounded map
    im2 = ax2.imshow(bounded_map, aspect='auto', cmap=cmap_name,
                     vmin=0, vmax=max_knee, interpolation='nearest')
    cbar2 = fig.colorbar(im2, ax=ax2, label=f'Bounded Knee (0-{max_knee})', shrink=0.8)
    ax2.set_xlabel('Range CPI Index', fontsize=13)
    ax2.set_ylabel('Pulse CPI Index', fontsize=13)
    n_rfi_bounded = np.sum(bounded_map > 0)
    ax2.set_title(f'Bounded Map (max_knee={max_knee})\n'
                  f'Recovered: {n_recovered:,} | Remaining RFI: {n_rfi_bounded:,}',
                  fontsize=14)
    ax2.grid(True, which='both', color='white', linewidth=0.3, alpha=0.3)

    fig.suptitle(f'NISAR RFI Knee Predictions - Comparison\nTotal CPIs: {stats["total_cpis"]:,}',
                 fontsize=16, y=0.98)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot NISAR knee predictions')
    parser.add_argument('--results-dir', required=True,
                        help='Path to results directory (output of predict_nisar.py)')
    parser.add_argument('--max-knee', type=int, default=4,
                        help='Maximum knee value for bounded map (default: 4)')
    parser.add_argument('--cmap', default='viridis',
                        help='Matplotlib colormap name (default: viridis). '
                             'Try: viridis, plasma, inferno, magma, cividis, turbo, jet')
    parser.add_argument('--no-histogram', action='store_true',
                        help='Skip histogram plot')
    parser.add_argument('--no-spatial', action='store_true',
                        help='Skip spatial map plot')
    parser.add_argument('--no-bounded', action='store_true',
                        help='Skip bounded map plot')
    parser.add_argument('--comparison', action='store_true',
                        help='Generate side-by-side comparison plot')

    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        print(f"ERROR: Results directory not found: {args.results_dir}")
        print(f"Run predict_nisar.py first to generate predictions.")
        sys.exit(1)

    # Load predictions
    predictions, pred_map, stats, metadata = load_predictions(args.results_dir)

    print(f"\n{'='*70}")
    print("Generating Plots")
    print(f"{'='*70}")

    # Generate plots
    if not args.no_histogram:
        plot_histogram(
            predictions, stats,
            os.path.join(args.results_dir, 'nisar_predictions_histogram.png')
        )

    if not args.no_spatial:
        plot_spatial_map(
            pred_map, stats,
            os.path.join(args.results_dir, 'nisar_predictions_spatial.png'),
            cmap_name=args.cmap
        )

    if not args.no_bounded:
        plot_bounded_spatial_map(
            pred_map, stats,
            os.path.join(args.results_dir, 'nisar_predictions_bounded.png'),
            max_knee=args.max_knee,
            cmap_name=args.cmap
        )

    if args.comparison:
        plot_comparison_maps(
            pred_map, stats,
            os.path.join(args.results_dir, 'nisar_predictions_comparison.png'),
            max_knee=args.max_knee,
            cmap_name=args.cmap
        )

    print(f"\n{'='*70}")
    print("Plotting Complete!")
    print(f"{'='*70}")
    print(f"Plots saved to: {args.results_dir}")
    if not args.no_histogram:
        print(f"  - nisar_predictions_histogram.png")
    if not args.no_spatial:
        print(f"  - nisar_predictions_spatial.png")
    if not args.no_bounded:
        print(f"  - nisar_predictions_bounded.png (max_knee={args.max_knee})")
    if args.comparison:
        print(f"  - nisar_predictions_comparison.png")


if __name__ == '__main__':
    main()
