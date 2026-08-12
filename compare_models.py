#!/usr/bin/env python
"""
compare_models.py

Compare 2-channel UNet vs 4-channel SegUNet performance.

This script compares:
1. Training metrics (test set IoU, F1, precision, recall)
2. Model complexity (parameters, architecture)
3. Scene-level predictions (contamination statistics)

Usage:
    # Compare training metrics
    python compare_models.py --training-only

    # Compare scene predictions
    python compare_models.py --scene-only \
        --scene-2ch score/two_channel/unet_vienna \
        --scene-4ch score/four_channel/unet_vienna

    # Full comparison (training + scenes)
    python compare_models.py \
        --scene-2ch score/two_channel/unet_vienna \
        --scene-4ch score/four_channel/unet_vienna
"""

import argparse
import json
from pathlib import Path

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_training_results(model_dir):
    """Load training metrics from test_results.npz and history.npz."""
    model_dir = Path(model_dir)

    # Load test results
    test_path = model_dir / 'test_results.npz'
    if not test_path.exists():
        print(f"Warning: {test_path} not found")
        return None

    test_data = np.load(test_path)
    test_metrics = {
        'iou': float(test_data['iou']),
        'precision': float(test_data['precision']),
        'recall': float(test_data['recall']),
        'f1': float(test_data['f1']),
        'loss': float(test_data['loss']),
    }

    # Load training history
    history_path = model_dir / 'history.npz'
    history = None
    if history_path.exists():
        history_data = np.load(history_path)
        history = {
            'train_loss': history_data['train_loss'],
            'val_loss': history_data['val_loss'],
            'val_iou': history_data['val_iou'],
            'val_f1': history_data['val_f1'],
        }

    # Load config
    config_path = model_dir / 'config.json'
    config = None
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    return {
        'test_metrics': test_metrics,
        'history': history,
        'config': config,
    }


def load_scene_results(scene_dir, freq='A', pol='HH'):
    """Load scene predictions and results.json."""
    scene_dir = Path(scene_dir)

    # Load predictions HDF5
    pred_path = scene_dir / f'predictions_{freq}_{pol}.h5'
    if not pred_path.exists():
        print(f"Warning: {pred_path} not found")
        return None

    with h5py.File(pred_path, 'r') as f:
        predictions = {
            'contamination_fraction': f['contamination_fraction'][:],
            'contaminated_samples': f['contaminated_samples'][:],
            'tile_pulse': f['tile_pulse'][:],
            'tile_range': f['tile_range'][:],
            'attrs': dict(f.attrs),
        }

    # Load results.json
    results_path = scene_dir / 'results.json'
    results = None
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

    return {
        'predictions': predictions,
        'results': results,
    }


def compare_training_metrics(results_2ch, results_4ch):
    """Print comparison of training metrics."""
    print("\n" + "="*70)
    print("TRAINING METRICS COMPARISON (Test Set)")
    print("="*70)

    if results_2ch is None or results_4ch is None:
        print("ERROR: Missing training results")
        print(f"  2-channel: {'found' if results_2ch else 'NOT FOUND'}")
        print(f"  4-channel: {'found' if results_4ch else 'NOT FOUND'}")
        return

    m2 = results_2ch['test_metrics']
    m4 = results_4ch['test_metrics']

    print("\n{:<20} {:>15} {:>15} {:>15}".format(
        "Metric", "2-Channel UNet", "4-Channel SegUNet", "Difference"))
    print("-" * 70)

    for metric in ['iou', 'precision', 'recall', 'f1', 'loss']:
        v2 = m2[metric]
        v4 = m4[metric]
        diff = v4 - v2
        sign = "+" if diff > 0 else ""
        print(f"{metric.upper():<20} {v2:>15.4f} {v4:>15.4f} {sign}{diff:>14.4f}")

    print("\n" + "="*70)

    # Model configuration
    if results_2ch['config'] and results_4ch['config']:
        c2 = results_2ch['config']
        c4 = results_4ch['config']

        print("\nMODEL CONFIGURATION")
        print("-" * 70)
        print(f"{'':20} {'2-Channel':>15} {'4-Channel':>15}")
        print(f"{'Architecture':20} {'UNet':>15} {'SegUNet':>15}")
        print(f"{'Input channels':20} {2:>15} {4:>15}")
        print(f"{'Batch size':20} {c2['batch_size']:>15} {c4['batch_size']:>15}")
        print(f"{'Learning rate':20} {c2['lr']:>15.6f} {c4['lr']:>15.6f}")
        print(f"{'Epochs':20} {c2['epochs']:>15} {c4['epochs']:>15}")


def compare_scene_predictions(scene_2ch, scene_4ch, freq='A', pol='HH'):
    """Print comparison of scene-level predictions."""
    print("\n" + "="*70)
    print(f"SCENE PREDICTIONS COMPARISON ({freq}-{pol})")
    print("="*70)

    if scene_2ch is None or scene_4ch is None:
        print("ERROR: Missing scene results")
        print(f"  2-channel: {'found' if scene_2ch else 'NOT FOUND'}")
        print(f"  4-channel: {'found' if scene_4ch else 'NOT FOUND'}")
        return

    p2 = scene_2ch['predictions']['contamination_fraction']
    p4 = scene_4ch['predictions']['contamination_fraction']

    print("\n{:<30} {:>15} {:>15}".format(
        "Statistic", "2-Channel", "4-Channel"))
    print("-" * 70)

    print(f"{'Mean contamination':30} {p2.mean():>14.2%} {p4.mean():>14.2%}")
    print(f"{'Median contamination':30} {np.median(p2):>14.2%} {np.median(p4):>14.2%}")
    print(f"{'Max contamination':30} {p2.max():>14.2%} {p4.max():>14.2%}")
    print(f"{'Tiles with RFI (>1%)':30} {(p2 > 0.01).sum():>15} {(p4 > 0.01).sum():>15}")
    print(f"{'Total tiles':30} {len(p2):>15} {len(p4):>15}")

    # Correlation
    if len(p2) == len(p4):
        correlation = np.corrcoef(p2, p4)[0, 1]
        print(f"\n{'Correlation (Pearson)':30} {correlation:>15.4f}")
    else:
        print(f"\nWARNING: Different number of tiles ({len(p2)} vs {len(p4)})")


def plot_training_curves(results_2ch, results_4ch, output_path):
    """Plot training curves comparison."""
    if results_2ch is None or results_4ch is None:
        print("Skipping training curves plot (missing data)")
        return

    h2 = results_2ch['history']
    h4 = results_4ch['history']

    if h2 is None or h4 is None:
        print("Skipping training curves plot (missing history)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs_2ch = np.arange(1, len(h2['train_loss']) + 1)
    epochs_4ch = np.arange(1, len(h4['train_loss']) + 1)

    # Loss curves
    axes[0, 0].plot(epochs_2ch, h2['train_loss'], label='2-ch Train', color='blue', alpha=0.7)
    axes[0, 0].plot(epochs_2ch, h2['val_loss'], label='2-ch Val', color='blue', linestyle='--')
    axes[0, 0].plot(epochs_4ch, h4['train_loss'], label='4-ch Train', color='red', alpha=0.7)
    axes[0, 0].plot(epochs_4ch, h4['val_loss'], label='4-ch Val', color='red', linestyle='--')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Validation IoU
    axes[0, 1].plot(epochs_2ch, h2['val_iou'], label='2-channel', color='blue')
    axes[0, 1].plot(epochs_4ch, h4['val_iou'], label='4-channel', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Validation IoU')
    axes[0, 1].set_title('Validation IoU Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Validation F1
    axes[1, 0].plot(epochs_2ch, h2['val_f1'], label='2-channel', color='blue')
    axes[1, 0].plot(epochs_4ch, h4['val_f1'], label='4-channel', color='red')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Validation F1')
    axes[1, 0].set_title('Validation F1 Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Test metrics bar chart
    m2 = results_2ch['test_metrics']
    m4 = results_4ch['test_metrics']
    metrics = ['IoU', 'Precision', 'Recall', 'F1']
    values_2ch = [m2['iou'], m2['precision'], m2['recall'], m2['f1']]
    values_4ch = [m4['iou'], m4['precision'], m4['recall'], m4['f1']]

    x = np.arange(len(metrics))
    width = 0.35
    axes[1, 1].bar(x - width/2, values_2ch, width, label='2-channel', color='blue', alpha=0.7)
    axes[1, 1].bar(x + width/2, values_4ch, width, label='4-channel', color='red', alpha=0.7)
    axes[1, 1].set_xlabel('Metric')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Test Set Metrics Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].set_ylim(0.85, 1.0)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved training curves: {output_path}")


def plot_scene_comparison(scene_2ch, scene_4ch, output_path, freq='A', pol='HH'):
    """Plot scene prediction comparison."""
    if scene_2ch is None or scene_4ch is None:
        print("Skipping scene comparison plot (missing data)")
        return

    p2 = scene_2ch['predictions']['contamination_fraction']
    p4 = scene_4ch['predictions']['contamination_fraction']

    if len(p2) != len(p4):
        print(f"WARNING: Different number of tiles ({len(p2)} vs {len(p4)})")
        print("Skipping scene comparison plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Scatter plot: 2-ch vs 4-ch
    axes[0, 0].scatter(p2, p4, alpha=0.3, s=10, color='steelblue')
    axes[0, 0].plot([0, 1], [0, 1], 'r--', linewidth=1, label='Perfect agreement')
    axes[0, 0].set_xlabel('2-channel contamination')
    axes[0, 0].set_ylabel('4-channel contamination')
    axes[0, 0].set_title(f'Per-Tile Contamination: 2-ch vs 4-ch ({freq}-{pol})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    correlation = np.corrcoef(p2, p4)[0, 1]
    axes[0, 0].text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                   transform=axes[0, 0].transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Histogram comparison
    bins = np.linspace(0, max(p2.max(), p4.max()), 50)
    axes[0, 1].hist(p2, bins=bins, alpha=0.5, label='2-channel', color='blue', edgecolor='black')
    axes[0, 1].hist(p4, bins=bins, alpha=0.5, label='4-channel', color='red', edgecolor='black')
    axes[0, 1].set_xlabel('Contamination fraction')
    axes[0, 1].set_ylabel('Number of tiles')
    axes[0, 1].set_title('Distribution Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Difference histogram
    diff = p4 - p2
    axes[1, 0].hist(diff, bins=50, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].axvline(diff.mean(), color='orange', linestyle='--', linewidth=2,
                      label=f'Mean: {diff.mean():.3f}')
    axes[1, 0].set_xlabel('Difference (4-ch - 2-ch)')
    axes[1, 0].set_ylabel('Number of tiles')
    axes[1, 0].set_title('Contamination Difference Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Cumulative distribution
    sorted_2ch = np.sort(p2)
    sorted_4ch = np.sort(p4)
    cumulative = np.arange(1, len(p2) + 1) / len(p2)
    axes[1, 1].plot(sorted_2ch, cumulative, label='2-channel', color='blue', linewidth=2)
    axes[1, 1].plot(sorted_4ch, cumulative, label='4-channel', color='red', linewidth=2)
    axes[1, 1].set_xlabel('Contamination fraction')
    axes[1, 1].set_ylabel('Cumulative fraction of tiles')
    axes[1, 1].set_title('Cumulative Distribution Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved scene comparison: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)

    # Model directories
    parser.add_argument('--model-2ch', default='model/two_channel',
                       help='2-channel model directory')
    parser.add_argument('--model-4ch', default='model/four_channel',
                       help='4-channel model directory')

    # Scene directories
    parser.add_argument('--scene-2ch', default=None,
                       help='2-channel scene results directory')
    parser.add_argument('--scene-4ch', default=None,
                       help='4-channel scene results directory')

    # Output
    parser.add_argument('--output-dir', default='comparison',
                       help='Output directory for plots')

    # Modes
    parser.add_argument('--training-only', action='store_true',
                       help='Only compare training metrics')
    parser.add_argument('--scene-only', action='store_true',
                       help='Only compare scene predictions')

    # Scene parameters
    parser.add_argument('--freq', default='A')
    parser.add_argument('--pol', default='HH')

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("MODEL COMPARISON: 2-CHANNEL vs 4-CHANNEL")
    print("="*70)

    # Load training results
    results_2ch = None
    results_4ch = None
    if not args.scene_only:
        print(f"\nLoading training results...")
        print(f"  2-channel: {args.model_2ch}")
        print(f"  4-channel: {args.model_4ch}")
        results_2ch = load_training_results(args.model_2ch)
        results_4ch = load_training_results(args.model_4ch)

        compare_training_metrics(results_2ch, results_4ch)

        if results_2ch and results_4ch:
            plot_training_curves(results_2ch, results_4ch,
                                Path(args.output_dir) / 'training_comparison.png')

    # Load scene results
    scene_2ch = None
    scene_4ch = None
    if not args.training_only and args.scene_2ch and args.scene_4ch:
        print(f"\nLoading scene results...")
        print(f"  2-channel: {args.scene_2ch}")
        print(f"  4-channel: {args.scene_4ch}")
        scene_2ch = load_scene_results(args.scene_2ch, args.freq, args.pol)
        scene_4ch = load_scene_results(args.scene_4ch, args.freq, args.pol)

        compare_scene_predictions(scene_2ch, scene_4ch, args.freq, args.pol)

        if scene_2ch and scene_4ch:
            plot_scene_comparison(scene_2ch, scene_4ch,
                                 Path(args.output_dir) / f'scene_comparison_{args.freq}_{args.pol}.png',
                                 args.freq, args.pol)

    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
