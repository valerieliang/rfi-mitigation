#!/usr/bin/env python
"""
score_unet.py

Run a PRETRAINED UNet segmentation model on real NISAR L0B tiles with NO LABELS.

Similar to score_scene.py but for the semantic segmentation model instead of the
eigenvalue-based knee classifier. There is no ground truth, so this produces
visualization of:
  1. SPATIAL MAPS - Predicted RFI contamination masks over the tile grid
  2. CONTAMINATION STATISTICS - Per-tile contamination fraction
  3. INDIVIDUAL EXAMPLES - Sample tiles showing the predicted masks

Usage:
    python score_unet.py \\
        --tiles la_blob_figs/raw_tiles_A_HH.h5 \\
        --model model/best_model.pth \\
        --output-dir results/unet_scene

Outputs:
    contamination_map.png       - Heatmap of contamination fraction per tile
    contamination_hist.png      - Distribution of contamination levels
    sample_predictions.png      - Grid showing individual tile predictions
    results.json                - Summary statistics
"""

import argparse
import json
from pathlib import Path

import numpy as np
import h5py
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Import the UNet model architecture
import sys
sys.path.insert(0, str(Path(__file__).parent))
from unet import SegUNet, build_input_channels


# ---------------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------------

def score_tiles(tiles, valid_masks, model, device='cuda', batch_size=8):
    """
    Run the UNet on a batch of tiles and return predictions.

    Parameters
    ----------
    tiles : (N, P, K) complex array
    valid_masks : (N, P, K) bool array
    model : trained SegUNet
    device : 'cuda' or 'cpu'
    batch_size : number of tiles to process at once

    Returns
    -------
    probs : (N, P, K) float array of predicted RFI probabilities
    """
    model.eval()
    n_tiles = len(tiles)
    probs_all = []

    with torch.no_grad():
        for i in range(0, n_tiles, batch_size):
            batch_tiles = tiles[i:i+batch_size]
            batch_valid = valid_masks[i:i+batch_size]

            # Build input channels for each tile
            inputs = []
            for tile, valid in zip(batch_tiles, batch_valid):
                x = build_input_channels(tile, valid, n_channels=4)
                inputs.append(x)

            inputs = torch.from_numpy(np.stack(inputs, axis=0)).to(device)

            # Forward pass
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_all.append(probs[:, 0])  # Remove channel dim

    return np.concatenate(probs_all, axis=0)


def compute_contamination_stats(probs, valid_masks, threshold=0.5):
    """
    Compute per-tile contamination statistics.

    Parameters
    ----------
    probs : (N, P, K) predicted probabilities
    valid_masks : (N, P, K) validity masks
    threshold : probability threshold for binary classification

    Returns
    -------
    dict with contamination statistics
    """
    n_tiles = len(probs)
    contamination_fractions = []

    for prob, valid in zip(probs, valid_masks):
        if valid.sum() == 0:
            contamination_fractions.append(0.0)
            continue

        # Contamination fraction over valid samples only
        flagged = (prob >= threshold) & valid
        frac = flagged.sum() / valid.sum()
        contamination_fractions.append(float(frac))

    contamination_fractions = np.array(contamination_fractions)

    return {
        'n_tiles': n_tiles,
        'mean_contamination': float(contamination_fractions.mean()),
        'median_contamination': float(np.median(contamination_fractions)),
        'max_contamination': float(contamination_fractions.max()),
        'tiles_with_contamination': int((contamination_fractions > 0.01).sum()),
        'contamination_fractions': contamination_fractions,
    }


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def plot_contamination_map(stats, tile_pulse, tile_range, output_dir):
    """
    Spatial heatmap of contamination fraction per tile.

    Similar to the knee_map from score_scene.py, but shows contamination
    fraction instead of number of RFI eigenvalues.
    """
    contamination = stats['contamination_fractions']

    # Determine grid structure from tile origins
    unique_pulse = np.unique(tile_pulse)
    unique_range = np.unique(tile_range)
    n_pt = len(unique_pulse)
    n_rt = len(unique_range)

    # Reshape to grid
    grid = contamination.reshape(n_pt, n_rt)

    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(grid, aspect='auto', cmap='YlOrRd', origin='upper',
                   vmin=0, vmax=0.5, interpolation='nearest',
                   extent=[0, n_rt, n_pt, 0])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Contamination fraction (RFI samples / valid samples)')

    ax.set_xlabel('Range tile index')
    ax.set_ylabel('Pulse tile index')
    ax.set_title(f'UNet RFI Contamination Map (REAL DATA, NO LABELS)\n'
                 f'Mean contamination: {stats["mean_contamination"]:.1%}, '
                 f'{stats["tiles_with_contamination"]}/{stats["n_tiles"]} tiles with RFI')

    fig.tight_layout()
    path = Path(output_dir) / 'contamination_map.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_contamination_histogram(stats, output_dir):
    """Distribution of contamination fractions across all tiles."""
    contamination = stats['contamination_fractions']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(contamination, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(stats['mean_contamination'], color='red', linestyle='--',
                linewidth=2, label=f'Mean: {stats["mean_contamination"]:.1%}')
    ax1.axvline(stats['median_contamination'], color='orange', linestyle='--',
                linewidth=2, label=f'Median: {stats["median_contamination"]:.1%}')
    ax1.set_xlabel('Contamination fraction')
    ax1.set_ylabel('Number of tiles')
    ax1.set_title('Distribution of tile contamination')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Cumulative distribution
    sorted_contam = np.sort(contamination)
    cumulative = np.arange(1, len(sorted_contam) + 1) / len(sorted_contam)
    ax2.plot(sorted_contam, cumulative, linewidth=2, color='steelblue')
    ax2.axhline(0.5, color='orange', linestyle='--', alpha=0.5)
    ax2.axhline(0.9, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Contamination fraction')
    ax2.set_ylabel('Cumulative fraction of tiles')
    ax2.set_title('Cumulative distribution')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = Path(output_dir) / 'contamination_hist.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_sample_predictions(tiles, valid_masks, probs, tile_pulse, tile_range,
                            output_dir, n_samples=12, seed=42):
    """
    Grid of sample tiles with their predicted masks.

    Shows both the magnitude (in dB) and the predicted RFI mask side by side.
    """
    rng = np.random.default_rng(seed)

    # Sample tiles with different contamination levels
    contamination = np.array([(p >= 0.5).sum() / v.sum() if v.sum() > 0 else 0.0
                              for p, v in zip(probs, valid_masks)])

    # Sample from different contamination ranges
    n_per_range = n_samples // 3
    low = rng.choice(np.where(contamination < 0.05)[0], size=min(n_per_range,
                     (contamination < 0.05).sum()), replace=False)
    medium = rng.choice(np.where((contamination >= 0.05) & (contamination < 0.2))[0],
                       size=min(n_per_range, ((contamination >= 0.05) & (contamination < 0.2)).sum()),
                       replace=False)
    high = rng.choice(np.where(contamination >= 0.2)[0],
                     size=min(n_per_range, (contamination >= 0.2).sum()), replace=False)

    indices = np.concatenate([low, medium, high])[:n_samples]

    n_rows = len(indices)
    fig = plt.figure(figsize=(14, 2.5 * n_rows))
    gs = GridSpec(n_rows, 3, figure=fig, width_ratios=[1, 1, 0.05])

    for i, idx in enumerate(indices):
        tile = tiles[idx]
        valid = valid_masks[idx]
        prob = probs[idx]

        # Magnitude in dB (20*log10 for complex voltage)
        mag_db = 20.0 * np.log10(np.abs(tile) + 1e-12)
        mag_db = np.where(valid, mag_db, np.nan)

        # Predicted mask
        mask = (prob >= 0.5) & valid

        # Plot magnitude
        ax1 = fig.add_subplot(gs[i, 0])
        vmin, vmax = np.nanpercentile(mag_db[valid], [1, 99])
        im1 = ax1.imshow(mag_db, aspect='auto', cmap='gray',
                        vmin=vmin, vmax=vmax, interpolation='nearest')
        ax1.set_title(f'Tile {idx}: Magnitude (dB)\np={tile_pulse[idx]}, r={tile_range[idx]}',
                     fontsize=9)
        ax1.set_ylabel('Pulse')
        ax1.set_xlabel('Range sample')

        # Plot predicted mask
        ax2 = fig.add_subplot(gs[i, 1])
        # Show mask overlaid on magnitude
        im2 = ax2.imshow(mag_db, aspect='auto', cmap='gray',
                        vmin=vmin, vmax=vmax, interpolation='nearest', alpha=0.6)
        im3 = ax2.imshow(mask, aspect='auto', cmap='Reds',
                        vmin=0, vmax=1, interpolation='nearest', alpha=0.6)
        contam_frac = mask.sum() / valid.sum() if valid.sum() > 0 else 0.0
        ax2.set_title(f'Predicted RFI Mask\nContamination: {contam_frac:.1%}',
                     fontsize=9)
        ax2.set_ylabel('Pulse')
        ax2.set_xlabel('Range sample')

    fig.suptitle(f'UNet Predictions on Real Tiles (no labels, seed={seed})',
                fontsize=12, y=0.995)
    fig.tight_layout()
    path = Path(output_dir) / 'sample_predictions.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Score real NISAR tiles with a pretrained UNet segmentation model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--tiles', required=True,
                       help='Input HDF5 file with tiles and valid masks')
    parser.add_argument('--model', required=True,
                       help='Trained PyTorch model (.pth)')
    parser.add_argument('--output-dir', default='results/unet_scene',
                       help='Output directory for results')
    parser.add_argument('--device', default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for inference')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Probability threshold for binary classification')
    parser.add_argument('--n-samples', type=int, default=12,
                       help='Number of sample tiles to visualize')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sample selection')
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print('UNet scoring on real tiles (UNLABELED)')
    print(f"{'='*70}")
    print(f"  tiles : {args.tiles}")
    print(f"  model : {args.model}")
    print(f"  device: {device}")
    print(f"  threshold: {args.threshold}")

    # Load data
    print(f"\nLoading tiles from {args.tiles}...")
    with h5py.File(args.tiles, 'r') as f:
        tiles = f['tiles'][:]
        valid_masks = f['valid'][:]
        tile_pulse = f['tile_pulse'][:] if 'tile_pulse' in f else np.arange(len(tiles)) * 256
        tile_range = f['tile_range'][:] if 'tile_range' in f else np.zeros(len(tiles), dtype=int)

    print(f"  Loaded {len(tiles)} tiles of shape {tiles[0].shape}")

    # Load model
    print(f"\nLoading model from {args.model}...")
    model = SegUNet(in_channels=4, out_channels=1, base_channels=16, depth=3)
    checkpoint = torch.load(args.model, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    print(f"  Model loaded with {model.n_parameters():,} parameters")

    # Run inference
    print(f"\nRunning inference on {len(tiles)} tiles...")
    probs = score_tiles(tiles, valid_masks, model, device=device,
                       batch_size=args.batch_size)
    print(f"  Predictions computed")

    # Compute statistics
    print(f"\nComputing contamination statistics...")
    stats = compute_contamination_stats(probs, valid_masks, threshold=args.threshold)

    print(f"\n{'='*60}")
    print('CONTAMINATION STATISTICS (unlabeled - no ground truth)')
    print(f"{'='*60}")
    print(f"  Total tiles              : {stats['n_tiles']}")
    print(f"  Tiles with contamination : {stats['tiles_with_contamination']} "
          f"({100 * stats['tiles_with_contamination'] / stats['n_tiles']:.1f}%)")
    print(f"  Mean contamination       : {stats['mean_contamination']:.2%}")
    print(f"  Median contamination     : {stats['median_contamination']:.2%}")
    print(f"  Max contamination        : {stats['max_contamination']:.2%}")

    # Save results
    results = {
        'tiles_file': str(args.tiles),
        'model': str(args.model),
        'threshold': args.threshold,
        'n_tiles': stats['n_tiles'],
        'tiles_with_contamination': int(stats['tiles_with_contamination']),
        'mean_contamination': float(stats['mean_contamination']),
        'median_contamination': float(stats['median_contamination']),
        'max_contamination': float(stats['max_contamination']),
    }

    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved results to {results_path}")

    # Generate plots
    print(f"\nGenerating visualizations...")
    plot_contamination_map(stats, tile_pulse, tile_range, output_dir)
    plot_contamination_histogram(stats, output_dir)
    plot_sample_predictions(tiles, valid_masks, probs, tile_pulse, tile_range,
                           output_dir, n_samples=args.n_samples, seed=args.seed)

    print(f"\n{'='*70}")
    print(f"All results saved to {output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
