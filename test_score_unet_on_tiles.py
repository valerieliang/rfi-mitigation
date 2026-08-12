#!/usr/bin/env python
"""
test_score_unet_on_tiles_remote.py

============================================================================
FOR TESTING ON CONTROLLED DATA ONLY (pre-extracted tiles from test suite)
FOR REAL L0B SCENES: Use score_unet_scene.py instead
============================================================================

This script is designed to run on the ISCE3 server where PyTorch is available.
It processes pre-extracted tile datasets (HDF5 files with tiles/masks/valid).

This is for testing on controlled data like la_blob_figs/raw_tiles_A_HH.h5.

For scoring real NISAR L0B granules, use score_unet_scene.py which reads
directly from the L0B file like score_scene.py does.

Usage:
    # On isce3 server (for testing pre-extracted tiles):
    python test_score_unet_on_tiles_remote.py \\
        --tiles /path/to/raw_tiles_A_HH.h5 \\
        --model /path/to/best_model.pth \\
        --output-dir results/test_tiles_A_HH

Outputs:
    predictions.h5              - Per-tile predictions and statistics
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

# Matplotlib backend for headless server
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Import the UNet model architecture
import sys
sys.path.insert(0, str(Path(__file__).parent))

# Import model and preprocessing functions
try:
    from unet import SegUNet
    from input_transforms import build_input_channels
    print("Using UNet from unet.py and input_transforms.py")
except ImportError:
    print("Required modules not found")
    raise


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

    print(f"  Processing {n_tiles} tiles in batches of {batch_size}...")

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

            if (i // batch_size + 1) % 10 == 0:
                print(f"    Processed {i + len(batch_tiles)}/{n_tiles} tiles")

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
    contaminated_samples_per_tile = []

    for prob, valid in zip(probs, valid_masks):
        if valid.sum() == 0:
            contamination_fractions.append(0.0)
            contaminated_samples_per_tile.append(0)
            continue

        # Contamination fraction over valid samples only
        flagged = (prob >= threshold) & valid
        frac = flagged.sum() / valid.sum()
        contamination_fractions.append(float(frac))
        contaminated_samples_per_tile.append(int(flagged.sum()))

    contamination_fractions = np.array(contamination_fractions)
    contaminated_samples = np.array(contaminated_samples_per_tile)

    return {
        'n_tiles': n_tiles,
        'mean_contamination': float(contamination_fractions.mean()),
        'median_contamination': float(np.median(contamination_fractions)),
        'max_contamination': float(contamination_fractions.max()),
        'tiles_with_contamination': int((contamination_fractions > 0.01).sum()),
        'contamination_fractions': contamination_fractions,
        'contaminated_samples': contaminated_samples,
    }


def save_predictions_h5(probs, tiles, valid_masks, tile_pulse, tile_range,
                       stats, args, output_dir):
    """Save per-tile predictions to HDF5 for later analysis."""
    path = Path(output_dir) / 'predictions.h5'
    with h5py.File(path, 'w') as f:
        f.attrs['tiles_file'] = str(args.tiles)
        f.attrs['model'] = str(args.model)
        f.attrs['threshold'] = args.threshold
        f.attrs['labeled'] = False
        f.attrs['note'] = ('Real scene, no labels: contamination fractions are '
                          'model predictions, not ground truth')

        # Per-tile data
        f.create_dataset('probabilities', data=probs.astype(np.float32),
                        compression='gzip', compression_opts=4)
        f.create_dataset('contamination_fraction',
                        data=stats['contamination_fractions'].astype(np.float32))
        f.create_dataset('contaminated_samples',
                        data=stats['contaminated_samples'].astype(np.int32))
        f.create_dataset('tile_pulse', data=tile_pulse)
        f.create_dataset('tile_range', data=tile_range)

        # Summary statistics
        f.attrs['n_tiles'] = stats['n_tiles']
        f.attrs['mean_contamination'] = stats['mean_contamination']
        f.attrs['median_contamination'] = stats['median_contamination']
        f.attrs['max_contamination'] = stats['max_contamination']
        f.attrs['tiles_with_contamination'] = stats['tiles_with_contamination']

    print(f"  Saved predictions to {path}")


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def plot_contamination_map(stats, tile_pulse, tile_range, channel_name, output_dir):
    """
    Spatial heatmap showing number of RFI-contaminated samples per tile.

    Uses power-law normalization to better visualize low RFI levels.
    """
    from matplotlib.colors import PowerNorm

    # Use contaminated sample counts instead of fractions
    contaminated_samples = stats['contaminated_samples']

    # Determine grid structure from tile origins
    unique_pulse = np.unique(tile_pulse)
    unique_range = np.unique(tile_range)
    n_pt = len(unique_pulse)
    n_rt = len(unique_range)

    # Reshape to grid
    try:
        grid = contaminated_samples.reshape(n_pt, n_rt)
    except ValueError:
        # Fallback: assume tiles are in order
        n_pt = int(np.sqrt(len(contaminated_samples)))
        n_rt = len(contaminated_samples) // n_pt
        grid = contaminated_samples[:n_pt * n_rt].reshape(n_pt, n_rt)

    # Assume 256x256 tiles (max possible)
    max_samples = 256 * 256

    fig, ax = plt.subplots(figsize=(13, 6))

    # Use power-law normalization to compress dynamic range
    # gamma < 1 expands low values, compresses high values
    im = ax.imshow(grid, aspect='auto', cmap='YlOrRd', origin='upper',
                   vmin=0, vmax=max_samples, interpolation='nearest',
                   extent=[0, n_rt, n_pt, 0],
                   norm=PowerNorm(gamma=0.5))

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Number of RFI-contaminated samples per tile')

    ax.set_xlabel('Range tile index')
    ax.set_ylabel('Pulse tile index')

    mean_count = contaminated_samples.mean()
    ax.set_title(f'UNet RFI Contamination Map - {channel_name} (REAL DATA, NO LABELS)\n'
                 f'Mean: {mean_count:.0f} samples/tile, '
                 f'{stats["tiles_with_contamination"]}/{stats["n_tiles"]} tiles with RFI')

    fig.tight_layout()
    path = Path(output_dir) / 'contamination_map.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_contamination_histogram(stats, channel_name, output_dir):
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
    ax1.set_title(f'Distribution of tile contamination - {channel_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Cumulative distribution
    sorted_contam = np.sort(contamination)
    cumulative = np.arange(1, len(sorted_contam) + 1) / len(sorted_contam)
    ax2.plot(sorted_contam, cumulative, linewidth=2, color='steelblue')
    ax2.axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='50th percentile')
    ax2.axhline(0.9, color='red', linestyle='--', alpha=0.5, label='90th percentile')
    ax2.set_xlabel('Contamination fraction')
    ax2.set_ylabel('Cumulative fraction of tiles')
    ax2.set_title('Cumulative distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = Path(output_dir) / 'contamination_hist.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_sample_predictions(tiles, valid_masks, probs, tile_pulse, tile_range,
                            channel_name, output_dir, n_samples=12, seed=42):
    """
    Grid of sample tiles with their predicted masks.

    Shows magnitude (in dB), predicted mask, and overlay side by side.
    """
    rng = np.random.default_rng(seed)

    # Sample tiles with different contamination levels
    contamination = np.array([(p >= 0.5).sum() / v.sum() if v.sum() > 0 else 0.0
                              for p, v in zip(probs, valid_masks)])

    # Sample from different contamination ranges
    n_per_range = max(1, n_samples // 3)
    indices = []

    # Low contamination
    low_idx = np.where(contamination < 0.05)[0]
    if len(low_idx) > 0:
        indices.extend(rng.choice(low_idx, size=min(n_per_range, len(low_idx)),
                                 replace=False))

    # Medium contamination
    medium_idx = np.where((contamination >= 0.05) & (contamination < 0.2))[0]
    if len(medium_idx) > 0:
        indices.extend(rng.choice(medium_idx, size=min(n_per_range, len(medium_idx)),
                                 replace=False))

    # High contamination
    high_idx = np.where(contamination >= 0.2)[0]
    if len(high_idx) > 0:
        indices.extend(rng.choice(high_idx, size=min(n_per_range, len(high_idx)),
                                 replace=False))

    if not indices:
        print("  Warning: No tiles available for sample predictions")
        return

    indices = indices[:n_samples]
    n_rows = len(indices)

    fig = plt.figure(figsize=(16, 2.5 * n_rows))
    gs = GridSpec(n_rows, 3, figure=fig, width_ratios=[1, 1, 1])

    for i, idx in enumerate(indices):
        tile = tiles[idx]
        valid = valid_masks[idx]
        prob = probs[idx]

        # Magnitude in dB (20*log10 for complex voltage)
        mag_db = 20.0 * np.log10(np.abs(tile) + 1e-12)
        mag_db = np.where(valid, mag_db, np.nan)

        # Predicted mask
        mask = (prob >= 0.5) & valid
        contam_frac = mask.sum() / valid.sum() if valid.sum() > 0 else 0.0

        # Plot magnitude
        ax1 = fig.add_subplot(gs[i, 0])
        vmin, vmax = np.nanpercentile(mag_db[valid], [1, 99])
        ax1.imshow(mag_db, aspect='auto', cmap='gray',
                  vmin=vmin, vmax=vmax, interpolation='nearest')
        ax1.set_title(f'Tile {idx}: Magnitude (dB)\np={tile_pulse[idx]}, r={tile_range[idx]}',
                     fontsize=9)
        ax1.set_ylabel('Pulse')
        if i == n_rows - 1:
            ax1.set_xlabel('Range sample')

        # Plot predicted mask only
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.imshow(mask, aspect='auto', cmap='Reds',
                  vmin=0, vmax=1, interpolation='nearest')
        ax2.set_title(f'Predicted RFI Mask\nContamination: {contam_frac:.1%}',
                     fontsize=9)
        ax2.set_ylabel('Pulse')
        if i == n_rows - 1:
            ax2.set_xlabel('Range sample')

        # Plot overlay
        ax3 = fig.add_subplot(gs[i, 2])
        ax3.imshow(mag_db, aspect='auto', cmap='gray',
                  vmin=vmin, vmax=vmax, interpolation='nearest', alpha=0.7)
        ax3.imshow(mask, aspect='auto', cmap='Reds',
                  vmin=0, vmax=1, interpolation='nearest', alpha=0.5)
        ax3.set_title(f'Overlay\n{mask.sum()} / {valid.sum()} samples flagged',
                     fontsize=9)
        ax3.set_ylabel('Pulse')
        if i == n_rows - 1:
            ax3.set_xlabel('Range sample')

    fig.suptitle(f'UNet Predictions on Real Tiles - {channel_name} '
                f'(no labels, seed={seed})',
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
        description='Score real NISAR tiles with a pretrained UNet (run on server).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--tiles', required=True,
                       help='Input HDF5 file with tiles and valid masks')
    parser.add_argument('--model', required=True,
                       help='Trained PyTorch model (.pth)')
    parser.add_argument('--output-dir', default='results/unet_scene',
                       help='Output directory for results')
    parser.add_argument('--channel', default='unknown',
                       help='Channel name for plot titles (e.g., "A-HH")')
    parser.add_argument('--device', default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    parser.add_argument('--batch-size', type=int, default=16,
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
    print(f"  tiles   : {args.tiles}")
    print(f"  model   : {args.model}")
    print(f"  channel : {args.channel}")
    print(f"  device  : {device}")
    print(f"  threshold: {args.threshold}")

    # Load data
    print(f"\nLoading tiles from {args.tiles}...")
    with h5py.File(args.tiles, 'r') as f:
        tiles = f['tiles'][:]
        valid_masks = f['valid'][:]
        tile_pulse = f['tile_pulse'][:] if 'tile_pulse' in f else np.arange(len(tiles)) * 256
        tile_range = f['tile_range'][:] if 'tile_range' in f else np.zeros(len(tiles), dtype=int)

    print(f"  Loaded {len(tiles)} tiles of shape {tiles[0].shape}")
    print(f"  Tile pulse range: [{tile_pulse.min()}, {tile_pulse.max()}]")
    print(f"  Tile range range: [{tile_range.min()}, {tile_range.max()}]")

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
    print(f"  Model loaded: {model.n_parameters():,} parameters, "
          f"RF {model.receptive_field()} px")

    # Run inference
    print(f"\nRunning inference...")
    probs = score_tiles(tiles, valid_masks, model, device=device,
                       batch_size=args.batch_size)
    print(f"  ✓ Predictions computed")

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

    # Save predictions
    print(f"\nSaving predictions...")
    save_predictions_h5(probs, tiles, valid_masks, tile_pulse, tile_range,
                       stats, args, output_dir)

    # Save JSON results
    results = {
        'tiles_file': str(args.tiles),
        'model': str(args.model),
        'channel': args.channel,
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
    print(f"  Saved {results_path}")

    # Generate plots
    print(f"\nGenerating visualizations...")
    plot_contamination_map(stats, tile_pulse, tile_range, args.channel, output_dir)
    plot_contamination_histogram(stats, args.channel, output_dir)
    plot_sample_predictions(tiles, valid_masks, probs, tile_pulse, tile_range,
                           args.channel, output_dir, n_samples=args.n_samples,
                           seed=args.seed)

    print(f"\n{'='*70}")
    print(f"All results saved to {output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
