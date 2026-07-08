#!/usr/bin/env python
"""
Plot NISAR predictions from read_nisar_isce3.py output.

Usage:
    python plot_nisar_results.py nisar_out/nisar_A_HV_eigenvalues.h5
    python plot_nisar_results.py nisar_out/nisar_A_HV_eigenvalues.h5 --output-dir plots/
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Plot NISAR RFI predictions')
    parser.add_argument('input_file', help='Input HDF5 file from read_nisar_isce3.py')
    parser.add_argument('--output-dir', default='plots', help='Output directory for plots')
    parser.add_argument('--knee-bound', type=int, default=4,
                        help='Knee threshold: RFI recoverable if knee <= threshold (default: 4)')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load data
    print(f"\nLoading data from: {args.input_file}")
    with h5py.File(args.input_file, 'r') as f:
        # Predictions
        knee_indices = f['predictions/knee_indices'][:]
        confidences = f['predictions/confidences'][:]
        probabilities = f['predictions/probabilities'][:]

        # Check if 2D spatial data
        is_spatial = len(knee_indices.shape) == 2

        # Eigenvalues
        eigenvalues = f['evd/eigenvalues'][:]

        # Metadata
        freq = f['metadata'].attrs.get('frequency', 'A')
        pol = f['metadata'].attrs.get('polarization', 'HV')
        cpi_len = f['metadata'].attrs.get('cpi_len', 16)
        num_cpi = f['metadata'].attrs.get('num_cpi', knee_indices.shape[0] if is_spatial else len(knee_indices))

        print(f"  Frequency: {freq}, Polarization: {pol}")
        print(f"  Number of CPIs: {num_cpi}")
        print(f"  CPI length: {cpi_len}")

        if is_spatial:
            n_pulse_tiles, n_range_tiles = knee_indices.shape
            print(f"  Spatial shape: {n_pulse_tiles} pulse tiles × {n_range_tiles} range tiles")
        else:
            print(f"  Data format: 1D (sequential CPIs only)")
            n_pulse_tiles = None
            n_range_tiles = None

    # Apply knee bound
    knee_bounded = np.minimum(knee_indices, args.knee_bound)

    # Statistics (use total tiles for spatial data)
    total_samples = knee_indices.size
    n_clean = np.sum(knee_indices == 0)
    n_rfi = np.sum(knee_indices > 0)
    n_recoverable = np.sum((knee_indices > 0) & (knee_indices <= args.knee_bound))
    n_severe = np.sum(knee_indices > args.knee_bound)

    print(f"\n=== Prediction Statistics ===")
    if is_spatial:
        print(f"Total tiles: {total_samples} ({n_pulse_tiles} pulse × {n_range_tiles} range)")
    else:
        print(f"Total CPIs: {total_samples}")
    print(f"Clean (knee=0): {n_clean} ({100*n_clean/total_samples:.1f}%)")
    print(f"RFI detected (knee>0): {n_rfi} ({100*n_rfi/total_samples:.1f}%)")
    print(f"  Recoverable (knee≤{args.knee_bound}): {n_recoverable} ({100*n_recoverable/total_samples:.1f}%)")
    print(f"  Severe (knee>{args.knee_bound}): {n_severe} ({100*n_severe/total_samples:.1f}%)")
    print(f"Mean confidence: {np.mean(confidences):.3f}")

    # Plot 1: Histogram of knee predictions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Original predictions
    bins = np.arange(-0.5, 17.5, 1)
    ax1.hist(knee_indices, bins=bins, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Knee Index', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title(f'Original Predictions: {freq}-{pol}\n{n_clean} clean, {n_rfi} RFI', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(args.knee_bound, color='red', linestyle='--', linewidth=2,
                label=f'Threshold={args.knee_bound}')
    ax1.legend()

    # Bounded predictions
    ax2.hist(knee_bounded, bins=bins, edgecolor='black', alpha=0.7, color='green')
    ax2.set_xlabel('Knee Index', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title(f'Bounded Predictions (knee≤{args.knee_bound})\n'
                  f'{n_clean} clean, {n_recoverable} recoverable, {n_severe} severe',
                  fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    hist_path = output_dir / f'knee_histogram_{freq}_{pol}.png'
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {hist_path}")
    plt.close()

    # Plot 2: Confidence distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(confidences, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Prediction Confidence', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'Prediction Confidence Distribution: {freq}-{pol}\n'
                 f'Mean: {np.mean(confidences):.3f}, Median: {np.median(confidences):.3f}',
                 fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axvline(np.mean(confidences), color='red', linestyle='--', linewidth=2,
               label=f'Mean={np.mean(confidences):.3f}')
    ax.legend()
    plt.tight_layout()
    conf_path = output_dir / f'confidence_distribution_{freq}_{pol}.png'
    plt.savefig(conf_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {conf_path}")
    plt.close()

    # Plot 3: Eigenvalue spectrum for sample CPIs
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # For spatial data, flatten for sampling; for 1D, use directly
    knee_flat = knee_indices.flatten()
    conf_flat = confidences.flatten()

    # Select 6 representative CPIs: 3 clean, 3 with RFI
    clean_indices = np.where(knee_flat == 0)[0]
    rfi_indices = np.where(knee_flat > 0)[0]

    sample_clean = np.random.choice(clean_indices, size=min(3, len(clean_indices)), replace=False)
    sample_rfi = np.random.choice(rfi_indices, size=min(3, len(rfi_indices)), replace=False)
    sample_indices = np.concatenate([sample_clean, sample_rfi])

    # For eigenvalues, we need to map back to CPI indices (not tile indices)
    # If spatial: each CPI has multiple range tiles, so divide by n_range_tiles
    if is_spatial:
        # Map flat tile index to CPI index (multiple tiles per CPI)
        cpi_indices = sample_indices // n_range_tiles
    else:
        cpi_indices = sample_indices

    for idx, (tile_idx, cpi_idx) in enumerate(zip(sample_indices, cpi_indices)):
        ax = axes[idx]
        eigvals = eigenvalues[cpi_idx, :]
        eigvals_db = 10 * np.log10(eigvals / eigvals[0] + 1e-12)

        knee = knee_flat[tile_idx]
        conf = conf_flat[tile_idx]

        ax.plot(range(1, cpi_len + 1), eigvals_db, 'o-', linewidth=2, markersize=6)
        if knee > 0:
            ax.axvline(knee, color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax.set_title(f'Tile {tile_idx}: knee={knee} (RFI)\nconf={conf:.3f}', fontsize=10)
        else:
            ax.set_title(f'Tile {tile_idx}: clean\nconf={conf:.3f}', fontsize=10)

        ax.set_xlabel('Eigenvalue Index', fontsize=9)
        ax.set_ylabel('Normalized Eigenvalue (dB)', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    eigval_path = output_dir / f'eigenvalue_samples_{freq}_{pol}.png'
    plt.savefig(eigval_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {eigval_path}")
    plt.close()

    # Plot 4: Time series of predictions (flatten if spatial)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Use flattened data for time series
    knee_plot = knee_flat if is_spatial else knee_indices
    conf_plot = conf_flat if is_spatial else confidences

    # Knee predictions over time
    ax1.plot(knee_plot, 'o', markersize=2, alpha=0.5)
    ax1.axhline(args.knee_bound, color='red', linestyle='--', linewidth=2, label=f'Threshold={args.knee_bound}')
    ax1.set_ylabel('Knee Index', fontsize=11)
    ax1.set_title(f'RFI Predictions Over Time: {freq}-{pol}', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Confidence over time
    ax2.plot(conf_plot, 'o', markersize=2, alpha=0.5, color='green')
    ax2.axhline(np.mean(conf_plot), color='red', linestyle='--', linewidth=2,
                label=f'Mean={np.mean(conf_plot):.3f}')
    ax2.set_xlabel('Tile Index' if is_spatial else 'CPI Index', fontsize=11)
    ax2.set_ylabel('Confidence', fontsize=11)
    ax2.set_title('Prediction Confidence Over Time', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    time_path = output_dir / f'time_series_{freq}_{pol}.png'
    plt.savefig(time_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {time_path}")
    plt.close()

    # Plot 5: Spatial maps (if data is 2D)
    if is_spatial:
        print(f"\nGenerating spatial maps...")

        # Create colormap (matplotlib 3.5+ compatible)
        n_colors = args.knee_bound + 1
        try:
            cmap = plt.colormaps['RdYlGn_r'].resampled(n_colors)
        except AttributeError:
            # Fallback for older matplotlib versions
            cmap = plt.cm.get_cmap('RdYlGn_r', n_colors)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Original predictions (swap axes: pulse on Y, range on X)
        im1 = ax1.imshow(knee_indices, aspect='auto', cmap=cmap,
                         vmin=0, vmax=args.knee_bound, origin='lower')
        ax1.set_xlabel('Range Tile Index', fontsize=11)
        ax1.set_ylabel('Pulse Tile Index', fontsize=11)
        ax1.set_title(f'Original RFI Predictions: {freq}-{pol}\n'
                      f'{n_clean} clean, {n_rfi} RFI detected',
                      fontsize=12)
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Knee Index', fontsize=11)

        # Bounded predictions
        im2 = ax2.imshow(knee_bounded, aspect='auto', cmap=cmap,
                         vmin=0, vmax=args.knee_bound, origin='lower')
        ax2.set_xlabel('Range Tile Index', fontsize=11)
        ax2.set_ylabel('Pulse Tile Index', fontsize=11)
        ax2.set_title(f'Bounded Predictions (knee≤{args.knee_bound}): {freq}-{pol}\n'
                      f'{n_recoverable} recoverable, {n_severe} severe',
                      fontsize=12)
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Knee Index', fontsize=11)

        plt.tight_layout()
        spatial_path = output_dir / f'spatial_map_{freq}_{pol}.png'
        plt.savefig(spatial_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {spatial_path}")
        plt.close()

        # Plot 6: Confidence spatial map
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(confidences, aspect='auto', cmap='viridis',
                       vmin=0, vmax=1, origin='lower')
        ax.set_xlabel('Range Tile Index', fontsize=11)
        ax.set_ylabel('Pulse Tile Index', fontsize=11)
        ax.set_title(f'Prediction Confidence Map: {freq}-{pol}\n'
                     f'Mean: {np.mean(confidences):.3f}',
                     fontsize=12)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Confidence', fontsize=11)

        plt.tight_layout()
        conf_spatial_path = output_dir / f'confidence_spatial_map_{freq}_{pol}.png'
        plt.savefig(conf_spatial_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {conf_spatial_path}")
        plt.close()

    print(f"\n=== All plots saved to: {output_dir} ===\n")


if __name__ == '__main__':
    main()
