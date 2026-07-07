"""
inspect_cpi_tiles.py

Inspect specific CPI tiles from NISAR data, compute their eigenvalue profiles,
extract global features used in train.py, and save plots + parameters to JSON.

This script takes CPI tile indices (in tile coordinates) and computes:
1. True global position (pulse, range) in the original NISAR dataset
2. Eigenvalue profile from the SCM
3. Global parameters used in train.py: condition_number, sigma_min, sigma_max, mu_min, f_factor

Usage:
    python inspect_cpi_tiles.py nisar.h5 \\
        --dataset /science/LSAR/RRSD/swaths/frequencyA/txH/rxV/HV \\
        --tiles "mountains:189,210" "urban_rfi:80,4577" "urban_clean:109,4635" \\
        --output-dir full_nisar_streaming/los_angeles/eigenvalue_inspection
"""

import os
import sys
import numpy as np
import h5py
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import constants and utilities from existing scripts
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from process_nisar_streaming import (
    CPI_HEIGHT, CPI_WIDTH,
    H5DataAccessor,
    compute_scm_and_eigenvalues,
    extract_model_features
)


def parse_tile_spec(tile_spec):
    """
    Parse a tile specification string like "mountains:189,210".

    Args:
        tile_spec (str): Format "name:pulse_tile_idx,range_tile_idx"

    Returns:
        tuple: (name, pulse_tile_idx, range_tile_idx)
    """
    parts = tile_spec.split(':')
    if len(parts) != 2:
        raise ValueError(f"Invalid tile spec: {tile_spec}. Expected format 'name:pulse_idx,range_idx'")

    name = parts[0]
    coords = parts[1].split(',')
    if len(coords) != 2:
        raise ValueError(f"Invalid coordinates in tile spec: {tile_spec}")

    pulse_tile_idx = int(coords[0])
    range_tile_idx = int(coords[1])

    return name, pulse_tile_idx, range_tile_idx


def tile_to_global_position(pulse_tile_idx, range_tile_idx, cpi_height=16, cpi_width=32):
    """
    Convert CPI tile indices to global pulse/range positions.

    Args:
        pulse_tile_idx (int): Pulse tile index (0-based)
        range_tile_idx (int): Range tile index (0-based)
        cpi_height (int): CPI height in pulses
        cpi_width (int): CPI width in range bins

    Returns:
        tuple: (pulse_start, pulse_end, range_start, range_end)
    """
    pulse_start = pulse_tile_idx * cpi_height
    pulse_end = pulse_start + cpi_height
    range_start = range_tile_idx * cpi_width
    range_end = range_start + cpi_width

    return pulse_start, pulse_end, range_start, range_end


def compute_train_global_features(cpi, eigvals_normalized, M):
    """
    Compute the exact global features used in train.py.

    This replicates the global feature extraction from train.py:extract_features()
    using NORMALIZED eigenvalues for scale invariance.

    Args:
        cpi (np.ndarray): Complex CPI tile, shape (M, K)
        eigvals_normalized (np.ndarray): Normalized eigenvalues (scaled by max eigenvalue)
        M (int): Number of pulses (CPI height)

    Returns:
        dict: Dictionary with all global parameters
    """
    K = cpi.shape[1]

    # Compute SCM
    R = (cpi @ cpi.conj().T) / K

    # Get diagonal of SCM
    diag = np.real(np.diag(R))

    # Get max eigenvalue for normalization (from original non-normalized eigenvalues)
    eigvals_original = np.linalg.eigvalsh(R)
    max_eigval = np.max(eigvals_original)
    diag_normalized = diag / max(max_eigval, 1e-12)

    # Split diagonal into top and bottom halves
    half = max(M // 2, 1)
    sigma_max = float(np.std(diag_normalized[:half]))
    sigma_min = float(np.std(diag_normalized[half:]))
    mu_min = float(np.mean(diag_normalized[half:]))

    # Condition number (ratio, scale-invariant)
    cond_number = float(eigvals_normalized[0] / max(eigvals_normalized[-1], 1e-12))

    # F-factor (ratio, scale-invariant)
    eps = 1e-6
    f_factor = float(sigma_max / (sigma_min + eps))

    return {
        'condition_number': cond_number,
        'sigma_min': sigma_min,
        'sigma_max': sigma_max,
        'mu_min': mu_min,
        'f_factor': f_factor,
    }


def inspect_cpi_tile(data, name, pulse_tile_idx, range_tile_idx, cpi_height, cpi_width, store_cpi=False):
    """
    Inspect a single CPI tile: extract data, compute features, and prepare for plotting.

    Args:
        data: H5DataAccessor
        name (str): Descriptive name for this tile
        pulse_tile_idx (int): Pulse tile index
        range_tile_idx (int): Range tile index
        cpi_height (int): CPI height
        cpi_width (int): CPI width
        store_cpi (bool): If True, store the CPI array in the result (for image plotting)

    Returns:
        dict: Dictionary with all inspection results
    """
    # Compute global position
    pulse_start, pulse_end, range_start, range_end = tile_to_global_position(
        pulse_tile_idx, range_tile_idx, cpi_height, cpi_width
    )

    print(f"\n{name}:")
    print(f"  Tile indices: ({pulse_tile_idx}, {range_tile_idx})")
    print(f"  Global position:")
    print(f"    Pulses: {pulse_start:,} to {pulse_end:,}")
    print(f"    Range:  {range_start:,} to {range_end:,}")

    # Extract CPI
    cpi = data[pulse_start:pulse_end, range_start:range_end]
    print(f"  CPI shape: {cpi.shape}")

    # Compute SCM and eigenvalues
    SCM, eigvals_sorted, eigvals_normalized, max_eigval_db = compute_scm_and_eigenvalues(cpi)
    diagonal = np.diag(SCM).real

    print(f"  Max eigenvalue: {max_eigval_db:.2f} dB")
    print(f"  Eigenvalue range (normalized): [{eigvals_normalized[-1]:.6f}, {eigvals_normalized[0]:.6f}]")

    # Extract model features (eigen, global_)
    eigen, global_features_vector = extract_model_features(cpi, eigvals_normalized)

    # Compute train.py global features with names
    M = cpi_height
    global_features_dict = compute_train_global_features(cpi, eigvals_normalized, M)

    print(f"  Global features:")
    for key, val in global_features_dict.items():
        print(f"    {key}: {val:.6f}")

    # Convert to dB scale
    eigvals_normalized_db = 10 * np.log10(eigvals_normalized + 1e-12)
    eigvals_sorted_db = 10 * np.log10(eigvals_sorted + 1e-12)
    diagonal_db = 10 * np.log10(diagonal + 1e-12)

    # Compute eigenvalue slopes in dB per index
    slopes_db = np.diff(eigvals_normalized_db)

    result = {
        'name': name,
        'tile_indices': {
            'pulse_tile': pulse_tile_idx,
            'range_tile': range_tile_idx,
        },
        'global_position': {
            'pulse_start': pulse_start,
            'pulse_end': pulse_end,
            'range_start': range_start,
            'range_end': range_end,
        },
        'cpi_dimensions': {
            'height': cpi_height,
            'width': cpi_width,
        },
        'eigenvalues': {
            'sorted': eigvals_sorted.tolist(),
            'sorted_db': eigvals_sorted_db.tolist(),
            'normalized': eigvals_normalized.tolist(),
            'normalized_db': eigvals_normalized_db.tolist(),
            'slopes_db': slopes_db.tolist(),
            'max_eigval_db': float(max_eigval_db),
        },
        'global_features': global_features_dict,
        'scm_diagonal': diagonal.tolist(),
        'scm_diagonal_db': diagonal_db.tolist(),
    }

    # Optionally store CPI array (not saved to JSON, only for image plotting)
    if store_cpi:
        result['_cpi_array'] = cpi

    return result


def plot_cpi_image(data, result, output_dir):
    """
    Plot the actual CPI tile as an image showing magnitude and phase.

    Creates a figure with:
    1. Magnitude (power) image in dB
    2. Phase image
    3. Range profile (averaged across pulses)
    4. Pulse profile (averaged across range)

    Args:
        data: H5DataAccessor to load the CPI
        result (dict): Inspection result dictionary for one tile
        output_dir (Path): Output directory for plots
    """
    tile_name = result['name']
    tile_idx = result['tile_indices']
    global_pos = result['global_position']
    cpi_dims = result['cpi_dimensions']

    # Extract CPI data
    pulse_start = global_pos['pulse_start']
    pulse_end = global_pos['pulse_end']
    range_start = global_pos['range_start']
    range_end = global_pos['range_end']

    cpi = data[pulse_start:pulse_end, range_start:range_end]

    # Compute magnitude and phase
    magnitude = np.abs(cpi)
    power_db = 10 * np.log10(magnitude**2 + 1e-12)  # Avoid log(0)
    phase = np.angle(cpi)

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Main title
    fig.suptitle(
        f"CPI Tile Image: {tile_name}\n"
        f"Tile Index: ({tile_idx['pulse_tile']}, {tile_idx['range_tile']}) | "
        f"Global Position: Pulses {pulse_start:,}-{pulse_end:,}, Range {range_start:,}-{range_end:,}",
        fontsize=14, fontweight='bold', y=0.98
    )

    # Plot 1: Power (magnitude) in dB
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(power_db, aspect='auto', cmap='viridis', interpolation='nearest')
    cbar1 = plt.colorbar(im1, ax=ax1, label='Power (dB)')
    ax1.set_xlabel('Range Bin (within CPI)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Pulse (within CPI)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Power (Magnitude²) in dB\nRange: [{power_db.min():.1f}, {power_db.max():.1f}] dB',
                  fontsize=12, fontweight='bold')

    # Plot 2: Phase
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(phase, aspect='auto', cmap='hsv', vmin=-np.pi, vmax=np.pi, interpolation='nearest')
    cbar2 = plt.colorbar(im2, ax=ax2, label='Phase (radians)', ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar2.ax.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax2.set_xlabel('Range Bin (within CPI)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Pulse (within CPI)', fontsize=11, fontweight='bold')
    ax2.set_title('Phase', fontsize=12, fontweight='bold')

    # Plot 3: Range profile (averaged across pulses)
    ax3 = fig.add_subplot(gs[1, 0])
    range_profile = np.mean(magnitude, axis=0)  # Average over pulses
    range_bins = np.arange(cpi_dims['width'])
    ax3.plot(range_bins, range_profile, color='#2ca02c', linewidth=2)
    ax3.set_xlabel('Range Bin (within CPI)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Average Magnitude', fontsize=11, fontweight='bold')
    ax3.set_title('Range Profile (Pulse-Averaged)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')

    # Plot 4: Pulse profile (averaged across range)
    ax4 = fig.add_subplot(gs[1, 1])
    pulse_profile = np.mean(magnitude, axis=1)  # Average over range bins
    pulses = np.arange(cpi_dims['height'])
    ax4.plot(pulses, pulse_profile, color='#d62728', linewidth=2, marker='o', markersize=6)
    ax4.set_xlabel('Pulse (within CPI)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Average Magnitude', fontsize=11, fontweight='bold')
    ax4.set_title('Pulse Profile (Range-Averaged)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')

    # Save plot
    output_path = output_dir / f'{tile_name}_image.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: {output_path}")
    return output_path


def plot_single_tile(result, output_dir):
    """
    Plot all features for a single CPI tile in one comprehensive figure.

    Creates four subplots:
    1. Normalized eigenvalues (descending)
    2. Eigenvalue slopes (finite differences)
    3. SCM diagonal values
    4. Unnormalized eigenvalues (log scale)

    Args:
        result (dict): Inspection result dictionary for one tile
        output_dir (Path): Output directory for plots
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    tile_name = result['name']
    tile_idx = result['tile_indices']
    global_pos = result['global_position']

    # Main title with tile info
    fig.suptitle(
        f"CPI Tile Inspection: {tile_name}\n"
        f"Tile Index: ({tile_idx['pulse_tile']}, {tile_idx['range_tile']}) | "
        f"Global Position: Pulses {global_pos['pulse_start']:,}-{global_pos['pulse_end']:,}, "
        f"Range {global_pos['range_start']:,}-{global_pos['range_end']:,}",
        fontsize=14, fontweight='bold', y=0.98
    )

    # Plot 1: Normalized eigenvalues in dB
    ax1 = fig.add_subplot(gs[0, 0])
    eigvals_db = np.array(result['eigenvalues']['normalized_db'])
    indices = np.arange(1, len(eigvals_db) + 1)
    ax1.plot(indices, eigvals_db, marker='o', color='#1f77b4', linewidth=2.5, markersize=8)
    ax1.set_xlabel('Eigenvalue Index (1-based)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Normalized Eigenvalue (dB)', fontsize=11, fontweight='bold')
    ax1.set_title('Eigenvalue Profile (Normalized, dB)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0.5, len(eigvals_db) + 0.5)
    # Add max eigenvalue info
    max_eig_db = result['eigenvalues']['max_eigval_db']
    ax1.text(0.02, 0.98, f"Max eigenvalue: {max_eig_db:.2f} dB",
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Eigenvalue slopes in dB per index
    ax2 = fig.add_subplot(gs[0, 1])
    slopes_db = np.array(result['eigenvalues']['slopes_db'])
    indices = np.arange(1, len(slopes_db) + 1)
    ax2.plot(indices, slopes_db, marker='s', color='#ff7f0e', linewidth=2.5, markersize=8)
    ax2.set_xlabel('Transition Index (k to k+1)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Eigenvalue Slope (dB per index)', fontsize=11, fontweight='bold')
    ax2.set_title('Eigenvalue Slopes (dB per index)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.0, alpha=0.5)
    # Highlight the steepest drop
    max_drop_idx = np.argmin(slopes_db)
    ax2.plot(max_drop_idx + 1, slopes_db[max_drop_idx], 'r*', markersize=15,
             label=f'Steepest drop at {max_drop_idx + 1}')
    ax2.legend(fontsize=9)

    # Plot 3: SCM diagonal in dB
    ax3 = fig.add_subplot(gs[1, 0])
    diagonal_db = np.array(result['scm_diagonal_db'])
    indices = np.arange(1, len(diagonal_db) + 1)
    ax3.plot(indices, diagonal_db, marker='^', color='#2ca02c', linewidth=2.5, markersize=8)
    ax3.set_xlabel('Pulse Index (within CPI)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('SCM Diagonal Value (dB)', fontsize=11, fontweight='bold')
    ax3.set_title('SCM Diagonal Elements (dB)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')

    # Plot 4: Unnormalized eigenvalues in dB
    ax4 = fig.add_subplot(gs[1, 1])
    eigvals_unnorm_db = np.array(result['eigenvalues']['sorted_db'])
    indices = np.arange(1, len(eigvals_unnorm_db) + 1)
    ax4.plot(indices, eigvals_unnorm_db, marker='D', color='#9467bd', linewidth=2.5, markersize=8)
    ax4.set_xlabel('Eigenvalue Index (1-based)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Eigenvalue (dB)', fontsize=11, fontweight='bold')
    ax4.set_title('Eigenvalue Profile (Unnormalized, dB)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_xlim(0.5, len(eigvals_unnorm_db) + 0.5)

    # Add global features as text annotation
    global_feats = result['global_features']
    info_text = "Global Features (train.py):\n"
    info_text += f"  Cond. #: {global_feats['condition_number']:.2f}\n"
    info_text += f"  σ_min: {global_feats['sigma_min']:.4f}\n"
    info_text += f"  σ_max: {global_feats['sigma_max']:.4f}\n"
    info_text += f"  μ_min: {global_feats['mu_min']:.4f}\n"
    info_text += f"  F-factor: {global_feats['f_factor']:.2f}"

    ax4.text(0.98, 0.02, info_text,
             transform=ax4.transAxes, fontsize=9, verticalalignment='bottom',
             horizontalalignment='right', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # Save plot
    output_path = output_dir / f'{tile_name}_profile.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: {output_path}")
    return output_path


def plot_global_features_comparison(results_list, output_dir):
    """
    Plot bar chart comparing global features across tiles.

    Args:
        results_list (list): List of inspection result dictionaries
        output_dir (Path): Output directory for plots
    """
    feature_names = ['condition_number', 'sigma_min', 'sigma_max', 'mu_min', 'f_factor']
    display_names = ['Condition\nNumber', 'σ_min', 'σ_max', 'μ_min', 'F-factor']

    n_tiles = len(results_list)
    n_features = len(feature_names)

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(n_features)
    width = 0.8 / n_tiles
    colors = plt.cm.tab10(np.linspace(0, 1, n_tiles))

    for i, result in enumerate(results_list):
        values = [result['global_features'][fname] for fname in feature_names]
        offset = (i - n_tiles / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=result['name'], color=colors[i], alpha=0.8, edgecolor='black')

    ax.set_xlabel('Global Feature', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('Global Features Comparison (Used in train.py)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=11)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')

    fig.tight_layout()

    output_path = output_dir / 'global_features_comparison.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved global features comparison: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Inspect CPI tiles and plot eigenvalue profiles with global features',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('nisar_file', help='Input NISAR HDF5 file')
    parser.add_argument('--dataset', required=True,
                        help='HDF5 dataset path (e.g., /science/LSAR/RRSD/swaths/frequencyA/txH/rxV/HV)')
    parser.add_argument('--tiles', nargs='+', required=True,
                        help='Tile specifications in format "name:pulse_tile_idx,range_tile_idx"')
    parser.add_argument('--cpi-height', type=int, default=CPI_HEIGHT,
                        help=f'CPI height in pulses (default: {CPI_HEIGHT})')
    parser.add_argument('--cpi-width', type=int, default=CPI_WIDTH,
                        help=f'CPI width in range bins (default: {CPI_WIDTH})')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for plots and JSON')

    args = parser.parse_args()

    # Validate input file
    nisar_path = Path(args.nisar_file)
    if not nisar_path.exists():
        print(f"ERROR: NISAR file not found: {args.nisar_file}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("CPI Tile Inspection")
    print("="*70)
    print(f"NISAR file:  {nisar_path}")
    print(f"Dataset:     {args.dataset}")
    print(f"CPI dims:    {args.cpi_height} x {args.cpi_width}")
    print(f"Output dir:  {output_dir}")
    print(f"Tiles to inspect: {len(args.tiles)}")

    # Parse tile specifications
    tile_specs = []
    for tile_spec in args.tiles:
        name, pulse_tile_idx, range_tile_idx = parse_tile_spec(tile_spec)
        tile_specs.append((name, pulse_tile_idx, range_tile_idx))
        print(f"  - {name}: tile ({pulse_tile_idx}, {range_tile_idx})")

    # Open NISAR file
    print(f"\nOpening NISAR file and loading dataset...")
    with h5py.File(nisar_path, 'r') as f:
        if args.dataset not in f:
            print(f"ERROR: Dataset not found: {args.dataset}")
            print(f"Available datasets:")
            f.visit(lambda name: print(f"  /{name}") if isinstance(f[name], h5py.Dataset) else None)
            sys.exit(1)

    # Create data accessor (needs to be outside the context manager)
    data = H5DataAccessor(str(nisar_path), args.dataset)
    print(f"Dataset shape: {data.shape}")

    # Inspect each tile (store CPI arrays for image plotting)
    results_list = []
    for name, pulse_tile_idx, range_tile_idx in tile_specs:
        result = inspect_cpi_tile(
            data, name, pulse_tile_idx, range_tile_idx,
            args.cpi_height, args.cpi_width, store_cpi=True
        )
        results_list.append(result)

    # Save all results to JSON (exclude CPI arrays)
    json_path = output_dir / 'tile_inspection_results.json'
    json_results = []
    for result in results_list:
        # Create a copy without the CPI array
        json_result = {k: v for k, v in result.items() if not k.startswith('_')}
        json_results.append(json_result)

    with open(json_path, 'w') as f:
        json.dump({
            'metadata': {
                'nisar_file': str(nisar_path),
                'dataset': args.dataset,
                'cpi_dimensions': {
                    'height': args.cpi_height,
                    'width': args.cpi_width,
                },
                'n_tiles': len(results_list),
            },
            'tiles': json_results,
        }, f, indent=2)

    print(f"\n{'='*70}")
    print("Saved inspection results to JSON:")
    print(f"  {json_path}")

    # Generate plots
    print(f"\n{'='*70}")
    print("Generating individual tile plots...")
    for result in results_list:
        plot_single_tile(result, output_dir)

    print(f"\nGenerating CPI image plots...")
    for result in results_list:
        plot_cpi_image(data, result, output_dir)

    print(f"\nGenerating comparison plot...")
    plot_global_features_comparison(results_list, output_dir)

    print(f"\n{'='*70}")
    print("Inspection Complete!")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"  - tile_inspection_results.json")
    for result in results_list:
        print(f"  - {result['name']}_profile.png (eigenvalue analysis)")
        print(f"  - {result['name']}_image.png (CPI visualization)")
    print(f"  - global_features_comparison.png")


if __name__ == '__main__':
    main()
