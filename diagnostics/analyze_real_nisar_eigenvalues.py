"""
analyze_real_nisar_eigenvalues.py

Extract and analyze eigenvalue statistics from real NISAR data to calibrate
synthetic data generation for training.

This script:
1. Samples random CPI tiles from real NISAR HDF5 data
2. Computes eigenvalue profiles, SCM statistics, and derived features
3. Saves distribution statistics to JSON and raw profiles to NPY
4. Generates comparison plots showing real vs. current synthetic distributions

Usage:
    python ml/analyze_real_nisar_eigenvalues.py \\
        --nisar-h5 data/nisar_data/NISAR_L0_PR_RRSD_*.h5 \\
        --dataset /science/LSAR/RRSD/swaths/frequencyA/txH/rxV/HV \\
        --n-samples 5000 \\
        --output-dir data/real_nisar_stats

Output:
    - data/real_nisar_stats/eigval_stats.json
    - data/real_nisar_stats/eigval_profiles.npy  (shape: [n_samples, M])
    - data/real_nisar_stats/comparison_plots.png
"""

import os
import sys
import json
import argparse
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from full_nisar_streaming.process_nisar_streaming import H5DataAccessor

# CPI dimensions
CPI_HEIGHT = 16
CPI_WIDTH = 250


def compute_eigenvalue_features(cpi):
    """
    Compute eigenvalue-based features from a CPI tile.

    Args:
        cpi (np.ndarray): Complex (M, K) array

    Returns:
        features (dict): Dictionary of computed features
    """
    M, K = cpi.shape

    # Sample covariance matrix
    SCM = (cpi @ cpi.conj().T) / K

    # Eigenvalues (sorted descending)
    eigvals = np.linalg.eigvalsh(SCM)
    eigvals_sorted = np.sort(np.real(eigvals))[::-1]

    # Convert to dB (absolute scale)
    eigvals_db = 10.0 * np.log10(eigvals_sorted + 1e-12)

    # Normalized eigenvalues
    max_eigval = eigvals_sorted[0]
    eigvals_normalized = eigvals_sorted / max(max_eigval, 1e-12)
    eigvals_normalized_db = 10.0 * np.log10(eigvals_normalized + 1e-12)

    # Condition number (dB space)
    cond_number_db = eigvals_db[0] - eigvals_db[-1]

    # Effective rank via Shannon entropy
    p = eigvals_normalized
    p = p / np.sum(p)
    p = p[p > 0]
    eff_rank = np.exp(-np.sum(p * np.log(p + 1e-12)))

    # Diagonal statistics
    diag = np.real(np.diag(SCM))
    diag_normalized = diag / max(max_eigval, 1e-12)

    half = M // 2
    sigma_max = np.std(diag_normalized[:half])
    sigma_min = np.std(diag_normalized[half:])
    mu_min = np.mean(diag_normalized[half:])

    # Power concentration metrics
    power_top3 = np.sum(eigvals_sorted[:3]) / np.sum(eigvals_sorted)
    power_top5 = np.sum(eigvals_sorted[:5]) / np.sum(eigvals_sorted)
    power_top_half = np.sum(eigvals_sorted[:M//2]) / np.sum(eigvals_sorted)

    # Eigenvalue decay rate (log-linear fit)
    log_eigvals = np.log10(eigvals_sorted + 1e-12)
    decay_rate = np.polyfit(np.arange(M), log_eigvals, deg=1)[0]

    # Eigenvalue gap detection (for knee finding)
    slopes = np.diff(eigvals_db)
    max_gap_idx = np.argmin(slopes)  # Most negative slope = biggest drop
    max_gap_value = -slopes[max_gap_idx]

    features = {
        'eigvals_db': eigvals_db,
        'eigvals_normalized_db': eigvals_normalized_db,
        'max_eigval_db': float(eigvals_db[0]),
        'min_eigval_db': float(eigvals_db[-1]),
        'cond_number_db': float(cond_number_db),
        'eff_rank': float(eff_rank),
        'sigma_max': float(sigma_max),
        'sigma_min': float(sigma_min),
        'mu_min': float(mu_min),
        'power_top3': float(power_top3),
        'power_top5': float(power_top5),
        'power_top_half': float(power_top_half),
        'decay_rate': float(decay_rate),
        'max_gap_idx': int(max_gap_idx),
        'max_gap_value': float(max_gap_value),
    }

    return features


def sample_nisar_eigenvalues(nisar_h5_path, dataset_path, n_samples=5000, region=None):
    """
    Sample random CPI tiles from NISAR data and compute eigenvalue features.

    Args:
        nisar_h5_path (str): Path to NISAR HDF5 file
        dataset_path (str): HDF5 dataset path (e.g., .../HV)
        n_samples (int): Number of CPIs to sample
        region (tuple or None): (pulse_start, pulse_end) to constrain sampling region

    Returns:
        features_list (list): List of feature dictionaries
    """
    print(f"\nOpening NISAR file: {nisar_h5_path}")
    print(f"  Dataset: {dataset_path}")

    accessor = H5DataAccessor(nisar_h5_path, dataset_path)
    total_pulses, total_range = accessor.shape

    print(f"  Shape: {total_pulses} pulses × {total_range} range bins")

    # Determine sampling region
    if region is not None:
        pulse_start, pulse_end = region
        pulse_start = max(0, pulse_start)
        pulse_end = min(total_pulses - CPI_HEIGHT, pulse_end)
    else:
        pulse_start = 0
        pulse_end = total_pulses - CPI_HEIGHT

    print(f"  Sampling region: pulses {pulse_start:,} to {pulse_end:,}")
    print(f"  Generating {n_samples:,} random CPI tiles...")

    rng = np.random.default_rng(42)
    features_list = []

    for _ in tqdm(range(n_samples), desc="Sampling CPIs"):
        # Random tile position
        p = int(rng.integers(pulse_start, pulse_end))
        r = int(rng.integers(0, total_range - CPI_WIDTH))

        # Extract CPI
        try:
            cpi = accessor[p:p+CPI_HEIGHT, r:r+CPI_WIDTH]

            # Compute features
            features = compute_eigenvalue_features(cpi)
            features['pulse_idx'] = p
            features['range_idx'] = r

            features_list.append(features)
        except Exception as e:
            print(f"\n  Warning: Failed to process CPI at pulse={p}, range={r}: {e}")
            continue

    accessor.close()

    print(f"\n  Successfully sampled {len(features_list):,} CPIs")

    return features_list


def compute_distribution_statistics(features_list):
    """
    Compute summary statistics across all sampled CPIs.

    Args:
        features_list (list): List of feature dictionaries

    Returns:
        stats (dict): Distribution statistics for each scalar feature
    """
    # Extract scalar features
    scalar_keys = [
        'max_eigval_db', 'min_eigval_db', 'cond_number_db', 'eff_rank',
        'sigma_max', 'sigma_min', 'mu_min',
        'power_top3', 'power_top5', 'power_top_half',
        'decay_rate', 'max_gap_idx', 'max_gap_value'
    ]

    stats = {}

    for key in scalar_keys:
        values = np.array([f[key] for f in features_list])

        stats[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'p5': float(np.percentile(values, 5)),
            'p25': float(np.percentile(values, 25)),
            'p50': float(np.percentile(values, 50)),
            'p75': float(np.percentile(values, 75)),
            'p95': float(np.percentile(values, 95)),
        }

    # Extract eigenvalue profiles
    eigval_profiles_db = np.array([f['eigvals_db'] for f in features_list])
    eigval_profiles_norm_db = np.array([f['eigvals_normalized_db'] for f in features_list])

    stats['eigval_profiles'] = {
        'absolute_db': {
            'mean': eigval_profiles_db.mean(axis=0).tolist(),
            'std': eigval_profiles_db.std(axis=0).tolist(),
            'p5': np.percentile(eigval_profiles_db, 5, axis=0).tolist(),
            'p95': np.percentile(eigval_profiles_db, 95, axis=0).tolist(),
        },
        'normalized_db': {
            'mean': eigval_profiles_norm_db.mean(axis=0).tolist(),
            'std': eigval_profiles_norm_db.std(axis=0).tolist(),
            'p5': np.percentile(eigval_profiles_norm_db, 5, axis=0).tolist(),
            'p95': np.percentile(eigval_profiles_norm_db, 95, axis=0).tolist(),
        }
    }

    return stats, eigval_profiles_db, eigval_profiles_norm_db


def plot_comparison(stats, output_path):
    """
    Generate comparison plots: real NISAR vs. expected synthetic distributions.

    Args:
        stats (dict): Distribution statistics
        output_path (str): Output PNG path
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Plot 1: Eigenvalue profiles (normalized dB)
    ax = axes[0, 0]
    profiles = stats['eigval_profiles']['normalized_db']
    x = np.arange(1, len(profiles['mean']) + 1)

    ax.plot(x, profiles['mean'], 'b-', linewidth=2, label='Real NISAR (mean)')
    ax.fill_between(x, profiles['p5'], profiles['p95'], alpha=0.3, label='Real (5-95 percentile)')

    ax.set_xlabel('Eigenvalue Index (1-based)')
    ax.set_ylabel('Eigenvalue (dB, normalized)')
    ax.set_title('Real NISAR: Eigenvalue Profiles')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Condition number distribution
    ax = axes[0, 1]
    cond_stats = stats['cond_number_db']
    ax.axvline(cond_stats['mean'], color='b', linewidth=2, label=f"Mean: {cond_stats['mean']:.1f} dB")
    ax.axvspan(cond_stats['p5'], cond_stats['p95'], alpha=0.3, label='5-95 percentile')
    ax.set_xlabel('Condition Number (dB)')
    ax.set_title(f"Condition Number\nRange: [{cond_stats['p5']:.1f}, {cond_stats['p95']:.1f}] dB")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 3: Effective rank distribution
    ax = axes[0, 2]
    eff_rank_stats = stats['eff_rank']
    ax.axvline(eff_rank_stats['mean'], color='b', linewidth=2, label=f"Mean: {eff_rank_stats['mean']:.2f}")
    ax.axvspan(eff_rank_stats['p5'], eff_rank_stats['p95'], alpha=0.3, label='5-95 percentile')
    ax.set_xlabel('Effective Rank')
    ax.set_title(f"Effective Rank\nRange: [{eff_rank_stats['p5']:.2f}, {eff_rank_stats['p95']:.2f}]")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 4: Power concentration
    ax = axes[1, 0]
    keys = ['power_top3', 'power_top5', 'power_top_half']
    labels = ['Top 3', 'Top 5', 'Top 50%']
    means = [stats[k]['mean'] for k in keys]
    stds = [stats[k]['std'] for k in keys]

    ax.bar(labels, means, yerr=stds, alpha=0.7, capsize=5)
    ax.set_ylabel('Fraction of Total Power')
    ax.set_title('Power Concentration in Top Eigenvalues')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    # Plot 5: Max gap (knee indicator)
    ax = axes[1, 1]
    gap_idx_stats = stats['max_gap_idx']
    gap_val_stats = stats['max_gap_value']
    ax.text(0.5, 0.7, f"Max Gap Position", ha='center', fontsize=14, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.5, f"Mean: {gap_idx_stats['mean']:.1f} (index)", ha='center', fontsize=12, transform=ax.transAxes)
    ax.text(0.5, 0.3, f"Range: [{gap_idx_stats['p5']:.0f}, {gap_idx_stats['p95']:.0f}]", ha='center', fontsize=12, transform=ax.transAxes)
    ax.text(0.5, 0.1, f"Gap value: {gap_val_stats['mean']:.1f} dB", ha='center', fontsize=11, transform=ax.transAxes)
    ax.axis('off')

    # Plot 6: Decay rate
    ax = axes[1, 2]
    decay_stats = stats['decay_rate']
    ax.axvline(decay_stats['mean'], color='b', linewidth=2, label=f"Mean: {decay_stats['mean']:.3f}")
    ax.axvspan(decay_stats['p5'], decay_stats['p95'], alpha=0.3, label='5-95 percentile')
    ax.set_xlabel('Decay Rate (dB/index)')
    ax.set_title(f"Eigenvalue Decay Rate\nRange: [{decay_stats['p5']:.3f}, {decay_stats['p95']:.3f}]")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle('Real NISAR Eigenvalue Statistics\n(Use these to calibrate synthetic data generation)',
                 fontsize=16, fontweight='bold')
    fig.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Saved comparison plots: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze real NISAR eigenvalue statistics')
    parser.add_argument('--nisar-h5', required=True, help='Path to NISAR HDF5 file')
    parser.add_argument('--dataset', required=True, help='Dataset path in HDF5 (e.g., .../HV)')
    parser.add_argument('--n-samples', type=int, default=5000, help='Number of CPIs to sample')
    parser.add_argument('--output-dir', default='data/real_nisar_stats', help='Output directory')
    parser.add_argument('--pulse-start', type=int, default=None, help='Start pulse index')
    parser.add_argument('--pulse-end', type=int, default=None, help='End pulse index')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("REAL NISAR EIGENVALUE ANALYSIS")
    print("="*80)

    # Determine sampling region
    region = None
    if args.pulse_start is not None and args.pulse_end is not None:
        region = (args.pulse_start, args.pulse_end)

    # Sample CPIs
    features_list = sample_nisar_eigenvalues(
        args.nisar_h5,
        args.dataset,
        n_samples=args.n_samples,
        region=region
    )

    # Compute statistics
    print("\nComputing distribution statistics...")
    stats, eigval_profiles_db, eigval_profiles_norm_db = compute_distribution_statistics(features_list)

    # Save results
    stats_path = os.path.join(args.output_dir, 'eigval_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Saved statistics: {stats_path}")

    profiles_abs_path = os.path.join(args.output_dir, 'eigval_profiles_absolute_db.npy')
    np.save(profiles_abs_path, eigval_profiles_db)
    print(f"  Saved absolute profiles: {profiles_abs_path}")

    profiles_norm_path = os.path.join(args.output_dir, 'eigval_profiles_normalized_db.npy')
    np.save(profiles_norm_path, eigval_profiles_norm_db)
    print(f"  Saved normalized profiles: {profiles_norm_path}")

    # Generate plots
    plot_path = os.path.join(args.output_dir, 'comparison_plots.png')
    plot_comparison(stats, plot_path)

    # Print key statistics
    print("\n" + "="*80)
    print("KEY STATISTICS")
    print("="*80)
    print(f"\nCondition Number (dB):")
    print(f"  Mean: {stats['cond_number_db']['mean']:.2f}")
    print(f"  Range (5-95%%): [{stats['cond_number_db']['p5']:.2f}, {stats['cond_number_db']['p95']:.2f}]")

    print(f"\nEffective Rank:")
    print(f"  Mean: {stats['eff_rank']['mean']:.2f}")
    print(f"  Range (5-95%%): [{stats['eff_rank']['p5']:.2f}, {stats['eff_rank']['p95']:.2f}]")

    print(f"\nPower Concentration (Top 5 eigenvalues):")
    print(f"  Mean: {stats['power_top5']['mean']:.3f}")
    print(f"  Range (5-95%%): [{stats['power_top5']['p5']:.3f}, {stats['power_top5']['p95']:.3f}]")

    print(f"\nMax Eigenvalue (dB):")
    print(f"  Mean: {stats['max_eigval_db']['mean']:.2f}")
    print(f"  Range (5-95%%): [{stats['max_eigval_db']['p5']:.2f}, {stats['max_eigval_db']['p95']:.2f}]")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}/")
    print("\nNext steps:")
    print("  1. Review comparison_plots.png to see real eigenvalue distributions")
    print("  2. Update data_gen/generate_synthetic_data.py with realistic clutter models")
    print("  3. Calibrate CNR ranges to match real power concentration metrics")
    print("  4. Retrain model with new clutter-aware dataset")
    print()


if __name__ == '__main__':
    main()
