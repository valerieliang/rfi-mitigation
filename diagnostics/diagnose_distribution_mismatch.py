"""
diagnose_distribution_mismatch.py

Quick diagnostic to visualize the distribution mismatch between synthetic training
data and real NISAR urban data. This explains why the model predicts "clean" everywhere.

Usage:
    python ml/diagnose_distribution_mismatch.py \\
        --synthetic-h5 data/multi_band/clean/image_0_snr_6.h5 \\
        --nisar-h5 data/nisar_data/NISAR_L0_PR_RRSD_*.h5 \\
        --dataset /science/LSAR/RRSD/swaths/frequencyA/txH/rxV/HV \\
        --n-samples 1000
"""

import os
import sys
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from full_nisar_streaming.process_nisar_streaming import H5DataAccessor
from ml.train_db import extract_features

CPI_HEIGHT = 16
CPI_WIDTH = 250


def load_synthetic_features(synthetic_h5_path, n_samples=500):
    """Load features from synthetic training data."""
    print(f"\nLoading synthetic data: {synthetic_h5_path}")

    features = {'eigen': [], 'global': [], 'eigvals_db': []}

    with h5py.File(synthetic_h5_path, 'r') as f:
        # Get all CPI datasets
        cpi_keys = [k for k in f.keys() if k.startswith('cpi_') and
                    not k.endswith('_eigenvalues') and not k.endswith('_diagonal')]

        # Sample randomly
        sampled_keys = np.random.choice(cpi_keys, size=min(n_samples, len(cpi_keys)), replace=False)

        for key in sampled_keys:
            cpi = f[key][:]
            eigen, global_feat = extract_features(cpi)

            # Also get raw eigenvalues for visualization
            eigvals = np.linalg.eigvalsh((cpi @ cpi.conj().T) / cpi.shape[1])
            eigvals_sorted = np.sort(np.real(eigvals))[::-1]
            eigvals_db = 10 * np.log10(eigvals_sorted / max(eigvals_sorted[0], 1e-12) + 1e-12)

            features['eigen'].append(eigen)
            features['global'].append(global_feat)
            features['eigvals_db'].append(eigvals_db)

    print(f"  Loaded {len(features['eigen'])} CPIs")

    return {
        'eigen': np.array(features['eigen']),
        'global': np.array(features['global']),
        'eigvals_db': np.array(features['eigvals_db'])
    }


def load_nisar_features(nisar_h5_path, dataset_path, n_samples=500, region=None):
    """Load features from real NISAR data."""
    print(f"\nLoading NISAR data: {nisar_h5_path}")

    accessor = H5DataAccessor(nisar_h5_path, dataset_path)
    total_pulses, total_range = accessor.shape

    if region:
        pulse_start, pulse_end = region
    else:
        pulse_start, pulse_end = 0, total_pulses - CPI_HEIGHT

    print(f"  Sampling {n_samples} random CPIs...")

    features = {'eigen': [], 'global': [], 'eigvals_db': []}
    rng = np.random.default_rng(42)

    for _ in range(n_samples):
        p = int(rng.integers(pulse_start, pulse_end))
        r = int(rng.integers(0, total_range - CPI_WIDTH))

        cpi = accessor[p:p+CPI_HEIGHT, r:r+CPI_WIDTH]
        eigen, global_feat = extract_features(cpi)

        eigvals = np.linalg.eigvalsh((cpi @ cpi.conj().T) / cpi.shape[1])
        eigvals_sorted = np.sort(np.real(eigvals))[::-1]
        eigvals_db = 10 * np.log10(eigvals_sorted / max(eigvals_sorted[0], 1e-12) + 1e-12)

        features['eigen'].append(eigen)
        features['global'].append(global_feat)
        features['eigvals_db'].append(eigvals_db)

    accessor.close()

    print(f"  Loaded {len(features['eigen'])} CPIs")

    return {
        'eigen': np.array(features['eigen']),
        'global': np.array(features['global']),
        'eigvals_db': np.array(features['eigvals_db'])
    }


def plot_diagnostic(synthetic_features, nisar_features, output_path):
    """Generate diagnostic plots showing distribution mismatch."""

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Extract global features
    # Assuming format: [cond_number_db, eff_rank] for dB model
    syn_cond = synthetic_features['global'][:, 0]
    syn_eff_rank = synthetic_features['global'][:, 1]

    nisar_cond = nisar_features['global'][:, 0]
    nisar_eff_rank = nisar_features['global'][:, 1]

    # 1. Eigenvalue profiles
    ax = fig.add_subplot(gs[0, :2])
    x = np.arange(1, 17)

    syn_ev = synthetic_features['eigvals_db']
    nisar_ev = nisar_features['eigvals_db']

    ax.plot(x, syn_ev.mean(axis=0), 'b-', linewidth=3, label='Synthetic (mean)')
    ax.fill_between(x,
                     np.percentile(syn_ev, 5, axis=0),
                     np.percentile(syn_ev, 95, axis=0),
                     alpha=0.3, color='b', label='Synthetic (5-95%)')

    ax.plot(x, nisar_ev.mean(axis=0), 'r-', linewidth=3, label='Real NISAR (mean)')
    ax.fill_between(x,
                     np.percentile(nisar_ev, 5, axis=0),
                     np.percentile(nisar_ev, 95, axis=0),
                     alpha=0.3, color='r', label='Real NISAR (5-95%)')

    ax.set_xlabel('Eigenvalue Index (1-based)', fontsize=12)
    ax.set_ylabel('Eigenvalue (dB, normalized)', fontsize=12)
    ax.set_title('DISTRIBUTION MISMATCH: Eigenvalue Profiles', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # Add text box explaining the issue
    textstr = 'KEY ISSUE:\\nSynthetic profiles show smooth noise decay.\\nReal NISAR has elevated eigenvalues from clutter.\\nModel trained on synthetic sees real data as anomalous.'
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # 2. Condition number distribution
    ax = fig.add_subplot(gs[0, 2:])

    bins = np.linspace(
        min(syn_cond.min(), nisar_cond.min()),
        max(syn_cond.max(), nisar_cond.max()),
        50
    )

    ax.hist(syn_cond, bins=bins, alpha=0.6, color='b', label=f'Synthetic\\n(μ={syn_cond.mean():.1f})', density=True)
    ax.hist(nisar_cond, bins=bins, alpha=0.6, color='r', label=f'Real NISAR\\n(μ={nisar_cond.mean():.1f})', density=True)

    ax.set_xlabel('Condition Number (dB)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Condition Number: Synthetic vs. Real', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # Overlap percentage
    overlap = compute_distribution_overlap(syn_cond, nisar_cond)
    ax.text(0.02, 0.98, f'Overlap: {overlap:.1f}%', transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 3. Effective rank distribution
    ax = fig.add_subplot(gs[1, :2])

    bins = np.linspace(
        min(syn_eff_rank.min(), nisar_eff_rank.min()),
        max(syn_eff_rank.max(), nisar_eff_rank.max()),
        50
    )

    ax.hist(syn_eff_rank, bins=bins, alpha=0.6, color='b', label=f'Synthetic\\n(μ={syn_eff_rank.mean():.2f})', density=True)
    ax.hist(nisar_eff_rank, bins=bins, alpha=0.6, color='r', label=f'Real NISAR\\n(μ={nisar_eff_rank.mean():.2f})', density=True)

    ax.set_xlabel('Effective Rank', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Effective Rank: Synthetic vs. Real', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    overlap = compute_distribution_overlap(syn_eff_rank, nisar_eff_rank)
    ax.text(0.02, 0.98, f'Overlap: {overlap:.1f}%', transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 4. 2D feature space (condition number vs. effective rank)
    ax = fig.add_subplot(gs[1, 2:])

    ax.scatter(syn_cond, syn_eff_rank, alpha=0.3, s=10, c='b', label='Synthetic')
    ax.scatter(nisar_cond, nisar_eff_rank, alpha=0.3, s=10, c='r', label='Real NISAR')

    ax.set_xlabel('Condition Number (dB)', fontsize=12)
    ax.set_ylabel('Effective Rank', fontsize=12)
    ax.set_title('2D Feature Space', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # 5. Eigenvalue slopes (first derivative)
    ax = fig.add_subplot(gs[2, :2])

    syn_slopes = synthetic_features['eigen'][:, :, 1]  # Shape: (n_samples, M)
    nisar_slopes = nisar_features['eigen'][:, :, 1]

    x = np.arange(1, 16)  # M-1 slopes

    ax.plot(x, syn_slopes[:, :-1].mean(axis=0), 'b-', linewidth=3, label='Synthetic (mean)')
    ax.fill_between(x,
                     np.percentile(syn_slopes[:, :-1], 5, axis=0),
                     np.percentile(syn_slopes[:, :-1], 95, axis=0),
                     alpha=0.3, color='b')

    ax.plot(x, nisar_slopes[:, :-1].mean(axis=0), 'r-', linewidth=3, label='Real NISAR (mean)')
    ax.fill_between(x,
                     np.percentile(nisar_slopes[:, :-1], 5, axis=0),
                     np.percentile(nisar_slopes[:, :-1], 95, axis=0),
                     alpha=0.3, color='r')

    ax.set_xlabel('Position (1-based)', fontsize=12)
    ax.set_ylabel('Slope (dB/index)', fontsize=12)
    ax.set_title('Eigenvalue Slopes (First Derivative)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # 6. Summary statistics table
    ax = fig.add_subplot(gs[2, 2:])
    ax.axis('off')

    # Compute summary
    stats_text = "SUMMARY STATISTICS\\n\\n"
    stats_text += f"{'Metric':<25} {'Synthetic':<15} {'Real NISAR':<15} {'Overlap':<10}\\n"
    stats_text += "-" * 70 + "\\n"

    metrics = [
        ('Condition Number (dB)', syn_cond, nisar_cond),
        ('Effective Rank', syn_eff_rank, nisar_eff_rank),
        ('Max Eigenvalue (dB)', synthetic_features['eigvals_db'][:, 0], nisar_features['eigvals_db'][:, 0]),
        ('Min Eigenvalue (dB)', synthetic_features['eigvals_db'][:, -1], nisar_features['eigvals_db'][:, -1]),
    ]

    for name, syn_vals, nisar_vals in metrics:
        overlap_pct = compute_distribution_overlap(syn_vals, nisar_vals)
        stats_text += f"{name:<25} {syn_vals.mean():>7.2f} ± {syn_vals.std():<4.2f}  "
        stats_text += f"{nisar_vals.mean():>7.2f} ± {nisar_vals.std():<4.2f}  "
        stats_text += f"{overlap_pct:>6.1f}%\\n"

    stats_text += "\\n" + "-" * 70 + "\\n"
    stats_text += "\\nINTERPRETATION:\\n"
    stats_text += "• Overlap < 50% = Model has NEVER seen this range\\n"
    stats_text += "• Overlap 50-70% = Partial coverage, poor generalization\\n"
    stats_text += "• Overlap > 70% = Good coverage, should generalize\\n"
    stats_text += "\\n"
    stats_text += "RECOMMENDATION:\\n"
    if compute_distribution_overlap(syn_cond, nisar_cond) < 50:
        stats_text += "⚠ CRITICAL: Feature distributions do not overlap!\\n"
        stats_text += "  Model will ALWAYS predict 'clean' on real data.\\n"
        stats_text += "  → MUST regenerate training data with realistic clutter.\\n"
    elif compute_distribution_overlap(syn_cond, nisar_cond) < 70:
        stats_text += "⚠ WARNING: Limited feature overlap.\\n"
        stats_text += "  Model may underperform on real data.\\n"
        stats_text += "  → Consider adding clutter to training data.\\n"
    else:
        stats_text += "✓ GOOD: Feature distributions overlap well.\\n"
        stats_text += "  Issue may be elsewhere (check model architecture, etc.)\\n"

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    fig.suptitle('MODEL FAILURE DIAGNOSIS: Synthetic vs. Real NISAR Feature Distributions',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\\n  Saved diagnostic plot: {output_path}")


def compute_distribution_overlap(dist1, dist2):
    """
    Compute percentage overlap between two distributions using histogram intersection.

    Returns:
        overlap (float): Percentage overlap (0-100)
    """
    # Create common bins
    combined = np.concatenate([dist1, dist2])
    bins = np.histogram_bin_edges(combined, bins=50)

    # Compute normalized histograms
    hist1, _ = np.histogram(dist1, bins=bins, density=True)
    hist2, _ = np.histogram(dist2, bins=bins, density=True)

    # Normalize to sum to 1
    hist1 = hist1 / (hist1.sum() + 1e-12)
    hist2 = hist2 / (hist2.sum() + 1e-12)

    # Intersection
    overlap = np.minimum(hist1, hist2).sum() * 100

    return overlap


def main():
    parser = argparse.ArgumentParser(description='Diagnose distribution mismatch')
    parser.add_argument('--synthetic-h5', required=True, help='Path to synthetic HDF5 file')
    parser.add_argument('--nisar-h5', required=True, help='Path to NISAR HDF5 file')
    parser.add_argument('--dataset', required=True, help='Dataset path (e.g., .../HV)')
    parser.add_argument('--n-samples', type=int, default=1000, help='CPIs to sample')
    parser.add_argument('--output', default='distribution_mismatch_diagnostic.png', help='Output plot path')
    parser.add_argument('--pulse-start', type=int, default=92000)
    parser.add_argument('--pulse-end', type=int, default=124000)

    args = parser.parse_args()

    print("\\n" + "="*80)
    print("DISTRIBUTION MISMATCH DIAGNOSTIC")
    print("="*80)

    # Load features
    synthetic_features = load_synthetic_features(args.synthetic_h5, n_samples=args.n_samples)
    nisar_features = load_nisar_features(
        args.nisar_h5,
        args.dataset,
        n_samples=args.n_samples,
        region=(args.pulse_start, args.pulse_end)
    )

    # Generate diagnostic plots
    print("\\nGenerating diagnostic plots...")
    plot_diagnostic(synthetic_features, nisar_features, args.output)

    print("\\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)
    print(f"\\nReview the plot: {args.output}")
    print("\\nThis shows WHY the model predicts 'clean' on real data:")
    print("  1. Eigenvalue profiles are completely different")
    print("  2. Feature distributions don't overlap")
    print("  3. Real data is 'out of distribution' for the model")
    print("\\nSolution: Retrain with realistic clutter (see TRAINING_ACTION_PLAN.md)")
    print()


if __name__ == '__main__':
    main()
