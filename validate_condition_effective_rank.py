"""
validate_condition_effective_rank.py

Validation script to cross-check condition number and effective rank computations
from train_db.py against eigenvalue profiles from the synthetic data generator.

This script:
1. Loads synthetic data HDF5 files
2. Extracts eigenvalues, condition number, and effective rank for each CPI
3. Compares computed values against stored eigenvalues
4. Generates validation plots showing:
   - Eigenvalue profiles with condition number annotations
   - Effective rank distributions by RFI count
   - Condition number vs RFI count correlations
   - Verification that condition number and effective rank match eigenvalue profiles

The goal is to verify that the global features (condition_number_db, eff_rank)
used in train_db.py correctly represent the eigenvalue spectrum characteristics.
"""

import os
import sys
import json
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_synthetic_data import BLOCK_HEIGHT, MAX_BANDS, JNR_RANGE_DB


# ---------------------------------------------------------------------------
# FEATURE EXTRACTION (mirrored from train_db.py)
# ---------------------------------------------------------------------------

def compute_eigenvalues(cpi):
    """
    Compute eigenvalues from the sample covariance matrix of a CPI tile.

    Args:
        cpi (np.ndarray): Complex array shape (M, K).

    Returns:
        eigvals (np.ndarray): Real eigenvalues, descending, shape (M,).
    """
    M, K = cpi.shape
    R = (cpi @ cpi.conj().T) / K
    eigvals = np.linalg.eigvalsh(R)
    eigvals = np.sort(np.real(eigvals))[::-1]
    return eigvals


def extract_features(cpi):
    """
    Extract condition number and effective rank from one CPI tile.
    This mirrors the train_db.py feature extraction logic.

    Args:
        cpi (np.ndarray): Complex array shape (M, K).

    Returns:
        eigvals_db (np.ndarray): shape (M,) -- dB eigenvalues
        condition_number_db (float): lambda_max - lambda_min in dB
        eff_rank (float): effective rank via Shannon entropy
    """
    eigvals = compute_eigenvalues(cpi)

    # Normalize by max eigenvalue, then convert to dB
    eigvals_db = 10 * np.log10(eigvals / max(eigvals[0], 1e-12) + 1e-12)

    # Condition number: difference in dB space (equivalent to ratio in linear space)
    cond_number_db = eigvals_db[0] - max(eigvals_db[-1], -100)

    # Effective rank via Shannon entropy of eigenvalue distribution
    p = np.maximum(eigvals, 1e-12)
    p = p / np.sum(p)
    p = p[p > 0]
    eff_rank = np.exp(-np.sum(p * np.log(p)))

    return eigvals_db, cond_number_db, eff_rank


# ---------------------------------------------------------------------------
# DATA LOADING AND VALIDATION
# ---------------------------------------------------------------------------

def load_and_validate_h5(h5_path):
    """
    Load all CPI tiles from an HDF5 file and extract validation data.

    Args:
        h5_path (str): Path to HDF5 file.

    Returns:
        validation_data (list[dict]): List of dicts, one per CPI tile:
            {
                'cpi_key': str,
                'knee': int,
                'pulse_positions': list[int],
                'jnr_db_list': list[int],
                'eigvals_computed': np.ndarray,
                'eigvals_stored': np.ndarray,
                'eigvals_db': np.ndarray,
                'condition_number_db': float,
                'eff_rank': float,
            }
    """
    validation_data = []

    with h5py.File(h5_path, 'r') as f:
        snr_db = f.attrs['snr_db']
        is_clean = f.attrs.get('is_clean', False)

        # Iterate over all CPI datasets
        for key in f.keys():
            if key.endswith('_eigenvalues') or key.endswith('_diagonal'):
                continue

            cpi = f[key][:]

            # Load stored eigenvalues
            eigvals_stored_key = f'{key}_eigenvalues'
            if eigvals_stored_key in f:
                eigvals_stored = f[eigvals_stored_key][:]
            else:
                eigvals_stored = None

            # Compute eigenvalues
            eigvals_computed = compute_eigenvalues(cpi)

            # Extract features
            eigvals_db, cond_number_db, eff_rank = extract_features(cpi)

            # Parse RFI metadata
            payload = json.loads(str(f[key].attrs['rfi_bands']))
            knee = payload['knee']
            pulse_positions = payload['pulse_positions']
            jnr_db_list = payload['jnr_db_list']

            # Count distinct pulse positions for knee validation
            n_distinct = len(set(pulse_positions)) if knee > 0 else 0

            validation_data.append({
                'cpi_key': key,
                'snr_db': snr_db,
                'is_clean': is_clean,
                'knee': knee,
                'n_distinct_pulses': n_distinct,
                'pulse_positions': pulse_positions,
                'jnr_db_list': jnr_db_list,
                'eigvals_computed': eigvals_computed,
                'eigvals_stored': eigvals_stored,
                'eigvals_db': eigvals_db,
                'condition_number_db': cond_number_db,
                'eff_rank': eff_rank,
            })

    return validation_data


# ---------------------------------------------------------------------------
# VALIDATION PLOTS
# ---------------------------------------------------------------------------

def plot_eigenvalue_with_metrics(validation_data, out_path):
    """
    Plot eigenvalue profiles in dB scale, color-coded by max RFI power.

    Shows all CPI tiles from the validation dataset overlaid on one plot.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Determine max JNR for color mapping
    max_jnr = JNR_RANGE_DB[1]
    norm_jnr = mcolors.Normalize(vmin=0, vmax=max_jnr)
    cmap = cm.plasma

    ev_index_1based = np.arange(1, BLOCK_HEIGHT + 1)

    for data in validation_data:
        jnr_db_list = data['jnr_db_list']
        max_jnr_power = max(jnr_db_list) if len(jnr_db_list) > 0 else 0

        # Compute eigenvalues in dB: 10*log10(abs(eigvals))
        eigvals_linear = data['eigvals_computed']
        eigvals_db_plot = 10 * np.log10(np.abs(eigvals_linear) + 1e-12)

        color = cmap(norm_jnr(max_jnr_power))
        ax.plot(ev_index_1based, eigvals_db_plot,
                color=color, alpha=0.4, linewidth=0.8)

    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm_jnr)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Max RFI Power (JNR dB, 0=clean)', fontsize=10)

    ax.set_xlabel('Eigenvalue Index (1-based)', fontsize=11)
    ax.set_ylabel('Eigenvalue (dB)', fontsize=11)
    ax.set_title('Eigenvalue Profiles in dB Scale',
                 fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_condition_number_distribution(validation_data, out_path):
    """
    Plot condition number distribution grouped by max RFI power.
    """
    # Group by max RFI power, with separate category for clean
    power_groups = defaultdict(list)
    for data in validation_data:
        jnr_db_list = data['jnr_db_list']
        if len(jnr_db_list) == 0:
            # Clean data
            max_jnr = -1  # Special marker for clean
        else:
            max_jnr = max(jnr_db_list)

        cond_num = data['condition_number_db']
        power_groups[max_jnr].append(cond_num)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Box plot
    powers = sorted(power_groups.keys())
    cond_nums = [power_groups[p] for p in powers]
    labels = ['clean' if p < 0 else f'{int(p)} dB' for p in powers]

    positions = list(range(1, len(powers) + 1))
    bp = ax.boxplot(cond_nums, positions=positions, patch_artist=True, widths=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # Color boxes by RFI power
    max_jnr = JNR_RANGE_DB[1]
    norm_jnr = mcolors.Normalize(vmin=0, vmax=max_jnr)
    cmap = cm.plasma
    for patch, power in zip(bp['boxes'], powers):
        color_val = 0 if power < 0 else power
        patch.set_facecolor(cmap(norm_jnr(color_val)))
        patch.set_alpha(0.6)

    ax.set_xlabel('Max RFI Power (JNR dB)', fontsize=11)
    ax.set_ylabel('Condition Number (dB)', fontsize=11)
    ax.set_title('Condition Number Distribution by Max RFI Power',
                 fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_effective_rank_distribution(validation_data, out_path):
    """
    Plot effective rank distribution grouped by max RFI power.
    """
    # Group by max RFI power, with separate category for clean
    power_groups = defaultdict(list)
    for data in validation_data:
        jnr_db_list = data['jnr_db_list']
        if len(jnr_db_list) == 0:
            # Clean data
            max_jnr = -1  # Special marker for clean
        else:
            max_jnr = max(jnr_db_list)

        eff_rank = data['eff_rank']
        power_groups[max_jnr].append(eff_rank)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Box plot
    powers = sorted(power_groups.keys())
    eff_ranks = [power_groups[p] for p in powers]
    labels = ['clean' if p < 0 else f'{int(p)} dB' for p in powers]

    positions = list(range(1, len(powers) + 1))
    bp = ax.boxplot(eff_ranks, positions=positions, patch_artist=True, widths=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # Color boxes by RFI power
    max_jnr = JNR_RANGE_DB[1]
    norm_jnr = mcolors.Normalize(vmin=0, vmax=max_jnr)
    cmap = cm.plasma
    for patch, power in zip(bp['boxes'], powers):
        color_val = 0 if power < 0 else power
        patch.set_facecolor(cmap(norm_jnr(color_val)))
        patch.set_alpha(0.6)

    ax.set_xlabel('Max RFI Power (JNR dB)', fontsize=11)
    ax.set_ylabel('Effective Rank', fontsize=11)
    ax.set_title('Effective Rank Distribution by Max RFI Power',
                 fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    # Add reference line at full rank
    ax.axhline(y=BLOCK_HEIGHT, color='red', linestyle='--', alpha=0.3,
               label=f'Full rank ({BLOCK_HEIGHT})')
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_scatter_cond_vs_eff_rank(validation_data, out_path):
    """
    Scatter plot of condition number vs effective rank, color-coded by max RFI power.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color mapping by max RFI power
    max_jnr = JNR_RANGE_DB[1]
    norm_jnr = mcolors.Normalize(vmin=0, vmax=max_jnr)
    cmap = cm.plasma

    for data in validation_data:
        jnr_db_list = data['jnr_db_list']
        max_jnr_power = max(jnr_db_list) if len(jnr_db_list) > 0 else 0

        cond_num = data['condition_number_db']
        eff_rank = data['eff_rank']

        ax.scatter(eff_rank, cond_num, c=[cmap(norm_jnr(max_jnr_power))],
                   alpha=0.5, s=30, edgecolors='none')

    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm_jnr)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Max RFI Power (JNR dB, 0=clean)', fontsize=10)

    ax.set_xlabel('Effective Rank', fontsize=11)
    ax.set_ylabel('Condition Number (dB)', fontsize=11)
    ax.set_title('Condition Number vs Effective Rank',
                 fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_eigenvalue_comparison(validation_data, out_path, n_samples=10):
    """
    Compare computed vs stored eigenvalues for a subset of CPI tiles.
    """
    # Select n_samples random tiles
    np.random.seed(42)
    selected = np.random.choice(len(validation_data),
                                size=min(n_samples, len(validation_data)),
                                replace=False)

    n_cols = 5
    n_rows = (len(selected) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    axes = axes.flat if n_rows > 1 else [axes]

    ev_index = np.arange(1, BLOCK_HEIGHT + 1)

    for ax, idx in zip(axes, selected):
        data = validation_data[idx]
        eigvals_computed = data['eigvals_computed']
        eigvals_stored = data['eigvals_stored']
        knee = data['n_distinct_pulses']
        cpi_key = data['cpi_key']

        if eigvals_stored is not None:
            # Plot both
            ax.plot(ev_index, eigvals_computed, 'b-', label='Computed', linewidth=1.5)
            ax.plot(ev_index, eigvals_stored, 'r--', label='Stored', linewidth=1.5, alpha=0.7)

            # Compute max difference
            max_diff = np.max(np.abs(eigvals_computed - eigvals_stored))
            ax.set_title(f'{cpi_key}\nknee={knee}, max_diff={max_diff:.2e}', fontsize=8)
        else:
            ax.plot(ev_index, eigvals_computed, 'b-', label='Computed', linewidth=1.5)
            ax.set_title(f'{cpi_key}\nknee={knee} (no stored)', fontsize=8)

        ax.set_xlabel('EV Index', fontsize=8)
        ax.set_ylabel('Eigenvalue (linear)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle='--', alpha=0.3)

    # Hide unused subplots
    for ax in axes[len(selected):]:
        ax.axis('off')

    fig.suptitle('Computed vs Stored Eigenvalues (Sample)', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# VALIDATION REPORT
# ---------------------------------------------------------------------------

def print_validation_report(validation_data):
    """
    Print a summary validation report to console.
    """
    print("\n" + "="*70)
    print("VALIDATION REPORT")
    print("="*70)

    print(f"\nTotal CPI tiles analyzed: {len(validation_data)}")

    # Group by knee count
    knee_groups = defaultdict(list)
    for data in validation_data:
        knee = data['n_distinct_pulses']
        knee_groups[knee].append(data)

    print(f"\nDistribution by knee count:")
    for knee in sorted(knee_groups.keys()):
        count = len(knee_groups[knee])
        label = "clean" if knee == 0 else f"knee@{knee}"
        print(f"  {label:10s}: {count:4d} samples")

    # Condition number statistics
    print(f"\nCondition Number (dB) Statistics:")
    for knee in sorted(knee_groups.keys()):
        cond_nums = [d['condition_number_db'] for d in knee_groups[knee]]
        label = "clean" if knee == 0 else f"knee@{knee}"
        print(f"  {label:10s}: mean={np.mean(cond_nums):6.2f}, "
              f"std={np.std(cond_nums):6.2f}, "
              f"min={np.min(cond_nums):6.2f}, "
              f"max={np.max(cond_nums):6.2f}")

    # Effective rank statistics
    print(f"\nEffective Rank Statistics:")
    for knee in sorted(knee_groups.keys()):
        eff_ranks = [d['eff_rank'] for d in knee_groups[knee]]
        label = "clean" if knee == 0 else f"knee@{knee}"
        print(f"  {label:10s}: mean={np.mean(eff_ranks):6.2f}, "
              f"std={np.std(eff_ranks):6.2f}, "
              f"min={np.min(eff_ranks):6.2f}, "
              f"max={np.max(eff_ranks):6.2f}")

    # Eigenvalue comparison (if stored values are available)
    stored_available = [d for d in validation_data if d['eigvals_stored'] is not None]
    if stored_available:
        print(f"\nEigenvalue Verification (computed vs stored):")
        max_diffs = []
        for data in stored_available:
            diff = np.max(np.abs(data['eigvals_computed'] - data['eigvals_stored']))
            max_diffs.append(diff)

        print(f"  Samples with stored eigenvalues: {len(stored_available)}")
        print(f"  Max difference (mean): {np.mean(max_diffs):.2e}")
        print(f"  Max difference (max):  {np.max(max_diffs):.2e}")

        if np.max(max_diffs) < 1e-6:
            print("  ✓ PASS: Computed eigenvalues match stored values")
        else:
            print("  ✗ WARNING: Computed eigenvalues differ from stored values")

    print("\n" + "="*70)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    """
    Main validation pipeline.

    Steps:
    1. Load synthetic data from HDF5 files (clean and contaminated)
    2. Extract eigenvalues, condition number, and effective rank
    3. Verify against stored eigenvalues
    4. Generate validation plots
    5. Print summary report
    """
    data_root = 'data/multi_band'
    output_dir = 'validation_plots'
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("CONDITION NUMBER AND EFFECTIVE RANK VALIDATION")
    print("="*70)
    print("\nThis script validates that condition number and effective rank")
    print("correctly represent eigenvalue spectrum characteristics.")

    # Load clean samples
    clean_dir = os.path.join(data_root, 'clean')
    print(f"\nLoading CLEAN samples from {clean_dir}...")

    all_validation_data = []

    if os.path.exists(clean_dir):
        h5_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.h5')])
        for h5_file in h5_files[:5]:  # Load first 5 files
            h5_path = os.path.join(clean_dir, h5_file)
            data = load_and_validate_h5(h5_path)
            all_validation_data.extend(data)
            print(f"  Loaded {len(data)} tiles from {h5_file}")

    # Load contaminated samples
    contaminated_dir = os.path.join(data_root, 'contaminated')
    print(f"\nLoading CONTAMINATED samples from {contaminated_dir}...")

    if os.path.exists(contaminated_dir):
        h5_files = sorted([f for f in os.listdir(contaminated_dir) if f.endswith('.h5')])
        for h5_file in h5_files[:5]:  # Load first 5 files
            h5_path = os.path.join(contaminated_dir, h5_file)
            data = load_and_validate_h5(h5_path)
            all_validation_data.extend(data)
            print(f"  Loaded {len(data)} tiles from {h5_file}")

    if not all_validation_data:
        print("\nERROR: No data found. Please run generate_synthetic_data.py first.")
        return

    print(f"\nTotal tiles loaded: {len(all_validation_data)}")

    # Generate validation plots
    print(f"\nGenerating validation plots...")

    plot_eigenvalue_with_metrics(
        all_validation_data,
        os.path.join(output_dir, 'eigenvalue_profiles_with_metrics.png')
    )

    plot_condition_number_distribution(
        all_validation_data,
        os.path.join(output_dir, 'condition_number_distribution.png')
    )

    plot_effective_rank_distribution(
        all_validation_data,
        os.path.join(output_dir, 'effective_rank_distribution.png')
    )

    plot_scatter_cond_vs_eff_rank(
        all_validation_data,
        os.path.join(output_dir, 'condition_number_vs_effective_rank.png')
    )

    plot_eigenvalue_comparison(
        all_validation_data,
        os.path.join(output_dir, 'eigenvalue_comparison_sample.png'),
        n_samples=10
    )

    # Print validation report
    print_validation_report(all_validation_data)

    print(f"\nValidation plots saved to: {output_dir}/")
    print("\nDone.")


if __name__ == '__main__':
    main()
