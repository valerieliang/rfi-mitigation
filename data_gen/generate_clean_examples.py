"""
generate_clean_examples.py

Generate a few example clean profiles for different scenarios to visualize
what the eigenvalue profiles look like for scattered targets, mountains, urban, etc.

This script generates 2 examples per scenario and saves to data/diff_targets/.
"""

import os
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import from generate_synthetic_data
import sys
sys.path.append(os.path.dirname(__file__))
from generate_synthetic_data import (
    NOISE_DB, BLOCK_HEIGHT, BLOCK_WIDTH,
    generate_clean_image, generate_mountain_clutter,
    generate_urban_clutter, generate_distributed_clutter,
    MOUNTAIN_CNR_RANGE_DB, URBAN_CNR_RANGE_DB, DISTRIBUTED_CNR_RANGE_DB,
    _compute_eigenvalues_normalized
)

# Output directory
OUTPUT_DIR = 'data/diff_targets'

def generate_and_plot_scenario(scenario_name, generator_func, cnr_db, seed, snr_db=12):
    """
    Generate a single CPI tile for a given scenario and plot its eigenvalue profile.

    Args:
        scenario_name (str): Name of scenario ('mountain', 'urban', 'distributed')
        generator_func: Function to generate clutter (generate_mountain_clutter, etc.)
        cnr_db (float): CNR in dB
        seed (int): Random seed
        snr_db (float): SNR in dB (default 12)

    Returns:
        cpi (np.ndarray): Generated CPI tile
        eigvals_db (np.ndarray): Eigenvalues in dB
    """
    # Generate noise + signal baseline
    noise_power_linear = 10.0 ** (NOISE_DB / 10.0)
    signal_power_linear = noise_power_linear * (10.0 ** (snr_db / 10.0))

    rng_noise = np.random.default_rng(seed)
    noise_matrix = (
        rng_noise.standard_normal((BLOCK_HEIGHT, BLOCK_WIDTH))
        + 1j * rng_noise.standard_normal((BLOCK_HEIGHT, BLOCK_WIDTH))
    ) * np.sqrt(noise_power_linear / 2.0)

    rng_signal = np.random.default_rng(seed + 1)
    signal_matrix = (
        rng_signal.standard_normal((BLOCK_HEIGHT, BLOCK_WIDTH))
        + 1j * rng_signal.standard_normal((BLOCK_HEIGHT, BLOCK_WIDTH))
    ) * np.sqrt(signal_power_linear / 2.0)

    # Add scenario-specific clutter
    clutter_matrix = generator_func(
        BLOCK_HEIGHT, BLOCK_WIDTH, cnr_db, noise_power_linear, seed + 2
    )

    cpi = (noise_matrix + signal_matrix + clutter_matrix).astype(np.complex64)

    # Compute eigenvalues
    eigvals_db, max_db, min_db, cond_num, eff_rank = _compute_eigenvalues_normalized(cpi)

    return cpi, eigvals_db, max_db, min_db, cond_num, eff_rank


def plot_comparison_grid(examples, output_path):
    """
    Plot a grid comparing all scenario examples.

    Args:
        examples (list): List of dicts with keys: scenario, seed, cnr_db, eigvals_db, stats
        output_path (str): Path to save the plot
    """
    n_examples = len(examples)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    ev_index = np.arange(1, BLOCK_HEIGHT + 1)

    for idx, ex in enumerate(examples):
        ax = axes[idx]
        scenario = ex['scenario']
        eigvals_db = ex['eigvals_db']
        cnr_db = ex['cnr_db']
        stats = ex['stats']
        seed = ex['seed']

        # Color by scenario
        if scenario == 'mountain':
            color = 'brown'
            label_str = 'Mountain'
        elif scenario == 'urban':
            color = 'blue'
            label_str = 'Urban'
        else:
            color = 'green'
            label_str = 'Distributed'

        ax.plot(ev_index, eigvals_db, color=color, linewidth=2, label=label_str)

        # Mark characteristic features
        if scenario == 'mountain':
            # Mark the steep drop (Z-shape)
            knee_idx = 3  # Approximate knee for mountain
            ax.axvline(x=knee_idx, color='red', linestyle='--', alpha=0.5, linewidth=1)

        ax.set_xlabel('Eigenvalue Index', fontsize=10)
        ax.set_ylabel('Eigenvalue (dB)', fontsize=10)
        ax.set_title(
            f'{label_str} (CNR={cnr_db:.1f} dB)\n'
            f'Cond#={stats["cond_num"]:.1f} | Eff Rank={stats["eff_rank"]:.1f}\n'
            f'λ: [{stats["min_db"]:.1f}, {stats["max_db"]:.1f}] dB',
            fontsize=9
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

    # Hide unused subplots
    for idx in range(n_examples, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(
        'Clean Eigenvalue Profiles - Different Scenarios\n'
        'SNR=12 dB | No RFI',
        fontsize=12, fontweight='bold'
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved comparison plot: {output_path}")


def main():
    """Generate example clean profiles for different scenarios."""

    print("="*80)
    print("GENERATING CLEAN EXAMPLE PROFILES")
    print("="*80)
    print()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Configuration: 2 examples per scenario
    scenarios = [
        {
            'name': 'mountain',
            'func': generate_mountain_clutter,
            'cnr_range': MOUNTAIN_CNR_RANGE_DB,
            'description': 'Mountain terrain (Z-shaped, few strong scatterers)',
            'seeds': [1000, 1001],
        },
        {
            'name': 'urban',
            'func': generate_urban_clutter,
            'cnr_range': URBAN_CNR_RANGE_DB,
            'description': 'Urban (point scatterers, elevated eigenvalues)',
            'seeds': [2000, 2001],
        },
        {
            'name': 'distributed',
            'func': generate_distributed_clutter,
            'cnr_range': DISTRIBUTED_CNR_RANGE_DB,
            'description': 'Distributed (water/vegetation, flat profile)',
            'seeds': [3000, 3001],
        },
    ]

    snr_db = 12  # Fixed SNR for all examples

    all_examples = []

    for scenario in scenarios:
        print(f"\n{scenario['name'].upper()}: {scenario['description']}")
        print(f"  CNR range: {scenario['cnr_range']} dB")

        for seed in scenario['seeds']:
            # Draw random CNR within range
            rng = np.random.default_rng(seed)
            cnr_db = float(rng.uniform(*scenario['cnr_range']))

            # Generate example
            cpi, eigvals_db, max_db, min_db, cond_num, eff_rank = generate_and_plot_scenario(
                scenario['name'],
                scenario['func'],
                cnr_db,
                seed,
                snr_db
            )

            stats = {
                'max_db': max_db,
                'min_db': min_db,
                'cond_num': cond_num,
                'eff_rank': eff_rank,
            }

            # Save to HDF5
            h5_path = os.path.join(OUTPUT_DIR, f"{scenario['name']}_seed{seed}_cnr{cnr_db:.1f}.h5")
            with h5py.File(h5_path, 'w') as f:
                f.create_dataset('cpi', data=cpi)
                f.attrs['scenario'] = scenario['name']
                f.attrs['snr_db'] = snr_db
                f.attrs['cnr_db'] = cnr_db
                f.attrs['seed'] = seed
                f.attrs['max_eigval_db'] = max_db
                f.attrs['min_eigval_db'] = min_db
                f.attrs['condition_number'] = cond_num
                f.attrs['effective_rank'] = eff_rank
                f.create_dataset('eigenvalues_db', data=eigvals_db)

            print(f"    seed={seed} CNR={cnr_db:.1f} dB | Cond#={cond_num:.1f} Eff Rank={eff_rank:.1f}")
            print(f"      -> {os.path.basename(h5_path)}")

            # Store for comparison plot
            all_examples.append({
                'scenario': scenario['name'],
                'seed': seed,
                'cnr_db': cnr_db,
                'eigvals_db': eigvals_db,
                'stats': stats,
            })

    # Generate comparison plot
    print("\nGenerating comparison plot...")
    comparison_path = os.path.join(OUTPUT_DIR, 'clean_profiles_comparison.png')
    plot_comparison_grid(all_examples, comparison_path)

    # Generate individual plots for each scenario
    print("\nGenerating individual scenario plots...")

    for scenario in scenarios:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        scenario_examples = [ex for ex in all_examples if ex['scenario'] == scenario['name']]

        for idx, ex in enumerate(scenario_examples):
            ax = axes[idx]
            ev_index = np.arange(1, BLOCK_HEIGHT + 1)
            eigvals_db = ex['eigvals_db']
            cnr_db = ex['cnr_db']
            stats = ex['stats']

            if scenario['name'] == 'mountain':
                color = 'brown'
            elif scenario['name'] == 'urban':
                color = 'blue'
            else:
                color = 'green'

            ax.plot(ev_index, eigvals_db, color=color, linewidth=2.5)
            ax.set_xlabel('Eigenvalue Index', fontsize=11)
            ax.set_ylabel('Eigenvalue (dB)', fontsize=11)
            ax.set_title(
                f'Example {idx+1} (CNR={cnr_db:.1f} dB)\n'
                f'Cond#={stats["cond_num"]:.1f} | Eff Rank={stats["eff_rank"]:.1f}',
                fontsize=10
            )
            ax.grid(True, alpha=0.4)

        fig.suptitle(
            f'{scenario["name"].capitalize()} Eigenvalue Profiles\n'
            f'{scenario["description"]}',
            fontsize=12, fontweight='bold'
        )
        fig.tight_layout()

        individual_path = os.path.join(OUTPUT_DIR, f'{scenario["name"]}_profiles.png')
        fig.savefig(individual_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {os.path.basename(individual_path)}")

    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}/")
    print(f"Total examples: {len(all_examples)}")
    print("\nFiles generated:")
    print(f"  - {len(all_examples)} HDF5 files (one per example)")
    print(f"  - 1 comparison plot (all scenarios)")
    print(f"  - 3 individual scenario plots")
    print("\nScenario characteristics:")
    print("  - Mountain: Z-shaped (steep drop after 1-3 strong eigenvalues)")
    print("  - Urban: Elevated curve (2-8 point scatterers, gradual drop)")
    print("  - Distributed: Flat profile (high effective rank, volume scattering)")
    print()


if __name__ == '__main__':
    main()