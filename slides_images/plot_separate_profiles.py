"""
Generate separate, clean eigenvalue and diagonal profile plots.
Each profile gets its own plot for maximum readability.

Usage:
    python slides_images/plot_separate_profiles.py <data_file.h5> --scene-name "Scene Name"
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import h5py
import sys
import os
import argparse

# Configure matplotlib
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']
rcParams['axes.linewidth'] = 1.0
rcParams['axes.edgecolor'] = '#c3c2b7'
rcParams['axes.labelcolor'] = '#0b0b0b'
rcParams['text.color'] = '#0b0b0b'
rcParams['xtick.color'] = '#898781'
rcParams['ytick.color'] = '#898781'
rcParams['grid.color'] = '#e1e0d9'
rcParams['grid.linewidth'] = 1.0

EPS = 1e-30

def compute_tight_ylim(values, margin_fraction=0.15):
    """Compute tight y-limits with margin."""
    v_min = np.min(values)
    v_max = np.max(values)
    data_range = v_max - v_min

    if data_range < 2.0:
        data_range = 2.0

    margin = data_range * margin_fraction
    y_min = v_min - margin
    y_max = v_max + margin

    # Round to nice values
    y_min = np.floor(y_min / 5) * 5
    y_max = np.ceil(y_max / 5) * 5

    return y_min, y_max


def plot_eigenvalue_profile(eig_db, knee, jsr, scene_name, output_path, cpi_pulse_start=None):
    """Plot a single eigenvalue profile."""
    n_show = 16
    x = np.arange(n_show)
    eig_trunc = eig_db[:n_show]

    fig, ax = plt.subplots(figsize=(9, 6), facecolor='white')
    ax.set_facecolor('white')

    # Plot eigenvalues
    ax.plot(x, eig_trunc,
            color='#2a78d6',
            linewidth=2.5,
            marker='o',
            markersize=5,
            markerfacecolor='#2a78d6',
            markeredgecolor='white',
            markeredgewidth=1.5,
            zorder=3)

    # Mark knee if contaminated
    if knee > 0:
        knee_x = knee - 0.5
        ax.axvline(x=knee_x, color='#eb6834', linestyle='--',
                  linewidth=2, alpha=0.7, zorder=2, label=f'Knee at {knee}')

    # Configure axes
    ax.set_xlabel('Sorted Eigenvalue Index', fontsize=13, color='#0b0b0b',
                  labelpad=10, weight='semibold')
    ax.set_ylabel('Eigenvalue (dB)', fontsize=13, color='#0b0b0b',
                  labelpad=10, weight='semibold')

    # Title
    pulse_info = f" (CPI pulses {cpi_pulse_start}:{cpi_pulse_start+16})" if cpi_pulse_start is not None else ""
    if knee == 0:
        title = f'{scene_name}\nClean (No RFI){pulse_info}'
    else:
        jsr_str = f", JSR = {jsr:.1f} dB" if jsr is not None else ""
        title = f'{scene_name}\n{knee} RFI Source' + ('s' if knee > 1 else '') + jsr_str + pulse_info

    ax.set_title(title, fontsize=14, weight='bold', color='#0b0b0b', pad=20)

    # Grid
    ax.grid(True, linewidth=1, alpha=0.5, zorder=1)
    ax.set_axisbelow(True)

    # Limits
    ax.set_xlim(-0.5, n_show - 0.5)
    y_min, y_max = compute_tight_ylim(eig_trunc, margin_fraction=0.20)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(x)

    # Legend if knee marked
    if knee > 0:
        legend = ax.legend(loc='upper right', frameon=True,
                          facecolor='white', edgecolor='#c3c2b7',
                          fontsize=11, framealpha=1.0)
        for text in legend.get_texts():
            text.set_color('#0b0b0b')

    ax.tick_params(axis='both', which='major', labelsize=11, length=5, width=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='white', edgecolor='none')
    plt.close(fig)


def plot_diagonal_profile(diag_db, diag_valid_idx, knee, jsr, scene_name, output_path, cpi_pulse_start=None):
    """Plot a single diagonal profile."""
    n_range = len(diag_db)
    x = np.arange(n_range)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    ax.set_facecolor('white')

    # Plot diagonal as scatter (dots) since position doesn't matter
    ax.scatter(x, diag_db,
               color='#1baf7a',
               s=20,
               alpha=0.7,
               edgecolors='none',
               zorder=3)

    # Mark valid region if available
    if diag_valid_idx is not None and len(diag_valid_idx) > 0:
        valid_mask = np.zeros(n_range, dtype=bool)
        valid_mask[diag_valid_idx] = True

        # Shade invalid regions
        for i in range(n_range):
            if not valid_mask[i]:
                ax.axvspan(i-0.5, i+0.5, color='#e1e0d9', alpha=0.3, zorder=1)

    # Configure axes
    ax.set_xlabel('Range Sample Index', fontsize=13, color='#0b0b0b',
                  labelpad=10, weight='semibold')
    ax.set_ylabel('Diagonal Power (dB)', fontsize=13, color='#0b0b0b',
                  labelpad=10, weight='semibold')

    # Title
    pulse_info = f" (CPI pulses {cpi_pulse_start}:{cpi_pulse_start+16})" if cpi_pulse_start is not None else ""
    if knee == 0:
        title = f'{scene_name}\nDiagonal Profile - Clean (No RFI){pulse_info}'
    else:
        jsr_str = f", JSR = {jsr:.1f} dB" if jsr is not None else ""
        title = f'{scene_name}\nDiagonal Profile - {knee} RFI Source' + ('s' if knee > 1 else '') + jsr_str + pulse_info

    ax.set_title(title, fontsize=14, weight='bold', color='#0b0b0b', pad=20)

    # Grid
    ax.grid(True, linewidth=1, alpha=0.5, zorder=2)
    ax.set_axisbelow(True)

    # Limits
    ax.set_xlim(-0.5, n_range - 0.5)
    y_min, y_max = compute_tight_ylim(diag_db, margin_fraction=0.15)
    ax.set_ylim(y_min, y_max)

    # Show fewer x-ticks for readability
    tick_stride = max(1, n_range // 10)
    ax.set_xticks(x[::tick_stride])

    ax.tick_params(axis='both', which='major', labelsize=11, length=5, width=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='white', edgecolor='none')
    plt.close(fig)


def generate_profiles(h5_path, scene_name, output_dir='slides_images',
                     num_profiles=5, target_knees=None, pulse_start=None, cpi_len=16):
    """
    Generate separate eigenvalue and diagonal plots.

    Args:
        h5_path: path to HDF5 file
        scene_name: scene name for titles
        output_dir: output directory
        num_profiles: number of profiles to generate
        target_knees: list of specific knees to generate (e.g., [0, 1, 2, 3])
        pulse_start: starting pulse index in raw data (for labeling CPIs)
        cpi_len: CPI length in pulses (default: 16)
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Generating profiles for {scene_name}")
    print(f"{'='*60}\n")

    with h5py.File(h5_path, 'r') as f:
        # Handle both data formats: synthetic (root level) and real (evd/ group)
        if 'eigenvalues' in f:
            # Synthetic data format
            eigenvalues = f['eigenvalues'][:]
            labels = f['labels'][:] if 'labels' in f else None
            jsr_db = f['jsr_db'][:] if 'jsr_db' in f else None
            diagonal = f['diagonal'][:] if 'diagonal' in f else None
            diag_valid_idx = f['diag_valid_idx'][:] if 'diag_valid_idx' in f else None
        elif 'evd' in f:
            # Real data format (from read_nisar_isce3.py)
            eigenvalues = f['evd/eigenvalues'][:]
            labels = None  # Real data doesn't have labels
            jsr_db = None  # Real data doesn't have JSR
            diagonal = f['evd/diagonal_power'][:]
            diag_valid_idx = f['evd/diagonal_valid'][:]
        else:
            raise ValueError(f"Unknown HDF5 format. Keys: {list(f.keys())}")

        # Convert to dB
        eig_db = 10.0 * np.log10(np.maximum(eigenvalues, EPS))
        if diagonal is not None:
            diag_db = 10.0 * np.log10(np.maximum(diagonal, EPS))
        else:
            diag_db = None

        # Select profiles to generate
        if target_knees is not None and labels is not None:
            # Synthetic data with labels
            profiles_to_generate = []
            for knee in target_knees:
                candidates = np.where(labels == knee)[0]
                if len(candidates) > 0:
                    idx = candidates[0]
                    profiles_to_generate.append(idx)
        else:
            # Real data without labels - just generate first num_profiles
            n_profiles = len(eigenvalues)
            profiles_to_generate = list(range(min(num_profiles, n_profiles)))

        scene_slug = scene_name.lower().replace(' ', '_').replace('(', '').replace(')', '')

        for idx in profiles_to_generate:
            knee = int(labels[idx]) if labels is not None else 0

            # Get JSR
            actual_jsr = None
            if jsr_db is not None:
                jsr_vals = jsr_db[idx][np.isfinite(jsr_db[idx])]
                if len(jsr_vals) > 0:
                    actual_jsr = np.mean(jsr_vals)

            # Calculate CPI pulse range
            cpi_pulse_start_abs = None
            if pulse_start is not None:
                cpi_pulse_start_abs = pulse_start + (idx * cpi_len)

            print(f"Profile {idx}: knee={knee}", end='')
            if actual_jsr is not None:
                print(f", JSR={actual_jsr:.1f} dB", end='')
            if cpi_pulse_start_abs is not None:
                print(f", CPI pulses [{cpi_pulse_start_abs}:{cpi_pulse_start_abs+cpi_len}]")
            else:
                print()

            # Plot eigenvalues
            eig_out = os.path.join(output_dir,
                                   f'{scene_slug}_eigenvalues_knee{knee}_idx{idx}.png')
            plot_eigenvalue_profile(eig_db[idx], knee, actual_jsr, scene_name, eig_out, cpi_pulse_start_abs)
            print(f"  [OK] {eig_out}")

            # Plot diagonal if available
            if diag_db is not None:
                diag_idx_valid = diag_valid_idx[idx] if diag_valid_idx is not None else None
                diag_out = os.path.join(output_dir,
                                       f'{scene_slug}_diagonal_knee{knee}_idx{idx}.png')
                plot_diagonal_profile(diag_db[idx], diag_idx_valid, knee,
                                     actual_jsr, scene_name, diag_out, cpi_pulse_start_abs)
                print(f"  [OK] {diag_out}")

            print()

    print(f"{'='*60}")
    print("Done!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate separate eigenvalue and diagonal profile plots'
    )
    parser.add_argument('h5_file', help='Path to HDF5 file')
    parser.add_argument('--scene-name', required=True, help='Scene name for titles')
    parser.add_argument('--output-dir', default='slides_images',
                       help='Output directory (default: slides_images)')
    parser.add_argument('--num-profiles', type=int, default=5,
                       help='Number of profiles to generate (default: 5)')
    parser.add_argument('--knees', type=int, nargs='+',
                       help='Specific knee values to generate (e.g., --knees 0 1 2 3)')
    parser.add_argument('--pulse-start', type=int,
                       help='Starting pulse index in raw data (for CPI labeling)')
    parser.add_argument('--cpi-len', type=int, default=16,
                       help='CPI length in pulses (default: 16)')

    args = parser.parse_args()

    if not os.path.exists(args.h5_file):
        print(f"ERROR: File not found: {args.h5_file}")
        sys.exit(1)

    generate_profiles(args.h5_file, args.scene_name, args.output_dir,
                     args.num_profiles, args.knees, args.pulse_start, args.cpi_len)
