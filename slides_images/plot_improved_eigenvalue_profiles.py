"""
Generate eigenvalue profile plots with improved y-axis scaling to avoid flat appearance.

Usage:
    python slides_images/plot_improved_eigenvalue_profiles.py <data_file.h5> [--scene-name SCENE]

Examples:
    python slides_images/plot_improved_eigenvalue_profiles.py data/amazon_contam/rfi_data_A_HH.h5 --scene-name Amazon
    python slides_images/plot_improved_eigenvalue_profiles.py data/czech_contam/rfi_data_A_HH.h5 --scene-name Czech
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

# Configure matplotlib for clean, professional appearance
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

def compute_ylim_tight(profiles, margin_fraction=0.15):
    """
    Compute tight y-axis limits that avoid flat appearance.

    Args:
        profiles: list of eigenvalue profiles in dB
        margin_fraction: fraction of data range to add as margin (default 0.15 = 15%)

    Returns:
        (y_min, y_max) tuple
    """
    all_values = np.concatenate([p for p in profiles if p is not None])
    data_min = np.min(all_values)
    data_max = np.max(all_values)

    # Calculate range
    data_range = data_max - data_min

    # If range is very small (< 2 dB), force a minimum visible range
    if data_range < 2.0:
        data_range = 2.0

    # Add margin as fraction of range
    margin = data_range * margin_fraction

    y_min = data_min - margin
    y_max = data_max + margin

    # Round to nice values
    y_min = np.floor(y_min / 5) * 5
    y_max = np.ceil(y_max / 5) * 5

    return y_min, y_max


def load_profile(h5_path, knee_target=None, jsr_target=None, jsr_tolerance=5.0):
    """Load eigenvalue profile from H5 file."""
    with h5py.File(h5_path, 'r') as f:
        eigenvalues = f['eigenvalues'][:]
        labels = f['labels'][:] if 'labels' in f else None
        jsr_db = f['jsr_db'][:] if 'jsr_db' in f else None

        # Convert to dB
        eig_db = 10.0 * np.log10(np.maximum(eigenvalues, EPS))

        # Find a profile matching criteria
        if knee_target is not None and labels is not None:
            # Find profiles with the target knee
            candidates = np.where(labels == knee_target)[0]

            if jsr_target is not None and jsr_db is not None:
                # Among those, find one with JSR close to target
                best_idx = None
                best_diff = float('inf')
                for idx in candidates:
                    jsr_vals = jsr_db[idx][np.isfinite(jsr_db[idx])]
                    if len(jsr_vals) > 0:
                        mean_jsr = np.mean(jsr_vals)
                        diff = abs(mean_jsr - jsr_target)
                        if diff < jsr_tolerance and diff < best_diff:
                            best_diff = diff
                            best_idx = idx
                if best_idx is not None:
                    actual_jsr = np.mean(jsr_db[best_idx][np.isfinite(jsr_db[best_idx])])
                    return eig_db[best_idx], int(labels[best_idx]), actual_jsr

            # Just take first matching knee if no JSR target
            if len(candidates) > 0:
                idx = candidates[0]
                actual_jsr = None
                if jsr_db is not None:
                    jsr_vals = jsr_db[idx][np.isfinite(jsr_db[idx])]
                    if len(jsr_vals) > 0:
                        actual_jsr = np.mean(jsr_vals)
                return eig_db[idx], int(labels[idx]), actual_jsr

        # Default: return first profile
        idx = 0
        knee = int(labels[idx]) if labels is not None else None
        actual_jsr = None
        if jsr_db is not None:
            jsr_vals = jsr_db[idx][np.isfinite(jsr_db[idx])]
            if len(jsr_vals) > 0:
                actual_jsr = np.mean(jsr_vals)
        return eig_db[idx], knee, actual_jsr


def plot_eigenvalue_comparison(h5_path, scene_name='Scene', output_dir='slides_images'):
    """
    Generate eigenvalue profile comparison plots with improved y-axis.

    Args:
        h5_path: path to HDF5 file with eigenvalues
        scene_name: name of the scene for labeling
        output_dir: directory to save output plots
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Generating eigenvalue profiles for {scene_name}")
    print(f"{'='*60}\n")

    # Load profiles
    print("Loading clean profile (knee=0)...")
    clean_profile, clean_knee, clean_jsr = load_profile(h5_path, knee_target=0)
    print(f"  Clean profile: knee={clean_knee}")

    print("\nLoading RFI contaminated profiles...")

    # Try to find contaminated profiles with different knees and JSRs
    contam_profiles = []

    # Knee=1, moderate JSR (~20-26 dB)
    try:
        profile, knee, jsr = load_profile(h5_path, knee_target=1, jsr_target=23, jsr_tolerance=5)
        contam_profiles.append((profile, knee, jsr, '1 RFI source'))
        print(f"  Found knee={knee}, JSR={jsr:.1f} dB")
    except Exception as e:
        print(f"  Could not find knee=1 profile: {e}")

    # Knee=2, moderate JSR
    try:
        profile, knee, jsr = load_profile(h5_path, knee_target=2, jsr_target=20, jsr_tolerance=5)
        contam_profiles.append((profile, knee, jsr, '2 RFI sources'))
        print(f"  Found knee={knee}, JSR={jsr:.1f} dB")
    except Exception as e:
        print(f"  Could not find knee=2 profile: {e}")

    # Knee=3, moderate JSR
    try:
        profile, knee, jsr = load_profile(h5_path, knee_target=3, jsr_target=20, jsr_tolerance=5)
        contam_profiles.append((profile, knee, jsr, '3 RFI sources'))
        print(f"  Found knee={knee}, JSR={jsr:.1f} dB")
    except Exception as e:
        print(f"  Could not find knee=3 profile: {e}")

    if not contam_profiles:
        print("  WARNING: No contaminated profiles found!")
        return

    # Generate plots for each contaminated profile
    n_show = 16
    x = np.arange(n_show)

    colors = ['#2a78d6', '#eb6834', '#1baf7a', '#b24bc8']  # blue, orange, green, purple

    for i, (contam_profile, contam_knee, contam_jsr, label) in enumerate(contam_profiles):
        # Truncate to n_show eigenvalues
        clean_trunc = clean_profile[:n_show]
        contam_trunc = contam_profile[:n_show]

        # Create figure
        fig, ax = plt.subplots(figsize=(9, 6), facecolor='white')
        ax.set_facecolor('white')

        # Plot lines with proper styling
        line1 = ax.plot(x, contam_trunc,
                        color=colors[1],  # orange for RFI Contaminated
                        linewidth=2,
                        label=f'RFI Contaminated ({label})',
                        zorder=3)

        line2 = ax.plot(x, clean_trunc,
                        color=colors[0],  # blue for Clean
                        linewidth=2,
                        label='RFI Free',
                        zorder=3)

        # Mark the "knee" point
        if contam_knee > 0:
            knee_x = contam_knee - 0.5
            ax.axvline(x=knee_x, color='#898781', linestyle=':', linewidth=1.5, alpha=0.7, zorder=2)

            # Add knee label
            ax.annotate(f'knee at {contam_knee}',
                        xy=(contam_knee, contam_trunc[contam_knee]),
                        xytext=(contam_knee + 1.5, contam_trunc[contam_knee] + 2),
                        fontsize=10,
                        color='#52514e',
                        fontstyle='italic',
                        arrowprops=dict(arrowstyle='-', color='#898781', lw=1))

        # Configure axes
        ax.set_xlabel('Sorted Eigenvalue Index', fontsize=12, color='#0b0b0b', labelpad=8)
        ax.set_ylabel('Eigenvalues (dB)', fontsize=12, color='#0b0b0b', labelpad=8)

        jsr_str = f", JSR≈{contam_jsr:.1f} dB" if contam_jsr is not None else ""
        title = f'RFI Contamination in Eigenvalue Spectrum\n({scene_name} scene, knee={contam_knee}{jsr_str})'
        ax.set_title(title, fontsize=14, weight='semibold', color='#0b0b0b', pad=15)

        # Configure grid: hairline, recessive
        ax.grid(True, linewidth=1, alpha=0.5, zorder=1)
        ax.set_axisbelow(True)

        # Set axis limits with improved y-axis
        ax.set_xlim(-0.5, n_show - 0.5)

        # Use tight y-limits to avoid flat appearance
        y_min, y_max = compute_ylim_tight([clean_trunc, contam_trunc], margin_fraction=0.15)
        ax.set_ylim(y_min, y_max)

        ax.set_xticks(x)

        # Configure legend
        legend = ax.legend(loc='upper right',
                           frameon=True,
                           facecolor='white',
                           edgecolor='#c3c2b7',
                           fontsize=11,
                           framealpha=1.0)
        for text in legend.get_texts():
            text.set_color('#0b0b0b')

        # Clean up tick styling
        ax.tick_params(axis='both', which='major', labelsize=10, length=4, width=1)

        # Ensure clean layout
        plt.tight_layout()

        # Save with descriptive filename
        scene_slug = scene_name.lower().replace(' ', '_')
        out_path = os.path.join(output_dir, f'eigenvalue_profile_{scene_slug}_knee{contam_knee}.png')
        plt.savefig(out_path, dpi=150, facecolor='white', edgecolor='none')
        print(f"\n[OK] Saved: {out_path}")
        print(f"  Y-axis range: [{y_min:.1f}, {y_max:.1f}] dB")
        print(f"  Data range: {np.min(contam_trunc):.1f} to {np.max(contam_trunc):.1f} dB")

        plt.close(fig)

    # Also generate a multi-profile comparison plot
    if len(contam_profiles) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
        ax.set_facecolor('white')

        # Plot clean profile
        clean_trunc = clean_profile[:n_show]
        ax.plot(x, clean_trunc,
                color=colors[0],
                linewidth=2,
                label='RFI Free',
                zorder=3)

        # Plot all contaminated profiles
        all_profiles = [clean_trunc]
        for i, (contam_profile, contam_knee, contam_jsr, label) in enumerate(contam_profiles[:3]):
            contam_trunc = contam_profile[:n_show]
            all_profiles.append(contam_trunc)

            jsr_str = f", JSR≈{contam_jsr:.0f} dB" if contam_jsr is not None else ""
            ax.plot(x, contam_trunc,
                    color=colors[i+1],
                    linewidth=2,
                    label=f'{label}{jsr_str}',
                    zorder=3)

            # Mark knee
            if contam_knee > 0:
                ax.axvline(x=contam_knee - 0.5, color=colors[i+1],
                          linestyle=':', linewidth=1, alpha=0.4, zorder=2)

        # Configure axes
        ax.set_xlabel('Sorted Eigenvalue Index', fontsize=12, color='#0b0b0b', labelpad=8)
        ax.set_ylabel('Eigenvalues (dB)', fontsize=12, color='#0b0b0b', labelpad=8)
        ax.set_title(f'RFI Eigenvalue Profiles - {scene_name}',
                     fontsize=14, weight='semibold', color='#0b0b0b', pad=15)

        ax.grid(True, linewidth=1, alpha=0.5, zorder=1)
        ax.set_axisbelow(True)

        ax.set_xlim(-0.5, n_show - 0.5)
        y_min, y_max = compute_ylim_tight(all_profiles, margin_fraction=0.12)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(x)

        legend = ax.legend(loc='upper right',
                           frameon=True,
                           facecolor='white',
                           edgecolor='#c3c2b7',
                           fontsize=10,
                           framealpha=1.0)
        for text in legend.get_texts():
            text.set_color('#0b0b0b')

        ax.tick_params(axis='both', which='major', labelsize=10, length=4, width=1)
        plt.tight_layout()

        scene_slug = scene_name.lower().replace(' ', '_')
        out_path = os.path.join(output_dir, f'eigenvalue_profiles_{scene_slug}_comparison.png')
        plt.savefig(out_path, dpi=150, facecolor='white', edgecolor='none')
        print(f"\n[OK] Saved comparison: {out_path}")
        print(f"  Y-axis range: [{y_min:.1f}, {y_max:.1f}] dB")

        plt.close(fig)

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate eigenvalue profile plots with improved y-axis scaling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python slides_images/plot_improved_eigenvalue_profiles.py data/amazon_contam/rfi_data_A_HH.h5
  python slides_images/plot_improved_eigenvalue_profiles.py data/amazon_contam/rfi_data_A_HH.h5 --scene-name "Amazon Rainforest"
  python slides_images/plot_improved_eigenvalue_profiles.py data/czech_contam/rfi_data_A_HH.h5 --scene-name "Czech Republic"
        '''
    )
    parser.add_argument('h5_file', help='Path to HDF5 file with eigenvalue data')
    parser.add_argument('--scene-name', default='Scene', help='Scene name for plot labels (default: "Scene")')
    parser.add_argument('--output-dir', default='slides_images', help='Output directory (default: slides_images)')

    args = parser.parse_args()

    if not os.path.exists(args.h5_file):
        print(f"ERROR: File not found: {args.h5_file}")
        sys.exit(1)

    plot_eigenvalue_comparison(args.h5_file, args.scene_name, args.output_dir)
