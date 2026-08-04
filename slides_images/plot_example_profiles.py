"""
Plot actual eigenvalue profiles: clean Amazon vs 1-knee RFI contaminated (JSR ~26 dB)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import rcParams
import h5py
import sys
import os

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

def load_profile(h5_path, knee_target=None, jsr_target=None):
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
                        if diff < best_diff:
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

# Load profiles
print("Loading clean Amazon profile...")
clean_profile, clean_knee, clean_jsr = load_profile('data/amazon_contam/rfi_data_A_HH.h5',
                                                      knee_target=0)
print(f"  Clean profile: knee={clean_knee}")

print("Loading RFI contaminated profile (knee=1, JSR~26 dB)...")
contam_profile, contam_knee, contam_jsr = load_profile('data/amazon_contam/rfi_data_A_HH.h5',
                                                         knee_target=1, jsr_target=26)
print(f"  Contaminated profile: knee={contam_knee}, JSR={contam_jsr:.1f} dB")

# Truncate to 16 eigenvalues
n_show = 16
clean_profile = clean_profile[:n_show]
contam_profile = contam_profile[:n_show]
x = np.arange(n_show)

# Create figure
fig, ax = plt.subplots(figsize=(9, 6), facecolor='white')
ax.set_facecolor('white')

# Plot lines with proper styling: 2px lines, series colors from palette
line1 = ax.plot(x, contam_profile,
                color='#eb6834',  # slot 2: orange for RFI Contaminated
                linewidth=2,
                label='RFI Contaminated',
                zorder=3)

line2 = ax.plot(x, clean_profile,
                color='#2a78d6',  # slot 1: blue for RFI Free
                linewidth=2,
                label='RFI Free',
                zorder=3)

# Mark the "knee" point at index 1 (after first RFI eigenvalue)
knee_idx = contam_knee
if knee_idx > 0:
    ax.axvline(x=knee_idx - 0.5, color='#898781', linestyle=':', linewidth=1.5, alpha=0.7, zorder=2)

# Configure axes
ax.set_xlabel('Sorted Eigenvalue Index', fontsize=12, color='#0b0b0b', labelpad=8)
ax.set_ylabel('Eigenvalues (dB)', fontsize=12, color='#0b0b0b', labelpad=8)
title = f'RFI Contamination in Eigenvalue Spectrum\n(Amazon scene, knee={contam_knee}, JSR≈{contam_jsr:.1f} dB)'
ax.set_title(title, fontsize=14, weight='semibold', color='#0b0b0b', pad=15)

# Configure grid: hairline, recessive
ax.grid(True, linewidth=1, alpha=0.5, zorder=1)
ax.set_axisbelow(True)

# Set axis limits and ticks
ax.set_xlim(-0.5, n_show - 0.5)
y_max = max(np.max(clean_profile), np.max(contam_profile)) + 2
y_min = min(np.min(clean_profile), np.min(contam_profile)) - 2
ax.set_ylim(y_min, y_max)
ax.set_xticks(x)

# Configure legend: always present for 2+ series
legend = ax.legend(loc='upper right',
                   frameon=True,
                   facecolor='white',
                   edgecolor='#c3c2b7',
                   fontsize=11,
                   framealpha=1.0)
# Legend text uses text tokens, not series colors
for text in legend.get_texts():
    text.set_color('#0b0b0b')

# Clean up tick styling
ax.tick_params(axis='both', which='major', labelsize=10, length=4, width=1)

# Ensure clean layout
plt.tight_layout()

# Save with high DPI for clarity
plt.savefig('eigenvalue_comparison.png', dpi=150, facecolor='white', edgecolor='none')
print("\nPlot saved as eigenvalue_comparison.png")
plt.close(fig)
