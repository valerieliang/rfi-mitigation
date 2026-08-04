"""
Visualize the synthetic RFI data generation process
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import rcParams
import numpy as np

# Configure matplotlib
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']

fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors
box_color = '#e8f4fd'
box_edge = '#2a78d6'
text_color = '#0b0b0b'
arrow_color = '#52514e'

# Helper function to create boxes
def create_box(x, y, width, height, label, sublabels=None):
    box = FancyBboxPatch((x, y), width, height,
                          boxstyle="round,pad=0.1",
                          edgecolor=box_edge, facecolor=box_color,
                          linewidth=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height - 0.3, label,
            ha='center', va='top', fontsize=12, weight='semibold', color=text_color)

    if sublabels:
        y_offset = y + height - 0.8
        for sublabel in sublabels:
            ax.text(x + width/2, y_offset, sublabel,
                    ha='center', va='top', fontsize=9, color=text_color)
            y_offset -= 0.35

# Helper function to create arrows
def create_arrow(x1, y1, x2, y2, label=''):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color=arrow_color)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label, fontsize=9, color=arrow_color,
                style='italic', bbox=dict(boxstyle='round,pad=0.3',
                facecolor='white', edgecolor='none'))

# Title
ax.text(5, 9.5, 'Synthetic RFI Data Generation Process',
        ha='center', fontsize=16, weight='bold', color=text_color)

# Step 1: Real L0B Data
create_box(0.5, 7, 2, 1.8, 'Real L0B Data',
           ['NISAR raw radar', 'CPI windows', 'Real scene clutter'])

# Step 2: RFI Parameters
create_box(0.5, 4.5, 2, 1.8, 'RFI Parameters',
           ['# uncorrelated sources', 'JSR per band', 'Wideband/Narrowband'])

# Step 3: Injection
create_box(3.5, 5.5, 2.5, 2.2, 'Synthetic RFI Injection',
           ['Add complex interference', 'to real CPI data',
            'JSR controls power', 'Sources → knee position'])

# Step 4: SCM Recomputation
create_box(7, 5.5, 2.5, 2.2, 'SCM Recomputation',
           ['Covariance matrix', 'from contaminated CPI',
            'Gap exclusion applied'])

# Step 5: Output
create_box(4, 2, 2.5, 1.8, 'Eigenvalue Profiles',
           ['Known ground truth', 'knee = # RFI sources', 'JSR per band'])

# Arrows
create_arrow(1.5, 7, 3.5, 6.5)
create_arrow(1.5, 5.4, 3.5, 6.2)
create_arrow(6, 6.6, 7, 6.6)
create_arrow(8.25, 5.5, 5.25, 3.8)

# Key points box
key_box_y = 0.3
ax.text(0.5, key_box_y + 1.2, 'Key Points:', fontsize=11, weight='bold', color=text_color)
key_points = [
    '• Eigenvalue profiles are NEVER synthesized directly',
    '• Real clutter statistics preserved from actual radar scenes',
    '• JSR (dB): Interference power relative to scene backscatter',
    '• Amazon: JSR ∈ [3, 30] dB  |  Mountain: JSR ∈ [3, 15] dB',
    '• # RFI sources determines knee position (# RFI eigenvalues)',
]
y_pos = key_box_y + 0.9
for point in key_points:
    ax.text(0.6, y_pos, point, fontsize=9, color=text_color)
    y_pos -= 0.25

plt.tight_layout()
plt.savefig('data_generation_process.png', dpi=150, facecolor='white', edgecolor='none', bbox_inches='tight')
print("Saved: data_generation_process.png")
plt.close()

# Also create a simple comparison figure showing the eigenvalue effect
fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor='white')

# Load example data
import h5py
EPS = 1e-30
with h5py.File('data/amazon_contam/rfi_data_A_HH.h5', 'r') as f:
    eigenvalues = f['eigenvalues'][:]
    labels = f['labels'][:]
    jsr_db = f['jsr_db'][:] if 'jsr_db' in f else None

    # Get clean and contaminated examples
    clean_idx = np.where(labels == 0)[0][0]

    # Find examples with different knees
    knee1_idx = None
    knee3_idx = None
    for idx in range(len(labels)):
        if labels[idx] == 1 and knee1_idx is None:
            if jsr_db is not None:
                jsr_vals = jsr_db[idx][np.isfinite(jsr_db[idx])]
                if len(jsr_vals) > 0 and 20 <= np.mean(jsr_vals) <= 30:
                    knee1_idx = idx
        elif labels[idx] == 3 and knee3_idx is None:
            if jsr_db is not None:
                jsr_vals = jsr_db[idx][np.isfinite(jsr_db[idx])]
                if len(jsr_vals) > 0 and 20 <= np.mean(jsr_vals) <= 30:
                    knee3_idx = idx
        if knee1_idx is not None and knee3_idx is not None:
            break

    n_show = 12
    clean_eig = 10 * np.log10(np.maximum(eigenvalues[clean_idx, :n_show], EPS))

    profiles = [clean_eig]
    labels_list = ['Clean (knee=0)']
    colors = ['#2a78d6']

    if knee1_idx is not None:
        knee1_eig = 10 * np.log10(np.maximum(eigenvalues[knee1_idx, :n_show], EPS))
        jsr1 = np.mean(jsr_db[knee1_idx][np.isfinite(jsr_db[knee1_idx])]) if jsr_db is not None else 0
        profiles.append(knee1_eig)
        labels_list.append(f'1 RFI source (JSR≈{jsr1:.0f} dB)')
        colors.append('#eb6834')

    if knee3_idx is not None:
        knee3_eig = 10 * np.log10(np.maximum(eigenvalues[knee3_idx, :n_show], EPS))
        jsr3 = np.mean(jsr_db[knee3_idx][np.isfinite(jsr_db[knee3_idx])]) if jsr_db is not None else 0
        profiles.append(knee3_eig)
        labels_list.append(f'3 RFI sources (JSR≈{jsr3:.0f} dB)')
        colors.append('#1baf7a')

    x = np.arange(n_show)

    # Left plot: JSR effect
    ax1 = axes[0]
    ax1.set_facecolor('white')
    if len(profiles) >= 2:
        ax1.plot(x, profiles[0], color=colors[0], linewidth=2, label=labels_list[0])
        ax1.plot(x, profiles[1], color=colors[1], linewidth=2, label=labels_list[1])
        ax1.axvline(x=0.5, color='#898781', linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('Eigenvalue Index', fontsize=11, color='#0b0b0b')
    ax1.set_ylabel('Eigenvalue (dB)', fontsize=11, color='#0b0b0b')
    ax1.set_title('JSR Controls RFI Eigenvalue Magnitude', fontsize=12, weight='semibold')
    ax1.grid(True, linewidth=1, alpha=0.5, color='#e1e0d9')
    ax1.legend(fontsize=9)
    ax1.set_xlim(-0.5, n_show - 0.5)
    ax1.set_xticks(x)

    # Right plot: Number of sources effect
    ax2 = axes[1]
    ax2.set_facecolor('white')
    if len(profiles) >= 3:
        ax2.plot(x, profiles[0], color=colors[0], linewidth=2, label=labels_list[0])
        ax2.plot(x, profiles[2], color=colors[2], linewidth=2, label=labels_list[2])
        ax2.axvline(x=2.5, color='#898781', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('Eigenvalue Index', fontsize=11, color='#0b0b0b')
    ax2.set_ylabel('Eigenvalue (dB)', fontsize=11, color='#0b0b0b')
    ax2.set_title('# RFI Sources Determines Knee Position', fontsize=12, weight='semibold')
    ax2.grid(True, linewidth=1, alpha=0.5, color='#e1e0d9')
    ax2.legend(fontsize=9)
    ax2.set_xlim(-0.5, n_show - 0.5)
    ax2.set_xticks(x)

plt.tight_layout()
plt.savefig('rfi_parameter_effects.png', dpi=150, facecolor='white', edgecolor='none')
print("Saved: rfi_parameter_effects.png")
plt.close()
