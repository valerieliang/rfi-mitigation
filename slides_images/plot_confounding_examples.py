"""
Plot eigenvalue profiles demonstrating adaptive thresholding confounding:
1. Clean majority, one strong RFI
2. Many deep RFIs, one clean
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import h5py

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

def load_profiles(h5_path, n_show=12):
    """Load all eigenvalue profiles from H5 file."""
    with h5py.File(h5_path, 'r') as f:
        eigenvalues = f['eigenvalues'][:]
        labels = f['labels'][:] if 'labels' in f else None
        jsr_db = f['jsr_db'][:] if 'jsr_db' in f else None

        # Convert to dB
        eig_db = 10.0 * np.log10(np.maximum(eigenvalues, EPS))

        return eig_db[:, :n_show], labels, jsr_db

def find_profile(eig_db, labels, jsr_db, knee_target, jsr_min=None, jsr_max=None):
    """Find a profile matching criteria."""
    candidates = np.where(labels == knee_target)[0]

    if jsr_min is not None or jsr_max is not None:
        filtered = []
        for idx in candidates:
            if jsr_db is not None:
                jsr_vals = jsr_db[idx][np.isfinite(jsr_db[idx])]
                if len(jsr_vals) > 0:
                    mean_jsr = np.mean(jsr_vals)
                    if jsr_min is not None and mean_jsr < jsr_min:
                        continue
                    if jsr_max is not None and mean_jsr > jsr_max:
                        continue
                    filtered.append(idx)
        candidates = np.array(filtered)

    if len(candidates) > 0:
        return candidates[0]
    return None

def plot_scenario(profiles, labels_list, colors, title, filename, n_show=12):
    """Plot multiple profiles overlaid."""
    x = np.arange(n_show)

    fig, ax = plt.subplots(figsize=(9, 6), facecolor='white')
    ax.set_facecolor('white')

    # Plot all profiles with faded majority and prominent outlier
    alphas = [0.3] * (len(profiles) - 1) + [1.0]
    for profile, label, color, alpha in zip(profiles, labels_list, colors, alphas):
        ax.plot(x, profile, color=color, linewidth=2, alpha=alpha, zorder=3)

    # Add legend with only unique labels
    unique_labels = []
    unique_colors = []
    for label, color in zip(labels_list, colors):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_colors.append(color)

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=c, linewidth=2, label=l)
                       for l, c in zip(unique_labels, unique_colors)]
    ax.legend(handles=legend_elements, loc='upper right',
              frameon=True, facecolor='white', edgecolor='#c3c2b7',
              fontsize=11, framealpha=1.0)

    # Configure axes
    ax.set_xlabel('Sorted Eigenvalue Index', fontsize=12, color='#0b0b0b', labelpad=8)
    ax.set_ylabel('Eigenvalues (dB)', fontsize=12, color='#0b0b0b', labelpad=8)
    ax.set_title(title, fontsize=14, weight='semibold', color='#0b0b0b', pad=15)

    # Configure grid
    ax.grid(True, linewidth=1, alpha=0.5, zorder=1)
    ax.set_axisbelow(True)

    # Set axis limits
    ax.set_xlim(-0.5, n_show - 0.5)
    y_max = max([np.max(p) for p in profiles]) + 2
    y_min = min([np.min(p) for p in profiles]) - 2
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(x)


    ax.tick_params(axis='both', which='major', labelsize=10, length=4, width=1)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, facecolor='white', edgecolor='none')
    print(f"Saved: {filename}")
    plt.close(fig)

# Load data
print("Loading Amazon data...")
eig_db, labels, jsr_db = load_profiles('data/amazon_contam/rfi_data_A_HH.h5')
print(f"Loaded {len(eig_db)} profiles")

# Scenario 1: Clean majority, one strong RFI
print("\nScenario 1: Clean majority, one strong RFI")
# Get 6 clean profiles
clean_mask = labels == 0
clean_candidates = np.where(clean_mask)[0]
clean_indices = clean_candidates[:6].tolist() if len(clean_candidates) >= 6 else []

# Find one strong RFI with knee around 1-2 (first few eigenvalues)
rfi_mask = (labels == 1) | (labels == 2)
rfi_candidates = np.where(rfi_mask)[0]
strong_rfi_idx = None
for idx in rfi_candidates:
    if jsr_db is not None:
        jsr_vals = jsr_db[idx][np.isfinite(jsr_db[idx])]
        if len(jsr_vals) > 0 and np.mean(jsr_vals) >= 20:
            strong_rfi_idx = idx
            break

if len(clean_indices) >= 6 and strong_rfi_idx is not None:
    profiles_1 = [eig_db[i] for i in clean_indices] + [eig_db[strong_rfi_idx]]
    # Simplified legend - just show category once
    labels_1 = ['RFI Free'] * 6 + ['RFI']
    colors_1 = ['#2a78d6'] * 6 + ['#eb6834']

    plot_scenario(profiles_1, labels_1, colors_1,
                  'Clean Majority, One Strong RFI\n(Block threshold swayed by clean majority)',
                  'confounding_clean_majority.png')
else:
    print("Could not find enough profiles for scenario 1")

# Scenario 2: Many deep RFIs, one clean
print("\nScenario 2: Many deep RFIs, one clean")
# Get 6 RFI profiles with knee around 1-2 (first few eigenvalues)
rfi_mask = (labels == 1) | (labels == 2)
rfi_candidates = np.where(rfi_mask)[0]
rfi_indices = []
for idx in rfi_candidates:
    if jsr_db is not None:
        jsr_vals = jsr_db[idx][np.isfinite(jsr_db[idx])]
        if len(jsr_vals) > 0 and np.mean(jsr_vals) >= 20:
            rfi_indices.append(idx)
            if len(rfi_indices) >= 6:
                break

clean_mask = labels == 0
clean_candidates = np.where(clean_mask)[0]
clean_idx = clean_candidates[0] if len(clean_candidates) > 0 else None

if len(rfi_indices) >= 6 and clean_idx is not None:
    profiles_2 = [eig_db[i] for i in rfi_indices] + [eig_db[clean_idx]]

    # Simplified legend - just show category once
    labels_2 = ['RFI'] * 6 + ['RFI Free']
    colors_2 = ['#eb6834'] * 6 + ['#2a78d6']

    plot_scenario(profiles_2, labels_2, colors_2,
                  'Many Deep RFIs, One Clean\n(Block threshold swayed by RFI majority)',
                  'confounding_rfi_majority.png')
else:
    print("Could not find enough profiles for scenario 2")

print("\nDone!")
