"""
Prettier eigenvalue plot showing the "knee" point between RFI-contaminated
and RFI-free signal components.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

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

# Create synthetic eigenvalue data similar to actual profile patterns
# Discrete eigenvalue indices (not continuous)
n_eigenvalues = 16
x = np.arange(n_eigenvalues)

# Generate realistic eigenvalue patterns with characteristic "knee"
# RFI Contaminated: high initial eigenvalues with sharp drop at knee
rfi_contaminated = np.array([42, 38, 30, 24, 16, 12, 8, 6, 5, 4.5, 4, 3.8, 3.6, 3.4, 3.2, 3.0])
# RFI Free: lower initial plateau with gradual decay
rfi_free = np.array([15, 14.5, 14, 13.5, 12, 10, 8, 6.5, 5.5, 4.8, 4.3, 4.0, 3.8, 3.6, 3.4, 3.2])

# Create figure with appropriate sizing
fig, ax = plt.subplots(figsize=(9, 6), facecolor='#fcfcfb')
ax.set_facecolor('#fcfcfb')

# Plot lines with proper styling: 2px lines, series colors from palette
line1 = ax.plot(x, rfi_contaminated,
                color='#eb6834',  # slot 2: orange for RFI Contaminated
                linewidth=2,
                label='RFI Contaminated',
                zorder=3)

line2 = ax.plot(x, rfi_free,
                color='#2a78d6',  # slot 1: blue for RFI Free
                linewidth=2,
                label='RFI Free',
                zorder=3)

# Add 8px markers at the endpoints with 2px surface rings
ax.plot(x[-1], rfi_contaminated[-1],
        'o', color='#eb6834', markersize=8,
        markeredgecolor='#fcfcfb', markeredgewidth=2, zorder=4)
ax.plot(x[-1], rfi_free[-1],
        'o', color='#2a78d6', markersize=8,
        markeredgecolor='#fcfcfb', markeredgewidth=2, zorder=4)

# Mark the "knee" point around index 4-5
knee_idx = 4
ax.annotate('"knee"',
            xy=(knee_idx, rfi_contaminated[knee_idx]),
            xytext=(knee_idx + 2, rfi_contaminated[knee_idx] - 8),
            fontsize=11,
            color='#52514e',  # secondary ink
            fontstyle='italic',
            arrowprops=dict(arrowstyle='-', color='#898781', lw=1))

# Configure axes
ax.set_xlabel('Sorted Eigenvalue Index', fontsize=12, color='#0b0b0b', labelpad=8)
ax.set_ylabel('Eigenvalues (dB)', fontsize=12, color='#0b0b0b', labelpad=8)
ax.set_title('RFI Contamination in Eigenvalue Spectrum',
             fontsize=14, weight='semibold', color='#0b0b0b', pad=15)

# Configure grid: hairline, recessive
ax.grid(True, linewidth=1, alpha=0.5, zorder=1)
ax.set_axisbelow(True)

# Set axis limits and ticks
ax.set_xlim(-0.5, n_eigenvalues - 0.5)
ax.set_ylim(0, 45)
ax.set_xticks(x)

# Configure legend: always present for 2+ series
legend = ax.legend(loc='upper right',
                   frameon=True,
                   facecolor='#fcfcfb',
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
plt.savefig('eigenvalue_spectrum.png', dpi=150, facecolor='#fcfcfb', edgecolor='none')
print("Plot saved as eigenvalue_spectrum.png")

# Display
plt.show()
