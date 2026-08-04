import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Create eigenvalue data (sorted by power, descending)
# 16 total eigenvalues: 6 RFI + 10 Signal
# RFI emitters: 70, 55, 48, 42, 35, 24 (less obvious gaps)
# Signal: ~20 for the rest (smoother decay)
eigenvalues = np.array([
    70,    # Strong RFI #1
    55,    # RFI #2
    48,    # RFI #3
    42,    # RFI #4
    35,    # RFI #5
    24,    # Last RFI (ambiguous) - very close to signal
    22,    # Signal start - gradual decay
    21.5,
    21,
    20.5,
    20.2,
    20,
    19.8,
    19.6,
    19.5,
    19.5
])

indices = np.arange(1, len(eigenvalues) + 1)

# Set up the plot
fig, ax = plt.subplots(figsize=(6, 5))

# Plot the eigenvalue profile
ax.plot(indices, eigenvalues, 'o-', color='#666666', linewidth=2,
        markersize=0, alpha=0.3, zorder=1)

# Color-code the points
# RFI (indices 1-6, including the ambiguous last one)
ax.scatter(indices[:6], eigenvalues[:6], s=100, color='#e34948',
          label='RFI', zorder=3, edgecolors='white', linewidths=1.5)

# Signal (indices 7+)
ax.scatter(indices[6:], eigenvalues[6:], s=100, color='#2a78d6',
          label='Signal', zorder=3, edgecolors='white', linewidths=1.5)

# Add value labels on key points
ax.text(1, 72, '70', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.text(2, 57, '55', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.text(3, 50, '48', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.text(4, 44, '42', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.text(5, 37, '35', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.text(6, 26, '24', ha='center', va='bottom', fontsize=10, fontweight='bold')



# Styling
ax.set_xlabel('Eigenvalue Index', fontsize=12, fontweight='500')
ax.set_ylabel('Eigenvalue Power', fontsize=12, fontweight='500')

# Grid
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# Set axis limits (tighter x-axis)
ax.set_xlim(0.5, len(eigenvalues) + 0.5)
ax.set_ylim(0, 75)

# Legend
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=10)

# Tight layout
plt.tight_layout()

# Save
output_path = 'eigenvalue_merge_problem.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved to {output_path}")

plt.close()
