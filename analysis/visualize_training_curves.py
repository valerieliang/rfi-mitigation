"""
Visualize training results and sample predictions

Usage:
    cd rfi-mitigation/
    python analysis/scripts/visualize_training_curves.py
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load training history (relative to repo root)
history = np.load('model/history.npz')

# Find best epoch (71 based on validation IoU)
best_epoch = np.argmax(history['val_iou']) + 1

# Create figure for training curves
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Training Curves to Best Model (Epoch 71)', fontsize=16, fontweight='bold')

epochs = np.arange(1, len(history['train_loss']) + 1)

# Plot 1: Loss curves
ax = axes[0, 0]
ax.plot(epochs, history['train_loss'], label='Train Loss', color='#2a78d6', linewidth=2)
ax.plot(epochs, history['val_loss'], label='Val Loss', color='#eb6834', linewidth=2)
ax.axvline(best_epoch, color='#0ca30c', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Best (Epoch {best_epoch})')
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Loss', fontsize=11)
ax.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
ax.legend(frameon=False)
ax.grid(True, alpha=0.3, linewidth=0.5)
ax.set_xlim(0, 100)

# Plot 2: IoU
ax = axes[0, 1]
ax.plot(epochs, history['val_iou'], label='Val IoU', color='#1baf7a', linewidth=2)
ax.axvline(best_epoch, color='#0ca30c', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Best (Epoch {best_epoch})')
ax.axhline(history['val_iou'][best_epoch-1], color='#898781', linestyle=':', linewidth=1, alpha=0.5)
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('IoU', fontsize=11)
ax.set_title('Validation IoU', fontsize=12, fontweight='bold')
ax.legend(frameon=False)
ax.grid(True, alpha=0.3, linewidth=0.5)
ax.set_xlim(0, 100)
ax.set_ylim(0.75, 0.95)

# Plot 3: F1 Score
ax = axes[1, 0]
ax.plot(epochs, history['val_f1'], label='Val F1', color='#e87ba4', linewidth=2)
ax.axvline(best_epoch, color='#0ca30c', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Best (Epoch {best_epoch})')
ax.axhline(history['val_f1'][best_epoch-1], color='#898781', linestyle=':', linewidth=1, alpha=0.5)
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('F1 Score', fontsize=11)
ax.set_title('Validation F1 Score', fontsize=12, fontweight='bold')
ax.legend(frameon=False)
ax.grid(True, alpha=0.3, linewidth=0.5)
ax.set_xlim(0, 100)
ax.set_ylim(0.85, 0.97)

# Plot 4: Summary metrics at best epoch
ax = axes[1, 1]
ax.axis('off')
best_metrics = f"""
Best Model Performance (Epoch {best_epoch})

Validation Metrics:
  • IoU:  {history['val_iou'][best_epoch-1]:.4f}
  • F1:   {history['val_f1'][best_epoch-1]:.4f}
  • Loss: {history['val_loss'][best_epoch-1]:.4f}

Final Model Performance (Epoch 100):
  • IoU:  {history['val_iou'][-1]:.4f}
  • F1:   {history['val_f1'][-1]:.4f}
  • Loss: {history['val_loss'][-1]:.4f}

Difference (Final - Best):
  • IoU:  {history['val_iou'][-1] - history['val_iou'][best_epoch-1]:.6f}
  • F1:   {history['val_f1'][-1] - history['val_f1'][best_epoch-1]:.6f}
"""
ax.text(0.1, 0.5, best_metrics, transform=ax.transAxes, fontsize=11,
        verticalalignment='center', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#f9f9f7', alpha=0.8, edgecolor='#c3c2b7'))

plt.tight_layout()
plt.savefig('analysis/results/training_curves.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved training curves to analysis/results/training_curves.png")
plt.close()

print(f"\nBest epoch: {best_epoch}")
print(f"Best Val IoU: {history['val_iou'][best_epoch-1]:.6f}")
print(f"Best Val F1: {history['val_f1'][best_epoch-1]:.6f}")
