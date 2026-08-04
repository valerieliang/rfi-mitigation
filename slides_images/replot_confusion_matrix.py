"""
Replot confusion matrix from synthetic test results with better label formatting.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import os

# Load results
results_path = 'score/urClean_mtnContam_amzContam/synthetic_test/results_combined.json'
with open(results_path, 'r') as f:
    results = json.load(f)

cm = np.array(results['confusion_matrix'])
class_names = results['class_names']
n_classes = results['n_classes']
acc = results['overall_accuracy']

# Plot confusion matrix with better formatting
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(cm, cmap='Blues', aspect='auto')
ax.set_xticks(range(n_classes))
ax.set_yticks(range(n_classes))
ax.set_xticklabels(class_names, rotation=45, ha='right')
ax.set_yticklabels(class_names)
ax.set_xlabel('Predicted', fontsize=12, labelpad=10)
ax.set_ylabel('True', fontsize=12, labelpad=10)
ax.set_title(f'Combined Confusion Matrix (Accuracy: {100*acc:.2f}%)', fontsize=14, pad=15)

# Annotate cells
for i in range(n_classes):
    for j in range(n_classes):
        text = ax.text(j, i, str(cm[i, j]),
                      ha='center', va='center',
                      color='white' if cm[i, j] > cm.max()/2 else 'black',
                      fontsize=10)

plt.colorbar(im, ax=ax)
fig.tight_layout()

# Save figure
out_dir = 'score/urClean_mtnContam_amzContam/synthetic_test'
out_path = os.path.join(out_dir, 'confusion_matrix_combined_fixed.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved {out_path}")

plt.show()