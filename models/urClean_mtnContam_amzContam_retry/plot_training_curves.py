import matplotlib.pyplot as plt
import numpy as np

# Parse the training log data
epochs = []
train_acc = []
train_loss = []
train_top2_acc = []
val_acc = []
val_loss = []
val_top2_acc = []
learning_rate = []

# Training data from the log
data = [
    (1, 0.8766, 0.3830, 0.9834, 0.8973, 0.2973, 0.9966, 3.0000e-04),
    (2, 0.8963, 0.3004, 0.9958, 0.8972, 0.3034, 0.9965, 3.0000e-04),
    (3, 0.8967, 0.2858, 0.9959, 0.8975, 0.3316, 0.9965, 3.0000e-04),
    (4, 0.8969, 0.2781, 0.9960, 0.8975, 0.2638, 0.9966, 3.0000e-04),
    (5, 0.8970, 0.2743, 0.9961, 0.8974, 0.2811, 0.9965, 3.0000e-04),
    (6, 0.8972, 0.2715, 0.9961, 0.8975, 0.2690, 0.9966, 3.0000e-04),
    (7, 0.8971, 0.2695, 0.9961, 0.8759, 0.4351, 0.9916, 3.0000e-04),
    (8, 0.8972, 0.2672, 0.9961, 0.8915, 0.2873, 0.9945, 3.0000e-04),
    (9, 0.8972, 0.2663, 0.9962, 0.8914, 0.4591, 0.9957, 3.0000e-04),
    (10, 0.8973, 0.2648, 0.9962, 0.8973, 0.2792, 0.9965, 3.0000e-04),
    (11, 0.8972, 0.2638, 0.9962, 0.8975, 0.2584, 0.9966, 3.0000e-04),
    (12, 0.8974, 0.2629, 0.9962, 0.8974, 0.3069, 0.9966, 3.0000e-04),
    (13, 0.8973, 0.2625, 0.9962, 0.8975, 0.2630, 0.9966, 3.0000e-04),
    (14, 0.8973, 0.2618, 0.9962, 0.8974, 0.2850, 0.9966, 3.0000e-04),
    (15, 0.8973, 0.2608, 0.9962, 0.8975, 0.2611, 0.9966, 3.0000e-04),
    (16, 0.8974, 0.2607, 0.9962, 0.8974, 0.2797, 0.9966, 3.0000e-04),
    (17, 0.8974, 0.2604, 0.9962, 0.8962, 0.2798, 0.9963, 3.0000e-04),
    (18, 0.8973, 0.2591, 0.9962, 0.8974, 0.3292, 0.9963, 3.0000e-04),
    (19, 0.8974, 0.2592, 0.9962, 0.8975, 0.3025, 0.9966, 3.0000e-04),
    (20, 0.8974, 0.2587, 0.9962, 0.8972, 0.3024, 0.9961, 3.0000e-04),
    (21, 0.8974, 0.2582, 0.9962, 0.8975, 0.2671, 0.9966, 3.0000e-04),
    (22, 0.8976, 0.2548, 0.9963, 0.8975, 0.2503, 0.9966, 1.5000e-04),
    (23, 0.8976, 0.2540, 0.9963, 0.8975, 0.2515, 0.9966, 1.5000e-04),
    (24, 0.8976, 0.2541, 0.9963, 0.8976, 0.2621, 0.9966, 1.5000e-04),
    (25, 0.8976, 0.2537, 0.9963, 0.8968, 0.2604, 0.9964, 1.5000e-04),
    (26, 0.8976, 0.2537, 0.9963, 0.8977, 0.2526, 0.9966, 1.5000e-04),
    (27, 0.8976, 0.2533, 0.9963, 0.8978, 0.2639, 0.9966, 1.5000e-04),
    (28, 0.8976, 0.2536, 0.9963, 0.8978, 0.2705, 0.9966, 1.5000e-04),
    (29, 0.8976, 0.2531, 0.9963, 0.8980, 0.2668, 0.9966, 1.5000e-04),
    (30, 0.8976, 0.2531, 0.9963, 0.8975, 0.2604, 0.9966, 1.5000e-04),
    (31, 0.8977, 0.2527, 0.9963, 0.8974, 0.2560, 0.9966, 1.5000e-04),
    (32, 0.8975, 0.2530, 0.9963, 0.8974, 0.2867, 0.9966, 1.5000e-04),
    (33, 0.8977, 0.2510, 0.9963, 0.8977, 0.2654, 0.9966, 7.5000e-05),
    (34, 0.8978, 0.2508, 0.9963, 0.8975, 0.2626, 0.9966, 7.5000e-05),
    (35, 0.8978, 0.2508, 0.9963, 0.8975, 0.2722, 0.9966, 7.5000e-05),
    (36, 0.8978, 0.2508, 0.9963, 0.8976, 0.2555, 0.9966, 7.5000e-05),
    (37, 0.8977, 0.2505, 0.9963, 0.8977, 0.2639, 0.9966, 7.5000e-05),
    (38, 0.8978, 0.2505, 0.9963, 0.8979, 0.2583, 0.9966, 7.5000e-05),
    (39, 0.8977, 0.2503, 0.9963, 0.8976, 0.2561, 0.9966, 7.5000e-05),
    (40, 0.8977, 0.2504, 0.9963, 0.8976, 0.2615, 0.9966, 7.5000e-05),
    (41, 0.8977, 0.2501, 0.9963, 0.8974, 0.2592, 0.9966, 7.5000e-05),
    (42, 0.8977, 0.2501, 0.9963, 0.8976, 0.2578, 0.9965, 7.5000e-05),
]

for row in data:
    epochs.append(row[0])
    train_acc.append(row[1])
    train_loss.append(row[2])
    train_top2_acc.append(row[3])
    val_acc.append(row[4])
    val_loss.append(row[5])
    val_top2_acc.append(row[6])
    learning_rate.append(row[7])

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Training Curves - urClean_mtnContam_amzContam_retry\nBest Model: Epoch 22 (val_loss=0.25032)',
             fontsize=14, fontweight='bold')

# Split data at epoch 22 (best model)
best_epoch_idx = 21  # 0-indexed, so epoch 22 is index 21

# Plot 1: Loss
axes[0, 0].plot(epochs[:best_epoch_idx+1], train_loss[:best_epoch_idx+1],
                color='#1f77b4', label='Train Loss', linewidth=2.5, alpha=0.8)
axes[0, 0].plot(epochs[:best_epoch_idx+1], val_loss[:best_epoch_idx+1],
                color='#ff7f0e', label='Val Loss', linewidth=2.5, alpha=0.8)
axes[0, 0].plot(epochs[best_epoch_idx:], train_loss[best_epoch_idx:],
                color='#1f77b4', linewidth=2, alpha=0.3, linestyle='--', label='Train (not used)')
axes[0, 0].plot(epochs[best_epoch_idx:], val_loss[best_epoch_idx:],
                color='#ff7f0e', linewidth=2, alpha=0.3, linestyle='--', label='Val (not used)')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Loss Curves')
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Accuracy
axes[0, 1].plot(epochs[:best_epoch_idx+1], train_acc[:best_epoch_idx+1],
                color='#1f77b4', label='Train Acc', linewidth=2.5, alpha=0.8)
axes[0, 1].plot(epochs[:best_epoch_idx+1], val_acc[:best_epoch_idx+1],
                color='#ff7f0e', label='Val Acc', linewidth=2.5, alpha=0.8)
axes[0, 1].plot(epochs[best_epoch_idx:], train_acc[best_epoch_idx:],
                color='#1f77b4', linewidth=2, alpha=0.3, linestyle='--', label='Train (not used)')
axes[0, 1].plot(epochs[best_epoch_idx:], val_acc[best_epoch_idx:],
                color='#ff7f0e', linewidth=2, alpha=0.3, linestyle='--', label='Val (not used)')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Accuracy Curves')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Top-2 Accuracy
axes[1, 0].plot(epochs[:best_epoch_idx+1], train_top2_acc[:best_epoch_idx+1],
                color='#1f77b4', label='Train Top-2 Acc', linewidth=2.5, alpha=0.8)
axes[1, 0].plot(epochs[:best_epoch_idx+1], val_top2_acc[:best_epoch_idx+1],
                color='#ff7f0e', label='Val Top-2 Acc', linewidth=2.5, alpha=0.8)
axes[1, 0].plot(epochs[best_epoch_idx:], train_top2_acc[best_epoch_idx:],
                color='#1f77b4', linewidth=2, alpha=0.3, linestyle='--', label='Train (not used)')
axes[1, 0].plot(epochs[best_epoch_idx:], val_top2_acc[best_epoch_idx:],
                color='#ff7f0e', linewidth=2, alpha=0.3, linestyle='--', label='Val (not used)')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Top-2 Accuracy')
axes[1, 0].set_title('Top-2 Accuracy Curves')
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Learning Rate
axes[1, 1].plot(epochs[:best_epoch_idx+1], learning_rate[:best_epoch_idx+1],
                color='#9467bd', linewidth=2.5, marker='o', markersize=4, alpha=0.8)
axes[1, 1].plot(epochs[best_epoch_idx:], learning_rate[best_epoch_idx:],
                color='#9467bd', linewidth=2, marker='o', markersize=3, alpha=0.3, linestyle='--')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Learning Rate')
axes[1, 1].set_title('Learning Rate Schedule')
axes[1, 1].set_yscale('log')
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('models/urClean_mtnContam_amzContam_retry/training_curves_complete.png', dpi=150, bbox_inches='tight')
print("Saved training curves to models/urClean_mtnContam_amzContam_retry/training_curves_complete.png")

# Create a summary text
print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)
print(f"Total epochs: {len(epochs)}")
print(f"Best epoch: 22")
print(f"Best val_loss: 0.25032")
print(f"Final train_acc: {train_acc[-1]:.4f}")
print(f"Final val_acc: {val_acc[-1]:.4f}")
print(f"Final train_loss: {train_loss[-1]:.4f}")
print(f"Final val_loss: {val_loss[-1]:.4f}")
print(f"Final top2_acc: {train_top2_acc[-1]:.4f}")
print(f"Final val_top2_acc: {val_top2_acc[-1]:.4f}")
print("="*60)
