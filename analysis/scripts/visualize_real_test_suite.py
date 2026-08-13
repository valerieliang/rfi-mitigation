#!/usr/bin/env python
"""
Visualize results from real-background test suite evaluation.

Usage:
    python analysis/scripts/visualize_real_test_suite.py \\
        --test-results model/four_channel/test/test_results_real_background.npz \\
        --output model/four_channel/test/real_test_suite_visualization.png
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import argparse
import sys
from pathlib import Path

def create_error_map(ground_truth, prediction, valid_mask):
    """Create color-coded error map"""
    error_map = np.zeros_like(ground_truth, dtype=np.uint8)
    error_map[valid_mask == 0] = 4  # Invalid region
    valid_region = valid_mask == 1
    error_map[(ground_truth == 0) & (prediction == 0) & valid_region] = 0  # TN
    error_map[(ground_truth == 1) & (prediction == 1) & valid_region] = 1  # TP
    error_map[(ground_truth == 0) & (prediction == 1) & valid_region] = 2  # FP
    error_map[(ground_truth == 1) & (prediction == 0) & valid_region] = 3  # FN
    return error_map

def main():
    parser = argparse.ArgumentParser(description='Visualize real-background test suite results')
    parser.add_argument('--test-results', type=str, required=True,
                        help='Path to test results (.npz)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output visualization path (.png)')
    parser.add_argument('--dpi', type=int, default=150, help='Figure DPI')
    args = parser.parse_args()

    # Load results
    print(f"Loading results from {args.test_results}...")
    data = np.load(args.test_results, allow_pickle=True)

    tiles = data['tiles']
    gt_masks = data['gt_masks']
    valid_masks = data['valid_masks']
    predictions = data['predictions']
    names = data['names']
    categories = data['categories']
    iou_scores = data['iou']
    f1_scores = data['f1']
    jsr_values = data.get('jsr_values', None)  # May not exist in older files

    n_cases = len(tiles)
    print(f"  Loaded {n_cases} test cases")

    # Create figure with subplots
    n_cols = 4
    n_rows = (n_cases + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    fig.suptitle('Real-Background Generalization Test Suite', fontsize=16, y=0.995)

    # Flatten axes for easier indexing
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    # Color map
    colors = [
        '#FFFFFF',  # 0: TN (Correct Background)
        '#2ECC71',  # 1: TP (Correct Detection)
        '#E74C3C',  # 2: FP (False Positive)
        '#F39C12',  # 3: FN (False Negative / Miss)
        '#95A5A6',  # 4: Invalid Region
    ]
    cmap = ListedColormap(colors)

    # Plot each test case
    for i in range(n_cases):
        ax = axes[i]

        # Create error map
        error_map = create_error_map(gt_masks[i], predictions[i], valid_masks[i])

        # Plot
        ax.imshow(error_map, cmap=cmap, vmin=0, vmax=4, aspect='auto', interpolation='nearest')

        # Title with metrics
        category_marker = 'x' if categories[i] == 'IN-DIST' else '✗'
        title = f"{category_marker} {categories[i]} | {names[i]}\n"
        if jsr_values is not None:
            title += f"JSR: {jsr_values[i]:.1f} dB | IoU: {iou_scores[i]:.3f} | F1: {f1_scores[i]:.3f}"
        else:
            title += f"IoU: {iou_scores[i]:.3f} | F1: {f1_scores[i]:.3f}"
        ax.set_title(title, fontsize=9, pad=4)

        ax.set_xlabel('Range', fontsize=8)
        ax.set_ylabel('Pulse', fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for i in range(n_cases, len(axes)):
        axes[i].axis('off')

    # Create legend
    legend_elements = [
        mpatches.Patch(facecolor=colors[1], label='Correct Detection (TP)'),
        mpatches.Patch(facecolor=colors[2], label='False Positive'),
        mpatches.Patch(facecolor=colors[3], label='False Negative (Miss)'),
        mpatches.Patch(facecolor=colors[0], label='Correct Background (TN)'),
        mpatches.Patch(facecolor=colors[4], label='Invalid Region'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=5, fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.02, 1, 0.99])

    # Save
    print(f"Saving visualization to {args.output}...")
    plt.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
    plt.close()

    print("\nVisualization complete!")

    # Print summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")

    in_dist_idx = categories == 'IN-DIST'
    ood_idx = categories == 'OOD'

    if in_dist_idx.any():
        print(f"\nIN-DISTRIBUTION (n={in_dist_idx.sum()}):")
        print(f"  Mean IoU: {iou_scores[in_dist_idx].mean():.4f} ± {iou_scores[in_dist_idx].std():.4f}")
        print(f"  Mean F1:  {f1_scores[in_dist_idx].mean():.4f} ± {f1_scores[in_dist_idx].std():.4f}")
        print(f"  Range:    IoU [{iou_scores[in_dist_idx].min():.3f}, {iou_scores[in_dist_idx].max():.3f}]")

    if ood_idx.any():
        print(f"\nOUT-OF-DISTRIBUTION (n={ood_idx.sum()}):")
        print(f"  Mean IoU: {iou_scores[ood_idx].mean():.4f} ± {iou_scores[ood_idx].std():.4f}")
        print(f"  Mean F1:  {f1_scores[ood_idx].mean():.4f} ± {f1_scores[ood_idx].std():.4f}")
        print(f"  Range:    IoU [{iou_scores[ood_idx].min():.3f}, {iou_scores[ood_idx].max():.3f}]")

        # Best and worst cases
        print(f"\n  WORST CASES (OOD):")
        worst_idx = np.argsort(iou_scores[ood_idx])[:3]
        for idx in worst_idx:
            ood_names = names[ood_idx]
            ood_ious = iou_scores[ood_idx]
            print(f"    {ood_names[idx]:30s}: IoU={ood_ious[idx]:.3f}")

        print(f"\n  BEST OOD CASES:")
        best_idx = np.argsort(iou_scores[ood_idx])[-3:][::-1]
        for idx in best_idx:
            ood_names = names[ood_idx]
            ood_ious = iou_scores[ood_idx]
            print(f"    {ood_names[idx]:30s}: IoU={ood_ious[idx]:.3f}")

    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
