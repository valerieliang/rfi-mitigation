#!/usr/bin/env python
"""
Create sample segmentation visualizations with color-coded errors using ACTUAL model predictions.
Using REALISTIC 2D Gaussian blobs that match actual training data.

Usage:
    # 4-channel model
    python analysis/scripts/visualize_segmentation_unified.py \
        --model model/four_channel/best_model.pth \
        --test-results model/four_channel/test_results.npz \
        --n-channels 4 \
        --output-dir model/four_channel/test

    # 2-channel model
    python analysis/scripts/visualize_segmentation_unified.py \
        --model model/two_channel/best_model.pth \
        --test-results model/two_channel/test_results.npz \
        --n-channels 2 \
        --output-dir model/two_channel/test
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from scipy import ndimage
import argparse
import sys
from pathlib import Path

# Add root directory to Python path to import unet and input_transforms
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def create_gaussian_blob_mask(height, width, center_pulse, center_range,
                               pulse_size, range_size, mask_threshold=0.3, sigma_scale=3.0):
    """Create a 2D Gaussian blob mask matching the training data generator."""
    y, x = np.ogrid[:height, :width]
    sigma_pulse = pulse_size / sigma_scale
    sigma_range = range_size / sigma_scale
    dist_sq = ((y - center_pulse) / sigma_pulse) ** 2 + ((x - center_range) / sigma_range) ** 2
    envelope = np.exp(-0.5 * dist_sq)
    mask = (envelope >= mask_threshold).astype(np.uint8)
    return mask, envelope

def create_realistic_sample(height=256, width=256, n_blobs=None, seed=None):
    """Create a realistic RFI mask with 2D Gaussian blobs matching training data."""
    if seed is not None:
        np.random.seed(seed)

    if n_blobs is None:
        n_blobs = np.random.randint(0, 9)

    mask = np.zeros((height, width), dtype=np.uint8)
    valid_mask = np.ones((height, width), dtype=np.uint8)
    gap_left = np.random.randint(5, 15)
    gap_right = np.random.randint(5, 15)
    valid_mask[:, :gap_left] = 0
    valid_mask[:, -gap_right:] = 0
    valid_width = width - gap_left - gap_right

    valid_pixels = np.sum(valid_mask)
    max_contaminated = int(0.30 * valid_pixels)
    current_contaminated = 0

    actual_blobs = 0
    for _ in range(n_blobs):
        if current_contaminated >= max_contaminated:
            break

        pulse_size = np.random.uniform(4, 24)
        range_frac = np.random.uniform(0.15, 0.90)
        range_size = range_frac * valid_width

        center_pulse = np.random.uniform(-pulse_size, height + pulse_size)
        center_range = np.random.uniform(gap_left - range_size/2, width - gap_right + range_size/2)

        blob_mask, envelope = create_gaussian_blob_mask(
            height, width, center_pulse, center_range,
            pulse_size, range_size, mask_threshold=0.3, sigma_scale=3.0
        )

        blob_mask = blob_mask & valid_mask
        new_contamination = np.sum(blob_mask & (mask == 0))
        if current_contaminated + new_contamination > max_contaminated:
            break

        mask = mask | blob_mask
        current_contaminated += new_contamination
        actual_blobs += 1

    return mask, valid_mask, actual_blobs

def create_test_tile(gt_mask, valid_mask, jsr_db=10.0):
    """Create complex test tile with RFI injected."""
    height, width = gt_mask.shape
    tile = (np.random.randn(height, width) + 1j * np.random.randn(height, width)).astype(np.complex64) * 0.1
    rfi_power = 10 ** (jsr_db / 20.0)
    for i in range(height):
        for j in range(width):
            if gt_mask[i, j]:
                tile[i, j] += rfi_power * (np.random.randn() + 1j * np.random.randn())
    return tile

def predict_with_model(tile, valid_mask, model, device, n_channels):
    """Run inference on tile using the actual model."""
    import torch
    from input_transforms import build_input_channels

    input_channels = build_input_channels(tile, valid_mask.astype(bool), n_channels=n_channels)
    input_tensor = torch.from_numpy(input_channels).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()

    pred_mask = (pred_prob >= 0.5).astype(np.uint8)
    return pred_mask, pred_prob

def create_error_map(ground_truth, prediction, valid_mask):
    """Create a color-coded error map"""
    error_map = np.zeros_like(ground_truth, dtype=np.uint8)
    error_map[valid_mask == 0] = 4
    valid_region = valid_mask == 1
    error_map[(ground_truth == 0) & (prediction == 0) & valid_region] = 0
    error_map[(ground_truth == 1) & (prediction == 1) & valid_region] = 1
    error_map[(ground_truth == 0) & (prediction == 1) & valid_region] = 2
    error_map[(ground_truth == 1) & (prediction == 0) & valid_region] = 3
    return error_map

def calculate_metrics(ground_truth, prediction, valid_mask):
    """Calculate IoU, precision, recall, F1 on valid region"""
    valid_region = valid_mask == 1
    gt_valid = ground_truth[valid_region]
    pred_valid = prediction[valid_region]

    tp = np.sum((gt_valid == 1) & (pred_valid == 1))
    fp = np.sum((gt_valid == 0) & (pred_valid == 1))
    fn = np.sum((gt_valid == 1) & (pred_valid == 0))
    tn = np.sum((gt_valid == 0) & (pred_valid == 0))

    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return iou, precision, recall, f1, tp, fp, fn, tn

def main():
    parser = argparse.ArgumentParser(description='Visualize segmentation samples with actual model predictions')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--test-results', type=str, required=True, help='Path to test_results.npz')
    parser.add_argument('--n-channels', type=int, default=4, choices=[2, 4], help='Number of input channels')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--height', type=int, default=256, help='Sample tile height')
    parser.add_argument('--width', type=int, default=256, help='Sample tile width')
    args = parser.parse_args()

    # Load model
    print(f"Loading {args.n_channels}-channel model from {args.model}...")
    try:
        import torch
        from unet import UNet

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = UNet(in_channels=args.n_channels, out_channels=1, features=[64, 128, 256, 512])
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.to(device)
        model.eval()
        print(f"  Model loaded on {device}")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        sys.exit(1)

    # Load test results
    test_results = np.load(args.test_results)
    print(f"\nTest set performance:")
    print(f"  IoU:       {test_results['iou']:.4f}")
    print(f"  F1:        {test_results['f1']:.4f}")
    print(f"  Precision: {test_results['precision']:.4f}")
    print(f"  Recall:    {test_results['recall']:.4f}")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # Visualization: 6 Sample Grid
    # ========================================================================
    print("\nGenerating 6-sample segmentation visualization...")

    np.random.seed(42)
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.25)

    colors = ['#fcfcfb', '#0ca30c', '#d03b3b', '#ec835a', '#898781']
    cmap = ListedColormap(colors)

    blob_counts = [2, 5, 3, 4, 6, 1]

    for i in range(6):
        row = i // 3
        col = i % 3

        gt_mask, valid_mask, n_blobs = create_realistic_sample(
            height=args.height, width=args.width, n_blobs=blob_counts[i], seed=42+i
        )

        if np.sum(gt_mask) == 0:
            gt_mask, valid_mask, n_blobs = create_realistic_sample(
                height=args.height, width=args.width, n_blobs=3, seed=100+i
            )

        # Create tile and predict
        tile = create_test_tile(gt_mask, valid_mask, jsr_db=10.0)
        pred_mask, pred_prob = predict_with_model(tile, valid_mask, model, device, args.n_channels)

        iou, precision, recall, f1, tp, fp, fn, tn = calculate_metrics(gt_mask, pred_mask, valid_mask)
        error_map = create_error_map(gt_mask, pred_mask, valid_mask)

        ax = fig.add_subplot(gs[row, col])
        ax.imshow(error_map, cmap=cmap, vmin=0, vmax=4, aspect='auto', interpolation='nearest')

        ax.set_title(f'Sample {i+1} ({n_blobs} blobs)\\nIoU: {iou:.3f} | F1: {f1:.3f} | Prec: {precision:.3f} | Rec: {recall:.3f}',
                     fontsize=10, fontweight='bold', pad=10)

        ax.set_xlabel('Range Sample', fontsize=9)
        ax.set_ylabel('Pulse', fontsize=9)
        ax.tick_params(labelsize=8)

        error_text = f'TP:{tp} FP:{fp} FN:{fn}'
        ax.text(0.02, 0.98, error_text, transform=ax.transAxes,
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#c3c2b7', pad=0.3))

    # Legend
    legend_elements = [
        mpatches.Patch(color='#0ca30c', label='Correct Detection (TP)'),
        mpatches.Patch(color='#d03b3b', label='False Positive'),
        mpatches.Patch(color='#ec835a', label='False Negative (Miss)'),
        mpatches.Patch(facecolor='#fcfcfb', label='Correct Background (TN)', edgecolor='#898781', linewidth=1),
        mpatches.Patch(color='#898781', label='Invalid Region'),
    ]

    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
              frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f'{args.n_channels}-Channel UNet Segmentation Results: Realistic 2D Gaussian Blob Patterns',
                fontsize=16, fontweight='bold', y=0.98)

    fig.text(0.5, 0.04,
             f'Model Test Performance: IoU={test_results["iou"]:.4f}, F1={test_results["f1"]:.4f}, '
             f'Precision={test_results["precision"]:.4f}, Recall={test_results["recall"]:.4f}',
             ha='center', fontsize=10, fontweight='bold', color='#0b0b0b')

    channel_str = 'mag_dB, cos_phase, sin_phase, valid' if args.n_channels == 4 else 'mag_dB, valid'
    fig.text(0.5, 0.01,
             f'Input: [{channel_str}] | 2D Gaussian blobs (4-24 pulses × 15-90% range)',
             ha='center', fontsize=9, style='italic', color='#52514e')

    output_file = Path(args.output_dir) / 'segmentation_samples_gaussian.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_file}")
    plt.close()

    # ========================================================================
    # Visualization: Single Detailed Sample
    # ========================================================================
    print("\nGenerating detailed single sample visualization...")

    np.random.seed(123)
    gt_mask, valid_mask, n_blobs = create_realistic_sample(
        height=args.height, width=args.width, n_blobs=5, seed=123
    )
    tile = create_test_tile(gt_mask, valid_mask, jsr_db=10.0)
    pred_mask, pred_prob = predict_with_model(tile, valid_mask, model, device, args.n_channels)
    iou, precision, recall, f1, tp, fp, fn, tn = calculate_metrics(gt_mask, pred_mask, valid_mask)
    error_map = create_error_map(gt_mask, pred_mask, valid_mask)

    fig = plt.figure(figsize=(18, 6))

    # Ground truth
    ax1 = plt.subplot(1, 4, 1)
    ax1.imshow(gt_mask * valid_mask, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='auto')
    ax1.set_title(f'Ground Truth\\n{n_blobs} Gaussian blobs', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Range Sample')
    ax1.set_ylabel('Pulse')

    # Prediction
    ax2 = plt.subplot(1, 4, 2)
    ax2.imshow(pred_mask * valid_mask, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='auto')
    ax2.set_title(f'{args.n_channels}-Ch UNet Prediction', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Range Sample')
    ax2.set_ylabel('Pulse')

    # Probability
    ax3 = plt.subplot(1, 4, 3)
    prob_masked = np.where(valid_mask, pred_prob, np.nan)
    im3 = ax3.imshow(prob_masked, cmap='plasma', vmin=0, vmax=1, aspect='auto')
    ax3.set_title('RFI Probability', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Range Sample')
    ax3.set_ylabel('Pulse')
    plt.colorbar(im3, ax=ax3, label='Probability')

    # Error map
    ax4 = plt.subplot(1, 4, 4)
    ax4.imshow(error_map, cmap=cmap, vmin=0, vmax=4, aspect='auto', interpolation='nearest')
    ax4.set_title(f'Error Map\\nIoU={iou:.3f}, F1={f1:.3f}\\nTP:{tp} FP:{fp} FN:{fn}',
                  fontsize=12, fontweight='bold')
    ax4.set_xlabel('Range Sample')
    ax4.set_ylabel('Pulse')

    fig.legend(handles=legend_elements, loc='lower center', ncol=5, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f'{args.n_channels}-Channel UNet: Detailed Segmentation Sample', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = Path(args.output_dir) / 'segmentation_samples.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_file}")
    plt.close()

    print("\nAll visualizations complete!")

if __name__ == '__main__':
    main()
