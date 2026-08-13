#!/usr/bin/env python
"""
Comprehensive generalization test suite for UNet RFI segmentation.
Tests both in-distribution and out-of-distribution patterns with ACTUAL model predictions.

Usage:
    # 4-channel model
    python analysis/scripts/evaluate_generalization_unified.py \
        --model model/four_channel/best_model.pth \
        --n-channels 4 \
        --output model/four_channel/test/generalization_test_suite.png

    # 2-channel model
    python analysis/scripts/evaluate_generalization_unified.py \
        --model model/two_channel/best_model.pth \
        --n-channels 2 \
        --output model/two_channel/test/generalization_test_suite.png
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

# Test case definitions
TEST_CASES = [
    # IN-DISTRIBUTION (seen during training)
    {
        'name': 'Standard Gaussian Blobs',
        'category': 'IN-DIST',
        'description': '4-24 pulses × 15-90% range',
        'type': 'gaussian_blobs',
        'params': {'n_blobs': 4, 'pulse_range': (4, 24), 'range_frac': (0.15, 0.90)}
    },
    {
        'name': 'Multiple Overlapping Blobs',
        'category': 'IN-DIST',
        'description': '6-8 blobs with overlap',
        'type': 'gaussian_blobs',
        'params': {'n_blobs': 7, 'pulse_range': (8, 20), 'range_frac': (0.20, 0.70)}
    },
    {
        'name': 'Small Blob Count',
        'category': 'IN-DIST',
        'description': '1-2 isolated blobs',
        'type': 'gaussian_blobs',
        'params': {'n_blobs': 2, 'pulse_range': (6, 16), 'range_frac': (0.25, 0.60)}
    },
    {
        'name': 'Edge Blobs',
        'category': 'IN-DIST',
        'description': 'Blobs at boundaries',
        'type': 'edge_blobs',
        'params': {'n_blobs': 3}
    },
    {
        'name': 'Wide Horizontal Blob',
        'category': 'IN-DIST',
        'description': 'Large range extent (85%)',
        'type': 'gaussian_blobs',
        'params': {'n_blobs': 1, 'pulse_range': (12, 16), 'range_frac': (0.80, 0.85)}
    },

    # OUT-OF-DISTRIBUTION (NOT seen during training)
    {
        'name': 'Very Small Point Sources',
        'category': 'OOD',
        'description': '2x2 pixel spots',
        'type': 'point_sources',
        'params': {'n_points': 8, 'size': 2}
    },
    {
        'name': 'Diagonal Streaks',
        'category': 'OOD',
        'description': '45° diagonal lines',
        'type': 'diagonal_lines',
        'params': {'n_lines': 3, 'width': 3}
    },
    {
        'name': 'Full-Width Vertical Stripes',
        'category': 'OOD',
        'description': 'Narrowband spanning all pulses',
        'type': 'vertical_stripes',
        'params': {'n_stripes': 4, 'width': 3}
    },
    {
        'name': 'Full-Height Horizontal Bands',
        'category': 'OOD',
        'description': 'Wideband spanning all ranges',
        'type': 'horizontal_bands',
        'params': {'n_bands': 3, 'height': 6}
    },
    {
        'name': 'Very Large Uniform RFI',
        'category': 'OOD',
        'description': '>90% coverage',
        'type': 'large_uniform',
        'params': {'coverage': 0.65}
    },
    {
        'name': 'Thin Lines',
        'category': 'OOD',
        'description': '1-pixel vertical/horizontal',
        'type': 'thin_lines',
        'params': {'n_lines': 5}
    },
    {
        'name': 'L-Shaped Pattern',
        'category': 'OOD',
        'description': 'Geometric corner shape',
        'type': 'l_shape',
        'params': {'n_shapes': 2}
    },
    {
        'name': 'Scattered Random Pixels',
        'category': 'OOD',
        'description': 'Salt-and-pepper noise',
        'type': 'scattered',
        'params': {'density': 0.08}
    },
]

def create_gaussian_blob_mask(height, width, center_pulse, center_range,
                               pulse_size, range_size, mask_threshold=0.3, sigma_scale=3.0):
    """Create a 2D Gaussian blob (matches training data)"""
    y, x = np.ogrid[:height, :width]
    sigma_pulse = pulse_size / sigma_scale
    sigma_range = range_size / sigma_scale
    dist_sq = ((y - center_pulse) / sigma_pulse) ** 2 + ((x - center_range) / sigma_range) ** 2
    envelope = np.exp(-0.5 * dist_sq)
    mask = (envelope >= mask_threshold).astype(np.uint8)
    return mask

def generate_test_case(case, height=128, width=200, seed=42):
    """Generate ground truth mask for a test case"""
    np.random.seed(seed)
    mask = np.zeros((height, width), dtype=np.uint8)

    # Create validity mask
    valid_mask = np.ones((height, width), dtype=np.uint8)
    gap_left, gap_right = 10, 10
    valid_mask[:, :gap_left] = 0
    valid_mask[:, -gap_right:] = 0
    valid_width = width - gap_left - gap_right

    case_type = case['type']
    params = case['params']

    if case_type == 'gaussian_blobs':
        n_blobs = params['n_blobs']
        pulse_range = params['pulse_range']
        range_frac = params['range_frac']

        for _ in range(n_blobs):
            pulse_size = np.random.uniform(*pulse_range)
            rf = np.random.uniform(*range_frac)
            range_size = rf * valid_width
            center_pulse = np.random.uniform(0, height)
            center_range = np.random.uniform(gap_left, width - gap_right)

            blob = create_gaussian_blob_mask(height, width, center_pulse, center_range,
                                            pulse_size, range_size)
            mask = mask | (blob & valid_mask)

    elif case_type == 'edge_blobs':
        positions = [
            (5, gap_left + 20),
            (height - 10, width - gap_right - 30),
            (height // 2, gap_left + 5),
        ]
        for i, (cp, cr) in enumerate(positions[:params['n_blobs']]):
            blob = create_gaussian_blob_mask(height, width, cp, cr, 14, 0.4 * valid_width)
            mask = mask | (blob & valid_mask)

    elif case_type == 'point_sources':
        size = params['size']
        for _ in range(params['n_points']):
            y = np.random.randint(0, height - size)
            x = np.random.randint(gap_left, width - gap_right - size)
            mask[y:y+size, x:x+size] = 1
        mask = mask & valid_mask

    elif case_type == 'diagonal_lines':
        for i in range(params['n_lines']):
            start_x = gap_left + i * (valid_width // params['n_lines'])
            for offset in range(-params['width']//2, params['width']//2 + 1):
                for y in range(height):
                    x = start_x + y + offset
                    if gap_left <= x < width - gap_right:
                        mask[y, x] = 1

    elif case_type == 'vertical_stripes':
        for i in range(params['n_stripes']):
            x = gap_left + (i + 1) * (valid_width // (params['n_stripes'] + 1))
            w = params['width']
            mask[:, max(gap_left, x-w//2):min(width-gap_right, x+w//2+1)] = 1
        mask = mask & valid_mask

    elif case_type == 'horizontal_bands':
        for i in range(params['n_bands']):
            y = (i + 1) * (height // (params['n_bands'] + 1))
            h = params['height']
            mask[y-h//2:y+h//2+1, gap_left:width-gap_right] = 1

    elif case_type == 'large_uniform':
        coverage = params['coverage']
        h_start = int(height * (1 - coverage) / 2)
        h_end = int(height * (1 + coverage) / 2)
        w_start = gap_left + int(valid_width * (1 - coverage) / 2)
        w_end = width - gap_right - int(valid_width * (1 - coverage) / 2)
        mask[h_start:h_end, w_start:w_end] = 1

    elif case_type == 'thin_lines':
        for i in range(params['n_lines']):
            if i % 2 == 0:  # vertical
                x = gap_left + np.random.randint(0, valid_width)
                mask[:, x] = 1
            else:  # horizontal
                y = np.random.randint(0, height)
                mask[y, gap_left:width-gap_right] = 1
        mask = mask & valid_mask

    elif case_type == 'l_shape':
        for i in range(params['n_shapes']):
            y = np.random.randint(20, height - 40)
            x = gap_left + np.random.randint(20, valid_width - 40)
            arm_len = 30
            thickness = 8
            mask[y:y+arm_len, x:x+thickness] = 1
            mask[y:y+thickness, x:x+arm_len] = 1
        mask = mask & valid_mask

    elif case_type == 'scattered':
        scattered = np.random.rand(height, width) < params['density']
        mask = (scattered & valid_mask).astype(np.uint8)

    return mask, valid_mask

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
    return pred_mask

def create_error_map(ground_truth, prediction, valid_mask):
    """Create color-coded error map"""
    error_map = np.zeros_like(ground_truth, dtype=np.uint8)
    error_map[valid_mask == 0] = 4
    valid_region = valid_mask == 1
    error_map[(ground_truth == 0) & (prediction == 0) & valid_region] = 0
    error_map[(ground_truth == 1) & (prediction == 1) & valid_region] = 1
    error_map[(ground_truth == 0) & (prediction == 1) & valid_region] = 2
    error_map[(ground_truth == 1) & (prediction == 0) & valid_region] = 3
    return error_map

def calculate_metrics(ground_truth, prediction, valid_mask):
    """Calculate IoU, precision, recall, F1"""
    valid_region = valid_mask == 1
    gt = ground_truth[valid_region]
    pred = prediction[valid_region]

    tp = np.sum((gt == 1) & (pred == 1))
    fp = np.sum((gt == 0) & (pred == 1))
    fn = np.sum((gt == 1) & (pred == 0))
    tn = np.sum((gt == 0) & (pred == 0))

    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return iou, precision, recall, f1, tp, fp, fn, tn

def main():
    parser = argparse.ArgumentParser(description='Run generalization test suite on trained UNet model')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--n-channels', type=int, default=4, choices=[2, 4], help='Number of input channels')
    parser.add_argument('--output', type=str, required=True, help='Output path for visualization')
    parser.add_argument('--height', type=int, default=256, help='Test tile height (should match training size)')
    parser.add_argument('--width', type=int, default=256, help='Test tile width (should match training size)')
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

    # Generate all test cases
    print("\nGenerating comprehensive test suite...")
    results = []

    for i, case in enumerate(TEST_CASES):
        gt_mask, valid_mask = generate_test_case(case, height=args.height, width=args.width, seed=42 + i)
        tile = create_test_tile(gt_mask, valid_mask, jsr_db=10.0)
        pred_mask = predict_with_model(tile, valid_mask, model, device, args.n_channels)
        error_map = create_error_map(gt_mask, pred_mask, valid_mask)
        iou, prec, rec, f1, tp, fp, fn, tn = calculate_metrics(gt_mask, pred_mask, valid_mask)

        results.append({
            'case': case,
            'gt': gt_mask,
            'pred': pred_mask,
            'error_map': error_map,
            'valid': valid_mask,
            'metrics': {'iou': iou, 'precision': prec, 'recall': rec, 'f1': f1,
                       'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}
        })

        print(f"  {case['name']:30s} ({case['category']:7s}): IoU={iou:.3f}, F1={f1:.3f}")

    # Create visualization
    print("\nCreating visualization...")
    n_cases = len(TEST_CASES)
    n_cols = 4
    n_rows = (n_cases + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(20, n_rows * 3.5))
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.25)

    colors = ['#fcfcfb', '#0ca30c', '#d03b3b', '#ec835a', '#898781']
    cmap = ListedColormap(colors)

    for i, result in enumerate(results):
        row = i // n_cols
        col = i % n_cols

        case = result['case']
        metrics = result['metrics']

        ax = fig.add_subplot(gs[row, col])
        ax.imshow(result['error_map'], cmap=cmap, vmin=0, vmax=4, aspect='auto', interpolation='nearest')

        title_color = '#0ca30c' if case['category'] == 'IN-DIST' else '#d03b3b'
        category_label = '✓ IN-DIST' if case['category'] == 'IN-DIST' else '✗ OOD'

        ax.set_title(f"{case['name']}\n{category_label} | {case['description']}\n"
                    f"IoU: {metrics['iou']:.3f} | F1: {metrics['f1']:.3f}",
                    fontsize=9, fontweight='bold', pad=8, color=title_color)

        ax.set_xlabel('Range', fontsize=8)
        ax.set_ylabel('Pulse', fontsize=8)
        ax.tick_params(labelsize=7)

        perf_text = f"P:{metrics['precision']:.2f} R:{metrics['recall']:.2f}"
        bbox_color = '#f0f9f0' if case['category'] == 'IN-DIST' else '#fef0f0'
        ax.text(0.02, 0.98, perf_text, transform=ax.transAxes, fontsize=7,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor=bbox_color, alpha=0.9,
                        edgecolor=title_color, linewidth=1, pad=0.3))

    # Legend
    legend_elements = [
        mpatches.Patch(color='#0ca30c', label='Correct Detection (TP)'),
        mpatches.Patch(color='#d03b3b', label='False Positive'),
        mpatches.Patch(color='#ec835a', label='False Negative'),
        mpatches.Patch(facecolor='#fcfcfb', label='Correct Background', edgecolor='#898781', linewidth=1),
        mpatches.Patch(color='#898781', label='Invalid Region'),
    ]

    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
              frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.015))

    fig.suptitle(f'{args.n_channels}-Channel UNet Generalization Test Suite: In-Distribution vs Out-of-Distribution Patterns',
                fontsize=16, fontweight='bold', y=0.995)

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved visualization to {args.output}")

    # Summary statistics
    print("\n" + "="*80)
    print("GENERALIZATION ANALYSIS SUMMARY")
    print("="*80)

    in_dist_results = [r for r in results if r['case']['category'] == 'IN-DIST']
    ood_results = [r for r in results if r['case']['category'] == 'OOD']

    print(f"\nIN-DISTRIBUTION (n={len(in_dist_results)}):")
    in_dist_iou = [r['metrics']['iou'] for r in in_dist_results]
    in_dist_f1 = [r['metrics']['f1'] for r in in_dist_results]
    print(f"  Mean IoU: {np.mean(in_dist_iou):.4f} ± {np.std(in_dist_iou):.4f}")
    print(f"  Mean F1:  {np.mean(in_dist_f1):.4f} ± {np.std(in_dist_f1):.4f}")
    print(f"  Range:    IoU [{min(in_dist_iou):.3f}, {max(in_dist_iou):.3f}]")

    print(f"\nOUT-OF-DISTRIBUTION (n={len(ood_results)}):")
    ood_iou = [r['metrics']['iou'] for r in ood_results]
    ood_f1 = [r['metrics']['f1'] for r in ood_results]
    print(f"  Mean IoU: {np.mean(ood_iou):.4f} ± {np.std(ood_iou):.4f}")
    print(f"  Mean F1:  {np.mean(ood_f1):.4f} ± {np.std(ood_f1):.4f}")
    print(f"  Range:    IoU [{min(ood_iou):.3f}, {max(ood_iou):.3f}]")

    print(f"\nPERFORMANCE DROP: {np.mean(in_dist_iou) - np.mean(ood_iou):.4f} IoU points")

    print("\nWORST CASES (OOD):")
    worst_cases = sorted(ood_results, key=lambda r: r['metrics']['iou'])[:3]
    for r in worst_cases:
        print(f"  {r['case']['name']:30s}: IoU={r['metrics']['iou']:.3f}, "
              f"Recall={r['metrics']['recall']:.3f}")

    print("\nBEST OOD CASES:")
    best_ood = sorted(ood_results, key=lambda r: r['metrics']['iou'], reverse=True)[:3]
    for r in best_ood:
        print(f"  {r['case']['name']:30s}: IoU={r['metrics']['iou']:.3f}, "
              f"F1={r['metrics']['f1']:.3f}")

    print("\n" + "="*80)

if __name__ == '__main__':
    main()
