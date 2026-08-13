#!/usr/bin/env python
"""
Evaluate UNet model on the real-background test suite.

Usage:
    python analysis/scripts/evaluate_on_real_test_suite.py \\
        --model model/four_channel/best_model.pth \\
        --test-suite model/test_suite_real_background.npz \\
        --n-channels 4 \\
        --output model/four_channel/test/test_results_real_background.npz
"""
import numpy as np
import argparse
import sys
from pathlib import Path

# Add root directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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

    return {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate model on real-background test suite')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--test-suite', type=str, required=True, help='Path to test suite (.npz)')
    parser.add_argument('--n-channels', type=int, default=4, choices=[2, 4], help='Number of input channels')
    parser.add_argument('--output', type=str, required=True, help='Output path for results (.npz)')
    args = parser.parse_args()

    # Load model
    print(f"Loading {args.n_channels}-channel model from {args.model}...")
    try:
        import torch
        from unet import UNet

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = UNet(in_channels=args.n_channels, out_channels=1, features=[64, 128, 256, 512])
        model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        print(f"  Model loaded on {device}")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        sys.exit(1)

    # Load test suite
    print(f"\nLoading test suite from {args.test_suite}...")
    try:
        data = np.load(args.test_suite, allow_pickle=True)
        tiles = data['tiles']
        gt_masks = data['masks']
        valid_masks = data['valid']
        names = data['names']
        categories = data['categories']
        descriptions = data['descriptions']
        jsr_values = data.get('jsr_values', None)
        print(f"  Loaded {len(tiles)} test cases")
    except Exception as e:
        print(f"ERROR: Failed to load test suite: {e}")
        sys.exit(1)

    # Run predictions
    print(f"\nRunning predictions...")
    predictions = []
    probabilities = []
    metrics = []

    for i, (tile, gt_mask, valid_mask, name, category) in enumerate(
            zip(tiles, gt_masks, valid_masks, names, categories)):

        pred_mask, pred_prob = predict_with_model(tile, valid_mask, model, device, args.n_channels)
        pred_metrics = calculate_metrics(gt_mask, pred_mask, valid_mask)

        predictions.append(pred_mask)
        probabilities.append(pred_prob)
        metrics.append(pred_metrics)

        jsr_str = f"JSR: {jsr_values[i]:4.1f} dB, " if jsr_values is not None else ""
        print(f"  [{i+1:2d}/{len(tiles)}] {name:35s} ({category:7s}) - "
              f"{jsr_str}IoU: {pred_metrics['iou']:.3f}, F1: {pred_metrics['f1']:.3f}, "
              f"Prec: {pred_metrics['precision']:.3f}, Rec: {pred_metrics['recall']:.3f}")

    # Calculate overall statistics
    in_dist_metrics = [m for m, c in zip(metrics, categories) if c == 'IN-DIST']
    ood_metrics = [m for m, c in zip(metrics, categories) if c == 'OOD']

    print(f"\n{'='*80}")
    print("OVERALL RESULTS")
    print(f"{'='*80}")

    if in_dist_metrics:
        in_dist_iou = np.mean([m['iou'] for m in in_dist_metrics])
        in_dist_f1 = np.mean([m['f1'] for m in in_dist_metrics])
        in_dist_prec = np.mean([m['precision'] for m in in_dist_metrics])
        in_dist_rec = np.mean([m['recall'] for m in in_dist_metrics])
        print(f"\nIN-DISTRIBUTION (n={len(in_dist_metrics)}):")
        print(f"  Mean IoU:       {in_dist_iou:.4f}")
        print(f"  Mean F1:        {in_dist_f1:.4f}")
        print(f"  Mean Precision: {in_dist_prec:.4f}")
        print(f"  Mean Recall:    {in_dist_rec:.4f}")

    if ood_metrics:
        ood_iou = np.mean([m['iou'] for m in ood_metrics])
        ood_f1 = np.mean([m['f1'] for m in ood_metrics])
        ood_prec = np.mean([m['precision'] for m in ood_metrics])
        ood_rec = np.mean([m['recall'] for m in ood_metrics])
        print(f"\nOUT-OF-DISTRIBUTION (n={len(ood_metrics)}):")
        print(f"  Mean IoU:       {ood_iou:.4f}")
        print(f"  Mean F1:        {ood_f1:.4f}")
        print(f"  Mean Precision: {ood_prec:.4f}")
        print(f"  Mean Recall:    {ood_rec:.4f}")

        if in_dist_metrics:
            print(f"\nPERFORMANCE DROP: {ood_iou - in_dist_iou:+.4f} IoU points")

    # Save results
    print(f"\nSaving results to {args.output}...")
    save_dict = {
        # Input data
        'tiles': tiles,
        'gt_masks': gt_masks,
        'valid_masks': valid_masks,
        'names': names,
        'categories': categories,
        'descriptions': descriptions,
        # Predictions
        'predictions': np.array(predictions, dtype=bool),
        'probabilities': np.array(probabilities, dtype=np.float32),
        # Metrics
        'iou': np.array([m['iou'] for m in metrics]),
        'precision': np.array([m['precision'] for m in metrics]),
        'recall': np.array([m['recall'] for m in metrics]),
        'f1': np.array([m['f1'] for m in metrics]),
        'tp': np.array([m['tp'] for m in metrics]),
        'fp': np.array([m['fp'] for m in metrics]),
        'fn': np.array([m['fn'] for m in metrics]),
        'tn': np.array([m['tn'] for m in metrics]),
        # Model info
        'model_path': args.model,
        'n_channels': args.n_channels
    }

    # Add JSR values if available
    if jsr_values is not None:
        save_dict['jsr_values'] = jsr_values

    np.savez_compressed(args.output, **save_dict)

    print(f"\nEvaluation complete!")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
