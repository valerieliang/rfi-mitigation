"""
eval_late_knee.py

Evaluate trained CNN knee-index classifier on late knee test data (knees at positions 7-8).

This script:
  1. Loads the trained model from models/multi_band/best_model.keras
  2. Loads late knee test data from data/test_late_knee/
  3. Computes predictions and evaluates performance
  4. Generates comprehensive visualizations and metrics

The goal is to assess model generalization to edge cases (late knees at positions 7-8)
that appear less frequently in the training distribution (which typically has knees at 1-6).

Outputs
-------
  - Console report with per-knee metrics
  - Confusion matrices saved to models/multi_band/late_knee_*.png
  - Detailed JSON results saved to models/multi_band/late_knee_eval_results.json
  - Per-sample predictions CSV for analysis
"""

import os
import sys
import json
import numpy as np
import h5py
import tensorflow as tf
from pathlib import Path
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_gen.generate_synthetic_data import BLOCK_HEIGHT


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

MODEL_PATH = 'models/multi_band/best_model.keras'
TEST_DATA_ROOT = 'data/test_late_knee'
OUTPUT_DIR = 'models/multi_band'

M = BLOCK_HEIGHT  # 16 pulses per CPI


# ---------------------------------------------------------------------------
# FEATURE EXTRACTION (must match training pipeline)
# ---------------------------------------------------------------------------

def _compute_eigenvalues(cpi):
    """
    Compute eigenvalues from the sample covariance matrix of a CPI tile.

    Args:
        cpi (np.ndarray): Complex array shape (M, K).

    Returns:
        eigvals (np.ndarray): Real eigenvalues, descending, shape (M,).
    """
    M, K = cpi.shape
    R = (cpi @ cpi.conj().T) / K
    eigvals = np.linalg.eigvalsh(R)
    eigvals = np.sort(np.real(eigvals))[::-1]
    return eigvals


def extract_features(cpi):
    """
    Extract eigen_input and global_input feature vectors from one CPI tile.
    Eigenvalues are normalized by max eigenvalue then converted to dB scale.

    Args:
        cpi (np.ndarray): Complex array shape (M, K).

    Returns:
        eigen   (np.ndarray): shape (M, 2)  -- [eigvals_db, slopes_db]
        global_ (np.ndarray): shape (2,)    -- [condition_number_db, eff_rank]
    """
    eigvals = _compute_eigenvalues(cpi)

    # Normalize by max eigenvalue, then convert to dB
    eigvals_db = 10 * np.log10(eigvals / max(eigvals[0], 1e-12) + 1e-12)

    # Eigenvalue branch: dB eigenvalues and their slopes
    slopes = np.diff(eigvals_db)
    slopes_padded = np.append(slopes, 0.0)
    eigen = np.stack([eigvals_db, slopes_padded], axis=-1).astype(np.float32)

    # Condition number: difference in dB space (equivalent to ratio in linear space)
    cond_number_db = eigvals_db[0] - max(eigvals_db[-1], -100)

    # Effective rank via Shannon entropy of eigenvalue distribution
    p = np.maximum(eigvals, 1e-12)
    p = p / np.sum(p)
    p = p[p > 0]
    eff_rank = np.exp(-np.sum(p * np.log(p)))

    global_ = np.array([cond_number_db, eff_rank], dtype=np.float32)
    return eigen, global_


def label_from_rfi_bands(rfi_bands_json):
    """
    Derive the knee label from the 'rfi_bands' JSON attribute.

    Args:
        rfi_bands_json (str): JSON string with 'pulse_positions' and 'knee' fields.

    Returns:
        label (int): Number of distinct RFI pulse positions (0 for clean, 1-M for RFI).
    """
    payload = json.loads(rfi_bands_json)
    if payload['knee'] == 0:
        return 0
    else:
        pulse_positions = payload['pulse_positions']
        n_distinct = len(set(pulse_positions))
        return n_distinct


# ---------------------------------------------------------------------------
# DATASET LOADING
# ---------------------------------------------------------------------------

def load_test_data_from_h5(h5_path):
    """
    Load all CPI tiles from a single HDF5 file.

    Args:
        h5_path (str): Path to HDF5 file.

    Returns:
        samples (list of dict): Each dict contains:
            - 'eigen': shape (M, 2)
            - 'global': shape (2,)
            - 'label': int (knee position)
            - 'cpi_id': str (dataset name)
            - 'metadata': dict (from HDF5 attributes)
    """
    samples = []

    with h5py.File(h5_path, 'r') as f:
        # Extract file-level metadata
        file_meta = {
            'snr_db': f.attrs.get('snr_db', None),
            'n_bands_fixed': f.attrs.get('n_bands_fixed', None),
            'seed': f.attrs.get('seed', None),
        }

        # Iterate over all CPI datasets
        cpi_keys = [k for k in f.keys() if k.startswith('cpi_') and not k.endswith('_eigenvalues') and not k.endswith('_diagonal')]

        for cpi_id in cpi_keys:
            dset = f[cpi_id]
            cpi = dset[:]

            # Extract features
            eigen, global_ = extract_features(cpi)

            # Extract label from metadata
            rfi_bands_json = dset.attrs.get('rfi_bands', '{"pulse_positions": [], "knee": 0, "jnr_db_list": []}')
            label = label_from_rfi_bands(rfi_bands_json)

            samples.append({
                'eigen': eigen,
                'global': global_,
                'label': label,
                'cpi_id': cpi_id,
                'metadata': {**file_meta, 'rfi_bands': json.loads(rfi_bands_json)},
            })

    return samples


def load_all_late_knee_test_data():
    """
    Load all test data from data/test_late_knee/.

    Returns:
        data_by_knee (dict): Maps knee position (7 or 8) to list of samples.
        all_samples (list): Flat list of all samples across all files.
    """
    test_root = Path(TEST_DATA_ROOT)

    if not test_root.exists():
        raise FileNotFoundError(f"Test data directory not found: {test_root}")

    data_by_knee = defaultdict(list)
    all_samples = []

    # Iterate over knee subdirectories (knee_7, knee_8)
    for knee_dir in sorted(test_root.iterdir()):
        if not knee_dir.is_dir():
            continue

        knee_value = int(knee_dir.name.split('_')[-1])

        # Load all HDF5 files in this knee directory
        h5_files = sorted(knee_dir.glob('*.h5'))

        print(f"Loading {len(h5_files)} files from {knee_dir.name}/")

        for h5_path in h5_files:
            samples = load_test_data_from_h5(str(h5_path))
            data_by_knee[knee_value].extend(samples)
            all_samples.extend(samples)

    return dict(data_by_knee), all_samples


# ---------------------------------------------------------------------------
# EVALUATION METRICS
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_probs):
    """
    Compute classification metrics.

    Args:
        y_true (np.ndarray): True labels, shape (N,)
        y_pred (np.ndarray): Predicted labels, shape (N,)
        y_probs (np.ndarray): Predicted probabilities, shape (N, n_classes)

    Returns:
        metrics (dict): Dictionary of metric name -> value
    """
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    accuracy = accuracy_score(y_true, y_pred)

    # Top-2 accuracy: true label in top 2 predictions
    top2_preds = np.argsort(y_probs, axis=-1)[:, -2:]
    top2_acc = np.mean([y_true[i] in top2_preds[i] for i in range(len(y_true))])

    # Mean absolute error
    mae = np.mean(np.abs(y_true - y_pred))

    # Exact match rate
    exact_match = accuracy

    # Off-by-one accuracy: |true - pred| <= 1
    off_by_one = np.mean(np.abs(y_true - y_pred) <= 1)

    # Mean confidence (max softmax probability)
    mean_confidence = np.mean(np.max(y_probs, axis=-1))

    # Entropy (uncertainty measure)
    eps = 1e-12
    entropy = -np.sum(y_probs * np.log(y_probs + eps), axis=-1)
    mean_entropy = np.mean(entropy)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Per-class metrics
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    metrics = {
        'accuracy': float(accuracy),
        'top2_accuracy': float(top2_acc),
        'mae': float(mae),
        'exact_match': float(exact_match),
        'off_by_one_accuracy': float(off_by_one),
        'mean_confidence': float(mean_confidence),
        'mean_entropy': float(mean_entropy),
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': report,
    }

    return metrics


# ---------------------------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm, title, output_path):
    """
    Plot and save confusion matrix.

    Args:
        cm (np.ndarray): Confusion matrix
        title (str): Plot title
        output_path (str): Output file path
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Normalize by row (true label)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-12)

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=range(cm.shape[1]),
                yticklabels=range(cm.shape[0]),
                ax=ax, cbar_kws={'label': 'Normalized Count'})

    ax.set_xlabel('Predicted Knee Position', fontsize=12)
    ax.set_ylabel('True Knee Position', fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"  Saved confusion matrix: {output_path}")


def plot_prediction_distribution(data_by_knee, predictions_by_knee, output_path):
    """
    Plot distribution of predictions for each true knee value.

    Args:
        data_by_knee (dict): True knee -> samples
        predictions_by_knee (dict): True knee -> predictions
        output_path (str): Output file path
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (knee_val, preds) in enumerate(sorted(predictions_by_knee.items())):
        ax = axes[idx]

        pred_labels = [p['pred_label'] for p in preds]
        true_labels = [p['true_label'] for p in preds]
        confidences = [p['confidence'] for p in preds]

        # Histogram of predictions
        counts = np.bincount(pred_labels, minlength=M+1)

        bars = ax.bar(range(M+1), counts, alpha=0.7, edgecolor='black')

        # Highlight the true knee position
        bars[knee_val].set_color('red')
        bars[knee_val].set_alpha(1.0)

        ax.set_xlabel('Predicted Knee Position', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'Predictions for True Knee = {knee_val}\n'
                     f'(N={len(pred_labels)}, Acc={np.mean(np.array(pred_labels)==knee_val):.2%})',
                     fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        # Add mean confidence text
        mean_conf = np.mean(confidences)
        ax.text(0.98, 0.98, f'Mean Confidence: {mean_conf:.3f}',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('Late Knee Test: Prediction Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"  Saved prediction distribution: {output_path}")


def plot_confidence_vs_error(all_predictions, output_path):
    """
    Plot relationship between prediction confidence and error magnitude.

    Args:
        all_predictions (list of dict): All predictions
        output_path (str): Output file path
    """
    import matplotlib.pyplot as plt

    confidences = np.array([p['confidence'] for p in all_predictions])
    errors = np.abs(np.array([p['true_label'] for p in all_predictions]) -
                   np.array([p['pred_label'] for p in all_predictions]))
    correct = errors == 0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot: confidence vs error
    ax = axes[0]
    scatter = ax.scatter(confidences[correct], errors[correct],
                         alpha=0.5, s=20, c='green', label='Correct')
    scatter = ax.scatter(confidences[~correct], errors[~correct],
                         alpha=0.5, s=20, c='red', label='Incorrect')
    ax.set_xlabel('Prediction Confidence', fontsize=11)
    ax.set_ylabel('Absolute Error (|true - pred|)', fontsize=11)
    ax.set_title('Confidence vs. Prediction Error', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

    # Histogram: confidence distribution by correctness
    ax = axes[1]
    ax.hist(confidences[correct], bins=30, alpha=0.6, color='green',
            label=f'Correct (N={np.sum(correct)})')
    ax.hist(confidences[~correct], bins=30, alpha=0.6, color='red',
            label=f'Incorrect (N={np.sum(~correct)})')
    ax.set_xlabel('Prediction Confidence', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Confidence Distribution', fontsize=12)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Late Knee Test: Confidence Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"  Saved confidence analysis: {output_path}")


# ---------------------------------------------------------------------------
# MAIN EVALUATION PIPELINE
# ---------------------------------------------------------------------------

def main():
    print("\n" + "="*80)
    print("LATE KNEE TEST EVALUATION")
    print("="*80)

    # Load model
    print(f"\nLoading trained model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"  Model loaded successfully")
    print(f"  Input shapes: {[inp.shape for inp in model.inputs]}")
    print(f"  Output shape: {model.output.shape}")

    # Load test data
    print(f"\nLoading test data from: {TEST_DATA_ROOT}")
    data_by_knee, all_samples = load_all_late_knee_test_data()

    print(f"\n  Total samples: {len(all_samples)}")
    for knee_val, samples in sorted(data_by_knee.items()):
        print(f"    Knee {knee_val}: {len(samples)} samples")

    # Prepare batches
    print("\nPreparing feature batches...")
    eigen_batch = np.array([s['eigen'] for s in all_samples])
    global_batch = np.array([s['global'] for s in all_samples])
    labels_batch = np.array([s['label'] for s in all_samples])

    print(f"  eigen_batch shape: {eigen_batch.shape}")
    print(f"  global_batch shape: {global_batch.shape}")
    print(f"  labels_batch shape: {labels_batch.shape}")

    # Run inference
    print("\nRunning model inference...")
    y_probs = model.predict([eigen_batch, global_batch], batch_size=128, verbose=1)
    y_pred = np.argmax(y_probs, axis=-1)
    y_conf = np.max(y_probs, axis=-1)

    # Compute entropy
    eps = 1e-12
    y_entropy = -np.sum(y_probs * np.log(y_probs + eps), axis=-1)

    # Store predictions with metadata
    all_predictions = []
    predictions_by_knee = defaultdict(list)

    for i, sample in enumerate(all_samples):
        pred_dict = {
            'true_label': int(labels_batch[i]),
            'pred_label': int(y_pred[i]),
            'confidence': float(y_conf[i]),
            'entropy': float(y_entropy[i]),
            'cpi_id': sample['cpi_id'],
            'metadata': sample['metadata'],
        }
        all_predictions.append(pred_dict)
        predictions_by_knee[sample['label']].append(pred_dict)

    # Compute overall metrics
    print("\n" + "="*80)
    print("OVERALL METRICS")
    print("="*80)

    overall_metrics = compute_metrics(labels_batch, y_pred, y_probs)

    print(f"\n  Accuracy:           {overall_metrics['accuracy']:.4f}")
    print(f"  Top-2 Accuracy:     {overall_metrics['top2_accuracy']:.4f}")
    print(f"  MAE:                {overall_metrics['mae']:.4f}")
    print(f"  Off-by-1 Accuracy:  {overall_metrics['off_by_one_accuracy']:.4f}")
    print(f"  Mean Confidence:    {overall_metrics['mean_confidence']:.4f}")
    print(f"  Mean Entropy:       {overall_metrics['mean_entropy']:.4f}")

    # Compute per-knee metrics
    print("\n" + "="*80)
    print("PER-KNEE METRICS")
    print("="*80)

    per_knee_metrics = {}

    for knee_val, preds in sorted(predictions_by_knee.items()):
        y_true_knee = np.array([p['true_label'] for p in preds])
        y_pred_knee = np.array([p['pred_label'] for p in preds])

        # Reconstruct probability distribution (not available in pred_dict)
        indices = [i for i, s in enumerate(all_samples) if s['label'] == knee_val]
        y_probs_knee = y_probs[indices]

        metrics_knee = compute_metrics(y_true_knee, y_pred_knee, y_probs_knee)
        per_knee_metrics[knee_val] = metrics_knee

        print(f"\n  Knee {knee_val}:")
        print(f"    N samples:          {len(preds)}")
        print(f"    Accuracy:           {metrics_knee['accuracy']:.4f}")
        print(f"    Top-2 Accuracy:     {metrics_knee['top2_accuracy']:.4f}")
        print(f"    MAE:                {metrics_knee['mae']:.4f}")
        print(f"    Off-by-1 Accuracy:  {metrics_knee['off_by_one_accuracy']:.4f}")
        print(f"    Mean Confidence:    {metrics_knee['mean_confidence']:.4f}")
        print(f"    Mean Entropy:       {metrics_knee['mean_entropy']:.4f}")

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save JSON results
    results = {
        'model_path': MODEL_PATH,
        'test_data_root': TEST_DATA_ROOT,
        'n_samples': len(all_samples),
        'samples_by_knee': {k: len(v) for k, v in data_by_knee.items()},
        'overall_metrics': overall_metrics,
        'per_knee_metrics': {str(k): v for k, v in per_knee_metrics.items()},
    }

    results_path = os.path.join(OUTPUT_DIR, 'late_knee_eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved JSON results: {results_path}")

    # Save per-sample predictions CSV
    import csv
    csv_path = os.path.join(OUTPUT_DIR, 'late_knee_predictions.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['cpi_id', 'true_label', 'pred_label',
                                                'confidence', 'entropy', 'snr_db', 'seed'])
        writer.writeheader()
        for pred in all_predictions:
            writer.writerow({
                'cpi_id': pred['cpi_id'],
                'true_label': pred['true_label'],
                'pred_label': pred['pred_label'],
                'confidence': pred['confidence'],
                'entropy': pred['entropy'],
                'snr_db': pred['metadata'].get('snr_db', 'N/A'),
                'seed': pred['metadata'].get('seed', 'N/A'),
            })
    print(f"  Saved predictions CSV: {csv_path}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Overall confusion matrix
    cm_path = os.path.join(OUTPUT_DIR, 'late_knee_confusion_matrix.png')
    plot_confusion_matrix(
        np.array(overall_metrics['confusion_matrix']),
        'Late Knee Test: Overall Confusion Matrix',
        cm_path
    )

    # Per-knee confusion matrices
    for knee_val, metrics_knee in per_knee_metrics.items():
        cm_path_knee = os.path.join(OUTPUT_DIR, f'late_knee_confusion_matrix_knee{knee_val}.png')
        plot_confusion_matrix(
            np.array(metrics_knee['confusion_matrix']),
            f'Late Knee Test: Confusion Matrix (True Knee = {knee_val})',
            cm_path_knee
        )

    # Prediction distributions
    pred_dist_path = os.path.join(OUTPUT_DIR, 'late_knee_prediction_distributions.png')
    plot_prediction_distribution(data_by_knee, predictions_by_knee, pred_dist_path)

    # Confidence analysis
    conf_path = os.path.join(OUTPUT_DIR, 'late_knee_confidence_analysis.png')
    plot_confidence_vs_error(all_predictions, conf_path)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print(f"  - late_knee_eval_results.json")
    print(f"  - late_knee_predictions.csv")
    print(f"  - late_knee_confusion_matrix.png")
    print(f"  - late_knee_prediction_distributions.png")
    print(f"  - late_knee_confidence_analysis.png")
    print()


if __name__ == '__main__':
    main()
