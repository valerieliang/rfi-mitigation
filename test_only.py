"""
test_only.py

Evaluate a trained RFI knee classifier on SYNTHETIC test data with known
ground-truth labels. Loads one or more test directories (each treated as a
separate dataset/source), runs the model, and reports a combined confusion
matrix + accuracy plus a per-dataset breakdown -- so you can see mountain-only
vs amazon-only vs combined for the same model.

This script is ONLY for labeled synthetic data. Scoring real NISAR scenes
(no ground truth) is done in score_scene.py.

Usage
-----
    python test_only.py \\
        --model models/<run>/best_model.keras \\
        --data-dirs data/mountain_test data/amazon_test \\
        --output-dir results/<run>_test
"""

import os
import sys
import json
import argparse

import numpy as np
import h5py


# ===========================================================================
# CONSTANTS
# ===========================================================================

EPS = 1e-12
M = 16                 # pulses per CPI = number of eigenvalues available
N_KEEP = 12            # eigenvalues actually used as features (the 12 largest)
N_GLOBAL = 3           # global features: [condition_number_db, eff_rank, diag_median_max_ratio]
DB_FLOOR = -100.0      # floor for dB values


# ===========================================================================
# FEATURE EXTRACTION (from train_only.py)
# ===========================================================================

def diag_median_max_ratio_feature(diag_lin, diag_valid_idx):
    """
    Ratio of the median VALID SCM diagonal entry to the max VALID entry, per
    tile, on the LINEAR scale.
    """
    single = (np.asarray(diag_lin).ndim == 1)
    diag = np.atleast_2d(np.asarray(diag_lin, dtype=np.float64))
    valid = np.atleast_2d(np.asarray(diag_valid_idx, dtype=bool))

    masked = np.where(valid, diag, np.nan)
    n_valid = valid.sum(axis=1)

    with np.errstate(invalid='ignore'):
        vmax = np.nanmax(masked, axis=1)
        vmed = np.nanmedian(masked, axis=1)

    ratio = np.where(n_valid >= 2, vmed / np.maximum(vmax, EPS), 1.0)
    ratio = ratio.astype(np.float32)

    if single:
        return float(ratio[0])
    return ratio


def features_from_eigenvalues(eigvals_linear, diag_lin, diag_valid_idx):
    """
    Build the eigen and global feature tensors from LINEAR eigenvalues plus
    the LINEAR SCM diagonal.

    This matches the signature from train_only.py and generate_amazon_data.py.
    """
    single = (np.asarray(eigvals_linear).ndim == 1)
    ev = np.atleast_2d(np.asarray(eigvals_linear, dtype=np.float64))

    # 1. Keep the 12 largest, drop the 4 dithering-exposed smallest
    ev = ev[:, :N_KEEP]

    # 2. Linear normalization by lambda_max (per tile)
    ev = np.maximum(ev, EPS)
    lam_max = np.maximum(ev[:, :1], EPS)
    ev_norm = ev / lam_max

    # 3. dB
    ev_db = 10.0 * np.log10(np.maximum(ev_norm, EPS))
    ev_db = np.maximum(ev_db, DB_FLOOR)

    # 4. Slopes, zero-padded so the tensor stays (N_KEEP, 2)
    slopes = np.diff(ev_db, axis=1)
    slopes = np.concatenate([slopes, np.zeros((ev_db.shape[0], 1))], axis=1)
    eigen = np.stack([ev_db, slopes], axis=-1).astype(np.float32)

    # 5. Global features, both over the kept 12 only
    cond_db = ev_db[:, 0] - np.maximum(ev_db[:, -1], DB_FLOOR)

    p = ev / np.maximum(ev.sum(axis=1, keepdims=True), EPS)
    p = np.maximum(p, EPS)
    eff_rank = np.exp(-np.sum(p * np.log(p), axis=1))

    # 6. Diagonal median/max ratio, over all valid rows
    diag_ratio = diag_median_max_ratio_feature(diag_lin, diag_valid_idx)
    diag_ratio = np.atleast_1d(np.asarray(diag_ratio, dtype=np.float64))

    global_ = np.stack([cond_db, eff_rank, diag_ratio], axis=-1).astype(np.float32)

    if single:
        return eigen[0], global_[0]
    return eigen, global_


# ===========================================================================
# MODE 1: COMBINED SYNTHETIC TEST
# ===========================================================================

def load_synthetic_data(data_dir):
    """Load synthetic test data from a data directory."""
    # Try standard test.h5 first (preprocessed features)
    test_file = os.path.join(data_dir, 'test.h5')
    if os.path.exists(test_file):
        with h5py.File(test_file, 'r') as f:
            eigen = f['eigen_features'][:]
            global_feat = f['global_features'][:]
            labels = f['labels'][:]
            if 'jsr_db' in f:
                jsr_db = f['jsr_db'][:]
            else:
                jsr_db = None
        return {
            'eigen': eigen,
            'global': global_feat,
            'labels': labels,
            'jsr_db': jsr_db,
            'source': os.path.basename(data_dir),
        }

    # Otherwise, look for separate polarization files with raw eigenvalues
    h5_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.h5')])
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {data_dir}")

    all_eigen, all_global, all_labels, all_jsr = [], [], [], []
    for h5_file in h5_files:
        fpath = os.path.join(data_dir, h5_file)
        with h5py.File(fpath, 'r') as f:
            # Load raw eigenvalues and preprocess them
            eigvals = f['eigenvalues'][:]
            diag = f['diagonal'][:]
            diag_valid_idx = f['diag_valid_idx'][:]
            labels_pol = f['labels'][:]

            # Preprocess raw eigenvalues into model features
            eigen_feat, global_feat = features_from_eigenvalues(
                eigvals, diag, diag_valid_idx
            )

            all_eigen.append(eigen_feat)
            all_global.append(global_feat)
            all_labels.append(labels_pol)

            if 'jsr_db' in f:
                all_jsr.append(f['jsr_db'][:])

    eigen = np.concatenate(all_eigen, axis=0)
    global_feat = np.concatenate(all_global, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # JSR: take max across bands (6 bands per sample, some may be NaN)
    if all_jsr:
        jsr_all_bands = np.concatenate(all_jsr, axis=0)
        jsr_db = np.nanmax(jsr_all_bands, axis=1)  # (N,) - max JSR per sample
    else:
        jsr_db = None

    return {
        'eigen': eigen,
        'global': global_feat,
        'labels': labels,
        'jsr_db': jsr_db,
        'source': os.path.basename(data_dir),
    }


def combined_synthetic_test(model, data_dirs, args):
    """
    Test model on synthetic data from multiple directories.
    Produces combined confusion matrix and accuracy vs JSR.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, classification_report

    print(f"\n{'='*70}")
    print('COMBINED SYNTHETIC TEST')
    print(f"{'='*70}")
    print(f"Data directories: {data_dirs}")

    all_data = [load_synthetic_data(d) for d in data_dirs]

    # Combine all datasets
    eigen_all = np.concatenate([d['eigen'] for d in all_data])
    global_all = np.concatenate([d['global'] for d in all_data])
    labels_all = np.concatenate([d['labels'] for d in all_data])

    # Track which dataset each sample came from
    dataset_ids = np.concatenate([
        np.full(len(d['labels']), i) for i, d in enumerate(all_data)
    ])

    # JSR (if available)
    has_jsr = all(d['jsr_db'] is not None for d in all_data)
    if has_jsr:
        jsr_all = np.concatenate([d['jsr_db'] for d in all_data])
    else:
        jsr_all = None

    print(f"\nTotal samples: {len(labels_all)}")
    for i, d in enumerate(all_data):
        print(f"  {d['source']}: {len(d['labels'])} samples")

    # Predict
    print("\nPredicting...")
    probs = model.predict([eigen_all, global_all], batch_size=args.batch_size, verbose=1)
    preds = np.argmax(probs, axis=-1)
    confidence = np.max(probs, axis=-1)

    n_classes = probs.shape[-1]

    # Overall accuracy
    acc = np.mean(preds == labels_all)
    print(f"\nOverall Accuracy: {100*acc:.2f}%")

    # Per-dataset accuracy
    print("\nPer-dataset accuracy:")
    for i, d in enumerate(all_data):
        mask = (dataset_ids == i)
        acc_i = np.mean(preds[mask] == labels_all[mask])
        print(f"  {d['source']}: {100*acc_i:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(labels_all, preds, labels=range(n_classes))

    # Classification report
    print("\nClassification Report:")
    class_names = ['clean'] + [(f'{k} RFI eigenvalue' if k == 1 else f'{k} RFI eigenvalues')
                                 for k in range(1, n_classes)]
    print(classification_report(labels_all, preds, target_names=class_names, digits=3))

    # Save results. Confusion matrices and per-class precision/recall/f1 are
    # stored for the COMBINED set and for EACH dataset (source), so the
    # mountain-only / amazon-only / combined metrics are all machine-readable.
    combined_report = classification_report(
        labels_all, preds, labels=range(n_classes),
        target_names=class_names, digits=3, output_dict=True, zero_division=0)

    results = {
        'mode': 'combined_synthetic',
        'data_dirs': data_dirs,
        'n_classes': n_classes,
        'class_names': class_names,
        'total_samples': int(len(labels_all)),
        'per_dataset_samples': {d['source']: int(len(d['labels'])) for d in all_data},
        'overall_accuracy': float(acc),
        'per_dataset_accuracy': {
            d['source']: float(np.mean(preds[dataset_ids == i] == labels_all[dataset_ids == i]))
            for i, d in enumerate(all_data)
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': combined_report,
        'per_dataset': {},   # populated in the per-dataset loop below
    }

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Combined Confusion Matrix (Accuracy: {100*acc:.2f}%)')

    # Annotate cells
    for i in range(n_classes):
        for j in range(n_classes):
            text = ax.text(j, i, str(cm[i, j]),
                          ha='center', va='center',
                          color='white' if cm[i, j] > cm.max()/2 else 'black')

    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path = os.path.join(args.output_dir, 'confusion_matrix_combined.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path}")

    # Accuracy vs JSR (if available)
    if has_jsr and jsr_all is not None:
        plot_accuracy_vs_jsr(labels_all, preds, jsr_all, n_classes, args.output_dir,
                            title='Combined Datasets')
        results['jsr_analysis'] = analyze_jsr_performance(labels_all, preds, jsr_all, n_classes)

    # Per-dataset confusion matrices
    for i, d in enumerate(all_data):
        mask = (dataset_ids == i)
        cm_i = confusion_matrix(labels_all[mask], preds[mask], labels=range(n_classes))

        # Machine-readable per-dataset metrics into the JSON.
        report_i = classification_report(
            labels_all[mask], preds[mask], labels=range(n_classes),
            target_names=class_names, digits=3, output_dict=True, zero_division=0)
        results['per_dataset'][d['source']] = {
            'n_samples': int(mask.sum()),
            'accuracy': float(np.mean(preds[mask] == labels_all[mask])),
            'confusion_matrix': cm_i.tolist(),
            'classification_report': report_i,
        }

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm_i, cmap='Blues', aspect='auto')
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        acc_i = np.mean(preds[mask] == labels_all[mask])
        ax.set_title(f'{d["source"]} Confusion Matrix (Accuracy: {100*acc_i:.2f}%)')

        for ii in range(n_classes):
            for jj in range(n_classes):
                text = ax.text(jj, ii, str(cm_i[ii, jj]),
                              ha='center', va='center',
                              color='white' if cm_i[ii, jj] > cm_i.max()/2 else 'black')

        plt.colorbar(im, ax=ax)
        fig.tight_layout()
        out_path = os.path.join(args.output_dir, f'confusion_matrix_{d["source"]}.png')
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")

    # Confidence distribution
    plot_confidence_distribution(confidence, labels_all, preds, n_classes, args.output_dir,
                                 title='Combined Datasets')

    # Save JSON results
    with open(os.path.join(args.output_dir, 'results_combined.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


def plot_accuracy_vs_jsr(labels, preds, jsr_db, n_classes, out_dir, title=''):
    """Plot accuracy vs JSR for each class."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    # JSR bins
    jsr_bins = np.arange(-10, 35, 2)
    bin_centers = (jsr_bins[:-1] + jsr_bins[1:]) / 2

    class_names = ['clean'] + [(f'{k} RFI eigenvalue' if k == 1 else f'{k} RFI eigenvalues')
                                 for k in range(1, n_classes)]

    for k in range(n_classes):
        mask = (labels == k)
        if mask.sum() < 10:
            continue

        accs = []
        for i in range(len(jsr_bins) - 1):
            bin_mask = mask & (jsr_db >= jsr_bins[i]) & (jsr_db < jsr_bins[i+1])
            if bin_mask.sum() > 0:
                accs.append(np.mean(preds[bin_mask] == labels[bin_mask]))
            else:
                accs.append(np.nan)

        ax.plot(bin_centers, accs, marker='o', label=class_names[k], linewidth=2)

    ax.set_xlabel('JSR (dB)')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    ax.set_title(f'Accuracy vs JSR - {title}')

    fig.tight_layout()
    out_path = os.path.join(out_dir, 'accuracy_vs_jsr.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def analyze_jsr_performance(labels, preds, jsr_db, n_classes):
    """Analyze performance across JSR ranges."""
    jsr_ranges = [(-10, 0), (0, 10), (10, 20), (20, 35)]

    results = {}
    for low, high in jsr_ranges:
        mask = (jsr_db >= low) & (jsr_db < high)
        if mask.sum() > 0:
            acc = float(np.mean(preds[mask] == labels[mask]))
            results[f'{low}to{high}dB'] = {
                'accuracy': acc,
                'n_samples': int(mask.sum()),
            }

    return results


def plot_confidence_distribution(confidence, labels, preds, n_classes, out_dir, title=''):
    """Plot confidence distribution by correctness."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    correct = (preds == labels)

    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(0, 1, 50)
    ax.hist(confidence[correct], bins=bins, alpha=0.6, label='Correct', color='green', density=True)
    ax.hist(confidence[~correct], bins=bins, alpha=0.6, label='Incorrect', color='red', density=True)

    ax.set_xlabel('Confidence (max softmax)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title(f'Confidence Distribution - {title}')

    fig.tight_layout()
    out_path = os.path.join(out_dir, 'confidence_distribution.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained model on labeled synthetic test data '
                    '(combined + per-dataset confusion matrices). For real '
                    'NISAR scenes use score_scene.py.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model', required=True,
                        help='Trained Keras model (best_model.keras)')
    parser.add_argument('--data-dirs', nargs='+', required=True,
                        help='One or more test directories, each with labeled .h5 tile '
                             'files. Every directory is scored as its own dataset AND '
                             'pooled into the combined metrics.')
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    import tensorflow as tf
    print(f"Loading model: {args.model}")
    model = tf.keras.models.load_model(args.model)

    combined_synthetic_test(model, args.data_dirs, args)

    print(f"\n{'='*70}")
    print(f"DONE. Results saved to {args.output_dir}")
    print(f"{'='*70}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
