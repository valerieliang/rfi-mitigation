"""
test_only.py

Unified test script with multiple modes:

1. COMBINED SYNTHETIC TEST
   Test synthetic data from multiple data/<folders>, produce confusion matrix,
   accuracy vs JSR, etc.

2. AMAZON CLEAN CHECK
   Test real Amazon NISAR data (assumed clean) against both HH and HV.
   Produces clean check accuracy and confusion matrices.

3. MOUNTAINS CLEAN CHECK
   Test real Mountains NISAR data (assumed clean) using H5 file to specify tiles.
   HH and HV tiles may be distinct. Produces clean check accuracy and confusion matrices.

4. SCORE SCENE (NO GROUND TRUTH)
   Score specific pulse/range windows with no ground truth labels.
   Analog to score_scene.py but integrated here for convenience.
   Outputs predictions, confidence maps, eigenvalue profiles, etc.

Usage Examples
--------------
# Combined synthetic test
py-isce3 test_only.py combined \\
    --model models/combined_amazon_mountain/best_model.keras \\
    --data-dirs data/amazon_synthetic data/mountain_synthetic \\
    --output-dir results/combined_test

# Amazon clean check
py-isce3 test_only.py amazon-clean \\
    --model models/combined_amazon_mountain/best_model.keras \\
    --nisar-file /path/to/amazon.h5 \\
    --pulse-start 46528 --pulse-end 124580 \\
    --range-start 0 --range-end 25600 \\
    --freq A \\
    --output-dir results/amazon_clean

# Mountains clean check
py-isce3 test_only.py mountains-clean \\
    --model models/combined_amazon_mountain/best_model.keras \\
    --h5-file data/mountains_test.h5 \\
    --nisar-file /path/to/berlin.h5 \\
    --output-dir results/mountains_clean

# Score scene (no ground truth)
py-isce3 test_only.py score-scene \\
    --model models/combined_amazon_mountain/best_model.keras \\
    --nisar-file /path/to/scene.h5 \\
    --pulse-start 0 --pulse-end 50000 \\
    --range-start 0 --range-end 25600 \\
    --freq A --pol HH \\
    --save-predictions --save-confidence --save-profiles --save-diagnostics \\
    --output-dir results/scene_score
"""

import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np
import h5py

# Reuse the project's feature extraction code
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_amazon_data import (
    read_raw_data_batch,
    get_subswath_mask,
    compute_scm_and_eigs,
    tile_signal_power,
    CPI_LEN_DEFAULT,
    CPI_WIDTH_DEFAULT,
    OFF_DIAG_OVERLAP_RATIO_DEFAULT,
    DIAG_VALID_RATIO_DEFAULT,
    PULSE_CHUNK_DEFAULT,
)
from nisar.products.readers.Raw import Raw


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
    test_file = os.path.join(data_dir, 'test.h5')
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"No test.h5 found in {data_dir}")

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
    class_names = ['clean'] + [f'knee@{k}' for k in range(1, n_classes)]
    print(classification_report(labels_all, preds, target_names=class_names, digits=3))

    # Save results
    results = {
        'mode': 'combined_synthetic',
        'data_dirs': data_dirs,
        'total_samples': int(len(labels_all)),
        'per_dataset_samples': {d['source']: int(len(d['labels'])) for d in all_data},
        'overall_accuracy': float(acc),
        'per_dataset_accuracy': {
            d['source']: float(np.mean(preds[dataset_ids == i] == labels_all[dataset_ids == i]))
            for i, d in enumerate(all_data)
        },
        'confusion_matrix': cm.tolist(),
        'n_classes': n_classes,
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

    class_names = ['clean'] + [f'knee@{k}' for k in range(1, n_classes)]

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


# ===========================================================================
# MODE 2: AMAZON CLEAN CHECK
# ===========================================================================

def amazon_clean_check(model, args):
    """
    Test model on real Amazon NISAR data assumed to be clean.
    Tests both HH and HV polarizations.
    Reports false positive rate (any knee>0 detection is a false positive).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    print(f"\n{'='*70}")
    print('AMAZON CLEAN CHECK')
    print(f"{'='*70}")
    print(f"NISAR file: {args.nisar_file}")
    print(f"Frequency: {args.freq}")
    print(f"Pulses: [{args.pulse_start}, {args.pulse_end})")
    print(f"Range: [{args.range_start}, {args.range_end})")
    print("\nASSUMPTION: Real data is CLEAN (label=0 ground truth)")

    raw = Raw(hdf5file=args.nisar_file)
    raw.parsePolarizations()

    freq = args.freq
    if freq not in raw.polarizations:
        raise ValueError(f"Frequency {freq} not found in NISAR file")

    pols = list(raw.polarizations[freq])
    print(f"Testing polarizations: {pols}")

    results = {
        'mode': 'amazon_clean',
        'nisar_file': args.nisar_file,
        'frequency': freq,
        'pulse_window': [args.pulse_start, args.pulse_end],
        'range_window': [args.range_start, args.range_end],
        'polarizations': {},
    }

    all_recs = []

    for pol in pols:
        print(f"\n--- Testing {freq}-{pol} ---")
        rec = score_clean_channel(raw, freq, pol, model, args, ground_truth_label=0)
        all_recs.append(rec)

        # Analyze results
        preds = rec['predictions']
        n_tiles = len(preds)
        false_positives = (preds > 0).sum()
        fpr = false_positives / n_tiles

        print(f"Total tiles: {n_tiles}")
        print(f"False positives (knee>0): {false_positives} ({100*fpr:.2f}%)")
        print(f"Mean confidence: {rec['confidence'].mean():.3f}")
        print(f"Mean confidence on false positives: {rec['confidence'][preds > 0].mean() if false_positives > 0 else 'N/A'}")

        # Confusion matrix (all should be class 0)
        n_classes = rec['n_classes']
        cm = confusion_matrix([0]*n_tiles, preds, labels=range(n_classes))

        results['polarizations'][pol] = {
            'n_tiles': int(n_tiles),
            'false_positives': int(false_positives),
            'false_positive_rate': float(fpr),
            'mean_confidence': float(rec['confidence'].mean()),
            'confusion_matrix': cm.tolist(),
        }

        # Plot confusion matrix
        plot_clean_confusion_matrix(cm, n_classes, f"{freq}-{pol}", args.output_dir)

        # Save predictions
        save_clean_check_h5(rec, args, args.output_dir)

    # Combined summary plot
    plot_clean_check_summary(all_recs, args.output_dir, 'Amazon')

    # Save results
    with open(os.path.join(args.output_dir, 'results_amazon_clean.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


def score_clean_channel(raw, freq, pol, model, args, ground_truth_label=0):
    """
    Score a channel assumed to be clean (or with known label).
    Similar to score_channel from score_scene.py but with ground truth.
    """
    cpi_len, cpi_width = args.cpi_len, args.cpi_width

    dataset = raw.getRawDataset(freq, pol)
    total_pulses, total_range = dataset.shape

    p_start = args.pulse_start
    p_end = min(args.pulse_end, total_pulses)
    r_start = args.range_start
    r_end = min(args.range_end, total_range)

    n_pt = (p_end - p_start) // cpi_len
    n_rt = (r_end - r_start) // cpi_width
    p_end = p_start + n_pt * cpi_len
    r_end = r_start + n_rt * cpi_width

    if n_pt <= 0 or n_rt <= 0:
        raise ValueError('Window is smaller than one CPI tile')

    n_tiles = n_pt * n_rt
    print(f"Tile grid: {n_pt} x {n_rt} = {n_tiles} tiles")

    eigen_all = np.zeros((n_tiles, N_KEEP, 2), dtype=np.float32)
    global_all = np.zeros((n_tiles, 2), dtype=np.float32)
    eigvals_all = np.zeros((n_tiles, M), dtype=np.float32)
    power_db = np.zeros(n_tiles, dtype=np.float32)
    tile_pulse = np.zeros(n_tiles, dtype=np.int32)
    tile_range = np.zeros(n_tiles, dtype=np.int32)

    chunk_tiles = max(1, args.pulse_chunk // cpi_len)
    k = 0

    for chunk_start in range(0, n_pt, chunk_tiles):
        n_here = min(chunk_tiles, n_pt - chunk_start)
        cp0 = p_start + chunk_start * cpi_len
        cp1 = cp0 + n_here * cpi_len

        raw_chunk = read_raw_data_batch(
            raw, freq, pol, slice(cp0, cp1), slice(r_start, r_end)
        )
        mask_chunk = (
            get_subswath_mask(raw, freq, pol,
                            np.arange(cp0, cp1), np.arange(r_start, r_end))
            if args.compute_subswath_mask else None
        )

        for lp in range(n_here):
            pt = chunk_start + lp
            lp0, lp1 = lp * cpi_len, (lp + 1) * cpi_len

            for rt in range(n_rt):
                lr0, lr1 = rt * cpi_width, (rt + 1) * cpi_width

                cpi = np.ascontiguousarray(
                    raw_chunk[lp0:lp1, lr0:lr1]).astype(np.complex64)
                cpi_mask = (np.ascontiguousarray(mask_chunk[lp0:lp1, lr0:lr1])
                           if mask_chunk is not None else None)

                _, eigvals, diag_lin, diag_valid = compute_scm_and_eigs(
                    cpi, cpi_mask,
                    args.off_diag_overlap_ratio, args.diag_valid_ratio
                )

                eigen_all[k], global_all[k] = features_from_eigenvalues(
                    eigvals, diag_lin, diag_valid
                )
                eigvals_all[k] = eigvals
                power_db[k] = 10.0 * np.log10(tile_signal_power(cpi, cpi_mask))
                tile_pulse[k] = p_start + pt * cpi_len
                tile_range[k] = r_start + lr0
                k += 1

    print("Predicting...")
    probs = model.predict([eigen_all, global_all],
                         batch_size=args.batch_size, verbose=0)

    preds = np.argmax(probs, axis=-1).astype(np.int8)
    confidence = np.max(probs, axis=-1).astype(np.float32)
    entropy = (-np.sum(probs * np.log(probs + EPS), axis=-1)).astype(np.float32)

    return {
        'freq': freq,
        'pol': pol,
        'predictions': preds,
        'confidence': confidence,
        'entropy': entropy,
        'eigvals': eigvals_all,
        'power_db': power_db,
        'tile_pulse': tile_pulse,
        'tile_range': tile_range,
        'n_pt': n_pt,
        'n_rt': n_rt,
        'pulse_window': [p_start, p_end],
        'range_window': [r_start, r_end],
        'n_classes': probs.shape[-1],
        'ground_truth_label': ground_truth_label,
    }


def plot_clean_confusion_matrix(cm, n_classes, channel_name, out_dir):
    """Plot confusion matrix for clean check (single ground truth class)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    class_names = ['clean'] + [f'knee@{k}' for k in range(1, n_classes)]

    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(cm[:1, :], cmap='Blues', aspect='auto')

    ax.set_xticks(range(n_classes))
    ax.set_yticks([0])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(['clean (truth)'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Clean Check Confusion Matrix - {channel_name}')

    for j in range(n_classes):
        text = ax.text(j, 0, str(cm[0, j]),
                      ha='center', va='center',
                      color='white' if cm[0, j] > cm.max()/2 else 'black',
                      fontsize=12)

    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f'confusion_matrix_{channel_name}_clean.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_clean_check_summary(recs, out_dir, dataset_name):
    """Summary plot for clean check across polarizations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # False positive rates
    channels = [f"{r['freq']}-{r['pol']}" for r in recs]
    fprs = [(r['predictions'] > 0).mean() for r in recs]

    axes[0].bar(channels, fprs, color='steelblue', alpha=0.7)
    axes[0].set_ylabel('False Positive Rate')
    axes[0].set_xlabel('Channel')
    axes[0].set_title('False Positive Rate by Channel')
    axes[0].grid(True, axis='y', linestyle='--', alpha=0.5)

    for i, (ch, fpr) in enumerate(zip(channels, fprs)):
        axes[0].text(i, fpr, f'{100*fpr:.2f}%', ha='center', va='bottom')

    # Confidence on false positives
    for rec in recs:
        fp_mask = (rec['predictions'] > 0)
        if fp_mask.sum() > 0:
            conf_fp = rec['confidence'][fp_mask]
            axes[1].hist(conf_fp, bins=30, alpha=0.6,
                        label=f"{rec['freq']}-{rec['pol']}", density=True)

    axes[1].set_xlabel('Confidence (max softmax)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Confidence Distribution on False Positives')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    fig.suptitle(f'{dataset_name} Clean Check Summary', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = os.path.join(out_dir, f'{dataset_name.lower()}_clean_summary.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def save_clean_check_h5(rec, args, out_dir):
    """Save clean check predictions to HDF5."""
    path = os.path.join(out_dir, f"predictions_{rec['freq']}_{rec['pol']}_clean.h5")
    with h5py.File(path, 'w') as f:
        f.attrs['mode'] = 'clean_check'
        f.attrs['nisar_file'] = args.nisar_file
        f.attrs['frequency'] = rec['freq']
        f.attrs['polarization'] = rec['pol']
        f.attrs['ground_truth_label'] = rec['ground_truth_label']
        f.attrs['pulse_start'] = rec['pulse_window'][0]
        f.attrs['pulse_end'] = rec['pulse_window'][1]
        f.attrs['range_start'] = rec['range_window'][0]
        f.attrs['range_end'] = rec['range_window'][1]
        f.attrs['n_pulse_tiles'] = rec['n_pt']
        f.attrs['n_range_tiles'] = rec['n_rt']

        f.create_dataset('predictions', data=rec['predictions'])
        f.create_dataset('confidence', data=rec['confidence'])
        f.create_dataset('entropy', data=rec['entropy'])
        f.create_dataset('eigenvalues', data=rec['eigvals'], compression='gzip')
        f.create_dataset('signal_power_db', data=rec['power_db'])
        f.create_dataset('tile_pulse', data=rec['tile_pulse'])
        f.create_dataset('tile_range', data=rec['tile_range'])

    print(f"Saved {path}")


# ===========================================================================
# MODE 3: MOUNTAINS CLEAN CHECK
# ===========================================================================

def mountains_clean_check(model, args):
    """
    Test model on real Mountains NISAR data assumed to be clean.
    Uses H5 file to specify which tiles to test (may be different for HH and HV).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    print(f"\n{'='*70}")
    print('MOUNTAINS CLEAN CHECK')
    print(f"{'='*70}")
    print(f"H5 file: {args.h5_file}")
    print(f"NISAR file: {args.nisar_file}")
    print("\nASSUMPTION: Real data is CLEAN (label=0 ground truth)")

    # Load tile specifications from H5
    with h5py.File(args.h5_file, 'r') as f:
        freq = str(f.attrs['frequency'])

        # Get polarizations and their tile specifications
        pols = []
        tile_specs = {}

        for key in f.keys():
            if key.endswith('_pulse_tiles'):
                pol = key.replace('_pulse_tiles', '')
                pols.append(pol)

                pulse_tiles = f[f'{pol}_pulse_tiles'][:]
                range_tiles = f[f'{pol}_range_tiles'][:]

                tile_specs[pol] = {
                    'pulse_tiles': pulse_tiles,
                    'range_tiles': range_tiles,
                }

    print(f"Frequency: {freq}")
    print(f"Polarizations: {pols}")
    for pol, spec in tile_specs.items():
        print(f"  {pol}: {len(spec['pulse_tiles'])} tiles")

    raw = Raw(hdf5file=args.nisar_file)
    raw.parsePolarizations()

    results = {
        'mode': 'mountains_clean',
        'h5_file': args.h5_file,
        'nisar_file': args.nisar_file,
        'frequency': freq,
        'polarizations': {},
    }

    all_recs = []

    for pol in pols:
        print(f"\n--- Testing {freq}-{pol} ---")
        rec = score_mountain_tiles(raw, freq, pol, model, args, tile_specs[pol],
                                   ground_truth_label=0)
        all_recs.append(rec)

        # Analyze results
        preds = rec['predictions']
        n_tiles = len(preds)
        false_positives = (preds > 0).sum()
        fpr = false_positives / n_tiles

        print(f"Total tiles: {n_tiles}")
        print(f"False positives (knee>0): {false_positives} ({100*fpr:.2f}%)")
        print(f"Mean confidence: {rec['confidence'].mean():.3f}")

        # Confusion matrix
        n_classes = rec['n_classes']
        cm = confusion_matrix([0]*n_tiles, preds, labels=range(n_classes))

        results['polarizations'][pol] = {
            'n_tiles': int(n_tiles),
            'false_positives': int(false_positives),
            'false_positive_rate': float(fpr),
            'mean_confidence': float(rec['confidence'].mean()),
            'confusion_matrix': cm.tolist(),
        }

        # Plot confusion matrix
        plot_clean_confusion_matrix(cm, n_classes, f"{freq}-{pol}", args.output_dir)

        # Save predictions
        save_clean_check_h5(rec, args, args.output_dir)

    # Combined summary plot
    plot_clean_check_summary(all_recs, args.output_dir, 'Mountains')

    # Save results
    with open(os.path.join(args.output_dir, 'results_mountains_clean.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


def score_mountain_tiles(raw, freq, pol, model, args, tile_spec, ground_truth_label=0):
    """
    Score specific tiles from mountains data.
    tile_spec contains pulse_tiles and range_tiles arrays.
    """
    cpi_len, cpi_width = args.cpi_len, args.cpi_width

    pulse_tiles = tile_spec['pulse_tiles']
    range_tiles = tile_spec['range_tiles']
    n_tiles = len(pulse_tiles)

    print(f"Scoring {n_tiles} specified tiles...")

    eigen_all = np.zeros((n_tiles, N_KEEP, 2), dtype=np.float32)
    global_all = np.zeros((n_tiles, 2), dtype=np.float32)
    eigvals_all = np.zeros((n_tiles, M), dtype=np.float32)
    power_db = np.zeros(n_tiles, dtype=np.float32)

    for i, (pt, rt) in enumerate(zip(pulse_tiles, range_tiles)):
        p_start = int(pt)
        p_end = p_start + cpi_len
        r_start = int(rt)
        r_end = r_start + cpi_width

        # Read tile
        raw_tile = read_raw_data_batch(
            raw, freq, pol, slice(p_start, p_end), slice(r_start, r_end)
        )

        tile_mask = (
            get_subswath_mask(raw, freq, pol,
                            np.arange(p_start, p_end), np.arange(r_start, r_end))
            if args.compute_subswath_mask else None
        )

        cpi = raw_tile.astype(np.complex64)

        _, eigvals, diag_lin, diag_valid = compute_scm_and_eigs(
            cpi, tile_mask,
            args.off_diag_overlap_ratio, args.diag_valid_ratio
        )

        eigen_all[i], global_all[i] = features_from_eigenvalues(
            eigvals, diag_lin, diag_valid
        )
        eigvals_all[i] = eigvals
        power_db[i] = 10.0 * np.log10(tile_signal_power(cpi, tile_mask))

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n_tiles} tiles")

    print("Predicting...")
    probs = model.predict([eigen_all, global_all],
                         batch_size=args.batch_size, verbose=0)

    preds = np.argmax(probs, axis=-1).astype(np.int8)
    confidence = np.max(probs, axis=-1).astype(np.float32)
    entropy = (-np.sum(probs * np.log(probs + EPS), axis=-1)).astype(np.float32)

    return {
        'freq': freq,
        'pol': pol,
        'predictions': preds,
        'confidence': confidence,
        'entropy': entropy,
        'eigvals': eigvals_all,
        'power_db': power_db,
        'tile_pulse': pulse_tiles,
        'tile_range': range_tiles,
        'n_tiles': n_tiles,
        'n_classes': probs.shape[-1],
        'ground_truth_label': ground_truth_label,
    }


# ===========================================================================
# MODE 4: SCORE SCENE (NO GROUND TRUTH)
# ===========================================================================

def score_scene_mode(model, args):
    """
    Score a scene with no ground truth labels.
    Analog to score_scene.py integrated into test_only.py.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print(f"\n{'='*70}")
    print('SCORE SCENE (NO GROUND TRUTH)')
    print(f"{'='*70}")
    print(f"NISAR file: {args.nisar_file}")
    print(f"Frequency: {args.freq}, Polarization: {args.pol}")
    print(f"Pulses: [{args.pulse_start}, {args.pulse_end})")
    print(f"Range: [{args.range_start}, {args.range_end})")

    raw = Raw(hdf5file=args.nisar_file)
    raw.parsePolarizations()

    rec = score_clean_channel(raw, args.freq, args.pol, model, args, ground_truth_label=None)

    # Report statistics
    preds = rec['predictions']
    n_tiles = len(preds)
    n_classes = rec['n_classes']

    print(f"\nTotal tiles: {n_tiles}")
    print(f"Prediction distribution:")
    dist = np.bincount(preds, minlength=n_classes)[:n_classes]
    for k, count in enumerate(dist):
        label = 'clean' if k == 0 else f'knee@{k}'
        print(f"  {label}: {count} ({100*count/n_tiles:.2f}%)")

    print(f"\nMean confidence: {rec['confidence'].mean():.3f}")

    results = {
        'mode': 'score_scene',
        'nisar_file': args.nisar_file,
        'frequency': rec['freq'],
        'polarization': rec['pol'],
        'pulse_window': rec['pulse_window'],
        'range_window': rec['range_window'],
        'n_tiles': int(n_tiles),
        'prediction_distribution': {str(k): int(v) for k, v in enumerate(dist)},
        'mean_confidence': float(rec['confidence'].mean()),
    }

    # Save outputs based on flags
    if args.save_predictions:
        save_scene_predictions_h5(rec, args, args.output_dir)

    if args.save_confidence:
        plot_confidence_maps(rec, args.output_dir)

    if args.save_profiles:
        plot_eigenvalue_profiles(rec, args.output_dir)

    if args.save_diagnostics:
        plot_diagnostic_plots(rec, args.output_dir)

    # Save results JSON
    with open(os.path.join(args.output_dir, 'results_scene.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")

    return results


def save_scene_predictions_h5(rec, args, out_dir):
    """Save scene predictions to HDF5."""
    path = os.path.join(out_dir, f"predictions_{rec['freq']}_{rec['pol']}_scene.h5")
    with h5py.File(path, 'w') as f:
        f.attrs['mode'] = 'score_scene'
        f.attrs['nisar_file'] = args.nisar_file
        f.attrs['frequency'] = rec['freq']
        f.attrs['polarization'] = rec['pol']
        f.attrs['labeled'] = False
        f.attrs['pulse_start'] = rec['pulse_window'][0]
        f.attrs['pulse_end'] = rec['pulse_window'][1]
        f.attrs['range_start'] = rec['range_window'][0]
        f.attrs['range_end'] = rec['range_window'][1]
        f.attrs['n_pulse_tiles'] = rec['n_pt']
        f.attrs['n_range_tiles'] = rec['n_rt']

        f.create_dataset('predictions', data=rec['predictions'])
        f.create_dataset('confidence', data=rec['confidence'])
        f.create_dataset('entropy', data=rec['entropy'])
        f.create_dataset('eigenvalues', data=rec['eigvals'], compression='gzip')
        f.create_dataset('signal_power_db', data=rec['power_db'])
        f.create_dataset('tile_pulse', data=rec['tile_pulse'])
        f.create_dataset('tile_range', data=rec['tile_range'])

    print(f"Saved predictions: {path}")


def plot_confidence_maps(rec, out_dir):
    """Plot knee map and confidence map."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_pt, n_rt = rec['n_pt'], rec['n_rt']

    # Knee map
    grid = rec['predictions'].reshape(n_pt, n_rt)
    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(grid, aspect='auto', cmap='inferno', origin='upper',
                   vmin=0, vmax=max(rec['n_classes'] - 1, 1),
                   interpolation='nearest',
                   extent=[rec['range_window'][0], rec['range_window'][1],
                          rec['pulse_window'][1], rec['pulse_window'][0]])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Predicted knee (0 = clean)')
    ax.set_xlabel('Range sample')
    ax.set_ylabel('Pulse')
    ax.set_title(f"Predicted Knee Map - {rec['freq']}-{rec['pol']}")
    fig.tight_layout()
    path = os.path.join(out_dir, f"knee_map_{rec['freq']}_{rec['pol']}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    # Confidence map
    grid = rec['confidence'].reshape(n_pt, n_rt)
    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(grid, aspect='auto', cmap='viridis', origin='upper',
                   vmin=0, vmax=1, interpolation='nearest',
                   extent=[rec['range_window'][0], rec['range_window'][1],
                          rec['pulse_window'][1], rec['pulse_window'][0]])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Confidence (max softmax)')
    ax.set_xlabel('Range sample')
    ax.set_ylabel('Pulse')
    ax.set_title(f"Confidence Map - {rec['freq']}-{rec['pol']}")
    fig.tight_layout()
    path = os.path.join(out_dir, f"confidence_map_{rec['freq']}_{rec['pol']}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_eigenvalue_profiles(rec, out_dir):
    """Plot eigenvalue profiles by predicted class."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    n_classes = rec['n_classes']
    preds = rec['predictions']

    ev = np.maximum(rec['eigvals'][:, :N_KEEP], EPS)
    ev_db = 10.0 * np.log10(ev / np.maximum(ev[:, :1], EPS))
    idx = np.arange(1, N_KEEP + 1)

    norm = mcolors.Normalize(vmin=0, vmax=max(n_classes - 1, 1))
    cmap_colors = cm.plasma

    fig, ax = plt.subplots(figsize=(10, 6))
    for k in range(n_classes):
        sel = (preds == k)
        if sel.sum() < 5:
            continue
        mean_prof = ev_db[sel].mean(axis=0)
        ax.plot(idx, mean_prof, color=cmap_colors(norm(k)), linewidth=2,
               label=f"{'clean' if k == 0 else f'knee@{k}'} (n={int(sel.sum())})")
        if k > 0:
            ax.plot(k, mean_prof[k - 1], 'rx', markersize=8, markeredgewidth=2)

    ax.set_xlabel('Eigenvalue index (1-based, descending)')
    ax.set_ylabel('Eigenvalue (dB, normalized to lambda_max)')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=9)
    ax.set_title(f"Mean Eigenvalue Profile by Predicted Class - {rec['freq']}-{rec['pol']}")

    fig.tight_layout()
    path = os.path.join(out_dir, f"eigen_profiles_{rec['freq']}_{rec['pol']}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_diagnostic_plots(rec, out_dir):
    """Plot diagnostic plots: power vs prediction, confidence by class."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_classes = rec['n_classes']
    preds = rec['predictions']
    power = rec['power_db']
    conf = rec['confidence']
    entropy = rec['entropy']

    # Power vs prediction
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    groups, labels, counts = [], [], []
    for k in range(n_classes):
        sel = (preds == k)
        if sel.sum() >= 10:
            groups.append(power[sel])
            labels.append('clean' if k == 0 else f'knee@{k}')
            counts.append(int(sel.sum()))

    if groups:
        bp = ax1.boxplot(groups, labels=labels, showfliers=False, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.6)
        ax1.set_ylabel('Tile baseline power (dB)')
        ax1.set_xlabel('Predicted class')
        ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax1.set_title('Baseline Power by Predicted Class')

    # Scatter
    n = len(preds)
    if n > 20000:
        idx = np.random.default_rng(0).choice(n, size=20000, replace=False)
    else:
        idx = np.arange(n)

    sc = ax2.scatter(preds[idx] + np.random.default_rng(1).uniform(-0.25, 0.25, len(idx)),
                    power[idx], c=conf[idx], cmap='viridis',
                    s=3, alpha=0.4, vmin=0, vmax=1)
    fig.colorbar(sc, ax=ax2, label='Confidence')
    ax2.set_xticks(range(n_classes))
    ax2.set_xticklabels(['clean'] + [f'{k}' for k in range(1, n_classes)])
    ax2.set_xlabel('Predicted knee')
    ax2.set_ylabel('Tile baseline power (dB)')
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.set_title('Power vs Prediction (subsample)')

    fig.suptitle(f"Power Diagnostics - {rec['freq']}-{rec['pol']}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(out_dir, f"power_diagnostics_{rec['freq']}_{rec['pol']}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    # Confidence by class
    conf_groups, ent_groups, labels2, counts2 = [], [], [], []
    for k in range(n_classes):
        sel = (preds == k)
        if sel.sum() >= 10:
            conf_groups.append(conf[sel])
            ent_groups.append(entropy[sel])
            labels2.append('clean' if k == 0 else f'knee@{k}')
            counts2.append(int(sel.sum()))

    if conf_groups:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        bp1 = ax1.boxplot(conf_groups, labels=labels2, showfliers=False, patch_artist=True)
        for patch in bp1['boxes']:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.6)
        ax1.set_ylabel('Max softmax probability')
        ax1.set_xlabel('Predicted class')
        ax1.set_ylim(0, 1.02)
        ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax1.set_title('Confidence per Prediction')

        bp2 = ax2.boxplot(ent_groups, labels=labels2, showfliers=False, patch_artist=True)
        for patch in bp2['boxes']:
            patch.set_facecolor('indianred')
            patch.set_alpha(0.6)
        ax2.set_ylabel('Posterior entropy (nats)')
        ax2.set_xlabel('Predicted class')
        ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax2.set_title('Uncertainty per Prediction')

        fig.suptitle(f"Confidence Diagnostics - {rec['freq']}-{rec['pol']}", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        path = os.path.join(out_dir, f"confidence_diagnostics_{rec['freq']}_{rec['pol']}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Unified test script with multiple modes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest='mode', help='Test mode')

    # Combined synthetic test
    combined = subparsers.add_parser('combined',
                                     help='Test synthetic data from multiple directories')
    combined.add_argument('--model', required=True, help='Trained Keras model')
    combined.add_argument('--data-dirs', nargs='+', required=True,
                         help='Data directories containing test.h5')
    combined.add_argument('--batch-size', type=int, default=4096)
    combined.add_argument('--output-dir', required=True)

    # Amazon clean check
    amazon = subparsers.add_parser('amazon-clean',
                                  help='Test real Amazon data (assumed clean)')
    amazon.add_argument('--model', required=True, help='Trained Keras model')
    amazon.add_argument('--nisar-file', required=True, help='NISAR L0B file')
    amazon.add_argument('--freq', required=True, choices=['A', 'B'])
    amazon.add_argument('--pulse-start', type=int, required=True)
    amazon.add_argument('--pulse-end', type=int, required=True)
    amazon.add_argument('--range-start', type=int, required=True)
    amazon.add_argument('--range-end', type=int, required=True)
    amazon.add_argument('--cpi-len', type=int, default=CPI_LEN_DEFAULT)
    amazon.add_argument('--cpi-width', type=int, default=CPI_WIDTH_DEFAULT)
    amazon.add_argument('--pulse-chunk', type=int, default=PULSE_CHUNK_DEFAULT)
    amazon.add_argument('--compute-subswath-mask', action='store_true')
    amazon.add_argument('--off-diag-overlap-ratio', type=float,
                       default=OFF_DIAG_OVERLAP_RATIO_DEFAULT)
    amazon.add_argument('--diag-valid-ratio', type=float,
                       default=DIAG_VALID_RATIO_DEFAULT)
    amazon.add_argument('--batch-size', type=int, default=4096)
    amazon.add_argument('--output-dir', required=True)

    # Mountains clean check
    mountains = subparsers.add_parser('mountains-clean',
                                     help='Test real Mountains data (assumed clean)')
    mountains.add_argument('--model', required=True, help='Trained Keras model')
    mountains.add_argument('--h5-file', required=True,
                          help='H5 file specifying tiles to test')
    mountains.add_argument('--nisar-file', required=True, help='NISAR L0B file')
    mountains.add_argument('--cpi-len', type=int, default=CPI_LEN_DEFAULT)
    mountains.add_argument('--cpi-width', type=int, default=CPI_WIDTH_DEFAULT)
    mountains.add_argument('--compute-subswath-mask', action='store_true')
    mountains.add_argument('--off-diag-overlap-ratio', type=float,
                          default=OFF_DIAG_OVERLAP_RATIO_DEFAULT)
    mountains.add_argument('--diag-valid-ratio', type=float,
                          default=DIAG_VALID_RATIO_DEFAULT)
    mountains.add_argument('--batch-size', type=int, default=4096)
    mountains.add_argument('--output-dir', required=True)

    # Score scene (no ground truth)
    scene = subparsers.add_parser('score-scene',
                                 help='Score scene with no ground truth')
    scene.add_argument('--model', required=True, help='Trained Keras model')
    scene.add_argument('--nisar-file', required=True, help='NISAR L0B file')
    scene.add_argument('--freq', required=True, choices=['A', 'B'])
    scene.add_argument('--pol', required=True, help='Polarization (e.g. HH, HV)')
    scene.add_argument('--pulse-start', type=int, required=True)
    scene.add_argument('--pulse-end', type=int, required=True)
    scene.add_argument('--range-start', type=int, required=True)
    scene.add_argument('--range-end', type=int, required=True)
    scene.add_argument('--cpi-len', type=int, default=CPI_LEN_DEFAULT)
    scene.add_argument('--cpi-width', type=int, default=CPI_WIDTH_DEFAULT)
    scene.add_argument('--pulse-chunk', type=int, default=PULSE_CHUNK_DEFAULT)
    scene.add_argument('--compute-subswath-mask', action='store_true')
    scene.add_argument('--off-diag-overlap-ratio', type=float,
                      default=OFF_DIAG_OVERLAP_RATIO_DEFAULT)
    scene.add_argument('--diag-valid-ratio', type=float,
                      default=DIAG_VALID_RATIO_DEFAULT)
    scene.add_argument('--save-predictions', action='store_true',
                      help='Save predictions to HDF5')
    scene.add_argument('--save-confidence', action='store_true',
                      help='Save knee and confidence maps')
    scene.add_argument('--save-profiles', action='store_true',
                      help='Save eigenvalue profiles')
    scene.add_argument('--save-diagnostics', action='store_true',
                      help='Save diagnostic plots (power, confidence)')
    scene.add_argument('--batch-size', type=int, default=4096)
    scene.add_argument('--output-dir', required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode is None:
        print("Error: No mode specified. Use -h for help.")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    import tensorflow as tf
    print(f"Loading model: {args.model}")
    model = tf.keras.models.load_model(args.model)

    # Dispatch to appropriate mode
    if args.mode == 'combined':
        combined_synthetic_test(model, args.data_dirs, args)
    elif args.mode == 'amazon-clean':
        amazon_clean_check(model, args)
    elif args.mode == 'mountains-clean':
        mountains_clean_check(model, args)
    elif args.mode == 'score-scene':
        score_scene_mode(model, args)
    else:
        print(f"Unknown mode: {args.mode}")
        return 1

    print(f"\n{'='*70}")
    print(f"DONE. Results saved to {args.output_dir}")
    print(f"{'='*70}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
