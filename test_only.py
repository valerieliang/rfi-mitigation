"""
test_only.py

Evaluate a trained RFI knee classifier on SYNTHETIC test data with known
ground-truth labels. Loads one or more test directories (each treated as a
separate dataset/source), runs the model, and reports a combined confusion
matrix + accuracy plus a per-dataset breakdown -- so you can see mountain-only
vs amazon-only vs combined for the same model.

Accuracy vs RFI strength is plotted for the pooled set AND for each individual
test .h5 file, so every polarization and every JSR band of a directory gets its
own curves. Each such figure has two panels: binned by the STRONGEST band in a
tile (how hard the tile was to notice) and by the WEAKEST band (how hard it was
to count). Files that record no `jsr_db` fall back to `rfi_power_db`; files
with neither are still scored, just not plotted against strength.

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
import warnings

import numpy as np
import h5py

from diag_features import diag_profile_features


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

def reduce_jsr_bands(jsr_bands):
    """
    Collapse the per-band JSR array (N, max_bands) to one number per tile.

    A tile with k bands carries k JSRs, each drawn independently, and the two
    ends of that spread answer different questions:

      strongest (nanmax)  the easiest band to see. Governs whether the tile is
                          flagged as RFI-bearing at all.
      weakest   (nanmin)  the hardest band to see. Governs whether the tile is
                          assigned the RIGHT knee, since the count is only
                          correct if even the faintest emitter clears the noise
                          floor. This is the pessimistic curve and the one that
                          explains knee under-counting.

    Clean tiles are all-NaN and reduce to NaN, which drops them out of every
    JSR bin downstream. That is intended: they have no JSR.

    Returns (jsr_max, jsr_min), or (None, None) if the input is None.
    """
    if jsr_bands is None:
        return None, None
    jsr_bands = np.asarray(jsr_bands)
    if jsr_bands.ndim == 1:
        # Already reduced upstream; nothing to spread over.
        return jsr_bands, jsr_bands
    with warnings.catch_warnings():
        # All-NaN rows (clean tiles) warn on nanmax/nanmin; NaN is the answer.
        warnings.simplefilter('ignore', category=RuntimeWarning)
        jsr_max = np.nanmax(jsr_bands, axis=1)
        jsr_min = np.nanmin(jsr_bands, axis=1)
    return jsr_max, jsr_min


#: Per-band strength datasets we know how to bin against, in preference order.
#: JSR is the right x-axis when it is there. Absolute RFI power is the fallback
#: for tile files written without a JSR record -- it answers a slightly
#: different question (how loud was the emitter, not how loud relative to the
#: scene) so the two are never mixed on one axis.
STRENGTH_DATASETS = [
    ('jsr_db', 'JSR (dB)'),
    ('rfi_power_db', 'RFI power (dB)'),
]


def _read_strength(f):
    """
    Pull the per-band strength array out of an open tile file.

    Returns (bands_array, metric_label), or (None, None) if the file records
    neither JSR nor RFI power.
    """
    for key, label in STRENGTH_DATASETS:
        if key in f:
            return f[key][:], label
    return None, None


def load_synthetic_file(fpath, dir_name):
    """
    Load ONE labeled tile file into the dict shape the rest of this script uses.

    'source' is '<dir>__<file stem>' so that files sharing a basename across
    directories (rfi_data_A_HH.h5 lives in half a dozen of them) stay distinct
    once they become plot filenames.
    """
    stem = os.path.splitext(os.path.basename(fpath))[0]
    diag_profile, diag_global = None, None
    with h5py.File(fpath, 'r') as f:
        if 'eigen_features' in f:
            # Preprocessed features, already model-ready. No raw diagonal here,
            # so a three-input model cannot be scored on this file; the check in
            # combined_synthetic_test reports that rather than failing obscurely.
            eigen = f['eigen_features'][:]
            global_feat = f['global_features'][:]
        else:
            # Raw eigenvalues; featurize exactly as training did.
            eigen, global_feat = features_from_eigenvalues(
                f['eigenvalues'][:], f['diagonal'][:], f['diag_valid_idx'][:]
            )
            # Also build the sorted-diagonal-profile inputs, for models trained
            # by train_diag_profile.py. Cheap relative to the h5 read, and it
            # keeps one code path for both model variants.
            diag_profile, valid_frac = diag_profile_features(
                f['diagonal'][:], f['diag_valid_idx'][:])
            # That variant swaps diag_median_max_ratio for valid_frac in the
            # global vector; the first two columns are identical.
            diag_global = np.stack(
                [global_feat[:, 0], global_feat[:, 1], valid_frac], axis=-1
            ).astype(np.float32)
        labels = f['labels'][:]
        strength_bands, metric_name = _read_strength(f)

    strength_max, strength_min = reduce_jsr_bands(strength_bands)

    return {
        'eigen': eigen,
        'global': global_feat,
        'diag': diag_profile,
        'global_diag': diag_global,
        'labels': labels,
        'jsr_db': strength_max,      # kept under the old key for compatibility
        'jsr_max': strength_max,
        'jsr_min': strength_min,
        'metric_name': metric_name,
        'source': f'{dir_name}__{stem}',
        'path': fpath,
    }


def load_synthetic_files(data_dir):
    """Load a data directory as a LIST of per-file datasets, in sorted order."""
    dir_name = os.path.basename(os.path.normpath(data_dir))

    # Standard test.h5 (preprocessed features) is a single-file directory.
    test_file = os.path.join(data_dir, 'test.h5')
    if os.path.exists(test_file):
        return [load_synthetic_file(test_file, dir_name)]

    # Otherwise, separate per-polarization files with raw eigenvalues.
    h5_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.h5')])
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {data_dir}")

    return [load_synthetic_file(os.path.join(data_dir, h), dir_name)
            for h in h5_files]


def load_synthetic_data(data_dir):
    """
    Load synthetic test data from a data directory, pooled across its files.

    The individual files stay reachable under 'files' so per-file plots can be
    made without re-reading or re-predicting anything.
    """
    files = load_synthetic_files(data_dir)

    eigen = np.concatenate([fd['eigen'] for fd in files], axis=0)
    global_feat = np.concatenate([fd['global'] for fd in files], axis=0)
    labels = np.concatenate([fd['labels'] for fd in files], axis=0)

    # Diagonal-profile inputs, only if EVERY file in the directory has them.
    if all(fd['diag'] is not None for fd in files):
        diag = np.concatenate([fd['diag'] for fd in files], axis=0)
        global_diag = np.concatenate([fd['global_diag'] for fd in files], axis=0)
    else:
        diag, global_diag = None, None

    # Pool the strength axis only if every file in the directory has it AND
    # they all recorded the same quantity. Mixed units get dropped rather than
    # silently plotted on a shared axis.
    metric_names = {fd['metric_name'] for fd in files}
    if len(metric_names) == 1 and None not in metric_names:
        metric_name = metric_names.pop()
        jsr_max = np.concatenate([fd['jsr_max'] for fd in files])
        jsr_min = np.concatenate([fd['jsr_min'] for fd in files])
    else:
        metric_name, jsr_max, jsr_min = None, None, None

    return {
        'eigen': eigen,
        'global': global_feat,
        'diag': diag,
        'global_diag': global_diag,
        'labels': labels,
        'jsr_db': jsr_max,
        'jsr_max': jsr_max,
        'jsr_min': jsr_min,
        'metric_name': metric_name,
        'source': os.path.basename(os.path.normpath(data_dir)),
        'files': files,
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

    # Where each individual test FILE lives in the pooled arrays, so per-file
    # plots reuse the single batched prediction below instead of re-running it.
    file_slices, offset = [], 0
    for d in all_data:
        for fd in d['files']:
            n = len(fd['labels'])
            file_slices.append((fd, offset, offset + n))
            offset += n
    assert offset == len(labels_all), 'per-file offsets do not cover the pooled set'

    # JSR (if available), kept as two reductions of the per-band spread:
    # strongest band (detectability) and weakest band (correct knee count).
    # The pooled plot needs every dataset to report the same quantity.
    metric_names = {d['metric_name'] for d in all_data}
    has_jsr = (all(d['jsr_max'] is not None for d in all_data)
               and len(metric_names) == 1 and None not in metric_names)
    if has_jsr:
        metric_name_all = metric_names.pop()
        jsr_max_all = np.concatenate([d['jsr_max'] for d in all_data])
        jsr_min_all = np.concatenate([d['jsr_min'] for d in all_data])
    else:
        metric_name_all = None
        jsr_max_all, jsr_min_all = None, None

    print(f"\nTotal samples: {len(labels_all)}")
    for i, d in enumerate(all_data):
        print(f"  {d['source']}: {len(d['labels'])} samples "
              f"({len(d['files'])} file(s))")

    # Pick the input set from the model's arity. Two inputs = the [eigen, global]
    # models from model.py; three = the [eigen, diag, global] models from
    # model_diag.py. The order here must match build_model_diag's Input order.
    n_model_inputs = len(model.inputs)
    if n_model_inputs == 3:
        missing = [d['source'] for d in all_data if d['diag'] is None]
        if missing:
            raise ValueError(
                f"Model takes 3 inputs (sorted diagonal profile), but these test "
                f"directories carry no raw 'diagonal' dataset to build it from: "
                f"{', '.join(missing)}. Preprocessed 'eigen_features' files cannot "
                f"be used with this model variant; regenerate them with raw "
                f"eigenvalues + diagonal."
            )
        diag_all = np.concatenate([d['diag'] for d in all_data])
        global_all = np.concatenate([d['global_diag'] for d in all_data])
        model_inputs = [eigen_all, diag_all, global_all]
        print(f"\n  model inputs: eigen {eigen_all.shape}, diag {diag_all.shape}, "
              f"global {global_all.shape}  [sorted diagonal profile variant]")
    elif n_model_inputs == 2:
        model_inputs = [eigen_all, global_all]
        print(f"\n  model inputs: eigen {eigen_all.shape}, "
              f"global {global_all.shape}  [baseline variant]")
    else:
        raise ValueError(f"Unsupported model with {n_model_inputs} inputs; "
                         f"expected 2 (baseline) or 3 (diagonal profile).")

    # Predict
    print("\nPredicting...")
    probs = model.predict(model_inputs, batch_size=args.batch_size, verbose=1)
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
        'per_file': {},      # populated in the per-file loop below
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
    if has_jsr and jsr_max_all is not None:
        plot_accuracy_vs_jsr(labels_all, preds, jsr_max_all, n_classes, args.output_dir,
                             title='Combined Datasets', jsr_min=jsr_min_all,
                             metric_name=metric_name_all)
        results['strength_metric'] = metric_name_all
        results['jsr_analysis'] = analyze_jsr_performance(
            labels_all, preds, jsr_max_all, n_classes, metric_name_all)
        results['jsr_analysis_weakest_band'] = analyze_jsr_performance(
            labels_all, preds, jsr_min_all, n_classes, metric_name_all)

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

    # Per-FILE accuracy vs strength. One figure per test .h5 (so each
    # polarization and each JSR band of a directory gets its own curves),
    # each with the strongest-band and weakest-band panels side by side.
    print(f"\n{'-'*70}")
    print('PER-FILE ACCURACY VS STRENGTH')
    print(f"{'-'*70}")
    for fd, lo, hi in file_slices:
        labels_f = labels_all[lo:hi]
        preds_f = preds[lo:hi]
        acc_f = float(np.mean(preds_f == labels_f))
        name = fd['source']

        entry = {
            'path': fd['path'],
            'n_samples': int(hi - lo),
            'accuracy': acc_f,
            'strength_metric': fd['metric_name'],
        }
        print(f"  {name}: {100*acc_f:.2f}%  ({hi - lo} samples)")

        if fd['jsr_max'] is None:
            print(f"    no JSR or RFI power recorded in this file; no plot")
        else:
            plot_accuracy_vs_jsr(
                labels_f, preds_f, fd['jsr_max'], n_classes, args.output_dir,
                title=name, jsr_min=fd['jsr_min'],
                out_name=f'accuracy_vs_jsr_{name}.png',
                metric_name=fd['metric_name'],
            )
            entry['strength_analysis'] = analyze_jsr_performance(
                labels_f, preds_f, fd['jsr_max'], n_classes, fd['metric_name'])
            entry['strength_analysis_weakest_band'] = analyze_jsr_performance(
                labels_f, preds_f, fd['jsr_min'], n_classes, fd['metric_name'])

        results['per_file'][name] = entry

    # Confidence distribution
    plot_confidence_distribution(confidence, labels_all, preds, n_classes, args.output_dir,
                                 title='Combined Datasets')

    # Save JSON results
    with open(os.path.join(args.output_dir, 'results_combined.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


def strength_bins(metric_name, *value_arrays):
    """
    Bin edges for the x-axis of the accuracy-vs-strength plots.

    JSR keeps the historical fixed -10..34 dB / 2 dB grid so new plots line up
    with the ones already in results/. Any other quantity (absolute RFI power,
    whose useful range depends on the scene) gets 22 equal bins spanning the
    1st-99th percentile of the data, which keeps a couple of wild tiles from
    squashing every curve into the leftmost bin.

    Returns None if there is nothing finite to bin.
    """
    if metric_name is not None and metric_name.startswith('JSR'):
        return np.arange(-10, 35, 2)

    finite = np.concatenate([
        np.asarray(v, dtype=np.float64)[np.isfinite(v)]
        for v in value_arrays if v is not None
    ]) if value_arrays else np.array([])

    if finite.size == 0:
        return None

    lo, hi = np.percentile(finite, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo:
            hi = lo + 1.0
    return np.linspace(lo, hi, 23)


def plot_accuracy_vs_jsr(labels, preds, jsr_db, n_classes, out_dir, title='',
                         jsr_min=None, out_name='accuracy_vs_jsr.png',
                         metric_name='JSR (dB)'):
    """
    Plot accuracy vs JSR (or whatever per-band strength the file recorded) for
    each class.

    jsr_db is the STRONGEST band in each tile. Pass jsr_min (the weakest band)
    to get a second panel beside it: same curves, but binned by the faintest
    emitter the tile contains. The two panels bracket the multi-band tiles --
    the left says how hard the tile was to notice, the right how hard it was to
    count -- and they are identical for single-band and clean tiles, which have
    no spread. A right panel that lags the left is the model missing the faint
    members of a multi-band tile and under-calling the knee.

    Returns the path written, or None if there was nothing to bin.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    panels = [(jsr_db, 'strongest band in tile')]
    if jsr_min is not None:
        panels.append((jsr_min, 'weakest band in tile'))

    bins = strength_bins(metric_name, *[v for v, _ in panels])
    if bins is None:
        print(f"  (no finite {metric_name} values for {title}; skipping plot)")
        return None
    bin_centers = (bins[:-1] + bins[1:]) / 2

    fig, axes = plt.subplots(1, len(panels), figsize=(10 * len(panels), 6),
                             squeeze=False, sharey=True)
    axes = axes[0]

    class_names = ['clean'] + [(f'{k} RFI eigenvalue' if k == 1 else f'{k} RFI eigenvalues')
                                 for k in range(1, n_classes)]

    for ax, (jsr_vals, panel_label) in zip(axes, panels):
        for k in range(n_classes):
            mask = (labels == k)
            if mask.sum() < 10:
                continue

            accs = []
            for i in range(len(bins) - 1):
                bin_mask = mask & (jsr_vals >= bins[i]) & (jsr_vals < bins[i+1])
                if bin_mask.sum() > 0:
                    accs.append(np.mean(preds[bin_mask] == labels[bin_mask]))
                else:
                    accs.append(np.nan)

            ax.plot(bin_centers, accs, marker='o', label=class_names[k], linewidth=2)

        ax.set_xlabel(f'{metric_name} - {panel_label}')
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_title(panel_label)

    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    fig.suptitle(f'Accuracy vs {metric_name} - {title}')

    fig.tight_layout()
    out_path = os.path.join(out_dir, out_name)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def analyze_jsr_performance(labels, preds, jsr_db, n_classes,
                            metric_name='JSR (dB)'):
    """
    Analyze performance across strength ranges.

    JSR uses the fixed decade-ish bands it always has. A non-JSR quantity
    (absolute RFI power) has no canonical bands, so it gets quartiles of its
    own finite values instead.
    """
    if metric_name is not None and metric_name.startswith('JSR'):
        jsr_ranges = [(-10, 0), (0, 10), (10, 20), (20, 35)]
    else:
        finite = np.asarray(jsr_db, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return {}
        edges = np.percentile(finite, [0, 25, 50, 75, 100])
        edges[-1] = np.nextafter(edges[-1], np.inf)   # keep the max in the top bin
        jsr_ranges = [(float(edges[i]), float(edges[i+1])) for i in range(4)]

    results = {}
    for low, high in jsr_ranges:
        mask = (jsr_db >= low) & (jsr_db < high)
        if mask.sum() > 0:
            acc = float(np.mean(preds[mask] == labels[mask]))
            results[f'{low:g}to{high:g}dB'] = {
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
