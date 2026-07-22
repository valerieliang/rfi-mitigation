"""
train_only.py

Training-only script for the CNN-based RFI knee-index classifier.

Loads training data from --data-dir (can combine Amazon and mountain datasets),
splits into train/val, trains the model, and saves it. Does NOT run any test
evaluations - use test_only.py for that.

Usage:
    python train_only.py \
        --data-dir data/amazon_train data/mountain_train \
        --run-name my_model \
        --epochs 50

Outputs:
    models/<run>/best_model.keras
    models/<run>/training_curves.png
    models/<run>/training_summary.json
"""

import os
import sys
import json
import argparse
import glob

import numpy as np
import h5py
import tensorflow as tf

# Import model builder
from model import build_model


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

MODELS_ROOT = 'models'

M = 16                 # pulses per CPI = number of eigenvalues available
N_KEEP = 12            # eigenvalues actually used as features (the 12 largest)
N_GLOBAL = 3           # global features: [condition_number_db, eff_rank, diag_median_max_ratio]

EPOCHS = 50
BATCH_SIZE = 256
LR = 3e-4
VAL_FRAC = 0.15        # of the training region

# Pulse tiles dropped at the train/val boundary
SPLIT_BUFFER_DEFAULT = 8

EPS = 1e-12
DB_FLOOR = -100.0      # floor for dB values

# File-name patterns. '*clean_data_*.h5' catches both the generic select_clean
# output (clean_data_A_HH.h5) and any scene-tagged variant (mountain_clean_data_*,
# amazon_clean_data_*, ...).
TILE_FILE_PATTERNS = ('rfi_data_*.h5', 'mountain_rfi_data_*.h5', '*clean_data_*.h5')
PAIRED_FILE_PATTERN = 'mountain_rfi_data_paired_*.h5'

# Multiplier for train/val split key
GROUP_OFFSET = 10_000_000


# ---------------------------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def _glob_tile_files(data_dir):
    """
    Non-paired tile-record files in a directory.
    """
    found = set()
    for pattern in TILE_FILE_PATTERNS:
        found.update(glob.glob(os.path.join(data_dir, pattern)))
    found -= set(glob.glob(os.path.join(data_dir, PAIRED_FILE_PATTERN)))
    return sorted(found)


def load_rfi_data_dir(data_dirs, max_samples=None, tag=''):
    """
    Load every tile-record file from one or more directories, concatenating
    them into a single pool.
    """
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    eigen_parts, global_parts = [], []
    label_parts, group_parts, source_parts = [], [], []
    n_classes = 0
    meta = {'dirs': list(data_dirs), 'files': [], 'channels': [], 'pulse_window': None,
            'source_names': [os.path.basename(os.path.normpath(d)) for d in data_dirs]}

    print(f"\nLoading {tag or data_dirs} ...")

    for source_id, data_dir in enumerate(data_dirs):
        paths = _glob_tile_files(data_dir)
        if not paths:
            raise FileNotFoundError(
                f"No tile record files (rfi_data_*.h5 / mountain_rfi_data_*.h5) "
                f"found in {data_dir}"
            )

        for path in paths:
            with h5py.File(path, 'r') as f:
                n_rec = int(f.attrs.get('n_records', f['labels'].shape[0]))
                freq = str(f.attrs.get('frequency', '?'))
                pol = str(f.attrs.get('polarization', '?'))
                chan = f'{freq}-{pol}'

                seed = int(f.attrs.get('seed', -1))
                p0 = int(f.attrs.get('pulse_start', -1)) if 'pulse_start' in f.attrs else -1
                p1 = int(f.attrs.get('pulse_end', -1)) if 'pulse_end' in f.attrs else -1
                min_b = int(f.attrs.get('min_bands', -1))
                max_b = int(f.attrs.get('max_bands', -1))

                if 'diagonal' not in f or 'diag_valid_idx' not in f:
                    raise ValueError(
                        f"{path} has no 'diagonal'/'diag_valid_idx' dataset. "
                        f"Regenerate it with the current generator scripts."
                    )

                sel = slice(None)
                if max_samples is not None and n_rec > max_samples:
                    stride = int(np.ceil(n_rec / max_samples))
                    sel = slice(0, n_rec, stride)

                eigvals = f['eigenvalues'][sel]              # (n, M) linear, descending
                labels = f['labels'][sel].astype(np.int32)
                groups_raw = f['tile_pulse'][sel].astype(np.int64)

                diag = np.asarray(f['diagonal'][sel], dtype=np.float64)
                diag_valid = np.asarray(f['diag_valid_idx'][sel], dtype=bool)

                eigen, global_ = features_from_eigenvalues(eigvals, diag, diag_valid)

                eigen_parts.append(eigen)
                global_parts.append(global_)
                label_parts.append(labels)
                group_parts.append(source_id * GROUP_OFFSET + groups_raw)
                source_parts.append(np.full(len(labels), source_id, dtype=np.int32))

                n_classes = max(n_classes, int(f.attrs.get('n_classes', 0)))

                print(f"  {os.path.basename(path)}: {len(labels)} tiles  "
                      f"[{chan}, pulses {p0}-{p1}, bands {min_b}..{max_b}, seed={seed}]")

                meta['files'].append(os.path.basename(path))
                if chan not in meta['channels']:
                    meta['channels'].append(chan)
                if p0 != -1:
                    meta['pulse_window'] = [p0, p1]
                meta['seed'] = seed
                meta['min_bands'] = min_b
                meta['max_bands'] = max_b

    labels = np.concatenate(label_parts, axis=0)

    return {
        'eigen': np.concatenate(eigen_parts, axis=0),
        'global': np.concatenate(global_parts, axis=0),
        'labels': labels,
        'groups': np.concatenate(group_parts, axis=0),
        'sources': np.concatenate(source_parts, axis=0),
        'n_classes': n_classes or int(labels.max()) + 1,
        'meta': meta,
    }


def split_train_val(groups, val_frac, buffer_tiles):
    """
    Split the training region into train / val by CONTIGUOUS pulse-tile blocks.
    """
    uniq = np.unique(groups)          # sorted ascending
    n = len(uniq)

    n_val = int(round(val_frac * n))
    n_train = n - n_val
    if n_train <= 0 or n_val <= 0:
        raise ValueError(f'Not enough distinct pulse tiles ({n}) to split')

    train_g = uniq[:n_train]
    val_g = uniq[n_train:]

    # Drop a buffer on each side of the single internal boundary
    if buffer_tiles > 0:
        if len(train_g) > buffer_tiles:
            train_g = train_g[:-buffer_tiles]
        if len(val_g) > buffer_tiles:
            val_g = val_g[buffer_tiles:]

    idx_train = np.where(np.isin(groups, train_g))[0]
    idx_val = np.where(np.isin(groups, val_g))[0]

    print(f"\n  Train/val block split on tile_pulse (buffer={buffer_tiles} pulse tiles):")
    print(f"    train pulses [{train_g.min()}, {train_g.max()}] -> {len(idx_train)} tiles")
    print(f"    val   pulses [{val_g.min()}, {val_g.max()}] -> {len(idx_val)} tiles")

    return idx_train, idx_val


def _class_tag(u):
    """Human label for a knee class (matches the reporting used elsewhere)."""
    return ('clean' if u == 0
            else (f'{u} RFI eig' if u == 1 else f'{u} RFI eigs'))


def report_class_by_source_split(labels, sources, source_names,
                                 idx_train, idx_val, n_classes):
    """
    Per-source class separation, shown for the train and val splits.

    For each source selection (e.g. the amazon dir and the mountain dir) print
    the number of clean / 1 RFI eig / ... tiles it contributed, and how those
    split into train and val, so the class balance of BOTH splits is visible
    per source. A final ALL block sums across sources. Tiles dropped in the
    split buffer count toward 'total' but neither 'train' nor 'val', so
    total >= train + val.

    Returns a nested dict suitable for the training summary JSON.
    """
    labels = np.asarray(labels)
    sources = np.asarray(sources)

    split_id = np.full(len(labels), -1, dtype=np.int8)
    split_id[idx_train] = 0
    split_id[idx_val] = 1

    print(f"\n{'='*70}")
    print("Class separation by source  (total = train + val + split-buffer drops)")
    print(f"{'='*70}")

    report = {}
    src_iter = list(enumerate(source_names)) + [('ALL', 'ALL')]

    for sid, name in src_iter:
        if sid == 'ALL':
            mask_src = np.ones(len(labels), dtype=bool)
        else:
            mask_src = (sources == sid)

        print(f"\n  {name}:")
        print(f"    {'class':<12}  {'total':>8}  {'train':>8}  {'val':>8}")
        src_report = {}
        tot_t = tot_tr = tot_va = 0
        for u in range(n_classes):
            in_class = mask_src & (labels == u)
            n_tot = int(np.count_nonzero(in_class))
            n_tr = int(np.count_nonzero(in_class & (split_id == 0)))
            n_va = int(np.count_nonzero(in_class & (split_id == 1)))
            tot_t += n_tot
            tot_tr += n_tr
            tot_va += n_va
            print(f"    {_class_tag(u):<12}  {n_tot:>8}  {n_tr:>8}  {n_va:>8}")
            src_report[str(u)] = {'tag': _class_tag(u),
                                  'total': n_tot, 'train': n_tr, 'val': n_va}
        print(f"    {'TOTAL':<12}  {tot_t:>8}  {tot_tr:>8}  {tot_va:>8}")
        src_report['total'] = {'total': tot_t, 'train': tot_tr, 'val': tot_va}
        report[name] = src_report

    return report


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def save_training_curves_png(history, out_dir):
    """Loss and accuracy vs epoch -> out_dir/training_curves.png."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = range(1, len(history.history['loss']) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(epochs, history.history['loss'], label='Train loss')
    ax_loss.plot(epochs, history.history['val_loss'], label='Val loss')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Training & Validation Loss')
    ax_loss.legend()
    ax_loss.grid(True, linestyle='--', alpha=0.5)

    acc_key = 'acc' if 'acc' in history.history else 'sparse_categorical_accuracy'
    val_acc_key = ('val_acc' if 'val_acc' in history.history
                   else 'val_sparse_categorical_accuracy')
    ax_acc.plot(epochs, history.history[acc_key], label='Train acc')
    ax_acc.plot(epochs, history.history[val_acc_key], label='Val acc')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_title('Training & Validation Accuracy')
    ax_acc.legend()
    ax_acc.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    path = os.path.join(out_dir, 'training_curves.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def train_model(run_name, n_classes,
                eigen_train, global_train, y_train,
                eigen_val, global_val, y_val,
                epochs, batch_size, learning_rate, dropout_rate, weight_decay,
                models_root=MODELS_ROOT):
    """
    Build and train the model on the training region, validating on the held-out
    pulse-tile block of that same region.

    Returns:
        model, out_dir
    """
    out_dir = os.path.join(models_root, run_name)
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'best_model.keras')

    print(f"\n{'='*60}")
    print(f"Run : {run_name}")
    print(f"  train={len(y_train)}  val={len(y_val)}")
    print(f"  eigen input : ({N_KEEP}, 2)   global: ({N_GLOBAL},)   classes: {n_classes}")
    print(f"{'='*60}")

    model = build_model(
        cpi_size=N_KEEP,
        n_global_features=N_GLOBAL,
        n_knee_classes=n_classes,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path, monitor='val_loss',
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10,
            min_lr=1e-6, verbose=1,
        ),
    ]

    history = model.fit(
        x=[eigen_train, global_train],
        y=y_train,
        validation_data=([eigen_val, global_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    save_training_curves_png(history, out_dir)
    return model, out_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description=(f'Train the RFI knee classifier on tile records from '
                     f'generate_amazon_data.py and generate_mountain_rfi_data.py '
                     f'(top {N_KEEP} eigenvalues). Saves trained model for later testing.'),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data-dir', type=str, nargs='+',
                        default=['data/amazon_train', 'data/mountain_train'],
                        help='One or more training directories, concatenated into a '
                             'single train+val pool.')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Model output folder. Defaults to the first data dir name.')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Cap tiles loaded per file (evenly strided), for quick runs.')
    parser.add_argument('--val-frac', type=float, default=VAL_FRAC,
                        help='Fraction of training-region pulse tiles held out for val.')
    parser.add_argument('--split-buffer', type=int, default=SPLIT_BUFFER_DEFAULT,
                        help='Pulse tiles dropped at the train/val boundary.')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--learning-rate', type=float, default=LR,
                        help='Initial learning rate.')
    parser.add_argument('--dropout-rate', type=float, default=0.6,
                        help='Dropout rate in the fusion head.')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='L2 regularization strength (AdamW weight decay).')
    return parser.parse_args()


def main():
    """
    Train the model and save it. Does NOT run any test evaluation.
    Use test_only.py to evaluate the saved model on test datasets.
    """
    args = parse_args()

    os.makedirs(MODELS_ROOT, exist_ok=True)
    run_name = args.run_name or os.path.basename(os.path.normpath(args.data_dir[0]))

    print(f"\n{'='*70}")
    print('RFI knee classifier - TRAINING ONLY')
    print(f"{'='*70}")
    print(f"  train region(s)   : {', '.join(args.data_dir)}")
    print(f"  features          : top {N_KEEP} of {M} eigenvalues, linear-normalized then dB")
    print(f"  run name          : {run_name}")
    print(f"  epochs            : {args.epochs}")
    print(f"  batch size        : {args.batch_size}")
    print(f"  learning rate     : {args.learning_rate}")
    print(f"  dropout rate      : {args.dropout_rate}")
    print(f"  weight decay      : {args.weight_decay}")

    # Load training data
    train_data = load_rfi_data_dir(args.data_dir, args.max_samples, tag='TRAIN region')
    n_classes = train_data['n_classes']

    print(f"\nTraining region: {len(train_data['labels'])} tiles   "
          f"classes: {n_classes}   channels: {', '.join(train_data['meta']['channels'])}")
    uniq, counts = np.unique(train_data['labels'], return_counts=True)
    for u, c in zip(uniq.tolist(), counts.tolist()):
        tag = ('clean' if u == 0
               else (f'{u} RFI eig' if u == 1 else f'{u} RFI eigs'))
        print(f"  label {u} ({tag}): {c}")

    # Split train/val
    idx_train, idx_val = split_train_val(
        train_data['groups'], args.val_frac, args.split_buffer
    )

    # Per-source class separation, broken out by train / val split
    class_distribution = report_class_by_source_split(
        train_data['labels'], train_data['sources'],
        train_data['meta']['source_names'],
        idx_train, idx_val, n_classes,
    )

    # Train model
    model, out_dir = train_model(
        run_name, n_classes,
        train_data['eigen'][idx_train], train_data['global'][idx_train],
        train_data['labels'][idx_train],
        train_data['eigen'][idx_val], train_data['global'][idx_val],
        train_data['labels'][idx_val],
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, dropout_rate=args.dropout_rate,
        weight_decay=args.weight_decay,
    )

    # Save training summary
    summary = {
        'run': run_name,
        'n_train': int(len(idx_train)),
        'n_val': int(len(idx_val)),
        'n_classes': n_classes,
        'n_keep': N_KEEP,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'dropout_rate': args.dropout_rate,
        'weight_decay': args.weight_decay,
        'class_distribution': class_distribution,
        'train_provenance': train_data['meta'],
    }

    summary_path = os.path.join(out_dir, 'training_summary.json')
    with open(summary_path, 'w') as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Model saved to: {os.path.join(out_dir, 'best_model.keras')}")
    print(f"  Summary saved to: {summary_path}")
    print(f"\nTo test this model, run:")
    print(f"  python test_only.py --model-dir models/{run_name} --test-dir <test_dir>")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
