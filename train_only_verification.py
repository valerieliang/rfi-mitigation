"""
train_only_verification.py

Training-only script for the CNN-based RFI knee-index classifier that combines
two very different training sources:

  1. Amazon dataset (rfi_data_*.h5 tile records from generate_rfi_data.py /
     generate_lumpy.py): scattered-target terrain that carries the FULL range
     of knee labels - clean sentinel (0) plus every injected knee class
     (1..N). This is where the model learns "all behavior".

  2. clean_mountains_filtered.h5 (identify_clean_profiles.py output): CPI
     eigenvalue profiles from a mountainous granule that has been filtered
     down to tiles verified to be clean. Every tile from this source is
     forced to label 0 (clean sentinel) regardless of channel, since the
     whole point of this source is "only clean behavior" on a terrain type
     that looks nothing like Amazon rainforest clutter.

The two sources are loaded and train/val split INDEPENDENTLY (each keeps its
own contiguous pulse-tile block split), then concatenated into one train pool
and one val pool for actual model fitting. After training, validation metrics
are reported BOTH combined and per-source, so the Amazon val split verifies
knee-detection behavior across all classes, and the mountain val split
verifies the false-alarm rate on out-of-distribution clean terrain (Type B
error in ST-EST terms: any non-zero prediction here is a false alarm).

Usage:
    python train_only_verification.py \
        --amazon-dir data/amazon_train \
        --mountains-file data/clean_mountains_filtered.h5 \
        --run-name amazon_mountains_verification \
        --epochs 50

Outputs:
    models/<run>/best_model.keras
    models/<run>/training_curves.png
    models/<run>/training_summary.json   (includes per-source val metrics)
"""

import os
import re
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
VAL_FRAC = 0.15        # of each source region, independently

# Pulse tiles dropped at the train/val boundary
SPLIT_BUFFER_DEFAULT = 8

EPS = 1e-12
DB_FLOOR = -100.0      # floor for dB values

# File-name patterns (Amazon-side tile record files)
TILE_FILE_PATTERNS = ('rfi_data_*.h5', 'mountain_rfi_data_*.h5')
PAIRED_FILE_PATTERN = 'mountain_rfi_data_paired_*.h5'

# Multiplier for train/val split key when combining multiple Amazon dirs
GROUP_OFFSET = 10_000_000

# Mountain-side: how much of the SCM diagonal must be flagged valid (as a
# per-tile fraction) for a clean-mountain tile to be trusted. Matches the
# PULSE_VALID_FRAC_THRESH convention used in cpi_preprocess.py.
MOUNTAIN_MIN_DIAG_VALID_FRAC_DEFAULT = 0.8

MOUNTAIN_GROUP_NAME_RE = re.compile(r'^freq_(?P<freq>.+)_pol_(?P<pol>.+)$')


# ---------------------------------------------------------------------------
# FEATURE EXTRACTION (identical to train_only.py, kept self-contained)
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
# AMAZON DATA LOADING (tile-record files, full knee-label range)
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


def load_amazon_data(data_dirs, max_samples=None):
    """
    Load every tile-record file from one or more Amazon directories,
    concatenating them into a single pool. Labels are read as-is (0 = clean
    sentinel, 1..N = knee at eigenvalue index 0..N-1), i.e. the FULL range of
    behavior for scattered-target terrain.
    """
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    eigen_parts, global_parts = [], []
    label_parts, group_parts = [], []
    n_classes = 0
    meta = {'dirs': list(data_dirs), 'files': [], 'channels': [], 'pulse_window': None}

    print(f"\nLoading Amazon data: {data_dirs} ...")

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
        'n_classes': n_classes or int(labels.max()) + 1,
        'meta': meta,
        'source': 'amazon',
    }


# ---------------------------------------------------------------------------
# MOUNTAIN DATA LOADING (clean_mountains_filtered.h5, clean-only)
# ---------------------------------------------------------------------------

def load_mountain_clean_data(path, pols=('HH', 'HV'),
                              min_diag_valid_frac=MOUNTAIN_MIN_DIAG_VALID_FRAC_DEFAULT,
                              max_samples=None):
    """
    Load a clean_mountains_filtered.h5 file produced by
    identify_clean_profiles.py. Every tile in this file is ALREADY verified
    to be clean (no RFI), so every tile is assigned label 0 regardless of
    channel - this is what makes the mountain source "clean-only" in
    contrast to the Amazon source's full label range.

    diag_valid_idx is not stored per-eigenvalue in this file, only a scalar
    diag_valid_frac per tile. Rather than guess which specific diagonal
    entries are valid, tiles below min_diag_valid_frac are dropped, and the
    remaining tiles use all 16 diagonal entries as valid (the filtering
    upstream in identify_clean_profiles.py already screened out low-quality
    tiles via min_valid_eigvals / std_threshold).
    """
    eigen_parts, global_parts, group_parts = [], [], []
    meta = {'file': os.path.basename(path), 'channels': [], 'pulse_window': None,
            'min_diag_valid_frac': min_diag_valid_frac}

    print(f"\nLoading clean-mountain data: {path} ...")

    with h5py.File(path, 'r') as f:
        for k, v in f.attrs.items():
            meta.setdefault('source_attrs', {})[k] = (
                v.item() if hasattr(v, 'item') else v
            )

        group_names = [name for name in f.keys()
                       if isinstance(f[name], h5py.Group)]

        for name in sorted(group_names):
            m = MOUNTAIN_GROUP_NAME_RE.match(name)
            if m is None:
                continue
            freq, pol = m.group('freq'), m.group('pol')
            if pol not in pols:
                continue

            grp = f[name]
            n_rec = grp['eigenvalues'].shape[0]

            sel = slice(None)
            if max_samples is not None and n_rec > max_samples:
                stride = int(np.ceil(n_rec / max_samples))
                sel = slice(0, n_rec, stride)

            eigvals = np.asarray(grp['eigenvalues'][sel], dtype=np.float64)
            diag = np.asarray(grp['diagonal'][sel], dtype=np.float64)
            diag_valid_frac = np.asarray(grp['diag_valid_frac'][sel], dtype=np.float64)
            pulse_idx = np.asarray(grp['pulse_idx'][sel], dtype=np.int64)

            keep = diag_valid_frac >= min_diag_valid_frac
            n_dropped = int((~keep).sum())
            if n_dropped:
                print(f"  {name}: dropping {n_dropped}/{n_rec} tiles below "
                      f"diag_valid_frac={min_diag_valid_frac}")

            eigvals = eigvals[keep]
            diag = diag[keep]
            pulse_idx = pulse_idx[keep]

            if len(eigvals) == 0:
                print(f"  {name}: no tiles remain after filtering, skipping")
                continue

            diag_valid = np.ones_like(diag, dtype=bool)

            eigen, global_ = features_from_eigenvalues(eigvals, diag, diag_valid)

            eigen_parts.append(eigen)
            global_parts.append(global_)
            group_parts.append(pulse_idx)

            chan = f'{freq}-{pol}'
            if chan not in meta['channels']:
                meta['channels'].append(chan)

            print(f"  {name}: {len(eigvals)} clean tiles kept")

    if not eigen_parts:
        raise FileNotFoundError(
            f"No usable freq_*_pol_{{{'/'.join(pols)}}} groups found in {path}"
        )

    eigen = np.concatenate(eigen_parts, axis=0)
    global_ = np.concatenate(global_parts, axis=0)
    groups = np.concatenate(group_parts, axis=0)

    # Every mountain tile is a verified-clean sentinel: label 0.
    labels = np.zeros((len(eigen),), dtype=np.int32)

    p0, p1 = int(groups.min()), int(groups.max())
    meta['pulse_window'] = [p0, p1]

    return {
        'eigen': eigen,
        'global': global_,
        'labels': labels,
        'groups': groups,
        'n_classes': 1,   # only label 0 present; real n_classes comes from Amazon
        'meta': meta,
        'source': 'mountain',
    }


# ---------------------------------------------------------------------------
# TRAIN / VAL SPLIT (per-source, contiguous pulse-tile blocks)
# ---------------------------------------------------------------------------

def split_train_val(groups, val_frac, buffer_tiles):
    """
    Split a source region into train / val by CONTIGUOUS pulse-tile blocks.
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

    print(f"    train pulses [{train_g.min()}, {train_g.max()}] -> {len(idx_train)} tiles")
    print(f"    val   pulses [{val_g.min()}, {val_g.max()}] -> {len(idx_val)} tiles")

    return idx_train, idx_val


def split_source(source_data, val_frac, buffer_tiles, label):
    print(f"\n  Train/val block split for {label} (buffer={buffer_tiles} pulse tiles):")
    idx_train, idx_val = split_train_val(source_data['groups'], val_frac, buffer_tiles)
    return idx_train, idx_val


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
                epochs, batch_size, learning_rate, dropout_rate, weight_decay):
    """
    Build and train the model on the combined training pool, validating on
    the combined held-out pool (Amazon val tiles + mountain val tiles).

    Returns:
        model, out_dir, history
    """
    out_dir = os.path.join(MODELS_ROOT, run_name)
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
    return model, out_dir, history


# ---------------------------------------------------------------------------
# PER-SOURCE VERIFICATION METRICS
# ---------------------------------------------------------------------------

def evaluate_source(model, eigen, global_, labels, source_name):
    """
    Run the trained model on one source's held-out tiles and report metrics
    appropriate to that source's role:

      - amazon: overall accuracy + per-class accuracy (verifies the full
        range of knee behavior was learned).
      - mountain: false-alarm rate = fraction of clean tiles predicted as
        non-zero (Type B error in ST-EST terms - the only error type
        possible here since every true label is 0).
    """
    if len(labels) == 0:
        return {'n_tiles': 0}

    probs = model.predict([eigen, global_], verbose=0)
    preds = np.argmax(probs, axis=1)

    overall_acc = float(np.mean(preds == labels))
    result = {'n_tiles': int(len(labels)), 'overall_accuracy': overall_acc}

    if source_name == 'mountain':
        false_alarm_rate = float(np.mean(preds != 0))
        result['false_alarm_rate'] = false_alarm_rate
        print(f"  [mountain] clean tiles: {len(labels)}  "
              f"false-alarm rate (pred != 0): {false_alarm_rate:.4f}")
    else:
        per_class = {}
        for cls in sorted(np.unique(labels).tolist()):
            mask = labels == cls
            per_class[int(cls)] = float(np.mean(preds[mask] == labels[mask]))
        result['per_class_accuracy'] = per_class
        print(f"  [{source_name}] tiles: {len(labels)}  overall accuracy: {overall_acc:.4f}")
        for cls, acc in per_class.items():
            tag = 'clean' if cls == 0 else f'knee@{cls}'
            print(f"      label {cls} ({tag}): acc={acc:.4f}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            f'Train the RFI knee classifier on Amazon tile records (full '
            f'knee-label behavior, scattered targets) combined with '
            f'clean_mountains_filtered.h5 (forced label 0 only, verifies '
            f'clean-terrain false-alarm behavior). Top {N_KEEP} eigenvalues used.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--amazon-dir', type=str, nargs='+',
                        default=['data/amazon_train'],
                        help='One or more Amazon training directories (rfi_data_*.h5 files).')
    parser.add_argument('--mountains-file', type=str,
                        default='data/clean_mountains_filtered.h5',
                        help='Path to the clean_mountains_filtered.h5 file.')
    parser.add_argument('--mountain-pols', type=str, nargs='+', default=['HH', 'HV'],
                        choices=['HH', 'HV'],
                        help='Which polarization groups to load from the mountain file.')
    parser.add_argument('--mountain-min-diag-valid-frac', type=float,
                        default=MOUNTAIN_MIN_DIAG_VALID_FRAC_DEFAULT,
                        help='Drop mountain tiles below this diag_valid_frac.')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Model output folder. Defaults to amazon_plus_mountains_verification.')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Cap tiles loaded per file/group (evenly strided), for quick runs.')
    parser.add_argument('--val-frac', type=float, default=VAL_FRAC,
                        help='Fraction of each source held out for val, applied independently.')
    parser.add_argument('--split-buffer', type=int, default=SPLIT_BUFFER_DEFAULT,
                        help='Pulse tiles dropped at each source train/val boundary.')
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
    Train the model on Amazon (full behavior) + clean mountains (clean-only)
    and save it, along with per-source held-out verification metrics. Does
    NOT run any external test-set evaluation - use test_only.py for that.
    """
    args = parse_args()

    os.makedirs(MODELS_ROOT, exist_ok=True)
    run_name = args.run_name or 'amazon_plus_mountains_verification'

    print(f"\n{'='*70}")
    print('RFI knee classifier - TRAINING ONLY (Amazon + clean-mountain verification)')
    print(f"{'='*70}")
    print(f"  amazon dir(s)     : {', '.join(args.amazon_dir)}")
    print(f"  mountains file    : {args.mountains_file}")
    print(f"  mountain pols     : {', '.join(args.mountain_pols)}")
    print(f"  features          : top {N_KEEP} of {M} eigenvalues, linear-normalized then dB")
    print(f"  run name          : {run_name}")
    print(f"  epochs            : {args.epochs}")
    print(f"  batch size        : {args.batch_size}")
    print(f"  learning rate     : {args.learning_rate}")
    print(f"  dropout rate      : {args.dropout_rate}")
    print(f"  weight decay      : {args.weight_decay}")

    # ------------------------------------------------------------------
    # Load both sources
    # ------------------------------------------------------------------
    amazon_data = load_amazon_data(args.amazon_dir, args.max_samples)
    mountain_data = load_mountain_clean_data(
        args.mountains_file,
        pols=tuple(args.mountain_pols),
        min_diag_valid_frac=args.mountain_min_diag_valid_frac,
        max_samples=args.max_samples,
    )

    n_classes = amazon_data['n_classes']   # mountain source only ever has label 0

    print(f"\nAmazon region: {len(amazon_data['labels'])} tiles   "
          f"classes: {n_classes}   channels: {', '.join(amazon_data['meta']['channels'])}")
    uniq, counts = np.unique(amazon_data['labels'], return_counts=True)
    for u, c in zip(uniq.tolist(), counts.tolist()):
        print(f"  label {u} ({'clean' if u == 0 else f'knee@{u}'}): {c}")

    print(f"\nMountain region: {len(mountain_data['labels'])} tiles   "
          f"(forced label 0, clean-only)   channels: {', '.join(mountain_data['meta']['channels'])}")

    # Sanity check: this source must be 100% clean by construction.
    assert np.all(mountain_data['labels'] == 0), (
        "Mountain source produced a non-zero label - this should be impossible "
        "since load_mountain_clean_data() forces label 0 for every tile."
    )

    # ------------------------------------------------------------------
    # Split each source independently, then merge
    # ------------------------------------------------------------------
    idx_train_amz, idx_val_amz = split_source(
        amazon_data, args.val_frac, args.split_buffer, 'Amazon')
    idx_train_mtn, idx_val_mtn = split_source(
        mountain_data, args.val_frac, args.split_buffer, 'Mountain')

    eigen_train = np.concatenate([
        amazon_data['eigen'][idx_train_amz], mountain_data['eigen'][idx_train_mtn]
    ], axis=0)
    global_train = np.concatenate([
        amazon_data['global'][idx_train_amz], mountain_data['global'][idx_train_mtn]
    ], axis=0)
    y_train = np.concatenate([
        amazon_data['labels'][idx_train_amz], mountain_data['labels'][idx_train_mtn]
    ], axis=0)

    eigen_val = np.concatenate([
        amazon_data['eigen'][idx_val_amz], mountain_data['eigen'][idx_val_mtn]
    ], axis=0)
    global_val = np.concatenate([
        amazon_data['global'][idx_val_amz], mountain_data['global'][idx_val_mtn]
    ], axis=0)
    y_val = np.concatenate([
        amazon_data['labels'][idx_val_amz], mountain_data['labels'][idx_val_mtn]
    ], axis=0)

    print(f"\nCombined pool: train={len(y_train)}  val={len(y_val)}  "
          f"(amazon train={len(idx_train_amz)}, mountain train={len(idx_train_mtn)}, "
          f"amazon val={len(idx_val_amz)}, mountain val={len(idx_val_mtn)})")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    model, out_dir, history = train_model(
        run_name, n_classes,
        eigen_train, global_train, y_train,
        eigen_val, global_val, y_val,
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, dropout_rate=args.dropout_rate,
        weight_decay=args.weight_decay,
    )

    # ------------------------------------------------------------------
    # Per-source verification on held-out tiles
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Per-source verification on held-out tiles")
    print(f"{'='*60}")
    amazon_val_metrics = evaluate_source(
        model, amazon_data['eigen'][idx_val_amz], amazon_data['global'][idx_val_amz],
        amazon_data['labels'][idx_val_amz], 'amazon')
    mountain_val_metrics = evaluate_source(
        model, mountain_data['eigen'][idx_val_mtn], mountain_data['global'][idx_val_mtn],
        mountain_data['labels'][idx_val_mtn], 'mountain')

    # ------------------------------------------------------------------
    # Save training summary
    # ------------------------------------------------------------------
    summary = {
        'run': run_name,
        'n_train': int(len(y_train)),
        'n_val': int(len(y_val)),
        'n_train_amazon': int(len(idx_train_amz)),
        'n_train_mountain': int(len(idx_train_mtn)),
        'n_val_amazon': int(len(idx_val_amz)),
        'n_val_mountain': int(len(idx_val_mtn)),
        'n_classes': n_classes,
        'n_keep': N_KEEP,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'dropout_rate': args.dropout_rate,
        'weight_decay': args.weight_decay,
        'amazon_provenance': amazon_data['meta'],
        'mountain_provenance': mountain_data['meta'],
        'amazon_val_metrics': amazon_val_metrics,
        'mountain_val_metrics': mountain_val_metrics,
    }

    summary_path = os.path.join(out_dir, 'training_summary.json')
    with open(summary_path, 'w') as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"  Model saved to: {os.path.join(out_dir, 'best_model.keras')}")
    print(f"  Summary saved to: {summary_path}")
    print(f"\nTo test this model, run:")
    print(f"  python test_only.py --model-dir models/{run_name} --test-dir <test_dir>")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()