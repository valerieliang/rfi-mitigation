"""
train_branch_updates.py

Trains model_branch_updates with two-branch structure:
  Branch 1: 12 EVs + 11 EV slopes + 12 SCM diagonal + [max EV, max pulse power]
            = 37 features (all normalized)
  Branch 2: 3 global scalars (condition_number_db, eff_rank, diag_median_max_ratio)

This follows the new convention where:
- EVs are the 12 valid eigenvalues (out of 16), max-normalized
- EV slopes are the 11 differences between consecutive EVs
- SCM diagonal are the 12 valid diagonal entries (max-normalized)
- The 2-element emphasis vector [max EV, max pulse power] reinforces the
  lossless transformation relationship between EV and SCM diagonal
- Global scalars provide contextual information about the overall tile structure

Branch 1 contains the detailed feature profiles (37 features).
Branch 2 contains summary statistics (3 scalars).

Usage:
    python train_branch_updates.py --run-name model_branch_updates --epochs 100 \
        --batch-size 256 --learning-rate 3e-4 --dropout-rate 0.6 \
        --weight-decay 1e-4

Outputs:
    models/<run>/best_model.keras
    models/<run>/training_curves.png
    models/<run>/training_summary.json
"""

import os
import json
import argparse

import numpy as np
import h5py
import tensorflow as tf

from model_branch_updates import build_model_branch_updates
from train_only import (
    MODELS_ROOT, M, EPOCHS, BATCH_SIZE, LR, VAL_FRAC,
    SPLIT_BUFFER_DEFAULT, GROUP_OFFSET, EPS, DB_FLOOR,
    _glob_tile_files,
    split_train_val,
    report_class_by_source_split,
    save_training_curves_png,
)

# Feature dimensions for the combined branch
N_KEEP = 12                      # valid EVs kept (out of 16)
N_EV_SLOPES = N_KEEP - 1         # slopes between consecutive EVs
N_DIAG = 12                      # valid diagonal entries kept
N_EMPHASIS = 2                   # [max EV, max pulse power]
N_BRANCH_FEATURES = N_KEEP + N_EV_SLOPES + N_DIAG + N_EMPHASIS  # 37 total

# The three directories the urClean_mtnContam_amzContam benchmark trained on
BENCHMARK_DIRS = ['data/czech_contam', 'data/amazon_contam', 'data/berlin_clean']


# ---------------------------------------------------------------------------
# FEATURE ASSEMBLY
# ---------------------------------------------------------------------------

def features_from_records(eigvals_linear, diag_lin, diag_valid_idx):
    """
    Build the two branch input tensors.

    Returns:
        branch1: (N, N_BRANCH_FEATURES) float32 array containing:
                 - 12 valid EVs (max-normalized, in dB)
                 - 11 EV slopes
                 - 12 valid diagonal entries (max-normalized, in dB)
                 - 2 emphasis values [max EV linear, max pulse power linear]
                   (normalized by global max across batch for scale-free comparison)
        global_: (N, 3) float32 array containing:
                 - condition number (dB)
                 - effective rank
                 - diagonal median/max ratio
    """
    single = (np.asarray(eigvals_linear).ndim == 1)
    ev = np.atleast_2d(np.asarray(eigvals_linear, dtype=np.float64))
    diag = np.atleast_2d(np.asarray(diag_lin, dtype=np.float64))
    valid = np.atleast_2d(np.asarray(diag_valid_idx, dtype=bool))

    n = ev.shape[0]

    # --- EVs: keep the 12 largest, normalize by lambda_max, convert to dB ---
    ev_keep = ev[:, :N_KEEP]
    ev_keep = np.maximum(ev_keep, EPS)
    lam_max = np.maximum(ev_keep[:, 0:1], EPS)
    ev_norm = ev_keep / lam_max
    ev_db = 10.0 * np.log10(np.maximum(ev_norm, EPS))
    ev_db = np.maximum(ev_db, DB_FLOOR)

    # --- EV slopes: differences between consecutive EVs ---
    ev_slopes = np.diff(ev_db, axis=1)  # shape (n, 11)

    # --- SCM diagonal: keep largest 12 valid entries, max-normalize, dB ---
    n_valid = valid.sum(axis=1)
    rows = np.arange(n)

    # Sort descending with invalid entries forced to the back
    keyed = np.where(valid, diag, -np.inf)
    order = np.argsort(-keyed, axis=1, kind='stable')
    srt = np.take_along_axis(keyed, order, axis=1)

    # Repeat the last valid value across the invalid tail
    last_valid = srt[rows, np.maximum(n_valid - 1, 0)]
    pad = np.arange(diag.shape[1])[None, :] >= n_valid[:, None]
    srt = np.where(pad, last_valid[:, None], srt)

    # Max over valid entries (first sorted entry)
    mx_diag = srt[:, 0]
    dead = (n_valid == 0) | ~(mx_diag > 0)

    # Normalize by max and convert to dB
    ratio = np.maximum(srt, EPS) / np.maximum(mx_diag, EPS)[:, None]
    diag_db = 10.0 * np.log10(np.maximum(ratio, EPS))
    diag_db = np.maximum(diag_db, DB_FLOOR)
    diag_db = np.where(dead[:, None], 0.0, diag_db)
    diag_db_keep = diag_db[:, :N_DIAG]

    # --- Emphasis vector: [max EV, max pulse power] ---
    # max EV is simply the first eigenvalue (already in ev_keep[:, 0])
    max_ev = ev_keep[:, 0]

    # max pulse power is the maximum valid diagonal entry (already computed as mx_diag)
    max_pulse_power = np.where(dead, 0.0, mx_diag)

    # Normalize emphasis values for scale-free comparison
    # Use batch-level normalization (per batch max)
    emphasis_raw = np.stack([max_ev, max_pulse_power], axis=1)
    emphasis_max = np.maximum(emphasis_raw.max(axis=0, keepdims=True), EPS)
    emphasis_norm = emphasis_raw / emphasis_max

    # --- Branch 1: concatenate EVs, slopes, diagonal, emphasis ---
    branch1 = np.concatenate([
        ev_db,           # 12 features
        ev_slopes,       # 11 features
        diag_db_keep,    # 12 features
        emphasis_norm    # 2 features
    ], axis=1).astype(np.float32)

    # --- Branch 2: global scalars ---
    # Condition number in dB (over the kept 12 EVs)
    cond_db = ev_db[:, 0] - np.maximum(ev_db[:, -1], DB_FLOOR)

    # Effective rank (over the kept 12 EVs)
    p = ev_keep / np.maximum(ev_keep.sum(axis=1, keepdims=True), EPS)
    p = np.maximum(p, EPS)
    eff_rank = np.exp(-np.sum(p * np.log(p), axis=1))

    # Diagonal median/max ratio
    masked = np.where(valid, diag, np.nan)
    with np.errstate(invalid='ignore'):
        vmax = np.nanmax(masked, axis=1)
        vmed = np.nanmedian(masked, axis=1)
    diag_ratio = np.where(n_valid >= 2, vmed / np.maximum(vmax, EPS), 1.0)

    global_ = np.stack([cond_db, eff_rank, diag_ratio], axis=-1).astype(np.float32)

    if single:
        return branch1[0], global_[0]
    return branch1, global_


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_rfi_data_branch_updates(data_dirs, max_samples=None, tag=''):
    """
    Load tile records and assemble the two-branch features.
    """
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    branch_parts = []
    global_parts = []
    label_parts, group_parts, source_parts = [], [], []
    n_classes = 0
    meta = {'dirs': list(data_dirs), 'files': [], 'channels': [],
            'pulse_window': None,
            'source_names': [os.path.basename(os.path.normpath(d)) for d in data_dirs]}

    print(f"\nLoading {tag or data_dirs} ...")

    for source_id, data_dir in enumerate(data_dirs):
        paths = _glob_tile_files(data_dir)
        if not paths:
            raise FileNotFoundError(f"No tile record files found in {data_dir}")

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
                        f"{path} has no 'diagonal'/'diag_valid_idx' dataset; this "
                        f"script needs them. Regenerate with the current generators."
                    )

                sel = slice(None)
                if max_samples is not None and n_rec > max_samples:
                    sel = slice(0, n_rec, int(np.ceil(n_rec / max_samples)))

                eigvals = f['eigenvalues'][sel]
                labels = f['labels'][sel].astype(np.int32)
                groups_raw = f['tile_pulse'][sel].astype(np.int64)
                diag = np.asarray(f['diagonal'][sel], dtype=np.float64)
                diag_valid = np.asarray(f['diag_valid_idx'][sel], dtype=bool)

                branch1, global_ = features_from_records(eigvals, diag, diag_valid)

                branch_parts.append(branch1)
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
        'branch1': np.concatenate(branch_parts, axis=0),
        'global': np.concatenate(global_parts, axis=0),
        'labels': labels,
        'groups': np.concatenate(group_parts, axis=0),
        'sources': np.concatenate(source_parts, axis=0),
        'n_classes': n_classes or int(labels.max()) + 1,
        'meta': meta,
    }


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def train_model(run_name, n_classes, train_inputs, y_train, val_inputs, y_val,
                epochs, batch_size, learning_rate, dropout_rate, weight_decay,
                models_root=MODELS_ROOT):
    """
    Build and train the model with the combined branch structure.
    """
    out_dir = os.path.join(models_root, run_name)
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'best_model.keras')

    print(f"\n{'='*60}")
    print(f"Run : {run_name}")
    print(f"  train={len(y_train)}  val={len(y_val)}")
    print(f"  branch1 input : ({N_BRANCH_FEATURES},)   global input: (3,)   classes: {n_classes}")
    print(f"    [12 EVs + 11 slopes + 12 diag + 2 emphasis = {N_BRANCH_FEATURES}]")
    print(f"    [cond_db, eff_rank, diag_median_max_ratio]")
    print(f"{'='*60}")

    model = build_model_branch_updates(
        n_branch_features=N_BRANCH_FEATURES,
        n_knee_classes=n_classes,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    model.summary(print_fn=lambda s: print('  ' + s))

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path, monitor='val_loss',
            save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10,
            min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        x=train_inputs, y=y_train,
        validation_data=(val_inputs, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=2,
    )

    save_training_curves_png(history, out_dir)
    return model, out_dir, history


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train the knee classifier with combined branch structure: '
                    '12 EVs + 11 EV slopes + 12 SCM diagonal + 2 emphasis features.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data-dir', type=str, nargs='+', default=BENCHMARK_DIRS,
                        help='Training directories. Defaults to the three the '
                             'urClean_mtnContam_amzContam benchmark used.')
    parser.add_argument('--run-name', type=str, default='model_branch_updates')
    parser.add_argument('--models-root', type=str, default=MODELS_ROOT)
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Cap tiles loaded per file (evenly strided), for quick runs.')
    parser.add_argument('--val-frac', type=float, default=VAL_FRAC)
    parser.add_argument('--split-buffer', type=int, default=SPLIT_BUFFER_DEFAULT,
                        help=f'CPI rows dropped each side of the train/val boundary '
                             f'(1 row = {M} pulses).')
    parser.add_argument('--split-mode', choices=['per-source', 'global'],
                        default='per-source')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--learning-rate', type=float, default=LR)
    parser.add_argument('--dropout-rate', type=float, default=0.6)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.models_root, exist_ok=True)
    run_name = args.run_name

    print(f"\n{'='*70}")
    print('RFI knee classifier - TRAINING with combined branch structure')
    print(f"{'='*70}")
    print(f"  train region(s)   : {', '.join(args.data_dir)}")
    print(f"  branch1 features  : {N_BRANCH_FEATURES} total")
    print(f"    - 12 EVs (max-normalized, dB)")
    print(f"    - 11 EV slopes")
    print(f"    - 12 SCM diagonal (max-normalized, dB)")
    print(f"    - 2 emphasis [max EV, max pulse power] (normalized)")
    print(f"  run name          : {run_name}")
    print(f"  epochs            : {args.epochs}")
    print(f"  batch size        : {args.batch_size}")
    print(f"  learning rate     : {args.learning_rate}")
    print(f"  dropout rate      : {args.dropout_rate}")
    print(f"  weight decay      : {args.weight_decay}")

    data = load_rfi_data_branch_updates(args.data_dir, args.max_samples,
                                        tag='TRAIN region')
    n_classes = data['n_classes']

    print(f"\nTraining region: {len(data['labels'])} tiles   classes: {n_classes}   "
          f"channels: {', '.join(data['meta']['channels'])}")

    idx_train, idx_val = split_train_val(
        data['groups'], args.val_frac, args.split_buffer, args.split_mode)

    class_distribution = report_class_by_source_split(
        data['labels'], data['sources'], data['meta']['source_names'],
        idx_train, idx_val, n_classes)

    train_inputs = [data['branch1'][idx_train], data['global'][idx_train]]
    val_inputs = [data['branch1'][idx_val], data['global'][idx_val]]

    model, out_dir, history = train_model(
        run_name, n_classes,
        train_inputs, data['labels'][idx_train],
        val_inputs, data['labels'][idx_val],
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, dropout_rate=args.dropout_rate,
        weight_decay=args.weight_decay, models_root=args.models_root,
    )

    summary = {
        'run': run_name,
        'variant': 'branch_updates',
        'feature_change': (
            'Two-branch architecture: Branch 1 combines 12 valid EVs + 11 EV slopes + '
            '12 valid SCM diagonal entries (all max-normalized, dB) + '
            '2 emphasis features [max EV, max pulse power] to reinforce '
            'the lossless transformation relationship. Branch 2 contains 3 global '
            'scalars (cond_db, eff_rank, diag_median_max_ratio).'
        ),
        'branch_structure': {
            'branch1': {
                'n_evs': N_KEEP,
                'n_ev_slopes': N_EV_SLOPES,
                'n_diag': N_DIAG,
                'n_emphasis': N_EMPHASIS,
                'total': N_BRANCH_FEATURES,
            },
            'branch2_global': 3,
        },
        'scope': 'per-CPI only',
        'inputs': {'branch1': [N_BRANCH_FEATURES], 'global': [3]},
        'n_train': int(len(idx_train)),
        'n_val': int(len(idx_val)),
        'n_classes': n_classes,
        'epochs_requested': args.epochs,
        'epochs_run': len(history.history.get('loss', [])),
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'dropout_rate': args.dropout_rate,
        'weight_decay': args.weight_decay,
        'class_distribution': class_distribution,
        'train_provenance': data['meta'],
    }

    summary_path = os.path.join(out_dir, 'training_summary.json')
    with open(summary_path, 'w') as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n{'='*60}")
    print('Training complete!')
    print(f"  Model saved to: {os.path.join(out_dir, 'best_model.keras')}")
    print(f"  Summary saved to: {summary_path}")
    print(f"\nNOTE: this model takes TWO inputs:")
    print(f"      - branch1: {N_BRANCH_FEATURES} features (EVs + slopes + diag + emphasis)")
    print(f"      - global: 3 scalars (cond_db, eff_rank, diag_median_max_ratio)")
    print(f"      Use score_scene_branch_updates.py to score real scenes.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
