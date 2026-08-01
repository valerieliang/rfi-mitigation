"""
train_new_globals.py

Trains model_new_globals: the urClean benchmark's eigenvalue branch, with the
global vector widened from 3 scalars to 5.

    [cond_db, eff_rank, diag_median_max_ratio, schur_horn_gap, participation_ratio]
                                               ^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^
                                               new                             new

The first three are byte-for-byte train_only.features_from_eigenvalues output --
that function is imported, not copied -- and the eigenvalue branch, split logic,
callbacks and hyperparameter defaults are unchanged. The only moving part is the
pair of added scalars.

Why this and not train_diag_profile.py
--------------------------------------
train_diag_profile.py puts the whole sorted diagonal on its own conv branch. In
the training set `label == number of elevated diagonal rows` exactly, because
generate_amazon_data.py drops each band in its own pulse row without
replacement -- so that branch has a near-noiseless shortcut to the label, and
roughly 155k parameters with which to fit it. The shortcut is an artifact of the
injection model: measured, the effective number of occupied rows at one RFI
eigenvalue is about 1.8 on synthetic tiles and about 9.9 on real scenes.

This script keeps two scale-free summaries of the diagonal and drops the profile.
That is a MITIGATION, not a fix -- the scalars carry the same domain shift, they
just come with 128 parameters instead of 155k, no normalization choice to get
wrong, and no padded invalid tail to manufacture structure. Removing the shortcut
itself needs a change to inject_rfi_bands.

See diag_features.new_global_features and model_new_globals for the full
reasoning and the measured numbers.

Usage:
    python train_new_globals.py --run-name model_new_globals --epochs 100 \
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

from model_new_globals import build_model_new_globals
from diag_features import N_GLOBAL_NEW, GAP_K, GAP_SCALE, new_global_features
from train_only import (
    MODELS_ROOT, M, N_KEEP, EPOCHS, BATCH_SIZE, LR, VAL_FRAC,
    SPLIT_BUFFER_DEFAULT, GROUP_OFFSET,
    _glob_tile_files,
    features_from_eigenvalues,
    split_train_val,
    report_class_by_source_split,
    save_training_curves_png,
)

# The three directories the urClean_mtnContam_amzContam benchmark trained on,
# per its training_summary.json. Default here so the comparison is like-for-like.
BENCHMARK_DIRS = ['data/czech_contam', 'data/amazon_contam', 'data/berlin_clean']

GLOBAL_NAMES = ['cond_db', 'eff_rank', 'diag_median_max_ratio',
                f'schur_horn_gap@{GAP_K}', 'participation_ratio']


# ---------------------------------------------------------------------------
# FEATURE ASSEMBLY
# ---------------------------------------------------------------------------

def features_from_records(eigvals_linear, diag_lin, diag_valid_idx):
    """
    Build the two input tensors.

    The eigen tensor and the first three globals come straight from
    train_only.features_from_eigenvalues; new_global_features appends the two
    scalar diagonal statistics. Both live in diag_features.py so test and scene
    scoring can import the exact same implementation without pulling in
    TensorFlow.
    """
    eigen, global_old = features_from_eigenvalues(
        eigvals_linear, diag_lin, diag_valid_idx)

    eigen = np.atleast_3d(eigen) if eigen.ndim == 2 else eigen
    global_ = new_global_features(
        np.atleast_2d(global_old), eigvals_linear, diag_lin, diag_valid_idx)

    return eigen, np.atleast_2d(global_).astype(np.float32)


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_rfi_data_new_globals(data_dirs, max_samples=None, tag=''):
    """
    Same contract as train_only.load_rfi_data_dir, but the global tensor is the
    5-vector.

    A separate loader is needed because the base one folds the diagonal into one
    scalar and discards the rest, and the Schur-Horn gap needs the raw diagonal
    AND the raw eigenvalues together. train_only.py is left untouched so the
    benchmark stays reproducible; globbing and the split/report helpers are
    imported rather than duplicated.
    """
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    eigen_parts, global_parts = [], []
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

                eigen, global_ = features_from_records(eigvals, diag, diag_valid)

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


def report_global_ranges(global_):
    """
    Print p1 / p50 / p99 of each global feature.

    The global branch is a plain Dense with no input normalization and no fitted
    scaler, so a feature that lands orders of magnitude below its neighbours is
    effectively ignored at initialization. GAP_SCALE exists to stop that
    happening to the Schur-Horn gap; this makes it visible rather than assumed.
    """
    print('\nGlobal feature ranges (p1 / p50 / p99):')
    for i, name in enumerate(GLOBAL_NAMES):
        col = global_[:, i]
        print(f'  {name:24s} {np.percentile(col, 1):10.3f} '
              f'{np.percentile(col, 50):10.3f} {np.percentile(col, 99):10.3f}')


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def train_model(run_name, n_classes, train_inputs, y_train, val_inputs, y_val,
                epochs, batch_size, learning_rate, dropout_rate, weight_decay,
                models_root=MODELS_ROOT):
    """
    Build and train. Callbacks and their patiences match train_only.train_model
    exactly so the comparison isolates the feature change.
    """
    out_dir = os.path.join(models_root, run_name)
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'best_model.keras')

    print(f"\n{'='*60}")
    print(f"Run : {run_name}")
    print(f"  train={len(y_train)}  val={len(y_val)}")
    print(f"  eigen input : ({N_KEEP}, 2)   global: ({N_GLOBAL_NEW},)   "
          f"classes: {n_classes}")
    print(f"{'='*60}")

    model = build_model_new_globals(
        cpi_size=N_KEEP,
        n_global_features=N_GLOBAL_NEW,
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
        description='Train the knee classifier with the Schur-Horn gap and the '
                    'diagonal participation ratio added to the global vector.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data-dir', type=str, nargs='+', default=BENCHMARK_DIRS,
                        help='Training directories. Defaults to the three the '
                             'urClean_mtnContam_amzContam benchmark used.')
    parser.add_argument('--run-name', type=str, default='model_new_globals')
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
    print('RFI knee classifier - TRAINING with the 5-scalar global vector')
    print(f"{'='*70}")
    print(f"  train region(s)   : {', '.join(args.data_dir)}")
    print(f"  eigen features    : top {N_KEEP} of {M} eigenvalues, "
          f"lambda_max-normalized then dB  [same as benchmark]")
    print(f"  global features   : {GLOBAL_NAMES}")
    print(f"                      first 3 same as benchmark; last 2 NEW, both "
          f"scale-free, computed over all {M} entries")
    print(f"  gap prefix        : k={GAP_K}, reported x{GAP_SCALE:g} (percent of trace)")
    print(f"  diag conv branch  : NONE  [dropped -- see model_new_globals docstring]")
    print(f"  scope             : strictly per-CPI, no cross-tile features")
    print(f"  run name          : {run_name}")
    print(f"  epochs            : {args.epochs}")
    print(f"  batch size        : {args.batch_size}")
    print(f"  learning rate     : {args.learning_rate}")
    print(f"  dropout rate      : {args.dropout_rate}")
    print(f"  weight decay      : {args.weight_decay}")

    data = load_rfi_data_new_globals(args.data_dir, args.max_samples,
                                     tag='TRAIN region')
    n_classes = data['n_classes']

    print(f"\nTraining region: {len(data['labels'])} tiles   classes: {n_classes}   "
          f"channels: {', '.join(data['meta']['channels'])}")

    report_global_ranges(data['global'])

    idx_train, idx_val = split_train_val(
        data['groups'], args.val_frac, args.split_buffer, args.split_mode)

    class_distribution = report_class_by_source_split(
        data['labels'], data['sources'], data['meta']['source_names'],
        idx_train, idx_val, n_classes)

    # Input order must match build_model: [eigen, global].
    train_inputs = [data['eigen'][idx_train], data['global'][idx_train]]
    val_inputs = [data['eigen'][idx_val], data['global'][idx_val]]

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
        'variant': 'new_globals',
        'feature_change': (
            'added schur_horn_gap@%d (percent of trace, computed over all %d '
            'eigenvalues and diagonal entries) and the diagonal participation '
            'ratio (sum(d)^2 / sum(d^2)) to the benchmark global vector; NO '
            'diagonal conv branch' % (GAP_K, M)),
        'global_features': GLOBAL_NAMES,
        'gap_k': GAP_K,
        'gap_scale': GAP_SCALE,
        'scope': 'per-CPI only',
        'inputs': {'eigen': [N_KEEP, 2], 'global': [N_GLOBAL_NEW]},
        'n_train': int(len(idx_train)),
        'n_val': int(len(idx_val)),
        'n_classes': n_classes,
        'n_keep': N_KEEP,
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
    print(f"\nNOTE: this model takes TWO inputs like the benchmark, but the")
    print(f"      global vector is {N_GLOBAL_NEW} wide, not 3. test_only.py and")
    print(f"      score_scene.py build a 3-vector and will fail to run it --")
    print(f"      they need diag_features.new_global_features applied first.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
