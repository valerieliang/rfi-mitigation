"""
train_diag_profile.py

Same job as train_only.py, one feature change: the scalar
`diag_median_max_ratio` global feature is replaced by the ENTIRE SORTED SCM
DIAGONAL PROFILE, fed through its own conv branch (see model_diag.py).

Why
---
generate_amazon_data.py guarantees `label == number of distinct contaminated
pulse rows == number of RFI eigenvalues` (bands are placed in distinct rows
without replacement). The SCM diagonal is per-pulse-row power and keeps all 16
entries with a validity mask, so the label is literally the count of elevated
entries in that vector. train_only.py compresses it to one scalar.

Both scalars that could be built from it are structurally incapable of counting:

  median/max  the median sits in the clean group for any label <= 8 (all of
              them, since max_bands=6), so this reports the STRONGEST band's
              JSR. It saturates at label=1 -- a presence detector, not a counter.
  min/median  both min and median sit in the clean group for label <= 8, so this
              measures clean-row speckle spread and nothing about RFI at all.

The sorted profile has a step at index k and the network can count it, the same
way it already finds the eigenvalue knee.

Why it should help at LOW JSR specifically
------------------------------------------
diag[i] is a plain mean of |x|^2 over the tile's K range samples, so its relative
error is ~1/sqrt(K), independent of matrix dimension. A 16x16 SCM estimated from
K samples spreads its noise EIGENvalues by ~sqrt(16/K) even on pure noise
(Marchenko-Pastur). A weak RFI eigenvalue drowns in that spread while its
diagonal row is still cleanly measurable.

CAVEAT: one row per band is a property of the INJECTION MODEL, not of physics.
Real RFI spanning many pulses would elevate many rows (or all of them, vanishing
under median normalization). Validate on real scenes before trusting a gain here.

Comparability with the urClean benchmark
----------------------------------------
The eigenvalue features come from train_only.features_from_eigenvalues -- the
same function the benchmark used, imported not copied -- and the eigenvalue
branch topology, split logic, callbacks and hyperparameter defaults are
unchanged. The default --data-dir is the exact three-directory set recorded in
models/urClean_mtnContam_amzContam/training_summary.json, so the only moving part
is the diagonal feature.

Usage:
    python train_diag_profile.py --run-name urClean_diagprofile --epochs 50

Outputs:
    models/<run>/best_model.keras
    models/<run>/training_curves.png
    models/<run>/training_summary.json
"""

import os
import json
import argparse
import warnings

import numpy as np
import h5py
import tensorflow as tf

from model_diag import build_model_diag
from diag_features import diag_profile_features
from train_only import (
    MODELS_ROOT, M, N_KEEP, N_GLOBAL, EPOCHS, BATCH_SIZE, LR, VAL_FRAC,
    SPLIT_BUFFER_DEFAULT, EPS, DB_FLOOR, GROUP_OFFSET,
    _glob_tile_files,
    features_from_eigenvalues,
    split_train_val,
    report_class_by_source_split,
    save_training_curves_png,
)

# The three directories the urClean_mtnContam_amzContam benchmark was trained on,
# per its training_summary.json. Default here so the comparison is like-for-like.
BENCHMARK_DIRS = ['data/czech_contam', 'data/amazon_contam', 'data/berlin_clean']


# ---------------------------------------------------------------------------
# FEATURE ASSEMBLY
#
# diag_profile_features lives in diag_features.py so test_only.py can import the
# exact same implementation without pulling in TensorFlow.
# ---------------------------------------------------------------------------

def features_from_records(eigvals_linear, diag_lin, diag_valid_idx):
    """
    Build the three input tensors.

    The eigen tensor and the first two global features come straight from
    train_only.features_from_eigenvalues, so they are identical to the benchmark
    model's. Only its third global column (diag_median_max_ratio) is dropped --
    no information is lost, since max/median is just profile[:, 0] of the new
    diagonal tensor -- and replaced by valid_frac.
    """
    eigen, global_old = features_from_eigenvalues(
        eigvals_linear, diag_lin, diag_valid_idx)
    profile, valid_frac = diag_profile_features(diag_lin, diag_valid_idx)

    eigen = np.atleast_3d(eigen) if eigen.ndim == 2 else eigen
    global_old = np.atleast_2d(global_old)
    global_ = np.stack(
        [global_old[:, 0], global_old[:, 1], np.atleast_1d(valid_frac)], axis=-1
    ).astype(np.float32)

    return eigen, profile, global_


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_rfi_data_dir_diag(data_dirs, max_samples=None, tag=''):
    """
    Same contract as train_only.load_rfi_data_dir, plus a 'diag' tensor.

    A separate loader is needed because the base one discards the raw diagonal
    after folding it into one scalar; train_only.py is left untouched so the
    benchmark stays reproducible. File globbing and the split/report helpers are
    imported rather than duplicated.
    """
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    eigen_parts, diag_parts, global_parts = [], [], []
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

                eigen, profile, global_ = features_from_records(
                    eigvals, diag, diag_valid)

                eigen_parts.append(eigen)
                diag_parts.append(profile)
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
        'diag': np.concatenate(diag_parts, axis=0),
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
    Build and train the three-branch model. Callbacks and their patiences match
    train_only.train_model exactly so the comparison isolates the feature change.
    """
    out_dir = os.path.join(models_root, run_name)
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'best_model.keras')

    print(f"\n{'='*60}")
    print(f"Run : {run_name}")
    print(f"  train={len(y_train)}  val={len(y_val)}")
    print(f"  eigen input : ({N_KEEP}, 2)   diag input : ({M}, 2)   "
          f"global: ({N_GLOBAL},)   classes: {n_classes}")
    print(f"{'='*60}")

    model = build_model_diag(
        cpi_size=N_KEEP,
        diag_size=M,
        n_global_features=N_GLOBAL,
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
        description='Train the knee classifier with the full sorted SCM diagonal '
                    'profile in place of the median/max scalar.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data-dir', type=str, nargs='+', default=BENCHMARK_DIRS,
                        help='Training directories. Defaults to the three the '
                             'urClean_mtnContam_amzContam benchmark used, for a '
                             'like-for-like comparison.')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Model output folder. Defaults to "<first dir>_diagprofile".')
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
    run_name = args.run_name or (
        os.path.basename(os.path.normpath(args.data_dir[0])) + '_diagprofile')

    print(f"\n{'='*70}")
    print('RFI knee classifier - TRAINING with SORTED DIAGONAL PROFILE')
    print(f"{'='*70}")
    print(f"  train region(s)   : {', '.join(args.data_dir)}")
    print(f"  eigen features    : top {N_KEEP} of {M} eigenvalues, "
          f"lambda_max-normalized then dB  [same as benchmark]")
    print(f"  diag features     : all {M} SCM diagonal entries, valid-only, sorted "
          f"descending, dB rel. median  [NEW]")
    print(f"  global features   : [cond_db, eff_rank, valid_frac]   "
          f"(diag_median_max_ratio removed -- now profile[0])")
    print(f"  run name          : {run_name}")
    print(f"  epochs            : {args.epochs}")
    print(f"  batch size        : {args.batch_size}")
    print(f"  learning rate     : {args.learning_rate}")
    print(f"  dropout rate      : {args.dropout_rate}")
    print(f"  weight decay      : {args.weight_decay}")

    data = load_rfi_data_dir_diag(args.data_dir, args.max_samples, tag='TRAIN region')
    n_classes = data['n_classes']

    print(f"\nTraining region: {len(data['labels'])} tiles   classes: {n_classes}   "
          f"channels: {', '.join(data['meta']['channels'])}")

    idx_train, idx_val = split_train_val(
        data['groups'], args.val_frac, args.split_buffer, args.split_mode)

    class_distribution = report_class_by_source_split(
        data['labels'], data['sources'], data['meta']['source_names'],
        idx_train, idx_val, n_classes)

    # Input order must match build_model_diag: [eigen, diag, global].
    train_inputs = [data['eigen'][idx_train], data['diag'][idx_train],
                    data['global'][idx_train]]
    val_inputs = [data['eigen'][idx_val], data['diag'][idx_val],
                  data['global'][idx_val]]

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
        'variant': 'sorted_diag_profile',
        'feature_change': ('replaced the scalar diag_median_max_ratio with the '
                           f'full sorted valid SCM diagonal profile ({M}, 2) on '
                           'its own conv branch; global is now '
                           '[cond_db, eff_rank, valid_frac]'),
        'inputs': {'eigen': [N_KEEP, 2], 'diag': [M, 2], 'global': [N_GLOBAL]},
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
    print(f"\nNOTE: this model takes THREE inputs. test_only.py builds only")
    print(f"      [eigen, global] and will fail to run it -- a matching test")
    print(f"      script is needed for the benchmark comparison.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
