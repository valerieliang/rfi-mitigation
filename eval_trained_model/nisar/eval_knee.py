"""
eval_knee.py  --  Stage 3: Model Evaluation on Real Data

Read eigenvalue features from cpi_blocks.h5 (produced by tile_cpi.py),
run the trained knee-index classifier, and write predictions back into a
new HDF5 file alongside the input features.

This script is the only one that imports TensorFlow.  It has no dependency
on tile_cpi.py or plot_eigenvalues.py at import time; it only needs the
output HDF5 from stage 1.

Output HDF5 layout
------------------
Mirrors cpi_blocks.h5 but adds three scalar datasets to each block group:
    knee_index   ()  int16    predicted knee class (-1 = invalid tile)
    confidence   ()  float32  max softmax probability
    entropy      ()  float32  Shannon entropy of the output distribution

Root attributes are copied from the input and extended with:
    model_path   str
    n_knee_classes int

Console output
--------------
Per-block summary: valid tile count, knee distribution, mean confidence,
fraction of low-confidence predictions (< 0.5).

Usage
-----
    python eval_knee.py
    python eval_knee.py --h5    nisar_data/processed/cpi_blocks.h5
    python eval_knee.py --model models/multi_band/best_model.keras
    python eval_knee.py --out   nisar_data/processed/knee_predictions.h5
    python eval_knee.py --batch 256
"""

import os
import argparse
import numpy as np
import h5py
import tensorflow as tf

DEFAULT_H5    = os.path.join('nisar_data', 'processed', 'cpi_blocks.h5')
DEFAULT_MODEL = os.path.join('models', 'multi_band', 'best_model.keras')
DEFAULT_OUT   = os.path.join('nisar_data', 'processed', 'knee_predictions.h5')
DEFAULT_BATCH = 512


# ---------------------------------------------------------------------------
# LOAD FEATURES
# ---------------------------------------------------------------------------

def load_all_features(h5_path):
    """
    Load every tile's features from cpi_blocks.h5 into flat arrays.

    Returns:
        eigen_buf  : float32 (n_tiles, M, 2)
        global_buf : float32 (n_tiles, 6)
        valid_mask : bool    (n_tiles,)
        keys       : list of str group names in row-major order
        meta       : dict of root attrs
    """
    with h5py.File(h5_path, 'r') as f:
        meta         = {k: f.attrs[k] for k in f.attrs}
        M            = int(meta['M'])
        n_cpi_rows   = int(meta['n_cpi_rows'])
        n_range_cols = int(meta['n_range_cols'])
        n_tiles      = n_cpi_rows * n_range_cols

        eigen_buf  = np.empty((n_tiles, M, 2), dtype=np.float32)
        global_buf = np.empty((n_tiles, 6),    dtype=np.float32)
        valid_mask = np.zeros(n_tiles, dtype=bool)
        keys       = []

        idx = 0
        for ci in range(n_cpi_rows):
            for ri in range(n_range_cols):
                key = f'cpi_{ci}_{ri}'
                keys.append(key)
                grp              = f[key]
                valid_mask[idx]  = bool(grp['valid'][()])
                eigen_buf[idx]   = grp['eigen_input'][:]
                global_buf[idx]  = grp['global_input'][:]
                idx += 1

    return eigen_buf, global_buf, valid_mask, keys, meta


# ---------------------------------------------------------------------------
# INFERENCE
# ---------------------------------------------------------------------------

def run_inference(model, eigen_buf, global_buf, valid_mask, batch_size):
    """
    Predict knee index for all valid tiles.

    Invalid tiles receive sentinel values:
        knee_index = -1, confidence = 0.0, entropy = 0.0

    Args:
        model      : loaded Keras model
        eigen_buf  : float32 (n_tiles, M, 2)
        global_buf : float32 (n_tiles, 6)
        valid_mask : bool    (n_tiles,)
        batch_size : int

    Returns:
        knee : int16   (n_tiles,)
        conf : float32 (n_tiles,)
        ent  : float32 (n_tiles,)
    """
    n_tiles = len(valid_mask)
    knee    = np.full(n_tiles, np.int16(-1),   dtype=np.int16)
    conf    = np.full(n_tiles, np.float32(0.), dtype=np.float32)
    ent     = np.full(n_tiles, np.float32(0.), dtype=np.float32)

    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        print('  Warning: no valid tiles found -- skipping inference.')
        return knee, conf, ent

    print(f'  Running inference on {len(valid_idx)} valid tiles '
          f'(batch={batch_size}) ...')
    probs = model.predict(
        [eigen_buf[valid_idx], global_buf[valid_idx]],
        batch_size=batch_size,
        verbose=1,
    )

    eps = 1e-12
    knee[valid_idx] = np.argmax(probs, axis=-1).astype(np.int16)
    conf[valid_idx] = np.max(probs, axis=-1).astype(np.float32)
    ent[valid_idx]  = (-np.sum(probs * np.log(probs + eps),
                               axis=-1)).astype(np.float32)
    return knee, conf, ent


# ---------------------------------------------------------------------------
# WRITE OUTPUT HDF5
# ---------------------------------------------------------------------------

def write_predictions(in_path, out_path, keys, knee, conf, ent,
                      valid_mask, meta, model_path):
    """
    Copy the input feature HDF5 and append prediction datasets to each group.

    Groups are written in the same key order as the input.  Root attributes
    are copied from meta and extended with model provenance fields.
    """
    n_classes = int(np.max(knee[valid_mask])) + 1 if valid_mask.any() else 0

    with h5py.File(in_path, 'r') as src, h5py.File(out_path, 'w') as dst:
        # Root attributes
        for k, v in meta.items():
            dst.attrs[k] = v
        dst.attrs['model_path']    = model_path
        dst.attrs['n_knee_classes'] = n_classes

        for idx, key in enumerate(keys):
            src_grp = src[key]
            dst_grp = dst.create_group(key)

            # Copy all attrs from source group
            for k, v in src_grp.attrs.items():
                dst_grp.attrs[k] = v

            # Copy input datasets
            for ds_name in ('eigen_input', 'global_input', 'valid'):
                dst_grp.create_dataset(ds_name,
                                       data=src_grp[ds_name][()])

            # Write predictions
            dst_grp.create_dataset('knee_index', data=knee[idx])
            dst_grp.create_dataset('confidence', data=conf[idx])
            dst_grp.create_dataset('entropy',    data=ent[idx])


# ---------------------------------------------------------------------------
# SUMMARY STATISTICS
# ---------------------------------------------------------------------------

def print_summary(knee, conf, ent, valid_mask, meta):
    """Print evaluation summary to stdout."""
    M            = int(meta['M'])
    n_cpi_rows   = int(meta['n_cpi_rows'])
    n_range_cols = int(meta['n_range_cols'])
    n_tiles      = len(valid_mask)
    n_valid      = int(valid_mask.sum())
    n_invalid    = n_tiles - n_valid

    print()
    print('=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f'  Tile grid      : {n_cpi_rows} CPI rows x {n_range_cols} range cols')
    print(f'  Total tiles    : {n_tiles}')
    print(f'  Valid tiles    : {n_valid}  ({100*n_valid/n_tiles:.1f}%)')
    print(f'  Invalid (gap)  : {n_invalid}  ({100*n_invalid/n_tiles:.1f}%)')
    print()

    if n_valid == 0:
        print('  No valid tiles -- nothing to report.')
        return

    vk = knee[valid_mask]
    vc = conf[valid_mask]
    ve = ent[valid_mask]

    print(f'  Knee index     : min={int(vk.min())}  max={int(vk.max())}  '
          f'mean={vk.mean():.2f}  median={np.median(vk):.1f}')

    # Top-5 most common knee values
    unique, counts = np.unique(vk, return_counts=True)
    top5 = sorted(zip(counts, unique), reverse=True)[:5]
    top5_str = '  '.join(f'knee={int(k)}:{int(c)}' for c, k in top5)
    print(f'  Top-5 knee     : {top5_str}')

    # Binary RFI detection (knee > 0 => RFI present)
    n_rfi    = int(np.sum(vk > 0))
    n_clean  = n_valid - n_rfi
    print(f'  RFI detected   : {n_rfi}  ({100*n_rfi/n_valid:.1f}%)')
    print(f'  Clean detected : {n_clean}  ({100*n_clean/n_valid:.1f}%)')

    print()
    print(f'  Mean confidence: {vc.mean():.4f}')
    print(f'  Min  confidence: {vc.min():.4f}')
    low_conf = int(np.sum(vc < 0.5))
    print(f'  Low-conf (<0.5): {low_conf}  ({100*low_conf/n_valid:.1f}%)')

    print()
    print(f'  Mean entropy   : {ve.mean():.4f}')
    print(f'  Max  entropy   : {ve.max():.4f}')
    print('=' * 60)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Stage 3: run the trained knee-index model on features '
                    'from cpi_blocks.h5 and write predictions.'
    )
    parser.add_argument('--h5',    default=DEFAULT_H5,
                        help='Path to cpi_blocks.h5 (output of tile_cpi.py).')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                        help='Path to the trained Keras model (.keras).')
    parser.add_argument('--out',   default=DEFAULT_OUT,
                        help='Output HDF5 path for predictions.')
    parser.add_argument('--batch', type=int, default=DEFAULT_BATCH,
                        help='Inference batch size (default 512).')
    args = parser.parse_args()

    if not os.path.exists(args.h5):
        raise FileNotFoundError(f'cpi_blocks.h5 not found: {args.h5}')
    if not os.path.exists(args.model):
        raise FileNotFoundError(f'Model not found: {args.model}')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print(f'Features    : {args.h5}')
    print(f'Model       : {args.model}')
    print(f'Output      : {args.out}')
    print()

    # --- Load features ---
    print('Loading features from cpi_blocks.h5 ...')
    eigen_buf, global_buf, valid_mask, keys, meta = load_all_features(args.h5)
    M = int(meta['M'])
    print(f'  Loaded {len(keys)} tiles  (M={M})')
    print(f'  Valid : {valid_mask.sum()}  Invalid : {(~valid_mask).sum()}')
    print()

    # --- Load model ---
    print('Loading model ...')
    model = tf.keras.models.load_model(args.model)
    model.summary(print_fn=lambda s: None)   # suppress verbose layer list

    # --- Inference ---
    knee, conf, ent = run_inference(
        model, eigen_buf, global_buf, valid_mask, args.batch
    )

    # --- Write output ---
    print()
    print('Writing predictions HDF5 ...')
    write_predictions(args.h5, args.out, keys, knee, conf, ent,
                      valid_mask, meta, args.model)
    print(f'  Written -> {args.out}')

    # --- Summary ---
    print_summary(knee, conf, ent, valid_mask, meta)

    print()
    print('To read a single tile prediction:')
    print('  import h5py')
    print(f"  with h5py.File('{args.out}', 'r') as f:")
    print("      grp  = f['cpi_12_5']")
    print("      knee = grp['knee_index'][()]   # int16")
    print("      conf = grp['confidence'][()]   # float32")
    print("      eig  = grp['eigen_input'][:]   # (M, 2)")


if __name__ == '__main__':
    main()
