"""
eval_knee.py  --  Stage 3: Model Evaluation on Real Data

Read eigenvalue features from cpi_blocks.h5 (produced by tile_cpi.py),
run the trained knee-index classifier over two targeted pulse regions
(RFI-contaminated and clean urban), and write predictions into a new HDF5.

Only CPI rows that fall within the declared pulse regions are evaluated.
Results are reported separately per region.

Output HDF5 layout
------------------
Mirrors the relevant subset of cpi_blocks.h5, adding to each group:
    knee_index   ()  int16    predicted knee class (-1 = invalid tile)
    confidence   ()  float32  max softmax probability
    entropy      ()  float32  Shannon entropy of the output distribution

Root attributes copied from input, extended with:
    model_path       str
    rfi_pulse_start  int
    rfi_pulse_stop   int
    clean_pulse_start int
    clean_pulse_stop  int

Usage
-----
    python eval_knee.py
    python eval_knee.py --h5    nisar_data/processed/cpi_blocks.h5
    python eval_knee.py --model models/multi_band/best_model.keras
    python eval_knee.py --out   nisar_data/processed/knee_predictions.h5
    python eval_knee.py --rfi-start 63000 --rfi-stop 87000
    python eval_knee.py --clean-start 106000 --clean-stop 122000
    python eval_knee.py --batch 512
"""

import os
import argparse
import numpy as np
import h5py
import tensorflow as tf

DEFAULT_H5          = os.path.join('nisar_data', 'processed', 'cpi_blocks.h5')
DEFAULT_MODEL       = os.path.join('models', 'multi_band', 'best_model.keras')
DEFAULT_OUT         = os.path.join('nisar_data', 'processed', 'knee_predictions.h5')
DEFAULT_BATCH       = 512
DEFAULT_RFI_START   = 63000
DEFAULT_RFI_STOP    = 87000
DEFAULT_CLEAN_START = 106000
DEFAULT_CLEAN_STOP  = 122000
DEFAULT_PULSE_OFFSET = 46528


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def pulse_to_ci(global_pulse, pulse_offset, M):
    return (global_pulse - pulse_offset) // M


def load_meta(h5_path):
    with h5py.File(h5_path, 'r') as f:
        return {k: f.attrs[k] for k in f.attrs}


# ---------------------------------------------------------------------------
# LOAD FEATURES FOR ONE REGION
# ---------------------------------------------------------------------------

def load_region_features(h5_path, ci_start, ci_stop, M, n_range_cols):
    """
    Load all tiles in CPI rows [ci_start, ci_stop) across all range columns.

    Returns:
        eigen_buf  : float32 (n_tiles, M, 2)
        global_buf : float32 (n_tiles, 6)
        valid_mask : bool    (n_tiles,)
        keys       : list of str  group names, row-major order
    """
    n_tiles    = (ci_stop - ci_start) * n_range_cols
    eigen_buf  = np.empty((n_tiles, M, 2), dtype=np.float32)
    global_buf = np.empty((n_tiles, 6),    dtype=np.float32)
    valid_mask = np.zeros(n_tiles,          dtype=bool)
    keys       = []

    with h5py.File(h5_path, 'r') as f:
        idx = 0
        for ci in range(ci_start, ci_stop):
            for ri in range(n_range_cols):
                key  = f'cpi_{ci}_{ri}'
                keys.append(key)
                grp  = f[key]
                valid_mask[idx] = bool(grp['valid'][()])
                eigen_buf[idx]  = grp['eigen_input'][:]
                global_buf[idx] = grp['global_input'][:]
                idx += 1

    return eigen_buf, global_buf, valid_mask, keys


# ---------------------------------------------------------------------------
# INFERENCE
# ---------------------------------------------------------------------------

def run_inference(model, eigen_buf, global_buf, valid_mask, batch_size):
    """
    Run model on all tiles; sentinels for invalid tiles.

    Returns:
        knee : int16   (n_tiles,)   -1 = invalid
        conf : float32 (n_tiles,)
        ent  : float32 (n_tiles,)
    """
    n_tiles = len(valid_mask)
    knee    = np.full(n_tiles, np.int16(-1),    dtype=np.int16)
    conf    = np.full(n_tiles, np.float32(0.),  dtype=np.float32)
    ent     = np.full(n_tiles, np.float32(0.),  dtype=np.float32)

    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        print('  Warning: no valid tiles -- skipping inference.')
        return knee, conf, ent

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

def write_region(dst, src_path, region_name, keys, knee, conf, ent,
                 valid_mask, ci_start, ci_stop, pulse_offset, M):
    """
    Write one HDF5 group per region containing all evaluated CPI tile groups.

    Layout:
        /{region_name}/
            cpi_{ci}_{ri}/
                eigen_input, global_input, valid  (copied from source)
                knee_index, confidence, entropy   (predictions)
    """
    blk = dst.create_group(region_name)
    blk.attrs['ci_start']     = ci_start
    blk.attrs['ci_stop']      = ci_stop
    blk.attrs['pulse_start']  = pulse_offset + ci_start * M
    blk.attrs['pulse_stop']   = pulse_offset + ci_stop  * M

    with h5py.File(src_path, 'r') as src:
        for idx, key in enumerate(keys):
            src_grp = src[key]
            dst_grp = blk.create_group(key)

            for k, v in src_grp.attrs.items():
                dst_grp.attrs[k] = v

            for ds_name in ('eigen_input', 'global_input', 'valid'):
                dst_grp.create_dataset(ds_name, data=src_grp[ds_name][()])

            dst_grp.create_dataset('knee_index', data=knee[idx])
            dst_grp.create_dataset('confidence', data=conf[idx])
            dst_grp.create_dataset('entropy',    data=ent[idx])


# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------

def print_region_summary(region_name, knee, conf, ent, valid_mask,
                         ci_start, ci_stop, pulse_offset, M):
    n_tiles  = len(valid_mask)
    n_valid  = int(valid_mask.sum())
    n_invalid = n_tiles - n_valid
    g_start  = pulse_offset + ci_start * M
    g_stop   = pulse_offset + ci_stop  * M

    print(f'  Region         : {region_name}')
    print(f'  Global pulses  : {g_start} -- {g_stop}')
    print(f'  Total tiles    : {n_tiles}')
    print(f'  Valid          : {n_valid}  ({100*n_valid/n_tiles:.1f}%)')
    print(f'  Invalid (gap)  : {n_invalid}  ({100*n_invalid/n_tiles:.1f}%)')

    if n_valid == 0:
        print('  No valid tiles.')
        return

    vk = knee[valid_mask]
    vc = conf[valid_mask]
    ve = ent[valid_mask]

    print(f'  Knee index     : min={int(vk.min())}  max={int(vk.max())}  '
          f'mean={vk.mean():.2f}  median={np.median(vk):.1f}')

    unique, counts = np.unique(vk, return_counts=True)
    top5 = sorted(zip(counts, unique), reverse=True)[:5]
    print(f'  Top-5 knee     : '
          + '  '.join(f'knee={int(k)}:{int(c)}' for c, k in top5))

    n_rfi = int(np.sum(vk > 0))
    print(f'  RFI detected   : {n_rfi}  ({100*n_rfi/n_valid:.1f}%)')
    print(f'  Clean detected : {n_valid - n_rfi}  '
          f'({100*(n_valid-n_rfi)/n_valid:.1f}%)')
    print(f'  Mean conf      : {vc.mean():.4f}  '
          f'low-conf (<0.5): {int(np.sum(vc<0.5))}  '
          f'({100*np.sum(vc<0.5)/n_valid:.1f}%)')
    print(f'  Mean entropy   : {ve.mean():.4f}  '
          f'max={ve.max():.4f}')


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Stage 3: evaluate knee-index model on RFI and clean '
                    'pulse regions from cpi_blocks.h5.'
    )
    parser.add_argument('--h5',           default=DEFAULT_H5)
    parser.add_argument('--model',        default=DEFAULT_MODEL)
    parser.add_argument('--out',          default=DEFAULT_OUT)
    parser.add_argument('--batch',        type=int, default=DEFAULT_BATCH)
    parser.add_argument('--rfi-start',    type=int, default=DEFAULT_RFI_START)
    parser.add_argument('--rfi-stop',     type=int, default=DEFAULT_RFI_STOP)
    parser.add_argument('--clean-start',  type=int, default=DEFAULT_CLEAN_START)
    parser.add_argument('--clean-stop',   type=int, default=DEFAULT_CLEAN_STOP)
    parser.add_argument('--pulse-offset', type=int, default=DEFAULT_PULSE_OFFSET,
                        help='Global pulse index of first pulse in file '
                             '(default 46528).')
    args = parser.parse_args()

    if not os.path.exists(args.h5):
        raise FileNotFoundError(f'cpi_blocks.h5 not found: {args.h5}')
    if not os.path.exists(args.model):
        raise FileNotFoundError(f'Model not found: {args.model}')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    meta         = load_meta(args.h5)
    M            = int(meta['M'])
    n_cpi_rows   = int(meta['n_cpi_rows'])
    n_range_cols = int(meta['n_range_cols'])

    print(f'Features     : {args.h5}')
    print(f'Model        : {args.model}')
    print(f'Output       : {args.out}')
    print(f'M            : {M}')
    print(f'Grid         : {n_cpi_rows} CPI rows x {n_range_cols} range cols')
    print()

    print('Loading model ...')
    model = tf.keras.models.load_model(args.model)
    model.summary(print_fn=lambda s: None)
    print()

    regions = [
        ('rfi',   args.rfi_start,   args.rfi_stop),
        ('clean', args.clean_start, args.clean_stop),
    ]

    with h5py.File(args.out, 'w') as dst:
        for k, v in meta.items():
            dst.attrs[k] = v
        dst.attrs['model_path']        = args.model
        dst.attrs['rfi_pulse_start']   = args.rfi_start
        dst.attrs['rfi_pulse_stop']    = args.rfi_stop
        dst.attrs['clean_pulse_start'] = args.clean_start
        dst.attrs['clean_pulse_stop']  = args.clean_stop

        print('=' * 60)
        for region_name, g_start, g_stop in regions:
            ci_start = max(0,          pulse_to_ci(g_start, args.pulse_offset, M))
            ci_stop  = min(n_cpi_rows, pulse_to_ci(g_stop,  args.pulse_offset, M))

            print(f'--- {region_name}  g=[{g_start}, {g_stop})  '
                  f'ci=[{ci_start}, {ci_stop}) ---')
            print(f'  Loading features ...')
            eigen_buf, global_buf, valid_mask, keys = load_region_features(
                args.h5, ci_start, ci_stop, M, n_range_cols
            )
            print(f'  Tiles: {len(keys)}  '
                  f'valid: {valid_mask.sum()}  '
                  f'invalid: {(~valid_mask).sum()}')

            print(f'  Running inference ...')
            knee, conf, ent = run_inference(
                model, eigen_buf, global_buf, valid_mask, args.batch
            )

            print(f'  Writing to HDF5 ...')
            write_region(dst, args.h5, region_name, keys, knee, conf, ent,
                         valid_mask, ci_start, ci_stop, args.pulse_offset, M)

            print()
            print_region_summary(region_name, knee, conf, ent, valid_mask,
                                 ci_start, ci_stop, args.pulse_offset, M)
            print('=' * 60)

    print()
    print(f'Written -> {args.out}')


if __name__ == '__main__':
    main()