"""
eval_knee.py  --  Stage 3: Model Evaluation on Real Data

Read eigenvalue features from cpi_blocks.h5 (produced by tile_cpi.py),
run the trained knee-index classifier over two targeted pulse regions
(RFI-contaminated and clean urban), and write a self-describing HDF5.

Label encoding (matches training convention):
    knee_index = 0       : no RFI detected (clean)
    knee_index = k (1..M): RFI occupies eigenvalue indices 0..k-1,
                           signal subspace starts at eigenvalue index k

For every tile the output records:
    knee_index          predicted class (0..M)
    n_rfi_eigenvalues   = knee_index  (number of RFI-dominated eigenvalues)
    ev_boundary         = knee_index - 1  (-1 if clean)
    global_pulse_start  first global pulse of this CPI tile
    global_range_start  first range sample of this CPI tile
    confidence          max softmax probability
    entropy             Shannon entropy of output distribution

Output HDF5 layout
------------------
Root attrs: source_file, M, BLOCK_WIDTH, pulse_offset, model_path,
            label_encoding (text), rfi/clean pulse bounds

/{region}/                          e.g. /rfi/  or  /clean/
    attrs: region_label, pulse_start, pulse_stop, ci_start, ci_stop,
           n_cpi_rows, n_range_cols, n_tiles, n_valid, n_rfi_detected

    /cpi_{ci}_{ri}/
        attrs: ci, ri,
               global_pulse_start,  global_pulse_end,
               global_range_start,  global_range_end
        datasets:
            knee_index          ()   int16
            n_rfi_eigenvalues   ()   int16   same as knee_index
            ev_boundary         ()   int16   knee_index-1, or -1 if clean
            confidence          ()   float32
            entropy             ()   float32
            valid               ()   bool
            eigen_input         (M,2) float32
            global_input        (6,)  float32

Usage
-----
    python eval_knee.py
    python eval_knee.py --h5    nisar_data/processed/cpi_blocks.h5
    python eval_knee.py --model models/multi_band/best_model.keras
    python eval_knee.py --out   nisar_data/processed/knee_predictions.h5
    python eval_knee.py --rfi-start 63000 --rfi-stop 87000
    python eval_knee.py --clean-start 106000 --clean-stop 122000
"""

import os
import argparse
import numpy as np
import h5py
import tensorflow as tf

DEFAULT_H5           = os.path.join('nisar_data', 'processed', 'cpi_blocks.h5')
DEFAULT_MODEL        = os.path.join('models', 'multi_band', 'best_model.keras')
DEFAULT_OUT          = os.path.join('nisar_data', 'processed', 'knee_predictions.h5')
DEFAULT_BATCH        = 512
DEFAULT_RFI_START    = 63000
DEFAULT_RFI_STOP     = 87000
DEFAULT_CLEAN_START  = 106000
DEFAULT_CLEAN_STOP   = 122000
DEFAULT_PULSE_OFFSET = 46528

LABEL_ENCODING = (
    'knee_index=0 : clean (no RFI); '
    'knee_index=k (1..M) : RFI in eigenvalue indices 0..k-1, '
    'signal subspace starts at eigenvalue index k. '
    'n_rfi_eigenvalues = knee_index. '
    'ev_boundary = knee_index - 1 (last RFI eigenvalue index), -1 if clean.'
)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def pulse_to_ci(global_pulse, pulse_offset, M):
    return (global_pulse - pulse_offset) // M


def load_meta(h5_path):
    with h5py.File(h5_path, 'r') as f:
        return {k: f.attrs[k] for k in f.attrs}


# ---------------------------------------------------------------------------
# LOAD FEATURES
# ---------------------------------------------------------------------------

def load_region_features(h5_path, ci_start, ci_stop, M, n_range_cols):
    """
    Load all tiles in CPI rows [ci_start, ci_stop) across all range columns.

    Returns:
        eigen_buf  : float32 (n_tiles, M, 2)
        global_buf : float32 (n_tiles, 6)
        valid_mask : bool    (n_tiles,)
        keys       : list[str]  group names in row-major order
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
    n_tiles = len(valid_mask)
    knee    = np.full(n_tiles, np.int16(-1),   dtype=np.int16)
    conf    = np.full(n_tiles, np.float32(0.), dtype=np.float32)
    ent     = np.full(n_tiles, np.float32(0.), dtype=np.float32)

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
# WRITE
# ---------------------------------------------------------------------------

def write_region(dst, src_path, region_name, keys, knee, conf, ent,
                 valid_mask, ci_start, ci_stop, pulse_offset, M,
                 BLOCK_WIDTH, n_range_cols):
    """
    Write a region group with fully annotated per-CPI subgroups.
    """
    n_tiles      = len(keys)
    n_valid      = int(valid_mask.sum())
    vk           = knee[valid_mask]
    n_rfi        = int(np.sum(vk > 0)) if len(vk) > 0 else 0
    g_pulse_start = pulse_offset + ci_start * M
    g_pulse_stop  = pulse_offset + ci_stop  * M

    blk = dst.create_group(region_name)
    blk.attrs['region_label']     = region_name
    blk.attrs['pulse_start']      = g_pulse_start
    blk.attrs['pulse_stop']       = g_pulse_stop
    blk.attrs['ci_start']         = ci_start
    blk.attrs['ci_stop']          = ci_stop
    blk.attrs['n_cpi_rows']       = ci_stop - ci_start
    blk.attrs['n_range_cols']     = n_range_cols
    blk.attrs['n_tiles']          = n_tiles
    blk.attrs['n_valid']          = n_valid
    blk.attrs['n_rfi_detected']   = n_rfi

    with h5py.File(src_path, 'r') as src:
        for idx, key in enumerate(keys):
            # Parse ci, ri from key name
            _, ci_str, ri_str = key.split('_')
            ci = int(ci_str)
            ri = int(ri_str)

            src_grp = src[key]
            dst_grp = blk.create_group(key)

            # --- Spatial coordinates as group attrs ---
            g_pulse = pulse_offset + ci * M
            g_range = ri * BLOCK_WIDTH
            dst_grp.attrs['ci']                 = ci
            dst_grp.attrs['ri']                 = ri
            dst_grp.attrs['global_pulse_start']  = g_pulse
            dst_grp.attrs['global_pulse_end']    = g_pulse + M
            dst_grp.attrs['global_range_start']  = g_range
            dst_grp.attrs['global_range_end']    = g_range + BLOCK_WIDTH

            # --- Copy features ---
            dst_grp.create_dataset('eigen_input',
                                   data=src_grp['eigen_input'][()])
            dst_grp.create_dataset('global_input',
                                   data=src_grp['global_input'][()])
            dst_grp.create_dataset('valid',
                                   data=src_grp['valid'][()])

            # --- Predictions with derived interpretive fields ---
            k = int(knee[idx])
            dst_grp.create_dataset('knee_index',
                                   data=np.int16(k))
            dst_grp.create_dataset('n_rfi_eigenvalues',
                                   data=np.int16(max(k, 0)))
            dst_grp.create_dataset('ev_boundary',
                                   data=np.int16(k - 1 if k > 0 else -1))
            dst_grp.create_dataset('confidence',
                                   data=conf[idx])
            dst_grp.create_dataset('entropy',
                                   data=ent[idx])


# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------

def print_region_summary(region_name, knee, conf, ent, valid_mask,
                         ci_start, ci_stop, pulse_offset, M):
    n_tiles   = len(valid_mask)
    n_valid   = int(valid_mask.sum())
    g_start   = pulse_offset + ci_start * M
    g_stop    = pulse_offset + ci_stop  * M

    print(f'  Region         : {region_name}')
    print(f'  Global pulses  : {g_start} -- {g_stop}')
    print(f'  Total tiles    : {n_tiles}')
    print(f'  Valid          : {n_valid}  ({100*n_valid/n_tiles:.1f}%)')
    print(f'  Invalid (gap)  : {n_tiles-n_valid}  '
          f'({100*(n_tiles-n_valid)/n_tiles:.1f}%)')

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
    print(f'  Clean          : {n_valid-n_rfi}  '
          f'({100*(n_valid-n_rfi)/n_valid:.1f}%)')
    print(f'  Mean conf      : {vc.mean():.4f}  '
          f'low-conf (<0.5): {int(np.sum(vc<0.5))}  '
          f'({100*np.mean(vc<0.5):.1f}%)')
    print(f'  Mean entropy   : {ve.mean():.4f}  max={ve.max():.4f}')


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Stage 3: evaluate knee-index model on targeted pulse '
                    'regions from cpi_blocks.h5.'
    )
    parser.add_argument('--h5',           default=DEFAULT_H5)
    parser.add_argument('--model',        default=DEFAULT_MODEL)
    parser.add_argument('--out',          default=DEFAULT_OUT)
    parser.add_argument('--batch',        type=int, default=DEFAULT_BATCH)
    parser.add_argument('--rfi-start',    type=int, default=DEFAULT_RFI_START)
    parser.add_argument('--rfi-stop',     type=int, default=DEFAULT_RFI_STOP)
    parser.add_argument('--clean-start',  type=int, default=DEFAULT_CLEAN_START)
    parser.add_argument('--clean-stop',   type=int, default=DEFAULT_CLEAN_STOP)
    parser.add_argument('--pulse-offset', type=int, default=DEFAULT_PULSE_OFFSET)
    args = parser.parse_args()

    if not os.path.exists(args.h5):
        raise FileNotFoundError(f'cpi_blocks.h5 not found: {args.h5}')
    if not os.path.exists(args.model):
        raise FileNotFoundError(f'Model not found: {args.model}')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    meta         = load_meta(args.h5)
    M            = int(meta['M'])
    BLOCK_WIDTH  = int(meta['BLOCK_WIDTH'])
    n_cpi_rows   = int(meta['n_cpi_rows'])
    n_range_cols = int(meta['n_range_cols'])

    print(f'Features     : {args.h5}')
    print(f'Model        : {args.model}')
    print(f'Output       : {args.out}')
    print(f'M            : {M}  BLOCK_WIDTH : {BLOCK_WIDTH}')
    print(f'Grid         : {n_cpi_rows} x {n_range_cols}')
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
        # Root-level metadata
        for k, v in meta.items():
            dst.attrs[k] = v
        dst.attrs['model_path']        = args.model
        dst.attrs['pulse_offset']      = args.pulse_offset
        dst.attrs['label_encoding']    = LABEL_ENCODING
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

            eigen_buf, global_buf, valid_mask, keys = load_region_features(
                args.h5, ci_start, ci_stop, M, n_range_cols
            )
            print(f'  Tiles  : {len(keys)}  '
                  f'valid : {valid_mask.sum()}  '
                  f'invalid : {(~valid_mask).sum()}')

            knee, conf, ent = run_inference(
                model, eigen_buf, global_buf, valid_mask, args.batch
            )

            write_region(dst, args.h5, region_name, keys, knee, conf, ent,
                         valid_mask, ci_start, ci_stop, args.pulse_offset,
                         M, BLOCK_WIDTH, n_range_cols)

            print()
            print_region_summary(region_name, knee, conf, ent, valid_mask,
                                 ci_start, ci_stop, args.pulse_offset, M)
            print('=' * 60)

    print(f'\nWritten -> {args.out}')
    print()
    print('To read a tile:')
    print('  import h5py')
    print(f"  with h5py.File('{args.out}', 'r') as f:")
    print("      grp  = f['rfi/cpi_500_105']")
    print("      print(grp.attrs['global_pulse_start'])  # first global pulse")
    print("      print(grp.attrs['global_range_start'])  # first range sample")
    print("      print(grp['knee_index'][()])            # predicted class")
    print("      print(grp['ev_boundary'][()])           # last RFI EV index")
    print("      print(grp['n_rfi_eigenvalues'][()])     # count of RFI EVs")


if __name__ == '__main__':
    main()