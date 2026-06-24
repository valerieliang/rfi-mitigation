"""
infer_knee_map.py

Read targeted pulse ranges directly from a raw NISAR L0B HDF5 file,
tile into CPI blocks, run the trained knee-index model, and write
results per-CPI tile into an output HDF5 file.

HDF5 layout
-----------
/block_mid/
    cpi_0_0/
        knee_index   ()       int16
        confidence   ()       float32
        entropy      ()       float32
        eigen_input  (M, 2)   float32
        global_input (6,)     float32
    cpi_0_1/
        ...
    cpi_499_210/
        ...
/block_bottom/
    cpi_0_0/
        ...

Group name format: cpi_{ci}_{ri}
    ci : CPI row index    (0 .. n_cpi_rows-1)
    ri : range tile index (0 .. n_range_cols-1)

Attrs on each block group (e.g. /block_mid):
    source_file, source_path, pulse_start, pulse_end,
    M, BLOCK_WIDTH, n_cpi_rows, n_range_cols

Usage
-----
    python infer_knee_map.py
    python infer_knee_map.py --l0    nisar_data/raw/NISAR_...h5
    python infer_knee_map.py --model models/my_run/best_model.keras
    python infer_knee_map.py --out   nisar_data/processed/knee_maps.h5
"""

import os
import sys
import argparse
import numpy as np
import h5py
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import extract_features

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Raw L0B dataset path inside the NISAR HDF5
L0B_DATASET = '/science/LSAR/RRSD/swaths/frequencyA/txH/rxV/HV'

# CPI / tiling layout
M           = 16    # pulses per CPI
BLOCK_WIDTH = 250   # range samples per tile

# Targeted pulse ranges to process: (pulse_start, pulse_end, block_name)
# pulse_end - pulse_start must be divisible by M
TARGETS = [
    (50000,  58000,  'block_mid'),
    (170000, 178000, 'block_bottom'),
]

DEFAULT_L0    = os.path.join('nisar_data', 'raw',
                             'NISAR_L0_PR_RRSD_006_112_D_197S_'
                             '20251006T024004_20251006T024139_'
                             'P00410_F_J_001.h5')
DEFAULT_MODEL = os.path.join('models', 'multi_band', 'best_model.keras')
DEFAULT_OUT   = os.path.join('nisar_data', 'processed', 'knee_maps.h5')


# ---------------------------------------------------------------------------
# RAW BLOCK LOADING
# ---------------------------------------------------------------------------

def load_block(l0_path, dataset_path, pulse_start, pulse_end):
    """
    Read a pulse range from the raw L0B HDF5 and return a complex64 array.

    Args:
        l0_path      : Path to the raw NISAR L0B HDF5 file.
        dataset_path : Internal HDF5 path to the raw dataset.
        pulse_start  : First pulse index (inclusive).
        pulse_end    : Last pulse index (exclusive).

    Returns:
        block (np.ndarray): complex64, shape (pulse_end-pulse_start, range_count)
    """
    with h5py.File(l0_path, 'r') as f:
        raw   = f[dataset_path]
        chunk = raw[pulse_start:pulse_end, :]

    # Structured dtype with 'r' and 'i' fields -> complex
    block = chunk['r'].astype(np.float32) + 1j * chunk['i'].astype(np.float32)
    return block


# ---------------------------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------------------------

def extract_all_features(block, n_cpi_rows, n_range_cols):
    """
    Tile block into CPI tiles and extract features for every tile.

    Args:
        block       : complex64, shape (n_cpi_rows*M, >=n_range_cols*BLOCK_WIDTH)
        n_cpi_rows  : Number of CPI rows derived from block pulse count.
        n_range_cols: Number of range tiles derived from block range width.

    Returns:
        eigen_buf  : float32, shape (n_cpi_rows*n_range_cols, M, 2)
        global_buf : float32, shape (n_cpi_rows*n_range_cols, 6)
    """
    range_used = n_range_cols * BLOCK_WIDTH
    block      = block[:n_cpi_rows * M, :range_used]

    n_tiles    = n_cpi_rows * n_range_cols
    eigen_buf  = np.empty((n_tiles, M, 2), dtype=np.float32)
    global_buf = np.empty((n_tiles, 6),    dtype=np.float32)

    idx = 0
    for ci in range(n_cpi_rows):
        p0 = ci * M
        for ri in range(n_range_cols):
            r0  = ri * BLOCK_WIDTH
            cpi = block[p0:p0 + M, r0:r0 + BLOCK_WIDTH]
            eigen_buf[idx], global_buf[idx] = extract_features(cpi)
            idx += 1

    return eigen_buf, global_buf


# ---------------------------------------------------------------------------
# INFERENCE
# ---------------------------------------------------------------------------

def run_inference(model, eigen_buf, global_buf):
    """
    Run model over pre-extracted feature buffers.

    Args:
        model      : Loaded Keras model.
        eigen_buf  : float32, shape (n_tiles, M, 2)
        global_buf : float32, shape (n_tiles, 6)

    Returns:
        knee : int16,   shape (n_tiles,)
        conf : float32, shape (n_tiles,)
        ent  : float32, shape (n_tiles,)
    """
    probs = model.predict(
        [eigen_buf, global_buf],
        batch_size=512,
        verbose=0,
    )   # (n_tiles, M+1)

    knee = np.argmax(probs, axis=-1).astype(np.int16)
    conf = np.max(probs, axis=-1).astype(np.float32)
    eps  = 1e-12
    ent  = (-np.sum(probs * np.log(probs + eps), axis=-1)).astype(np.float32)

    return knee, conf, ent


# ---------------------------------------------------------------------------
# HDF5 WRITING
# ---------------------------------------------------------------------------

def write_cpi_groups(blk_grp, eigen_buf, global_buf, knee, conf, ent,
                     n_cpi_rows, n_range_cols):
    """
    Write one HDF5 group per CPI tile under blk_grp.

    Group naming: cpi_{ci}_{ri}

    Args:
        blk_grp      : h5py.Group for the current block (e.g. /block_mid)
        eigen_buf    : float32, shape (n_tiles, M, 2)
        global_buf   : float32, shape (n_tiles, 6)
        knee         : int16,   shape (n_tiles,)
        conf         : float32, shape (n_tiles,)
        ent          : float32, shape (n_tiles,)
        n_cpi_rows   : int
        n_range_cols : int
    """
    idx = 0
    for ci in range(n_cpi_rows):
        for ri in range(n_range_cols):
            cpi_grp = blk_grp.create_group(f'cpi_{ci}_{ri}')

            cpi_grp.create_dataset('knee_index',   data=knee[idx])
            cpi_grp.create_dataset('confidence',   data=conf[idx])
            cpi_grp.create_dataset('entropy',      data=ent[idx])
            cpi_grp.create_dataset('eigen_input',  data=eigen_buf[idx])
            cpi_grp.create_dataset('global_input', data=global_buf[idx])

            idx += 1


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Infer knee-index map directly from a raw NISAR L0B file, '
                    'storing results per CPI tile in HDF5.'
    )
    parser.add_argument('--l0',    default=DEFAULT_L0,
                        help='Path to raw NISAR L0B HDF5 file.')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--out',   default=DEFAULT_OUT)
    args = parser.parse_args()

    if not os.path.exists(args.l0):
        raise FileNotFoundError(f'L0B file not found: {args.l0}')
    if not os.path.exists(args.model):
        raise FileNotFoundError(f'Model not found: {args.model}')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f'L0B source : {args.l0}')
    print(f'Model      : {args.model}')
    print(f'Output     : {args.out}')
    print()

    print('Loading model ...')
    model = tf.keras.models.load_model(args.model)

    with h5py.File(args.out, 'w') as out_f:
        for pulse_start, pulse_end, block_name in TARGETS:
            n_pulses     = pulse_end - pulse_start
            assert n_pulses % M == 0, \
                f'{block_name}: pulse count {n_pulses} not divisible by M={M}'
            n_cpi_rows   = n_pulses // M

            print(f'--- {block_name}  pulses [{pulse_start}:{pulse_end}] ---')
            print(f'  Loading block from L0B ...')
            block        = load_block(args.l0, L0B_DATASET,
                                      pulse_start, pulse_end)
            n_range_cols = block.shape[1] // BLOCK_WIDTH
            print(f'  Block shape  : {block.shape}')
            print(f'  Tiling       : {n_cpi_rows} CPI rows '
                  f'x {n_range_cols} range cols '
                  f'= {n_cpi_rows * n_range_cols} tiles')

            print(f'  Extracting features ...')
            eigen_buf, global_buf = extract_all_features(
                block, n_cpi_rows, n_range_cols
            )

            print(f'  Running inference ...')
            knee, conf, ent = run_inference(model, eigen_buf, global_buf)

            print(f'  Writing per-CPI groups ...')
            blk_grp = out_f.create_group(block_name)
            blk_grp.attrs['source_file']  = args.l0
            blk_grp.attrs['source_path']  = L0B_DATASET
            blk_grp.attrs['pulse_start']  = pulse_start
            blk_grp.attrs['pulse_end']    = pulse_end
            blk_grp.attrs['M']            = M
            blk_grp.attrs['BLOCK_WIDTH']  = BLOCK_WIDTH
            blk_grp.attrs['n_cpi_rows']   = n_cpi_rows
            blk_grp.attrs['n_range_cols'] = n_range_cols

            write_cpi_groups(blk_grp, eigen_buf, global_buf, knee, conf, ent,
                             n_cpi_rows, n_range_cols)

            print(f'  knee_index : min={knee.min()}  max={knee.max()}  '
                  f'mean={knee.mean():.2f}')
            print(f'  mean conf  : {conf.mean():.4f}')
            print(f'  mean ent   : {ent.mean():.4f}')
            print()

    print(f'Done. Wrote {args.out}')
    print('\nTo read back a single tile:')
    print('  import h5py')
    print(f"  with h5py.File('{args.out}', 'r') as f:")
    print("      grp  = f['block_mid/cpi_12_5']")
    print("      knee = grp['knee_index'][()]")
    print("      eig  = grp['eigen_input'][:]   # (M, 2)")


if __name__ == '__main__':
    main()