"""
infer_knee_map.py

Run the trained knee-index model over every processed NISAR block in
nisar_data/processed/ and save a 2-D knee-index map per block.

Block layout (per file):
    - Shape  : (8000 pulses, 52866 range samples) complex64
    - CPI    : M = 16 pulses
    - BLOCK_WIDTH : 250 range samples
    - CPI rows   : 8000 / 16  = 500
    - Range tiles : floor(52866 / 250) = 211  (columns 0:52750 used)
    - Output map : (500, 211)  int16  knee index per tile

Output HDF5  nisar_data/processed/knee_maps.h5
    /block_mid/knee_map        int16  (500, 211)
    /block_mid/confidence      float32 (500, 211)
    /block_mid/entropy         float32 (500, 211)
    /block_mid attrs: source_file, pulse_start, range_start, M, BLOCK_WIDTH,
                      n_cpi_rows, n_range_cols

    /block_bottom/ ...  (same structure)

Usage
-----
    python infer_knee_map.py
    python infer_knee_map.py --model  models/my_run/best_model.keras
    python infer_knee_map.py --blocks nisar_data/processed
    python infer_knee_map.py --out    nisar_data/processed/knee_maps.h5
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
# LAYOUT CONSTANTS
# ---------------------------------------------------------------------------

M           = 16     # pulses per CPI
BLOCK_WIDTH = 250    # range samples per tile
N_CPI_ROWS  = 500    # 8000 / 16
N_RANGE_COL = 211    # floor(52866 / 250)
RANGE_USED  = BLOCK_WIDTH * N_RANGE_COL   # 52750

DEFAULT_MODEL  = os.path.join('models', 'multi_band', 'best_model.keras')
DEFAULT_BLOCKS = os.path.join('nisar_data', 'processed')
DEFAULT_OUT    = os.path.join('nisar_data', 'processed', 'knee_maps.h5')


# ---------------------------------------------------------------------------
# INFERENCE
# ---------------------------------------------------------------------------

def infer_block(model, block):
    """
    Tile a single complex block into CPI tiles and run inference.

    Args:
        model : Loaded Keras model.
        block (np.ndarray): complex64, shape (8000, >=RANGE_USED)

    Returns:
        knee_map   (np.ndarray): int16,   shape (N_CPI_ROWS, N_RANGE_COL)
        conf_map   (np.ndarray): float32, shape (N_CPI_ROWS, N_RANGE_COL)
        entropy_map(np.ndarray): float32, shape (N_CPI_ROWS, N_RANGE_COL)
    """
    # Trim to exact tiling extent
    block = block[:N_CPI_ROWS * M, :RANGE_USED]   # (8000, 52750)

    n_tiles = N_CPI_ROWS * N_RANGE_COL
    eigen_buf  = np.empty((n_tiles, M, 2),  dtype=np.float32)
    global_buf = np.empty((n_tiles, 6),     dtype=np.float32)

    idx = 0
    for ci in range(N_CPI_ROWS):
        p0 = ci * M
        for ri in range(N_RANGE_COL):
            r0 = ri * BLOCK_WIDTH
            cpi = block[p0:p0 + M, r0:r0 + BLOCK_WIDTH]
            eigen_buf[idx], global_buf[idx] = extract_features(cpi)
            idx += 1

    probs = model.predict(
        [eigen_buf, global_buf],
        batch_size=512,
        verbose=0,
    )   # (n_tiles, M+1)

    preds = np.argmax(probs, axis=-1).astype(np.int16)
    conf  = np.max(probs, axis=-1).astype(np.float32)
    eps   = 1e-12
    ent   = (-np.sum(probs * np.log(probs + eps), axis=-1)).astype(np.float32)

    knee_map    = preds.reshape(N_CPI_ROWS, N_RANGE_COL)
    conf_map    = conf.reshape(N_CPI_ROWS,  N_RANGE_COL)
    entropy_map = ent.reshape(N_CPI_ROWS,   N_RANGE_COL)

    return knee_map, conf_map, entropy_map


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Infer knee-index map for every processed NISAR block.'
    )
    parser.add_argument('--model',  default=DEFAULT_MODEL)
    parser.add_argument('--blocks', default=DEFAULT_BLOCKS)
    parser.add_argument('--out',    default=DEFAULT_OUT)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    print(f"Loading model from {args.model} ...")
    model = tf.keras.models.load_model(args.model)

    h5_files = sorted(
        f for f in os.listdir(args.blocks)
        if f.endswith('.h5') and f != os.path.basename(args.out)
    )
    if not h5_files:
        raise FileNotFoundError(f"No block .h5 files in: {args.blocks}")

    print(f"Output -> {args.out}\n")

    with h5py.File(args.out, 'w') as out_f:
        for fname in h5_files:
            fpath = os.path.join(args.blocks, fname)
            group_name = os.path.splitext(fname)[0]   # e.g. 'block_mid'

            print(f"Processing {fname} -> /{group_name}/")

            with h5py.File(fpath, 'r') as src:
                ds          = src['data']
                block       = ds[:]
                pulse_start = int(ds.attrs['pulse_start'])
                range_start = int(ds.attrs['range_start'])
                source_file = str(ds.attrs['source_file'])

            print(f"  Block shape  : {block.shape}")
            print(f"  Tiling       : {N_CPI_ROWS} CPI rows x {N_RANGE_COL} range cols")
            print(f"  Total tiles  : {N_CPI_ROWS * N_RANGE_COL}")

            knee_map, conf_map, entropy_map = infer_block(model, block)

            grp = out_f.create_group(group_name)

            ds_knee = grp.create_dataset('knee_map',    data=knee_map,    compression='gzip')
            ds_conf = grp.create_dataset('confidence',  data=conf_map,    compression='gzip')
            ds_ent  = grp.create_dataset('entropy',     data=entropy_map, compression='gzip')

            for ds_out in (ds_knee, ds_conf, ds_ent):
                ds_out.attrs['source_file']  = source_file
                ds_out.attrs['source_block'] = fname
                ds_out.attrs['pulse_start']  = pulse_start
                ds_out.attrs['range_start']  = range_start
                ds_out.attrs['M']            = M
                ds_out.attrs['BLOCK_WIDTH']  = BLOCK_WIDTH
                ds_out.attrs['n_cpi_rows']   = N_CPI_ROWS
                ds_out.attrs['n_range_cols'] = N_RANGE_COL

            print(f"  knee_map     : {knee_map.shape}  "
                  f"min={knee_map.min()}  max={knee_map.max()}  "
                  f"mean={knee_map.mean():.2f}")
            print(f"  mean conf    : {conf_map.mean():.4f}")
            print(f"  mean entropy : {entropy_map.mean():.4f}")
            print()

    print(f"Done. Wrote {args.out}")
    print("\nTo read back:")
    print("  import h5py, numpy as np")
    print(f"  with h5py.File('{args.out}', 'r') as f:")
    print("      knee = f['block_mid/knee_map'][:]   # (500, 211) int16")
    print("      conf = f['block_mid/confidence'][:] # (500, 211) float32")


if __name__ == '__main__':
    main()