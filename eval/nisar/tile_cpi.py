"""
tile_cpi.py  --  Stage 1: Tile and Extract

Read the full HV channel from a raw NISAR L0B HDF5 file, decode via the
BFPQLUT lookup table, divide into non-overlapping CPI blocks of shape
(M, BLOCK_WIDTH), compute SCM eigenvalues and global features for every
block, and write a compact feature HDF5 that downstream scripts consume.

This script has no model dependency.  It produces cpi_blocks.h5 which is
the sole input to plot_eigenvalues.py and eval_knee.py.

HDF5 output layout
------------------
Attributes on root:
    source_file   str   path to the L0B file
    source_path   str   internal HDF5 dataset path used
    M             int   pulses per CPI block
    BLOCK_WIDTH   int   range samples per CPI block
    n_cpi_rows    int   total CPI rows across the full image
    n_range_cols  int   total range tiles per row
    n_pulses      int   total pulses read from the file
    n_range       int   total range samples read from the file

One group per CPI block:
    /cpi_{ci}_{ri}/
        eigen_input  (M, 2)  float32   [ev_db, slope_db]
        global_input (6,)    float32   [F_factor, sigma_min, sigma_max,
                                        mu_min, trace_db, cond_number]
        valid        ()      bool      False = satellite gap / fill region

Block index convention:
    ci : CPI row index   -- azimuth (0 = first M pulses of the image)
    ri : range tile index -- range  (0 = first BLOCK_WIDTH samples)

Usage
-----
    python tile_cpi.py
    python tile_cpi.py --l0   /path/to/NISAR_L0B.h5
    python tile_cpi.py --out  nisar_data/processed/cpi_blocks.h5
    python tile_cpi.py --m    16 --bw 250
"""

import os
import argparse
import numpy as np
import h5py

# ---------------------------------------------------------------------------
# DEFAULTS
# ---------------------------------------------------------------------------

# HV channel path inside the NISAR L0B HDF5
L0B_DATASET_HV = '/science/LSAR/RRSD/swaths/frequencyA/txH/rxV/HV'

DEFAULT_L0  = os.path.join('nisar_data', 'raw',
                           'NISAR_L0_PR_RRSD_006_112_D_197S_'
                           '20251006T024004_20251006T024139_'
                           'P00410_F_J_001.h5')
DEFAULT_OUT = os.path.join('nisar_data', 'processed', 'cpi_blocks.h5')
DEFAULT_M   = 16
DEFAULT_BW  = 250


# ---------------------------------------------------------------------------
# DECODE
# ---------------------------------------------------------------------------

def decode_hv(l0_path, dataset_path):
    """
    Read and decode the full HV dataset from an L0B HDF5 file.

    Supports three storage formats in this priority order:
      1. BFPQLUT  -- uint16 compound {r, i} indexing a float32 lookup table
      2. complex32 -- float16 compound {r, i}
      3. complex64 -- native; no decoding needed

    Args:
        l0_path      : path to the L0B HDF5 file
        dataset_path : internal HDF5 path to the raw dataset

    Returns:
        data (np.ndarray): complex64, shape (n_pulses, n_range)
    """
    with h5py.File(l0_path, 'r', libver='latest', swmr=True) as f:
        dataset = f[dataset_path]
        group   = dataset.parent
        chunk   = dataset[:]               # read entire dataset

        # --- Format 1: BFPQLUT ---
        if 'BFPQLUT' in group:
            lut  = np.asarray(group['BFPQLUT'], dtype=np.float32)
            data = (lut[chunk['r']].astype(np.float32)
                    + 1j * lut[chunk['i']].astype(np.float32))
            return data.astype(np.complex64)

        # --- Format 2: float16 compound (complex32) ---
        try:
            dt = dataset.dtype
            is_c32 = (
                dt.names is not None
                and set(dt.names) == {'r', 'i'}
                and dt['r'].itemsize == 2
            )
        except TypeError:
            is_c32 = True

        if is_c32:
            data = (chunk['r'].astype(np.float32)
                    + 1j * chunk['i'].astype(np.float32))
            return data.astype(np.complex64)

        # --- Format 3: native complex ---
        if np.issubdtype(chunk.dtype, np.complexfloating):
            return chunk.astype(np.complex64)

        raise ValueError(
            f'Unsupported dtype at {dataset_path}: {dataset.dtype}'
        )


# ---------------------------------------------------------------------------
# VALIDITY CHECK
# ---------------------------------------------------------------------------

def _is_valid_tile(cpi):
    """
    Return False if the tile is a satellite gap or fill region.

    A tile is declared invalid when more than half its pulse rows have
    exactly zero total power (they are fill zeros injected at the sensor).
    """
    row_power = np.sum(np.abs(cpi) ** 2, axis=1)
    return int(np.sum(row_power == 0.0)) <= (cpi.shape[0] // 2)


# ---------------------------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------------------------

def extract_features(cpi):
    """
    Compute SCM eigenvalue features for a single CPI tile.

    Args:
        cpi (np.ndarray): complex64, shape (M, K)

    Returns:
        eigen_input  (np.ndarray): float32, shape (M, 2)
            col 0 -- SCM eigenvalues in dB, sorted descending
            col 1 -- first-difference slopes (padded at end to length M)
        global_input (np.ndarray): float32, shape (6,)
            [F_factor, sigma_min, sigma_max, mu_min, trace_db, cond_number]
    """
    eps = 1e-12
    M_  = cpi.shape[0]

    scm     = (cpi @ cpi.conj().T) / cpi.shape[1]
    eigvals = np.linalg.eigvalsh(scm).real[::-1]
    eigvals = np.maximum(eigvals, eps)

    ev_db  = 10.0 * np.log10(eigvals)
    slopes = np.diff(ev_db)
    slopes = np.append(slopes, slopes[-1])     # pad to length M

    eigen_input = np.stack([ev_db, slopes], axis=1).astype(np.float32)

    half      = max(M_ // 2, 1)
    diffs_top = np.diff(eigvals[:half])
    diffs_bot = np.diff(eigvals[half:])

    sigma_max   = float(np.std(diffs_top))  if len(diffs_top) > 0 else 0.0
    sigma_min   = float(np.std(diffs_bot))  if len(diffs_bot) > 0 else 0.0
    mu_min      = float(np.mean(diffs_bot)) if len(diffs_bot) > 0 else 0.0
    f_factor    = sigma_max / max(sigma_min, eps)
    trace_db    = 10.0 * np.log10(max(float(np.real(np.trace(scm))), eps))
    cond_number = float(eigvals[0] / max(eigvals[-1], eps))

    global_input = np.array(
        [f_factor, sigma_min, sigma_max, mu_min, trace_db, cond_number],
        dtype=np.float32,
    )
    return eigen_input, global_input


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Stage 1: tile raw NISAR L0B HV data into CPI blocks '
                    'and extract eigenvalue features.'
    )
    parser.add_argument('--l0',  default=DEFAULT_L0,
                        help='Path to raw NISAR L0B HDF5 file.')
    parser.add_argument('--out', default=DEFAULT_OUT,
                        help='Output HDF5 path for CPI feature blocks.')
    parser.add_argument('--m',   type=int, default=DEFAULT_M,
                        help='Pulses per CPI block (default 16).')
    parser.add_argument('--bw',  type=int, default=DEFAULT_BW,
                        help='Range samples per tile (default 250).')
    parser.add_argument('--dataset', default=L0B_DATASET_HV,
                        help='Internal HDF5 path to the raw dataset '
                             '(default: HV channel).')
    args = parser.parse_args()

    if not os.path.exists(args.l0):
        raise FileNotFoundError(f'L0B file not found: {args.l0}')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    M          = args.m
    BLOCK_WIDTH = args.bw

    print(f'L0B source  : {args.l0}')
    print(f'Dataset     : {args.dataset}')
    print(f'M           : {M}')
    print(f'BLOCK_WIDTH : {BLOCK_WIDTH}')
    print(f'Output      : {args.out}')
    print()

    # --- Decode full HV image ---
    print('Decoding HV dataset ...')
    data = decode_hv(args.l0, args.dataset)
    n_pulses, n_range = data.shape
    print(f'  Raw shape : {n_pulses} pulses x {n_range} range samples')

    # --- Compute tile grid ---
    n_cpi_rows   = n_pulses  // M
    n_range_cols = n_range   // BLOCK_WIDTH
    n_pulses_used = n_cpi_rows  * M
    n_range_used  = n_range_cols * BLOCK_WIDTH
    n_tiles = n_cpi_rows * n_range_cols

    print(f'  Tile grid : {n_cpi_rows} CPI rows x {n_range_cols} range cols '
          f'= {n_tiles} tiles')
    print(f'  Trailing pulses discarded : {n_pulses - n_pulses_used}')
    print(f'  Trailing range bins discarded : {n_range - n_range_used}')
    print()

    # Trim to exact tile boundary
    data = data[:n_pulses_used, :n_range_used]

    # --- Write output HDF5 ---
    # Features are extracted for every tile including gap/fill zones so that
    # eval_knee.py can observe how the model responds to degenerate SCMs.
    # The valid flag is stored for reference but does not gate extraction.
    print('Extracting features and writing HDF5 ...')
    n_valid   = 0
    n_invalid = 0

    with h5py.File(args.out, 'w') as f:
        # Root attributes
        f.attrs['source_file']   = args.l0
        f.attrs['source_path']   = args.dataset
        f.attrs['M']             = M
        f.attrs['BLOCK_WIDTH']   = BLOCK_WIDTH
        f.attrs['n_cpi_rows']    = n_cpi_rows
        f.attrs['n_range_cols']  = n_range_cols
        f.attrs['n_pulses']      = n_pulses
        f.attrs['n_range']       = n_range

        for ci in range(n_cpi_rows):
            p0 = ci * M
            for ri in range(n_range_cols):
                r0  = ri * BLOCK_WIDTH
                cpi = data[p0:p0 + M, r0:r0 + BLOCK_WIDTH]

                is_valid = _is_valid_tile(cpi)
                if is_valid:
                    n_valid += 1
                else:
                    n_invalid += 1

                eigen_in, global_in = extract_features(cpi)

                grp = f.create_group(f'cpi_{ci}_{ri}')
                grp.attrs['ci'] = ci
                grp.attrs['ri'] = ri
                grp.attrs['pulse_start']  = p0
                grp.attrs['pulse_end']    = p0 + M
                grp.attrs['range_start']  = r0
                grp.attrs['range_end']    = r0 + BLOCK_WIDTH

                grp.create_dataset('eigen_input',  data=eigen_in)
                grp.create_dataset('global_input', data=global_in)
                grp.create_dataset('valid', data=np.bool_(is_valid))

            if (ci + 1) % 100 == 0 or ci == n_cpi_rows - 1:
                print(f'  CPI row {ci + 1:5d} / {n_cpi_rows}  '
                      f'(valid so far: {n_valid}, invalid: {n_invalid})')

    print()
    print(f'Done.')
    print(f'  Total tiles : {n_tiles}')
    print(f'  Valid       : {n_valid}')
    print(f'  Invalid     : {n_invalid}  ({100*n_invalid/n_tiles:.1f}%)')
    print(f'  Output      : {args.out}')
    print()
    print('Next steps:')
    print(f'  python plot_eigenvalues.py --h5 {args.out}')
    print(f'  python eval_knee.py        --h5 {args.out} --model <path>')


if __name__ == '__main__':
    main()