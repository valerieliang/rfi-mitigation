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
import argparse
import numpy as np
import h5py
import tensorflow as tf

# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def plot_eigenvalue_profiles(out_h5_path, block_name, plot_dir):
    """
    Generate two eigenvalue profile plots for one block group in the output
    HDF5 and save as PNG.

    Reads eigen_input[:, 0] (eigenvalues in dB) and knee_index / confidence
    directly from the per-CPI groups written by write_cpi_groups.

    Only range tile ri=0 is used so the number of profiles equals n_cpi_rows,
    matching the spaghetti-plot style from the reference script.

    Plot 1 -- All CPI rows overlaid (color-coded by predicted knee index)
        One line per CPI row at ri=0, colored by knee_index. A vertical
        dashed line marks the mean predicted knee across all tiles in the block.

    Plot 2 -- 2x5 grid of 10 evenly-spaced CPI rows
        CPI rows sampled at equal spacing, colored by knee_index, with the
        predicted knee marked as a vertical dashed line per subplot.

    Args:
        out_h5_path : Path to the knee_maps.h5 output file.
        block_name  : Group name inside the HDF5 (e.g. 'block_mid').
        plot_dir    : Directory where the two PNG files are saved.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    os.makedirs(plot_dir, exist_ok=True)

    with h5py.File(out_h5_path, 'r') as f:
        blk        = f[block_name]
        n_cpi_rows = int(blk.attrs['n_cpi_rows'])
        m_pulses   = int(blk.attrs['M'])

        profiles   = []   # (n_cpi_rows,) each entry shape (M,)
        knees      = []   # int per CPI row
        confs      = []   # float per CPI row

        for ci in range(n_cpi_rows):
            grp      = blk[f'cpi_{ci}_0']
            ev_db    = grp['eigen_input'][:, 0]   # eigenvalues dB, shape (M,)
            knee_val = int(grp['knee_index'][()])
            conf_val = float(grp['confidence'][()])
            profiles.append(ev_db)
            knees.append(knee_val)
            confs.append(conf_val)

    knees    = np.array(knees)
    confs    = np.array(confs)
    ev_index = np.arange(m_pulses)

    # Color scale spans [0, M] (the full knee label range)
    norm = mcolors.Normalize(vmin=0, vmax=m_pulses)
    cmap = cm.plasma

    # ------------------------------------------------------------------
    # Plot 1: all CPI rows overlaid, colored by predicted knee index
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(9, 5))

    for ev_db, knee_val in zip(profiles, knees):
        ax1.plot(ev_index, ev_db, color=cmap(norm(knee_val)),
                 alpha=0.35, linewidth=0.7)

    mean_knee = knees.mean()
    ax1.axvline(mean_knee, color='white', linewidth=1.2, linestyle='--',
                label=f'mean knee = {mean_knee:.1f}')

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig1.colorbar(sm, ax=ax1)
    cbar.set_label('Predicted Knee Index', fontsize=11)

    ax1.set_xlabel('Eigenvalue Index', fontsize=11)
    ax1.set_ylabel('Eigenvalue Power (dB)', fontsize=11)
    ax1.set_title(
        f'Eigenvalue Profiles -- All CPI Rows  (ri=0, color = knee index)\n'
        f'{block_name}  |  M={m_pulses}  |  '
        f'mean knee={mean_knee:.1f}  mean conf={confs.mean():.3f}',
        fontsize=10,
    )
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.4)
    fig1.tight_layout()

    out1 = os.path.join(plot_dir, f'{block_name}_ev_all_cpi.png')
    fig1.savefig(out1, dpi=150)
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Plot 2: 2x5 grid of 10 evenly-spaced CPI rows
    # ------------------------------------------------------------------
    step         = max(n_cpi_rows // 10, 1)
    selected_cis = [i * step for i in range(10)]
    n_cols, n_rows = 5, 2
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6), sharey=True)

    for ax, ci in zip(axes.flat, selected_cis):
        ev_db    = profiles[ci]
        knee_val = knees[ci]
        conf_val = confs[ci]
        ax.plot(ev_index, ev_db, color=cmap(norm(knee_val)), linewidth=1.4)
        ax.axvline(knee_val, color='red', linewidth=1.0, linestyle='--')
        ax.set_title(f'ci={ci}  knee={knee_val}  conf={conf_val:.2f}',
                     fontsize=9)
        ax.set_xlabel('EV Index', fontsize=8)
        ax.set_ylabel('dB', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle='--', alpha=0.4)

    fig2.suptitle(
        f'Eigenvalue Profiles -- Selected CPI Rows  (ri=0, color = knee index)\n'
        f'{block_name}',
        fontsize=11,
    )
    fig2.tight_layout()

    out2 = os.path.join(plot_dir, f'{block_name}_ev_selected_cpi.png')
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)

    print(f'    plots -> {os.path.basename(out1)}, {os.path.basename(out2)}')

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

def _decode_chunk(chunk, dataset, group):
    """
    Decode a raw chunk to complex64 following the same logic as isce3's
    DataDecoder (python/packages/nisar/products/readers/Raw/DataDecoder.py).

    Three encoding formats exist in NISAR L0B files:

    1. BFPQLUT  -- Block Floating Point Quantization with lookup table.
                   A sibling dataset BFPQLUT (float32) is present in the
                   same HDF5 group. Each sample is a compound {r, i} index
                   into the LUT: complex = LUT[r] + j*LUT[i].

    2. complex32 (float16 pairs) -- Compound dtype {r: float16, i: float16}.
                   h5py >= 3.8 exposes this as a compound dtype; older h5py
                   raises TypeError on .dtype access and we fall back to
                   treating it as complex32.

    3. complex64 -- Already native; no decoding needed.

    The NISAR RRSD (L0B) products encountered here use encoding (1) or (3).
    The compound U16 dtype ({r: U16, i: U16}) is the BFPQLUT index format:
    each sample is a pair of uint16 indices into the LUT.

    Args:
        chunk   : Raw numpy array as returned by h5py slice.
        dataset : The h5py dataset object (for dtype introspection).
        group   : The parent h5py group (to check for BFPQLUT sibling).

    Returns:
        block (np.ndarray): complex64
    """
    # Path 1: BFPQLUT encoding (U16 compound indices into float32 LUT)
    if "BFPQLUT" in group:
        lut   = np.asarray(group["BFPQLUT"], dtype=np.float32)
        block = lut[chunk['r']].astype(np.float32)               + 1j * lut[chunk['i']].astype(np.float32)
        return block.astype(np.complex64)

    # Path 2: float16 compound (complex32)
    # h5py >= 3.8 exposes compound dtype; older versions raise TypeError
    try:
        storage_dtype = dataset.dtype
        is_complex32  = (
            storage_dtype.names is not None
            and set(storage_dtype.names) == {'r', 'i'}
            and storage_dtype['r'] == np.float16
        )
    except TypeError:
        is_complex32 = True   # older h5py with complex32

    if is_complex32:
        block = chunk['r'].astype(np.float32)               + 1j * chunk['i'].astype(np.float32)
        return block.astype(np.complex64)

    # Path 3: already complex64
    if np.issubdtype(chunk.dtype, np.complexfloating):
        return chunk.astype(np.complex64)

    raise ValueError(
        f"Unsupported raw data dtype: {dataset.dtype}. "
        "Expected BFPQLUT compound, float16 compound, or complex64."
    )


def load_block(l0_path, dataset_path, pulse_start, pulse_end):
    """
    Read a pulse range from the raw NISAR L0B HDF5 and return complex64.

    Handles all three NISAR L0B encoding formats (BFPQLUT, complex32,
    complex64) by delegating to _decode_chunk.

    Args:
        l0_path      : Path to the raw NISAR L0B HDF5 file.
        dataset_path : Internal HDF5 path to the raw dataset (e.g.
                       /science/LSAR/RRSD/swaths/frequencyA/txH/rxV/HV).
        pulse_start  : First pulse index (inclusive).
        pulse_end    : Last pulse index (exclusive).

    Returns:
        block (np.ndarray): complex64, shape (pulse_end-pulse_start, range_count)
    """
    with h5py.File(l0_path, 'r', libver='latest', swmr=True) as f:
        dataset = f[dataset_path]
        group   = dataset.parent
        chunk   = dataset[pulse_start:pulse_end, :]
        block   = _decode_chunk(chunk, dataset, group)
    return block


# ---------------------------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------------------------

def _is_valid_tile(cpi):
    """
    Return False if a CPI tile is invalid (satellite gap / fill zone).

    Invalid tiles are all-zero or near-zero rows from black bands between
    data collection segments. They produce degenerate SCMs and must not
    be passed to the model.

    A tile is marked invalid if more than half its pulse rows have zero
    total power (sum of |sample|^2 == 0).
    """
    row_power = np.sum(np.abs(cpi) ** 2, axis=1)   # (M,)
    zero_rows = np.sum(row_power == 0.0)
    return zero_rows <= (cpi.shape[0] // 2)


def _extract_features(cpi):
    """
    Compute per-CPI features from a raw complex tile.

    Mirrors extract_features() in train.py so this script is self-contained.

    Args:
        cpi (np.ndarray): complex64, shape (M, K)

    Returns:
        eigen_input  (np.ndarray): float32, shape (M, 2)
            channel 0 -- eigenvalues of the SCM in dB, sorted descending
            channel 1 -- finite-differenced slopes, last value repeated to
                         pad to length M
        global_input (np.ndarray): float32, shape (6,)
            [F_factor, sigma_min, sigma_max, mu_min, trace_db, condition_number]
    """
    eps = 1e-12

    # Sample Covariance Matrix (M x M)
    scm     = (cpi @ cpi.conj().T) / cpi.shape[1]

    # Eigenvalues (real, descending order)
    eigvals = np.linalg.eigvalsh(scm).real[::-1]
    eigvals = np.maximum(eigvals, eps)

    ev_db  = 10.0 * np.log10(eigvals)         # (M,)
    slopes = np.diff(ev_db)                    # (M-1,)
    slopes = np.append(slopes, slopes[-1])     # pad to (M,)

    eigen_input = np.stack([ev_db, slopes], axis=1).astype(np.float32)

    # Global / ST-EST proxy features
    # sigma_max: std of the top-half eigenvalue diffs (most likely RFI region)
    # sigma_min: std of the bottom-half eigenvalue diffs (signal/noise floor)
    # mu_min   : mean of the bottom-half eigenvalue diffs
    # These approximate the TB-level sigma_max/sigma_min used by ST-EST.
    half      = max(len(eigvals) // 2, 1)
    diffs_top = np.diff(eigvals[:half])
    diffs_bot = np.diff(eigvals[half:])

    sigma_max   = float(np.std(diffs_top))  if len(diffs_top) > 0 else 0.0
    sigma_min   = float(np.std(diffs_bot))  if len(diffs_bot) > 0 else 0.0
    mu_min      = float(np.mean(diffs_bot)) if len(diffs_bot) > 0 else 0.0
    f_factor    = sigma_max / max(sigma_min, eps)

    trace       = float(np.real(np.trace(scm)))
    trace_db    = 10.0 * np.log10(max(trace, eps))
    cond_number = float(eigvals[0] / max(eigvals[-1], eps))

    global_input = np.array(
        [f_factor, sigma_min, sigma_max, mu_min, trace_db, cond_number],
        dtype=np.float32,
    )

    return eigen_input, global_input


# Sentinel values written for invalid (gap) tiles so the HDF5 is complete
# but downstream code can filter by the valid flag.
_INVALID_EIGEN  = np.zeros((M, 2),  dtype=np.float32)
_INVALID_GLOBAL = np.zeros((6,),    dtype=np.float32)
_INVALID_KNEE   = np.int16(-1)
_INVALID_CONF   = np.float32(0.0)
_INVALID_ENT    = np.float32(0.0)


def extract_all_features(block, n_cpi_rows, n_range_cols):
    """
    Tile block into CPI tiles and extract features for every valid tile.

    Invalid tiles (satellite gap rows, all-zero pulses) are detected via
    _is_valid_tile and stored with sentinel values. The valid flag in each
    HDF5 group identifies them downstream.

    Args:
        block        : complex64, shape (n_cpi_rows*M, >=n_range_cols*BLOCK_WIDTH)
        n_cpi_rows   : Number of CPI rows derived from block pulse count.
        n_range_cols : Number of range tiles derived from block range width.

    Returns:
        eigen_buf  : float32, shape (n_tiles, M, 2)
        global_buf : float32, shape (n_tiles, 6)
        valid_mask : bool,    shape (n_tiles,)  True = valid data
    """
    range_used = n_range_cols * BLOCK_WIDTH
    block      = block[:n_cpi_rows * M, :range_used]

    n_tiles    = n_cpi_rows * n_range_cols
    eigen_buf  = np.empty((n_tiles, M, 2), dtype=np.float32)
    global_buf = np.empty((n_tiles, 6),    dtype=np.float32)
    valid_mask = np.ones(n_tiles, dtype=bool)

    idx = 0
    for ci in range(n_cpi_rows):
        p0 = ci * M
        for ri in range(n_range_cols):
            r0  = ri * BLOCK_WIDTH
            cpi = block[p0:p0 + M, r0:r0 + BLOCK_WIDTH]
            if _is_valid_tile(cpi):
                eigen_buf[idx], global_buf[idx] = _extract_features(cpi)
            else:
                eigen_buf[idx]  = _INVALID_EIGEN
                global_buf[idx] = _INVALID_GLOBAL
                valid_mask[idx] = False
            idx += 1

    return eigen_buf, global_buf, valid_mask


# ---------------------------------------------------------------------------
# INFERENCE
# ---------------------------------------------------------------------------

def run_inference(model, eigen_buf, global_buf, valid_mask):
    """
    Run model over pre-extracted feature buffers, skipping invalid tiles.

    Args:
        model      : Loaded Keras model.
        eigen_buf  : float32, shape (n_tiles, M, 2)
        global_buf : float32, shape (n_tiles, 6)
        valid_mask : bool,    shape (n_tiles,)

    Returns:
        knee : int16,   shape (n_tiles,)  -1 for invalid tiles
        conf : float32, shape (n_tiles,)   0 for invalid tiles
        ent  : float32, shape (n_tiles,)   0 for invalid tiles
    """
    n_tiles = len(valid_mask)
    knee    = np.full(n_tiles, _INVALID_KNEE,  dtype=np.int16)
    conf    = np.full(n_tiles, _INVALID_CONF,  dtype=np.float32)
    ent     = np.full(n_tiles, _INVALID_ENT,   dtype=np.float32)

    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        return knee, conf, ent

    probs = model.predict(
        [eigen_buf[valid_idx], global_buf[valid_idx]],
        batch_size=512,
        verbose=0,
    )   # (n_valid, M+1)

    eps = 1e-12
    knee[valid_idx] = np.argmax(probs, axis=-1).astype(np.int16)
    conf[valid_idx] = np.max(probs, axis=-1).astype(np.float32)
    ent[valid_idx]  = (-np.sum(probs * np.log(probs + eps), axis=-1)).astype(np.float32)

    return knee, conf, ent


# ---------------------------------------------------------------------------
# HDF5 WRITING
# ---------------------------------------------------------------------------

def write_cpi_groups(blk_grp, eigen_buf, global_buf, knee, conf, ent,
                     valid_mask, n_cpi_rows, n_range_cols):
    """
    Write one HDF5 group per CPI tile under blk_grp.

    Group naming: cpi_{ci}_{ri}

    Invalid tiles (satellite gap zones) are written with sentinel values and
    valid=0 so downstream code can filter them out without special-casing
    missing groups.

    Args:
        blk_grp      : h5py.Group for the current block (e.g. /block_mid)
        eigen_buf    : float32, shape (n_tiles, M, 2)
        global_buf   : float32, shape (n_tiles, 6)
        knee         : int16,   shape (n_tiles,)  -1 = invalid
        conf         : float32, shape (n_tiles,)
        ent          : float32, shape (n_tiles,)
        valid_mask   : bool,    shape (n_tiles,)
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
            cpi_grp.create_dataset('valid',        data=np.bool_(valid_mask[idx]))

            idx += 1


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Infer knee-index map directly from a raw NISAR L0B file, '
                    'storing results per CPI tile in HDF5.'
    )
    parser.add_argument('--l0',      default=DEFAULT_L0,
                        help='Path to raw NISAR L0B HDF5 file.')
    parser.add_argument('--model',   default=DEFAULT_MODEL)
    parser.add_argument('--out',     default=DEFAULT_OUT)
    parser.add_argument('--plot',    action='store_true',
                        help='Save eigenvalue profile PNGs after inference.')
    parser.add_argument('--plot-dir', default=None,
                        help='Directory for plots (default: same dir as --out).')
    args = parser.parse_args()

    if not os.path.exists(args.l0):
        raise FileNotFoundError(f'L0B file not found: {args.l0}')
    if not os.path.exists(args.model):
        raise FileNotFoundError(f'Model not found: {args.model}')

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    plot_dir = args.plot_dir if args.plot_dir else out_dir

    print(f'L0B source : {args.l0}')
    print(f'Model      : {args.model}')
    print(f'Output     : {args.out}')
    if args.plot:
        print(f'Plot dir   : {plot_dir}')
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
            eigen_buf, global_buf, valid_mask = extract_all_features(
                block, n_cpi_rows, n_range_cols
            )
            n_valid   = int(valid_mask.sum())
            n_invalid = len(valid_mask) - n_valid
            print(f'  Valid tiles  : {n_valid}  /  Invalid (gap): {n_invalid}')

            print(f'  Running inference ...')
            knee, conf, ent = run_inference(model, eigen_buf, global_buf,
                                            valid_mask)

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
                             valid_mask, n_cpi_rows, n_range_cols)

            valid_knee = knee[valid_mask]
            valid_conf = conf[valid_mask]
            valid_ent  = ent[valid_mask]
            if len(valid_knee) > 0:
                print(f'  knee_index : min={valid_knee.min()}  '
                      f'max={valid_knee.max()}  mean={valid_knee.mean():.2f}')
                print(f'  mean conf  : {valid_conf.mean():.4f}')
                print(f'  mean ent   : {valid_ent.mean():.4f}')
            else:
                print(f'  (no valid tiles in this block)')

            if args.plot:
                print(f'  Plotting ...')
                # HDF5 file must be flushed before reading back for plots
                out_f.flush()
                plot_eigenvalue_profiles(args.out, block_name, plot_dir)

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
