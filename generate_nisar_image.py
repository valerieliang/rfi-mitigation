"""
generate_nisar_image.py

Generates synthetic complex-valued NISAR-like raw data frames for use in
RFI CNN training and evaluation pipelines.

Conventions:
  - Matrix shape: (TOTAL_PULSES, RANGE_BINS) = (rows, cols)
  - Pulses run along rows (slow-time / azimuth direction)
  - Range bins run along columns (fast-time / range direction)
  - All matrices are complex-valued (complex64)
  - Power convention: 10 * log10(E[|x|^2]) in dB
  - Noise floor: NOISE_DB = 3 dB
  - Signal SNR above noise: SNR_DB = 6 dB

RFI block structure:
  - The pulse axis is divided into non-overlapping blocks of BLOCK_SIZE pulses.
  - TOTAL_PULSES must be divisible by BLOCK_SIZE.
  - RFI is injected into every block at a randomly chosen position within that
    block. generate_rfi_image returns an RfiMeta object carrying JNR and the
    per-block injection positions (see RfiMeta for format details).

HDF5 layout:
  Root attributes  : file-level config (dimensions, RFI type, seed, JNR, etc.)
  Dataset per CPI  : name "cpi_{i}_{j}", complex64 array of shape
                     (cpi_height, cpi_width).
  Dataset attributes: per-CPI RFI metadata:
      single_tone  -- rfi_row (int): absolute pulse index of the active row
      wideband     -- rfi_start (int), rfi_end (int): inclusive pulse range
                      of the active run within this CPI's pulse band.
"""

import os
import numpy as np
import h5py
from dataclasses import dataclass, field
from typing import List, Tuple, Union

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

NOISE_DB = 3
SNR_DB   = 6        # signal power = noise power * 10^(SNR_DB/10)

LOW_POWER_JNR_RANGE  = (6,  12)
HIGH_POWER_JNR_RANGE = (20, 30)

TOTAL_PULSES = 1600     # rows  (slow-time / azimuth)
RANGE_BINS   = 10000    # cols  (fast-time / range)
BLOCK_SIZE   = 16       # pulses per CPI block; TOTAL_PULSES must be divisible

assert TOTAL_PULSES % BLOCK_SIZE == 0, (
    f"TOTAL_PULSES ({TOTAL_PULSES}) must be divisible by BLOCK_SIZE ({BLOCK_SIZE})"
)
N_BLOCKS = TOTAL_PULSES // BLOCK_SIZE

# Default CPI tile dimensions used by divide_cpi_and_save
BLOCK_HEIGHT = BLOCK_SIZE   # pulse rows per CPI tile
BLOCK_WIDTH  = 1000         # range cols per CPI tile

# Child-seed offsets for SeedSequence-style isolation
NOISE_SEED  = 0
SIGNAL_SEED = 1
RFI_SEED    = 2
JNR_SEED    = 3     # separate offset so JNR draw does not perturb RFI state


# ---------------------------------------------------------------------------
# RFI METADATA CONTAINER
# ---------------------------------------------------------------------------

@dataclass
class RfiMeta:
    """
    Carries all RFI injection metadata for one generated image.

    Attributes:
        rfi_type      (str): 'single_tone' or 'wideband'.
        jnr_per_block (list[int]): Per-block JNR in dB (N_BLOCKS entries).
            Each block draws its own JNR independently from jnr_range, so
            power varies across the image. Overlaps between blocks are allowed.

        For single_tone:
            rfi_rows (list[int]): One absolute pulse row per block (N_BLOCKS
                entries), the row where the CW tone was injected.
            rfi_spans is empty.

        For wideband:
            rfi_spans (list[tuple[int,int]]): One (start, end) tuple per block
                (N_BLOCKS entries). start and end are absolute pulse indices,
                both inclusive, of the contiguous run of active rows.
            rfi_rows is empty.
    """
    rfi_type      : str
    jnr_per_block : List[int]
    rfi_rows      : List[int]             = field(default_factory=list)
    rfi_spans     : List[Tuple[int, int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# UTILITY
# ---------------------------------------------------------------------------

def verify_matrix_power(matrix, expected_power_db, label="matrix"):
    """
    Print and assert that the complex matrix power is within 1 dB of expected.

    Power is computed as 10 * log10(mean(|x|^2)), consistent with the
    complex-baseband convention used throughout this file.

    Intended as a presentation-ready sanity check; not called automatically
    during dataset generation.

    Args:
        matrix (np.ndarray): Complex-valued input matrix.
        expected_power_db (float): Expected power in dB.
        label (str): Human-readable name for the matrix (used in printout).
    """
    actual_power_db = 10.0 * np.log10(np.mean(np.abs(matrix) ** 2))
    print(
        f"[{label}] actual power: {actual_power_db:.2f} dB, "
        f"expected: {expected_power_db:.2f} dB"
    )
    assert abs(actual_power_db - expected_power_db) < 1.0, (
        f"[{label}] power {actual_power_db:.2f} dB deviates more than 1 dB "
        f"from expected {expected_power_db:.2f} dB"
    )


# ---------------------------------------------------------------------------
# CLEAN IMAGE
# ---------------------------------------------------------------------------

def generate_clean_image(seed=0):
    """
    Generate a complex-valued clean image: spatially white noise + signal.

    Noise power : NOISE_DB dB
    Signal power: NOISE_DB + SNR_DB dB

    Args:
        seed (int): Base seed. Child seeds are seed+NOISE_SEED, seed+SIGNAL_SEED.

    Returns:
        clean_image (np.ndarray): Complex64 array of shape (TOTAL_PULSES, RANGE_BINS).
    """
    noise_power_linear  = 10.0 ** (NOISE_DB / 10.0)
    signal_power_linear = noise_power_linear * (10.0 ** (SNR_DB / 10.0))

    # Complex Gaussian: real and imag each carry half the power.
    rng_noise = np.random.default_rng(seed + NOISE_SEED)
    noise_matrix = (
        rng_noise.standard_normal((TOTAL_PULSES, RANGE_BINS))
        + 1j * rng_noise.standard_normal((TOTAL_PULSES, RANGE_BINS))
    ) * np.sqrt(noise_power_linear / 2.0)

    rng_signal = np.random.default_rng(seed + SIGNAL_SEED)
    signal_matrix = (
        rng_signal.standard_normal((TOTAL_PULSES, RANGE_BINS))
        + 1j * rng_signal.standard_normal((TOTAL_PULSES, RANGE_BINS))
    ) * np.sqrt(signal_power_linear / 2.0)

    return (noise_matrix + signal_matrix).astype(np.complex64)


# ---------------------------------------------------------------------------
# RFI IMAGE
# ---------------------------------------------------------------------------

def generate_rfi_image(seed=0, jnr_range=(20, 30), rfi_type='single_tone'):
    """
    Generate a complex-valued image with additive RFI injected into a clean frame.

    RFI is injected into every block of BLOCK_SIZE pulses along the pulse axis.
    Each block draws its own JNR independently from jnr_range so that power
    varies across the image (overlaps are allowed).

    Args:
        seed (int): Base seed. JNR draws use seed+JNR_SEED; RFI uses seed+RFI_SEED.
        jnr_range (tuple): (low, high) JNR in dB, both ends inclusive.
        rfi_type (str): 'single_tone' or 'wideband'.

    Returns:
        rfi_image (np.ndarray): Complex64 array of shape (TOTAL_PULSES, RANGE_BINS).
        meta      (RfiMeta): Per-block JNR, type, and injection positions.
    """
    clean_image = generate_clean_image(seed)

    # One JNR RNG shared by both generators; each block advances it by one draw
    rng_jnr = np.random.default_rng(seed + JNR_SEED)

    if rfi_type == 'single_tone':
        rfi_signal, rfi_rows, jnr_per_block = _generate_single_tone_rfi(
            seed, jnr_range, rng_jnr
        )
        meta = RfiMeta(rfi_type=rfi_type, jnr_per_block=jnr_per_block,
                       rfi_rows=rfi_rows)
    elif rfi_type == 'wideband':
        rfi_signal, rfi_spans, jnr_per_block = _generate_wideband_rfi(
            seed, jnr_range, rng_jnr
        )
        meta = RfiMeta(rfi_type=rfi_type, jnr_per_block=jnr_per_block,
                       rfi_spans=rfi_spans)
    else:
        raise ValueError(
            f"Unknown rfi_type '{rfi_type}'. Use 'single_tone' or 'wideband'."
        )

    rfi_image = (clean_image + rfi_signal).astype(np.complex64)
    return rfi_image, meta


# ---------------------------------------------------------------------------
# RFI SIGNAL GENERATORS
# ---------------------------------------------------------------------------

def _generate_single_tone_rfi(seed, jnr_range, rng_jnr):
    """
    Generate a narrowband (CW) RFI signal injected into every block.

    In each block of BLOCK_SIZE pulses, exactly one pulse row is chosen at
    random to carry the RFI tone. Each block draws its own JNR independently
    from jnr_range so that power varies across the image. The same Doppler
    frequency is reused across all blocks (same physical emitter); the range
    coefficient vector is rescaled per block to match that block's JNR.

    Args:
        seed      (int): Base seed; RFI RNG uses seed+RFI_SEED.
        jnr_range (tuple): (low, high) JNR in dB, both ends inclusive.
        rng_jnr   (np.random.Generator): Seeded JNR RNG; advanced one integer
                  draw per block.

    Returns:
        rfi_matrix    (np.ndarray): Complex64, shape (TOTAL_PULSES, RANGE_BINS).
        rfi_rows      (list[int]): One absolute row index per block (N_BLOCKS entries).
        jnr_per_block (list[int]): JNR in dB for each block (N_BLOCKS entries).
    """
    rng = np.random.default_rng(seed + RFI_SEED)

    doppler_freq = rng.uniform(-0.5, 0.5)
    # Unit-variance range direction template; scaled per block below
    range_template = (
        rng.standard_normal(RANGE_BINS) + 1j * rng.standard_normal(RANGE_BINS)
    )

    noise_power_linear = 10.0 ** (NOISE_DB / 10.0)
    rfi_matrix    = np.zeros((TOTAL_PULSES, RANGE_BINS), dtype=np.complex64)
    rfi_rows      = []
    jnr_per_block = []

    for b in range(N_BLOCKS):
        block_start = b * BLOCK_SIZE
        local_idx   = int(rng.integers(0, BLOCK_SIZE))
        abs_row     = block_start + local_idx

        jnr_db          = int(rng_jnr.integers(jnr_range[0], jnr_range[1] + 1))
        rfi_power_linear = noise_power_linear * (10.0 ** (jnr_db / 10.0))
        sigma            = np.sqrt(rfi_power_linear / 2.0)

        phase = np.exp(1j * 2.0 * np.pi * doppler_freq * abs_row)
        rfi_matrix[abs_row, :] = (phase * range_template * sigma).astype(np.complex64)

        rfi_rows.append(abs_row)
        jnr_per_block.append(jnr_db)

    return rfi_matrix, rfi_rows, jnr_per_block


def _generate_wideband_rfi(seed, jnr_range, rng_jnr, width=4):
    """
    Generate a wideband RFI signal injected into every block.

    In each block of BLOCK_SIZE pulses, 'width' consecutive rows are chosen
    at a random start position such that all rows fit within the block. Each
    block draws its own JNR independently from jnr_range so that power varies
    across the image. Each block also gets independent Doppler frequencies and
    log-normal power jitter with the sum renormalised to that block's total
    rfi_power_linear.

    Args:
        seed      (int): Base seed; RFI RNG uses seed+RFI_SEED.
        jnr_range (tuple): (low, high) JNR in dB, both ends inclusive.
        rng_jnr   (np.random.Generator): Seeded JNR RNG; advanced one integer
                  draw per block.
        width     (int): Number of consecutive active rows per block (= SCM rank).

    Returns:
        rfi_matrix    (np.ndarray): Complex64, shape (TOTAL_PULSES, RANGE_BINS).
        rfi_spans     (list[tuple[int,int]]): One (start, end) inclusive tuple
                      per block (N_BLOCKS entries).
        jnr_per_block (list[int]): JNR in dB for each block (N_BLOCKS entries).
    """
    assert width <= BLOCK_SIZE, (
        f"width ({width}) must be <= BLOCK_SIZE ({BLOCK_SIZE})"
    )

    noise_power_linear = 10.0 ** (NOISE_DB / 10.0)
    rng = np.random.default_rng(seed + RFI_SEED)

    rfi_matrix    = np.zeros((TOTAL_PULSES, RANGE_BINS), dtype=np.complex64)
    rfi_spans     = []
    jnr_per_block = []

    for b in range(N_BLOCKS):
        block_start     = b * BLOCK_SIZE
        max_local_start = BLOCK_SIZE - width
        local_start     = int(rng.integers(0, max_local_start + 1))
        abs_start       = block_start + local_start
        abs_end         = abs_start + width - 1   # inclusive
        active_rows     = list(range(abs_start, abs_end + 1))

        jnr_db           = int(rng_jnr.integers(jnr_range[0], jnr_range[1] + 1))
        rfi_power_linear  = noise_power_linear * (10.0 ** (jnr_db / 10.0))

        freqs = rng.uniform(-0.5, 0.5, size=width)

        log_jitter       = np.exp(rng.standard_normal(width) * 0.5)
        component_powers = (rfi_power_linear / width) * log_jitter
        component_powers *= rfi_power_linear / component_powers.sum()

        for i in range(width):
            row   = active_rows[i]
            phase = np.exp(1j * 2.0 * np.pi * freqs[i] * row)
            sigma = np.sqrt(component_powers[i] / 2.0)
            range_coeff = (
                rng.standard_normal(RANGE_BINS) + 1j * rng.standard_normal(RANGE_BINS)
            ) * sigma
            rfi_matrix[row, :] += (phase * range_coeff).astype(np.complex64)

        rfi_spans.append((abs_start, abs_end))
        jnr_per_block.append(jnr_db)

    return rfi_matrix, rfi_spans, jnr_per_block


# ---------------------------------------------------------------------------
# DIVIDE CPI AND SAVE AS HDF5
# ---------------------------------------------------------------------------

def divide_cpi_and_save(
    matrix,
    meta,
    seed,
    jnr_range,
    cpi_height=BLOCK_HEIGHT,
    cpi_width=BLOCK_WIDTH,
    output_path=None,
):
    """
    Divide a complex-valued matrix into non-overlapping CPI tiles and save to HDF5.

    HDF5 layout
    -----------
    Root attributes (file-level config):
        total_pulses  (int)   : total rows in the source image
        range_bins    (int)   : total columns in the source image
        cpi_height    (int)   : pulse rows per CPI tile
        cpi_width     (int)   : range columns per CPI tile
        block_size    (int)   : pulse rows per RFI injection block
        n_blocks      (int)   : number of RFI injection blocks
        noise_db      (float) : noise floor in dB
        snr_db        (float) : signal-to-noise ratio in dB
        rfi_type      (str)   : 'single_tone' or 'wideband'
        jnr_db        (int)   : actual JNR drawn for this image
        jnr_range_low (int)   : lower bound of JNR sampling range
        jnr_range_high(int)   : upper bound of JNR sampling range
        seed          (int)   : base seed used for generation

    Dataset per CPI tile, name "cpi_{i}_{j}":
        Data: complex64 array of shape (cpi_height, cpi_width).
        Attributes (single_tone):
            rfi_row (int): absolute pulse row of the CW tone within this tile's
                           pulse band. Present only when that band contains RFI.
        Attributes (wideband):
            rfi_start (int): absolute start row (inclusive) of the active run.
            rfi_end   (int): absolute end row (inclusive) of the active run.
            Both present only when the band's span falls within this tile's range.

    Args:
        matrix     (np.ndarray): Complex64 array of shape (TOTAL_PULSES, RANGE_BINS).
        meta       (RfiMeta)   : RFI injection metadata from generate_rfi_image.
        seed       (int)       : Base seed used to generate the image.
        jnr_range  (tuple)     : (low, high) JNR range passed to generate_rfi_image.
        cpi_height (int)       : Pulse rows per CPI tile. Must divide TOTAL_PULSES.
        cpi_width  (int)       : Range columns per CPI tile. Must divide RANGE_BINS.
        output_path(str|None)  : Path for the HDF5 file. No file is written if None.
    """
    if output_path is None:
        return

    assert TOTAL_PULSES % cpi_height == 0, (
        f"cpi_height ({cpi_height}) must divide TOTAL_PULSES ({TOTAL_PULSES})"
    )
    assert RANGE_BINS % cpi_width == 0, (
        f"cpi_width ({cpi_width}) must divide RANGE_BINS ({RANGE_BINS})"
    )

    # Build lookup structures for fast per-CPI metadata attachment.
    # For single_tone: map block_index -> absolute rfi_row.
    # For wideband:    map block_index -> (abs_start, abs_end).
    # A CPI tile at pulse offset i spans rows [i, i+cpi_height).
    # The block index for that row band is i // BLOCK_SIZE (assuming
    # cpi_height == BLOCK_SIZE, which is the expected configuration).

    with h5py.File(output_path, 'w') as f:

        # --- Root attributes -----------------------------------------------
        f.attrs['total_pulses']   = TOTAL_PULSES
        f.attrs['range_bins']     = RANGE_BINS
        f.attrs['cpi_height']     = cpi_height
        f.attrs['cpi_width']      = cpi_width
        f.attrs['block_size']     = BLOCK_SIZE
        f.attrs['n_blocks']       = N_BLOCKS
        f.attrs['noise_db']       = NOISE_DB
        f.attrs['snr_db']         = SNR_DB
        f.attrs['rfi_type']       = meta.rfi_type
        f.attrs['jnr_range_low']  = jnr_range[0]
        f.attrs['jnr_range_high'] = jnr_range[1]
        f.attrs['seed']           = seed

        # --- CPI datasets ---------------------------------------------------
        for i in range(0, TOTAL_PULSES, cpi_height):
            # Block index that this pulse band belongs to.
            # Assumes cpi_height divides BLOCK_SIZE or vice versa so each
            # tile maps cleanly to one injection block.
            block_idx = i // BLOCK_SIZE

            for j in range(0, RANGE_BINS, cpi_width):
                cpi      = matrix[i:i + cpi_height, j:j + cpi_width]
                dset     = f.create_dataset(f"cpi_{i}_{j}", data=cpi)

                # Attach per-CPI RFI metadata
                if meta.rfi_type == 'single_tone':
                    # rfi_rows is in block order; index directly by block_idx
                    dset.attrs['rfi_row'] = meta.rfi_rows[block_idx]
                    dset.attrs['jnr_db']  = meta.jnr_per_block[block_idx]

                elif meta.rfi_type == 'wideband':
                    start, end = meta.rfi_spans[block_idx]
                    dset.attrs['rfi_start'] = start
                    dset.attrs['rfi_end']   = end
                    dset.attrs['jnr_db']    = meta.jnr_per_block[block_idx]


# ---------------------------------------------------------------------------
# PIPELINE CONFIGURATION
# ---------------------------------------------------------------------------

# Each entry defines one output folder and the RFI parameters used to fill it.
# Folder structure: data/<power_level>_power/<rfi_type>/image_<seed>.h5
_PIPELINE_CONFIGS = [
    {
        'rfi_type'  : 'single_tone',
        'jnr_range' : LOW_POWER_JNR_RANGE,
        'power_level': 'low',
    },
    {
        'rfi_type'  : 'single_tone',
        'jnr_range' : HIGH_POWER_JNR_RANGE,
        'power_level': 'high',
    },
    {
        'rfi_type'  : 'wideband',
        'jnr_range' : LOW_POWER_JNR_RANGE,
        'power_level': 'low',
    },
    {
        'rfi_type'  : 'wideband',
        'jnr_range' : HIGH_POWER_JNR_RANGE,
        'power_level': 'high',
    },
]

N_IMAGES   = 10     # number of raw images (seeds 0..N_IMAGES-1)
DATA_ROOT  = 'data' # root output directory


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def _compute_eigenvalues_db(cpi):
    """
    Compute sorted descending eigenvalues in dB from a complex CPI tile.

    Forms the sample covariance matrix R = CPI @ CPI^H / K where K is the
    number of range bins (columns), then returns eigenvalues sorted descending
    converted to dB via 10 * log10.

    Args:
        cpi (np.ndarray): Complex array of shape (M, K) where M = BLOCK_SIZE.

    Returns:
        eigvals_db (np.ndarray): Shape (M,), eigenvalues in dB descending order.
    """
    M, K  = cpi.shape
    R     = (cpi @ cpi.conj().T) / K           # (M, M) sample covariance
    eigvals = np.linalg.eigvalsh(R)             # ascending real eigenvalues
    eigvals = np.sort(eigvals)[::-1]            # descending
    eigvals_db = 10.0 * np.log10(np.maximum(eigvals, 1e-12))
    return eigvals_db


def plot_eigenvalue_profiles(h5_path, out_dir):
    """
    Generate two eigenvalue profile plots for a single HDF5 file and save as PNG.

    Plot 1 -- All-blocks overlay (color-coded by JNR)
        All 100 pulse-blocks from range tile j=0 are overlaid on one axes.
        Lines are drawn in a colormap keyed to the file's JNR value so that
        when this function is called across multiple files (different JNR draws)
        the color conveys power level. Within a single file all lines share the
        same JNR, so the spread shows spatial variation across pulse-blocks.
        Colormap range is fixed to the full JNR axis (LOW_POWER_JNR_RANGE[0]
        to HIGH_POWER_JNR_RANGE[1]) so colours are comparable across files.

    Plot 2 -- Grid of 10 individual block profiles
        10 evenly-spaced pulse-blocks (indices 0, 10, 20, ..., 90) from range
        tile j=0 are shown in a 2x5 subplot grid. Each panel shows one block's
        eigenvalue profile so fine-grained shape differences are visible.

    Args:
        h5_path (str): Path to the source HDF5 file.
        out_dir (str): Directory where the two PNG files are saved.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    jnr_min_global = LOW_POWER_JNR_RANGE[0]
    jnr_max_global = HIGH_POWER_JNR_RANGE[1]

    # Block pulse offsets in this file (multiples of BLOCK_SIZE, j=0 column)
    block_pulse_offsets = [b * BLOCK_SIZE for b in range(N_BLOCKS)]
    ev_index            = np.arange(BLOCK_SIZE)

    with h5py.File(h5_path, 'r') as f:
        rfi_type = str(f.attrs['rfi_type'])

        # Collect eigenvalue profiles and per-block JNR for all blocks at j=0
        all_profiles  = []
        all_jnr       = []
        for pulse_offset in block_pulse_offsets:
            dset        = f[f"cpi_{pulse_offset}_0"]
            cpi         = dset[:]
            ev_db       = _compute_eigenvalues_db(cpi)
            all_profiles.append(ev_db)
            all_jnr.append(int(dset.attrs['jnr_db']))

    norm = mcolors.Normalize(vmin=jnr_min_global, vmax=jnr_max_global)
    cmap = cm.plasma

    # ------------------------------------------------------------------
    # Plot 1: all 100 blocks overlaid, each line colored by its block JNR
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(9, 5))

    for ev_db, jnr in zip(all_profiles, all_jnr):
        ax1.plot(ev_index, ev_db, color=cmap(norm(jnr)), alpha=0.45, linewidth=0.8)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig1.colorbar(sm, ax=ax1)
    cbar.set_label('JNR (dB)', fontsize=11)

    ax1.set_xlabel('Eigenvalue Index', fontsize=11)
    ax1.set_ylabel('Eigenvalue Power (dB)', fontsize=11)
    ax1.set_title(
        f'Eigenvalue Profiles -- All Blocks  (color = JNR per block)\n'
        f'{rfi_type}  |  JNR range {jnr_min_global}-{jnr_max_global} dB  |  '
        f'{os.path.basename(h5_path)}',
        fontsize=11,
    )
    ax1.grid(True, linestyle='--', alpha=0.4)
    fig1.tight_layout()

    stem1    = os.path.splitext(os.path.basename(h5_path))[0]
    out_path1 = os.path.join(out_dir, f"{stem1}_ev_all_blocks.png")
    fig1.savefig(out_path1, dpi=150)
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Plot 2: 2x5 grid of 10 evenly-spaced blocks
    # ------------------------------------------------------------------
    selected_block_indices = [b * (N_BLOCKS // 10) for b in range(10)]  # 0,10,20,...,90
    n_cols, n_rows = 5, 2
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6), sharey=True)

    for ax, block_idx in zip(axes.flat, selected_block_indices):
        ev_db = all_profiles[block_idx]
        jnr   = all_jnr[block_idx]
        ax.plot(ev_index, ev_db, color=cmap(norm(jnr)), linewidth=1.4)
        ax.set_title(f'Block {block_idx}  |  {jnr} dB', fontsize=9)
        ax.set_xlabel('EV Index', fontsize=8)
        ax.set_ylabel('dB', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle='--', alpha=0.4)

    fig2.suptitle(
        f'Eigenvalue Profiles -- Selected Blocks  (color = JNR per block)\n'
        f'{rfi_type}  |  {os.path.basename(h5_path)}',
        fontsize=11,
    )
    fig2.tight_layout()

    out_path2 = os.path.join(out_dir, f"{stem1}_ev_selected_blocks.png")
    fig2.savefig(out_path2, dpi=150)
    plt.close(fig2)

    print(f"    plots -> {os.path.basename(out_path1)}, {os.path.basename(out_path2)}")


def main():
    """
    End-to-end data generation pipeline.

    Steps
    -----
    1. For each of N_IMAGES base seeds, generate a clean image (implicitly
       embedded inside generate_rfi_image via generate_clean_image).
    2. For each pipeline config (4 combinations of power level x RFI type),
       inject RFI into every image and save it as an HDF5 file under:
           data/<power_level>_power/<rfi_type>/image_<seed>.h5
    3. Generate eigenvalue profile plots for each saved file and save as PNG
       in the same directory as the HDF5 file.

    Seeds are used directly as the image index (0 to N_IMAGES-1) so that
    each image is fully reproducible from its filename alone.
    """
    seeds = list(range(N_IMAGES))

    for cfg in _PIPELINE_CONFIGS:
        rfi_type    = cfg['rfi_type']
        jnr_range   = cfg['jnr_range']
        power_level = cfg['power_level']

        out_dir = os.path.join(DATA_ROOT, f"{power_level}_power", rfi_type)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n[{power_level}_power / {rfi_type}]  JNR range: {jnr_range} dB")

        for seed in seeds:
            out_path = os.path.join(out_dir, f"image_{seed}.h5")

            rfi_image, meta = generate_rfi_image(
                seed      = seed,
                jnr_range = jnr_range,
                rfi_type  = rfi_type,
            )

            divide_cpi_and_save(
                matrix      = rfi_image,
                meta        = meta,
                seed        = seed,
                jnr_range   = jnr_range,
                output_path = out_path,
            )

            print(f"  seed={seed:02d}  jnr range={jnr_range}  -> {out_path}")
            plot_eigenvalue_profiles(h5_path=out_path, out_dir=out_dir)

    print("\nDone.")


if __name__ == '__main__':
    main()