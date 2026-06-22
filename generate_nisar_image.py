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
        rfi_type (str): 'single_tone' or 'wideband'.
        jnr_db   (int): Jammer-to-noise ratio used for this image, in dB.

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
    rfi_type  : str
    jnr_db    : int
    rfi_rows  : List[int]             = field(default_factory=list)
    rfi_spans : List[Tuple[int, int]] = field(default_factory=list)


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

    Args:
        seed (int): Base seed. JNR draw uses seed+JNR_SEED; RFI uses seed+RFI_SEED.
        jnr_range (tuple): (low, high) JNR in dB, both ends inclusive.
        rfi_type (str): 'single_tone' or 'wideband'.

    Returns:
        rfi_image (np.ndarray): Complex64 array of shape (TOTAL_PULSES, RANGE_BINS).
        meta      (RfiMeta): JNR, type, and per-block injection positions.
    """
    clean_image = generate_clean_image(seed)

    rng_jnr = np.random.default_rng(seed + JNR_SEED)
    jnr_db = int(rng_jnr.integers(jnr_range[0], jnr_range[1] + 1))

    noise_power_linear = 10.0 ** (NOISE_DB / 10.0)
    rfi_power_linear   = noise_power_linear * (10.0 ** (jnr_db / 10.0))

    if rfi_type == 'single_tone':
        rfi_signal, rfi_rows = _generate_single_tone_rfi(seed, rfi_power_linear)
        meta = RfiMeta(rfi_type=rfi_type, jnr_db=jnr_db, rfi_rows=rfi_rows)
    elif rfi_type == 'wideband':
        rfi_signal, rfi_spans = _generate_wideband_rfi(seed, rfi_power_linear)
        meta = RfiMeta(rfi_type=rfi_type, jnr_db=jnr_db, rfi_spans=rfi_spans)
    else:
        raise ValueError(
            f"Unknown rfi_type '{rfi_type}'. Use 'single_tone' or 'wideband'."
        )

    rfi_image = (clean_image + rfi_signal).astype(np.complex64)
    return rfi_image, meta


# ---------------------------------------------------------------------------
# RFI SIGNAL GENERATORS
# ---------------------------------------------------------------------------

def _generate_single_tone_rfi(seed, rfi_power_linear):
    """
    Generate a narrowband (CW) RFI signal injected into every block.

    In each block of BLOCK_SIZE pulses, exactly one pulse row is chosen at
    random to carry the RFI tone. The same Doppler frequency and range
    coefficient vector are reused across all blocks (same physical emitter).

    Args:
        seed (int): Base seed; RFI RNG uses seed+RFI_SEED.
        rfi_power_linear (float): Desired per-active-row RFI power (linear).

    Returns:
        rfi_matrix (np.ndarray): Complex64, shape (TOTAL_PULSES, RANGE_BINS).
        rfi_rows   (list[int]): One absolute row index per block (N_BLOCKS entries).
    """
    rng = np.random.default_rng(seed + RFI_SEED)

    doppler_freq = rng.uniform(-0.5, 0.5)
    sigma        = np.sqrt(rfi_power_linear / 2.0)
    range_coeff  = (
        rng.standard_normal(RANGE_BINS) + 1j * rng.standard_normal(RANGE_BINS)
    ) * sigma

    rfi_matrix = np.zeros((TOTAL_PULSES, RANGE_BINS), dtype=np.complex64)
    rfi_rows   = []

    for b in range(N_BLOCKS):
        block_start = b * BLOCK_SIZE
        local_idx   = int(rng.integers(0, BLOCK_SIZE))
        abs_row     = block_start + local_idx
        phase       = np.exp(1j * 2.0 * np.pi * doppler_freq * abs_row)
        rfi_matrix[abs_row, :] = (phase * range_coeff).astype(np.complex64)
        rfi_rows.append(abs_row)

    # rfi_rows is already in block order (one entry per block, ascending)
    return rfi_matrix, rfi_rows


def _generate_wideband_rfi(seed, rfi_power_linear, width=4):
    """
    Generate a wideband RFI signal injected into every block.

    In each block of BLOCK_SIZE pulses, 'width' consecutive rows are chosen
    at a random start position such that all rows fit within the block.
    Each block gets independent Doppler frequencies and log-normal power jitter
    with the sum renormalised to rfi_power_linear.

    Args:
        seed (int): Base seed; RFI RNG uses seed+RFI_SEED.
        rfi_power_linear (float): Desired total RFI power per block (linear).
        width (int): Number of consecutive active rows per block (= SCM rank).

    Returns:
        rfi_matrix (np.ndarray): Complex64, shape (TOTAL_PULSES, RANGE_BINS).
        rfi_spans  (list[tuple[int,int]]): One (start, end) tuple per block
            (N_BLOCKS entries). Both indices are absolute and inclusive.
    """
    assert width <= BLOCK_SIZE, (
        f"width ({width}) must be <= BLOCK_SIZE ({BLOCK_SIZE})"
    )

    rng = np.random.default_rng(seed + RFI_SEED)

    rfi_matrix = np.zeros((TOTAL_PULSES, RANGE_BINS), dtype=np.complex64)
    rfi_spans  = []

    for b in range(N_BLOCKS):
        block_start     = b * BLOCK_SIZE
        max_local_start = BLOCK_SIZE - width
        local_start     = int(rng.integers(0, max_local_start + 1))
        abs_start       = block_start + local_start
        abs_end         = abs_start + width - 1   # inclusive
        active_rows     = list(range(abs_start, abs_end + 1))

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

    return rfi_matrix, rfi_spans


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
        f.attrs['jnr_db']         = meta.jnr_db
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

                elif meta.rfi_type == 'wideband':
                    start, end = meta.rfi_spans[block_idx]
                    dset.attrs['rfi_start'] = start
                    dset.attrs['rfi_end']   = end