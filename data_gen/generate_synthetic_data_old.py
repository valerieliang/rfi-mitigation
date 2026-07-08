"""
generate_synthetic_data.py

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

Labeling convention:
  - knee = 0: CLEAN (no RFI contamination)
  - knee = 1-16: CONTAMINATED (knee indicates number of RFI signals present)

Training set structure:
  - data/multi_band/clean/: clean samples (knee=0)
  - data/multi_band/contaminated/: RFI-contaminated samples (knee=1-16)
  - Equal number of clean and contaminated samples across SNR levels

RFI block structure:
  - The pulse axis is divided into non-overlapping blocks of BLOCK_HEIGHT pulses.
  - TOTAL_PULSES must be divisible by BLOCK_HEIGHT.
  - Each block receives 1-6 independent RFI bands at randomly chosen positions
    within that block. Bands need not be adjacent; each has its own JNR drawn
    from JNR_RANGE_DB. Bands are mutually uncorrelated (independent RNG state
    per band). generate_rfi_image returns an RfiMeta object carrying per-block
    band descriptors (see RfiMeta for format details).

HDF5 layout:
  Root attributes  : file-level config (dimensions, seed, JNR range, is_clean flag, etc.)
  Dataset per CPI  : name "cpi_{i}_{j}", complex64 array of shape
                     (cpi_height, cpi_width).
  Dataset attributes: per-CPI RFI metadata serialised as JSON string under
                      the key 'rfi_bands'. Top-level keys:
                          pulse_positions (list[int]): 1-based pulse positions of RFI bands
                          knee (int): RFI count (0=clean, 1-16=contaminated)
                          jnr_db_list (list[int]): JNR in dB for each band
"""

import os
import json
import numpy as np
import h5py
from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

NOISE_DB = 3        # fixed noise power in dB
SNR_RANGE_DB = (6, 10)   # SNR range in dB: [6, 7, 8, 9, 10]

JNR_RANGE_DB = (10, 30)  # per-band JNR range, both ends inclusive

TOTAL_PULSES = 1600     # rows  (slow-time / azimuth)
RANGE_BINS   = 10000    # cols  (fast-time / range)

# CPI tile dimensions
BLOCK_HEIGHT = 16       # pulse rows per CPI block; TOTAL_PULSES must be divisible
BLOCK_WIDTH  = 250      # range cols per CPI tile

assert TOTAL_PULSES % BLOCK_HEIGHT == 0, (
    f"TOTAL_PULSES ({TOTAL_PULSES}) must be divisible by BLOCK_HEIGHT ({BLOCK_HEIGHT})"
)
N_BLOCKS = TOTAL_PULSES // BLOCK_HEIGHT

# Band count range per block (both ends inclusive)
MIN_BANDS = 1
MAX_BANDS = 6

# Child-seed offsets for SeedSequence-style isolation
NOISE_SEED  = 0
SIGNAL_SEED = 1
RFI_SEED    = 2
JNR_SEED    = 3     # separate offset so JNR draw does not perturb RFI state


# ---------------------------------------------------------------------------
# RFI METADATA CONTAINER
# ---------------------------------------------------------------------------

@dataclass
class BandMeta:
    """Descriptor for a single RFI band within one block."""
    local_idx : int   # pulse index within the block [0, BLOCK_HEIGHT)
    row       : int   # absolute pulse row in the full image
    jnr_db    : int   # JNR for this band in dB


@dataclass
class RfiMeta:
    """
    Carries all RFI injection metadata for one generated image.

    Attributes:
        bands_per_block (list[list[BandMeta]]): N_BLOCKS outer entries.
            Each inner list holds 1-6 BandMeta objects, one per injected band.
            Bands within a block are uncorrelated and may occupy any row.
            len(bands_per_block[b]) is the number of bands injected in block b.
    """
    bands_per_block : List[List[BandMeta]] = field(default_factory=list)


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

def generate_clean_image(seed=0, snr_db=6):
    """
    Generate a complex-valued clean image: spatially white noise + signal.

    Noise power : NOISE_DB dB
    Signal power: NOISE_DB + snr_db dB

    Args:
        seed (int): Base seed. Child seeds are seed+NOISE_SEED, seed+SIGNAL_SEED.
        snr_db (float): Signal-to-noise ratio in dB.

    Returns:
        clean_image (np.ndarray): Complex64 array of shape (TOTAL_PULSES, RANGE_BINS).
    """
    noise_power_linear  = 10.0 ** (NOISE_DB / 10.0)
    signal_power_linear = noise_power_linear * (10.0 ** (snr_db / 10.0))

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

def generate_rfi_image(seed=0, snr_db=6):
    """
    Generate a complex-valued image with additive multi-band RFI injected into
    a clean frame.

    For every block of BLOCK_HEIGHT pulses, 1-6 independent RFI bands are
    injected. Each band occupies a single randomly chosen pulse row within the
    block (bands need not be adjacent). Each band draws its own JNR
    independently from JNR_RANGE_DB. All bands are mutually uncorrelated
    (separate RNG streams).

    Args:
        seed (int): Base seed. JNR draws use seed+JNR_SEED; RFI uses
                    seed+RFI_SEED.
        snr_db (float): Signal-to-noise ratio in dB for this image.

    Returns:
        rfi_image (np.ndarray): Complex64 array of shape (TOTAL_PULSES, RANGE_BINS).
        meta      (RfiMeta)   : Per-block band descriptors.
    """
    clean_image = generate_clean_image(seed, snr_db)
    rfi_signal, meta = _generate_multi_band_rfi(seed)
    rfi_image = (clean_image + rfi_signal).astype(np.complex64)
    return rfi_image, meta


# ---------------------------------------------------------------------------
# RFI SIGNAL GENERATOR
# ---------------------------------------------------------------------------

def _generate_multi_band_rfi(seed):
    """
    Generate a multi-band RFI signal and inject it into every CPI block.

    For each block:
      1. Draw the number of bands uniformly from [MIN_BANDS, MAX_BANDS].
      2. For each band, pick a random local pulse index within [0, BLOCK_HEIGHT)
         (with replacement -- two bands may land on the same row and their
         contributions sum incoherently because they use independent range
         coefficient vectors and independent Doppler frequencies).
      3. Draw a per-band JNR from JNR_RANGE_DB (integer, both ends inclusive)
         using the shared JNR RNG.
      4. Generate the range coefficient vector from an independent RNG child
         stream so that bands are statistically uncorrelated.

    Uncorrelation guarantee:
      Each band within every block is generated from a fresh sub-stream
      derived from (seed, block_index, band_index) so that no two bands share
      any RNG state.

    Args:
        seed (int): Base seed; band RNGs use (seed+RFI_SEED, block, band) via
                    SeedSequence; JNR RNG uses seed+JNR_SEED.

    Returns:
        rfi_matrix (np.ndarray): Complex64, shape (TOTAL_PULSES, RANGE_BINS).
        meta       (RfiMeta)   : Per-block band descriptor lists.
    """
    noise_power_linear = 10.0 ** (NOISE_DB / 10.0)

    # Shared RNG for block-level structural draws (n_bands, local row positions).
    rng_struct = np.random.default_rng(seed + RFI_SEED)

    # Separate RNG for JNR so the JNR stream is decoupled from placement draws.
    rng_jnr = np.random.default_rng(seed + JNR_SEED)

    rfi_matrix      = np.zeros((TOTAL_PULSES, RANGE_BINS), dtype=np.complex64)
    bands_per_block : List[List[BandMeta]] = []

    for b in range(N_BLOCKS):
        block_start = b * BLOCK_HEIGHT

        # Number of bands for this block
        n_bands = int(rng_struct.integers(MIN_BANDS, MAX_BANDS + 1))

        # Local pulse indices within [0, BLOCK_HEIGHT) (sampled with replacement)
        local_indices = rng_struct.integers(0, BLOCK_HEIGHT, size=n_bands)

        block_bands : List[BandMeta] = []

        for band_idx in range(n_bands):
            local_idx = int(local_indices[band_idx])
            abs_row   = block_start + local_idx

            # Per-band JNR (independent draw from shared JNR RNG)
            jnr_db           = int(rng_jnr.integers(
                JNR_RANGE_DB[0], JNR_RANGE_DB[1] + 1
            ))
            rfi_power_linear = noise_power_linear * (10.0 ** (jnr_db / 10.0))
            sigma            = np.sqrt(rfi_power_linear / 2.0)

            # Derive a deterministic child seed from (seed, b, band_idx) so
            # that each band's range coefficients are fully reproducible and
            # uncorrelated regardless of how many bands other blocks have.
            child_seed = (seed + RFI_SEED) * 10_000 + b * MAX_BANDS + band_idx
            rng_band   = np.random.default_rng(child_seed)

            doppler_freq = rng_band.uniform(-0.5, 0.5)
            range_coeff  = (
                rng_band.standard_normal(RANGE_BINS)
                + 1j * rng_band.standard_normal(RANGE_BINS)
            ) * sigma

            phase = np.exp(1j * 2.0 * np.pi * doppler_freq * abs_row)
            rfi_matrix[abs_row, :] += (phase * range_coeff).astype(np.complex64)

            block_bands.append(BandMeta(local_idx=local_idx, row=abs_row,
                                        jnr_db=jnr_db))

        bands_per_block.append(block_bands)

    meta = RfiMeta(bands_per_block=bands_per_block)
    return rfi_matrix, meta


# ---------------------------------------------------------------------------
# DIVIDE CPI AND SAVE AS HDF5
# ---------------------------------------------------------------------------

def divide_cpi_and_save(
    matrix,
    meta,
    seed,
    snr_db,
    cpi_height=BLOCK_HEIGHT,
    cpi_width=BLOCK_WIDTH,
    output_path=None,
    is_clean=False,
):
    """
    Divide a complex-valued matrix into non-overlapping CPI tiles and save to HDF5.

    HDF5 layout
    -----------
    Root attributes (file-level config):
        total_pulses   (int)   : total rows in the source image
        range_bins     (int)   : total columns in the source image
        cpi_height     (int)   : pulse rows per CPI tile (== BLOCK_HEIGHT)
        cpi_width      (int)   : range columns per CPI tile
        block_height   (int)   : pulse rows per RFI injection block
        n_blocks       (int)   : number of RFI injection blocks
        noise_db       (float) : noise floor in dB
        snr_db         (float) : signal-to-noise ratio in dB for this file
        jnr_range_low  (int)   : lower bound of per-band JNR sampling range
        jnr_range_high (int)   : upper bound of per-band JNR sampling range
        min_bands      (int)   : minimum bands per block
        max_bands      (int)   : maximum bands per block
        seed           (int)   : base seed used for generation
        is_clean       (bool)  : True if this is a clean (no RFI) sample

    Dataset per CPI tile, name "cpi_{i}_{j}":
        Data: complex64 array of shape (cpi_height, cpi_width).
        Attribute 'rfi_bands' (str): JSON object with keys:
            pulse_positions (list[int]): 1-based pulse positions of RFI bands
            knee (int): number of RFI pulses (0 if no RFI)
            jnr_db_list (list[int]): JNR in dB for each band
        Additional datasets for eigenvalue analysis:
            "cpi_{i}_{j}_eigenvalues": sorted eigenvalues (largest to smallest)
            "cpi_{i}_{j}_diagonal": diagonal values of SCM

    Args:
        matrix     (np.ndarray): Complex64 array of shape (TOTAL_PULSES, RANGE_BINS).
        meta       (RfiMeta|None): RFI injection metadata from generate_rfi_image (None for clean).
        seed       (int)       : Base seed used to generate the image.
        snr_db     (float)     : SNR in dB for this image.
        cpi_height (int)       : Pulse rows per CPI tile. Must divide TOTAL_PULSES.
        cpi_width  (int)       : Range columns per CPI tile. Must divide RANGE_BINS.
        output_path(str|None)  : Path for the HDF5 file. No file is written if None.
        is_clean   (bool)      : True if this is a clean (no RFI) sample.
    """
    if output_path is None:
        return

    assert TOTAL_PULSES % cpi_height == 0, (
        f"cpi_height ({cpi_height}) must divide TOTAL_PULSES ({TOTAL_PULSES})"
    )
    assert RANGE_BINS % cpi_width == 0, (
        f"cpi_width ({cpi_width}) must divide RANGE_BINS ({RANGE_BINS})"
    )

    with h5py.File(output_path, 'w') as f:

        # --- Root attributes -----------------------------------------------
        f.attrs['total_pulses']    = TOTAL_PULSES
        f.attrs['range_bins']      = RANGE_BINS
        f.attrs['cpi_height']      = cpi_height
        f.attrs['cpi_width']       = cpi_width
        f.attrs['block_height']    = BLOCK_HEIGHT
        f.attrs['n_blocks']        = N_BLOCKS
        f.attrs['noise_db']        = NOISE_DB
        f.attrs['snr_db']          = snr_db
        f.attrs['jnr_range_low']   = JNR_RANGE_DB[0]
        f.attrs['jnr_range_high']  = JNR_RANGE_DB[1]
        f.attrs['min_bands']       = MIN_BANDS
        f.attrs['max_bands']       = MAX_BANDS
        f.attrs['seed']            = seed
        f.attrs['is_clean']        = is_clean

        # --- CPI datasets ---------------------------------------------------
        for i in range(0, TOTAL_PULSES, cpi_height):
            # For clean samples, use empty RFI metadata
            if is_clean:
                pulse_positions = []
                jnr_db_list = []
                knee = 0
            else:
                # Each tile's pulse band maps 1-to-1 to one injection block when
                # cpi_height == BLOCK_HEIGHT (the standard configuration).
                block_idx = i // BLOCK_HEIGHT
                bands     = meta.bands_per_block[block_idx]

                # Convert to 1-based pulse positions and extract JNR values
                pulse_positions = [b.local_idx + 1 for b in bands]  # 1-based indexing
                jnr_db_list = [b.jnr_db for b in bands]
                knee = len(bands)  # Number of RFI pulses, 0 if no RFI

            # Serialise to JSON with new format
            bands_json = json.dumps({
                'pulse_positions': pulse_positions,
                'knee': knee,
                'jnr_db_list': jnr_db_list,
            })

            for j in range(0, RANGE_BINS, cpi_width):
                cpi  = matrix[i:i + cpi_height, j:j + cpi_width]

                # Compute SCM: M * M^H / cpi_width
                M = cpi
                SCM = (M @ M.conj().T) / cpi_width

                # Extract diagonal and eigenvalues
                diagonal = np.diag(SCM).real
                eigvals = np.linalg.eigvalsh(SCM)
                eigvals_sorted = np.sort(eigvals)[::-1]  # Largest to smallest

                # Save CPI data
                dset = f.create_dataset(f"cpi_{i}_{j}", data=cpi)
                dset.attrs['rfi_bands'] = bands_json

                # Save eigenvalue analysis
                f.create_dataset(f"cpi_{i}_{j}_eigenvalues", data=eigvals_sorted)
                f.create_dataset(f"cpi_{i}_{j}_diagonal", data=diagonal)


# ---------------------------------------------------------------------------
# PIPELINE CONFIGURATION
# ---------------------------------------------------------------------------

N_IMAGES_PER_SNR = 10          # number of images per SNR level (for each type: clean & contaminated)
SNR_LEVELS = [6, 7, 8, 9, 10]  # SNR levels in dB
N_IMAGES_CLEAN = N_IMAGES_PER_SNR * len(SNR_LEVELS)  # total clean images
N_IMAGES_RFI = N_IMAGES_PER_SNR * len(SNR_LEVELS)    # total contaminated images
DATA_ROOT  = 'data' # root output directory


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def _compute_eigenvalues_normalized(cpi):
    """
    Compute sorted descending eigenvalues in dB scale from a complex CPI tile.

    Forms the sample covariance matrix SCM = M @ M^H / K where K is the
    number of range bins (columns), then returns eigenvalues sorted descending
    in absolute and relative dB scales, along with matrix health metrics.

    Args:
        cpi (np.ndarray): Complex array of shape (M, K) where M = BLOCK_HEIGHT.

    Returns:
        eigvals_abs_db (np.ndarray): Eigenvalues in absolute dB scale
        max_eigval_db (float): Largest eigenvalue in absolute dB
        min_eigval_db (float): Smallest eigenvalue in absolute dB
        condition_number (float): Ratio of max to min eigenvalue
        effective_rank (float): Normalized participation ratio
    """
    M, K  = cpi.shape
    SCM   = (cpi @ cpi.conj().T) / K            # (M, M) sample covariance
    eigvals = np.linalg.eigvalsh(SCM)            # ascending real eigenvalues
    eigvals = np.sort(eigvals)[::-1]             # descending

    max_eigval = eigvals[0]
    min_eigval = eigvals[-1]

    # Condition number: ratio of max to min eigenvalue
    condition_number = max_eigval / max(min_eigval, 1e-12)

    # Effective rank: normalized participation ratio
    # effective_rank = (sum of eigenvalues)^2 / (sum of eigenvalues^2)
    eigvals_normalized = eigvals / max(np.sum(eigvals), 1e-12)
    effective_rank = 1.0 / np.sum(eigvals_normalized ** 2)

    # Absolute dB scale
    eigvals_abs_db = 10.0 * np.log10(eigvals + 1e-12)
    max_eigval_db = 10.0 * np.log10(max(max_eigval, 1e-12))
    min_eigval_db = 10.0 * np.log10(max(min_eigval, 1e-12))

    return eigvals_abs_db, max_eigval_db, min_eigval_db, condition_number, effective_rank


def plot_eigenvalue_profiles(h5_path, out_dir):
    """
    Generate two eigenvalue profile plots for a single HDF5 file and save as PNG.

    Plot 1 -- All-blocks overlay (color-coded by RFI count or clean)
        All N_BLOCKS pulse-blocks from range tile j=0 are overlaid on one axes.
        Lines are drawn in a colormap keyed to the RFI count (knee) of that block.
        For clean samples, all blocks show knee=0.
        Y-axis is in dB relative to max eigenvalue. X-axis uses 1-based indexing.

    Plot 2 -- Grid of 10 individual block profiles
        10 evenly-spaced pulse-blocks (indices 0, 10, 20, ..., 90) from range
        tile j=0 are shown in a 2x5 subplot grid. Shows true largest eigenvalue,
        SNR, and RFI count (knee: 0=clean, 1-16=contaminated) in title.

    Args:
        h5_path (str): Path to the source HDF5 file.
        out_dir (str): Directory where the two PNG files are saved.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    jnr_min_global, jnr_max_global = JNR_RANGE_DB

    block_pulse_offsets = [b * BLOCK_HEIGHT for b in range(N_BLOCKS)]
    ev_index_1based = np.arange(1, BLOCK_HEIGHT + 1)  # 1-based indexing

    with h5py.File(h5_path, 'r') as f:
        snr_db = f.attrs['snr_db']
        is_clean = f.attrs.get('is_clean', False)
        all_profiles = []
        all_max_eigval_db = []
        all_min_eigval_db = []
        all_condition_numbers = []
        all_effective_ranks = []
        all_max_jnr  = []
        all_knees = []

        for pulse_offset in block_pulse_offsets:
            dset      = f[f"cpi_{pulse_offset}_0"]
            cpi       = dset[:]
            eigvals_abs_db, max_eigval_db, min_eigval_db, cond_num, eff_rank = _compute_eigenvalues_normalized(cpi)
            all_profiles.append(eigvals_abs_db)
            all_max_eigval_db.append(max_eigval_db)
            all_min_eigval_db.append(min_eigval_db)
            all_condition_numbers.append(cond_num)
            all_effective_ranks.append(eff_rank)

            payload  = json.loads(str(dset.attrs['rfi_bands']))
            if payload['knee'] > 0:
                max_jnr  = float(np.max(payload['jnr_db_list']))
                # Compute knee: number of distinct pulse positions
                n_distinct = len(set(payload['pulse_positions']))
                knee = n_distinct  # Knee index in 1-based indexing
            else:
                max_jnr = 0.0
                knee = 0  # No RFI (clean)
            all_max_jnr.append(max_jnr)
            all_knees.append(knee)

    # Determine global y-axis range for consistent scaling
    all_eigvals = np.concatenate(all_profiles)
    global_min_db = np.percentile(all_eigvals, 1)  # 1st percentile to avoid outliers
    global_max_db = np.percentile(all_eigvals, 99)  # 99th percentile
    y_margin = 5  # dB margin
    ylim = [global_min_db - y_margin, global_max_db + y_margin]

    # Compute average statistics
    avg_condition = np.mean(all_condition_numbers)
    avg_eff_rank = np.mean(all_effective_ranks)
    avg_max_jnr = np.mean([j for j in all_max_jnr if j > 0]) if any(j > 0 for j in all_max_jnr) else 0.0

    # Color mapping: use knee count (0=clean, 1-16=RFI contaminated)
    norm_knee = mcolors.Normalize(vmin=0, vmax=MAX_BANDS)
    cmap = cm.plasma

    # ------------------------------------------------------------------
    # Plot 1: all blocks overlaid, each line colored by RFI count (knee)
    # ------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    for eigvals_db, max_jnr, knee in zip(all_profiles, all_max_jnr, all_knees):
        ax1.plot(ev_index_1based, eigvals_db, color=cmap(norm_knee(knee)), alpha=0.5,
                 linewidth=0.9)
        # Mark knee position if RFI is present
        if knee > 0:
            ax1.plot(knee, eigvals_db[knee-1], 'rx', markersize=5, alpha=0.4)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm_knee)
    sm.set_array([])
    cbar = fig1.colorbar(sm, ax=ax1)
    cbar.set_label('RFI Count (knee: 0=clean, 1-16=contaminated)', fontsize=11)

    ax1.set_xlabel('Eigenvalue Index (1-based)', fontsize=11)
    ax1.set_ylabel('Eigenvalue (dB, absolute scale)', fontsize=11)
    ax1.set_ylim(ylim)

    # Descriptive title indicating clean vs contaminated
    sample_type = "CLEAN (No RFI)" if is_clean else f"CONTAMINATED (RFI: {MIN_BANDS}-{MAX_BANDS} bands, JNR {jnr_min_global}-{jnr_max_global} dB)"
    jnr_str = "" if is_clean else f" | Avg Max RFI={avg_max_jnr:.1f} dB"
    ax1.set_title(
        f'Eigenvalue Profiles -- All Blocks -- {sample_type}\n'
        f'SNR={snr_db:.1f} dB{jnr_str} | Avg Cond#={avg_condition:.1f} | Avg Eff Rank={avg_eff_rank:.1f}\n'
        f'{os.path.basename(h5_path)}',
        fontsize=10,
    )
    ax1.grid(True, linestyle='--', alpha=0.4)
    fig1.tight_layout()

    stem1     = os.path.splitext(os.path.basename(h5_path))[0]
    out_path1 = os.path.join(out_dir, f"{stem1}_ev_all_blocks.png")
    fig1.savefig(out_path1, dpi=150)
    plt.close(fig1)

    # ------------------------------------------------------------------
    # Plot 2: 2x5 grid of 10 evenly-spaced blocks
    # ------------------------------------------------------------------
    selected_block_indices = [b * (N_BLOCKS // 10) for b in range(10)]
    n_cols, n_rows = 5, 2
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(18, 7))

    for ax, block_idx in zip(axes.flat, selected_block_indices):
        eigvals_db   = all_profiles[block_idx]
        max_eigval_db = all_max_eigval_db[block_idx]
        min_eigval_db = all_min_eigval_db[block_idx]
        cond_num = all_condition_numbers[block_idx]
        eff_rank = all_effective_ranks[block_idx]
        max_jnr = all_max_jnr[block_idx]
        knee = all_knees[block_idx]

        ax.plot(ev_index_1based, eigvals_db, color=cmap(norm_knee(knee)), linewidth=1.4)

        # Mark knee position if RFI is present
        if knee > 0:
            ax.plot(knee, eigvals_db[knee-1], 'rx', markersize=8, markeredgewidth=2)
            ax.axvline(x=knee, color='red', linestyle='--', alpha=0.3, linewidth=1)

        # Clear labeling: knee 0=clean, 1-16=contaminated
        status_label = "CLEAN" if knee == 0 else f"RFI={knee}"
        jnr_str = "" if knee == 0 else f" | Max RFI={max_jnr:.0f} dB"
        ax.set_title(
            f'Block {block_idx} [{status_label}]{jnr_str}\n'
            f'Cond#={cond_num:.1f} | Eff Rank={eff_rank:.1f}\n'
            f'λ: [{min_eigval_db:.1f}, {max_eigval_db:.1f}] dB',
            fontsize=7
        )
        ax.set_xlabel('EV Index', fontsize=8)
        ax.set_ylabel('Eigenvalue (dB)', fontsize=8)
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle='--', alpha=0.4)

    # Clear suptitle indicating clean vs contaminated
    fig2.suptitle(
        f'Eigenvalue Profiles -- Selected Blocks -- {sample_type}\n'
        f'SNR={snr_db:.1f} dB  |  {os.path.basename(h5_path)}',
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
    1. Generate 50 images total:
       - 25 CLEAN images: 10 images per SNR level (6-10 dB)
       - 25 CONTAMINATED images: 10 images per SNR level (6-10 dB)
    2. For clean images:
       - Generate 16 signal pulses per CPI block
       - NO RFI injection
    3. For contaminated images:
       - Generate 16 signal pulses per CPI block
       - Generate 1-6 random RFI pulses per CPI block
       - Each RFI pulse has random JNR in range [10, 30] dB
    4. Save clean as HDF5 under data/multi_band/clean/image_<seed>_snr_<snr>.h5
    5. Save contaminated as HDF5 under data/multi_band/contaminated/image_<seed>_snr_<snr>.h5
    6. Generate eigenvalue profile plots for both types.

    All random generations (noise, signal, RFI placement, JNR) are uncorrelated.
    """
    base_dir = os.path.join(DATA_ROOT, 'multi_band_original')
    clean_dir = os.path.join(base_dir, 'clean')
    contaminated_dir = os.path.join(base_dir, 'contaminated')

    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(contaminated_dir, exist_ok=True)

    print(f"\n[multi_band_original]  Generating training data with SNR 6-10 dB")
    print(f"  CLEAN images: {N_IMAGES_CLEAN}")
    print(f"  CONTAMINATED images: {N_IMAGES_RFI}")
    print(f"  SNR range: {SNR_RANGE_DB[0]}-{SNR_RANGE_DB[1]} dB  ({N_IMAGES_PER_SNR} images per level per type)")
    print(f"  JNR range: {JNR_RANGE_DB} dB  bands per block: {MIN_BANDS}-{MAX_BANDS}")
    print(f"  Noise: {NOISE_DB} dB (fixed)")

    # Generate CLEAN samples
    print(f"\n[CLEAN SAMPLES]")
    seed = 0
    for snr_db in SNR_LEVELS:
        print(f"\n  Generating CLEAN SNR = {snr_db} dB:")
        for img_idx in range(N_IMAGES_PER_SNR):
            out_path = os.path.join(clean_dir, f"image_{seed}_snr_{snr_db}.h5")

            clean_image = generate_clean_image(seed=seed, snr_db=snr_db)

            divide_cpi_and_save(
                matrix      = clean_image,
                meta        = None,
                seed        = seed,
                snr_db      = snr_db,
                output_path = out_path,
                is_clean    = True,
            )

            print(f"    seed={seed:02d}  -> {os.path.basename(out_path)}")
            plot_eigenvalue_profiles(h5_path=out_path, out_dir=clean_dir)

            seed += 1

    # Generate CONTAMINATED samples
    print(f"\n[CONTAMINATED SAMPLES]")
    seed = 0
    for snr_db in SNR_LEVELS:
        print(f"\n  Generating CONTAMINATED SNR = {snr_db} dB:")
        for img_idx in range(N_IMAGES_PER_SNR):
            out_path = os.path.join(contaminated_dir, f"image_{seed}_snr_{snr_db}.h5")

            rfi_image, meta = generate_rfi_image(seed=seed, snr_db=snr_db)

            divide_cpi_and_save(
                matrix      = rfi_image,
                meta        = meta,
                seed        = seed,
                snr_db      = snr_db,
                output_path = out_path,
                is_clean    = False,
            )

            print(f"    seed={seed:02d}  -> {os.path.basename(out_path)}")
            plot_eigenvalue_profiles(h5_path=out_path, out_dir=contaminated_dir)

            seed += 1

    print("\nDone.")
    print(f"\nTotal samples generated:")
    print(f"  Clean: {N_IMAGES_CLEAN} in {clean_dir}")
    print(f"  Contaminated: {N_IMAGES_RFI} in {contaminated_dir}")


if __name__ == '__main__':
    main()