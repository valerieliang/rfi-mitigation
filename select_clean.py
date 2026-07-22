#!/usr/bin/env python
"""
select_clean.py

Scan ANY NISAR L0B scene (mountain, rainforest, urban, ...) for "clean" CPI
tiles and write them directly as a CLEAN (label-0) training set that
train_only.py can load, with no separate filtering or generation step.

Nothing here is scene-specific: the cleanliness test is a per-tile signal
statistic, so the same command works on any granule. Point it at a scene, give
it a pulse window, and it emits that scene's clean background tiles.

Pipeline per tile:
1. Subtract the instrument caltone from the raw data (on by default), so
   cleanliness and the stored features are computed on caltone-free data.
2. Compute the gap-excluded SCM, its eigenvalues, its diagonal, and the
   per-index diagonal validity mask.
3. IQR cleanliness test: reject a tile if its top n_check eigenvalues exceed
   the IQR mean + std_threshold * std_dev of the valid diagonal.
4. Power / valid-eigenvalue filter (folded in from the old
   filter_clean_mountains.py): reject a tile whose max eigenvalue power is below
   min_power_db, or that has fewer than min_valid_eigvals eigenvalues above 0 dB.

Tiles that pass BOTH are written, one file per channel, in the exact layout
train_only.py consumes (<name-prefix>_<freq>_<pol>.h5, default clean_data_*):

    labels          int8    (N,)         all 0 (clean)
    eigenvalues     float32 (N, cpi_len) descending, LINEAR scale
    diagonal        float32 (N, cpi_len) SCM diagonal, LINEAR, unnormalized
    diag_valid_idx  bool    (N, cpi_len) per-index diagonal validity
    signal_power_db float32 (N,)         tile baseline power, 10*log10
    valid_fraction  float32 (N,)         fraction of valid samples
    tile_pulse      int32   (N,)         absolute pulse index of tile row 0
    tile_range      int32   (N,)         absolute range index of tile col 0
    (plus iqr_mean / iqr_std / n_std_above / diag_valid_frac / max_power_db /
     n_valid_eigvals as clean-selection provenance)

By default, the script processes both polarizations (if available) and
allows frequency selection via --freq.

Examples
--------
    # Clean tiles from a mountain scene
    python select_clean.py mountain.h5 --pulse-start 435777 --pulse-end 489617 \
        --compute-subswath-mask --output-dir data/mountain_clean

    # Clean tiles from an Amazon scene, tagged so the files are distinguishable
    python select_clean.py amazon.h5 --pulse-start 813924 --pulse-end 888222 \
        --compute-subswath-mask --name-prefix amazon_clean_data \
        --output-dir data/amazon_clean
"""

import argparse
import os
from datetime import datetime, timezone

import numpy as np
import h5py

# ---------------------------------------------------------------------------
# CONSTANTS (copied from anomaly_features.py to make this script standalone)
# ---------------------------------------------------------------------------

CPI_LEN_DEFAULT = 16
CPI_WIDTH_DEFAULT = 250

# Permissive gap-exclusion ratios: preserve as much dithered data as possible
OFF_DIAG_OVERLAP_RATIO_DEFAULT = 0.20
DIAG_VALID_RATIO_DEFAULT = 0.15

# Number of leading (largest) eigenvalues to keep as features.
N_KEEP_DEFAULT = 12

# Number of values to show in plots (all 16)
N_SHOW = 16

# Power / valid-eigenvalue filter defaults (folded in from filter_clean_mountains.py)
MIN_POWER_DB_DEFAULT = 4.0
MIN_VALID_EIGVALS_DEFAULT = 12

EPS = 1e-12

# Pulse chunk size for reading L0B data
PULSE_CHUNK_DEFAULT = 512


# ---------------------------------------------------------------------------
# CALTONE REMOVAL
# ---------------------------------------------------------------------------
#
# The instrument caltone is a narrowband sinusoid in fast time (range). Left in
# place it adds a rank-1 term to every slow-time CPI, lifting the eigenvalue /
# diagonal structure the IQR cleanliness test keys on -- i.e. it makes clean
# tiles look marginally less clean. Because this file defines the CLEAN BASELINE
# that the mountain training tiles are drawn from, the caltone is subtracted
# from the raw data BEFORE the SCM / diagonal / cleanliness are computed, so the
# baseline matches the caltone-removed generate_mountain_data.py path.
#
# ToneRemover builds an ABSOLUTE phase reference exp(-1j*2*pi*f*arange(n))
# anchored at range sample 0, so the tone must be removed from the FULL-WIDTH
# range line (aligned to sample 0); the range window is sliced out afterward.

CALTONE_WINDOW_SIZE = 64
# Fallback default matches isce3's caltone_frequency_from_raw.
CALTONE_DEFAULT_FREQ_HZ = 1214.883e6
CALTONE_LO_HZ = 1200e6
CALTONE_CLOCK_HZ = 240e6


def parse_caltone_freq_from_drt(raw, txrx_pol):
    """
    Local fallback for isce3's caltone_frequency_from_raw: caltone frequency
    (Hz) for one TxRx polarization from the DRT CALTONE phase-step telemetry,
    with CALTONE_DEFAULT_FREQ_HZ when the path is missing. Only used when the
    official helper cannot be imported.
    """
    path = f'{raw.TelemetryPath}/DRT/MISC/CP_IFSW_CALTONE_PHASE_STEP_{txrx_pol[1]}'
    with h5py.File(raw.filename, mode='r', swmr=True) as f:
        try:
            ds = f[path]
        except KeyError:
            print(f'    caltone: missing "{path}"; using default '
                  f'{CALTONE_DEFAULT_FREQ_HZ} Hz')
            return CALTONE_DEFAULT_FREQ_HZ
        i_cal = np.median(ds[()]).astype(int)
        return (i_cal / 2 ** 32) * CALTONE_CLOCK_HZ + CALTONE_LO_HZ


def build_tone_remover(raw, freq, pol, num_rng_samples):
    """
    ToneRemover sized to the full range width for one channel, plus the caltone
    frequency used. ToneRemover is imported lazily so --help works without isce3.
    The caltone frequency prefers isce3's official caltone_frequency_from_raw,
    falling back to the local parse_caltone_freq_from_drt if it cannot import.
    """
    from isce3.focus import ToneRemover
    try:
        from nisar.products.readers.Raw import caltone_frequency_from_raw
    except ImportError:
        caltone_frequency_from_raw = None

    tx_pol = pol[0]
    fc, fs, _, _ = raw.getChirpParameters(freq, tx_pol)
    if caltone_frequency_from_raw is not None:
        caltone_freq = caltone_frequency_from_raw(raw, pol)
    else:
        caltone_freq = parse_caltone_freq_from_drt(raw, pol)
    remover = ToneRemover((caltone_freq - fc) / fs, num_rng_samples,
                          CALTONE_WINDOW_SIZE)
    return remover, caltone_freq


# ---------------------------------------------------------------------------
# GAP-EXCLUSION SCM (standalone - copied from anomaly_features.py)
# ---------------------------------------------------------------------------

def compute_gap_exclusion_scm(
    data: np.ndarray,
    *,
    mask_valid_cpi: np.ndarray = None,
    off_diag_overlap_ratio: float = OFF_DIAG_OVERLAP_RATIO_DEFAULT,
    diag_valid_ratio: float = DIAG_VALID_RATIO_DEFAULT,
):
    """
    Compute a gap-excluded slow-time sample covariance matrix (SCM).

    Parameters
    ----------
    data : (num_pulses, num_rng_samples) complex array
        Slow-time CPI block: M pulses x K range samples.
    mask_valid_cpi : (num_pulses, num_rng_samples) bool array, optional
        True indicates a valid sample. If None, all samples are valid.
    off_diag_overlap_ratio : float, default 0.20
        Minimum fraction of overlapping valid range samples required to
        compute an off-diagonal SCM entry R_ij.
    diag_valid_ratio : float, default 0.15
        Minimum fraction of valid samples required to compute a diagonal
        SCM entry R_ii.

    Returns
    -------
    scm : (num_pulses, num_pulses) complex64
        Gap-excluded sample covariance matrix.
    diag_valid_idx : (num_pulses,) bool array
        True where the diagonal term had enough valid samples to be trusted.
    diag_valid_frac : float
        Fraction of diagonal entries that were valid (0 to 1).
    """
    num_pulses, num_rng_samples = data.shape

    if mask_valid_cpi is None:
        mask_valid_cpi = np.ones(data.shape, dtype=bool)
    else:
        mask_valid_cpi = mask_valid_cpi.astype(bool, copy=False)

    if mask_valid_cpi.shape != data.shape:
        raise ValueError(f"CPI mask shape {mask_valid_cpi.shape} != CPI data shape {data.shape}")

    if not (0.0 < off_diag_overlap_ratio <= 1.0):
        raise ValueError("off_diag_overlap_ratio must be between 0 and 1.")
    if not (0.0 < diag_valid_ratio <= 1.0):
        raise ValueError("diag_valid_ratio must be between 0 and 1.")

    min_valid_off_diag = max(1, int(np.ceil(off_diag_overlap_ratio * num_rng_samples)))
    min_valid_diag = max(1, int(np.ceil(diag_valid_ratio * num_rng_samples)))

    # Zero-out invalid samples
    x_valid = data * mask_valid_cpi

    # Overlap counts per SCM entry
    mask_int = mask_valid_cpi.astype(np.int32)
    overlap_counts = mask_int @ mask_int.T

    # Unnormalized conjugate products
    scm_sum = x_valid @ x_valid.conj().T

    scm = np.zeros((num_pulses, num_pulses), dtype=np.complex64)

    diag_idx = np.diag_indices(num_pulses)
    diag_counts = overlap_counts[diag_idx]
    diag_sum = scm_sum[diag_idx]

    diag_valid_idx = diag_counts >= min_valid_diag
    diag_vals = np.zeros(num_pulses, dtype=np.complex64)
    diag_vals[diag_valid_idx] = diag_sum[diag_valid_idx] / diag_counts[diag_valid_idx]
    scm[diag_idx] = diag_vals

    off_diag_valid = overlap_counts >= min_valid_off_diag
    np.fill_diagonal(off_diag_valid, False)
    scm[off_diag_valid] = scm_sum[off_diag_valid] / overlap_counts[off_diag_valid]

    # Ensure Hermitian numerically
    scm = (0.5 * (scm + scm.conj().T)).astype(np.complex64)

    diag_valid_frac = float(np.mean(diag_valid_idx))

    return scm, diag_valid_idx, diag_valid_frac


def eigen_decompose_descending(scm: np.ndarray) -> np.ndarray:
    """
    Eigenvalue decomposition of a Hermitian SCM, returned in descending order.

    Parameters
    ----------
    scm : (M, M) complex array

    Returns
    -------
    eigvals : (M,) float64 array, descending order
    """
    eigvals = np.linalg.eigvalsh(scm)
    return np.sort(np.real(eigvals))[::-1]


def eigvals_to_db(eigvals: np.ndarray) -> np.ndarray:
    """
    Convert eigenvalues to dB (un-normalized power in dB).

    Parameters
    ----------
    eigvals : (M,) float array, linear scale

    Returns
    -------
    eigvals_db : (M,) float32 array
    """
    eigvals_db = 10.0 * np.log10(np.clip(eigvals, EPS, None))
    return eigvals_db.astype(np.float32)


# ---------------------------------------------------------------------------
# CLEAN TILE SELECTION
# ---------------------------------------------------------------------------

def is_clean_tile(
    diag_lin: np.ndarray,
    diag_valid_idx: np.ndarray,
    n_check: int = 1,
    std_threshold: float = 1.0
) -> tuple[bool, dict]:
    """
    Determine if a CPI tile is "clean" based on SCM diagonal statistics using IQR method.

    Steps:
    1. Remove invalid diagonal entries (from diag_valid_idx)
    2. Select middle 50% of remaining valid diagonals (IQR)
    3. Compute mean and std dev of the middle 50%
    4. Check if the top n_check eigenvalues exceed mean + std_threshold * std_dev
    5. If any exceed, flag as unclean (RFI contamination)

    Parameters
    ----------
    diag_lin : (M,) float array
        SCM diagonal in linear scale (unnormalized).
    diag_valid_idx : (M,) bool array
        True where diagonal entry had enough valid samples.
    n_check : int, default 1
        Number of top eigenvalues to check against IQR statistics.
    std_threshold : float, default 1.0
        Number of standard deviations above the IQR mean to allow.

    Returns
    -------
    is_clean : bool
        True if tile is clean (no outliers detected).
    stats : dict
        Dictionary containing:
        - 'iqr_mean': mean of middle 50% (linear scale)
        - 'iqr_std': std dev of middle 50% (linear scale)
        - 'max_value': maximum value checked
        - 'n_std_above': number of std devs the max is above mean
    """
    # Step 1: Filter out invalid diagonals
    valid_diag = diag_lin[diag_valid_idx]

    if len(valid_diag) < 4:
        # Not enough valid samples to compute IQR
        return False, {
            'iqr_mean': 0.0,
            'iqr_std': 0.0,
            'max_value': 0.0,
            'n_std_above': np.inf
        }

    # Step 2: Select middle 50% (IQR)
    n_valid = len(valid_diag)
    q1_idx = n_valid // 4
    q3_idx = 3 * n_valid // 4

    # Sort to get IQR
    sorted_diag = np.sort(valid_diag)
    iqr_diag = sorted_diag[q1_idx:q3_idx]

    if len(iqr_diag) == 0:
        return False, {
            'iqr_mean': 0.0,
            'iqr_std': 0.0,
            'max_value': 0.0,
            'n_std_above': np.inf
        }

    # Step 3: Compute mean and std dev of IQR
    iqr_mean = np.mean(iqr_diag)
    iqr_std = np.std(iqr_diag, ddof=1) if len(iqr_diag) > 1 else 0.0

    # Step 4: Check top n_check values against threshold
    # Top values are at the beginning (descending order from SCM diag)
    n_check_actual = min(n_check, len(valid_diag))
    top_values = np.sort(valid_diag)[-n_check_actual:]  # Get largest values

    max_value = np.max(top_values)

    # Compute how many std devs above mean
    if iqr_std > EPS:
        n_std_above = (max_value - iqr_mean) / iqr_std
    else:
        # If std is zero, any value above mean is considered an outlier
        n_std_above = np.inf if max_value > iqr_mean else 0.0

    # Step 5: Flag as clean if within threshold
    is_clean = n_std_above <= std_threshold

    stats = {
        'iqr_mean': float(iqr_mean),
        'iqr_std': float(iqr_std),
        'max_value': float(max_value),
        'n_std_above': float(n_std_above)
    }

    return is_clean, stats


def tile_signal_power(cpi, cpi_mask):
    """Baseline power of a tile: mean(|x|^2) over its valid samples (EPS-floored)."""
    if cpi_mask is not None and cpi_mask.any():
        vals = cpi[cpi_mask]
    else:
        vals = cpi.ravel()
    return max(float(np.mean(np.abs(vals) ** 2)), EPS)


def process_freq_pol(data_block, mask_block, p0, r0, cpi_len, cpi_width,
                     off_diag_ratio, diag_ratio, n_check, std_threshold,
                     min_power_db, min_valid_eigvals):
    """
    Tile data_block into non-overlapping cpi_len x cpi_width CPIs and keep the
    tiles that pass BOTH the IQR cleanliness test and the power / valid-eigenvalue
    filter, returning per-tile arrays in the layout train_only.py consumes.

    A tile is kept when:
      1. is_clean_tile(...) finds no diagonal outlier above the IQR threshold,
      2. its max eigenvalue power is >= min_power_db, and
      3. at least min_valid_eigvals eigenvalues sit above 0 dB.

    Returns a dict of arrays for the kept (clean, label-0) tiles, or None.
    """
    n_p, n_r = data_block.shape
    n_pt = n_p // cpi_len
    n_rt = n_r // cpi_width

    eig_lin_list = []          # (16,) linear eigenvalues, descending
    diag_lin_list = []         # (16,) SCM diagonal power (linear, unnormalized)
    diag_valid_idx_list = []   # (16,) per-index bool -- required by train_only.py
    diag_valid_frac_list = []  # scalar fraction, kept as provenance
    sig_db_list = []           # tile baseline power, 10*log10
    vfrac_list = []            # valid-sample fraction
    pulse_idx_list = []
    range_idx_list = []
    iqr_mean_list = []
    iqr_std_list = []
    n_std_above_list = []

    rejection_counts = {'zero': 0, 'unclean': 0}

    for pt in range(n_pt):
        ps = pt * cpi_len
        pe = ps + cpi_len
        for rt in range(n_rt):
            rs = rt * cpi_width
            re = rs + cpi_width

            cpi = data_block[ps:pe, rs:re]
            cpi_mask = None if mask_block is None else mask_block[ps:pe, rs:re]

            scm, diag_valid_idx, diag_valid_frac = compute_gap_exclusion_scm(
                cpi,
                mask_valid_cpi=cpi_mask,
                off_diag_overlap_ratio=off_diag_ratio,
                diag_valid_ratio=diag_ratio,
            )

            # Normalize SCM by range width (CPI^H CPI / cpi_width), matching
            # generate_mountain_data.py / generate_amazon_data.py.
            scm = scm / cpi_width

            eigvals = eigen_decompose_descending(scm)          # (16,) linear
            diag_lin = np.real(np.diag(scm)).astype(np.float64)

            # Skip zero tiles (invalid / near-range data)
            if np.max(np.abs(eigvals)) < EPS:
                rejection_counts['zero'] += 1
                continue

            is_clean, stats = is_clean_tile(
                diag_lin, diag_valid_idx,
                n_check=n_check, std_threshold=std_threshold,
            )
            if not is_clean:
                rejection_counts['unclean'] += 1
                continue

            eig_lin_list.append(eigvals.astype(np.float64))
            diag_lin_list.append(diag_lin)
            diag_valid_idx_list.append(diag_valid_idx.astype(bool))
            diag_valid_frac_list.append(diag_valid_frac)
            sig_db_list.append(10.0 * np.log10(tile_signal_power(cpi, cpi_mask)))
            vfrac_list.append(float(cpi_mask.mean()) if cpi_mask is not None else 1.0)
            pulse_idx_list.append(p0 + ps)
            range_idx_list.append(r0 + rs)
            iqr_mean_list.append(stats['iqr_mean'])
            iqr_std_list.append(stats['iqr_std'])
            n_std_above_list.append(stats['n_std_above'])

    total_tiles = n_pt * n_rt
    n_iqr_clean = len(eig_lin_list)
    print(f"    IQR clean test: {total_tiles} tiles -> {n_iqr_clean} clean "
          f"(rejected zero={rejection_counts['zero']}, unclean={rejection_counts['unclean']})")

    if not eig_lin_list:
        return None

    eig_lin = np.stack(eig_lin_list)                    # (N, 16)
    diag_lin = np.stack(diag_lin_list)                  # (N, 16)
    diag_valid_idx = np.stack(diag_valid_idx_list)      # (N, 16) bool
    diag_valid_frac = np.array(diag_valid_frac_list, dtype=np.float32)
    sig_db = np.array(sig_db_list, dtype=np.float32)
    vfrac = np.array(vfrac_list, dtype=np.float32)
    pulse_idx = np.array(pulse_idx_list, dtype=np.int64)
    range_idx = np.array(range_idx_list, dtype=np.int64)
    iqr_mean = np.array(iqr_mean_list, dtype=np.float32)
    iqr_std = np.array(iqr_std_list, dtype=np.float32)
    n_std_above = np.array(n_std_above_list, dtype=np.float32)

    # Power / valid-eigenvalue filter (folded in from filter_clean_mountains.py):
    # drop tiles whose max eigenvalue power is too low or that have too few
    # eigenvalues above the 0 dB noise floor.
    eig_db = eigvals_to_db(eig_lin)                     # (N, 16)
    max_power_db = np.max(eig_db, axis=1)
    n_valid_eigvals = np.sum(eig_db > 0, axis=1)
    keep = (max_power_db >= min_power_db) & (n_valid_eigvals >= min_valid_eigvals)

    print(f"    power/eigval filter (max >= {min_power_db} dB, >= {min_valid_eigvals} valid): "
          f"{n_iqr_clean} -> {int(np.sum(keep))} kept ({int(np.sum(~keep))} removed)")

    if not np.any(keep):
        return None

    return {
        "eig_lin": eig_lin[keep].astype(np.float32),
        "diag_lin": diag_lin[keep].astype(np.float32),
        "diag_valid_idx": diag_valid_idx[keep],
        "diag_valid_frac": diag_valid_frac[keep],
        "signal_power_db": sig_db[keep],
        "valid_fraction": vfrac[keep],
        "pulse_idx": pulse_idx[keep],
        "range_idx": range_idx[keep],
        "iqr_mean": iqr_mean[keep],
        "iqr_std": iqr_std[keep],
        "n_std_above": n_std_above[keep],
        "max_power_db": max_power_db[keep].astype(np.float32),
        "n_valid_eigvals": n_valid_eigvals[keep].astype(np.int32),
    }


# ---------------------------------------------------------------------------
# L0B READING (integration point)
# ---------------------------------------------------------------------------

def open_raw(l0b_file):
    """Open a NISAR L0B granule with the ISCE3 / nisar Raw reader."""
    try:
        from nisar.products.readers.Raw import Raw
    except Exception as exc:
        raise RuntimeError(
            "Could not import the nisar Raw reader. Align open_raw() with the "
            "L0B reader used by read_nisar_swaths_isce3.py in this project. "
            "Underlying import error: {}".format(exc)
        )
    return Raw(hdf5file=l0b_file)


def resolve_freq_pols(raw, freq_arg, pol_arg):
    """Return the list of (freq, pol) pairs to process."""
    try:
        pol_map = dict(raw.polarizations)
    except Exception as exc:
        raise RuntimeError(
            "Could not read frequency/polarization map from the granule: {}".format(exc)
        )

    freqs = [freq_arg] if freq_arg is not None else sorted(pol_map.keys())
    pairs = []
    for freq in freqs:
        if freq not in pol_map:
            print("[warn] frequency {} not in granule; skipping".format(freq))
            continue
        pols = list(pol_map[freq])
        if pol_arg is not None:
            if pol_arg in pols:
                pols = [pol_arg]
            else:
                print("[warn] pol {} not in frequency {}; skipping".format(pol_arg, freq))
                continue
        for pol in pols:
            pairs.append((freq, pol))
    if not pairs:
        raise RuntimeError("No matching frequency/polarization to process.")
    return pairs


def read_scene_block(ds, p0, p1, r0, r1, pulse_chunk, remover=None):
    """
    Read and BFPQLUT-decode a [p0:p1, r0:r1] window in pulse chunks.

    When `remover` is given, the caltone is subtracted from each FULL-WIDTH
    range line before the [r0:r1] window is sliced out, so the tone is removed
    at its true absolute range phase (the remover is anchored at sample 0).
    """
    n_p = p1 - p0
    n_r = r1 - r0
    out = np.empty((n_p, n_r), dtype=np.complex64)
    for cs in range(p0, p1, pulse_chunk):
        ce = min(cs + pulse_chunk, p1)
        if remover is None:
            out[cs - p0:ce - p0, :] = np.asarray(ds[cs:ce, r0:r1], dtype=np.complex64)
        else:
            lines = np.asarray(ds[cs:ce, :], dtype=np.complex64)   # full width
            for ip in range(lines.shape[0]):
                lines[ip] = remover.remove_tone(lines[ip])
            out[cs - p0:ce - p0, :] = lines[:, r0:r1]
    return out


# ---------------------------------------------------------------------------
# VALIDITY MASK
# ---------------------------------------------------------------------------

def amplitude_gap_mask(data_block, gap_frac=0.10):
    """
    Self-contained validity mask based on ADC fill level.

    Inter-subswath transmission gaps show up as near-zero (not exactly zero)
    ADC fill. Mark range samples whose mean magnitude across pulses falls below
    gap_frac of the peak mean magnitude as invalid. Returns a bool mask the same
    shape as data_block (True = valid).
    """
    mean_mag = np.mean(np.abs(data_block), axis=0)
    peak = np.max(mean_mag)
    if peak < EPS:
        return np.ones(data_block.shape, dtype=bool)
    threshold = gap_frac * peak
    valid_rng = mean_mag >= threshold
    valid_mask = np.tile(valid_rng, (data_block.shape[0], 1))
    return valid_mask


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('l0b_file', help='Path to NISAR L0B HDF5 granule.')
    parser.add_argument('--freq', default=None,
                        help='Frequency band (e.g., "A" or "B"). Default: process all.')
    parser.add_argument('--pol', default=None,
                        help='Polarization (e.g., "HH", "VV"). Default: process all available.')

    parser.add_argument('--pulse-start', type=int, required=True)
    parser.add_argument('--pulse-end', type=int, required=True)
    parser.add_argument('--range-start', type=int, default=None,
                        help='Default: 0 (start of the swath).')
    parser.add_argument('--range-end', type=int, default=None,
                        help='Default: full range extent.')

    parser.add_argument('--cpi-len', type=int, default=CPI_LEN_DEFAULT)
    parser.add_argument('--cpi-width', type=int, default=CPI_WIDTH_DEFAULT)
    parser.add_argument('--pulse-chunk', type=int, default=PULSE_CHUNK_DEFAULT)

    parser.add_argument('--remove-caltone', dest='remove_caltone',
                        action='store_true', default=True,
                        help='Subtract the instrument caltone from the raw data '
                             'before computing cleanliness. This defines the clean '
                             'baseline for the scene (default: on).')
    parser.add_argument('--no-remove-caltone', dest='remove_caltone',
                        action='store_false',
                        help='Compute cleanliness on the raw data with the caltone '
                             'still in it (legacy behavior).')

    parser.add_argument('--compute-subswath-mask', action='store_true',
                        help='Apply gap-exclusion via amplitude-based subswath masking.')
    parser.add_argument('--off-diag-overlap-ratio', type=float,
                        default=OFF_DIAG_OVERLAP_RATIO_DEFAULT)
    parser.add_argument('--diag-valid-ratio', type=float,
                        default=DIAG_VALID_RATIO_DEFAULT)

    parser.add_argument('--n-check', type=int, default=1,
                        help='Number of top eigenvalues to check against IQR statistics.')
    parser.add_argument('--std-threshold', type=float, default=1.0,
                        help='Number of standard deviations above IQR mean to allow before flagging as unclean.')

    parser.add_argument('--min-power-db', type=float, default=MIN_POWER_DB_DEFAULT,
                        help='Drop tiles whose max eigenvalue power is below this (dB).')
    parser.add_argument('--min-valid-eigvals', type=int, default=MIN_VALID_EIGVALS_DEFAULT,
                        help='Drop tiles with fewer than this many eigenvalues above 0 dB.')

    parser.add_argument('--name-prefix', default='clean_data',
                        help='Filename prefix for the per-channel output files: '
                             '<name-prefix>_<freq>_<pol>.h5. Use a scene-specific '
                             'prefix (e.g. mountain_clean_data) to keep sets apart.')
    parser.add_argument('--output-dir', default='data/clean',
                        help='Output directory. One train-ready file per channel is '
                             'written here: <name-prefix>_<freq>_<pol>.h5')
    args = parser.parse_args()

    # Open L0B granule
    print("[+] Opening L0B granule: {}".format(args.l0b_file))
    raw = open_raw(args.l0b_file)

    # Resolve frequency/polarization pairs
    freq_pols = resolve_freq_pols(raw, args.freq, args.pol)
    print("[+] Processing {} frequency/polarization pair(s): {}".format(len(freq_pols), freq_pols))

    os.makedirs(args.output_dir, exist_ok=True)

    written = []
    group_totals = {}
    for freq, pol in freq_pols:
        print("\n[+] Processing freq={}, pol={}".format(freq, pol))

        # Get raw dataset
        try:
            rds = raw.getRawDataset(freq, pol)
        except Exception as exc:
            print("[error] Could not get raw dataset for freq={}, pol={}: {}".format(freq, pol, exc))
            continue

        # Determine range window
        r0 = args.range_start if args.range_start is not None else 0
        r1 = args.range_end if args.range_end is not None else rds.shape[1]

        # Build the caltone remover once per channel, sized to the FULL range
        # width so remove_tone() sees each range line at its true sample offset.
        if args.remove_caltone:
            remover, caltone_freq = build_tone_remover(raw, freq, pol, rds.shape[1])
            print("    caltone removal ON  (f_caltone = {:.4f} MHz, window = {})".format(
                caltone_freq / 1e6, CALTONE_WINDOW_SIZE))
        else:
            remover, caltone_freq = None, None
            print("    caltone removal OFF")

        # Read scene block
        print("    Reading pulse range [{}:{}], range [{}:{}] ...".format(
            args.pulse_start, args.pulse_end, r0, r1))
        data_block = read_scene_block(rds, args.pulse_start, args.pulse_end, r0, r1,
                                      args.pulse_chunk, remover)
        print("    Data block shape: {}".format(data_block.shape))

        # Compute validity mask if requested
        mask_block = None
        if args.compute_subswath_mask:
            mask_block = amplitude_gap_mask(data_block)
            print("    amplitude validity mask: {:.2%} valid".format(np.mean(mask_block)))

        # Tile, apply the IQR cleanliness test AND the power/eigval filter
        result = process_freq_pol(
            data_block, mask_block,
            args.pulse_start, r0,
            args.cpi_len, args.cpi_width,
            args.off_diag_overlap_ratio, args.diag_valid_ratio,
            args.n_check, args.std_threshold,
            args.min_power_db, args.min_valid_eigvals,
        )

        if result is None:
            print("    [warn] No clean tiles survived for freq={}, pol={}".format(freq, pol))
            continue

        n_clean = result['eig_lin'].shape[0]
        print("    Kept {} clean tile(s)".format(n_clean))

        # Write one train-ready file per channel (label 0, no RFI).
        out_path = os.path.join(args.output_dir,
                                "{}_{}_{}.h5".format(args.name_prefix, freq, pol))
        with h5py.File(out_path, 'w') as f:
            f.attrs['granule'] = os.path.basename(args.l0b_file)
            f.attrs['granule_path'] = args.l0b_file
            f.attrs['frequency'] = freq
            f.attrs['polarization'] = pol
            f.attrs['pulse_start'] = args.pulse_start
            f.attrs['pulse_end'] = args.pulse_end
            f.attrs['range_start'] = r0
            f.attrs['range_end'] = r1
            f.attrs['n_tiles'] = n_clean
            f.attrs['n_records'] = n_clean
            f.attrs['cpi_len'] = args.cpi_len
            f.attrs['cpi_width'] = args.cpi_width
            # Clean-only: label space is the single class {0}. train_only.py takes
            # the max n_classes across loaded files, so this stays compatible with
            # any RFI set that declares a larger n_classes (e.g. 7).
            f.attrs['min_bands'] = 0
            f.attrs['max_bands'] = 0
            f.attrs['n_classes'] = 1
            f.attrs['seed'] = 0
            f.attrs['content'] = 'clean scene background, no RFI injected (all label 0)'
            f.attrs['gap_exclusion_used'] = bool(args.compute_subswath_mask)
            f.attrs['off_diag_overlap_ratio'] = args.off_diag_overlap_ratio
            f.attrs['diag_valid_ratio'] = args.diag_valid_ratio
            f.attrs['n_check'] = args.n_check
            f.attrs['std_threshold'] = args.std_threshold
            f.attrs['min_power_db'] = args.min_power_db
            f.attrs['min_valid_eigvals'] = args.min_valid_eigvals
            f.attrs['eigenvalue_scale'] = 'linear, descending'
            f.attrs['diagonal_scale'] = 'linear, unnormalized'
            f.attrs['caltone_removed'] = bool(args.remove_caltone)
            if caltone_freq is not None:
                f.attrs['caltone_freq_hz'] = float(caltone_freq)
                f.attrs['caltone_window_size'] = CALTONE_WINDOW_SIZE
            f.attrs['generated_utc'] = datetime.now(timezone.utc).isoformat()

            # train_only.py required datasets
            f.create_dataset('labels', data=np.zeros(n_clean, dtype=np.int8))
            f.create_dataset('eigenvalues', data=result['eig_lin'], compression='gzip')
            f.create_dataset('diagonal', data=result['diag_lin'], compression='gzip')
            f.create_dataset('diag_valid_idx', data=result['diag_valid_idx'], compression='gzip')
            f.create_dataset('tile_pulse', data=result['pulse_idx'].astype(np.int32))
            f.create_dataset('tile_range', data=result['range_idx'].astype(np.int32))
            f.create_dataset('signal_power_db', data=result['signal_power_db'])
            f.create_dataset('valid_fraction', data=result['valid_fraction'])
            # Clean-selection provenance (ignored by train_only.py)
            f.create_dataset('diag_valid_frac', data=result['diag_valid_frac'], compression='gzip')
            f.create_dataset('iqr_mean', data=result['iqr_mean'], compression='gzip')
            f.create_dataset('iqr_std', data=result['iqr_std'], compression='gzip')
            f.create_dataset('n_std_above', data=result['n_std_above'], compression='gzip')
            f.create_dataset('max_power_db', data=result['max_power_db'], compression='gzip')
            f.create_dataset('n_valid_eigvals', data=result['n_valid_eigvals'], compression='gzip')

        written.append(out_path)
        group_totals[f'{freq}-{pol}'] = n_clean
        print("    Saved train-ready clean file: {}".format(out_path))

    if not written:
        raise RuntimeError("No clean tiles were written for any channel.")

    print("\n" + "=" * 60)
    print("[+] Done. Clean training files (all label 0):")
    for chan, tot in group_totals.items():
        print("    {:<8}: {} clean tiles".format(chan, tot))
    print("    total   : {} clean tiles".format(sum(group_totals.values())))
    for path in written:
        print("    -> {}".format(path))


if __name__ == '__main__':
    main()
