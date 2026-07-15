#!/usr/bin/env python
"""
select_clean_mountain.py

Scan a NISAR L0B scene for "clean" CPI tiles and save them to HDF5.

A tile is considered clean using an IQR-based method:
1. Compute gap-excluded SCM and identify valid diagonal entries
2. Remove invalid diagonal entries
3. Select middle 50% of remaining valid diagonals (IQR)
4. Compute mean and std dev of the IQR
5. Check if top n_check eigenvalues exceed mean + std_threshold * std_dev
6. Flag as unclean if any exceed the threshold

This approach uses signal noise statistics (IQR) free from drop-offs and RFI
to detect outliers, rather than an absolute threshold.

For each clean tile, we store:
  - Full eigenvalue profile (16 values, unnormalized)
  - Full diagonal (16 values, unnormalized)
  - IQR mean and std dev (statistics used for cleanliness check)
  - Number of std devs above mean for the max value checked
  - Pulse start and range start indices
  - Frequency and polarization

By default, the script processes both polarizations (if available) and
allows frequency selection via --freq.
"""

import argparse
import os

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

EPS = 1e-12

# Pulse chunk size for reading L0B data
PULSE_CHUNK_DEFAULT = 512


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


def process_freq_pol(data_block, mask_block, p0, r0, cpi_len, cpi_width,
                     off_diag_ratio, diag_ratio, n_check, std_threshold):
    """
    Tile data_block into non-overlapping cpi_len x cpi_width CPIs, extract
    eigenvalues and diagonal for each, and filter for clean tiles only.

    Returns a dict of arrays for clean tiles, or None if no clean tiles found.
    """
    n_p, n_r = data_block.shape
    n_pt = n_p // cpi_len
    n_rt = n_r // cpi_width

    eig_lin_list = []       # (16,) linear eigenvalues, descending, unnormalized
    diag_lin_list = []      # (16,) per-pulse SCM diagonal power (linear, unnormalized)
    diag_valid_list = []
    pulse_idx_list = []
    range_idx_list = []
    iqr_mean_list = []      # IQR mean for each CPI
    iqr_std_list = []       # IQR std dev for each CPI
    n_std_above_list = []   # Number of std devs above mean for max value

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

            # Normalize SCM by number of range samples (CPI^H * CPI / 250)
            scm = scm / cpi_width

            eigvals = eigen_decompose_descending(scm)      # (16,) linear, unnormalized
            diag_lin = np.real(np.diag(scm)).astype(np.float64)

            # DEBUG: Check for zero eigenvalues/diagonal
            if len(eig_lin_list) == 0:  # Only print for first clean tile
                print(f"    [DEBUG] First clean tile at p={ps}, r={rs}:")
                print(f"      SCM max: {np.max(np.abs(scm)):.6e}")
                print(f"      Eigvals: {eigvals[:5]}")
                print(f"      Diagonal: {diag_lin[:5]}")

            # Filter: only keep clean tiles using new IQR-based method
            is_clean, stats = is_clean_tile(
                diag_lin,
                diag_valid_idx,
                n_check=n_check,
                std_threshold=std_threshold
            )

            if not is_clean:
                continue

            eig_lin_list.append(eigvals.astype(np.float64))
            diag_lin_list.append(diag_lin)
            diag_valid_list.append(diag_valid_frac)
            pulse_idx_list.append(p0 + ps)
            range_idx_list.append(r0 + rs)
            iqr_mean_list.append(stats['iqr_mean'])
            iqr_std_list.append(stats['iqr_std'])
            n_std_above_list.append(stats['n_std_above'])

    if not eig_lin_list:
        return None

    eig_lin = np.stack(eig_lin_list)             # (N, 16)
    diag_lin = np.stack(diag_lin_list)           # (N, 16)

    # DEBUG: Check if stacked arrays are non-zero
    print(f"    [DEBUG] After stacking {len(eig_lin_list)} clean tiles:")
    print(f"      eig_lin max: {np.max(eig_lin):.6e}, all_zero: {np.all(eig_lin == 0)}")
    print(f"      diag_lin max: {np.max(diag_lin):.6e}, all_zero: {np.all(diag_lin == 0)}")

    return {
        "eig_lin": eig_lin,
        "diag_lin": diag_lin.astype(np.float64),
        "diag_valid_frac": np.array(diag_valid_list, dtype=np.float32),
        "pulse_idx": np.array(pulse_idx_list, dtype=np.int64),
        "range_idx": np.array(range_idx_list, dtype=np.int64),
        "iqr_mean": np.array(iqr_mean_list, dtype=np.float32),
        "iqr_std": np.array(iqr_std_list, dtype=np.float32),
        "n_std_above": np.array(n_std_above_list, dtype=np.float32),
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


def read_scene_block(ds, p0, p1, r0, r1, pulse_chunk):
    """Read and BFPQLUT-decode a [p0:p1, r0:r1] window in pulse chunks."""
    n_p = p1 - p0
    n_r = r1 - r0
    out = np.empty((n_p, n_r), dtype=np.complex64)
    for cs in range(p0, p1, pulse_chunk):
        ce = min(cs + pulse_chunk, p1)
        out[cs - p0:ce - p0, :] = np.asarray(ds[cs:ce, r0:r1], dtype=np.complex64)
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

    parser.add_argument('--output-h5', default='clean_mountains.h5',
                        help='Output HDF5 file path.')
    args = parser.parse_args()

    # Open L0B granule
    print("[+] Opening L0B granule: {}".format(args.l0b_file))
    raw = open_raw(args.l0b_file)

    # Resolve frequency/polarization pairs
    freq_pols = resolve_freq_pols(raw, args.freq, args.pol)
    print("[+] Processing {} frequency/polarization pair(s): {}".format(len(freq_pols), freq_pols))

    # Prepare output HDF5
    os.makedirs(os.path.dirname(args.output_h5) or '.', exist_ok=True)
    h5_out = h5py.File(args.output_h5, 'w')
    h5_out.attrs['l0b_file'] = os.path.basename(args.l0b_file)
    h5_out.attrs['pulse_start'] = args.pulse_start
    h5_out.attrs['pulse_end'] = args.pulse_end
    h5_out.attrs['range_start'] = args.range_start if args.range_start is not None else 0
    h5_out.attrs['range_end'] = args.range_end if args.range_end is not None else -1
    h5_out.attrs['n_check'] = args.n_check
    h5_out.attrs['std_threshold'] = args.std_threshold

    # Process each frequency/polarization pair
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

        # Read scene block
        print("    Reading pulse range [{}:{}], range [{}:{}] ...".format(
            args.pulse_start, args.pulse_end, r0, r1))
        data_block = read_scene_block(rds, args.pulse_start, args.pulse_end, r0, r1, args.pulse_chunk)
        print("    Data block shape: {}".format(data_block.shape))

        # Compute validity mask if requested
        mask_block = None
        if args.compute_subswath_mask:
            print("    Computing amplitude-based validity mask ...")
            mask_block = amplitude_gap_mask(data_block)
            valid_frac = np.mean(mask_block)
            print("    Valid fraction: {:.2%}".format(valid_frac))

        # Process tiles and filter for clean ones
        print("    Tiling and filtering for clean tiles (std_threshold={} std devs) ...".format(args.std_threshold))
        result = process_freq_pol(
            data_block, mask_block,
            args.pulse_start, r0,
            args.cpi_len, args.cpi_width,
            args.off_diag_overlap_ratio, args.diag_valid_ratio,
            args.n_check, args.std_threshold
        )

        if result is None:
            print("    [warn] No clean tiles found for freq={}, pol={}".format(freq, pol))
            continue

        n_clean = result['eig_lin'].shape[0]
        print("    Found {} clean tile(s)".format(n_clean))

        # Save to HDF5
        grp_name = "freq_{}_pol_{}".format(freq, pol)
        grp = h5_out.create_group(grp_name)
        grp.create_dataset('eigenvalues', data=result['eig_lin'], compression='gzip')
        grp.create_dataset('diagonal', data=result['diag_lin'], compression='gzip')
        grp.create_dataset('diag_valid_frac', data=result['diag_valid_frac'], compression='gzip')
        grp.create_dataset('pulse_idx', data=result['pulse_idx'], compression='gzip')
        grp.create_dataset('range_idx', data=result['range_idx'], compression='gzip')
        grp.create_dataset('iqr_mean', data=result['iqr_mean'], compression='gzip')
        grp.create_dataset('iqr_std', data=result['iqr_std'], compression='gzip')
        grp.create_dataset('n_std_above', data=result['n_std_above'], compression='gzip')
        grp.attrs['frequency'] = freq
        grp.attrs['polarization'] = pol
        grp.attrs['n_clean_tiles'] = n_clean

        print("    Saved to HDF5 group: {}".format(grp_name))

    h5_out.close()
    print("\n[+] Done. Clean tiles saved to: {}".format(args.output_h5))


if __name__ == '__main__':
    main()
