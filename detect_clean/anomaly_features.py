"""
anomaly_features.py

Feature extraction for per-CPI anomaly detection (clean vs RFI-contaminated),
robust to dithered / gappy L0B data.

Standard CPI dimensions: 16 pulses x 250 range samples (M=16, K=250).

Design decisions (per project requirements)
--------------------------------------------
1. Gap-exclusion SCM uses very permissive validity ratios (3% off-diagonal
   overlap, 2% diagonal valid) so dithered CPIs are preserved rather than
   dropped. This trades some SCM estimate quality for data yield; the
   feature normalization below is what makes the trade-off usable.

2. Eigenvalues are normalized to the linear scale first (lambda_i / lambda_max,
   so lambda_max -> 1), THEN converted to dB. This removes absolute power
   level as a factor (dithered vs full-power CPIs), leaving spectral SHAPE as
   the signal the model learns from.

3. Only the first N_KEEP=12 eigenvalues (the 12 largest, in descending order)
   are used as features. The remaining 4 (smallest) eigenvalues are the ones
   most exposed to dithering / dropout artifacts (see last-eigenvalue
   collapse: ~38.6% of healthy tiles show a >10 dB collapse in the smallest
   eigenvalue). Truncating to 12 sidesteps that artifact entirely rather than
   trying to anchor around it.

4. Condition number (dB) and effective rank are BOTH computed using only the
   kept 12 eigenvalues, for every tile, clean or dithered alike. This keeps
   these two global features on a consistent basis regardless of whether a
   given tile happened to have valid data in eigenvalue positions 12-15.

Label convention
----------------
This module does not assign clean/RFI labels. It only extracts features.
Labeling is the responsibility of the calling script (see train_anomaly.py),
which trains a one-class (autoencoder) model on tiles assumed clean.
"""

import numpy as np
import warnings

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

CPI_LEN_DEFAULT = 16
CPI_WIDTH_DEFAULT = 250

# Permissive gap-exclusion ratios: preserve as much dithered data as possible
OFF_DIAG_OVERLAP_RATIO_DEFAULT = 0.03
DIAG_VALID_RATIO_DEFAULT = 0.02

# Number of leading (largest) eigenvalues to keep as features.
# 16 - 12 = 4 eigenvalues assumed potentially affected by dithered/dropped rows.
N_KEEP_DEFAULT = 12

EPS = 1e-12


# ---------------------------------------------------------------------------
# GAP-EXCLUSION SAMPLE COVARIANCE MATRIX
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
    off_diag_overlap_ratio : float, default 0.03
        Minimum fraction of overlapping valid range samples required to
        compute an off-diagonal SCM entry R_ij. Kept deliberately low to
        preserve dithered CPIs.
    diag_valid_ratio : float, default 0.02
        Minimum fraction of valid samples required to compute a diagonal
        SCM entry R_ii. Kept deliberately low to preserve dithered CPIs.

    Returns
    -------
    scm : (num_pulses, num_pulses) complex64
        Gap-excluded sample covariance matrix.
    diag_valid_idx : (num_pulses,) bool array
        True where the diagonal term had enough valid samples to be trusted.
    diag_valid_frac : float
        Fraction of diagonal entries that were valid (0 to 1). Useful as a
        tile-level diagnostic for how dithered a given CPI block was.
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


# ---------------------------------------------------------------------------
# EIGENVALUE FEATURE EXTRACTION
# ---------------------------------------------------------------------------

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


def normalize_eigvals_db(eigvals: np.ndarray) -> np.ndarray:
    """
    Normalize eigenvalues to the linear scale (lambda_i / lambda_max), then
    convert to dB. lambda_max maps to 0 dB by construction.

    Parameters
    ----------
    eigvals : (M,) float array, descending order, linear scale

    Returns
    -------
    eigvals_db : (M,) float32 array
    """
    lambda_max = max(float(eigvals[0]), EPS)
    eigvals_norm = np.clip(eigvals / lambda_max, EPS, None)
    eigvals_db = 10.0 * np.log10(eigvals_norm)
    return eigvals_db.astype(np.float32)


def extract_anomaly_features(
    cpi_data: np.ndarray,
    *,
    mask_valid_cpi: np.ndarray = None,
    n_keep: int = N_KEEP_DEFAULT,
    off_diag_overlap_ratio: float = OFF_DIAG_OVERLAP_RATIO_DEFAULT,
    diag_valid_ratio: float = DIAG_VALID_RATIO_DEFAULT,
):
    """
    Extract anomaly-detection features from a single 16x250 CPI tile.

    Parameters
    ----------
    cpi_data : (M, K) complex array
        Slow-time CPI block, standard M=16, K=250.
    mask_valid_cpi : (M, K) bool array, optional
        True indicates valid samples (e.g. from subswath masking). If None,
        SCM is computed with no gap exclusion.
    n_keep : int, default 12
        Number of leading (largest) eigenvalues to keep as features.
    off_diag_overlap_ratio, diag_valid_ratio : float
        Gap-exclusion validity ratios, see compute_gap_exclusion_scm.

    Returns
    -------
    eigen_feat : (n_keep, 2) float32 array
        Channel 0: dB-normalized eigenvalues (max eigenvalue -> 0 dB).
        Channel 1: slope in dB per eigenvalue index, zero-padded at the end.
    global_feat : (2,) float32 array
        [condition_number_db, effective_rank], both computed using only the
        first n_keep eigenvalues so dropout in eigenvalues n_keep..M-1 never
        affects these two features.
    diag_valid_frac : float
        Fraction of the 16 diagonal SCM entries that had enough valid range
        samples. Useful as a per-tile "how dithered was this" diagnostic;
        NOT used to gate the feature computation itself.
    eigvals_db_full : (M,) float32 array
        Full dB-normalized eigenvalue spectrum (all M values), returned for
        diagnostics / plotting only. Not fed to the model.
    """
    M, K = cpi_data.shape

    if n_keep > M:
        raise ValueError(f"n_keep ({n_keep}) cannot exceed number of pulses M ({M})")

    scm, diag_valid_idx, diag_valid_frac = compute_gap_exclusion_scm(
        cpi_data,
        mask_valid_cpi=mask_valid_cpi,
        off_diag_overlap_ratio=off_diag_overlap_ratio,
        diag_valid_ratio=diag_valid_ratio,
    )

    eigvals = eigen_decompose_descending(scm)
    eigvals_db_full = normalize_eigvals_db(eigvals)

    # Truncate to the first n_keep (largest) eigenvalues
    eigvals_db_kept = eigvals_db_full[:n_keep]

    # Slope in dB per eigenvalue index, zero-padded to keep length n_keep
    slopes = np.diff(eigvals_db_kept)
    slopes_padded = np.append(slopes, 0.0).astype(np.float32)

    eigen_feat = np.stack([eigvals_db_kept, slopes_padded], axis=-1).astype(np.float32)
    eigen_feat = np.nan_to_num(eigen_feat, nan=0.0, posinf=0.0, neginf=-120.0)

    # Condition number (dB): max eigenvalue (0 dB by construction) minus the
    # n_keep-th eigenvalue. Never touches eigenvalues n_keep..M-1.
    condition_number_db = float(eigvals_db_full[0] - eigvals_db_full[n_keep - 1])

    # Effective rank via Shannon entropy, computed on the linear-scale
    # eigenvalues restricted to the first n_keep, renormalized to sum to 1
    # over that subset. Max possible value is n_keep (full spread).
    eigvals_kept_linear = np.maximum(eigvals[:n_keep], EPS)
    p = eigvals_kept_linear / np.sum(eigvals_kept_linear)
    p = p[p > 0]
    eff_rank = float(np.exp(-np.sum(p * np.log(p))))

    global_feat = np.array([condition_number_db, eff_rank], dtype=np.float32)
    global_feat = np.nan_to_num(global_feat, nan=0.0, posinf=0.0, neginf=0.0)

    return eigen_feat, global_feat, diag_valid_frac, eigvals_db_full


def tile_and_extract_features(
    raw_data: np.ndarray,
    *,
    mask_valid: np.ndarray = None,
    cpi_len: int = CPI_LEN_DEFAULT,
    cpi_width: int = CPI_WIDTH_DEFAULT,
    n_keep: int = N_KEEP_DEFAULT,
    off_diag_overlap_ratio: float = OFF_DIAG_OVERLAP_RATIO_DEFAULT,
    diag_valid_ratio: float = DIAG_VALID_RATIO_DEFAULT,
    min_tile_valid_frac: float = 0.0,
):
    """
    Tile a (pulses, range) raw data array into non-overlapping 16x250 CPI
    blocks and extract anomaly features from each tile.

    Parameters
    ----------
    raw_data : (n_pulses, n_range) complex array
    mask_valid : (n_pulses, n_range) bool array, optional
        Subswath / gap validity mask, same shape as raw_data.
    cpi_len : int, default 16
    cpi_width : int, default 250
    n_keep : int, default 12
    off_diag_overlap_ratio, diag_valid_ratio : float
        Gap-exclusion validity ratios.
    min_tile_valid_frac : float, default 0.0
        Minimum fraction of valid diagonal entries (0 to 1) required to keep
        a tile at all. Default 0.0 keeps everything (dithered data is
        preserved); raise this only if you need to discard tiles that are
        almost entirely dropped out.

    Returns
    -------
    eigen_feats : (N, n_keep, 2) float32 array
    global_feats : (N, 2) float32 array
    diag_valid_fracs : (N,) float32 array
    tile_pulse_idx : (N,) int32 array
        Starting pulse index of each tile (for traceability).
    tile_range_idx : (N,) int32 array
        Starting range index of each tile (for traceability).
    """
    n_pulses, n_range = raw_data.shape
    n_pulse_tiles = n_pulses // cpi_len
    n_range_tiles = n_range // cpi_width

    if n_pulse_tiles == 0 or n_range_tiles == 0:
        warnings.warn(
            f"Raw data shape {raw_data.shape} too small for CPI size "
            f"{cpi_len}x{cpi_width}; no tiles extracted."
        )
        empty_eigen = np.zeros((0, n_keep, 2), dtype=np.float32)
        empty_global = np.zeros((0, 2), dtype=np.float32)
        empty_frac = np.zeros((0,), dtype=np.float32)
        empty_idx = np.zeros((0,), dtype=np.int32)
        return empty_eigen, empty_global, empty_frac, empty_idx, empty_idx

    eigen_list, global_list, frac_list = [], [], []
    pulse_idx_list, range_idx_list = [], []

    for pt in range(n_pulse_tiles):
        p_start = pt * cpi_len
        p_end = p_start + cpi_len

        for rt in range(n_range_tiles):
            r_start = rt * cpi_width
            r_end = r_start + cpi_width

            cpi_data = raw_data[p_start:p_end, r_start:r_end]
            cpi_mask = None
            if mask_valid is not None:
                cpi_mask = mask_valid[p_start:p_end, r_start:r_end]

            eigen_feat, global_feat, diag_valid_frac, _ = extract_anomaly_features(
                cpi_data,
                mask_valid_cpi=cpi_mask,
                n_keep=n_keep,
                off_diag_overlap_ratio=off_diag_overlap_ratio,
                diag_valid_ratio=diag_valid_ratio,
            )

            if diag_valid_frac < min_tile_valid_frac:
                continue

            eigen_list.append(eigen_feat)
            global_list.append(global_feat)
            frac_list.append(diag_valid_frac)
            pulse_idx_list.append(p_start)
            range_idx_list.append(r_start)

    eigen_feats = np.stack(eigen_list).astype(np.float32) if eigen_list else \
        np.zeros((0, n_keep, 2), dtype=np.float32)
    global_feats = np.stack(global_list).astype(np.float32) if global_list else \
        np.zeros((0, 2), dtype=np.float32)
    diag_valid_fracs = np.array(frac_list, dtype=np.float32)
    tile_pulse_idx = np.array(pulse_idx_list, dtype=np.int32)
    tile_range_idx = np.array(range_idx_list, dtype=np.int32)

    return eigen_feats, global_feats, diag_valid_fracs, tile_pulse_idx, tile_range_idx
