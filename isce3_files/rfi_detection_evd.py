"""
RFI Detection using DUAL-CHECK: Condition Number + Power Stationarity
Enhanced version - uses BOTH condition number variability AND power spread check
"""
import numpy as np
from isce3.signal.compute_evd_cpi import compute_evd_tb
from dataclasses import dataclass, field
from typing import List
import warnings

@dataclass
class ThresholdParams:
    """This dataclass computes the interpolated value of the number
    of sigmas (standard deviation) of the first difference of minimum
    Eigenvalues across threshold block.

    Parameters
    ----------
    x: list of floats
        This is the computed sigma ratio of maximum and minimum Eigenvalue
        first differences. It is a dimensionless figure of merit. Larger value
        of x indicates higher likelihood of RFI presence.
        Defaults are [2.0, 20.0]
    y: list of floats
        Estimated range of number of sigmas of the first difference of
        minimum Eigenvalues across threshold block as a function of input x,
        e.g., smaller x results in larger value of y, therefore relaxing the
        final threshold. The values of x outside of the defined range of y are
        extrapolated.
        Defaults are [5.0, 2.0]
    """
    x: List[float] = field(default_factory=lambda: [2.0, 20.0])
    y: List[float] = field(default_factory=lambda: [5.0, 2.0])

    def __post_init__(self) -> None:
        if len(self.x) != len(self.y):
            raise ValueError("length mismatch: x and y must have the same size")
        if len(self.x) < 2:
            raise ValueError("At least two points are required")

def rfi_detect(
    raw_data,
    cpi_len,
    max_deg_freedom,
    min_ev_valid_idx,
    *,
    num_rfi_buffer=2,
    num_max_trim=0,
    num_min_trim=0,
    max_num_rfi_ev=2,
    off_diag_overlap_ratio=0.25,
    diag_valid_ratio=0.20,
    rx_dynamic_range_db=50.0,
    mask_valid=None,
    threshold_method='max_ev',
    threshold_params: ThresholdParams = None,
    rfi_check=True,
    bt_cond_num_std_thresh_db=3.0,
    max_ev_spread_thresh_db=2.0,
    eff_rank_std_thresh=1.0,
    pwr_ref_percentile=50,
    pwr_upper_percentile=99.5,
    sig_ev_margin_db=1.0,
    rfi_candidate_tolerance_db=3.0,
):

    """This wrapper performs Eigenvalue Decomposition of input raw data as well as
    RFI Eigenvalue slope threshold estimation and detection.

    DUAL-CHECK VERSION: Uses condition number check AND power spread check

    Workflow:
    1. Perform EVD on all CPIs in the threshold block
    2. Validate TB (check for sufficient usable eigenvalues)
    3. RFI-presence characterization based on Eigenvalue spread, condition number
       variability, and effective rank variability:
       SKIP if none of these traits indicate RFI presence
    4. If RFI traits are present, apply selected threshold method (ev_slope/max_ev)
    5. Detect RFI eigenvalues per CPI based on computed threshold

    Parameters
    ------------
    raw_data: array-like complex [num_pulses x num_rng_samples]
        raw data to be processed, supports all numpy complex formats
    cpi_len: int
        Number of slow-time pulses within a Coherent Processing Interval or CPI
    max_deg_freedom: int
        Max number of independent RFI emitters designed to be detected and mitigated.
        This number should be less than cpi_len to avoid unintended removal of signal data.
    min_ev_valid_idx: int
        Eigenvalue index used by threshold estimation to estimate the slow-time minimum
        Eigenvalue slope. This parameter is also used to validate that the threshold block
        has enough usable eigenvalues for robust sample covaraince estimation of a CPI.
    num_rfi_buffer: int, default=3
        Number of buffer eigenvalue indices to skip after last possible RFI EV before
        starting clean segment interpolation for 'max_ev' method. The clean segment
        starts at index (max_deg_freedom + num_rfi_buffer - 1).
    num_max_trim: int, default=0
        Number of large-value outliers to be trimmed in slow-time minimum Eigenvalues.
    num_min_trim: int, default=0
        Number of small-value outliers to be trimmed in slow-time minimum Eigenvalues
    max_num_rfi_ev: int, default=2
        A detection error (miss) happens when a maximum power RFI emitter contaminates
        multiple consecutive CPIs, resulting in a flat maximum Eigenvalue slope in slow
        time. Hence the standard (STD) deviation of multiple dominant EVs across slow time
        defined by this parameter are compared. The one with the maximum STD is used for RFI
        Eigenvalue first difference computation.
    off_diag_overlap_ratio : float, default=0.25
        Minimum overlap ratio used by gap exclusion covariance estimation
    diag_valid_ratio : float, default=0.20
        Minimum fraction of valid samples required to compute a diagonal term in the
        sample covariance matrix entry R_ii.
    rx_dynamic_range_db: float, default=50.0
        Radar platform receiver dynamic range. This is applied as a threshold
        to determine if the Eigenvalue under test is meaningfully signficant. If the
        Eigenvalue under test is less than this threshold, it will be viewed as unusable.
    mask_valid : np.ndarray bool or None, default=None
        Valid-sample mask with same shape as raw_data If provided, it has the shape of
        [num_pulses x num_rng_samples]. CPI sample covariance matrix will be normalized
        differently by excluding the invalid data gaps.
    threshold_method : str, default='max_ev'
        Detection method: 'ev_slope', 'max_ev', or 'aer'. ('ev_est' is implemented
        but not currently wired up as a selectable option.)
        'max_ev' now uses per-CPI adaptive thresholds with fixed margin.
    threshold_params: ThresholdParams dataclass object or None, default=None
        RFI detection threshold interpolation parameters. If None, a method-specific
        default is selected: ThresholdParams() (x=[2.0, 20.0], y=[5.0, 2.0]) for
        'ev_slope', or ThresholdParams(x=[1.0, 10.0], y=[1.0, 0.95]) for 'aer'.
        For 'ev_slope', the x field defines the STD ratio between maximum and
        minimum Eigenvalue slopes (MMES) of the slow-time threshold interval, and
        the y field defines the number of sigma (STD) from the mean of MMES. For
        'aer', the x field defines the STD of the per-CPI max/min Eigenvalue dB
        spread (AER figure of merit), and the y field defines the quantile
        probability (0 to 1) used to derive the absolute Eigenvalue detection
        threshold.
    rfi_check : bool, default=True
        Controls RFI-presence characterization. If False, no check is performed
        and all TBs proceed to RFI detection. Otherwise, a TB is screened for
        RFI-like Eigenvalue traits (dominant Eigenvalue spread, condition number
        variability, and effective rank variability); the TB is skipped if none
        of these traits indicate RFI is present.
    bt_cond_num_std_thresh_db : float, default=3.0
        Condition number std threshold in dB. TBs with std(cond#) > this value,
        combined with eff_rank_std > eff_rank_std_thresh, indicate RFI presence.
        Only used when rfi_check=True.
    max_ev_spread_thresh_db : float, default=2.0
        Threshold in dB for the spread (std across CPIs) of the dominant Eigenvalues
        in a TB. If the maximum spread among dominant EVs exceeds this value, RFI
        is determined to be present. Only used when rfi_check=True.
    eff_rank_std_thresh : float, default=1.0
        Threshold for the standard deviation of per-CPI effective rank across a TB.
        Combined with bt_cond_num_std_thresh_db, indicates RFI presence.
        Only used when rfi_check=True.
    pwr_ref_percentile : float, default=50
        Reference (baseline) percentile for power spread computation.
        Anchors the lower end of the spread measurement.
    pwr_upper_percentile : float, default=99.5
        Upper tail percentile for power spread computation (recommended: 98.0 or 99.5).
        99.5 = top 0.5% of power samples, 98.0 = top 2% of power samples.
    sig_ev_margin_db : float, default=1.0
        Aggressive safety margin in dB added to the extrapolated estimate for RFI candidates
        in 'max_ev' method. Only used when threshold_method='max_ev'.
    rfi_candidate_tolerance_db : float, default=3.0
        Tolerance in dB for RFI candidate selection in 'max_ev' method. CPIs where
        abs(actual_EV0 - predicted_clean_EV0) > this value are flagged as RFI candidates
        and receive aggressive thresholds. Only used when threshold_method='max_ev'.

    RFI-PRESENCE CHARACTERIZATION LOGIC:
    -------------------------------------
    rfi_check=False:
        All TBs proceed to RFI detection (no RFI-trait screening)
    rfi_check=True:
        rfi_present = True  if max_ev_spread_db > max_ev_spread_thresh_db
        rfi_present = True  elif (condition_number_std_db > bt_cond_num_std_thresh_db
                                   and eff_rank_std > eff_rank_std_thresh)
        rfi_present = False otherwise
        SKIP TB if rfi_present is False

    Returns
    --------
    rfi_cpi_flag_array: 2D array of bool, [num_cpi x cpi_len]
        RFI flag array that marks each Eigenvalue index in a CPI as either RFI or signal.
        1 = RFI Eigenvalue index; 0 = Signal Eigenvalue index
    eig_vec_sort: 3D array of complex, [num_cpi x cpi_len x cpi_len]
        Sorted column vector Eigenvectors of all CPIs based on indices of sorted Eigenvalues
    bt_skipped: bool
        True if the TB was skipped because none of the RFI-trait checks (dominant
        Eigenvalue spread, condition number variability, effective rank variability)
        indicated RFI presence. False in all other cases (invalid TB, low FoM,
        or normal detection path).
    condition_number_std_db: float
        Standard deviation of per-CPI condition numbers (in dB) across the TB, as
        computed by compute_tb_condition_number_std(). NaN if rfi_check
        is False or the TB is invalid (check not performed).
    upper_tail_spread_db: float
        Upper-tail power spread (in dB) across the TB, as computed by
        compute_tb_upper_tail(). NaN if the TB is invalid (check not performed).
    rfi_present: bool
        True if the RFI-presence characterization determined the TB exhibits
        RFI-like Eigenvalue traits (or if rfi_check is False, in which case all
        TBs are assumed to potentially contain RFI). False if the check determined
        the TB does not exhibit RFI traits (TB skipped), or if the TB is invalid.
    num_rfi_candidate_cpi: int or nan
        For 'max_ev' threshold method: Number of CPIs flagged as RFI candidates in
        Step 1 (those receiving aggressive thresholds extrapolated to max_deg_freedom - 1).
        NaN for all other threshold methods or when the TB is skipped/invalid before
        threshold estimation is reached.
    """
    num_pulses = raw_data.shape[0]

    # Verify total number of pulses is greater than number of pulses per CPI
    if num_pulses < cpi_len:
        raise ValueError(
            "Total number of pulses must be greater or equal to number of pulses per single CPI."
        )

    # Select method-specific threshold_params default if not explicitly provided
    if threshold_params is None:
        if threshold_method == 'aer':
            threshold_params = ThresholdParams(x=[1.0, 10.0], y=[1.0, 0.95])
        else:
            threshold_params = ThresholdParams()

    (
        eig_val_sort_array,
        eig_vec_sort_array,
        diag_power_array,
        diag_valid_array,
        tb_is_valid,
    ) = compute_evd_tb(
        raw_data,
        cpi_len=cpi_len,
        mask_valid=mask_valid,
        off_diag_overlap_ratio=off_diag_overlap_ratio,
        diag_valid_ratio=diag_valid_ratio,
        min_ev_valid_idx=min_ev_valid_idx,
        rx_dynamic_range_db=rx_dynamic_range_db,
    )

    # Default bright-target check stats: only populated when the check actually runs
    condition_number_std_db = np.nan
    upper_tail_spread_db = np.nan
    # Default rfi_present: assume RFI may be present unless the check determines otherwise
    rfi_present = True
    # Default RFI candidate CPI count: only populated for 'max_ev' threshold estimation
    num_rfi_candidate_cpi = np.nan

    # If any CPI within a threshold block is determined to be invalid
    # Then skip threshold computation for this block by setting rfi_cpi_flag_array
    # to all zeros
    if not tb_is_valid:
        num_cpi = eig_val_sort_array.shape[0]
        rfi_cpi_flag_array = np.zeros((num_cpi, cpi_len), dtype=np.bool_)
        fig_merit_detect_tb = 0
        bt_skipped = False  # invalid TB, not an RFI-presence skip
        rfi_present = False  # invalid TB: no RFI determination possible

        return (
            rfi_cpi_flag_array,
            eig_vec_sort_array,
            diag_power_array,
            diag_valid_array,
            fig_merit_detect_tb,
            bt_skipped,
            condition_number_std_db,
            upper_tail_spread_db,
            rfi_present,
            num_rfi_candidate_cpi,
        )

    upper_tail_spread_db = compute_tb_upper_tail(
        diag_power_array,
        diag_valid_array,
        pwr_ref_percentile=pwr_ref_percentile,
        pwr_upper_percentile=pwr_upper_percentile,
    )

    # Bright target check: controlled by rfi_check
    if rfi_check:
        max_ev_spread_db = compute_tb_max_ev_spread(
            eig_val_sort_array,
            max_deg_freedom,
        )

        condition_number_std_db = compute_tb_condition_number_std(
            eig_val_sort_array,
            min_ev_valid_idx,
        )

        eff_rank_std = compute_tb_eff_rank_std(
            eig_val_sort_array,
            min_ev_valid_idx,
        )

        if max_ev_spread_db > max_ev_spread_thresh_db:
            rfi_present = True
        elif (condition_number_std_db > bt_cond_num_std_thresh_db and
              eff_rank_std > eff_rank_std_thresh):
            rfi_present = True
        else:
            rfi_present = False

        if not rfi_present:
            # No RFI-like Eigenvalue traits detected: stable structure + stationary power
            num_cpi = eig_val_sort_array.shape[0]
            rfi_cpi_flag_array = np.zeros((num_cpi, cpi_len), dtype=np.bool_)
            fig_merit_detect_tb = 0
            bt_skipped = True  # TB skipped: RFI-presence characterization found no RFI traits

            return (
                rfi_cpi_flag_array,
                eig_vec_sort_array,
                diag_power_array,
                diag_valid_array,
                fig_merit_detect_tb,
                bt_skipped,
                condition_number_std_db,
                upper_tail_spread_db,
                rfi_present,
                num_rfi_candidate_cpi,
            )

    # TB has potential RFI (at least one check failed): proceed with detection
    cpi_detect_mask = None

    if threshold_method == 'ev_slope':
        # Estimate a single threshold for all CPIs
        detect_threshold, fig_merit_detect_tb = threshold_estimate_evd(
            eig_val_sort_array,
            num_max_trim,
            num_min_trim,
            max_num_rfi_ev,
            min_ev_valid_idx,
            threshold_params,
        )
    elif threshold_method == 'max_ev':
        detect_threshold, fig_merit_detect_tb, num_rfi_candidate_cpi, rfi_cpi_candidate = threshold_estimate_max_ev(
            eig_val_sort_array,
            max_deg_freedom=max_deg_freedom,
            num_rfi_buffer=num_rfi_buffer,
            num_max_trim=num_max_trim,
            num_min_trim=num_min_trim,
            max_num_rfi_ev=max_num_rfi_ev,
            min_ev_valid_idx=min_ev_valid_idx,
            sig_ev_margin_db=sig_ev_margin_db,
            rfi_candidate_tolerance_db=rfi_candidate_tolerance_db,
        )
        cpi_detect_mask = rfi_cpi_candidate
    elif threshold_method == 'aer':
        detect_threshold, fig_merit_detect_tb = threshold_estimate_aer(
            eig_val_sort_array,
            num_max_trim=num_max_trim,
            num_min_trim=num_min_trim,
            min_ev_valid_idx=min_ev_valid_idx,
            threshold_params=threshold_params,
        )
    else:
        raise ValueError(f"Unsupported threshold method: {threshold_method}")

    # Detect RFI Eigenvalues of each CPI based on input detection threshold
    rfi_cpi_flag_array = rfi_detect_evd_tb(
        eig_val_sort_array,
        detect_threshold,
        max_deg_freedom,
        threshold_method,
        cpi_detect_mask=cpi_detect_mask,
    )

    bt_skipped = False  # TB processed normally (not skipped)
    return (
        rfi_cpi_flag_array,
        eig_vec_sort_array,
        diag_power_array,
        diag_valid_array,
        fig_merit_detect_tb,
        bt_skipped,
        condition_number_std_db,
        upper_tail_spread_db,
        rfi_present,
        num_rfi_candidate_cpi,
    )

def threshold_estimate_max_ev(
    eig_val_sort_array,
    max_deg_freedom=4,
    num_rfi_buffer=2,
    num_max_trim=0,
    num_min_trim=0,
    max_num_rfi_ev=2,
    min_ev_valid_idx=10,
    sig_ev_margin_db=1.0,
    rfi_candidate_tolerance_db=3.0,
):
    """Estimate signal max eigenvalue threshold using max_ev detection algorithm.

    TWO-STEP ALGORITHM:
    1. RFI CPI Candidate Selection:
       - For each CPI, perform linear regression on clean segment
       - Extrapolate to index 0 to get predicted clean EV0
       - Compare with actual EV0: if abs(actual - predicted) > rfi_candidate_tolerance_db,
         flag as RFI candidate
    2. Aggressive Threshold for RFI Candidates:
       - For RFI candidates: extrapolate to sig_ev_ref_idx (max_deg_freedom - 1) with
         aggressive margin (sig_ev_margin_db, default 1.0 dB)
       - For good CPIs (not RFI candidates): threshold comparison is skipped via mask,
         resulting in all eigenvalues marked as signal (rfi_cpi_flag_array = all False)

    Parameters
    ----------
    eig_val_sort_array : 2D array of float, [num_cpi x cpi_len]
        Sorted eigenvalues in descending order
    max_deg_freedom : int, default=4
        Maximum number of independent RFI emitters designed to be detected.
        Eigenvalue indices [0, ..., max_deg_freedom-1] can potentially be RFI.
    num_rfi_buffer : int, default=2
        Number of buffer eigenvalue indices to skip after last possible RFI EV
        before starting clean segment interpolation.
        start_idx = max_deg_freedom + num_rfi_buffer - 1.
    num_max_trim : int, default=0
        Number of large-value outliers to trim for FoM computation
    num_min_trim : int, default=0
        Number of small-value outliers to trim for FoM computation
    max_num_rfi_ev : int, default=2
        Maximum number of dominant EVs to check for FoM computation
    min_ev_valid_idx : int, default=10
        Eigenvalue index for noise floor estimation
    sig_ev_margin_db : float, default=1.0
        Aggressive safety margin in dB added to the extrapolated estimate for RFI candidates.
    rfi_candidate_tolerance_db : float, default=3.0
        Tolerance in dB for RFI candidate selection. If abs(actual_EV0 - predicted_clean_EV0)
        exceeds this value, the CPI is flagged as an RFI candidate.

    Returns
    -------
    detect_threshold : ndarray of float, shape (num_cpi,)
        Per-CPI signal max eigenvalue thresholds in dB. All CPIs receive finite thresholds
        computed as ev_extrap_db_array + sig_ev_margin_db. The rfi_cpi_candidate mask
        determines which CPIs actually undergo threshold comparison.
    fig_merit : float
        Figure of merit value computed
    num_rfi_candidates : int
        Number of CPIs flagged as RFI candidates in Step 1 (those receiving aggressive thresholds).
    rfi_cpi_candidate : ndarray of bool, shape (num_cpi,)
        Per-CPI mask for 'max_ev' method. True = RFI candidate (apply threshold comparison);
        False = good CPI (skip comparison, mark as clean/signal).
    """

    # Compute FoM using EST method (for diagnostics only)
    _, fig_merit = threshold_estimate_evd(
        eig_val_sort_array,
        num_max_trim=num_max_trim,
        num_min_trim=num_min_trim,
        max_num_rfi_ev=max_num_rfi_ev,
        min_ev_valid_idx=min_ev_valid_idx,
        threshold_params=ThresholdParams(x=[2.0, 20.0], y=[5.0, 2.0]),
    )

    # Convert eigenvalues to dB
    eps = np.finfo(np.float64).tiny
    eig_val = np.maximum(np.real(eig_val_sort_array), eps)
    eig_val_sort_db_array = 10.0 * np.log10(eig_val)

    num_cpi, cpi_len = eig_val_sort_db_array.shape

    # Define clean eigenvalue segment
    start_idx = max_deg_freedom + num_rfi_buffer - 1
    end_idx = min_ev_valid_idx

    if end_idx - start_idx < 2:
        raise ValueError(
            f"Invalid clean eigenvalue segment: need at least 2 points, got "
            f"{end_idx - start_idx}. start_idx={start_idx}, end_idx={end_idx}."
        )

    # Reference index for aggressive threshold (RFI candidates)
    sig_ev_ref_idx = max_deg_freedom - 1

    # Initialize arrays
    ev0_est_db_array = np.zeros(num_cpi, dtype=np.float64)
    ev_extrap_db_array = np.zeros(num_cpi, dtype=np.float64)
    rfi_cpi_candidate = np.zeros(num_cpi, dtype=bool)

    # Single loop: compute both ev0_est_db and ev_extrap_db for each CPI
    for idx_cpi in range(num_cpi):
        # Extract clean eigenvalue segment for this CPI
        ev_segment = eig_val_sort_db_array[idx_cpi, start_idx:end_idx]
        indices = np.arange(start_idx, end_idx)

        # Linear regression: ev_db = slope * index + intercept
        coeffs = np.polyfit(indices, ev_segment, deg=1)

        # Extrapolate to index 0 (for RFI candidate selection)
        ev0_est_db_array[idx_cpi] = np.polyval(coeffs, 0)

        # Extrapolate to sig_ev_ref_idx (for aggressive threshold)
        ev_extrap_db_array[idx_cpi] = np.polyval(coeffs, sig_ev_ref_idx)

    # Step 1: Select RFI CPI candidates
    # Compare actual EV0 with predicted clean EV0
    for idx_cpi in range(num_cpi):
        ev0_excess_db = eig_val_sort_db_array[idx_cpi, 0] - ev0_est_db_array[idx_cpi]
        if np.abs(ev0_excess_db) > rfi_candidate_tolerance_db:
            rfi_cpi_candidate[idx_cpi] = True

    num_rfi_candidates = np.sum(rfi_cpi_candidate)

    # Step 2: Generate finite per-CPI thresholds.
    # The final max_ev comparison is applied only where rfi_cpi_candidate is True.
    # For good CPIs, this finite placeholder is not used because comparison is skipped.
    signal_ev_max_estimate_array = ev_extrap_db_array + sig_ev_margin_db

    return signal_ev_max_estimate_array, fig_merit, num_rfi_candidates, rfi_cpi_candidate

def threshold_estimate_ev_est(
    eig_val_sort_array,
    num_max_trim=0,
    num_min_trim=0,
    max_num_rfi_ev=2,
    min_ev_valid_idx=10,
    sig_ev_margin_db=3.0,
):
    """Estimate signal max eigenvalue threshold using the 'ev_est' method.

    Similar to 'max_ev' but uses only two eigenvalue points — at indices
    (min_ev_valid_idx - 1) and (min_ev_valid_idx) — to estimate the local
    slope and extrapolate to EV[0] per CPI.

    Algorithm:
    1. Compute FoM using EST method (returned for diagnostics only)
    2. For each CPI:
       a. Use EV[min_ev_valid_idx - 1] and EV[min_ev_valid_idx] to compute slope.
       b. Extrapolate the fitted line to index 0 and add sig_ev_margin_db.

    Parameters
    ----------
    eig_val_sort_array : 2D array of float, [num_cpi x cpi_len]
        Sorted eigenvalues in descending order.
    num_max_trim : int, default=0
        Number of large-value outliers to trim for FoM computation.
    num_min_trim : int, default=0
        Number of small-value outliers to trim for FoM computation.
    max_num_rfi_ev : int, default=2
        Maximum number of dominant EVs to check for FoM computation.
    min_ev_valid_idx : int, default=10
        Eigenvalue index of the noise floor. The two estimation points are
        (min_ev_valid_idx - 1) and (min_ev_valid_idx).
    sig_ev_margin_db : float, default=3.0
        Safety margin in dB added to the extrapolated EV[0] estimate.

    Returns
    -------
    detect_threshold : float or ndarray
        Per-CPI signal max eigenvalue estimate in dB.
    fig_merit : float
        Figure of merit value computed.
    """
    # Step 1: Compute FoM
    _, fig_merit = threshold_estimate_evd(
        eig_val_sort_array,
        num_max_trim=num_max_trim,
        num_min_trim=num_min_trim,
        max_num_rfi_ev=max_num_rfi_ev,
        min_ev_valid_idx=min_ev_valid_idx,
        threshold_params=ThresholdParams(x=[2.0, 20.0], y=[5.0, 2.0]),
    )

    # Step 2: Per-CPI EV[0] estimation using two points
    # (RFI presence is now determined by the rfi_check logic in rfi_detect, not FoM)
    eps = np.finfo(np.float64).tiny
    eig_val = np.maximum(np.real(eig_val_sort_array), eps)
    eig_val_sort_db_array = 10.0 * np.log10(eig_val)

    num_cpi = eig_val_sort_db_array.shape[0]
    signal_ev_max_estimate = np.zeros(num_cpi)

    idx_lo = min_ev_valid_idx - 1  # lower index (start)
    idx_hi = min_ev_valid_idx      # higher index (end)

    for i in range(num_cpi):
        # Slope from the two points at idx_lo and idx_hi
        ev_lo = eig_val_sort_db_array[i, idx_lo]
        ev_hi = eig_val_sort_db_array[i, idx_hi]
        slope = ev_hi - ev_lo  # dB per index (negative for descending spectrum)

        # Extrapolate to index 0: intercept = ev_lo - slope * idx_lo
        intercept = ev_lo - slope * idx_lo

        signal_ev_max_estimate[i] = intercept + sig_ev_margin_db

    return signal_ev_max_estimate, fig_merit


def threshold_estimate_aer(
    eig_val_sort_array,
    num_max_trim=0,
    num_min_trim=0,
    min_ev_valid_idx=10,
    threshold_params: ThresholdParams = ThresholdParams(x=[1.0, 10.0], y=[1.0, 0.95]),
):
    """Estimate AER eigenvalue threshold in dB.

    AER:
        alpha_k = lambda_max_db,k - lambda_min_db,k
        F = std(alpha_k)
        p = interp(F, threshold_params.x, threshold_params.y)
        tau = quantile(eigenvalues in TB up to min_ev_valid_idx, p)

    Parameters
    ----------
    eig_val_sort_array : 2D array of float, [num_cpi x cpi_len]
        Sorted Eigenvalues in descending order in linear units of all CPIs in raw data.
    num_max_trim : int, default=0
        Number of large-value outliers to trim from the per-CPI alpha (max-min EV
        dB spread) array.
    num_min_trim : int, default=0
        Number of small-value outliers to trim from the per-CPI alpha array.
    min_ev_valid_idx : int, default=10
        Eigenvalue index used as the minimum usable Eigenvalue for alpha computation.
    threshold_params : ThresholdParams dataclass object,
            default=ThresholdParams(x=[1.0, 10.0], y=[1.0, 0.95])
        Interpolation parameters. The x field is the AER figure of merit (STD of
        alpha in dB). The y field is the quantile probability (0 to 1) used to
        derive the absolute Eigenvalue detection threshold.

    Returns
    -------
    detect_threshold : float
        Eigenvalue threshold in dB (absolute, not slope threshold).
    aer_fom : float
        AER figure of merit (STD of per-CPI alpha) computed for the TB.
    """

    eps = np.finfo(np.float64).tiny
    eig_val = np.maximum(np.real(eig_val_sort_array), eps)
    eig_val_sort_db_array = 10.0 * np.log10(eig_val)

    # Use dominant usable eigenvalue and selected minimum usable eigenvalue.
    ev_max_db = eig_val_sort_db_array[:, 0]
    ev_min_db = eig_val_sort_db_array[:, min_ev_valid_idx]

    # Optional trimming of alpha outliers.
    alpha_db = ev_max_db - ev_min_db

    if num_min_trim > 0:
        trim_idx = np.argsort(alpha_db)[:num_min_trim]
        alpha_db = np.delete(alpha_db, trim_idx)

    if num_max_trim > 0:
        trim_idx = np.argsort(alpha_db)[-num_max_trim:]
        alpha_db = np.delete(alpha_db, trim_idx)

    if alpha_db.size < 2:
        return np.inf, 0.0

    # AER figure of merit.
    aer_fom = np.std(alpha_db)

    # threshold_params.x = F interval, e.g. [1, 10]
    # threshold_params.y = quantile probability, e.g. [1.0, 0.95]
    confidence_level = np.interp(
        aer_fom,
        threshold_params.x,
        threshold_params.y,
    )
    confidence_level = np.clip(confidence_level, 0.0, 1.0)

    detect_threshold = np.quantile(
        eig_val_sort_db_array[:, :min_ev_valid_idx + 1], confidence_level
    )

    return detect_threshold, aer_fom


def threshold_estimate_evd(
    eig_val_sort_array,
    num_max_trim=0,
    num_min_trim=0,
    max_num_rfi_ev=2,
    min_ev_valid_idx=10,
    threshold_params: ThresholdParams = ThresholdParams(),
):
    """Perform data-centric thresholding algorithm: "Slow-Time Eigenvalue Slope
    Thresholding "(ST-EST)"[1]_ based on the assumption that first difference of
    minimum Eigenvalue across slow time is an estimate of signal power variation
    as if there is no RFI"

    Algorithm Overview for applying ST-EST on one raw data block:
    1. Remove a specified number of outliers in maximum and minimum Eigenvalues
    2. Compute slow-time standard deviation (STD) of maximum Eigenvalue slope.
    3. Compute slow-time standard deviation (STD) of minimum Eigenvalue slope.
    4. Compute STD ratio of maximum and minimum Eigenvalue slopes (SRMMES).
       SRMMES will be applied as input to a linear interpolator to derive
       the detection threshold in dB / Eigenvalue Index.
    5. Apply SRMMES in step #4 as input to a linear interpolator
       defined by threshold_params. The final detection threshold tau is a
       function of sigma and mu (mean) computed in step #4 such that:
       tau = alpha * sigma(min_EV_slope) + mu(min_EV_slope)
       where alpha is the curve-fitted value.

    Parameters
    ----------
    eig_val_sort_array: 2D array of float, [num_cpi x cpi_len]
        Sorted Eigenvalues in descending order in linear units of all CPIs in raw data
        Eigenvalues will subsequently converted into dB for threshold estimation.
    num_max_trim: int, default = 0
        Number of large value outliers to be trimmed in slow-time minimum Eigenvalues.
    num_min_trim: int, default = 0
        Number of small value outlliers to be trimmed in slow-time minimum Eigenvalues
    max_num_rfi_ev: int, default = 2
        A detection error (miss) happens when a maximum power RFI emitter contaminates
        multiple consecutive CPIs, resulting in a flat maximum Eigenvalue slope in slow
        time. Hence the standard (STD) deviation of multiple dominant EVs across slow time
        defined by this parameter are compared. The one with the maximum STD is used for RFI
        Eigenvalue first difference computation.
    min_ev_valid_idx: int
        Eigenvalue index used by threshold estimation to estimate the slow-time minimum
        Eigenvalue slope. This parameter is also used to validate that the threshold block
        has enough usable eigenvalues for robust sample covaraince estimation of a CPI.
    threshold_params: ThresholdParams dataclass object, default=ThresholdParams()
        RFI detection threshold interpolation parameters

    Returns
    -------
    detect_threshold: float
        RFI detection threshold used by EVD detection algorithm in dB/Eigenvalue index
        All CPIs shares a common threshold.

    References
    ----------
    ..[1] Bo Huang, Heresh Fattahi, Hirad Ghaemi, Brian Hawkins, Geoffrey Gunter,
    "Radio Frequency Interference Detection and Mitigation of NISAR DATA using
    Slow Time Eigenvalue Decomposition", IGARSS 2023.'
    """

    if max_num_rfi_ev < 1:
        raise ValueError('max_num_rfi_ev" shall be larger than zero.')

    # Max power Eigenvalue (RFI) can appear in multiple consecutive CPIs which results
    # in zero (flat) slope across slow time, compute slow-time standard deviation (STD)
    # of top N max power Eigenvalues, default=2, and use the one with highest STD.
    eval_sort_max_db = 10 * np.log10(np.abs(eig_val_sort_array[:, 0:max_num_rfi_ev]))
    eval_sort_max_std = np.std(eval_sort_max_db, axis=0)

    # Find slow-time Principal Component Eigenvalue array with largest STD
    ev_max_std_idx = np.argmax(eval_sort_max_std)
    ev_max_db = eval_sort_max_db[:, ev_max_std_idx]

    # For Dithered PRF mode, min_ev_valid_idx is selected to avoid zeros in the tail
    # of the Eigenvalue spectrum due to invalid data gaps for each pulse.
    ev_min_db = 10 * np.log10(np.abs(eig_val_sort_array[:, min_ev_valid_idx]))

    # Remove possible outliers in max and min Eigenvalues without reordering.
    if num_min_trim > 0:
        ev_min_trim_idx = np.argsort(ev_min_db)[:num_min_trim]
        ev_min_db = np.delete(ev_min_db, ev_min_trim_idx)

    if num_max_trim > 0:
        ev_max_trim_idx = np.argsort(ev_max_db)[-num_max_trim:]
        ev_max_db = np.delete(ev_max_db, ev_max_trim_idx)

    # Compute STD of the slope of max and min Eigenvalues
    ev_slope_max = np.diff(ev_max_db)
    ev_slope_min = np.diff(ev_min_db)

    ev_slope_max_std = ev_slope_max.std()
    ev_slope_min_std = ev_slope_min.std()
    ev_slope_min_mean = ev_slope_min.mean()

    # Max(dB)/min(dB) Eigenvalue slope STD ratio: indicator of RFI severity and
    # input to RFI linear interpolator for final detection threshold
    std_ratio_ev_slope = ev_slope_max_std / ev_slope_min_std

    # Threshold interpolation parameters
    std_ratio = threshold_params.x
    threshold_sigma = threshold_params.y

    num_sigma = np.interp(std_ratio_ev_slope, std_ratio, threshold_sigma)

    detect_threshold = ev_slope_min_mean + num_sigma * ev_slope_min_std

    return detect_threshold, std_ratio_ev_slope


def rfi_detect_evd(
    eig_val_db_slope,
    detect_threshold,
    max_deg_freedom=8,
):
    """Perform RFI detection of Eigenvalues within a CPI based on input detection
    threshold in dB/Eigenvalue index. The threshold is set to be a negative value.
    If the magnitude of an Eigenvalue slope exceeds this threshold, then it is identified
    as RFI.

    Parameters
    ----------
    eig_val_db_slope: 1D array of float
        Eigenvalue slope (first difference of Eigenvalues)
    detect_threshold: float
        A positive RFI detection threshold used by EVD detection algorithm to identify
        RFI Eigenvalue slope valules.
    max_deg_freedom: int, default=8
        Max number of independent RFI emitters designed to be detected and mitigated.
        This number should be less than cpi_len.

    Returns
    -------
    sig_ev_idx_start: int
        Start index of signal Eigenvalues
    """

    sig_ev_idx_start = 0
    rfi_ev_idx = np.where(eig_val_db_slope[:max_deg_freedom] < -detect_threshold)[0]

    if rfi_ev_idx.size:
        sig_ev_idx_start = rfi_ev_idx[-1] + 1

    return sig_ev_idx_start


def rfi_detect_evd_tb(
    eig_val_sort_array,
    detect_threshold,
    max_deg_freedom=4,
    threshold_method='max_ev',
    cpi_detect_mask=None,
):
    """Wrapper function which performs RFI detection of data within a Threshold Block (TB)
    one CPI at a time based on input detection threshold.

    Supports three detection methods:
    - 'ev_slope': Eigenvalue Slope Thresholding (slope-based detection)
    - 'max_ev': EV Maximum Estimation (absolute eigenvalue threshold from noise extrapolation)
    - 'aer': Absolute eigenvalue threshold from quantile-based estimation

    Parameters
    ----------
    eig_val_sort_array: 2D array of float, [num_cpi x cpi_len]
        Sorted Eigenvalues in descending order of all CPIs in raw data
    detect_threshold: float or ndarray
        RFI detection threshold. For 'ev_slope': positive slope threshold in dB/index.
        For 'max_ev' and 'aer': absolute eigenvalue threshold in dB.
        Can be a scalar (same threshold for all CPIs) or a 1D array of length num_cpi
        (one threshold per CPI).
    max_deg_freedom: int, default = 4
        Max number of independent RFI emitters designed to be detected and mitigated.
        This number should be less than cpi_len.
    threshold_method: str, default='max_ev'
        Detection method: 'ev_slope', 'max_ev', or 'aer'
    cpi_detect_mask : ndarray of bool or None, default=None
        Optional per-CPI mask for 'max_ev' method. When provided, threshold comparison
        is applied only where this mask is True (RFI candidates). CPIs where the mask is
        False (good CPIs) are skipped and returned with all-False RFI flags (no RFI detected).
        For other methods ('ev_slope', 'aer'), this parameter is ignored (should be None).

    Returns
    -------
    rfi_cpi_flag_array: 2D array of bool, [num_cpi x cpi_len]
        RFI flag array that marks each Eigenvalue index in a CPI as either RFI or signal.
        True = RFI Eigenvalue index; False = Signal Eigenvalue index
    """

    # Number of pulses, CPI length, and number of range blocks in a CPI
    num_cpi, cpi_len = eig_val_sort_array.shape

    # Convert threshold to array for uniform handling
    detect_threshold_arr = np.atleast_1d(detect_threshold)

    # Validate threshold array shape
    if detect_threshold_arr.size == 1:
        # Scalar threshold - broadcast to all CPIs
        detect_threshold_arr = np.full(num_cpi, detect_threshold_arr[0])
    elif detect_threshold_arr.size != num_cpi:
        raise ValueError(f"Threshold array size ({detect_threshold_arr.size}) must match num_cpi ({num_cpi})")

    # Validate optional CPI detection mask
    if cpi_detect_mask is not None:
        cpi_detect_mask = np.asarray(cpi_detect_mask, dtype=bool)
        if cpi_detect_mask.shape != (num_cpi,):
            raise ValueError(
                f"cpi_detect_mask shape {cpi_detect_mask.shape} must be ({num_cpi},)"
            )

    # Validate threshold values
    if threshold_method == 'ev_slope':
        # Ensure detection threshold is a positive value
        if np.any(~np.isfinite(detect_threshold_arr)) or np.any(detect_threshold_arr <= 0):
            warnings.warn("Warning: Non-positive detection threshold. Skipping TB detection.")
            return np.zeros((num_cpi, cpi_len), dtype=np.bool_)
    elif threshold_method in ('max_ev', 'aer'):
        if np.any(~np.isfinite(detect_threshold_arr)):
            warnings.warn(f"Warning: Non-finite {threshold_method.upper()} detection threshold. Skipping TB detection.")
            return np.zeros((num_cpi, cpi_len), dtype=np.bool_)
    else:
        raise ValueError(f"Unsupported threshold_method: {threshold_method}")

    # Maximum number of degrees of freedom must be less than cpi_len
    if max_deg_freedom >= cpi_len:
        raise ValueError(
            "Max number of deg. of freedom must be less than number of pulses in a CPI."
        )

    # Compute Eigenvalue in dB for all CPIs
    eig_val_sort_db_array = 10 * np.log10(np.abs(eig_val_sort_array))

    if threshold_method == 'ev_slope':
        # RFI flag for each eigenvalue index in all CPIs: RFI=1, signal=0
        rfi_cpi_flag_array = np.ones((num_cpi, cpi_len), dtype=np.bool_)
        eig_val_db_slope_array = np.diff(eig_val_sort_db_array, axis=1)
        for idx_cpi in range(num_cpi):
            eig_val_db_slope = eig_val_db_slope_array[idx_cpi]

            # Use per-CPI threshold
            sig_ev_idx_start = rfi_detect_evd(eig_val_db_slope, detect_threshold_arr[idx_cpi], max_deg_freedom)

            # Sets signal eigenvalue indices to False
            rfi_cpi_flag_array[idx_cpi, sig_ev_idx_start:] = False
    else:
        # Absolute eigenvalue comparison ('max_ev' and 'aer').
        # Start with all CPIs marked clean. If a max_ev mask is provided, only
        # masked/bad CPIs are compared against the aggressive threshold.
        rfi_cpi_flag_array = np.zeros((num_cpi, cpi_len), dtype=np.bool_)
        if cpi_detect_mask is None:
            cpi_detect_mask = np.ones(num_cpi, dtype=bool)

        for idx_cpi in range(num_cpi):
            if not cpi_detect_mask[idx_cpi]:
                continue

            eig_val_db_valid = eig_val_sort_db_array[idx_cpi, :max_deg_freedom]

            # Use per-CPI threshold
            rfi_ev_idx = np.where(eig_val_db_valid > detect_threshold_arr[idx_cpi])[0]

            if rfi_ev_idx.size:
                sig_ev_idx_start = rfi_ev_idx[-1] + 1
                rfi_cpi_flag_array[idx_cpi, :sig_ev_idx_start] = True

    return rfi_cpi_flag_array

def compute_tb_upper_tail(
    diag_power_array,
    diag_valid_array,
    pwr_ref_percentile=50,
    pwr_upper_percentile=99.5,
    eps=1e-12,
):
    pwr = np.asarray(diag_power_array)[diag_valid_array]

    pwr = pwr[np.isfinite(pwr)]

    if pwr.size == 0:
        return np.nan

    # Convert to dB
    pwr_db = 10 * np.log10(np.maximum(pwr, eps))

    # Remove non-finite values
    pwr_db = pwr_db[np.isfinite(pwr_db)]

    # Compute upper-tail spread
    baseline = np.percentile(pwr_db, pwr_ref_percentile)
    upper = np.percentile(pwr_db, pwr_upper_percentile)

    upper_tail_db = upper - baseline

    return upper_tail_db

def compute_tb_condition_number_std(
    eig_val_sort_array,
    min_ev_valid_idx,
    eps=1e-12,
):
    """Compute standard deviation of condition numbers across CPIs in a TB.

    Parameters
    ----------
    eig_val_sort_array : 2D array [num_cpi x cpi_len]
        Sorted eigenvalues in descending order
    min_ev_valid_idx : int
        Index of minimum valid eigenvalue
    eps : float
        Small value to avoid log(0)

    Returns
    -------
    condition_number_std_db : float
        Standard deviation of condition numbers in dB
    """
    num_cpi = eig_val_sort_array.shape[0]
    condition_numbers_db = np.zeros(num_cpi)

    for i in range(num_cpi):
        ev_max = np.maximum(eig_val_sort_array[i, 0], eps)
        ev_min = np.maximum(eig_val_sort_array[i, min_ev_valid_idx], eps)
        condition_numbers_db[i] = 10 * np.log10(ev_max) - 10 * np.log10(ev_min)

    # Remove non-finite values
    condition_numbers_db = condition_numbers_db[np.isfinite(condition_numbers_db)]

    if condition_numbers_db.size == 0:
        return np.nan

    condition_number_std_db = np.std(condition_numbers_db)

    return condition_number_std_db

def compute_tb_max_ev_spread(
    eig_val_sort_array,
    max_deg_freedom,
    eps=1e-12,
):
    """Compute the maximum spread (std across CPIs) among dominant Eigenvalues in a TB.

    Parameters
    ----------
    eig_val_sort_array : 2D array [num_cpi x cpi_len]
        Sorted eigenvalues in descending order
    max_deg_freedom : int
        Number of dominant Eigenvalue indices considered (indices [0, max_deg_freedom))
    eps : float
        Small value to avoid log(0)

    Returns
    -------
    max_ev_spread : float
        Maximum standard deviation (in dB) among the dominant Eigenvalues across CPIs
    """
    eig_val_db_array = 10.0 * np.log10(
        np.maximum(np.real(eig_val_sort_array), eps)
    )

    # Use dominant EVs only
    dominant_ev_db = eig_val_db_array[:, :max_deg_freedom]

    # Standard deviation of each EV index across CPIs
    ev_std = np.std(dominant_ev_db, axis=0)

    # Maximum spread among dominant EVs
    max_ev_spread = np.max(ev_std)

    return max_ev_spread

def compute_tb_eff_rank_std(
    eig_val_sort_array,
    min_ev_valid_idx,
    eps=1e-12,
):
    """Compute standard deviation of per-CPI effective rank across a TB.

    Effective rank is computed via Shannon entropy of the normalized eigenvalue
    distribution over the valid eigenvalue indices [0, min_ev_valid_idx].

    Parameters
    ----------
    eig_val_sort_array : 2D array [num_cpi x cpi_len]
        Sorted eigenvalues in descending order
    min_ev_valid_idx : int
        Index of minimum valid eigenvalue
    eps : float
        Small value to avoid log(0)

    Returns
    -------
    eff_rank_std : float
        Standard deviation of effective rank across CPIs in the TB
    """
    num_cpi = eig_val_sort_array.shape[0]
    eff_rank_array = np.zeros(num_cpi)

    for i in range(num_cpi):
        ev = np.maximum(np.real(eig_val_sort_array[i, :min_ev_valid_idx + 1]), eps)
        p = ev / np.sum(ev)
        p = p[p > 0]
        entropy = -np.sum(p * np.log(p))
        eff_rank_array[i] = np.exp(entropy)

    eff_rank_array = eff_rank_array[np.isfinite(eff_rank_array)]

    if eff_rank_array.size == 0:
        return np.nan

    eff_rank_std = np.std(eff_rank_array)

    return eff_rank_std
