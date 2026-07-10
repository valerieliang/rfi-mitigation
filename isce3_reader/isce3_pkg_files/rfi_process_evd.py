"""
Perform RFI detection and mitigation of input raw data using Slow-Time Eigenvalue Decomposition
(ST-EVD).

DUAL-CHECK VERSION: Uses condition number check AND power spread check
"""
import numpy as np
from isce3.signal.compute_evd_cpi import slice_gen
from isce3.signal.rfi_detection_evd import rfi_detect, ThresholdParams
from isce3.signal.rfi_mitigation_evd import rfi_mitigate_tb
import os
import scipy.io
import warnings
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

def run_slow_time_evd(
    raw_data: np.ndarray,
    cpi_len,
    max_deg_freedom,
    *,
    num_rfi_buffer=2,
    num_max_trim=0,
    num_min_trim=0,
    max_num_rfi_ev=2,
    num_samples_rng_blk=250,
    use_entire_pulse=False,
    threshold_params: ThresholdParams = None,
    num_cpi_per_threshold_block=12,
    off_diag_overlap_ratio=0.20,
    diag_valid_ratio=0.15,
    mitigate_enable=False,
    min_rank_frac=0.70,
    rx_dynamic_range_db=50.0,
    swaths=None,
    threshold_method='max_ev',
    rfi_check=True,
    bt_cond_num_std_thresh_db=3.0,
    max_ev_spread_thresh_db=2.0,
    eff_rank_std_thresh=1.0,
    pwr_ref_percentile=50,
    pwr_upper_percentile=99.5,
    sig_ev_margin_db=1.0,
    rfi_candidate_tolerance_db=3.0,
    raw_data_mitigated=None,
):

    """This is the top-level wrapper which takes raw data in whole or part and does the following:
    1. Partition data into smaller blocks defined as Threshold Block (TB)
       Each TB is consisted of M Coherent Processing Intervals (CPI) and N range samples
       Each CPI is consisted of K slow-time pulses.
    2. Derive slow-time RFI detection threshold for each TB.
       Note: Bright target check (dual condition number + power spread check) is performed
       inside rfi_detect() before threshold computation for all methods.
    3. Mitigate RFI of all CPIs above the detection threshold if mitigation is enabled.

    Parameters
    ------------
    raw_data: array-like complex [num_pulses x num_rng_samples]
        raw data to be processed, supports all numpy complex formats
    cpi_len: int
        Number of slow-time pulses within a CPI
    max_deg_freedom: int
        Max number of independent RFI emitters designed to be detected and mitigated.
        This number should be less than cpi_len to avoid unintended removal of signal data.
    num_rfi_buffer: int, default=2
        Number of buffer eigenvalue indices to skip after last possible RFI EV before
        starting clean segment interpolation for 'max_ev' method. The clean segment
        starts at index (max_deg_freedom + num_rfi_buffer - 1).
    num_max_trim: int, default=0
        Number of large value outliers to be trimmed in slow-time minimum Eigenvalues.
    num_min_trim: int, default=0
        Number of small value outliers to be trimmed in slow-time minimum Eigenvalues
    max_num_rfi_ev: int, default=2
        A detection error (miss) happens when a maximum power RFI emitter contaminates
        multiple consecutive CPIs, resulting in a flat maximum Eigenvalue slope in slow
        time. Hence the standard (STD) deviation of multiple dominant EVs across slow time
        defined by this parameter are compared. The one with the maximum STD is used for RFI
        Eigenvalue first difference computation.
    num_samples_rng_blk: int, default=250
        Number of range samples per range block when data blockin is applied in range direction
        for sample covariance matrix estimation. It is recommended that this parameter is at
        least 5 x cpi_len to avoid discrepancy from true sample covaraince matrix. In addition,
        in order to avoid a run-time error for ST-EVD, this parameter needs to be
        at least 2 x cpi_len.
    use_entire_pulse: bool, default=False
        Ignore any value passed for num_samples_rng_blk and instead use all samples
        in the slow-time pulses for detection if this is True.
    threshold_params: ThresholdParams object or None, default=None
        RFI detection threshold interpolation parameters. If None, a method-specific
        default is selected: ThresholdParams() (x=[2.0, 20.0], y=[5.0, 2.0]) for
        'ev_slope', or ThresholdParams(x=[1.0, 10.0], y=[1.0, 0.95]) for 'aer'.
        For 'ev_slope', the x field defines the STD ratio between maximum and
        minimum Eigenvalue slopes (MMES) of the slow-time threshold interval, and
        the y field defines the number of sigma (STD) from the mean of MMES. For
        'aer', the x field is the AER figure of merit (STD of per-CPI max/min
        Eigenvalue dB spread), and the y field is the quantile probability (0 to 1)
        used to derive the absolute Eigenvalue detection threshold.
    num_cpi_per_threshold_block: int, default=12
        Number of slow-time CPIs in a TB
    off_diag_overlap_ratio : float, optional, default=0.20
        Minimum overlap ratio used by gap exclusion covariance estimation
    diag_valid_ratio : float, optional, default=0.15
        Minimum fraction of valid samples required to compute a diagonal term in the
        sample covariance matrix entry R_ii.
    mitigate_enable: bool, default=False
        Enable mitigation
    min_rank_frac: float, default = 0.7
        This fraction will be used to determine the minimum number of valid Eigenvalues
        required for a CPI. min_ev_valid_idx = int(np.floor(min_rank_frac * cpi_len))
        Must be a value within (0,1]
    rx_dynamic_range_db: float, optional, default = 50 dB
        radar platform receiver dynamic range in dB. This is applied as a threshold
        to determine if the Eigenvalue under test is meaningfully signficant. If the
        Eigenvalue under test is less than this threshold, it will be viewed as unusable.
    swaths : np.ndarray [int], optional
        Valid subswath samples, dims = (ns, nt, 2) where ns is the number of
        sub-swaths, nt is the number of pulses, and the trailing dimension is
        the [start, stop) indices of the sub-swath.  It's recommended to supply
        this for modes with dithered PRI, where it will be used to normalize
        the sample covariance matrix.
    threshold_method : str, default='max_ev'
        RFI detection method: 'ev_slope' (Eigenvalue Slope Thresholding),
        'max_ev' (TB-averaged EV regression extrapolated to EV[0]), or
        'aer' (absolute Eigenvalue quantile threshold based on per-CPI
        max/min Eigenvalue dB spread).
    rfi_check : bool, default=True
        Controls RFI-presence characterization. If False, no check is performed
        and all TBs proceed to RFI detection. Otherwise, a TB is screened for
        RFI-like Eigenvalue traits and skipped if RFI is
        determined to not be present, based on dominant Eigenvalue spread, condition
        number variability, and effective rank variability.
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
        Represents the median power level of the diagonal covariance entries.
    pwr_upper_percentile : float, default=99.5
        Upper percentile for power spread computation.
        Recommended values: 98.0 (top 2%) or 99.5 (top 0.5%).
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
        All TBs proceed to RFI detection (no RFI-trait screening).
    rfi_check=True:
        rfi_present = True  if max_ev_spread_db > max_ev_spread_thresh_db
        rfi_present = True  elif (condition_number_std_db > bt_cond_num_std_thresh_db
                                   and eff_rank_std > eff_rank_std_thresh)
        rfi_present = False otherwise
        SKIP TB if rfi_present is False
    raw_data_mitigated: array-like complex [num_pulses x num_rng_samples] or None, optional
        output array in which the mitigated data values is placed. It
        must be an array-like object supporting `multidimensional array access
        <https://numpy.org/doc/stable/user/basics.indexing.html>`_.
        The array should have the same shape and dtype as the input raw data array.
        If None (the default), the input 'raw_data' will be modified in-place.


    Returns
    --------
    rfi_likelihood: float
        Ratio of number of CPIs detected with RFI Eigenvalues over that of total number
        of CPIs.
    Notes
    -----
    If the number of pulses is not an integer multiple of the CPI length,
    any remaining pulses after the last full CPI will be unmitigated.

    References
    ----------
    ..[1] Bo Huang, Heresh Fattahi, Hirad Ghaemi, Brian Hawkins, Geoffrey Gunter,
    "Radio Frequency Interference Detection and Mitigation of NISAR DATA using
    Slow Time Eigenvalue Decomposition", IGARSS 2023.'
    """

    # Select method-specific threshold_params default if not explicitly provided
    if threshold_params is None:
        if threshold_method == 'aer':
            threshold_params = ThresholdParams(x=[1.0, 10.0], y=[1.0, 0.95])
        else:
            threshold_params = ThresholdParams()

    num_pulses, num_rng_samples = raw_data.shape

    # Override num_rng_samples_blk if use_entire_pulse is True
    if use_entire_pulse:
        num_samples_rng_blk = num_rng_samples

    # If the number of pulses is not an integer multiple of TB size, following
    # operations will ensue. If the number of remaining pulses is greater than
    # the CPI length, additional CPI(s) will be constructed, the very last TB will
    # include the additional CPI(s). The rest of the remaining pulses not enough to
    # construct a full CPI will not be processed. If the number of remaining pulses
    # is less than CPI length, then they will not be processed. In both cases,
    # at most cpi_len-1 number of pulses will be ignored.

    num_cpi = num_pulses // cpi_len
    num_pulses_proc = cpi_len * num_cpi
    num_pulses_tb = cpi_len * num_cpi_per_threshold_block
    num_tb = num_pulses_proc // num_pulses_tb

    # Figue out how many range slices are there
    rng_slices = list(
        slice_gen(num_rng_samples, num_samples_rng_blk, combine_rem=True)
    )
    num_rng_blks = len(rng_slices)

    # RFI EV Count Map
    rfi_ev_count_map = np.zeros(
        (num_cpi, num_rng_blks),
        dtype=np.int16,
    )

    # Boolean Detection Map
    rfi_cpi_detection_map = np.zeros(
        (num_cpi, num_rng_blks),
        dtype=bool,
    )

    figure_merit_array = np.zeros((num_tb, num_rng_blks), dtype=np.float32)

    tb_skipped_map = np.zeros((num_tb, num_rng_blks), dtype=np.bool_)

    rfi_present_map = np.zeros((num_tb, num_rng_blks), dtype=np.bool_)

    cond_num_std_db_map = np.full((num_tb, num_rng_blks), np.nan, dtype=np.float32)

    power_spread_db_map = np.full((num_tb, num_rng_blks), np.nan, dtype=np.float32)

    rfi_candidate_cpi_count_map = np.full((num_tb, num_rng_blks), np.nan, dtype=np.float32)

    # Modify raw_data in-place
    if raw_data_mitigated is None:
        raw_data_mitigated = raw_data
    else:
        if raw_data_mitigated.shape != raw_data.shape:
            raise ValueError(
                "Shape mismatch: output mitigated data array must have the same shape"
                " as the input data"
            )

    # Verify min_rank_frac
    if not (0.0 < min_rank_frac <= 1.0):
        raise ValueError(
            f"min_rank_frac must be in (0, 1], got {min_rank_frac}."
        )

    # Collect total number of CPI range blocks contaminated by RFI
    rfi_cpi_count_sum = 0

    # Verify total number of pulses is equal or greater than number of pulses per TB
    if num_pulses < num_pulses_tb:
        raise ValueError(
            "Total number of pulses must be greater or equal to that of a threshold block."
        )

    # Maximum number of degrees of freedom must be less than cpi_len
    if max_deg_freedom >= cpi_len:
        raise ValueError(
            "Max number of deg. of freedom must be less than number of pulses in a CPI."
        )

    # Verify mask_valid: check to see if it is None or populated.
    if (swaths is not None) and (swaths.shape[1] != raw_data.shape[0]):
        raise ValueError("Require same number of rows in swaths and raw_data")

    # Determine a valid Eigenvalue index to estimate minimum-Eigenvalue statistics,
    # ensuring robustness against zero Eigenvalues caused by insufficient valid samples in a CPI.
    min_ev_valid_idx = max(1, int(np.round(min_rank_frac * cpi_len)) - 1)

    # Maximum number of degrees of freedom must be less than max_deg_freedom
    if max_deg_freedom >= min_ev_valid_idx:
        warnings.warn(
            f"max_deg_freedom ({max_deg_freedom}) >= min_ev_valid_idx ({min_ev_valid_idx})."
            "This is not recommended since it may lead to insufficient noise subspace.",
            RuntimeWarning
        )

    # ========== DEBUG DATA SAVE CONFIGURATION ==========
    # Change these directories as needed for debugging
    raw_data_dir = './data_07_03/raw_data/north_texas/hv'
    mask_dir = './data_07_03/mask/north_texas/hv'
    # Create directories if they don't exist
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    # ====================================================

    # Run RFI Detection and Mitigation
    for idx_tb, tb_slow_time in enumerate(slice_gen(num_pulses_proc, num_pulses_tb)):
        # Get valid data mask for all rows in current block.
        if swaths is not None:
            swaths_tb = swaths[:, tb_slow_time, :]
            mask_valid = np.zeros((swaths_tb.shape[1], raw_data.shape[1]), dtype=bool)
            for i in range(mask_valid.shape[0]):
                for start, end in swaths_tb[:, i, :]:
                    mask_valid[i, start:end] = True

        for idx_rng, tb_fast_time in enumerate(rng_slices):
            print(f'{idx_tb = } {idx_rng = }\n')
            raw_tb_blk = raw_data[tb_slow_time, tb_fast_time]
            mask_valid_tb = None if swaths is None else mask_valid[:, tb_fast_time]

            #if 33 < idx_tb <= 46 and 235 < idx_rng <= 241:
            #    scipy.io.savemat(
            #         f'{raw_data_dir}/raw_tb_blk_az_{idx_tb}_rng_{idx_rng}.mat',
            #         {'raw_tb_blk': raw_tb_blk},
            #     )
            #    if mask_valid_tb is not None:
            #        scipy.io.savemat(
            #             f'{mask_dir}/mask_valid_az_{idx_tb}_rng_{idx_rng}.mat',
            #             {'mask_valid_tb': mask_valid_tb},
            #        )

            #if 87 < idx_tb <= 96 and 218 < idx_rng <= 225:
            #    scipy.io.savemat(
            #         f'{raw_data_dir}/raw_tb_blk_az_{idx_tb}_rng_{idx_rng}.mat',
            #         {'raw_tb_blk': raw_tb_blk},
            #     )
            #    if mask_valid_tb is not None:
            #        scipy.io.savemat(
            #             f'{mask_dir}/mask_valid_az_{idx_tb}_rng_{idx_rng}.mat',
            #             {'mask_valid_tb': mask_valid_tb},
            #        )

            (
                rfi_cpi_flag_tb,
                evec_sort_tb,
                _,  # diag_power_array (unused, power check now in rfi_detect)
                _,  # diag_valid_array (unused, power check now in rfi_detect)
                figure_merit_tb,
                bt_skipped_tb,
                cond_num_std_db_tb,
                upper_tail_spread_db_tb,
                rfi_present_tb,
                num_rfi_candidate_cpi_tb,
            ) = rfi_detect(
                raw_tb_blk,
                cpi_len,
                max_deg_freedom,
                min_ev_valid_idx,
                num_rfi_buffer=num_rfi_buffer,
                num_max_trim=num_max_trim,
                num_min_trim=num_min_trim,
                max_num_rfi_ev=max_num_rfi_ev,
                off_diag_overlap_ratio=off_diag_overlap_ratio,
                diag_valid_ratio=diag_valid_ratio,
                rx_dynamic_range_db=rx_dynamic_range_db,
                mask_valid=mask_valid_tb,
                threshold_method=threshold_method,
                threshold_params=threshold_params,
                rfi_check=rfi_check,
                bt_cond_num_std_thresh_db=bt_cond_num_std_thresh_db,
                max_ev_spread_thresh_db=max_ev_spread_thresh_db,
                eff_rank_std_thresh=eff_rank_std_thresh,
                pwr_ref_percentile=pwr_ref_percentile,
                pwr_upper_percentile=pwr_upper_percentile,
                sig_ev_margin_db=sig_ev_margin_db,
                rfi_candidate_tolerance_db=rfi_candidate_tolerance_db,
            )

            # Global CPI indices for this threshold block
            cpi_start = idx_tb * num_cpi_per_threshold_block
            cpi_end = cpi_start + rfi_cpi_flag_tb.shape[0]

            # Power stationarity check is now handled inside rfi_detect()
            # Detection returns no RFI (all zeros) for TBs with no RFI traits automatically
            has_rfi = np.any(rfi_cpi_flag_tb)

            # Compute number of CPIs detected with RFI presence
            num_rfi_ev_cpi = np.sum(rfi_cpi_flag_tb, axis=1).astype(np.int16)
            rfi_cpi_count = np.sum(num_rfi_ev_cpi != 0)
            rfi_cpi_count_sum += rfi_cpi_count

            figure_merit_array[idx_tb, idx_rng] = figure_merit_tb
            tb_skipped_map[idx_tb, idx_rng] = bt_skipped_tb
            rfi_present_map[idx_tb, idx_rng] = rfi_present_tb
            cond_num_std_db_map[idx_tb, idx_rng] = cond_num_std_db_tb
            power_spread_db_map[idx_tb, idx_rng] = upper_tail_spread_db_tb
            rfi_candidate_cpi_count_map[idx_tb, idx_rng] = num_rfi_candidate_cpi_tb
            rfi_ev_count_map[cpi_start:cpi_end, idx_rng] = num_rfi_ev_cpi
            rfi_cpi_detection_map[cpi_start:cpi_end, idx_rng] = num_rfi_ev_cpi > 0

            #if 467 < idx_tb <= 472 and 74 < idx_rng <= 90:
            #    os.makedirs('./data_12_29/HH/det', exist_ok=True)
                # scipy.io.savemat(
                #     f'./data_12_29/HH/det/bt_map_tb_blk_az_{idx_tb}_rng_{idx_rng}.mat',
                #     {'bt_skipped_tb': bt_skipped_tb},
                # )
                # scipy.io.savemat(
                #     f'./data_12_29/HH/det/fom_tb_blk_az_{idx_tb}_rng_{idx_rng}.mat',
                #     {'figure_merit_tb': figure_merit_tb},
                # )
                # scipy.io.savemat(
                #     f'./data_12_29/HH/det/cond_num_std_db_az_{idx_tb}_rng_{idx_rng}.mat',
                #     {'cond_num_std_db_tb': cond_num_std_db_tb},
                # )
                # scipy.io.savemat(
                #     f'./data_12_29/HH/det/upper_tail_spread_db_az_{idx_tb}_rng_{idx_rng}.mat',
                #     {'upper_tail_spread_db_tb': upper_tail_spread_db_tb},
                # )

            # Run Mitigation:
            if mitigate_enable and has_rfi:
                rfi_mitigate_tb(
                    raw_tb_blk,
                    evec_sort_tb,
                    rfi_cpi_flag_tb,
                    raw_data_mitigated[tb_slow_time, tb_fast_time],
                )
            else:
                raw_data_mitigated[tb_slow_time, tb_fast_time] = raw_tb_blk

   #os.makedirs('./data_12_29/HH/det', exist_ok=True)
    # scipy.io.savemat(
    #     './data_12_29/HH/det/figure_merit_array.mat',
    #     {'figure_merit_array': figure_merit_array},
    # )
    # scipy.io.savemat(
    #     './data_12_29/HH/det/tb_skipped_map.mat',
    #     {'tb_skipped_map': tb_skipped_map},
    # )
    # scipy.io.savemat(
    #     './data_12_29/HH/det/rfi_present_map.mat',
    #     {'rfi_present_map': rfi_present_map},
    # )
    # scipy.io.savemat(
    #     './data_12_29/HH/det/rfi_ev_count_map.mat',
    #     {'rfi_ev_count_map': rfi_ev_count_map},
    # )
    # scipy.io.savemat(
    #     './data_12_29/HH/det/rfi_cpi_detection_map.mat',
    #     {'rfi_cpi_detection_map': rfi_cpi_detection_map},
    # )
    # scipy.io.savemat(
    #     './data_12_29/HH/det/cond_num_std_db_map.mat',
    #     {'cond_num_std_db_map': cond_num_std_db_map},
    # )
    # scipy.io.savemat(
    #     './data_12_29/HH/det/power_spread_db_map.mat',
    #     {'power_spread_db_map': power_spread_db_map},
    # )

    # Percentage of RFI Eigenvalues based on slow-time min EV slope detection
    rfi_likelihood = rfi_cpi_count_sum / (num_cpi * num_rng_blks)
    print(f'{rfi_likelihood = } \n')

    # Fill the remaining few pulses with original raw data samples
    if num_pulses > num_pulses_proc:
        raw_data_mitigated[num_pulses_proc:] = raw_data[num_pulses_proc:]

    #plot_detect = 0
    plot_detect = 1

    if plot_detect:
        # Map rfi_check to display label and filename suffix
        rfi_check_mode_label = str(rfi_check)
        rfi_check_mode_fname = str(rfi_check).lower()

        #plot_dir = os.path.join(
        #    '/home/bohuang/plots/06_28/la_12_29/HV/',
        #    threshold_method,
        #    f'rfi_check_{rfi_check_mode_fname}',
        #)

        plot_dir = os.path.join(
            '/scratch/bohuang/rfi/data/amazon/HV/',
            threshold_method,
            f'rfi_check_{rfi_check_mode_fname}',
        )

        os.makedirs(plot_dir, exist_ok=True)

        # TB skipped map: red = TB skipped (True), black = not skipped (False)
        cmap_bt = ListedColormap(['black', 'red'])
        legend_elements_bt = [
            Patch(facecolor='black', label='Not Skipped'),
            Patch(facecolor='red', label='Skipped'),
        ]
        plt.figure()
        plt.imshow(tb_skipped_map, cmap=cmap_bt, aspect='auto')
        plt.title(f'TB Skipped Map: {threshold_method}, cpi_len={cpi_len} (rfi_check={rfi_check_mode_label})')
        plt.xlabel('Range Block Index')
        plt.ylabel('Threshold Block Index')
        plt.legend(handles=legend_elements_bt, loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'tb_skipped_map_{threshold_method}_cpi{cpi_len}_{rfi_check_mode_fname}.png'))

        # RFI present map: red = RFI present (True), black = no RFI (False)
        cmap_rfi_present = ListedColormap(['black', 'red'])
        legend_elements_rfi_present = [
            Patch(facecolor='black', label='No RFI'),
            Patch(facecolor='red', label='RFI Present'),
        ]
        plt.figure()
        plt.imshow(rfi_present_map, cmap=cmap_rfi_present, aspect='auto')
        plt.title(f'RFI Present Map: {threshold_method}, cpi_len={cpi_len} (rfi_check={rfi_check_mode_label})')
        plt.xlabel('Range Block Index')
        plt.ylabel('Threshold Block Index')
        plt.legend(handles=legend_elements_rfi_present, loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'rfi_present_map_{threshold_method}_cpi{cpi_len}_{rfi_check_mode_fname}.png'))

        #plt.figure()
        #plt.imshow(figure_merit_array, interpolation='nearest', cmap='viridis', vmin=0, vmax=np.percentile(figure_merit_array,99), aspect='auto')
        #plt.title(f'RFI Detection Figure of Merit: {threshold_method}, cpi_len={cpi_len}, Likelihood: {rfi_likelihood:.2%}\nrfi_check={rfi_check_mode_label}')
        #plt.xlabel('Range Block')
        #plt.ylabel('Azimuth Block')
        #plt.colorbar(label='Detection Figure of Merit')
        #plt.tight_layout()
        #plt.savefig(os.path.join(plot_dir, f'fig_merit_{threshold_method}_cpi{cpi_len}_{rfi_check_mode_fname}.png'))

        # One color for each RFI EV count
        colors = [
            "#dbe9f6",  # 0 : Clean (light blue)
            "#ffe680",  # 1 : Yellow
            "#ffb347",  # 2 : Orange
            "#ff6666",  # 3 : Red
            "#cc66cc",  # 4 : Magenta
            "#8a2be2",  # 5 : Purple
            "#333333",  # 6 : Black
        ]

        # Extend automatically if max_deg_freedom > 6
        if max_deg_freedom + 1 > len(colors):
            colors.extend(["#000000"] * (max_deg_freedom + 1 - len(colors)))

        cmap = ListedColormap(colors[:max_deg_freedom + 1])
        norm = BoundaryNorm(
            np.arange(-0.5, max_deg_freedom + 1.5, 1),
            cmap.N
        )

        plt.figure(figsize=(12, 8))

        plt.imshow(
            rfi_ev_count_map,
            interpolation='nearest',
            cmap=cmap,
            norm=norm,
            aspect='auto'
        )

        plt.title(
            f'RFI Eigenvalue Count Map: {threshold_method}, cpi_len={cpi_len}\n'
            f'rfi_check={rfi_check_mode_label}'
        )

        plt.xlabel('Range Block Index')
        plt.ylabel('CPI Index')

        cbar = plt.colorbar(
            ticks=np.arange(max_deg_freedom + 1)
        )
        cbar.set_label('Number of RFI Eigenvalues')

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'rfi_ev_count_{threshold_method}_cpi{cpi_len}_{rfi_check_mode_fname}.png'))

        cmap_rfi = ListedColormap(['black', 'red'])
        legend_elements_rfi = [
            Patch(facecolor='black', label='No RFI'),
            Patch(facecolor='red', label='RFI Detected'),
        ]
        plt.figure()
        plt.imshow(rfi_cpi_detection_map, cmap=cmap_rfi, aspect='auto')
        plt.title(f'RFI Detection Map: {threshold_method}, cpi_len={cpi_len} (rfi_check={rfi_check_mode_label})')
        plt.xlabel('Range Block Index')
        plt.ylabel('CPI Index')
        plt.legend(handles=legend_elements_rfi, loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'rfi_det_map_{threshold_method}_cpi{cpi_len}_{rfi_check_mode_fname}.png'))

        #plt.figure()
        #plt.imshow(cond_num_std_db_map, interpolation='nearest', cmap='viridis', aspect='auto')
        #plt.title(f'Condition Number STD Map (dB): {threshold_method}, cpi_len={cpi_len}\nrfi_check={rfi_check_mode_label}')
        #plt.xlabel('Range Block Index')
        #plt.ylabel('Threshold Block Index')
        #plt.colorbar(label='Condition Number STD (dB)')
        #plt.tight_layout()
        #plt.savefig(os.path.join(plot_dir, f'cond_num_std_db_map_{threshold_method}_cpi{cpi_len}_{rfi_check_mode_fname}.png'))

        #plt.figure()
        #plt.imshow(power_spread_db_map, interpolation='nearest', cmap='viridis', aspect='auto')
        #plt.title(f'Upper Tail Power Spread Map (dB): {threshold_method}, cpi_len={cpi_len}\nrfi_check={rfi_check_mode_label}')
        #plt.xlabel('Range Block Index')
        #plt.ylabel('Threshold Block Index')
        #plt.colorbar(label='Upper Tail Spread (dB)')
        #plt.tight_layout()
        #plt.savefig(os.path.join(plot_dir, f'power_spread_db_map_{threshold_method}_cpi{cpi_len}_{rfi_check_mode_fname}.png'))

        if threshold_method == 'max_ev':
            max_rfi_candidate_cpi = num_cpi_per_threshold_block
            rfi_candidate_cpi_colors = [
                "#dbe9f6",  # 0  : light blue (no RFI candidates)
                "#ffe680",  # 1  : yellow
                "#ffb347",  # 2  : orange
                "#ff6666",  # 3  : red
                "#cc66cc",  # 4  : magenta
                "#8a2be2",  # 5  : purple
                "#4b0082",  # 6  : indigo
                "#2e8b57",  # 7  : sea green
                "#008080",  # 8  : teal
                "#191970",  # 9  : midnight blue
                "#8b4513",  # 10 : saddle brown
                "#556b2f",  # 11 : dark olive green
                "#000000",  # 12 : black (all CPIs are RFI candidates)
            ]
            if max_rfi_candidate_cpi + 1 > len(rfi_candidate_cpi_colors):
                rfi_candidate_cpi_colors.extend(["#000000"] * (max_rfi_candidate_cpi + 1 - len(rfi_candidate_cpi_colors)))
            cmap_rfi_candidate_cpi = ListedColormap(rfi_candidate_cpi_colors[:max_rfi_candidate_cpi + 1])
            norm_rfi_candidate_cpi = BoundaryNorm(
                np.arange(-0.5, max_rfi_candidate_cpi + 1.5, 1),
                cmap_rfi_candidate_cpi.N,
            )
            plt.figure(figsize=(12, 8))
            plt.imshow(
                rfi_candidate_cpi_count_map,
                interpolation='nearest',
                cmap=cmap_rfi_candidate_cpi,
                norm=norm_rfi_candidate_cpi,
                aspect='auto',
            )
            plt.title(f'RFI Candidate CPI Count Map (max_ev): cpi_len={cpi_len}\nrfi_check={rfi_check_mode_label}')
            plt.xlabel('Range Block Index')
            plt.ylabel('Threshold Block Index')
            cbar_rfi_candidate = plt.colorbar(ticks=np.arange(max_rfi_candidate_cpi + 1))
            cbar_rfi_candidate.set_label('Number of RFI Candidate CPIs')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'rfi_candidate_cpi_count_map_{threshold_method}_cpi{cpi_len}_{rfi_check_mode_fname}.png'))

        plt.show()

    return rfi_likelihood
