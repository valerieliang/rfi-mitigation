import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import argparse
import time
import tempfile
import h5py
from os import path
from nisar.products.readers.Raw import Raw, open_rrsd
from isce3.core.types import to_complex32, is_complex32, read_c4_dataset_as_c8
from isce3.signal.rfi_process_evd import run_slow_time_evd
from isce3.signal.rfi_detection_evd import (
    ThresholdParams,
)
from isce3.signal.compute_evd_cpi import slice_gen
from numpy.fft import fft, fftshift, ifft, ifftshift, fftfreq
from isce3.signal.rfi_freq_null import run_freq_notch
#import nisar.workflows.focus
from isce3 import focus

from numpy.fft import fft, fftshift
import scipy.io as sio
import time
import shutil
import scipy.io as sio
from scipy.fft import fftshift, fft, ifft



def cmd_line_parse():
    parser = argparse.ArgumentParser(description="Input L0B file.")

    parser.add_argument(
        '-i',
        '--input', 
        dest='input_file', 
        type=str,
        required=True, 
        help='Input L0B file'
    )
    parser.add_argument(
        "-c",
        "--cpi",
        dest="cpi_len",
        type=int,
        default="16",
        required=False,
        help="Number of pulses in a coherent processing interval. Deafult: 32",
    )
    parser.add_argument(
        "-p",
        "--pols",
        dest="pols",
        type=str,
        default="HH",
        required=False,
        help="Polarization(s) to process. Default: HH"
    )
    #parser.add_argument(
    #    "-r",
    #    "--nrs",
    #    dest="num_rng_samples",
    #    type=int,
    #    default=256,
    #    required=False,
    #    help="Number of range samples per range block. Default: 256.",
    #)
    parser.add_argument(
        "-r",
        "--nrs",
        dest="num_rng_samp_blk",
        type=int,
        default=1000,
        required=False,
        help="Number of range samples per range block. Default: 1000.",
    )
    parser.add_argument(
        "-t",
        "--tb",
        dest="num_cpi_tb",
        type=int,
        default="12",
        required=False,
        help="RFI detection threshold block size in number of CPI."
    )
    parser.add_argument(
        "-a",
        "--at",
        dest="num_tb_az_blk",
        type=int,
        default="1",
        required=False,
        help="Number of threshold block in a Azimuth processing block"
    )
    parser.add_argument(
        "-m",
        "--miti",
        dest="mitigation",
        type=str,
        default="evd",
        required=False,
        help="Select RFI mitigation algorithm"
    )
    parser.add_argument(
        "-w",
        "--caltone",
        dest="caltone_rm_algo",
        type=str,
        default='wavelet',
        required=False,
        help="Select caltone removal algorithm, default: wavelet"
    )
    return parser.parse_args()


def read_raw_data(raw_file, pol, freq_group):
    raw = Raw(hdf5file=raw_file)
    raw_data = raw.getRawDataset(freq_group, pol)

    raw_data = raw_data[:]

    return raw_data


def read_raw_h5py(
    raw_file, pols, 
    freq_group, 
    #num_pulses_blk,
    pulse_start,
    pulse_stop,
):
    tx_pol = pols[0]
    rx_pol = pols[1]

    dset_path = f'/science/LSAR/RRSD/swaths/frequency{freq_group}/tx{tx_pol}/rx{rx_pol}/{pols}'

    print(f'{dset_path = }')

    #with h5py.File(raw_file, "r") as f:
    #    dset = f[dset_path]

    #    for pulse in range(pulse_start, pulse_stop, num_pulse_blk):
    #        pulse_blk_stop = min(pulse + num_pulses_blk, pulse_stop)

    #        pulse_blk = dset[pulse : pulse_blk_stop, :]

    #        pulse_blk_data = (pulse_blk["r"].astype(np.float32) + 1j * pulse_blk["i"].astype(np.float32)).astype(np.complex64)

    if pulse_stop <= pulse_start:
            raise ValueError("row1 must be > row0")

    with h5py.File(raw_file, "r") as f:
        dset = f[dset_path]

        blk = dset[pulse_start : pulse_stop, :]

        pulse_blk = np.empty(blk.shape, dtype=np.complex64)

        pulse_blk.real = blk["r"]
        pulse_blk.imag = blk["i"]


    return pulse_blk

def read_subswath(raw, freq, pols, pulse_idx):
    tx_pol = pols[0]
    swaths = raw.getSubSwaths(freq, tx_pol)
    swaths = swaths[:, pulse_idx, :]

    return swaths

def read_subswath_mask(raw, freq, pols, pulse_idx):
    #raw = Raw(hdf5file=input_file)
    #raw.parsePolarizations()
    zero_as_gap = True

    tx_pol = pols[0]
    print(f'Start Processing: frequency{freq} {pols}')

    raw_data = raw.getRawDataset(freq, pols)
    subswaths = raw.getSubSwaths(freq, tx_pol)

    _, grid = raw.getRadarGrid(freq, tx_pol)
    num_pulses, num_rng_samples = raw_data.shape

    print(f'{raw_data.shape = }')
    print(f'{subswaths.shape = }')
    print(f'{grid.shape = }')
    print(f'{grid.width = }')
    print()

    mask_subswath = generate_valid_data_mask(
        pulse_idx,
        subswaths, 
        num_rng_samples
    )

    # Read a portion of raw data using pulse_idx
    raw_data_blk = raw_data[pulse_idx, :]
    print(f'{raw_data_blk.shape = }\n')

    mask_from_data = build_gap_mask_from_data(
        raw_data_blk,
        zero_as_gap,
    )

    #plt.figure()
    #plt.imshow(mask_subswath[:, ::150])
    #plt.title('Mask from data: Re-grouped Pulses')
    #plt.ylabel('Azimuth')
    #plt.xlabel('Range')

    #plt.figure()
    #plt.imshow(mask_from_data[:, ::150])
    #plt.title('Mask from data: Re-grouped Pulses')
    #plt.ylabel('Azimuth')
    #plt.xlabel('Range')
    #plt.show()

    return mask_subswath, mask_from_data


def generate_valid_data_mask(
    pulse_idx: np.ndarray,
    subswaths: np.ndarray, 
    num_rng_samples: int
):

    #print(f'{pulse_idx = }\n')
    print(f'{subswaths.shape = }\n')
    print()
    num_pulses_mask = len(pulse_idx)
    mask = np.zeros((num_pulses_mask, num_rng_samples), dtype=bool)

    for imask, ipulse in enumerate(pulse_idx):
        #print()
        #print(f'{imask = }')
        #print(f'{ipulse = }\n')
        for subswath in subswaths:
            #print()
            #print(f'{subswath.shape = }')
            start, end = subswath[ipulse]
            #print(f'{start = }')
            #print(f'{end = }')
            #print()
            mask[imask, start:end] = True

    return mask


def build_gap_mask_from_data(
    X: np.ndarray,
    zero_as_gap: bool = True
) -> np.ndarray:

    finite = np.isfinite(X)
    if zero_as_gap:
        nonzero = (np.abs(X) > 0)
        mask = finite & nonzero

        return mask
    else:
        return finite

def parse_caltone_freq_from_drt(
    raw: Raw,
    txrx_pol: str
) -> float:

    default = 1214.88e6

    lo = 1200e6

    clock = 240e6

    c_p = (f'{raw.TelemetryPath}/DRT/MISC/CP_IFSW_CALTONE_PHASE_STEP_'
        f'{txrx_pol[1]}')

    with h5py.File(raw.filename, mode='r', swmr=True) as f5:
        try:
            ds_caltone_phase = f5[c_p]
        except KeyError:
            warn(f'Missing path "{c_p}" in L0B! Caltone frequency will '
                f'be set to {default} (Hz)')
            return default
        else:
            i_cal = np.median(ds_caltone_phase[()]).astype(int)
            caltone_freq = (i_cal / 2**32) * clock + lo
            return caltone_freq
        

def process_nisar_data(
    raw: Raw,
    input_file: str,
    num_samples_rng_blk: int = 200,
    cpi_len: int = 15,
    num_cpi_tb: int = 12,
    num_tb_az_proc_blk: int = 2,
    mitigation_algo: str = 'evd',
    caltone_rm_algo: str = 'wavelet',
    freq: str = 'A',
    pols: str = 'HH',
):

    # Input chirp paramters
    pol_tx = pols[0]
    fc, fs, _, _ = raw.getChirpParameters(freq, pol_tx)
    #fc = raw.getCenterFrequency(freq, pol_tx)
    #fcal = 1216.904e6
    #fcal = 1214.882813e6
    fcal = parse_caltone_freq_from_drt(raw, pols)

    print(f'{mitigation_algo = }')
    print(f'{caltone_rm_algo = }')
    print(f'{fc = }')
    print(f'{fs = }')
    print(f'{pols = }')
    #print(f'{caltone_freq = }')
    print(f'{fcal = }\n')
    print(f'{raw.isDithered(freq, pol_tx) = }\n')
    

    # ST-EVD Parameters
    max_deg_freedom = 3
    num_max_trim = 0
    num_min_trim = 0
    max_num_rfi_ev = 2
    num_samples_rng_blk = 200
    num_cpi_per_threshold_block = 10
    use_entire_pulse = False
    mitigate_enable = False
    prf_dither_mode = raw.isDithered(freq, pol_tx) 
    #prf_dither_mode = True
    min_rank_frac = 0.70
    rx_dynamic_range_db = 50
    off_diag_overlap_ratio=0.2
    diag_valid_ratio=0.15
    #threshold_method='ev_slope'
    #threshold_method='aer'
    threshold_method='max_ev'
    #rfi_fig_merit_thresh = 1.5
    #bright_tgt_check_mode = 'condition_num'
    #bright_tgt_check_mode = 'both'
    rfi_check = True
    bt_cond_num_std_thresh_db = 2.5
    max_ev_spread_thresh_db = 2
    eff_rank_std_thresh = 1
    #bt_pwr_spread_thresh_db = 6.0
    pwr_ref_percentile = 50.0
    pwr_upper_percentile = 99.5
    sig_ev_margin_db = 1.0
    #cond_alpha = 2.0
    rfi_candidate_tolerance_db = 4.0
    #sig_ev_margin_upper_db = 3.0
    #sig_ev_margin_lower_db = 0
    #pcr_range = [0.1, 0.35]
    num_rfi_buffer = 1
    swaths = None

    if threshold_method == 'aer':
        threshold_params = ThresholdParams([1, 10], [1, 0.8])
    else:
        threshold_params = ThresholdParams([2, 20],[5, 2])

    #if threshold_method == 'max_ev':
    #    if fom_method == 'est':
    #        threshold_params = ThresholdParams([2, 20],[5, 2])
    #    elif fom_method == 'aer':
    #        threshold_params = ThresholdParams([1, 10], [1, 0.9])


    # Read a chunk of L0B data

    # Colorado: [623753, 697802]
    #pulse_start = 623753
    #pulse_stop = 697802

    #pulse_stop = 642440

    #pulse_stop = 630000


    # Texas: [130581, 204600]
    #pulse_start = 130581 
    #pulse_stop = 204600
    #pulse_stop = 156860

    # LA Constant PRF: 10/06/2025
    #pulse_start = 46528
    #pulse_stop = 124580

    #pulse_start = 64000
    #pulse_stop = 86000

    # LA 12/23/2025: [72930, 148895]
    #pulse_start = 72930
    #pulse_stop = 148895

    # LA 12/29/2025: [681445, 757408]
    #pulse_start = 681445
    #pulse_stop = 757408

    #pulse_start = 700000
    #pulse_stop = 730000

    # LA 01/04/2026:  [74192, 148222]
    #pulse_start = 74192
    #pulse_stop = 148222

    # Montana: [804782, 880766]
    #pulse_start = 804782
    #pulse_stop = 880766

    # Northern Texas: [357079, 429252]
    #pulse_start = 357079
    #pulse_stop = 429252
    #pulse_start =  393361
    #pulse_stop = 421612

    #[418184, 444532]
    # Lubbock Texas: [418184, 490370]
    #pulse_start = 418184
    #pulse_stop = 490370
    
    #pulse_stop = 444532

    # Mexico Coast: [51507, 123688]
    #pulse_start = 51507
    #pulse_stop = 123688

    #pulse_start = 51507
    #pulse_stop = 86688

    # Medicine Hat Canada: [968193, 1040344]
    #pulse_start = 968193 
    #pulse_stop = 1040344

    # Hormuz: [942311, 1001439]
    #pulse_start = 94211
    #pulse_stop = 1001439

    # Canada: [804639, 880632]
    #pulse_start = 804639
    #pulse_stop = 880632

    # Berlin: [493860, 572023]
    #pulse_start = 493860
    #pulse_stop = 572023

    # Vienna: [363793, 440011]
    #pulse_start = 363793
    #pulse_stop = 440011
    # Amazon1: [813924, 888222]
    pulse_start = 813924
    pulse_stop = 888222


    #raw_data = read_raw_h5py(
    #    input_file, 
    #    pols,
    #    freq_group,
        #num_pulses_blk,
    #    pulse_start,
    #    pulse_stop,
    #)

        
    raw_data = raw.getRawDataset(freq, pols)

    #range_valid_start = 827

    #range_valid_start = 942
    range_valid_start = 0

    print(f'{prf_dither_mode = }')
    print(f'{raw_data.shape = }')

    #raw_data = raw_data[:, range_valid_start:]

    num_pulses_proc = pulse_stop - pulse_start
    num_pulses_tb = cpi_len * num_cpi_tb
    num_tb_proc = num_pulses_proc // num_pulses_tb
    num_pulses, num_rng_samples = raw_data.shape
    num_pulses_az_blk = num_tb_az_proc_blk * num_cpi_tb * cpi_len
    #slices = list(slice_gen(num_pulses, num_pulses_az_blk, combine_rem=False))

    slices = list(slice_gen(num_pulses_proc, num_pulses_az_blk, combine_rem=False))
    
    num_az_blocks = len(slices)
    
    # number of range blocks
    num_rng_blks = num_rng_samples // num_samples_rng_blk 

    print()
    print('ST-EVD parameters')
    print(f'{cpi_len = }')
    print(f'{max_deg_freedom = }')
    print(f'{num_cpi_tb = }')
    print(f'{num_tb_az_proc_blk = }')
    #print(f'{raw_data.dtype = }')
    print(f'{num_pulses = }')
    print(f'{num_pulses_proc = }')
    print(f'{num_tb_proc = }')
    print(f'{num_pulses_az_blk = }')
    print(f'{num_az_blocks = }')
    print(f'{num_rng_samples = }')
    print(f'{num_samples_rng_blk = }')
    print(f'{num_rng_blks = }')
    print(f'{pulse_start = }')
    print(f'{pulse_stop = }')
    #print(f'{pulse_pwr_stationarity_check = }')
    print(f'{threshold_method = }')
    print(f'{rfi_check = }')
    print(f'{bt_cond_num_std_thresh_db = }')
    #print(f'{slices = }')
    print()

    # Freq
    n = num_rng_samples
    df = fs / n
    freq_axis = (fc + (np.arange(n) - n/2) * df) / 1e6

    for az_idx, az_slice in enumerate(slices):
        az_start_global = az_slice.start + pulse_start
        az_stop_global  = az_slice.stop + pulse_start

        s = slice(az_start_global, az_stop_global)
        az_blk = raw_data[s, :]

        az_indices = np.arange(az_start_global, az_stop_global)
        print(f'{az_start_global = }')
        print(f'{az_stop_global = }')
        print()

        #mask_subswath_az_blk, mask_from_data_az_blk = read_subswath_mask(
        #    raw, 
        #    freq, 
        #    pols, 
        #    az_indices
        #)

        subswath_az_blk = read_subswath(
            raw, 
            freq, 
            pols, 
            az_indices
        )

        #mask_subswath_az_blk = mask_subswath_az_blk[:, range_valid_start:]
            
        #if az_idx < 3:
        #    sio.savemat(f'mask_subswath_az_blk_{az_idx}.mat', {'mask_subswath_az_blk':mask_subswath_az_blk})
        #    sio.savemat(f'mask_from_data_az_blk_{az_idx}.mat', {'mask_from_data_az_blk':mask_from_data_az_blk})
        
        #if az_idx < 3:
        #    sio.savemat(f'az_blk_{az_idx}.mat', {'az_blk':az_blk})

        az_blk_cal_free = np.zeros(az_blk.shape, dtype=az_blk.dtype)
        #print()
        #print(f'{mask_subswath_az_blk.shape = }')
        print(f'{az_blk.shape = }')
        print(f'{az_blk_cal_free.shape = }')


        if caltone_rm_algo == 'wavelet':
            wavelet_size = 64
            print('Running Wavelet algorithm to remove caltone.\n')

            remover = focus.ToneRemover((fcal - fc) / fs, az_blk.shape[1], wavelet_size)
            #test_pulse = 0
            #raw_pulse_clean = remover.remove_tone(az_blk[test_pulse])

            #print(f'{remover = }')
            for idx_pulse in range(az_blk.shape[0]):
                az_blk_cal_free[idx_pulse] = remover.remove_tone(az_blk[idx_pulse])

                #az_blk_raw = az_blk_cal_free
            #az_blk_cal_free = az_blk_cal_free[:, range_valid_start:]

            #raw_data_mitigated_az_blk = np.zeros(az_blk_cal_free.shape, dtype=raw_data.dtype)

            #if az_idx < 3:
            #    sio.savemat(f'az_blk_cal_free_{az_idx}.mat', {'az_blk_cal_free':az_blk_cal_free})
            print('\nCompleted Wavelet algorithm to remove caltone.\n')

            #plt.figure()
            #plt.plot(freq_axis, 20*np.log10(np.abs(fftshift(fft(az_blk[10]))) / num_rng_samples))
            #plt.plot(20*np.log10(np.abs(fftshift(fft(az_blk_cal_free[10])))), 'r')
            #plt.grid(True)
            #plt.title('Freq: Before Caltone Removal')
            #plt.xlabel('MHz')
            #plt.ylabel('Power (dB)') 

            #plt.figure()
            #plt.plot(freq_axis, 20*np.log10(np.abs(fftshift(fft(az_blk_cal_free[10])))/ num_rng_samples), 'r')
            #plt.grid(True)
            #plt.title('Freq: After Caltone Removal')
            #plt.xlabel('MHz')
            #plt.ylabel('Power (dB)') 

            #plt.figure()
            #plt.imshow(mask_subswath_az_blk)
            #plt.title('Gap Mask from Subswath')
            #plt.grid(True)

            #plt.figure()
            #plt.imshow(mask_from_data_az_blk)
            #plt.title('Gap Mask Estimated from Data')
            #plt.grid(True)
            #plt.show()

            print(f'Finished processing block {az_idx} caltone wavelet removal\n')

        elif caltone_rm_algo == None:
            az_blk_cal_free = az_blk;
            #az_blk_raw = az_blk
            print('No caltone removal algorithm is performed.\n')

            #test_pulse = 1000
            #plt.figure()
            #plt.plot(20*np.log10(np.abs(fftshift(fft(az_blk[-1])))))
            #plt.plot(20*np.log10(np.abs(fftshift(fft(raw_pulse_clean)))), 'r')
            #plt.grid(True)
            #plt.show()


        # Compare power of pulses with and without caltone
        #num_valid = mask_subswath_az_blk.sum(axis = 1)
        #az_blk_pwr = (np.abs(az_blk)**2 * mask_subswath_az_blk).sum(axis=1) / num_valid
        #az_blk_pwr_db = 10*np.log10(az_blk_pwr)

        #az_blk_cal_free_pwr = (np.abs(az_blk_cal_free)**2 * mask_subswath_az_blk).sum(axis=1) / num_valid
        #az_blk_cal_free_pwr_db = 10*np.log10(az_blk_cal_free_pwr)

        #plt.figure()
        #plt.plot(az_blk_pwr_db, label='caltone')
        #plt.title(f'Raw Echo Pulse Power: Az Block {az_idx}')
        #plt.xlabel('Pulse')
        #plt.ylabel('Power (dB)')
        #plt.ylim([20, 50])
        #plt.legend(loc='lower left')
        #plt.grid(True)

        #plt.figure()
        #plt.plot(az_blk_cal_free_pwr_db, 'r', label='no caltone')
        #plt.title(f'Caltone Removed Pulse Power: Az Block {az_idx}')
        #plt.xlabel('Pulse')
        #plt.ylabel('Power (dB)')
        #plt.ylim([20, 50])
        #plt.legend(loc='lower left')
        #plt.grid(True)
        #plt.show()

        #plt.figure()
        #plt.plot(mitigated_data_pulse_pwr_az_blk)
        #plt.title(f'Mitigated Echo Pulse Power: Az Block {az_idx}')
        #plt.xlabel('Pulse')
        #plt.ylabel('Power (dB)')
        #plt.ylim([20, 50])
        #plt.grid(True)
        #plt.savefig(f'pulse_pwr_freq_{freq_group}_{pols}_{mitigation_algo}.png')
        #plt.show()

        raw_data_mitigated_az_blk = np.zeros(az_blk_cal_free.shape, dtype=az_blk.dtype)

        #print(f'{az_blk[0, 0] = }')
        #print(f'{az_blk_cal_free[0, 0] = }')
        #print(f'{az_blk.dtype = }')
        #print(f'{az_blk_cal_free.dtype = }')
        #print()

        #pulse_rfi_test = 68300
        #pulse_rfi_fft = fft(az_blk_cal_free[pulse_rfi_test]) / num_rng_samples

        #pulse_rfi_fft = fftshift(fft(az_blk_cal_free[pulse_rfi_test])) / num_rng_samples
        #pulse_rfi_fft_pwr_db = 20*np.log10(np.abs(pulse_rfi_fft))
        #pulse_rfi_fft_pwr_db2 = 20*np.log10(np.abs(pulse_rfi_fft2))

        #f_base = fftshift(fftfreq(num_rng_samples, d =1/fs))
        #freq_axis = fc + f_base

        #plt.figure()
        #plt.plot(pulse_rfi_fft_pwr_db, label='Before RFI Mitigation')
        #plt.grid(True)
        #plt.title(f'Frequency Response of pulse {pulse_rfi_test}: No RFI Mitigation')
        #plt.xlabel('Range Frequency Bin')
        #plt.ylabel('Power (dB)')

        #plt.figure()
        #plt.plot(freq_axis / 1e6, pulse_rfi_fft_pwr_db, label='Before RFI Mitigation')
        #plt.grid(True)
        #plt.title(f'Range-Frequency Response of pulse {pulse_rfi_test}: No RFI Mitigation')
        #plt.xlabel('Frequency (MHz)')
        #plt.ylabel('Power (dB)')
        #plt.ylim((-80, 20))
        #plt.show()

        
        if mitigation_algo == 'evd':
            rfi_likelihood = run_slow_time_evd(
                az_blk_cal_free,
                cpi_len,
                max_deg_freedom,
                num_rfi_buffer=num_rfi_buffer,
                num_max_trim=num_max_trim,
                num_min_trim=num_min_trim,
                max_num_rfi_ev=max_num_rfi_ev,
                num_samples_rng_blk=num_samples_rng_blk,
                use_entire_pulse=use_entire_pulse,
                threshold_params=threshold_params,
                num_cpi_per_threshold_block=num_cpi_per_threshold_block,
                off_diag_overlap_ratio=off_diag_overlap_ratio,
                diag_valid_ratio=diag_valid_ratio,
                mitigate_enable=mitigate_enable,
                min_rank_frac=min_rank_frac,
                rx_dynamic_range_db=rx_dynamic_range_db,
                swaths=subswath_az_blk,
                threshold_method=threshold_method,
                #rfi_fig_merit_thresh=rfi_fig_merit_thresh,
                rfi_check=rfi_check,
                bt_cond_num_std_thresh_db=bt_cond_num_std_thresh_db,
                max_ev_spread_thresh_db=max_ev_spread_thresh_db,
                eff_rank_std_thresh=eff_rank_std_thresh,
                #bt_pwr_spread_thresh_db=bt_pwr_spread_thresh_db,
                pwr_ref_percentile=pwr_ref_percentile,
                pwr_upper_percentile=pwr_upper_percentile,
                sig_ev_margin_db=sig_ev_margin_db,
                rfi_candidate_tolerance_db=rfi_candidate_tolerance_db,
                #sig_ev_margin_upper_db=sig_ev_margin_upper_db,
                #sig_ev_margin_lower_db=sig_ev_margin_lower_db,
                #pcr_range = pcr_range,
                raw_data_mitigated=raw_data_mitigated_az_blk,
            )

        print(f'{rfi_likelihood = }\n')

        #raw_data_pulse_pwr_az_blk = 10*np.log10(np.var(az_blk_cal_free, axis=1))
        #mitigated_data_pulse_pwr_az_blk = 10*np.log10(np.var(raw_data_mitigated_az_blk, axis=1))

        #pulse_rfi_fft = fft(az_blk_cal_free[pulse_rfi_test]) / num_rng_samples

        #pulse_rfi_miti_fft = fftshift(fft(raw_data_mitigated_az_blk[pulse_rfi_test])) / num_rng_samples
        #pulse_rfi_miti_fft_pwr_db = 20*np.log10(np.abs(pulse_rfi_miti_fft))

        #plt.figure()
        #plt.plot(freq_axis / 1e6, pulse_rfi_miti_fft_pwr_db, label='After RFI Mitigation')
        #plt.grid(True)
        #plt.title(f'Range-Frequency Response of pulse {pulse_rfi_test}: After RFI Mitigation')
        #plt.xlabel('Frequency (MHz)')
        #plt.ylabel('Power (dB)')
        #plt.ylim((-80, 20))
        #plt.show()
         

        #plt.figure()
        #plt.plot(raw_data_pulse_pwr_az_blk)
        #plt.title(f'Raw Echo Pulse Power: Az Block {az_idx}')
        #plt.xlabel('Pulse')
        #plt.ylabel('Power (dB)')
        #plt.ylim([20, 50])
        #plt.grid(True)
        #plt.savefig(f'pulse_pwr_freq_{freq_group}_{pols}_raw.png')

        #plt.figure()
        #plt.plot(mitigated_data_pulse_pwr_az_blk)
        #plt.title(f'Mitigated Echo Pulse Power: Az Block {az_idx}')
        #plt.xlabel('Pulse')
        #plt.ylabel('Power (dB)')
        #plt.ylim([20, 50])
        #plt.grid(True)
        #plt.savefig(f'pulse_pwr_freq_{freq_group}_{pols}_{mitigation_algo}.png')
        #plt.show()


        plot_2d = False

        if plot_2d == True:
            plt.figure()
            plt.imshow(20*np.log10(np.abs(az_blk_cal_free)), interpolation='nearest', cmap='gray', aspect='auto')
            plt.title(f'Raw Echo {freq_group} {pols}: Az Block {az_idx}')
            plt.xlabel('Range (Bins)')
            plt.ylabel('Azimuth (Pulses)')
            plt.colorbar()
            plt.ylim([0, 60])
            ##plt.savefig(f'{plt_path}ree_{pol}_{freq_group}_pwr_raw.png')

            plt.figure()
            plt.imshow(20*np.log10(np.abs(raw_data_mitigated_az_blk)), interpolation='nearest', cmap='gray', aspect='auto')
            plt.title(f'Mitigated Echo {freq_group} {pols}: Az Block {az_idx}')
            plt.xlabel('Range (Bins)')
            plt.ylabel('Azimuth (Pulses)')
            plt.colorbar()
            plt.ylim([0, 60])
            ##plt.savefig(f'{plt_path}ree_{pol}_{freq_group}_pwr_raw.png')
            plt.show()




    #az_mean = np.mean(raw_data, axis=0)
    #raw_data = raw_data - az_mean

if __name__ == "__main__":
    tStart = time.time()

    inputs = cmd_line_parse()

    input_file = inputs.input_file
    cpi_len = inputs.cpi_len
    num_samples_rng_blk = inputs.num_rng_samp_blk
    num_cpi_tb = inputs.num_cpi_tb
    num_tb_az_blk = inputs.num_tb_az_blk
    mitigation_algo = inputs.mitigation
    caltone_rm_algo = inputs.caltone_rm_algo

    #raw_file = f'/scratch2/bohuang/NISAR_L0_PR_RRSD_002_159_A_137S_20250822T093038_20250822T093104_P00408_F_J_001.h5'
    #raw_file = f'/scratch/bohuang/rfi/texas/NISAR_L0_PR_RRSD_004_169_D_156S_20250916T013246_20250916T013546_P00408_F_J_001.h5'


    # Tx/RX Polarizations
    #pols = "HH"
    pols = inputs.pols 
    print(f'{pols = }')

    # RFI: Group A
    freq_group = "A"


    #cpi_len = 32
    #num_samples_rng_blk = 256

    #Extract Raw Data
    print()
    print('Extract Raw Data')

    raw = Raw(hdf5file=input_file)
    
    process_nisar_data(
        raw,
        input_file,
        num_samples_rng_blk,
        cpi_len,
        num_cpi_tb,
        num_tb_az_blk,
        mitigation_algo,
        caltone_rm_algo,
        freq_group,
        pols,
    )



    tEnd = time.time()
    tstring = (
        str(int((tEnd - tStart) / 60))
        + "m "
        + str(round((tEnd - tStart) % 60, 2))
        + "s"
    )

    print("\nTotal run-time:", tstring)
