"""
decode_nisar_data.py

Decode NISAR L0 RRSD quantized radar data using the BFPQLUT (BeamForming & Quantization LUT).

The raw HV/HH/VV/VH data is stored as compound uint16 {r, i} and needs to be decoded
using the BFPQLUT to convert from Digital Numbers (DN) to floating-point physical values.

IMPORTANT: The BFPQLUT is IDENTICAL for all polarizations (HH, HV, VH, VV) and
frequencies (A, B). You only need to extract it once and can reuse it for all data.

Usage:
    hv_decoded = decode_polarization(h5_file, 'HV')

    # Or extract BFPQLUT once and reuse:
    bfpqlut = extract_bfpqlut(h5_file)
    hv_decoded = decode_complex_data(hv_quantized, bfpqlut)
    hh_decoded = decode_complex_data(hh_quantized, bfpqlut)  # Same LUT!
"""

import h5py
import numpy as np
from typing import Tuple


def extract_bfpqlut(h5_file_path: str, tx_pol: str = 'H', rx_pol: str = 'V', frequency: str = 'A') -> np.ndarray:
    """
    Extract the BFPQLUT (BeamForming & Quantization Lookup Table) from NISAR HDF5 file.

    NOTE: The BFPQLUT is identical for ALL polarizations (HH, HV, VH, VV) and both
    frequency bands (A, B) in NISAR L0 RRSD data. The tx_pol, rx_pol, and frequency
    parameters are provided for flexibility, but you can extract from any location
    and reuse it for all data.

    Parameters:
    -----------
    h5_file_path : str
        Path to NISAR L0 RRSD HDF5 file
    tx_pol : str
        Transmit polarization ('H' or 'V'), default 'H'
    rx_pol : str
        Receive polarization ('H' or 'V'), default 'V'
    frequency : str
        Frequency band ('A' or 'B'), default 'A'

    Returns:
    --------
    np.ndarray : Lookup table of shape (65536,) with float32 values
                 Index with uint16 value to get decoded float
    """

    with h5py.File(h5_file_path, 'r') as f:
        lut_path = f'/science/LSAR/RRSD/swaths/frequency{frequency}/tx{tx_pol}/rx{rx_pol}/BFPQLUT'

        if lut_path not in f:
            raise ValueError(f"BFPQLUT not found at {lut_path}")

        bfpqlut = f[lut_path][:]

        # Verify it's the right size (2^16 entries for uint16)
        if len(bfpqlut) != 65536:
            raise ValueError(f"Expected 65536 LUT entries, got {len(bfpqlut)}")

        return bfpqlut


def decode_quantized_data(quantized_data: np.ndarray, bfpqlut: np.ndarray) -> np.ndarray:
    """
    Decode quantized uint16 data using BFPQLUT.

    Parameters:
    -----------
    quantized_data : np.ndarray
        Array of uint16 quantized values (can be any shape)
    bfpqlut : np.ndarray
        Lookup table of shape (65536,) with float32 decode values

    Returns:
    --------
    np.ndarray : Decoded floating-point values (same shape as input)
    """

    # Use the quantized values as indices into the LUT
    return bfpqlut[quantized_data]


def decode_complex_data(compound_data: np.ndarray, bfpqlut: np.ndarray) -> np.ndarray:
    """
    Decode compound {r, i} uint16 data to complex float.

    Parameters:
    -----------
    compound_data : np.ndarray
        Structured array with fields 'r' and 'i' (both uint16)
    bfpqlut : np.ndarray
        Lookup table for decoding

    Returns:
    --------
    np.ndarray : Complex-valued array (same shape as input)
    """

    # Decode real and imaginary parts separately
    real_decoded = decode_quantized_data(compound_data['r'], bfpqlut)
    imag_decoded = decode_quantized_data(compound_data['i'], bfpqlut)

    # Combine into complex array
    return real_decoded + 1j * imag_decoded


def decode_polarization(h5_file_path: str,
                        pol: str = 'HV',
                        frequency: str = 'A',
                        range_lines: Tuple[int, int] = None,
                        range_samples: Tuple[int, int] = None,
                        dataset_path: str = None) -> np.ndarray:
    """
    Decode a polarization channel from NISAR L0 RRSD data.

    Parameters:
    -----------
    h5_file_path : str
        Path to NISAR L0 RRSD HDF5 file
    pol : str
        Polarization channel: 'HH', 'HV', 'VH', or 'VV' (ignored if dataset_path provided)
    frequency : str
        Frequency band: 'A' or 'B', default 'A' (ignored if dataset_path provided)
    range_lines : tuple of (start, end), optional
        Subset of range lines to read (azimuth dimension)
    range_samples : tuple of (start, end), optional
        Subset of range samples to read (range dimension)
    dataset_path : str, optional
        Full dataset path (e.g., '/science/LSAR/RRSD/swaths/frequencyA/txH/rxV/HV')
        If provided, overrides pol and frequency parameters

    Returns:
    --------
    np.ndarray : Decoded complex-valued radar data
    """

    if dataset_path is None:
        # Use pol and frequency to construct path
        if len(pol) != 2 or pol[0] not in 'HV' or pol[1] not in 'HV':
            raise ValueError(f"Invalid polarization: {pol}. Must be HH, HV, VH, or VV")

        tx_pol = pol[0]
        rx_pol = pol[1]
    else:
        # Extract pol from dataset path
        import re
        match = re.search(r'/tx([HV])/rx([HV])/([HV]{2})$', dataset_path)
        if not match:
            raise ValueError(f"Could not extract polarization from dataset path: {dataset_path}")
        tx_pol = match.group(1)
        rx_pol = match.group(2)
        pol = match.group(3)
        # Extract frequency from path
        freq_match = re.search(r'/frequency([AB])/', dataset_path)
        if freq_match:
            frequency = freq_match.group(1)

    with h5py.File(h5_file_path, 'r') as f:
        # Construct paths
        if dataset_path is None:
            data_path = f'/science/LSAR/RRSD/swaths/frequency{frequency}/tx{tx_pol}/rx{rx_pol}/{pol}'
        else:
            data_path = dataset_path

        # Get the BFPQLUT (same for all polarizations and frequencies)
        parent_path = '/'.join(data_path.split('/')[:-1])
        lut_path = f'{parent_path}/BFPQLUT'
        bfpqlut = f[lut_path][:]

        # Get the quantized data
        dataset = f[data_path]

        print(f"Reading {pol} data from: {data_path}")
        print(f"  Shape: {dataset.shape}")
        print(f"  Dtype: {dataset.dtype}")

        # Read subset or full data
        if range_lines is not None and range_samples is not None:
            quantized_data = dataset[range_lines[0]:range_lines[1],
                                    range_samples[0]:range_samples[1]]
        elif range_lines is not None:
            quantized_data = dataset[range_lines[0]:range_lines[1], :]
        elif range_samples is not None:
            quantized_data = dataset[:, range_samples[0]:range_samples[1]]
        else:
            print(f"  WARNING: Reading full dataset - this may be large!")
            quantized_data = dataset[:]

        print(f"  Read subset shape: {quantized_data.shape}")

        # Decode to complex float
        decoded = decode_complex_data(quantized_data, bfpqlut)

        return decoded


def analyze_bfpqlut(bfpqlut: np.ndarray):
    """Print statistics about the BFPQLUT."""

    print("="*70)
    print("BFPQLUT Analysis")
    print("="*70)
    print(f"Length: {len(bfpqlut)}")
    print(f"Dtype: {bfpqlut.dtype}")
    print(f"\nValue range:")
    print(f"  Min: {bfpqlut.min()}")
    print(f"  Max: {bfpqlut.max()}")
    print(f"  Mean: {bfpqlut.mean()}")
    print(f"  Std: {bfpqlut.std()}")

    # Count zeros (often many trailing zeros)
    num_zeros = np.sum(bfpqlut == 0.0)
    print(f"\nNumber of zero entries: {num_zeros} ({100*num_zeros/len(bfpqlut):.2f}%)")

    # Count non-zero entries
    non_zero = bfpqlut[bfpqlut != 0.0]
    if len(non_zero) > 0:
        print(f"\nNon-zero entries:")
        print(f"  Count: {len(non_zero)}")
        print(f"  Min: {non_zero.min()}")
        print(f"  Max: {non_zero.max()}")
        print(f"  Mean: {non_zero.mean()}")

    print(f"\nFirst 20 entries:")
    for i in range(20):
        print(f"  LUT[{i:5d}] = {bfpqlut[i]:10.6f}")

    print(f"\nLast 20 non-zero entries:")
    last_nonzero_idx = np.where(bfpqlut != 0.0)[0][-1]
    for i in range(max(0, last_nonzero_idx - 19), last_nonzero_idx + 1):
        print(f"  LUT[{i:5d}] = {bfpqlut[i]:10.6f}")


if __name__ == '__main__':
    import sys

    # Default file
    default_file = 'nisar_data/NISAR_L0_PR_RRSD_006_112_D_197S_20251006T024004_20251006T024139_P00410_F_J_001.h5'
    h5_file = sys.argv[1] if len(sys.argv) > 1 else default_file

    print(f"Reading NISAR file: {h5_file}\n")

    # Extract and analyze the BFPQLUT
    print("Extracting BFPQLUT for HV polarization...")
    bfpqlut = extract_bfpqlut(h5_file, tx_pol='H', rx_pol='V')

    analyze_bfpqlut(bfpqlut)

    # Save the LUT
    lut_file = 'bfpqlut_hv.npy'
    np.save(lut_file, bfpqlut)
    print(f"\nSaved BFPQLUT to: {lut_file}")

    # Demonstrate decoding a small subset
    print("\n" + "="*70)
    print("Decoding HV data subset (first 100x100 pixels)")
    print("="*70)

    try:
        hv_decoded = decode_polarization(h5_file, pol='HV',
                                        range_lines=(0, 100),
                                        range_samples=(0, 100))

        print(f"\nDecoded data:")
        print(f"  Shape: {hv_decoded.shape}")
        print(f"  Dtype: {hv_decoded.dtype}")
        print(f"  Range: [{hv_decoded.real.min():.3f}, {hv_decoded.real.max():.3f}] (real)")
        print(f"         [{hv_decoded.imag.min():.3f}, {hv_decoded.imag.max():.3f}] (imag)")
        print(f"\nFirst 5x5 complex values:")
        print(hv_decoded[:5, :5])

    except Exception as e:
        print(f"Error decoding data: {e}")
