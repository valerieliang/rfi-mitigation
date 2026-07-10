# BFPQLUT: NISAR Data Decoding Guide

## What is BFPQLUT?

**BFPQLUT** = **BeamForming & Quantization Lookup Table**

This is the lookup table that decodes NISAR L0 RRSD quantized radar data from unsigned integers to floating-point physical values.

## The Problem

NISAR stores radar echo data in a compressed format to save space:
- Raw data type: `compound {r: uint16, i: uint16}` (real + imaginary parts)
- Each value is a **quantized integer** (0-65535)
- These integers are **NOT the actual radar values**

## The Solution

The BFPQLUT provides the decoding mapping:
```python
decoded_value = BFPQLUT[quantized_uint16]
```

- **Size**: 65,536 entries (one for each possible uint16 value)
- **Type**: float32
- **Active entries**: Only 512 out of 65,536 are non-zero (99.22% are zeros)
- **Range**: ±33,388 (decoded values)

## Key Finding: Same LUT for ALL Polarizations

**IMPORTANT**: The BFPQLUT is **IDENTICAL** for:
- All polarizations: HH, HV, VH, VV
- Both frequency bands: A and B

This means:
1. Extract the BFPQLUT **once** from any location
2. Reuse it to decode **all** polarization channels
3. No need to store multiple copies

## How to Use

### Method 1: Decode polarization directly (recommended)

```python
from decode_nisar_data import decode_polarization

# Decode HV data (frequency A)
hv_complex = decode_polarization(
    'nisar_data/your_file.h5',
    pol='HV',
    frequency='A',
    range_lines=(0, 1000),      # Optional: subset
    range_samples=(0, 10000)    # Optional: subset
)

# Decode HH data (frequency B)
hh_complex = decode_polarization(
    'nisar_data/your_file.h5',
    pol='HH',
    frequency='B',
    range_lines=(0, 1000),
    range_samples=(0, 10000)
)
```

### Method 2: Extract LUT once and reuse

```python
from decode_nisar_data import extract_bfpqlut, decode_complex_data
import h5py

# Extract BFPQLUT once
bfpqlut = extract_bfpqlut('nisar_data/your_file.h5')

# Save for later use
import numpy as np
np.save('bfpqlut.npy', bfpqlut)

# Decode multiple polarizations with the same LUT
with h5py.File('nisar_data/your_file.h5', 'r') as f:
    # Read quantized HV data
    hv_quantized = f['/science/LSAR/RRSD/swaths/frequencyA/txH/rxV/HV'][0:1000, 0:10000]
    hv_decoded = decode_complex_data(hv_quantized, bfpqlut)
    
    # Read quantized HH data
    hh_quantized = f['/science/LSAR/RRSD/swaths/frequencyA/txH/rxH/HH'][0:1000, 0:10000]
    hh_decoded = decode_complex_data(hh_quantized, bfpqlut)  # Same LUT!
```

## BFPQLUT Characteristics

### Quantization Scheme

The BFPQLUT uses a **non-linear quantization**:
- **Small values**: Linear spacing (0.5, 1.5, 2.5, 3.5, ...)
- **Large values**: Logarithmic/exponential spacing (up to ±33,388)
- **Sign encoding**: Positive and negative values are interleaved

Example entries:
```
LUT[0]   = 0.5
LUT[1]   = 1.5
LUT[7]   = 8.151415
LUT[8]   = -0.5       (negative starts)
LUT[9]   = -1.5
LUT[15]  = -8.151415
...
LUT[503] = 33388.25   (max positive)
LUT[511] = -33388.25  (max negative)
LUT[512-65535] = 0.0  (unused)
```

### Statistics

- **Total entries**: 65,536 (2^16)
- **Non-zero entries**: 512 (0.78%)
- **Zero entries**: 65,024 (99.22%)
- **Value range**: [-33388.25, +33388.25]
- **Data type**: float32

## Integration with RFI Detection

For eigenvalue-based RFI detection:

```python
from decode_nisar_data import decode_polarization

# 1. Decode quantized data → complex float
hv_decoded = decode_polarization('file.h5', 'HV', 
                                 range_lines=(i, i+window))

# 2. Form covariance matrix
cov_matrix = compute_covariance(hv_decoded)

# 3. Compute eigenvalues
import numpy as np
eigvals = np.linalg.eigvalsh(cov_matrix)

# 4. Normalize in LINEAR scale (IMPORTANT!)
eigvals_normalized = eigvals / eigvals[0]

# 5. Use for RFI detection
rfi_mask = detect_rfi(eigvals_normalized)
```

**Remember**: Always normalize eigenvalues in **LINEAR scale**, not dB! See `normalization_example.py` for details.

## File Locations

The BFPQLUT can be found at:
```
/science/LSAR/RRSD/swaths/frequencyA/txH/rxH/BFPQLUT  (HH)
/science/LSAR/RRSD/swaths/frequencyA/txH/rxV/BFPQLUT  (HV)
/science/LSAR/RRSD/swaths/frequencyB/txH/rxH/BFPQLUT  (HH, freq B)
/science/LSAR/RRSD/swaths/frequencyB/txH/rxV/BFPQLUT  (HV, freq B)
```

But since they're all identical, just use any one of them.

## Data Format Reference

### Quantized (stored in HDF5):
```
Dtype: compound {
  'r': uint16,  # Real part (0-65535)
  'i': uint16   # Imaginary part (0-65535)
}
```

### Decoded (after BFPQLUT):
```
Dtype: complex64 (float32 real + float32 imag)
Values: Complex numbers in range ±33388
```

## Files

- **`decode_nisar_data.py`**: Python decoder functions
- **`bfpqlut_hv.npy`**: Saved BFPQLUT (can be reused for all polarizations)
- **`normalization_example.py`**: Eigenvalue normalization (linear vs dB)

## Questions?

If you encounter issues with decoding:
1. Check that the HDF5 file contains the BFPQLUT dataset
2. Verify the data path matches your file structure
3. Ensure you're reading the compound dtype correctly (`data['r']` and `data['i']`)
