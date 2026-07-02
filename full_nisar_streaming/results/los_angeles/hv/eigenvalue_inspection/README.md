# CPI Tile Eigenvalue Inspection

This directory contains detailed eigenvalue analysis and visualization for selected CPI tiles from NISAR data.

## Files Generated

### JSON Data
- **`tile_inspection_results.json`** - Complete numerical results including:
  - Tile indices and global positions
  - Unnormalized eigenvalues (sorted descending)
  - Normalized eigenvalues (scaled 0-1)
  - Eigenvalue slopes (finite differences)
  - Global features used in train.py (condition_number, sigma_min, sigma_max, mu_min, f_factor)
  - SCM diagonal elements

### Eigenvalue Analysis Plots (per tile)
Each tile has a `*_profile.png` showing:
1. **Normalized eigenvalues** - Scaled to [0, 1] for comparing shape
2. **Eigenvalue slopes** - Finite differences with steepest drop highlighted
3. **SCM diagonal** - Real diagonal elements across pulses
4. **Unnormalized eigenvalues (log scale)** - Absolute power values with global features annotated

### CPI Image Visualization (per tile)
Each tile has a `*_image.png` showing the actual radar data:
1. **Power (dB)** - Magnitude² in dB scale showing signal intensity
2. **Phase** - Complex phase in radians (-π to π)
3. **Range profile** - Average magnitude across pulses (shows range structure)
4. **Pulse profile** - Average magnitude across range (shows temporal structure)

### Comparison Plot
- **`global_features_comparison.png`** - Bar chart comparing the 5 global features across all tiles

## Tiles Analyzed

### 1. Mountains (Tile 189, 210)
- **Global Position**: Pulses 3,024-3,040, Range 52,500-52,750
- **Max Eigenvalue**: 45.36 dB
- **Condition Number**: 2707.36
- **Characteristics**: Low signal power, high condition number (noisy/weak signal)

### 2. Urban RFI (Tile 80, 45)
- **Global Position**: Pulses 1,280-1,296, Range 11,250-11,500
- **Max Eigenvalue**: 54.69 dB
- **Condition Number**: 1940.51
- **Characteristics**: High signal power, likely contains RFI contamination

### 3. Urban Clean (Tile 109, 46)
- **Global Position**: Pulses 1,744-1,760, Range 11,500-11,750
- **Max Eigenvalue**: 54.56 dB
- **Condition Number**: 1782.20
- **Characteristics**: High signal power, similar to urban_rfi but cleaner

## Global Features (from train.py)

All features use **normalized eigenvalues** (scaled by max eigenvalue) for scale invariance:

1. **condition_number** = λ_max / λ_min
   - Ratio of largest to smallest eigenvalue
   - Higher values indicate more ill-conditioned covariance (noisy or RFI)

2. **sigma_min** = std(bottom-half SCM diagonal / λ_max)
   - Standard deviation of the lower-power diagonal elements
   - Indicates noise floor variability

3. **sigma_max** = std(top-half SCM diagonal / λ_max)
   - Standard deviation of the higher-power diagonal elements
   - Indicates signal power variability

4. **mu_min** = mean(bottom-half SCM diagonal / λ_max)
   - Mean of the lower-power diagonal elements
   - Baseline noise floor level

5. **f_factor** = sigma_max / (sigma_min + ε)
   - Ratio of signal to noise variability
   - Higher values indicate stronger signal structure

## Understanding the Eigenvalue Scale

### Unnormalized Eigenvalues
- Represent **power per range bin** from the SCM computation
- Formula: SCM = (CPI @ CPI^H) / K, where K=250 (range bins)
- Typical ranges:
  - Clean noise floor: ~10-100 (10-20 dB)
  - Clean signal: ~1,000-100,000 (30-50 dB)
  - RFI-contaminated: >100,000 (>50 dB)

### dB Conversion
- Formula: `dB = 10 * log10(power_linear)`
- This is the standard **power** dB scale
- Example: 294,202 → 54.69 dB

### Normalized Eigenvalues
- Scaled to [0, 1] by dividing by λ_max
- Used in train.py for scale-invariant classification
- Preserves the eigenvalue profile shape regardless of absolute power

## Dataset Information

- **NISAR File**: `NISAR_L0_PR_RRSD_006_112_D_197S_20251006T024004_20251006T024139_P00410_F_J_001.h5`
- **Dataset Path**: `/science/LSAR/RRSD/swaths/frequencyA/txH/rxV/HV`
- **Polarization**: HV (transmit H, receive V)
- **Full Dataset Shape**: 182,760 pulses × 52,866 range bins
- **CPI Dimensions**: 16 pulses × 250 range bins
- **Valid Tile Ranges**:
  - Pulse tiles: 0 to 11,421
  - Range tiles: 0 to 210
  - LA region pulse tiles: 2,908 to 7,786

## How to Regenerate

```bash
cd full_nisar_streaming

MSYS_NO_PATHCONV=1 python inspect_cpi_tiles.py \
    ../nisar_data/NISAR_L0_PR_RRSD_006_112_D_197S_20251006T024004_20251006T024139_P00410_F_J_001.h5 \
    --dataset /science/LSAR/RRSD/swaths/frequencyA/txH/rxV/HV \
    --tiles "mountains:189,210" "urban_rfi:80,45" "urban_clean:109,46" \
    --output-dir los_angeles/eigenvalue_inspection
```

## Notes

- CPI arrays are **not** stored in the JSON file (too large)
- The script uses the same eigenvalue computation as `generate_synthetic_data.py` and `train.py`
- All computations match the training pipeline exactly for consistency
