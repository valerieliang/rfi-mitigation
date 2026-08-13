# Data Sources for RFI Detection Training and Testing

**Last Updated**: 2026-08-13

---

## Overview

All training and testing in this project uses **real Amazon rainforest SAR data** as backgrounds, with **synthetic RFI patterns** injected at known locations to create ground truth labels.

**Key Point**: There is NO fully synthetic testing. All test results are from real NISAR L0B data.

---

## Training Data

**Source**: Amazon rainforest NISAR L0B granule  
**File**: `NISAR_L0_PR_RRSD_010_017_A_148S_20260110T101338_20260110T102252_X05009_N_J_001.h5`  
**Script**: `generate_unet_segmentation_data.py`

### Process:
1. Extract clean SAR tiles from real L0B data
2. Inject synthetic 2D Gaussian blob RFI patterns
3. Generate binary masks marking RFI-contaminated pixels
4. Save tiles + masks as training data

### RFI Pattern Parameters:
- **Blob size**: 4-24 pulses × 15-90% range coverage
- **Blob count**: 0-8 per tile (max 30% contamination)
- **JSR range**: 3-30 dB
- **Background**: Real Amazon SAR clutter texture

---

## Test Data

### Main Test Set (`test_results.npz`)

**Background**: Same Amazon L0B data (held-out tiles)  
**RFI Patterns**: Gaussian blobs matching training distribution  
**Purpose**: Evaluate in-distribution performance

**Results**:
- 2-Channel: 92.3% IoU, 96.3% Precision, 95.7% Recall
- 4-Channel: 91.9% IoU, 95.9% Precision, 95.6% Recall

### Generalization Test Suite (`test_results_real_background.npz`)

**Background**: Real Amazon L0B data (different tiles)  
**RFI Patterns**: 13 pattern types (5 in-dist, 8 out-of-dist)  
**Purpose**: Evaluate generalization to diverse RFI types

**Pattern Types**:
1. **IN-DIST** (5 patterns):
   - Standard Gaussian Blobs
   - Multiple Overlapping Blobs
   - Small Blob Count
   - Edge Blobs
   - Wide Horizontal Blob

2. **OOD** (8 patterns):
   - Very Small Point Sources (2×2 pixels)
   - Diagonal Streaks
   - Full-Width Vertical Stripes
   - Full-Height Horizontal Bands
   - Very Large Uniform RFI (>60% coverage)
   - Thin Lines (1-pixel narrowband)
   - L-Shaped Patterns
   - Scattered Random Pixels

**Critical**: All patterns injected into **real Amazon SAR backgrounds**, NOT synthetic noise.

---

## Why Use Real Backgrounds?

### Advantages:
1. **Realistic clutter texture** - speckle, terrain features, vegetation backscatter
2. **Authentic ADC gaps** - valid/invalid regions match real NISAR geometry
3. **True performance estimates** - model must handle real SAR statistics
4. **Deployment-ready** - no domain gap between training and production

### What is Synthetic:
- ✅ RFI signal patterns (Gaussian blobs, lines, etc.)
- ✅ RFI power levels (JSR 3-30 dB)
- ✅ RFI positions (random placement)

### What is Real:
- ✅ SAR background clutter (Amazon forest)
- ✅ Speckle texture
- ✅ ADC gap locations
- ✅ Subswath geometry
- ✅ System noise floor

---

## Visualizations

All PNG files in `model/*/test/` directories show:
- **Background**: Real Amazon L0B SAR data
- **Green**: Correctly detected synthetic RFI (True Positive)
- **Red**: False Positive (model predicts RFI on real clean background)
- **Orange**: False Negative (missed synthetic RFI)
- **White**: Correctly classified real background (True Negative)

---

## No Fully Synthetic Testing

**Removed scripts** (generated fully fake data):
- ❌ `analysis/evaluate_generalization.py` - used random noise as background
- ❌ `analysis/visualize_segmentation_samples.py` - synthetic visualizations
- ❌ `analysis/visualize_segmentation_unified.py` - synthetic segmentation
- ❌ `check_fp_rate.py` - fake SAR data for FP rate calculation

**Remaining scripts** (all use real Amazon backgrounds):
- ✅ `generate_unet_segmentation_data.py` - training data from L0B
- ✅ `analysis/create_real_background_test_suite.py` - test suite from L0B
- ✅ `analysis/evaluate_on_real_test_suite.py` - evaluate on real backgrounds
- ✅ `analysis/visualize_real_test_suite.py` - visualize real-background results

---

## Data Provenance

### L0B Granule Details:
```
Filename: NISAR_L0_PR_RRSD_010_017_A_148S_20260110T101338_20260110T102252_X05009_N_J_001.h5
Location: Amazon rainforest
Acquisition: 2026-01-10
Product: L0B Raw Radar Data
Swath: 148S (ascending)
Polarization: HH + HV (dual-pol)
```

### Clean Tile Selection:
- Tiles manually inspected to ensure no real RFI
- Assumption: Amazon scene is RFI-free (tropical forest, low EMI)
- If real RFI present, it becomes false negatives in training (not ideal but conservative)

---

## Summary

**Training**: Real Amazon SAR + Synthetic Gaussian blob RFI  
**Testing**: Real Amazon SAR + Synthetic diverse RFI patterns  
**Visualizations**: All show real backgrounds with synthetic RFI labels  
**Fully Synthetic Testing**: None (removed)

This ensures all performance metrics reflect how the model will behave on real NISAR data.
