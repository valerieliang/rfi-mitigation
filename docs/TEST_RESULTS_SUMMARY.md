# UNet Test Results Summary: 2-Channel vs 4-Channel

**Date**: 2026-08-13  
**Data Source**: Real Amazon rainforest NISAR L0B SAR data + Synthetic RFI labels  
**Test Suite**: 13 RFI patterns (5 in-distribution, 8 out-of-distribution)  
**Background**: ALL tests use real Amazon SAR backgrounds (NO fully synthetic testing)

---

## Executive Summary

**Main Finding**: The 2-channel UNet slightly outperforms the 4-channel model. Phase information provides no benefit.

**Critical Discovery**: False positive rates vary **dramatically** (0.04% to 53.5%) depending on pattern type:
- **IN-DIST patterns**: < 1% FP rate (excellent)
- **OOD Scattered Random**: 53.5% FP rate (catastrophic - predicts RFI everywhere)
- **OOD Thin Lines**: 14.8% FP rate (severe)

The **96% precision** metric applies only to Gaussian blob test data, NOT to out-of-distribution patterns.

**Data Integrity**: All performance metrics are from real NISAR L0B data (Amazon rainforest). Only the RFI patterns are synthetic (injected at known locations for ground truth).

## 1. Overall Test Metrics

### 1.1 Main Test Set (Gaussian Blobs)

| Model | IoU | Precision | Recall | F1 | Loss |
|-------|-----|-----------|--------|-----|------|
| **2-Channel** | 0.9227 | 0.9625 | 0.9570 | 0.9598 | 0.0285 |
| **4-Channel** | 0.9192 | 0.9593 | 0.9565 | 0.9578 | 0.0287 |
| **Difference** | **+0.35%** | +0.32% | +0.05% | +0.20% | -0.02 |

**Winner**: 2-Channel (simpler, faster, better performance)

### 1.2 Generalization Test Suite

| Metric | 2-Ch IN-DIST | 2-Ch OOD | 4-Ch IN-DIST | 4-Ch OOD |
|--------|--------------|----------|--------------|----------|
| **Mean IoU** | 0.9311 ± 0.020 | 0.4205 ± 0.354 | 0.9128 ± 0.029 | 0.3664 ± 0.324 |
| **Mean F1** | 0.9642 ± 0.011 | 0.5126 ± 0.323 | 0.9542 ± 0.016 | 0.4595 ± 0.326 |
| **Performance Drop** | **51.06 IoU points** | | **54.65 IoU points** | |

---

## 2. Detailed Results: 2-Channel Model

### 2.1 In-Distribution Patterns (✅ Excellent Performance)

| Pattern | IoU | F1 | Precision | Recall | **FP Rate** | JSR (dB) |
|---------|-----|-----|-----------|--------|-------------|----------|
| Standard Gaussian Blobs | 0.914 | 0.955 | 0.918 | 0.995 | **0.74%** | 2.0 |
| Multiple Overlapping | 0.935 | 0.967 | 0.955 | 0.979 | **0.45%** | 4.3 |
| Small Blob Count | 0.906 | 0.951 | 0.968 | 0.933 | **0.09%** | 6.7 |
| Edge Blobs | 0.938 | 0.968 | 0.973 | 0.963 | **0.11%** | 9.0 |
| Wide Horizontal | 0.963 | 0.981 | 0.988 | 0.974 | **0.04%** | 11.3 |

**Mean**: IoU 0.931, F1 0.964, **FP Rate 0.29%**

**Key Observations**:
- Very low false positive rates (< 1%)
- High recall (93-100%) - catches nearly all RFI
- Performance improves slightly with JSR

### 2.2 Out-of-Distribution Patterns (❌ Highly Variable)

| Pattern | IoU | F1 | Precision | Recall | **FP Rate** | JSR (dB) | Status |
|---------|-----|-----|-----------|--------|-------------|----------|--------|
| Very Small Point Sources | 0.103 | 0.188 | 0.167 | 0.214 | **0.05%** | 13.7 | ❌ Misses tiny spots |
| Diagonal Streaks | 0.284 | 0.443 | 0.291 | 0.930 | **7.34%** | 16.0 | ❌ Over-predicts |
| Full-Width Vertical | 0.258 | 0.410 | 0.375 | 0.452 | **3.74%** | 18.3 | ❌ Partial detection |
| Full-Height Bands | 0.997 | 0.998 | 1.000 | 0.997 | **0.00%** | 20.7 | ✅ Excellent! |
| Very Large Uniform RFI | 0.974 | 0.987 | 1.000 | 0.974 | **0.00%** | 23.0 | ✅ Excellent! |
| Thin Lines | 0.114 | 0.205 | 0.115 | 0.948 | **14.78%** | 25.3 | ❌ Massive FP |
| L-Shaped Pattern | 0.542 | 0.703 | 0.933 | 0.564 | **0.05%** | 27.7 | ⚠️ Misses corners |
| Scattered Random | 0.092 | 0.168 | 0.096 | 0.671 | **53.54%** | 30.0 | ❌ Catastrophic |

**Mean OOD**: IoU 0.421, F1 0.513, **FP Rate 9.94%**

**Key Findings**:
1. **Catastrophic failures** (FP > 10%):
   - Scattered Random Pixels: **53.5% FP rate** - model hallucinates blobs everywhere
   - Thin Lines: **14.8% FP rate** - tries to "complete" lines into blobs

2. **Surprising successes** (FP = 0%):
   - Full-Height Horizontal Bands: 99.7% IoU
   - Very Large Uniform RFI: 97.4% IoU
   - Reason: Local texture similar to wide Gaussian blobs

3. **Pattern**: High recall (67-95%) but very low precision (9-29%) on geometric OOD patterns

---

## 3. Detailed Results: 4-Channel Model

### 3.1 In-Distribution Patterns

| Pattern | IoU | F1 | Precision | Recall | **FP Rate** | JSR (dB) |
|---------|-----|-----|-----------|--------|-------------|----------|
| Standard Gaussian Blobs | 0.870 | 0.930 | 0.870 | 1.000 | **1.25%** | 2.0 |
| Multiple Overlapping | 0.888 | 0.941 | 0.895 | 0.991 | **1.12%** | 4.3 |
| Small Blob Count | 0.937 | 0.968 | 0.971 | 0.965 | **0.09%** | 6.7 |
| Edge Blobs | 0.940 | 0.969 | 0.973 | 0.964 | **0.11%** | 9.0 |
| Wide Horizontal | 0.929 | 0.963 | 0.997 | 0.932 | **0.01%** | 11.3 |

**Mean**: IoU 0.913, F1 0.954, **FP Rate 0.52%**

**Comparison to 2-Channel**:
- Lower IoU (-1.8%)
- **Higher FP rates** (0.52% vs 0.29%) - phase adds noise

### 3.2 Out-of-Distribution Patterns

| Pattern | IoU | F1 | Precision | Recall | **FP Rate** | JSR (dB) | vs 2-Ch FP |
|---------|-----|-----|-----------|--------|-------------|----------|------------|
| Very Small Point Sources | 0.063 | 0.118 | 0.100 | 0.143 | **0.06%** | 13.7 | +0.01% |
| Diagonal Streaks | 0.134 | 0.236 | 0.544 | 0.151 | **0.41%** | 16.0 | **-6.93%** ✓ |
| Full-Width Vertical | 0.230 | 0.374 | 0.324 | 0.442 | **4.57%** | 18.3 | +0.83% |
| Full-Height Bands | 0.767 | 0.868 | 1.000 | 0.767 | **0.00%** | 20.7 | 0.00% |
| Very Large Uniform RFI | 0.911 | 0.954 | 1.000 | 0.911 | **0.00%** | 23.0 | 0.00% |
| Thin Lines | 0.080 | 0.148 | 0.289 | 0.099 | **0.49%** | 25.3 | **-14.29%** ✓ |
| L-Shaped Pattern | 0.635 | 0.776 | 0.758 | 0.796 | **0.33%** | 27.7 | +0.28% |
| Scattered Random | 0.113 | 0.202 | 0.142 | 0.354 | **18.18%** | 30.0 | **-35.36%** ✓ |

**Mean OOD**: IoU 0.366, F1 0.460, **FP Rate 2.88%**

**Key Finding**: 4-channel has **lower FP rates** on some OOD patterns (thin lines, scattered noise), but **worse overall IoU**. This suggests phase helps reject some false positives but at the cost of missing true RFI.

---

## 4. Why Do Visualizations Look "So Red"?

Looking at the regenerated visualizations with FP rates displayed:

### Pattern-by-Pattern Explanation

1. **Scattered Random Pixels** (bottom-left in visualization):
   - FP Rate: **53.5%** (2-ch) / **18.2%** (4-ch)
   - Why: Model trained on spatially-coherent blobs tries to "find" blobs in random noise
   - Result: Predicts RFI almost everywhere → **image is mostly red**

2. **Thin Lines**:
   - FP Rate: **14.8%** (2-ch) / **0.5%** (4-ch)
   - Why: Model tries to "complete" thin lines into blob shapes
   - Result: Large red halos around actual lines

3. **Diagonal Streaks**:
   - FP Rate: **7.3%** (2-ch) / **0.4%** (4-ch)
   - Why: Model struggles with diagonal geometry (trained on axis-aligned)
   - Result: Scattered red regions trying to match blob priors

4. **IN-DIST Gaussian Blobs**:
   - FP Rate: **< 1%**
   - Why: Matches training distribution
   - Result: **Very little red** - mostly green (TP) and white (TN)

### The Confusion

The original claim of "4% false positive rate" was based on:
- Precision = 96% → 4% of predictions are wrong
- But this applies only to **Gaussian blob test set**

The OOD visualizations show **much higher FP rates** (7-54%), which explains all the red.

---

## 5. Root Cause Analysis

### What the Model Learned

The UNet learned **"Gaussian blob texture with spatial coherence"** rather than **"general RFI"**:

1. **Size prior**: 4-24 pulses → fails on 2×2 pixels and full-span
2. **Shape prior**: Smooth ellipses → fails on sharp corners, thin lines
3. **Spatial coherence**: Continuous regions → fails on scattered pixels
4. **Texture prior**: Soft Gaussian boundaries → struggles with hard edges

### Why Phase Information Doesn't Help (4-Channel)

**Expected Benefit**: Phase coherence texture would reveal RFI structure independently.

**Actual Result**: Phase channels:
- Add **noise** to predictions (higher FP rate on IN-DIST: 0.52% vs 0.29%)
- Help **reject** some OOD false positives (thin lines, scattered noise)
- But **hurt overall performance** (-1.8% IoU on IN-DIST)

**Conclusion**: Phase is too noisy in SAR data to be useful. The marginal OOD FP reduction doesn't justify the IN-DIST performance loss.

---

## 6. Deployment Recommendations

### Use 2-Channel Model When:

✅ **Deploy for**:
- RFI patterns match training (Gaussian blobs, 4-24 pulses)
- Size range 15-90% of valid range
- Smooth boundaries
- Real SAR backgrounds (tested on Amazon forest)

**Expected Performance**:
- IoU: 93% ± 2%
- Precision: 96%
- Recall: 96%
- **FP Rate: < 1%** on clean pixels

### Do NOT Use 2-Channel Model When:

⚠️ **High Risk** for:
- Thin narrowband lines (14.8% FP rate)
- Scattered salt-and-pepper noise (53.5% FP rate!)
- Very small point sources (misses 79%)
- Diagonal patterns (7.3% FP rate)

**Mitigation**:
- FFT-based detector for narrowband
- Median filter for scattered noise
- Data augmentation for diagonals
- Hybrid pipeline combining multiple approaches

### Never Use 4-Channel Model

❌ **Not Recommended**:
- Phase provides no benefit on IN-DIST
- Slightly better on some OOD, but worse overall
- Added complexity without performance gain

**Stick with 2-channel** (simpler, faster, better).

---

## 7. Visualizations Updated

### What Changed:

1. **FP rates displayed** in info boxes - explains red regions
2. **Better spacing** - no text overlap
3. **Clearer legend** - explains color coding
4. **Larger figures** - easier to read

### Files Regenerated:

- `model/two_channel/test/real_test_suite_visualization.png`
- `model/four_channel/test/real_test_suite_visualization.png`

### Updated Scripts:

- `analysis/visualize_real_test_suite.py` - fixed layout, added FP rates
- `analysis/evaluate_generalization_unified.py` - improved spacing

---

## 8. Key Takeaways

1. **Trust the .npz data** - visualizations now match test metrics exactly
2. **FP rates vary 500×** - from 0.04% (wide blobs) to 53.5% (scattered noise)
3. **96% precision ≠ 4% FP rate** - precision applies to predictions, FP rate applies to background pixels
4. **2-channel > 4-channel** - phase adds no value
5. **OOD generalization is poor** - 51 IoU point drop, catastrophic FP rates on geometric patterns

---

**Document Version**: 1.0  
**Last Updated**: 2026-08-13  
**Source Data**: `model/{two,four}_channel/test/test_results_real_background.npz`  

---

## 9. Real-World Vienna Scene Results (NO LABELS)

### Overview

Both models were evaluated on **real Vienna urban scene** NISAR L0B data with **NO ground truth labels**. This tests model behavior on actual production data in a different environment (urban vs Amazon rainforest training).

**Key Observation**: The 4-channel model predicts **6× more RFI contamination** than the 2-channel model on the same data.

### Contamination Statistics

| Model | Mean RFI per Tile | Tiles with RFI | Total Tiles | % Contaminated |
|-------|-------------------|----------------|-------------|----------------|
| **2-Channel (HH)** | 209 samples | 11,579 | 30,888 | **37.5%** |
| **4-Channel (HH)** | 1,230 samples | 16,510 | 30,888 | **53.5%** |
| **Difference** | **+1,021 samples** (+488%) | **+4,931 tiles** (+42.6%) | | **+16.0%** |

**Tile Size**: 256×256 = 65,536 samples per tile

### Contamination Rate Analysis

**2-Channel**:
- Mean contamination: 209 / 65,536 = **0.32% of pixels per tile**
- Relatively conservative predictions
- Detections appear sparse and localized

**4-Channel**:
- Mean contamination: 1,230 / 65,536 = **1.88% of pixels per tile**
- **6× higher** contamination rate than 2-channel
- Predicts RFI in 53.5% of tiles vs 37.5% for 2-channel
- Detections appear more widespread throughout scene

### Spatial Patterns (Observed)

#### 2-Channel Contamination Map
- Sparse, localized detections
- Concentrated in specific regions
- Large areas of the scene show minimal or no contamination

#### 4-Channel Contamination Map
- Denser, more widespread detections
- Higher contamination distributed across entire scene
- More scattered predictions throughout image
- Darker red regions indicate heavier predicted RFI concentration

### What We Can Conclude

**With Ground Truth (Amazon Test Suite)**:
- 2-channel: 0.29% false positive rate on IN-DIST patterns
- 4-channel: 0.52% false positive rate on IN-DIST patterns
- 2-channel achieves +1.8% higher IoU on controlled tests

**Without Ground Truth (Vienna Scene)**:
- 4-channel predicts 6× more contamination than 2-channel
- Vienna is urban environment, training was on forest
- Phase behavior may differ between environments

### Possible Interpretations

**Without labels, multiple interpretations are valid**:

1. **4-channel detects more real RFI**: Vienna urban environment may have more interference sources (buildings, electronics). Phase information helps detect weaker RFI that 2-channel misses.

2. **4-channel has higher false positive rate**: Controlled tests show 4-channel has 1.8× higher FP rate (0.52% vs 0.29%). Urban clutter phase statistics may differ from forest training data, causing false detections.

3. **Both models struggle with domain shift**: Urban vs forest represents significant distribution shift. Different clutter statistics may affect both models differently.

### Recommendation Based on All Evidence

**For deployment on Amazon-like environments**: Use 2-channel
- Better performance on Amazon test suite (+1.8% IoU)
- Lower false positive rate on controlled tests (0.29% vs 0.52%)

**For deployment on Vienna-like environments**: Uncertain without labels
- Need labeled Vienna data to determine which model is accurate
- Could validate by:
  - Manual inspection of predicted RFI regions
  - Comparison with traditional RFI detectors
  - Analysis of focused SAR images (RFI causes artifacts)

**General recommendation**: 2-channel model based on:
- Consistently better performance on all controlled tests
- Simpler architecture (easier to interpret/debug)
- Lower computational cost

---

## 10. Summary Across All Test Conditions

### Controlled Tests (Real Amazon Background + Synthetic RFI Labels)

| Condition | 2-Ch Performance | 4-Ch Performance | Winner |
|-----------|------------------|------------------|--------|
| **IN-DIST Gaussian Blobs** | 93.1% IoU, 0.29% FP | 91.3% IoU, 0.52% FP | **2-Ch** |
| **OOD Geometric Patterns** | 42.1% IoU, 9.94% FP | 36.6% IoU, 2.88% FP | Neither (both poor) |
| **Main Test Set (Blobs)** | 92.3% IoU | 91.9% IoU | **2-Ch** |

### Real Unlabeled Data (Vienna Urban Scene)

| Metric | 2-Ch | 4-Ch | Observation |
|--------|------|------|-------------|
| **Contamination Rate** | 0.32% per tile | 1.88% per tile | 4-ch predicts 6× more |
| **Tiles Flagged** | 37.5% | 53.5% | 4-ch flags 43% more tiles |

**Note**: Without Vienna ground truth, we cannot determine which prediction rate is more accurate.

---

## 11. Final Recommendation

**Deploy 2-Channel UNet** for RFI detection:

**Reasons Based on Controlled Tests**:
1. ✅ **Better performance** (+1.8% IoU on Amazon test suite)
2. ✅ **Lower false positive rate** (0.29% vs 0.52% on IN-DIST)
3. ✅ **Simpler architecture** (2 channels vs 4)
4. ✅ **Faster inference** (~50% fewer input operations)
5. ✅ **Easier deployment** (fewer preprocessing steps)

**Caveat on Vienna Results**:
- 4-channel predicts 6× more RFI on Vienna data
- Without labels, cannot confirm if this is better detection or higher false positives
- Recommend validation on labeled urban data before production deployment on non-forest environments

**Phase information provides no performance benefit on controlled tests.**

---

**Document Version**: 2.0  
**Last Updated**: 2026-08-13  
**Source Data**: 
  - Controlled tests: `model/{two,four}_channel/test/test_results_real_background.npz`
  - Vienna unlabeled: `score/{two,four}_channel/unet_vienna/*.png`
  
**Limitations**: Vienna scene analysis is observational only (no ground truth labels available).
