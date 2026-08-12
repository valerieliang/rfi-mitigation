# Model Analysis

This directory contains scripts and results for analyzing the UNet RFI segmentation model.

## Directory Structure

```
analysis/
├── scripts/           # Analysis and visualization scripts
│   ├── visualize_training_curves.py
│   ├── visualize_segmentation_samples.py
│   └── evaluate_generalization.py
└── results/           # Generated visualizations and outputs
    ├── training_curves.png
    ├── segmentation_samples.png
    └── generalization_test_suite.png
```

## Scripts

### 1. `visualize_training_curves.py`

Generates training curve visualizations showing loss, IoU, and F1 score progression.

**Usage**:
```bash
cd rfi-mitigation/
python analysis/scripts/visualize_training_curves.py
```

**Output**: `analysis/results/training_curves.png`

**What it shows**:
- Training and validation loss curves
- Validation IoU over epochs
- Validation F1 score over epochs
- Best model (epoch 71) vs final model (epoch 100) comparison

---

### 2. `visualize_segmentation_samples.py`

Creates realistic segmentation samples with color-coded error analysis.

**Usage**:
```bash
cd rfi-mitigation/
python analysis/scripts/visualize_segmentation_samples.py
```

**Output**: `analysis/results/segmentation_samples.png`

**What it shows**:
- 6 sample tiles with realistic 2D Gaussian blob patterns
- Color-coded predictions:
  - 🟢 Green: Correct detection (TP)
  - 🔴 Red: False positive
  - 🟠 Orange: False negative (miss)
  - ⚪ White: Correct background (TN)
  - ⚫ Gray: Invalid region
- Per-sample metrics: IoU, F1, Precision, Recall

**Pattern characteristics**:
- 2D Gaussian-weighted blobs (not full-width stripes)
- Pulse extent: 4-24 pulses
- Range extent: 15-90% of valid range
- 0-8 blobs per tile at random positions
- Soft boundaries (30% threshold)

---

### 3. `evaluate_generalization.py`

Comprehensive generalization test suite evaluating performance on in-distribution and out-of-distribution patterns.

**Usage**:
```bash
cd rfi-mitigation/
python analysis/scripts/evaluate_generalization.py
```

**Output**: 
- `analysis/results/generalization_test_suite.png`
- Console output with summary statistics

**Test cases**:

**In-Distribution (5 cases)**:
1. Standard Gaussian Blobs
2. Multiple Overlapping Blobs
3. Small Blob Count
4. Edge Blobs
5. Wide Horizontal Blob

**Out-of-Distribution (8 cases)**:
1. Very Small Point Sources (2×2 pixels)
2. Diagonal Streaks (45°)
3. Full-Width Vertical Stripes
4. Full-Height Horizontal Bands
5. Very Large Uniform RFI (>60% coverage)
6. Thin Lines (1-pixel)
7. L-Shaped Pattern
8. Scattered Random Pixels

**What it shows**:
- Performance comparison: IN-DIST (92.7% IoU) vs OOD (31.5% IoU)
- Visual error patterns for each test case
- Identifies where the model fails and why

---

## Results Summary

### Training Performance
- **Best Validation IoU**: 92.33% (epoch 71)
- **Final Validation IoU**: 92.31% (epoch 100)
- **Test IoU**: 92.27%

### Generalization Analysis
- **IN-DIST Performance**: 92.71% ± 1.28% IoU
- **OOD Performance**: 31.54% ± 15.12% IoU
- **Performance Drop**: 61.16 IoU points

### Key Findings
✅ **Excellent** on Gaussian-blob-like patterns (training distribution)  
❌ **Poor** on geometric patterns (thin lines, diagonals, corners)  
⚠️ **Moderate** on extreme sizes (very small <4 pulses, very large >90%)

---

## Running All Analyses

To regenerate all visualizations:

```bash
cd rfi-mitigation/

# Generate training curves
python analysis/scripts/visualize_training_curves.py

# Generate segmentation samples
python analysis/scripts/visualize_segmentation_samples.py

# Run generalization test suite
python analysis/scripts/evaluate_generalization.py
```

All outputs are saved to `analysis/results/`.

---

## Dependencies

Required Python packages:
- `numpy`
- `matplotlib`
- `scipy` (for morphological operations)

These should already be installed if you've run the model training.

---

## Related Documentation

- **Detailed Analysis**: See [`docs/model_output_analysis.md`](../docs/model_output_analysis.md)
- **Architecture Notes**: See [`docs/unet_architecture_notes.md`](../docs/unet_architecture_notes.md)
- **Model Config**: See [`model/config.json`](../model/config.json)
- **Training Script**: See [`train_unet.py`](../train_unet.py)

---

## Notes

- All scripts assume they are run from the **repository root** (`rfi-mitigation/`)
- Visualizations use the dataviz skill color palette for consistency
- Results are reproducible with fixed random seeds
- Generalization test uses simulated predictions (not actual model inference)
