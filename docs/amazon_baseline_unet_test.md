# Amazon Baseline U-Net Test

## Objective
Test if U-Net semantic segmentation is viable for RFI detection using synthetic RFI overlaid on clean Amazon rainforest data.

## Dataset

**Granule**: `/scratch2/bohuang/amazon/NISAR_L0_PR_RRSD_010_017_A_148S_20260110T101338_20260110T102252_X05009_N_J_001.h5`

**Clean Region**:
- Pulse range: [813924, 888222] (74,298 pulses)
- Range samples: [1000, 26000] (25,000 samples)
- Frequency: A
- Polarization: HH (extendable to others)

**Assumption**: Entire Amazon scene is RFI-free → perfect ground truth for synthetic injection.

## Tile Configuration

- **Size**: 256×256 pixels (square, required for U-Net)
- **Total tiles**: ~11,280 tiles
  - Pulse dimension: 74298 ÷ 256 = 290 tiles
  - Range dimension: 25000 ÷ 256 = 97 tiles
  - Total: 290 × 97 ≈ 28,130 tiles

## Synthetic RFI Parameters

### Blob Configuration
- **Blobs per tile**: 0-8 (includes clean tiles for class balance)
- **Pulse extent**: 4-24 pulses (azimuth)
- **Range extent**: 15%-90% of valid range samples
- **JSR**: 3-30 dB above local signal power
- **Contamination limit**: ≤30% of pixels per tile

### Ground Truth
- **Binary masks**: Pixel-level RFI segmentation (1=RFI, 0=clean)
- **Blob metadata**: Centers, sizes, JSR values for each blob

## Running the Pipeline

### Quick Start
```bash
# On isce3 server
cd /path/to/rfi-mitigation
bash run_amazon_unet_data_generation.sh
```

### Manual Steps

**Step 1: Create clean tiles index**
```bash
python create_amazon_clean_tiles_index.py \
    /scratch2/bohuang/amazon/NISAR_L0_PR_RRSD_010_017_A_148S_20260110T101338_20260110T102252_X05009_N_J_001.h5 \
    --pulse-start 813924 --pulse-end 888222 \
    --range-start 1000 --range-end 26000 \
    --cpi-len 256 --cpi-width 256 \
    --freq A --pol HH \
    --output data/amazon_clean_tiles_index.h5
```

**Step 2: Generate synthetic RFI training data**
```bash
python generate_unet_segmentation_data.py \
    data/amazon_clean_tiles_index.h5 \
    /scratch2/bohuang/amazon/NISAR_L0_PR_RRSD_010_017_A_148S_20260110T101338_20260110T102252_X05009_N_J_001.h5 \
    --target-tag amazon \
    --cpi-width 256 \
    --min-blobs 0 --max-blobs 8 \
    --min-pulse-size 4 --max-pulse-size 24 \
    --min-range-frac 0.15 --max-range-frac 0.90 \
    --jsr-min-db 3 --jsr-max-db 30 \
    --mask-threshold 0.3 \
    --max-contamination-frac 0.30 \
    --output-dir data/unet_amazon_train \
    --seed 0
```

## Output Files

**Location**: `data/unet_amazon_train/`

**Files**:
- `amazon_unet_seg_A_HH.h5` - Training data for frequency A, polarization HH

**HDF5 Structure**:
```
tiles         (N, 256, 256) complex64 - radar tiles with synthetic RFI
masks         (N, 256, 256) bool      - binary segmentation ground truth
valid         (N, 256, 256) bool      - validity mask (subswath/ADC gaps)
n_blobs       (N,)          int8      - number of blobs per tile (0=clean)
blob_jsr_db   (N, 8)        float32   - JSR per blob (NaN-padded)
signal_power_db (N,)        float32   - baseline tile power
... [additional blob metadata]
```

## Expected Results

### Class Distribution
- **Clean tiles** (n_blobs=0): ~12.5% (1/8 on average)
- **Contaminated tiles** (n_blobs≥1): ~87.5%
- **Contamination per tile**: ≤30% of valid pixels

### Validation Metrics
After training U-Net:
- **Pixel-level IoU**: Intersection-over-Union for RFI vs clean
- **Precision/Recall**: At various thresholds
- **F1 Score**: Harmonic mean of precision/recall
- **Visual inspection**: Predicted masks vs ground truth

## Why This Tests U-Net Viability

1. **Perfect ground truth**: Synthetic injection = exact RFI locations known
2. **Controlled difficulty**: Variable blob sizes, JSR, spatial distributions
3. **Realistic scenarios**: Blobs respect subswath gaps, ADC boundaries
4. **Class balance**: 30% contamination limit prevents degenerate solutions

If U-Net can't segment synthetic blobs on clean Amazon data, it won't work on real RFI.

## Next Steps After Data Generation

1. **Verify output**: Check tile shapes, contamination distribution, blob metadata
2. **Split train/val**: 80/20 or 90/10 split by tile indices
3. **Train U-Net**: PyTorch/TensorFlow implementation
4. **Evaluate**: Metrics + visual inspection of predicted masks
5. **Iterate**: Adjust blob parameters if too easy/hard
