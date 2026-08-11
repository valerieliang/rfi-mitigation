# U-Net Architecture Requirements for RFI Segmentation

## Tile Size Constraints

**U-Net requires large, square input tiles** for effective semantic segmentation.

### Minimum Requirements
- **Tile dimensions**: 256×256 or larger (128×128 absolute minimum)
- **Aspect ratio**: Square or near-square (1:1 preferred)
- **Rationale**: U-Net uses multiple downsampling stages (typically 4-5 levels), each halving spatial dimensions via max-pooling or strided convolutions

### Why Small/Asymmetric Tiles Don't Work

Previous configuration used **16×250 tiles**, which caused:
1. **Insufficient downsampling depth**: Only 2-3 pooling operations possible before collapsing the 16-row dimension
2. **Asymmetric feature maps**: 16×250 → 8×125 → 4×62 → unusable
3. **Limited spatial context**: Only 16 pulses in azimuth provides minimal contextual information
4. **Architecture mismatch**: Standard U-Net encoder/decoder paths expect symmetric, multi-scale feature hierarchies

### Current Configuration
- **CPI_LEN_DEFAULT = 256** (azimuth/pulse dimension)
- **CPI_WIDTH_DEFAULT = 256** (range dimension)
- Allows 4 downsampling stages: 256 → 128 → 64 → 32 → 16
- Provides sufficient spatial context for blob detection

### Alternative Approaches for Small Tiles
If memory or data constraints require smaller tiles:
- **1D CNNs**: Process range lines independently
- **Shallow networks**: Custom architecture with fewer pooling layers
- **Tile aggregation**: Concatenate multiple small tiles into larger batches
- **Different architectures**: ResNet-based segmentation, DeepLabv3, etc.

## Training Implications
- Larger tiles → higher memory requirements during training
- May need to reduce batch size accordingly
- Trade-off between spatial context and GPU memory

## Mixed Precision Training for Speed

### What is Mixed Precision (AMP)?

**Automatic Mixed Precision (AMP)** uses 16-bit floating point (FP16) for most operations while keeping critical operations in 32-bit (FP32):

- **FP16 operations**: Convolutions, matrix multiplications (99% of compute)
- **FP32 operations**: Loss calculation, batch normalization, gradient updates
- **Gradient scaler**: Prevents numerical underflow in backpropagation

### Speed Improvements on Modern GPUs

Modern GPUs (H100, A100, V100) have specialized **Tensor Cores** optimized for FP16:

| GPU | FP32 TFLOPS | FP16 TFLOPS | Speedup |
|-----|-------------|-------------|---------|
| H100 | ~60 | ~1000 | ~16x |
| A100 | ~19 | ~312 | ~16x |
| V100 | ~15 | ~125 | ~8x |

**Real-world training speedup**: 2-3x faster (limited by memory bandwidth and CPU overhead)

### Does It Affect Accuracy?

**No.** Mixed precision is now industry standard with no observed accuracy loss:

✅ **Proven safe for CNNs** (ResNet, U-Net, etc.) - NVIDIA 2018 paper  
✅ **Used in production** by all major AI labs since 2018  
✅ **Critical operations stay FP32** - loss, batch norm, final weights  
✅ **Gradient scaling prevents underflow** - no numerical instability  

**For RFI segmentation**: Binary segmentation is robust to minor precision differences. Your JSR range (3-30 dB) doesn't require extreme numerical precision.

### Usage

The `train_unet.py` script enables AMP automatically on GPU:

```bash
# AMP enabled by default (2-3x faster)
python train_unet.py data/*.h5 --epochs 100 --batch-size 192

# Disable if needed (slower, same accuracy)
python train_unet.py data/*.h5 --epochs 100 --batch-size 128 --no-amp
```

### Recommendations

1. **Always use AMP on modern GPUs** (H100, A100, V100, RTX 30xx/40xx)
2. **Increase batch size** with memory savings (AMP uses ~40% less GPU memory)
3. **Monitor first few epochs** - if loss is NaN, gradient scaling may need tuning (rare)

### Verification

To verify AMP doesn't hurt your model:

```bash
# Train 10 epochs with AMP
python train_unet.py data/*.h5 --epochs 10 --run-name test_amp

# Train 10 epochs without AMP
python train_unet.py data/*.h5 --epochs 10 --no-amp --run-name test_fp32
```

Compare validation IoU - should be within 0.01-0.02 (negligible difference).

## RFI Contamination Regions

### Metadata Saved Per Tile
The solution saves RFI contaminated regions in two complementary formats:

1. **Binary segmentation masks** (`masks` dataset): Pixel-level ground truth showing which pixels are RFI (1) vs clean (0)
2. **Blob metadata** (per-blob arrays): Centers, sizes, JSR values, and pixel counts for each injected blob

### Contamination Limit: 30% Rule
**At most 30% of valid pixels can be RFI contaminated per tile** (`max_contamination_frac = 0.30`).

#### Rationale
- Ensures realistic class balance for U-Net training (avoids pathological 90% RFI tiles)
- Prevents model from learning degenerate solutions where "predict all RFI" achieves high accuracy
- Reflects realistic radar scenarios where RFI is minority class
- Blob injection stops early if adding another blob would exceed the 30% limit

#### Configurable via CLI
```bash
--max-contamination-frac 0.25  # Limit to 25% contaminated pixels
```
