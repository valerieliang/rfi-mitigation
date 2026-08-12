# RFI Segmentation Training Paradigms

This document describes the two training approaches for RFI semantic segmentation: **2-channel** and **4-channel** input representations.

## Overview

The RFI segmentation model can be trained with two different input representations, each with different architectural requirements and trade-offs:

| Paradigm | Input Channels | Architecture | Status | Training Script |
|----------|----------------|--------------|--------|----------------|
| **2-channel** | [real, imag] | UNet with BatchNorm | ✅ **Current trained model** | [train_unet_2channel.py](train_unet_2channel.py) |
| **4-channel** | [mag_dB, cos_phase, sin_phase, valid] | SegUNet with GroupNorm | ⏳ Not yet implemented | [train_unet_4channel.py](train_unet_4channel.py) |

---

## 2-Channel Training: Real/Imaginary (Current)

### Input Representation

The 2-channel approach directly uses the real and imaginary components of the complex radar data:

- **Channel 0**: Real part (normalized)
- **Channel 1**: Imaginary part (normalized)

### Normalization Strategy

```python
# Convert complex to [real, imag]
tile_real_imag = np.stack([tile.real, tile.imag], axis=0)

# Normalize by 99th percentile of magnitude over VALID samples
magnitude = np.sqrt(real**2 + imag**2)
scale = np.percentile(magnitude[valid > 0.5], 99)
tile_normalized = tile_real_imag / scale
```

**Key point:** Normalization uses the 99th percentile of magnitude computed only over valid samples, preventing ADC gaps from distorting the scale estimate.

### Architecture: UNet with BatchNorm

```python
from unet import UNet

model = UNet(
    in_channels=2,           # [real, imag]
    out_channels=1,          # binary segmentation
    features=[64, 128, 256, 512]  # default encoder widths
)
```

**Architecture details:**
- **Normalization**: BatchNorm2d after each convolution
- **Encoder**: 4 levels with max pooling downsampling
- **Bottleneck**: 1024 channels (512 × 2)
- **Decoder**: Transposed convolutions for upsampling with skip connections
- **Parameters**: ~31M (default features)

### Training Configuration

```bash
python train_unet_2channel.py \
    data/train_tiles.h5 \
    --epochs 100 \
    --batch-size 128 \
    --lr 1e-3 \
    --loss combined \         # BCE + Dice
    --features 64 128 256 512
```

**Loss function:** Combined BCE + Dice loss, masked to valid regions only
```python
loss = 0.5 * BCE(pred, target, valid) + 0.5 * Dice(pred, target, valid)
```

### Advantages

- ✅ **Simple**: Direct use of raw complex data
- ✅ **Proven**: Current trained model (`model/best_model.pth`) uses this approach
- ✅ **Fast training**: BatchNorm is efficient with large batch sizes
- ✅ **No information loss**: Preserves all information from complex data

### Limitations

- ❌ **Rotation variant**: Model must learn rotation invariance from data
- ❌ **Magnitude-phase coupled**: Real/imag encoding couples magnitude and phase information
- ❌ **Implicit validity**: Valid mask affects loss but isn't an explicit input channel

---

## 4-Channel Training: Magnitude + Phase + Valid (Future)

### Input Representation

The 4-channel approach uses a more sophisticated encoding that explicitly separates magnitude, phase, and validity:

- **Channel 0**: Floor-relative log magnitude in dB (scaled)
- **Channel 1**: cos(Δφ) — cosine of adjacent-pulse phase difference
- **Channel 2**: sin(Δφ) — sine of adjacent-pulse phase difference
- **Channel 3**: Validity mask (0 or 1)

### Preprocessing Pipeline

```python
from input_transforms import build_input_channels

# Convert complex tile to 4-channel input
input_4ch = build_input_channels(tile, valid, n_channels=4)
```

**Channel 0: Floor-relative magnitude (dB)**
```python
mag_db = 20 * log₁₀(|tile| + ε)  # magnitude in dB
floor = quantile(mag_db[valid], 0.5)  # median of valid samples
channel_0 = clip((mag_db - floor) / 20.0, -2.0, 6.0)  # scaled and clipped
```

**Channels 1-2: Adjacent-pulse phase difference**
```python
prod = tile[m] * conj(tile[m-1])  # m is pulse index
Δφ = angle(prod)
channel_1 = cos(Δφ)
channel_2 = sin(Δφ)
```

**Channel 3: Validity mask**
```python
channel_3 = valid.astype(float32)  # explicit 0/1 mask
```

### Architecture: SegUNet with GroupNorm

```python
from unet import SegUNet

model = SegUNet(
    in_channels=4,           # [mag_dB, cos_phase, sin_phase, valid]
    out_channels=1,          # binary segmentation
    base_channels=16,        # starting width (doubles each level)
    depth=3                  # downsampling levels
)
```

**Architecture details:**
- **Normalization**: GroupNorm (better for small batch sizes)
- **Encoder**: Configurable depth with max pooling
- **Decoder**: Bilinear upsampling + skip connections
- **Weight initialization**: Kaiming normal for convs, negative bias on head (-4.0)
- **Parameters**: Fewer than 2-channel UNet due to narrower base

### Training Configuration (Planned)

```bash
python train_unet_4channel.py \
    data/train_tiles.h5 \
    --epochs 100 \
    --batch-size 64 \         # smaller due to GroupNorm
    --lr 1e-3 \
    --base-channels 16 \
    --depth 3
```

### Advantages

- ✅ **Rotation invariant**: Magnitude encoding is naturally rotation invariant
- ✅ **Phase information preserved**: cos/sin encoding captures phase differences
- ✅ **Explicit validity**: Valid mask is an input channel, allowing network to learn different behavior for gaps
- ✅ **Better small-batch performance**: GroupNorm works well with batch size < 32
- ✅ **Interpretable channels**: Each channel has clear physical meaning

### Limitations

- ❌ **Not yet implemented**: Training script is a placeholder
- ❌ **More complex preprocessing**: Requires magnitude/phase computation
- ❌ **Potential information loss**: Phase difference loses absolute phase (though absolute phase is likely not useful for RFI)
- ❌ **Unvalidated**: No empirical comparison with 2-channel approach yet

---

## Implementation Status

### Current Model (`model/best_model.pth`)

- **Architecture**: 2-channel UNet with BatchNorm
- **Training script**: [train_unet_2channel.py](train_unet_2channel.py)
- **Input preprocessing**: [input_transforms.py](input_transforms.py):`prepare_tile_2channel()`
- **Inference script**: `score_unet_scene.py` (uses 2-channel input)

### Future Work (4-channel)

To implement 4-channel training:

1. **Complete training script**: Implement [train_unet_4channel.py](train_unet_4channel.py)
   - Import `SegUNet` from [unet.py](unet.py)
   - Use `build_input_channels()` from [input_transforms.py](input_transforms.py)
   - Adjust loss masking for explicit validity channel

2. **Empirical comparison**: Train both models on the same dataset and compare:
   - Segmentation metrics (IoU, F1, precision, recall)
   - Training speed and stability
   - Generalization to unseen data
   - Performance on edge cases (ADC gaps, low SNR, etc.)

3. **Update inference**: Modify `score_unet_scene.py` to support 4-channel input

---

## Code Organization

```
.
├── train_unet_2channel.py      # 2-channel training (CURRENT)
├── train_unet_4channel.py      # 4-channel training (TODO)
├── unet.py                     # Model architectures ONLY
│   ├── UNet                    # 2-channel, BatchNorm
│   └── SegUNet                 # 4-channel, GroupNorm
├── input_transforms.py         # Preprocessing functions ONLY
│   ├── prepare_tile_2channel() # Real/imag normalization
│   └── build_input_channels()  # 4-channel transformation
└── model/
    └── best_model.pth          # Trained 2-channel UNet
```

**Design principle:** Models and preprocessing are strictly separated to maximize compatibility between training paradigms. Both UNet and SegUNet live in [unet.py](unet.py), while all input transforms live in [input_transforms.py](input_transforms.py).

---

## Decision Guide: Which Paradigm?

**Use 2-channel (current):**
- You need a working model NOW
- Training on GPU with large batch sizes (≥64)
- Simple preprocessing pipeline is preferred
- Baseline performance is sufficient

**Use 4-channel (future):**
- Better interpretability is desired (separate magnitude/phase channels)
- Training with small batch sizes (<32)
- Explicit validity modeling is important
- Rotation invariance is critical
- Willing to invest in implementation and validation

---

## References

- **2-channel UNet**: Standard U-Net architecture from Ronneberger et al. (2015)
- **4-channel SegUNet**: Custom architecture with GroupNorm for semantic segmentation
- **Magnitude dB convention**: 20 log₁₀(|z|) per project standards
- **Phase difference**: Adjacent-pulse phase difference captures temporal structure
