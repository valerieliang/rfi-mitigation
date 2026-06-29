# NISAR Processing Workflow

All scripts now automatically detect polarization (HH/HV/VH/VV) and handle paths appropriately.

## 1. Process NISAR Data to CPI Tiles

Use the unified `process_nisar_to_cpi.py` script for any polarization:

### Process HV data:
```bash
python preprocess_nisar/process_nisar_to_cpi.py \
  nisar_data/NISAR_L0_PR_RRSD_006_112_D_197S_20251006T024004_20251006T024139_P00410_F_J_001.h5 \
  nisar_cpi_hv.h5 \
  --dataset /science/LSAR/RRSD/swaths/frequencyA/txH/rxV/HV
```

### Process HH data:
```bash
python preprocess_nisar/process_nisar_to_cpi.py \
  nisar_data/NISAR_L0_PR_RRSD_006_112_D_197S_20251006T024004_20251006T024139_P00410_F_J_001.h5 \
  nisar_cpi_hh.h5 \
  --dataset /science/LSAR/RRSD/swaths/frequencyA/txH/rxH/HH
```

### With range selection (for testing):
```bash
python preprocess_nisar/process_nisar_to_cpi.py \
  nisar_data/NISAR_L0_PR_RRSD_006_112_D_197S_20251006T024004_20251006T024139_P00410_F_J_001.h5 \
  nisar_cpi_hh.h5 \
  --dataset /science/LSAR/RRSD/swaths/frequencyA/txH/rxH/HH \
  --range-start 0 \
  --range-end 1600
```

**Features:**
- Polarization auto-detected from dataset path
- Polarization stored as HDF5 attribute: `f.attrs['polarization']`
- Works with any polarization (HH, HV, VH, VV)
- BFPQLUT automatically extracted from correct parent path
- Single script for all polarizations

## 2. Evaluate with Model (Two-Step Process)

### Step 2a: Run Predictions

Use `predict_nisar.py` to run the model and save predictions:

```bash
# HV data
python predict_nisar.py --nisar-h5 nisar_cpi_hv.h5
```
- **Auto-detects**: Polarization = HV
- **Auto-saves to**: `results/nisar_eval_HV/`
- **Outputs**: predictions.npy, probabilities.npy, prediction_map.npy, nisar_stats.json, metadata.json

```bash
# HH data
python predict_nisar.py --nisar-h5 nisar_cpi_hh.h5
```
- **Auto-detects**: Polarization = HH
- **Auto-saves to**: `results/nisar_eval_HH/`

### Step 2b: Generate Plots

Use `plot_nisar_predictions.py` to visualize results:

```bash
# Generate all plots (histogram, spatial map, bounded map)
python plot_nisar_predictions.py --results-dir results/nisar_eval_HV

# With custom max_knee threshold for bounded map
python plot_nisar_predictions.py --results-dir results/nisar_eval_HV --max-knee 4

# Use different colormap (viridis, plasma, inferno, magma, turbo, jet)
python plot_nisar_predictions.py --results-dir results/nisar_eval_HV --cmap plasma

# Generate comparison plot (original vs bounded)
python plot_nisar_predictions.py --results-dir results/nisar_eval_HV --comparison
```

**Plot Types:**
- **Histogram**: Distribution of knee predictions across all CPIs
- **Spatial Map**: Geographic distribution of knee positions (sequential colormap)
- **Bounded Map**: "Recovered" CPIs with knee > max_knee set to 0 (preserves data with weak RFI)
- **Comparison**: Side-by-side original vs bounded

**Features:**
- Reads `polarization` attribute from input HDF5
- Auto-creates output directory: `results/nisar_eval_{POLARIZATION}/`
- Terminology changed from "tiles" to "CPIs" for clarity
- Sequential colormaps (viridis, plasma, etc.) instead of random tab20
- Bounded map allows data preservation for weakly contaminated CPIs

## 3. Decode Raw Data (Optional)

The `decode_nisar_data.py` script now accepts dataset paths directly:

### Using dataset path (recommended):
```python
from preprocess_nisar.decode_nisar_data import decode_polarization

# HV data
hv_data = decode_polarization(
    'nisar_data/NISAR_L0_PR_RRSD_*.h5',
    dataset_path='/science/LSAR/RRSD/swaths/frequencyA/txH/rxV/HV'
)

# HH data
hh_data = decode_polarization(
    'nisar_data/NISAR_L0_PR_RRSD_*.h5',
    dataset_path='/science/LSAR/RRSD/swaths/frequencyA/txH/rxH/HH'
)
```

### Using pol parameter (legacy):
```python
hv_data = decode_polarization('nisar.h5', pol='HV', frequency='A')
hh_data = decode_polarization('nisar.h5', pol='HH', frequency='A')
```

**Features:**
- Accepts full dataset path (invariant to HH/HV)
- Auto-extracts BFPQLUT from parent path
- Backwards compatible with `pol` parameter

## Output Structure

```
results/
├── nisar_eval_HV/          # HV polarization results
│   ├── nisar_stats.json                      # Statistics (RFI rate, distribution)
│   ├── metadata.json                         # CPI dimensions, tile indices
│   ├── predictions.npy                       # Predicted knee indices (N,)
│   ├── probabilities.npy                     # Class probabilities (N, M+1)
│   ├── prediction_map.npy                    # 2D spatial map
│   ├── nisar_predictions_histogram.png       # Distribution histogram
│   ├── nisar_predictions_spatial.png         # Spatial map (sequential colormap)
│   ├── nisar_predictions_bounded.png         # Bounded map (max_knee threshold)
│   └── nisar_predictions_comparison.png      # Side-by-side comparison (optional)
└── nisar_eval_HH/          # HH polarization results
    └── (same structure as HV)
```

## Key Changes

### 1. Unified Processing Script
- Single script `process_nisar_to_cpi.py` handles all polarizations (HH, HV, VH, VV)

### 2. Auto-Detection
- Polarization extracted from dataset path using regex: `/(HH|HV|VH|VV)$`
- Stored in output HDF5: `f.attrs['polarization'] = 'HV'` or `'HH'`

### 3. Output Directory Management
- **Old**: Manual `--output-dir results/nisar_eval`
- **New**: Auto-detects from polarization → `results/nisar_eval_{POLARIZATION}/`
- Still allows manual override if needed

### 4. Path Invariance
All scripts now work with full dataset paths, making them invariant to:
- Polarization (HH, HV, VH, VV)
- Frequency band (A, B)
- Transmit/receive configuration (txH/rxH, txH/rxV, etc.)

## Verification

Check polarization in processed file:
```python
import h5py
with h5py.File('nisar_cpi_hh.h5', 'r') as f:
    print(f"Polarization: {f.attrs['polarization']}")  # Should print: HH
```

Check output directory:
```bash
ls results/
# Should show: nisar_eval_HV/ and nisar_eval_HH/
```

