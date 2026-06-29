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

## 2. Evaluate with Model

Use `eval_nisar.py` which now auto-detects polarization and saves to the correct directory:

### Evaluate HV data:
```bash
python eval_nisar.py --nisar-h5 nisar_cpi_hv.h5
```
- **Auto-detects**: Polarization = HV
- **Auto-saves to**: `results/nisar_eval_HV/`

### Evaluate HH data:
```bash
python eval_nisar.py --nisar-h5 nisar_cpi_hh.h5
```
- **Auto-detects**: Polarization = HH
- **Auto-saves to**: `results/nisar_eval_HH/`

### Manual output directory (optional):
```bash
python eval_nisar.py --nisar-h5 nisar_cpi_hh.h5 --output-dir results/my_custom_dir
```

**Features:**
- Reads `polarization` attribute from input HDF5
- Auto-creates output directory: `results/nisar_eval_{POLARIZATION}/`
- No manual directory specification needed
- Prevents accidental overwrites between polarizations

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
│   ├── nisar_stats.json
│   ├── predictions.npy
│   ├── probabilities.npy
│   ├── prediction_map.npy
│   ├── nisar_predictions_spatial.png
│   └── nisar_predictions_histogram.png
└── nisar_eval_HH/          # HH polarization results
    ├── nisar_stats.json
    ├── predictions.npy
    ├── probabilities.npy
    ├── prediction_map.npy
    ├── nisar_predictions_spatial.png
    └── nisar_predictions_histogram.png
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

