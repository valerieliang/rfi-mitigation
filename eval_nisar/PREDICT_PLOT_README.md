# Two-Step Prediction and Plotting Workflow

This document describes the new separated workflow for NISAR evaluation, which splits prediction and plotting into two independent scripts.

## Why Separate?

**Before:** `eval_nisar.py` did everything in one script (load data → predict → plot)
- Hard to regenerate plots with different settings
- Expensive to recompute predictions just to change visualization
- Mixed concerns (ML inference + matplotlib config)

**After:** Two-step workflow
1. `predict_nisar.py` - Run model inference once, save predictions
2. `plot_nisar_predictions.py` - Generate plots multiple times with different settings

**Benefits:**
- Run predictions once, generate plots many times
- Experiment with colormaps, thresholds, plot types without re-running model
- Cleaner separation of concerns
- Faster iteration on visualizations

## Workflow

### Step 1: Generate Predictions

Run the trained model on NISAR CPI tiles:

```bash
python predict_nisar.py --nisar-h5 nisar_cpi_hv.h5
```

**What it does:**
1. Loads trained model (`models/multi_band/best_model.keras`)
2. Loads NISAR CPI tiles from HDF5
3. Extracts features (eigenvalues, global features)
4. Runs inference to predict knee positions
5. Computes statistics (RFI rate, distribution)
6. Saves everything to `results/nisar_eval_{POLARIZATION}/`

**Outputs:**
- `predictions.npy` - Knee indices for each CPI (N,)
- `probabilities.npy` - Model confidence scores (N, M+1)
- `prediction_map.npy` - 2D spatial map
- `nisar_stats.json` - Statistics (distribution, RFI rate, counts)
- `metadata.json` - CPI dimensions, tile indices, polarization

**This step is slow** (~5-30 minutes for large datasets) but only needs to run once.

### Step 2: Generate Plots

Create visualizations from saved predictions:

```bash
# Basic usage (all default plots)
python plot_nisar_predictions.py --results-dir results/nisar_eval_HV

# With custom max_knee threshold
python plot_nisar_predictions.py --results-dir results/nisar_eval_HV --max-knee 4

# Different colormap
python plot_nisar_predictions.py --results-dir results/nisar_eval_HV --cmap plasma

# Add comparison plot
python plot_nisar_predictions.py --results-dir results/nisar_eval_HV --comparison

# Skip certain plots
python plot_nisar_predictions.py --results-dir results/nisar_eval_HV --no-histogram
```

**What it does:**
1. Loads predictions from `results/nisar_eval_{POLARIZATION}/`
2. Generates visualization plots
3. Saves plots to same directory

**Outputs:**
- `nisar_predictions_histogram.png` - Distribution of knee values
- `nisar_predictions_spatial.png` - Spatial map with sequential colormap
- `nisar_predictions_bounded.png` - Bounded map (max_knee threshold)
- `nisar_predictions_comparison.png` - Side-by-side comparison (if `--comparison` flag)

**This step is fast** (~5-30 seconds) and can be run many times with different settings.

## Key Improvements

### 1. CPI Terminology
- **Before:** "tiles" everywhere
- **After:** "CPIs" (Coherent Processing Intervals)
- **Why:** More accurate terminology for radar processing

### 2. Sequential Colormaps
- **Before:** `tab20` colormap (random assortment of colors)
- **After:** Sequential colormaps (`viridis`, `plasma`, `inferno`, etc.)
- **Why:** Knee value increases → Color intensity increases (intuitive)

**Available colormaps:**
- `viridis` (default) - Perceptually uniform, colorblind-friendly
- `plasma` - High contrast, good for printing
- `inferno` - Warm colors, good for dark backgrounds
- `magma` - Purple-red, similar to inferno
- `cividis` - Blue-yellow, optimized for colorblind viewers
- `turbo` - Rainbow-like, high dynamic range
- `jet` - Classic rainbow (not recommended, but available)

### 3. Bounded Spatial Map (NEW)

The bounded map "recovers" CPIs with weak RFI by treating them as clean.

**Concept:**
- If `knee > max_knee`, set knee to 0 (treat as clean)
- Preserves data from CPIs with weak/high-frequency RFI
- Allows balancing data quality vs quantity

**Example:**
```bash
python plot_nisar_predictions.py --results-dir results/nisar_eval_HV --max-knee 4
```

Original predictions:
- 290,000 CPIs with RFI detected
- Only 111 CPIs are clean

Bounded map (max_knee=4):
- CPIs with knee=5,6,7,... → set to knee=0
- Recovers ~59,000 CPIs (20% of dataset)
- Remaining RFI: ~231,000 CPIs (80% instead of 99.96%)

**Use case:** If you can tolerate a few contaminated samples, recover more data by increasing max_knee.

## Command Reference

### predict_nisar.py

```bash
python predict_nisar.py --nisar-h5 <path> [OPTIONS]

Required:
  --nisar-h5 PATH          Path to processed NISAR HDF5 file

Optional:
  --model PATH             Path to trained model (default: models/multi_band/best_model.keras)
  --output-dir PATH        Output directory (default: auto-detect from polarization)
  --max-tiles N            Limit to first N CPIs for testing
  --batch-size N           Batch size for inference (default: 512)
  --save-to-h5             Save predictions back to HDF5 as attributes
```

**Examples:**
```bash
# Basic usage
python predict_nisar.py --nisar-h5 nisar_cpi_hv.h5

# Test on first 1000 CPIs
python predict_nisar.py --nisar-h5 nisar_cpi_hv.h5 --max-tiles 1000

# Custom output directory
python predict_nisar.py --nisar-h5 nisar_cpi_hv.h5 --output-dir results/test_run

# Save predictions back to HDF5
python predict_nisar.py --nisar-h5 nisar_cpi_hv.h5 --save-to-h5
```

### plot_nisar_predictions.py

```bash
python plot_nisar_predictions.py --results-dir <path> [OPTIONS]

Required:
  --results-dir PATH       Path to results directory (output of predict_nisar.py)

Optional:
  --max-knee N             Maximum knee value for bounded map (default: 4)
  --cmap NAME              Colormap name (default: viridis)
  --comparison             Generate side-by-side comparison plot
  --no-histogram           Skip histogram plot
  --no-spatial             Skip spatial map plot
  --no-bounded             Skip bounded map plot
```

**Examples:**
```bash
# Basic usage (all default plots)
python plot_nisar_predictions.py --results-dir results/nisar_eval_HV

# Custom max_knee and colormap
python plot_nisar_predictions.py --results-dir results/nisar_eval_HV --max-knee 6 --cmap plasma

# Only generate bounded map
python plot_nisar_predictions.py --results-dir results/nisar_eval_HV --no-histogram --no-spatial

# Add comparison plot
python plot_nisar_predictions.py --results-dir results/nisar_eval_HV --comparison

# Try different colormaps quickly
python plot_nisar_predictions.py --results-dir results/nisar_eval_HV --cmap viridis
python plot_nisar_predictions.py --results-dir results/nisar_eval_HV --cmap plasma
python plot_nisar_predictions.py --results-dir results/nisar_eval_HV --cmap inferno
```

## Interpreting Results

### Histogram (`nisar_predictions_histogram.png`)

**What it shows:** Distribution of knee values across all CPIs

**How to read:**
- **Green bar (knee=0):** Clean CPIs (no RFI detected)
- **Red bars (knee=1,2,3,...):** RFI-contaminated CPIs, grouped by knee position
- **Y-axis:** Number of CPIs
- **Title:** Shows RFI rate (percentage and count)

**Example interpretation:**
- If most CPIs have knee=1 or knee=2: Weak RFI, consider higher max_knee
- If distribution is uniform across knees: Diverse RFI strengths
- If knee=0 is rare (<1%): Heavy contamination, may need better mitigation

### Spatial Map (`nisar_predictions_spatial.png`)

**What it shows:** Geographic distribution of knee positions

**How to read:**
- **X-axis:** Range CPI index (across-track direction)
- **Y-axis:** Pulse CPI index (along-track direction)
- **Color:** Knee value (darker = clean, brighter = higher knee)
- Sequential colormap: intensity increases with knee value

**Example interpretation:**
- **Horizontal/vertical stripes:** Systematic RFI source (repeating in time/space)
- **Clustered regions:** Localized RFI (e.g., urban areas, transmitters)
- **Random speckle:** Intermittent RFI (e.g., mobile emitters)
- **Uniform color:** Persistent RFI across entire scene

### Bounded Map (`nisar_predictions_bounded.png`)

**What it shows:** "Recovered" CPIs after applying max_knee threshold

**How to read:**
- Same as spatial map, but knee > max_knee → knee=0
- **Title shows:**
  - Original RFI CPIs
  - Recovered CPIs (set to clean)
  - Remaining RFI CPIs
  - Effective RFI rate (after recovery)

**Example interpretation:**
- **High recovery rate (>30%):** Many CPIs have weak RFI, safe to increase max_knee
- **Low recovery rate (<5%):** Most RFI is strong, max_knee won't help much
- **Visual comparison:** If bounded map looks mostly clean, recovery strategy works

### Comparison Plot (`nisar_predictions_comparison.png`)

**What it shows:** Side-by-side original vs bounded maps

**How to read:**
- **Left:** Original predictions (all knee values)
- **Right:** Bounded map (knee > max_knee → 0)
- Visually assess impact of max_knee threshold

**Example interpretation:**
- **Large color change:** max_knee is recovering many CPIs
- **Small color change:** max_knee threshold is too low, increase it
- **Spatial patterns preserved:** RFI structure is still visible after bounding

## Choosing max_knee

The `max_knee` parameter controls the trade-off between data quantity and quality.

**Guidelines:**
- **max_knee = 0:** No recovery, only truly clean CPIs (most conservative)
- **max_knee = 2:** Weak RFI tolerated, moderate recovery
- **max_knee = 4:** (Default) Good balance for most applications
- **max_knee = 6:** Aggressive recovery, tolerates moderate RFI
- **max_knee = 8+:** Very aggressive, may include heavily contaminated CPIs

**Decision criteria:**
- **Application:** If downstream processing is RFI-sensitive, use low max_knee (0-2)
- **Data scarcity:** If you need more samples, increase max_knee (4-6)
- **Visual inspection:** Look at spatial map - if most RFI is knee=1-3, use max_knee=3

**Workflow:**
1. Run predictions once: `python predict_nisar.py --nisar-h5 nisar_cpi_hv.h5`
2. Try different thresholds:
   ```bash
   python plot_nisar_predictions.py --results-dir results/nisar_eval_HV --max-knee 2
   python plot_nisar_predictions.py --results-dir results/nisar_eval_HV --max-knee 4
   python plot_nisar_predictions.py --results-dir results/nisar_eval_HV --max-knee 6
   ```
3. Compare recovery rates in plot titles
4. Choose threshold that balances quantity and quality

## Migration from Old eval_nisar.py

If you have existing code that uses `eval_nisar.py`:

**Old workflow:**
```bash
python eval_nisar.py --nisar-h5 nisar_cpi_hv.h5
```

**New workflow:**
```bash
# Step 1: Generate predictions (run once)
python predict_nisar.py --nisar-h5 nisar_cpi_hv.h5

# Step 2: Generate plots (run many times)
python plot_nisar_predictions.py --results-dir results/nisar_eval_HV
```

**Output compatibility:**
- ✅ `predictions.npy` - Same format
- ✅ `probabilities.npy` - Same format
- ✅ `prediction_map.npy` - Same format
- ✅ `nisar_stats.json` - Same format (different key names: "tiles" → "cpis")
- ➕ `metadata.json` - NEW, includes tile_indices and CPI dimensions
- ✅ `nisar_predictions_histogram.png` - Same content, better formatting
- ✅ `nisar_predictions_spatial.png` - Same content, better colormap
- ➕ `nisar_predictions_bounded.png` - NEW
- ➕ `nisar_predictions_comparison.png` - NEW (optional)

**Breaking changes:**
- JSON keys renamed: `total_tiles` → `total_cpis`, `rfi_tiles` → `rfi_cpis`
- Colormap changed: `tab20` → `viridis` (sequential instead of categorical)
- Labels changed: "tiles" → "CPIs" in all plots

**If you need the old behavior:**
- Keep `eval_nisar.py` in your codebase (it still works)
- Or update your downstream code to use new JSON keys

## Troubleshooting

### "ERROR: Results directory not found"
**Solution:** Run `predict_nisar.py` first to generate predictions.

### "ERROR: Model not found"
**Solution:** Train the model first with `python train.py`.

### "ERROR: NISAR HDF5 not found"
**Solution:** Process NISAR data first with `python preprocess_nisar/process_nisar_to_cpi.py`.

### Plots look too dark/bright
**Solution:** Try a different colormap with `--cmap`:
```bash
python plot_nisar_predictions.py --results-dir results/nisar_eval_HV --cmap plasma
```

### Want to change max_knee after plotting
**Solution:** Just re-run the plotting script (it's fast):
```bash
python plot_nisar_predictions.py --results-dir results/nisar_eval_HV --max-knee 6
```

### Need to regenerate predictions
**Solution:** Delete old results and run predict_nisar.py again:
```bash
rm -rf results/nisar_eval_HV
python predict_nisar.py --nisar-h5 nisar_cpi_hv.h5
```

## Performance

**predict_nisar.py:**
- Small dataset (10k CPIs): ~30 seconds
- Medium dataset (100k CPIs): ~5 minutes
- Large dataset (500k CPIs): ~20-30 minutes

**plot_nisar_predictions.py:**
- Any dataset size: ~5-30 seconds (independent of dataset size)

**Memory usage:**
- predict_nisar.py: ~2-4 GB (model + features + predictions)
- plot_nisar_predictions.py: ~500 MB (just predictions + matplotlib)

## See Also

- [NISAR_WORKFLOW.md](NISAR_WORKFLOW.md) - Full NISAR processing pipeline
- [SCENE_CLASSIFIER_README.md](SCENE_CLASSIFIER_README.md) - Two-stage scene classification
- [train.py](train.py) - Model training pipeline
- [model.py](model.py) - Knee estimator architecture
