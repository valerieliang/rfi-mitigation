# HH Polarization - Los Angeles Region Focused Plots

This folder contains plots that focus **only on the Los Angeles region** extracted from the full NISAR processing results.

## Data Extraction

### Full File (Parent Directory)
- **File**: `../hh_outputs.h5`
- **Shape**: `[11,422 × 211]` predictions
- **Coverage**: Entire NISAR file (pulses 0 to 182,752)

### Los Angeles Region (These Plots)
- **Pulse range**: 46,528 to 124,580
- **Tile indices**: Rows 2,908 to 7,786 (extracted from full array)
- **Shape**: `[4,878 × 211]` predictions
- **Y-axis range**: 0 to 4,878 (instead of 0 to 11,422)

## Files

### knee_la_histogram_comparison.png
Histogram comparing original vs bounded knee predictions for the LA region only.

### knee_la_spatial_comparison.png
Side-by-side spatial maps showing original and bounded predictions for the LA region.

### knee_la_original.png
Full spatial map of original predictions (LA region only).

### knee_la_bounded.png
Full spatial map of bounded predictions with knee_bound=4 (LA region only).

## Statistics (LA Region Only)

- **Total CPIs**: 1,029,258
- **Original RFI CPIs**: 1,029,193 (99.99%)
- **Recovered CPIs** (knee_bound=4): 553,823 (53.81%)
- **Remaining RFI CPIs**: 475,370 (46.19%)

## How to Regenerate

```bash
cd full_nisar_streaming/results/los_angeles/hh
python ../../../plot_nisar_predictions.py hh_outputs.h5 --knee-bound 4 --la-only --output-dir focused_image --prefix knee_la
```

## Difference from Parent Directory Plots

The plots in the parent directory (`../knee_*.png`) show the **entire NISAR file** with 11,422 tiles, including regions before and after Los Angeles. These focused plots extract and display **only the 4,878 tiles** that correspond to the Los Angeles geographic region (pulses 46,528 to 124,580).
