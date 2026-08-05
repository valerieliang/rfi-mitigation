# RFI Knee Detection

This directory contains the final trained RFI eigenvalue knee detection model for NISAR L0B data, preserved for reproducibility and future reference.

## Model Overview

**Architecture:** Two-branch CNN classifier  
**Task:** Predict the number of RFI-contaminated eigenvalues in covariance matrix eigenvalue profiles  
**Framework:** TensorFlow/Keras

### Model Inputs
1. **Eigenvalue branch** (`eigen_input`): (12, 2) tensor
   - Channel 1: Top 12 eigenvalues, normalized by λ_max, converted to dB
   - Channel 2: First-difference slopes, zero-padded
   
2. **Global branch** (`global_input`): (3,) vector
   - Condition number (dB): `10*log10(λ_1/λ_12)`
   - Effective rank: Shannon entropy of eigenvalue distribution
   - Diagonal median/max ratio: `median(diag)/max(diag)` over valid SCM diagonal entries

### Model Output
Softmax over 17 classes (knee indices 0–16), where:
- Class 0 = no RFI present (all eigenvalues are noise)
- Class k (k > 0) = k eigenvalues are RFI-contaminated

The model outputs a full posterior distribution over knee locations. Use `argmax` for point estimates and Shannon entropy for per-CPI confidence.

## Training Data

Training data was generated from three NISAR L0B scenes using synthetic RFI injection:

### 1. Amazon (Descending Pass)
**Scene:** `NISAR_L0_PR_RRSD_010_017_A_148S_20260110T101338_20260110T102252_X05009_N_J_001.h5`  
**Processing:** Used directly without additional inspection  
**Training region:** [813924, 888222]

### 2. Berlin (Ascending Pass)
**Scene:** `NISAR_L0_PR_RRSD_016_012_A_148S_20260127T034309_20260127T034853_P05006_F_J_001.h5`  
**Processing:** Clean tiles selected via `select_clean.py` after manual inspection  
**Clean region (pulse indices):** `[493860, 572023]`

### 3. Czech Republic (Ascending Pass)
**Scene:** `NISAR_L0_PR_RRSD_016_012_A_148S_20260127T034309_20260127T034853_P05006_F_J_001.h5`  
**Processing:** Clean tiles selected via `select_clean.py` after manual inspection  
**Clean region (pulse indices):** `[435777, 489625]`

**Note:** Berlin and Czech data come from the same ascending pass but represent spatially distinct regions. Both underwent additional inspection to ensure clean background tiles before synthetic RFI injection. The Amazon scene was from a different orbit geometry and backscatter regime.

### Data Generation Pipeline
1. `select_clean.py` identifies clean background tiles (low off-diagonal contamination, high diagonal validity)
2. `generate_data_from_preprocessed.py` or `generate_amazon_data.py` injects synthetic RFI bands into selected tiles
3. Eigenvalues and features are extracted from the contaminated covariance matrices
4. Labels correspond to the number of injected RFI bands

## Training Results

**Model checkpoint:** [`model/best_model.keras`](model/best_model.keras)  
**Training curves:** [`model/training_curves.png`](model/training_curves.png)  
**Training summary:** [`model/training_summary.json`](model/training_summary.json)

The model was trained using:
- **Loss:** Sparse categorical cross-entropy
- **Optimizer:** AdamW with weight decay
- **Validation strategy:** Held-out spatial blocks from training scenes
- **Epochs:** 50 (early stopping on validation loss)

## Global Feature Statistics

The [`metrics/`](metrics/) directory contains distributional statistics for the three global features used by the model, computed separately for clean and RFI-contaminated tiles across all training datasets:

- **Condition number (dB):** [`condition_number_db.csv`](metrics/condition_number_db.csv)  
  - Clean tiles: mean ≈ 7 dB (λ_1/λ_12 ≈ 5×)
  - RFI-contaminated: mean ≈ 15–25 dB (λ_1/λ_12 ≈ 30–300×)
  
- **Effective rank:** [`effective_rank.csv`](metrics/effective_rank.csv)  
  - Shannon entropy of the eigenvalue distribution
  - Lower values indicate more spectral concentration (common in RFI)
  
- **Diagonal median/max ratio:** [`median_max_ratio.csv`](metrics/median_max_ratio.csv)  
  - Ratio of median to maximum diagonal power across valid SCM entries
  - Captures spatial variability in tile power distribution

Each CSV includes summary statistics (mean, median, std, min, max, p05, p95, IQR, count) per dataset and RFI condition. Visualizations are available in [`metrics/plots/`](metrics/plots/), including box plots and number line distributions for comparing clean vs. contaminated feature distributions.

**Use case:** These statistics provide interpretability for the model's global features and can inform threshold selection for anomaly detection or manual inspection workflows.

## Usage

### 1. Testing on Labeled Data

Evaluate the model on synthetic test sets with known ground truth:

```bash
python test_only.py \
    --model model/best_model.keras \
    --data-dirs data/mountain_test data/amazon_test \
    --output-dir results/test_results
```

**Outputs:**
- Confusion matrix and accuracy metrics
- Accuracy vs. RFI strength curves (per-file and pooled)
- Per-dataset performance breakdown

### 2. Scoring Real NISAR Scenes

Run the model on unlabeled L0B granules:

```bash
python score_scene.py \
    /path/to/NISAR_L0_PR_*.h5 \
    --model model/best_model.keras \
    --off-diag-overlap-ratio 0.03 \
    --diag-valid-ratio 0.02 \
    --output-dir results/scene_predictions
```

**Outputs (per channel):**
- `predictions_{freq}_{pol}.h5` - Per-tile predictions, confidence, eigenvalues
- `knee_map_{freq}_{pol}.png` - Spatial map of predicted knee indices
- `confidence_map_{freq}_{pol}.png` - Model confidence across the scene
- `power_vs_knee_{freq}_{pol}.png` - Tile power distribution by predicted class
- `eigen_profiles_{freq}_{pol}.png` - Mean eigenvalue profiles per predicted class

**Important:** Real scene predictions are *candidate detections*, not ground truth. Real RFI has spatial structure (coherent streaks in range or time); scattered single-tile detections are likely model noise.

### 3. Training a New Model

Retrain from scratch on new data:

```bash
python train_only.py \
    --data-dir data/amazon_train data/czech_train data/berlin_train \
    --run-name my_new_model \
    --epochs 50
```

**Outputs:**
- `models/{run_name}/best_model.keras`
- `models/{run_name}/training_curves.png`
- `models/{run_name}/training_summary.json`

## File Structure

```
.
├── model.py                              # Model architecture definition
├── model/
│   ├── best_model.keras                  # Trained model weights
│   ├── training_curves.png               # Loss/accuracy curves
│   └── training_summary.json             # Training hyperparameters and final metrics
├── metrics/                              # Global feature statistics
│   ├── condition_number_db.csv           # Condition number (dB) stats by dataset
│   ├── effective_rank.csv                # Effective rank stats by dataset
│   ├── median_max_ratio.csv              # Diagonal median/max ratio stats by dataset
│   └── plots/                            # Box plots and number line distributions
│       ├── condition_number_db_*.png
│       ├── effective_rank_*.png
│       └── median_max_ratio_*.png
├── train_only.py                         # Training script (fresh model)
├── test_only.py                          # Evaluation script (labeled data)
├── score_scene.py                        # Inference script (unlabeled NISAR scenes)
├── generate_data_from_preprocessed.py    # Synthetic RFI injection (general)
├── generate_amazon_data.py               # Synthetic RFI injection (Amazon-specific)
├── select_clean.py                       # Clean tile selection
├── plotters/                             # Visualization utilities
│   ├── _common.py
│   ├── plot_metrics.py
│   ├── plot_profiles.py
│   └── plot_scene_profiles.py
├── knee_maps/                            # Example knee detection maps
│   ├── amazon/
│   ├── la/
│   ├── medhat/
│   └── vienna/
└── requirements.txt                      # Python dependencies
```

## Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

**Core dependencies:**
- TensorFlow 2.x
- NumPy
- h5py
- matplotlib
- scikit-learn
- tqdm

**NISAR-specific dependencies** (for `score_scene.py`):
- `isce3` (NISAR radar processing library)
- `nisar` (NISAR product readers)

## Model Architecture Details

From [`model.py`](model.py):

**Eigenvalue branch:**
1. Conv1D(128, kernel=5) → BatchNorm → ReLU
2. Residual block (128 filters) with Squeeze-and-Excitation
3. MaxPooling1D(2)
4. Residual block (256 filters) with Squeeze-and-Excitation
5. GlobalAveragePooling1D → 256-dimensional embedding

**Global branch:**
1. Dense(64, ReLU) → Dropout(0.3)
2. Dense(32, ReLU) → 32-dimensional embedding

**Fusion head:**
1. Concatenate(eigenvalue_embedding, global_embedding) → 288 dims
2. Dense(128, ReLU) → Dropout(0.5)
3. Dense(17, softmax) → posterior over knee indices

**Key features:**
- Residual skip connections for gradient flow
- Squeeze-and-Excitation blocks for channel attention
- Scale-invariant features (eigenvalues normalized by λ_max)

## Notes

- The model was trained on CPI size M=16 pulses; only the top 12 eigenvalues are used as features (the bottom 4 are dropped to avoid quantization dithering artifacts).
- Eigenvalue normalization (divide by λ_max) provides scale invariance across different signal-to-noise regimes.
