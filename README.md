# NISAR RFI Mitigation — CNN Knee Classifier

Deep learning replacement for the adaptive slope threshold (ST-EST) in the Slow-Time Eigenvalue Decomposition (ST-EVD) RFI mitigation pipeline for NISAR L-band SAR. The classifier predicts the eigenvalue "knee" index that separates RFI-dominated from signal-dominated eigenvalues on a per-CPI basis, replacing the threshold-block approach with a per-CPI prediction backed by a confidence estimate.

---

## Background

NISAR L-band SAR data is susceptible to RFI from ground-based radars and communication platforms. The ST-EVD approach (Huang et al., IGARSS 2023) removes RFI by decomposing each Coherent Processing Interval (CPI) into its eigenvalue subspace and projecting out the RFI-dominated eigenvectors. The critical parameter is the **knee index** — the boundary between the steep RFI-dominated portion and the flatter signal portion of the sorted eigenvalue profile.

The original ST-EST algorithm estimates this threshold adaptively per threshold block (a group of CPIs sharing the same range extent) using a sigma-ratio heuristic. The limitation is that a single threshold is applied to all CPIs in a block, which causes false alarms and misses for individual CPIs whose RFI levels deviate from the block average.

This project trains a CNN to predict the knee index independently for every CPI, using both the full eigenvalue+slope profile and threshold-block context features as input.

---

## Architecture

Two-branch CNN defined in `ml/model.py`:

**Eigenvalue branch** — 1D CNN over the full M=32 eigenvalue profile (2 channels: eigenvalues in dB and first-difference slopes). Two residual blocks with Squeeze-and-Excitation attention, max pooling, and global average pooling.

**Global branch** — dense network over 6 scalar threshold-block features: F-factor (σ_max/σ_min), σ_min, σ_max, μ_min, trace (dB), and log condition number.

Both branches are concatenated and passed through a dropout + dense head with a 33-class softmax output (knee index 0 = clean, 1–32 = RFI boundary after eigenvalue i). The expected knee is the argmax; the Shannon entropy of the output distribution is a per-CPI confidence score used to gate fallback to ST-EST.

Total parameters: 594,993 (2.27 MB).

---

## Data Pipeline

### 1. Scene acquisition

NISAR L1 RSLC granules were fetched from ASF Vertex using `fetch_nisar.py` with an integrated screen-and-keep loop: each candidate granule is downloaded, screened for RFI contamination and data quality by `check_nisar_clean.py`, and either kept or immediately discarded. This avoids accumulating dirty scenes on disk during the multi-hour fetch run.

Starting from 192 granules matching mode code `_2005_` over the Amazon basin (cycles 4–10, Oct 2025–Jan 2026), 40 clean DHDH/SHSH dual-polarisation scenes were retained after screening (27% yield). Total download: approximately 550 GB screened, ~540 GB discarded.

### 2. Scene-level train/val/test split

Scenes were split at the **scene level** before any CPI extraction using `split_scenes.py`, stratified by polarisation:

| Split | Scenes | Polarisation |
|-------|--------|-------------|
| Train | 28     | 20 DHDH + 8 SHSH |
| Val   | 6      | 4 DHDH + 2 SHSH |
| Test  | 6      | 4 DHDH + 2 SHSH |

All CPIs from a given granule appear in exactly one split. This prevents the model from learning scene-specific speckle textures rather than eigenvalue structure, which was identified as the root cause of the train/val accuracy gap seen in earlier experiments with only 7 scenes.

### 3. Synthetic RFI augmentation

Clean CPI blocks extracted from the RSLC granules serve as background. Synthetic RFI is overlaid using `rfi_gen`, a purpose-built augmentation library with six RFI styles:

| Style | Description | Typical knee range |
|-------|-------------|-------------------|
| `cw_tone` | Single continuous-wave narrowband tone | 1 |
| `multitone` | N simultaneous CW tones | 1–28 |
| `pulsed` | Burst of coherent pulses, variable duty cycle | 6–26 |
| `wideband` | Decaying multi-mode broadband interference | 3–6 |
| `chirp` | Doppler-sweeping tone, variable bandwidth | 14–32 |
| `subtle` | Near-noise-floor single tone | 1–2 |

Each style independently controls the eigenvalue spectrum shape (which sets the knee index label) and the INR (which sets the contrast above the noise floor). This decoupling ensures the model learns the structural feature — the knee — rather than raw power level.

A quota-driven balancer fills 9 knee bands to equal target counts, correcting the natural tendency for low-knee RFI styles (CW tones) to dominate unbalanced generation. Normalization statistics are computed on the train split only and applied to val and test.

**Dataset statistics:**

| Split | Samples | Clean | RFI | Samples/band |
|-------|---------|-------|-----|-------------|
| Train | 4,554 | 510 | 4,044 | ~500 |
| Val   | 4,571 | 510 | 4,061 | ~500 |
| Test  | 4,566 | 510 | 4,056 | ~500 |

---

## Training

```bash
python ml/train.py \
    --data      data/model/train.npz \
    --val-data  data/model/val.npz \
    --test-data data/model/test.npz \
    --label-smoothing 0.1 --weight-decay 1e-4 \
    --epochs 100 --out-dir ml/checkpoints
```

**Key training decisions:**

- **Label smoothing (ε=0.1)** — prevents the model from becoming over-confident on wrong predictions, which was observed to drive val loss upward in earlier runs. Switching from sparse to categorical cross-entropy with smoothed targets stabilised val loss and improved tolerance-2 accuracy by ~7 percentage points.
- **AdamW weight decay (λ=1e-4)** — mild L2 regularisation to reduce overfitting from the limited scene count.
- **Class weighting disabled** — with a balanced dataset (equal samples per band), inverse-frequency weighting is redundant and incompatible with one-hot targets. It is retained as an option for unbalanced runs.
- Early stopping fired at epoch 50 (patience=20); best checkpoint was epoch 30.

---

## Results

### Val set (scene-held-out)

| Metric | Value |
|--------|-------|
| Exact match | 0.709 |
| Tolerance-1 (±1 EV index) | 0.819 |
| Tolerance-2 (±2 EV indices) | 0.909 |
| Mean absolute error | 0.68 EV indices |
| Binary RFI precision | 0.988 |
| Binary RFI recall | 0.984 |
| Binary RFI F1 | 0.986 |

### Test set (fully held-out, never seen during training)

| Metric | Value |
|--------|-------|
| Exact match | 0.706 |
| Tolerance-1 (±1 EV index) | 0.813 |
| Tolerance-2 (±2 EV indices) | 0.892 |
| Mean absolute error | 0.75 EV indices |
| Binary RFI precision | 0.994 |
| Binary RFI recall | 0.990 |
| Binary RFI F1 | 0.992 |

Val and test results are consistent to within 2 percentage points across all metrics, confirming genuine generalisation to unseen scenes rather than memorisation of training scene characteristics.

### Confidence calibration

The Shannon entropy of the softmax output is a reliable gating signal:

| Confidence tier | Samples (val) | Exact accuracy |
|----------------|--------------|----------------|
| High (≥0.6 max prob) | 3,020 (66%) | 0.894 |
| Low (<0.6 max prob) | 1,551 (34%) | 0.349 |

Low-confidence predictions can be routed to the ST-EST baseline. The 66/34 split at the 0.6 threshold means the CNN handles two-thirds of CPIs autonomously at high accuracy, deferring only the genuinely ambiguous cases.

### Comparison to previous results (7-scene baseline)

| | 7-scene baseline | 40-scene model |
|--|---------|------------|
| Train scenes | 7 | 28 |
| Val scenes | 0 (same as train) | 6 (held out) |
| Val tol-2 | 0.77 | 0.909 |
| Train/val tol-1 gap | 0.33 | 0.15 |
| Binary F1 | 0.986 | 0.992 |

The improvement from 0.77 to 0.909 tol-2 came primarily from fixing the scene-level data split, not from architecture changes. The earlier model was measuring val accuracy on scenes it had already seen during training.

---

## Repository structure

```
rfi-mitigation/
├── ml/
│   ├── model.py               two-branch CNN, build_model(), predict_knee_with_confidence()
│   ├── augment.py             quota-balanced RFI augmentation, produces split .npz files
│   ├── build_dataset.py       orchestrator: scene split -> augment train/val/test
│   ├── split_scenes.py        scene-level stratified train/val/test assignment
│   ├── train.py               training loop, evaluation, report
│   ├── inspect_dataset.py     pre-training sanity check
│   └── checkpoints/           saved models and training artefacts
├── rfi_gen/                   synthetic RFI augmentation library
│   ├── base.py                shared helpers (SCM, eigenvalue, features)
│   ├── canvas.py              RSLC block extraction
│   ├── cw_tone.py             CW tone generator
│   ├── multitone.py           multi-tone generator
│   ├── wideband.py            wideband generator
│   ├── pulsed.py              pulsed burst generator
│   ├── subtle.py              near-noise generator
│   └── chirp.py               chirp sweep generator
├── nisar/
│   ├── fetch_nisar.py         ASF search, screen-and-keep download loop
│   └── check_nisar_clean.py   per-granule RFI and quality screen
└── data/
    ├── nisar_out/             clean RSLC granules (40 scenes, ~540 GB)
    └── model/                 train.npz, val.npz, test.npz, scene_split.csv
```

---

## Usage

**Fetch clean scenes** (screen-and-keep, runs until target is met):
```bash
python nisar/fetch_nisar.py --screen-and-keep --target-clean 40 \
    --name-contains _2005_ --max-results 0 --out-dir data/nisar_out
```

**Build dataset** (scene split + augmentation, ~10 min):
```bash
python ml/build_dataset.py \
    --data-dir data/nisar_out --out-dir data/model \
    --target-per-band 500 --figure
```

**Inspect before training:**
```bash
python ml/inspect_dataset.py \
    --train data/model/train.npz \
    --val   data/model/val.npz \
    --test  data/model/test.npz
```

**Train:**
```bash
python ml/train.py \
    --data data/model/train.npz --val-data data/model/val.npz \
    --test-data data/model/test.npz \
    --label-smoothing 0.1 --weight-decay 1e-4 \
    --epochs 100 --out-dir ml/checkpoints
```