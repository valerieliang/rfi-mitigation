# RFI Knee Classifier — L0 Raw Data Pipeline

CNN-based Radio Frequency Interference (RFI) knee-index classifier for
NISAR L-band SAR raw (L0B) data. Replaces the adaptive ST-EST thresholding
step in the ST-EVD mitigation pipeline with a per-CPI deep learning
prediction, enabling finer-grained RFI/signal subspace separation.

---

## Background

The ST-EVD pipeline suppresses RFI by decomposing the slow-time sample
covariance matrix (SCM) of each Coherent Processing Interval (CPI) into
RFI-dominated and signal-dominated eigenspaces. The boundary between those
subspaces is the **knee index**: eigenvalues 1..k are RFI, eigenvalues
k+1..M are signal. The current production algorithm (ST-EST) estimates a
single adaptive threshold per Threshold Block (TB) of 16–20 CPIs, causing:

- **Type A errors** — missed RFI when the knee varies within the TB
- **Type B errors** — false alarms that remove real signal

This project trains a CNN to predict an individualized knee index per CPI
directly from the raw pulse data, matching the granularity of the underlying
SCM decomposition.

### Why L0 instead of RSLC?

The ST-EVD SCM is formed from raw slow-time pulse blocks `S ∈ C^{M×K}`:

```
R = S @ S^H / K
```

RSLC azimuth lines approximate this but carry range migration and Doppler
history artifacts from focusing. L0B raw pulses are exactly what the ST-EVD
math assumes, making L0 the correct input domain for this classifier.

---

## Repository Layout

```
rfi-mitigation-l0/
├── rfi_gen/               # Installable RFI synthesis package
│   ├── base.py            # SCM utilities, effective_rank, synthetic_clean
│   ├── canvas_l0.py       # L0B HDF5 block extraction (mirrors canvas.py)
│   ├── cw_tone.py         # Narrowband CW RFI generator
│   ├── multitone.py       # Multi-tone RFI generator
│   ├── wideband.py        # Wideband RFI generator
│   ├── pulsed.py          # Pulsed RFI generator
│   ├── chirp.py           # Chirp RFI generator
│   └── pyproject.toml
├── ml/
│   ├── augment.py         # Dataset builder (RFI injection + feature extraction)
│   ├── model.py           # Two-branch CNN architecture
│   ├── split_scenes.py    # Scene-level train/val/test splitter
│   ├── train.py           # Training loop with evaluation
│   └── checkpoints/       # Saved models and evaluation outputs
├── nisar/
│   ├── gen_raw_l0.py      # Synthetic L0B HDF5 generator (white noise)
│   └── check_l0_clean.py  # L0B granule QC screener
└── data/
    ├── l0_out/            # Raw L0B .h5 files (real or synthetic)
    ├── l0_qc/             # QC outputs from check_l0_clean.py
    │   └── l0_qc_summary.csv
    └── l0_model/          # Generated dataset splits and metadata
```

---

## Method

### Data generation

Clean L0B pulse blocks `S ∈ C^{M×K}` are extracted from screened granules
(real or synthetic white noise). Synthetic RFI is injected at a target knee
index using one of five generator types (CW tone, multitone, wideband,
pulsed, chirp), producing a contaminated block `S_rfi`. The SCM is computed
and two feature vectors are extracted per CPI:

| Feature | Shape | Description |
|---|---|---|
| `eigen_input` | `(M, 2)` | Eigenvalue profile (dB) + slope profile |
| `global_input` | `(6,)` | F-factor, σ_min, σ_max, μ_min, trace (dB), condition number |

These are the same statistics ST-EST uses, which guarantees the CNN's
worst-case behavior is no worse than the baseline.

### Knee cap policy

RFI is constrained to contaminate **at most half the eigenvalues** per CPI.
This policy is enforced at three independent layers:

1. **Parameter level** — `draw_style_params` bounds rank-controlling
   parameters (`n_tones`, `n_modes`, `duty`) before the generator runs
2. **Target level** — `style_for_knee` clamps the requested target to
   `[1, M//2]` before selecting a generator style
3. **Post-generation** — `build_tb` demotes any CPI whose realized
   `knee_truth > M//2` to clean (label 0) rather than mislabeling it

Hard caps by CPI size:

| M | Max knee |
|---|---|
| 12 | 6 |
| 16 | 8 |
| 20 | 10 |
| 32 | 16 |

### Balanced dataset

The knee axis `[0, cap]` is divided into bands. A quota-driven scheduler
steers generation toward the most under-filled band at each step, producing
a class-balanced dataset across all reachable knee indices.

Bands for M=32 (cap=16):

| Band | Knee indices |
|---|---|
| clean | 0 |
| 1 | 1 |
| 2 | 2 |
| 3–5 | 3, 4, 5 |
| 6–9 | 6, 7, 8, 9 |
| 10–14 | 10, 11, 12, 13, 14 |
| 15–16 | 15, 16 |

### Model architecture

Two-branch CNN with softmax output over `n_classes = M + 1` knee indices:

- **Eigenvalue branch** — 1D CNN with residual blocks and
  Squeeze-and-Excitation attention over the `(M, 2)` eigenvalue/slope profile
- **Global branch** — dense network over the 6 scalar TB-context features
- **Fusion head** — concatenation → dense → dropout → softmax

Output is a full posterior over knee locations. `argmax` is the point
estimate; Shannon entropy of the distribution is a per-CPI confidence score
usable to defer to the ST-EST baseline on ambiguous CPIs.

---

## Results

Trained on 500 synthetic L0B scenes (white noise canvas, 50 CPIs each),
300 samples per band, M=32.

### Synthetic evaluation (val / test)

| Metric | Val | Test |
|---|---|---|
| Exact accuracy | 0.961 | 0.951 |
| Tolerance-1 (±1) | 0.979 | 0.979 |
| Tolerance-2 (±2) | 0.987 | 0.989 |
| Mean abs error | 0.08 indices | 0.09 indices |
| Binary F1 (RFI present/absent) | 1.000 | 1.000 |
| False alarms | 0 | 0 |
| Missed RFI | 0 | 1 / 1857 |

### Comparison to RSLC baseline

The RSLC version of this classifier (trained on focused azimuth-line
blocks) achieved val tol-2 = 0.909 and test F1 = 0.992. The L0 version
trained on the same synthetic augmentation strategy reaches tol-2 = 0.989
and F1 = 1.000, with lower mean absolute error (0.09 vs 0.75 indices).

The improvement reflects the tighter match between the L0 input domain
and the ST-EVD math, not a larger training set. Sim-to-real transfer on
actual contaminated L0B granules has not yet been evaluated.

---

## Installation

```bash
pip install -e rfi_gen
pip install tensorflow numpy h5py
```

---

## Run Order

### 1. Generate synthetic L0B scenes

```bash
python nisar/gen_raw_l0.py \
    --out data/l0_out/ --n-files 500 \
    --n-cpis 50 --cpi-size 32 --n-range 256 \
    --signal-power-db -20 --pols HH --seed 0
```

Produces `data/l0_out/synthetic_l0_0000.h5` … `synthetic_l0_0499.h5`,
each a `[1600 × 256]` complex64 HDF5 under the NISAR L0B path convention.
Skip this step and drop real L0B files into `data/l0_out/` instead.

### 2. Screen granules for RFI / data quality

```bash
python nisar/check_l0_clean.py data/l0_out/ --out-dir data/l0_qc/
```

Produces `data/l0_qc/l0_qc_summary.csv` with flags CLEAN / REVIEW / NOISY
based on eigenvalue contrast, F-factor, per-pulse energy MAD, and zero-fill
fraction. REVIEW and NOISY files are excluded from training.

### 3. Scene-level train / val / test split

```bash
python ml/split_scenes.py \
    --manifest data/l0_qc/l0_qc_summary.csv \
    --data-dir data/l0_out/ \
    --out data/l0_model/scene_split.csv
```

Splits at the scene (file) level so no granule appears in more than one
split. Default ratio: 70 / 15 / 15.

### 4. Build dataset

```bash
python ml/augment.py \
    --split-csv data/l0_model/scene_split.csv --split train \
    --data-dir data/l0_out/ --l0 \
    --out data/l0_model/train.npz \
    --balance --target-per-band 300 --M 32

python ml/augment.py \
    --split-csv data/l0_model/scene_split.csv --split val \
    --data-dir data/l0_out/ --l0 \
    --out data/l0_model/val.npz \
    --balance --target-per-band 300 --M 32

python ml/augment.py \
    --split-csv data/l0_model/scene_split.csv --split test \
    --data-dir data/l0_out/ --l0 \
    --out data/l0_model/test.npz \
    --balance --target-per-band 300 --M 32
```

The `--l0` flag routes block extraction through `canvas_l0` instead of
`canvas`. Everything downstream (RFI injection, SCM computation, TB
assembly, feature extraction) is unchanged from the RSLC pipeline.

### 5. Train

```bash
python ml/train.py \
    --data data/l0_model/train.npz \
    --val-data data/l0_model/val.npz \
    --test-data data/l0_model/test.npz
```

Saves best checkpoint to `ml/checkpoints/best_model.keras` and writes
`evaluation_report.txt`, `training_curves.png`, and `confusion_matrix.png`.

---

## Key Design Decisions

**Feeding the model the same statistics ST-EST uses** guarantees that the
CNN's worst-case behavior is no worse than the baseline — a strong
architectural argument for deployment.

**White-noise synthetic canvas** (rather than real L0B scenes) is used for
the initial proof of concept. The approach requires relatively few source
scenes to generate large labeled datasets via augmentation. Real L0B scenes
can be substituted at step 1 with no other pipeline changes.

**Chirp generators are excluded from the targeted (balanced) path** because
chirp rank depends on sweep, K, and covariance structure together in a way
that cannot be reliably inverted. Chirp is still used in the random
`draw_style_params` path where the post-generation demotion layer handles
any over-contaminated output.

---

## Next Steps

- Sim-to-real transfer evaluation on real contaminated L0B granules
- End-to-end coherence comparison against ST-EST baseline on full ALOS frames