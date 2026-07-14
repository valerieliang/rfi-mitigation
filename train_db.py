"""
train_db.py

Training pipeline for the CNN-based RFI knee-index classifier.

Consumes ONLY the tile records written by generate_rfi_data.py: one HDF5 per
channel (rfi_data_<freq>_<pol>.h5) holding flat, per-tile arrays. The SCM was
already gap-excluded at generation time and its eigenvalues stored in LINEAR
scale, so this script never touches a complex CPI tile -- it truncates,
normalizes, and trains.

Standard CPI dimensions: 16 pulses x 250 range samples (M=16, K=250).

Data layout: two regions, two generator runs
--------------------------------------------
The training directory is split into TRAIN and VAL only. The test data is not
carved out of it; it comes from a different, disjoint pulse window, generated
separately, so no test tile shares a background with training.

    --data-dir   pulses [813924, 888222)   bands 0..6   -> train + val
    --test-dir   pulses [888222, 896222)   bands 0..6   -> held-out evaluation

BOTH regions use the same label scheme: each tile draws its band count from
[0, 6] and clean tiles (knee = 0) are interspersed at random, roughly 1/7 of
the set. Clean data is never a separate file or a separate run -- it is simply
the tiles that drew zero bands, sitting in the same distribution as everything
else. The confusion matrix therefore has a real clean row, and clean recall /
false-positive rate fall straight out of it rather than needing their own set.

Generating the test region (use a seed distinct from the training run, so the
held-out injections are an independent realization):

    py-isce3 generate_rfi_data.py granule.h5 --freq A \\
        --pulse-start 888222 --pulse-end 896222 \\
        --range-start 2000 --range-end 25000 \\
        --compute-subswath-mask \\
        --off-diag-overlap-ratio 0.03 --diag-valid-ratio 0.02 \\
        --seed 7 --output-dir data/rfi_test

Design decisions
----------------
1. The generating run is expected to have used very permissive gap-exclusion
   validity ratios (3% off-diagonal overlap, 2% diagonal valid) so dithered
   CPIs are preserved rather than dropped. That trades some SCM estimate
   quality for data yield; the feature normalization below is what makes the
   trade-off usable. The ratios actually used live in the file attrs and are
   echoed at load time, so a mismatch is visible instead of silent.
2. Eigenvalues are normalized on the LINEAR scale first (lambda_i / lambda_max,
   so lambda_max -> 1), THEN converted to dB. This removes absolute power level
   as a factor (dithered vs full-power CPIs), leaving spectral SHAPE as the
   signal the model learns from.
3. Only the first N_KEEP = 12 eigenvalues (the 12 largest, descending) are used
   as features. The remaining 4 (smallest) are the ones most exposed to
   dithering / dropout artifacts -- roughly 38.6% of healthy tiles show a
   >10 dB collapse in the smallest eigenvalue. Truncation happens BEFORE
   normalization, so a collapsed eigenvalue never reaches lambda_max, the dB
   floor, or the global features.
4. Condition number (dB) and effective rank are BOTH computed from the kept 12
   eigenvalues only, for every tile, clean or dithered alike, so the two global
   features stay on a consistent basis.

The model architecture is untouched. build_model is called with
cpi_size = N_KEEP, which only sets the length of the eigen-branch input;
n_knee_classes stays tied to the LABEL range, so truncating the features cannot
silently reshape the output head.

Feature extraction (per CPI tile)
---------------------------------
eigen_input  shape (N_KEEP, 2):
    channel 0 -- top N_KEEP eigenvalues, normalized by lambda_max on the linear
                 scale, then 10*log10 (so channel 0 always starts at 0 dB)
    channel 1 -- finite differences of channel 0 (slopes in dB per index),
                 zero-padded at index N_KEEP-1 so the tensor stays (N_KEEP, 2)

global_input shape (2,):
    0  condition_number_db -- ev_db[0] - ev_db[N_KEEP-1], over the kept 12
    1  eff_rank            -- Shannon-entropy effective rank, over the kept 12

Train / val split
-----------------
Split by CONTIGUOUS BLOCKS OF PULSE TILES, not by random row. Tiles at the same
tile_pulse in different polarizations are the same physical background, and
neighbouring tiles share clutter, so a random split would leak a background's
clean eigenvalue floor from train into val. Splitting on tile_pulse keeps every
co-located tile together; --split-buffer pulse tiles are dropped at the
boundary so the two sides are not immediate spatial neighbours.

Usage
-----
    python train_db.py --data-dir data/rfi_train --test-dir data/rfi_test

Outputs
-------
    models/<run>/best_model.keras
    models/<run>/training_curves.png
    models/<run>/confusion_matrix.png    -- full 0..6 grid, clean row included
    models/<run>/metrics.png
    models/<run>/accuracy_vs_jsr.png
    models/<run>/eval_results.json
    models/<run>_summary.json
"""

import os
import sys
import glob
import json
import argparse

import numpy as np
import h5py
import tensorflow as tf

# Add project root to path to import model
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from model import build_model


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

MODELS_ROOT = 'models'

M = 16                 # pulses per CPI = number of eigenvalues available
N_KEEP = 12            # eigenvalues actually used as features (the 12 largest)
N_GLOBAL = 2           # global features: [condition_number_db, eff_rank]

# Permissive gap-exclusion ratios the generating run is expected to have used.
# Checked against the file attrs; the SCM itself is computed in the generator.
EXPECTED_OFF_DIAG_OVERLAP_RATIO = 0.03
EXPECTED_DIAG_VALID_RATIO = 0.02

EPOCHS = 50
BATCH_SIZE = 256
LR = 3e-4
VAL_FRAC = 0.15        # of the training region; test comes from its own region

# Pulse tiles dropped at the train/val boundary
SPLIT_BUFFER_DEFAULT = 8

EPS = 1e-12
DB_FLOOR = -100.0      # floor for dB values, guards the log of a collapsed eigenvalue


# ---------------------------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------------------------

def features_from_eigenvalues(eigvals_linear):
    """
    Build the eigen and global feature tensors from LINEAR eigenvalues.

    Single place the N_KEEP truncation and normalization order are defined, so
    train and test features are guaranteed identical.

    Steps:
      1. Take the top N_KEEP eigenvalues (input is already descending). The
         bottom 4 are dropped here, before anything else touches them, so the
         dithering-induced collapse in the smallest eigenvalue cannot reach
         lambda_max or the global features.
      2. Normalize on the LINEAR scale by lambda_max, so lambda_max -> 1. This
         removes absolute power level and leaves spectral shape as the signal.
      3. Convert to dB. Channel 0 therefore always starts at exactly 0 dB.
      4. Slopes = finite differences of the dB profile, zero-padded to N_KEEP.
      5. Condition number and effective rank from the SAME kept 12.

    A gap-excluded SCM can be slightly indefinite (entries with too little valid
    overlap are zeroed), so eigenvalues are clipped at EPS before the log.

    Args:
        eigvals_linear (np.ndarray): (M,) or (N, M) real eigenvalues, descending.

    Returns:
        eigen (np.ndarray): (N_KEEP, 2) or (N, N_KEEP, 2), float32.
        global_ (np.ndarray): (N_GLOBAL,) or (N, N_GLOBAL), float32.
    """
    single = (np.asarray(eigvals_linear).ndim == 1)
    ev = np.atleast_2d(np.asarray(eigvals_linear, dtype=np.float64))

    # 1. Keep the 12 largest, drop the 4 dithering-exposed smallest
    ev = ev[:, :N_KEEP]

    # 2. Linear normalization by lambda_max (per tile)
    ev = np.maximum(ev, EPS)
    lam_max = np.maximum(ev[:, :1], EPS)
    ev_norm = ev / lam_max

    # 3. dB
    ev_db = 10.0 * np.log10(np.maximum(ev_norm, EPS))
    ev_db = np.maximum(ev_db, DB_FLOOR)

    # 4. Slopes, zero-padded so the tensor stays (N_KEEP, 2)
    slopes = np.diff(ev_db, axis=1)
    slopes = np.concatenate([slopes, np.zeros((ev_db.shape[0], 1))], axis=1)
    eigen = np.stack([ev_db, slopes], axis=-1).astype(np.float32)

    # 5. Global features, both over the kept 12 only
    cond_db = ev_db[:, 0] - np.maximum(ev_db[:, -1], DB_FLOOR)

    p = ev / np.maximum(ev.sum(axis=1, keepdims=True), EPS)
    p = np.maximum(p, EPS)
    eff_rank = np.exp(-np.sum(p * np.log(p), axis=1))

    global_ = np.stack([cond_db, eff_rank], axis=-1).astype(np.float32)

    if single:
        return eigen[0], global_[0]
    return eigen, global_


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_rfi_data_dir(data_dir, max_samples=None, tag=''):
    """
    Load every rfi_data_*.h5 written by generate_rfi_data.py in a directory.

    All channels present (A-HH, A-HV, ...) are concatenated. Each channel got an
    independent RFI injection realization, so their labels are uncorrelated even
    where they share a physical background. Clean tiles (knee = 0) are already
    interspersed through these files; there is no separate clean source.

    Args:
        data_dir (str): Directory of rfi_data_<freq>_<pol>.h5 files.
        max_samples (int|None): Cap on tiles loaded per file (evenly strided).
        tag (str): Label used in the printout (e.g. 'TRAIN region').

    Returns:
        data (dict) with keys:
            eigen (N, N_KEEP, 2) float32
            global (N, N_GLOBAL) float32
            labels (N,) int32       -- knee = number of injected bands, 0 = clean
            groups (N,) int32       -- tile_pulse, the block-split key
            channels (N,) object    -- 'A-HH' / 'A-HV' / ...
            jsr (N, max_bands) float32 or None -- NaN-padded per-band JSR
            n_classes (int)
            meta (dict)
    """
    paths = sorted(glob.glob(os.path.join(data_dir, 'rfi_data_*.h5')))
    if not paths:
        raise FileNotFoundError(f"No rfi_data_*.h5 files found in {data_dir}")

    eigen_parts, global_parts = [], []
    label_parts, group_parts, chan_parts, jsr_parts = [], [], [], []
    n_classes = 0
    meta = {'dir': data_dir, 'files': [], 'channels': [], 'pulse_window': None}

    print(f"\nLoading {tag or data_dir} ...")

    for path in paths:
        with h5py.File(path, 'r') as f:
            n_rec = int(f.attrs.get('n_records', f['labels'].shape[0]))
            freq = str(f.attrs.get('frequency', '?'))
            pol = str(f.attrs.get('polarization', '?'))
            chan = f'{freq}-{pol}'

            odr = float(f.attrs.get('off_diag_overlap_ratio', float('nan')))
            dvr = float(f.attrs.get('diag_valid_ratio', float('nan')))
            gap = bool(f.attrs.get('gap_exclusion_used', False))
            seed = int(f.attrs.get('seed', -1))
            p0 = int(f.attrs.get('pulse_start', -1))
            p1 = int(f.attrs.get('pulse_end', -1))
            min_b = int(f.attrs.get('min_bands', -1))
            max_b = int(f.attrs.get('max_bands', -1))

            sel = slice(None)
            if max_samples is not None and n_rec > max_samples:
                stride = int(np.ceil(n_rec / max_samples))
                sel = slice(0, n_rec, stride)

            eigvals = f['eigenvalues'][sel]              # (n, M) linear, descending
            labels = f['labels'][sel].astype(np.int32)
            groups = f['tile_pulse'][sel].astype(np.int32)
            jsr = np.asarray(f['jsr_db'][sel], dtype=np.float32) if 'jsr_db' in f else None

            eigen, global_ = features_from_eigenvalues(eigvals)

            eigen_parts.append(eigen)
            global_parts.append(global_)
            label_parts.append(labels)
            group_parts.append(groups)
            chan_parts.append(np.full(len(labels), chan, dtype=object))
            if jsr is not None:
                jsr_parts.append(jsr)

            n_classes = max(n_classes, int(f.attrs.get('n_classes', 0)))

            print(f"  {os.path.basename(path)}: {len(labels)} tiles  "
                  f"[{chan}, pulses {p0}-{p1}, bands {min_b}..{max_b}, seed={seed}, "
                  f"gap={gap}, off_diag={odr:.2f}, diag={dvr:.2f}]")

            if min_b != 0:
                print(f"    WARNING: min_bands={min_b}, so this set has NO clean tiles. "
                      f"Both regions should be generated with --min-bands 0 so clean "
                      f"data stays interspersed in the same distribution.")

            if gap and not (np.isclose(odr, EXPECTED_OFF_DIAG_OVERLAP_RATIO)
                            and np.isclose(dvr, EXPECTED_DIAG_VALID_RATIO)):
                print(f"    NOTE: SCM validity ratios differ from the permissive "
                      f"target ({EXPECTED_OFF_DIAG_OVERLAP_RATIO}/"
                      f"{EXPECTED_DIAG_VALID_RATIO}). Dithered CPIs may have had "
                      f"SCM entries zeroed at generation time.")

            meta['files'].append(os.path.basename(path))
            if chan not in meta['channels']:
                meta['channels'].append(chan)
            meta['pulse_window'] = [p0, p1]
            meta['seed'] = seed
            meta['min_bands'] = min_b
            meta['max_bands'] = max_b
            meta['off_diag_overlap_ratio'] = odr
            meta['diag_valid_ratio'] = dvr

    labels = np.concatenate(label_parts, axis=0)

    return {
        'eigen': np.concatenate(eigen_parts, axis=0),
        'global': np.concatenate(global_parts, axis=0),
        'labels': labels,
        'groups': np.concatenate(group_parts, axis=0),
        'channels': np.concatenate(chan_parts, axis=0),
        'jsr': np.concatenate(jsr_parts, axis=0) if jsr_parts else None,
        'n_classes': n_classes or int(labels.max()) + 1,
        'meta': meta,
    }


def check_disjoint(train_meta, test_meta):
    """
    Confirm the test window does not overlap the training window.

    A shared pulse would mean a test tile's background was seen in training,
    which is what the separate-region design exists to prevent.
    """
    tp = train_meta.get('pulse_window')
    xp = test_meta.get('pulse_window')
    if not tp or not xp or -1 in tp or -1 in xp:
        return

    overlap = min(tp[1], xp[1]) - max(tp[0], xp[0])
    if overlap > 0:
        print(f"\n  WARNING: test window {xp} overlaps the training window {tp} by "
              f"{overlap} pulses. Those tiles' backgrounds were seen in training; "
              f"the held-out metrics will be optimistic.")
    else:
        print(f"  test pulses {xp} disjoint from training {tp}  OK")

    if test_meta.get('seed') == train_meta.get('seed'):
        print(f"  NOTE: the test set was generated with the same --seed as the "
              f"training set. The per-tile injection stream is keyed on the "
              f"WINDOW-RELATIVE pulse tile, so the two sets replay the same band "
              f"counts/rows/JSRs tile-for-tile. Backgrounds still differ, so this "
              f"is not label leakage, but a distinct seed gives a genuinely "
              f"independent injection realization.")


def split_train_val(groups, val_frac, buffer_tiles):
    """
    Split the training region into train / val by CONTIGUOUS pulse-tile blocks.

    No test split is carved out here: the test data comes from its own disjoint
    pulse window, loaded separately.

    Every record at a given tile_pulse -- including the same tile in another
    polarization -- lands on the same side, so a background's clean eigenvalue
    floor cannot appear in both. A buffer of pulse tiles is discarded at the
    boundary so the two sides are not immediate spatial neighbours.

    Args:
        groups (np.ndarray): (N,) tile_pulse of each record.
        val_frac (float): fraction of the pulse-tile axis held out for val.
        buffer_tiles (int): pulse tiles dropped at the boundary.

    Returns:
        idx_train, idx_val (np.ndarray of int)
    """
    uniq = np.unique(groups)          # sorted ascending
    n = len(uniq)

    n_val = int(round(val_frac * n))
    n_train = n - n_val
    if n_train <= 0 or n_val <= 0:
        raise ValueError(f'Not enough distinct pulse tiles ({n}) to split')

    train_g = uniq[:n_train]
    val_g = uniq[n_train:]

    # Drop a buffer on each side of the single internal boundary
    if buffer_tiles > 0:
        if len(train_g) > buffer_tiles:
            train_g = train_g[:-buffer_tiles]
        if len(val_g) > buffer_tiles:
            val_g = val_g[buffer_tiles:]

    idx_train = np.where(np.isin(groups, train_g))[0]
    idx_val = np.where(np.isin(groups, val_g))[0]

    print(f"\n  Train/val block split on tile_pulse (buffer={buffer_tiles} pulse tiles):")
    print(f"    train pulses [{train_g.min()}, {train_g.max()}] -> {len(idx_train)} tiles")
    print(f"    val   pulses [{val_g.min()}, {val_g.max()}] -> {len(idx_val)} tiles")

    return idx_train, idx_val


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------

def save_training_curves_png(history, out_dir):
    """Loss and accuracy vs epoch -> out_dir/training_curves.png."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = range(1, len(history.history['loss']) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(epochs, history.history['loss'], label='Train loss')
    ax_loss.plot(epochs, history.history['val_loss'], label='Val loss')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Training & Validation Loss')
    ax_loss.legend()
    ax_loss.grid(True, linestyle='--', alpha=0.5)

    acc_key = 'acc' if 'acc' in history.history else 'sparse_categorical_accuracy'
    val_acc_key = ('val_acc' if 'val_acc' in history.history
                   else 'val_sparse_categorical_accuracy')
    ax_acc.plot(epochs, history.history[acc_key], label='Train acc')
    ax_acc.plot(epochs, history.history[val_acc_key], label='Val acc')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_title('Training & Validation Accuracy')
    ax_acc.legend()
    ax_acc.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    path = os.path.join(out_dir, 'training_curves.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def save_confusion_matrix_png(y_true, y_pred, n_classes, out_dir, title):
    """
    Row-normalised knee confusion matrix -> out_dir/confusion_matrix.png.

    The full 0..6 grid, clean row included: clean tiles are interspersed in the
    test set like any other class, so clean recall and the false-positive rate
    are just row 0 of this matrix.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = ['clean'] + [f'knee@{k}' for k in range(1, n_classes)]

    matrix = np.zeros((n_classes, n_classes), dtype=np.int32)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        matrix[t, p] += 1

    normed = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(max(7, n_classes), max(6, n_classes * 0.9)))
    im = ax.imshow(normed, vmin=0.0, vmax=1.0, cmap='Blues')
    fig.colorbar(im, ax=ax, label='Recall (row-normalised)')

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

    for ri in range(n_classes):
        for ci in range(n_classes):
            color = 'white' if normed[ri, ci] > 0.5 else 'black'
            ax.text(ci, ri, str(matrix[ri, ci]), ha='center', va='center',
                    fontsize=7, color=color)

    fig.tight_layout()
    path = os.path.join(out_dir, 'confusion_matrix.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def save_metrics_png(results, out_dir):
    """Bar chart of the scalar metrics -> out_dir/metrics.png."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names, values = [], []
    for label, key in (
        ('Exact accuracy', 'exact_acc'),
        ('Tol-1 accuracy', 'tol1_acc'),
        ('RFI exact', 'exact_rfi'),
        ('RFI tol-1', 'tol1_rfi'),
        ('Clean recall', 'clean_recall'),
        ('Detection rate', 'detection_rate'),
    ):
        if key in results and results[key] == results[key]:   # skip NaN
            names.append(label)
            values.append(results[key])

    if not names:
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.barh(names, values, color='steelblue')
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Value')
    ax.set_title(f"Held-out metrics  --  {results['run']}\n"
                 f"test pulses {results.get('test_pulse_window')}  |  "
                 f"N={results['n_test']}  |  top {N_KEEP} eigenvalues")
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    for bar, val in zip(bars, values):
        ax.text(min(val + 0.01, 1.0), bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=9)

    fig.tight_layout()
    path = os.path.join(out_dir, 'metrics.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def save_accuracy_vs_jsr_png(y_true, y_pred, jsr, out_dir):
    """
    Exact accuracy vs the WEAKEST band's realized JSR, contaminated tiles only
    -> out_dir/accuracy_vs_jsr.png.

    Bands draw independent JSRs from [3, 30] dB, so a knee=3 tile whose weakest
    band landed at 4 dB is a far harder sample than one whose bands all landed
    at 25 dB: the weak band barely lifts its eigenvalue off the clutter floor,
    making the tile look like a lower knee. A single aggregate accuracy hides
    this; binning by minimum band JSR shows where the model actually fails.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if jsr is None:
        return

    rfi = y_true > 0
    if not rfi.any():
        return

    with np.errstate(invalid='ignore'):
        min_jsr = np.nanmin(jsr[rfi], axis=1)

    finite = np.isfinite(min_jsr)
    if not finite.any():
        return

    min_jsr = min_jsr[finite]
    correct = (y_pred[rfi][finite] == y_true[rfi][finite])

    edges = np.arange(np.floor(min_jsr.min()), np.ceil(min_jsr.max()) + 3.0, 3.0)
    centers, accs, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (min_jsr >= lo) & (min_jsr < hi)
        if sel.sum() < 20:
            continue
        centers.append(0.5 * (lo + hi))
        accs.append(float(correct[sel].mean()))
        counts.append(int(sel.sum()))

    if not centers:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(centers, accs, 'o-', color='steelblue')
    ax.set_xlabel('Weakest band JSR in the tile (dB)')
    ax.set_ylabel('Exact knee accuracy')
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title('Accuracy vs realized RFI strength (contaminated test tiles)')

    for x, y, c in zip(centers, accs, counts):
        ax.annotate(f'n={c}', (x, y), textcoords='offset points',
                    xytext=(0, 7), ha='center', fontsize=7)

    fig.tight_layout()
    path = os.path.join(out_dir, 'accuracy_vs_jsr.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

def evaluate(model, data, n_classes, run_name, out_dir):
    """
    Evaluate on the held-out test region.

    Clean tiles are part of this set (they are the tiles that drew 0 bands), so
    clean recall and the false-positive rate come out of the same pass as the
    knee-counting metrics -- row 0 of the confusion matrix.

    Reports the counting task (exact / tol-1 knee accuracy) and the detection
    task (contaminated vs not, ignoring the count) separately, because a model
    can be a strong detector while still confusing adjacent knee counts.

    Args:
        model: trained Keras model.
        data (dict): output of load_rfi_data_dir on the test dir.
        n_classes (int): label space size, taken from the TRAINING set.
        run_name (str): used in figure titles.
        out_dir (str): output directory.

    Returns:
        results (dict)
    """
    y_true = data['labels']
    y_pred = np.argmax(model.predict([data['eigen'], data['global']], verbose=0),
                       axis=-1)

    exact = float(np.mean(y_pred == y_true))
    tol1 = float(np.mean(np.abs(y_pred - y_true) <= 1))

    rfi_mask = y_true > 0
    clean_mask = y_true == 0

    exact_rfi = float(np.mean(y_pred[rfi_mask] == y_true[rfi_mask])) \
        if rfi_mask.any() else float('nan')
    tol1_rfi = float(np.mean(np.abs(y_pred[rfi_mask] - y_true[rfi_mask]) <= 1)) \
        if rfi_mask.any() else float('nan')

    # Clean recall and FPR are just row 0 of the confusion matrix
    clean_recall = float(np.mean(y_pred[clean_mask] == 0)) \
        if clean_mask.any() else float('nan')
    fpr = float(np.mean(y_pred[clean_mask] > 0)) if clean_mask.any() else float('nan')
    detection = float(np.mean(y_pred[rfi_mask] > 0)) if rfi_mask.any() else float('nan')

    per_class = {}
    for k in range(n_classes):
        sel = (y_true == k)
        if sel.any():
            per_class[str(k)] = {
                'n': int(sel.sum()),
                'recall': float(np.mean(y_pred[sel] == k)),
                'tol1': float(np.mean(np.abs(y_pred[sel] - k) <= 1)),
                'called_clean': float(np.mean(y_pred[sel] == 0)),
            }

    per_channel = {}
    for ch in sorted(set(data['channels'].tolist())):
        sel = (data['channels'] == ch)
        c_clean = sel & clean_mask
        c_rfi = sel & rfi_mask
        per_channel[ch] = {
            'n': int(sel.sum()),
            'exact_acc': float(np.mean(y_pred[sel] == y_true[sel])),
            'clean_recall': float(np.mean(y_pred[c_clean] == 0)) if c_clean.any() else float('nan'),
            'false_positive_rate': float(np.mean(y_pred[c_clean] > 0)) if c_clean.any() else float('nan'),
            'detection_rate': float(np.mean(y_pred[c_rfi] > 0)) if c_rfi.any() else float('nan'),
        }

    results = {
        'run': run_name,
        'n_test': int(len(y_true)),
        'n_keep': N_KEEP,
        'n_classes': n_classes,
        'test_pulse_window': data['meta'].get('pulse_window'),
        'exact_acc': exact,
        'tol1_acc': tol1,
        'exact_rfi': exact_rfi,
        'tol1_rfi': tol1_rfi,
        'clean_recall': clean_recall,
        'false_positive_rate': fpr,
        'detection_rate': detection,
        'per_class': per_class,
        'per_channel': per_channel,
    }

    print(f"\n=== HELD-OUT TEST REGION {data['meta'].get('pulse_window')} ===")
    print(f"  N tiles        : {len(y_true)}  (features: top {N_KEEP} eigenvalues)")
    print(f"  Exact knee acc : {exact:.4f}")
    print(f"  Tol-1 knee acc : {tol1:.4f}")
    print(f"  RFI exact / tol-1 : {exact_rfi:.4f} / {tol1_rfi:.4f}")
    print(f"  Clean recall   : {clean_recall:.4f}   (false positives: {fpr:.4f})")
    print(f"  Detection rate : {detection:.4f}   (contaminated called contaminated)")
    for k, s in per_class.items():
        tag = 'clean' if k == '0' else f'knee@{k}'
        print(f"    {tag}: recall={s['recall']:.3f}  tol1={s['tol1']:.3f}  "
              f"called clean={s['called_clean']:.3f}  (n={s['n']})")
    for ch, s in per_channel.items():
        print(f"    {ch}: exact={s['exact_acc']:.4f}  clean_recall={s['clean_recall']:.4f}  "
              f"FPR={s['false_positive_rate']:.4f}  detection={s['detection_rate']:.4f}")
    print("  Saving evaluation plots ...")

    save_confusion_matrix_png(
        y_true, y_pred, n_classes, out_dir,
        f"Knee Confusion Matrix -- held-out region "
        f"(pulses {data['meta'].get('pulse_window')})",
    )
    save_metrics_png(results, out_dir)
    save_accuracy_vs_jsr_png(y_true, y_pred, data['jsr'], out_dir)

    return results


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def train_model(run_name, n_classes,
                eigen_train, global_train, y_train,
                eigen_val, global_val, y_val,
                epochs, batch_size):
    """
    Build and train the model on the training region, validating on the held-out
    pulse-tile block of that same region.

    The architecture is untouched: build_model is called with cpi_size = N_KEEP,
    which only sets the eigen-branch input length. n_knee_classes is passed
    explicitly from the data, so truncating the features to 12 does not resize
    the output head (its default would otherwise become cpi_size + 1 = 13).

    Returns:
        model, out_dir
    """
    out_dir = os.path.join(MODELS_ROOT, run_name)
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'best_model.keras')

    print(f"\n{'='*60}")
    print(f"Run : {run_name}")
    print(f"  train={len(y_train)}  val={len(y_val)}")
    print(f"  eigen input : ({N_KEEP}, 2)   global: ({N_GLOBAL},)   classes: {n_classes}")
    print(f"{'='*60}")

    model = build_model(
        cpi_size=N_KEEP,                # feature length, not the pulse count
        n_global_features=N_GLOBAL,
        n_knee_classes=n_classes,       # stays tied to the label range
        dropout_rate=0.5,
        learning_rate=LR,
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path, monitor='val_loss',
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5,
            min_lr=1e-6, verbose=1,
        ),
    ]

    model.fit(
        x=[eigen_train, global_train],
        y=y_train,
        validation_data=([eigen_val, global_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    save_training_curves_png(model.history, out_dir)
    return model, out_dir


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(f'Train the RFI knee classifier on tile records from '
                     f'generate_rfi_data.py (top {N_KEEP} eigenvalues). Train and '
                     f'val come from the training region; the test set is a '
                     f'separate generator run over a disjoint pulse window.'),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data-dir', type=str, default='data/rfi_train',
                        help='Training region: rfi_data_<freq>_<pol>.h5, bands 0..6.')
    parser.add_argument('--test-dir', type=str, default='data/rfi_test',
                        help='Held-out region, same band range, clean interspersed.')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Model output folder. Defaults to the data dir name.')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Cap tiles loaded per file (evenly strided), for quick runs.')
    parser.add_argument('--val-frac', type=float, default=VAL_FRAC,
                        help='Fraction of training-region pulse tiles held out for val.')
    parser.add_argument('--split-buffer', type=int, default=SPLIT_BUFFER_DEFAULT,
                        help='Pulse tiles dropped at the train/val boundary.')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    return parser.parse_args()


def main():
    """
    Train on the training region (train/val split only), then evaluate on the
    held-out region, which is a separate generator run over a disjoint pulse
    window with the same 0..6 band range -- clean tiles interspersed, not split
    out into their own set.

    Only the top N_KEEP = 12 eigenvalues ever reach the model.
    """
    args = parse_args()

    os.makedirs(MODELS_ROOT, exist_ok=True)
    run_name = args.run_name or os.path.basename(os.path.normpath(args.data_dir))

    print(f"\n{'='*70}")
    print('RFI knee classifier training')
    print(f"{'='*70}")
    print(f"  train region : {args.data_dir}")
    print(f"  test  region : {args.test_dir}")
    print(f"  features     : top {N_KEEP} of {M} eigenvalues, linear-normalized then dB")
    print(f"  run name     : {run_name}")

    # ---------------- Training region -------------------------------------
    train_data = load_rfi_data_dir(args.data_dir, args.max_samples, tag='TRAIN region')
    n_classes = train_data['n_classes']

    print(f"\nTraining region: {len(train_data['labels'])} tiles   "
          f"classes: {n_classes}   channels: {', '.join(train_data['meta']['channels'])}")
    uniq, counts = np.unique(train_data['labels'], return_counts=True)
    for u, c in zip(uniq.tolist(), counts.tolist()):
        print(f"  label {u} ({'clean' if u == 0 else f'knee@{u}'}): {c}")

    idx_train, idx_val = split_train_val(
        train_data['groups'], args.val_frac, args.split_buffer
    )

    model, out_dir = train_model(
        run_name, n_classes,
        train_data['eigen'][idx_train], train_data['global'][idx_train],
        train_data['labels'][idx_train],
        train_data['eigen'][idx_val], train_data['global'][idx_val],
        train_data['labels'][idx_val],
        epochs=args.epochs, batch_size=args.batch_size,
    )

    # ---------------- Held-out test region --------------------------------
    test_data = load_rfi_data_dir(args.test_dir, args.max_samples, tag='TEST region')
    check_disjoint(train_data['meta'], test_data['meta'])

    uniq, counts = np.unique(test_data['labels'], return_counts=True)
    print(f"\nTest region: {len(test_data['labels'])} tiles")
    for u, c in zip(uniq.tolist(), counts.tolist()):
        print(f"  label {u} ({'clean' if u == 0 else f'knee@{u}'}): {c}")

    results = evaluate(model, test_data, n_classes, run_name, out_dir)

    results['n_train'] = int(len(idx_train))
    results['n_val'] = int(len(idx_val))
    results['train_provenance'] = train_data['meta']
    results['test_provenance'] = test_data['meta']

    with open(os.path.join(out_dir, 'eval_results.json'), 'w') as fh:
        json.dump(results, fh, indent=2)
    summary_path = os.path.join(MODELS_ROOT, f'{run_name}_summary.json')
    with open(summary_path, 'w') as fh:
        json.dump(results, fh, indent=2)

    print(f"\nSummary saved to {summary_path}")


if __name__ == '__main__':
    main()