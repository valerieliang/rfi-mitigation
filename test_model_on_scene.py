"""
test_model_on_scene.py

Run a PRETRAINED knee classifier over a NISAR L0B scene. No training here.

Two modes, and the distinction matters:

  --mode inference   (default)
      Runs the model on the REAL, UNMODIFIED data. There is no ground truth, so
      nothing here is an accuracy number. Every predicted knee > 0 is a CANDIDATE
      RFI detection, not a false positive: this scene may genuinely contain RFI
      (an urban descending pass very well might). The outputs are a knee map over
      the scene, a prediction histogram, and a per-tile HDF5 of predictions with
      confidences, so detections can be inspected against the eigenvalue profiles.

  --mode labeled
      Overlays synthetic RFI (0-6 bands, same generator as training) on the same
      real tiles, giving ground-truth labels on a scene the model has never seen.
      THIS is the generalization test: it answers "does the model transfer to a
      different scene with different backscatter statistics", which the held-out
      pulse window of the training granule cannot answer.

Feature extraction, SCM convention and the RFI injection model are imported from
train_db.py and generate_rfi_data.py rather than reimplemented, so the features
fed to the model here are byte-for-byte what it was trained on.

Outputs (per channel)
---------------------
    <out>/predictions_<freq>_<pol>.h5   per-tile knee, confidence, entropy, tile origin
    <out>/knee_map_<freq>_<pol>.png     predicted knee across the scene grid
    <out>/pred_hist_<freq>_<pol>.png    prediction histogram
    <out>/confusion_matrix.png          labeled mode only
    <out>/accuracy_vs_jsr.png           labeled mode only
    <out>/results.json
"""

import os
import sys
import json
import argparse

import numpy as np
import h5py

# Project modules: the SCM / injection / feature code lives there and is reused
# verbatim so the model sees exactly the features it was trained on.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_rfi_data import (  # noqa: E402
    read_raw_data_batch,
    get_subswath_mask,
    compute_scm_and_eigs,
    inject_rfi_bands,
    draw_n_bands,
    make_tile_seed_seq,
    channel_id,
    CPI_LEN_DEFAULT,
    CPI_WIDTH_DEFAULT,
    JSR_MIN_DB_DEFAULT,
    JSR_MAX_DB_DEFAULT,
    MIN_BANDS_DEFAULT,
    MAX_BANDS_DEFAULT,
    OFF_DIAG_OVERLAP_RATIO_DEFAULT,
    DIAG_VALID_RATIO_DEFAULT,
    PULSE_CHUNK_DEFAULT,
)
from train_db import features_from_eigenvalues, N_KEEP  # noqa: E402

from nisar.products.readers.Raw import Raw  # noqa: E402


EPS = 1e-12


# ---------------------------------------------------------------------------
# SCENE SCORING
# ---------------------------------------------------------------------------

def score_channel(raw, freq, pol, model, args):
    """
    Stream one channel of the scene, featurize every CPI tile, and predict.

    In labeled mode each tile is contaminated with 0-6 synthetic bands first,
    exactly as in training, and the drawn count is the ground-truth label. In
    inference mode the tile is left untouched and there is no label.

    Args:
        raw (Raw): ISCE3 Raw reader.
        freq (str): 'A' or 'B'.
        pol (str): 'HH' / 'HV' / ...
        model: loaded Keras model.
        args: parsed CLI args.

    Returns:
        rec (dict): per-tile arrays -- knee, confidence, entropy, labels (or None),
                    jsr (or None), tile_pulse, tile_range, grid shape.
    """
    cpi_len, cpi_width = args.cpi_len, args.cpi_width
    chan = channel_id(freq, pol)

    dataset = raw.getRawDataset(freq, pol)
    total_pulses, total_range = dataset.shape

    p_start = args.pulse_start
    p_end = min(args.pulse_end, total_pulses)
    r_start = args.range_start
    r_end = min(args.range_end, total_range) if args.range_end else total_range

    n_pt = (p_end - p_start) // cpi_len
    n_rt = (r_end - r_start) // cpi_width
    p_end = p_start + n_pt * cpi_len
    r_end = r_start + n_rt * cpi_width

    if n_pt <= 0 or n_rt <= 0:
        raise ValueError('Window smaller than one CPI tile')

    n_tiles = n_pt * n_rt
    print(f"\n[{freq}-{pol}]  pulses [{p_start}:{p_end}]  range [{r_start}:{r_end}]")
    print(f"  tile grid: {n_pt} x {n_rt} = {n_tiles} tiles")

    labeled = (args.mode == 'labeled')

    eigen_all = np.zeros((n_tiles, N_KEEP, 2), dtype=np.float32)
    global_all = np.zeros((n_tiles, 2), dtype=np.float32)
    labels = np.zeros(n_tiles, dtype=np.int8) if labeled else None
    jsr = np.full((n_tiles, args.max_bands), np.nan, dtype=np.float32) if labeled else None
    tile_pulse = np.zeros(n_tiles, dtype=np.int32)
    tile_range = np.zeros(n_tiles, dtype=np.int32)
    valid_frac = np.zeros(n_tiles, dtype=np.float32)

    chunk_tiles = max(1, args.pulse_chunk // cpi_len)
    k = 0

    for chunk_start in range(0, n_pt, chunk_tiles):
        n_here = min(chunk_tiles, n_pt - chunk_start)
        cp0 = p_start + chunk_start * cpi_len
        cp1 = cp0 + n_here * cpi_len

        raw_chunk = read_raw_data_batch(
            raw, freq, pol, slice(cp0, cp1), slice(r_start, r_end)
        )
        mask_chunk = (
            get_subswath_mask(raw, freq, pol,
                              np.arange(cp0, cp1), np.arange(r_start, r_end))
            if args.compute_subswath_mask else None
        )

        for lp in range(n_here):
            pt = chunk_start + lp
            lp0, lp1 = lp * cpi_len, (lp + 1) * cpi_len

            for rt in range(n_rt):
                lr0, lr1 = rt * cpi_width, (rt + 1) * cpi_width

                cpi = np.ascontiguousarray(raw_chunk[lp0:lp1, lr0:lr1]).astype(np.complex64)
                cpi_mask = (np.ascontiguousarray(mask_chunk[lp0:lp1, lr0:lr1])
                            if mask_chunk is not None else None)

                if labeled:
                    ss = make_tile_seed_seq(args.seed, chan, pt, rt)
                    n_bands = draw_n_bands(ss, args.min_bands, args.max_bands)
                    tile, meta = inject_rfi_bands(
                        cpi, cpi_mask, n_bands, args.jsr_min_db, args.jsr_max_db, ss
                    )
                    labels[k] = meta.knee
                    for bi, band in enumerate(meta.bands):
                        jsr[k, bi] = band.jsr_db
                else:
                    tile = cpi

                _, eigvals = compute_scm_and_eigs(
                    tile, cpi_mask,
                    args.off_diag_overlap_ratio, args.diag_valid_ratio
                )
                eigen_all[k], global_all[k] = features_from_eigenvalues(eigvals)

                tile_pulse[k] = p_start + pt * cpi_len
                tile_range[k] = r_start + lr0
                valid_frac[k] = (float(cpi_mask.sum()) / cpi_mask.size
                                 if cpi_mask is not None else 1.0)
                k += 1

        print(f"    pulse tiles {chunk_start + n_here}/{n_pt}")

    print(f"  predicting on {n_tiles} tiles ...")
    probs = model.predict([eigen_all, global_all], batch_size=args.batch_size, verbose=0)
    knee = np.argmax(probs, axis=-1).astype(np.int8)
    confidence = np.max(probs, axis=-1).astype(np.float32)
    entropy = (-np.sum(probs * np.log(probs + EPS), axis=-1)).astype(np.float32)

    return {
        'freq': freq, 'pol': pol, 'chan': f'{freq}-{pol}',
        'knee': knee, 'confidence': confidence, 'entropy': entropy,
        'labels': labels, 'jsr': jsr,
        'tile_pulse': tile_pulse, 'tile_range': tile_range,
        'valid_frac': valid_frac,
        'n_pt': n_pt, 'n_rt': n_rt,
        'pulse_window': [p_start, p_end], 'range_window': [r_start, r_end],
        'n_classes': probs.shape[-1],
    }


def save_predictions_h5(rec, args, out_dir):
    """Write the per-tile predictions so detections can be traced back to tiles."""
    path = os.path.join(out_dir, f"predictions_{rec['freq']}_{rec['pol']}.h5")
    with h5py.File(path, 'w') as f:
        f.attrs['granule'] = os.path.basename(args.l0b_file)
        f.attrs['model'] = os.path.basename(args.model)
        f.attrs['mode'] = args.mode
        f.attrs['frequency'] = rec['freq']
        f.attrs['polarization'] = rec['pol']
        f.attrs['pulse_start'] = rec['pulse_window'][0]
        f.attrs['pulse_end'] = rec['pulse_window'][1]
        f.attrs['range_start'] = rec['range_window'][0]
        f.attrs['range_end'] = rec['range_window'][1]
        f.attrs['n_pulse_tiles'] = rec['n_pt']
        f.attrs['n_range_tiles'] = rec['n_rt']
        f.attrs['gap_exclusion_used'] = bool(args.compute_subswath_mask)
        f.attrs['off_diag_overlap_ratio'] = args.off_diag_overlap_ratio
        f.attrs['diag_valid_ratio'] = args.diag_valid_ratio
        f.attrs['n_keep'] = N_KEEP

        f.create_dataset('knee', data=rec['knee'])
        f.create_dataset('confidence', data=rec['confidence'])
        f.create_dataset('entropy', data=rec['entropy'])
        f.create_dataset('tile_pulse', data=rec['tile_pulse'])
        f.create_dataset('tile_range', data=rec['tile_range'])
        f.create_dataset('valid_fraction', data=rec['valid_frac'])
        if rec['labels'] is not None:
            f.create_dataset('labels', data=rec['labels'])
            f.create_dataset('jsr_db', data=rec['jsr'])
            f.attrs['seed'] = args.seed

    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------

def plot_knee_map(rec, out_dir, mode):
    """
    Predicted knee across the scene grid (pulse tile x range tile).

    In inference mode this is the useful picture: real RFI shows up as coherent
    structure -- streaks along range, blocks in slow time -- while scattered
    isolated hits are more likely model noise on a hard tile.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    grid = rec['knee'].reshape(rec['n_pt'], rec['n_rt'])

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(grid, aspect='auto', cmap='inferno', vmin=0,
                   vmax=max(rec['n_classes'] - 1, 1), interpolation='nearest',
                   origin='lower',
                   extent=[rec['range_window'][0], rec['range_window'][1],
                           rec['pulse_window'][0], rec['pulse_window'][1]])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Predicted knee (0 = clean)')

    ax.set_xlabel('Range sample')
    ax.set_ylabel('Pulse')
    title = ('Predicted knee across scene -- UNMODIFIED data'
             if mode == 'inference' else
             'Predicted knee across scene -- synthetic RFI overlaid')
    ax.set_title(f"{title}\n{rec['chan']}")

    fig.tight_layout()
    path = os.path.join(out_dir, f"knee_map_{rec['freq']}_{rec['pol']}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_pred_hist(recs, out_dir, mode):
    """Prediction histogram per channel."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_classes = recs[0]['n_classes']
    x = np.arange(n_classes)
    width = 0.8 / len(recs)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, rec in enumerate(recs):
        counts = np.bincount(rec['knee'], minlength=n_classes)[:n_classes]
        frac = counts / max(counts.sum(), 1)
        ax.bar(x + i * width - 0.4 + width / 2, frac, width,
               label=f"{rec['chan']} (n={len(rec['knee'])})")

    ax.set_xticks(x)
    ax.set_xticklabels(['clean'] + [f'knee@{k}' for k in range(1, n_classes)])
    ax.set_ylabel('Fraction of tiles')
    ax.set_xlabel('Predicted class')
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend()

    if mode == 'inference':
        ax.set_title('Predictions on UNMODIFIED scene\n'
                     'no ground truth: knee > 0 are CANDIDATE detections, '
                     'not errors')
    else:
        ax.set_title('Predictions with synthetic RFI overlaid')

    fig.tight_layout()
    path = os.path.join(out_dir, 'pred_hist.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_confusion(y_true, y_pred, n_classes, out_dir, title):
    """Row-normalised confusion matrix (labeled mode only)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = ['clean'] + [f'knee@{k}' for k in range(1, n_classes)]
    matrix = np.zeros((n_classes, n_classes), dtype=np.int32)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        matrix[t, p] += 1
    normed = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(max(7, n_classes), max(6, n_classes * 0.9)))
    im = ax.imshow(normed, vmin=0, vmax=1, cmap='Blues')
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


def plot_accuracy_vs_jsr(y_true, y_pred, jsr, out_dir):
    """Exact accuracy vs the weakest band's realized JSR (labeled mode only)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

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
    ax.set_title('Accuracy vs realized RFI strength (new scene)')
    for x, y, c in zip(centers, accs, counts):
        ax.annotate(f'n={c}', (x, y), textcoords='offset points',
                    xytext=(0, 7), ha='center', fontsize=7)

    fig.tight_layout()
    path = os.path.join(out_dir, 'accuracy_vs_jsr.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# REPORTING
# ---------------------------------------------------------------------------

def report_inference(recs, results):
    """
    Summarize predictions on unmodified data.

    Deliberately avoids the words accuracy / false positive: there is no ground
    truth on a real scene. A knee > 0 here is a candidate detection that may well
    be real RFI, and the only way to tell is to look at the flagged tiles.
    """
    print(f"\n{'='*60}")
    print('INFERENCE ON UNMODIFIED SCENE  (no ground truth)')
    print(f"{'='*60}")

    for rec in recs:
        knee = rec['knee']
        n = len(knee)
        flagged = int((knee > 0).sum())
        dist = np.bincount(knee, minlength=rec['n_classes'])[:rec['n_classes']]

        # Coherent structure is the signature of real RFI: a range column that is
        # flagged across many pulse tiles is far more convincing than scattered hits
        grid = (knee > 0).reshape(rec['n_pt'], rec['n_rt'])
        col_rate = grid.mean(axis=0)
        hot_cols = int((col_rate > 0.5).sum())

        s = {
            'n_tiles': n,
            'flagged': flagged,
            'flagged_fraction': float(flagged / n),
            'prediction_distribution': {str(k): int(v) for k, v in enumerate(dist)},
            'mean_confidence': float(rec['confidence'].mean()),
            'mean_confidence_flagged': float(rec['confidence'][knee > 0].mean())
            if flagged else float('nan'),
            'range_tiles_flagged_over_half_the_scene': hot_cols,
            'n_range_tiles': rec['n_rt'],
        }
        results['channels'][rec['chan']] = s

        print(f"\n  {rec['chan']}: {n} tiles")
        print(f"    flagged knee>0 : {flagged} ({100*flagged/n:.2f}%)")
        print(f"    distribution   : "
              + ", ".join(f"{'clean' if k == 0 else f'knee@{k}'}={v}"
                          for k, v in enumerate(dist) if v))
        print(f"    mean confidence: {s['mean_confidence']:.3f}"
              + (f"  (flagged tiles: {s['mean_confidence_flagged']:.3f})"
                 if flagged else ''))
        print(f"    range tiles flagged over >50% of the scene: {hot_cols} / {rec['n_rt']}")
        print(f"      (a persistent range column is the signature of real RFI; "
              f"scattered isolated hits are more likely model noise)")

    print("\n  These are CANDIDATE detections, not errors. Inspect the knee map and "
          "the flagged tiles' eigenvalue profiles before drawing conclusions.")


def report_labeled(recs, results, out_dir):
    """Ground-truth evaluation on the new scene: does the model transfer?"""
    y_true = np.concatenate([r['labels'] for r in recs]).astype(np.int32)
    y_pred = np.concatenate([r['knee'] for r in recs]).astype(np.int32)
    jsr = np.concatenate([r['jsr'] for r in recs])
    n_classes = recs[0]['n_classes']

    exact = float(np.mean(y_pred == y_true))
    tol1 = float(np.mean(np.abs(y_pred - y_true) <= 1))

    clean = y_true == 0
    rfi = y_true > 0
    clean_recall = float(np.mean(y_pred[clean] == 0)) if clean.any() else float('nan')
    fpr = float(np.mean(y_pred[clean] > 0)) if clean.any() else float('nan')
    exact_rfi = float(np.mean(y_pred[rfi] == y_true[rfi])) if rfi.any() else float('nan')
    detection = float(np.mean(y_pred[rfi] > 0)) if rfi.any() else float('nan')

    per_class = {}
    for k in range(n_classes):
        sel = y_true == k
        if sel.any():
            per_class[str(k)] = {
                'n': int(sel.sum()),
                'recall': float(np.mean(y_pred[sel] == k)),
                'tol1': float(np.mean(np.abs(y_pred[sel] - k) <= 1)),
            }

    results.update({
        'exact_acc': exact, 'tol1_acc': tol1,
        'clean_recall': clean_recall, 'false_positive_rate': fpr,
        'exact_rfi': exact_rfi, 'detection_rate': detection,
        'per_class': per_class,
    })

    for rec in recs:
        t, p = rec['labels'].astype(np.int32), rec['knee'].astype(np.int32)
        results['channels'][rec['chan']] = {
            'n_tiles': int(len(t)),
            'exact_acc': float(np.mean(p == t)),
            'tol1_acc': float(np.mean(np.abs(p - t) <= 1)),
            'clean_recall': float(np.mean(p[t == 0] == 0)) if (t == 0).any() else float('nan'),
            'detection_rate': float(np.mean(p[t > 0] > 0)) if (t > 0).any() else float('nan'),
        }

    print(f"\n{'='*60}")
    print('LABELED TEST ON NEW SCENE  (synthetic RFI, real background)')
    print(f"{'='*60}")
    print(f"  N tiles        : {len(y_true)}")
    print(f"  Exact knee acc : {exact:.4f}")
    print(f"  Tol-1 knee acc : {tol1:.4f}")
    print(f"  Clean recall   : {clean_recall:.4f}   (flagged clean: {fpr:.4f})")
    print(f"  RFI exact      : {exact_rfi:.4f}")
    print(f"  Detection rate : {detection:.4f}")
    for k, s in per_class.items():
        tag = 'clean' if k == '0' else f'knee@{k}'
        print(f"    {tag}: recall={s['recall']:.3f}  tol1={s['tol1']:.3f}  (n={s['n']})")
    for ch, s in results['channels'].items():
        print(f"    {ch}: exact={s['exact_acc']:.4f}  detection={s['detection_rate']:.4f}")

    print("\n  NOTE: the 'clean' tiles here are real data assumed to be RFI-free. If "
          "this scene contains genuine RFI, some of those tiles are mislabeled and "
          "the clean recall shown is pessimistic.")

    plot_confusion(y_true, y_pred, n_classes, out_dir,
                   'Knee Confusion Matrix -- new scene (synthetic RFI overlaid)')
    plot_accuracy_vs_jsr(y_true, y_pred, jsr, out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Test a pretrained RFI knee classifier on a NISAR L0B scene.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('l0b_file', help='Input NISAR L0B HDF5 granule')
    parser.add_argument('--model', required=True,
                        help='Path to the trained Keras model (best_model.keras).')
    parser.add_argument('--mode', choices=['inference', 'labeled'], default='inference',
                        help="'inference' = unmodified data, no ground truth. "
                             "'labeled' = overlay synthetic RFI for a real accuracy test.")

    parser.add_argument('--freq', choices=['A', 'B'], default=None,
                        help='Default: every frequency in the granule.')
    parser.add_argument('--pol', default=None,
                        help='Default: every polarization in the granule.')

    parser.add_argument('--pulse-start', type=int, required=True)
    parser.add_argument('--pulse-end', type=int, required=True)
    parser.add_argument('--range-start', type=int, default=2000)
    parser.add_argument('--range-end', type=int, default=25000)

    parser.add_argument('--cpi-len', type=int, default=CPI_LEN_DEFAULT)
    parser.add_argument('--cpi-width', type=int, default=CPI_WIDTH_DEFAULT)
    parser.add_argument('--pulse-chunk', type=int, default=PULSE_CHUNK_DEFAULT)

    parser.add_argument('--compute-subswath-mask', action='store_true',
                        help='Use ISCE3 subswath boundaries for the gap-exclusion SCM. '
                             'Must match how the training data was generated.')
    parser.add_argument('--off-diag-overlap-ratio', type=float,
                        default=OFF_DIAG_OVERLAP_RATIO_DEFAULT)
    parser.add_argument('--diag-valid-ratio', type=float,
                        default=DIAG_VALID_RATIO_DEFAULT)

    # labeled mode only
    parser.add_argument('--min-bands', type=int, default=MIN_BANDS_DEFAULT)
    parser.add_argument('--max-bands', type=int, default=MAX_BANDS_DEFAULT)
    parser.add_argument('--jsr-min-db', type=float, default=JSR_MIN_DB_DEFAULT)
    parser.add_argument('--jsr-max-db', type=float, default=JSR_MAX_DB_DEFAULT)
    parser.add_argument('--seed', type=int, default=11,
                        help='Injection seed for labeled mode. Use a seed distinct '
                             'from the training and test runs.')

    parser.add_argument('--batch-size', type=int, default=4096,
                        help='Prediction batch size.')
    parser.add_argument('--output-dir', default='results/scene_test')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    import tensorflow as tf
    print(f"\n{'='*70}")
    print('Pretrained model -- scene test')
    print(f"{'='*70}")
    print(f"  granule : {args.l0b_file}")
    print(f"  model   : {args.model}")
    print(f"  mode    : {args.mode}")
    print(f"  pulses  : [{args.pulse_start}, {args.pulse_end})")
    print(f"  range   : [{args.range_start}, {args.range_end})")
    print(f"  features: top {N_KEEP} eigenvalues, gap_exclusion="
          f"{args.compute_subswath_mask} ({args.off_diag_overlap_ratio}/"
          f"{args.diag_valid_ratio})")

    model = tf.keras.models.load_model(args.model)

    raw = Raw(hdf5file=args.l0b_file)
    raw.parsePolarizations()

    freqs = [args.freq] if args.freq else list(raw.polarizations.keys())
    channels = []
    for freq in freqs:
        if freq not in raw.polarizations:
            continue
        pols = ([args.pol] if args.pol and args.pol in raw.polarizations[freq]
                else list(raw.polarizations[freq]))
        channels.extend((freq, p) for p in pols)

    if not channels:
        raise ValueError('No matching channels in this granule')
    print(f"  channels: " + ', '.join(f'{f}-{p}' for f, p in channels))

    recs = [score_channel(raw, f, p, model, args) for f, p in channels]

    results = {
        'granule': os.path.basename(args.l0b_file),
        'model': args.model,
        'mode': args.mode,
        'pulse_window': [args.pulse_start, args.pulse_end],
        'range_window': [args.range_start, args.range_end],
        'n_keep': N_KEEP,
        'gap_exclusion_used': bool(args.compute_subswath_mask),
        'off_diag_overlap_ratio': args.off_diag_overlap_ratio,
        'diag_valid_ratio': args.diag_valid_ratio,
        'channels': {},
    }

    for rec in recs:
        save_predictions_h5(rec, args, args.output_dir)
        plot_knee_map(rec, args.output_dir, args.mode)
    plot_pred_hist(recs, args.output_dir, args.mode)

    if args.mode == 'labeled':
        results['seed'] = args.seed
        report_labeled(recs, results, args.output_dir)
    else:
        report_inference(recs, results)

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved to {os.path.join(args.output_dir, 'results.json')}")


if __name__ == '__main__':
    main()