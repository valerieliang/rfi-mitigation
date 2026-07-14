"""
score_scene.py

Run a PRETRAINED knee classifier over a NISAR L0B scene that has NO LABELS.

There is no ground truth on a real scene, so there is no accuracy number to
report, and this script never pretends otherwise. A predicted knee > 0 is a
CANDIDATE DETECTION -- it may be genuine RFI (an urban descending pass very
plausibly contains some) or a model error, and nothing in the data alone
distinguishes the two. What the script produces instead is the evidence you need
to judge the detections yourself:

  1. SPATIAL MAPS
     Predicted knee and model confidence over the (pulse tile x range tile) grid.
     This is the single most informative output. Real RFI has STRUCTURE: it
     persists along range (a jammer occupies a band, so a range column lights up
     across many pulses) or arrives in bursts in slow time. Model noise is
     scattered and isolated. A coherent streak is believable; confetti is not.

  2. POWER vs PREDICTION
     Tile baseline power (the mean |x|^2 the model's JSR was always referenced
     to) broken out by predicted class. The model was trained on a scale-invariant
     feature (eigenvalues normalized by lambda_max, then dB), so its prediction
     should NOT correlate with absolute tile power. If it does -- if high-power
     tiles are systematically flagged -- that is the model keying on backscatter
     brightness rather than eigenvalue structure, and it is the failure mode to
     watch for when moving to a new scene.

  3. CONFIDENCE PER GUESS
     Max softmax and the entropy of the full posterior, per predicted class.
     Low-confidence / high-entropy detections are the ones to distrust first.
     Training showed clean-vs-knee@1 is the genuinely hard boundary, so expect
     the knee@1 column to carry the least confident calls.

  4. EIGENVALUE PROFILES BY PREDICTED CLASS
     The mean normalized eigenvalue profile for each predicted class. A tile
     called knee@3 should show three eigenvalues standing above the floor. If
     the knee@3 profile does not have a visible knee at index 3, the model is
     not doing what its label says.

The per-tile HDF5 keeps everything (knee, confidence, entropy, baseline power,
the full linear eigenvalue vector, tile origin), so any flagged tile can be
pulled back out and inspected against the raw data.

If you want an actual NUMBER on this scene, the only honest way is to overlay
synthetic RFI on it -- real background, known labels -- which is what
generate_rfi_data.py does. Point it at this granule and retest. That measures
transfer to this scene's backscatter statistics; it cannot tell you what RFI is
already there.

Feature extraction and the SCM convention are imported from train_db.py and
generate_rfi_data.py rather than reimplemented, so the model sees exactly the
features it was trained on.

By default the FULL range extent of the granule is scored. --range-start and
--range-end are there to narrow it, not to define it. Note that the training set
was built over a restricted range window, so scoring the full swath will include
near- and far-range geometry the model never saw; the gap-exclusion mask handles
the inter-subswath gaps, but tiles at the extreme edges are worth treating with
extra suspicion.

Usage
-----
    py-isce3 score_scene.py \\
        /scratch/bohuang/rfi/la/NISAR_L0_PR_RRSD_006_112_D_197S_20251006T024004_20251006T024139_P00410_F_J_001.h5 \\
        --model models/rfi_train/best_model.keras \\
        --pulse-start 46528 --pulse-end 124580 \\
        --compute-subswath-mask \\
        --off-diag-overlap-ratio 0.03 --diag-valid-ratio 0.02 \\
        --output-dir results/la_scene

Outputs (per channel unless noted)
----------------------------------
    predictions_<freq>_<pol>.h5     per-tile knee, confidence, entropy, power, eigenvalues
    knee_map_<freq>_<pol>.png       predicted knee over the scene grid
    confidence_map_<freq>_<pol>.png model confidence over the scene grid
    power_vs_knee_<freq>_<pol>.png  tile baseline power by predicted class
    confidence_by_knee_<freq>_<pol>.png  confidence + entropy by predicted class
    eigen_profiles_<freq>_<pol>.png mean eigenvalue profile by predicted class
    selected_predictions_<freq>_<pol>.png  individual tiles: profile + what the
                                           model called it, one row per class
    pred_hist.png                   prediction histogram, all channels
    results.json                    summary statistics
"""

import os
import sys
import json
import argparse

import numpy as np
import h5py

# Reuse the project's SCM / feature code so the model sees identical inputs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_rfi_data import (  # noqa: E402
    read_raw_data_batch,
    get_subswath_mask,
    compute_scm_and_eigs,
    tile_signal_power,
    CPI_LEN_DEFAULT,
    CPI_WIDTH_DEFAULT,
    OFF_DIAG_OVERLAP_RATIO_DEFAULT,
    DIAG_VALID_RATIO_DEFAULT,
    PULSE_CHUNK_DEFAULT,
)
from train_db import features_from_eigenvalues, N_KEEP  # noqa: E402

from nisar.products.readers.Raw import Raw  # noqa: E402


EPS = 1e-12
M = 16  # eigenvalues per CPI


# ---------------------------------------------------------------------------
# SCENE SCORING
# ---------------------------------------------------------------------------

def score_channel(raw, freq, pol, model, args):
    """
    Stream one channel of the scene, featurize every CPI tile, and predict.

    The tiles are left completely unmodified -- this is the real scene as
    recorded. Per tile we keep the prediction, its confidence and entropy, the
    baseline power (so power can be cross-plotted against the prediction), and
    the full linear eigenvalue vector (so any tile can be re-examined later
    without re-reading the granule).

    Args:
        raw (Raw): ISCE3 Raw reader.
        freq (str): 'A' or 'B'.
        pol (str): 'HH' / 'HV' / ...
        model: loaded Keras model.
        args: parsed CLI args.

    Returns:
        rec (dict): per-tile arrays plus the grid geometry.
    """
    cpi_len, cpi_width = args.cpi_len, args.cpi_width

    dataset = raw.getRawDataset(freq, pol)
    total_pulses, total_range = dataset.shape

    p_start = args.pulse_start
    p_end = min(args.pulse_end, total_pulses)

    # No range window given -> process the full swath width. Only whole CPI tiles
    # are scored, so a trailing partial tile at the far edge is dropped.
    r_start = args.range_start if args.range_start is not None else 0
    r_end = (min(args.range_end, total_range) if args.range_end is not None
             else total_range)

    n_pt = (p_end - p_start) // cpi_len
    n_rt = (r_end - r_start) // cpi_width
    p_end = p_start + n_pt * cpi_len
    r_end = r_start + n_rt * cpi_width

    if n_pt <= 0 or n_rt <= 0:
        raise ValueError('Window is smaller than one CPI tile')

    n_tiles = n_pt * n_rt
    print(f"\n[{freq}-{pol}]  pulses [{p_start}:{p_end}]  range [{r_start}:{r_end}]")
    print(f"  tile grid: {n_pt} x {n_rt} = {n_tiles} tiles")

    eigen_all = np.zeros((n_tiles, N_KEEP, 2), dtype=np.float32)
    global_all = np.zeros((n_tiles, 2), dtype=np.float32)
    eigvals_all = np.zeros((n_tiles, M), dtype=np.float32)   # linear, descending
    power_db = np.zeros(n_tiles, dtype=np.float32)
    valid_frac = np.zeros(n_tiles, dtype=np.float32)
    tile_pulse = np.zeros(n_tiles, dtype=np.int32)
    tile_range = np.zeros(n_tiles, dtype=np.int32)

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

                cpi = np.ascontiguousarray(
                    raw_chunk[lp0:lp1, lr0:lr1]).astype(np.complex64)
                cpi_mask = (np.ascontiguousarray(mask_chunk[lp0:lp1, lr0:lr1])
                            if mask_chunk is not None else None)

                _, eigvals = compute_scm_and_eigs(
                    cpi, cpi_mask,
                    args.off_diag_overlap_ratio, args.diag_valid_ratio
                )

                eigen_all[k], global_all[k] = features_from_eigenvalues(eigvals)
                eigvals_all[k] = eigvals
                power_db[k] = 10.0 * np.log10(tile_signal_power(cpi, cpi_mask))
                valid_frac[k] = (float(cpi_mask.sum()) / cpi_mask.size
                                 if cpi_mask is not None else 1.0)
                tile_pulse[k] = p_start + pt * cpi_len
                tile_range[k] = r_start + lr0
                k += 1

        print(f"    pulse tiles {chunk_start + n_here}/{n_pt}")

    print(f"  predicting on {n_tiles} tiles ...")
    probs = model.predict([eigen_all, global_all],
                          batch_size=args.batch_size, verbose=0)

    knee = np.argmax(probs, axis=-1).astype(np.int8)
    confidence = np.max(probs, axis=-1).astype(np.float32)
    entropy = (-np.sum(probs * np.log(probs + EPS), axis=-1)).astype(np.float32)

    return {
        'freq': freq, 'pol': pol, 'chan': f'{freq}-{pol}',
        'knee': knee, 'confidence': confidence, 'entropy': entropy,
        'eigvals': eigvals_all, 'power_db': power_db, 'valid_frac': valid_frac,
        'tile_pulse': tile_pulse, 'tile_range': tile_range,
        'n_pt': n_pt, 'n_rt': n_rt,
        'pulse_window': [p_start, p_end], 'range_window': [r_start, r_end],
        'n_classes': probs.shape[-1],
    }


def save_predictions_h5(rec, args, out_dir):
    """Per-tile record, so any flagged tile can be traced back and re-examined."""
    path = os.path.join(out_dir, f"predictions_{rec['freq']}_{rec['pol']}.h5")
    with h5py.File(path, 'w') as f:
        f.attrs['granule'] = os.path.basename(args.l0b_file)
        f.attrs['model'] = os.path.basename(args.model)
        f.attrs['labeled'] = False
        f.attrs['note'] = ('unlabeled scene: knee > 0 are candidate detections, '
                           'not verified RFI and not errors')
        f.attrs['frequency'] = rec['freq']
        f.attrs['polarization'] = rec['pol']
        f.attrs['pulse_start'] = rec['pulse_window'][0]
        f.attrs['pulse_end'] = rec['pulse_window'][1]
        f.attrs['range_start'] = rec['range_window'][0]
        f.attrs['range_end'] = rec['range_window'][1]
        f.attrs['n_pulse_tiles'] = rec['n_pt']
        f.attrs['n_range_tiles'] = rec['n_rt']
        f.attrs['cpi_len'] = args.cpi_len
        f.attrs['cpi_width'] = args.cpi_width
        f.attrs['gap_exclusion_used'] = bool(args.compute_subswath_mask)
        f.attrs['off_diag_overlap_ratio'] = args.off_diag_overlap_ratio
        f.attrs['diag_valid_ratio'] = args.diag_valid_ratio
        f.attrs['n_keep'] = N_KEEP

        f.create_dataset('knee', data=rec['knee'])
        f.create_dataset('confidence', data=rec['confidence'])
        f.create_dataset('entropy', data=rec['entropy'])
        f.create_dataset('eigenvalues', data=rec['eigvals'], compression='gzip')
        f.create_dataset('signal_power_db', data=rec['power_db'])
        f.create_dataset('valid_fraction', data=rec['valid_frac'])
        f.create_dataset('tile_pulse', data=rec['tile_pulse'])
        f.create_dataset('tile_range', data=rec['tile_range'])

    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------

def _boxplot(ax, groups, labels, **kwargs):
    """
    Version-agnostic boxplot.

    Matplotlib 3.9 renamed the 'labels' kwarg to 'tick_labels' and 3.11 removed
    the old name outright, so passing either one directly breaks on some installs.
    Try the new name, fall back to the old.
    """
    try:
        return ax.boxplot(groups, tick_labels=labels, **kwargs)
    except TypeError:
        return ax.boxplot(groups, labels=labels, **kwargs)


def plot_knee_map(rec, out_dir):
    """
    Predicted knee over the scene grid.

    The key read: real RFI is STRUCTURED. A jammer sits in a frequency band, so
    it lights up a range column across many pulse tiles; a burst emitter lights
    up a slow-time block. Scattered single-tile hits with no neighbours are much
    more likely to be model noise on a hard tile.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    grid = rec['knee'].reshape(rec['n_pt'], rec['n_rt'])

    fig, ax = plt.subplots(figsize=(13, 6))
    # origin='upper' plus a top-down extent puts the FIRST pulse at the top of the
    # axis and runs time downward, matching how a radar swath is normally read.
    im = ax.imshow(grid, aspect='auto', cmap='inferno', origin='upper',
                   vmin=0, vmax=max(rec['n_classes'] - 1, 1),
                   interpolation='nearest',
                   extent=[rec['range_window'][0], rec['range_window'][1],
                           rec['pulse_window'][1], rec['pulse_window'][0]])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Predicted knee (0 = clean)')

    ax.set_xlabel('Range sample')
    ax.set_ylabel('Pulse')
    ax.set_title(f"Predicted knee across scene -- {rec['chan']} (UNMODIFIED data)\n"
                 f"no ground truth: knee > 0 are candidate detections. "
                 f"Look for coherent range columns / slow-time blocks.")

    fig.tight_layout()
    path = os.path.join(out_dir, f"knee_map_{rec['freq']}_{rec['pol']}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_confidence_map(rec, out_dir):
    """
    Model confidence over the same grid, so it can be laid alongside the knee map.

    A detection that is both spatially coherent AND high-confidence is the
    strongest evidence of real RFI. A high-knee call with low confidence sitting
    alone is the weakest.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    grid = rec['confidence'].reshape(rec['n_pt'], rec['n_rt'])

    fig, ax = plt.subplots(figsize=(13, 6))
    # Same top-down pulse axis as the knee map so the two can be read side by side
    im = ax.imshow(grid, aspect='auto', cmap='viridis', origin='upper',
                   vmin=0, vmax=1, interpolation='nearest',
                   extent=[rec['range_window'][0], rec['range_window'][1],
                           rec['pulse_window'][1], rec['pulse_window'][0]])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Max softmax probability')

    ax.set_xlabel('Range sample')
    ax.set_ylabel('Pulse')
    ax.set_title(f"Model confidence across scene -- {rec['chan']}\n"
                 f"compare against the knee map: coherent AND confident is the "
                 f"strongest evidence")

    fig.tight_layout()
    path = os.path.join(out_dir, f"confidence_map_{rec['freq']}_{rec['pol']}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_power_vs_knee(rec, out_dir):
    """
    Tile baseline power, broken out by predicted class.

    This is a sanity check on WHAT the model is keying on. Its features are
    scale-invariant by construction (eigenvalues normalized by lambda_max before
    the dB conversion), so the predicted knee should be roughly INDEPENDENT of
    absolute tile power. If the boxes march upward with knee -- if bright tiles
    get flagged and dim ones do not -- the model is riding backscatter brightness
    rather than eigenvalue structure, which would not transfer to a new scene.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_classes = rec['n_classes']
    knee, power = rec['knee'], rec['power_db']

    groups, labels, counts = [], [], []
    for k in range(n_classes):
        sel = (knee == k)
        if sel.sum() >= 10:
            groups.append(power[sel])
            labels.append('clean' if k == 0 else f'knee@{k}')
            counts.append(int(sel.sum()))

    if not groups:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bp = _boxplot(ax1, groups, labels, showfliers=False, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.6)
    ax1.set_ylabel('Tile baseline power (dB)')
    ax1.set_xlabel('Predicted class')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax1.set_title('Baseline power by predicted class\n'
                  'features are scale-invariant, so these should be FLAT')
    for i, c in enumerate(counts):
        ax1.annotate(f'n={c}', (i + 1, ax1.get_ylim()[1]),
                     textcoords='offset points', xytext=(0, -12),
                     ha='center', fontsize=7)

    # Scatter of a subsample, colored by confidence
    n = len(knee)
    idx = np.random.default_rng(0).choice(n, size=min(20000, n), replace=False)
    sc = ax2.scatter(knee[idx] + np.random.default_rng(1).uniform(-0.25, 0.25, len(idx)),
                     power[idx], c=rec['confidence'][idx], cmap='viridis',
                     s=3, alpha=0.4, vmin=0, vmax=1)
    fig.colorbar(sc, ax=ax2, label='Confidence')
    ax2.set_xticks(range(n_classes))
    ax2.set_xticklabels(['clean'] + [f'{k}' for k in range(1, n_classes)])
    ax2.set_xlabel('Predicted knee')
    ax2.set_ylabel('Tile baseline power (dB)')
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.set_title('Power vs prediction (subsample), colored by confidence')

    fig.suptitle(f"Power vs prediction -- {rec['chan']}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(out_dir, f"power_vs_knee_{rec['freq']}_{rec['pol']}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_confidence_by_knee(rec, out_dir):
    """
    Confidence and posterior entropy per predicted class.

    Without labels, confidence is the closest thing to a per-guess quality score.
    Low confidence / high entropy marks the calls to distrust first. Training
    showed clean-vs-knee@1 is the genuinely hard boundary (the weakest band
    barely clears the clutter floor), so a soft knee@1 column is expected, not
    alarming.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_classes = rec['n_classes']
    knee = rec['knee']

    conf_groups, ent_groups, labels, counts = [], [], [], []
    for k in range(n_classes):
        sel = (knee == k)
        if sel.sum() >= 10:
            conf_groups.append(rec['confidence'][sel])
            ent_groups.append(rec['entropy'][sel])
            labels.append('clean' if k == 0 else f'knee@{k}')
            counts.append(int(sel.sum()))

    if not conf_groups:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bp1 = _boxplot(ax1, conf_groups, labels, showfliers=False, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.6)
    ax1.set_ylabel('Max softmax probability')
    ax1.set_xlabel('Predicted class')
    ax1.set_ylim(0, 1.02)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax1.set_title('Confidence per guess')
    for i, c in enumerate(counts):
        ax1.annotate(f'n={c}', (i + 1, 0.02), ha='center', fontsize=7)

    bp2 = _boxplot(ax2, ent_groups, labels, showfliers=False, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('indianred')
        patch.set_alpha(0.6)
    ax2.set_ylabel('Posterior entropy (nats)')
    ax2.set_xlabel('Predicted class')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax2.set_title('Uncertainty per guess (high = distrust first)')

    fig.suptitle(f"Confidence by predicted class -- {rec['chan']}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(out_dir, f"confidence_by_knee_{rec['freq']}_{rec['pol']}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_eigen_profiles_by_class(rec, out_dir):
    """
    Mean normalized eigenvalue profile per predicted class -- the physical check.

    A tile the model calls knee@k should show k eigenvalues standing above the
    clutter floor, with the drop-off right after index k. If the knee@3 curve has
    no visible knee at 3, the model is not doing what its label claims, and no
    amount of spatial coherence in the map would redeem that.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    n_classes = rec['n_classes']
    knee = rec['knee']

    # Same normalization the model sees: linear / lambda_max, then dB
    ev = np.maximum(rec['eigvals'][:, :N_KEEP], EPS)
    ev_db = 10.0 * np.log10(ev / np.maximum(ev[:, :1], EPS))
    idx = np.arange(1, N_KEEP + 1)

    norm = mcolors.Normalize(vmin=0, vmax=max(n_classes - 1, 1))
    cmap = cm.plasma

    fig, ax = plt.subplots(figsize=(10, 6))
    for k in range(n_classes):
        sel = (knee == k)
        if sel.sum() < 10:
            continue
        mean_prof = ev_db[sel].mean(axis=0)
        ax.plot(idx, mean_prof, color=cmap(norm(k)), linewidth=2,
                label=f"{'clean' if k == 0 else f'knee@{k}'} (n={int(sel.sum())})")
        if k > 0:
            ax.plot(k, mean_prof[k - 1], 'rx', markersize=8, markeredgewidth=2)

    ax.set_xlabel('Eigenvalue index (1-based, descending)')
    ax.set_ylabel('Eigenvalue (dB, normalized to lambda_max)')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=9)
    ax.set_title(f"Mean eigenvalue profile by predicted class -- {rec['chan']}\n"
                 f"a knee@k call should show its drop-off right after index k "
                 f"(red x)")

    fig.tight_layout()
    path = os.path.join(out_dir, f"eigen_profiles_{rec['freq']}_{rec['pol']}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_selected_predictions(rec, out_dir, n_per_class, seed):
    """
    Grid of individual tiles: eigenvalue profile + what the model called it.

    The mean-profile plot shows the model is right ON AVERAGE. This shows what
    single tiles actually look like, which is where the failures live -- a mean
    curve happily hides a class whose members are half convincing and half
    nonsense.

    One row per predicted class, n_per_class randomly chosen examples across the
    row (fixed seed, so the selection is reproducible). Each panel draws the
    tile's normalized eigenvalue profile with a red marker at the predicted knee
    index. The panel is believable when the drop-off sits right after that
    marker, and suspicious when it does not.

    Panel titles carry the numbers needed to triage a detection without labels:
    confidence, tile baseline power, and the tile's (pulse, range) origin so it
    can be found in the knee map and pulled out of the predictions HDF5.

    Args:
        rec (dict): scored channel record.
        out_dir (str): output directory.
        n_per_class (int): examples per predicted class (grid columns).
        seed (int): selection seed, independent of everything else.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    n_classes = rec['n_classes']
    knee = rec['knee']
    rng = np.random.default_rng(seed)

    # Same normalization the model sees: linear / lambda_max, then dB
    ev = np.maximum(rec['eigvals'][:, :N_KEEP], EPS)
    ev_db = 10.0 * np.log10(ev / np.maximum(ev[:, :1], EPS))
    idx = np.arange(1, N_KEEP + 1)

    present = [k for k in range(n_classes) if (knee == k).sum() > 0]
    if not present:
        return

    norm = mcolors.Normalize(vmin=0, vmax=max(n_classes - 1, 1))
    cmap = cm.plasma

    # Shared y-range so panels are directly comparable to each other
    sample_all = ev_db[rng.choice(len(knee), size=min(5000, len(knee)), replace=False)]
    ylim = [float(np.percentile(sample_all, 0.5)) - 3.0, 3.0]

    n_rows, n_cols = len(present), n_per_class
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.9 * n_cols, 2.5 * n_rows),
                             squeeze=False)

    for ri, k in enumerate(present):
        pool = np.where(knee == k)[0]
        pick = rng.choice(pool, size=min(n_per_class, len(pool)), replace=False)

        for ci in range(n_cols):
            ax = axes[ri][ci]

            if ci >= len(pick):
                ax.axis('off')
                continue

            t = int(pick[ci])
            ax.plot(idx, ev_db[t], color=cmap(norm(k)), linewidth=1.6)

            if k > 0:
                # Red marker at the predicted knee: the drop-off should follow it
                ax.plot(k, ev_db[t, k - 1], 'rx', markersize=8, markeredgewidth=2)
                ax.axvline(x=k, color='red', linestyle='--', alpha=0.35, linewidth=1)

            label = 'CLEAN' if k == 0 else f'knee@{k}'
            ax.set_title(
                f"{label}  conf={rec['confidence'][t]:.2f}\n"
                f"p={rec['tile_pulse'][t]} r={rec['tile_range'][t]}  "
                f"pow={rec['power_db'][t]:.1f} dB",
                fontsize=7,
            )
            ax.set_ylim(ylim)
            ax.set_xticks([1, 4, 8, 12])
            ax.tick_params(labelsize=6)
            ax.grid(True, linestyle='--', alpha=0.35)

            if ci == 0:
                ax.set_ylabel('EV (dB)', fontsize=8)
            if ri == n_rows - 1:
                ax.set_xlabel('EV index', fontsize=8)

    fig.suptitle(
        f"Selected predictions -- {rec['chan']}  "
        f"({n_per_class} random tiles per predicted class, seed={seed})\n"
        f"red x = predicted knee; the profile should drop off just after it",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(out_dir, f"selected_predictions_{rec['freq']}_{rec['pol']}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_pred_hist(recs, out_dir):
    """Prediction histogram across channels."""
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
    ax.set_title('Predictions on UNMODIFIED scene\n'
                 'no ground truth: knee > 0 are candidate detections, not errors')

    fig.tight_layout()
    path = os.path.join(out_dir, 'pred_hist.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# REPORTING
# ---------------------------------------------------------------------------

def report(recs, results):
    """
    Summarize what the model saw. No accuracy language: there is no ground truth.

    The one quantitative structure cue reported here is per-range-column flag
    rate. A jammer occupies a frequency band, which maps to a persistent range
    column, so a column flagged across most of the scene is a much stronger RFI
    candidate than the same number of scattered hits.
    """
    print(f"\n{'='*60}")
    print('SCENE SCORING  (unlabeled -- no accuracy can be computed)')
    print(f"{'='*60}")

    for rec in recs:
        knee, conf = rec['knee'], rec['confidence']
        n = len(knee)
        flagged = knee > 0
        n_flag = int(flagged.sum())
        dist = np.bincount(knee, minlength=rec['n_classes'])[:rec['n_classes']]

        grid = flagged.reshape(rec['n_pt'], rec['n_rt'])
        col_rate = grid.mean(axis=0)
        row_rate = grid.mean(axis=1)
        hot_cols = np.where(col_rate > 0.5)[0]

        # Power correlation: should be near zero if the model is scale-invariant
        if n_flag and n_flag < n:
            corr = float(np.corrcoef(knee.astype(float), rec['power_db'])[0, 1])
        else:
            corr = float('nan')

        s = {
            'n_tiles': n,
            'flagged': n_flag,
            'flagged_fraction': float(n_flag / n),
            'prediction_distribution': {str(k): int(v) for k, v in enumerate(dist)},
            'mean_confidence': float(conf.mean()),
            'mean_confidence_flagged': float(conf[flagged].mean()) if n_flag else float('nan'),
            'mean_confidence_clean': float(conf[~flagged].mean()) if (~flagged).any() else float('nan'),
            'knee_vs_power_correlation': corr,
            'range_columns_flagged_over_half': [int(c) for c in hot_cols],
            'n_range_tiles': rec['n_rt'],
            'max_slowtime_block_flag_rate': float(row_rate.max()),
        }
        results['channels'][rec['chan']] = s

        print(f"\n  {rec['chan']}: {n} tiles")
        print(f"    flagged knee>0   : {n_flag} ({100 * n_flag / n:.2f}%)")
        print(f"    distribution     : "
              + ", ".join(f"{'clean' if k == 0 else f'knee@{k}'}={v}"
                          for k, v in enumerate(dist) if v))
        print(f"    mean confidence  : {s['mean_confidence']:.3f}  "
              f"(flagged {s['mean_confidence_flagged']:.3f}, "
              f"clean {s['mean_confidence_clean']:.3f})")
        print(f"    knee vs power r  : {corr:.3f}   "
              f"<- should be near 0; a large value means the model is keying on "
              f"brightness, not structure")
        print(f"    range columns flagged in >50% of the scene: "
              f"{len(hot_cols)} / {rec['n_rt']}"
              + (f"  -> {hot_cols[:12].tolist()}" if len(hot_cols) else ''))
        print(f"    busiest slow-time block: {100 * s['max_slowtime_block_flag_rate']:.1f}% "
              f"of its range tiles flagged")

    print("\n  Reading the result:")
    print("    - Coherent range columns or slow-time blocks  -> likely REAL RFI.")
    print("    - Scattered, isolated, low-confidence hits    -> likely model noise.")
    print("    - knee-vs-power correlation far from 0        -> the model is riding")
    print("      backscatter brightness; distrust everything on this scene.")
    print("    For a hard number on this scene, overlay synthetic RFI on it with")
    print("    generate_rfi_data.py and retest -- real background, known labels.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Score an UNLABELED NISAR L0B scene with a pretrained knee '
                    'classifier. Produces spatial maps, power-vs-prediction and '
                    'confidence diagnostics. No accuracy is reported: there is no '
                    'ground truth on a real scene.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('l0b_file', help='Input NISAR L0B HDF5 granule')
    parser.add_argument('--model', required=True,
                        help='Trained Keras model (e.g. models/rfi_train/best_model.keras)')

    parser.add_argument('--freq', choices=['A', 'B'], default=None,
                        help='Default: every frequency in the granule.')
    parser.add_argument('--pol', default=None,
                        help='Default: every polarization in the granule.')

    parser.add_argument('--pulse-start', type=int, required=True)
    parser.add_argument('--pulse-end', type=int, required=True)
    parser.add_argument('--range-start', type=int, default=None,
                        help='Default: 0 (start of the swath).')
    parser.add_argument('--range-end', type=int, default=None,
                        help='Default: the full range extent of the granule.')

    parser.add_argument('--cpi-len', type=int, default=CPI_LEN_DEFAULT)
    parser.add_argument('--cpi-width', type=int, default=CPI_WIDTH_DEFAULT)
    parser.add_argument('--pulse-chunk', type=int, default=PULSE_CHUNK_DEFAULT)

    parser.add_argument('--compute-subswath-mask', action='store_true',
                        help='Gap-exclusion SCM via ISCE3 subswaths. Must match how '
                             'the training data was generated.')
    parser.add_argument('--off-diag-overlap-ratio', type=float,
                        default=OFF_DIAG_OVERLAP_RATIO_DEFAULT)
    parser.add_argument('--diag-valid-ratio', type=float,
                        default=DIAG_VALID_RATIO_DEFAULT)

    parser.add_argument('--n-examples-per-class', type=int, default=6,
                        help='Tiles per predicted class in the selected-predictions grid.')
    parser.add_argument('--example-seed', type=int, default=42,
                        help='Fixed seed for choosing which tiles to show.')

    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--output-dir', default='results/scene')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    import tensorflow as tf

    print(f"\n{'='*70}")
    print('Scene scoring with a pretrained model (UNLABELED)')
    print(f"{'='*70}")
    print(f"  granule : {args.l0b_file}")
    print(f"  model   : {args.model}")
    print(f"  pulses  : [{args.pulse_start}, {args.pulse_end})")
    rng_str = (f"[{args.range_start if args.range_start is not None else 0}, "
               f"{args.range_end if args.range_end is not None else 'full swath'})")
    print(f"  range   : {rng_str}")
    print(f"  features: top {N_KEEP} eigenvalues, gap_exclusion="
          f"{args.compute_subswath_mask} "
          f"({args.off_diag_overlap_ratio}/{args.diag_valid_ratio})")

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
    print("  channels: " + ', '.join(f'{f}-{p}' for f, p in channels))

    recs = [score_channel(raw, f, p, model, args) for f, p in channels]

    results = {
        'granule': os.path.basename(args.l0b_file),
        'model': args.model,
        'labeled': False,
        'pulse_window': recs[0]['pulse_window'],
        'range_window': recs[0]['range_window'],
        'n_keep': N_KEEP,
        'gap_exclusion_used': bool(args.compute_subswath_mask),
        'off_diag_overlap_ratio': args.off_diag_overlap_ratio,
        'diag_valid_ratio': args.diag_valid_ratio,
        'channels': {},
    }

    for rec in recs:
        save_predictions_h5(rec, args, args.output_dir)
        plot_knee_map(rec, args.output_dir)
        plot_confidence_map(rec, args.output_dir)
        plot_power_vs_knee(rec, args.output_dir)
        plot_confidence_by_knee(rec, args.output_dir)
        plot_eigen_profiles_by_class(rec, args.output_dir)
        plot_selected_predictions(rec, args.output_dir,
                                  args.n_examples_per_class, args.example_seed)

    plot_pred_hist(recs, args.output_dir)
    report(recs, results)

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved to {os.path.join(args.output_dir, 'results.json')}")


if __name__ == '__main__':
    main()