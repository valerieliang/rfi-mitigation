"""
replot_predictions.py

Reprint every plot that score_scene.py produces for a channel, using only the
predictions_<freq>_<pol>.h5 file it saved. No L0B granule, no model, and no
ISCE3 import is needed: everything a plot depends on (knee, confidence,
entropy, per-tile eigenvalues, baseline power, tile grid geometry) is already
sitting in that HDF5 file.

This intentionally duplicates the plotting logic in score_scene.py rather than
importing it, since importing score_scene.py would drag in the ISCE3 / Raw /
nisar_utils dependency chain that score_scene.py needs to read a granule but
this script does not.

Per-channel outputs (one predictions_<freq>_<pol>.h5 in -> five PNGs out):
    knee_map_<freq>_<pol>.png
    confidence_map_<freq>_<pol>.png
    power_vs_knee_<freq>_<pol>.png
    confidence_by_knee_<freq>_<pol>.png
    eigen_profiles_<freq>_<pol>.png
    selected_predictions_<freq>_<pol>.png

If more than one predictions_*.h5 file is given, two cross-channel outputs are
also produced:
    pred_hist.png
    (a console report identical in content to score_scene.py's report())

Usage
-----
    # Single channel
    python replot_predictions.py results/la_scene/predictions_A_HH.h5 \\
        --output-dir results/la_scene/replot

    # Every channel from one scene run (also gets pred_hist.png + report)
    python replot_predictions.py results/la_scene/predictions_*.h5 \\
        --output-dir results/la_scene/replot

    # Selected-predictions grid uses a random subsample of tiles per class;
    # match the original figure by reusing the same n/seed score_scene.py used
    python replot_predictions.py results/la_scene/predictions_A_HH.h5 \\
        --output-dir results/la_scene/replot \\
        --n-examples-per-class 6 --example-seed 42
"""

import os
import sys
import argparse

import numpy as np
import h5py


EPS = 1e-12


# ---------------------------------------------------------------------------
# LOADING
# ---------------------------------------------------------------------------

def load_predictions_h5(path):
    """
    Rebuild the same 'rec' dict that score_channel() produced in score_scene.py,
    reading it back out of a saved predictions_<freq>_<pol>.h5 file.

    Everything downstream (all plot_* functions, report()) consumes this dict
    exactly the way it did when it came straight out of scoring, so no plot
    code needs to know or care that the data was reloaded from disk.
    """
    with h5py.File(path, 'r') as f:
        knee = f['knee'][()]
        confidence = f['confidence'][()]
        entropy = f['entropy'][()]
        eigvals = f['eigenvalues'][()]
        power_db = f['signal_power_db'][()]
        valid_frac = f['valid_fraction'][()]
        tile_pulse = f['tile_pulse'][()]
        tile_range = f['tile_range'][()]

        freq = f.attrs['frequency']
        pol = f.attrs['polarization']
        pulse_start = int(f.attrs['pulse_start'])
        pulse_end = int(f.attrs['pulse_end'])
        range_start = int(f.attrs['range_start'])
        range_end = int(f.attrs['range_end'])
        n_pt = int(f.attrs['n_pulse_tiles'])
        n_rt = int(f.attrs['n_range_tiles'])
        cpi_len = int(f.attrs['cpi_len'])
        cpi_width = int(f.attrs['cpi_width'])
        n_keep = int(f.attrs['n_keep'])
        granule = f.attrs.get('granule', 'unknown')
        model = f.attrs.get('model', 'unknown')

    n_classes = int(knee.max()) + 1 if knee.size else 1

    return {
        'freq': freq, 'pol': pol, 'chan': f'{freq}-{pol}',
        'knee': knee, 'confidence': confidence, 'entropy': entropy,
        'eigvals': eigvals, 'power_db': power_db, 'valid_frac': valid_frac,
        'tile_pulse': tile_pulse, 'tile_range': tile_range,
        'n_pt': n_pt, 'n_rt': n_rt,
        'pulse_window': [pulse_start, pulse_end],
        'range_window': [range_start, range_end],
        'n_classes': n_classes,
        'cpi_len': cpi_len, 'cpi_width': cpi_width, 'n_keep': n_keep,
        'granule': granule, 'model': model,
        'source_path': path,
    }


# ---------------------------------------------------------------------------
# PLOTS (same logic and figures as score_scene.py, driven by the reloaded rec)
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
    """Predicted knee over the scene grid."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    grid = rec['knee'].reshape(rec['n_pt'], rec['n_rt'])

    cpi_width = rec['cpi_width']

    cpi_start = 0
    cpi_end = rec['n_pt']
    range_block_start = rec['range_window'][0] / cpi_width
    range_block_end = rec['range_window'][1] / cpi_width

    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(grid, aspect='auto', cmap='inferno', origin='upper',
                   vmin=0, vmax=max(rec['n_classes'] - 1, 1),
                   interpolation='nearest',
                   extent=[range_block_start, range_block_end,
                           cpi_end, cpi_start])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Number of RFI eigenvalues')

    ax.set_xlabel('Range blocks')
    ax.set_ylabel('CPI index')
    ax.set_title(f"Number of RFI eigenvalues across scene -- {rec['chan']} (UNMODIFIED data)\n"
                 f"no ground truth: k > 0 are candidate detections. "
                 f"Look for coherent range columns / slow-time blocks.")

    fig.tight_layout()
    path = os.path.join(out_dir, f"knee_map_{rec['freq']}_{rec['pol']}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_confidence_map(rec, out_dir):
    """Model confidence over the same grid as the knee map."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    grid = rec['confidence'].reshape(rec['n_pt'], rec['n_rt'])

    cpi_width = rec['cpi_width']

    cpi_start = 0
    cpi_end = rec['n_pt']
    range_block_start = rec['range_window'][0] / cpi_width
    range_block_end = rec['range_window'][1] / cpi_width

    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(grid, aspect='auto', cmap='viridis', origin='upper',
                   vmin=0, vmax=1, interpolation='nearest',
                   extent=[range_block_start, range_block_end,
                           cpi_end, cpi_start])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Max softmax probability')

    ax.set_xlabel('Range blocks')
    ax.set_ylabel('CPI index')
    ax.set_title(f"Model confidence across scene -- {rec['chan']}\n"
                 f"compare against the knee map: coherent AND confident is the "
                 f"strongest evidence")

    fig.tight_layout()
    path = os.path.join(out_dir, f"confidence_map_{rec['freq']}_{rec['pol']}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_power_vs_knee(rec, out_dir):
    """Tile baseline power, broken out by predicted class."""
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
            label = ('clean' if k == 0
                    else (f'{k} RFI\neigenvalue' if k == 1 else f'{k} RFI\neigenvalues'))
            labels.append(label)
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

    n = len(knee)
    idx = np.random.default_rng(0).choice(n, size=min(20000, n), replace=False)
    sc = ax2.scatter(knee[idx] + np.random.default_rng(1).uniform(-0.25, 0.25, len(idx)),
                     power[idx], c=rec['confidence'][idx], cmap='viridis',
                     s=3, alpha=0.4, vmin=0, vmax=1)
    fig.colorbar(sc, ax=ax2, label='Confidence')
    ax2.set_xticks(range(n_classes))
    ax2.set_xticklabels(['clean'] + [f'{k}' for k in range(1, n_classes)])
    ax2.set_xlabel('Number of RFI eigenvalues')
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
    """Confidence and posterior entropy per predicted class."""
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
            label = ('clean' if k == 0
                    else (f'{k} RFI\neigenvalue' if k == 1 else f'{k} RFI\neigenvalues'))
            labels.append(label)
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
    """Mean normalized eigenvalue profile per predicted class."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    n_classes = rec['n_classes']
    n_keep = rec['n_keep']
    knee = rec['knee']

    # Same normalization the model sees: linear / lambda_max, then dB
    ev = np.maximum(rec['eigvals'][:, :n_keep], EPS)
    ev_db = 10.0 * np.log10(ev / np.maximum(ev[:, :1], EPS))
    idx = np.arange(1, n_keep + 1)

    norm = mcolors.Normalize(vmin=0, vmax=max(n_classes - 1, 1))
    cmap = cm.plasma

    fig, ax = plt.subplots(figsize=(10, 6))
    for k in range(n_classes):
        sel = (knee == k)
        if sel.sum() < 10:
            continue
        mean_prof = ev_db[sel].mean(axis=0)
        label = ('clean' if k == 0
                 else (f'{k} RFI eigenvalue' if k == 1 else f'{k} RFI eigenvalues'))
        ax.plot(idx, mean_prof, color=cmap(norm(k)), linewidth=2,
                label=f"{label} (n={int(sel.sum())})")

        if k > 0:
            ax.axvline(x=k + 1, color=cmap(norm(k)), linestyle=':',
                      linewidth=2, alpha=0.7)

    ax.set_xlabel('Eigenvalue index (1-based, descending)')
    ax.set_ylabel('Eigenvalue (dB, normalized to lambda_max)')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=9)
    ax.set_title(f"Mean eigenvalue profile by predicted class -- {rec['chan']}\n"
                 f"dotted line marks the separation point between RFI and noise subspace")

    fig.tight_layout()
    path = os.path.join(out_dir, f"eigen_profiles_{rec['freq']}_{rec['pol']}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_selected_predictions(rec, out_dir, n_per_class, seed):
    """
    Grid of individual tiles: eigenvalue profile + what the model called it.

    Reproducible from a saved h5 as long as the same n_per_class and seed used
    at score_scene.py time are passed back in here; those two values are the
    only pieces of the original figure not stored in the predictions file.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    n_classes = rec['n_classes']
    n_keep = rec['n_keep']
    knee = rec['knee']
    rng = np.random.default_rng(seed)

    ev = np.maximum(rec['eigvals'][:, :n_keep], EPS)
    ev_db = 10.0 * np.log10(ev / np.maximum(ev[:, :1], EPS))
    idx = np.arange(1, n_keep + 1)

    present = [k for k in range(n_classes) if (knee == k).sum() > 0]
    if not present:
        return

    norm = mcolors.Normalize(vmin=0, vmax=max(n_classes - 1, 1))
    cmap = cm.plasma

    sample_all = ev_db[rng.choice(len(knee), size=min(5000, len(knee)), replace=False)]
    # Extend y-limits to ensure all 12 eigenvalues are visible
    ylim = [float(np.percentile(sample_all.ravel(), 0.1)) - 5.0, 3.0]

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
                ax.axvline(x=k + 1, color='red', linestyle=':', linewidth=1.5, alpha=0.7)

            label = ('CLEAN' if k == 0
                    else (f'{k} RFI eigenvalue' if k == 1 else f'{k} RFI eigenvalues'))
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
        f"dotted line marks the separation point between RFI and noise subspace",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(out_dir, f"selected_predictions_{rec['freq']}_{rec['pol']}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_pred_hist(recs, out_dir):
    """Prediction histogram across channels (needs 2+ predictions files)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_classes = max(rec['n_classes'] for rec in recs)
    x = np.arange(n_classes)
    width = 0.8 / len(recs)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, rec in enumerate(recs):
        counts = np.bincount(rec['knee'], minlength=n_classes)[:n_classes]
        frac = counts / max(counts.sum(), 1)
        ax.bar(x + i * width - 0.4 + width / 2, frac, width,
               label=f"{rec['chan']} (n={len(rec['knee'])})")

    ax.set_xticks(x)
    ax.set_xticklabels(['clean'] + [(f'{k} RFI\neigenvalue' if k == 1 else f'{k} RFI\neigenvalues')
                                      for k in range(1, n_classes)])
    ax.set_ylabel('Fraction of tiles')
    ax.set_xlabel('Predicted class')
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend()
    ax.set_title('Predictions on UNMODIFIED scene\n'
                 'no ground truth: k RFI eigenvalues are candidate detections, not errors')

    fig.tight_layout()
    path = os.path.join(out_dir, 'pred_hist.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# REPORTING (console summary only, same numbers as score_scene.py's report())
# ---------------------------------------------------------------------------

def report(recs):
    print(f"\n{'='*60}")
    print('SCENE SCORING  (reprinted from saved predictions -- no accuracy)')
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

        if n_flag and n_flag < n:
            corr = float(np.corrcoef(knee.astype(float), rec['power_db'])[0, 1])
        else:
            corr = float('nan')

        print(f"\n  {rec['chan']} (from {rec['source_path']}): {n} tiles")
        print(f"    flagged (k>0)    : {n_flag} ({100 * n_flag / n:.2f}%)")
        print(f"    distribution     : "
              + ", ".join(f"{'clean' if k == 0 else (f'{k} RFI eig' if k == 1 else f'{k} RFI eigs')}={v}"
                          for k, v in enumerate(dist) if v))
        print(f"    mean confidence  : {conf.mean():.3f}  "
              f"(flagged {conf[flagged].mean() if n_flag else float('nan'):.3f}, "
              f"clean {conf[~flagged].mean() if (~flagged).any() else float('nan'):.3f})")
        print(f"    RFI eigs vs power r : {corr:.3f}   "
              f"<- should be near 0; a large value means the model is keying on "
              f"brightness, not structure")
        print(f"    range columns flagged in >50% of the scene: "
              f"{len(hot_cols)} / {rec['n_rt']}"
              + (f"  -> {hot_cols[:12].tolist()}" if len(hot_cols) else ''))
        print(f"    busiest slow-time block: {100 * row_rate.max():.1f}% "
              f"of its range tiles flagged")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Reprint all score_scene.py plots for one or more channels '
                    'from their saved predictions_<freq>_<pol>.h5 files, with '
                    'no need to reprocess the original L0B granule.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('predictions_h5', nargs='+',
                        help='Path(s) to predictions_<freq>_<pol>.h5 file(s) '
                             'saved by score_scene.py.')
    parser.add_argument('--output-dir', default='results/replot',
                        help='Where to write the regenerated PNGs.')
    parser.add_argument('--n-examples-per-class', type=int, default=6,
                        help='Tiles per predicted class in the selected-predictions '
                             'grid. Use the same value score_scene.py was run with '
                             'to match the original figure exactly.')
    parser.add_argument('--example-seed', type=int, default=42,
                        help='Selection seed for the selected-predictions grid. '
                             'Use the same value score_scene.py was run with '
                             'to match the original figure exactly.')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print('Reprinting score_scene.py plots from saved predictions HDF5')
    print(f"{'='*70}")

    recs = []
    for path in args.predictions_h5:
        if not os.path.isfile(path):
            print(f"  Skipping (not found): {path}")
            continue
        rec = load_predictions_h5(path)
        print(f"  Loaded {path}  ({rec['chan']}, {len(rec['knee'])} tiles, "
              f"granule={rec['granule']}, model={rec['model']})")
        recs.append(rec)

    if not recs:
        print('No valid predictions files were loaded.')
        sys.exit(1)

    for rec in recs:
        plot_knee_map(rec, args.output_dir)
        plot_confidence_map(rec, args.output_dir)
        plot_power_vs_knee(rec, args.output_dir)
        plot_confidence_by_knee(rec, args.output_dir)
        plot_eigen_profiles_by_class(rec, args.output_dir)
        plot_selected_predictions(rec, args.output_dir,
                                  args.n_examples_per_class, args.example_seed)

    if len(recs) > 1:
        plot_pred_hist(recs, args.output_dir)

    report(recs)


if __name__ == '__main__':
    main()
