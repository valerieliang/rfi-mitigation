"""
diff_predictions.py

Compare the knee predictions two different models made on the SAME scene, tile
by tile. Both runs must come from the same granule and the same tiling, so the
two predictions_<freq>_<pol>.h5 files line up row for row; the script checks
that before differencing anything.

Sign convention (fixed by the argument order):

    diff = knee(A) - knee(B)

    diff > 0  ->  A predicted MORE RFI eigenvalues than B
    diff < 0  ->  B predicted MORE RFI eigenvalues than A
    diff == 0 ->  the two models agree on that tile

So with A = score/aus_sa/medhat and B = score/urClean_mtnContam_amzContam/medhat,
negative means urClean called a higher knee and positive means aus_sa did.

Per-channel outputs:
    knee_diff_map_<freq>_<pol>.png       signed difference over the scene grid
    knee_diff_hist_<freq>_<pol>.png      histogram of the signed difference
    knee_diff_confusion_<freq>_<pol>.png A-knee vs B-knee joint counts
    knee_diff_<freq>_<pol>.h5            per-tile diff + both knees (--save-h5)

Plus a console report per channel: agreement rate, mean/median signed diff,
where the disagreements sit, and how confident each model was when they split.

Usage
-----
    # One channel
    python diff_predictions.py \\
        score/aus_sa/medhat/predictions_A_HH.h5 \\
        score/urClean_mtnContam_amzContam/medhat/predictions_A_HH.h5 \\
        --output-dir results/knee_diff

    # Both channels at once: pass the two run directories instead of files and
    # every channel they share is matched up automatically
    python diff_predictions.py \\
        score/aus_sa/medhat \\
        score/urClean_mtnContam_amzContam/medhat \\
        --output-dir results/knee_diff --save-h5
"""

import os
import sys
import glob
import argparse

import numpy as np
import h5py


# ---------------------------------------------------------------------------
# LOADING
# ---------------------------------------------------------------------------

def load_predictions_h5(path):
    """
    Pull out everything the diff needs from one predictions_<freq>_<pol>.h5.

    This is deliberately a thinner read than replot_predictions.load_predictions_h5():
    the diff never touches the eigenvalue profiles, so there is no reason to pay
    for a (n_tiles, 16) float array on both sides of the comparison.
    """
    with h5py.File(path, 'r') as f:
        rec = {
            'knee': f['knee'][()].astype(np.int16),
            'confidence': f['confidence'][()],
            'tile_pulse': f['tile_pulse'][()],
            'tile_range': f['tile_range'][()],
            'freq': f.attrs['frequency'],
            'pol': f.attrs['polarization'],
            'n_pt': int(f.attrs['n_pulse_tiles']),
            'n_rt': int(f.attrs['n_range_tiles']),
            'cpi_width': int(f.attrs['cpi_width']),
            'range_window': [int(f.attrs['range_start']), int(f.attrs['range_end'])],
            'granule': f.attrs.get('granule', 'unknown'),
            'model': f.attrs.get('model', 'unknown'),
            'source_path': path,
        }
    rec['chan'] = f"{rec['freq']}-{rec['pol']}"
    return rec


def check_comparable(rec_a, rec_b, strict_granule=True):
    """
    Refuse to difference two runs that are not tile-for-tile aligned.

    A silent misalignment here would produce a diff map that looks perfectly
    plausible and is entirely meaningless, so every mismatch is fatal except
    the granule check, which can be relaxed with --allow-granule-mismatch.
    """
    problems = []

    if rec_a['chan'] != rec_b['chan']:
        problems.append(f"channel mismatch: {rec_a['chan']} vs {rec_b['chan']}")
    if rec_a['knee'].shape != rec_b['knee'].shape:
        problems.append(f"tile count mismatch: {rec_a['knee'].shape[0]} vs "
                        f"{rec_b['knee'].shape[0]}")
    if (rec_a['n_pt'], rec_a['n_rt']) != (rec_b['n_pt'], rec_b['n_rt']):
        problems.append(f"grid mismatch: {rec_a['n_pt']}x{rec_a['n_rt']} vs "
                        f"{rec_b['n_pt']}x{rec_b['n_rt']}")
    if rec_a['knee'].shape == rec_b['knee'].shape:
        if not np.array_equal(rec_a['tile_pulse'], rec_b['tile_pulse']) or \
           not np.array_equal(rec_a['tile_range'], rec_b['tile_range']):
            problems.append('tile coordinates differ: the two runs did not tile '
                            'the scene the same way')
    if rec_a['granule'] != rec_b['granule']:
        msg = f"granule mismatch:\n      A: {rec_a['granule']}\n      B: {rec_b['granule']}"
        if strict_granule:
            problems.append(msg)
        else:
            print(f"  WARNING: {msg}")

    return problems


def pair_inputs(path_a, path_b):
    """
    Accept either two predictions files or two run directories.

    Directory form matches channels by the predictions_<freq>_<pol>.h5 filename,
    so only channels present in BOTH runs are compared; anything unpaired is
    reported and skipped rather than quietly dropped.
    """
    if os.path.isfile(path_a) and os.path.isfile(path_b):
        return [(path_a, path_b)]

    if os.path.isdir(path_a) and os.path.isdir(path_b):
        a_files = {os.path.basename(p): p
                   for p in sorted(glob.glob(os.path.join(path_a, 'predictions_*.h5')))}
        b_files = {os.path.basename(p): p
                   for p in sorted(glob.glob(os.path.join(path_b, 'predictions_*.h5')))}
        shared = sorted(set(a_files) & set(b_files))
        for name in sorted(set(a_files) ^ set(b_files)):
            print(f"  Skipping unpaired channel file: {name}")
        return [(a_files[n], b_files[n]) for n in shared]

    print('Both arguments must be predictions .h5 files, or both must be '
          'directories containing them.')
    sys.exit(1)


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------

def plot_diff_map(diff, rec_a, rec_b, label_a, label_b, out_dir):
    """Signed knee difference over the scene grid, on a symmetric diverging scale."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    grid = diff.reshape(rec_a['n_pt'], rec_a['n_rt'])

    # Symmetric limits so that zero is always the neutral middle color; without
    # this, an asymmetric diff makes "agree" render as a nonzero-looking shade.
    lim = max(int(np.abs(diff).max()), 1)

    cpi_width = rec_a['cpi_width']
    range_block_start = rec_a['range_window'][0] / cpi_width
    range_block_end = rec_a['range_window'][1] / cpi_width

    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(grid, aspect='auto', cmap='RdBu_r', origin='upper',
                   vmin=-lim, vmax=lim, interpolation='nearest',
                   extent=[range_block_start, range_block_end,
                           rec_a['n_pt'], 0])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('knee(A) - knee(B)   [RFI eigenvalues]')

    ax.set_xlabel('Range blocks')
    ax.set_ylabel('CPI index')
    ax.set_title(f"Knee difference across scene -- {rec_a['chan']}\n"
                 f"red / positive = {label_a} predicted higher\n"
                 f"blue / negative = {label_b} predicted higher\n"
                 f"white = the two runs agree",
                 fontsize=10)

    fig.tight_layout()
    path = os.path.join(out_dir, f"knee_diff_map_{rec_a['freq']}_{rec_a['pol']}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_diff_hist(diff, rec_a, label_a, label_b, out_dir):
    """How the signed difference is distributed, one bar per integer offset."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lo, hi = int(diff.min()), int(diff.max())
    values = np.arange(lo, hi + 1)
    counts = np.array([(diff == v).sum() for v in values])
    frac = counts / max(counts.sum(), 1)

    colors = ['indianred' if v > 0 else ('steelblue' if v < 0 else '0.6')
              for v in values]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(values, frac, width=0.8, color=colors, alpha=0.85)
    for v, f_, c in zip(values, frac, counts):
        if f_ > 0:
            ax1.annotate(f'n={c}', (v, f_), textcoords='offset points',
                         xytext=(0, 3), ha='center', fontsize=7)
    ax1.set_xticks(values)
    ax1.set_xlabel('knee(A) - knee(B)')
    ax1.set_ylabel('Fraction of all tiles')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax1.set_title('All tiles\nagreement dominates, so the tails are invisible here')

    # Same data with the agreement bar dropped and renormalized: the shape of
    # the disagreement is the whole point, and it is unreadable next to a bar
    # holding ~97% of the tiles.
    nz = values != 0
    nz_frac = counts[nz] / max(counts[nz].sum(), 1)
    ax2.bar(values[nz], nz_frac, width=0.8,
            color=[c for c, keep in zip(colors, nz) if keep], alpha=0.85)
    for v, f_, c in zip(values[nz], nz_frac, counts[nz]):
        if f_ > 0:
            ax2.annotate(f'n={c}', (v, f_), textcoords='offset points',
                         xytext=(0, 3), ha='center', fontsize=7)
    ax2.set_xticks(values[nz])
    ax2.set_xlabel('knee(A) - knee(B)')
    ax2.set_ylabel('Fraction of DISAGREEING tiles')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax2.set_title(f'Disagreements only (n={int(counts[nz].sum())})\n'
                  f'renormalized, agreement bar removed')

    fig.suptitle(f"Knee difference distribution -- {rec_a['chan']}   "
                 f"positive = {label_a} higher,  negative = {label_b} higher",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(out_dir, f"knee_diff_hist_{rec_a['freq']}_{rec_a['pol']}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_diff_confusion(rec_a, rec_b, label_a, label_b, out_dir):
    """
    Joint counts of knee(A) vs knee(B).

    Neither run is ground truth, so this is not a confusion matrix in the usual
    sense -- it just shows WHICH classes the two models trade against each
    other, which the signed histogram alone cannot tell you.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    n_classes = int(max(rec_a['knee'].max(), rec_b['knee'].max())) + 1
    joint = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(joint, (rec_a['knee'], rec_b['knee']), 1)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(joint, cmap='viridis', origin='upper',
                   norm=mcolors.LogNorm(vmin=max(joint.min(), 1), vmax=max(joint.max(), 1)))
    fig.colorbar(im, ax=ax, label='Tile count (log scale)')

    for i in range(n_classes):
        for j in range(n_classes):
            if joint[i, j]:
                ax.annotate(f'{joint[i, j]}', (j, i), ha='center', va='center',
                            fontsize=6,
                            color='white' if joint[i, j] < joint.max() / 10 else 'black')

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xlabel(f'knee -- B: {label_b}')
    ax.set_ylabel(f'knee -- A: {label_a}')
    ax.set_title(f"Joint predictions -- {rec_a['chan']}\n"
                 f"diagonal = agreement; below diagonal = A higher, "
                 f"above = B higher")

    fig.tight_layout()
    path = os.path.join(out_dir, f"knee_diff_confusion_{rec_a['freq']}_{rec_a['pol']}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------

def save_diff_h5(diff, rec_a, rec_b, label_a, label_b, out_dir):
    """Per-tile diff alongside both source knees, for downstream analysis."""
    path = os.path.join(out_dir, f"knee_diff_{rec_a['freq']}_{rec_a['pol']}.h5")
    with h5py.File(path, 'w') as f:
        f.create_dataset('knee_diff', data=diff.astype(np.int16), compression='gzip')
        f.create_dataset('knee_a', data=rec_a['knee'].astype(np.int8), compression='gzip')
        f.create_dataset('knee_b', data=rec_b['knee'].astype(np.int8), compression='gzip')
        f.create_dataset('confidence_a', data=rec_a['confidence'], compression='gzip')
        f.create_dataset('confidence_b', data=rec_b['confidence'], compression='gzip')
        f.create_dataset('tile_pulse', data=rec_a['tile_pulse'], compression='gzip')
        f.create_dataset('tile_range', data=rec_a['tile_range'], compression='gzip')

        f.attrs['convention'] = 'knee_diff = knee_a - knee_b; >0 means A predicted higher'
        f.attrs['label_a'] = label_a
        f.attrs['label_b'] = label_b
        f.attrs['source_a'] = rec_a['source_path']
        f.attrs['source_b'] = rec_b['source_path']
        f.attrs['model_a'] = rec_a['model']
        f.attrs['model_b'] = rec_b['model']
        f.attrs['frequency'] = rec_a['freq']
        f.attrs['polarization'] = rec_a['pol']
        f.attrs['granule'] = rec_a['granule']
        f.attrs['n_pulse_tiles'] = rec_a['n_pt']
        f.attrs['n_range_tiles'] = rec_a['n_rt']
        f.attrs['cpi_width'] = rec_a['cpi_width']
        f.attrs['range_start'] = rec_a['range_window'][0]
        f.attrs['range_end'] = rec_a['range_window'][1]
    print(f"  Saved {path}")


def report(diff, rec_a, rec_b, label_a, label_b):
    n = diff.size
    same = diff == 0
    a_higher = diff > 0
    b_higher = diff < 0

    print(f"\n  {rec_a['chan']}: {n} tiles")
    print(f"    A = {label_a}   ({rec_a['source_path']})")
    print(f"    B = {label_b}   ({rec_b['source_path']})")
    print(f"    agree (diff = 0) : {int(same.sum())} ({100 * same.mean():.2f}%)")
    print(f"    A higher (> 0)   : {int(a_higher.sum())} ({100 * a_higher.mean():.2f}%)"
          + (f"   mean +{diff[a_higher].mean():.2f}, max +{int(diff.max())}"
             if a_higher.any() else ''))
    print(f"    B higher (< 0)   : {int(b_higher.sum())} ({100 * b_higher.mean():.2f}%)"
          + (f"   mean {diff[b_higher].mean():.2f}, min {int(diff.min())}"
             if b_higher.any() else ''))
    print(f"    signed diff      : mean {diff.mean():+.3f}, median "
          f"{np.median(diff):+.1f}   <- net bias of A relative to B")
    print(f"    absolute diff    : mean {np.abs(diff).mean():.3f}, "
          f"max {int(np.abs(diff).max())}")

    flag_a = int((rec_a['knee'] > 0).sum())
    flag_b = int((rec_b['knee'] > 0).sum())
    print(f"    flagged (k > 0)  : A {flag_a} ({100 * flag_a / n:.2f}%)  vs  "
          f"B {flag_b} ({100 * flag_b / n:.2f}%)")

    # Where a model was confident and still disagreed is the interesting case:
    # a low-confidence split is just the model hedging near a class boundary.
    disagree = ~same
    if disagree.any():
        ca = rec_a['confidence'][disagree]
        cb = rec_b['confidence'][disagree]
        print(f"    on disagreements : mean confidence A {ca.mean():.3f}, "
              f"B {cb.mean():.3f}")
        both_conf = int(((ca > 0.9) & (cb > 0.9)).sum())
        print(f"    both > 0.9 conf and still disagree: {both_conf} "
              f"({100 * both_conf / n:.2f}% of all tiles)")

    # Coherent structure in the disagreement usually means one model is keying
    # on a whole range column or slow-time block the other one is not.
    grid = disagree.reshape(rec_a['n_pt'], rec_a['n_rt'])
    col_rate = grid.mean(axis=0)
    row_rate = grid.mean(axis=1)
    hot_cols = np.where(col_rate > 0.5)[0]
    print(f"    range columns where they disagree on >50% of tiles: "
          f"{len(hot_cols)} / {rec_a['n_rt']}"
          + (f"  -> {hot_cols[:12].tolist()}" if len(hot_cols) else ''))
    print(f"    busiest slow-time block: {100 * row_rate.max():.1f}% of its "
          f"range tiles disagree")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Difference the per-tile knee predictions of two scoring runs '
                    'on the same scene. Positive = the FIRST argument predicted a '
                    'higher knee, negative = the SECOND did.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('run_a',
                        help='Run A: a predictions_<freq>_<pol>.h5 file, or a '
                             'directory containing them. Positive diffs mean A '
                             'predicted more RFI eigenvalues.')
    parser.add_argument('run_b',
                        help='Run B: matching predictions file or directory. '
                             'Negative diffs mean B predicted more.')
    parser.add_argument('--label-a', default=None,
                        help='Name for run A in titles and reports. Defaults to '
                             'the run directory name.')
    parser.add_argument('--label-b', default=None,
                        help='Name for run B in titles and reports. Defaults to '
                             'the run directory name.')
    parser.add_argument('--output-dir', default='results/knee_diff',
                        help='Where to write the PNGs (and the h5, if requested).')
    parser.add_argument('--save-h5', action='store_true',
                        help='Also write knee_diff_<freq>_<pol>.h5 with the '
                             'per-tile difference and both source knees.')
    parser.add_argument('--allow-granule-mismatch', action='store_true',
                        help='Warn instead of aborting when the two runs record '
                             'different source granules.')
    parser.add_argument('--no-plots', action='store_true',
                        help='Console report only.')
    return parser.parse_args()


def default_label(path):
    """
    Name a run after the directory that identifies it.

    score/<dataset>/<model>/predictions_A_HH.h5 -> '<dataset>/<model>', which is
    what actually distinguishes two runs here; the leaf directory alone is often
    the same name ('medhat') on both sides.
    """
    d = path if os.path.isdir(path) else os.path.dirname(path)
    d = os.path.normpath(d)
    parts = [p for p in d.split(os.sep) if p not in ('.', '')]
    return '/'.join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else d)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    label_a = args.label_a or default_label(args.run_a)
    label_b = args.label_b or default_label(args.run_b)

    print(f"\n{'='*70}")
    print('KNEE PREDICTION DIFF')
    print(f"{'='*70}")
    print(f"  A (positive diffs): {label_a}")
    print(f"  B (negative diffs): {label_b}")

    pairs = pair_inputs(args.run_a, args.run_b)
    if not pairs:
        print('No channels in common between the two runs.')
        sys.exit(1)

    n_done = 0
    for path_a, path_b in pairs:
        rec_a = load_predictions_h5(path_a)
        rec_b = load_predictions_h5(path_b)

        problems = check_comparable(rec_a, rec_b,
                                    strict_granule=not args.allow_granule_mismatch)
        if problems:
            print(f"\n  Skipping {os.path.basename(path_a)} -- not comparable:")
            for p in problems:
                print(f"    - {p}")
            continue

        diff = rec_a['knee'].astype(np.int16) - rec_b['knee'].astype(np.int16)

        report(diff, rec_a, rec_b, label_a, label_b)

        if not args.no_plots:
            plot_diff_map(diff, rec_a, rec_b, label_a, label_b, args.output_dir)
            plot_diff_hist(diff, rec_a, label_a, label_b, args.output_dir)
            plot_diff_confusion(rec_a, rec_b, label_a, label_b, args.output_dir)
        if args.save_h5:
            save_diff_h5(diff, rec_a, rec_b, label_a, label_b, args.output_dir)

        n_done += 1

    if n_done == 0:
        print('\nNo channel pairs could be compared.')
        sys.exit(1)


if __name__ == '__main__':
    main()
