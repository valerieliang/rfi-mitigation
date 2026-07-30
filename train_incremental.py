"""
train_incremental.py

Continue training an existing knee-classifier checkpoint on data it has NOT seen,
without letting the already-fit old data drown out the new data.

Why this is a separate script from train_only.py
------------------------------------------------
Naively concatenating old and new regions and calling fit() again fails quietly.
In the low_jsr_dist_chile_incremental run, 59% of the training pool was
amazon_contam -- data the base model already fit -- so the average 256-tile batch
carried ~151 already-fit tiles against ~81 new low-JSR tiles, and the gradient
mostly said "keep doing what you already do". Aggregate val_loss had the same
problem: 69% of the val set was converged old data, so val_loss was dominated by
a term that could not improve, which is what drove ReduceLROnPlateau and early
stopping while the new-domain metrics were never separately visible.

This script addresses that with four things:

  1. --old-data-dir / --new-data-dir are declared explicitly and cross-checked
     against the base model's training_summary.json, so a directory the model has
     never seen cannot be mislabelled as old.
  2. Per-sample loss weights rebalance old vs new to --new-weight-frac. No tiles
     are discarded, so the full variety of the old domain still anchors the model
     against forgetting.
  3. Validation is reported PER SOURCE every epoch, against a baseline measured
     from the loaded model before training starts, so "did the new domain
     actually improve" is answerable from the run itself.
  4. Checkpointing and early stopping track the NEW-source val loss, guarded by a
     forgetting check on the old sources.

The data path (feature extraction, file globbing, block split) is imported from
train_only.py rather than duplicated, so the two cannot drift apart.

Usage:
    python train_incremental.py \
        --model models/urClean_mtnContam_amzContam/best_model.keras \
        --old-data-dir data/czech_contam/ data/amazon_contam/ data/berlin_clean/ \
        --new-data-dir data/amz_low_jsr/ data/chile_mountain_clean/ \
        --run-name low_jsr_incremental_v2 \
        --epochs 100

Outputs:
    models/<run>/best_model.keras
    models/<run>/training_curves.png
    models/<run>/per_source_curves.png
    models/<run>/incremental_summary.json
"""

import os
import json
import argparse

import numpy as np
import tensorflow as tf

from train_only import (
    MODELS_ROOT,
    N_KEEP,
    N_GLOBAL,
    VAL_FRAC,
    SPLIT_BUFFER_DEFAULT,
    EPS,
    load_rfi_data_dir,
    split_train_val,
    report_class_by_source_split,
    save_training_curves_png,
)


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

EPOCHS = 100
BATCH_SIZE = 256
LR = 3e-4

# Target share of total training loss weight assigned to the new sources.
NEW_WEIGHT_FRAC = 0.5

# Old-source val accuracy is allowed to sit this far below the baseline before a
# checkpoint is refused.
MAX_FORGET = 0.01

PATIENCE = 20
LR_PATIENCE = 10


# ---------------------------------------------------------------------------
# PROVENANCE
# ---------------------------------------------------------------------------

def _dir_name(d):
    """Basename of a data dir, tolerant of trailing slashes."""
    return os.path.basename(os.path.normpath(d))


def check_provenance(model_path, old_dirs, new_dirs, strict=True):
    """
    Reconcile the declared old/new split against what the base model was really
    trained on, read from <model_dir>/training_summary.json.

    Catches the failure mode where a directory the checkpoint has never seen is
    passed as old data -- it then gets down-weighted as "already learned" and is
    excluded from the improvement metric, which is exactly backwards.

    Returns a dict for the run summary. Raises SystemExit on a mismatch when
    strict, after printing what is wrong.
    """
    summary_path = os.path.join(os.path.dirname(model_path), 'training_summary.json')

    result = {'summary_path': summary_path, 'checked': False,
              'base_sources': None, 'misdeclared_old': [], 'misdeclared_new': []}

    print(f"\n{'='*70}")
    print('Provenance check')
    print(f"{'='*70}")

    if not os.path.isfile(summary_path):
        print(f"  WARNING: no training_summary.json next to the checkpoint")
        print(f"           ({summary_path})")
        print(f"           Cannot verify the old/new split -- trusting the flags as given.")
        return result

    with open(summary_path) as fh:
        base = json.load(fh)

    base_sources = base.get('train_provenance', {}).get('source_names')
    if not base_sources:
        print(f"  WARNING: {summary_path} has no train_provenance.source_names")
        print(f"           Cannot verify the old/new split -- trusting the flags as given.")
        return result

    base_set = set(base_sources)
    old_names = [_dir_name(d) for d in old_dirs]
    new_names = [_dir_name(d) for d in new_dirs]

    # Declared old but the base model never saw it -> it is really new data.
    misdeclared_old = [n for n in old_names if n not in base_set]
    # Declared new but the base model already trained on it -> not new.
    misdeclared_new = [n for n in new_names if n in base_set]
    # Seen during base training but not declared at all: fine, just report it.
    dropped = [n for n in base_sources if n not in set(old_names) | set(new_names)]

    result.update({'checked': True, 'base_sources': base_sources,
                   'misdeclared_old': misdeclared_old,
                   'misdeclared_new': misdeclared_new,
                   'dropped_from_base': dropped})

    print(f"  base model : {model_path}")
    print(f"  trained on : {', '.join(base_sources)}")
    print()
    print(f"    {'source':<26}  {'declared':>10}  {'in base model':>14}")
    for n in old_names:
        print(f"    {n:<26}  {'old':>10}  {('yes' if n in base_set else 'NO'):>14}")
    for n in new_names:
        print(f"    {n:<26}  {'new':>10}  {('YES' if n in base_set else 'no'):>14}")

    if dropped:
        print(f"\n  Note: base model also saw {', '.join(dropped)}, not passed to this run.")
        print(f"        Those regions get no replay here and may drift.")

    if misdeclared_old or misdeclared_new:
        print(f"\n  MISMATCH:")
        for n in misdeclared_old:
            print(f"    '{n}' passed as --old-data-dir, but the base model never "
                  f"trained on it.\n      -> it is NEW data; move it to --new-data-dir.")
        for n in misdeclared_new:
            print(f"    '{n}' passed as --new-data-dir, but the base model already "
                  f"trained on it.\n      -> it is OLD data; move it to --old-data-dir.")
        if strict:
            print(f"\n  Aborting. Re-run with corrected flags, or pass "
                  f"--no-provenance-check to override.")
            raise SystemExit(2)
        print(f"\n  --no-provenance-check set: continuing anyway.")
    else:
        print(f"\n  OK: declared split matches the base model's provenance.")

    return result


# ---------------------------------------------------------------------------
# LOSS REBALANCING
# ---------------------------------------------------------------------------

def compute_sample_weights(is_new_train, new_weight_frac):
    """
    Per-sample loss weights giving the new sources `new_weight_frac` of the total
    training loss weight, with the old sources taking the remainder.

    Solves  n_new*w_new / (n_new*w_new + n_old*w_old) = new_weight_frac,
    then scales so the mean weight is 1.0 -- that keeps the reported loss on the
    same scale as an unweighted run, so it stays comparable to previous training
    curves.

    All tiles are kept. Only their gradient contribution is rescaled, so the full
    variety of the old domain still anchors the model against forgetting.

    Returns (weights, info_dict).
    """
    is_new = np.asarray(is_new_train, dtype=bool)
    n_new = int(np.count_nonzero(is_new))
    n_old = int(len(is_new) - n_new)
    n_tot = n_new + n_old

    if n_new == 0:
        raise ValueError('No new-source tiles in the training split -- nothing to learn.')
    if n_old == 0:
        print('\n  No old-source tiles; skipping rebalancing (weights all 1.0).')
        return np.ones(n_tot, dtype=np.float32), {
            'new_weight_frac_requested': new_weight_frac,
            'new_weight_frac_actual': 1.0,
            'w_new': 1.0, 'w_old': 0.0, 'n_new': n_new, 'n_old': n_old,
        }

    # Total weight mass each side should carry, out of n_tot.
    mass_new = new_weight_frac * n_tot
    mass_old = (1.0 - new_weight_frac) * n_tot
    w_new = mass_new / max(n_new, 1)
    w_old = mass_old / max(n_old, 1)

    weights = np.where(is_new, w_new, w_old).astype(np.float32)

    # Mean is 1.0 by construction; renormalize against float drift.
    weights /= max(float(weights.mean()), EPS)

    frac_new_actual = float(n_new * w_new / max(n_new * w_new + n_old * w_old, EPS))

    print(f"\n{'='*70}")
    print(f"Loss rebalancing  (target new share = {new_weight_frac:.2f})")
    print(f"{'='*70}")
    print(f"    {'':<12}  {'tiles':>10}  {'tile share':>11}  {'weight':>8}  {'loss share':>11}")
    print(f"    {'new':<12}  {n_new:>10}  {n_new/n_tot:>10.1%}  "
          f"{w_new:>8.3f}  {frac_new_actual:>10.1%}")
    print(f"    {'old':<12}  {n_old:>10}  {n_old/n_tot:>10.1%}  "
          f"{w_old:>8.3f}  {1-frac_new_actual:>10.1%}")
    print(f"\n  All {n_tot} tiles retained; only gradient contribution is rescaled.")

    info = {
        'new_weight_frac_requested': float(new_weight_frac),
        'new_weight_frac_actual': frac_new_actual,
        'w_new': float(w_new), 'w_old': float(w_old),
        'n_new': n_new, 'n_old': n_old,
    }
    return weights, info


# ---------------------------------------------------------------------------
# CALLBACKS
# ---------------------------------------------------------------------------

class PerSourceVal(tf.keras.callbacks.Callback):
    """
    Per-source validation metrics, injected into `logs` so later callbacks can
    monitor them.

    Runs ONE predict pass over the whole val split per epoch and aggregates it
    with numpy, rather than calling evaluate() once per source. Also runs once in
    on_train_begin to capture the loaded model's baseline -- without that there is
    no reference for whether the new domain improved.

    Adds to logs:
        val_<source>_loss / val_<source>_acc   per declared source
        val_new_loss / val_new_acc             pooled over new sources
        val_old_loss / val_old_acc             pooled over old sources

    Metrics are UNWEIGHTED, so they stay comparable to non-incremental runs.
    """

    def __init__(self, eigen_val, global_val, y_val, sources_val,
                 source_names, n_old_sources, batch_size=1024):
        super().__init__()
        self.x = [eigen_val, global_val]
        self.y = np.asarray(y_val).astype(np.int64)
        self.sources = np.asarray(sources_val)
        self.source_names = list(source_names)
        self.n_old = n_old_sources
        self.batch_size = batch_size

        self.is_new = self.sources >= n_old_sources
        self.baseline = None
        self.history = []

        # Precompute per-source masks once.
        self._masks = [(name, self.sources == sid)
                       for sid, name in enumerate(self.source_names)]

    def _evaluate(self):
        probs = self.model.predict(self.x, batch_size=self.batch_size, verbose=0)
        p_true = np.clip(probs[np.arange(len(self.y)), self.y], EPS, 1.0)
        nll = -np.log(p_true)
        correct = (probs.argmax(axis=1) == self.y)

        out = {}
        for name, mask in self._masks:
            if not mask.any():
                continue
            out[f'val_{name}_loss'] = float(nll[mask].mean())
            out[f'val_{name}_acc'] = float(correct[mask].mean())

        for tag, mask in (('new', self.is_new), ('old', ~self.is_new)):
            if mask.any():
                out[f'val_{tag}_loss'] = float(nll[mask].mean())
                out[f'val_{tag}_acc'] = float(correct[mask].mean())
        return out

    def _print(self, metrics, title):
        print(f"\n  {title}")
        print(f"    {'source':<26}  {'loss':>8}  {'acc':>8}   {'vs baseline':>12}")
        rows = [(n, n) for n, m in self._masks if m.any()]
        rows += [('new (pooled)', 'new'), ('old (pooled)', 'old')]
        for label, key in rows:
            lk, ak = f'val_{key}_loss', f'val_{key}_acc'
            if lk not in metrics:
                continue
            delta = ''
            if self.baseline is not None and ak in self.baseline:
                d = metrics[ak] - self.baseline[ak]
                delta = f'{d:+.4f} acc'
            sep = '  --' if label.endswith('(pooled)') and key == 'new' else ''
            print(f"    {label:<26}  {metrics[lk]:>8.4f}  {metrics[ak]:>8.4f}   "
                  f"{delta:>12}{sep}")

    def on_train_begin(self, logs=None):
        self.baseline = self._evaluate()
        self._print(self.baseline, 'BASELINE (loaded model, before any training)')

    def on_epoch_end(self, epoch, logs=None):
        metrics = self._evaluate()
        if logs is not None:
            logs.update(metrics)
        self.history.append(metrics)
        self._print(metrics, f'Per-source validation (epoch {epoch + 1})')


class GuardedCheckpoint(tf.keras.callbacks.Callback):
    """
    Save on new-source val loss improvement, but refuse if the old sources have
    regressed more than `max_forget` in accuracy below their baseline.

    A plain ModelCheckpoint on val_new_loss would happily save a model that has
    learned low-JSR by wrecking the regions it already handled. The guard makes
    the tradeoff explicit and visible in the log rather than silent.
    """

    def __init__(self, filepath, per_source_cb, max_forget,
                 monitor='val_new_loss', guard_metric='val_old_acc'):
        super().__init__()
        self.filepath = filepath
        self.per_source = per_source_cb
        self.max_forget = max_forget
        self.monitor = monitor
        self.guard_metric = guard_metric
        self.best = np.inf
        self.best_epoch = None
        self.n_blocked = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        if current >= self.best:
            print(f"\n  Epoch {epoch + 1}: {self.monitor} did not improve "
                  f"from {self.best:.5f}")
            return

        baseline_guard = (self.per_source.baseline or {}).get(self.guard_metric)
        guard_now = logs.get(self.guard_metric)

        if baseline_guard is not None and guard_now is not None:
            floor = baseline_guard - self.max_forget
            if guard_now < floor:
                self.n_blocked += 1
                print(f"\n  Epoch {epoch + 1}: {self.monitor} improved "
                      f"{self.best:.5f} -> {current:.5f}, but NOT saved: "
                      f"{self.guard_metric} {guard_now:.4f} is below the "
                      f"forgetting floor {floor:.4f} "
                      f"(baseline {baseline_guard:.4f} - {self.max_forget:.4f}).")
                return

        print(f"\n  Epoch {epoch + 1}: {self.monitor} improved "
              f"{self.best:.5f} -> {current:.5f}, saving to {self.filepath}")
        self.best = current
        self.best_epoch = epoch + 1
        self.model.save(self.filepath)


# ---------------------------------------------------------------------------
# MODEL PREP
# ---------------------------------------------------------------------------

def load_and_prepare(model_path, learning_rate, weight_decay, dropout_rate=None):
    """
    Load the checkpoint and rebuild the optimizer from the current
    --learning-rate/--weight-decay, so the LR schedule and decay restart cleanly
    rather than resuming whatever state was saved with the model.

    dropout_rate, when given, overrides the LAST Dropout layer -- the fusion head
    dropout from model.py's build_model. The global branch's fixed 0.3 is left
    alone. This has to be done by hand: passing --dropout-rate to a LOADED model
    is otherwise a silent no-op, since the rate is baked in at build time.
    """
    print(f"\nLoading base model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    if dropout_rate is not None:
        dropouts = [l for l in model.layers if isinstance(l, tf.keras.layers.Dropout)]
        if not dropouts:
            print(f"  WARNING: --dropout-rate given but the model has no Dropout layers.")
        else:
            head = dropouts[-1]
            was = head.rate
            try:
                head.rate = dropout_rate
                stuck = (float(head.rate) == float(dropout_rate))
            except AttributeError:
                stuck = False
            if stuck:
                print(f"  Head dropout '{head.name}': {was} -> {head.rate}")
            else:
                # Better to say so than to repeat the silent no-op this script
                # exists to fix.
                print(f"  WARNING: could not set '{head.name}'.rate on this Keras "
                      f"version; it is still {was}. Rebuild from scratch with "
                      f"train_only.py if the dropout change matters.")

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top2_acc'),
        ],
    )
    return model


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def save_per_source_curves_png(per_source_cb, out_dir):
    """
    New vs old validation loss and accuracy per epoch, with dashed baseline
    reference lines from the loaded model. This is the plot that answers whether
    the new domain improved and whether the old one held.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    hist = per_source_cb.history
    if not hist:
        return None
    baseline = per_source_cb.baseline or {}
    epochs = range(1, len(hist) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    for tag, color in (('new', 'tab:blue'), ('old', 'tab:orange')):
        lk, ak = f'val_{tag}_loss', f'val_{tag}_acc'
        if lk not in hist[0]:
            continue
        ax_loss.plot(epochs, [h[lk] for h in hist], color=color, label=f'{tag} val loss')
        ax_acc.plot(epochs, [h[ak] for h in hist], color=color, label=f'{tag} val acc')
        if lk in baseline:
            ax_loss.axhline(baseline[lk], color=color, ls='--', alpha=0.6,
                            label=f'{tag} baseline')
            ax_acc.axhline(baseline[ak], color=color, ls='--', alpha=0.6,
                           label=f'{tag} baseline')

    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Validation loss: new vs old sources')
    ax_loss.legend()
    ax_loss.grid(True, linestyle='--', alpha=0.5)

    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_title('Validation accuracy: new vs old sources')
    ax_acc.legend()
    ax_acc.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    path = os.path.join(out_dir, 'per_source_curves.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def print_final_table(per_source_cb, final_metrics):
    """Baseline -> final per-source comparison, the headline result of the run."""
    baseline = per_source_cb.baseline or {}

    print(f"\n{'='*70}")
    print('Baseline -> final  (per source, on the held-out val split)')
    print(f"{'='*70}")
    print(f"    {'source':<26}  {'base acc':>9}  {'final acc':>9}  {'delta':>8}  "
          f"{'base loss':>9}  {'final loss':>10}")

    rows = [(n, n) for n, m in per_source_cb._masks if m.any()]
    rows += [('new (pooled)', 'new'), ('old (pooled)', 'old')]

    for label, key in rows:
        ak, lk = f'val_{key}_acc', f'val_{key}_loss'
        if ak not in final_metrics or ak not in baseline:
            continue
        d = final_metrics[ak] - baseline[ak]
        if label.startswith('new (') or label.startswith('old ('):
            print(f"    {'-'*26}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*9}  {'-'*10}")
        print(f"    {label:<26}  {baseline[ak]:>9.4f}  {final_metrics[ak]:>9.4f}  "
              f"{d:>+8.4f}  {baseline[lk]:>9.4f}  {final_metrics[lk]:>10.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Continue training an existing knee-classifier checkpoint on '
                    'unseen regions, rebalancing the loss so the new data is not '
                    'drowned out by the already-fit old data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Base .keras checkpoint to continue training from.')
    parser.add_argument('--old-data-dir', type=str, nargs='+', required=True,
                        help='Directories the base model was ALREADY trained on. '
                             'Kept in training as replay against forgetting, but '
                             'down-weighted.')
    parser.add_argument('--new-data-dir', type=str, nargs='+', required=True,
                        help='Directories the base model has NOT seen. These drive '
                             'the loss, the checkpoint metric and early stopping.')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Model output folder. Defaults to the first new data '
                             'dir name plus "_incremental".')
    parser.add_argument('--models-root', type=str, default=MODELS_ROOT,
                        help='Parent directory for the run folder.')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Cap tiles loaded per file (evenly strided), for quick runs.')
    parser.add_argument('--val-frac', type=float, default=VAL_FRAC,
                        help='Fraction of pulse tiles held out for val, per source.')
    parser.add_argument('--split-buffer', type=int, default=SPLIT_BUFFER_DEFAULT,
                        help='Pulse tiles dropped at each train/val boundary.')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--learning-rate', type=float, default=LR,
                        help='Initial learning rate. The optimizer is rebuilt, so '
                             'this restarts the schedule rather than resuming it.')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='L2 regularization strength (AdamW weight decay).')
    parser.add_argument('--dropout-rate', type=float, default=None,
                        help='Override the fusion-head dropout of the loaded model. '
                             'Default: inherit whatever the checkpoint was built with.')
    parser.add_argument('--new-weight-frac', type=float, default=NEW_WEIGHT_FRAC,
                        help='Share of total training loss weight given to the new '
                             'sources. 0.5 = new and old contribute equally.')
    parser.add_argument('--max-forget', type=float, default=MAX_FORGET,
                        help='Max allowed drop in pooled old-source val accuracy '
                             'below baseline before a checkpoint is refused.')
    parser.add_argument('--patience', type=int, default=PATIENCE,
                        help='Early-stopping patience on val_new_loss.')
    parser.add_argument('--lr-patience', type=int, default=LR_PATIENCE,
                        help='ReduceLROnPlateau patience on val_new_loss.')
    parser.add_argument('--no-provenance-check', action='store_true',
                        help='Skip aborting when the declared old/new split '
                             'contradicts the base model training_summary.json.')
    return parser.parse_args()


def main():
    args = parse_args()

    old_dirs = list(args.old_data_dir)
    new_dirs = list(args.new_data_dir)
    n_old_sources = len(old_dirs)

    overlap = {_dir_name(d) for d in old_dirs} & {_dir_name(d) for d in new_dirs}
    if overlap:
        raise SystemExit(f"Directory listed as both old and new: {', '.join(sorted(overlap))}")

    run_name = args.run_name or f'{_dir_name(new_dirs[0])}_incremental'
    os.makedirs(args.models_root, exist_ok=True)
    out_dir = os.path.join(args.models_root, run_name)
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'best_model.keras')

    print(f"\n{'='*70}")
    print('RFI knee classifier - INCREMENTAL TRAINING')
    print(f"{'='*70}")
    print(f"  base model        : {args.model}")
    print(f"  old region(s)     : {', '.join(old_dirs)}")
    print(f"  new region(s)     : {', '.join(new_dirs)}")
    print(f"  features          : top {N_KEEP} eigenvalues, linear-normalized then dB")
    print(f"  run name          : {run_name}")
    print(f"  epochs            : {args.epochs}")
    print(f"  batch size        : {args.batch_size}")
    print(f"  learning rate     : {args.learning_rate}")
    print(f"  weight decay      : {args.weight_decay}")
    print(f"  dropout rate      : "
          f"{'inherit from checkpoint' if args.dropout_rate is None else args.dropout_rate}")
    print(f"  new loss share    : {args.new_weight_frac}")
    print(f"  max forgetting    : {args.max_forget} (old-source val acc below baseline)")

    provenance = check_provenance(args.model, old_dirs, new_dirs,
                                  strict=not args.no_provenance_check)

    # Old dirs FIRST so source ids 0..n_old-1 are old and the rest are new.
    data = load_rfi_data_dir(old_dirs + new_dirs, args.max_samples,
                             tag='OLD + NEW regions')
    n_classes = data['n_classes']
    source_names = data['meta']['source_names']

    print(f"\nCombined pool: {len(data['labels'])} tiles   classes: {n_classes}   "
          f"channels: {', '.join(data['meta']['channels'])}")

    idx_train, idx_val = split_train_val(
        data['groups'], args.val_frac, args.split_buffer, 'per-source'
    )

    class_distribution = report_class_by_source_split(
        data['labels'], data['sources'], source_names,
        idx_train, idx_val, n_classes,
    )

    is_new_train = data['sources'][idx_train] >= n_old_sources
    sample_weights, weight_info = compute_sample_weights(
        is_new_train, args.new_weight_frac)

    model = load_and_prepare(args.model, args.learning_rate,
                             args.weight_decay, args.dropout_rate)

    per_source = PerSourceVal(
        data['eigen'][idx_val], data['global'][idx_val], data['labels'][idx_val],
        data['sources'][idx_val], source_names, n_old_sources,
    )
    checkpoint = GuardedCheckpoint(model_path, per_source, args.max_forget)

    # PerSourceVal must come first: it injects val_new_loss / val_old_acc into
    # logs, which the two callbacks after it monitor.
    #
    # restore_best_weights is False on purpose. EarlyStopping would otherwise
    # restore ITS best epoch by val_new_loss, which can differ from the epoch
    # GuardedCheckpoint actually saved once the forgetting guard blocks one. The
    # saved file is reloaded from disk below instead, so the returned model and
    # best_model.keras are guaranteed to be the same weights.
    callbacks = [
        per_source,
        checkpoint,
        # mode='min' is required, not optional: Keras 3 infers the direction from
        # a table of known metric names, and 'val_new_loss' is a custom key
        # injected by PerSourceVal, so inference fails with a ValueError.
        tf.keras.callbacks.EarlyStopping(
            monitor='val_new_loss', mode='min', patience=args.patience,
            restore_best_weights=False, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_new_loss', mode='min', factor=0.5,
            patience=args.lr_patience, min_lr=1e-6, verbose=1,
        ),
    ]

    print(f"\n{'='*60}")
    print(f"Run : {run_name}")
    print(f"  train={len(idx_train)}  val={len(idx_val)}")
    print(f"  eigen input : ({N_KEEP}, 2)   global: ({N_GLOBAL},)   classes: {n_classes}")
    print(f"  monitoring  : val_new_loss, guarded by val_old_acc")
    print(f"{'='*60}")

    history = model.fit(
        x=[data['eigen'][idx_train], data['global'][idx_train]],
        y=data['labels'][idx_train],
        sample_weight=sample_weights,
        validation_data=([data['eigen'][idx_val], data['global'][idx_val]],
                         data['labels'][idx_val]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    if checkpoint.best_epoch is not None:
        print(f"\nReloading best checkpoint (epoch {checkpoint.best_epoch}) "
              f"from {model_path}")
        model = tf.keras.models.load_model(model_path)
        # set_model(), not `per_source.model = ...`: in Keras 3 Callback.model is
        # a read-only property backed by _model, so direct assignment raises.
        per_source.set_model(model)
        final_metrics = per_source._evaluate()
    else:
        print(f"\nWARNING: no checkpoint was ever saved -- val_new_loss never "
              f"improved on the loaded model within the forgetting budget.")
        print(f"         Saving the final-epoch weights so the run is inspectable.")
        model.save(model_path)
        final_metrics = per_source.history[-1] if per_source.history else {}

    save_training_curves_png(history, out_dir)
    save_per_source_curves_png(per_source, out_dir)
    print_final_table(per_source, final_metrics)

    summary = {
        'run': run_name,
        'mode': 'incremental',
        'base_model': args.model,
        'old_data_dirs': old_dirs,
        'new_data_dirs': new_dirs,
        'provenance_check': provenance,
        'n_train': int(len(idx_train)),
        'n_val': int(len(idx_val)),
        'n_classes': n_classes,
        'n_keep': N_KEEP,
        'epochs_requested': args.epochs,
        'epochs_run': len(history.history.get('loss', [])),
        'best_epoch': checkpoint.best_epoch,
        'checkpoints_blocked_by_forgetting_guard': checkpoint.n_blocked,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'dropout_rate': args.dropout_rate,
        'max_forget': args.max_forget,
        'loss_weighting': weight_info,
        'baseline_metrics': per_source.baseline,
        'final_metrics': final_metrics,
        'per_source_history': per_source.history,
        'class_distribution': class_distribution,
        'train_provenance': data['meta'],
    }

    summary_path = os.path.join(out_dir, 'incremental_summary.json')
    with open(summary_path, 'w') as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n{'='*60}")
    print('Incremental training complete!')
    print(f"  Model saved to  : {model_path}")
    print(f"  Summary saved to: {summary_path}")
    if checkpoint.n_blocked:
        print(f"  NOTE: {checkpoint.n_blocked} improvement(s) were blocked by the "
              f"forgetting guard.\n        Raise --max-forget to trade old-region "
              f"accuracy for new-region gains.")
    print(f"\nTo test this model, run:")
    print(f"  python test_only.py --model {model_path} ...")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
