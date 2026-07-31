"""
train_incremental_diag.py

train_incremental.py for the THREE-INPUT sorted-diagonal-profile model, i.e. a
checkpoint produced by train_diag_profile.py ([eigen, diag, global]).

Why this is a separate script and not a --diag flag
---------------------------------------------------
Only two things actually change: the loader (train_diag_profile's, which keeps
the raw SCM diagonal instead of folding it into one scalar) and the input list
handed to fit/predict. Everything that makes incremental training work -- the
provenance cross-check, the old/new loss rebalancing, the per-source validation
callback, the forgetting-guarded checkpoint, the optimizer rebuild, the plots
and the final table -- is IMPORTED from train_incremental.py, not copied. The two
scripts therefore cannot drift on the part that is easy to get subtly wrong, and
train_incremental.py's docstring keeps describing exactly the code it contains.

This mirrors how test_only_diag.py and score_scene_diag.py already relate to
their 2-input counterparts.

Read train_incremental.py's module docstring for WHY incremental training needs
loss rebalancing and per-source validation at all. Nothing about that reasoning
changes here: the diagonal branch does not alter the fact that a 59%-old
training pool produces a gradient that mostly says "keep doing what you already
do", nor that an aggregate val_loss dominated by converged old data drives early
stopping on a term that cannot improve.

Fails loudly on a 2-input checkpoint rather than crashing inside fit() with a
shape error, and points at train_incremental.py instead.

Usage:
    python train_incremental_diag.py \
        --model models/urClean_diagprofile/best_model.keras \
        --old-data-dir data/czech_contam/ data/amazon_contam/ data/berlin_clean/ \
        --new-data-dir data/low_jsr_amazon/ \
        --run-name urClean_diagprofile_lowjsr \
        --epochs 100 \
        --new-weight-frac 0.5

Outputs:
    models/<run>/best_model.keras
    models/<run>/training_curves.png
    models/<run>/per_source_curves.png
    models/<run>/incremental_summary.json
"""

import os
import json
import argparse

import tensorflow as tf

from diag_features import DIAG_CHANNELS, N_GLOBAL_DIAG, N_KEEP_DIAG
from train_diag_profile import load_rfi_data_dir_diag
from train_only import (
    MODELS_ROOT,
    N_KEEP,
    VAL_FRAC,
    SPLIT_BUFFER_DEFAULT,
    split_train_val,
    report_class_by_source_split,
    save_training_curves_png,
)
from train_incremental import (
    EPOCHS,
    BATCH_SIZE,
    LR,
    NEW_WEIGHT_FRAC,
    MAX_FORGET,
    PATIENCE,
    LR_PATIENCE,
    PerSourceVal,
    GuardedCheckpoint,
    _dir_name,
    check_provenance,
    compute_sample_weights,
    load_and_prepare,
    save_per_source_curves_png,
    print_final_table,
)


# ---------------------------------------------------------------------------
# CALLBACK
# ---------------------------------------------------------------------------

class PerSourceValDiag(PerSourceVal):
    """
    PerSourceVal with the three-input feed.

    The parent does all the work -- one predict pass per epoch, per-source
    aggregation in numpy, the on_train_begin baseline, and injection of
    val_new_loss / val_old_acc into `logs` for the checkpoint, early stopping and
    LR schedule to monitor. The only thing that differs for a diagonal model is
    the shape of `self.x`, so that is the only thing overridden here.

    Passing global_val to super() and then reassigning self.x looks redundant but
    is not: the parent's constructor is what builds the label/source masks and the
    baseline slot, and it needs a well-formed x only after construction.
    """

    def __init__(self, eigen_val, diag_val, global_val, y_val, sources_val,
                 source_names, n_old_sources, batch_size=1024):
        super().__init__(eigen_val, global_val, y_val, sources_val,
                         source_names, n_old_sources, batch_size)
        self.x = [eigen_val, diag_val, global_val]


# ---------------------------------------------------------------------------
# CHECKPOINT VALIDATION
# ---------------------------------------------------------------------------

def assert_diag_model(model, model_path):
    """
    Refuse a checkpoint that is not the three-input diagonal-profile model.

    Without this the failure surfaces as a Keras shape/arity error several
    hundred lines into the run, after the whole data pool has been loaded. The
    2-input case is the likely mistake -- both scripts take the same flags -- so
    it gets an explicit redirect.
    """
    n_inputs = len(model.inputs)
    if n_inputs != 3:
        raise SystemExit(
            f"\n{model_path} takes {n_inputs} input(s), not 3, so it is not a "
            f"sorted-diagonal-profile model.\n"
            f"Continue training it with train_incremental.py instead:\n"
            f"  python train_incremental.py --model {model_path} ..."
        )

    expected = [(N_KEEP, 2), (N_KEEP_DIAG, DIAG_CHANNELS), (N_GLOBAL_DIAG,)]
    actual = [tuple(t.shape[1:]) for t in model.inputs]
    if actual != expected:
        raise SystemExit(
            f"\n{model_path} has 3 inputs but shapes {actual}, expected "
            f"{expected}.\nThe checkpoint was built with different feature "
            f"constants than diag_features.py currently defines "
            f"(N_KEEP={N_KEEP}, N_KEEP_DIAG={N_KEEP_DIAG}, "
            f"DIAG_CHANNELS={DIAG_CHANNELS}, N_GLOBAL_DIAG={N_GLOBAL_DIAG}); "
            f"the features it learned and the ones this run would feed it are "
            f"not the same quantity."
        )

    print(f"  3 inputs confirmed: {', '.join(str(s) for s in actual)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Continue training a sorted-diagonal-profile (3-input) knee '
                    'classifier on unseen regions, rebalancing the loss so the new '
                    'data is not drowned out by the already-fit old data. For the '
                    '2-input baseline use train_incremental.py.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Base .keras checkpoint from train_diag_profile.py.')
    parser.add_argument('--old-data-dir', type=str, nargs='+', required=True,
                        help='Directories the base model was ALREADY trained on. '
                             'Kept in training as replay against forgetting, but '
                             'down-weighted.')
    parser.add_argument('--new-data-dir', type=str, nargs='+', required=True,
                        help='Directories the base model has NOT seen. These drive '
                             'the loss, the checkpoint metric and early stopping. '
                             "Files must carry raw 'diagonal' and 'diag_valid_idx' "
                             'datasets.')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Model output folder. Defaults to the first new data '
                             'dir name plus "_incremental_diag".')
    parser.add_argument('--models-root', type=str, default=MODELS_ROOT,
                        help='Parent directory for the run folder.')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Cap tiles loaded per file (evenly strided), for quick runs.')
    parser.add_argument('--val-frac', type=float, default=VAL_FRAC,
                        help='Fraction of pulse tiles held out for val, per source.')
    parser.add_argument('--split-buffer', type=int, default=SPLIT_BUFFER_DEFAULT,
                        help='CPI ROWS dropped on each side of the train/val '
                             'boundary (1 row = one tile_pulse value = 16 '
                             'pulses). Too small a buffer makes val_new_loss '
                             'track memorization of the training scenes rather '
                             'than generalization.')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--learning-rate', type=float, default=LR,
                        help='Initial learning rate. The optimizer is rebuilt, so '
                             'this restarts the schedule rather than resuming it.')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='L2 regularization strength (AdamW weight decay).')
    parser.add_argument('--dropout-rate', type=float, default=None,
                        help='Override the fusion-head dropout of the loaded model '
                             '(the last Dropout layer; the global branch\'s fixed '
                             '0.3 is left alone). Default: inherit whatever the '
                             'checkpoint was built with.')
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

    run_name = args.run_name or f'{_dir_name(new_dirs[0])}_incremental_diag'
    os.makedirs(args.models_root, exist_ok=True)
    out_dir = os.path.join(args.models_root, run_name)
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'best_model.keras')

    print(f"\n{'='*70}")
    print('RFI knee classifier - INCREMENTAL TRAINING (sorted diagonal profile)')
    print(f"{'='*70}")
    print(f"  base model        : {args.model}")
    print(f"  old region(s)     : {', '.join(old_dirs)}")
    print(f"  new region(s)     : {', '.join(new_dirs)}")
    print(f"  eigen features    : top {N_KEEP} eigenvalues, linear-normalized then dB")
    print(f"  diag features     : top {N_KEEP_DIAG} valid SCM diagonal entries, "
          f"sorted descending, dB rel. max, {DIAG_CHANNELS} ch")
    print(f"  global features   : [cond_db, eff_rank, diag_median_max_ratio]")
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
    data = load_rfi_data_dir_diag(old_dirs + new_dirs, args.max_samples,
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
    assert_diag_model(model, args.model)

    per_source = PerSourceValDiag(
        data['eigen'][idx_val], data['diag'][idx_val], data['global'][idx_val],
        data['labels'][idx_val], data['sources'][idx_val],
        source_names, n_old_sources,
    )
    checkpoint = GuardedCheckpoint(model_path, per_source, args.max_forget)

    # PerSourceValDiag must come first: it injects val_new_loss / val_old_acc into
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
    print(f"  eigen input : ({N_KEEP}, 2)   diag input : ({N_KEEP_DIAG}, "
          f"{DIAG_CHANNELS})   global: ({N_GLOBAL_DIAG},)   classes: {n_classes}")
    print(f"  monitoring  : val_new_loss, guarded by val_old_acc")
    print(f"{'='*60}")

    # Input order must match build_model_diag: [eigen, diag, global].
    train_inputs = [data['eigen'][idx_train], data['diag'][idx_train],
                    data['global'][idx_train]]
    val_inputs = [data['eigen'][idx_val], data['diag'][idx_val],
                  data['global'][idx_val]]

    history = model.fit(
        x=train_inputs,
        y=data['labels'][idx_train],
        sample_weight=sample_weights,
        validation_data=(val_inputs, data['labels'][idx_val]),
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
        'variant': 'sorted_diag_profile',
        'base_model': args.model,
        'old_data_dirs': old_dirs,
        'new_data_dirs': new_dirs,
        'provenance_check': provenance,
        'inputs': {'eigen': [N_KEEP, 2], 'diag': [N_KEEP_DIAG, DIAG_CHANNELS],
                   'global': [N_GLOBAL_DIAG]},
        'diag_normalization': 'divide by max valid entry (linear), then dB',
        'scope': 'per-CPI only',
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

    # check_provenance reads training_summary.json, so a SECOND incremental run
    # stacked on this one would find nothing next to the checkpoint and fall back
    # to trusting its flags. Write the same content under that name too, so the
    # chain stays verifiable.
    with open(os.path.join(out_dir, 'training_summary.json'), 'w') as fh:
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
    print(f"  python test_only_diag.py --model {model_path} \\")
    print(f"      --data-dirs <labeled test dirs> --output-dir results/{run_name}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
