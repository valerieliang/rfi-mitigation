"""
score_scene_new_globals.py

Score a REAL NISAR granule with a model from train_new_globals.py -- the
benchmark eigenvalue branch with a 5-wide global vector:

    [cond_db, eff_rank, diag_median_max_ratio, schur_horn_gap, participation_ratio]

No ground truth: k > 0 predictions are candidate detections, not verified RFI.

Why this exists
---------------
model_new_globals takes TWO inputs, exactly like the urClean benchmark, so
arity alone cannot tell them apart -- the difference is the global vector's
width, 5 against 3. Handing the benchmark's 3-vector to this model fails inside
predict(), which on a granule this size is many minutes after the mistake was
made. score_scene.score_channel now splits the two-input case on the global
input's width, and this entry point checks the same thing up front so a wrong
--model argument costs a second instead of a scene read.

The two added scalars are computed by diag_features.new_global_features, the
same function train_new_globals.py uses, so scoring cannot drift from training.
Both need the RAW SCM diagonal over all M entries, which the baseline path folds
into one scalar and discards, so score_channel retains it when this variant is
active.

Thin on purpose
---------------
All the work happens in score_scene.score_channel. Forking that 1100-line module
would mean two copies of the tiling, caltone-removal and SCM code, and any drift
between them would invalidate the comparison this script exists to support. So

    python score_scene.py <granule> --model models/model_new_globals/best_model.keras ...

does the same thing and produces identical output.

Comparing models is a separate, explicit step -- run this per model, then
diff_predictions.py on the resulting predictions_<freq>_<pol>.h5 files. Each file
now also carries a 'global_features' dataset with a 'columns' attribute, so the
gap and participation ratio can be examined per tile without re-reading the
granule.

Interpreting the result
-----------------------
Both new scalars carry the same synthetic-to-real shift the sorted profile did --
the participation ratio measures about 1.8 on synthetic single-band tiles and
about 9.9 on real scenes. This variant reduces how much of that shift the model
can fit (128 parameters instead of roughly 155k), it does not remove it. A
difference against the benchmark on a real scene is evidence about robustness,
not about correctness: these scenes are unlabeled, so neither model's disagreement
can be scored as right or wrong without hand-checked tiles.

Frequency default
-----------------
score_scene.py defaults --freq to "every frequency in the granule". Here the
default is frequency A only, matching score_scene_diag.py: the variant is being
evaluated against A-band baselines, and silently pulling in a B channel would add
per-channel outputs that have nothing to diff against while roughly doubling the
runtime. Pass --freq B explicitly to score B.

Usage
-----
    python score_scene_new_globals.py /path/to/NISAR_L0_..._001.h5 \\
        --model models/model_new_globals/best_model.keras \\
        --pulse-start 681445 --pulse-end 757408 \\
        --output-dir score/model_new_globals/la
"""

import sys

import score_scene
from diag_features import N_GLOBAL_NEW


def _default_freq_a():
    """Inject --freq A into argv unless the caller already chose a frequency.

    score_scene.main() re-parses sys.argv rather than taking an args object, so
    the default has to live in argv for both parses to agree. Mutating argv
    before either parse is what keeps them consistent.
    """
    if any(a == '--freq' or a.startswith('--freq=') for a in sys.argv[1:]):
        return
    sys.argv.extend(['--freq', 'A'])


def main():
    _default_freq_a()
    args = score_scene.parse_args()

    import tensorflow as tf
    model = tf.keras.models.load_model(args.model)

    n_inputs = len(model.inputs)
    if n_inputs != 2:
        raise SystemExit(
            f"{args.model} takes {n_inputs} input(s), not 2, so it is not a "
            f"new-globals model. A 3-input model is the sorted diagonal "
            f"profile:\n"
            f"  python score_scene_diag.py {args.l0b_file} --model {args.model} "
            f"--output-dir {args.output_dir}"
        )

    n_global = int(model.inputs[-1].shape[-1])
    if n_global != N_GLOBAL_NEW:
        raise SystemExit(
            f"{args.model} wants a {n_global}-wide global vector, not "
            f"{N_GLOBAL_NEW}, so it is the baseline model rather than a "
            f"new-globals one. Score it with score_scene.py:\n"
            f"  python score_scene.py {args.l0b_file} --model {args.model} "
            f"--output-dir {args.output_dir}"
        )

    print(f"2 inputs confirmed, global vector {n_global} wide: "
          f"{', '.join(str(tuple(t.shape[1:])) for t in model.inputs)}")
    del model

    # Delegate. score_scene.main() re-parses argv, which is idempotent here.
    return score_scene.main()


if __name__ == '__main__':
    sys.exit(main())
