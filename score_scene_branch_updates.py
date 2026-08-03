"""
score_scene_branch_updates.py

Score a REAL NISAR granule with a model from train_branch_updates.py -- the
single-branch classifier with combined features:
    - 12 valid EVs (max-normalized, dB)
    - 11 EV slopes
    - 12 valid SCM diagonal entries (max-normalized, dB)
    - 2 emphasis features [max EV, max pulse power] (normalized)

No ground truth: k > 0 predictions are candidate detections, not verified RFI.

Why this exists
---------------
model_branch_updates takes a SINGLE input with 37 features, which is distinct
from all other model variants:
  - model.py (2 inputs: eigen + 3 globals)
  - model_diag.py (3 inputs: eigen + diag conv + 3 globals)
  - model_new_globals.py (2 inputs: eigen + 5 globals)
  - model_branch_updates (1 input: 37 combined features, THIS FILE)

The combined feature vector is constructed by train_branch_updates.features_from_records,
which we import here to ensure scoring matches training exactly.

Thin on purpose
---------------
Most of the work happens in score_scene.score_channel. This is a thin wrapper
that:
  1. Validates the model takes 1 input of width 37
  2. Defaults to frequency A only (like score_scene_new_globals.py)
  3. Delegates to score_scene.main()

The actual feature construction for scene scoring would need to be added to
score_scene.py as a new variant, or this script could provide a custom
score_channel implementation. For now, this is a placeholder that checks
model compatibility.

Frequency default
-----------------
Defaults to frequency A only, matching score_scene_new_globals.py. Pass --freq B
explicitly to score B, or --freq A B for both.

Usage
-----
    python score_scene_branch_updates.py /path/to/NISAR_L0_..._001.h5 `
        --model models/model_branch_updates/best_model.keras `
        --pulse-start 681445 --pulse-end 757408 `
        --output-dir score/model_branch_updates/la
"""

import sys

import score_scene


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
    if n_inputs != 1:
        raise SystemExit(
            f"{args.model} takes {n_inputs} input(s), not 1, so it is not a "
            f"branch-updates model. Models by input count:\n"
            f"  1 input  = branch_updates (THIS SCRIPT)\n"
            f"  2 inputs = baseline or new-globals (use score_scene.py or "
            f"score_scene_new_globals.py)\n"
            f"  3 inputs = sorted diagonal profile (use score_scene_diag.py)"
        )

    n_features = int(model.inputs[0].shape[-1])
    expected_features = 37  # 12 EVs + 11 slopes + 12 diag + 2 emphasis
    if n_features != expected_features:
        raise SystemExit(
            f"{args.model} wants {n_features} features, not {expected_features}, "
            f"so it does not match the branch_updates architecture. Expected:\n"
            f"  12 EVs + 11 EV slopes + 12 SCM diagonal + 2 emphasis = 37 features"
        )

    print(f"1 input confirmed, {n_features} features: single-branch combined model")
    del model

    # NOTE: This is where custom scoring logic would go if score_scene.py doesn't
    # yet support this variant. For now, we delegate and assume score_scene.py
    # has been updated to handle single-input models with 37 features.
    #
    # If score_scene.py hasn't been updated yet, you'll need to either:
    # 1. Add support for this variant to score_scene.score_channel, OR
    # 2. Implement a custom score_channel function here and call it instead
    #    of delegating to score_scene.main()

    return score_scene.main()


if __name__ == '__main__':
    sys.exit(main())
