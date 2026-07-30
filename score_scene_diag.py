"""
score_scene_diag.py

Score a REAL NISAR granule with a three-input model from train_diag_profile.py
(the sorted SCM diagonal profile variant). No ground truth: k > 0 predictions are
candidate detections, not verified RFI.

Why this exists
---------------
The diagonal profile earned +0.76 points on low-JSR SYNTHETIC test data, and the
reason it works is that generate_amazon_data.py places each injected band in its
own pulse row -- which makes the label the count of elevated diagonal entries.
That is a property of the INJECTION MODEL, not of physics. Real RFI spanning many
pulses would elevate many rows, or all of them (vanishing under the median
normalization). So the synthetic gain may not transfer, and this script is how you
find out: score a real scene with both models and diff them.

Thin on purpose
---------------
All the work happens in score_scene.score_channel, which now selects its input set
from the loaded model's arity. Forking that 1100-line module would mean two copies
of the tiling, caltone-removal and SCM code -- and any drift between them would
invalidate exactly the comparison this script is for. So:

    python score_scene.py --model models/<diag run>/best_model.keras ...

does the same thing and produces identical output. This entry point exists to
fail loudly on a 2-input model and to print the matching baseline command.

Usage
-----
    python score_scene_diag.py \\
        --l0b-file /path/to/NISAR_L0_..._001.h5 \\
        --model models/urClean_diagprofile/best_model.keras \\
        --output-dir score/urClean_diagprofile/la/12_29 \\
        --baseline-model models/urClean_mtnContam_amzContam/best_model.keras
"""

import sys

import score_scene


def main():
    # score_scene.parse_args() builds and parses in one call, so rather than
    # rebuild its parser (and risk the two drifting apart on scene-geometry,
    # caltone or SCM options) intercept argv: pull --baseline-model out, then
    # delegate everything else untouched.
    baseline = None
    argv = sys.argv[1:]
    if '--baseline-model' in argv:
        i = argv.index('--baseline-model')
        if i + 1 >= len(argv):
            raise SystemExit('--baseline-model needs a path')
        baseline = argv[i + 1]
        del argv[i:i + 2]
        sys.argv = [sys.argv[0]] + argv

    args = score_scene.parse_args()

    import tensorflow as tf
    model = tf.keras.models.load_model(args.model)
    n_inputs = len(model.inputs)
    if n_inputs != 3:
        raise SystemExit(
            f"{args.model} takes {n_inputs} input(s), not 3, so it is not a "
            f"sorted-diagonal-profile model. Score it with score_scene.py:\n"
            f"  python score_scene.py --l0b-file {args.l0b_file} "
            f"--model {args.model} --output-dir {args.output_dir}"
        )
    print(f"3 inputs confirmed: "
          f"{', '.join(str(tuple(t.shape[1:])) for t in model.inputs)}")
    del model, tf

    rc = score_scene.main()

    if baseline:
        out = args.output_dir.rstrip('/')
        print(f"\nFor the like-for-like baseline, run:")
        print(f"  python score_scene.py --l0b-file {args.l0b_file} \\")
        print(f"      --model {baseline} \\")
        print(f"      --output-dir {out}_baseline")
        print(f"\nThen diff the two, per channel:")
        print(f"  python diff_predictions.py \\")
        print(f"      {out}/predictions_A_HH.h5 \\")
        print(f"      {out}_baseline/predictions_A_HH.h5")
        print(f"\nWhat to look for: the synthetic gain came from promoting true-6")
        print(f"tiles that were being called 5, so on a real scene the diagonal")
        print(f"model should predict HIGHER knees on genuinely contaminated tiles")
        print(f"and agree on clean ones. A broad shift on quiet tiles instead")
        print(f"means it is keying on the one-row-per-band injection artifact.")
    return rc


if __name__ == '__main__':
    sys.exit(main())
