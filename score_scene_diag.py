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
normalization). Whether the synthetic gain transfers to real data is exactly what
scoring a real scene is for.

Thin on purpose
---------------
All the work happens in score_scene.score_channel, which selects its input set
from the loaded model's arity. Forking that 1100-line module would mean two copies
of the tiling, caltone-removal and SCM code, and any drift between them would
invalidate the comparison this script exists to support. So

    python score_scene.py <granule> --model models/<diag run>/best_model.keras ...

does the same thing and produces identical output. This entry point exists to fail
loudly and early when handed a 2-input model, rather than deep inside a scene
stream after the granule has already been read.

Comparing two models is a separate, explicit step -- run this per model, then
diff_predictions.py on the resulting predictions_<freq>_<pol>.h5 files.

Usage
-----
    python score_scene_diag.py /path/to/NISAR_L0_..._001.h5 \\
        --model models/urClean_diagprofile/best_model.keras \\
        --pulse-start 46528 --pulse-end 124580 \\
        --output-dir score/urClean_diagprofile/la
"""

import sys

import score_scene


def main():
    args = score_scene.parse_args()

    import tensorflow as tf
    model = tf.keras.models.load_model(args.model)
    n_inputs = len(model.inputs)
    if n_inputs != 3:
        raise SystemExit(
            f"{args.model} takes {n_inputs} input(s), not 3, so it is not a "
            f"sorted-diagonal-profile model. Score it with score_scene.py:\n"
            f"  python score_scene.py {args.l0b_file} --model {args.model} "
            f"--output-dir {args.output_dir}"
        )
    print(f"3 inputs confirmed: "
          f"{', '.join(str(tuple(t.shape[1:])) for t in model.inputs)}")
    del model

    # Delegate. score_scene.main() re-parses argv, which is idempotent here.
    return score_scene.main()


if __name__ == '__main__':
    sys.exit(main())
