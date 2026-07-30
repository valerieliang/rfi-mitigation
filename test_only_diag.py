"""
test_only_diag.py

Evaluate a model trained by train_diag_profile.py -- the three-input variant that
consumes the sorted SCM diagonal profile ([eigen, diag, global]).

This is a thin entry point on purpose. It reuses test_only.combined_synthetic_test
rather than reimplementing it: the whole reason to run this is an eye-to-eye
comparison against the baseline, and that comparison is only trustworthy if both
models go through identical binning, confusion-matrix, classification-report and
accuracy-vs-JSR code. A forked copy of that harness would drift.

test_only.py already selects its input set from the loaded model's arity, so

    python test_only.py --model models/<diag run>/best_model.keras ...

works too and produces byte-identical results. This script exists to make the
intent explicit, to fail loudly if handed a 2-input model, and to print the
matching baseline command for the comparison.

Usage
-----
    python test_only_diag.py \\
        --model models/urClean_diagprofile/best_model.keras \\
        --data-dirs data/amazon_test_low_jsr data/amazon_test_high_jsr \\
        --output-dir results/urClean_diagprofile_test
"""

import os
import sys
import argparse

from test_only import combined_synthetic_test


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a sorted-diagonal-profile (3-input) knee classifier '
                    'on labeled synthetic test data. For the 2-input baseline use '
                    'test_only.py; for real NISAR scenes use score_scene.py.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model', required=True,
                        help='Model from train_diag_profile.py (best_model.keras)')
    parser.add_argument('--data-dirs', nargs='+', required=True,
                        help='Test directories with labeled .h5 tile files. Each is '
                             'scored separately AND pooled. Files must carry raw '
                             "'diagonal' and 'diag_valid_idx' datasets.")
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--baseline-model', default=None,
                        help='Optional 2-input baseline checkpoint. Only used to '
                             'print the matching test_only.py command for a '
                             'like-for-like comparison; it is not run here.')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    import tensorflow as tf
    print(f"Loading model: {args.model}")
    model = tf.keras.models.load_model(args.model)

    n_inputs = len(model.inputs)
    if n_inputs != 3:
        raise SystemExit(
            f"{args.model} takes {n_inputs} input(s), not 3, so it is not a "
            f"sorted-diagonal-profile model. Score it with test_only.py instead:\n"
            f"  python test_only.py --model {args.model} "
            f"--data-dirs {' '.join(args.data_dirs)} "
            f"--output-dir {args.output_dir}"
        )

    print(f"  3 inputs confirmed: "
          f"{', '.join(str(tuple(t.shape[1:])) for t in model.inputs)}")

    combined_synthetic_test(model, args.data_dirs, args)

    print(f"\n{'='*70}")
    print(f"DONE. Results saved to {args.output_dir}")
    if args.baseline_model:
        print(f"\nFor the like-for-like baseline number, run:")
        print(f"  python test_only.py --model {args.baseline_model} \\")
        print(f"      --data-dirs {' '.join(args.data_dirs)} \\")
        print(f"      --output-dir {args.output_dir.rstrip('/')}_baseline")
        print(f"\nCompare overall_accuracy, per_dataset_accuracy and the class-6")
        print(f"recall in each results_combined.json.")
    print(f"{'='*70}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
