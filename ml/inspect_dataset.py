#!/usr/bin/env python3
"""
inspect_dataset.py

Print a pre-training sanity check for the three split .npz files produced
by build_dataset.py. Run this before training to confirm the dataset looks
right.

Usage:
    python ml/inspect_dataset.py \
        --train data/model/train.npz \
        --val   data/model/val.npz \
        --test  data/model/test.npz
"""

import argparse
import collections
import os
import sys

import numpy as np


BANDS = [
    (0,   0,  "clean "),
    (1,   1,  "1     "),
    (2,   2,  "2     "),
    (3,   5,  "3-5   "),
    (6,   9,  "6-9   "),
    (10, 14,  "10-14 "),
    (15, 20,  "15-20 "),
    (21, 26,  "21-26 "),
    (27, 32,  "27-32 "),
]
BAR_WIDTH = 30


def band_count(labels, lo, hi):
    return int(((labels >= lo) & (labels <= hi)).sum())


def bar(n, target, width=BAR_WIDTH):
    filled = int(round(width * min(n / target, 1.0))) if target > 0 else 0
    pct = n / target * 100 if target > 0 else 0
    flag = " OK " if n >= target * 0.95 else "WARN"
    return "[{}{}] {:4d} ({:5.1f}%) [{}]".format(
        "#" * filled, "." * (width - filled), n, pct, flag)


def check_split(path, name, target_per_band=None, norm_mean=None, norm_std=None):
    if not os.path.exists(path):
        print("  {} not found: {}".format(name.upper(), path))
        return None

    d = np.load(path, allow_pickle=True)
    labels  = d["labels"]
    eigen   = d["eigen_input"]
    glob_r  = d["global_input_raw"] if "global_input_raw" in d.files else d["global_input"]
    M       = int(d["M"])
    n       = len(labels)

    print("=" * 68)
    print("{} -- {}  ({} samples, M={})".format(name.upper(), path, n, M))
    print("=" * 68)

    # --- sample counts ---
    n_clean = int((labels == 0).sum())
    n_rfi   = int((labels >  0).sum())
    print("\nSplit summary:")
    print("  total    : {:6d}".format(n))
    print("  clean    : {:6d}  ({:.1f}%)".format(n_clean, 100 * n_clean / n))
    print("  RFI      : {:6d}  ({:.1f}%)".format(n_rfi,   100 * n_rfi   / n))

    # --- knee band balance ---
    target = target_per_band or max(band_count(labels, lo, hi) for lo, hi, _ in BANDS)
    print("\nKnee band balance  (target ~{}):".format(target))
    for lo, hi, tag in BANDS:
        c = band_count(labels, lo, hi)
        print("  {} {}".format(tag, bar(c, target)))

    # --- per-class counts for sparse classes ---
    hist = collections.Counter(labels.tolist())
    sparse = [(k, v) for k, v in sorted(hist.items()) if v < 10]
    if sparse:
        print("\n  Classes with <10 samples (watch these):")
        for k, v in sparse:
            print("    knee={:2d} : {}".format(k, v))
    else:
        print("\n  No classes with <10 samples.")

    # --- eigenvalue profile sanity ---
    print("\nEigenvalue profile sanity:")
    mean_profile = eigen[:, :, 0].mean(axis=0)
    std_profile  = eigen[:, :, 0].std(axis=0)
    print("  index 0  mean={:6.2f} dB  std={:.2f}".format(mean_profile[0],  std_profile[0]))
    print("  index 15 mean={:6.2f} dB  std={:.2f}".format(mean_profile[15], std_profile[15]))
    print("  index 31 mean={:6.2f} dB  std={:.2f}".format(mean_profile[31], std_profile[31]))
    # Check that eigenvalues are descending on average (they should be).
    descending = bool(np.all(np.diff(mean_profile) < 0.5))
    print("  mean profile descending: {}{}".format(
        descending, "" if descending else "  <-- WARN: unexpected"))
    # Check for NaN / Inf.
    n_bad = int(np.isnan(eigen).sum() + np.isinf(eigen).sum())
    print("  NaN/Inf in eigen_input : {}{}".format(
        n_bad, "" if n_bad == 0 else "  <-- WARN"))

    # --- global features ---
    print("\nGlobal feature stats (raw, before normalisation):")
    feat_names = (list(d["feature_names"]) if "feature_names" in d.files
                  else ["F", "sigma_min", "sigma_max", "mu_min",
                        "trace_db", "log10_cond"])
    for i, fname in enumerate(feat_names):
        col = glob_r[:, i]
        print("  {:22s}  mean={:8.3f}  std={:7.3f}  "
              "min={:8.3f}  max={:8.3f}".format(
                  str(fname), col.mean(), col.std(), col.min(), col.max()))

    # --- normalisation check (val/test only) ---
    if norm_mean is not None:
        glob_n = d["global_input"]
        residual = np.abs(glob_n.mean(axis=0))
        print("\nNormalisation residual (should be ~0 if stats match):")
        for i, fname in enumerate(feat_names):
            flag = "" if residual[i] < 0.15 else "  <-- WARN: check norm_mean/std"
            print("  {:22s}  mean of normed col = {:+.4f}{}".format(
                str(fname), glob_n[:, i].mean(), flag))

    # --- F-factor separation ---
    F_clean = glob_r[labels == 0, 0]
    F_rfi   = glob_r[labels >  0, 0]
    if len(F_clean) and len(F_rfi):
        print("\nF-factor (sigma_max/sigma_min):")
        print("  clean  mean={:.2f}  median={:.2f}  p95={:.2f}".format(
            F_clean.mean(), np.median(F_clean), np.percentile(F_clean, 95)))
        print("  RFI    mean={:.2f}  median={:.2f}  p95={:.2f}".format(
            F_rfi.mean(),   np.median(F_rfi),   np.percentile(F_rfi, 95)))

    # --- n_bad global features ---
    n_bad_g = int(np.isnan(glob_r).sum() + np.isinf(glob_r).sum())
    if n_bad_g:
        print("\nWARN: {} NaN/Inf values in global features".format(n_bad_g))

    print()
    return d


def main():
    ap = argparse.ArgumentParser(description="Pre-training dataset sanity check.")
    ap.add_argument("--train", default="data/model/train.npz")
    ap.add_argument("--val",   default="data/model/val.npz")
    ap.add_argument("--test",  default="data/model/test.npz")
    ap.add_argument("--target-per-band", type=int, default=500)
    args = ap.parse_args()

    d_train = check_split(args.train, "train",
                          target_per_band=args.target_per_band)

    # Extract train norm stats to verify val/test are normed the same way.
    nm = d_train["norm_mean"] if d_train and "norm_mean" in d_train.files else None
    ns = d_train["norm_std"]  if d_train and "norm_std"  in d_train.files else None

    check_split(args.val,  "val",
                target_per_band=args.target_per_band, norm_mean=nm, norm_std=ns)
    check_split(args.test, "test",
                target_per_band=args.target_per_band, norm_mean=nm, norm_std=ns)

    # --- cross-split consistency ---
    print("=" * 68)
    print("CROSS-SPLIT CONSISTENCY")
    print("=" * 68)
    if d_train:
        for path, name in ((args.val, "val"), (args.test, "test")):
            if not os.path.exists(path):
                continue
            d = np.load(path, allow_pickle=True)
            if "norm_mean" in d.files and nm is not None:
                diff = np.abs(d["norm_mean"] - nm)
                match = bool(diff.max() < 1e-4)
                print("  {} norm_mean matches train: {}{}".format(
                    name, match,
                    "" if match else "  <-- WARN: normalisation mismatch"))
    print()
    print("All checks done. If everything shows OK, run:")
    print("  python ml/train.py \\")
    print("    --data      {} \\".format(args.train))
    print("    --val-data  {} \\".format(args.val))
    print("    --test-data {} \\".format(args.test))
    print("    --label-smoothing 0.1 --weight-decay 1e-4 --epochs 100 \\")
    print("    --out-dir ml/checkpoints")


if __name__ == "__main__":
    main()