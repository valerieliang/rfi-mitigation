#!/usr/bin/env python3
"""
build_dataset.py

One command: scan clean scenes -> scene-level split -> balanced RFI
augmentation -> train.npz / val.npz / test.npz.

Usage (40 scenes on disk, no manifest needed):

    python ml/build_dataset.py \
        --data-dir  nisar_out \
        --out-dir   data/model \
        --target-per-band 500 \
        --figure

Outputs:
    data/model/train.npz        train samples + norm stats
    data/model/val.npz          val samples, normed by train stats
    data/model/test.npz         test samples, normed by train stats
    data/model/scene_split.csv  which granule went to which split
    data/model/dataset_meta.json

Design:
    Scene-level split: all CPIs from a granule go to exactly one split,
    preventing the model from memorising scene-specific speckle.

    Balanced knee classes: quota-driven generation fills all 9 knee bands
    to --target-per-band samples in every split.

    Normalization locked to train: mean/std computed on train only,
    applied identically to val and test.

With 40 scenes the 70/15/15 default gives 28 train / 6 val / 6 test.
Requires augment.py, split_scenes.py, and rfi_gen/ alongside this file.
"""

import argparse
import collections
import csv
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# ---------------------------------------------------------------------------
# Lazy imports of sibling modules
# ---------------------------------------------------------------------------
def _import_augment():
    import importlib.util
    path = os.path.join(_HERE, "augment.py")
    if not os.path.exists(path):
        sys.exit("augment.py not found alongside build_dataset.py")
    spec = importlib.util.spec_from_file_location("augment", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _import_split():
    import importlib.util
    path = os.path.join(_HERE, "split_scenes.py")
    if not os.path.exists(path):
        sys.exit("split_scenes.py not found alongside build_dataset.py")
    spec = importlib.util.spec_from_file_location("split_scenes", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Per-split builder
# ---------------------------------------------------------------------------
def build_split(scenes, split_name, args, M, K, inr_range, styles,
                rng_seed, norm_mean=None, norm_std=None):
    """
    Augment one split's scenes with synthetic RFI and return a data dict.

    scenes          list of .h5 filenames in args.data_dir
    split_name      "train" | "val" | "test"  (logging only)
    norm_mean/std   if None, compute from this split (train);
                    otherwise apply provided stats (val/test)
    """
    aug = _import_augment()
    rng = np.random.default_rng(rng_seed)

    # Recycle scenes round-robin into a large pool; each TB samples a
    # random window so repeats still produce different blocks.
    pool_size = 100_000
    tb_jobs = [scenes[i % len(scenes)] for i in range(pool_size)]
    rng.shuffle(tb_jobs)

    class _A:
        pass
    a = _A()
    a.cpi_per_tb        = args.cpi_per_tb
    a.target_per_band   = args.target_per_band
    a.max_tbs           = 0
    a.intermittent_frac = args.intermittent_frac
    a.clean_frac        = None
    a.data_dir          = args.data_dir
    a.synthetic         = False
    a._styles           = styles
    a._inr_range        = inr_range

    print("\n--- {} split: {} scene(s), target {}/band ---".format(
        split_name.upper(), len(scenes), args.target_per_band))

    all_records, tb_id, bands, counts, target = aug.build_balanced(
        tb_jobs, a, M, K, rng)

    print("Per-band fill:")
    for i, (lo, hi) in enumerate(bands):
        tag = "clean" if i == 0 else "{}-{}".format(lo, hi)
        print("  {:>7}: {:4d} / {}".format(tag, counts[i], target[i]))

    if not all_records:
        sys.exit("No samples built for '{}'. Check --data-dir."
                 .format(split_name))

    # Assemble numpy arrays.
    eigen_input = np.stack([r["eigen"] for r in all_records]).astype(np.float32)
    labels      = np.array([r["label"] for r in all_records], dtype=np.int32)
    tb_ids      = np.array([r["tb_id"] for r in all_records], dtype=np.int32)
    inr_arr     = np.array(
        [r["inr"] if not np.isnan(r.get("inr", float("nan"))) else -1.0
         for r in all_records], dtype=np.float32)

    feat_names = ["F", "sigma_min", "sigma_max", "mu_min",
                  "trace_db", "log10_condition_number"]
    global_raw = np.array(
        [[r["F"], r["sigma_min"], r["sigma_max"], r["mu_min"],
          r["trace_db"], r["log_cond"]] for r in all_records],
        dtype=np.float32)

    style_list  = sorted(set(r["style"] for r in all_records))
    style_to_id = {s: i for i, s in enumerate(style_list)}
    style_id    = np.array([style_to_id[r["style"]] for r in all_records],
                            dtype=np.int32)
    granules    = np.array([r.get("granule", "") for r in all_records])

    # Normalization.
    if norm_mean is None:
        norm_mean = global_raw.mean(axis=0).astype(np.float32)
        norm_std  = global_raw.std(axis=0).astype(np.float32)
        norm_std[norm_std < 1e-6] = 1.0
        print("Computed normalization stats from {} split.".format(split_name))

    global_input = ((global_raw - norm_mean) / norm_std).astype(np.float32)

    n = len(all_records)
    print("Built {} samples across {} threshold blocks.".format(n, tb_id))

    return dict(
        eigen_input      = eigen_input,
        global_input     = global_input,
        global_input_raw = global_raw,
        labels           = labels,
        tb_id            = tb_ids,
        style_id         = style_id,
        inr              = inr_arr,
        granules         = granules,
        norm_mean        = norm_mean,
        norm_std         = norm_std,
        feature_names    = np.array(feat_names),
        style_names      = np.array(style_list),
        M                = np.int32(M),
        n_knee_classes   = np.int32(M + 1),
        _n_samples       = n,
        _n_tbs           = tb_id,
    )


def save_split(data, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # Strip private bookkeeping keys before saving.
    payload = {k: v for k, v in data.items() if not k.startswith("_")}
    np.savez_compressed(path, **payload)
    sz = os.path.getsize(path) / 1e6
    print("Saved {} ({:.0f} MB, {} samples)".format(
        path, sz, data["_n_samples"]))


def write_figure(path, data, split_name):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figure for {}"
              .format(split_name))
        return

    eigen  = data["eigen_input"]
    labels = data["labels"]
    glob_r = data["global_input_raw"]
    M      = eigen.shape[1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Dataset diagnostics: {}".format(split_name))

    axes[0].hist(labels, bins=np.arange(labels.max() + 2) - 0.5)
    axes[0].set_title("knee label distribution")
    axes[0].set_xlabel("knee index (0 = clean)")
    axes[0].set_ylabel("samples")

    F = glob_r[:, 0]
    axes[1].hist(F[labels == 0], bins=30, alpha=0.6, label="clean")
    axes[1].hist(F[labels >  0], bins=30, alpha=0.6, label="RFI")
    axes[1].set_title("F factor by RFI presence")
    axes[1].set_xlabel("F = sigma_max / sigma_min")
    axes[1].legend(fontsize=8)

    x = np.arange(1, M + 1)
    for tag, cond in (("clean",   labels == 0),
                      ("knee=1-2",(labels >= 1) & (labels <= 2)),
                      ("knee>=8", labels >= 8)):
        idx = np.where(cond)[0]
        if idx.size:
            axes[2].plot(x, eigen[idx[0], :, 0], marker=".", ms=3,
                         label="{} (label {})".format(tag, labels[idx[0]]))
    axes[2].set_title("example eigenvalue profiles")
    axes[2].set_xlabel("eigenvalue index")
    axes[2].set_ylabel("eigenvalue (dB)")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("Wrote figure: {}".format(path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Scene-split + balanced RFI augmentation -> "
                    "data/model/{train,val,test}.npz")
    ap.add_argument("--manifest", default=None,
                    help="Screen manifest CSV from fetch_nisar.py. "
                         "Omit to scan --data-dir for all .h5 files.")
    ap.add_argument("--data-dir", default="nisar_out",
                    help="Directory holding the clean .h5 granules.")
    ap.add_argument("--out-dir", default="data/model",
                    help="Output directory.")
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val",   type=float, default=0.15)
    ap.add_argument("--test",  type=float, default=0.15)
    ap.add_argument("--target-per-band", type=int, default=500,
                    help="CPI samples per knee band per split "
                         "(9 bands -> 9x this many total samples). "
                         "Default 500.")
    ap.add_argument("--M", type=int, default=32,
                    help="CPI size in pulses.")
    ap.add_argument("--k-range", type=int, default=128,
                    help="Range samples per CPI block.")
    ap.add_argument("--cpi-per-tb", type=int, default=20,
                    help="CPIs per threshold block.")
    ap.add_argument("--inr-min", type=float, default=2.0)
    ap.add_argument("--inr-max", type=float, default=30.0)
    ap.add_argument("--intermittent-frac", type=float, default=0.4,
                    help="Fraction of RFI threshold blocks that are "
                         "intermittent (mix of clean and RFI CPIs).")
    ap.add_argument("--styles", default="all",
                    help="Comma list of RFI styles, or 'all'.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Base augmentation seed. val/test get seed+1/+2.")
    ap.add_argument("--split-seed", type=int, default=42,
                    help="Seed for the scene-level split assignment.")
    ap.add_argument("--figure", action="store_true",
                    help="Write a diagnostic PNG for each split.")
    args = ap.parse_args()

    total = args.train + args.val + args.test
    if abs(total - 1.0) > 0.01:
        sys.exit("--train + --val + --test must sum to 1.0")
    fracs = (args.train / total, args.val / total, args.test / total)

    import rfi_gen
    styles = (list(rfi_gen.STYLES) if args.styles == "all"
              else [s.strip() for s in args.styles.split(",") if s.strip()])
    bad = [s for s in styles if s not in rfi_gen.STYLES]
    if bad:
        sys.exit("Unknown styles: {}".format(bad))

    inr_range = (args.inr_min, args.inr_max)
    M, K = args.M, args.k_range

    # -----------------------------------------------------------------------
    # Step 1: scene-level split
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: scene-level split")
    print("=" * 60)

    ss = _import_split()

    if args.manifest:
        if not os.path.exists(args.manifest):
            sys.exit("Manifest not found: {}\n"
                     "Omit --manifest to scan --data-dir directly."
                     .format(args.manifest))
        rows = ss.read_manifest(args.manifest)
        if not rows:
            sys.exit("No clean scenes in manifest.")
        print("Read {} scene(s) from manifest.".format(len(rows)))
    else:
        rows = ss.scan_dir(args.data_dir)
        if not rows:
            sys.exit("No .h5 files in: {}".format(args.data_dir))
        print("Scanned {} .h5 file(s) from {}.".format(
            len(rows), args.data_dir))

    rng_split = np.random.default_rng(args.split_seed)
    scenes_meta, missing = [], []
    for row in rows:
        fname = row["filename"]
        if not os.path.exists(os.path.join(args.data_dir, fname)):
            missing.append(fname)
            continue
        meta = ss.parse_granule_name(row["granule_name"])
        scenes_meta.append(dict(
            filename     = fname,
            granule_name = row["granule_name"],
            pol          = meta["pol"]   or "",
            track        = meta["track"] or "",
            cycle        = meta["cycle"] or "",
        ))

    if missing:
        print("Warning: {} file(s) listed but not on disk (skipped)."
              .format(len(missing)))
    if len(scenes_meta) < 6:
        sys.exit("Need at least 6 on-disk scenes. Found {}."
                 .format(len(scenes_meta)))

    train_sc, val_sc, test_sc = ss.stratified_split(
        scenes_meta, fracs, rng_split)

    for sc, name in ((train_sc, "train"), (val_sc, "val"), (test_sc, "test")):
        if len(sc) < 2:
            sys.exit("Split '{}' has only {} scene(s); need >= 2."
                     .format(name, len(sc)))

    print("  train : {} scenes".format(len(train_sc)))
    print("  val   : {} scenes".format(len(val_sc)))
    print("  test  : {} scenes".format(len(test_sc)))
    for name, grp in (("train", train_sc), ("val", val_sc), ("test", test_sc)):
        pols = collections.Counter(s["pol"] for s in grp)
        print("    {} pols: {}".format(
            name, ", ".join("{} {}".format(v, k)
                            for k, v in sorted(pols.items()))))

    os.makedirs(args.out_dir, exist_ok=True)
    split_csv = os.path.join(args.out_dir, "scene_split.csv")
    cols = ["filename", "split", "granule_name", "track", "cycle", "pol"]
    with open(split_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for sname, grp in (("train", train_sc), ("val", val_sc),
                            ("test", test_sc)):
            for s in grp:
                w.writerow({k: s.get(k, "") for k in cols}
                           | {"split": sname})
    print("Scene split: {}".format(split_csv))

    # -----------------------------------------------------------------------
    # Step 2: augment train (and record norm stats)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: TRAIN")
    print("=" * 60)
    train_data = build_split(
        [s["filename"] for s in train_sc],
        "train", args, M, K, inr_range, styles, rng_seed=args.seed)
    norm_mean = train_data["norm_mean"]
    norm_std  = train_data["norm_std"]
    train_path = os.path.join(args.out_dir, "train.npz")
    save_split(train_data, train_path)
    if args.figure:
        write_figure(train_path.replace(".npz", "_diagnostics.png"),
                     train_data, "train")

    # -----------------------------------------------------------------------
    # Step 3: augment val (normed by train stats)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: VAL")
    print("=" * 60)
    val_data = build_split(
        [s["filename"] for s in val_sc],
        "val", args, M, K, inr_range, styles, rng_seed=args.seed + 1,
        norm_mean=norm_mean, norm_std=norm_std)
    val_path = os.path.join(args.out_dir, "val.npz")
    save_split(val_data, val_path)
    if args.figure:
        write_figure(val_path.replace(".npz", "_diagnostics.png"),
                     val_data, "val")

    # -----------------------------------------------------------------------
    # Step 4: augment test (normed by train stats)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: TEST")
    print("=" * 60)
    test_data = build_split(
        [s["filename"] for s in test_sc],
        "test", args, M, K, inr_range, styles, rng_seed=args.seed + 2,
        norm_mean=norm_mean, norm_std=norm_std)
    test_path = os.path.join(args.out_dir, "test.npz")
    save_split(test_data, test_path)
    if args.figure:
        write_figure(test_path.replace(".npz", "_diagnostics.png"),
                     test_data, "test")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Done")
    print("=" * 60)
    for sname, data in (("train", train_data), ("val", val_data),
                         ("test", test_data)):
        hist = collections.Counter(data["labels"].tolist())
        n_rfi = sum(v for k, v in hist.items() if k > 0)
        print("  {:5s}: {:5d} samples  clean={:4d}  rfi={:4d}".format(
            sname, data["_n_samples"], hist.get(0, 0), n_rfi))

    meta = dict(
        n_train_scenes  = len(train_sc),
        n_val_scenes    = len(val_sc),
        n_test_scenes   = len(test_sc),
        n_train         = int(train_data["_n_samples"]),
        n_val           = int(val_data["_n_samples"]),
        n_test          = int(test_data["_n_samples"]),
        M               = M,
        k_range         = K,
        cpi_per_tb      = args.cpi_per_tb,
        target_per_band = args.target_per_band,
        styles          = styles,
        inr_range       = [args.inr_min, args.inr_max],
        norm_mean       = norm_mean.tolist(),
        norm_std        = norm_std.tolist(),
        split_fracs     = {"train": args.train,
                           "val":   args.val,
                           "test":  args.test},
        seeds           = {"split": args.split_seed,
                           "augment": args.seed},
    )
    meta_path = os.path.join(args.out_dir, "dataset_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print("\nMetadata: {}".format(meta_path))
    print("\nNext:")
    print("  python ml/train.py \\")
    print("    --data      {} \\".format(train_path))
    print("    --val-data  {} \\".format(val_path))
    print("    --test-data {} \\".format(test_path))
    print("    --label-smoothing 0.1 --weight-decay 1e-4 --epochs 100 \\")
    print("    --out-dir ml/checkpoints")


if __name__ == "__main__":
    main()
