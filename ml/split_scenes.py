#!/usr/bin/env python3
"""
split_scenes.py

Assign clean granule files to train / val / test splits AT THE SCENE
LEVEL so no granule ever appears in more than one split.

Input: either --manifest (the screen manifest CSV from fetch_nisar.py)
OR just --data-dir on its own -- in which case every .h5 file found in
that directory is treated as a clean scene. Use the latter when you have
the files but the manifest was not saved.

Produces scene_split.csv with columns:
    filename, split, granule_name, track, cycle, pol

Usage -- with manifest:
    python split_scenes.py \
        --manifest nisar_out/nisar_screen_manifest.csv \
        --data-dir nisar_out --out data/model/scene_split.csv

Usage -- manifest missing, scan dir directly:
    python split_scenes.py \
        --data-dir nisar_out --out data/model/scene_split.csv

The split is stratified by polarization (DHDH vs SHSH) so both pols
are represented in every split. Reproducible with --seed.
"""

import argparse
import collections
import csv
import os
import sys


def parse_granule_name(name):
    """Decode track, cycle, pol from a NISAR RSLC granule name."""
    tokens = name.split("_")
    fields = {"cycle": None, "track": None, "pol": None}
    try:
        i = tokens.index("RSLC")
        fields["cycle"] = tokens[i + 1]
        fields["track"] = tokens[i + 2]
        fields["pol"]   = tokens[i + 5]
    except (ValueError, IndexError):
        pass
    return fields


def read_manifest(csv_path):
    """
    Return scene dicts from a screen manifest or legacy QC CSV.
    Screen manifest: keeps rows where kept==1.
    QC CSV: keeps rows where flag==CLEAN.
    """
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        for row in reader:
            if "kept" in cols:
                if str(row.get("kept", "0")).strip() == "1":
                    name  = row.get("name", "").strip()
                    fname = row.get("file", name + ".h5").strip()
                    rows.append({"filename": fname, "granule_name": name})
            elif "flag" in cols:
                if row.get("flag", "").strip().upper() == "CLEAN":
                    fname = row.get("file", "").strip()
                    rows.append({"filename": fname, "granule_name": fname})
    return rows


def scan_dir(data_dir):
    """
    Fallback: treat every .h5 in data_dir as a clean scene.
    Used when the manifest was not saved but the files are on disk.
    """
    rows = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.lower().endswith(".h5"):
            stem = os.path.splitext(fname)[0]
            rows.append({"filename": fname, "granule_name": stem})
    return rows


def stratified_split(scenes, fracs, rng):
    """
    Split into (train, val, test) preserving pol distribution.
    fracs = (train_frac, val_frac, test_frac) summing to 1.
    """
    import numpy as np
    by_pol = collections.defaultdict(list)
    for s in scenes:
        by_pol[s["pol"] or "unknown"].append(s)

    train, val, test = [], [], []
    for pol, group in sorted(by_pol.items()):
        arr = list(group)
        rng.shuffle(arr)
        n = len(arr)
        n_val  = max(1, round(fracs[1] * n))
        n_test = max(1, round(fracs[2] * n))
        n_train = n - n_val - n_test
        if n_train < 1:
            n_train = 1
            n_val   = max(0, (n - 1) // 2)
            n_test  = n - 1 - n_val
        train.extend(arr[:n_train])
        val.extend(arr[n_train:n_train + n_val])
        test.extend(arr[n_train + n_val:])

    return train, val, test


def main():
    ap = argparse.ArgumentParser(
        description="Assign clean NISAR granules to train/val/test splits "
                    "at the scene level.")
    ap.add_argument("--manifest", default=None,
                    help="Screen manifest CSV (nisar_screen_manifest.csv) "
                         "or legacy QC CSV. If omitted, every .h5 file in "
                         "--data-dir is used (manifest-free mode).")
    ap.add_argument("--data-dir", default=".",
                    help="Directory holding the .h5 files.")
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val",   type=float, default=0.15)
    ap.add_argument("--test",  type=float, default=0.15)
    ap.add_argument("--out",   default="scene_split.csv")
    ap.add_argument("--seed",  type=int, default=42)
    args = ap.parse_args()

    total = args.train + args.val + args.test
    if abs(total - 1.0) > 0.01:
        sys.exit("--train + --val + --test must sum to 1.0 (got {:.3f})"
                 .format(total))
    fracs = (args.train / total, args.val / total, args.test / total)

    # Load scene list from manifest or by scanning the directory.
    if args.manifest:
        if not os.path.exists(args.manifest):
            sys.exit("Manifest not found: {}\n"
                     "If you don't have the manifest, omit --manifest and "
                     "the script will scan --data-dir for .h5 files."
                     .format(args.manifest))
        rows = read_manifest(args.manifest)
        if not rows:
            sys.exit("No kept/clean scenes found in {}".format(args.manifest))
        print("Read {} scene(s) from manifest.".format(len(rows)))
    else:
        if not os.path.isdir(args.data_dir):
            sys.exit("--data-dir not found: {}".format(args.data_dir))
        rows = scan_dir(args.data_dir)
        if not rows:
            sys.exit("No .h5 files found in {}".format(args.data_dir))
        print("Scanned {} .h5 file(s) from {}.".format(
            len(rows), args.data_dir))

    import numpy as np
    rng = np.random.default_rng(args.seed)

    # Verify on disk and decode metadata.
    scenes = []
    missing = []
    for row in rows:
        fname = row["filename"]
        path  = os.path.join(args.data_dir, fname)
        if not os.path.exists(path):
            missing.append(fname)
            continue
        meta = parse_granule_name(row["granule_name"])
        scenes.append({
            "filename":     fname,
            "granule_name": row["granule_name"],
            "track":        meta["track"] or "",
            "cycle":        meta["cycle"] or "",
            "pol":          meta["pol"] or "",
        })

    if missing:
        print("Warning: {} file(s) not found on disk (skipped):"
              .format(len(missing)))
        for m in missing[:5]:
            print("  ", m)
        if len(missing) > 5:
            print("  ... and {} more".format(len(missing) - 5))

    if len(scenes) < 6:
        sys.exit("Need at least 6 on-disk scenes for a 3-way split. "
                 "Found {}.".format(len(scenes)))

    train, val, test = stratified_split(scenes, fracs, rng)

    for split_list, name in ((train, "train"), (val, "val"), (test, "test")):
        if len(split_list) < 2:
            sys.exit("Split '{}' has only {} scene(s); need at least 2. "
                     "Increase --{} or add more scenes."
                     .format(name, len(split_list), name))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fieldnames = ["filename", "split", "granule_name", "track", "cycle", "pol"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for split_name, group in (("train", train), ("val", val),
                                   ("test",  test)):
            for s in group:
                w.writerow({k: s.get(k, "") for k in fieldnames}
                           | {"split": split_name})

    print("Scene split written to: {}".format(args.out))
    print("  train : {} scenes".format(len(train)))
    print("  val   : {} scenes".format(len(val)))
    print("  test  : {} scenes".format(len(test)))
    print()
    for split_name, group in (("train", train), ("val", val), ("test", test)):
        pols = collections.Counter(s["pol"] for s in group)
        pol_str = ", ".join("{} {}".format(v, k)
                            for k, v in sorted(pols.items()))
        print("  {} pols: {}".format(split_name, pol_str or "unknown"))


if __name__ == "__main__":
    main()
