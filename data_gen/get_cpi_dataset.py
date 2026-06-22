#!/usr/bin/env python3
"""
get_cpi_dataset.py

Sub-divide synthetic STAP CPI blocks along the range axis.

NISAR ST-EVD convention
-----------------------
M : pulses per CPI  (rows)    -- SCM dimension; fixed at generation time
K : range bins      (columns) -- snapshot count; may be subdivided here

Each sample in the source file has shape (M, K_full).
This script splits K_full columns into n_blocks = K_full // K_cpi non-
overlapping sub-blocks, producing samples of shape (M, K_cpi).

The SCM for each output block is R = S @ S.conj().T / K_cpi, shape (M, M).

Constraint: K_cpi >= 2 * M (non-degenerate SCM). Recommended: K_cpi >= 4*M.

When to use this script
-----------------------
Use it only if you want more training samples at reduced snapshot count.
With the default K_full=128, M=16 (8x overdetermined), chopping to
K_cpi=64 gives 2 blocks per sample at 4x overdetermination -- still good.
Chopping to K_cpi=32 gives 4 blocks at 2x -- borderline.

If your dataset is large enough, skip this and train on the full (M, K_full)
blocks directly.

Input HDF5 layout (produced by save_stap_dataset.py)
-----------------------------------------------------
/data/contaminated   (N, M, K_full)  complex64
/data/clean          (N, M, K_full)  complex64
/data/signal         (N, M, K_full)  complex64
/data/noise          (N, M, K_full)  complex64
/data/rfi            (N, M, K_full)  complex64
/labels/label        (N,)            bytes
/labels/rfi_style    (N,)            bytes
/labels/jnr_db       (N,)            int32
/meta/               group           attributes (M, K, ...)

Output HDF5 layout
------------------
Same structure; shapes become (N * n_blocks, M, K_cpi).
/meta/ attributes copied; K updated to K_cpi.
Added: K_full, n_blocks, source_file.

Usage
-----
    # Single file -- K_cpi=64 (4x overdetermined for M=16)
    python get_cpi_dataset.py --input cw_high_jnr.h5 --output cw_high_jnr_cpi.h5 --K-cpi 64

    # All files in a directory
    python get_cpi_dataset.py --input-dir data/full_stap --output-dir data/per_cpi --K-cpi 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


DATA_KEYS  = ("contaminated", "clean", "signal", "noise", "rfi")
LABEL_KEYS = ("label", "rfi_style", "jnr_db")

MIN_OVERDETERMINATION = 2


# ---------------------------------------------------------------------------
# Core reshape helpers
# ---------------------------------------------------------------------------

def chop_matrix(arr: np.ndarray, k_cpi: int) -> np.ndarray:
    """
    Reshape (N, M, K_full) into (N * n_blocks, M, K_cpi).

    The chop is along the range (column) axis, axis=2.
    Trailing columns that do not fill a complete block are discarded.
    """
    n, m, k_full = arr.shape
    n_blocks = k_full // k_cpi
    usable_k = n_blocks * k_cpi

    arr = arr[:, :, :usable_k]               # (N, M, n_blocks*k_cpi)
    arr = arr.reshape(n, m, n_blocks, k_cpi) # (N, M, n_blocks, k_cpi)
    arr = arr.transpose(0, 2, 1, 3)          # (N, n_blocks, M, k_cpi)
    arr = arr.reshape(n * n_blocks, m, k_cpi)
    return arr


def chop_labels(arr: np.ndarray, n_blocks: int) -> np.ndarray:
    """Repeat each label n_blocks times to match expanded data shape."""
    return np.repeat(arr, n_blocks, axis=0)


# ---------------------------------------------------------------------------
# Single-file converter
# ---------------------------------------------------------------------------

def convert_file(
    input_path: str | Path,
    output_path: str | Path,
    k_cpi: int,
) -> None:
    """
    Chop one HDF5 file into sub-blocks of K_cpi range bins each.

    Parameters
    ----------
    input_path  : Source .h5 file from save_stap_dataset.py.
    output_path : Destination .h5 file.
    k_cpi       : Range bins per output block (snapshot count for SCM).
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)

    print(f"\n{input_path.name}")

    with h5py.File(input_path, "r") as fin:
        for grp in ("data", "labels"):
            if grp not in fin:
                raise KeyError(
                    f"Group '{grp}' not found in {input_path.name}. "
                    f"File must be produced by save_stap_dataset.py."
                )

        n, m, k_full = fin["data/contaminated"].shape
        file_m = int(fin["meta"].attrs.get("M", m))

        if m != file_m:
            raise ValueError(
                f"Data M dimension {m} disagrees with meta M={file_m}."
            )
        if k_cpi < MIN_OVERDETERMINATION * m:
            raise ValueError(
                f"K_cpi={k_cpi} < {MIN_OVERDETERMINATION}*M={m} "
                f"({MIN_OVERDETERMINATION * m}). "
                f"SCM would be rank-deficient. "
                f"Use at least --K-cpi {MIN_OVERDETERMINATION * m}."
            )
        if k_cpi > k_full:
            raise ValueError(
                f"K_cpi={k_cpi} > K_full={k_full}: cannot form even one block."
            )

        n_blocks = k_full // k_cpi
        n_out    = n * n_blocks
        ratio    = k_cpi / m

        print(f"  M (pulses, rows)    : {m}")
        print(f"  K_full (range cols) : {k_full}")
        print(f"  K_cpi (output cols) : {k_cpi}  "
              f"(overdetermination = {ratio:.1f}x)")
        print(f"  n_blocks per sample : {n_blocks}")
        print(f"  shape in            : ({n}, {m}, {k_full})")
        print(f"  shape out           : ({n_out}, {m}, {k_cpi})")
        print(f"  SCM formula         : R = S @ S.conj().T / {k_cpi}")

        with h5py.File(output_path, "w") as fout:

            grp_data = fout.create_group("data")
            for key in DATA_KEYS:
                raw     = fin[f"data/{key}"][:]
                chopped = chop_matrix(raw, k_cpi)
                grp_data.create_dataset(
                    key, data=chopped,
                    compression="gzip", compression_opts=4,
                )

            grp_lbl = fout.create_group("labels")
            for key in LABEL_KEYS:
                raw     = fin[f"labels/{key}"][:]
                chopped = chop_labels(raw, n_blocks)
                grp_lbl.create_dataset(
                    key, data=chopped, compression="gzip",
                )

            grp_meta = fout.create_group("meta")
            if "meta" in fin:
                for k, v in fin["meta"].attrs.items():
                    grp_meta.attrs[k] = v
            grp_meta.attrs["K"]           = k_cpi
            grp_meta.attrs["K_full"]      = k_full
            grp_meta.attrs["n_blocks"]    = n_blocks
            grp_meta.attrs["source_file"] = input_path.name

    with h5py.File(output_path, "r") as fout:
        n_written = fout["data/contaminated"].shape[0]
        labels    = fout["labels/label"][:]
        n_cont    = int(np.sum(labels == b"contaminated"))
        n_clean   = int(np.sum(labels == b"clean"))

    print(f"  written             : {n_written} blocks  "
          f"({n_cont} contaminated, {n_clean} clean)")
    print(f"  saved to            : {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Sub-divide STAP CPI blocks along the range axis. "
            "Chops (N, M, K_full) -> (N*n_blocks, M, K_cpi)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input",     type=str, help="Single input .h5 file.")
    mode.add_argument("--input-dir", type=str, help="Directory of .h5 files.")

    p.add_argument("--output",     type=str, default=None,
                   help="Output .h5 path (single-file mode only).")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory.")
    p.add_argument("--K-cpi",      type=int, default=64,
                   help=(
                       "Range bins per output block (snapshot count for SCM). "
                       "Must be >= 2*M. Recommended: 4*M (=64 for M=16)."
                   ))
    return p.parse_args()


def main() -> None:
    args = _parse()

    if args.input:
        input_path = Path(args.input)
        if args.output:
            output_path = Path(args.output)
        elif args.output_dir:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / f"{input_path.stem}_cpi.h5"
        else:
            output_path = input_path.parent / f"{input_path.stem}_cpi.h5"

        convert_file(input_path, output_path, args.K_cpi)

    else:
        input_dir = Path(args.input_dir)
        out_dir   = (Path(args.output_dir) if args.output_dir
                     else input_dir.parent / (input_dir.name + "_cpi"))
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(input_dir.glob("*.h5"))
        if not files:
            raise RuntimeError(f"No .h5 files found in {input_dir}")

        print(f"Found {len(files)} file(s) in {input_dir}")
        for f in files:
            convert_file(f, out_dir / f"{f.stem}_cpi.h5", args.K_cpi)

    print("\nDone.")


if __name__ == "__main__":
    main()
