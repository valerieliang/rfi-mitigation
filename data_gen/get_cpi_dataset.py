#!/usr/bin/env python3
"""
get_cpi_dataset.py

Divide full STAP images (N, K_full, M_full) into CPI blocks
(N * n_blocks, K_cpi, M_cpi).

CPI dimensions
--------------
K_cpi : range bins per CPI block  -- number of snapshots for the SCM estimate.
        Must satisfy K_cpi >= 2 * M_cpi for a non-degenerate SCM.
        Recommended: 4 * M_cpi or higher.

M_cpi : pulses per CPI             -- SCM dimension; yields M_cpi eigenvalues.
        Must equal the M stored in the source file (checked at runtime).

The overdetermination ratio K_cpi / M_cpi is printed for each file so you
can verify SCM quality before committing to a chop configuration.

Input HDF5 layout (produced by save_stap_dataset.py)
-----------------------------------------------------
/data/contaminated   (N, K_full, M)  complex64
/data/clean          (N, K_full, M)  complex64
/data/signal         (N, K_full, M)  complex64
/data/noise          (N, K_full, M)  complex64
/data/rfi            (N, K_full, M)  complex64
/labels/label        (N,)            bytes
/labels/rfi_style    (N,)            bytes
/labels/jnr_db       (N,)            int32
/meta/               group           attributes (K, M, ...)

Output HDF5 layout
------------------
Same structure; data shapes become (N * n_blocks, K_cpi, M_cpi).
/meta/ attributes copied from source; K and M updated to CPI values.
Added attributes: K_full, M_full, n_blocks, cpi_k, cpi_m, source_file.

Usage
-----
    # Single file, CPI = (64 range bins) x (16 pulses)
    python get_cpi_dataset.py \\
        --input cw_high_jnr.h5 --output cw_high_jnr_cpi.h5 \\
        --cpi-k 64 --cpi-m 16

    # All files in a directory
    python get_cpi_dataset.py \\
        --input-dir data/ --output-dir data_cpi/ \\
        --cpi-k 64 --cpi-m 16

Recommended K_cpi values for M=16
-----------------------------------
    K_cpi=32   2x overdetermined   minimum usable
    K_cpi=64   4x overdetermined   recommended
    K_cpi=128  8x overdetermined   use full image (no chopping)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


DATA_KEYS  = ("contaminated", "clean", "signal", "noise", "rfi")
LABEL_KEYS = ("label", "rfi_style", "jnr_db")

MIN_OVERDETERMINATION = 2   # K_cpi / M_cpi must be >= this


# ---------------------------------------------------------------------------
# Core reshape helpers
# ---------------------------------------------------------------------------

def chop_matrix(arr: np.ndarray, k_cpi: int) -> np.ndarray:
    """
    Reshape (N, K_full, M) into (N * n_blocks, K_cpi, M).

    Trailing range bins that do not fill a complete block are discarded.
    """
    n, k_full, m = arr.shape
    n_blocks = k_full // k_cpi
    usable_k = n_blocks * k_cpi
    arr = arr[:, :usable_k, :]
    arr = arr.reshape(n, n_blocks, k_cpi, m)
    arr = arr.reshape(n * n_blocks, k_cpi, m)
    return arr


def chop_labels(arr: np.ndarray, n_blocks: int) -> np.ndarray:
    """
    Repeat each label n_blocks times to match expanded data shape.
    Works for numeric and object (bytes) arrays.
    """
    return np.repeat(arr, n_blocks, axis=0)


# ---------------------------------------------------------------------------
# Single-file converter
# ---------------------------------------------------------------------------

def convert_file(
    input_path: str | Path,
    output_path: str | Path,
    cpi_k: int,
    cpi_m: int,
) -> None:
    """
    Chop one HDF5 file into CPI blocks of shape (cpi_k, cpi_m).

    Parameters
    ----------
    input_path  : Source .h5 file from save_stap_dataset.py.
    output_path : Destination .h5 file.
    cpi_k       : Range bins per CPI block (snapshot count for SCM).
    cpi_m       : Pulses per CPI (SCM dimension). Must match file's M.
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

        n, k_full, m_full = fin["data/contaminated"].shape

        # Validate M
        file_m = int(fin["meta"].attrs.get("M", m_full))
        if cpi_m != file_m:
            raise ValueError(
                f"--cpi-m {cpi_m} does not match M={file_m} stored in "
                f"{input_path.name}. They must be equal."
            )
        if m_full != file_m:
            raise ValueError(
                f"Data M dimension {m_full} disagrees with meta M={file_m}."
            )

        # Validate K
        if cpi_k < MIN_OVERDETERMINATION * cpi_m:
            raise ValueError(
                f"--cpi-k {cpi_k} < {MIN_OVERDETERMINATION} * M={cpi_m} "
                f"({MIN_OVERDETERMINATION * cpi_m}). "
                f"SCM would be rank-deficient. "
                f"Use at least --cpi-k {MIN_OVERDETERMINATION * cpi_m}."
            )
        if cpi_k > k_full:
            raise ValueError(
                f"--cpi-k {cpi_k} > K_full={k_full}: cannot form even one block."
            )

        n_blocks = k_full // cpi_k
        n_out    = n * n_blocks
        ratio    = cpi_k / cpi_m

        print(f"  CPI dims        : K_cpi={cpi_k}  M_cpi={cpi_m}  "
              f"(overdetermination = {ratio:.1f}x)")
        print(f"  shape in        : ({n}, {k_full}, {m_full})")
        print(f"  n_blocks/image  : {n_blocks}")
        print(f"  shape out       : ({n_out}, {cpi_k}, {cpi_m})")

        with h5py.File(output_path, "w") as fout:

            # Data
            grp_data = fout.create_group("data")
            for key in DATA_KEYS:
                raw     = fin[f"data/{key}"][:]
                chopped = chop_matrix(raw, cpi_k)
                grp_data.create_dataset(
                    key, data=chopped,
                    compression="gzip", compression_opts=4,
                )

            # Labels
            grp_lbl = fout.create_group("labels")
            for key in LABEL_KEYS:
                raw     = fin[f"labels/{key}"][:]
                chopped = chop_labels(raw, n_blocks)
                grp_lbl.create_dataset(
                    key, data=chopped, compression="gzip",
                )

            # Meta
            grp_meta = fout.create_group("meta")
            if "meta" in fin:
                for k, v in fin["meta"].attrs.items():
                    grp_meta.attrs[k] = v
            # Override / add CPI-specific fields
            grp_meta.attrs["K"]           = cpi_k
            grp_meta.attrs["M"]           = cpi_m
            grp_meta.attrs["K_full"]      = k_full
            grp_meta.attrs["M_full"]      = m_full
            grp_meta.attrs["n_blocks"]    = n_blocks
            grp_meta.attrs["cpi_k"]       = cpi_k
            grp_meta.attrs["cpi_m"]       = cpi_m
            grp_meta.attrs["source_file"] = input_path.name

    # Read-back summary
    with h5py.File(output_path, "r") as fout:
        n_written = fout["data/contaminated"].shape[0]
        labels    = fout["labels/label"][:]
        n_cont    = int(np.sum(labels == b"contaminated"))
        n_clean   = int(np.sum(labels == b"clean"))

    print(f"  written         : {n_written} CPI blocks  "
          f"({n_cont} contaminated, {n_clean} clean)")
    print(f"  saved to        : {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Divide STAP images into CPI blocks by (K_cpi, M_cpi) dimensions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--input", type=str,
        help="Single input .h5 file.",
    )
    mode.add_argument(
        "--input-dir", type=str,
        help="Directory of .h5 files (all processed).",
    )

    p.add_argument(
        "--output", type=str, default=None,
        help="Output .h5 path (single-file mode only).",
    )
    p.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory.",
    )
    p.add_argument(
        "--cpi-k", type=int, default=64,
        help=(
            "Range bins per CPI block (snapshot count for the SCM estimate). "
            "Must be >= 2 * cpi-m. Recommended: 4 * cpi-m."
        ),
    )
    p.add_argument(
        "--cpi-m", type=int, default=16,
        help=(
            "Pulses per CPI (SCM dimension; number of eigenvalues). "
            "Must match M stored in the source file."
        ),
    )
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

        convert_file(input_path, output_path, args.cpi_k, args.cpi_m)

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
            convert_file(f, out_dir / f"{f.stem}_cpi.h5", args.cpi_k, args.cpi_m)

    print("\nDone.")


if __name__ == "__main__":
    main()