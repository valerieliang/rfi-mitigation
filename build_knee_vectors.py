#!/usr/bin/env python3
"""
build_knee_vectors.py

Translates per-CPI knee-index predictions (as produced by the anomaly /
knee classifier pipeline) into fixed-length binary occupancy vectors and
stores them as matrices in .npz files.

Label convention (matches train_db.py):
    knee == 0            -> clean CPI, vector is all zeros
    knee == k (1 <= k<=16)-> RFI occupies eigenvalue indices 0..k-1,
                             vector has the first k entries set to 1 and
                             the remainder set to 0

Two outputs are produced:
    1. Full vectors    : one 1 in the vector for every unit of knee index,
                         up to cpi_len entries (no capping other than the
                         natural cpi_len ceiling).
    2. Bounded vectors  : same as above, but the number of 1s is capped at
                         --max-bands. This is useful when a downstream
                         consumer only wants to consider up to N RFI bands
                         regardless of what the model predicted.

Usage:
    python build_knee_vectors.py \
        --input predictions_A_HH.h5 \
        --output-dir out \
        --cpi-len 16 \
        --max-bands 4
"""

import argparse
import os

import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Convert knee predictions to binary vectors")
    parser.add_argument("--input", required=True, help="Path to predictions .h5 file")
    parser.add_argument("--output-dir", default=".", help="Directory to write .npz outputs")
    parser.add_argument("--cpi-len", type=int, default=16, help="Length of each CPI vector (pulses per CPI, fixed at 16 for this pipeline)")
    parser.add_argument("--max-bands", type=int, default=4, help="Cap on number of RFI bands (1s) for the bounded vector matrix")
    parser.add_argument("--prefix", default=None, help="Optional filename prefix; defaults to the input file stem")
    return parser.parse_args()


def load_predictions(h5_path):
    """Load knee predictions and useful metadata from the predictions h5 file."""
    with h5py.File(h5_path, "r") as f:
        knee = f["knee"][:].astype(np.int64)
        tile_pulse = f["tile_pulse"][:]
        tile_range = f["tile_range"][:]
        confidence = f["confidence"][:] if "confidence" in f else None
        attrs = dict(f.attrs)
    return knee, tile_pulse, tile_range, confidence, attrs


def knee_to_vectors(knee, cpi_len, max_bands=None):
    """
    Convert an array of knee indices into an (N, cpi_len) binary matrix.

    knee[i] == 0            -> row of zeros
    knee[i] == k (k >= 1)    -> first min(k, cap) entries are 1, rest are 0
        where cap = cpi_len if max_bands is None else min(max_bands, cpi_len)
    """
    n = knee.shape[0]
    cap = cpi_len if max_bands is None else min(max_bands, cpi_len)

    vectors = np.zeros((n, cpi_len), dtype=np.uint8)

    # number of bands to set to 1 for each row, clipped at cap and cpi_len
    n_bands = np.clip(knee, 0, cap)

    # build vectors via broadcasting: column index < n_bands[i]
    col_idx = np.arange(cpi_len)[None, :]
    vectors = (col_idx < n_bands[:, None]).astype(np.uint8)

    return vectors


def main():
    args = parse_args()

    if args.prefix is None:
        stem = os.path.splitext(os.path.basename(args.input))[0]
        args.prefix = stem

    os.makedirs(args.output_dir, exist_ok=True)

    knee, tile_pulse, tile_range, confidence, attrs = load_predictions(args.input)

    print("Loaded {} predictions from {}".format(knee.shape[0], args.input))
    print("knee value range: min={} max={}".format(knee.min(), knee.max()))

    # 1. Full vectors: no cap beyond cpi_len itself
    full_vectors = knee_to_vectors(knee, args.cpi_len, max_bands=None)

    # 2. Bounded vectors: capped at --max-bands
    bounded_vectors = knee_to_vectors(knee, args.cpi_len, max_bands=args.max_bands)

    n_capped = int(np.sum(knee > args.max_bands))
    print("Rows where knee > max_bands ({}): {} of {} ({:.2f} percent)".format(
        args.max_bands, n_capped, knee.shape[0], 100.0 * n_capped / knee.shape[0]))

    full_path = os.path.join(args.output_dir, "{}_vectors_full.npz".format(args.prefix))
    bounded_path = os.path.join(args.output_dir, "{}_vectors_bounded_{}.npz".format(args.prefix, args.max_bands))

    np.savez_compressed(
        full_path,
        vectors=full_vectors,
        knee=knee.astype(np.int8),
        tile_pulse=tile_pulse,
        tile_range=tile_range,
        confidence=confidence if confidence is not None else np.array([]),
        cpi_len=args.cpi_len,
        max_bands=args.cpi_len,  # full vectors are only capped by cpi_len
        source_file=os.path.basename(args.input),
    )

    np.savez_compressed(
        bounded_path,
        vectors=bounded_vectors,
        knee=knee.astype(np.int8),
        tile_pulse=tile_pulse,
        tile_range=tile_range,
        confidence=confidence if confidence is not None else np.array([]),
        cpi_len=args.cpi_len,
        max_bands=args.max_bands,
        source_file=os.path.basename(args.input),
    )

    print("Wrote full vectors    : {} shape={}".format(full_path, full_vectors.shape))
    print("Wrote bounded vectors : {} shape={}".format(bounded_path, bounded_vectors.shape))


if __name__ == "__main__":
    main()
