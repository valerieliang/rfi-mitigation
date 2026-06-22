"""
gen_stap_dataset.py
-------------------
Generate synthetic STAP CPI blocks and write them to an HDF5 file.

Imports from clean_data.py (primitives + dataset iterator) and
rfi_gen_stap.py (RFI field generators). No other local dependencies.

NISAR ST-EVD convention
-----------------------
M : pulses per CPI  (rows)    -- SCM dimension; yields M eigenvalues
K : range bins      (columns) -- snapshot count; must be >= 2*M

CPI block S has shape (M, K).
SCM: R = S @ S.conj().T / K,  shape (M, M)

HDF5 layout
-----------
/data/
    contaminated   (N, M, K)  complex64  -- signal + noise + rfi
    clean          (N, M, K)  complex64  -- signal + noise
    signal         (N, M, K)  complex64
    noise          (N, M, K)  complex64
    rfi            (N, M, K)  complex64  -- zeros for clean samples

/labels/
    label          (N,)  bytes   -- b"clean" or b"contaminated"
    rfi_style      (N,)  bytes   -- b"cw_tone", b"wideband", or b""
    jnr_db         (N,)  int32   -- JNR in dB; -1 sentinel for clean

/meta/                           -- scalar attributes
    n_samples, M, K
    noise_db, signal_db
    jnr_min_db, jnr_max_db
    contaminated_fraction, base_seed
    rfi_style  ("cw_tone", "wideband", or "mixed")

Usage
-----
    python gen_stap_dataset.py
    python gen_stap_dataset.py --style cw_tone  --jnr-min 20 --jnr-max 30 --out cw_high.h5
    python gen_stap_dataset.py --style cw_tone  --jnr-min  6 --jnr-max 10 --out cw_low.h5
    python gen_stap_dataset.py --style wideband --jnr-min 20 --jnr-max 30 --out wb_high.h5
    python gen_stap_dataset.py --style wideband --jnr-min  6 --jnr-max 10 --out wb_low.h5

Reading back
------------
    import h5py, numpy as np
    with h5py.File("cw_high.h5", "r") as f:
        S      = f["data/contaminated"][:]  # (N, M, K) rows=pulses, cols=range
        labels = f["labels/label"][:]
        jnr    = f["labels/jnr_db"][:]      # -1 = clean
        M      = f["meta"].attrs["M"]
        K      = f["meta"].attrs["K"]
        # SCM for sample i: S[i] @ S[i].conj().T / K
"""

from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np

from clean_data import iter_dataset


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------

def save_dataset(
    out_path: str,
    n_samples: int = 1000,
    M: int = 16,
    K: int = 128,
    noise_db: float = 3.0,
    signal_db: float = 9.0,
    jnr_min_db: int = 10,
    jnr_max_db: int = 30,
    contaminated_fraction: float = 0.5,
    base_seed: int = 42,
    chunk_size: int = 64,
    rfi_style: str | None = None,
    verbose: bool = True,
) -> None:
    """
    Generate STAP CPI blocks and write them to an HDF5 file.

    Samples are written in batches of chunk_size to avoid holding the
    full dataset in memory at once.

    Parameters
    ----------
    out_path              : Output .h5 file path.
    n_samples             : Number of samples.
    M                     : Pulses per CPI (rows). SCM is M x M.
    K                     : Range bins per CPI (columns). Must be >= 2*M.
    noise_db              : Noise power in dB.
    signal_db             : Signal power in dB.
    jnr_min_db            : Lower bound of integer JNR draw (inclusive).
    jnr_max_db            : Upper bound of integer JNR draw (inclusive).
    contaminated_fraction : Fraction of samples with RFI.
    base_seed             : Root RNG seed.
    chunk_size            : Samples per HDF5 chunk and write batch.
    rfi_style             : "cw_tone", "wideband", or None (random mix).
    verbose               : Print progress.
    """
    if K < 2 * M:
        raise ValueError(f"K={K} must be >= 2*M={2*M}.")

    mat_chunks = (chunk_size, M, K)
    style_tag  = rfi_style if rfi_style is not None else "mixed"

    t0 = time.time()
    if verbose:
        print(f"Writing {n_samples} samples to {out_path}")
        print(f"  CPI shape : (M={M} pulses, K={K} range bins)")
        print(f"  SCM       : R = S @ S.conj().T / K,  shape ({M}, {M})")
        print(f"  noise={noise_db} dB  signal={signal_db} dB")
        print(f"  JNR [{jnr_min_db}, {jnr_max_db}] dB  "
              f"style={style_tag}  frac={contaminated_fraction}  seed={base_seed}")
        print()

    with h5py.File(out_path, "w") as f:

        dt_cx  = np.dtype("complex64")
        dt_str = h5py.special_dtype(vlen=bytes)

        grp_data = f.create_group("data")
        for name in ("contaminated", "clean", "signal", "noise", "rfi"):
            grp_data.create_dataset(
                name,
                shape=(n_samples, M, K),
                dtype=dt_cx,
                chunks=mat_chunks,
                compression="gzip",
                compression_opts=4,
            )

        grp_lbl = f.create_group("labels")
        grp_lbl.create_dataset("label",     shape=(n_samples,), dtype=dt_str)
        grp_lbl.create_dataset("rfi_style", shape=(n_samples,), dtype=dt_str)
        grp_lbl.create_dataset("jnr_db",    shape=(n_samples,), dtype=np.int32)

        grp_meta = f.create_group("meta")
        for attr, val in {
            "n_samples":             n_samples,
            "M":                     M,
            "K":                     K,
            "noise_db":              noise_db,
            "signal_db":             signal_db,
            "jnr_min_db":            jnr_min_db,
            "jnr_max_db":            jnr_max_db,
            "contaminated_fraction": contaminated_fraction,
            "base_seed":             base_seed,
            "rfi_style":             style_tag,
        }.items():
            grp_meta.attrs[attr] = val

        gen = iter_dataset(
            n_samples             = n_samples,
            M                     = M,
            K                     = K,
            noise_db              = noise_db,
            signal_db             = signal_db,
            jnr_min_db            = jnr_min_db,
            jnr_max_db            = jnr_max_db,
            contaminated_fraction = contaminated_fraction,
            base_seed             = base_seed,
            rfi_style             = rfi_style,
        )

        buf: dict[str, list] = {
            "contaminated": [], "clean": [], "signal": [], "noise": [], "rfi": [],
            "label": [], "rfi_style": [], "jnr_db": [],
        }

        def _flush(write_start: int) -> None:
            n  = len(buf["label"])
            sl = slice(write_start, write_start + n)
            for name in ("contaminated", "clean", "signal", "noise", "rfi"):
                grp_data[name][sl] = np.stack(buf[name]).astype(dt_cx)
            grp_lbl["label"][sl]     = np.array(buf["label"],     dtype=object)
            grp_lbl["rfi_style"][sl] = np.array(buf["rfi_style"], dtype=object)
            grp_lbl["jnr_db"][sl]    = np.array(buf["jnr_db"],    dtype=np.int32)
            for lst in buf.values():
                lst.clear()

        written = 0
        for sample in gen:
            buf["contaminated"].append(sample.contaminated)
            buf["clean"].append(sample.clean)
            buf["signal"].append(sample.signal)
            buf["noise"].append(sample.noise)
            buf["rfi"].append(sample.rfi)
            buf["label"].append(sample.label.encode())
            buf["rfi_style"].append((sample.rfi_style or "").encode())
            buf["jnr_db"].append(
                sample.jnr_db if sample.jnr_db is not None else -1
            )

            if len(buf["label"]) == chunk_size:
                _flush(written)
                written += chunk_size
                if verbose:
                    elapsed = time.time() - t0
                    pct     = 100.0 * written / n_samples
                    print(f"  [{written:>{len(str(n_samples))}}/{n_samples}]"
                          f"  {pct:5.1f}%  {elapsed:.1f}s elapsed")

        if buf["label"]:
            remaining = len(buf["label"])
            _flush(written)
            written += remaining

    elapsed = time.time() - t0
    size_mb = os.path.getsize(out_path) / 1024 ** 2
    if verbose:
        print()
        print(f"Done.  {written} samples  {elapsed:.1f}s  "
              f"{size_mb:.1f} MB  ->  {out_path}")


# ---------------------------------------------------------------------------
# Read-back verification
# ---------------------------------------------------------------------------

def verify_h5(path: str) -> None:
    """Print a summary of HDF5 file contents."""
    with h5py.File(path, "r") as f:
        labels     = f["labels/label"][:]
        jnr        = f["labels/jnr_db"][:]
        styles_raw = f["labels/rfi_style"][:]
        shape      = tuple(f["data/contaminated"].shape)

        n_cont  = int(np.sum(labels == b"contaminated"))
        n_clean = int(np.sum(labels == b"clean"))
        styles, counts = np.unique(
            styles_raw[styles_raw != b""], return_counts=True
        )
        jnr_valid = jnr[jnr >= 0]

        print(f"\nVerification: {path}")
        print(f"  shape (N, M, K) : {shape}")
        print(f"  contaminated    : {n_cont}   clean: {n_clean}")
        print(f"  RFI styles      : " + (
            ", ".join(f"{s.decode()} ({c})" for s, c in zip(styles, counts))
            if len(styles) else "none"
        ))
        if len(jnr_valid):
            print(f"  JNR range       : {jnr_valid.min()} .. "
                  f"{jnr_valid.max()} dB  (mean {jnr_valid.mean():.1f} dB)")
        print(f"  meta:")
        for k, v in sorted(f["meta"].attrs.items()):
            print(f"    {k:<25} = {v}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic STAP CPI blocks and save to HDF5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n",       type=int,   default=1000,
                   help="Number of samples")
    p.add_argument("--M",       type=int,   default=16,
                   help="Pulses per CPI (rows, SCM dimension)")
    p.add_argument("--K",       type=int,   default=128,
                   help="Range bins per CPI (columns, snapshot count)")
    p.add_argument("--noise",   type=float, default=3.0,
                   help="Noise power (dB)")
    p.add_argument("--signal",  type=float, default=9.0,
                   help="Signal power (dB)")
    p.add_argument("--jnr-min", type=int,   default=10,
                   help="JNR lower bound (dB, inclusive)")
    p.add_argument("--jnr-max", type=int,   default=30,
                   help="JNR upper bound (dB, inclusive)")
    p.add_argument("--frac",    type=float, default=0.5,
                   help="Contaminated fraction [0, 1]")
    p.add_argument("--seed",    type=int,   default=42,
                   help="Base RNG seed")
    p.add_argument("--chunk",   type=int,   default=64,
                   help="Write chunk size")
    p.add_argument("--style",   type=str,   default=None,
                   choices=["cw_tone", "wideband"],
                   help="Pin RFI style. Omit for random mix.")
    p.add_argument("--out",     type=str,   default="stap_dataset.h5",
                   help="Output HDF5 file path")
    p.add_argument("--verify",  action="store_true",
                   help="Run read-back check after writing")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()

    save_dataset(
        out_path              = args.out,
        n_samples             = args.n,
        M                     = args.M,
        K                     = args.K,
        noise_db              = args.noise,
        signal_db             = args.signal,
        jnr_min_db            = args.jnr_min,
        jnr_max_db            = args.jnr_max,
        contaminated_fraction = args.frac,
        base_seed             = args.seed,
        chunk_size            = args.chunk,
        rfi_style             = args.style,
    )

    if args.verify:
        verify_h5(args.out)
