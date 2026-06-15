"""
gen_raw_l0.py  --  Synthetic L0 raw data generator for RFI pipeline testing.

Produces HDF5 files that mimic the NISAR L0B raw data layout:

    /science/LSAR/L0B/swaths/frequencyA/HH   [M_total x N]  complex64
    /science/LSAR/L0B/swaths/frequencyA/VV   [M_total x N]  complex64  (optional)

where:
    M_total = n_cpis * cpi_size          (total pulses)
    N       = n_range_samples            (range samples per pulse)

When --n-files > 1, the output argument is treated as a directory and
files are named synthetic_l0_0000.h5, synthetic_l0_0001.h5, etc.
Each file gets a unique seed derived from --seed + file index so the
noise is independent across files.

Signal model
------------
Each range sample is drawn from CN(0, sigma_s^2).  This is white Gaussian
noise standing in for real thermal noise + clutter; no SAR chirp modulation
is applied because the RFI injector operates on the raw pulse domain, not
on range-compressed data.

The INR regime is controlled by --signal-power-db.  A typical value for
NISAR L-band is -20 dB to -30 dB (relative to the RFI power normalised to
0 dB), so RFI will be ~20-30 dB above the noise floor.

Usage
-----
    # Single file
    python gen_raw_l0.py --out data/l0_out/synthetic_l0.h5 \\
        --n-cpis 50 --cpi-size 32 --n-range 256 \\
        --signal-power-db -20 --pols HH --seed 0

    # 500 files into a directory
    python gen_raw_l0.py --out data/l0_out/ --n-files 500 \\
        --n-cpis 50 --cpi-size 32 --n-range 256 \\
        --signal-power-db -20 --pols HH --seed 0
"""

import argparse
import os
import sys

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# HDF5 path constants (mirrors NISAR L0B convention)
# ---------------------------------------------------------------------------
_L0B_ROOT = "science/LSAR/L0B/swaths/frequencyA"


def _make_group(f: h5py.File, path: str) -> h5py.Group:
    """Create nested groups, return the leaf."""
    return f.require_group(path)


def generate_white_noise_raw(
    n_cpis: int,
    cpi_size: int,
    n_range: int,
    signal_power_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Return a (n_cpis * cpi_size, n_range) complex64 array of circularly
    symmetric white Gaussian noise with the specified power.

    Parameters
    ----------
    n_cpis : int
        Number of CPIs in the scene.
    cpi_size : int
        Pulses per CPI (M).
    n_range : int
        Range samples per pulse (N / K in pipeline notation).
    signal_power_db : float
        Per-sample signal power in dB (relative to 1.0).
    rng : np.random.Generator
        Seeded random number generator for reproducibility.
    """
    M_total = n_cpis * cpi_size
    sigma = 10.0 ** (signal_power_db / 20.0)   # amplitude sigma
    # Complex circular Gaussian: real + imag each ~ N(0, sigma/sqrt(2))
    s = (sigma / np.sqrt(2.0)) * (
        rng.standard_normal((M_total, n_range)).astype(np.float32)
        + 1j * rng.standard_normal((M_total, n_range)).astype(np.float32)
    )
    return s.astype(np.complex64)


def write_l0b(
    out_path: str,
    pols: list,
    n_cpis: int,
    cpi_size: int,
    n_range: int,
    signal_power_db: float,
    seed: int,
) -> None:
    """
    Generate and write one synthetic L0B HDF5 file.

    Parameters
    ----------
    out_path : str
        Output HDF5 file path.
    pols : list of str
        Polarisation channels to generate, e.g. ['HH', 'VV'].
    n_cpis : int
        Number of CPIs.
    cpi_size : int
        Pulses per CPI (M).
    n_range : int
        Range samples per pulse.
    signal_power_db : float
        Signal power in dB relative to 1.0.
    seed : int
        Random seed.
    """
    rng = np.random.default_rng(seed)
    M_total = n_cpis * cpi_size

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    print(f"Generating synthetic L0B: {M_total} pulses x {n_range} samples "
          f"({n_cpis} CPIs of size {cpi_size}), "
          f"signal power = {signal_power_db:.1f} dB  seed={seed}")

    with h5py.File(out_path, "w") as f:
        grp = _make_group(f, _L0B_ROOT)

        # Store acquisition metadata as attributes
        grp.attrs["cpi_size"]        = cpi_size
        grp.attrs["n_cpis"]          = n_cpis
        grp.attrs["n_range_samples"] = n_range
        grp.attrs["signal_power_db"] = signal_power_db
        grp.attrs["synthetic"]       = True
        grp.attrs["seed"]            = seed

        for pol in pols:
            raw = generate_white_noise_raw(
                n_cpis, cpi_size, n_range, signal_power_db, rng
            )
            ds = grp.create_dataset(
                pol, data=raw,
                chunks=(cpi_size, n_range),
                compression="gzip", compression_opts=4,
            )
            ds.attrs["polarisation"]     = pol
            ds.attrs["signal_power_db"]  = signal_power_db
            print(f"  Written /{pol}: shape={raw.shape}, dtype={raw.dtype}")

    print(f"Saved -> {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Generate synthetic L0 raw data (white noise) for RFI pipeline "
            "testing. When --n-files > 1, --out is treated as a directory and "
            "files are named synthetic_l0_NNNN.h5 with independent seeds."
        )
    )
    p.add_argument("--out", default="synthetic_l0.h5",
                   help="Output path: a .h5 file when --n-files 1 (default), "
                        "or a directory when --n-files > 1.")
    p.add_argument("--n-files", type=int, default=1,
                   help="Number of independent scene files to generate (default: 1).")
    p.add_argument("--n-cpis", type=int, default=50,
                   help="Number of CPIs per file (default: 50).")
    p.add_argument("--cpi-size", type=int, default=32,
                   help="Pulses per CPI / M (default: 32).")
    p.add_argument("--n-range", type=int, default=256,
                   help="Range samples per pulse / K (default: 256).")
    p.add_argument("--signal-power-db", type=float, default=-20.0,
                   help="Per-sample signal power in dB rel. to 1.0 (default: -20).")
    p.add_argument("--pols", nargs="+", default=["HH"],
                   choices=["HH", "VV", "HV", "VH"],
                   help="Polarisation channels to generate (default: HH).")
    p.add_argument("--seed", type=int, default=0,
                   help="Base random seed. File i uses seed + i (default: 0).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.n_files == 1:
        # Single-file mode: --out is the full file path.
        write_l0b(
            out_path=args.out,
            pols=args.pols,
            n_cpis=args.n_cpis,
            cpi_size=args.cpi_size,
            n_range=args.n_range,
            signal_power_db=args.signal_power_db,
            seed=args.seed,
        )
    else:
        # Multi-file mode: --out is a directory.
        out_dir = args.out
        os.makedirs(out_dir, exist_ok=True)
        print(f"Generating {args.n_files} files into {out_dir}/")
        for i in range(args.n_files):
            fname = f"synthetic_l0_{i:04d}.h5"
            out_path = os.path.join(out_dir, fname)
            write_l0b(
                out_path=out_path,
                pols=args.pols,
                n_cpis=args.n_cpis,
                cpi_size=args.cpi_size,
                n_range=args.n_range,
                signal_power_db=args.signal_power_db,
                seed=args.seed + i,
            )
            print(f"  [{i+1}/{args.n_files}] {fname}")
        print(f"Done. {args.n_files} files written to {out_dir}/")


if __name__ == "__main__":
    main()