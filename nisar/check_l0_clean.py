#!/usr/bin/env python3
"""
check_l0_clean.py

Screen NISAR L0B (raw) granules for RFI contamination and data-quality
problems, producing the same QC CSV that check_nisar_clean.py produces
for RSLC files.

Why a separate screener for L0?
--------------------------------
RSLC screening (check_nisar_clean.py) looks for RFI artifacts that
survive focusing: range-spectrum spikes and azimuth-brightness stripes.
At the raw / L0B stage the same RFI shows up differently:

  - A CW tone appears as a coherent pulse-to-pulse modulation: the
    slow-time eigenvalue profile of any CPI extracted from an affected
    range bin will have a dominant eigenvalue well above the others.
  - Wideband interference raises the overall noise floor and creates
    multiple elevated eigenvalues.
  - Transient (pulsed) RFI causes one or a few pulses in a CPI to have
    anomalously high energy.

This screener probes a set of random CPI blocks from each L0B file and
computes:

  eigenvalue_max_db   -- max top-eigenvalue height across sampled CPIs
  f_factor_max        -- max F = sigma_max/sigma_min across sampled TBs
  pulse_energy_mad    -- MAD robustness score for per-pulse energy
  zero_frac           -- fraction of raw samples that are exactly zero

Thresholds below are heuristic starting points. As with the RSLC
screener, treat the flag as triage; confirm borderline files by
inspecting their eigenvalue profiles directly.

Output
------
A CSV file (same columns as nisar_qc_summary.csv) whose 'flag' column
is CLEAN / REVIEW / NOISY and which can be passed to augment.py via
--qc-csv.

Usage
-----
    python check_l0_clean.py data/l0_out/ --out-dir data/l0_qc/

No non-ASCII characters are used in this file.

Dependencies: numpy, h5py.
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("h5py is required: pip install h5py numpy")

from rfi_gen.canvas_l0 import _find_swaths_l0, _first_pol_dataset, _to_complex

# ---------------------------------------------------------------------------
# Heuristic thresholds -- tune for your data
# ---------------------------------------------------------------------------
TH = {
    # Top eigenvalue (dB) relative to the plateau median of the SAME CPI.
    # A clean CPI with only thermal noise has eigenvalue contrast near 0.
    # Narrowband RFI at 15+ dB INR lifts the top eigenvalue well above that.
    "eigen_contrast_review": 10.0,   # dB above plateau median
    "eigen_contrast_noisy":  18.0,

    # F-factor: sigma_max / sigma_min across a mini-TB of n_cpi CPIs.
    # Large F signals fluctuating eigenvalue structure across time -> RFI.
    "f_factor_review": 4.0,
    "f_factor_noisy":  8.0,

    # Per-pulse energy anomaly in MAD units.  A pulsed RFI burst makes one
    # or a few pulses stand out dramatically.
    "pulse_mad_review": 6.0,
    "pulse_mad_noisy":  12.0,

    # Zero fill / data gaps fraction.
    "zero_frac_review": 0.02,
    "zero_frac_noisy":  0.10,
}

# Number of random CPI blocks sampled per file for the eigenvalue/F checks.
N_PROBE_CPIS = 8
# Pulses per probed CPI (M).
PROBE_M = 32
# Range samples per probed CPI (K).
PROBE_K = 128
# Number of CPIs grouped into one mini-TB for the F-factor computation.
MINI_TB = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mad(v):
    """Median absolute deviation, normalized to Gaussian sigma (MAD/0.6745)."""
    v = np.asarray(v, dtype=np.float64)
    med = np.median(v)
    return float(np.median(np.abs(v - med)) / 0.6745)


def _slow_time_scm(S):
    """Sample covariance R = S @ S^H / K."""
    K = S.shape[1]
    return (S @ S.conj().T) / float(K)


def _eig_descending(R):
    """Real eigenvalues of a Hermitian matrix, descending order."""
    w = np.linalg.eigvalsh(R)
    return w[::-1].copy()


def _eigenvalue_contrast(block):
    """
    Return the top-eigenvalue contrast (dB) above the median plateau
    for a single (M, K) complex block.

    A clean block of white noise has a flat eigenvalue profile; contrast
    near 0 dB. A block contaminated by a single CW tone has the top
    eigenvalue sitting 10-30 dB above the rest.
    """
    R = _slow_time_scm(block)
    w = np.clip(_eig_descending(R), 1e-20, None)
    w_db = 10.0 * np.log10(w)
    if len(w_db) < 2:
        return 0.0
    plateau = float(np.median(w_db[1:]))  # exclude top eigenvalue
    return float(w_db[0] - plateau)


def _f_factor(blocks):
    """
    Compute the F-factor over a mini-TB (list of (M, K) blocks).

    F = sigma_max / sigma_min where sigma_max and sigma_min are the
    standard deviations of the first differences of lambda_max and
    lambda_min across the CPIs in the TB.
    """
    lam_max = []
    lam_min = []
    for b in blocks:
        R = _slow_time_scm(b)
        w = np.clip(_eig_descending(R), 1e-20, None)
        w_db = 10.0 * np.log10(w)
        lam_max.append(float(w_db[0]))
        lam_min.append(float(w_db[-1]))

    if len(lam_max) < 2:
        return 0.0

    d_max = np.diff(lam_max)
    d_min = np.diff(lam_min)
    sigma_max = float(np.std(d_max))
    sigma_min = float(np.std(d_min))
    return float(np.clip(sigma_max / max(sigma_min, 1e-3), 0.0, 100.0))


def _pulse_energy_mad(block):
    """
    MAD score for per-pulse energy within a (M, K) block.

    A transient RFI burst makes one pulse much brighter than the others;
    this shows up as a high MAD score.
    """
    per_pulse = np.mean(np.abs(block) ** 2, axis=1)  # (M,)
    mad_val = _mad(per_pulse)
    median_val = float(np.median(per_pulse))
    if median_val < 1e-30:
        return 0.0
    # Normalise: how many MAD units is the brightest pulse above the median?
    return float((per_pulse.max() - median_val) / (mad_val + 1e-30))


def _zero_fraction(dset, max_pulses=2000):
    """
    Estimate the fraction of raw samples that are identically zero.

    Reads at most max_pulses pulses (strided) to keep I/O bounded.
    """
    n_total = dset.shape[0]
    stride = max(1, n_total // max_pulses)
    raw = dset[::stride, :]
    block = _to_complex(raw)
    total = block.size
    zeros = int(np.sum(block == 0))
    return float(zeros) / max(total, 1)


# ---------------------------------------------------------------------------
# Per-file assessment
# ---------------------------------------------------------------------------

def assess_file(h5_path, rng):
    """
    Assess one L0B HDF5 file. Returns a metrics dict and a (flag, reasons)
    tuple.

    Parameters
    ----------
    h5_path : str
        Path to the L0B HDF5 file.
    rng : np.random.Generator
        For reproducible random block selection.

    Returns
    -------
    metrics : dict
    flag : str   -- 'CLEAN', 'REVIEW', or 'NOISY'
    reasons : str
    """
    with h5py.File(h5_path, "r") as h5:
        swaths = _find_swaths_l0(h5)
        if swaths is None:
            return {}, "ERROR", "no L0B swaths group found"

        freqs = [k for k in h5[swaths].keys()
                 if k.lower().startswith("frequency")]
        if not freqs:
            return {}, "ERROR", "no frequency groups found"

        freq_key = sorted(freqs)[0]
        try:
            dset, pol = _first_pol_dataset(h5, swaths, freq_key)
        except ValueError as e:
            return {}, "ERROR", str(e)

        n_total_pulses, n_total_range = dset.shape
        need = PROBE_M * (N_PROBE_CPIS + MINI_TB)  # conservative lower bound
        if n_total_pulses < need or n_total_range < PROBE_K:
            return {}, "ERROR", (
                "scene too small ({} x {}); need at least {} pulses x {} range"
                .format(n_total_pulses, n_total_range, need, PROBE_K))

        # Zero-fill check (read strided, not all data).
        zero_frac = _zero_fraction(dset)

        # Draw N_PROBE_CPIS + MINI_TB random CPI blocks for eigenvalue probing.
        n_blocks = N_PROBE_CPIS + MINI_TB
        stride = PROBE_M  # non-overlapping CPIs

        # Pick a random starting offset (keep away from edges).
        max_az_start = n_total_pulses - n_blocks * stride
        if max_az_start <= 0:
            max_az_start = 1
        az_start = int(rng.integers(0, max_az_start))
        rg_start = int(rng.integers(0, n_total_range - PROBE_K + 1))

        raw_strip = dset[az_start:az_start + n_blocks * stride,
                         rg_start:rg_start + PROBE_K]

    raw_strip = _to_complex(raw_strip)
    blocks = [raw_strip[k * PROBE_M:(k + 1) * PROBE_M, :]
              for k in range(n_blocks)]

    # Eigenvalue contrast per CPI.
    contrasts = [_eigenvalue_contrast(b) for b in blocks]
    eigen_contrast_max = float(max(contrasts))
    eigen_contrast_mean = float(np.mean(contrasts))

    # F-factor over the mini-TB subset.
    f_factor = _f_factor(blocks[:MINI_TB])

    # Per-pulse energy MAD (worst CPI).
    pulse_mads = [_pulse_energy_mad(b) for b in blocks]
    pulse_mad_max = float(max(pulse_mads))

    metrics = {
        "freq": freq_key,
        "pol": pol,
        "n_pulses": n_total_pulses,
        "n_range": n_total_range,
        "eigen_contrast_max_db": round(eigen_contrast_max, 2),
        "eigen_contrast_mean_db": round(eigen_contrast_mean, 2),
        "f_factor_max": round(f_factor, 3),
        "pulse_energy_mad": round(pulse_mad_max, 2),
        "zero_frac": round(zero_frac, 4),
    }

    # Verdict: worst failing threshold sets the flag.
    level = 0   # 0=CLEAN, 1=REVIEW, 2=NOISY
    reasons = []

    def bump(lvl, reason):
        nonlocal level
        if lvl > level:
            level = lvl
        reasons.append(reason)

    if eigen_contrast_max >= TH["eigen_contrast_noisy"]:
        bump(2, "eigenvalue contrast {:.1f} dB (>= {})".format(
            eigen_contrast_max, TH["eigen_contrast_noisy"]))
    elif eigen_contrast_max >= TH["eigen_contrast_review"]:
        bump(1, "eigenvalue contrast {:.1f} dB (>= {})".format(
            eigen_contrast_max, TH["eigen_contrast_review"]))

    if f_factor >= TH["f_factor_noisy"]:
        bump(2, "F-factor {:.2f} (>= {})".format(
            f_factor, TH["f_factor_noisy"]))
    elif f_factor >= TH["f_factor_review"]:
        bump(1, "F-factor {:.2f} (>= {})".format(
            f_factor, TH["f_factor_review"]))

    if pulse_mad_max >= TH["pulse_mad_noisy"]:
        bump(2, "pulse energy MAD {:.1f} (>= {})".format(
            pulse_mad_max, TH["pulse_mad_noisy"]))
    elif pulse_mad_max >= TH["pulse_mad_review"]:
        bump(1, "pulse energy MAD {:.1f} (>= {})".format(
            pulse_mad_max, TH["pulse_mad_review"]))

    if zero_frac >= TH["zero_frac_noisy"]:
        bump(2, "{:.1f}% zero samples".format(100 * zero_frac))
    elif zero_frac >= TH["zero_frac_review"]:
        bump(1, "{:.1f}% zero samples".format(100 * zero_frac))

    flag = ["CLEAN", "REVIEW", "NOISY"][level]
    reason_str = "; ".join(reasons) if reasons else "no issues flagged"
    return metrics, flag, reason_str


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Screen NISAR L0B raw granules for RFI / quality problems.")
    ap.add_argument("inputs", nargs="*",
                    help="Specific .h5 files. If omitted, uses --glob.")
    ap.add_argument("--glob", default="data/l0_out/*.h5",
                    help="Glob for L0B granules when no files are listed.")
    ap.add_argument("--out-dir", default="data/l0_qc",
                    help="Directory for the QC summary CSV.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed for reproducible block selection.")
    args = ap.parse_args()

    # If a positional argument is a directory, expand it to all .h5 files
    # inside it so the user can pass the data dir directly.
    expanded = []
    for p in args.inputs:
        if os.path.isdir(p):
            expanded.extend(sorted(glob.glob(os.path.join(p, "*.h5"))))
        else:
            expanded.append(p)
    files = expanded if expanded else sorted(glob.glob(args.glob))
    if not files:
        sys.exit("No input files. Pass .h5 files or a directory containing them.")

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = []

    for path in files:
        print("Assessing {} ...".format(os.path.basename(path)))
        try:
            metrics, flag, reasons = assess_file(path, rng)
        except Exception as exc:
            metrics, flag, reasons = (
                {}, "ERROR", "{}: {}".format(type(exc).__name__, exc))
        row = {"file": os.path.basename(path), "flag": flag,
               "reasons": reasons}
        row.update(metrics)
        rows.append(row)
        print("  -> {}  {}".format(flag, reasons))

    # Write summary CSV with the same 'file' / 'flag' columns that
    # canvas_l0.read_clean_list and augment.py --qc-csv expect.
    fields = ["file", "flag", "reasons", "freq", "pol",
              "n_pulses", "n_range",
              "eigen_contrast_max_db", "eigen_contrast_mean_db",
              "f_factor_max", "pulse_energy_mad", "zero_frac"]
    csv_path = os.path.join(args.out_dir, "l0_qc_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    clean  = [r for r in rows if r["flag"] == "CLEAN"]
    review = [r for r in rows if r["flag"] == "REVIEW"]
    noisy  = [r for r in rows if r["flag"] == "NOISY"]
    err    = [r for r in rows if r["flag"] == "ERROR"]
    print("\nSummary: {} CLEAN  {} REVIEW  {} NOISY  {} ERROR".format(
        len(clean), len(review), len(noisy), len(err)))
    print("Wrote {}".format(csv_path))
    if clean:
        print("\nClean L0B canvases:")
        for r in clean:
            print("  {}".format(r["file"]))


if __name__ == "__main__":
    main()