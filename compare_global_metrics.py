#!/usr/bin/env python3
"""
compare_global_metrics.py

Compute per-CPI eigenvalue-based cleanliness metrics -- condition number,
effective rank, and median/max eigenvalue ratio -- from either:

  1. "preprocessed" mode: an already-extracted per-CPI eigenvalue/SCM-diagonal
     HDF5 file (e.g. clean_mountains_filtered.h5), or
  2. "nisar" mode: a raw NISAR L0B HDF5 file, tiled on the fly into
     16-pulse x 250-range CPI blocks.

Both modes report the same summary statistics (mean, median, std, min, max,
plus IQR and 5th/95th percentiles) for the three metrics, so results from
the two modes can be compared directly.

Only valid data is used in every metric:
  - only the first N_KEEP (default 12) eigenvalues of each 16-eigenvalue
    profile are used (this drops the dithered/collapsed tail, per project
    convention)
  - only CPI tiles whose SCM diagonal validity fraction meets
    --diag-valid-frac-thresh (default 0.8) are included

SYNTHETIC RFI-ONLY METRIC (--mode preprocessed only):
If the input file carries a per-tile "labels" dataset (knee = number of
injected RFI bands, 0 = clean; e.g. the rfi_data_<freq>_<pol>.h5 output of
generate_amazon_data.py / generate_mountain_data.py, or a grouped
freq_X_pol_Y file with a "labels" dataset alongside "eigenvalues"), a
second report is printed alongside the usual "all valid tiles" report. This
second report computes the exact same three metrics (condition number,
effective rank, median/max ratio) but restricted to tiles whose label falls
in [--rfi-label-min, --rfi-label-max] (default 1..6) -- i.e. only tiles that
actually received injected RFI. Clean tiles (label 0) are disregarded for
this metric so they cannot dilute the RFI-only statistics.

If no "labels" dataset is present (e.g. clean_mountains_filtered.h5, or any
--mode nisar input, since raw NISAR data has no injected-RFI ground truth),
the RFI-only report is skipped and a note is printed instead.

IMPORTANT NOTE ON THE NISAR PATH:
The gap-exclusion covariance construction used here (see
`compute_gap_exclusion_cov_simple`) is a best-effort reconstruction based on
project notes (mean-magnitude gap detection at 10 percent of the max, per-CPI
pulse validity). It does NOT reproduce the exact per-element (3 percent
off-diagonal / 2 percent diagonal) gap-exclusion ratios used in the project's
real `compute_gap_exclusion_cov` (read_nisar_swaths_isce3.py /
cpi_preprocess.py). If exact parity with that pipeline matters, swap
`compute_gap_exclusion_cov_simple` out for the real function -- by project
convention these reader/covariance functions are self-contained and should
drop in directly in place of the one here.

Usage:
  Preprocessed file:
    python compare_global_metrics.py --mode preprocessed \\
        --input clean_mountains_filtered.h5 --freq A --pol HH

  Raw NISAR L0B file:
    python compare_global_metrics.py --mode nisar \\
        --input NISAR_L0_..._h5 --freq A --pol HH \\
        --pulse-start 0 --pulse-end 10000 --range-start 0 --range-end 5000
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np

# CPI tile size -- always 16 pulses x 250 range samples, per project convention
CPI_PULSES = 16
CPI_RANGE = 250

DEFAULT_N_KEEP = 12
DEFAULT_DIAG_VALID_FRAC_THRESH = 0.8
DEFAULT_GAP_MAG_FRAC = 0.10  # transmission-gap detection threshold, per project notes

# Label convention for generate_amazon_data.py / generate_mountain_data.py
# output: labels[i] == 0 means a clean tile, labels[i] in [1, max_bands] means
# knee = number of injected RFI bands. Default RFI-only range matches the
# project's default --max-bands of 6.
DEFAULT_RFI_LABEL_MIN = 1
DEFAULT_RFI_LABEL_MAX = 6


# ---------------------------------------------------------------------------
# Shared metric core (used by both preprocessed and nisar modes)
# ---------------------------------------------------------------------------

def compute_condition_number_db(eig_kept):
    """
    Condition number in dB: ratio of the largest to the smallest of the
    kept eigenvalues, expressed in dB.

    eig_kept : (n_cpi, n_keep) array, linear-scale eigenvalues, sorted
               descending along axis 1.
    Returns  : (n_cpi,) array in dB.
    """
    eig_max = eig_kept[:, 0]
    eig_min_kept = eig_kept[:, -1]
    with np.errstate(divide="ignore", invalid="ignore"):
        cond_db = 10.0 * np.log10(eig_max / eig_min_kept)
    return cond_db


def compute_effective_rank(eig_kept):
    """
    Effective rank via Shannon entropy of the normalized kept eigenvalues
    (Roy and Vetterli definition): treat eig_kept / sum(eig_kept) as a
    probability distribution, take its Shannon entropy, then exponentiate.

    eig_kept : (n_cpi, n_keep) array, linear-scale eigenvalues.
    Returns  : (n_cpi,) array, effective rank in [1, n_keep].
    """
    eig_sum = eig_kept.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = eig_kept / eig_sum
        # 0 * log(0) = 0 by convention
        plogp = np.where(p > 0, p * np.log(p), 0.0)
        entropy = -plogp.sum(axis=1)
    return np.exp(entropy)


def compute_median_max_ratio(eig_kept, definition="median_over_max"):
    """
    Median/max eigenvalue ratio among the kept eigenvalues.

    definition:
      "median_over_max" (default): median(eig_kept) / max(eig_kept), in (0, 1]
      "max_over_median": max(eig_kept) / median(eig_kept), in [1, inf)

    eig_kept : (n_cpi, n_keep) array, linear-scale eigenvalues.
    Returns  : (n_cpi,) array.
    """
    eig_median = np.median(eig_kept, axis=1)
    eig_max = eig_kept[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        if definition == "median_over_max":
            ratio = eig_median / eig_max
        elif definition == "max_over_median":
            ratio = eig_max / eig_median
        else:
            raise ValueError(f"Unknown median_max_ratio definition: {definition}")
    return ratio


def compute_all_metrics(eig_kept, ratio_definition="median_over_max"):
    """Compute all three metrics for a (n_cpi, n_keep) eigenvalue array."""
    return {
        "condition_number_db": compute_condition_number_db(eig_kept),
        "effective_rank": compute_effective_rank(eig_kept),
        "median_max_ratio": compute_median_max_ratio(eig_kept, ratio_definition),
    }


def summarize(values, name):
    """Summary statistics for one metric across all valid CPI tiles."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"metric": name, "count": 0}
    return {
        "metric": name,
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
    }


# ---------------------------------------------------------------------------
# Preprocessed-file loader (e.g. clean_mountains_filtered.h5)
# ---------------------------------------------------------------------------

def _read_preprocessed_arrays(path, freq, pol):
    """
    Read the raw per-tile arrays from a preprocessed HDF5 file, supporting
    both layouts used in this project:

      1. Grouped layout (e.g. clean_mountains_filtered.h5):
             freq_{freq}_pol_{pol}/eigenvalues       (n_cpi, 16)
             freq_{freq}_pol_{pol}/diag_valid_frac   (n_cpi,)
             freq_{freq}_pol_{pol}/pulse_idx         (n_cpi,)
             freq_{freq}_pol_{pol}/range_idx         (n_cpi,)
             freq_{freq}_pol_{pol}/labels            (n_cpi,)  [optional]

      2. Flat, one-file-per-channel layout (rfi_data_<freq>_<pol>.h5, written
         by generate_amazon_data.py / generate_mountain_data.py):
             /eigenvalues       (n_cpi, cpi_len)
             /diag_valid_idx    (n_cpi, cpi_len)  bool, per-index validity
             /tile_pulse        (n_cpi,)
             /tile_range        (n_cpi,)
             /labels            (n_cpi,)          knee, 0 = clean

    Returns eigenvalues, diag_valid_frac, pulse_idx, range_idx, labels
    (labels is None if no "labels" dataset is present in either layout).
    """
    group_name = f"freq_{freq}_pol_{pol}"
    with h5py.File(path, "r") as f:
        if group_name in f:
            g = f[group_name]
            eigenvalues = g["eigenvalues"][:]
            diag_valid_frac = g["diag_valid_frac"][:]
            pulse_idx = g["pulse_idx"][:]
            range_idx = g["range_idx"][:]
            labels = g["labels"][:] if "labels" in g else None
        elif "eigenvalues" in f:
            eigenvalues = f["eigenvalues"][:]
            diag_valid_idx = f["diag_valid_idx"][:]  # (n_cpi, cpi_len) bool
            diag_valid_frac = diag_valid_idx.mean(axis=1)
            pulse_idx = f["tile_pulse"][:]
            range_idx = f["tile_range"][:]
            labels = f["labels"][:] if "labels" in f else None
        else:
            available = list(f.keys())
            raise KeyError(
                f"Neither group '{group_name}' nor a root-level 'eigenvalues' "
                f"dataset was found in {path}. Available top-level keys: "
                f"{available}"
            )
    return eigenvalues, diag_valid_frac, pulse_idx, range_idx, labels


def load_preprocessed(path, freq, pol, diag_valid_frac_thresh, n_keep,
                       pulse_start=None, pulse_end=None,
                       range_start=None, range_end=None,
                       rfi_label_min=DEFAULT_RFI_LABEL_MIN,
                       rfi_label_max=DEFAULT_RFI_LABEL_MAX):
    """
    Load per-CPI eigenvalues from a preprocessed HDF5 file (either the
    grouped freq_X_pol_Y layout or the flat rfi_data_<freq>_<pol>.h5 layout;
    see _read_preprocessed_arrays for both schemas).

    Returns:
        eig_kept          : (n_valid, n_keep) all valid tiles, linear scale
        eig_kept_rfi_only : (n_rfi, n_keep) subset of eig_kept whose label is
                             in [rfi_label_min, rfi_label_max], or None if the
                             file carries no "labels" dataset
        counts            : dict of tile counts for reporting
    """
    eigenvalues, diag_valid_frac, pulse_idx, range_idx, labels = \
        _read_preprocessed_arrays(path, freq, pol)

    n_total = eigenvalues.shape[0]

    scene_mask = np.ones(n_total, dtype=bool)
    if pulse_start is not None:
        scene_mask &= pulse_idx >= pulse_start
    if pulse_end is not None:
        scene_mask &= pulse_idx < pulse_end
    if range_start is not None:
        scene_mask &= range_idx >= range_start
    if range_end is not None:
        scene_mask &= range_idx < range_end

    n_in_scene = int(scene_mask.sum())

    valid_mask = scene_mask & (diag_valid_frac >= diag_valid_frac_thresh)
    n_valid = int(valid_mask.sum())

    eig_valid = eigenvalues[valid_mask]
    eig_kept = eig_valid[:, :n_keep]

    counts = {
        "n_total_in_file": n_total,
        "n_in_requested_scene": n_in_scene,
        "n_passing_diag_valid_frac": n_valid,
    }

    eig_kept_rfi_only = None
    if labels is not None:
        labels_valid = labels[valid_mask]
        rfi_mask = (labels_valid >= rfi_label_min) & (labels_valid <= rfi_label_max)
        eig_kept_rfi_only = eig_kept[rfi_mask]
        counts["n_rfi_labeled_1_to_6"] = int(rfi_mask.sum())
        counts["n_clean_labeled_0"] = int((labels_valid == 0).sum())

    return eig_kept, eig_kept_rfi_only, counts


# ---------------------------------------------------------------------------
# Raw NISAR L0B loader (tiles CPIs on the fly)
# ---------------------------------------------------------------------------

def compute_gap_exclusion_cov_simple(pulse_block, gap_mag_frac=DEFAULT_GAP_MAG_FRAC):
    """
    Simplified gap-exclusion sample covariance matrix for one CPI x
    range-tile block.

    pulse_block : (n_pulses, n_range) complex array (decoded raw samples)

    Detects transmission-gap pulses via mean magnitude (per project notes:
    inter-subswath gaps produce near-zero, not exactly-zero, ADC fill), then
    forms the SCM using only the valid pulses:
        R = S_valid @ S_valid^H / n_range

    This is a per-pulse (not per-element) validity criterion, and is a
    simplification of the project's real compute_gap_exclusion_cov (which
    applies separate 3 percent off-diagonal / 2 percent diagonal validity
    ratios per matrix element). Swap in the real function for exact parity.

    Returns:
        cov            : (n_pulses, n_pulses) complex covariance matrix,
                          entries involving an invalid pulse set to NaN
        diag_valid_frac : scalar, fraction of pulses considered valid
    """
    n_pulses, n_range = pulse_block.shape
    mean_mag = np.abs(pulse_block).mean(axis=1)
    max_mag = mean_mag.max() if mean_mag.size else 0.0
    gap_thresh = max_mag * gap_mag_frac
    valid_pulse = mean_mag >= gap_thresh

    diag_valid_frac = float(valid_pulse.mean()) if n_pulses else 0.0

    cov = np.full((n_pulses, n_pulses), np.nan, dtype=complex)
    valid_idx = np.where(valid_pulse)[0]
    if valid_idx.size > 0:
        s_valid = pulse_block[valid_idx, :]
        r_valid = (s_valid @ s_valid.conj().T) / n_range
        cov[np.ix_(valid_idx, valid_idx)] = r_valid

    return cov, diag_valid_frac


def load_raw_cpi_block(raw_dataset, pulse0, range0, cpi_pulses, cpi_range):
    """
    Read one (cpi_pulses, cpi_range) complex block starting at (pulse0, range0)
    from a decoded raw dataset object.

    This is the integration point for the project's actual L0B reader
    (see read_nisar_swaths_isce3.py / tile_cpi.py). It is left as a thin
    wrapper so it can be swapped for the real reader without touching the
    rest of this script.

    raw_dataset : an object supporting raw_dataset[p0:p1, r0:r1] slicing
                  that returns decoded complex samples (e.g. the object
                  returned by isce3's Raw.getRawDataset(frequency, polarization))
    """
    p1 = pulse0 + cpi_pulses
    r1 = range0 + cpi_range
    return raw_dataset[pulse0:p1, range0:r1]


def load_nisar(path, freq, pol, diag_valid_frac_thresh, n_keep,
                pulse_start=None, pulse_end=None,
                range_start=None, range_end=None,
                gap_mag_frac=DEFAULT_GAP_MAG_FRAC):
    """
    Load and tile a raw NISAR L0B HDF5 file into 16-pulse x 250-range CPI
    blocks over the requested (or full) pulse/range scene extent, form the
    gap-exclusion SCM per tile, eigendecompose, and return the per-CPI
    eigenvalue array (kept to n_keep) plus counts.

    Requires isce3 to decode the BFPQLUT-compressed raw data.
    """
    try:
        from nisar.products.readers.Raw import Raw
    except ImportError as exc:
        raise RuntimeError(
            "nisar.products.readers.Raw is required for --mode nisar (BFPQLUT decoding). "
            "Run this script inside the project's 'isce3' conda environment "
            "(py-isce3) on nisar-adt-dev-5."
        ) from exc

    raw = Raw(hdf5file=path)
    raw.parsePolarizations()
    raw_dataset = raw.getRawDataset(freq, pol)
    n_pulses_total, n_range_total = raw_dataset.shape

    p_start = 0 if pulse_start is None else pulse_start
    p_end = n_pulses_total if pulse_end is None else min(pulse_end, n_pulses_total)
    r_start = 0 if range_start is None else range_start
    r_end = n_range_total if range_end is None else min(range_end, n_range_total)

    eig_rows = []
    diag_valid_fracs = []

    n_cpi_pulse = (p_end - p_start) // CPI_PULSES
    n_cpi_range = (r_end - r_start) // CPI_RANGE

    # Read data in larger chunks (1000 pulses at a time) to avoid slow random access
    CHUNK_PULSES = 1000
    chunk_cpi = CHUNK_PULSES // CPI_PULSES

    for chunk_idx in range(0, n_cpi_pulse, chunk_cpi):
        chunk_n = min(chunk_cpi, n_cpi_pulse - chunk_idx)
        chunk_p0 = p_start + chunk_idx * CPI_PULSES
        chunk_p1 = chunk_p0 + chunk_n * CPI_PULSES

        # Read entire chunk once
        raw_chunk = raw_dataset[chunk_p0:chunk_p1, r_start:r_end]

        for ip_local in range(chunk_n):
            lp0 = ip_local * CPI_PULSES
            lp1 = lp0 + CPI_PULSES

            for ir in range(n_cpi_range):
                lr0 = ir * CPI_RANGE
                lr1 = lr0 + CPI_RANGE
                block = raw_chunk[lp0:lp1, lr0:lr1]

                cov, dvf = compute_gap_exclusion_cov_simple(block, gap_mag_frac)
                diag_valid_fracs.append(dvf)

                if dvf < diag_valid_frac_thresh:
                    eig_rows.append(np.full(CPI_PULSES, np.nan))
                    continue

                valid_idx = np.where(~np.isnan(np.diag(cov)))[0]
                if valid_idx.size < 2:
                    eig_rows.append(np.full(CPI_PULSES, np.nan))
                    continue

                sub_cov = cov[np.ix_(valid_idx, valid_idx)]
                eigvals = np.linalg.eigvalsh(sub_cov)  # ascending, real (Hermitian)
                eigvals = np.sort(eigvals)[::-1]        # descending
                padded = np.full(CPI_PULSES, np.nan)
                padded[: eigvals.size] = np.real(eigvals)
                eig_rows.append(padded)

    eigenvalues = np.array(eig_rows)
    diag_valid_frac_arr = np.array(diag_valid_fracs)

    n_total = eigenvalues.shape[0]
    valid_mask = diag_valid_frac_arr >= diag_valid_frac_thresh
    valid_mask &= ~np.any(np.isnan(eigenvalues[:, :n_keep]), axis=1) if n_total else valid_mask
    n_valid = int(valid_mask.sum())

    eig_valid = eigenvalues[valid_mask]
    eig_kept = eig_valid[:, :n_keep]

    counts = {
        "n_total_cpi_tiles": n_total,
        "n_passing_diag_valid_frac": n_valid,
    }
    # Raw NISAR L0B data carries no injected-RFI ground truth, so there is no
    # "labels" dataset to restrict to here. Kept as None for a consistent
    # return signature with load_preprocessed.
    eig_kept_rfi_only = None
    return eig_kept, eig_kept_rfi_only, counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_report(mode, args, counts, summaries, ratio_definition, report_name="All valid CPI tiles"):
    print("=" * 70)
    print(f"Mode: {mode}")
    print(f"Report: {report_name}")
    print(f"Input: {args.input}")
    print(f"Frequency / Polarization: {args.freq} / {args.pol}")
    print(f"n_keep (eigenvalues used): {args.n_keep}")
    print(f"diag_valid_frac threshold: {args.diag_valid_frac_thresh}")
    print(f"median_max_ratio definition: {ratio_definition}")
    for k, v in counts.items():
        print(f"{k}: {v}")
    print("-" * 70)
    header = f"{'metric':22s}{'count':>8s}{'mean':>10s}{'median':>10s}{'std':>10s}{'min':>10s}{'max':>10s}{'p05':>10s}{'p95':>10s}{'iqr':>10s}"
    print(header)
    for s in summaries:
        if s.get("count", 0) == 0:
            print(f"{s['metric']:22s}{'0':>8s} (no valid CPI tiles)")
            continue
        print(
            f"{s['metric']:22s}{s['count']:>8d}{s['mean']:>10.3f}{s['median']:>10.3f}"
            f"{s['std']:>10.3f}{s['min']:>10.3f}{s['max']:>10.3f}"
            f"{s['p05']:>10.3f}{s['p95']:>10.3f}{s['iqr']:>10.3f}"
        )
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-CPI condition number / effective rank / "
                    "median-max ratio statistics from a preprocessed or "
                    "raw NISAR HDF5 file."
    )
    parser.add_argument("--mode", choices=["preprocessed", "nisar"], required=True,
                        help="Whether --input is a preprocessed per-CPI file "
                             "or a raw NISAR L0B file.")
    parser.add_argument("--input", required=True, help="Path to the HDF5 file.")
    parser.add_argument("--freq", default="A", help="Frequency band (default: A)")
    parser.add_argument("--pol", default="HH", help="Polarization, e.g. HH/HV/VV/VH (default: HH)")

    parser.add_argument("--pulse-start", type=int, default=None,
                        help="Scene start pulse index (default: full extent)")
    parser.add_argument("--pulse-end", type=int, default=None,
                        help="Scene end pulse index, exclusive (default: full extent)")
    parser.add_argument("--range-start", type=int, default=None,
                        help="Scene start range index (default: full extent)")
    parser.add_argument("--range-end", type=int, default=None,
                        help="Scene end range index, exclusive (default: full extent)")

    parser.add_argument("--n-keep", type=int, default=DEFAULT_N_KEEP,
                        help=f"Number of leading eigenvalues to keep (default: {DEFAULT_N_KEEP})")
    parser.add_argument("--diag-valid-frac-thresh", type=float,
                        default=DEFAULT_DIAG_VALID_FRAC_THRESH,
                        help=f"Minimum SCM diagonal valid fraction to keep a CPI tile "
                             f"(default: {DEFAULT_DIAG_VALID_FRAC_THRESH})")
    parser.add_argument("--median-max-ratio-def", choices=["median_over_max", "max_over_median"],
                        default="median_over_max",
                        help="Definition of the median/max ratio metric (default: median_over_max)")
    parser.add_argument("--gap-mag-frac", type=float, default=DEFAULT_GAP_MAG_FRAC,
                        help=f"[nisar mode only] fraction of max mean-magnitude used as the "
                             f"transmission-gap threshold (default: {DEFAULT_GAP_MAG_FRAC})")

    parser.add_argument("--rfi-label-min", type=int, default=DEFAULT_RFI_LABEL_MIN,
                        help="[preprocessed mode only] minimum knee label (inclusive) counted "
                             f"as synthetic RFI for the RFI-only report (default: {DEFAULT_RFI_LABEL_MIN})")
    parser.add_argument("--rfi-label-max", type=int, default=DEFAULT_RFI_LABEL_MAX,
                        help="[preprocessed mode only] maximum knee label (inclusive) counted "
                             f"as synthetic RFI for the RFI-only report (default: {DEFAULT_RFI_LABEL_MAX})")

    parser.add_argument("--output-json", default=None,
                        help="Optional path to write summary statistics as JSON.")
    parser.add_argument("--output-csv", default=None,
                        help="Optional path to write per-CPI metric values as CSV.")

    args = parser.parse_args()

    eig_kept_rfi_only = None
    if args.mode == "preprocessed":
        eig_kept, eig_kept_rfi_only, counts = load_preprocessed(
            args.input, args.freq, args.pol,
            args.diag_valid_frac_thresh, args.n_keep,
            args.pulse_start, args.pulse_end,
            args.range_start, args.range_end,
            args.rfi_label_min, args.rfi_label_max,
        )
    else:
        eig_kept, eig_kept_rfi_only, counts = load_nisar(
            args.input, args.freq, args.pol,
            args.diag_valid_frac_thresh, args.n_keep,
            args.pulse_start, args.pulse_end,
            args.range_start, args.range_end,
            args.gap_mag_frac,
        )

    if eig_kept.shape[0] == 0:
        print("No valid CPI tiles found with the given filters.", file=sys.stderr)
        sys.exit(1)

    metrics = compute_all_metrics(eig_kept, args.median_max_ratio_def)
    summaries = [summarize(v, name) for name, v in metrics.items()]

    print_report(args.mode, args, counts, summaries, args.median_max_ratio_def,
                 report_name="All valid CPI tiles")

    # Synthetic RFI-only report: same three metrics, restricted to tiles
    # whose label (knee) falls in [rfi_label_min, rfi_label_max]. Clean
    # tiles (label 0) are excluded so they cannot dilute these statistics.
    metrics_rfi_only = None
    summaries_rfi_only = None
    if eig_kept_rfi_only is None:
        print()
        print(f"[RFI-only report skipped: no 'labels' dataset found for "
              f"{args.mode} input -- this report requires a labeled RFI "
              f"training file such as rfi_data_<freq>_<pol>.h5, produced by "
              f"generate_amazon_data.py / generate_mountain_data.py]")
    elif eig_kept_rfi_only.shape[0] == 0:
        print()
        print(f"[RFI-only report skipped: no valid tiles with label in "
              f"[{args.rfi_label_min}, {args.rfi_label_max}] found]")
    else:
        metrics_rfi_only = compute_all_metrics(eig_kept_rfi_only, args.median_max_ratio_def)
        summaries_rfi_only = [summarize(v, name) for name, v in metrics_rfi_only.items()]
        rfi_counts = {
            "n_rfi_only_tiles": int(eig_kept_rfi_only.shape[0]),
            "rfi_label_range": f"[{args.rfi_label_min}, {args.rfi_label_max}]",
        }
        print()
        print_report(args.mode, args, rfi_counts, summaries_rfi_only, args.median_max_ratio_def,
                     report_name=f"Synthetic RFI tiles only (labels {args.rfi_label_min}-{args.rfi_label_max})")

    if args.output_json:
        payload = {
            "mode": args.mode,
            "input": args.input,
            "freq": args.freq,
            "pol": args.pol,
            "n_keep": args.n_keep,
            "diag_valid_frac_thresh": args.diag_valid_frac_thresh,
            "median_max_ratio_def": args.median_max_ratio_def,
            "counts": counts,
            "summaries_all": summaries,
            "rfi_label_min": args.rfi_label_min,
            "rfi_label_max": args.rfi_label_max,
            "summaries_rfi_only": summaries_rfi_only,
        }
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote summary JSON to {args.output_json}")

    if args.output_csv:
        import csv
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(metrics.keys()))
            for row in zip(*metrics.values()):
                writer.writerow(row)
        print(f"Wrote per-CPI metrics CSV to {args.output_csv}")

        if metrics_rfi_only is not None:
            root, ext = os.path.splitext(args.output_csv)
            rfi_csv_path = f"{root}_rfi_only{ext or '.csv'}"
            with open(rfi_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(list(metrics_rfi_only.keys()))
                for row in zip(*metrics_rfi_only.values()):
                    writer.writerow(row)
            print(f"Wrote RFI-only per-CPI metrics CSV to {rfi_csv_path}")


if __name__ == "__main__":
    main()