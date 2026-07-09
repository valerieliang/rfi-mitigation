"""
identify_clean_profiles.py

Given a NISAR L0B HDF5 file and a pulse/range interval, compute per-tile
slow-time eigenvalue profiles (in dB) and flag tiles whose profile is
consistent with an RFI-free (eigenvalue-clean) signal, using a conservative
physics-based heuristic rather than the CNN model.

The heuristic follows directly from the ST-EVD RFI assumption (Huang et al.,
IGARSS 2023): RFI power exceeds signal power, so a genuine RFI-contaminated
eigenvalue must show a clear dominant step relative to the rest of the
profile. If no such step exists anywhere in the profile, and the overall
spread is modest, the tile is very unlikely to contain RFI regardless of
whether the underlying scene is noise-limited (flat tail) or clutter-limited
(smooth decline, e.g. dense forest volume scattering).

This is intentionally conservative: the thresholds are set to avoid ever
mislabeling an RFI-contaminated tile as clean, at the cost of possibly
rejecting some genuinely clean tiles near the threshold. A spatial
consistency filter is then applied along the pulse (azimuth) axis, since
real RFI is azimuth-banded, so an isolated single clean tile surrounded by
contaminated neighbors is more likely measurement noise than a true gap in
the interference.

Usage
-----
  # Screen a pulse/range interval for clean tiles
  python identify_clean_profiles.py input.h5 --freq A --pol HV \\
      --pulse-start 112000 --pulse-end 123000 \\
      --range-start 0 --range-end 10000 \\
      --output-dir ./clean_regions

  # Loosen or tighten the heuristic
  python identify_clean_profiles.py input.h5 --freq A --pol HV \\
      --pulse-start 0 --pulse-end 50000 \\
      --max-first-step-db 1.5 --max-any-step-db 2.0 --max-spread-db 10.0
"""

import argparse
import os
import sys
import time
from datetime import datetime

import h5py
import numpy as np

from nisar.products.readers.Raw import Raw


def parse_args():
    parser = argparse.ArgumentParser(
        description="Identify eigenvalue-clean tiles in a NISAR L0B file using a conservative heuristic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_file", help="Input NISAR L0B HDF5 file path")
    parser.add_argument("--freq", choices=["A", "B"], default="A")
    parser.add_argument("--pol", choices=["HH", "HV", "VH", "VV"], default="HV")
    parser.add_argument("--output-dir", type=str, default="./clean_regions")

    parser.add_argument("--pulse-start", type=int, default=None, help="Start pulse index (slow time)")
    parser.add_argument("--pulse-end", type=int, default=None, help="End pulse index (slow time)")
    parser.add_argument("--range-start", type=int, default=None, help="Start range sample index")
    parser.add_argument("--range-end", type=int, default=None, help="End range sample index")

    parser.add_argument("--cpi-len", type=int, default=16, help="Pulses per CPI (default: 16)")
    parser.add_argument("--cpi-width", type=int, default=250, help="Range samples per tile (default: 250)")

    parser.add_argument("--db-mode", choices=["top_anchored", "floor_anchored", "raw"],
                         default="top_anchored",
                         help="dB normalization convention for the saved/plotted profiles. "
                              "Gap-based heuristic thresholds are shift-invariant and unaffected "
                              "by this choice (default: top_anchored, matches existing plots)")

    # Conservative heuristic thresholds
    parser.add_argument("--max-first-step-db", type=float, default=2.0,
                         help="Max allowed dB drop from eigenvalue 1 to 2 for a tile to be clean")
    parser.add_argument("--max-any-step-db", type=float, default=2.5,
                         help="Max allowed dB drop between any adjacent eigenvalue pair for a tile to be clean")
    parser.add_argument("--max-spread-db", type=float, default=12.0,
                         help="Max allowed total dB spread (max minus min eigenvalue) for a tile to be clean")
    parser.add_argument("--step-search-depth", type=int, default=None,
                         help="Restrict the max-step and spread checks to the top N eigenvalue "
                              "indices, where a real RFI component would appear. Default: half of "
                              "cpi-len. Set to cpi-len - 1 to check the full profile (not recommended "
                              "at typical range-tile sample sizes, see docstring)")

    # Spatial consistency filter (along pulse/azimuth axis, per range tile column)
    parser.add_argument("--consistency-window", type=int, default=5,
                         help="Number of adjacent CPIs (same range tile) considered for the consistency vote")
    parser.add_argument("--consistency-frac", type=float, default=0.8,
                         help="Minimum fraction of tiles in the window that must also be clean")

    # Global clean-window summary
    parser.add_argument("--min-range-frac", type=float, default=0.95,
                         help="Minimum fraction of range tiles in a CPI row that must be clean for the "
                              "row to count as a full-frame clean pulse interval")
    parser.add_argument("--min-window-cpi", type=int, default=5,
                         help="Minimum number of consecutive clean CPI rows to report as a clean window")

    parser.add_argument("--save-plots", action="store_true",
                         help="Save a few example clean and rejected profile plots for visual sanity check")

    parser.add_argument("--gap-threshold-frac", type=float, default=0.1,
                         help="Range tiles whose mean magnitude falls below this fraction of the "
                              "median tile's mean magnitude are treated as transmission-gap or "
                              "edge fill and excluded from all clean/RFI statistics (default: 0.1)")

    return parser.parse_args()


def read_raw_data_batch(raw, freq, pol, pulse_slice, range_slice):
    """Read a batch of raw pulses using ISCE3's Raw reader (handles BFPQLUT decoding)."""
    dataset = raw.getRawDataset(freq, pol)

    pulse_start = pulse_slice.start if pulse_slice.start is not None else 0
    pulse_stop = pulse_slice.stop if pulse_slice.stop is not None else dataset.shape[0]
    range_start = range_slice.start if range_slice.start is not None else 0
    range_stop = range_slice.stop if range_slice.stop is not None else dataset.shape[1]

    return dataset[pulse_start:pulse_stop, range_start:range_stop]


def detect_gap_range_tiles(raw_data, cpi_width, threshold_frac=0.1):
    """
    Flag range tiles that are transmission-gap or edge fill rather than real
    signal, using mean magnitude relative to the median tile's mean
    magnitude. This is the ADC near-zero-fill test: gap columns are not
    exactly zero, so an absolute-zero test is not reliable, but their mean
    magnitude is far below a typical signal-bearing tile.

    Returns
    -------
    valid : np.ndarray of bool, shape (n_range_tiles,)
        True for tiles that appear to contain real signal.
    """
    n_range = raw_data.shape[1]
    n_range_tiles = n_range // cpi_width
    mag = np.abs(raw_data)

    tile_means = np.zeros(n_range_tiles, dtype=np.float64)
    for rt in range(n_range_tiles):
        r0 = rt * cpi_width
        tile_means[rt] = mag[:, r0:r0 + cpi_width].mean()

    median_mag = np.median(tile_means)
    valid = tile_means >= threshold_frac * median_mag

    return valid


def compute_eigenvalue_tiles(raw_data, cpi_len, cpi_width):
    """
    Compute the sorted (descending) slow-time SCM eigenvalues for every
    CPI x range-tile in raw_data.

    Returns
    -------
    eigvals : np.ndarray, shape (num_cpi, n_range_tiles, cpi_len), linear power
    """
    n_pulses, n_range = raw_data.shape
    num_cpi = n_pulses // cpi_len
    n_range_tiles = n_range // cpi_width

    eigvals = np.zeros((num_cpi, n_range_tiles, cpi_len), dtype=np.float64)

    for cpi_idx in range(num_cpi):
        p0 = cpi_idx * cpi_len
        cpi_block = raw_data[p0:p0 + cpi_len, :]
        for rt_idx in range(n_range_tiles):
            r0 = rt_idx * cpi_width
            tile = cpi_block[:, r0:r0 + cpi_width]
            M, K = tile.shape
            scm = (tile @ tile.conj().T) / K
            ev = np.linalg.eigvalsh(scm)
            eigvals[cpi_idx, rt_idx, :] = np.sort(ev)[::-1]

    return eigvals


def eigenvalues_to_db(eigvals, mode="top_anchored", eps=1e-12):
    """Convert linear-power eigenvalues to dB under the requested anchoring convention."""
    lin = np.maximum(eigvals, eps)
    db = 10.0 * np.log10(lin)

    if mode == "top_anchored":
        db = db - db[..., :1]
    elif mode == "floor_anchored":
        db = db - db[..., -1:]
    elif mode == "raw":
        pass
    else:
        raise ValueError("Unknown db_mode: %s" % mode)

    return db.astype(np.float32)


def classify_clean_tiles(eigvals_db, max_first_step_db, max_any_step_db, max_spread_db,
                          step_search_depth=None):
    """
    Conservative per-tile clean classification.

    A tile is marked clean only if:
      1. The first eigenvalue step (largest-to-second-largest) is below
         max_first_step_db, i.e. there is no dominant component.
      2. No step within the top step_search_depth indices exceeds
         max_any_step_db, i.e. no staircase break among the eigenvalues
         where a real RFI component could plausibly sit.
      3. The spread from eigenvalue 1 down to step_search_depth is below
         max_spread_db.

    step_search_depth restricts checks 2 and 3 to the upper part of the
    sorted spectrum. RFI eigenvalues are by construction the dominant ones
    (ST-EVD assumes RFI power exceeds signal power), so a genuine additional
    RFI component always appears near the top of the profile, never buried
    deep in the tail. The smallest eigenvalues sit near the noise floor and
    have real sample-covariance estimation variance, especially at modest
    K/M ratios (few hundred range samples over a CPI length of order 10),
    so evaluating the full profile depth produces false rejections from
    tail noise rather than real contamination. Default is half of cpi_len.

    Full-profile diagnostics are still returned for comparison/tuning even
    when step_search_depth restricts the actual decision.

    Steps are differences of dB values and are therefore invariant to the
    db_mode anchoring convention used for eigvals_db.
    """
    cpi_len = eigvals_db.shape[-1]
    if step_search_depth is None:
        step_search_depth = max(cpi_len // 2, 1)
    step_search_depth = int(min(step_search_depth, cpi_len - 1))

    steps = -np.diff(eigvals_db, axis=-1)  # positive dB drop per index step
    first_step = steps[..., 0]

    max_step_full = np.max(steps, axis=-1)
    max_step_depth = np.max(steps[..., :step_search_depth], axis=-1)

    spread_full = eigvals_db[..., 0] - eigvals_db[..., -1]
    spread_depth = eigvals_db[..., 0] - eigvals_db[..., step_search_depth]

    is_clean = (
        (first_step <= max_first_step_db)
        & (max_step_depth <= max_any_step_db)
        & (spread_depth <= max_spread_db)
    )

    diagnostics = {
        "first_step_db": first_step.astype(np.float32),
        "max_step_db": max_step_full.astype(np.float32),
        "max_step_depth_db": max_step_depth.astype(np.float32),
        "spread_db": spread_full.astype(np.float32),
        "spread_depth_db": spread_depth.astype(np.float32),
    }

    return is_clean, diagnostics


def apply_spatial_consistency(mask, window, min_frac):
    """
    Require a tile to also have a majority-clean neighborhood along the
    pulse (azimuth) axis, within the same range-tile column, since real RFI
    is azimuth-banded rather than isolated to a single CPI.
    """
    num_cpi, n_range_tiles = mask.shape
    half = window // 2

    padded = np.zeros((num_cpi + 2 * half, n_range_tiles), dtype=bool)
    padded[half:half + num_cpi, :] = mask

    refined = np.zeros_like(mask)
    for i in range(num_cpi):
        local = padded[i:i + window, :]
        frac_clean = local.mean(axis=0)
        refined[i, :] = mask[i, :] & (frac_clean >= min_frac)

    return refined


def find_full_frame_clean_windows(mask, p_start, cpi_len, min_range_frac, min_window_cpi):
    """
    Find contiguous pulse-index ranges where at least min_range_frac of the
    range tiles are clean, i.e. genuinely full-frame clean intervals such as
    the known-clean pulse range used elsewhere in this project.
    """
    num_cpi, n_range_tiles = mask.shape
    frac_per_cpi = mask.mean(axis=1)
    is_globally_clean = frac_per_cpi >= min_range_frac

    windows = []
    i = 0
    while i < num_cpi:
        if is_globally_clean[i]:
            j = i
            while j < num_cpi and is_globally_clean[j]:
                j += 1
            length = j - i
            if length >= min_window_cpi:
                pulse_lo = p_start + i * cpi_len
                pulse_hi = p_start + j * cpi_len
                mean_frac = float(frac_per_cpi[i:j].mean())
                windows.append((pulse_lo, pulse_hi, length, mean_frac))
            i = j
        else:
            i += 1

    return windows


def save_sample_plots(eigvals_db, mask, output_dir, n_samples=6):
    """Save a small panel of example clean and rejected profiles for a quick visual check."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    flat_db = eigvals_db.reshape(-1, eigvals_db.shape[-1])
    flat_mask = mask.reshape(-1)

    clean_idx = np.where(flat_mask)[0]
    rejected_idx = np.where(~flat_mask)[0]

    rng = np.random.default_rng(0)
    n_each = n_samples // 2
    picks_clean = rng.choice(clean_idx, size=min(n_each, len(clean_idx)), replace=False) if len(clean_idx) else []
    picks_rejected = rng.choice(rejected_idx, size=min(n_each, len(rejected_idx)), replace=False) if len(rejected_idx) else []

    picks = list(picks_clean) + list(picks_rejected)
    labels = ["clean"] * len(picks_clean) + ["rejected"] * len(picks_rejected)

    if not picks:
        return

    ncols = 3
    nrows = int(np.ceil(len(picks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, idx, label in zip(axes, picks, labels):
        profile = flat_db[idx]
        ax.plot(np.arange(1, len(profile) + 1), profile, marker="o")
        ax.set_title("tile %d: %s" % (idx, label))
        ax.set_xlabel("Eigenvalue Index")
        ax.set_ylabel("Eigenvalue (dB)")
        ax.grid(True, alpha=0.3)

    for ax in axes[len(picks):]:
        ax.axis("off")

    fig.tight_layout()
    out_path = os.path.join(output_dir, "clean_profile_samples.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print("  Saved sample plots to %s" % out_path)


def main():
    args = parse_args()

    if not os.path.exists(args.input_file):
        print("ERROR: Input file not found: %s" % args.input_file)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("Clean Eigenvalue Profile Screening")
    print("=" * 70)
    print("Input file: %s" % args.input_file)
    print("Freq/Pol: %s-%s" % (args.freq, args.pol))

    raw = Raw(hdf5file=str(args.input_file))
    raw.parsePolarizations()

    dataset = raw.getRawDataset(args.freq, args.pol)
    total_pulses, total_range = dataset.shape

    p_start = args.pulse_start if args.pulse_start is not None else 0
    p_end = args.pulse_end if args.pulse_end is not None else total_pulses
    r_start = args.range_start if args.range_start is not None else 0
    r_end = args.range_end if args.range_end is not None else total_range

    print("Pulse interval: [%d:%d] of %d total" % (p_start, p_end, total_pulses))
    print("Range interval: [%d:%d] of %d total" % (r_start, r_end, total_range))

    n_pulses = p_end - p_start
    num_cpi = n_pulses // args.cpi_len
    tb_size = num_cpi * args.cpi_len

    print("CPI length: %d, number of CPIs: %d" % (args.cpi_len, num_cpi))

    print("Reading raw data...")
    t0 = time.time()
    raw_data = read_raw_data_batch(
        raw, args.freq, args.pol,
        pulse_slice=slice(p_start, p_start + tb_size),
        range_slice=slice(r_start, r_end),
    )
    print("  Read %.3f GB in %.2fs" % (raw_data.nbytes / 1e9, time.time() - t0))

    print("Detecting transmission-gap / edge-fill range tiles...")
    valid_range_tiles = detect_gap_range_tiles(raw_data, args.cpi_width, args.gap_threshold_frac)
    n_gap_tiles = int((~valid_range_tiles).sum())
    if n_gap_tiles:
        gap_indices = np.where(~valid_range_tiles)[0]
        print("  Excluding %d of %d range tiles as gap/edge fill: %s"
              % (n_gap_tiles, len(valid_range_tiles), gap_indices.tolist()))
    else:
        print("  No gap/edge range tiles detected")

    print("Computing eigenvalue profiles...")
    t0 = time.time()
    eigvals = compute_eigenvalue_tiles(raw_data, args.cpi_len, args.cpi_width)
    print("  Computed %d x %d tiles in %.2fs" % (eigvals.shape[0], eigvals.shape[1], time.time() - t0))

    eigvals_db = eigenvalues_to_db(eigvals, mode=args.db_mode)

    print("Applying conservative clean-tile heuristic...")
    resolved_depth = args.step_search_depth if args.step_search_depth is not None else max(args.cpi_len // 2, 1)
    print("  Step/spread search depth: top %d of %d eigenvalue indices" % (resolved_depth, args.cpi_len))
    is_clean_raw, diagnostics = classify_clean_tiles(
        eigvals_db,
        max_first_step_db=args.max_first_step_db,
        max_any_step_db=args.max_any_step_db,
        max_spread_db=args.max_spread_db,
        step_search_depth=args.step_search_depth,
    )

    # Gap/edge tiles are never real signal, so they are forced non-clean and
    # excluded from the denominator of every reported statistic below.
    is_clean_raw = is_clean_raw & valid_range_tiles[None, :]

    valid_2d = np.broadcast_to(valid_range_tiles[None, :], is_clean_raw.shape)
    total_valid_tiles = int(valid_2d.sum())
    n_clean_raw = int(is_clean_raw.sum())
    print("  Per-tile clean (before spatial filter): %d / %d valid tiles (%.1f%%), %d gap tiles excluded"
          % (n_clean_raw, total_valid_tiles, 100.0 * n_clean_raw / max(total_valid_tiles, 1),
             is_clean_raw.size - total_valid_tiles))

    print("  Diagnostic percentiles over valid tiles (10th / 50th / 90th):")
    for key in ("first_step_db", "max_step_db", "max_step_depth_db", "spread_db", "spread_depth_db"):
        vals = diagnostics[key][valid_2d]
        p10, p50, p90 = np.percentile(vals, [10, 50, 90])
        print("    %-18s %.3f / %.3f / %.3f dB" % (key, p10, p50, p90))

    is_clean_consistent = apply_spatial_consistency(
        is_clean_raw,
        window=args.consistency_window,
        min_frac=args.consistency_frac,
    )
    is_clean_consistent = is_clean_consistent & valid_range_tiles[None, :]
    n_clean_consistent = int(is_clean_consistent.sum())
    print("  Per-tile clean (after spatial filter):  %d / %d valid tiles (%.1f%%)"
          % (n_clean_consistent, total_valid_tiles, 100.0 * n_clean_consistent / max(total_valid_tiles, 1)))

    windows = find_full_frame_clean_windows(
        is_clean_consistent[:, valid_range_tiles],
        p_start=p_start,
        cpi_len=args.cpi_len,
        min_range_frac=args.min_range_frac,
        min_window_cpi=args.min_window_cpi,
    )

    print("Full-frame clean pulse windows (>= %.0f%% of range tiles clean, >= %d CPIs long):"
          % (100 * args.min_range_frac, args.min_window_cpi))
    if windows:
        for pulse_lo, pulse_hi, length_cpi, mean_frac in windows:
            print("  pulses [%d:%d]  (%d CPIs, mean clean fraction %.3f)"
                  % (pulse_lo, pulse_hi, length_cpi, mean_frac))
    else:
        print("  none found with the current thresholds")

    output_file = os.path.join(
        args.output_dir,
        "clean_profiles_%s_%s_%d_%d.h5" % (args.freq, args.pol, p_start, p_end),
    )
    print("Saving results to %s..." % output_file)

    with h5py.File(output_file, "w") as f:
        evd_grp = f.create_group("evd")
        evd_grp.create_dataset("eigenvalues_db", data=eigvals_db, compression="gzip")

        mask_grp = f.create_group("clean_mask")
        mask_grp.create_dataset("raw", data=is_clean_raw, compression="gzip")
        mask_grp.create_dataset("consistent", data=is_clean_consistent, compression="gzip")
        mask_grp.create_dataset("valid_range_tile", data=valid_range_tiles, compression="gzip")

        diag_grp = f.create_group("diagnostics")
        for key, val in diagnostics.items():
            diag_grp.create_dataset(key, data=val, compression="gzip")

        if windows:
            windows_arr = np.array(
                [(w[0], w[1], w[2], w[3]) for w in windows],
                dtype=[("pulse_start", "i8"), ("pulse_end", "i8"),
                       ("length_cpi", "i8"), ("mean_clean_fraction", "f4")],
            )
            f.create_dataset("clean_windows", data=windows_arr)

        meta_grp = f.create_group("metadata")
        meta_grp.attrs["frequency"] = args.freq
        meta_grp.attrs["polarization"] = args.pol
        meta_grp.attrs["cpi_len"] = args.cpi_len
        meta_grp.attrs["cpi_width"] = args.cpi_width
        meta_grp.attrs["db_mode"] = args.db_mode
        meta_grp.attrs["pulse_start"] = p_start
        meta_grp.attrs["pulse_end"] = p_start + tb_size
        meta_grp.attrs["range_start"] = r_start
        meta_grp.attrs["range_end"] = r_end
        meta_grp.attrs["max_first_step_db"] = args.max_first_step_db
        meta_grp.attrs["max_any_step_db"] = args.max_any_step_db
        meta_grp.attrs["max_spread_db"] = args.max_spread_db
        meta_grp.attrs["step_search_depth"] = resolved_depth
        meta_grp.attrs["consistency_window"] = args.consistency_window
        meta_grp.attrs["consistency_frac"] = args.consistency_frac
        meta_grp.attrs["min_range_frac"] = args.min_range_frac
        meta_grp.attrs["min_window_cpi"] = args.min_window_cpi
        meta_grp.attrs["gap_threshold_frac"] = args.gap_threshold_frac
        meta_grp.attrs["n_gap_tiles_excluded"] = n_gap_tiles
        meta_grp.attrs["processing_date"] = datetime.now().isoformat()

    if args.save_plots:
        save_sample_plots(eigvals_db, is_clean_consistent, args.output_dir)

    print("=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()