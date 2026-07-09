"""
compare_dither_masked_profiles.py

Generate per-tile slow-time eigenvalue profiles from a NISAR L0B file with
an optional dither-aware masking step, and plot N profiles comparing the
masked computation against the standard (unmasked) computation.

Motivation
----------
NISAR transmission gaps are dithered: the gap position shifts from pulse to
pulse, so within a single CPI tile some pulses may have a fraction of their
range samples filled with near-zero ADC values while other pulses are fully
valid. The standard SCM computation averages over all K range samples and
silently mixes gap fill into the covariance estimate, biasing eigenvalues
low and distorting the profile shape.

Masking approach
----------------
1. Per-sample validity: a sample is invalid if its magnitude falls below
   sample_mag_frac times the global median magnitude of the batch. Gap fill
   is near-zero ADC output, far below the signal median, while genuine
   low-magnitude signal samples (Rayleigh tail) are flagged at a rate below
   about one percent, which has negligible effect on the estimate.
2. Row rule: if the invalid fraction of a pulse row within the tile meets
   or exceeds row_invalid_frac (default 0.25), the entire row is treated as
   invalid, per the project convention that a heavily gapped pulse should
   not contribute to the covariance at all. Set --row-invalid-frac 1.0 to
   disable the row rule and mask only individual samples.
3. Pairwise-valid SCM: the masked SCM is still M x M (16 x 16). Entry
   (i, j) is the sum of S[i, k] * conj(S[j, k]) over samples k that are
   valid in BOTH rows, divided by the count of such samples. Rows that are
   fully masked produce structurally zero rows/columns, so they contribute
   eigenvalues at the numerical floor; when comparing profiles, ignore the
   bottom D eigenvalues where D is the number of dead rows (annotated on
   each plot).

Note: pairwise normalization does not guarantee positive semidefiniteness.
Tiny negative eigenvalues can occur and are clamped before dB conversion.
In practice with modest masked fractions the effect is negligible.

Usage
-----
  # Compare masked vs normal on the 12 most-masked tiles of an interval
  python compare_dither_masked_profiles.py input.h5 --freq A --pol HV \
      --pulse-start 813924 --pulse-end 830000 \
      --num-profiles 12 --selection most_masked \
      --output-dir ./dither_compare

  # Plot only the standard computation on random tiles
  python compare_dither_masked_profiles.py input.h5 --freq A --pol HV \
      --pulse-start 813924 --pulse-end 830000 \
      --mode normal --selection random --num-profiles 9

Keep the pulse/range interval modest (a few thousand pulses): the batch and
its magnitude are held in memory.
"""

import argparse
import os
import sys
import time

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nisar.products.readers.Raw import Raw

EPS = 1e-12


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare dither-masked vs standard eigenvalue profiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_file", help="Input NISAR L0B HDF5 file path")
    parser.add_argument("--freq", choices=["A", "B"], default="A")
    parser.add_argument("--pol", choices=["HH", "HV", "VH", "VV"], default="HV")
    parser.add_argument("--output-dir", type=str, default="./dither_compare")

    parser.add_argument("--pulse-start", type=int, default=None)
    parser.add_argument("--pulse-end", type=int, default=None)
    parser.add_argument("--range-start", type=int, default=None)
    parser.add_argument("--range-end", type=int, default=None)

    parser.add_argument("--cpi-len", type=int, default=16, help="Pulses per CPI (default: 16)")
    parser.add_argument("--cpi-width", type=int, default=250, help="Range samples per tile (default: 250)")

    parser.add_argument("--mode", choices=["compare", "normal", "masked"], default="compare",
                        help="compare overlays both computations; normal or masked plots one (default: compare)")

    parser.add_argument("--num-profiles", type=int, default=12,
                        help="Number of tile profiles to plot (default: 12)")
    parser.add_argument("--selection", choices=["most_masked", "random"], default="most_masked",
                        help="most_masked picks tiles with the highest invalid-sample fraction so the "
                             "comparison is informative; random samples uniformly (default: most_masked)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for random selection")

    parser.add_argument("--sample-mag-frac", type=float, default=0.1,
                        help="A sample is invalid if |sample| < this fraction of the global median "
                             "magnitude of the batch (default: 0.1)")
    parser.add_argument("--row-invalid-frac", type=float, default=0.25,
                        help="A pulse row with at least this fraction of invalid samples is fully "
                             "masked out of the covariance (default: 0.25; set 1.0 to disable)")

    parser.add_argument("--db-mode", choices=["raw", "top_anchored", "floor_anchored"], default="raw",
                        help="dB convention for plotting. raw keeps absolute power, which is the most "
                             "informative for masked-vs-normal comparison (default: raw)")

    return parser.parse_args()


def read_raw_data_batch(raw, freq, pol, pulse_slice, range_slice):
    """Read a batch of raw pulses using ISCE3's Raw reader (handles BFPQLUT decoding)."""
    dataset = raw.getRawDataset(freq, pol)

    pulse_start = pulse_slice.start if pulse_slice.start is not None else 0
    pulse_stop = pulse_slice.stop if pulse_slice.stop is not None else dataset.shape[0]
    range_start = range_slice.start if range_slice.start is not None else 0
    range_stop = range_slice.stop if range_slice.stop is not None else dataset.shape[1]

    return dataset[pulse_start:pulse_stop, range_start:range_stop]


def compute_scm_standard(tile):
    """Standard slow-time SCM: tile is (M pulses, K range samples), SCM = S S^H / K."""
    M, K = tile.shape
    return (tile @ tile.conj().T) / K


def compute_scm_masked(tile, sample_valid):
    """
    Pairwise-valid slow-time SCM, still M x M.

    sample_valid is an (M, K) bool array. Invalid samples are zeroed so they
    drop out of the numerator sums; each entry is normalized by the count of
    jointly valid samples for that pulse pair. Entries with zero jointly
    valid samples are set to zero.

    Returns (scm, pair_counts).
    """
    S = np.where(sample_valid, tile, 0.0)
    V = sample_valid.astype(np.float64)

    num = S @ S.conj().T
    cnt = V @ V.T

    scm = np.where(cnt > 0, num / np.maximum(cnt, 1.0), 0.0)
    return scm, cnt


def eigenvalues_desc(scm):
    """Real eigenvalues of a Hermitian(ish) SCM, sorted descending, clamped at EPS."""
    ev = np.linalg.eigvalsh(scm)
    ev = np.sort(ev.real)[::-1]
    return np.maximum(ev, EPS)


def to_db(ev, mode):
    db = 10.0 * np.log10(ev)
    if mode == "top_anchored":
        db = db - db[0]
    elif mode == "floor_anchored":
        db = db - db[-1]
    return db


def build_sample_validity(tile, global_median_mag, sample_mag_frac, row_invalid_frac):
    """
    Per-sample validity for one tile, with the row rule applied.

    Returns (sample_valid, row_invalid_fraction, dead_rows).
    """
    mag = np.abs(tile)
    sample_valid = mag >= (sample_mag_frac * global_median_mag)

    row_invalid = 1.0 - sample_valid.mean(axis=1)
    dead = row_invalid >= row_invalid_frac
    sample_valid[dead, :] = False

    return sample_valid, row_invalid, dead


def main():
    args = parse_args()

    if not os.path.exists(args.input_file):
        print("ERROR: Input file not found: %s" % args.input_file)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("Dither-Masked vs Standard Eigenvalue Profile Comparison")
    print("=" * 70)
    print("Input file: %s" % args.input_file)
    print("Freq/Pol: %s-%s, mode: %s" % (args.freq, args.pol, args.mode))

    raw = Raw(hdf5file=str(args.input_file))
    raw.parsePolarizations()

    dataset = raw.getRawDataset(args.freq, args.pol)
    total_pulses, total_range = dataset.shape

    p_start = args.pulse_start if args.pulse_start is not None else 0
    p_end = args.pulse_end if args.pulse_end is not None else total_pulses
    r_start = args.range_start if args.range_start is not None else 0
    r_end = args.range_end if args.range_end is not None else total_range

    n_pulses = p_end - p_start
    num_cpi = n_pulses // args.cpi_len
    tb_size = num_cpi * args.cpi_len

    print("Pulse interval: [%d:%d] of %d total" % (p_start, p_end, total_pulses))
    print("Range interval: [%d:%d] of %d total" % (r_start, r_end, total_range))
    print("CPI length: %d, CPIs: %d, range tile width: %d" % (args.cpi_len, num_cpi, args.cpi_width))

    print("Reading raw data...")
    t0 = time.time()
    raw_data = read_raw_data_batch(
        raw, args.freq, args.pol,
        pulse_slice=slice(p_start, p_start + tb_size),
        range_slice=slice(r_start, r_end),
    )
    print("  Read %.3f GB in %.2fs" % (raw_data.nbytes / 1e9, time.time() - t0))

    n_range_tiles = raw_data.shape[1] // args.cpi_width
    n_tiles = num_cpi * n_range_tiles

    print("Computing global magnitude reference and per-tile invalid fractions...")
    t0 = time.time()
    mag = np.abs(raw_data).astype(np.float32)
    global_median_mag = float(np.median(mag))
    sample_thr = args.sample_mag_frac * global_median_mag
    print("  Global median magnitude: %.4f, per-sample threshold: %.4f" % (global_median_mag, sample_thr))

    # Per-tile invalid-sample fraction, used for tile selection and reporting.
    invalid = mag < sample_thr
    trimmed = invalid[:num_cpi * args.cpi_len, :n_range_tiles * args.cpi_width]
    tile_invalid_frac = (
        trimmed.reshape(num_cpi, args.cpi_len, n_range_tiles, args.cpi_width)
        .mean(axis=(1, 3))
    )
    del invalid, trimmed
    print("  Done in %.2fs" % (time.time() - t0))

    frac_flat = tile_invalid_frac.reshape(-1)
    p50, p90, p99 = np.percentile(frac_flat, [50, 90, 99])
    print("  Tile invalid-fraction percentiles: p50=%.4f p90=%.4f p99=%.4f max=%.4f"
          % (p50, p90, p99, frac_flat.max()))

    # Select tiles to plot.
    n_plot = min(args.num_profiles, n_tiles)
    if args.selection == "most_masked":
        order = np.argsort(frac_flat)[::-1]
        picks = order[:n_plot]
        if frac_flat[picks[0]] <= 0:
            print("  WARNING: no tiles contain invalid samples; falling back to random selection")
            rng = np.random.default_rng(args.seed)
            picks = rng.choice(n_tiles, size=n_plot, replace=False)
    else:
        rng = np.random.default_rng(args.seed)
        picks = rng.choice(n_tiles, size=n_plot, replace=False)

    do_normal = args.mode in ("compare", "normal")
    do_masked = args.mode in ("compare", "masked")

    print("Computing eigenvalue profiles for %d selected tiles..." % n_plot)
    results = []
    for flat_idx in picks:
        cpi_idx = int(flat_idx // n_range_tiles)
        rt_idx = int(flat_idx % n_range_tiles)

        p0 = cpi_idx * args.cpi_len
        r0 = rt_idx * args.cpi_width
        tile = np.asarray(raw_data[p0:p0 + args.cpi_len, r0:r0 + args.cpi_width], dtype=np.complex128)

        entry = {
            "cpi_idx": cpi_idx,
            "rt_idx": rt_idx,
            "pulse_lo": p_start + p0,
            "invalid_frac": float(tile_invalid_frac[cpi_idx, rt_idx]),
            "ev_normal_db": None,
            "ev_masked_db": None,
            "dead_rows": 0,
        }

        if do_normal:
            ev_n = eigenvalues_desc(compute_scm_standard(tile))
            entry["ev_normal_db"] = to_db(ev_n, args.db_mode)

        if do_masked:
            sample_valid, row_invalid, dead = build_sample_validity(
                tile, global_median_mag, args.sample_mag_frac, args.row_invalid_frac)
            scm_m, _ = compute_scm_masked(tile, sample_valid)
            ev_m = eigenvalues_desc(scm_m)
            entry["ev_masked_db"] = to_db(ev_m, args.db_mode)
            entry["dead_rows"] = int(dead.sum())

        results.append(entry)

    # Console summary.
    print()
    print("%-6s %-6s %-10s %-9s %-6s %-12s" % ("CPI", "RngTB", "pulse_lo", "inval%", "dead", "max|d| top8"))
    for e in results:
        if e["ev_normal_db"] is not None and e["ev_masked_db"] is not None:
            top = min(8, args.cpi_len - e["dead_rows"])
            dmax = float(np.max(np.abs(e["ev_normal_db"][:top] - e["ev_masked_db"][:top]))) if top > 0 else float("nan")
            dstr = "%.3f dB" % dmax
        else:
            dstr = "-"
        print("%-6d %-6d %-10d %-9.4f %-6d %-12s"
              % (e["cpi_idx"], e["rt_idx"], e["pulse_lo"], 100.0 * e["invalid_frac"], e["dead_rows"], dstr))

    # Plot grid.
    ncols = 3
    nrows = int(np.ceil(n_plot / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    x = np.arange(1, args.cpi_len + 1)
    for ax, e in zip(axes, results):
        if e["ev_normal_db"] is not None:
            ax.plot(x, e["ev_normal_db"], marker="o", color="tab:blue", label="normal")
        if e["ev_masked_db"] is not None:
            ax.plot(x, e["ev_masked_db"], marker="s", linestyle="--", color="tab:orange", label="masked")
            if e["dead_rows"] > 0:
                # Eigenvalues below this line are structural zeros from fully
                # masked rows, not physical estimates.
                ax.axvline(args.cpi_len - e["dead_rows"] + 0.5, color="gray", linestyle=":", alpha=0.7)
        ax.set_title("CPI %d, RngTB %d (pulse %d)\ninvalid %.2f%%, dead rows %d"
                     % (e["cpi_idx"], e["rt_idx"], e["pulse_lo"], 100.0 * e["invalid_frac"], e["dead_rows"]),
                     fontsize=9)
        ax.set_xlabel("Eigenvalue Index")
        ax.set_ylabel("Eigenvalue (dB, %s)" % args.db_mode)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes[n_plot:]:
        ax.axis("off")

    fig.tight_layout()
    out_path = os.path.join(
        args.output_dir,
        "dither_compare_%s_%s_%d_%d_%s.png" % (args.freq, args.pol, p_start, p_end, args.mode),
    )
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print()
    print("Saved plot: %s" % out_path)

    print("=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()