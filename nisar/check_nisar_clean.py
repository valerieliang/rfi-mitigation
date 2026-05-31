#!/usr/bin/env python3
"""
check_nisar_clean.py

Screen NISAR L1 RSLC granules for RFI / quality problems and render a
human-readable RGB quicklook for each one.

Purpose
-------
The intended use is to find verified-CLEAN focused scenes that can serve
as a blank canvas for overlaying synthetic, self-labeled RFI. A scene
that already shows RFI in the focused product is also contaminated at
the raw (L0B) stage, so this acts as a screening gate before any
injection / training work.

What "clean" means here
-----------------------
A focused RSLC does not expose the slow-time eigenvalue knee directly
(focusing has already combined the pulses). What RFI DOES leave in a
focused product is what the IGARSS / EVD slides describe:
  - narrowband / wideband spikes in the range-frequency spectrum
  - bright stripes or haze in the image (transient or persistent)
This tool screens for those signatures, plus basic quality issues that
would make a scene a poor injection canvas:
  - large dark / low-backscatter fraction (rivers, water: low SNR)
  - zero-fill / data gaps
  - near-saturation

All thresholds are HEURISTIC and exposed as constants below. They are a
starting point, not calibrated truth. Treat the CLEAN / REVIEW / NOISY
flag as a triage aid; confirm borderline scenes visually with the
quicklook PNG before trusting them.

Reading notes
-------------
NISAR RSLC pixels are complex. Real spaceborne products store them as
half-precision complex (compound HDF5 type with 'r' and 'i' fields);
simulated / airborne products use full complex64. Both are handled.
Files are large, so pixels are decimated on read (strided slicing) to a
target quicklook size rather than fully loaded.

HDF5 layout (auto-discovered):
  science/<BAND>/<RSLC|SLC>/swaths/frequency<A|B>/<POL>
  where BAND is LSAR or SSAR and POL is HH, HV, VH, VV, ...

Dependencies: numpy, h5py, matplotlib.
  pip install numpy h5py matplotlib

Examples
--------
  # screen everything downloaded into data/nisar_out, write quicklooks there
  python check_nisar_clean.py

  # a specific file, larger quicklook, only frequency A
  python check_nisar_clean.py path/to/granule.h5 --max-dim 2400 --freq A

  # screen a folder and only keep scenes flagged CLEAN in the summary
  python check_nisar_clean.py --glob "data/nisar_out/*.h5"
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
    sys.exit("h5py is required. Install with: pip install h5py numpy matplotlib")

import matplotlib
matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Heuristic thresholds. Tune these for your data. They decide the
# CLEAN / REVIEW / NOISY flag; they do not affect the raw metrics.
# ----------------------------------------------------------------------
TH = {
    # Range-spectrum narrowband RFI: robust spike height in MAD units.
    "spike_review": 8.0,    # above this -> at least REVIEW
    "spike_noisy": 14.0,    # above this -> NOISY
    # Number of strong spectral spikes (narrowband tones).
    "nspike_review": 1,
    "nspike_noisy": 4,
    # Azimuth-line brightness stripes (transient RFI bursts), MAD units.
    "row_stripe_review": 8.0,
    "row_stripe_noisy": 14.0,
    # Dark / low-backscatter fraction (water, very low SNR canvas).
    "dark_frac_review": 0.35,
    "dark_frac_noisy": 0.60,
    # Zero-fill / data-gap fraction.
    "zero_frac_review": 0.02,
    "zero_frac_noisy": 0.10,
}

KNOWN_POLS = {"HH", "HV", "VH", "VV", "RH", "RV", "LH", "LV"}


# ----------------------------------------------------------------------
# HDF5 structure discovery
# ----------------------------------------------------------------------
def find_swaths_group(h5):
    """
    Locate the RSLC (or SLC) swaths group and return its path plus the
    band name. Tries the standard layout first, then walks the tree as a
    fallback for non-standard arrangements.
    """
    for band in ("LSAR", "SSAR"):
        for prod in ("RSLC", "SLC"):
            path = "science/{}/{}/swaths".format(band, prod)
            if path in h5:
                return path, band
    # Fallback: search for any ".../swaths" group containing a frequency.
    hit = {"path": None, "band": None}

    def visit(name, obj):
        if hit["path"] is not None:
            return
        if isinstance(obj, h5py.Group) and name.endswith("swaths"):
            for key in obj.keys():
                if key.lower().startswith("frequency"):
                    hit["path"] = name
                    parts = name.split("/")
                    hit["band"] = parts[1] if len(parts) > 1 else "?"
                    return

    h5.visititems(visit)
    return hit["path"], hit["band"]


def list_frequencies(h5, swaths_path):
    """Return available frequency sub-group names, e.g. ['frequencyA']."""
    grp = h5[swaths_path]
    freqs = [k for k in grp.keys() if k.lower().startswith("frequency")]
    return sorted(freqs)


def list_pols(h5, freq_path):
    """Return polarization dataset names present under a frequency group."""
    grp = h5[freq_path]
    pols = []
    # Prefer the declared list if present.
    if "listOfPolarizations" in grp:
        try:
            raw = grp["listOfPolarizations"][()]
            for v in np.atleast_1d(raw):
                s = v.decode() if isinstance(v, bytes) else str(v)
                s = s.strip().upper()
                if s in KNOWN_POLS:
                    pols.append(s)
        except Exception:
            pass
    if not pols:
        for k in grp.keys():
            if k.upper() in KNOWN_POLS and isinstance(grp[k], h5py.Dataset):
                pols.append(k.upper())
    # Stable order: co-pol first, then cross-pol.
    return sorted(pols, key=lambda p: (p[0] != p[1], p))


def is_copol(pol):
    return len(pol) == 2 and pol[0] == pol[1]


# ----------------------------------------------------------------------
# Reading + complex handling
# ----------------------------------------------------------------------
def read_slc_decimated(dset, max_dim):
    """
    Read a strided / decimated copy of a complex SLC dataset as
    complex64. Handles compound (r, i) half/float, native complex, and a
    real-only fallback (amplitude).
    """
    if dset.ndim != 2:
        raise ValueError("expected 2-D SLC, got shape {}".format(dset.shape))
    naz, nrg = dset.shape
    saz = max(1, naz // max_dim)
    srg = max(1, nrg // max_dim)
    raw = dset[::saz, ::srg]

    if raw.dtype.names:
        names = {n.lower(): n for n in raw.dtype.names}
        if "r" in names and "i" in names:
            arr = (raw[names["r"]].astype(np.float32)
                   + 1j * raw[names["i"]].astype(np.float32))
        elif "real" in names and "imag" in names:
            arr = (raw[names["real"]].astype(np.float32)
                   + 1j * raw[names["imag"]].astype(np.float32))
        else:
            first = raw.dtype.names[0]
            arr = raw[first].astype(np.float32).astype(np.complex64)
    elif np.iscomplexobj(raw):
        arr = raw.astype(np.complex64)
    else:
        # Real-valued: treat as amplitude with zero phase.
        arr = raw.astype(np.float32).astype(np.complex64)

    return arr.astype(np.complex64), saz, srg


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def mad_units(v):
    """Robust spike height of the max of v, in MAD units above median."""
    med = np.median(v)
    mad = np.median(np.abs(v - med)) + 1e-20
    return (np.max(v) - med) / (1.4826 * mad), med, mad


def range_spectrum_db(slc):
    """Mean range-frequency power spectrum (dB), averaged over azimuth."""
    s = slc - slc.mean(axis=1, keepdims=True)
    n = s.shape[1]
    win = np.hanning(max(n, 2))[None, :n]
    f = np.fft.fftshift(np.fft.fft(s * win, axis=1), axes=1)
    p = (np.abs(f) ** 2).mean(axis=0)
    return 10.0 * np.log10(p + 1e-20)


def spectral_flatness(spec_db):
    """Spectral flatness in [0, 1]; ~1 is white, low is peaky."""
    p = 10.0 ** (spec_db / 10.0)
    gmean = np.exp(np.mean(np.log(p + 1e-20)))
    amean = np.mean(p) + 1e-20
    return float(gmean / amean)


def compute_metrics(slc):
    """Compute RFI / quality metrics for one polarization channel."""
    inten = (np.abs(slc) ** 2).astype(np.float32)

    # Range-spectrum narrowband RFI.
    spec_db = range_spectrum_db(slc)
    spike, smed, smad = mad_units(spec_db)
    n_spikes = int(np.sum(spec_db > smed + 6.0 * 1.4826 * smad))
    flat = spectral_flatness(spec_db)

    # Brightness stripes. Rows = azimuth lines (transient bursts),
    # cols = range bins (persistent features).
    row_mean = inten.mean(axis=1)
    col_mean = inten.mean(axis=0)
    row_stripe, _, _ = mad_units(row_mean)
    col_stripe, _, _ = mad_units(col_mean)

    # Quality: dark fraction (water / low SNR), zero-fill, dynamic range.
    amp_db = 10.0 * np.log10(inten + 1e-20)
    med_db = np.median(amp_db)
    dark_frac = float(np.mean(amp_db < med_db - 6.0))
    zero_frac = float(np.mean(inten == 0))

    return {
        "spec_db": spec_db,
        "row_mean": row_mean,
        "col_mean": col_mean,
        "spike": float(spike),
        "n_spikes": n_spikes,
        "flatness": float(flat),
        "row_stripe": float(row_stripe),
        "col_stripe": float(col_stripe),
        "dark_frac": dark_frac,
        "zero_frac": zero_frac,
        "mean_db": float(med_db),
    }


def verdict(m):
    """Turn metrics into a CLEAN / REVIEW / NOISY flag with reasons."""
    reasons = []
    level = 0  # 0 clean, 1 review, 2 noisy

    def bump(to, reason):
        nonlocal level
        level = max(level, to)
        reasons.append(reason)

    if m["spike"] >= TH["spike_noisy"]:
        bump(2, "strong range-spectrum spike ({:.1f} MAD)".format(m["spike"]))
    elif m["spike"] >= TH["spike_review"]:
        bump(1, "range-spectrum spike ({:.1f} MAD)".format(m["spike"]))

    if m["n_spikes"] >= TH["nspike_noisy"]:
        bump(2, "{} spectral tones".format(m["n_spikes"]))
    elif m["n_spikes"] >= TH["nspike_review"]:
        bump(1, "{} spectral tone(s)".format(m["n_spikes"]))

    if m["row_stripe"] >= TH["row_stripe_noisy"]:
        bump(2, "bright azimuth stripe ({:.1f} MAD)".format(m["row_stripe"]))
    elif m["row_stripe"] >= TH["row_stripe_review"]:
        bump(1, "azimuth stripe ({:.1f} MAD)".format(m["row_stripe"]))

    if m["dark_frac"] >= TH["dark_frac_noisy"]:
        bump(2, "{:.0f}% dark/low-backscatter".format(100 * m["dark_frac"]))
    elif m["dark_frac"] >= TH["dark_frac_review"]:
        bump(1, "{:.0f}% dark/low-backscatter".format(100 * m["dark_frac"]))

    if m["zero_frac"] >= TH["zero_frac_noisy"]:
        bump(2, "{:.1f}% zero-fill/gaps".format(100 * m["zero_frac"]))
    elif m["zero_frac"] >= TH["zero_frac_review"]:
        bump(1, "{:.1f}% zero-fill/gaps".format(100 * m["zero_frac"]))

    flag = ["CLEAN", "REVIEW", "NOISY"][level]
    return flag, ("; ".join(reasons) if reasons else "no issues flagged")


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------
def db_stretch(inten, plow=5.0, phigh=95.0, gamma=1.6):
    """Convert intensity to a normalized [0, 1] display layer in dB."""
    db = 10.0 * np.log10(inten + 1e-20)
    lo, hi = np.nanpercentile(db, [plow, phigh])
    y = np.clip((db - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    return y ** (1.0 / gamma)


def make_rgb(channels):
    """
    Build an RGB quicklook from a dict {pol: slc}.

    Dual-pol false color: R = co-pol, G = cross-pol, B = co-pol. Volume
    scatterers (vegetation, high cross-pol) appear green; smooth
    surfaces appear magenta. Single-pol: grayscale.
    """
    copol = next((p for p in channels if is_copol(p)), None)
    crosspol = next((p for p in channels if not is_copol(p)), None)

    if copol is not None and crosspol is not None:
        r = db_stretch(np.abs(channels[copol]) ** 2)
        g = db_stretch(np.abs(channels[crosspol]) ** 2)
        b = r
        rgb = np.dstack([r, g, b])
        label = "R={} G={} B={}".format(copol, crosspol, copol)
    else:
        only = copol or crosspol or list(channels)[0]
        gray = db_stretch(np.abs(channels[only]) ** 2)
        rgb = np.dstack([gray, gray, gray])
        label = "grayscale ({})".format(only)
    return rgb, label


def save_quicklook(out_png, rgb, rgb_label, metrics, flag, reasons, title):
    """Write a diagnostics figure: quicklook + range spectrum + stripes."""
    fig = plt.figure(figsize=(14, 6))

    ax0 = fig.add_subplot(1, 3, 1)
    ax0.imshow(rgb, aspect="auto", origin="upper")
    ax0.set_title("Quicklook  [{}]".format(rgb_label), fontsize=9)
    ax0.set_xlabel("range (decimated)")
    ax0.set_ylabel("azimuth (decimated)")

    ax1 = fig.add_subplot(1, 3, 2)
    spec = metrics["spec_db"]
    ax1.plot(spec, lw=0.7)
    ax1.set_title("Range power spectrum  (spike {:.1f} MAD, "
                  "{} tones)".format(metrics["spike"], metrics["n_spikes"]),
                  fontsize=9)
    ax1.set_xlabel("range frequency bin")
    ax1.set_ylabel("power (dB)")

    ax2 = fig.add_subplot(1, 3, 3)
    ax2.plot(metrics["row_mean"], lw=0.7, label="per-azimuth mean")
    ax2.set_title("Azimuth brightness  (stripe {:.1f} "
                  "MAD)".format(metrics["row_stripe"]), fontsize=9)
    ax2.set_xlabel("azimuth line")
    ax2.set_ylabel("mean intensity")
    ax2.legend(fontsize=7)

    fig.suptitle("{}\n{}  -  {}".format(title, flag, reasons),
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)

    # Also save a standalone RGB image.
    rgb_only = os.path.splitext(out_png)[0] + "_rgb.png"
    plt.imsave(rgb_only, np.clip(rgb, 0, 1))
    return rgb_only


# ----------------------------------------------------------------------
# Per-file assessment
# ----------------------------------------------------------------------
def assess_file(path, out_dir, max_dim, want_freq):
    """Open one granule, compute metrics, render quicklook, return a row."""
    base = os.path.splitext(os.path.basename(path))[0]
    result = {"file": os.path.basename(path)}

    with h5py.File(path, "r") as h5:
        swaths, band = find_swaths_group(h5)
        if swaths is None:
            result.update({"flag": "ERROR",
                           "reasons": "no RSLC/SLC swaths group found"})
            return result

        freqs = list_frequencies(h5, swaths)
        if want_freq:
            freqs = [f for f in freqs
                     if f.lower().endswith(want_freq.lower())] or freqs
        freq = freqs[0]
        freq_path = "{}/{}".format(swaths, freq)
        pols = list_pols(h5, freq_path)
        if not pols:
            result.update({"flag": "ERROR",
                           "reasons": "no polarization datasets found"})
            return result

        # Read each pol decimated; metrics come from the co-pol if present.
        channels = {}
        saz = srg = 1
        for pol in pols:
            dset = h5["{}/{}".format(freq_path, pol)]
            slc, saz, srg = read_slc_decimated(dset, max_dim)
            channels[pol] = slc

        metric_pol = next((p for p in pols if is_copol(p)), pols[0])
        m = compute_metrics(channels[metric_pol])
        flag, reasons = verdict(m)

        rgb, rgb_label = make_rgb(channels)
        title = "{}  [{} {} {}]  decim {}x{}".format(
            base, band, freq, "/".join(pols), saz, srg)
        out_png = os.path.join(out_dir, base + "_quicklook.png")
        rgb_only = save_quicklook(out_png, rgb, rgb_label, m, flag,
                                  reasons, title)

        result.update({
            "band": band,
            "freq": freq,
            "pols": "/".join(pols),
            "metric_pol": metric_pol,
            "spike_mad": round(m["spike"], 2),
            "n_spikes": m["n_spikes"],
            "flatness": round(m["flatness"], 4),
            "row_stripe_mad": round(m["row_stripe"], 2),
            "col_stripe_mad": round(m["col_stripe"], 2),
            "dark_frac": round(m["dark_frac"], 3),
            "zero_frac": round(m["zero_frac"], 4),
            "median_db": round(m["mean_db"], 2),
            "flag": flag,
            "reasons": reasons,
            "quicklook": os.path.basename(out_png),
            "rgb": os.path.basename(rgb_only),
        })
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Screen NISAR RSLC granules for RFI / quality and "
                    "render RGB quicklooks.")
    parser.add_argument("inputs", nargs="*",
                        help="Specific .h5 files. If omitted, uses --glob.")
    parser.add_argument("--glob", default="data/nisar_out/*.h5",
                        help="Glob for granules when no files are listed.")
    parser.add_argument("--out-dir", default="data/nisar_qc",
                        help="Directory for quicklook PNGs and summary CSV.")
    parser.add_argument("--max-dim", type=int, default=1600,
                        help="Target longest decimated dimension for read.")
    parser.add_argument("--freq", default=None,
                        help="Frequency to use: A or B (default: first found).")
    args = parser.parse_args()

    files = args.inputs if args.inputs else sorted(glob.glob(args.glob))
    if not files:
        sys.exit("No input files. Pass .h5 paths or set --glob.")

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for path in files:
        print("Assessing {} ...".format(os.path.basename(path)))
        try:
            row = assess_file(path, args.out_dir, args.max_dim, args.freq)
        except Exception as exc:
            row = {"file": os.path.basename(path), "flag": "ERROR",
                   "reasons": "{}: {}".format(type(exc).__name__, exc)}
        rows.append(row)
        print("  -> {}  {}".format(row.get("flag", "?"),
                                   row.get("reasons", "")))

    # Write summary CSV.
    fields = ["file", "flag", "reasons", "band", "freq", "pols", "metric_pol",
              "spike_mad", "n_spikes", "flatness", "row_stripe_mad",
              "col_stripe_mad", "dark_frac", "zero_frac", "median_db",
              "quicklook", "rgb"]
    csv_path = os.path.join(args.out_dir, "nisar_qc_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    clean = [r for r in rows if r.get("flag") == "CLEAN"]
    review = [r for r in rows if r.get("flag") == "REVIEW"]
    noisy = [r for r in rows if r.get("flag") == "NOISY"]
    err = [r for r in rows if r.get("flag") == "ERROR"]
    print("\nSummary: {} CLEAN, {} REVIEW, {} NOISY, {} ERROR".format(
        len(clean), len(review), len(noisy), len(err)))
    print("Wrote {} and quicklook PNGs in {}/".format(csv_path, args.out_dir))
    if clean:
        print("\nClean canvases to inject into:")
        for r in clean:
            print("  {}".format(r["file"]))


if __name__ == "__main__":
    main()