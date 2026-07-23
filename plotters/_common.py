#!/usr/bin/env python
"""
plotters/_common.py

Shared helpers for the data-structure plotters (plot_metrics.py, plot_profiles.py).

Everything in this project's plotting flow speaks the same canonical, flat,
per-CPI HDF5 layout written by generate_amazon_data.py / generate_mountain_data.py
/ select_clean.py:

    eigenvalues      (n_tiles, cpi_len)   linear, descending
    diagonal         (n_tiles, cpi_len)   linear, unnormalized SCM diagonal   [optional]
    diag_valid_idx   (n_tiles, cpi_len)   bool, per-index validity            [optional]
    diag_valid_frac  (n_tiles,)           float                              [optional]
    labels           (n_tiles,)           knee, 0 = clean                     [optional]
    jsr_db           (n_tiles, max_bands) per-band JSR, NaN-padded            [optional]
    band_rows        (n_tiles, max_bands) local pulse row of each band, -1    [optional]
    signal_power_db  (n_tiles,)                                               [optional]
    valid_fraction   (n_tiles,)                                               [optional]
    tile_pulse       (n_tiles,)           absolute pulse index of the tile
    tile_range       (n_tiles,)           absolute range index of the tile
    cpi              (n_tiles, cpi_len, cpi_width) complex64 (--save-cpi only) [optional]

plus attrs including granule_path, pulse_start/pulse_end, range_start/range_end,
frequency, polarization, cpi_len, cpi_width, caltone_removed, gap_exclusion_used,
off_diag_overlap_ratio, diag_valid_ratio.

DESIGN: read everything from the H5. Only reach back to the original L0B granule
(via the granule_path attr) when a needed field is genuinely NOT stored -- e.g. a
file with no `diagonal` dataset, or an SCM heatmap when no `cpi` dataset was saved.
On that fallback path the instrument caltone is removed first (mirroring
score_scene.py / select_clean.py), so recomputed features match the stored,
caltone-removed ones. The real gap-exclusion covariance from read_nisar_isce3.py is
reused rather than reconstructed.
"""

import os
import sys

import numpy as np
import h5py


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

EPS = 1e-12
N_KEEP_DEFAULT = 12          # leading eigenvalues used as features (drops dithered tail)
CPI_PULSES = 16              # pulses per CPI = number of eigenvalues
CPI_RANGE = 250              # range samples per CPI

# Caltone removal (identical to score_scene.py / select_clean.py). Only used on
# the rare L0B fallback path; the preprocessed data is already caltone-removed.
CALTONE_WINDOW_SIZE = 64
CALTONE_DEFAULT_FREQ_HZ = 1214.883e6   # matches isce3's caltone_frequency_from_raw
CALTONE_LO_HZ = 1200e6
CALTONE_CLOCK_HZ = 240e6


# ---------------------------------------------------------------------------
# dB helpers
# ---------------------------------------------------------------------------

def eigvals_to_db(eigvals):
    """Convert linear eigenvalues (or any linear power) to dB."""
    out = 10.0 * np.log10(np.clip(np.asarray(eigvals, dtype=np.float64), EPS, None))
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# GLOBAL CLEANLINESS METRICS (moved verbatim from compare_global_metrics.py)
# ---------------------------------------------------------------------------

def compute_condition_number_db(eig_kept):
    """
    Condition number in dB: largest / smallest kept eigenvalue, in dB.

    eig_kept : (n_cpi, n_keep) linear eigenvalues, descending along axis 1.
    Returns  : (n_cpi,) dB.
    """
    eig_max = eig_kept[:, 0]
    eig_min_kept = eig_kept[:, -1]
    with np.errstate(divide="ignore", invalid="ignore"):
        cond_db = 10.0 * np.log10(eig_max / eig_min_kept)
    return cond_db


def compute_effective_rank(eig_kept):
    """
    Effective rank via Shannon entropy of the normalized kept eigenvalues
    (Roy & Vetterli): exp(entropy(eig_kept / sum(eig_kept))).

    eig_kept : (n_cpi, n_keep) linear eigenvalues.
    Returns  : (n_cpi,) effective rank in [1, n_keep].
    """
    eig_sum = eig_kept.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = eig_kept / eig_sum
        plogp = np.where(p > 0, p * np.log(p), 0.0)
        entropy = -plogp.sum(axis=1)
    return np.exp(entropy)


def compute_median_max_ratio(eig_kept, definition="median_over_max"):
    """
    Median/max eigenvalue ratio among the kept eigenvalues.

    definition:
      "median_over_max" (default): median/max, in (0, 1]
      "max_over_median": max/median, in [1, inf)
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


# Column order used by both the tracking CSVs and every row written into them.
TRACKING_TABLE_FIELDS = ["run", "mean", "median", "std", "min", "max", "p05", "p95", "iqr", "count"]


def update_metrics_table(metrics_dir, metric_name, run_name, summary):
    """
    Insert or update a single named row in metrics_dir/<metric_name>.csv.

    Re-using a run_name overwrites its row in place; a new run_name appends a
    new row. Row order otherwise follows first appearance.
    """
    import csv

    os.makedirs(metrics_dir, exist_ok=True)
    path = os.path.join(metrics_dir, f"{metric_name}.csv")

    rows = {}
    order = []
    if os.path.exists(path):
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows[r["run"]] = r
                order.append(r["run"])

    if run_name not in rows:
        order.append(run_name)

    if summary.get("count", 0) == 0:
        rows[run_name] = {"run": run_name, "count": 0}
    else:
        rows[run_name] = {
            "run": run_name,
            "mean": summary["mean"],
            "median": summary["median"],
            "std": summary["std"],
            "min": summary["min"],
            "max": summary["max"],
            "p05": summary["p05"],
            "p95": summary["p95"],
            "iqr": summary["iqr"],
            "count": summary["count"],
        }

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRACKING_TABLE_FIELDS)
        writer.writeheader()
        for name in order:
            writer.writerow(rows[name])

    return path


def update_all_metrics_tables(metrics_dir, run_name, summaries):
    """Call update_metrics_table once per metric summary. Returns CSV paths."""
    return [update_metrics_table(metrics_dir, s["metric"], run_name, s) for s in summaries]


# ---------------------------------------------------------------------------
# FLAT-H5 CHANNEL LOADER
# ---------------------------------------------------------------------------

class Channel:
    """
    In-memory view of one flat per-CPI HDF5 file. Small per-tile arrays are read
    eagerly; the (potentially large) complex `cpi` dataset is left on disk and
    fetched on demand via fetch_cpi_tiles().
    """

    def __init__(self, path=None):
        # path=None builds an empty shell to be filled by an alternate
        # constructor (e.g. channel_from_l0b); all per-tile fields default None.
        self.path = path
        self.attrs = {}
        self.eigenvalues = None
        self.diagonal = None
        self.diag_valid_idx = None
        self.diag_valid_frac = None
        self.labels = None
        self.jsr_db = None
        self.band_rows = None
        self.signal_power_db = None
        self.valid_fraction = None
        self.tile_pulse = None
        self.tile_range = None
        self.has_cpi = False
        if path is None:
            return
        with h5py.File(path, "r") as f:
            self.attrs = dict(f.attrs)
            if "eigenvalues" not in f:
                raise KeyError(
                    f"{path} has no 'eigenvalues' dataset; not a per-CPI feature "
                    f"file. Top-level keys: {list(f.keys())}"
                )
            self.eigenvalues = f["eigenvalues"][:]
            self.diagonal = f["diagonal"][:] if "diagonal" in f else None
            self.diag_valid_idx = f["diag_valid_idx"][:] if "diag_valid_idx" in f else None
            self.labels = f["labels"][:] if "labels" in f else None
            self.jsr_db = f["jsr_db"][:] if "jsr_db" in f else None
            self.band_rows = f["band_rows"][:] if "band_rows" in f else None
            self.signal_power_db = f["signal_power_db"][:] if "signal_power_db" in f else None
            self.valid_fraction = f["valid_fraction"][:] if "valid_fraction" in f else None
            self.tile_pulse = f["tile_pulse"][:] if "tile_pulse" in f else None
            self.tile_range = f["tile_range"][:] if "tile_range" in f else None
            self.has_cpi = "cpi" in f

            if "diag_valid_frac" in f:
                self.diag_valid_frac = f["diag_valid_frac"][:]
            elif self.diag_valid_idx is not None:
                self.diag_valid_frac = self.diag_valid_idx.mean(axis=1).astype(np.float32)
            else:
                self.diag_valid_frac = None

    # -- convenience -------------------------------------------------------

    @property
    def freq(self):
        return self.attrs.get("frequency", "?")

    @property
    def pol(self):
        return self.attrs.get("polarization", "?")

    @property
    def n_tiles(self):
        return self.eigenvalues.shape[0]

    @property
    def cpi_len(self):
        return int(self.attrs.get("cpi_len", self.eigenvalues.shape[1]))

    @property
    def cpi_width(self):
        return int(self.attrs.get("cpi_width", CPI_RANGE))

    @property
    def tag(self):
        return f"{self.freq}_{self.pol}"


def load_channel(path):
    """Load one flat per-CPI HDF5 file into a Channel."""
    return Channel(path)


def describe_coverage(ch, printer=print):
    """
    Report the pulse/range extent and tile grid covered by a Channel, derived
    from its tile_pulse/tile_range datasets and attrs. Returns the info dict.
    """
    info = {
        "path": ch.path,
        "freq": ch.freq,
        "pol": ch.pol,
        "n_tiles": ch.n_tiles,
        "cpi_len": ch.cpi_len,
        "cpi_width": ch.cpi_width,
        "caltone_removed": bool(ch.attrs.get("caltone_removed", False)),
        "granule": ch.attrs.get("granule", ch.attrs.get("granule_path", "?")),
    }
    if ch.tile_pulse is not None and ch.tile_pulse.size:
        info["pulse_min"] = int(ch.tile_pulse.min())
        info["pulse_max"] = int(ch.tile_pulse.max()) + ch.cpi_len
        info["n_pulse_tiles"] = int(np.unique(ch.tile_pulse).size)
    if ch.tile_range is not None and ch.tile_range.size:
        info["range_min"] = int(ch.tile_range.min())
        info["range_max"] = int(ch.tile_range.max()) + ch.cpi_width
        info["n_range_tiles"] = int(np.unique(ch.tile_range).size)

    if printer is not None:
        printer(f"  [{ch.tag}] {ch.n_tiles} tiles from {os.path.basename(str(info['granule']))}")
        printer(f"    caltone_removed={info['caltone_removed']}  "
                f"cpi={ch.cpi_len}x{ch.cpi_width}")
        if "pulse_min" in info:
            printer(f"    pulse  [{info['pulse_min']}:{info['pulse_max']}]  "
                    f"({info['n_pulse_tiles']} distinct pulse tiles)")
        if "range_min" in info:
            printer(f"    range  [{info['range_min']}:{info['range_max']}]  "
                    f"({info['n_range_tiles']} distinct range tiles)")
    return info


# ---------------------------------------------------------------------------
# CALTONE (L0B fallback only) -- copied from score_scene.py / select_clean.py
# ---------------------------------------------------------------------------

def parse_caltone_freq_from_drt(raw, txrx_pol):
    """
    Local fallback for isce3's caltone_frequency_from_raw: caltone frequency
    (Hz) for one TxRx polarization from the DRT CALTONE phase-step telemetry,
    with CALTONE_DEFAULT_FREQ_HZ when the path is missing.
    """
    path = (f'{raw.TelemetryPath}/DRT/MISC/'
            f'CP_IFSW_CALTONE_PHASE_STEP_{txrx_pol[1]}')
    with h5py.File(raw.filename, mode='r', swmr=True) as f:
        try:
            ds = f[path]
        except KeyError:
            print(f'    caltone: missing "{path}"; using default '
                  f'{CALTONE_DEFAULT_FREQ_HZ} Hz')
            return CALTONE_DEFAULT_FREQ_HZ
        i_cal = np.median(ds[()]).astype(int)
        return (i_cal / 2 ** 32) * CALTONE_CLOCK_HZ + CALTONE_LO_HZ


def build_tone_remover(raw, freq, pol, num_rng_samples):
    """
    Construct a ToneRemover sized to the full range width for one channel, plus
    the caltone frequency used (for provenance). ToneRemover is imported lazily
    so this module imports without isce3.
    """
    from isce3.focus import ToneRemover
    try:
        from nisar.products.readers.Raw import caltone_frequency_from_raw
    except ImportError:
        caltone_frequency_from_raw = None

    tx_pol = pol[0]
    fc, fs, _, _ = raw.getChirpParameters(freq, tx_pol)
    if caltone_frequency_from_raw is not None:
        caltone_freq = caltone_frequency_from_raw(raw, pol)
    else:
        caltone_freq = parse_caltone_freq_from_drt(raw, pol)
    remover = ToneRemover((caltone_freq - fc) / fs, num_rng_samples, CALTONE_WINDOW_SIZE)
    return remover, caltone_freq


# ---------------------------------------------------------------------------
# L0B FALLBACK -- recompute a field from the source granule when it is missing
# ---------------------------------------------------------------------------

def _import_reader():
    """
    Lazily import the project's real NISAR reader/covariance utilities (the same
    set score_scene.py uses). read_nisar_isce3.py lives in the repo root, one
    level up from this plotters/ package, so put the parent on sys.path first.
    """
    parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from read_nisar_isce3 import (
        read_raw_data_batch,
        get_subswath_mask,
        compute_scm_and_eigs,
    )
    from nisar.products.readers.Raw import Raw
    return read_raw_data_batch, get_subswath_mask, compute_scm_and_eigs, Raw


def _scm_unmasked(cpi):
    """Plain (M @ M^H) / K SCM for one complex CPI tile (no gap exclusion)."""
    k = cpi.shape[1]
    return ((cpi @ cpi.conj().T) / k).astype(np.complex64)


def _granule_path(ch):
    """Resolve the source granule for a Channel, or raise a clear error."""
    gp = ch.attrs.get("granule_path")
    if gp and os.path.exists(gp):
        return gp
    # granule_path may be a cluster path; fall back to a local file named after it.
    gname = ch.attrs.get("granule")
    if gname and os.path.exists(gname):
        return gname
    raise FileNotFoundError(
        f"{ch.path} is missing a field that must be recomputed from the source "
        f"L0B granule, but the granule could not be found (granule_path="
        f"{gp!r}, granule={gname!r}). Point --l0b at the granule, or use a file "
        f"that stores the field."
    )


def _tone_removed_reader(ch):
    """
    Open the source granule and return (raw, remover, total_range, off_ratio,
    diag_ratio, use_mask). The caltone remover is built only if the stored data
    was caltone-removed.
    """
    read_raw_data_batch, get_subswath_mask, compute_scm_and_eigs, Raw = _import_reader()
    granule = _granule_path(ch)
    raw = Raw(hdf5file=granule)
    total_range = raw.getRawDataset(ch.freq, ch.pol).shape[1]

    remover = None
    if ch.attrs.get("caltone_removed", False):
        remover, caltone_freq = build_tone_remover(raw, ch.freq, ch.pol, total_range)
        print(f"    caltone removal ON  (f_caltone = {caltone_freq/1e6:.4f} MHz)")
    else:
        print("    caltone removal OFF (source data was not caltone-removed)")

    off_ratio = float(ch.attrs.get("off_diag_overlap_ratio", 0.25))
    diag_ratio = float(ch.attrs.get("diag_valid_ratio", 0.20))
    use_mask = bool(ch.attrs.get("gap_exclusion_used", False))
    return (raw, remover, total_range, off_ratio, diag_ratio, use_mask,
            read_raw_data_batch, get_subswath_mask, compute_scm_and_eigs)


def _read_cpi_from_l0b(ch, indices):
    """
    Read the complex CPI tiles at the given tile indices from the source L0B,
    caltone-removed, grouped by shared pulse block for efficiency.

    Yields (idx, cpi_complex, cpi_mask_or_None) for each requested index.
    """
    (raw, remover, total_range, off_ratio, diag_ratio, use_mask,
     read_raw_data_batch, get_subswath_mask, _scm) = _tone_removed_reader(ch)

    cpi_len, cpi_width = ch.cpi_len, ch.cpi_width
    # Group requested tiles by their absolute pulse start so each 16-pulse,
    # full-width read serves every range window that shares it.
    by_pulse = {}
    for idx in indices:
        p0 = int(ch.tile_pulse[idx])
        by_pulse.setdefault(p0, []).append(idx)

    for p0, idx_list in sorted(by_pulse.items()):
        p1 = p0 + cpi_len
        lines = np.ascontiguousarray(
            read_raw_data_batch(raw, ch.freq, ch.pol, slice(p0, p1), slice(0, total_range))
        ).astype(np.complex64)
        if remover is not None:
            for ip in range(lines.shape[0]):
                lines[ip] = remover.remove_tone(lines[ip])

        mask_full = None
        if use_mask:
            mask_full = get_subswath_mask(
                raw, ch.freq, ch.pol, np.arange(p0, p1), np.arange(0, total_range)
            )

        for idx in idx_list:
            r0 = int(ch.tile_range[idx])
            r1 = r0 + cpi_width
            cpi = np.ascontiguousarray(lines[:, r0:r1])
            m = np.ascontiguousarray(mask_full[:, r0:r1]) if mask_full is not None else None
            yield idx, cpi, m


def ensure_diagonal(ch):
    """
    Make sure ch.diagonal and ch.diag_valid_idx are populated. If the stored
    file has no `diagonal`, recompute it (and the per-index validity) for every
    tile from the caltone-removed source L0B. No-op when already present.
    """
    if ch.diagonal is not None:
        return ch.diagonal

    print(f"  [{ch.tag}] no 'diagonal' dataset stored; recomputing from L0B ...")
    n = ch.n_tiles
    diag = np.zeros((n, ch.cpi_len), dtype=np.float32)
    valid = np.ones((n, ch.cpi_len), dtype=bool)

    # _import_reader() puts the repo root on sys.path and returns the real
    # gap-exclusion SCM used by the rest of the project.
    _rrb, _gsm, compute_scm_and_eigs, _Raw = _import_reader()

    off_ratio = float(ch.attrs.get("off_diag_overlap_ratio", 0.25))
    diag_ratio = float(ch.attrs.get("diag_valid_ratio", 0.20))

    done = 0
    for idx, cpi, mask in _read_cpi_from_l0b(ch, range(n)):
        _scm, _eig, diag_lin, diag_valid = compute_scm_and_eigs(cpi, mask, off_ratio, diag_ratio)
        diag[idx] = diag_lin.astype(np.float32)
        valid[idx] = diag_valid
        done += 1
        if done % 1000 == 0:
            print(f"    recomputed diagonal for {done}/{n} tiles")

    ch.diagonal = diag
    ch.diag_valid_idx = valid
    if ch.diag_valid_frac is None:
        ch.diag_valid_frac = valid.mean(axis=1).astype(np.float32)
    return ch.diagonal


def fetch_cpi_tiles(ch, indices):
    """
    Return a list of (idx, scm_complex) for the given tile indices, sourcing the
    complex SCM from the stored `cpi` dataset when present (no isce3 needed),
    else from the caltone-removed source L0B. Used by the SCM heatmap plot.
    """
    out = []
    if ch.has_cpi:
        # The stored cpi tiles were selected as valid at generation time, so an
        # unmasked (M @ M^H)/K SCM matches how they were featurized -- and this
        # keeps stored-cpi heatmaps free of any isce3 dependency.
        with h5py.File(ch.path, "r") as f:
            cpi_ds = f["cpi"]
            for idx in indices:
                cpi = np.asarray(cpi_ds[idx], dtype=np.complex64)
                out.append((idx, _scm_unmasked(cpi)))
        return out

    # No cpi dataset -> read (caltone-removed) from L0B and use the real SCM.
    _rrb, _gsm, compute_scm_and_eigs, _Raw = _import_reader()
    off_ratio = float(ch.attrs.get("off_diag_overlap_ratio", 0.25))
    diag_ratio = float(ch.attrs.get("diag_valid_ratio", 0.20))
    for idx, cpi, mask in _read_cpi_from_l0b(ch, list(indices)):
        scm, _eig, _diag, _dv = compute_scm_and_eigs(cpi, mask, off_ratio, diag_ratio)
        out.append((idx, scm))
    return out


# ---------------------------------------------------------------------------
# DIRECT L0B -> Channel (for plotting a granule with no preprocessed file yet)
# ---------------------------------------------------------------------------

def channel_from_l0b(granule, freq, pol, *,
                     pulse_start=None, pulse_end=None,
                     range_start=None, range_end=None,
                     cpi_len=CPI_PULSES, cpi_width=CPI_RANGE,
                     remove_caltone=True, use_mask=False,
                     off_diag_overlap_ratio=0.25, diag_valid_ratio=0.20,
                     pulse_chunk=1600):
    """
    Tile a raw NISAR L0B granule into cpi_len x cpi_width CPIs over the requested
    (or full) window, caltone-remove it, and return a Channel populated with the
    per-tile eigenvalues / diagonal / validity -- the same shape the flat H5
    files carry -- so the profile plots work identically on a raw granule.
    """
    read_raw_data_batch, get_subswath_mask, compute_scm_and_eigs, Raw = _import_reader()

    raw = Raw(hdf5file=granule)
    dataset = raw.getRawDataset(freq, pol)
    total_pulses, total_range = dataset.shape

    p0 = 0 if pulse_start is None else max(0, pulse_start)
    p1 = total_pulses if pulse_end is None else min(pulse_end, total_pulses)
    r0 = 0 if range_start is None else max(0, range_start)
    r1 = total_range if range_end is None else min(range_end, total_range)

    n_pt = (p1 - p0) // cpi_len
    n_rt = (r1 - r0) // cpi_width
    if n_pt <= 0 or n_rt <= 0:
        raise ValueError(f"window {p1 - p0}x{r1 - r0} smaller than one CPI {cpi_len}x{cpi_width}")

    remover = None
    if remove_caltone:
        remover, caltone_freq = build_tone_remover(raw, freq, pol, total_range)
        print(f"    caltone removal ON  (f_caltone = {caltone_freq/1e6:.4f} MHz)")
    else:
        print("    caltone removal OFF")

    n_tiles = n_pt * n_rt
    eig = np.zeros((n_tiles, cpi_len), dtype=np.float32)
    diag = np.zeros((n_tiles, cpi_len), dtype=np.float32)
    valid = np.ones((n_tiles, cpi_len), dtype=bool)
    tile_pulse = np.zeros(n_tiles, dtype=np.int32)
    tile_range = np.zeros(n_tiles, dtype=np.int32)

    chunk_tiles = max(1, pulse_chunk // cpi_len)
    k = 0
    for chunk_start in range(0, n_pt, chunk_tiles):
        n_here = min(chunk_tiles, n_pt - chunk_start)
        cp0 = p0 + chunk_start * cpi_len
        cp1 = cp0 + n_here * cpi_len

        lines = np.ascontiguousarray(
            read_raw_data_batch(raw, freq, pol, slice(cp0, cp1), slice(0, total_range))
        ).astype(np.complex64)
        if remover is not None:
            for ip in range(lines.shape[0]):
                lines[ip] = remover.remove_tone(lines[ip])
        chunk = lines[:, r0:r1]

        mask_chunk = (get_subswath_mask(raw, freq, pol, np.arange(cp0, cp1),
                                        np.arange(r0, r1)) if use_mask else None)

        for lp in range(n_here):
            lp0, lp1 = lp * cpi_len, (lp + 1) * cpi_len
            for rt in range(n_rt):
                lr0, lr1 = rt * cpi_width, (rt + 1) * cpi_width
                cpi = np.ascontiguousarray(chunk[lp0:lp1, lr0:lr1])
                cmask = (np.ascontiguousarray(mask_chunk[lp0:lp1, lr0:lr1])
                         if mask_chunk is not None else None)
                _scm, eigvals, diag_lin, diag_valid = compute_scm_and_eigs(
                    cpi, cmask, off_diag_overlap_ratio, diag_valid_ratio)
                eig[k] = eigvals
                diag[k] = diag_lin.astype(np.float32)
                valid[k] = diag_valid
                tile_pulse[k] = cp0 + lp * cpi_len
                tile_range[k] = r0 + lr0
                k += 1
        print(f"    pulse tiles {chunk_start + n_here}/{n_pt}")

    ch = Channel(path=None)
    ch.eigenvalues = eig
    ch.diagonal = diag
    ch.diag_valid_idx = valid
    ch.diag_valid_frac = valid.mean(axis=1).astype(np.float32)
    ch.tile_pulse = tile_pulse
    ch.tile_range = tile_range
    ch.attrs = {
        "frequency": freq, "polarization": pol,
        "granule_path": granule, "granule": os.path.basename(str(granule)),
        "pulse_start": p0, "pulse_end": p1, "range_start": r0, "range_end": r1,
        "cpi_len": cpi_len, "cpi_width": cpi_width,
        "caltone_removed": bool(remove_caltone),
        "gap_exclusion_used": bool(use_mask),
        "off_diag_overlap_ratio": off_diag_overlap_ratio,
        "diag_valid_ratio": diag_valid_ratio,
    }
    return ch
