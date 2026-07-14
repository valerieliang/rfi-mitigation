"""
generate_rfi_data.py

Builds two CPI-tile test sets from REAL NISAR L0B data:

  1. clean/       -- real CPI tiles taken as-is, every tile labeled knee = 0.
                     This is the "current data" set: no injection at all, and
                     the whole region is assumed clean by construction of the
                     label (see CAVEAT below).

  2. contaminated/-- the SAME real CPI tiles with synthetic RFI overlaid.
                     Each tile independently draws n_bands uniformly from
                     [MIN_BANDS, MAX_BANDS] = [0, 6]; the label (knee) is
                     exactly that drawn number. A tile that draws 0 bands is
                     an in-set clean tile (knee = 0).

Why JSR (not JNR)
-----------------
generate_synthetic_data.py builds fully synthetic frames (synthetic noise +
synthetic signal), so RFI strength is naturally expressed relative to a
synthetic noise floor (JNR). Here the background is real L0B data: there is
no separate synthetic noise component to reference. The only meaningful
reference is the tile's own observed baseline power, so RFI strength is
expressed as JSR = jammer-to-signal ratio, where "signal" is the mean power
of the real CPI matrix over its valid (non-gap) samples.

Per the requested configuration the JSR is FIXED at JSR_DB = 3.0 dB, i.e.
every injected band carries 3 dB more power than the tile's own baseline.

RFI injection model (adapted from generate_synthetic_data.py and
score_anomaly_jsr_sweep.py)
---------------------------------------------------------------------
For each band injected into a tile:
  1. Estimate the tile's baseline power from its valid samples:
         signal_power = mean(|x|^2) over mask == True
  2. Band power = signal_power * 10^(JSR_DB / 10).
  3. Pick a random local pulse row in [0, cpi_len) (rows may repeat across
     bands; two bands on the same row sum incoherently because they use
     independent range-coefficient vectors and independent Doppler phases).
  4. Draw an independent complex Gaussian range-coefficient vector scaled to
     the band power, modulate by a random Doppler phase, add onto the tile.

SCM / eigenvalue convention
---------------------------
All covariance and eigenvalue math follows read_nisar_isce3.py:
  - SCM is the gap-exclusion slow-time sample covariance
    (compute_gap_exclusion_cov), which normalizes each entry by its own
    valid-overlap count using the ISCE3 subswath mask. This is what keeps
    the inter-subswath gaps from poisoning the covariance.
  - Eigenvalues come from np.linalg.eigvalsh on that Hermitian SCM, sorted
    descending.
  - When no subswath mask is available/requested, the SCM degrades to the
    plain (M @ M^H) / K estimate.

Seeding / uncorrelation
-----------------------
Every random stream is derived from a SeedSequence whose entropy tuple
uniquely identifies its role, so no two streams share state:

    injection structure (n_bands, band rows) : [seed, SET_INJECT, pt, rt]
    per-band coefficients + Doppler          : spawned children of the above
    plot tile selection                      : [plot_seed, SALT_PLOT]

The plot RNG is a completely separate stream, so drawing plot tiles cannot
perturb the injection draws. The injection stream for tile (pt, rt) depends
only on (seed, pt, rt), so a tile's contamination is reproducible regardless
of how many bands any other tile drew.

CAVEAT
------
The "clean" set is only clean by assumption: the background region is
presumed mostly RFI-free, it is not independently verified ground truth. Any
real RFI already present in the region will show up as a false positive
against the knee = 0 label.

HDF5 layout (one file per polarization per set)
-----------------------------------------------
Root attributes : region config (pulse/range window, cpi dims, jsr, seed,
                  set name, freq, pol, gap-exclusion ratios, ...)
Dataset         : "subswath_mask" (bool, region-sized, gzip) if computed
Per CPI tile at absolute (pulse p0, range r0):
    "cpi_{p0}_{r0}"              complex64 (cpi_len, cpi_width)
        attr 'rfi_bands' (str, JSON):
            knee            (int)       : number of injected bands, 0..6
            pulse_positions (list[int]) : 1-based local pulse rows of bands
            jsr_db_list     (list[float]): per-band JSR in dB
            n_distinct_rows (int)       : distinct rows actually hit
            signal_power_db (float)     : tile baseline power, 10*log10
            valid_fraction  (float)     : fraction of valid (unmasked) samples
    "cpi_{p0}_{r0}_eigenvalues"  float32 (cpi_len,)  descending, linear
    "cpi_{p0}_{r0}_diagonal"     float32 (cpi_len,)  |diag(SCM)|
    "cpi_{p0}_{r0}_scm"          complex64 (cpi_len, cpi_len)  [--save-scm]

Usage
-----
    python generate_rfi_data.py granule.h5 --freq A --pol HH \
        --pulse-start 888222 --pulse-end 896222 \
        --range-start 2000 --range-end 25000 \
        --output-dir data/real_testsets \
        --compute-subswath-mask \
        --seed 1234 --plot-seed 20240101
"""

import os
import json
import argparse
from dataclasses import dataclass
from typing import List

import numpy as np
import h5py

from nisar.products.readers.Raw import Raw


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Standard CPI tile: 16 pulses x 250 range samples
CPI_LEN_DEFAULT = 16
CPI_WIDTH_DEFAULT = 250

# Test region (slow time). Matches the region used by score_anomaly_jsr_sweep.
PULSE_START_DEFAULT = 888222
PULSE_END_DEFAULT = 896222

# RFI band count drawn per tile in the contaminated set (both ends inclusive).
# The drawn count IS the label.
MIN_BANDS_DEFAULT = 0
MAX_BANDS_DEFAULT = 6

# Fixed jammer-to-signal ratio: every band sits 3 dB above the tile baseline.
JSR_DB_DEFAULT = 3.0

# Gap-exclusion covariance thresholds (same defaults as read_nisar_isce3.py)
OFF_DIAG_OVERLAP_RATIO_DEFAULT = 0.25
DIAG_VALID_RATIO_DEFAULT = 0.20

# Seed-namespace salts. These keep independent roles in disjoint SeedSequence
# entropy namespaces so no two random streams can ever alias.
SALT_INJECT = 0xA1
SALT_PLOT = 0xB2

# Number of pulses read per chunk (must be a multiple of cpi_len; enforced).
PULSE_CHUNK_DEFAULT = 1600

N_PLOT_BLOCKS_DEFAULT = 12

EPS = 1e-12


# ---------------------------------------------------------------------------
# METADATA CONTAINERS
# ---------------------------------------------------------------------------

@dataclass
class BandMeta:
    """Descriptor for a single injected RFI band within one CPI tile."""
    local_row: int    # local pulse row within the tile, [0, cpi_len)
    jsr_db: float     # jammer-to-signal ratio for this band, in dB


@dataclass
class TileMeta:
    """Label + provenance for one CPI tile."""
    knee: int                       # number of injected bands (0..MAX_BANDS)
    bands: List[BandMeta]
    signal_power_db: float          # tile baseline power, 10*log10(mean|x|^2)
    valid_fraction: float


# ---------------------------------------------------------------------------
# GAP-EXCLUSION COVARIANCE (verbatim convention from read_nisar_isce3.py)
# ---------------------------------------------------------------------------

def compute_gap_exclusion_cov(
    data: np.ndarray,
    *,
    mask_valid_cpi: np.ndarray = None,
    off_diag_overlap_ratio: float = OFF_DIAG_OVERLAP_RATIO_DEFAULT,
    diag_valid_ratio: float = DIAG_VALID_RATIO_DEFAULT,
):
    """
    Gap-excluded slow-time sample covariance matrix.

    Each SCM entry R_ij is normalized by the number of range samples where
    pulses i and j are BOTH valid, rather than by the nominal range width.
    Entries without enough overlapping valid samples are left at zero.

    Parameters
    ----------
    data : (M, K) complex array
        CPI tile: M pulses x K range samples, contiguous in slow time.
    mask_valid_cpi : (M, K) bool array, optional
        True where the sample is valid. None means "all valid".
    off_diag_overlap_ratio : float
        Minimum fraction of overlapping valid samples to compute R_ij, i != j.
    diag_valid_ratio : float
        Minimum fraction of valid samples to compute R_ii.

    Returns
    -------
    cov : (M, M) complex64, Hermitian
    diag_valid_idx : (M,) bool
    """
    num_pulses, num_rng_samples = data.shape

    if mask_valid_cpi is None:
        mask_valid_cpi = np.ones(data.shape, dtype=bool)
    else:
        mask_valid_cpi = mask_valid_cpi.astype(bool, copy=False)

    if mask_valid_cpi.shape != data.shape:
        raise ValueError(
            f"CPI mask shape {mask_valid_cpi.shape} != CPI data shape {data.shape}"
        )

    min_valid_off_diag = max(1, int(np.ceil(off_diag_overlap_ratio * num_rng_samples)))
    min_valid_diag = max(1, int(np.ceil(diag_valid_ratio * num_rng_samples)))

    x_valid = data * mask_valid_cpi

    mask_int = mask_valid_cpi.astype(np.int32)
    overlap_counts = mask_int @ mask_int.T
    cov_sum = x_valid @ x_valid.conj().T

    cov = np.zeros((num_pulses, num_pulses), dtype=np.complex64)

    diag_idx = np.diag_indices(num_pulses)
    diag_counts = overlap_counts[diag_idx]
    diag_cov_sum = cov_sum[diag_idx]
    diag_valid_idx = diag_counts >= min_valid_diag

    diag_vals = np.zeros(num_pulses, dtype=np.complex64)
    diag_vals[diag_valid_idx] = (
        diag_cov_sum[diag_valid_idx] / diag_counts[diag_valid_idx]
    )
    cov[diag_idx] = diag_vals

    off_diag_valid = overlap_counts >= min_valid_off_diag
    np.fill_diagonal(off_diag_valid, False)
    cov[off_diag_valid] = cov_sum[off_diag_valid] / overlap_counts[off_diag_valid]

    # Enforce Hermitian symmetry numerically
    cov = (0.5 * (cov + cov.conj().T)).astype(np.complex64)

    return cov, diag_valid_idx


def compute_scm_and_eigs(cpi, cpi_mask, off_diag_overlap_ratio, diag_valid_ratio):
    """
    Compute the SCM and its descending eigenvalues for one CPI tile.

    Uses the gap-exclusion covariance when a mask is supplied, otherwise the
    plain (M @ M^H) / K estimate.

    Returns
    -------
    scm : (M, M) complex64
    eigvals : (M,) float32, descending, linear scale
    diag_power : (M,) float32, |diag(SCM)|
    """
    if cpi_mask is not None:
        scm, _ = compute_gap_exclusion_cov(
            cpi,
            mask_valid_cpi=cpi_mask,
            off_diag_overlap_ratio=off_diag_overlap_ratio,
            diag_valid_ratio=diag_valid_ratio,
        )
    else:
        M, K = cpi.shape
        scm = ((cpi @ cpi.conj().T) / K).astype(np.complex64)

    eigvals = np.linalg.eigvalsh(scm)          # ascending, real
    eigvals = np.sort(eigvals)[::-1]           # descending
    diag_power = np.abs(np.diag(scm))

    return scm, eigvals.astype(np.float32), diag_power.astype(np.float32)


# ---------------------------------------------------------------------------
# ISCE3 RAW READ / SUBSWATH MASK
# ---------------------------------------------------------------------------

def read_raw_data_batch(raw: Raw, freq: str, pol: str, pulse_slice: slice, range_slice: slice):
    """Read a (pulse, range) window; ISCE3 handles BFPQLUT decoding."""
    dataset = raw.getRawDataset(freq, pol)

    p0 = pulse_slice.start if pulse_slice.start is not None else 0
    p1 = pulse_slice.stop if pulse_slice.stop is not None else dataset.shape[0]
    r0 = range_slice.start if range_slice.start is not None else 0
    r1 = range_slice.stop if range_slice.stop is not None else dataset.shape[1]

    return dataset[p0:p1, r0:r1]


def get_subswath_mask(raw: Raw, freq: str, pol: str,
                      pulse_indices: np.ndarray, range_indices: np.ndarray) -> np.ndarray:
    """
    Boolean valid-sample mask from the ISCE3 subswath boundaries.

    Subswath boundaries are absolute range-sample indices, so they are shifted
    by the window's first range index and clipped to the window extent.
    """
    tx_pol = pol[0]
    subswaths = raw.getSubSwaths(freq, tx_pol)
    swaths = subswaths[:, pulse_indices, :]

    num_pulses = len(pulse_indices)
    num_range_samples = len(range_indices)
    mask = np.zeros((num_pulses, num_range_samples), dtype=bool)

    r_offset = int(range_indices[0])

    if swaths is not None:
        for i in range(num_pulses):
            for start, end in swaths[:, i, :]:
                s = max(int(start) - r_offset, 0)
                e = min(int(end) - r_offset, num_range_samples)
                if e > s:
                    mask[i, s:e] = True

    return mask


# ---------------------------------------------------------------------------
# RFI INJECTION
# ---------------------------------------------------------------------------

def tile_signal_power(cpi, cpi_mask):
    """
    Baseline power of a real CPI tile: mean(|x|^2) over its valid samples.
    This is the "signal" that JSR is referenced to.
    """
    if cpi_mask is not None and cpi_mask.any():
        vals = cpi[cpi_mask]
    else:
        vals = cpi.ravel()
    return max(float(np.mean(np.abs(vals) ** 2)), EPS)


def make_tile_seed_seq(seed: int, pulse_tile: int, range_tile: int) -> np.random.SeedSequence:
    """
    Injection SeedSequence for one tile. Depends only on (seed, pt, rt), so a
    tile's contamination is reproducible independently of every other tile.
    """
    return np.random.SeedSequence([int(seed), SALT_INJECT, int(pulse_tile), int(range_tile)])


def inject_rfi_bands(cpi, cpi_mask, n_bands, jsr_db, tile_seed_seq):
    """
    Overlay n_bands synthetic Gaussian RFI bands onto a real CPI tile.

    Every band sits jsr_db dB above the tile's own baseline power. Each band
    gets its own spawned RNG child stream, so bands are mutually uncorrelated;
    the structural draws (band rows) come from the tile's own stream.

    Parameters
    ----------
    cpi : (M, K) complex array          real background tile
    cpi_mask : (M, K) bool or None      valid-sample mask
    n_bands : int                       number of bands to inject (may be 0)
    jsr_db : float                      fixed JSR per band, in dB
    tile_seed_seq : np.random.SeedSequence

    Returns
    -------
    contaminated : (M, K) complex64
    meta : TileMeta
    """
    M, K = cpi.shape

    signal_power = tile_signal_power(cpi, cpi_mask)
    signal_power_db = 10.0 * np.log10(signal_power)
    valid_fraction = (
        float(cpi_mask.sum()) / cpi_mask.size if cpi_mask is not None else 1.0
    )

    if n_bands == 0:
        meta = TileMeta(knee=0, bands=[],
                        signal_power_db=signal_power_db,
                        valid_fraction=valid_fraction)
        return cpi.astype(np.complex64), meta

    rng_struct = np.random.default_rng(tile_seed_seq)
    band_seeds = tile_seed_seq.spawn(n_bands)

    band_power = signal_power * (10.0 ** (jsr_db / 10.0))
    sigma = np.sqrt(band_power / 2.0)   # half the power in real, half in imag

    rfi = np.zeros((M, K), dtype=np.complex64)
    bands: List[BandMeta] = []

    for b in range(n_bands):
        local_row = int(rng_struct.integers(0, M))
        rng_band = np.random.default_rng(band_seeds[b])

        doppler_freq = rng_band.uniform(-0.5, 0.5)
        range_coeff = (
            rng_band.standard_normal(K) + 1j * rng_band.standard_normal(K)
        ) * sigma
        phase = np.exp(1j * 2.0 * np.pi * doppler_freq * local_row)

        rfi[local_row, :] += (phase * range_coeff).astype(np.complex64)
        bands.append(BandMeta(local_row=local_row, jsr_db=float(jsr_db)))

    contaminated = (cpi + rfi).astype(np.complex64)
    meta = TileMeta(knee=n_bands, bands=bands,
                    signal_power_db=signal_power_db,
                    valid_fraction=valid_fraction)
    return contaminated, meta


def draw_n_bands(tile_seed_seq, min_bands, max_bands):
    """
    Draw the band count (== the label) for one tile from its own stream.

    Uses a dedicated child of the tile seed sequence so that the count draw is
    decoupled from the band-row / coefficient draws.
    """
    count_seed = tile_seed_seq.spawn(1)[0]
    rng_count = np.random.default_rng(count_seed)
    return int(rng_count.integers(min_bands, max_bands + 1))


# ---------------------------------------------------------------------------
# TEST SET GENERATION
# ---------------------------------------------------------------------------

def tile_meta_json(meta: TileMeta) -> str:
    """Serialize a TileMeta to the 'rfi_bands' JSON attribute string."""
    rows = [b.local_row for b in meta.bands]
    return json.dumps({
        'knee': meta.knee,
        'pulse_positions': [r + 1 for r in rows],   # 1-based, as in the synthetic generator
        'jsr_db_list': [b.jsr_db for b in meta.bands],
        'n_distinct_rows': len(set(rows)),
        'signal_power_db': meta.signal_power_db,
        'valid_fraction': meta.valid_fraction,
    })


def write_root_attrs(f, args, freq, pol, set_name, p_start, p_end, r_start, r_end,
                     cpi_len, cpi_width, n_pulse_tiles, n_range_tiles, use_mask):
    """File-level configuration attributes."""
    f.attrs['set_name'] = set_name
    f.attrs['is_clean_set'] = (set_name == 'clean')
    f.attrs['l0b_file'] = os.path.basename(args.l0b_file)
    f.attrs['frequency'] = freq
    f.attrs['polarization'] = pol
    f.attrs['pulse_start'] = p_start
    f.attrs['pulse_end'] = p_end
    f.attrs['range_start'] = r_start
    f.attrs['range_end'] = r_end
    f.attrs['cpi_len'] = cpi_len
    f.attrs['cpi_width'] = cpi_width
    f.attrs['n_pulse_tiles'] = n_pulse_tiles
    f.attrs['n_range_tiles'] = n_range_tiles
    f.attrs['jsr_db'] = args.jsr_db
    f.attrs['min_bands'] = 0 if set_name == 'clean' else args.min_bands
    f.attrs['max_bands'] = 0 if set_name == 'clean' else args.max_bands
    f.attrs['seed'] = args.seed
    f.attrs['plot_seed'] = args.plot_seed
    f.attrs['gap_exclusion_used'] = bool(use_mask)
    f.attrs['off_diag_overlap_ratio'] = args.off_diag_overlap_ratio
    f.attrs['diag_valid_ratio'] = args.diag_valid_ratio
    f.attrs['jsr_reference'] = 'tile baseline power over valid samples'


def generate_set(raw, freq, pol, args, set_name, plot_tiles, out_dir):
    """
    Build one test set (clean or contaminated) for one polarization.

    The region is streamed in pulse chunks so the full window never has to sit
    in memory at once. Tiles are written to HDF5 as they are produced.

    Returns
    -------
    plot_records : list[dict]
        Per-selected-tile records (cpi, scm, eigenvalues, meta) for plotting.
    knee_counts : dict[int, int]
        Label histogram for this set.
    """
    cpi_len = args.cpi_len
    cpi_width = args.cpi_width

    dataset = raw.getRawDataset(freq, pol)
    total_pulses, total_range = dataset.shape

    p_start = args.pulse_start
    p_end = min(args.pulse_end, total_pulses)
    r_start = args.range_start if args.range_start is not None else 0
    r_end = args.range_end if args.range_end is not None else total_range

    # Truncate the window to whole CPI tiles
    n_pulse_tiles = (p_end - p_start) // cpi_len
    n_range_tiles = (r_end - r_start) // cpi_width
    p_end = p_start + n_pulse_tiles * cpi_len
    r_end = r_start + n_range_tiles * cpi_width

    if n_pulse_tiles == 0 or n_range_tiles == 0:
        raise ValueError(
            f"Requested window is smaller than one CPI tile "
            f"({cpi_len} x {cpi_width})"
        )

    out_path = os.path.join(out_dir, f"testset_{set_name}_{freq}_{pol}.h5")
    print(f"\n[{set_name}] {freq}-{pol}")
    print(f"  pulses [{p_start}:{p_end}]  range [{r_start}:{r_end}]")
    print(f"  tile grid: {n_pulse_tiles} pulse tiles x {n_range_tiles} range tiles "
          f"= {n_pulse_tiles * n_range_tiles} tiles")
    print(f"  -> {out_path}")

    use_mask = args.compute_subswath_mask
    plot_lookup = {(pt, rt) for (pt, rt) in plot_tiles}
    plot_records = []
    knee_counts = {}

    # Chunk size in pulses, snapped to a whole number of CPI tiles
    chunk_tiles = max(1, args.pulse_chunk // cpi_len)
    chunk_pulses = chunk_tiles * cpi_len

    with h5py.File(out_path, 'w') as f:
        write_root_attrs(f, args, freq, pol, set_name, p_start, p_end, r_start, r_end,
                         cpi_len, cpi_width, n_pulse_tiles, n_range_tiles, use_mask)

        for chunk_start_tile in range(0, n_pulse_tiles, chunk_tiles):
            n_tiles_here = min(chunk_tiles, n_pulse_tiles - chunk_start_tile)
            cp0 = p_start + chunk_start_tile * cpi_len
            cp1 = cp0 + n_tiles_here * cpi_len

            raw_chunk = read_raw_data_batch(
                raw, freq, pol,
                pulse_slice=slice(cp0, cp1),
                range_slice=slice(r_start, r_end),
            )

            if use_mask:
                mask_chunk = get_subswath_mask(
                    raw, freq, pol,
                    np.arange(cp0, cp1),
                    np.arange(r_start, r_end),
                )
            else:
                mask_chunk = None

            for local_pt in range(n_tiles_here):
                pt = chunk_start_tile + local_pt
                lp0 = local_pt * cpi_len
                lp1 = lp0 + cpi_len
                abs_p0 = p_start + pt * cpi_len

                for rt in range(n_range_tiles):
                    lr0 = rt * cpi_width
                    lr1 = lr0 + cpi_width
                    abs_r0 = r_start + lr0

                    cpi = np.ascontiguousarray(raw_chunk[lp0:lp1, lr0:lr1]).astype(np.complex64)
                    cpi_mask = (
                        np.ascontiguousarray(mask_chunk[lp0:lp1, lr0:lr1])
                        if mask_chunk is not None else None
                    )

                    tile_ss = make_tile_seed_seq(args.seed, pt, rt)

                    if set_name == 'clean':
                        # No injection; the label is 0 by definition of the set.
                        sig_db = 10.0 * np.log10(tile_signal_power(cpi, cpi_mask))
                        vfrac = (
                            float(cpi_mask.sum()) / cpi_mask.size
                            if cpi_mask is not None else 1.0
                        )
                        tile = cpi
                        meta = TileMeta(knee=0, bands=[],
                                        signal_power_db=sig_db,
                                        valid_fraction=vfrac)
                    else:
                        n_bands = draw_n_bands(tile_ss, args.min_bands, args.max_bands)
                        tile, meta = inject_rfi_bands(
                            cpi, cpi_mask, n_bands, args.jsr_db, tile_ss
                        )

                    scm, eigvals, diag_power = compute_scm_and_eigs(
                        tile, cpi_mask,
                        args.off_diag_overlap_ratio,
                        args.diag_valid_ratio,
                    )

                    knee_counts[meta.knee] = knee_counts.get(meta.knee, 0) + 1

                    base = f"cpi_{abs_p0}_{abs_r0}"
                    if args.save_cpi:
                        dset = f.create_dataset(base, data=tile, compression='gzip',
                                                compression_opts=4)
                    else:
                        dset = f.create_group(base)
                    dset.attrs['rfi_bands'] = tile_meta_json(meta)

                    f.create_dataset(f"{base}_eigenvalues", data=eigvals)
                    f.create_dataset(f"{base}_diagonal", data=diag_power)
                    if args.save_scm:
                        f.create_dataset(f"{base}_scm", data=scm)

                    if (pt, rt) in plot_lookup:
                        plot_records.append({
                            'pt': pt, 'rt': rt,
                            'abs_p0': abs_p0, 'abs_r0': abs_r0,
                            'cpi': tile.copy(),
                            'scm': scm.copy(),
                            'eigvals': eigvals.copy(),
                            'meta': meta,
                        })

            print(f"    pulse tiles {chunk_start_tile}..{chunk_start_tile + n_tiles_here - 1} done")

    plot_records.sort(key=lambda rec: (rec['pt'], rec['rt']))

    print(f"  label histogram: "
          + ", ".join(f"knee={k}: {knee_counts[k]}" for k in sorted(knee_counts)))

    return plot_records, knee_counts, out_path


# ---------------------------------------------------------------------------
# PLOT TILE SELECTION (fixed, independent seed)
# ---------------------------------------------------------------------------

def select_plot_tiles(plot_seed, n_pulse_tiles, n_range_tiles, n_blocks):
    """
    Pick n_blocks distinct (pulse_tile, range_tile) positions using a stream
    that is fully independent of the injection streams. The SAME positions are
    used for the clean and contaminated sets, so the plots are directly
    comparable tile-for-tile.
    """
    ss = np.random.SeedSequence([int(plot_seed), SALT_PLOT])
    rng = np.random.default_rng(ss)

    n_available = n_pulse_tiles * n_range_tiles
    n_pick = min(n_blocks, n_available)
    flat = rng.choice(n_available, size=n_pick, replace=False)

    tiles = [(int(idx // n_range_tiles), int(idx % n_range_tiles)) for idx in flat]
    return sorted(tiles)


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------

def plot_eigenvalue_profiles(records, set_name, freq, pol, out_dir, max_bands):
    """
    Two eigenvalue figures for the randomly selected blocks:

      Figure 1 -- all selected blocks overlaid on one axes, each line colored
                  by its label (knee). The knee position is marked so the
                  drop-off after the injected bands is visible.
      Figure 2 -- 2 x N grid of the individual profiles, one panel per block.

    Y axis is absolute dB: 10 * log10(eigenvalue).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    if not records:
        return

    cpi_len = len(records[0]['eigvals'])
    ev_index = np.arange(1, cpi_len + 1)   # 1-based eigenvalue index

    profiles_db = [10.0 * np.log10(np.maximum(r['eigvals'], EPS)) for r in records]
    knees = [r['meta'].knee for r in records]

    all_db = np.concatenate(profiles_db)
    ylim = [np.percentile(all_db, 1) - 5.0, np.percentile(all_db, 99) + 5.0]

    norm = mcolors.Normalize(vmin=0, vmax=max(max_bands, 1))
    cmap = cm.plasma

    # --- Figure 1: overlay ---------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(11, 6))
    for prof, knee in zip(profiles_db, knees):
        ax1.plot(ev_index, prof, color=cmap(norm(knee)), alpha=0.75, linewidth=1.2)
        if knee > 0:
            ax1.plot(knee, prof[knee - 1], 'rx', markersize=6, alpha=0.7)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig1.colorbar(sm, ax=ax1)
    cbar.set_label('Label (knee = number of injected RFI bands)', fontsize=10)

    ax1.set_xlabel('Eigenvalue index (1-based, descending)', fontsize=11)
    ax1.set_ylabel('Eigenvalue (dB, absolute)', fontsize=11)
    ax1.set_ylim(ylim)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.set_title(
        f'Eigenvalue profiles -- {set_name.upper()} set -- {freq}-{pol}\n'
        f'{len(records)} randomly selected CPI blocks (fixed plot seed), '
        f'gap-exclusion SCM',
        fontsize=11,
    )
    fig1.tight_layout()
    path1 = os.path.join(out_dir, f'{set_name}_{freq}_{pol}_ev_overlay.png')
    fig1.savefig(path1, dpi=150)
    plt.close(fig1)

    # --- Figure 2: per-block grid -------------------------------------------
    n = len(records)
    n_cols = min(6, n)
    n_rows = int(np.ceil(n / n_cols))
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 3.1 * n_rows),
                              squeeze=False)

    for ax, rec, prof in zip(axes.flat, records, profiles_db):
        meta = rec['meta']
        knee = meta.knee
        ax.plot(ev_index, prof, color=cmap(norm(knee)), linewidth=1.5)
        if knee > 0:
            ax.plot(knee, prof[knee - 1], 'rx', markersize=7, markeredgewidth=2)
            ax.axvline(x=knee, color='red', linestyle='--', alpha=0.35, linewidth=1)
        label = 'CLEAN' if knee == 0 else f'RFI={knee}'
        ax.set_title(
            f'p={rec["abs_p0"]} r={rec["abs_r0"]} [{label}]\n'
            f'baseline={meta.signal_power_db:.1f} dB | valid={meta.valid_fraction*100:.0f}%',
            fontsize=7,
        )
        ax.set_xlabel('EV index', fontsize=8)
        ax.set_ylabel('Eigenvalue (dB)', fontsize=8)
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle='--', alpha=0.4)

    for ax in axes.flat[n:]:
        ax.axis('off')

    fig2.suptitle(
        f'Eigenvalue profiles per selected block -- {set_name.upper()} -- {freq}-{pol}',
        fontsize=12,
    )
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    path2 = os.path.join(out_dir, f'{set_name}_{freq}_{pol}_ev_blocks.png')
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)

    print(f"  plots -> {os.path.basename(path1)}, {os.path.basename(path2)}")


def plot_scm_matrices(records, set_name, freq, pol, out_dir):
    """
    Grid of SCM magnitude heatmaps (20 * log10 |R_ij|, dB) for the same
    randomly selected blocks. Injected bands show up as bright rows/columns
    and as raised off-diagonal structure.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not records:
        return

    mags_db = [20.0 * np.log10(np.abs(r['scm']) + EPS) for r in records]
    finite = np.concatenate([m[np.isfinite(m)].ravel() for m in mags_db])
    vmin = np.percentile(finite, 5)
    vmax = np.percentile(finite, 100)

    n = len(records)
    n_cols = min(6, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.2 * n_rows),
                             squeeze=False)

    im = None
    for ax, rec, mag in zip(axes.flat, records, mags_db):
        meta = rec['meta']
        im = ax.imshow(mag, cmap='inferno', vmin=vmin, vmax=vmax,
                       origin='upper', interpolation='nearest')
        label = 'CLEAN' if meta.knee == 0 else f'RFI={meta.knee}'
        rows = sorted({b.local_row for b in meta.bands})
        rows_str = '' if not rows else f'\nrows={rows}'
        ax.set_title(
            f'p={rec["abs_p0"]} r={rec["abs_r0"]} [{label}]{rows_str}',
            fontsize=7,
        )
        ax.set_xlabel('Pulse j', fontsize=8)
        ax.set_ylabel('Pulse i', fontsize=8)
        ax.tick_params(labelsize=6)

    for ax in axes.flat[n:]:
        ax.axis('off')

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
        cbar.set_label('|SCM| (dB)', fontsize=10)

    fig.suptitle(
        f'Gap-exclusion SCM magnitude -- {set_name.upper()} -- {freq}-{pol}',
        fontsize=12,
    )
    path = os.path.join(out_dir, f'{set_name}_{freq}_{pol}_scm.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  plots -> {os.path.basename(path)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=('Generate a clean test set and an RFI-contaminated test set '
                     'from real NISAR L0B CPI tiles.'),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('l0b_file', help='Input NISAR L0B HDF5 granule')
    parser.add_argument('--freq', choices=['A', 'B'], default='A')
    parser.add_argument('--pol', default=None,
                        help='Polarization (HH/HV/VH/VV). Default: all available.')

    parser.add_argument('--pulse-start', type=int, default=PULSE_START_DEFAULT)
    parser.add_argument('--pulse-end', type=int, default=PULSE_END_DEFAULT)
    parser.add_argument('--range-start', type=int, default=None)
    parser.add_argument('--range-end', type=int, default=None)

    parser.add_argument('--cpi-len', type=int, default=CPI_LEN_DEFAULT)
    parser.add_argument('--cpi-width', type=int, default=CPI_WIDTH_DEFAULT)
    parser.add_argument('--pulse-chunk', type=int, default=PULSE_CHUNK_DEFAULT,
                        help='Pulses read per chunk; snapped down to whole CPI tiles.')

    parser.add_argument('--min-bands', type=int, default=MIN_BANDS_DEFAULT)
    parser.add_argument('--max-bands', type=int, default=MAX_BANDS_DEFAULT)
    parser.add_argument('--jsr-db', type=float, default=JSR_DB_DEFAULT,
                        help='Fixed jammer-to-signal ratio per band, in dB.')

    parser.add_argument('--compute-subswath-mask', action='store_true',
                        help='Use ISCE3 subswath boundaries for gap-exclusion SCM.')
    parser.add_argument('--off-diag-overlap-ratio', type=float,
                        default=OFF_DIAG_OVERLAP_RATIO_DEFAULT)
    parser.add_argument('--diag-valid-ratio', type=float,
                        default=DIAG_VALID_RATIO_DEFAULT)

    parser.add_argument('--output-dir', default='data/real_testsets')
    parser.add_argument('--no-save-cpi', dest='save_cpi', action='store_false',
                        help='Skip storing the complex CPI tiles (labels/eigs only).')
    parser.add_argument('--save-scm', action='store_true',
                        help='Also store the full complex SCM per tile.')

    parser.add_argument('--seed', type=int, default=1234,
                        help='Master seed for RFI injection streams.')
    parser.add_argument('--plot-seed', type=int, default=20240101,
                        help='Fixed, independent seed for selecting plotted blocks.')
    parser.add_argument('--n-plot-blocks', type=int, default=N_PLOT_BLOCKS_DEFAULT)

    parser.set_defaults(save_cpi=True)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.min_bands < 0 or args.max_bands < args.min_bands:
        raise ValueError('Require 0 <= min_bands <= max_bands')

    os.makedirs(args.output_dir, exist_ok=True)
    clean_dir = os.path.join(args.output_dir, 'clean')
    cont_dir = os.path.join(args.output_dir, 'contaminated')
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(cont_dir, exist_ok=True)

    print('=' * 70)
    print('Real-background RFI / clean test set generation')
    print('=' * 70)
    print(f'  granule        : {args.l0b_file}')
    print(f'  pulse window   : [{args.pulse_start}, {args.pulse_end})')
    print(f'  range window   : [{args.range_start}, {args.range_end})')
    print(f'  CPI tile       : {args.cpi_len} x {args.cpi_width}')
    print(f'  bands per tile : {args.min_bands}..{args.max_bands} (label = drawn count)')
    print(f'  JSR            : {args.jsr_db} dB above each tile baseline power')
    print(f'  gap exclusion  : {args.compute_subswath_mask}')
    print(f'  seeds          : injection={args.seed}, plot={args.plot_seed}')

    raw = Raw(hdf5file=args.l0b_file)
    raw.parsePolarizations()

    pols = [args.pol] if args.pol else list(raw.polarizations[args.freq])

    for pol in pols:
        dataset = raw.getRawDataset(args.freq, pol)
        total_pulses, total_range = dataset.shape

        p_end = min(args.pulse_end, total_pulses)
        r_start = args.range_start if args.range_start is not None else 0
        r_end = args.range_end if args.range_end is not None else total_range

        n_pulse_tiles = (p_end - args.pulse_start) // args.cpi_len
        n_range_tiles = (r_end - r_start) // args.cpi_width

        plot_tiles = select_plot_tiles(
            args.plot_seed, n_pulse_tiles, n_range_tiles, args.n_plot_blocks
        )

        # Set 1: current data as-is, all tiles labeled clean (knee = 0)
        clean_records, _, clean_path = generate_set(
            raw, args.freq, pol, args, 'clean', plot_tiles, clean_dir
        )
        plot_eigenvalue_profiles(clean_records, 'clean', args.freq, pol, clean_dir,
                                 args.max_bands)
        plot_scm_matrices(clean_records, 'clean', args.freq, pol, clean_dir)

        # Set 2: same tiles, RFI overlaid, label = number of bands drawn
        cont_records, _, cont_path = generate_set(
            raw, args.freq, pol, args, 'contaminated', plot_tiles, cont_dir
        )
        plot_eigenvalue_profiles(cont_records, 'contaminated', args.freq, pol, cont_dir,
                                 args.max_bands)
        plot_scm_matrices(cont_records, 'contaminated', args.freq, pol, cont_dir)

        print(f'\n  {pol}: wrote\n    {clean_path}\n    {cont_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()