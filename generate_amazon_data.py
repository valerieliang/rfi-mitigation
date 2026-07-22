"""
generate_amazon_data.py

Builds ONE labeled CPI training set from real NISAR L0B data by overlaying
synthetic Gaussian RFI bands on top of real CPI tiles.

Labeling
--------
Every tile independently draws n_bands uniformly from [MIN_BANDS, MAX_BANDS]
= [0, 6], and that drawn number IS the label (knee). A tile that draws 0 gets
no injection at all, so clean tiles are randomly interspersed through the set
rather than living in a separate file. With a uniform draw over 7 values the
classes come out balanced at roughly 1/7 each.

The band rows are then drawn WITHOUT REPLACEMENT (see step 3 below), so the
label is exactly the number of elevated eigenvalues the SCM actually contains.
Class balance is preserved -- the count is still drawn uniformly first, and only
the row placement changed.

Why JSR (not JNR)
-----------------
generate_amazon_data.py builds fully synthetic frames (synthetic noise +
synthetic signal), so RFI strength is naturally expressed against a synthetic
noise floor (JNR). Here the background is real L0B data: there is no separate
synthetic noise component to reference. The only meaningful reference is the
tile's own observed baseline power, so RFI strength is expressed as
JSR = jammer-to-signal ratio, where "signal" is the mean power of the real CPI
matrix over its valid (non-gap) samples.

Each band draws its own JSR independently and uniformly from
[JSR_MIN_DB, JSR_MAX_DB] = [3, 30] dB, so the weakest interferer sits 3 dB above
the tile's own baseline power and the strongest sits 30 dB above it. Bands
within a tile therefore generally differ in strength, matching how
generate_amazon_data.py draws an independent per-band JNR from JNR_RANGE_DB.

RFI injection model (adapted from generate_amazon_data.py and
score_anomaly_jsr_sweep.py)
--------------------------------------------------------------------
For each band injected into a tile:
  1. Estimate the tile's baseline power from its valid samples:
         signal_power = mean(|x|^2) over mask == True
  2. Draw this band's JSR uniformly from [jsr_min_db, jsr_max_db], then
     band power = signal_power * 10^(JSR_dB / 10).
  3. Pick the band's local pulse row WITHOUT REPLACEMENT from [0, cpi_len):
     the n_bands rows for a tile are drawn as a distinct subset, i.e. one of
     the C(cpi_len, n_bands) possible row combinations, chosen uniformly.

     This matters for label validity. Each occupied pulse row contributes one
     rank-1 term to the SCM and therefore one elevated eigenvalue. If rows were
     drawn WITH replacement, two bands could land on the same row, sum into a
     single row, and produce only ONE elevated eigenvalue while the tile was
     still labeled with the full band count. The label would then be asking the
     model to count something the covariance does not contain. Sampling without
     replacement guarantees

         knee (label) == number of distinct rows == number of RFI eigenvalues

     exactly, for every tile. n_bands <= cpi_len is required and asserted.
  4. Draw an independent complex Gaussian range-coefficient vector scaled to
     the band power, modulate by a random Doppler phase, add onto the tile.

SCM / eigenvalue convention
---------------------------
All covariance and eigenvalue math follows read_nisar_isce3.py:
  - SCM is the gap-exclusion slow-time sample covariance
    (compute_gap_exclusion_cov), which normalizes each entry by its own
    valid-overlap count using the ISCE3 subswath mask, so inter-subswath gaps
    do not poison the covariance.
  - Eigenvalues come from np.linalg.eigvalsh on that Hermitian SCM, sorted
    descending. They are stored in LINEAR scale (not dB); take
    10 * log10(.) downstream if a dB profile is wanted.
  - Without --compute-subswath-mask the SCM degrades to (M @ M^H) / K.

Seeding / uncorrelation
-----------------------
Every random stream comes from a SeedSequence whose entropy tuple uniquely
identifies its role, so no two streams can share state:

    band count for tile (pt, rt)  : child of [seed, SALT_INJECT, chan, pt, rt]
    per-band JSR draws            : a separate child of the same
    band rows for tile (pt, rt)   : [seed, SALT_INJECT, chan, pt, rt] itself
    per-band Doppler + coeffs     : spawned children of the same
    plot tile selection           : [plot_seed, SALT_PLOT, chan]

The JSR draws come from their own child stream (mirroring the JNR_SEED offset in
generate_amazon_data.py) so that changing the JSR range does not perturb band
placement or the coefficient vectors.

'chan' is a channel id derived from (frequency, polarization) via CHANNEL_IDS.
It matters because the script runs over every frequency and polarization in the
granule: without it, the same (pt, rt) grid position would receive an identical
band count, identical band rows and identical Doppler in A-HH, A-HV, B-HH, ...
Those tiles would then be near-duplicate training samples with correlated
labels. With the channel id in the entropy tuple, every channel gets its own
independent injection realization over the same real background.

A tile's contamination depends only on (seed, chan, pt, rt), so it is
reproducible regardless of what any other tile or channel drew. The plot stream
is disjoint, so selecting blocks to plot cannot perturb the injection draws.

Storage
-------
By default only compact per-tile records are stored (about 100 bytes/tile):

    labels          int8      (N,)              knee = number of RFI bands, 0..6
    jsr_db          float32   (N, max_bands)    per-band JSR in dB (each drawn
                                                from [jsr_min_db, jsr_max_db]),
                                                NaN-padded; all-NaN row for a
                                                clean tile
    band_rows       int8      (N, max_bands)    local pulse row per band, -1 pad
    eigenvalues     float32   (N, cpi_len)      descending, LINEAR scale
    diagonal        float32   (N, cpi_len)      SCM diagonal, LINEAR scale,
                                                unnormalized power per pulse row.
                                                RFI shows up as a jump on the
                                                injected row(s); a clean row is
                                                comparatively flat.
    diag_valid_idx  bool      (N, cpi_len)      per-index diagonal validity mask
                                                (True = enough non-gap samples)
    signal_power_db float32   (N,)              tile baseline power, 10*log10
    valid_fraction  float32   (N,)              fraction of valid samples
    tile_pulse      int32     (N,)              absolute pulse index of tile row 0
    tile_range      int32     (N,)              absolute range index of tile col 0

With --save-cpi the raw complex tiles are stored as well:

    cpi             complex64 (N, cpi_len, cpi_width)   gzip, chunked per tile

That array is roughly 32 kB/tile (about 13.7 GB for the full default window),
so it is off by default.

Root attributes carry the full generation config: seed, plot seed, pulse and
range windows, CPI dimensions, JSR range, band range, gap-exclusion ratios,
granule name, frequency, polarization, and the final label histogram.

Usage
-----
    # Every frequency and polarization in the granule (the default)
    python generate_amazon_data.py granule.h5 \
        --pulse-start 813924 --pulse-end 888222 \
        --range-start 2000 --range-end 25000 \
        --compute-subswath-mask \
        --jsr-min-db 3 --jsr-max-db 30 \
        --output-dir data/rfi_train \
        --seed 0 --plot-seed 99

    # Restrict to one channel, and also keep the complex CPI tiles
    python generate_amazon_data.py granule.h5 --freq A --pol HH --save-cpi

One HDF5 file per channel is written: rfi_data_<freq>_<pol>.h5
"""

import os
import json
import argparse
import warnings
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List

import numpy as np
import h5py


def _silence_third_party_noise():
    """
    Quiet the warnings the readers emit on every run.

    Three separate sources, none of them actionable here:
      1. The nisar Identification reader warns that hasInputDataException is
         absent from the product metadata; it then correctly assumes no
         anomalies. Expected for these granules.
      2. ISCE3 routes the same message through its 'journal' channels, which
         bypass the warnings module entirely and must be deactivated directly.
      3. Assorted DeprecationWarnings from the isce3/h5py/numpy stack.

    Set RFI_SHOW_WARNINGS=1 in the environment to keep all of them.
    """
    if os.environ.get('RFI_SHOW_WARNINGS'):
        return

    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings(
        'ignore',
        message='.*hasInputDataException.*',
        category=UserWarning,
    )

    # ISCE3 journal channels print outside the warnings machinery
    try:
        import journal
        for channel in ('nisar.reader', 'isce3.io', 'isce3.core'):
            journal.info(channel).deactivate()
            journal.warning(channel).deactivate()
    except Exception:
        # journal is an ISCE3 dependency; if its API shifts, the messages are
        # cosmetic and not worth failing the run over
        pass


_silence_third_party_noise()

from nisar.products.readers.Raw import Raw  # noqa: E402  (import after silencing)
from isce3.focus import ToneRemover          # noqa: E402  (import after silencing)


# ---------------------------------------------------------------------------
# CALTONE REMOVAL
# ---------------------------------------------------------------------------
#
# NISAR L0B raw data carries an instrument calibration tone (caltone) as a
# narrowband sinusoid in fast time (range). Left in place it injects an extra
# rank-1 term into every slow-time CPI, which biases the eigenvalue/knee
# structure the model learns from -- in practice it nudges the effective RFI
# count up by one. We coherently estimate and subtract that tone from the raw
# complex data BEFORE any RFI injection or SCM computation, exactly as the
# ST-EVD detection path (test_rfi_check.py) does.
#
# ToneRemover builds an ABSOLUTE phase reference exp(-1j*2*pi*f*arange(n))
# anchored at range sample 0, so remove_tone() must be handed the FULL-WIDTH
# range line (length == the dataset's range width), aligned to sample 0. It is
# therefore applied to whole range lines and the CPI window is sliced out only
# afterwards -- never fed a pre-sliced sub-tile, which would carry the wrong
# tone phase.

# Estimation block length for ToneRemover (matches the ST-EVD detection path).
CALTONE_WINDOW_SIZE = 64

# Fallback caltone frequency (Hz) when the DRT phase-step telemetry is absent.
CALTONE_DEFAULT_FREQ_HZ = 1214.88e6
CALTONE_LO_HZ = 1200e6
CALTONE_CLOCK_HZ = 240e6


def parse_caltone_freq_from_drt(raw: Raw, txrx_pol: str) -> float:
    """
    Caltone frequency (Hz) for one TxRx polarization, read from the DRT
    CALTONE phase-step telemetry. Falls back to CALTONE_DEFAULT_FREQ_HZ when
    the telemetry path is missing. (Mirrors test_rfi_check.py.)
    """
    path = (f'{raw.TelemetryPath}/DRT/MISC/'
            f'CP_IFSW_CALTONE_PHASE_STEP_{txrx_pol[1]}')
    with h5py.File(raw.filename, mode='r', swmr=True) as f:
        try:
            ds = f[path]
        except KeyError:
            print(f'  caltone: missing "{path}"; using default '
                  f'{CALTONE_DEFAULT_FREQ_HZ} Hz')
            return CALTONE_DEFAULT_FREQ_HZ
        i_cal = np.median(ds[()]).astype(int)
        return (i_cal / 2 ** 32) * CALTONE_CLOCK_HZ + CALTONE_LO_HZ


def build_tone_remover(raw: Raw, freq: str, pol: str, num_rng_samples: int):
    """
    Construct a ToneRemover sized to the full range width for one channel,
    plus the caltone frequency used (for provenance).

    Returns
    -------
    remover : ToneRemover
    caltone_freq : float   caltone frequency in Hz
    """
    tx_pol = pol[0]
    fc, fs, _, _ = raw.getChirpParameters(freq, tx_pol)
    caltone_freq = parse_caltone_freq_from_drt(raw, pol)
    remover = ToneRemover((caltone_freq - fc) / fs, num_rng_samples,
                          CALTONE_WINDOW_SIZE)
    return remover, caltone_freq


def remove_caltone_lines(lines: np.ndarray, remover: ToneRemover) -> np.ndarray:
    """
    Subtract the caltone from each full-width range line (pulse row) in place.
    `lines` is (num_pulses, num_rng_samples) and its width MUST equal the width
    the remover was built with.
    """
    for ip in range(lines.shape[0]):
        lines[ip] = remover.remove_tone(lines[ip])
    return lines


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Standard CPI tile: 16 pulses x 250 range samples
CPI_LEN_DEFAULT = 16
CPI_WIDTH_DEFAULT = 250

# Valid training region (slow time): 74298 pulses -> 4643 pulse tiles
PULSE_START_DEFAULT = 813924
PULSE_END_DEFAULT = 888222

# Default range window (fast time): 23000 samples -> 92 range tiles
RANGE_START_DEFAULT = 2000
RANGE_END_DEFAULT = 25000

# RFI band count drawn per tile (both ends inclusive). The drawn count IS the
# label; drawing 0 leaves the tile untouched, which is how clean tiles get
# randomly interspersed through the set.
MIN_BANDS_DEFAULT = 0
MAX_BANDS_DEFAULT = 6

# Per-band jammer-to-signal ratio range, both ends inclusive. Each band draws its
# own JSR uniformly from this range, relative to the tile's own baseline power.
JSR_MIN_DB_DEFAULT = 3.0
JSR_MAX_DB_DEFAULT = 30.0

# Gap-exclusion covariance thresholds (same defaults as read_nisar_isce3.py)
OFF_DIAG_OVERLAP_RATIO_DEFAULT = 0.25
DIAG_VALID_RATIO_DEFAULT = 0.20

# Seed-namespace salts. These keep independent roles in disjoint SeedSequence
# entropy namespaces so no two random streams can ever alias.
SALT_INJECT = 0xA1
SALT_PLOT = 0xB2

# Channel ids also enter the seed entropy tuples, so the same (pt, rt) tile gets
# an independent injection realization in each frequency/polarization channel
# instead of the identical one repeated across channels.
FREQ_IDS = {'A': 0, 'B': 1}
POL_IDS = {'HH': 0, 'HV': 1, 'VH': 2, 'VV': 3}

SEED_DEFAULT = 0
PLOT_SEED_DEFAULT = 99

# Pulses read per chunk; snapped down to a whole number of CPI tiles.
PULSE_CHUNK_DEFAULT = 1600

N_PLOT_BLOCKS_DEFAULT = 12

# Y-axis for the eigenvalue plots, in dB. The bottom is pinned at 0 dB; the top
# is the largest eigenvalue seen across ALL selected blocks, plus a small margin
# so the peak is not drawn flush against the axis. Both eigenvalue figures share
# this range, so every block is on the same scale and a high-power block is never
# clipped. Set EV_YLIM_TOP_MARGIN_DB to 0.0 for the bare maximum.
EV_YLIM_BOTTOM_DB = 0.0
EV_YLIM_TOP_MARGIN_DB = 2.0
EV_YLIM_MIN_TOP_DB = 10.0   # floor on the top, so a flat low-power set is not squashed

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
    """Label and provenance for one CPI tile."""
    knee: int                       # number of injected bands (0..MAX_BANDS)
    bands: List[BandMeta]
    signal_power_db: float          # tile baseline power, 10*log10(mean|x|^2)
    valid_fraction: float


# ---------------------------------------------------------------------------
# GAP-EXCLUSION COVARIANCE (convention taken from read_nisar_isce3.py)
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
    Compute the SCM, its descending LINEAR eigenvalues, its LINEAR diagonal,
    and the per-index diagonal validity mask for one CPI tile.

    Uses the gap-exclusion covariance when a mask is supplied, otherwise the
    plain (M @ M^H) / K estimate.

    The diagonal is the entry-level counterpart to the eigenvalue profile: RFI
    shows up as a jump in specific pulse rows of the diagonal, whereas a clean
    tile's diagonal stays comparatively flat across rows. diag_valid_idx marks
    which rows had enough non-gap samples to be trusted, so a downstream jump
    metric can skip entries that are invalid rather than real.

    Returns
    -------
    scm : (M, M) complex64
    eigvals : (M,) float32, descending, linear scale
    diag_lin : (M,) float64, linear scale, unnormalized power per pulse row
    diag_valid_idx : (M,) bool
    """
    if cpi_mask is not None:
        scm, diag_valid_idx = compute_gap_exclusion_cov(
            cpi,
            mask_valid_cpi=cpi_mask,
            off_diag_overlap_ratio=off_diag_overlap_ratio,
            diag_valid_ratio=diag_valid_ratio,
        )
    else:
        M, K = cpi.shape
        scm = ((cpi @ cpi.conj().T) / K).astype(np.complex64)
        diag_valid_idx = np.ones(M, dtype=bool)

    eigvals = np.linalg.eigvalsh(scm)          # ascending, real
    eigvals = np.sort(eigvals)[::-1]           # descending

    diag_lin = np.real(np.diag(scm)).astype(np.float64)

    return scm, eigvals.astype(np.float32), diag_lin, diag_valid_idx


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


def channel_id(freq: str, pol: str) -> int:
    """
    Stable integer id for a (frequency, polarization) channel, used to keep the
    random streams of different channels independent.
    """
    if freq not in FREQ_IDS:
        raise ValueError(f"Unknown frequency '{freq}'")
    if pol not in POL_IDS:
        raise ValueError(f"Unknown polarization '{pol}'")
    return FREQ_IDS[freq] * len(POL_IDS) + POL_IDS[pol]


def make_tile_seed_seq(seed: int, chan: int, pulse_tile: int,
                       range_tile: int) -> np.random.SeedSequence:
    """
    Injection SeedSequence for one tile of one channel. Depends only on
    (seed, chan, pt, rt), so a tile's contamination is reproducible
    independently of every other tile and every other channel.
    """
    return np.random.SeedSequence(
        [int(seed), SALT_INJECT, int(chan), int(pulse_tile), int(range_tile)]
    )


def draw_n_bands(tile_seed_seq, min_bands, max_bands):
    """
    Draw the band count (== the label) for one tile from a dedicated child of
    the tile stream, so the count draw is decoupled from the band-row and
    coefficient draws.
    """
    count_seed = tile_seed_seq.spawn(1)[0]
    rng_count = np.random.default_rng(count_seed)
    return int(rng_count.integers(min_bands, max_bands + 1))


def inject_rfi_bands(cpi, cpi_mask, n_bands, jsr_min_db, jsr_max_db, tile_seed_seq):
    """
    Overlay n_bands synthetic Gaussian RFI bands onto a real CPI tile.

    The n_bands pulse rows are drawn WITHOUT REPLACEMENT, as a uniformly chosen
    distinct subset of [0, cpi_len) -- equivalently, one of the
    C(cpi_len, n_bands) row combinations. Each occupied row contributes exactly
    one rank-1 term to the SCM, so the label equals the number of elevated
    eigenvalues by construction. (With replacement, two bands could collide on a
    row and yield fewer elevated eigenvalues than the label claims: at
    cpi_len=16 that happens for 50% of 5-band tiles and 66% of 6-band tiles,
    which is unlearnable label noise.)

    Each band draws its own JSR uniformly from [jsr_min_db, jsr_max_db] dB,
    relative to the tile's own baseline power, so bands within a tile generally
    differ in strength. Each band also gets its own spawned RNG child stream, so
    bands are mutually uncorrelated; the row subset comes from the tile's own
    stream, and the JSR draws from a separate child stream.

    n_bands == 0 returns the tile untouched with a knee = 0 label: that is the
    clean case, interspersed at random through the set.

    Returns
    -------
    tile : (M, K) complex64
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

    if n_bands > M:
        raise ValueError(
            f"n_bands ({n_bands}) exceeds cpi_len ({M}); rows are drawn without "
            f"replacement, so at most {M} bands can be placed"
        )

    rng_struct = np.random.default_rng(tile_seed_seq)

    # Dedicated JSR stream: keeps the strength draws decoupled from placement
    # and from the per-band coefficient vectors.
    rng_jsr = np.random.default_rng(tile_seed_seq.spawn(1)[0])
    band_seeds = tile_seed_seq.spawn(n_bands)

    # Rows WITHOUT replacement: a uniformly chosen distinct subset of the M pulse
    # rows, so every band occupies its own row and knee == number of elevated
    # eigenvalues exactly. Sorted only so the stored row list is deterministic.
    local_rows = np.sort(rng_struct.choice(M, size=n_bands, replace=False))

    rfi = np.zeros((M, K), dtype=np.complex64)
    bands: List[BandMeta] = []

    for b in range(n_bands):
        jsr_db = float(rng_jsr.uniform(jsr_min_db, jsr_max_db))
        band_power = signal_power * (10.0 ** (jsr_db / 10.0))
        sigma = np.sqrt(band_power / 2.0)   # half the power in real, half in imag

        local_row = int(local_rows[b])
        rng_band = np.random.default_rng(band_seeds[b])

        doppler_freq = rng_band.uniform(-0.5, 0.5)
        range_coeff = (
            rng_band.standard_normal(K) + 1j * rng_band.standard_normal(K)
        ) * sigma
        phase = np.exp(1j * 2.0 * np.pi * doppler_freq * local_row)

        rfi[local_row, :] += (phase * range_coeff).astype(np.complex64)
        bands.append(BandMeta(local_row=local_row, jsr_db=jsr_db))

    tile = (cpi + rfi).astype(np.complex64)
    meta = TileMeta(knee=n_bands, bands=bands,
                    signal_power_db=signal_power_db,
                    valid_fraction=valid_fraction)
    return tile, meta


# ---------------------------------------------------------------------------
# HDF5 WRITER
# ---------------------------------------------------------------------------

class TileWriter:
    """
    Append-as-you-go writer for the per-tile record arrays.

    Every dataset is resizable along the tile axis so the region can be
    streamed in pulse chunks without ever holding the whole set in memory.
    """

    def __init__(self, f, cpi_len, cpi_width, max_bands, save_cpi):
        self.f = f
        self.n = 0
        self.save_cpi = save_cpi

        def mk(name, shape, dtype, chunks, **kw):
            return f.create_dataset(
                name,
                shape=(0,) + shape,
                maxshape=(None,) + shape,
                dtype=dtype,
                chunks=(chunks,) + shape,
                **kw,
            )

        self.labels = mk('labels', (), np.int8, 4096)
        self.jsr_db = mk('jsr_db', (max_bands,), np.float32, 4096)
        self.band_rows = mk('band_rows', (max_bands,), np.int8, 4096)
        self.eigenvalues = mk('eigenvalues', (cpi_len,), np.float32, 2048)
        self.diagonal = mk('diagonal', (cpi_len,), np.float32, 2048)
        self.diag_valid_idx = mk('diag_valid_idx', (cpi_len,), bool, 2048)
        self.signal_power_db = mk('signal_power_db', (), np.float32, 4096)
        self.valid_fraction = mk('valid_fraction', (), np.float32, 4096)
        self.tile_pulse = mk('tile_pulse', (), np.int32, 4096)
        self.tile_range = mk('tile_range', (), np.int32, 4096)

        self.labels.attrs['description'] = 'knee: number of injected RFI bands (0 = clean)'
        self.jsr_db.attrs['description'] = (
            'per-band JSR in dB, each drawn uniformly from [jsr_min_db, jsr_max_db]; '
            'NaN-padded to max_bands; all-NaN row means clean tile'
        )
        self.band_rows.attrs['description'] = 'local pulse row of each band, -1 padded'
        self.eigenvalues.attrs['description'] = (
            'SCM eigenvalues, descending, LINEAR scale (take 10*log10 for dB)'
        )
        self.diagonal.attrs['description'] = (
            'SCM diagonal, LINEAR scale, unnormalized power per pulse row; '
            'RFI tends to show up as a jump on the injected row(s), a clean '
            'row is comparatively flat'
        )
        self.diag_valid_idx.attrs['description'] = (
            'per-index bool mask: True where that diagonal entry had enough '
            'valid (non-gap) samples to be trusted'
        )

        if save_cpi:
            self.cpi = f.create_dataset(
                'cpi',
                shape=(0, cpi_len, cpi_width),
                maxshape=(None, cpi_len, cpi_width),
                dtype=np.complex64,
                chunks=(16, cpi_len, cpi_width),
                compression='gzip',
                compression_opts=4,
            )
            self.cpi.attrs['description'] = 'complex CPI tile, RFI already overlaid'
        else:
            self.cpi = None

    def append(self, labels, jsr_db, band_rows, eigenvalues, diagonal, diag_valid_idx,
               signal_power_db, valid_fraction, tile_pulse, tile_range, cpi=None):
        """Append one batch of tile records (arrays with a leading tile axis)."""
        m = len(labels)
        new_n = self.n + m

        for dset, arr in (
            (self.labels, labels),
            (self.jsr_db, jsr_db),
            (self.band_rows, band_rows),
            (self.eigenvalues, eigenvalues),
            (self.diagonal, diagonal),
            (self.diag_valid_idx, diag_valid_idx),
            (self.signal_power_db, signal_power_db),
            (self.valid_fraction, valid_fraction),
            (self.tile_pulse, tile_pulse),
            (self.tile_range, tile_range),
        ):
            dset.resize(new_n, axis=0)
            dset[self.n:new_n] = arr

        if self.cpi is not None and cpi is not None:
            self.cpi.resize(new_n, axis=0)
            self.cpi[self.n:new_n] = cpi

        self.n = new_n


def write_root_attrs(f, args, freq, pol, p_start, p_end, r_start, r_end,
                     n_pulse_tiles, n_range_tiles, use_mask):
    """File-level generation config: everything needed to reproduce this set."""
    f.attrs['granule'] = os.path.basename(args.l0b_file)
    f.attrs['granule_path'] = args.l0b_file
    f.attrs['frequency'] = freq
    f.attrs['polarization'] = pol

    f.attrs['pulse_start'] = p_start
    f.attrs['pulse_end'] = p_end
    f.attrs['range_start'] = r_start
    f.attrs['range_end'] = r_end
    f.attrs['n_pulse_tiles'] = n_pulse_tiles
    f.attrs['n_range_tiles'] = n_range_tiles
    f.attrs['n_tiles'] = n_pulse_tiles * n_range_tiles

    f.attrs['cpi_len'] = args.cpi_len
    f.attrs['cpi_width'] = args.cpi_width

    f.attrs['min_bands'] = args.min_bands
    f.attrs['max_bands'] = args.max_bands
    # Labels are the drawn band count, so the label SPACE is always 0..max_bands
    # even when min_bands > 0 (an RFI-only set simply never emits label 0). The
    # classifier head must be sized to the label space, not to the drawn range.
    f.attrs['n_classes'] = args.max_bands + 1
    f.attrs['jsr_min_db'] = args.jsr_min_db
    f.attrs['jsr_max_db'] = args.jsr_max_db
    f.attrs['jsr_draw'] = 'per band, uniform in [jsr_min_db, jsr_max_db]'
    f.attrs['jsr_reference'] = 'tile baseline power, mean(|x|^2) over valid samples'

    f.attrs['seed'] = args.seed
    f.attrs['plot_seed'] = args.plot_seed
    f.attrs['channel_id'] = channel_id(freq, pol)
    f.attrs['salt_inject'] = SALT_INJECT
    f.attrs['salt_plot'] = SALT_PLOT
    f.attrs['seed_scheme'] = (
        'per-tile SeedSequence entropy = '
        '[seed, salt_inject, channel_id, pulse_tile, range_tile]; '
        'band count from a spawned child; each band coeff/Doppler from its own spawned child'
    )

    f.attrs['gap_exclusion_used'] = bool(use_mask)
    f.attrs['off_diag_overlap_ratio'] = args.off_diag_overlap_ratio
    f.attrs['diag_valid_ratio'] = args.diag_valid_ratio

    f.attrs['eigenvalue_scale'] = 'linear, descending'
    f.attrs['diagonal_scale'] = 'linear, unnormalized'
    f.attrs['cpi_stored'] = bool(args.save_cpi)
    f.attrs['generated_utc'] = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# CLASS-BALANCE REPORTING
# ---------------------------------------------------------------------------

def format_class_distribution(knee_counts, n_classes, indent='  '):
    """
    Human-readable class-separation report for one set of generated tiles.

    'Class separation' here is the per-label sample count and its share of the
    group -- i.e. how the generated tiles separate across the knee classes
    (0 = clean .. n_classes - 1). Returns (text, total).
    """
    total = int(sum(int(v) for v in knee_counts.values()))
    parts = []
    for k in range(n_classes):
        c = int(knee_counts.get(k, 0))
        frac = (100.0 * c / total) if total else 0.0
        parts.append(f"knee={k}: {c} ({frac:.1f}%)")
    text = (f"{indent}total samples: {total}\n"
            f"{indent}class separation: " + ", ".join(parts))
    return text, total


# ---------------------------------------------------------------------------
# SET GENERATION
# ---------------------------------------------------------------------------

def resolve_window(args, total_pulses, total_range):
    """Resolve the pulse/range window and snap it down to whole CPI tiles."""
    p_start = args.pulse_start
    p_end = min(args.pulse_end, total_pulses)
    r_start = args.range_start
    r_end = min(args.range_end, total_range) if args.range_end is not None else total_range

    n_pulse_tiles = (p_end - p_start) // args.cpi_len
    n_range_tiles = (r_end - r_start) // args.cpi_width
    p_end = p_start + n_pulse_tiles * args.cpi_len
    r_end = r_start + n_range_tiles * args.cpi_width

    if n_pulse_tiles <= 0 or n_range_tiles <= 0:
        raise ValueError(
            f"Window is smaller than one CPI tile ({args.cpi_len} x {args.cpi_width})"
        )

    return p_start, p_end, r_start, r_end, n_pulse_tiles, n_range_tiles


def generate_dataset(raw, freq, pol, args, out_dir):
    """
    Build the labeled training set for one polarization.

    Streams the region in pulse chunks, injects RFI per tile, computes the
    gap-exclusion SCM and its eigenvalues, and appends the per-tile records to
    HDF5 as it goes.

    Returns
    -------
    plot_records : list[dict]   selected tiles retained for plotting
    knee_counts : dict[int, int]
    out_path : str
    """
    cpi_len = args.cpi_len
    cpi_width = args.cpi_width
    max_bands = args.max_bands

    dataset = raw.getRawDataset(freq, pol)
    total_pulses, total_range = dataset.shape

    p_start, p_end, r_start, r_end, n_pulse_tiles, n_range_tiles = resolve_window(
        args, total_pulses, total_range
    )
    n_tiles = n_pulse_tiles * n_range_tiles

    chan = channel_id(freq, pol)
    plot_tiles = select_plot_tiles(
        args.plot_seed, chan, n_pulse_tiles, n_range_tiles, args.n_plot_blocks
    )
    plot_lookup = set(plot_tiles)

    out_path = os.path.join(out_dir, f"rfi_data_{freq}_{pol}.h5")
    print(f"\n[{freq}-{pol}]")
    print(f"  pulses [{p_start}:{p_end}]  range [{r_start}:{r_end}]")
    print(f"  tile grid: {n_pulse_tiles} x {n_range_tiles} = {n_tiles} tiles")
    print(f"  storing CPI tiles: {args.save_cpi}")
    print(f"  -> {out_path}")

    use_mask = args.compute_subswath_mask
    plot_records = []
    knee_counts = {}

    # Build the caltone remover once per channel, sized to the FULL range width
    # so remove_tone() sees each range line at its true sample offset.
    if args.remove_caltone:
        remover, caltone_freq = build_tone_remover(raw, freq, pol, total_range)
        print(f"  caltone removal ON  (f_caltone = {caltone_freq/1e6:.4f} MHz, "
              f"window = {CALTONE_WINDOW_SIZE})")
    else:
        remover, caltone_freq = None, None
        print("  caltone removal OFF")

    chunk_tiles = max(1, args.pulse_chunk // cpi_len)

    with h5py.File(out_path, 'w') as f:
        write_root_attrs(f, args, freq, pol, p_start, p_end, r_start, r_end,
                         n_pulse_tiles, n_range_tiles, use_mask)
        f.attrs['caltone_removed'] = bool(args.remove_caltone)
        if caltone_freq is not None:
            f.attrs['caltone_freq_hz'] = float(caltone_freq)
            f.attrs['caltone_window_size'] = CALTONE_WINDOW_SIZE
        writer = TileWriter(f, cpi_len, cpi_width, max_bands, args.save_cpi)

        for chunk_start_tile in range(0, n_pulse_tiles, chunk_tiles):
            n_tiles_here = min(chunk_tiles, n_pulse_tiles - chunk_start_tile)
            cp0 = p_start + chunk_start_tile * cpi_len
            cp1 = cp0 + n_tiles_here * cpi_len

            if remover is not None:
                # Read the full-width lines, subtract the caltone at the correct
                # absolute range phase, then slice out the CPI window.
                raw_full = np.ascontiguousarray(
                    read_raw_data_batch(
                        raw, freq, pol,
                        pulse_slice=slice(cp0, cp1),
                        range_slice=slice(0, total_range),
                    )
                ).astype(np.complex64)
                remove_caltone_lines(raw_full, remover)
                raw_chunk = raw_full[:, r_start:r_end]
            else:
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

            n_batch = n_tiles_here * n_range_tiles
            b_labels = np.zeros(n_batch, dtype=np.int8)
            b_jsr = np.full((n_batch, max_bands), np.nan, dtype=np.float32)
            b_rows = np.full((n_batch, max_bands), -1, dtype=np.int8)
            b_eigs = np.zeros((n_batch, cpi_len), dtype=np.float32)
            b_diag = np.zeros((n_batch, cpi_len), dtype=np.float32)
            b_diag_valid = np.zeros((n_batch, cpi_len), dtype=bool)
            b_sig = np.zeros(n_batch, dtype=np.float32)
            b_vfrac = np.zeros(n_batch, dtype=np.float32)
            b_pulse = np.zeros(n_batch, dtype=np.int32)
            b_range = np.zeros(n_batch, dtype=np.int32)
            b_cpi = (
                np.zeros((n_batch, cpi_len, cpi_width), dtype=np.complex64)
                if args.save_cpi else None
            )

            k = 0
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

                    tile_ss = make_tile_seed_seq(args.seed, chan, pt, rt)
                    n_bands = draw_n_bands(tile_ss, args.min_bands, args.max_bands)
                    tile, meta = inject_rfi_bands(
                        cpi, cpi_mask, n_bands,
                        args.jsr_min_db, args.jsr_max_db, tile_ss
                    )

                    scm, eigvals, diag_lin, diag_valid_idx = compute_scm_and_eigs(
                        tile, cpi_mask,
                        args.off_diag_overlap_ratio,
                        args.diag_valid_ratio,
                    )

                    b_labels[k] = meta.knee
                    for bi, band in enumerate(meta.bands):
                        b_jsr[k, bi] = band.jsr_db      # stays NaN for a clean tile
                        b_rows[k, bi] = band.local_row
                    b_eigs[k] = eigvals                 # linear scale, descending
                    b_diag[k] = diag_lin                # linear scale, per pulse row
                    b_diag_valid[k] = diag_valid_idx
                    b_sig[k] = meta.signal_power_db
                    b_vfrac[k] = meta.valid_fraction
                    b_pulse[k] = abs_p0
                    b_range[k] = abs_r0
                    if b_cpi is not None:
                        b_cpi[k] = tile

                    knee_counts[meta.knee] = knee_counts.get(meta.knee, 0) + 1

                    if (pt, rt) in plot_lookup:
                        plot_records.append({
                            'pt': pt, 'rt': rt,
                            'abs_p0': abs_p0, 'abs_r0': abs_r0,
                            'scm': scm.copy(),
                            'eigvals': eigvals.copy(),
                            'meta': meta,
                        })

                    k += 1

            writer.append(b_labels, b_jsr, b_rows, b_eigs, b_diag, b_diag_valid,
                          b_sig, b_vfrac, b_pulse, b_range, cpi=b_cpi)

            done = chunk_start_tile + n_tiles_here
            print(f"    pulse tiles {done}/{n_pulse_tiles}  ({writer.n} records written)")

        # Label histogram lives in the file so downstream code can weight classes
        hist = {str(k): int(v) for k, v in sorted(knee_counts.items())}
        f.attrs['label_histogram'] = json.dumps(hist)
        f.attrs['n_records'] = writer.n

    plot_records.sort(key=lambda rec: (rec['pt'], rec['rt']))

    report_text, _ = format_class_distribution(knee_counts, max_bands + 1)
    print(report_text)

    return plot_records, knee_counts, out_path


# ---------------------------------------------------------------------------
# PLOT TILE SELECTION (fixed, independent seed)
# ---------------------------------------------------------------------------

def select_plot_tiles(plot_seed, chan, n_pulse_tiles, n_range_tiles, n_blocks):
    """
    Pick n_blocks distinct (pulse_tile, range_tile) positions from a stream that
    is fully independent of the injection streams, so choosing plot blocks
    cannot perturb any tile's contamination. The channel id keeps each
    frequency/polarization looking at its own random selection of blocks.
    """
    ss = np.random.SeedSequence([int(plot_seed), SALT_PLOT, int(chan)])
    rng = np.random.default_rng(ss)

    n_available = n_pulse_tiles * n_range_tiles
    n_pick = min(n_blocks, n_available)
    flat = rng.choice(n_available, size=n_pick, replace=False)

    return sorted((int(i // n_range_tiles), int(i % n_range_tiles)) for i in flat)


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------

def plot_eigenvalue_profiles(records, freq, pol, out_dir, max_bands):
    """
    Two eigenvalue figures for the randomly selected blocks (fixed plot seed):

      Figure 1 -- all selected blocks overlaid, each line colored by its label
                  (knee), with the knee index marked so the drop-off after the
                  injected bands is visible.
      Figure 2 -- grid of the individual profiles, one panel per block.

    Eigenvalues are stored linear; they are plotted as 10 * log10(.).
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

    # Bottom pinned at 0 dB; top driven by the largest eigenvalue anywhere in
    # this set of blocks, so a high-power block is never cut off and every panel
    # stays on the same scale. Note eigenvalues below 0 dB fall off the bottom of
    # the axis by design.
    global_max_db = float(np.max(np.concatenate(profiles_db)))
    top = max(global_max_db + EV_YLIM_TOP_MARGIN_DB, EV_YLIM_MIN_TOP_DB)
    ylim = [EV_YLIM_BOTTOM_DB, top]

    n_below = int(np.sum(np.concatenate(profiles_db) < EV_YLIM_BOTTOM_DB))
    if n_below:
        print(f"  note: {n_below} eigenvalue points fall below "
              f"{EV_YLIM_BOTTOM_DB:.0f} dB and are clipped off the bottom of the axis")

    norm = mcolors.Normalize(vmin=0, vmax=max(max_bands, 1))
    cmap = cm.plasma

    # --- Figure 1: overlay ---------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(11, 6))
    for prof, knee in zip(profiles_db, knees):
        ax1.plot(ev_index, prof, color=cmap(norm(knee)), alpha=0.75, linewidth=1.2)
        if knee > 0:
            ax1.axvline(x=knee + 0.5, color=cmap(norm(knee)), linestyle=':',
                       linewidth=1.5, alpha=0.5)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig1.colorbar(sm, ax=ax1)
    cbar.set_label('Number of RFI eigenvalues (injected RFI bands)', fontsize=10)

    ax1.set_xlabel('Eigenvalue index (1-based, descending)', fontsize=11)
    ax1.set_ylabel('Eigenvalue (dB)', fontsize=11)
    ax1.set_ylim(ylim)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.set_title(
        f'Eigenvalue profiles -- {freq}-{pol}\n'
        f'{len(records)} randomly selected CPI blocks (fixed plot seed), '
        f'gap-exclusion SCM',
        fontsize=11,
    )
    fig1.tight_layout()
    path1 = os.path.join(out_dir, f'{freq}_{pol}_ev_overlay.png')
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
            ax.axvline(x=knee + 0.5, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
        label = ('CLEAN' if knee == 0
                else (f'{knee} RFI eigenvalue' if knee == 1 else f'{knee} RFI eigenvalues'))
        # Bands now differ in strength, so show the range actually realized here
        if meta.bands:
            jsrs = [b.jsr_db for b in meta.bands]
            jsr_str = f' | JSR {min(jsrs):.0f}-{max(jsrs):.0f} dB'
        else:
            jsr_str = ''
        ax.set_title(
            f'p={rec["abs_p0"]} r={rec["abs_r0"]} [{label}]{jsr_str}\n'
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

    fig2.suptitle(f'Eigenvalue profiles per selected block -- {freq}-{pol}', fontsize=12)
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    path2 = os.path.join(out_dir, f'{freq}_{pol}_ev_blocks.png')
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)

    print(f"  plots -> {os.path.basename(path1)}, {os.path.basename(path2)}")


def plot_scm_matrices(records, freq, pol, out_dir):
    """
    Grid of SCM magnitude heatmaps (20 * log10 |R_ij|, dB) for the same randomly
    selected blocks. Injected bands show up as bright rows/columns and as raised
    off-diagonal structure.
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
        if meta.bands:
            jsrs = [b.jsr_db for b in meta.bands]
            jsr_str = f'\nJSR {min(jsrs):.0f}-{max(jsrs):.0f} dB'
        else:
            jsr_str = ''
        ax.set_title(f'p={rec["abs_p0"]} r={rec["abs_r0"]} [{label}]{rows_str}{jsr_str}',
                     fontsize=7)
        ax.set_xlabel('Pulse j', fontsize=8)
        ax.set_ylabel('Pulse i', fontsize=8)
        ax.tick_params(labelsize=6)

    for ax in axes.flat[n:]:
        ax.axis('off')

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
        cbar.set_label('|SCM| (dB)', fontsize=10)

    fig.suptitle(f'Gap-exclusion SCM magnitude -- {freq}-{pol}', fontsize=12)
    path = os.path.join(out_dir, f'{freq}_{pol}_scm.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  plots -> {os.path.basename(path)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=('Generate a labeled RFI CPI training set from real NISAR L0B data. '
                     'Clean tiles (knee = 0) are interspersed at random.'),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('l0b_file', help='Input NISAR L0B HDF5 granule')
    parser.add_argument('--freq', choices=['A', 'B'], default=None,
                        help='Frequency to process. Default: every frequency in the granule.')
    parser.add_argument('--pol', default=None,
                        help='Polarization (HH/HV/VH/VV). Default: every pol in the granule.')

    parser.add_argument('--pulse-start', type=int, default=PULSE_START_DEFAULT)
    parser.add_argument('--pulse-end', type=int, default=PULSE_END_DEFAULT)
    parser.add_argument('--range-start', type=int, default=RANGE_START_DEFAULT)
    parser.add_argument('--range-end', type=int, default=RANGE_END_DEFAULT)

    parser.add_argument('--cpi-len', type=int, default=CPI_LEN_DEFAULT)
    parser.add_argument('--cpi-width', type=int, default=CPI_WIDTH_DEFAULT)
    parser.add_argument('--pulse-chunk', type=int, default=PULSE_CHUNK_DEFAULT,
                        help='Pulses read per chunk; snapped down to whole CPI tiles.')

    parser.add_argument('--min-bands', type=int, default=MIN_BANDS_DEFAULT)
    parser.add_argument('--max-bands', type=int, default=MAX_BANDS_DEFAULT)
    parser.add_argument('--jsr-min-db', type=float, default=JSR_MIN_DB_DEFAULT,
                        help='Lower bound of the per-band jammer-to-signal ratio, in dB.')
    parser.add_argument('--jsr-max-db', type=float, default=JSR_MAX_DB_DEFAULT,
                        help='Upper bound of the per-band jammer-to-signal ratio, in dB.')

    parser.add_argument('--remove-caltone', dest='remove_caltone',
                        action='store_true', default=True,
                        help='Subtract the instrument caltone from the raw data '
                             'before RFI injection / SCM (default: on).')
    parser.add_argument('--no-remove-caltone', dest='remove_caltone',
                        action='store_false',
                        help='Leave the caltone in the raw data (legacy behavior).')

    parser.add_argument('--compute-subswath-mask', action='store_true',
                        help='Use ISCE3 subswath boundaries for gap-exclusion SCM.')
    parser.add_argument('--off-diag-overlap-ratio', type=float,
                        default=OFF_DIAG_OVERLAP_RATIO_DEFAULT)
    parser.add_argument('--diag-valid-ratio', type=float,
                        default=DIAG_VALID_RATIO_DEFAULT)

    parser.add_argument('--output-dir', default='data/rfi_train')
    parser.add_argument('--save-cpi', action='store_true',
                        help='Also store the full complex CPI tiles (about 32 kB/tile).')

    parser.add_argument('--seed', type=int, default=SEED_DEFAULT,
                        help='Master seed for all RFI injection streams.')
    parser.add_argument('--plot-seed', type=int, default=PLOT_SEED_DEFAULT,
                        help='Fixed, independent seed for selecting plotted blocks.')
    parser.add_argument('--n-plot-blocks', type=int, default=N_PLOT_BLOCKS_DEFAULT)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.min_bands < 0 or args.max_bands < args.min_bands:
        raise ValueError('Require 0 <= min_bands <= max_bands')
    if args.max_bands > args.cpi_len:
        raise ValueError(
            f'max_bands ({args.max_bands}) cannot exceed cpi_len ({args.cpi_len}): '
            f'band rows are drawn without replacement'
        )
    if args.jsr_max_db < args.jsr_min_db:
        raise ValueError('Require jsr_min_db <= jsr_max_db')

    # The per-tile injection stream is keyed on (seed, channel, pulse_tile,
    # range_tile), where pulse_tile is RELATIVE to the start of the window. Two
    # different pulse windows generated with the SAME seed therefore replay the
    # same band counts, rows and JSRs tile-for-tile. Backgrounds differ, so this
    # is not label leakage, but a held-out set should not share an injection
    # realization with the set the model trained on. Use a distinct --seed for
    # every window.
    if args.seed == 0 and args.pulse_start != PULSE_START_DEFAULT:
        print('\n  NOTE: this is not the default training window but --seed is still 0. '
              'Use a distinct seed for held-out windows so their injection '
              'realizations are independent of the training set.\n')

    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 70)
    print('RFI CPI training set generation (real background, synthetic RFI)')
    print('=' * 70)
    print(f'  granule        : {args.l0b_file}')
    print(f'  pulse window   : [{args.pulse_start}, {args.pulse_end})')
    print(f'  range window   : [{args.range_start}, {args.range_end})')
    print(f'  CPI tile       : {args.cpi_len} x {args.cpi_width}')
    print(f'  bands per tile : {args.min_bands}..{args.max_bands} '
          f'(label = drawn count; 0 = clean, interspersed at random)')
    print(f'  JSR            : [{args.jsr_min_db}, {args.jsr_max_db}] dB above each tile '
          f'baseline power (drawn per band)')
    print(f'  gap exclusion  : {args.compute_subswath_mask}')
    print(f'  caltone removal: {args.remove_caltone}')
    print(f'  save CPI       : {args.save_cpi}')
    print(f'  seeds          : injection={args.seed}, plot={args.plot_seed}')

    raw = Raw(hdf5file=args.l0b_file)
    raw.parsePolarizations()

    # Default: every frequency in the granule, and within each, every pol.
    freqs = [args.freq] if args.freq else list(raw.polarizations.keys())

    channels = []
    for freq in freqs:
        if freq not in raw.polarizations:
            print(f"  WARNING: frequency {freq} not present in granule; skipping")
            continue
        avail = list(raw.polarizations[freq])
        if args.pol:
            if args.pol not in avail:
                print(f"  WARNING: {args.pol} not present in frequency {freq}; skipping")
                continue
            pols = [args.pol]
        else:
            pols = avail
        channels.extend((freq, pol) for pol in pols)

    if not channels:
        raise ValueError('No matching frequency/polarization channels in this granule')

    print(f'  channels       : ' + ', '.join(f'{f}-{p}' for f, p in channels))

    written = []
    overall_counts = {}
    group_totals = {}
    for freq, pol in channels:
        records, knee_counts, out_path = generate_dataset(raw, freq, pol, args, args.output_dir)
        plot_eigenvalue_profiles(records, freq, pol, args.output_dir, args.max_bands)
        plot_scm_matrices(records, freq, pol, args.output_dir)
        written.append(out_path)

        group_totals[f'{freq}-{pol}'] = int(sum(int(v) for v in knee_counts.values()))
        for k, v in knee_counts.items():
            overall_counts[k] = overall_counts.get(k, 0) + int(v)

    # Combined pool = every group concatenated. train_only.py splits this into
    # train/val by pulse tile (val_frac + split_buffer), so this is the total
    # sample count and class separation that feed train/val downstream.
    print('\n' + '=' * 70)
    print('TRAIN/VAL POOL SUMMARY (all groups combined)')
    print('=' * 70)
    for chan, tot in group_totals.items():
        print(f'  {chan:<8}: {tot} samples')
    overall_text, grand_total = format_class_distribution(overall_counts, args.max_bands + 1)
    print(f'  {"-"*40}')
    print(overall_text)

    print('\nDone. Wrote:')
    for path in written:
        print(f'  {path}')


if __name__ == '__main__':
    main()