#!/usr/bin/env python
"""
generate_unet_segmentation_data.py

Generate semantic segmentation training data for U-Net RFI detection.

Key differences from generate_data_from_preprocessed.py:
  1. Injects 2D RFI BLOBS (not full-pulse bands) with variable spatial extent
  2. Range contamination is PARTIAL: controlled by range_frac parameter
  3. Generates BINARY MASKS as ground truth for pixel-level segmentation
  4. JSR computed relative to VALID samples in the blob region only
  5. Saves complex tiles + binary masks (not eigenvalues/diagonal)

RFI Blob Model
--------------
Each blob is a 2D Gaussian-weighted region spanning:
  - Pulse extent: min_pulse_size to max_pulse_size rows
  - Range extent: min_range_frac to max_range_frac of VALID range samples

The blob does NOT necessarily extend across the entire pulse. Range coverage
is drawn as a FRACTION of valid samples per pulse, so blobs respect the ADC
gap / subswath geometry.

JSR is computed relative to the local signal power within the blob's valid
region, ensuring that weak and strong interference are calibrated against
the actual clutter return in that part of the tile, not the tile-wide average.

Binary Mask Generation
-----------------------
The ground truth mask is 1 where the Gaussian envelope exceeds mask_threshold
(default 0.3, meaning pixels at 30% of the blob's peak power are flagged).
This produces soft boundaries suitable for U-Net training while preserving
the blob's spatial structure.

Contamination Limit
-------------------
At most max_contamination_frac (default 0.30 = 30%) of valid pixels can be
flagged as RFI contaminated per tile. Blob injection stops early if adding
another blob would exceed this limit. This ensures realistic class balance
for training and prevents pathological cases where the entire tile is RFI.

Seeding and Reproducibility
----------------------------
Identical to generate_data_from_preprocessed.py: per-tile SeedSequence keyed
on [seed, SALT_INJECT, channel_id, pulse_tile, range_tile], with blob count
and each blob's spatial/power parameters spawned as independent children.

Usage
-----
    # Training set: variable blob counts, sizes, and JSR
    python generate_unet_segmentation_data.py clean_tiles.h5 granule.h5 \\
        --min-blobs 0 --max-blobs 8 \\
        --min-pulse-size 4 --max-pulse-size 24 \\
        --min-range-frac 0.15 --max-range-frac 0.90 \\
        --jsr-min-db 3 --jsr-max-db 30 \\
        --mask-threshold 0.3 \\
        --output-dir data/unet_train \\
        --seed 0

    # Paired test set: clean + forced contamination
    python generate_unet_segmentation_data.py clean_tiles.h5 granule.h5 \\
        --paired-test --max-blobs 8 \\
        --output-dir data/unet_paired_test \\
        --seed 1
"""

import os
import glob
import json
import argparse
import warnings
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import h5py


def _silence_third_party_noise():
    """Quiet routine warnings from L0B readers."""
    if os.environ.get('RFI_SHOW_WARNINGS'):
        return
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', message='.*hasInputDataException.*',
                            category=UserWarning)
    try:
        import journal
        for channel in ('nisar.reader', 'isce3.io', 'isce3.core'):
            journal.info(channel).deactivate()
            journal.warning(channel).deactivate()
    except Exception:
        pass


_silence_third_party_noise()

from nisar.products.readers.Raw import Raw  # noqa: E402
from isce3.focus import ToneRemover          # noqa: E402

try:
    from nisar.products.readers.Raw import caltone_frequency_from_raw  # noqa: E402
except ImportError:
    caltone_frequency_from_raw = None


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

CPI_LEN_DEFAULT = 256
CPI_WIDTH_DEFAULT = 256

MIN_BLOBS_DEFAULT = 0
MAX_BLOBS_DEFAULT = 8

# Blob size in PULSE dimension (azimuth): 4-24 pulses by default
MIN_PULSE_SIZE_DEFAULT = 4
MAX_PULSE_SIZE_DEFAULT = 24

# Blob size in RANGE dimension: fraction of valid range samples per pulse
# 0.15 = narrowband (15% of valid range), 0.90 = wideband (90% of valid range)
MIN_RANGE_FRAC_DEFAULT = 0.15
MAX_RANGE_FRAC_DEFAULT = 0.90

JSR_MIN_DB_DEFAULT = 2.0
JSR_MAX_DB_DEFAULT = 30.0

# Binary mask threshold: Gaussian envelope must exceed this fraction of the
# blob peak to be flagged as RFI (0.3 = 30% of peak, producing soft boundaries)
MASK_THRESHOLD_DEFAULT = 0.3

# Maximum fraction of valid pixels that can be RFI contaminated per tile
MAX_CONTAMINATION_FRAC_DEFAULT = 0.30

# Gaussian sigma scale: blob_size / SIGMA_SCALE
# 3.0 means the Gaussian extends to ~3σ at the blob's nominal size
SIGMA_SCALE_DEFAULT = 3.0

CALTONE_WINDOW_SIZE = 64
CALTONE_DEFAULT_FREQ_HZ = 1214.883e6
CALTONE_LO_HZ = 1200e6
CALTONE_CLOCK_HZ = 240e6

SALT_INJECT = 0xA1
FREQ_IDS = {'A': 0, 'B': 1}
POL_IDS = {'HH': 0, 'HV': 1, 'VH': 2, 'VV': 3}
SEED_DEFAULT = 0

EPS = 1e-12


# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------

@dataclass
class BlobMeta:
    """Descriptor for one RFI blob."""
    center_pulse: int       # pulse row center (can be out of bounds for edge blobs)
    center_range: int       # range sample center (can be out of bounds)
    pulse_size: float       # azimuth extent (pulses)
    range_size: float       # range extent (samples)
    jsr_db: float           # jammer-to-signal ratio in dB
    n_pixels: int           # number of pixels flagged in the binary mask


@dataclass
class TileMeta:
    """Label and provenance for one tile."""
    n_blobs: int                    # number of injected blobs
    blobs: List[BlobMeta]
    signal_power_db: float          # tile baseline power, 10*log10(mean|x|^2)
    valid_fraction: float


# ---------------------------------------------------------------------------
# CALTONE REMOVAL
# ---------------------------------------------------------------------------

def parse_caltone_freq_from_drt(raw: Raw, txrx_pol: str) -> float:
    """Caltone frequency from DRT telemetry, with fallback."""
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
    """Construct ToneRemover for full range width."""
    tx_pol = pol[0]
    fc, fs, _, _ = raw.getChirpParameters(freq, tx_pol)
    if caltone_frequency_from_raw is not None:
        caltone_freq = caltone_frequency_from_raw(raw, pol)
    else:
        caltone_freq = parse_caltone_freq_from_drt(raw, pol)
    remover = ToneRemover((caltone_freq - fc) / fs, num_rng_samples,
                          CALTONE_WINDOW_SIZE)
    return remover, caltone_freq


# ---------------------------------------------------------------------------
# L0B READING
# ---------------------------------------------------------------------------

def read_raw_tile(raw: Raw, freq: str, pol: str, p0: int, cpi_len: int,
                  r0: int, cpi_width: int, remover: ToneRemover = None):
    """Read and decode one tile, with optional caltone removal."""
    dataset = raw.getRawDataset(freq, pol)
    if remover is None:
        return np.asarray(dataset[p0:p0 + cpi_len, r0:r0 + cpi_width],
                          dtype=np.complex64)
    # Remove tone from full-width lines before slicing
    lines = np.asarray(dataset[p0:p0 + cpi_len, :], dtype=np.complex64)
    for ip in range(lines.shape[0]):
        lines[ip] = remover.remove_tone(lines[ip])
    return np.ascontiguousarray(lines[:, r0:r0 + cpi_width])


def get_subswath_mask(raw: Raw, freq: str, pol: str, p0: int, cpi_len: int,
                      r0: int, cpi_width: int):
    """Boolean valid-sample mask from subswath boundaries."""
    tx_pol = pol[0]
    pulse_indices = np.arange(p0, p0 + cpi_len)
    subswaths = raw.getSubSwaths(freq, tx_pol)
    swaths = subswaths[:, pulse_indices, :]
    mask = np.zeros((cpi_len, cpi_width), dtype=bool)
    if swaths is not None:
        for i in range(cpi_len):
            for start, end in swaths[:, i, :]:
                s = max(int(start) - r0, 0)
                e = min(int(end) - r0, cpi_width)
                if e > s:
                    mask[i, s:e] = True
    return mask


def amplitude_gap_mask(cpi, gap_frac=0.10):
    """Self-contained validity mask based on ADC fill level."""
    mean_mag = np.mean(np.abs(cpi), axis=0)
    peak = np.max(mean_mag)
    if peak < EPS:
        return np.ones(cpi.shape, dtype=bool)
    threshold = gap_frac * peak
    valid_rng = mean_mag >= threshold
    return np.tile(valid_rng, (cpi.shape[0], 1))


# ---------------------------------------------------------------------------
# POWER UTILITIES
# ---------------------------------------------------------------------------

def tile_signal_power(cpi, cpi_mask):
    """Baseline power: mean(|x|^2) over valid samples."""
    if cpi_mask is not None and cpi_mask.any():
        vals = cpi[cpi_mask]
    else:
        vals = cpi.ravel()
    return max(float(np.mean(np.abs(vals) ** 2)), EPS)


def channel_id(freq: str, pol: str) -> int:
    """Stable integer id for (frequency, polarization)."""
    if freq not in FREQ_IDS:
        raise ValueError(f"Unknown frequency '{freq}'")
    if pol not in POL_IDS:
        raise ValueError(f"Unknown polarization '{pol}'")
    return FREQ_IDS[freq] * len(POL_IDS) + POL_IDS[pol]


def make_tile_seed_seq(seed: int, chan: int, pulse_tile: int,
                       range_tile: int) -> np.random.SeedSequence:
    """Per-tile SeedSequence for reproducible injection."""
    return np.random.SeedSequence(
        [int(seed), SALT_INJECT, int(chan), int(pulse_tile), int(range_tile)]
    )


def draw_n_blobs(tile_seed_seq, min_blobs, max_blobs):
    """Draw blob count for one tile."""
    count_seed = tile_seed_seq.spawn(1)[0]
    rng_count = np.random.default_rng(count_seed)
    return int(rng_count.integers(min_blobs, max_blobs + 1))


# ---------------------------------------------------------------------------
# RFI BLOB INJECTION
# ---------------------------------------------------------------------------

def inject_rfi_blobs(
    cpi: np.ndarray,
    cpi_mask: np.ndarray,
    n_blobs: int,
    min_pulse_size: int,
    max_pulse_size: int,
    min_range_frac: float,
    max_range_frac: float,
    jsr_min_db: float,
    jsr_max_db: float,
    mask_threshold: float,
    sigma_scale: float,
    tile_seed_seq: np.random.SeedSequence,
    max_contamination_frac: float = MAX_CONTAMINATION_FRAC_DEFAULT,
) -> Tuple[np.ndarray, np.ndarray, TileMeta]:
    """
    Inject n_blobs 2D Gaussian RFI blobs into a clean tile.

    Parameters
    ----------
    cpi : (M, K) complex64
    cpi_mask : (M, K) bool or None, valid sample mask
    n_blobs : number of blobs to inject (0 = clean tile)
    min_pulse_size, max_pulse_size : blob extent in pulse dimension (rows)
    min_range_frac, max_range_frac : blob extent as fraction of valid range
        samples per pulse. 0.15 = narrowband, 0.90 = wideband.
    jsr_min_db, jsr_max_db : per-blob JSR draw range
    mask_threshold : Gaussian envelope threshold for binary mask (0.0-1.0)
    sigma_scale : blob_size / sigma_scale gives Gaussian sigma
    tile_seed_seq : SeedSequence for this tile's injection
    max_contamination_frac : maximum fraction of valid pixels that can be RFI
        contaminated (default 0.30). Blob injection stops early if exceeded.

    Returns
    -------
    tile : (M, K) complex64, contaminated
    binary_mask : (M, K) bool, ground truth segmentation mask
    meta : TileMeta
    """
    M, K = cpi.shape
    if cpi_mask is None:
        cpi_mask = np.ones((M, K), dtype=bool)

    signal_power = tile_signal_power(cpi, cpi_mask)
    signal_power_db = 10.0 * np.log10(signal_power)
    valid_fraction = float(cpi_mask.sum()) / cpi_mask.size

    if n_blobs == 0:
        meta = TileMeta(n_blobs=0, blobs=[], signal_power_db=signal_power_db,
                        valid_fraction=valid_fraction)
        binary_mask = np.zeros((M, K), dtype=bool)
        return cpi.astype(np.complex64), binary_mask, meta

    rng_struct = np.random.default_rng(tile_seed_seq)
    rng_jsr = np.random.default_rng(tile_seed_seq.spawn(1)[0])
    blob_seeds = tile_seed_seq.spawn(n_blobs)

    rfi = np.zeros((M, K), dtype=np.complex64)
    binary_mask = np.zeros((M, K), dtype=bool)
    blobs: List[BlobMeta] = []

    # Per-pulse valid range count for range_frac calculation
    valid_per_pulse = cpi_mask.sum(axis=1)  # (M,)

    # Contamination limit enforcement
    n_valid = cpi_mask.sum()
    max_contaminated_pixels = int(max_contamination_frac * n_valid)

    for b in range(n_blobs):
        # Check contamination limit before adding more blobs
        current_contamination = binary_mask.sum()
        if current_contamination >= max_contaminated_pixels:
            # Already at contamination limit; stop adding blobs
            break

        # 1. Draw blob spatial parameters
        pulse_size = rng_struct.uniform(min_pulse_size, max_pulse_size)
        range_frac = rng_struct.uniform(min_range_frac, max_range_frac)

        # Blob center: can be outside tile bounds for edge blobs
        center_pulse = rng_struct.uniform(-pulse_size, M + pulse_size)
        center_range = rng_struct.uniform(-K * range_frac, K * (1 + range_frac))

        # 2. Draw JSR for this blob
        jsr_db = float(rng_jsr.uniform(jsr_min_db, jsr_max_db))

        # 3. Compute blob power relative to VALID samples in blob region only
        # Create Gaussian envelope
        p_coords = np.arange(M, dtype=np.float32)[:, None] - center_pulse
        k_coords = np.arange(K, dtype=np.float32)[None, :] - center_range

        sigma_p = pulse_size / sigma_scale
        sigma_k = (range_frac * K) / sigma_scale  # range size in samples

        gauss_p = np.exp(-0.5 * (p_coords / sigma_p) ** 2)
        gauss_k = np.exp(-0.5 * (k_coords / sigma_k) ** 2)
        envelope = gauss_p * gauss_k  # (M, K)

        # Binary mask: envelope > threshold AND valid
        blob_mask = (envelope > mask_threshold) & cpi_mask

        if not blob_mask.any():
            # Blob fell entirely outside valid region; skip it
            continue

        # Check if adding this blob would exceed contamination limit
        new_contamination = (binary_mask | blob_mask).sum()
        if new_contamination > max_contaminated_pixels:
            # This blob would exceed limit; stop here
            break

        # Local signal power in blob region
        local_signal_power = np.mean(np.abs(cpi[blob_mask]) ** 2)
        blob_power = max(local_signal_power, signal_power) * (10.0 ** (jsr_db / 10.0))
        sigma = np.sqrt(blob_power / 2.0)

        # 4. Generate complex Gaussian RFI with Doppler modulation
        rng_blob = np.random.default_rng(blob_seeds[b])
        doppler = rng_blob.uniform(-0.5, 0.5)

        # Complex Gaussian coefficients
        real_noise = rng_blob.standard_normal((M, K)).astype(np.float32)
        imag_noise = rng_blob.standard_normal((M, K)).astype(np.float32)

        # Phase modulation: exp(2j * pi * doppler * pulse_index)
        phase_mod = np.exp(2j * np.pi * doppler * np.arange(M))[:, None]

        # Apply envelope and phase
        rfi_blob = sigma * envelope * phase_mod * (real_noise + 1j * imag_noise)

        # Only inject where valid
        rfi_blob *= cpi_mask.astype(np.float32)
        rfi += rfi_blob.astype(np.complex64)

        # Update binary mask
        binary_mask |= blob_mask

        # Store metadata
        n_pixels = int(blob_mask.sum())
        blobs.append(BlobMeta(
            center_pulse=int(center_pulse),
            center_range=int(center_range),
            pulse_size=float(pulse_size),
            range_size=float(range_frac * K),
            jsr_db=jsr_db,
            n_pixels=n_pixels,
        ))

    tile = (cpi + rfi).astype(np.complex64)
    meta = TileMeta(n_blobs=len(blobs), blobs=blobs,
                    signal_power_db=signal_power_db,
                    valid_fraction=valid_fraction)

    return tile, binary_mask, meta


# ---------------------------------------------------------------------------
# HDF5 WRITER
# ---------------------------------------------------------------------------

class SegmentationWriter:
    """Append-as-you-go writer for segmentation tiles + masks."""

    def __init__(self, f, cpi_len, cpi_width, max_blobs):
        self.f = f
        self.n = 0
        self.max_blobs = max_blobs

        def mk(name, shape, dtype, chunks, **kw):
            return f.create_dataset(
                name,
                shape=(0,) + shape,
                maxshape=(None,) + shape,
                dtype=dtype,
                chunks=(chunks,) + shape,
                **kw,
            )

        # Tiles and masks
        self.tiles = f.create_dataset(
            'tiles',
            shape=(0, cpi_len, cpi_width),
            maxshape=(None, cpi_len, cpi_width),
            dtype=np.complex64,
            chunks=(16, cpi_len, cpi_width),
            compression='gzip',
            compression_opts=4,
        )
        self.tiles.attrs['description'] = 'complex CPI tiles, RFI overlaid'

        self.masks = f.create_dataset(
            'masks',
            shape=(0, cpi_len, cpi_width),
            maxshape=(None, cpi_len, cpi_width),
            dtype=bool,
            chunks=(16, cpi_len, cpi_width),
            compression='gzip',
            compression_opts=4,
        )
        self.masks.attrs['description'] = (
            'binary ground truth masks: 1=RFI, 0=clean, per pixel'
        )

        self.valid = f.create_dataset(
            'valid',
            shape=(0, cpi_len, cpi_width),
            maxshape=(None, cpi_len, cpi_width),
            dtype=bool,
            chunks=(16, cpi_len, cpi_width),
            compression='gzip',
            compression_opts=4,
        )
        self.valid.attrs['description'] = 'validity mask: ADC gap / subswath'

        # Metadata
        self.n_blobs = mk('n_blobs', (), np.int8, 4096)
        self.blob_jsr_db = mk('blob_jsr_db', (max_blobs,), np.float32, 4096)
        self.blob_pulse_center = mk('blob_pulse_center', (max_blobs,), np.int16, 4096)
        self.blob_range_center = mk('blob_range_center', (max_blobs,), np.int16, 4096)
        self.blob_pulse_size = mk('blob_pulse_size', (max_blobs,), np.float32, 4096)
        self.blob_range_size = mk('blob_range_size', (max_blobs,), np.float32, 4096)
        self.blob_n_pixels = mk('blob_n_pixels', (max_blobs,), np.int32, 4096)

        self.signal_power_db = mk('signal_power_db', (), np.float32, 4096)
        self.valid_fraction = mk('valid_fraction', (), np.float32, 4096)
        self.tile_pulse = mk('tile_pulse', (), np.int32, 4096)
        self.tile_range = mk('tile_range', (), np.int32, 4096)
        self.pair_id = mk('pair_id', (), np.int32, 4096)

        # Add descriptions
        self.n_blobs.attrs['description'] = 'number of injected blobs (0=clean)'
        self.blob_jsr_db.attrs['description'] = 'per-blob JSR in dB, NaN-padded'
        self.pair_id.attrs['description'] = (
            'index into source clean file; in --paired-test mode, clean and '
            'contaminated records share the same pair_id'
        )

    def append(self, tiles_batch, masks_batch, valid_batch, meta_batch,
               tile_pulse_batch, tile_range_batch, pair_id_batch):
        """Append one batch of tiles."""
        m = len(tiles_batch)
        new_n = self.n + m

        # Resize and write tiles/masks
        self.tiles.resize(new_n, axis=0)
        self.masks.resize(new_n, axis=0)
        self.valid.resize(new_n, axis=0)
        self.tiles[self.n:new_n] = tiles_batch
        self.masks[self.n:new_n] = masks_batch
        self.valid[self.n:new_n] = valid_batch

        # Pack metadata
        n_blobs_arr = np.array([meta.n_blobs for meta in meta_batch], dtype=np.int8)
        signal_power_db_arr = np.array([meta.signal_power_db for meta in meta_batch],
                                       dtype=np.float32)
        valid_frac_arr = np.array([meta.valid_fraction for meta in meta_batch],
                                  dtype=np.float32)

        jsr_arr = np.full((m, self.max_blobs), np.nan, dtype=np.float32)
        pulse_center_arr = np.full((m, self.max_blobs), -1, dtype=np.int16)
        range_center_arr = np.full((m, self.max_blobs), -1, dtype=np.int16)
        pulse_size_arr = np.full((m, self.max_blobs), np.nan, dtype=np.float32)
        range_size_arr = np.full((m, self.max_blobs), np.nan, dtype=np.float32)
        n_pixels_arr = np.full((m, self.max_blobs), -1, dtype=np.int32)

        for i, meta in enumerate(meta_batch):
            for bi, blob in enumerate(meta.blobs):
                jsr_arr[i, bi] = blob.jsr_db
                pulse_center_arr[i, bi] = blob.center_pulse
                range_center_arr[i, bi] = blob.center_range
                pulse_size_arr[i, bi] = blob.pulse_size
                range_size_arr[i, bi] = blob.range_size
                n_pixels_arr[i, bi] = blob.n_pixels

        # Write metadata
        for dset, arr in [
            (self.n_blobs, n_blobs_arr),
            (self.blob_jsr_db, jsr_arr),
            (self.blob_pulse_center, pulse_center_arr),
            (self.blob_range_center, range_center_arr),
            (self.blob_pulse_size, pulse_size_arr),
            (self.blob_range_size, range_size_arr),
            (self.blob_n_pixels, n_pixels_arr),
            (self.signal_power_db, signal_power_db_arr),
            (self.valid_fraction, valid_frac_arr),
            (self.tile_pulse, tile_pulse_batch),
            (self.tile_range, tile_range_batch),
            (self.pair_id, pair_id_batch),
        ]:
            dset.resize(new_n, axis=0)
            dset[self.n:new_n] = arr

        self.n = new_n


def write_root_attrs(f, args, freq, pol, source_group, n_clean_tiles,
                     cpi_len, cpi_width, use_mask):
    """File-level generation config."""
    f.attrs['clean_h5'] = os.path.basename(args.clean_h5)
    f.attrs['clean_h5_path'] = args.clean_h5
    f.attrs['clean_h5_group'] = source_group
    f.attrs['granule'] = os.path.basename(args.l0b_file)
    f.attrs['granule_path'] = args.l0b_file
    f.attrs['frequency'] = freq
    f.attrs['polarization'] = pol
    f.attrs['target_tag'] = args.target_tag if args.target_tag else ''

    f.attrs['n_clean_tiles'] = n_clean_tiles
    f.attrs['cpi_len'] = cpi_len
    f.attrs['cpi_width'] = cpi_width

    f.attrs['min_blobs'] = args.min_blobs
    f.attrs['max_blobs'] = args.max_blobs
    f.attrs['min_pulse_size'] = args.min_pulse_size
    f.attrs['max_pulse_size'] = args.max_pulse_size
    f.attrs['min_range_frac'] = args.min_range_frac
    f.attrs['max_range_frac'] = args.max_range_frac
    f.attrs['jsr_min_db'] = args.jsr_min_db
    f.attrs['jsr_max_db'] = args.jsr_max_db
    f.attrs['jsr_reference'] = (
        'local signal power within each blob\'s valid region, '
        'mean(|x|^2) over blob_mask & cpi_mask'
    )
    f.attrs['mask_threshold'] = args.mask_threshold
    f.attrs['max_contamination_frac'] = args.max_contamination_frac
    f.attrs['sigma_scale'] = args.sigma_scale

    f.attrs['seed'] = args.seed
    f.attrs['channel_id'] = channel_id(freq, pol)
    f.attrs['salt_inject'] = SALT_INJECT
    f.attrs['seed_scheme'] = (
        'per-tile SeedSequence entropy = '
        '[seed, salt_inject, channel_id, pulse_tile, range_tile]; '
        'blob count from spawned child; each blob spatial/power params from '
        'its own spawned child'
    )

    f.attrs['mask_mode'] = args.mask_mode
    f.attrs['gap_exclusion_used'] = bool(use_mask)

    f.attrs['paired_test'] = bool(args.paired_test)
    if args.paired_test:
        f.attrs['paired_test_scheme'] = (
            'each source tile (pair_id) produced 2 records: one untouched '
            '(n_blobs=0) and one with blobs forced (n_blobs >= 1), both from '
            'identical raw CPI. Match rows by pair_id for clean vs contaminated.'
        )

    f.attrs['caltone_removed'] = bool(args.remove_caltone)
    f.attrs['generated_utc'] = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# CLASS BALANCE REPORTING
# ---------------------------------------------------------------------------

def format_blob_distribution(blob_counts, max_blobs, indent='  '):
    """Report blob count distribution."""
    total = int(sum(int(v) for v in blob_counts.values()))
    parts = []
    for k in range(max_blobs + 1):
        c = int(blob_counts.get(k, 0))
        frac = (100.0 * c / total) if total else 0.0
        parts.append(f"n_blobs={k}: {c} ({frac:.1f}%)")
    text = (f"{indent}total samples: {total}\n"
            f"{indent}blob count distribution: " + ", ".join(parts))
    return text, total


# ---------------------------------------------------------------------------
# GENERATION
# ---------------------------------------------------------------------------

def generate_dataset_for_group(raw, freq, pol, grp_name, pulse_idx, range_idx,
                                cpi_len, cpi_width, args, out_dir):
    """
    Generate segmentation training data for one freq/pol group.

    Returns
    -------
    blob_counts : dict[int, int], distribution of n_blobs
    out_path : str
    """
    chan = channel_id(freq, pol)
    n_tiles = len(pulse_idx)
    paired = bool(args.paired_test)

    suffix = '_paired' if paired else ''
    base_name = f"{args.target_tag}_unet_seg" if args.target_tag else "unet_seg"
    out_path = os.path.join(out_dir, f"{base_name}{suffix}_{freq}_{pol}.h5")
    print(f"\n[{freq}-{pol}] source group: {grp_name}  ({n_tiles} clean tiles)"
          + ("  [paired-test mode]" if paired else ""))
    print(f"  -> {out_path}")

    use_mask = args.mask_mode != 'none'
    blob_counts = {}

    # Caltone removal
    if args.remove_caltone:
        full_range = raw.getRawDataset(freq, pol).shape[1]
        remover, caltone_freq = build_tone_remover(raw, freq, pol, full_range)
        print(f"  caltone removal ON  (f_caltone = {caltone_freq/1e6:.4f} MHz, "
              f"window = {CALTONE_WINDOW_SIZE})")
    else:
        remover, caltone_freq = None, None
        print("  caltone removal OFF")

    with h5py.File(out_path, 'w') as f:
        write_root_attrs(f, args, freq, pol, grp_name, n_tiles, cpi_len,
                         cpi_width, use_mask)
        if caltone_freq is not None:
            f.attrs['caltone_freq_hz'] = float(caltone_freq)
            f.attrs['caltone_window_size'] = CALTONE_WINDOW_SIZE

        writer = SegmentationWriter(f, cpi_len, cpi_width, args.max_blobs)
        report_every = max(1, n_tiles // 10)

        for i in range(n_tiles):
            p0 = int(pulse_idx[i])
            r0 = int(range_idx[i])

            cpi = read_raw_tile(raw, freq, pol, p0, cpi_len, r0, cpi_width, remover)

            if args.mask_mode == 'subswath':
                cpi_mask = get_subswath_mask(raw, freq, pol, p0, cpi_len, r0, cpi_width)
            elif args.mask_mode == 'amplitude':
                cpi_mask = amplitude_gap_mask(cpi)
            else:
                cpi_mask = None

            pulse_tile = p0 // cpi_len
            range_tile = r0 // cpi_width
            tile_ss = make_tile_seed_seq(args.seed, chan, pulse_tile, range_tile)

            # In paired-test mode, contaminated copy must have blobs >= 1
            contam_min_blobs = max(args.min_blobs, 1) if paired else args.min_blobs
            n_blobs = draw_n_blobs(tile_ss, contam_min_blobs, args.max_blobs)

            tiles_batch = []
            masks_batch = []
            valid_batch = []
            meta_batch = []
            tile_pulse_batch = []
            tile_range_batch = []
            pair_id_batch = []

            if paired:
                # Untouched clean copy
                clean_tile = cpi.astype(np.complex64)
                clean_mask = np.zeros((cpi_len, cpi_width), dtype=bool)
                clean_meta = TileMeta(
                    n_blobs=0, blobs=[],
                    signal_power_db=10.0 * np.log10(tile_signal_power(cpi, cpi_mask)),
                    valid_fraction=(float(cpi_mask.sum()) / cpi_mask.size
                                    if cpi_mask is not None else 1.0),
                )
                tiles_batch.append(clean_tile)
                masks_batch.append(clean_mask)
                valid_batch.append(cpi_mask if cpi_mask is not None
                                   else np.ones((cpi_len, cpi_width), dtype=bool))
                meta_batch.append(clean_meta)
                tile_pulse_batch.append(p0)
                tile_range_batch.append(r0)
                pair_id_batch.append(i)
                blob_counts[0] = blob_counts.get(0, 0) + 1

            # Contaminated copy
            contam_tile, contam_mask, contam_meta = inject_rfi_blobs(
                cpi, cpi_mask, n_blobs,
                args.min_pulse_size, args.max_pulse_size,
                args.min_range_frac, args.max_range_frac,
                args.jsr_min_db, args.jsr_max_db,
                args.mask_threshold, args.sigma_scale,
                tile_ss,
                args.max_contamination_frac,
            )
            tiles_batch.append(contam_tile)
            masks_batch.append(contam_mask)
            valid_batch.append(cpi_mask if cpi_mask is not None
                               else np.ones((cpi_len, cpi_width), dtype=bool))
            meta_batch.append(contam_meta)
            tile_pulse_batch.append(p0)
            tile_range_batch.append(r0)
            pair_id_batch.append(i)
            blob_counts[contam_meta.n_blobs] = blob_counts.get(contam_meta.n_blobs, 0) + 1

            writer.append(
                np.array(tiles_batch),
                np.array(masks_batch),
                np.array(valid_batch),
                meta_batch,
                np.array(tile_pulse_batch, dtype=np.int32),
                np.array(tile_range_batch, dtype=np.int32),
                np.array(pair_id_batch, dtype=np.int32),
            )

            if (i + 1) % report_every == 0 or (i + 1) == n_tiles:
                print(f"    {i + 1}/{n_tiles} tiles processed ({writer.n} records written)")

        hist = {str(k): int(v) for k, v in sorted(blob_counts.items())}
        f.attrs['blob_count_histogram'] = json.dumps(hist)
        f.attrs['n_records'] = writer.n

    report_text, _ = format_blob_distribution(blob_counts, args.max_blobs)
    print(report_text)

    return blob_counts, out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('clean_h5',
                        help='Clean-tile source: directory of select_clean.py '
                             'outputs, a single flat file, or legacy grouped file')
    parser.add_argument('l0b_file', help='Source NISAR L0B HDF5 granule')

    parser.add_argument('--target-tag', default=None,
                        help='Optional label for background/scene type (e.g. "mountain")')
    parser.add_argument('--freq', default=None, help='Restrict to one frequency (e.g. "A")')
    parser.add_argument('--pol', default=None, help='Restrict to one polarization (e.g. "HH")')

    parser.add_argument('--cpi-width', type=int, default=None,
                        help=f'CPI width (default: {CPI_WIDTH_DEFAULT})')

    parser.add_argument('--min-blobs', type=int, default=MIN_BLOBS_DEFAULT)
    parser.add_argument('--max-blobs', type=int, default=MAX_BLOBS_DEFAULT)

    parser.add_argument('--min-pulse-size', type=int, default=MIN_PULSE_SIZE_DEFAULT,
                        help='Min blob extent in pulse dimension (azimuth)')
    parser.add_argument('--max-pulse-size', type=int, default=MAX_PULSE_SIZE_DEFAULT,
                        help='Max blob extent in pulse dimension')

    parser.add_argument('--min-range-frac', type=float, default=MIN_RANGE_FRAC_DEFAULT,
                        help='Min blob extent as fraction of valid range samples '
                             '(0.15 = narrowband, 15%% of range)')
    parser.add_argument('--max-range-frac', type=float, default=MAX_RANGE_FRAC_DEFAULT,
                        help='Max blob extent as fraction of valid range samples '
                             '(0.90 = wideband, 90%% of range)')

    parser.add_argument('--jsr-min-db', type=float, default=JSR_MIN_DB_DEFAULT)
    parser.add_argument('--jsr-max-db', type=float, default=JSR_MAX_DB_DEFAULT)

    parser.add_argument('--mask-threshold', type=float, default=MASK_THRESHOLD_DEFAULT,
                        help='Gaussian envelope threshold for binary mask (0.0-1.0); '
                             '0.3 = flag pixels at 30%% of blob peak')
    parser.add_argument('--max-contamination-frac', type=float, default=MAX_CONTAMINATION_FRAC_DEFAULT,
                        help='Maximum fraction of valid pixels that can be RFI contaminated '
                             '(default 0.30 = 30%%)')
    parser.add_argument('--sigma-scale', type=float, default=SIGMA_SCALE_DEFAULT,
                        help='Gaussian sigma = blob_size / sigma_scale (default 3.0)')

    parser.add_argument('--remove-caltone', dest='remove_caltone',
                        action='store_true', default=True,
                        help='Subtract caltone from raw data (default: on)')
    parser.add_argument('--no-remove-caltone', dest='remove_caltone',
                        action='store_false',
                        help='Leave caltone in raw data')

    parser.add_argument('--mask-mode', choices=['none', 'subswath', 'amplitude'],
                        default='subswath',
                        help='Validity mask: "subswath" uses ISCE3 boundaries, '
                             '"amplitude" uses ADC fill heuristic')

    parser.add_argument('--output-dir', default='data/unet_train')

    parser.add_argument('--paired-test', action='store_true',
                        help='Generate paired test set: each tile -> '
                             'untouched (n_blobs=0) + forced contamination (n_blobs>=1), '
                             'same pair_id')

    parser.add_argument('--seed', type=int, default=SEED_DEFAULT,
                        help='Master seed for RFI injection streams')

    return parser.parse_args()


def _read_flat_clean_file(path):
    """(freq, pol, name, pulse_idx, range_idx, cpi_len) from flat file."""
    with h5py.File(path, 'r') as f:
        freq = str(f.attrs['frequency'])
        pol = str(f.attrs['polarization'])
        pulse_idx = f['tile_pulse'][:]
        range_idx = f['tile_range'][:]
        eig_shape = f['eigenvalues'].shape if 'eigenvalues' in f else None
    cpi_len = eig_shape[1] if (eig_shape and len(eig_shape) == 2) else CPI_LEN_DEFAULT
    return freq, pol, os.path.basename(path), pulse_idx, range_idx, cpi_len


def iter_clean_channels(clean_path):
    """Yield (freq, pol, name, pulse_idx, range_idx, cpi_len) for all channels."""
    if os.path.isdir(clean_path):
        paths = sorted(glob.glob(os.path.join(clean_path, '*.h5')))
        if not paths:
            raise FileNotFoundError(f"No *.h5 files found in {clean_path}")
        n_ok = 0
        for p in paths:
            try:
                rec = _read_flat_clean_file(p)
            except (OSError, KeyError) as exc:
                print(f"[warn] skipping '{p}': not a readable flat clean file "
                      f"({type(exc).__name__}: {exc})")
                continue
            n_ok += 1
            yield rec
        if n_ok == 0:
            raise RuntimeError(f"No readable flat clean *.h5 files in {clean_path}")
        return

    with h5py.File(clean_path, 'r') as f:
        is_flat = ('tile_pulse' in f and 'frequency' in f.attrs)
        grouped = [k for k in f.keys() if isinstance(f[k], h5py.Group)]

    if is_flat:
        yield _read_flat_clean_file(clean_path)
        return

    for grp_name in grouped:
        with h5py.File(clean_path, 'r') as f:
            grp = f[grp_name]
            freq = str(grp.attrs['frequency'])
            pol = str(grp.attrs['polarization'])
            pulse_idx = grp['pulse_idx'][:]
            range_idx = grp['range_idx'][:]
            eig_shape = grp['eigenvalues'].shape
            cpi_len = eig_shape[1] if len(eig_shape) == 2 else CPI_LEN_DEFAULT
        yield freq, pol, grp_name, pulse_idx, range_idx, cpi_len


def main():
    args = parse_args()

    if args.min_blobs < 0 or args.max_blobs < args.min_blobs:
        raise ValueError('Require 0 <= min_blobs <= max_blobs')
    if args.jsr_max_db < args.jsr_min_db:
        raise ValueError('Require jsr_min_db <= jsr_max_db')
    if args.min_pulse_size < 1 or args.max_pulse_size < args.min_pulse_size:
        raise ValueError('Require 1 <= min_pulse_size <= max_pulse_size')
    if not (0.0 < args.min_range_frac <= args.max_range_frac <= 1.0):
        raise ValueError('Require 0 < min_range_frac <= max_range_frac <= 1.0')
    if not (0.0 <= args.mask_threshold <= 1.0):
        raise ValueError('Require 0.0 <= mask_threshold <= 1.0')
    if not (0.0 < args.max_contamination_frac <= 1.0):
        raise ValueError('Require 0.0 < max_contamination_frac <= 1.0')
    if args.paired_test and args.max_blobs < 1:
        raise ValueError('max_blobs must be >= 1 in --paired-test mode')

    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 70)
    if args.paired_test:
        print('U-Net Segmentation PAIRED test set generation')
        print('(each clean tile -> untouched + forced contamination, same pair_id)')
    else:
        print('U-Net Segmentation training set generation')
        print('(2D RFI blobs injected onto clean tiles with binary masks)')
    print('=' * 70)
    print(f'  clean tiles from : {args.clean_h5}')
    print(f'  source granule   : {args.l0b_file}')
    print(f'  target tag       : {args.target_tag or "(none)"}')
    if args.paired_test:
        print(f'  blobs per tile   : 0 (untouched) paired with '
              f'{max(args.min_blobs, 1)}..{args.max_blobs} (forced)')
    else:
        print(f'  blobs per tile   : {args.min_blobs}..{args.max_blobs} '
              f'(0 = clean, interspersed)')
    print(f'  blob pulse size  : {args.min_pulse_size}..{args.max_pulse_size} pulses')
    print(f'  blob range frac  : {args.min_range_frac:.2f}..{args.max_range_frac:.2f} '
          f'of valid range')
    print(f'  JSR              : [{args.jsr_min_db}, {args.jsr_max_db}] dB above '
          f'local blob signal power')
    print(f'  mask threshold   : {args.mask_threshold} (Gaussian envelope fraction)')
    print(f'  caltone removal  : {args.remove_caltone}')
    print(f'  mask mode        : {args.mask_mode}')
    print(f'  seed             : {args.seed}')

    raw = Raw(hdf5file=args.l0b_file)
    raw.parsePolarizations()

    written = []
    overall_counts = {}
    group_totals = {}

    for freq, pol, src_name, pulse_idx, range_idx, cpi_len in iter_clean_channels(args.clean_h5):
        if args.freq is not None and freq != args.freq:
            continue
        if args.pol is not None and pol != args.pol:
            continue

        cpi_width = args.cpi_width if args.cpi_width is not None else CPI_WIDTH_DEFAULT

        if len(pulse_idx) == 0:
            print(f"\n[warn] {src_name} ({freq}-{pol}) has no clean tiles; skipping")
            continue

        blob_counts, out_path = generate_dataset_for_group(
            raw, freq, pol, src_name, pulse_idx, range_idx,
            cpi_len, cpi_width, args, args.output_dir
        )
        written.append(out_path)

        group_totals[f'{freq}-{pol}'] = int(sum(int(v) for v in blob_counts.values()))
        for k, v in blob_counts.items():
            overall_counts[k] = overall_counts.get(k, 0) + int(v)

    if not written:
        raise RuntimeError('No matching freq/pol groups processed; check --freq/--pol')

    print('\n' + '=' * 70)
    print('TRAIN/VAL POOL SUMMARY (all groups combined)')
    print('=' * 70)
    for chan, tot in group_totals.items():
        print(f'  {chan:<8}: {tot} samples')
    overall_text, grand_total = format_blob_distribution(overall_counts, args.max_blobs)
    print(f'  {"-"*40}')
    print(overall_text)

    print('\nDone. Wrote:')
    for path in written:
        print(f'  {path}')


if __name__ == '__main__':
    main()
