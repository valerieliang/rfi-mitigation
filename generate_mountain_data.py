#!/usr/bin/env python
"""
generate_mountain_data.py

Build a labeled RFI training set by overlaying synthetic RFI bands onto the
EXACT pulse tiles that select_clean_mountain.py / filter_clean_mountains.py
already identified as clean mountain background.

Why re-read the raw CPI
------------------------
clean_mountains_filtered.h5 only stores the reduced eigenvalue and diagonal
profiles for each clean tile, not the underlying complex CPI matrix or the
full SCM. RFI injection (adding a rank-1 interference term to specific pulse
rows, then recomputing the covariance) has to happen on the raw complex data,
so this script uses the tile_pulse / tile_range locations recorded in the
filtered file to go back to the original L0B granule and re-read each tile's
raw CPI matrix before injecting.

Pipeline
--------
1. Open clean_mountains_filtered.h5 (or an unfiltered clean_mountains.h5; the
   schema is the same). For each freq_X_pol_Y group, read pulse_idx[] and
   range_idx[]: the absolute (row0, col0) location of every clean tile.
2. Open the source L0B granule and, for each recorded location, read back the
   cpi_len x cpi_width raw complex tile.
3. Draw a band count (the label) for the tile and inject that many synthetic
   Gaussian RFI bands, using the same JSR-referenced model as
   generate_rfi_data.py (see that file's docstring for the full derivation).
   n_bands == 0 leaves the tile untouched, so clean examples (knee = 0) are
   naturally interspersed alongside the RFI examples.
4. Recompute the gap-exclusion SCM on the (possibly) contaminated tile and
   store, per tile:
     - eigenvalues   (cpi_len,) linear scale, descending
     - diagonal      (cpi_len,) linear scale, unnormalized
     - diag_valid_idx (cpi_len,) bool, per-index diagonal validity mask
     - label / band metadata / signal power / valid fraction / tile location

Seeding
-------
Each tile's injection stream is keyed on
    [seed, SALT_INJECT, channel_id(freq, pol), pulse_tile, range_tile]
where pulse_tile = tile_pulse // cpi_len and range_tile = tile_range //
cpi_width (derived from the tile's own absolute location, since these tiles
are a scattered subset rather than a dense grid). This keeps injection fully
reproducible per tile and independent across channels, matching the seeding
convention in generate_rfi_data.py.

Usage
-----
    # Training set: one record per tile, band count drawn from [0, 6]
    python generate_mountain_data.py clean_mountains_filtered.h5 granule.h5 \
        --min-bands 0 --max-bands 6 \
        --jsr-min-db 3 --jsr-max-db 30 \
        --mask-mode subswath \
        --output-dir data/mountain_rfi_train \
        --seed 0

    # Paired test set: two records per tile (clean + forced-RFI), same background
    python generate_mountain_data.py clean_mountains_filtered.h5 granule.h5 \
        --paired-test --max-bands 6 \
        --jsr-min-db 3 --jsr-max-db 30 \
        --mask-mode subswath \
        --output-dir data/mountain_rfi_paired_test \
        --seed 1
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
    """Quiet the routine, non-actionable warnings the L0B readers emit."""
    if os.environ.get('RFI_SHOW_WARNINGS'):
        return

    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings(
        'ignore',
        message='.*hasInputDataException.*',
        category=UserWarning,
    )

    try:
        import journal
        for channel in ('nisar.reader', 'isce3.io', 'isce3.core'):
            journal.info(channel).deactivate()
            journal.warning(channel).deactivate()
    except Exception:
        pass


_silence_third_party_noise()

from nisar.products.readers.Raw import Raw  # noqa: E402  (import after silencing)


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

CPI_LEN_DEFAULT = 16
CPI_WIDTH_DEFAULT = 250

MIN_BANDS_DEFAULT = 0
MAX_BANDS_DEFAULT = 6

JSR_MIN_DB_DEFAULT = 3.0
JSR_MAX_DB_DEFAULT = 30.0

OFF_DIAG_OVERLAP_RATIO_DEFAULT = 0.20
DIAG_VALID_RATIO_DEFAULT = 0.15

SALT_INJECT = 0xA1

FREQ_IDS = {'A': 0, 'B': 1}
POL_IDS = {'HH': 0, 'HV': 1, 'VH': 2, 'VV': 3}

SEED_DEFAULT = 0

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
    knee: int                       # number of injected bands (0..max_bands)
    bands: List[BandMeta]
    signal_power_db: float          # tile baseline power, 10*log10(mean|x|^2)
    valid_fraction: float


# ---------------------------------------------------------------------------
# GAP-EXCLUSION SCM (returns per-index diagonal validity, unlike the
# generate_rfi_data.py variant which only returns the boolean mask used
# internally -- here we need it as an output, so the diagonal values and the
# mask are both surfaced explicitly)
# ---------------------------------------------------------------------------

def compute_gap_exclusion_scm(
    data: np.ndarray,
    *,
    mask_valid_cpi: np.ndarray = None,
    off_diag_overlap_ratio: float = OFF_DIAG_OVERLAP_RATIO_DEFAULT,
    diag_valid_ratio: float = DIAG_VALID_RATIO_DEFAULT,
):
    """
    Gap-excluded slow-time sample covariance matrix (SCM).

    Returns
    -------
    scm : (M, M) complex64, Hermitian, NOT yet normalized by range width.
    diag_valid_idx : (M,) bool array, True where the diagonal term had
        enough valid samples to be trusted.
    """
    num_pulses, num_rng_samples = data.shape

    if mask_valid_cpi is None:
        mask_valid_cpi = np.ones(data.shape, dtype=bool)
    else:
        mask_valid_cpi = mask_valid_cpi.astype(bool, copy=False)

    if mask_valid_cpi.shape != data.shape:
        raise ValueError(f"CPI mask shape {mask_valid_cpi.shape} != CPI data shape {data.shape}")

    min_valid_off_diag = max(1, int(np.ceil(off_diag_overlap_ratio * num_rng_samples)))
    min_valid_diag = max(1, int(np.ceil(diag_valid_ratio * num_rng_samples)))

    x_valid = data * mask_valid_cpi

    mask_int = mask_valid_cpi.astype(np.int32)
    overlap_counts = mask_int @ mask_int.T
    scm_sum = x_valid @ x_valid.conj().T

    scm = np.zeros((num_pulses, num_pulses), dtype=np.complex64)

    diag_idx = np.diag_indices(num_pulses)
    diag_counts = overlap_counts[diag_idx]
    diag_sum = scm_sum[diag_idx]

    diag_valid_idx = diag_counts >= min_valid_diag
    diag_vals = np.zeros(num_pulses, dtype=np.complex64)
    diag_vals[diag_valid_idx] = diag_sum[diag_valid_idx] / diag_counts[diag_valid_idx]
    scm[diag_idx] = diag_vals

    off_diag_valid = overlap_counts >= min_valid_off_diag
    np.fill_diagonal(off_diag_valid, False)
    scm[off_diag_valid] = scm_sum[off_diag_valid] / overlap_counts[off_diag_valid]

    scm = (0.5 * (scm + scm.conj().T)).astype(np.complex64)

    return scm, diag_valid_idx


def compute_scm_eigs_and_diag(cpi, cpi_mask, cpi_width, off_diag_overlap_ratio, diag_valid_ratio):
    """
    Compute the normalized SCM, its descending linear eigenvalues, the linear
    diagonal, and the per-index diagonal validity mask for one CPI tile.

    Returns
    -------
    eigvals : (M,) float32, descending, linear scale
    diag_lin : (M,) float64, linear scale, unnormalized power
    diag_valid_idx : (M,) bool
    """
    if cpi_mask is not None:
        scm, diag_valid_idx = compute_gap_exclusion_scm(
            cpi,
            mask_valid_cpi=cpi_mask,
            off_diag_overlap_ratio=off_diag_overlap_ratio,
            diag_valid_ratio=diag_valid_ratio,
        )
    else:
        M, K = cpi.shape
        scm = ((cpi @ cpi.conj().T)).astype(np.complex64)
        diag_valid_idx = np.ones(M, dtype=bool)

    # Normalize by number of range samples (CPI^H * CPI / K)
    scm = scm / cpi_width

    eigvals = np.linalg.eigvalsh(scm)
    eigvals = np.sort(eigvals)[::-1].astype(np.float32)

    diag_lin = np.real(np.diag(scm)).astype(np.float64)

    return eigvals, diag_lin, diag_valid_idx


# ---------------------------------------------------------------------------
# L0B READING
# ---------------------------------------------------------------------------

def read_raw_tile(raw: Raw, freq: str, pol: str, p0: int, cpi_len: int, r0: int, cpi_width: int):
    """Read and BFPQLUT-decode one cpi_len x cpi_width tile at (p0, r0)."""
    dataset = raw.getRawDataset(freq, pol)
    return np.asarray(dataset[p0:p0 + cpi_len, r0:r0 + cpi_width], dtype=np.complex64)


def get_subswath_mask(raw: Raw, freq: str, pol: str, p0: int, cpi_len: int, r0: int, cpi_width: int):
    """
    Boolean valid-sample mask from the ISCE3 subswath boundaries for one tile.
    """
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
    """Self-contained validity mask based on ADC fill level (per tile)."""
    mean_mag = np.mean(np.abs(cpi), axis=0)
    peak = np.max(mean_mag)
    if peak < EPS:
        return np.ones(cpi.shape, dtype=bool)
    threshold = gap_frac * peak
    valid_rng = mean_mag >= threshold
    return np.tile(valid_rng, (cpi.shape[0], 1))


# ---------------------------------------------------------------------------
# RFI INJECTION (same model as generate_rfi_data.py)
# ---------------------------------------------------------------------------

def tile_signal_power(cpi, cpi_mask):
    """Baseline power of a real CPI tile: mean(|x|^2) over its valid samples."""
    if cpi_mask is not None and cpi_mask.any():
        vals = cpi[cpi_mask]
    else:
        vals = cpi.ravel()
    return max(float(np.mean(np.abs(vals) ** 2)), EPS)


def channel_id(freq: str, pol: str) -> int:
    """Stable integer id for a (frequency, polarization) channel."""
    if freq not in FREQ_IDS:
        raise ValueError(f"Unknown frequency '{freq}'")
    if pol not in POL_IDS:
        raise ValueError(f"Unknown polarization '{pol}'")
    return FREQ_IDS[freq] * len(POL_IDS) + POL_IDS[pol]


def make_tile_seed_seq(seed: int, chan: int, pulse_tile: int, range_tile: int) -> np.random.SeedSequence:
    """
    Injection SeedSequence for one tile of one channel. Depends only on
    (seed, chan, pulse_tile, range_tile), so a tile's contamination is
    reproducible independently of every other tile.
    """
    return np.random.SeedSequence(
        [int(seed), SALT_INJECT, int(chan), int(pulse_tile), int(range_tile)]
    )


def draw_n_bands(tile_seed_seq, min_bands, max_bands):
    """Draw the band count (== the label) for one tile from a dedicated child stream."""
    count_seed = tile_seed_seq.spawn(1)[0]
    rng_count = np.random.default_rng(count_seed)
    return int(rng_count.integers(min_bands, max_bands + 1))


def inject_rfi_bands(cpi, cpi_mask, n_bands, jsr_min_db, jsr_max_db, tile_seed_seq):
    """
    Overlay n_bands synthetic Gaussian RFI bands onto a real CPI tile.

    Rows are drawn WITHOUT REPLACEMENT so knee (label) == number of elevated
    eigenvalues exactly. See generate_rfi_data.py's module docstring for the
    full derivation of this injection model.

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
    rng_jsr = np.random.default_rng(tile_seed_seq.spawn(1)[0])
    band_seeds = tile_seed_seq.spawn(n_bands)

    local_rows = np.sort(rng_struct.choice(M, size=n_bands, replace=False))

    rfi = np.zeros((M, K), dtype=np.complex64)
    bands: List[BandMeta] = []

    for b in range(n_bands):
        jsr_db = float(rng_jsr.uniform(jsr_min_db, jsr_max_db))
        band_power = signal_power * (10.0 ** (jsr_db / 10.0))
        sigma = np.sqrt(band_power / 2.0)

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
    """Append-as-you-go writer for the per-tile record arrays."""

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
        self.pair_id = mk('pair_id', (), np.int32, 4096)

        self.pair_id.attrs['description'] = (
            'index into the source clean_mountains file for this tile. In '
            '--paired-test mode the clean record and its contaminated '
            'counterpart share the same pair_id, so they can be matched up '
            'as the SAME background under the two conditions.'
        )
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
            'SCM diagonal, LINEAR scale, unnormalized power per pulse row'
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
               signal_power_db, valid_fraction, tile_pulse, tile_range, pair_id, cpi=None):
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
            (self.pair_id, pair_id),
        ):
            dset.resize(new_n, axis=0)
            dset[self.n:new_n] = arr

        if self.cpi is not None and cpi is not None:
            self.cpi.resize(new_n, axis=0)
            self.cpi[self.n:new_n] = cpi

        self.n = new_n


def write_root_attrs(f, args, freq, pol, source_group, n_tiles, cpi_len, cpi_width, use_mask):
    """File-level generation config: everything needed to reproduce this set."""
    f.attrs['clean_h5'] = os.path.basename(args.clean_h5)
    f.attrs['clean_h5_path'] = args.clean_h5
    f.attrs['clean_h5_group'] = source_group
    f.attrs['granule'] = os.path.basename(args.l0b_file)
    f.attrs['granule_path'] = args.l0b_file
    f.attrs['frequency'] = freq
    f.attrs['polarization'] = pol

    f.attrs['n_tiles'] = n_tiles
    f.attrs['cpi_len'] = cpi_len
    f.attrs['cpi_width'] = cpi_width

    f.attrs['min_bands'] = args.min_bands
    f.attrs['max_bands'] = args.max_bands
    f.attrs['n_classes'] = args.max_bands + 1
    f.attrs['jsr_min_db'] = args.jsr_min_db
    f.attrs['jsr_max_db'] = args.jsr_max_db
    f.attrs['jsr_draw'] = 'per band, uniform in [jsr_min_db, jsr_max_db]'
    f.attrs['jsr_reference'] = 'tile baseline power, mean(|x|^2) over valid samples'

    f.attrs['seed'] = args.seed
    f.attrs['channel_id'] = channel_id(freq, pol)
    f.attrs['salt_inject'] = SALT_INJECT
    f.attrs['seed_scheme'] = (
        'per-tile SeedSequence entropy = '
        '[seed, salt_inject, channel_id, pulse_tile, range_tile]; '
        'pulse_tile = tile_pulse // cpi_len, range_tile = tile_range // cpi_width; '
        'band count from a spawned child; each band coeff/Doppler from its own spawned child'
    )

    f.attrs['mask_mode'] = args.mask_mode
    f.attrs['gap_exclusion_used'] = bool(use_mask)
    f.attrs['off_diag_overlap_ratio'] = args.off_diag_overlap_ratio
    f.attrs['diag_valid_ratio'] = args.diag_valid_ratio

    f.attrs['eigenvalue_scale'] = 'linear, descending'
    f.attrs['diagonal_scale'] = 'linear, unnormalized'
    f.attrs['paired_test'] = bool(args.paired_test)
    if args.paired_test:
        f.attrs['paired_test_scheme'] = (
            'each source tile (pair_id) produced exactly 2 records: one '
            'untouched (label 0) and one with RFI forced (label >= 1, '
            'band count drawn from [max(min_bands,1), max_bands]), both from '
            'the identical raw CPI. Match rows by pair_id to compare behavior '
            'on the same background clean vs contaminated.'
        )
    f.attrs['cpi_stored'] = bool(args.save_cpi)
    f.attrs['generated_utc'] = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# SET GENERATION
# ---------------------------------------------------------------------------

def _build_record(tile, meta, cpi_mask, cpi_width, args, pair_id, p0, r0):
    """Compute features for one (tile, meta) pair and return the append() kwargs."""
    eigvals, diag_lin, diag_valid_idx = compute_scm_eigs_and_diag(
        tile, cpi_mask, cpi_width,
        args.off_diag_overlap_ratio, args.diag_valid_ratio,
    )

    jsr_row = np.full(args.max_bands, np.nan, dtype=np.float32)
    rows_row = np.full(args.max_bands, -1, dtype=np.int8)
    for bi, band in enumerate(meta.bands):
        jsr_row[bi] = band.jsr_db
        rows_row[bi] = band.local_row

    return dict(
        labels=np.array([meta.knee], dtype=np.int8),
        jsr_db=jsr_row[None, :],
        band_rows=rows_row[None, :],
        eigenvalues=eigvals[None, :],
        diagonal=diag_lin.astype(np.float32)[None, :],
        diag_valid_idx=diag_valid_idx[None, :],
        signal_power_db=np.array([meta.signal_power_db], dtype=np.float32),
        valid_fraction=np.array([meta.valid_fraction], dtype=np.float32),
        tile_pulse=np.array([p0], dtype=np.int32),
        tile_range=np.array([r0], dtype=np.int32),
        pair_id=np.array([pair_id], dtype=np.int32),
        cpi=tile[None, ...] if args.save_cpi else None,
    )


def generate_dataset_for_group(raw, freq, pol, grp_name, pulse_idx, range_idx,
                                cpi_len, cpi_width, args, out_dir):
    """
    Inject RFI onto every tile location listed for one freq_X_pol_Y group of
    the clean-mountain source file, and write the labeled records to HDF5.

    Default mode: one record per tile, band count drawn from
    [min_bands, max_bands] (0 = clean, interspersed at random) -- suitable
    for training.

    --paired-test mode: two records per tile, both from the IDENTICAL raw
    CPI: one left untouched (label 0) and one with RFI forced onto it (band
    count drawn from [max(min_bands, 1), max_bands], so it is never 0). Both
    records share the same pair_id, so a downstream consumer can evaluate the
    model (and any diagonal-based metric) on the very same background clean
    vs. contaminated, rather than on two different tiles that happen to have
    drawn different band counts.

    Returns
    -------
    knee_counts : dict[int, int]
    out_path : str
    """
    chan = channel_id(freq, pol)
    n_tiles = len(pulse_idx)
    paired = bool(args.paired_test)

    suffix = '_paired' if paired else ''
    out_path = os.path.join(out_dir, f"mountain_rfi_data{suffix}_{freq}_{pol}.h5")
    print(f"\n[{freq}-{pol}] source group: {grp_name}  ({n_tiles} clean tiles)"
          + ("  [paired-test mode]" if paired else ""))
    print(f"  -> {out_path}")

    use_mask = args.mask_mode != 'none'
    knee_counts = {}

    with h5py.File(out_path, 'w') as f:
        write_root_attrs(f, args, freq, pol, grp_name, n_tiles, cpi_len, cpi_width, use_mask)
        writer = TileWriter(f, cpi_len, cpi_width, args.max_bands, args.save_cpi)

        report_every = max(1, n_tiles // 10)

        for i in range(n_tiles):
            p0 = int(pulse_idx[i])
            r0 = int(range_idx[i])

            cpi = read_raw_tile(raw, freq, pol, p0, cpi_len, r0, cpi_width)

            if args.mask_mode == 'subswath':
                cpi_mask = get_subswath_mask(raw, freq, pol, p0, cpi_len, r0, cpi_width)
            elif args.mask_mode == 'amplitude':
                cpi_mask = amplitude_gap_mask(cpi)
            else:
                cpi_mask = None

            pulse_tile = p0 // cpi_len
            range_tile = r0 // cpi_width
            tile_ss = make_tile_seed_seq(args.seed, chan, pulse_tile, range_tile)

            # In paired-test mode the contaminated copy must never draw 0
            # bands, or it would be indistinguishable from its own clean
            # counterpart. The draw still comes from tile_ss, so it stays
            # reproducible per tile.
            contam_min_bands = max(args.min_bands, 1) if paired else args.min_bands
            n_bands = draw_n_bands(tile_ss, contam_min_bands, args.max_bands)

            tiles_metas = []

            if paired:
                # Untouched copy of the identical raw tile. n_bands=0 returns
                # immediately inside inject_rfi_bands without touching
                # tile_ss's random stream, so calling it here does not
                # perturb the contaminated draw below.
                clean_tile, clean_meta = inject_rfi_bands(
                    cpi, cpi_mask, 0, args.jsr_min_db, args.jsr_max_db, tile_ss
                )
                tiles_metas.append((clean_tile, clean_meta))

            contam_tile, contam_meta = inject_rfi_bands(
                cpi, cpi_mask, n_bands, args.jsr_min_db, args.jsr_max_db, tile_ss
            )
            tiles_metas.append((contam_tile, contam_meta))

            for tile, meta in tiles_metas:
                kwargs = _build_record(tile, meta, cpi_mask, cpi_width, args,
                                       pair_id=i, p0=p0, r0=r0)
                writer.append(**kwargs)
                knee_counts[meta.knee] = knee_counts.get(meta.knee, 0) + 1

            if (i + 1) % report_every == 0 or (i + 1) == n_tiles:
                print(f"    {i + 1}/{n_tiles} tiles processed ({writer.n} records written)")

        hist = {str(k): int(v) for k, v in sorted(knee_counts.items())}
        f.attrs['label_histogram'] = json.dumps(hist)
        f.attrs['n_records'] = writer.n

    print("  label histogram: "
          + ", ".join(f"knee={k}: {knee_counts[k]}" for k in sorted(knee_counts)))

    return knee_counts, out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('clean_h5', help='clean_mountains_filtered.h5 (or clean_mountains.h5)')
    parser.add_argument('l0b_file', help='Source NISAR L0B HDF5 granule the clean tiles were drawn from')

    parser.add_argument('--freq', default=None,
                        help='Restrict to one frequency group (e.g. "A"). Default: all groups present.')
    parser.add_argument('--pol', default=None,
                        help='Restrict to one polarization group (e.g. "HH"). Default: all groups present.')

    parser.add_argument('--cpi-width', type=int, default=None,
                        help='CPI width used when the clean tiles were selected. '
                             'Default: CPI_WIDTH_DEFAULT (250); this is NOT stored in '
                             'clean_mountains.h5, so pass it explicitly if it differs.')

    parser.add_argument('--min-bands', type=int, default=MIN_BANDS_DEFAULT)
    parser.add_argument('--max-bands', type=int, default=MAX_BANDS_DEFAULT)
    parser.add_argument('--jsr-min-db', type=float, default=JSR_MIN_DB_DEFAULT)
    parser.add_argument('--jsr-max-db', type=float, default=JSR_MAX_DB_DEFAULT)

    parser.add_argument('--mask-mode', choices=['none', 'subswath', 'amplitude'], default='subswath',
                        help='Validity mask used for gap-exclusion SCM recompute and for the '
                             'injection JSR reference. "subswath" uses ISCE3 subswath boundaries '
                             '(matches generate_rfi_data.py); "amplitude" uses the ADC fill-level '
                             'heuristic (matches select_clean_mountain.py --compute-subswath-mask).')
    parser.add_argument('--off-diag-overlap-ratio', type=float, default=OFF_DIAG_OVERLAP_RATIO_DEFAULT)
    parser.add_argument('--diag-valid-ratio', type=float, default=DIAG_VALID_RATIO_DEFAULT)

    parser.add_argument('--output-dir', default='data/mountain_rfi_train')
    parser.add_argument('--save-cpi', action='store_true',
                        help='Also store the raw complex CPI tiles (RFI already overlaid).')

    parser.add_argument('--paired-test', action='store_true',
                        help='Generate a PAIRED test set instead of a training set: for '
                             'every clean tile, write both an untouched copy (label 0) and '
                             'a forced-contaminated copy (label >= 1) of the IDENTICAL raw '
                             'CPI, sharing a pair_id. Use this output with train_db.py '
                             '--paired-test-dir to compare model (and diagonal-metric) '
                             'behavior on the same background clean vs. contaminated.')

    parser.add_argument('--seed', type=int, default=SEED_DEFAULT,
                        help='Master seed for all RFI injection streams.')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.min_bands < 0 or args.max_bands < args.min_bands:
        raise ValueError('Require 0 <= min_bands <= max_bands')
    if args.jsr_max_db < args.jsr_min_db:
        raise ValueError('Require jsr_min_db <= jsr_max_db')
    if args.paired_test and args.max_bands < 1:
        raise ValueError(
            'max_bands must be >= 1 in --paired-test mode: the contaminated '
            'copy of each tile is forced to draw at least 1 band'
        )

    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 70)
    if args.paired_test:
        print('Mountain-tile PAIRED test set generation')
        print('(each clean tile -> one untouched record + one forced-RFI record, same pair_id)')
    else:
        print('Mountain-tile RFI training set generation')
        print('(RFI injected onto the exact tiles select_clean_mountain.py found clean)')
    print('=' * 70)
    print(f'  clean tiles from : {args.clean_h5}')
    print(f'  source granule   : {args.l0b_file}')
    if args.paired_test:
        print(f'  bands per tile   : untouched (0) paired with '
              f'{max(args.min_bands, 1)}..{args.max_bands} (forced RFI)')
    else:
        print(f'  bands per tile   : {args.min_bands}..{args.max_bands} '
              f'(label = drawn count; 0 = clean, interspersed at random)')
    print(f'  JSR              : [{args.jsr_min_db}, {args.jsr_max_db}] dB above tile baseline power')
    print(f'  mask mode        : {args.mask_mode}')
    print(f'  save CPI         : {args.save_cpi}')
    print(f'  seed             : {args.seed}')

    raw = Raw(hdf5file=args.l0b_file)
    raw.parsePolarizations()

    h5_in = h5py.File(args.clean_h5, 'r')

    written = []
    for grp_name in h5_in.keys():
        grp = h5_in[grp_name]
        freq = str(grp.attrs['frequency'])
        pol = str(grp.attrs['polarization'])

        if args.freq is not None and freq != args.freq:
            continue
        if args.pol is not None and pol != args.pol:
            continue

        pulse_idx = grp['pulse_idx'][:]
        range_idx = grp['range_idx'][:]
        eig_shape = grp['eigenvalues'].shape
        cpi_len = eig_shape[1] if len(eig_shape) == 2 else CPI_LEN_DEFAULT
        cpi_width = args.cpi_width if args.cpi_width is not None else CPI_WIDTH_DEFAULT

        if len(pulse_idx) == 0:
            print(f"\n[warn] group {grp_name} has no clean tiles; skipping")
            continue

        _, out_path = generate_dataset_for_group(
            raw, freq, pol, grp_name, pulse_idx, range_idx,
            cpi_len, cpi_width, args, args.output_dir
        )
        written.append(out_path)

    h5_in.close()

    if not written:
        raise RuntimeError('No matching freq/pol groups were processed; check --freq/--pol filters')

    print('\nDone. Wrote:')
    for path in written:
        print(f'  {path}')


if __name__ == '__main__':
    main()