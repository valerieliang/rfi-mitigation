#!/usr/bin/env python
"""
plot_raw_tiles.py

Extract raw complex L0B tiles in the TIME DOMAIN (pulse x range sample),
store the matrices, and render them as images.

This is the Step 1 visual check that comes BEFORE any segmentation data
generation: the question it answers is whether RFI is separable from terrain
by eye in raw fast time, at the tile sizes and JSRs that matter. If it is
not, that is worth knowing before building a generator and a training loop.

What this script deliberately does NOT do
-----------------------------------------
No sample covariance matrix, no eigenvalue decomposition, no Fourier
transform. The stored and plotted arrays are the raw complex data and
quantities derived pointwise from it. The SCM path already exists in the
detection scripts; the point here is to look at what the segmentation model
would actually see.

Panels
------
Four panels are rendered for each tile, corresponding to the UNet input channels:

  magnitude   Absolute log magnitude in dB (power). Wideband RFI raises
              the speckle level. This matches UNet input channel 0 (after
              scaling).

  phase_cos   Cosine of the adjacent-pulse phase difference. This matches
              UNet input channel 1 exactly. Fixed-Doppler emitters appear
              as constant-valued regions.

  phase_sin   Sine of the adjacent-pulse phase difference. This matches
              UNet input channel 2 exactly. Used with phase_cos to encode
              phase coherence without discontinuities at ±π.

  valid       ADC gap / subswath validity mask. This matches UNet input
              channel 3 exactly.

Tile selection
--------------
Three ways to choose locations, in decreasing order of usefulness:

  --from-clean PATH   Use the tile locations recorded by a select_clean.py
                      output, and find runs of consecutive clean CPI blocks
                      long enough for the requested pulse extent. This also
                      REPORTS how many such runs exist, which is the Step 2
                      question about whether wide tiles are feasible at all.

  --at P0,R0          Explicit locations, repeatable. Use this for the known
                      contaminated scenes, where the interesting locations
                      are picked by hand off the browse image.

  --grid              Dense strided grid over the granule.

Usage
-----
    # Process specific pulse and range windows (defaults to freq A, all pols)
    # Large windows are automatically tiled into 256x256 blocks
    python plot_raw_tiles.py granule.h5 \\
        --pulse-start 700435 --pulse-end 715206 \\
        --range-start 0 --range-end 512 \\
        --output-dir figs/la_blob

    # Custom tile size for auto-tiling
    python plot_raw_tiles.py granule.h5 \\
        --pulse-start 700435 --pulse-end 715206 \\
        --tile-pulses 512 --tile-range 512 \\
        --output-dir figs/la_blob_512

    # Process entire granule with specific frequency and polarization
    python plot_raw_tiles.py granule.h5 --freq A --pol HH \\
        --output-dir figs/full_granule

    # Process all polarizations (default behavior when --pol is omitted)
    python plot_raw_tiles.py granule.h5 \\
        --pulse-start 700435 --pulse-end 715206 \\
        --output-dir figs/all_pols

    # Legacy: Look at known contaminated locations in a scene
    python plot_raw_tiles.py granule.h5 --freq A --pol HV \\
        --at 12000,3000 --at 12256,3000 \\
        --pulses 256 --range-width 512 \\
        --output-dir figs/vienna

    # Clean-run census plus a contact sheet, no per-tile figures
    python plot_raw_tiles.py granule.h5 --freq A --pol HH \\
        --from-clean clean_tiles/ --pulses 256 --range-width 512 \\
        --max-tiles 36 --no-per-tile --output-dir figs/clean_survey

    # Render whatever is already stored, without touching the granule
    python plot_raw_tiles.py --replot figs/vienna/raw_tiles_A_HV.h5 \\
        --output-dir figs/vienna_replot

    # Self test on synthetic data, no L0B needed
    python plot_raw_tiles.py --demo --output-dir figs/demo

Outputs
-------
    <output-dir>/raw_tiles_<freq>_<pol>.h5    complex tiles + validity + meta
    <output-dir>/tile_<idx>_p<P0>_r<R0>.png   per-tile panel figure
    <output-dir>/contact_sheet.png            magnitude thumbnails, all tiles
"""

import os
import glob
import argparse
import warnings
from datetime import datetime, timezone

import numpy as np


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

import h5py                                    # noqa: E402
import matplotlib                              # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                # noqa: E402

# The L0B readers are only needed when actually reading a granule. Importing
# them lazily keeps --replot and --demo usable in an environment without
# isce3 installed.
Raw = None
ToneRemover = None
caltone_frequency_from_raw = None


def _import_l0b_readers():
    """Import the isce3 / nisar readers on demand."""
    global Raw, ToneRemover, caltone_frequency_from_raw
    if Raw is not None:
        return
    from nisar.products.readers.Raw import Raw as _Raw
    from isce3.focus import ToneRemover as _ToneRemover
    try:
        from nisar.products.readers.Raw import caltone_frequency_from_raw as _ctf
    except ImportError:
        _ctf = None
    Raw, ToneRemover, caltone_frequency_from_raw = _Raw, _ToneRemover, _ctf


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

CPI_LEN_DEFAULT = 16          # pulses per CPI block, for clean-run bookkeeping
CPI_WIDTH_DEFAULT = 250       # range samples per clean-tile record

PULSES_DEFAULT = 256
RANGE_WIDTH_DEFAULT = 512

# Default tile size for automatic tiling of large windows
DEFAULT_TILE_PULSES = 256
DEFAULT_TILE_RANGE = 256

# Caltone removal, matching the detection and generation paths.
CALTONE_WINDOW_SIZE = 64
CALTONE_DEFAULT_FREQ_HZ = 1214.883e6
CALTONE_LO_HZ = 1200e6
CALTONE_CLOCK_HZ = 240e6

# Amplitude gap mask threshold, as a fraction of the peak mean magnitude.
GAP_FRAC_DEFAULT = 0.10

EPS = 1e-12


# ---------------------------------------------------------------------------
# CALTONE
# ---------------------------------------------------------------------------

def parse_caltone_freq_from_drt(raw, txrx_pol):
    """
    Fallback for isce3's caltone_frequency_from_raw, reading the DRT CALTONE
    phase-step telemetry directly. Only used when the official helper is not
    importable.
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


def build_tone_remover(raw, freq, pol, num_rng_samples):
    """ToneRemover sized to the full range width, plus the frequency used."""
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
# TILE READING
# ---------------------------------------------------------------------------

def read_raw_tile(raw, freq, pol, p0, n_pulses, r0, range_width, remover=None):
    """
    Read and BFPQLUT-decode one n_pulses x range_width tile at (p0, r0).

    When a remover is given, the caltone is subtracted from each FULL-WIDTH
    range line before the tile is sliced out. ToneRemover builds an absolute
    phase reference anchored at range sample 0, so a pre-sliced sub-tile
    would carry the wrong tone phase.
    """
    dataset = raw.getRawDataset(freq, pol)

    if remover is None:
        return np.asarray(dataset[p0:p0 + n_pulses, r0:r0 + range_width],
                          dtype=np.complex64)

    lines = np.asarray(dataset[p0:p0 + n_pulses, :], dtype=np.complex64)
    for ip in range(lines.shape[0]):
        lines[ip] = remover.remove_tone(lines[ip])
    return np.ascontiguousarray(lines[:, r0:r0 + range_width])


def get_subswath_mask(raw, freq, pol, p0, n_pulses, r0, range_width):
    """Boolean valid-sample mask from the ISCE3 subswath boundaries."""
    tx_pol = pol[0]
    pulse_indices = np.arange(p0, p0 + n_pulses)
    subswaths = raw.getSubSwaths(freq, tx_pol)
    swaths = subswaths[:, pulse_indices, :]

    mask = np.zeros((n_pulses, range_width), dtype=bool)
    if swaths is not None:
        for i in range(n_pulses):
            for start, end in swaths[:, i, :]:
                s = max(int(start) - r0, 0)
                e = min(int(end) - r0, range_width)
                if e > s:
                    mask[i, s:e] = True
    return mask


def amplitude_gap_mask(tile, gap_frac=GAP_FRAC_DEFAULT):
    """Self-contained validity mask based on ADC fill level, per tile."""
    mean_mag = np.mean(np.abs(tile), axis=0)
    peak = np.max(mean_mag)
    if peak < EPS:
        return np.ones(tile.shape, dtype=bool)
    valid_rng = mean_mag >= gap_frac * peak
    return np.tile(valid_rng, (tile.shape[0], 1))


# ---------------------------------------------------------------------------
# DERIVED DISPLAY QUANTITIES (all pointwise, no SCM, no FFT)
# ---------------------------------------------------------------------------

def absolute_log_magnitude(tile):
    """Absolute log magnitude in dB, no floor subtraction."""
    return (20.0 * np.log10(np.abs(tile) + EPS)).astype(np.float32)




# ---------------------------------------------------------------------------
# TILE LOCATION SELECTION
# ---------------------------------------------------------------------------

def parse_at(values):
    """Parse repeated --at P0,R0 arguments into a list of (p0, r0)."""
    out = []
    for v in values or []:
        parts = v.replace(' ', '').split(',')
        if len(parts) != 2:
            raise ValueError(f"--at expects 'P0,R0', got '{v}'")
        out.append((int(parts[0]), int(parts[1])))
    return out


def grid_locations(n_pulses_total, n_range_total, n_pulses, range_width,
                   stride_pulse, stride_range, max_tiles):
    """Dense strided grid of tile origins over the granule."""
    p_starts = range(0, max(n_pulses_total - n_pulses, 0) + 1, stride_pulse)
    r_starts = range(0, max(n_range_total - range_width, 0) + 1, stride_range)
    out = []
    for p0 in p_starts:
        for r0 in r_starts:
            out.append((int(p0), int(r0)))
            if len(out) >= max_tiles:
                return out
    return out


def _read_flat_clean_file(path):
    """(freq, pol, tile_pulse, tile_range) from one select_clean.py output."""
    with h5py.File(path, 'r') as f:
        freq = str(f.attrs['frequency'])
        pol = str(f.attrs['polarization'])
        tile_pulse = f['tile_pulse'][:]
        tile_range = f['tile_range'][:]
    return freq, pol, tile_pulse, tile_range


def load_clean_locations(clean_path, freq, pol):
    """
    Gather clean tile origins for one channel from a select_clean.py output,
    accepting either a directory of flat per-channel files or a single file.
    """
    if os.path.isdir(clean_path):
        paths = sorted(glob.glob(os.path.join(clean_path, '*.h5')))
        if not paths:
            raise FileNotFoundError(f"No *.h5 files found in {clean_path}")
    else:
        paths = [clean_path]

    tp_all, tr_all = [], []
    for p in paths:
        try:
            f_freq, f_pol, tp, tr = _read_flat_clean_file(p)
        except (OSError, KeyError) as exc:
            print(f"[warn] skipping '{p}': not a readable flat clean file "
                  f"({type(exc).__name__}: {exc})")
            continue
        if freq is not None and f_freq != freq:
            continue
        if pol is not None and f_pol != pol:
            continue
        tp_all.append(tp)
        tr_all.append(tr)

    if not tp_all:
        raise RuntimeError(
            f"No clean tiles for freq={freq} pol={pol} in {clean_path}")

    return np.concatenate(tp_all), np.concatenate(tr_all)


def find_clean_runs(tile_pulse, tile_range, n_pulses, cpi_len,
                    min_clean_frac=1.0):
    """
    Find tile origins where enough consecutive CPI blocks are clean.

    This doubles as the Step 2 census: wide tiles are only usable if long
    unbroken clean runs actually exist, and the clean tiles produced by
    select_clean.py are a scattered subset rather than a dense grid.

    Parameters
    ----------
    tile_pulse, tile_range : absolute origins of the clean CPI blocks.
    n_pulses : requested pulse extent of the output tile.
    cpi_len : pulses per clean-tile record.
    min_clean_frac : fraction of the constituent blocks that must be present
        in the clean set. 1.0 requires an unbroken run.

    Returns
    -------
    list of (p0, r0)
    """
    n_blocks = n_pulses // cpi_len
    if n_blocks < 1:
        raise ValueError(f"n_pulses ({n_pulses}) < cpi_len ({cpi_len})")

    present = set()
    for tp, tr in zip(tile_pulse, tile_range):
        present.add((int(tp) // cpi_len, int(tr)))

    need = max(1, int(np.ceil(min_clean_frac * n_blocks)))

    by_range = {}
    for blk, r0 in present:
        by_range.setdefault(r0, set()).add(blk)

    out = []
    for r0, blocks in sorted(by_range.items()):
        if not blocks:
            continue
        lo, hi = min(blocks), max(blocks)
        blk = lo
        while blk + n_blocks - 1 <= hi:
            count = sum(1 for b in range(blk, blk + n_blocks) if b in blocks)
            if count >= need:
                out.append((blk * cpi_len, int(r0)))
                blk += n_blocks          # non-overlapping runs
            else:
                blk += 1
    return out


# ---------------------------------------------------------------------------
# STORAGE
# ---------------------------------------------------------------------------

def write_tiles(path, tiles, valids, locations, meta):
    """
    Store the raw complex tiles and their validity masks.

    The complex data is stored as-is so that any later analysis (including
    analysis this script deliberately omits) can be run without re-reading
    the granule, which is the slow part.
    """
    tiles = np.asarray(tiles, dtype=np.complex64)
    valids = np.asarray(valids, dtype=bool)
    locations = np.asarray(locations, dtype=np.int32)

    n, p, k = tiles.shape

    # Calculate safe chunk size to avoid HDF5 4GB limit
    # complex64 = 8 bytes, bool = 1 byte
    # For tiles: 1 * p * k * 8 bytes must be < 4GB
    # For valid: 1 * p * k * 1 byte must be < 4GB
    max_chunk_elements = (4 * 1024**3) // 8  # 4GB / 8 bytes

    if p * k > max_chunk_elements:
        # Need to chunk in smaller pieces
        chunk_p = min(p, 256)
        chunk_k = min(k, min(max_chunk_elements // chunk_p, 4096))
    else:
        chunk_p = p
        chunk_k = k

    with h5py.File(path, 'w') as f:
        f.create_dataset('tiles', data=tiles, chunks=(1, chunk_p, chunk_k),
                         compression='gzip', compression_opts=1)
        f.create_dataset('valid', data=valids, chunks=(1, chunk_p, chunk_k),
                         compression='gzip', compression_opts=1)
        f.create_dataset('tile_pulse', data=locations[:, 0])
        f.create_dataset('tile_range', data=locations[:, 1])
        for key, val in meta.items():
            if val is not None:
                f.attrs[key] = val
        f.attrs['n_tiles'] = n
        f.attrs['n_pulses'] = p
        f.attrs['range_width'] = k
        f.attrs['domain'] = 'time'
        f.attrs['contains_scm'] = False
        f.attrs['created'] = datetime.now(timezone.utc).isoformat()
    return path


def read_tiles(path):
    """Read back a file written by write_tiles."""
    with h5py.File(path, 'r') as f:
        tiles = f['tiles'][:]
        valids = f['valid'][:]
        locations = np.stack([f['tile_pulse'][:], f['tile_range'][:]], axis=1)
        meta = {k: f.attrs[k] for k in f.attrs}
    return tiles, valids, locations, meta


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def adjacent_pulse_phase_cos_sin(tile):
    """
    cos and sin of arg(x[m] * conj(x[m-1])) at every range sample.

    This matches the UNet input channels 1 and 2 exactly.
    """
    if tile.shape[0] < 2:
        zeros = np.zeros(tile.shape, dtype=np.float32)
        return zeros, zeros.copy()

    prod = tile[1:] * np.conj(tile[:-1])
    phase = np.angle(prod)

    cos_d = np.empty(tile.shape, dtype=np.float32)
    sin_d = np.empty(tile.shape, dtype=np.float32)

    cos_d[1:] = np.cos(phase)
    sin_d[1:] = np.sin(phase)
    cos_d[0] = cos_d[1]
    sin_d[0] = sin_d[1]

    return cos_d, sin_d


def plot_tile(tile, valid, p0, r0, out_path, title_extra=''):
    """Render one tile as a row of 4 panels (UNet input channels) and save it."""
    mag_db = absolute_log_magnitude(tile)
    cos_d, sin_d = adjacent_pulse_phase_cos_sin(tile)

    fig_w = 4.2 * 4 + 1.0
    fig, axes = plt.subplots(1, 4, figsize=(fig_w, 4.8), squeeze=False)
    axes = axes[0]

    # Panel 0: magnitude (power)
    im0 = axes[0].imshow(mag_db, aspect='auto', origin='upper', cmap='gray',
                         interpolation='nearest')
    axes[0].set_title('magnitude')
    axes[0].set_xlabel('range sample')
    axes[0].set_ylabel('pulse')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label='dB absolute')

    # Panel 1: phase_cos
    im1 = axes[1].imshow(cos_d, aspect='auto', origin='upper', cmap='gray',
                         vmin=-1.0, vmax=1.0, interpolation='nearest')
    axes[1].set_title('phase_cos')
    axes[1].set_xlabel('range sample')
    axes[1].set_ylabel('pulse')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='cos(phase diff)')

    # Panel 2: phase_sin
    im2 = axes[2].imshow(sin_d, aspect='auto', origin='upper', cmap='gray',
                         vmin=-1.0, vmax=1.0, interpolation='nearest')
    axes[2].set_title('phase_sin')
    axes[2].set_xlabel('range sample')
    axes[2].set_ylabel('pulse')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label='sin(phase diff)')

    # Panel 3: valid
    im3 = axes[3].imshow(valid.astype(np.float32), aspect='auto', origin='upper',
                         cmap='gray', vmin=0.0, vmax=1.0, interpolation='nearest')
    axes[3].set_title('valid')
    axes[3].set_xlabel('range sample')
    axes[3].set_ylabel('pulse')
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04, label='valid')

    valid_frac = float(valid.mean())
    title = (f'pulse {p0}, range {r0}   |   {tile.shape[0]} x {tile.shape[1]}   |   '
             f'valid {100 * valid_frac:.0f}%{title_extra}')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_contact_sheet(tiles, valids, locations, out_path, n_cols=6, max_tiles=60):
    """Grid of magnitude thumbnails for fast scanning across many tiles."""
    n = min(len(tiles), max_tiles)
    if n == 0:
        return None
    n_cols = max(1, min(n_cols, n))
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.4 * n_cols, 2.6 * n_rows),
                             squeeze=False)

    for idx in range(n_rows * n_cols):
        ax = axes[idx // n_cols][idx % n_cols]
        ax.set_xticks([])
        ax.set_yticks([])
        if idx >= n:
            ax.axis('off')
            continue
        mag_db = absolute_log_magnitude(tiles[idx])
        ax.imshow(mag_db, aspect='auto', origin='upper', cmap='gray',
                  interpolation='nearest')
        ax.set_title(f'{idx}: p{locations[idx][0]} r{locations[idx][1]}',
                     fontsize=7)

    fig.suptitle('absolute magnitude, dB', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def render_all(tiles, valids, locations, out_dir, args, tag=''):
    """Per-tile figures plus the contact sheet."""
    written = []

    if not args.no_per_tile:
        for idx, (tile, valid, (p0, r0)) in enumerate(
                zip(tiles, valids, locations)):
            name = f'tile_{idx:03d}_p{int(p0)}_r{int(r0)}.png'
            written.append(plot_tile(
                tile, valid, int(p0), int(r0),
                os.path.join(out_dir, name),
                title_extra=tag))

    sheet = plot_contact_sheet(
        tiles, valids, locations,
        os.path.join(out_dir, 'contact_sheet.png'),
        n_cols=args.sheet_cols,
        max_tiles=args.max_tiles)
    if sheet:
        written.append(sheet)
    return written


# ---------------------------------------------------------------------------
# MODES
# ---------------------------------------------------------------------------

def generate_tile_grid(pulse_start, pulse_end, range_start, range_end,
                       tile_pulses=DEFAULT_TILE_PULSES, tile_range=DEFAULT_TILE_RANGE):
    """
    Divide a large pulse/range window into smaller tiles for visualization.

    Returns list of (p0, r0, n_pulses, range_width) tuples.
    """
    tiles = []
    p0 = pulse_start
    while p0 < pulse_end:
        p_len = min(tile_pulses, pulse_end - p0)
        r0 = range_start
        while r0 < range_end:
            r_len = min(tile_range, range_end - r0)
            tiles.append((p0, r0, p_len, r_len))
            r0 += tile_range
        p0 += tile_pulses
    return tiles


def process_single_channel(raw, freq, pol, args, pulse_start, pulse_end, range_start, range_end,
                          n_pulses, range_width, locations):
    """Process a single frequency/polarization channel."""
    dataset = raw.getRawDataset(freq, pol)
    n_pulses_total, n_range_total = dataset.shape

    print(f'\n  Processing {freq} {pol}')
    print(f'  granule shape    : {n_pulses_total} pulses x {n_range_total} range samples')

    # Validate bounds for this channel
    actual_pulse_end = min(pulse_end, n_pulses_total)
    actual_range_end = min(range_end, n_range_total)
    actual_n_pulses = min(n_pulses, n_pulses_total - pulse_start)
    actual_range_width = min(range_width, n_range_total - range_start)

    print(f'  processing window: pulses [{pulse_start}, {actual_pulse_end}) x range [{range_start}, {actual_range_end})')
    print(f'  window size      : {actual_n_pulses} pulses x {actual_range_width} range samples')

    if args.remove_caltone:
        remover, caltone_freq = build_tone_remover(raw, freq, pol, n_range_total)
        print(f'  caltone removal  : ON (f = {caltone_freq / 1e6:.4f} MHz)')
    else:
        remover, caltone_freq = None, None
        print('  caltone removal  : OFF')

    tiles, valids, kept = [], [], []
    for p0, r0 in locations:
        # Use explicit tile sizes if given, otherwise use auto-tile sizes
        tile_n_pulses = args.pulses if args.pulses else args.tile_pulses
        tile_range_width = args.range_width if args.range_width else args.tile_range

        # Clip to actual bounds
        tile_n_pulses = min(tile_n_pulses, n_pulses_total - p0)
        tile_range_width = min(tile_range_width, n_range_total - r0)

        if tile_n_pulses <= 0 or tile_range_width <= 0:
            print(f'  [skip] ({p0}, {r0}) extends past the granule bounds')
            continue

        tile = read_raw_tile(raw, freq, pol, p0, tile_n_pulses, r0,
                             tile_range_width, remover)

        if args.mask_mode == 'subswath':
            valid = get_subswath_mask(raw, freq, pol, p0, tile_n_pulses, r0,
                                      tile_range_width)
        elif args.mask_mode == 'amplitude':
            valid = amplitude_gap_mask(tile, args.gap_frac)
        else:
            valid = np.ones(tile.shape, dtype=bool)

        tiles.append(tile)
        valids.append(valid)
        kept.append((p0, r0))

    if not tiles:
        print(f'  [skip] No tiles could be read for {freq} {pol}')
        return []

    # Create polarization-specific subdirectory
    pol_output_dir = os.path.join(args.output_dir, pol)
    os.makedirs(pol_output_dir, exist_ok=True)

    out_h5 = os.path.join(args.output_dir, f'raw_tiles_{freq}_{pol}.h5')
    meta = dict(
        l0b_file=os.path.basename(args.l0b_file),
        frequency=freq,
        polarization=pol,
        cpi_len=args.cpi_len,
        mask_mode=args.mask_mode,
        caltone_removed=bool(args.remove_caltone),
        caltone_freq_hz=float(caltone_freq) if caltone_freq else None,
        scene_tag=args.scene_tag or '',
    )
    write_tiles(out_h5, tiles, valids, kept, meta)
    print(f'  wrote            : {out_h5}')

    tag = f'   |   {args.scene_tag}' if args.scene_tag else ''
    figs = render_all(np.asarray(tiles), np.asarray(valids),
                      np.asarray(kept), pol_output_dir, args, tag)
    return [out_h5] + figs


def run_from_granule(args):
    """Read tiles from an L0B granule, store them, and render them."""
    _import_l0b_readers()

    raw = Raw(hdf5file=args.l0b_file)
    raw.parsePolarizations()

    freq = args.freq

    # Determine which polarizations to process
    if args.pol:
        pols = [args.pol]
    else:
        # Get all available polarizations for this frequency
        available_pols = []
        for pol_candidate in ['HH', 'HV', 'VH', 'VV']:
            try:
                raw.getRawDataset(freq, pol_candidate)
                available_pols.append(pol_candidate)
            except (KeyError, RuntimeError):
                pass

        if not available_pols:
            raise ValueError(f'No polarizations found for frequency {freq}')

        pols = available_pols
        print(f'  polarizations    : {", ".join(pols)} (all available)')

    # Get initial dataset to determine bounds
    dataset = raw.getRawDataset(freq, pols[0])
    n_pulses_total, n_range_total = dataset.shape

    # Handle new pulse-start/pulse-end and range-start/range-end arguments
    pulse_start = args.pulse_start if args.pulse_start is not None else 0
    pulse_end = args.pulse_end if args.pulse_end is not None else n_pulses_total
    range_start = args.range_start if args.range_start is not None else 0
    range_end = args.range_end if args.range_end is not None else n_range_total

    if args.pulse_start is None and args.pulse_end is None and args.range_start is None and args.range_end is None:
        if not (args.at or args.from_clean or args.grid):
            print('  WARNING: No pulse/range bounds specified. Processing ENTIRE granule.')
            print(f'           This will process {n_pulses_total} x {n_range_total} samples.')

    if pulse_start < 0 or pulse_end > n_pulses_total or pulse_start >= pulse_end:
        raise ValueError(f'Invalid pulse range: [{pulse_start}, {pulse_end}) for granule with {n_pulses_total} pulses')
    if range_start < 0 or range_end > n_range_total or range_start >= range_end:
        raise ValueError(f'Invalid range: [{range_start}, {range_end}) for granule with {n_range_total} samples')

    n_pulses = pulse_end - pulse_start
    range_width = range_end - range_start

    locations = parse_at(args.at)

    # If using new-style pulse/range bounds, tile the window if it's large
    if args.pulse_start is not None or args.pulse_end is not None or args.range_start is not None or args.range_end is not None:
        if not locations:
            # Check if window is large enough to warrant auto-tiling
            if n_pulses * range_width > (args.tile_pulses * args.tile_range * 4):
                tile_grid = generate_tile_grid(pulse_start, pulse_end, range_start, range_end,
                                              args.tile_pulses, args.tile_range)
                locations = [(p0, r0) for p0, r0, _, _ in tile_grid]
                print(f'  auto-tiling      : {len(locations)} tiles of {args.tile_pulses}x{args.tile_range}')
            else:
                locations = [(pulse_start, range_start)]

    if args.from_clean:
        tile_pulse, tile_range = load_clean_locations(args.from_clean, freq, pols[0])
        print(f'  clean CPI blocks : {len(tile_pulse)}')
        tile_pulses_to_check = {args.pulses, 128, 256, 512} if args.pulses else {n_pulses}
        for n_p in sorted(tile_pulses_to_check):
            runs = find_clean_runs(tile_pulse, tile_range, n_p,
                                   args.cpi_len, args.min_clean_frac)
            marker = ' <-- requested' if n_p == (args.pulses or n_pulses) else ''
            print(f'  clean runs of {n_p:>4} pulses : {len(runs)}{marker}')
        locations += find_clean_runs(tile_pulse, tile_range, args.pulses or n_pulses,
                                     args.cpi_len, args.min_clean_frac)

    if args.grid:
        locations += grid_locations(
            n_pulses_total, n_range_total, args.pulses or n_pulses, args.range_width or range_width,
            args.stride_pulse, args.stride_range, args.max_tiles)

    if not locations:
        raise RuntimeError(
            'No tile locations selected. Pass --at, --from-clean, or --grid.')

    # De-duplicate while preserving order, then cap.
    seen, ordered = set(), []
    for loc in locations:
        if loc not in seen:
            seen.add(loc)
            ordered.append(loc)
    locations = ordered[:args.max_tiles]
    print(f'  tiles to extract : {len(locations)}')

    # Process each polarization
    all_written = []
    for pol in pols:
        written = process_single_channel(raw, freq, pol, args,
                                        pulse_start, pulse_end, range_start, range_end,
                                        n_pulses, range_width, locations)
        all_written.extend(written)

    return all_written


def run_replot(args):
    """Render from a stored tile file without touching the granule."""
    tiles, valids, locations, meta = read_tiles(args.replot)
    print(f'  loaded           : {len(tiles)} tiles from {args.replot}')
    print(f'  tile geometry    : {tiles.shape[1]} pulses x {tiles.shape[2]} range samples')
    tag = f"   |   {meta.get('scene_tag', '')}" if meta.get('scene_tag') else ''
    return render_all(tiles, valids, locations, args.output_dir, args, tag)


def run_demo(args):
    """
    Self test on synthetic data, so the script can be exercised without an
    L0B granule. The synthetic tile contains speckle clutter, a narrowband
    constant modulus emitter on a run of pulses, a wideband emitter on a
    different run, and an ADC gap.
    """
    rng = np.random.default_rng(0)
    P, K = args.pulses, args.range_width

    tile = (rng.standard_normal((P, K)) +
            1j * rng.standard_normal((P, K))).astype(np.complex64) / np.sqrt(2)

    # Narrowband: constant modulus in range, fixed Doppler across pulses.
    nb_rows = slice(P // 4, P // 2)
    ramp = np.exp(2j * np.pi * (0.031 * np.arange(K)[None, :] +
                                0.13 * np.arange(P)[:, None]))
    tile[nb_rows] += (2.0 * ramp[nb_rows]).astype(np.complex64)

    # Wideband: speckly, higher power, a shorter run of pulses.
    wb_rows = slice(int(0.70 * P), int(0.80 * P))
    n_wb = wb_rows.stop - wb_rows.start
    wb = (rng.standard_normal((n_wb, K)) +
          1j * rng.standard_normal((n_wb, K))) / np.sqrt(2)
    tile[wb_rows] += (2.5 * wb).astype(np.complex64)

    valid = np.ones((P, K), dtype=bool)
    valid[:, -K // 12:] = False
    tile[~valid] = 0.0

    locations = [(0, 0)]
    out_h5 = os.path.join(args.output_dir, 'raw_tiles_demo.h5')
    write_tiles(out_h5, [tile], [valid], locations,
                dict(l0b_file='(demo)', frequency='A', polarization='HH',
                     cpi_len=args.cpi_len, mask_mode='demo',
                     caltone_removed=False, scene_tag='synthetic demo'))
    print(f'  wrote            : {out_h5}')

    figs = render_all(np.asarray([tile]), np.asarray([valid]),
                      np.asarray(locations), args.output_dir, args,
                      '   |   synthetic demo')

    # Round trip check.
    t2, v2, l2, _ = read_tiles(out_h5)
    assert np.allclose(t2[0], tile), 'stored tile does not round trip'
    assert np.array_equal(v2[0], valid), 'stored mask does not round trip'
    assert l2[0][0] == 0 and l2[0][1] == 0

    cv_nb = float(np.median(amplitude_texture(tile)[nb_rows]))
    cv_clean = float(np.median(amplitude_texture(tile)[:P // 8]))
    print(f'  amplitude CV     : clean {cv_clean:.3f}, narrowband {cv_nb:.3f}')
    print('  (narrowband should be LOWER: a constant modulus component '
          'suppresses speckle fluctuation)')

    return [out_h5] + figs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Extract, store, and plot raw time-domain L0B tiles '
                    '(no SCM, no eigenvalues, no FFT).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('l0b_file', nargs='?', default=None,
                   help='NISAR L0B granule (omit with --replot or --demo)')
    p.add_argument('--freq', default='A', help="frequency band, 'A' or 'B' (default: A)")
    p.add_argument('--pol', default=None, help="polarization, e.g. 'HH', 'HV'; if omitted, processes all available polarizations")

    p.add_argument('--pulse-start', type=int, default=None,
                   help='starting pulse index (0-based); if omitted, starts at 0')
    p.add_argument('--pulse-end', type=int, default=None,
                   help='ending pulse index (exclusive); if omitted, processes to end')
    p.add_argument('--range-start', type=int, default=None,
                   help='starting range sample index (0-based); if omitted, starts at 0')
    p.add_argument('--range-end', type=int, default=None,
                   help='ending range sample index (exclusive); if omitted, processes to end')

    p.add_argument('--pulses', type=int, default=None,
                   help='(legacy) pulse (azimuth) extent of each tile')
    p.add_argument('--range-width', type=int, default=None,
                   help='(legacy) range sample (fast time) extent of each tile')
    p.add_argument('--tile-pulses', type=int, default=DEFAULT_TILE_PULSES,
                   help='pulse extent for auto-generated tiles')
    p.add_argument('--tile-range', type=int, default=DEFAULT_TILE_RANGE,
                   help='range extent for auto-generated tiles')
    p.add_argument('--cpi-len', type=int, default=CPI_LEN_DEFAULT,
                   help='pulses per CPI block, for clean-run bookkeeping')

    p.add_argument('--at', action='append', default=None, metavar='P0,R0',
                   help='explicit tile origin, repeatable')
    p.add_argument('--from-clean', default=None, metavar='PATH',
                   help='select_clean.py output file or directory')
    p.add_argument('--min-clean-frac', type=float, default=1.0,
                   help='fraction of constituent CPI blocks that must be '
                        'clean; 1.0 requires an unbroken run')
    p.add_argument('--grid', action='store_true',
                   help='add a dense strided grid of tile origins')
    p.add_argument('--stride-pulse', type=int, default=4096)
    p.add_argument('--stride-range', type=int, default=2048)
    p.add_argument('--max-tiles', type=int, default=50,
                   help='cap on the number of tiles extracted')

    p.add_argument('--mask-mode', choices=['subswath', 'amplitude', 'none'],
                   default='subswath', help='validity mask source')
    p.add_argument('--gap-frac', type=float, default=GAP_FRAC_DEFAULT,
                   help='amplitude gap threshold, fraction of peak')
    p.add_argument('--remove-caltone', action='store_true', default=True,
                   help='subtract the instrument caltone before storing')
    p.add_argument('--no-remove-caltone', dest='remove_caltone',
                   action='store_false')

    p.add_argument('--sheet-cols', type=int, default=6,
                   help='number of columns in contact sheet')
    p.add_argument('--no-per-tile', action='store_true',
                   help='contact sheet only, skip the per-tile figures')

    p.add_argument('--replot', default=None, metavar='TILES_H5',
                   help='render from a stored tile file instead of a granule')
    p.add_argument('--demo', action='store_true',
                   help='self test on synthetic data, no granule needed')

    p.add_argument('--scene-tag', default=None,
                   help='free-form label recorded in the file and figures')
    p.add_argument('--output-dir', default='raw_tile_figs')

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 70)
    print('Raw time-domain tile extraction and rendering')
    print('(pulse x range sample; no SCM, no eigenvalues, no FFT)')
    print('=' * 70)

    print(f'  output dir       : {args.output_dir}')
    print(f'  panels           : magnitude (power), phase_cos, phase_sin, valid (UNet inputs)')

    if args.demo:
        written = run_demo(args)
    elif args.replot:
        written = run_replot(args)
    else:
        if not args.l0b_file:
            raise SystemExit(
                'An L0B granule is required unless --replot or --demo is given')
        written = run_from_granule(args)

    print('\nDone. Wrote:')
    for path in written:
        print(f'  {path}')


if __name__ == '__main__':
    main()
