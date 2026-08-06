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
For each tile, up to four panels are rendered:

  magnitude   Floor relative log magnitude in dB. The general brightness
              cue. Wideband RFI raises the speckle level.

  phasediff   Adjacent-pulse phase difference, arg(x[m] * conj(x[m-1])),
              in radians. The slow-time coherence cue, and the direct
              pointwise analog of what the SCM off-diagonal terms measure.
              An emitter with a fixed Doppler shows a near constant value
              across range samples. Terrain returns in RAW data are also
              pulse-to-pulse correlated, so this is not an RFI-only cue.

  texture     Local coefficient of variation of the amplitude, computed in a
              small range window. This is the narrowband cue in the time
              domain: a constant modulus tone REDUCES local amplitude
              fluctuation relative to speckle, so narrowband RFI appears as
              a dark (low variation) region rather than a bright one.

  valid       ADC gap / subswath validity mask.

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
    # Look at known contaminated locations in a scene
    python plot_raw_tiles.py granule.h5 --freq A --pol HV \\
        --at 12000,3000 --at 12256,3000 \\
        --pulses 256 --range-width 512 \\
        --output-dir figs/vienna

    # Sweep tile geometry on the same underlying data
    python plot_raw_tiles.py granule.h5 --freq A --pol HH \\
        --at 40000,5000 --pulses 256 --range-width 250 --output-dir figs/geom_256x250
    python plot_raw_tiles.py granule.h5 --freq A --pol HH \\
        --at 40000,5000 --pulses 256 --range-width 512 --output-dir figs/geom_256x512
    python plot_raw_tiles.py granule.h5 --freq A --pol HH \\
        --at 40000,5000 --pulses 512 --range-width 512 --output-dir figs/geom_512x512

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

# Caltone removal, matching the detection and generation paths.
CALTONE_WINDOW_SIZE = 64
CALTONE_DEFAULT_FREQ_HZ = 1214.883e6
CALTONE_LO_HZ = 1200e6
CALTONE_CLOCK_HZ = 240e6

# Display defaults for the magnitude panel, in dB above the per-tile floor.
# The upper limit is deliberately modest: high power RFI saturates any scale,
# but moderate and low JSR contamination is only a few dB above the floor and
# is invisible on a wide scale. Raise --vmax-db for the strong blob scenes.
MAG_VMIN_DB = -5.0
MAG_VMAX_DB = 15.0

# Range window (in samples) for the local amplitude coefficient of variation.
TEXTURE_WINDOW_DEFAULT = 9

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

def floor_relative_db(tile, valid=None, quantile=0.5):
    """
    Log magnitude in dB relative to the per-tile floor.

    The floor is estimated over valid samples only, so an ADC gap cannot drag
    the estimate down and inflate the apparent contrast everywhere else.
    """
    mag_db = 20.0 * np.log10(np.abs(tile) + EPS)
    if valid is not None and valid.any():
        floor = float(np.quantile(mag_db[valid], quantile))
    else:
        floor = float(np.quantile(mag_db, quantile))
    return (mag_db - floor).astype(np.float32), floor


def adjacent_pulse_phase_diff(tile):
    """
    arg(x[m] * conj(x[m-1])) at every range sample, in radians.

    Row 0 has no predecessor and is replicated from row 1 so the output keeps
    the tile shape.
    """
    if tile.shape[0] < 2:
        return np.zeros(tile.shape, dtype=np.float32)
    phase = np.angle(tile[1:] * np.conj(tile[:-1])).astype(np.float32)
    out = np.empty(tile.shape, dtype=np.float32)
    out[1:] = phase
    out[0] = phase[0]
    return out


def amplitude_texture(tile, window=TEXTURE_WINDOW_DEFAULT):
    """
    Local coefficient of variation of the amplitude along range.

    For fully developed speckle this sits near 0.52 (the Rayleigh value). A
    constant modulus component such as a narrowband tone pulls it DOWN, so
    narrowband RFI reads as a dark region here even where the magnitude panel
    shows nothing obvious. This is the main reason the panel exists.
    """
    if window < 3:
        raise ValueError("texture window must be >= 3")
    if window % 2 == 0:
        window += 1

    amp = np.abs(tile).astype(np.float32)
    kernel = np.ones(window, dtype=np.float32) / window

    # Reflect-pad along range so the edges are not biased by zero padding.
    pad = window // 2
    padded = np.pad(amp, ((0, 0), (pad, pad)), mode='reflect')

    mean = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode='valid'), 1, padded)
    mean_sq = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode='valid'), 1, padded ** 2)

    var = np.maximum(mean_sq - mean ** 2, 0.0)
    return (np.sqrt(var) / np.maximum(mean, EPS)).astype(np.float32)


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
    with h5py.File(path, 'w') as f:
        f.create_dataset('tiles', data=tiles, chunks=(1, p, k),
                         compression='gzip', compression_opts=1)
        f.create_dataset('valid', data=valids, chunks=(1, p, k),
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

def plot_tile(tile, valid, p0, r0, out_path, panels=('magnitude', 'phasediff',
                                                     'texture', 'valid'),
              texture_window=TEXTURE_WINDOW_DEFAULT,
              vmin_db=MAG_VMIN_DB, vmax_db=MAG_VMAX_DB, title_extra=''):
    """Render one tile as a row of panels and save it."""
    panels = [p for p in panels]
    n_panels = len(panels)
    if n_panels == 0:
        raise ValueError("at least one panel is required")

    mag_db, floor = floor_relative_db(tile, valid)

    fig_w = 4.2 * n_panels + 1.0
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 4.8),
                             squeeze=False)
    axes = axes[0]

    for ax, name in zip(axes, panels):
        if name == 'magnitude':
            img, cmap, vmin, vmax, label = (mag_db, 'viridis', vmin_db,
                                            vmax_db, 'dB above floor')
        elif name == 'phasediff':
            img, cmap, vmin, vmax, label = (adjacent_pulse_phase_diff(tile),
                                            'twilight', -np.pi, np.pi,
                                            'radians')
        elif name == 'texture':
            img, cmap, vmin, vmax, label = (
                amplitude_texture(tile, texture_window), 'magma', 0.0, 1.0,
                'amplitude CV')
        elif name == 'valid':
            img, cmap, vmin, vmax, label = (valid.astype(np.float32), 'gray',
                                            0.0, 1.0, 'valid')
        else:
            raise ValueError(f"unknown panel '{name}'")

        im = ax.imshow(img, aspect='auto', origin='upper', cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title(name)
        ax.set_xlabel('range sample')
        ax.set_ylabel('pulse')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)

    valid_frac = float(valid.mean())
    fig.suptitle(
        f'pulse {p0}, range {r0}   |   {tile.shape[0]} x {tile.shape[1]}   |   '
        f'floor {floor:.1f} dB   |   valid {100 * valid_frac:.0f}%{title_extra}',
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def plot_contact_sheet(tiles, valids, locations, out_path, n_cols=6,
                       vmin_db=MAG_VMIN_DB, vmax_db=MAG_VMAX_DB,
                       max_tiles=60):
    """
    Grid of magnitude thumbnails for fast scanning across many tiles.

    Every thumbnail uses the SAME colour limits, relative to its own floor,
    so relative contamination is comparable across the sheet.
    """
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
        mag_db, _ = floor_relative_db(tiles[idx], valids[idx])
        ax.imshow(mag_db, aspect='auto', origin='upper', cmap='viridis',
                  vmin=vmin_db, vmax=vmax_db, interpolation='nearest')
        ax.set_title(f'{idx}: p{locations[idx][0]} r{locations[idx][1]}',
                     fontsize=7)

    fig.suptitle('floor relative magnitude, dB', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def render_all(tiles, valids, locations, out_dir, args, tag=''):
    """Per-tile figures plus the contact sheet."""
    written = []
    panels = tuple(args.panels)

    if not args.no_per_tile:
        for idx, (tile, valid, (p0, r0)) in enumerate(
                zip(tiles, valids, locations)):
            name = f'tile_{idx:03d}_p{int(p0)}_r{int(r0)}.png'
            written.append(plot_tile(
                tile, valid, int(p0), int(r0),
                os.path.join(out_dir, name),
                panels=panels, texture_window=args.texture_window,
                vmin_db=args.vmin_db, vmax_db=args.vmax_db,
                title_extra=tag))

    sheet = plot_contact_sheet(
        tiles, valids, locations,
        os.path.join(out_dir, 'contact_sheet.png'),
        n_cols=args.sheet_cols, vmin_db=args.vmin_db, vmax_db=args.vmax_db)
    if sheet:
        written.append(sheet)
    return written


# ---------------------------------------------------------------------------
# MODES
# ---------------------------------------------------------------------------

def run_from_granule(args):
    """Read tiles from an L0B granule, store them, and render them."""
    _import_l0b_readers()

    raw = Raw(hdf5file=args.l0b_file)
    raw.parsePolarizations()

    freq = args.freq
    pol = args.pol
    if freq is None or pol is None:
        raise ValueError('--freq and --pol are required when reading a granule')

    dataset = raw.getRawDataset(freq, pol)
    n_pulses_total, n_range_total = dataset.shape
    print(f'  granule shape    : {n_pulses_total} pulses x {n_range_total} range samples')

    locations = parse_at(args.at)

    if args.from_clean:
        tile_pulse, tile_range = load_clean_locations(args.from_clean, freq, pol)
        print(f'  clean CPI blocks : {len(tile_pulse)}')
        for n_p in sorted({args.pulses, 128, 256, 512}):
            runs = find_clean_runs(tile_pulse, tile_range, n_p,
                                   args.cpi_len, args.min_clean_frac)
            marker = ' <-- requested' if n_p == args.pulses else ''
            print(f'  clean runs of {n_p:>4} pulses : {len(runs)}{marker}')
        locations += find_clean_runs(tile_pulse, tile_range, args.pulses,
                                     args.cpi_len, args.min_clean_frac)

    if args.grid:
        locations += grid_locations(
            n_pulses_total, n_range_total, args.pulses, args.range_width,
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

    if args.remove_caltone:
        remover, caltone_freq = build_tone_remover(raw, freq, pol, n_range_total)
        print(f'  caltone removal  : ON (f = {caltone_freq / 1e6:.4f} MHz)')
    else:
        remover, caltone_freq = None, None
        print('  caltone removal  : OFF')

    tiles, valids, kept = [], [], []
    for p0, r0 in locations:
        if p0 + args.pulses > n_pulses_total or r0 + args.range_width > n_range_total:
            print(f'  [skip] ({p0}, {r0}) extends past the granule bounds')
            continue

        tile = read_raw_tile(raw, freq, pol, p0, args.pulses, r0,
                             args.range_width, remover)

        if args.mask_mode == 'subswath':
            valid = get_subswath_mask(raw, freq, pol, p0, args.pulses, r0,
                                      args.range_width)
        elif args.mask_mode == 'amplitude':
            valid = amplitude_gap_mask(tile, args.gap_frac)
        else:
            valid = np.ones(tile.shape, dtype=bool)

        tiles.append(tile)
        valids.append(valid)
        kept.append((p0, r0))

    if not tiles:
        raise RuntimeError('No tiles could be read; check the locations given')

    out_h5 = os.path.join(args.output_dir,
                          f'raw_tiles_{freq}_{pol}.h5')
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
                      np.asarray(kept), args.output_dir, args, tag)
    return [out_h5] + figs


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
    p.add_argument('--freq', default=None, help="frequency band, 'A' or 'B'")
    p.add_argument('--pol', default=None, help="polarization, e.g. 'HH', 'HV'")

    p.add_argument('--pulses', type=int, default=PULSES_DEFAULT,
                   help='pulse (azimuth) extent of each tile')
    p.add_argument('--range-width', type=int, default=RANGE_WIDTH_DEFAULT,
                   help='range sample (fast time) extent of each tile')
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
    p.add_argument('--max-tiles', type=int, default=24,
                   help='cap on the number of tiles extracted')

    p.add_argument('--mask-mode', choices=['subswath', 'amplitude', 'none'],
                   default='subswath', help='validity mask source')
    p.add_argument('--gap-frac', type=float, default=GAP_FRAC_DEFAULT,
                   help='amplitude gap threshold, fraction of peak')
    p.add_argument('--remove-caltone', action='store_true', default=True,
                   help='subtract the instrument caltone before storing')
    p.add_argument('--no-remove-caltone', dest='remove_caltone',
                   action='store_false')

    p.add_argument('--panels', nargs='+',
                   default=['magnitude', 'phasediff', 'texture', 'valid'],
                   choices=['magnitude', 'phasediff', 'texture', 'valid'],
                   help='panels rendered per tile')
    p.add_argument('--texture-window', type=int, default=TEXTURE_WINDOW_DEFAULT,
                   help='range window for the amplitude CV panel')
    p.add_argument('--vmin-db', type=float, default=MAG_VMIN_DB)
    p.add_argument('--vmax-db', type=float, default=MAG_VMAX_DB)
    p.add_argument('--sheet-cols', type=int, default=6)
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
    print(f'  tile geometry    : {args.pulses} pulses x {args.range_width} range samples')
    print(f'  output dir       : {args.output_dir}')

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
