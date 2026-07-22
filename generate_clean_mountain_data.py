#!/usr/bin/env python
"""
generate_clean_mountain_data.py

Build a CLEAN mountain training set: one label-0 record per clean tile, with NO
RFI injected. Use this when the clean class should come entirely from real
mountain background -- e.g. training on contaminated Amazon (RFI classes 1..6)
paired with clean mountain (class 0).

This is the no-injection sibling of generate_mountain_data.py. It deliberately
does NOT go through the RFI-injection path, so there is no max_bands=0 corner
case (that path builds zero-width jsr/band datasets, which h5py cannot chunk).
Instead it reuses generate_mountain_data.py's raw read (with caltone removal),
gap-exclusion SCM, and feature layout, then writes only the datasets
train_only.py actually consumes:

    labels          int8    (N,)        all 0 (clean)
    eigenvalues     float32 (N, cpi_len) descending, LINEAR scale
    diagonal        float32 (N, cpi_len) SCM diagonal, LINEAR, unnormalized
    diag_valid_idx  bool    (N, cpi_len) per-index diagonal validity
    signal_power_db float32 (N,)        tile baseline power, 10*log10
    valid_fraction  float32 (N,)        fraction of valid samples
    tile_pulse      int32   (N,)        absolute pulse index of tile row 0
    tile_range      int32   (N,)        absolute range index of tile col 0

The tile locations come from the same clean_mountains_filtered.h5 that
generate_mountain_data.py reads, so the two sets are drawn from the identical
clean background; this one just never contaminates it.

Usage
-----
    python generate_clean_mountain_data.py clean_mountains_filtered.h5 granule.h5 \
        --mask-mode subswath \
        --off-diag-overlap-ratio 0.2 --diag-valid-ratio 0.15 \
        --output-dir data/mountain_clean

One HDF5 file per channel is written: mountain_clean_data_<freq>_<pol>.h5
"""

import os
import json
import argparse
from datetime import datetime, timezone

import numpy as np
import h5py

# Reuse the exact raw read (caltone removal), SCM, feature and mask code from
# the RFI mountain generator so the clean records match it bit-for-bit apart
# from the (absent) injection.
from generate_mountain_data import (
    read_raw_tile,
    get_subswath_mask,
    amplitude_gap_mask,
    compute_scm_eigs_and_diag,
    tile_signal_power,
    build_tone_remover,
    format_class_distribution,
    CALTONE_WINDOW_SIZE,
    CPI_LEN_DEFAULT,
    CPI_WIDTH_DEFAULT,
    OFF_DIAG_OVERLAP_RATIO_DEFAULT,
    DIAG_VALID_RATIO_DEFAULT,
)
from nisar.products.readers.Raw import Raw  # noqa: E402


def generate_clean_group(raw, freq, pol, grp_name, pulse_idx, range_idx,
                         cpi_len, cpi_width, args, out_dir):
    """
    Emit one clean (label-0) record per clean tile of one freq_X_pol_Y group.

    Returns
    -------
    n_tiles : int
    out_path : str
    """
    n_tiles = len(pulse_idx)
    out_path = os.path.join(out_dir, f"mountain_clean_data_{freq}_{pol}.h5")
    print(f"\n[{freq}-{pol}] source group: {grp_name}  ({n_tiles} clean tiles)")
    print(f"  -> {out_path}")

    # Caltone remover, sized to the FULL range width (remove_tone is anchored at
    # range sample 0), built once per channel. On by default.
    if args.remove_caltone:
        full_range = raw.getRawDataset(freq, pol).shape[1]
        remover, caltone_freq = build_tone_remover(raw, freq, pol, full_range)
        print(f"  caltone removal ON  (f_caltone = {caltone_freq/1e6:.4f} MHz, "
              f"window = {CALTONE_WINDOW_SIZE})")
    else:
        remover, caltone_freq = None, None
        print("  caltone removal OFF")

    eigs = np.zeros((n_tiles, cpi_len), dtype=np.float32)
    diag = np.zeros((n_tiles, cpi_len), dtype=np.float32)
    dvalid = np.zeros((n_tiles, cpi_len), dtype=bool)
    sig_db = np.zeros(n_tiles, dtype=np.float32)
    vfrac = np.zeros(n_tiles, dtype=np.float32)
    tpulse = np.zeros(n_tiles, dtype=np.int32)
    trange = np.zeros(n_tiles, dtype=np.int32)

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

        eigvals, diag_lin, diag_valid_idx = compute_scm_eigs_and_diag(
            cpi, cpi_mask, cpi_width,
            args.off_diag_overlap_ratio, args.diag_valid_ratio,
        )

        eigs[i] = eigvals
        diag[i] = diag_lin.astype(np.float32)
        dvalid[i] = diag_valid_idx
        sig_db[i] = 10.0 * np.log10(tile_signal_power(cpi, cpi_mask))
        vfrac[i] = (float(cpi_mask.sum()) / cpi_mask.size
                    if cpi_mask is not None else 1.0)
        tpulse[i] = p0
        trange[i] = r0

        if (i + 1) % report_every == 0 or (i + 1) == n_tiles:
            print(f"    {i + 1}/{n_tiles} clean tiles processed")

    use_mask = args.mask_mode != 'none'
    with h5py.File(out_path, 'w') as f:
        # Provenance
        f.attrs['clean_h5'] = os.path.basename(args.clean_h5)
        f.attrs['clean_h5_path'] = args.clean_h5
        f.attrs['clean_h5_group'] = grp_name
        f.attrs['granule'] = os.path.basename(args.l0b_file)
        f.attrs['granule_path'] = args.l0b_file
        f.attrs['frequency'] = freq
        f.attrs['polarization'] = pol
        f.attrs['n_tiles'] = n_tiles
        f.attrs['n_records'] = n_tiles
        f.attrs['cpi_len'] = cpi_len
        f.attrs['cpi_width'] = cpi_width
        # Clean-only: label space is a single class {0}. train_only.py takes the
        # max n_classes across all loaded files, so this stays compatible with an
        # Amazon RFI set that declares n_classes = 7.
        f.attrs['min_bands'] = 0
        f.attrs['max_bands'] = 0
        f.attrs['n_classes'] = 1
        f.attrs['seed'] = args.seed
        f.attrs['content'] = 'clean mountain background, no RFI injected (all label 0)'
        f.attrs['mask_mode'] = args.mask_mode
        f.attrs['gap_exclusion_used'] = bool(use_mask)
        f.attrs['off_diag_overlap_ratio'] = args.off_diag_overlap_ratio
        f.attrs['diag_valid_ratio'] = args.diag_valid_ratio
        f.attrs['eigenvalue_scale'] = 'linear, descending'
        f.attrs['diagonal_scale'] = 'linear, unnormalized'
        f.attrs['caltone_removed'] = bool(args.remove_caltone)
        if caltone_freq is not None:
            f.attrs['caltone_freq_hz'] = float(caltone_freq)
            f.attrs['caltone_window_size'] = CALTONE_WINDOW_SIZE
        f.attrs['label_histogram'] = json.dumps({'0': n_tiles})
        f.attrs['generated_utc'] = datetime.now(timezone.utc).isoformat()

        def mk(name, data, **kw):
            d = f.create_dataset(name, data=data, **kw)
            return d

        mk('labels', np.zeros(n_tiles, dtype=np.int8))
        mk('eigenvalues', eigs, compression='gzip')
        mk('diagonal', diag, compression='gzip')
        mk('diag_valid_idx', dvalid, compression='gzip')
        mk('signal_power_db', sig_db)
        mk('valid_fraction', vfrac)
        mk('tile_pulse', tpulse)
        mk('tile_range', trange)

        f['labels'].attrs['description'] = 'knee: number of injected RFI bands (0 = clean); always 0 here'
        f['eigenvalues'].attrs['description'] = 'SCM eigenvalues, descending, LINEAR scale'
        f['diagonal'].attrs['description'] = 'SCM diagonal, LINEAR scale, unnormalized power per pulse row'
        f['diag_valid_idx'].attrs['description'] = 'per-index bool: enough non-gap samples to trust the diagonal entry'

    report_text, _ = format_class_distribution({0: n_tiles}, 1)
    print(report_text)

    return n_tiles, out_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('clean_h5', help='clean_mountains_filtered.h5 (or clean_mountains.h5)')
    parser.add_argument('l0b_file', help='Source NISAR L0B HDF5 granule the clean tiles were drawn from')

    parser.add_argument('--freq', default=None,
                        help='Restrict to one frequency group. Default: all groups present.')
    parser.add_argument('--pol', default=None,
                        help='Restrict to one polarization group. Default: all groups present.')

    parser.add_argument('--cpi-width', type=int, default=None,
                        help='CPI width used when the clean tiles were selected. '
                             'Default: CPI_WIDTH_DEFAULT (250).')

    parser.add_argument('--remove-caltone', dest='remove_caltone',
                        action='store_true', default=True,
                        help='Subtract the instrument caltone from the raw data '
                             'before the SCM/features (default: on).')
    parser.add_argument('--no-remove-caltone', dest='remove_caltone',
                        action='store_false',
                        help='Leave the caltone in the raw data (legacy behavior).')

    parser.add_argument('--mask-mode', choices=['none', 'subswath', 'amplitude'], default='subswath',
                        help='Validity mask for the gap-exclusion SCM (match generate_mountain_data.py).')
    parser.add_argument('--off-diag-overlap-ratio', type=float, default=OFF_DIAG_OVERLAP_RATIO_DEFAULT)
    parser.add_argument('--diag-valid-ratio', type=float, default=DIAG_VALID_RATIO_DEFAULT)

    parser.add_argument('--output-dir', default='data/mountain_clean')
    parser.add_argument('--seed', type=int, default=0,
                        help='Recorded for provenance only; clean generation has no randomness.')

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 70)
    print('Clean mountain training set generation (label 0 only, no RFI injected)')
    print('=' * 70)
    print(f'  clean tiles from : {args.clean_h5}')
    print(f'  source granule   : {args.l0b_file}')
    print(f'  caltone removal  : {args.remove_caltone}')
    print(f'  mask mode        : {args.mask_mode}')

    raw = Raw(hdf5file=args.l0b_file)
    raw.parsePolarizations()

    h5_in = h5py.File(args.clean_h5, 'r')

    written = []
    grand_total = 0
    group_totals = {}
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

        n_tiles, out_path = generate_clean_group(
            raw, freq, pol, grp_name, pulse_idx, range_idx,
            cpi_len, cpi_width, args, args.output_dir
        )
        written.append(out_path)
        group_totals[f'{freq}-{pol}'] = n_tiles
        grand_total += n_tiles

    h5_in.close()

    if not written:
        raise RuntimeError('No matching freq/pol groups were processed; check --freq/--pol filters')

    print('\n' + '=' * 70)
    print('TRAIN/VAL POOL SUMMARY (clean mountain, all groups combined)')
    print('=' * 70)
    for chan, tot in group_totals.items():
        print(f'  {chan:<8}: {tot} samples')
    print(f'  {"-"*40}')
    print(f'  total clean samples: {grand_total}  (all label 0)')

    print('\nDone. Wrote:')
    for path in written:
        print(f'  {path}')


if __name__ == '__main__':
    main()
