#!/usr/bin/env python
"""
Generate U-Net training data from Amazon dataset.

Simplified workflow:
1. Reads clean Amazon region directly from L0B
2. Injects synthetic RFI blobs
3. Outputs training data with binary masks

Usage:
    python generate_amazon_data.py <l0b_file> \\
        --pulse-start 813924 --pulse-end 888222 \\
        --range-start 1000 --range-end 26000 \\
        --output data/amazon_unet
"""

import os
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import h5py

# Import from existing script
from generate_unet_segmentation_data import (
    _silence_third_party_noise,
    Raw, ToneRemover,
    build_tone_remover, read_raw_tile, get_subswath_mask, amplitude_gap_mask,
    inject_rfi_blobs, channel_id, make_tile_seed_seq,
    tile_signal_power, TileMeta,
    CPI_LEN_DEFAULT, CPI_WIDTH_DEFAULT,
    MIN_BLOBS_DEFAULT, MAX_BLOBS_DEFAULT,
    MIN_PULSE_SIZE_DEFAULT, MAX_PULSE_SIZE_DEFAULT,
    MIN_RANGE_FRAC_DEFAULT, MAX_RANGE_FRAC_DEFAULT,
    JSR_MIN_DB_DEFAULT, JSR_MAX_DB_DEFAULT,
    MASK_THRESHOLD_DEFAULT, MAX_CONTAMINATION_FRAC_DEFAULT,
    SIGMA_SCALE_DEFAULT, SEED_DEFAULT
)

_silence_third_party_noise()


def generate_tiles_from_region(raw, freq, pol, pulse_start, pulse_end,
                                range_start, range_end, cpi_len, cpi_width,
                                args, output_path):
    """Generate training data from a clean region."""

    # Generate tile indices
    pulse_tiles = []
    range_tiles = []
    for p0 in range(pulse_start, pulse_end, cpi_len):
        if p0 + cpi_len > pulse_end:
            break
        for r0 in range(range_start, range_end, cpi_width):
            if r0 + cpi_width > range_end:
                break
            pulse_tiles.append(p0)
            range_tiles.append(r0)

    n_tiles = len(pulse_tiles)
    print(f"\n{freq}-{pol}: {n_tiles} tiles ({cpi_len}x{cpi_width})")

    # Caltone removal
    remover = None
    if args.remove_caltone:
        full_range = raw.getRawDataset(freq, pol).shape[1]
        remover, caltone_freq = build_tone_remover(raw, freq, pol, full_range)
        print(f"  Caltone removal: ON ({caltone_freq/1e6:.4f} MHz)")

    # Create output
    with h5py.File(output_path, 'w') as f:
        # Datasets
        tiles_ds = f.create_dataset('tiles', shape=(0, cpi_len, cpi_width),
                                     maxshape=(None, cpi_len, cpi_width),
                                     dtype=np.complex64, chunks=(16, cpi_len, cpi_width),
                                     compression='gzip', compression_opts=4)
        masks_ds = f.create_dataset('masks', shape=(0, cpi_len, cpi_width),
                                     maxshape=(None, cpi_len, cpi_width),
                                     dtype=bool, chunks=(16, cpi_len, cpi_width),
                                     compression='gzip', compression_opts=4)
        valid_ds = f.create_dataset('valid', shape=(0, cpi_len, cpi_width),
                                     maxshape=(None, cpi_len, cpi_width),
                                     dtype=bool, chunks=(16, cpi_len, cpi_width),
                                     compression='gzip', compression_opts=4)

        # Metadata
        n_blobs_ds = f.create_dataset('n_blobs', shape=(0,), maxshape=(None,),
                                       dtype=np.int8, chunks=(4096,))
        signal_power_db_ds = f.create_dataset('signal_power_db', shape=(0,), maxshape=(None,),
                                               dtype=np.float32, chunks=(4096,))

        # Attributes
        f.attrs['source_granule'] = args.l0b_file
        f.attrs['frequency'] = freq
        f.attrs['polarization'] = pol
        f.attrs['pulse_range'] = [pulse_start, pulse_end]
        f.attrs['range_samples'] = [range_start, range_end]
        f.attrs['cpi_len'] = cpi_len
        f.attrs['cpi_width'] = cpi_width
        f.attrs['min_blobs'] = args.min_blobs
        f.attrs['max_blobs'] = args.max_blobs
        f.attrs['jsr_range_db'] = [args.jsr_min_db, args.jsr_max_db]
        f.attrs['max_contamination_frac'] = args.max_contamination_frac
        f.attrs['seed'] = args.seed
        f.attrs['generated_utc'] = datetime.now(timezone.utc).isoformat()

        # Process tiles
        blob_counts = {}
        chan = channel_id(freq, pol)

        for i, (p0, r0) in enumerate(zip(pulse_tiles, range_tiles)):
            # Read tile
            cpi = read_raw_tile(raw, freq, pol, p0, cpi_len, r0, cpi_width, remover)

            # Validity mask
            if args.mask_mode == 'subswath':
                cpi_mask = get_subswath_mask(raw, freq, pol, p0, cpi_len, r0, cpi_width)
            elif args.mask_mode == 'amplitude':
                cpi_mask = amplitude_gap_mask(cpi)
            else:
                cpi_mask = None

            # Generate seed
            pulse_tile = p0 // cpi_len
            range_tile = r0 // cpi_width
            tile_ss = make_tile_seed_seq(args.seed, chan, pulse_tile, range_tile)

            # Draw number of blobs
            count_seed = tile_ss.spawn(1)[0]
            rng = np.random.default_rng(count_seed)
            n_blobs = int(rng.integers(args.min_blobs, args.max_blobs + 1))

            # Inject RFI
            tile, mask, meta = inject_rfi_blobs(
                cpi, cpi_mask, n_blobs,
                args.min_pulse_size, args.max_pulse_size,
                args.min_range_frac, args.max_range_frac,
                args.jsr_min_db, args.jsr_max_db,
                args.mask_threshold, args.sigma_scale,
                tile_ss, args.max_contamination_frac
            )

            # Append
            n = tiles_ds.shape[0]
            tiles_ds.resize(n + 1, axis=0)
            masks_ds.resize(n + 1, axis=0)
            valid_ds.resize(n + 1, axis=0)
            n_blobs_ds.resize(n + 1, axis=0)
            signal_power_db_ds.resize(n + 1, axis=0)

            tiles_ds[n] = tile
            masks_ds[n] = mask
            valid_ds[n] = cpi_mask if cpi_mask is not None else np.ones_like(mask)
            n_blobs_ds[n] = meta.n_blobs
            signal_power_db_ds[n] = meta.signal_power_db

            blob_counts[meta.n_blobs] = blob_counts.get(meta.n_blobs, 0) + 1

            if (i + 1) % 1000 == 0 or (i + 1) == n_tiles:
                print(f"  {i+1}/{n_tiles} tiles processed")

        # Save blob distribution
        f.attrs['blob_distribution'] = json.dumps({str(k): int(v) for k, v in blob_counts.items()})
        print(f"  Blob distribution: {dict(blob_counts)}")

    print(f"✓ Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # Required
    parser.add_argument('l0b_file', help='NISAR L0B granule')
    parser.add_argument('--pulse-start', type=int, required=True)
    parser.add_argument('--pulse-end', type=int, required=True)
    parser.add_argument('--range-start', type=int, required=True)
    parser.add_argument('--range-end', type=int, required=True)

    # Optional
    parser.add_argument('--freq', default='A')
    parser.add_argument('--pols', nargs='+', default=['HH', 'HV'])
    parser.add_argument('--cpi-len', type=int, default=CPI_LEN_DEFAULT)
    parser.add_argument('--cpi-width', type=int, default=CPI_WIDTH_DEFAULT)

    # RFI parameters
    parser.add_argument('--min-blobs', type=int, default=MIN_BLOBS_DEFAULT)
    parser.add_argument('--max-blobs', type=int, default=MAX_BLOBS_DEFAULT)
    parser.add_argument('--min-pulse-size', type=int, default=MIN_PULSE_SIZE_DEFAULT)
    parser.add_argument('--max-pulse-size', type=int, default=MAX_PULSE_SIZE_DEFAULT)
    parser.add_argument('--min-range-frac', type=float, default=MIN_RANGE_FRAC_DEFAULT)
    parser.add_argument('--max-range-frac', type=float, default=MAX_RANGE_FRAC_DEFAULT)
    parser.add_argument('--jsr-min-db', type=float, default=JSR_MIN_DB_DEFAULT)
    parser.add_argument('--jsr-max-db', type=float, default=JSR_MAX_DB_DEFAULT)
    parser.add_argument('--mask-threshold', type=float, default=MASK_THRESHOLD_DEFAULT)
    parser.add_argument('--max-contamination-frac', type=float, default=MAX_CONTAMINATION_FRAC_DEFAULT)
    parser.add_argument('--sigma-scale', type=float, default=SIGMA_SCALE_DEFAULT)

    # System
    parser.add_argument('--remove-caltone', action='store_true', default=True)
    parser.add_argument('--no-remove-caltone', dest='remove_caltone', action='store_false')
    parser.add_argument('--mask-mode', choices=['none', 'subswath', 'amplitude'], default='subswath')
    parser.add_argument('--output', '-o', default='data/amazon_unet')
    parser.add_argument('--seed', type=int, default=SEED_DEFAULT)

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print("="*60)
    print("Amazon U-Net Data Generation")
    print("="*60)
    print(f"L0B: {args.l0b_file}")
    print(f"Region: pulses [{args.pulse_start}, {args.pulse_end}), "
          f"range [{args.range_start}, {args.range_end})")
    print(f"Tile size: {args.cpi_len}x{args.cpi_width}")
    print(f"Polarizations: {', '.join(args.pols)}")
    print(f"Blobs per tile: {args.min_blobs}-{args.max_blobs}")
    print(f"Max contamination: {args.max_contamination_frac*100:.0f}%")

    # Load raw
    raw = Raw(hdf5file=args.l0b_file)
    raw.parsePolarizations()

    # Generate for each polarization
    for pol in args.pols:
        output_path = os.path.join(args.output, f'amazon_{args.freq}_{pol}.h5')
        generate_tiles_from_region(
            raw, args.freq, pol,
            args.pulse_start, args.pulse_end,
            args.range_start, args.range_end,
            args.cpi_len, args.cpi_width,
            args, output_path
        )

    print(f"\n{'='*60}")
    print("✓ Done!")
    print(f"Output directory: {args.output}")
    print("="*60)


if __name__ == '__main__':
    main()
