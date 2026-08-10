#!/usr/bin/env python
"""
Create a clean tiles index file for Amazon dataset.

Assumes the entire specified region is clean (no RFI), which is used
as ground truth baseline for synthetic RFI injection.
"""

import argparse
import h5py
import numpy as np


def create_clean_tiles_index(l0b_path, pulse_start, pulse_end, range_start, range_end,
                              cpi_len, cpi_width, output_path, freq='A', pol='HH'):
    """
    Generate tile indices for a clean region.

    Parameters
    ----------
    l0b_path : str
        Path to L0B granule
    pulse_start, pulse_end : int
        Clean pulse range (inclusive)
    range_start, range_end : int
        Clean range sample range (inclusive)
    cpi_len, cpi_width : int
        Tile dimensions
    output_path : str
        Output HDF5 file path
    freq : str
        Frequency band ('A' or 'B')
    pol : str
        Polarization ('HH', 'HV', 'VH', 'VV')
    """

    # Generate tile grid
    pulse_tiles = []
    range_tiles = []

    for p0 in range(pulse_start, pulse_end, cpi_len):
        if p0 + cpi_len > pulse_end:
            break  # Skip incomplete tiles at the end
        for r0 in range(range_start, range_end, cpi_width):
            if r0 + cpi_width > range_end:
                break  # Skip incomplete tiles at the end
            pulse_tiles.append(p0)
            range_tiles.append(r0)

    pulse_tiles = np.array(pulse_tiles, dtype=np.int32)
    range_tiles = np.array(range_tiles, dtype=np.int32)

    n_tiles = len(pulse_tiles)

    print(f"Generated {n_tiles} clean tile indices:")
    print(f"  Pulse range: [{pulse_start}, {pulse_end})")
    print(f"  Range samples: [{range_start}, {range_end})")
    print(f"  Tile size: {cpi_len} x {cpi_width}")
    print(f"  Tiles in pulse dimension: {(pulse_end - pulse_start) // cpi_len}")
    print(f"  Tiles in range dimension: {(range_end - range_start) // cpi_width}")
    print(f"  Total tiles: {n_tiles}")

    # Write to HDF5 (flat format matching select_clean.py output)
    with h5py.File(output_path, 'w') as f:
        f.attrs['source_granule'] = l0b_path
        f.attrs['frequency'] = freq
        f.attrs['polarization'] = pol
        f.attrs['pulse_start'] = pulse_start
        f.attrs['pulse_end'] = pulse_end
        f.attrs['range_start'] = range_start
        f.attrs['range_end'] = range_end
        f.attrs['cpi_len'] = cpi_len
        f.attrs['cpi_width'] = cpi_width
        f.attrs['assumed_clean'] = True
        f.attrs['description'] = (
            'Clean tile indices for synthetic RFI injection. '
            'Assumes entire region is RFI-free.'
        )

        f.create_dataset('tile_pulse', data=pulse_tiles, dtype=np.int32)
        f.create_dataset('tile_range', data=range_tiles, dtype=np.int32)

        # Create dummy eigenvalues dataset for shape inference compatibility
        f.create_dataset('eigenvalues',
                        shape=(n_tiles, cpi_len),
                        dtype=np.float32)

    print(f"\nWrote clean tiles index to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('l0b_file', help='NISAR L0B granule path')
    parser.add_argument('--pulse-start', type=int, required=True,
                       help='Start pulse index (inclusive)')
    parser.add_argument('--pulse-end', type=int, required=True,
                       help='End pulse index (exclusive)')
    parser.add_argument('--range-start', type=int, required=True,
                       help='Start range sample (inclusive)')
    parser.add_argument('--range-end', type=int, required=True,
                       help='End range sample (exclusive)')
    parser.add_argument('--cpi-len', type=int, default=256,
                       help='Tile height (pulses)')
    parser.add_argument('--cpi-width', type=int, default=256,
                       help='Tile width (range samples)')
    parser.add_argument('--freq', default='A', choices=['A', 'B'],
                       help='Frequency band')
    parser.add_argument('--pol', default='HH', choices=['HH', 'HV', 'VH', 'VV'],
                       help='Polarization')
    parser.add_argument('--output', '-o', required=True,
                       help='Output HDF5 file path')

    args = parser.parse_args()

    create_clean_tiles_index(
        args.l0b_file,
        args.pulse_start, args.pulse_end,
        args.range_start, args.range_end,
        args.cpi_len, args.cpi_width,
        args.output,
        args.freq, args.pol
    )


if __name__ == '__main__':
    main()
