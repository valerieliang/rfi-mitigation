#!/usr/bin/env python
"""
Create a multi-polarization clean tiles index file for Amazon dataset.

Stores multiple polarizations in a single HDF5 file as separate groups.
"""

import argparse
import h5py
import numpy as np


def create_multipol_clean_index(l0b_path, pulse_start, pulse_end, range_start, range_end,
                                cpi_len, cpi_width, output_path, freq='A', pols=None):
    """
    Generate tile indices for multiple polarizations in one file.

    Parameters
    ----------
    pols : list of str
        Polarizations to include (e.g., ['HH', 'HV'])
    """
    if pols is None:
        pols = ['HH', 'HV']

    # Generate tile grid (same for all polarizations)
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

    pulse_tiles = np.array(pulse_tiles, dtype=np.int32)
    range_tiles = np.array(range_tiles, dtype=np.int32)
    n_tiles = len(pulse_tiles)

    print(f"Generated {n_tiles} clean tile indices:")
    print(f"  Pulse range: [{pulse_start}, {pulse_end})")
    print(f"  Range samples: [{range_start}, {range_end})")
    print(f"  Tile size: {cpi_len} x {cpi_width}")
    print(f"  Polarizations: {', '.join(pols)}")
    print(f"  Total tiles per pol: {n_tiles}")

    # Write to HDF5 with groups per polarization
    with h5py.File(output_path, 'w') as f:
        # Root-level attributes
        f.attrs['source_granule'] = l0b_path
        f.attrs['frequency'] = freq
        f.attrs['polarizations'] = ','.join(pols)
        f.attrs['pulse_start'] = pulse_start
        f.attrs['pulse_end'] = pulse_end
        f.attrs['range_start'] = range_start
        f.attrs['range_end'] = range_end
        f.attrs['cpi_len'] = cpi_len
        f.attrs['cpi_width'] = cpi_width
        f.attrs['assumed_clean'] = True

        # Create a group for each polarization
        for pol in pols:
            grp = f.create_group(f'{freq}_{pol}')
            grp.attrs['frequency'] = freq
            grp.attrs['polarization'] = pol
            grp.attrs['description'] = (
                f'Clean tile indices for {freq}-{pol}. '
                'Assumes entire region is RFI-free.'
            )

            grp.create_dataset('pulse_idx', data=pulse_tiles, dtype=np.int32)
            grp.create_dataset('range_idx', data=range_tiles, dtype=np.int32)

            # Dummy eigenvalues for shape inference
            grp.create_dataset('eigenvalues',
                              shape=(n_tiles, cpi_len),
                              dtype=np.float32)

    print(f"\nWrote multi-pol clean tiles index to: {output_path}")
    print(f"Groups created: {', '.join([f'{freq}_{pol}' for pol in pols])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('l0b_file', help='NISAR L0B granule path')
    parser.add_argument('--pulse-start', type=int, required=True)
    parser.add_argument('--pulse-end', type=int, required=True)
    parser.add_argument('--range-start', type=int, required=True)
    parser.add_argument('--range-end', type=int, required=True)
    parser.add_argument('--cpi-len', type=int, default=256)
    parser.add_argument('--cpi-width', type=int, default=256)
    parser.add_argument('--freq', default='A', choices=['A', 'B'])
    parser.add_argument('--pols', nargs='+', default=['HH', 'HV'],
                       help='Polarizations to include (default: HH HV)')
    parser.add_argument('--output', '-o', required=True)

    args = parser.parse_args()

    create_multipol_clean_index(
        args.l0b_file,
        args.pulse_start, args.pulse_end,
        args.range_start, args.range_end,
        args.cpi_len, args.cpi_width,
        args.output,
        args.freq, args.pols
    )


if __name__ == '__main__':
    main()
