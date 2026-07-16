#!/usr/bin/env python
"""
filter_clean_mountains.py

Filter clean_mountains.h5 to remove profiles where max power is below 4 dB.

Reads a clean_mountains.h5 file produced by select_clean_mountain.py and
filters out any CPI profiles where the maximum eigenvalue power is below
a specified threshold (default: 4 dB). This removes profiles with too low
signal power that likely have insufficient data.

The filtered results are saved to a new HDF5 file with the same structure.
"""

import argparse
import os

import numpy as np
import h5py

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

EPS = 1e-12

# ---------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------------------------

def eigvals_to_db(eigvals: np.ndarray) -> np.ndarray:
    """Convert eigenvalues to dB (un-normalized power in dB)."""
    eigvals_db = 10.0 * np.log10(np.clip(eigvals, EPS, None))
    return eigvals_db.astype(np.float32)


def filter_by_max_power(
    eig_lin: np.ndarray,
    diag_lin: np.ndarray,
    diag_valid_frac: np.ndarray,
    pulse_idx: np.ndarray,
    range_idx: np.ndarray,
    iqr_mean: np.ndarray,
    iqr_std: np.ndarray,
    n_std_above: np.ndarray,
    min_power_db: float = 4.0,
    min_valid_eigvals: int = 12
):
    """
    Filter clean mountain profiles by maximum eigenvalue power and number of valid eigenvalues.

    Parameters
    ----------
    eig_lin : (N, 16) float array
        Eigenvalues in linear scale.
    diag_lin : (N, 16) float array
        SCM diagonal in linear scale.
    diag_valid_frac : (N,) float array
        Fraction of valid diagonal entries per CPI.
    pulse_idx : (N,) int array
        Pulse start index per CPI.
    range_idx : (N,) int array
        Range start index per CPI.
    iqr_mean : (N,) float array
        IQR mean per CPI.
    iqr_std : (N,) float array
        IQR std dev per CPI.
    n_std_above : (N,) float array
        Number of std devs above mean per CPI.
    min_power_db : float, default 4.0
        Minimum max eigenvalue power in dB to keep.
    min_valid_eigvals : int, default 12
        Minimum number of eigenvalues that must be > 0 dB (valid, above noise floor).

    Returns
    -------
    filtered_data : dict
        Dictionary with filtered arrays, or None if all profiles filtered out.
    n_removed_power : int
        Number of profiles removed due to low max power.
    n_removed_eigvals : int
        Number of profiles removed due to insufficient valid eigenvalues.
    """
    # Convert eigenvalues to dB to check max power
    eig_db = eigvals_to_db(eig_lin)
    max_power_db = np.max(eig_db, axis=1)  # Max power per CPI (across all 16 eigenvalues)

    # Count number of valid eigenvalues per CPI (> 0 dB, not just > 0 in linear scale)
    # This filters out noise-floor eigenvalues that are technically positive but negligible
    n_valid_eigvals = np.sum(eig_db > 0, axis=1)  # (N,)

    # Filter 1: keep only profiles where max power >= min_power_db
    power_mask = max_power_db >= min_power_db

    # Filter 2: keep only profiles where at least min_valid_eigvals are > 0
    eigval_mask = n_valid_eigvals >= min_valid_eigvals

    # Combined filter: both conditions must be satisfied
    keep_mask = power_mask & eigval_mask

    n_removed_power = np.sum(~power_mask & eigval_mask)  # Failed power check but passed eigval check
    n_removed_eigvals = np.sum(power_mask & ~eigval_mask)  # Failed eigval check but passed power check
    n_removed_both = np.sum(~power_mask & ~eigval_mask)  # Failed both checks
    n_kept = np.sum(keep_mask)

    if n_kept == 0:
        return None, n_removed_power, n_removed_eigvals, n_removed_both

    # Filter all arrays
    filtered_data = {
        'eig_lin': eig_lin[keep_mask],
        'diag_lin': diag_lin[keep_mask],
        'diag_valid_frac': diag_valid_frac[keep_mask],
        'pulse_idx': pulse_idx[keep_mask],
        'range_idx': range_idx[keep_mask],
        'iqr_mean': iqr_mean[keep_mask],
        'iqr_std': iqr_std[keep_mask],
        'n_std_above': n_std_above[keep_mask],
        'max_power_db': max_power_db[keep_mask],
        'n_valid_eigvals': n_valid_eigvals[keep_mask],
    }

    return filtered_data, n_removed_power, n_removed_eigvals, n_removed_both


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('input_h5', help='Input clean_mountains.h5 file')
    parser.add_argument('--min-power-db', type=float, default=4.0,
                        help='Minimum max eigenvalue power in dB to keep (default: 4.0).')
    parser.add_argument('--min-valid-eigvals', type=int, default=12,
                        help='Minimum number of eigenvalues that must be > 0 dB (default: 12).')
    parser.add_argument('--output-h5', default=None,
                        help='Output HDF5 file path. Default: <input>_filtered.h5')
    args = parser.parse_args()

    # Determine output path
    if args.output_h5 is None:
        base, ext = os.path.splitext(args.input_h5)
        args.output_h5 = "{}_filtered{}".format(base, ext)

    print("[+] Reading clean mountain profiles from: {}".format(args.input_h5))
    print("[+] Filtering profiles with:")
    print("    - max power < {:.1f} dB".format(args.min_power_db))
    print("    - fewer than {} valid eigenvalues (> 0 dB)".format(args.min_valid_eigvals))

    h5_in = h5py.File(args.input_h5, 'r')

    # Prepare output HDF5
    os.makedirs(os.path.dirname(args.output_h5) or '.', exist_ok=True)
    h5_out = h5py.File(args.output_h5, 'w')

    # Copy global attributes
    for key, val in h5_in.attrs.items():
        h5_out.attrs[key] = val
    h5_out.attrs['min_power_db'] = args.min_power_db
    h5_out.attrs['min_valid_eigvals'] = args.min_valid_eigvals
    h5_out.attrs['filtered_from'] = os.path.basename(args.input_h5)

    total_removed = 0
    total_kept = 0

    # Iterate over all groups (freq_X_pol_Y)
    for grp_name in h5_in.keys():
        print("\n[+] Processing group: {}".format(grp_name))
        grp_in = h5_in[grp_name]

        # Read attributes
        freq = grp_in.attrs['frequency']
        pol = grp_in.attrs['polarization']
        n_clean = grp_in.attrs['n_clean_tiles']
        print("    freq={}, pol={}, n_clean_tiles={}".format(freq, pol, n_clean))

        # Read datasets
        eig_lin = grp_in['eigenvalues'][:]
        diag_lin = grp_in['diagonal'][:]
        diag_valid_frac = grp_in['diag_valid_frac'][:]
        pulse_idx = grp_in['pulse_idx'][:]
        range_idx = grp_in['range_idx'][:]
        iqr_mean = grp_in['iqr_mean'][:]
        iqr_std = grp_in['iqr_std'][:]
        n_std_above = grp_in['n_std_above'][:]

        # Filter by max power and valid eigenvalues
        filtered_data, n_removed_power, n_removed_eigvals, n_removed_both = filter_by_max_power(
            eig_lin, diag_lin, diag_valid_frac, pulse_idx, range_idx,
            iqr_mean, iqr_std, n_std_above,
            min_power_db=args.min_power_db,
            min_valid_eigvals=args.min_valid_eigvals
        )

        if filtered_data is None:
            print("    [warn] All profiles removed by filters")
            continue

        n_kept = filtered_data['eig_lin'].shape[0]
        n_removed_total = n_removed_power + n_removed_eigvals + n_removed_both
        total_removed += n_removed_total
        total_kept += n_kept

        print("    Removed: {} profiles total".format(n_removed_total))
        print("      - {} due to max power < {:.1f} dB".format(n_removed_power, args.min_power_db))
        print("      - {} due to < {} valid eigenvalues".format(n_removed_eigvals, args.min_valid_eigvals))
        print("      - {} due to both conditions".format(n_removed_both))
        print("    Kept: {} profiles".format(n_kept))

        # Show statistics for kept profiles
        max_power_db = filtered_data['max_power_db']
        n_valid_eigvals = filtered_data['n_valid_eigvals']
        print("    Max power (dB) stats for kept profiles:")
        print("      min: {:.2f}, max: {:.2f}, mean: {:.2f}, median: {:.2f}".format(
            np.min(max_power_db), np.max(max_power_db),
            np.mean(max_power_db), np.median(max_power_db)))
        print("    Valid eigenvalues stats for kept profiles:")
        print("      min: {}, max: {}, mean: {:.1f}, median: {}".format(
            np.min(n_valid_eigvals), np.max(n_valid_eigvals),
            np.mean(n_valid_eigvals), int(np.median(n_valid_eigvals))))

        # Save to output HDF5
        grp_out = h5_out.create_group(grp_name)
        grp_out.create_dataset('eigenvalues', data=filtered_data['eig_lin'], compression='gzip')
        grp_out.create_dataset('diagonal', data=filtered_data['diag_lin'], compression='gzip')
        grp_out.create_dataset('diag_valid_frac', data=filtered_data['diag_valid_frac'], compression='gzip')
        grp_out.create_dataset('pulse_idx', data=filtered_data['pulse_idx'], compression='gzip')
        grp_out.create_dataset('range_idx', data=filtered_data['range_idx'], compression='gzip')
        grp_out.create_dataset('iqr_mean', data=filtered_data['iqr_mean'], compression='gzip')
        grp_out.create_dataset('iqr_std', data=filtered_data['iqr_std'], compression='gzip')
        grp_out.create_dataset('n_std_above', data=filtered_data['n_std_above'], compression='gzip')

        # Copy attributes and update n_clean_tiles
        grp_out.attrs['frequency'] = freq
        grp_out.attrs['polarization'] = pol
        grp_out.attrs['n_clean_tiles'] = n_kept
        grp_out.attrs['n_removed_by_power'] = n_removed_power
        grp_out.attrs['n_removed_by_eigvals'] = n_removed_eigvals
        grp_out.attrs['n_removed_by_both'] = n_removed_both
        grp_out.attrs['original_n_clean_tiles'] = n_clean

        print("    Saved to HDF5 group: {}".format(grp_name))

    h5_in.close()
    h5_out.close()

    print("\n" + "="*60)
    print("[+] Done.")
    print("    Total profiles removed: {}".format(total_removed))
    print("    Total profiles kept: {}".format(total_kept))
    print("    Filtered clean mountains saved to: {}".format(args.output_h5))


if __name__ == '__main__':
    main()
