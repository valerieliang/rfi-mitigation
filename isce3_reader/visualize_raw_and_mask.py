"""
visualize_raw_and_mask.py

Visualize raw NISAR data power and subswath mask side by side.

Usage:
    python visualize_raw_and_mask.py nisar_A_HV_raw.h5 --output-dir ./plots
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize NISAR raw data power and subswath mask',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('input_file', help='Input HDF5 file containing raw data')
    parser.add_argument('--output-dir', type=str, default='./plots',
                        help='Output directory for plots (default: ./plots)')
    parser.add_argument('--vmin', type=float, default=None,
                        help='Minimum power value in dB for color scale')
    parser.add_argument('--vmax', type=float, default=None,
                        help='Maximum power value in dB for color scale')
    parser.add_argument('--pulse-start', type=int, default=None,
                        help='Start pulse index for subset visualization')
    parser.add_argument('--pulse-end', type=int, default=None,
                        help='End pulse index for subset visualization')
    parser.add_argument('--range-start', type=int, default=None,
                        help='Start range sample index for subset visualization')
    parser.add_argument('--range-end', type=int, default=None,
                        help='End range sample index for subset visualization')
    parser.add_argument('--overlay-alpha', type=float, default=0.6,
                        help='Opacity of black overlay on invalid regions (default: 0.6)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for saved figures (default: 150)')
    parser.add_argument('--show', action='store_true',
                        help='Display plots interactively after saving')

    return parser.parse_args()


def load_data(input_file):
    """
    Load raw data and subswath mask from HDF5 file.

    Parameters
    ----------
    input_file : str
        Path to input HDF5 file

    Returns
    -------
    raw_data : np.ndarray
        Complex raw data
    mask : np.ndarray or None
        Subswath mask if available
    metadata : dict
        Metadata from file
    """
    with h5py.File(input_file, 'r') as f:
        # Load raw data
        if 'raw_data' not in f:
            raise ValueError(f"No 'raw_data' dataset found in {input_file}")

        raw_data = f['raw_data'][:]

        # Load mask if available
        mask = None
        if 'subswath_mask/mask' in f:
            mask = f['subswath_mask/mask'][:]

        # Load metadata
        metadata = {}
        if 'metadata' in f:
            meta_grp = f['metadata']
            for key in meta_grp.attrs:
                metadata[key] = meta_grp.attrs[key]

    return raw_data, mask, metadata


def compute_power_db(data):
    """
    Compute power in dB: 20*log10(abs(data)).

    Parameters
    ----------
    data : np.ndarray
        Complex data

    Returns
    -------
    power_db : np.ndarray
        Power in dB
    """
    power_db = 20 * np.log10(np.abs(data) + 1e-12)  # Add small epsilon to avoid log(0)
    return power_db


def plot_raw_and_mask(raw_data, mask, metadata, output_file, vmin=None, vmax=None,
                      dpi=150, show=False, overlay_alpha=0.6):
    """
    Plot raw data power and subswath mask side by side.

    Parameters
    ----------
    raw_data : np.ndarray
        Complex raw data
    mask : np.ndarray or None
        Subswath mask
    metadata : dict
        Metadata
    output_file : str
        Output file path
    vmin, vmax : float, optional
        Color scale limits in dB
    dpi : int
        DPI for saved figure
    show : bool
        Whether to display the plot interactively
    overlay_alpha : float
        Opacity of the black overlay on invalid regions in the third
        panel (0 = fully transparent, 1 = solid black)
    """
    # Compute power in dB
    power_db = compute_power_db(raw_data)

    # Determine figure layout
    if mask is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        ax_power, ax_mask, ax_masked = axes
    else:
        fig, ax_power = plt.subplots(1, 1, figsize=(8, 6))

    # Determine color scale
    if vmin is None:
        vmin = np.percentile(power_db[np.isfinite(power_db)], 1)
    if vmax is None:
        vmax = np.percentile(power_db[np.isfinite(power_db)], 99)

    # Plot raw power
    im_power = ax_power.imshow(power_db, aspect='auto', cmap='viridis',
                                vmin=vmin, vmax=vmax, origin='lower')
    ax_power.set_xlabel('Range Sample')
    ax_power.set_ylabel('Pulse (Slow Time)')
    ax_power.set_title('Raw Data Power (dB)')
    plt.colorbar(im_power, ax=ax_power, label='Power (dB)')

    if mask is not None:
        # Plot subswath mask
        im_mask = ax_mask.imshow(mask, aspect='auto', cmap='gray',
                                 vmin=0, vmax=1, origin='lower')
        ax_mask.set_xlabel('Range Sample')
        ax_mask.set_ylabel('Pulse (Slow Time)')
        ax_mask.set_title('Subswath Mask')
        plt.colorbar(im_mask, ax=ax_mask, label='Valid (1) / Gap (0)')

        # Plot raw power with a transparent overlay marking invalid regions.
        # Invalid samples are covered with semi-transparent black; valid
        # samples are fully transparent so the raw power shows through.
        im_masked = ax_masked.imshow(power_db, aspect='auto', cmap='viridis',
                                     vmin=vmin, vmax=vmax, origin='lower')

        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[~mask] = [0.0, 0.0, 0.0, overlay_alpha]
        ax_masked.imshow(overlay, aspect='auto', origin='lower',
                         interpolation='nearest')

        ax_masked.set_xlabel('Range Sample')
        ax_masked.set_ylabel('Pulse (Slow Time)')
        ax_masked.set_title('Raw Power with Invalid Regions Overlaid')
        plt.colorbar(im_masked, ax=ax_masked, label='Power (dB)')

    # Add metadata as title
    freq = metadata.get('frequency', 'N/A')
    pol = metadata.get('polarization', 'N/A')
    shape = raw_data.shape
    fig.suptitle(f'NISAR Data: Freq {freq}, Pol {pol}, Shape {shape}', fontsize=14, y=0.98)

    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    if show:
        plt.show()
    plt.close()


def plot_power_histogram(raw_data, mask, metadata, output_file, dpi=150, show=False):
    """
    Plot histogram of power values.

    Parameters
    ----------
    raw_data : np.ndarray
        Complex raw data
    mask : np.ndarray or None
        Subswath mask
    metadata : dict
        Metadata
    output_file : str
        Output file path
    dpi : int
        DPI for saved figure
    show : bool
        Whether to display the plot interactively
    """
    power_db = compute_power_db(raw_data)
    power_db_flat = power_db[np.isfinite(power_db)].flatten()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram for all data
    ax.hist(power_db_flat, bins=100, alpha=0.5, label='All Data', density=True)

    # Plot histogram for valid data if mask is available
    if mask is not None:
        power_valid = power_db[mask]
        power_valid_flat = power_valid[np.isfinite(power_valid)].flatten()
        ax.hist(power_valid_flat, bins=100, alpha=0.5, label='Valid (Masked)', density=True)

    ax.set_xlabel('Power (dB)')
    ax.set_ylabel('Density')
    ax.set_title('Power Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add metadata
    freq = metadata.get('frequency', 'N/A')
    pol = metadata.get('polarization', 'N/A')
    fig.suptitle(f'NISAR Data: Freq {freq}, Pol {pol}', fontsize=12, y=0.98)

    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"Saved histogram to {output_file}")
    if show:
        plt.show()
    plt.close()


def main():
    """Main execution function."""
    args = parse_args()

    # Validate input file
    input_file = Path(args.input_file)
    if not input_file.exists():
        print(f"ERROR: Input file not found: {args.input_file}")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.input_file}...")
    raw_data, mask, metadata = load_data(str(input_file))

    print(f"  Raw data shape: {raw_data.shape}")
    if mask is not None:
        print(f"  Mask shape: {mask.shape}")
        valid_fraction = mask.sum() / mask.size
        print(f"  Valid data fraction: {valid_fraction:.1%}")
    else:
        print("  No subswath mask found in file")

    # Apply subsetting if requested
    p_start = args.pulse_start if args.pulse_start is not None else 0
    p_end = args.pulse_end if args.pulse_end is not None else raw_data.shape[0]
    r_start = args.range_start if args.range_start is not None else 0
    r_end = args.range_end if args.range_end is not None else raw_data.shape[1]

    if (p_start != 0 or p_end != raw_data.shape[0] or
        r_start != 0 or r_end != raw_data.shape[1]):
        print(f"\nApplying subset: pulses [{p_start}:{p_end}], range [{r_start}:{r_end}]")
        raw_data = raw_data[p_start:p_end, r_start:r_end]
        if mask is not None:
            mask = mask[p_start:p_end, r_start:r_end]
        print(f"  New data shape: {raw_data.shape}")

    # Generate output filename
    freq = metadata.get('frequency', 'A')
    pol = metadata.get('polarization', 'HH')
    output_file = output_dir / f'raw_power_and_mask_{freq}_{pol}.png'
    hist_file = output_dir / f'power_histogram_{freq}_{pol}.png'

    # Create visualizations
    print(f"\nCreating visualizations...")
    plot_raw_and_mask(raw_data, mask, metadata, str(output_file),
                      vmin=args.vmin, vmax=args.vmax, dpi=args.dpi, show=args.show,
                      overlay_alpha=args.overlay_alpha)
    plot_power_histogram(raw_data, mask, metadata, str(hist_file), dpi=args.dpi, show=args.show)

    print(f"\nDone! Plots saved to {output_dir}")


if __name__ == '__main__':
    main()