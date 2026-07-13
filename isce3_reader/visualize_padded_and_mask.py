"""
visualize_padded_and_mask.py

Visualize padded NISAR raw data power and the original subswath mask side
by side, and plot Sample Covariance Matrix (SCM) magnitude comparisons
between an originally-clean CPI and an originally-gappy (now padded) CPI.

Usage:
    python visualize_padded_and_mask.py nisar_padded_A_HV.h5 --output-dir ./plots
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Standard CPI dimensions: 16 pulses x 250 range samples
CPI_LEN_DEFAULT = 16
CPI_WIDTH_DEFAULT = 250

# Fixed color scale for SCM magnitude plots, in dB
SCM_DB_VMIN = 0.0
SCM_DB_VMAX = 60.0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize padded NISAR raw data power, subswath mask, and SCM matrices',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('input_file', help='Input HDF5 file containing padded raw data')
    parser.add_argument('--output-dir', type=str, default='./plots',
                        help='Output directory for plots (default: ./plots)')
    parser.add_argument('--vmin', type=float, default=None,
                        help='Minimum power value in dB for raw power color scale')
    parser.add_argument('--vmax', type=float, default=None,
                        help='Maximum power value in dB for raw power color scale')
    parser.add_argument('--pulse-start', type=int, default=None,
                        help='Start pulse index for subset visualization')
    parser.add_argument('--pulse-end', type=int, default=None,
                        help='End pulse index for subset visualization')
    parser.add_argument('--range-start', type=int, default=None,
                        help='Start range sample index for subset visualization')
    parser.add_argument('--range-end', type=int, default=None,
                        help='End range sample index for subset visualization')
    parser.add_argument('--overlay-alpha', type=float, default=0.6,
                        help='Opacity of black overlay on originally-invalid regions (default: 0.6)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for saved figures (default: 150)')
    parser.add_argument('--show', action='store_true',
                        help='Display plots interactively after saving')
    parser.add_argument('--cpi-len', type=int, default=CPI_LEN_DEFAULT,
                        help=f'CPI length (pulses) for SCM visualization (default: {CPI_LEN_DEFAULT})')
    parser.add_argument('--cpi-width', type=int, default=CPI_WIDTH_DEFAULT,
                        help=f'CPI width (range samples) for SCM visualization (default: {CPI_WIDTH_DEFAULT})')

    return parser.parse_args()


def load_data(input_file):
    """
    Load padded raw data and the original subswath mask from HDF5 file.

    Parameters
    ----------
    input_file : str
        Path to input HDF5 file

    Returns
    -------
    raw_data : np.ndarray
        Complex padded raw data
    mask : np.ndarray or None
        Original (pre-padding) subswath mask if available
    metadata : dict
        Metadata from file
    """
    with h5py.File(input_file, 'r') as f:
        if 'raw_data' not in f:
            raise ValueError(f"No 'raw_data' dataset found in {input_file}")

        raw_data = f['raw_data'][:]

        mask = None
        if 'subswath_mask/mask' in f:
            mask = f['subswath_mask/mask'][:]

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


def plot_padded_and_mask(raw_data, mask, metadata, output_file, vmin=None, vmax=None,
                          dpi=150, show=False, overlay_alpha=0.6):
    """
    Plot padded raw data power and the original subswath mask side by side.

    Parameters
    ----------
    raw_data : np.ndarray
        Complex padded raw data (invalid samples already filled)
    mask : np.ndarray or None
        Original subswath mask, before padding
    metadata : dict
        Metadata
    output_file : str
        Output file path
    vmin, vmax : float, optional
        Color scale limits in dB for the raw power panels
    dpi : int
        DPI for saved figure
    show : bool
        Whether to display the plot interactively
    overlay_alpha : float
        Opacity of the black overlay marking originally-invalid regions in
        the third panel (0 = fully transparent, 1 = solid black)
    """
    power_db = compute_power_db(raw_data)

    if mask is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        ax_power, ax_mask, ax_overlay = axes
    else:
        fig, ax_power = plt.subplots(1, 1, figsize=(8, 6))

    if vmin is None:
        vmin = np.percentile(power_db[np.isfinite(power_db)], 1)
    if vmax is None:
        vmax = np.percentile(power_db[np.isfinite(power_db)], 99)

    # Panel 1: padded raw power (fully filled, no gaps)
    im_power = ax_power.imshow(power_db, aspect='auto', cmap='viridis',
                                vmin=vmin, vmax=vmax, origin='lower')
    ax_power.set_xlabel('Range Sample')
    ax_power.set_ylabel('Pulse (Slow Time)')
    ax_power.set_title('Padded Data Power (dB)')
    plt.colorbar(im_power, ax=ax_power, label='Power (dB)')

    if mask is not None:
        # Panel 2: original subswath mask (before padding)
        im_mask = ax_mask.imshow(mask, aspect='auto', cmap='gray',
                                  vmin=0, vmax=1, origin='lower')
        ax_mask.set_xlabel('Range Sample')
        ax_mask.set_ylabel('Pulse (Slow Time)')
        ax_mask.set_title('Original Subswath Mask (Pre-Padding)')
        plt.colorbar(im_mask, ax=ax_mask, label='Valid (1) / Gap (0)')

        # Panel 3: padded power with a transparent overlay marking regions
        # that were originally invalid and are now filled.
        im_overlay = ax_overlay.imshow(power_db, aspect='auto', cmap='viridis',
                                        vmin=vmin, vmax=vmax, origin='lower')

        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[~mask] = [0.0, 0.0, 0.0, overlay_alpha]
        ax_overlay.imshow(overlay, aspect='auto', origin='lower',
                           interpolation='nearest')

        ax_overlay.set_xlabel('Range Sample')
        ax_overlay.set_ylabel('Pulse (Slow Time)')
        ax_overlay.set_title('Padded Power with Originally-Invalid Regions Overlaid')
        plt.colorbar(im_overlay, ax=ax_overlay, label='Power (dB)')

    freq = metadata.get('frequency', 'N/A')
    pol = metadata.get('polarization', 'N/A')
    shape = raw_data.shape
    num_filled = metadata.get('num_samples_filled', None)
    title = f'NISAR Padded Data: Freq {freq}, Pol {pol}, Shape {shape}'
    if num_filled is not None:
        title += f', Samples Filled: {num_filled}'
    fig.suptitle(title, fontsize=14, y=0.98)

    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    if show:
        plt.show()
    plt.close()


def find_cpi_with_dropouts(mask, cpi_len=CPI_LEN_DEFAULT, min_dropout_pct=5.0, max_dropout_pct=50.0):
    """
    Find a CPI block that originally had dropouts (invalid samples), based
    on the pre-padding subswath mask.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask where True = originally valid, False = originally a
        dropout/gap that has since been padded
    cpi_len : int
        CPI length
    min_dropout_pct : float
        Minimum percentage of dropout samples to consider interesting
    max_dropout_pct : float
        Maximum percentage of dropout samples (avoid mostly invalid blocks)

    Returns
    -------
    cpi_idx : int or None
        Index of a CPI that originally had dropouts, or None if none found
    dropout_pct : float
        Percentage of originally-dropout samples in the returned CPI
    """
    num_pulses = mask.shape[0]
    num_cpi = num_pulses // cpi_len

    candidates = []

    for cpi_idx in range(num_cpi):
        cpi_start = cpi_idx * cpi_len
        cpi_mask = mask[cpi_start:cpi_start + cpi_len, :]

        valid_frac = cpi_mask.sum() / cpi_mask.size
        dropout_pct = 100 * (1 - valid_frac)

        if min_dropout_pct <= dropout_pct <= max_dropout_pct:
            candidates.append((cpi_idx, dropout_pct))

    if candidates:
        target_pct = (min_dropout_pct + max_dropout_pct) / 2
        best = min(candidates, key=lambda x: abs(x[1] - target_pct))
        return best[0], best[1]

    return None, 0.0


def plot_scm_comparison(raw_data, mask, metadata, output_file,
                         cpi_len=CPI_LEN_DEFAULT, cpi_width=CPI_WIDTH_DEFAULT,
                         dpi=150, show=False):
    """
    Plot SCM magnitude comparison: one originally-clean CPI and one
    originally-gappy (now padded) CPI, each alongside its eigenvalue
    spectrum. Since the data has already been padded, the standard sample
    covariance matrix is used directly (no gap-exclusion weighting is
    needed). The SCM magnitude color scale is fixed to 0-60 dB so that
    different CPIs and files can be compared directly.

    Parameters
    ----------
    raw_data : np.ndarray
        Complex padded raw data
    mask : np.ndarray or None
        Original (pre-padding) subswath mask, used only to pick
        representative CPIs
    metadata : dict
        Metadata
    output_file : str
        Output file path
    cpi_len : int
        CPI length (number of pulses), default=16
    cpi_width : int
        CPI width (number of range samples), default=250
    dpi : int
        DPI for saved figure
    show : bool
        Whether to display the plot interactively
    """
    num_pulses, num_range_full = raw_data.shape
    num_cpi = num_pulses // cpi_len

    num_range = min(cpi_width, num_range_full)

    pulse_offset = metadata.get('pulse_start', 0)
    range_offset = metadata.get('range_start', 0)

    # Find a CPI that was originally mostly valid (clean)
    clean_cpi_idx = 0
    if mask is not None:
        best_valid_frac = 0
        for i in range(min(num_cpi, 10)):
            cpi_mask = mask[i * cpi_len:(i + 1) * cpi_len, :]
            valid_frac = cpi_mask.sum() / cpi_mask.size
            if valid_frac > best_valid_frac:
                best_valid_frac = valid_frac
                clean_cpi_idx = i

    # Find a CPI that originally had dropouts (now padded)
    dropout_cpi_idx = None
    dropout_pct = 0.0
    if mask is not None:
        dropout_cpi_idx, dropout_pct = find_cpi_with_dropouts(mask, cpi_len)
        if dropout_cpi_idx is None:
            print("  No CPI found with significant original dropouts. Plotting only clean CPI.")
            dropout_cpi_idx = clean_cpi_idx

    def compute_scm_and_eigvals(cpi_idx):
        """Standard sample covariance matrix and its eigenvalues for one CPI."""
        cpi_start = cpi_idx * cpi_len
        cpi_data = raw_data[cpi_start:cpi_start + cpi_len, :num_range].copy()

        K = cpi_data.shape[1]
        SCM = (cpi_data @ cpi_data.conj().T) / K

        eigvals = np.linalg.eigvalsh(SCM)
        return SCM, eigvals[::-1]  # Descending order

    _, clean_eigvals = compute_scm_and_eigvals(clean_cpi_idx)
    _, dropout_eigvals = compute_scm_and_eigvals(dropout_cpi_idx) if dropout_cpi_idx is not None else (None, clean_eigvals)

    clean_eigvals_db = 10 * np.log10(np.maximum(clean_eigvals, 1e-12))
    dropout_eigvals_db = 10 * np.log10(np.maximum(dropout_eigvals, 1e-12))
    eigval_ymin = min(clean_eigvals_db.min(), dropout_eigvals_db.min())
    eigval_ymax = max(clean_eigvals_db.max(), dropout_eigvals_db.max())
    eigval_yrange = eigval_ymax - eigval_ymin
    eigval_ylim = (eigval_ymin - 0.05 * eigval_yrange, eigval_ymax + 0.05 * eigval_yrange)

    # Create figure: 2 rows (clean vs padded), 2 columns (SCM magnitude, eigenvalues)
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    def plot_scm_row(row_idx, cpi_idx, row_label):
        """Helper to plot one row of SCM visualizations."""
        cpi_start = cpi_idx * cpi_len
        cpi_data = raw_data[cpi_start:cpi_start + cpi_len, :num_range].copy()

        valid_pct = 100.0
        if mask is not None:
            cpi_mask = mask[cpi_start:cpi_start + cpi_len, :num_range]
            valid_pct = 100 * cpi_mask.sum() / cpi_mask.size

        SCM, eigvals_sorted = compute_scm_and_eigvals(cpi_idx)

        global_pulse = pulse_offset + cpi_start
        global_range = range_offset
        cpi_dims = f"{cpi_len}x{num_range}"

        # Plot 1: SCM magnitude, fixed color scale (0-60 dB)
        ax1 = fig.add_subplot(gs[row_idx, 0])
        scm_db = 10 * np.log10(np.abs(SCM) + 1e-12)
        im1 = ax1.imshow(scm_db, aspect='auto', cmap='viridis',
                          vmin=SCM_DB_VMIN, vmax=SCM_DB_VMAX, origin='lower')
        ax1.set_xlabel('Pulse Index')
        ax1.set_ylabel('Pulse Index')
        ax1.set_title(f'{row_label}\nSCM Magnitude (dB) | [{global_pulse}, {global_range}] {cpi_dims} | Originally Valid: {valid_pct:.1f}%')
        plt.colorbar(im1, ax=ax1, label='Magnitude (dB)')

        # Plot 2: Eigenvalue spectrum
        ax2 = fig.add_subplot(gs[row_idx, 1])
        eigvals_db = 10 * np.log10(np.maximum(eigvals_sorted, 1e-12))
        ax2.plot(eigvals_db, 'o-', linewidth=2, markersize=6)
        ax2.set_xlabel('Eigenvalue Index')
        ax2.set_ylabel('Eigenvalue (dB)')
        ax2.set_title(f'{row_label}\nEigenvalue Spectrum | [{global_pulse}, {global_range}] {cpi_dims}')
        ax2.set_ylim(eigval_ylim)
        ax2.grid(True, alpha=0.3)

        max_ev = eigvals_sorted[0]
        min_ev = eigvals_sorted[-1]
        cond_num_db = 10 * np.log10(max_ev / max(min_ev, 1e-12))
        ax2.text(0.05, 0.95, f'Condition #: {cond_num_db:.1f} dB',
                  transform=ax2.transAxes, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plot_scm_row(0, clean_cpi_idx, 'Originally Clean CPI')

    if dropout_cpi_idx is not None:
        label = f'Padded CPI (Originally {dropout_pct:.1f}% Invalid)' if dropout_pct > 0 else 'Reference CPI'
        plot_scm_row(1, dropout_cpi_idx, label)

    freq = metadata.get('frequency', 'N/A')
    pol = metadata.get('polarization', 'N/A')
    fig.suptitle(f'SCM Comparison (Padded Data): Freq {freq}, Pol {pol} (CPI length={cpi_len})',
                 fontsize=14, y=0.995)

    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"Saved SCM comparison plot to {output_file}")

    clean_global_pulse = pulse_offset + clean_cpi_idx * cpi_len
    print("  Originally Clean CPI:")
    print(f"    Index: {clean_cpi_idx}")
    print(f"    Global coords: [{clean_global_pulse}, {range_offset}]")
    print(f"    Dimensions: {cpi_len}x{num_range}")

    if dropout_cpi_idx is not None:
        dropout_global_pulse = pulse_offset + dropout_cpi_idx * cpi_len
        if dropout_pct > 0:
            print("  Padded CPI:")
            print(f"    Index: {dropout_cpi_idx}")
            print(f"    Global coords: [{dropout_global_pulse}, {range_offset}]")
            print(f"    Dimensions: {cpi_len}x{num_range}")
            print(f"    Originally invalid: {dropout_pct:.1f}%")
        else:
            print("  Reference CPI:")
            print(f"    Index: {dropout_cpi_idx}")
            print(f"    Global coords: [{dropout_global_pulse}, {range_offset}]")

    if show:
        plt.show()
    plt.close()


def plot_power_histogram(raw_data, mask, metadata, output_file, dpi=150, show=False):
    """
    Plot histogram of power values for the padded data.

    Parameters
    ----------
    raw_data : np.ndarray
        Complex padded raw data
    mask : np.ndarray or None
        Original subswath mask, before padding
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

    ax.hist(power_db_flat, bins=100, alpha=0.5, label='All Data (Padded)', density=True)

    if mask is not None:
        power_originally_valid = power_db[mask]
        power_originally_valid_flat = power_originally_valid[np.isfinite(power_originally_valid)].flatten()
        ax.hist(power_originally_valid_flat, bins=100, alpha=0.5, label='Originally Valid', density=True)

    ax.set_xlabel('Power (dB)')
    ax.set_ylabel('Density')
    ax.set_title('Power Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    freq = metadata.get('frequency', 'N/A')
    pol = metadata.get('polarization', 'N/A')
    fig.suptitle(f'NISAR Padded Data: Freq {freq}, Pol {pol}', fontsize=12, y=0.98)

    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"Saved histogram to {output_file}")
    if show:
        plt.show()
    plt.close()


def main():
    """Main execution function."""
    args = parse_args()

    input_file = Path(args.input_file)
    if not input_file.exists():
        print(f"ERROR: Input file not found: {args.input_file}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.input_file}...")
    raw_data, mask, metadata = load_data(str(input_file))

    print(f"  Padded data shape: {raw_data.shape}")
    if mask is not None:
        print(f"  Original mask shape: {mask.shape}")
        valid_fraction = mask.sum() / mask.size
        print(f"  Originally valid data fraction: {valid_fraction:.1%}")
    else:
        print("  No subswath mask found in file")

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

    freq = metadata.get('frequency', 'A')
    pol = metadata.get('polarization', 'HH')
    output_file = output_dir / f'padded_power_and_mask_{freq}_{pol}.png'
    hist_file = output_dir / f'padded_power_histogram_{freq}_{pol}.png'
    scm_file = output_dir / f'padded_scm_comparison_{freq}_{pol}.png'

    print("\nCreating visualizations...")
    plot_padded_and_mask(raw_data, mask, metadata, str(output_file),
                         vmin=args.vmin, vmax=args.vmax, dpi=args.dpi, show=args.show,
                         overlay_alpha=args.overlay_alpha)
    plot_power_histogram(raw_data, mask, metadata, str(hist_file), dpi=args.dpi, show=args.show)

    print(f"\nCreating SCM comparison visualization (CPI size: {args.cpi_len}x{args.cpi_width})...")
    plot_scm_comparison(raw_data, mask, metadata, str(scm_file),
                        cpi_len=args.cpi_len, cpi_width=args.cpi_width,
                        dpi=args.dpi, show=args.show)

    print(f"\nDone! Plots saved to {output_dir}")


if __name__ == '__main__':
    main()
