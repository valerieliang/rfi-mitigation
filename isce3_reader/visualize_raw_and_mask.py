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
import warnings

# Standard CPI dimensions: 16 pulses x 250 range samples
CPI_LEN_DEFAULT = 16
CPI_WIDTH_DEFAULT = 250

# Default gap-exclusion covariance thresholds
# Off-diagonal terms need >= 25% overlapping valid samples
# Diagonal terms need >= 20% valid samples
OFF_DIAG_OVERLAP_RATIO_DEFAULT = 0.25
DIAG_VALID_RATIO_DEFAULT = 0.20


def compute_gap_exclusion_cov(
    data: np.ndarray,
    *,
    mask_valid_cpi: np.ndarray = None,
    off_diag_overlap_ratio: float = OFF_DIAG_OVERLAP_RATIO_DEFAULT,
    diag_valid_ratio: float = DIAG_VALID_RATIO_DEFAULT,
):
    """
    Compute a gap-excluded slow-time sample covariance matrix.

    Refactored from compute_evd_cpi.py so this script is self-contained.

    Parameters
    ----------
    data: (num_pulses, num_rng_samples) complex array
        Slow-time block: K pulses x M range samples.
        Pulses should be contiguous in slow time for ST-EVD.
    mask_valid_cpi: (num_pulses, num_rng_samples) bool array, optional
        True indicates valid samples. False indicates invalid samples or gaps.
        If None, an all true boolean mask is created and all samples are
        assumed to be valid.
    off_diag_overlap_ratio: float, default = 0.25
        Minimum fraction of overlapping valid range samples required to compute
        an off-diagonal term in the sample covariance matrix entry R_ij.
    diag_valid_ratio : float, default = 0.20
        Minimum fraction of valid samples required to compute a diagonal term
        in the sample covariance matrix entry R_ii.

    Returns
    -------
    cov : (num_pulses, num_pulses) complex64
        Gap-excluded sample covariance matrix.
    diag_valid_idx : (num_pulses,) bool array
        True where the diagonal term had enough valid samples.

    Notes
    -----
    The number of range samples per pulse should be equal or larger than
    2 x number of pulses to have a reliable sample covariance matrix estimate.
    """

    num_pulses, num_rng_samples = data.shape

    if mask_valid_cpi is None:
        mask_valid_cpi = np.ones(data.shape, dtype=bool)
    else:
        mask_valid_cpi = mask_valid_cpi.astype(bool, copy=False)

    if mask_valid_cpi.shape != data.shape:
        raise ValueError(f"CPI mask shape {mask_valid_cpi.shape} != CPI data shape {data.shape}")

    if not (0.0 < off_diag_overlap_ratio <= 1.0):
        raise ValueError("off_diag_overlap_ratio must be between 0 and 1.")

    if not (0.0 < diag_valid_ratio <= 1.0):
        raise ValueError("diag_valid_ratio must be between 0 and 1.")

    # Minimum samples required to compute diagonal and off-diagonal terms
    min_valid_off_diag = max(1, int(np.ceil(off_diag_overlap_ratio * num_rng_samples)))
    min_valid_diag = max(1, int(np.ceil(diag_valid_ratio * num_rng_samples)))

    # The number of range samples per pulse should be >= 2 x number of pulses
    # for a reliable estimate; only warn to allow exploring the trade-off
    rng_samples_min = 2 * num_pulses

    if min_valid_off_diag < rng_samples_min:
        warnings.warn(f"""
            Minimum number of samples required per pulse to estimate sample covariance matrix
            is {rng_samples_min}. The number of valid overlapping off-diagonal samples is
            {min_valid_off_diag}.
        """)

    if min_valid_diag < rng_samples_min:
        warnings.warn(f"""
            Minimum number of samples required per pulse to estimate sample covariance matrix
            is {rng_samples_min}. The number of valid diagonal samples is {min_valid_diag}.
        """)

    # Zero-out invalid samples
    x_valid = data * mask_valid_cpi

    # Count valid sample overlap count for each element of the covariance matrix
    mask_int = mask_valid_cpi.astype(np.int32)
    overlap_counts = mask_int @ mask_int.T  # shape (pulse x pulse)

    # Sum of conjugate products over overlapping valid samples (unnormalized)
    cov_sum = x_valid @ x_valid.conj().T

    # Initialize gap-excluded sample covariance matrix
    cov = np.zeros((num_pulses, num_pulses), dtype=np.complex64)

    # Diagonal terms
    diag_idx = np.diag_indices(num_pulses)
    diag_counts = overlap_counts[diag_idx]
    diag_cov_sum = cov_sum[diag_idx]

    # Check if there are enough valid samples
    diag_valid_idx = diag_counts >= min_valid_diag

    diag_vals = np.zeros(num_pulses, dtype=np.complex64)
    diag_vals[diag_valid_idx] = diag_cov_sum[diag_valid_idx] / diag_counts[diag_valid_idx]
    cov[diag_idx] = diag_vals

    # Off-diagonal terms: verify there are enough overlapping valid samples
    off_diag_valid = overlap_counts >= min_valid_off_diag
    np.fill_diagonal(off_diag_valid, False)
    cov[off_diag_valid] = cov_sum[off_diag_valid] / overlap_counts[off_diag_valid]

    # Ensure Hermitian numerically
    cov = (0.5 * (cov + cov.conj().T)).astype(np.complex64)

    return cov, diag_valid_idx


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
    parser.add_argument('--plot-scm', action='store_true',
                        help='Also plot Sample Covariance Matrix (SCM) comparison: clean vs dropout CPIs')
    parser.add_argument('--cpi-len', type=int, default=CPI_LEN_DEFAULT,
                        help=f'CPI length (pulses) for SCM visualization (default: {CPI_LEN_DEFAULT})')
    parser.add_argument('--cpi-width', type=int, default=CPI_WIDTH_DEFAULT,
                        help=f'CPI width (range samples) for SCM visualization (default: {CPI_WIDTH_DEFAULT})')
    parser.add_argument('--off-diag-overlap-ratio', type=float, default=OFF_DIAG_OVERLAP_RATIO_DEFAULT,
                        help=f'Minimum overlap ratio for off-diagonal SCM terms when using gap-exclusion (default: {OFF_DIAG_OVERLAP_RATIO_DEFAULT})')
    parser.add_argument('--diag-valid-ratio', type=float, default=DIAG_VALID_RATIO_DEFAULT,
                        help=f'Minimum valid ratio for diagonal SCM terms when using gap-exclusion (default: {DIAG_VALID_RATIO_DEFAULT})')

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


def find_cpi_with_dropouts(mask, cpi_len=CPI_LEN_DEFAULT, min_dropout_pct=5.0, max_dropout_pct=50.0):
    """
    Find a CPI block that has dropouts (invalid samples).

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask where True = valid, False = dropout/gap
    cpi_len : int
        CPI length
    min_dropout_pct : float
        Minimum percentage of dropout samples to consider interesting
    max_dropout_pct : float
        Maximum percentage of dropout samples (avoid mostly invalid blocks)

    Returns
    -------
    cpi_idx : int or None
        Index of a CPI with dropouts, or None if none found
    dropout_pct : float
        Percentage of dropout samples in the returned CPI
    """
    num_pulses = mask.shape[0]
    num_cpi = num_pulses // cpi_len

    candidates = []

    for cpi_idx in range(num_cpi):
        cpi_start = cpi_idx * cpi_len
        cpi_mask = mask[cpi_start:cpi_start + cpi_len, :]

        # Calculate dropout percentage
        valid_frac = cpi_mask.sum() / cpi_mask.size
        dropout_pct = 100 * (1 - valid_frac)

        if min_dropout_pct <= dropout_pct <= max_dropout_pct:
            candidates.append((cpi_idx, dropout_pct))

    if candidates:
        # Return CPI with dropout percentage closest to middle of range
        target_pct = (min_dropout_pct + max_dropout_pct) / 2
        best = min(candidates, key=lambda x: abs(x[1] - target_pct))
        return best[0], best[1]

    return None, 0.0


def plot_scm_comparison(raw_data, mask, metadata, output_file,
                        cpi_len=CPI_LEN_DEFAULT, cpi_width=CPI_WIDTH_DEFAULT,
                        off_diag_overlap_ratio=OFF_DIAG_OVERLAP_RATIO_DEFAULT,
                        diag_valid_ratio=DIAG_VALID_RATIO_DEFAULT, dpi=150, show=False):
    """
    Plot SCM comparison: one clean CPI and one CPI with dropouts.

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
    cpi_len : int
        CPI length (number of pulses), default=16
    cpi_width : int
        CPI width (number of range samples), default=250
    off_diag_overlap_ratio : float
        Minimum overlap ratio for off-diagonal SCM terms (default=0.25)
    diag_valid_ratio : float
        Minimum valid ratio for diagonal SCM terms (default=0.20)
    dpi : int
        DPI for saved figure
    show : bool
        Whether to display the plot interactively
    """
    num_pulses, num_range_full = raw_data.shape
    num_cpi = num_pulses // cpi_len

    # Limit range samples to cpi_width for SCM computation
    num_range = min(cpi_width, num_range_full)

    # Get global offset coordinates from metadata
    pulse_offset = metadata.get('pulse_start', 0)
    range_offset = metadata.get('range_start', 0)

    # Find a clean CPI (high valid fraction)
    clean_cpi_idx = 0
    if mask is not None:
        best_valid_frac = 0
        for i in range(min(num_cpi, 10)):  # Check first 10 CPIs
            cpi_mask = mask[i*cpi_len:(i+1)*cpi_len, :]
            valid_frac = cpi_mask.sum() / cpi_mask.size
            if valid_frac > best_valid_frac:
                best_valid_frac = valid_frac
                clean_cpi_idx = i

    # Find a CPI with dropouts
    dropout_cpi_idx = None
    dropout_pct = 0.0
    if mask is not None:
        dropout_cpi_idx, dropout_pct = find_cpi_with_dropouts(mask, cpi_len)
        if dropout_cpi_idx is None:
            print(f"  No CPI found with significant dropouts. Plotting only clean CPI.")
            dropout_cpi_idx = clean_cpi_idx

    # Pre-compute eigenvalues for both CPIs to get consistent y-axis limits
    def compute_eigenvalues_for_cpi(cpi_idx):
        """Helper to compute eigenvalues for a CPI using gap-exclusion covariance if mask provided."""
        cpi_start = cpi_idx * cpi_len
        cpi_data = raw_data[cpi_start:cpi_start + cpi_len, :num_range].copy()

        if mask is not None:
            cpi_mask = mask[cpi_start:cpi_start + cpi_len, :num_range]
            # Use gap-exclusion covariance computation
            SCM, _ = compute_gap_exclusion_cov(
                cpi_data,
                mask_valid_cpi=cpi_mask,
                off_diag_overlap_ratio=off_diag_overlap_ratio,
                diag_valid_ratio=diag_valid_ratio,
            )
        else:
            # Standard covariance without mask
            K = cpi_data.shape[1]
            SCM = (cpi_data @ cpi_data.conj().T) / K

        eigvals = np.linalg.eigvalsh(SCM)
        return eigvals[::-1]  # Descending order

    # Compute eigenvalues for both CPIs
    clean_eigvals = compute_eigenvalues_for_cpi(clean_cpi_idx)
    dropout_eigvals = compute_eigenvalues_for_cpi(dropout_cpi_idx) if dropout_cpi_idx is not None else clean_eigvals

    # Compute global y-axis limits for eigenvalue plots
    clean_eigvals_db = 10 * np.log10(np.maximum(clean_eigvals, 1e-12))
    dropout_eigvals_db = 10 * np.log10(np.maximum(dropout_eigvals, 1e-12))
    eigval_ymin = min(clean_eigvals_db.min(), dropout_eigvals_db.min())
    eigval_ymax = max(clean_eigvals_db.max(), dropout_eigvals_db.max())
    eigval_yrange = eigval_ymax - eigval_ymin
    eigval_ylim = (eigval_ymin - 0.05 * eigval_yrange, eigval_ymax + 0.05 * eigval_yrange)

    # Create figure: 2 rows (clean vs dropout), 3 columns (SCM mag, SCM phase, eigenvalues)
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    def plot_scm_row(row_idx, cpi_idx, row_label):
        """Helper to plot one row of SCM visualizations."""
        # Extract CPI data (limit to cpi_width range samples)
        cpi_start = cpi_idx * cpi_len
        cpi_data = raw_data[cpi_start:cpi_start + cpi_len, :num_range].copy()

        # Get mask and compute valid percentage
        cpi_mask = None
        valid_pct = 100.0
        if mask is not None:
            cpi_mask = mask[cpi_start:cpi_start + cpi_len, :num_range]
            valid_pct = 100 * cpi_mask.sum() / cpi_mask.size

        # Compute SCM using gap-exclusion method if mask is available
        if mask is not None:
            SCM, _ = compute_gap_exclusion_cov(
                cpi_data,
                mask_valid_cpi=cpi_mask,
                off_diag_overlap_ratio=off_diag_overlap_ratio,
                diag_valid_ratio=diag_valid_ratio,
            )
        else:
            # Standard covariance without mask
            K = cpi_data.shape[1]
            SCM = (cpi_data @ cpi_data.conj().T) / K

        # Compute eigenvalues
        eigvals = np.linalg.eigvalsh(SCM)
        eigvals_sorted = eigvals[::-1]

        # Global coordinates (top-left of CPI block)
        global_pulse = pulse_offset + cpi_start
        global_range = range_offset
        cpi_dims = f"{cpi_len}x{num_range}"

        # Plot 1: SCM magnitude
        ax1 = fig.add_subplot(gs[row_idx, 0])
        scm_db = 10 * np.log10(np.abs(SCM) + 1e-12)
        im1 = ax1.imshow(scm_db, aspect='auto', cmap='viridis', origin='lower')
        ax1.set_xlabel('Pulse Index')
        ax1.set_ylabel('Pulse Index')
        ax1.set_title(f'{row_label}\nSCM Magnitude (dB) | [{global_pulse}, {global_range}] {cpi_dims} | Valid: {valid_pct:.1f}%')
        plt.colorbar(im1, ax=ax1, label='Magnitude (dB)')

        # Plot 2: SCM phase
        ax2 = fig.add_subplot(gs[row_idx, 1])
        scm_phase = np.angle(SCM)
        im2 = ax2.imshow(scm_phase, aspect='auto', cmap='hsv', vmin=-np.pi, vmax=np.pi, origin='lower')
        ax2.set_xlabel('Pulse Index')
        ax2.set_ylabel('Pulse Index')
        ax2.set_title(f'{row_label}\nSCM Phase | [{global_pulse}, {global_range}] {cpi_dims}')
        plt.colorbar(im2, ax=ax2, label='Phase (radians)')

        # Plot 3: Eigenvalue spectrum
        ax3 = fig.add_subplot(gs[row_idx, 2])
        eigvals_db = 10 * np.log10(np.maximum(eigvals_sorted, 1e-12))
        ax3.plot(eigvals_db, 'o-', linewidth=2, markersize=6)
        ax3.set_xlabel('Eigenvalue Index')
        ax3.set_ylabel('Eigenvalue (dB)')
        ax3.set_title(f'{row_label}\nEigenvalue Spectrum | [{global_pulse}, {global_range}] {cpi_dims}')
        ax3.set_ylim(eigval_ylim)  # Use consistent y-axis limits for comparison
        ax3.grid(True, alpha=0.3)

        # Add statistics
        max_ev = eigvals_sorted[0]
        min_ev = eigvals_sorted[-1]
        cond_num_db = 10 * np.log10(max_ev / max(min_ev, 1e-12))
        ax3.text(0.05, 0.95, f'Condition #: {cond_num_db:.1f} dB',
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot clean CPI
    plot_scm_row(0, clean_cpi_idx, 'Clean CPI')

    # Plot dropout CPI
    if dropout_cpi_idx is not None:
        label = f'CPI with Dropouts ({dropout_pct:.1f}%)' if dropout_pct > 0 else 'Reference CPI'
        plot_scm_row(1, dropout_cpi_idx, label)

    # Add overall title
    freq = metadata.get('frequency', 'N/A')
    pol = metadata.get('polarization', 'N/A')
    fig.suptitle(f'SCM Comparison: Freq {freq}, Pol {pol} (CPI length={cpi_len})',
                 fontsize=14, y=0.995)

    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"Saved SCM comparison plot to {output_file}")

    # Clean CPI info
    clean_global_pulse = pulse_offset + clean_cpi_idx * cpi_len
    print(f"  Clean CPI:")
    print(f"    Index: {clean_cpi_idx}")
    print(f"    Global coords: [{clean_global_pulse}, {range_offset}]")
    print(f"    Dimensions: {cpi_len}x{num_range}")
    print(f"    Pulse range: {clean_global_pulse} to {clean_global_pulse + cpi_len - 1}")

    # Dropout CPI info
    if dropout_cpi_idx is not None:
        dropout_global_pulse = pulse_offset + dropout_cpi_idx * cpi_len
        if dropout_pct > 0:
            print(f"  Dropout CPI:")
            print(f"    Index: {dropout_cpi_idx}")
            print(f"    Global coords: [{dropout_global_pulse}, {range_offset}]")
            print(f"    Dimensions: {cpi_len}x{num_range}")
            print(f"    Pulse range: {dropout_global_pulse} to {dropout_global_pulse + cpi_len - 1}")
            print(f"    Dropout: {dropout_pct:.1f}%")
        else:
            print(f"  Reference CPI:")
            print(f"    Index: {dropout_cpi_idx}")
            print(f"    Global coords: [{dropout_global_pulse}, {range_offset}]")

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
    scm_file = output_dir / f'scm_comparison_{freq}_{pol}.png'

    # Create visualizations
    print(f"\nCreating visualizations...")
    plot_raw_and_mask(raw_data, mask, metadata, str(output_file),
                      vmin=args.vmin, vmax=args.vmax, dpi=args.dpi, show=args.show,
                      overlay_alpha=args.overlay_alpha)
    plot_power_histogram(raw_data, mask, metadata, str(hist_file), dpi=args.dpi, show=args.show)

    # Plot SCM comparison if requested
    if args.plot_scm:
        print(f"\nCreating SCM comparison visualization (CPI size: {args.cpi_len}x{args.cpi_width})...")
        if mask is not None:
            print(f"  Using gap-exclusion covariance with:")
            print(f"    Off-diagonal overlap ratio: {args.off_diag_overlap_ratio}")
            print(f"    Diagonal valid ratio: {args.diag_valid_ratio}")
        plot_scm_comparison(raw_data, mask, metadata, str(scm_file),
                           cpi_len=args.cpi_len, cpi_width=args.cpi_width,
                           off_diag_overlap_ratio=args.off_diag_overlap_ratio,
                           diag_valid_ratio=args.diag_valid_ratio,
                           dpi=args.dpi, show=args.show)

    print(f"\nDone! Plots saved to {output_dir}")


if __name__ == '__main__':
    main()