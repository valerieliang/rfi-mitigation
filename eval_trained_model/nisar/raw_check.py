"""
raw_check.py

Sanity check the raw NISAR L0B decoded data before running inference.

Loads a sample of CPI tiles directly from the L0B file, bypassing feature
extraction and the model entirely, and plots:

    Plot 1 -- Eigenvalue profiles (4x4 grid)
        SCM eigenvalue spectra for 16 CPI tiles sampled evenly across
        azimuth at range tile 0. Each subplot shows the dB-sorted profile
        and its dynamic span. A healthy SAR signal should show a smoothly
        descending curve with >5 dB span. A flat profile or near-zero span
        indicates a degenerate SCM (bad decode, zero data, or fill region).

    Plot 2 -- Range power spectra (4x4 grid)
        FFT of the first pulse in each tile. Healthy SAR data shows a
        broadband chirp spectrum. A single DC spike or flat white noise
        floor indicates a decode problem.

Use this to verify:
    - The BFPQLUT decode produced valid IQ (not quantised flat or all-zero)
    - The SCM has meaningful structure before the model ever runs

Usage
-----
    python raw_check.py
    python raw_check.py --l0   nisar_data/raw/NISAR_...h5
    python raw_check.py --out  nisar_data/plots/
    python raw_check.py --n    32   (sample 32 tiles instead of 16)
    python raw_check.py --target block_bottom
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from infer_knee_map import (
    L0B_DATASET, M, BLOCK_WIDTH, TARGETS,
    DEFAULT_L0,
    load_block, _is_valid_tile,
)

DEFAULT_OUT = os.path.join('nisar_data', 'processed')


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------

def plot_eigenvalue_profiles(block, block_name, sample_cis, out_dir, l0_path):
    """
    4xN grid of raw SCM eigenvalue profiles for sampled CPI tiles.
    """
    n_samples = len(sample_cis)
    n_cols    = 4
    n_rows    = int(np.ceil(n_samples / n_cols))
    ev_index  = np.arange(M)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3),
                             sharey=False)
    fig.suptitle(
        f'Raw Eigenvalue Profiles  --  {block_name}  ri=0\n'
        f'{n_samples} CPI tiles sampled across azimuth\n'
        f'L0B: {os.path.basename(l0_path)}',
        fontsize=11,
    )

    for ax, ci in zip(axes.flat, sample_cis):
        p0  = ci * M
        cpi = block[p0:p0 + M, :]
        if not _is_valid_tile(cpi):
            ax.set_title(f'ci={ci}  INVALID (gap)', fontsize=8)
            ax.axis('off')
            continue
        scm     = (cpi @ cpi.conj().T) / cpi.shape[1]
        eigvals = np.linalg.eigvalsh(scm).real[::-1]
        eigvals = np.maximum(eigvals, 1e-12)
        ev_db   = 10.0 * np.log10(eigvals)
        span    = ev_db[0] - ev_db[-1]
        ax.plot(ev_index, ev_db, linewidth=1.2, color='steelblue')
        ax.set_title(f'ci={ci}  span={span:.1f} dB', fontsize=8)
        ax.set_xlabel('EV index', fontsize=7)
        ax.set_ylabel('dB', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, linestyle='--', alpha=0.4)

    for ax in axes.flat[n_samples:]:
        ax.axis('off')

    fig.tight_layout()
    out_path = os.path.join(out_dir, f'{block_name}_raw_eigenvalues.png')
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f'  eigenvalue profiles -> {os.path.basename(out_path)}')


def plot_range_spectra(block, block_name, sample_cis, out_dir, l0_path):
    """
    4xN grid of range FFT power spectra for the first pulse of each sampled tile.
    """
    n_samples = len(sample_cis)
    n_cols    = 4
    n_rows    = int(np.ceil(n_samples / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3),
                             sharey=False)
    fig.suptitle(
        f'Range Power Spectra (first pulse)  --  {block_name}  ri=0\n'
        f'{n_samples} CPI tiles sampled across azimuth\n'
        f'L0B: {os.path.basename(l0_path)}',
        fontsize=11,
    )

    for ax, ci in zip(axes.flat, sample_cis):
        p0  = ci * M
        cpi = block[p0:p0 + M, :]
        if not _is_valid_tile(cpi):
            ax.set_title(f'ci={ci}  INVALID (gap)', fontsize=8)
            ax.axis('off')
            continue
        pulse = cpi[0, :]
        spec  = np.abs(np.fft.fftshift(np.fft.fft(pulse))) ** 2
        spec  = 10.0 * np.log10(np.maximum(spec, 1e-12))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(pulse)))
        ax.plot(freqs, spec, linewidth=0.8, color='darkorange')
        ax.set_title(f'ci={ci}  peak={spec.max():.1f} dB', fontsize=8)
        ax.set_xlabel('Norm. freq', fontsize=7)
        ax.set_ylabel('dB', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, linestyle='--', alpha=0.4)

    for ax in axes.flat[n_samples:]:
        ax.axis('off')

    fig.tight_layout()
    out_path = os.path.join(out_dir, f'{block_name}_raw_range_spectra.png')
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f'  range spectra       -> {os.path.basename(out_path)}')


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Sanity check raw NISAR L0B decoded data before inference.'
    )
    parser.add_argument('--l0',     default=DEFAULT_L0,
                        help='Path to raw NISAR L0B HDF5 file.')
    parser.add_argument('--out',    default=DEFAULT_OUT,
                        help='Output directory for PNGs.')
    parser.add_argument('--n',      type=int, default=16,
                        help='Number of CPI tiles to sample (default 16).')
    parser.add_argument('--target', default=None,
                        help='Block name to check (default: first in TARGETS).')
    args = parser.parse_args()

    if not os.path.exists(args.l0):
        raise FileNotFoundError(f'L0B file not found: {args.l0}')

    os.makedirs(args.out, exist_ok=True)

    # Pick target block
    target_map = {name: (ps, pe) for ps, pe, name in TARGETS}
    if args.target is None:
        _, _, block_name = TARGETS[0]
    else:
        if args.target not in target_map:
            raise ValueError(f'Unknown target "{args.target}". '
                             f'Available: {list(target_map)}')
        block_name = args.target

    pulse_start, pulse_end = target_map[block_name]
    n_pulses   = pulse_end - pulse_start
    n_cpi_rows = n_pulses // M

    print(f'L0B      : {args.l0}')
    print(f'Block    : {block_name}  pulses [{pulse_start}:{pulse_end}]')
    print(f'Samples  : {args.n}')
    print(f'Out dir  : {args.out}')
    print()

    print(f'Loading block ...')
    block      = load_block(args.l0, L0B_DATASET, pulse_start, pulse_end)
    # Trim to ri=0 only (first BLOCK_WIDTH range samples)
    block      = block[:n_cpi_rows * M, :BLOCK_WIDTH]
    print(f'Block shape (trimmed to ri=0): {block.shape}')

    sample_cis = np.linspace(0, n_cpi_rows - 1, args.n, dtype=int)

    print(f'Plotting ...')
    plot_eigenvalue_profiles(block, block_name, sample_cis, args.out, args.l0)
    plot_range_spectra(block, block_name, sample_cis, args.out, args.l0)

    print()
    print('Done.')


if __name__ == '__main__':
    main()