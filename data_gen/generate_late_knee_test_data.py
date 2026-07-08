"""
generate_late_knee_test_data.py

Generates synthetic test data with "late knees" at positions 7 or 8 to evaluate
model performance on these edge cases.

Key differences from training data:
  - Fixed knee positions: exactly 7 or 8 RFI bands per block
  - Test data only (no training usage)
  - Saved to separate directory: data/test_late_knee/

The objective is to observe how the trained model responds to cases where the
knee appears later in the eigenvalue profile (positions 7-8) compared to typical
training data (knees at positions 1-6).
"""

import os
import json
import numpy as np
import h5py
from dataclasses import dataclass, field
from typing import List

# Import constants and utilities from the main generation script
# We'll reuse the same infrastructure
NOISE_DB = 3
SNR_RANGE_DB = (6, 20)
JNR_MAX_DB = 30  # Absolute max for JNR
JNR_MIN_OFFSET_DB = 3  # JNR must be at least 3 dB above SNR

TOTAL_PULSES = 1600
RANGE_BINS = 10000
BLOCK_HEIGHT = 16
BLOCK_WIDTH = 250

assert TOTAL_PULSES % BLOCK_HEIGHT == 0
N_BLOCKS = TOTAL_PULSES // BLOCK_HEIGHT

# For late knee test: fixed knee positions
LATE_KNEE_POSITIONS = [7, 8]  # Test knees at positions 7 and 8

# Seed offsets
NOISE_SEED = 0
SIGNAL_SEED = 1
RFI_SEED = 2
JNR_SEED = 3


@dataclass
class BandMeta:
    """Descriptor for a single RFI band within one block."""
    local_idx: int   # pulse index within the block [0, BLOCK_HEIGHT)
    row: int         # absolute pulse row in the full image
    jnr_db: int      # JNR for this band in dB


@dataclass
class RfiMeta:
    """Carries all RFI injection metadata for one generated image."""
    bands_per_block: List[List[BandMeta]] = field(default_factory=list)


def generate_clean_image(seed=0, snr_db=6):
    """
    Generate a complex-valued clean image: spatially white noise + signal.

    Noise power : NOISE_DB dB
    Signal power: NOISE_DB + snr_db dB
    """
    noise_power_linear = 10.0 ** (NOISE_DB / 10.0)
    signal_power_linear = noise_power_linear * (10.0 ** (snr_db / 10.0))

    rng_noise = np.random.default_rng(seed + NOISE_SEED)
    noise_matrix = (
        rng_noise.standard_normal((TOTAL_PULSES, RANGE_BINS))
        + 1j * rng_noise.standard_normal((TOTAL_PULSES, RANGE_BINS))
    ) * np.sqrt(noise_power_linear / 2.0)

    rng_signal = np.random.default_rng(seed + SIGNAL_SEED)
    signal_matrix = (
        rng_signal.standard_normal((TOTAL_PULSES, RANGE_BINS))
        + 1j * rng_signal.standard_normal((TOTAL_PULSES, RANGE_BINS))
    ) * np.sqrt(signal_power_linear / 2.0)

    return (noise_matrix + signal_matrix).astype(np.complex64)


def _generate_late_knee_rfi(seed, n_bands, snr_db):
    """
    Generate multi-band RFI with a fixed number of bands per block.

    This is different from the training data generator which uses random
    band counts (1-6). Here we fix the number of bands to create specific
    knee positions (7 or 8).

    Args:
        seed (int): Base seed for RNG
        n_bands (int): Fixed number of bands per block (7 or 8)
        snr_db (float): Signal-to-noise ratio in dB, used to determine JNR range

    Returns:
        rfi_matrix (np.ndarray): Complex64, shape (TOTAL_PULSES, RANGE_BINS)
        meta (RfiMeta): Per-block band descriptors
    """
    noise_power_linear = 10.0 ** (NOISE_DB / 10.0)

    rng_struct = np.random.default_rng(seed + RFI_SEED)
    rng_jnr = np.random.default_rng(seed + JNR_SEED)

    rfi_matrix = np.zeros((TOTAL_PULSES, RANGE_BINS), dtype=np.complex64)
    bands_per_block: List[List[BandMeta]] = []

    # Calculate JNR range based on SNR
    # JNR must be at least 3 dB above SNR and at most 30 dB
    jnr_min = snr_db + JNR_MIN_OFFSET_DB
    jnr_max = JNR_MAX_DB

    for b in range(N_BLOCKS):
        block_start = b * BLOCK_HEIGHT

        # Sample n_bands distinct pulse positions without replacement
        # This ensures each band is on a different pulse
        local_indices = rng_struct.choice(BLOCK_HEIGHT, size=n_bands, replace=False)

        block_bands: List[BandMeta] = []

        for band_idx in range(n_bands):
            local_idx = int(local_indices[band_idx])
            abs_row = block_start + local_idx

            # Per-band JNR: at least 3 dB above SNR, max 30 dB
            jnr_db = int(rng_jnr.integers(jnr_min, jnr_max + 1))
            rfi_power_linear = noise_power_linear * (10.0 ** (jnr_db / 10.0))
            sigma = np.sqrt(rfi_power_linear / 2.0)

            # Derive deterministic child seed
            child_seed = (seed + RFI_SEED) * 10_000 + b * 10 + band_idx
            rng_band = np.random.default_rng(child_seed)

            doppler_freq = rng_band.uniform(-0.5, 0.5)
            range_coeff = (
                rng_band.standard_normal(RANGE_BINS)
                + 1j * rng_band.standard_normal(RANGE_BINS)
            ) * sigma

            phase = np.exp(1j * 2.0 * np.pi * doppler_freq * abs_row)
            rfi_matrix[abs_row, :] += (phase * range_coeff).astype(np.complex64)

            block_bands.append(BandMeta(local_idx=local_idx, row=abs_row,
                                        jnr_db=jnr_db))

        bands_per_block.append(block_bands)

    meta = RfiMeta(bands_per_block=bands_per_block)
    return rfi_matrix, meta


def generate_late_knee_rfi_image(seed=0, snr_db=6, n_bands=7):
    """
    Generate a complex-valued image with late knee RFI.

    Args:
        seed (int): Base seed
        snr_db (float): Signal-to-noise ratio in dB
        n_bands (int): Number of RFI bands per block (7 or 8)

    Returns:
        rfi_image (np.ndarray): Complex64 array
        meta (RfiMeta): Per-block band descriptors
    """
    clean_image = generate_clean_image(seed, snr_db)
    rfi_signal, meta = _generate_late_knee_rfi(seed, n_bands, snr_db)
    rfi_image = (clean_image + rfi_signal).astype(np.complex64)
    return rfi_image, meta


def divide_cpi_and_save(
    matrix,
    meta,
    seed,
    snr_db,
    n_bands,
    cpi_height=BLOCK_HEIGHT,
    cpi_width=BLOCK_WIDTH,
    output_path=None,
):
    """
    Divide matrix into CPI tiles and save to HDF5.

    Similar to the training data format but includes n_bands in metadata.
    """
    if output_path is None:
        return

    assert TOTAL_PULSES % cpi_height == 0
    assert RANGE_BINS % cpi_width == 0

    with h5py.File(output_path, 'w') as f:
        # Root attributes
        f.attrs['total_pulses'] = TOTAL_PULSES
        f.attrs['range_bins'] = RANGE_BINS
        f.attrs['cpi_height'] = cpi_height
        f.attrs['cpi_width'] = cpi_width
        f.attrs['block_height'] = BLOCK_HEIGHT
        f.attrs['n_blocks'] = N_BLOCKS
        f.attrs['noise_db'] = NOISE_DB
        f.attrs['snr_db'] = snr_db
        # JNR range depends on SNR: at least 3 dB above SNR, max 30 dB
        f.attrs['jnr_range_low'] = snr_db + JNR_MIN_OFFSET_DB
        f.attrs['jnr_range_high'] = JNR_MAX_DB
        f.attrs['n_bands_fixed'] = n_bands  # Fixed for late knee test
        f.attrs['seed'] = seed
        f.attrs['is_test'] = True  # Mark as test data

        # CPI datasets
        for i in range(0, TOTAL_PULSES, cpi_height):
            block_idx = i // BLOCK_HEIGHT
            bands = meta.bands_per_block[block_idx]

            pulse_positions = [b.local_idx + 1 for b in bands]  # 1-based
            jnr_db_list = [b.jnr_db for b in bands]
            knee = len(bands)

            bands_json = json.dumps({
                'pulse_positions': pulse_positions,
                'knee': knee,
                'jnr_db_list': jnr_db_list,
            })

            for j in range(0, RANGE_BINS, cpi_width):
                cpi = matrix[i:i + cpi_height, j:j + cpi_width]

                # Compute SCM and eigenvalues
                M = cpi
                SCM = (M @ M.conj().T) / cpi_width
                diagonal = np.diag(SCM).real
                eigvals = np.linalg.eigvalsh(SCM)
                eigvals_sorted = np.sort(eigvals)[::-1]

                # Save datasets
                dset = f.create_dataset(f"cpi_{i}_{j}", data=cpi)
                dset.attrs['rfi_bands'] = bands_json
                f.create_dataset(f"cpi_{i}_{j}_eigenvalues", data=eigvals_sorted)
                f.create_dataset(f"cpi_{i}_{j}_diagonal", data=diagonal)


def _compute_eigenvalues_db(cpi):
    """
    Compute sorted descending eigenvalues in dB scale from a complex CPI tile.

    Returns:
        eigvals_db (np.ndarray): Eigenvalues in dB relative to max
        eigvals_abs_db (np.ndarray): Eigenvalues in absolute dB
        max_eigval_db (float): Largest eigenvalue in absolute dB
        min_eigval_db (float): Smallest eigenvalue in absolute dB
        condition_number (float): Ratio of max to min eigenvalue
        effective_rank (float): Normalized participation ratio
    """
    M, K = cpi.shape
    SCM = (cpi @ cpi.conj().T) / K
    eigvals = np.linalg.eigvalsh(SCM)
    eigvals = np.sort(eigvals)[::-1]

    max_eigval = eigvals[0]
    min_eigval = eigvals[-1]

    # Condition number: ratio of max to min eigenvalue
    condition_number = max_eigval / max(min_eigval, 1e-12)

    # Effective rank: normalized participation ratio
    # effective_rank = (sum of eigenvalues)^2 / (sum of eigenvalues^2)
    eigvals_normalized = eigvals / max(np.sum(eigvals), 1e-12)
    effective_rank = 1.0 / np.sum(eigvals_normalized ** 2)

    # Absolute dB scale
    eigvals_abs_db = 10.0 * np.log10(eigvals + 1e-12)
    max_eigval_db = 10.0 * np.log10(max(max_eigval, 1e-12))
    min_eigval_db = 10.0 * np.log10(max(min_eigval, 1e-12))

    # Relative dB scale
    eigvals_rel_db = 10.0 * np.log10(eigvals / max(max_eigval, 1e-12) + 1e-12)

    return eigvals_rel_db, eigvals_abs_db, max_eigval_db, min_eigval_db, condition_number, effective_rank


def plot_late_knee_profiles(h5_path, out_dir, n_bands):
    """
    Generate eigenvalue profile plots for late knee test data.

    Creates two plots:
    1. All blocks overlay with knee markers at position n_bands (absolute dB scale)
    2. Grid of selected blocks showing late knee structure (absolute dB scale)

    Both plots include condition number, effective rank, and max RFI power.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Read JNR range from file attributes (depends on SNR)
    with h5py.File(h5_path, 'r') as f_temp:
        jnr_min_global = f_temp.attrs['jnr_range_low']
        jnr_max_global = f_temp.attrs['jnr_range_high']

    block_pulse_offsets = [b * BLOCK_HEIGHT for b in range(N_BLOCKS)]
    ev_index_1based = np.arange(1, BLOCK_HEIGHT + 1)

    with h5py.File(h5_path, 'r') as f:
        snr_db = f.attrs['snr_db']
        all_profiles_abs = []  # Absolute dB
        all_max_eigval_db = []
        all_min_eigval_db = []
        all_condition_numbers = []
        all_effective_ranks = []
        all_max_jnr = []
        all_knees = []

        for pulse_offset in block_pulse_offsets:
            dset = f[f"cpi_{pulse_offset}_0"]
            cpi = dset[:]
            eigvals_rel_db, eigvals_abs_db, max_eigval_db, min_eigval_db, cond_num, eff_rank = _compute_eigenvalues_db(cpi)

            all_profiles_abs.append(eigvals_abs_db)
            all_max_eigval_db.append(max_eigval_db)
            all_min_eigval_db.append(min_eigval_db)
            all_condition_numbers.append(cond_num)
            all_effective_ranks.append(eff_rank)

            payload = json.loads(str(dset.attrs['rfi_bands']))
            knee = payload['knee']
            max_jnr = float(np.max(payload['jnr_db_list'])) if knee > 0 else 0.0

            all_max_jnr.append(max_jnr)
            all_knees.append(knee)

    # Determine global y-axis range for consistent scaling
    all_eigvals = np.concatenate(all_profiles_abs)
    global_min_db = np.percentile(all_eigvals, 1)  # 1st percentile to avoid outliers
    global_max_db = np.percentile(all_eigvals, 99)  # 99th percentile
    y_margin = 5  # dB margin
    ylim = [global_min_db - y_margin, global_max_db + y_margin]

    # Compute average statistics for the file
    avg_condition = np.mean(all_condition_numbers)
    avg_eff_rank = np.mean(all_effective_ranks)
    avg_max_jnr = np.mean([j for j in all_max_jnr if j > 0])

    # Plot 1: All blocks overlay (absolute dB scale with colorbar)
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    # Color by max eigenvalue
    cmap = cm.viridis
    norm = plt.Normalize(vmin=min(all_max_eigval_db), vmax=max(all_max_eigval_db))

    for idx, (eigvals_abs_db, max_eigval_db, knee) in enumerate(zip(all_profiles_abs, all_max_eigval_db, all_knees)):
        color = cmap(norm(max_eigval_db))
        line = ax1.plot(ev_index_1based, eigvals_abs_db, color=color, alpha=0.5, linewidth=1.0)

        # Mark knee position (should be at n_bands)
        if knee > 0 and knee <= BLOCK_HEIGHT:
            ax1.plot(knee, eigvals_abs_db[knee-1], 'rx', markersize=6, alpha=0.6,
                    markeredgewidth=2)

    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig1.colorbar(sm, ax=ax1, label='Max Eigenvalue (dB)')

    # Add vertical line at expected knee position
    ax1.axvline(x=n_bands, color='red', linestyle='--', linewidth=2, alpha=0.7,
                label=f'Expected knee = {n_bands}')

    ax1.set_xlabel('Eigenvalue Index (1-based)', fontsize=12)
    ax1.set_ylabel('Eigenvalue (dB, absolute scale)', fontsize=12)
    ax1.set_ylim(ylim)
    ax1.set_title(
        f'Late Knee Test: All Blocks (knee={n_bands})\n'
        f'SNR={snr_db:.1f} dB | JNR={jnr_min_global}-{jnr_max_global} dB (avg={avg_max_jnr:.1f} dB) | '
        f'Avg Cond#={avg_condition:.1f} | Avg Eff Rank={avg_eff_rank:.1f}\n'
        f'{os.path.basename(h5_path)}',
        fontsize=10,
    )
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(fontsize=10)
    fig1.tight_layout()

    stem1 = os.path.splitext(os.path.basename(h5_path))[0]
    out_path1 = os.path.join(out_dir, f"{stem1}_ev_all_blocks.png")
    fig1.savefig(out_path1, dpi=150)
    plt.close(fig1)

    # Plot 2: Grid of selected blocks (absolute dB scale)
    selected_block_indices = [b * (N_BLOCKS // 10) for b in range(10)]
    n_cols, n_rows = 5, 2
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(18, 7))

    for ax, block_idx in zip(axes.flat, selected_block_indices):
        eigvals_abs_db = all_profiles_abs[block_idx]
        max_eigval_db = all_max_eigval_db[block_idx]
        min_eigval_db = all_min_eigval_db[block_idx]
        cond_num = all_condition_numbers[block_idx]
        eff_rank = all_effective_ranks[block_idx]
        max_jnr = all_max_jnr[block_idx]
        knee = all_knees[block_idx]

        ax.plot(ev_index_1based, eigvals_abs_db, color='steelblue', linewidth=1.5)

        # Mark knee position
        if knee > 0 and knee <= BLOCK_HEIGHT:
            ax.plot(knee, eigvals_abs_db[knee-1], 'rx', markersize=10, markeredgewidth=2.5)
            ax.axvline(x=knee, color='red', linestyle='--', alpha=0.4, linewidth=1.5)

        ax.set_title(
            f'Block {block_idx} [knee={knee}]\n'
            f'Max RFI={max_jnr:.0f} dB | Cond#={cond_num:.1f} | Eff Rank={eff_rank:.1f}\n'
            f'λ: [{min_eigval_db:.1f}, {max_eigval_db:.1f}] dB',
            fontsize=7
        )
        ax.set_xlabel('EV Index', fontsize=8)
        ax.set_ylabel('Eigenvalue (dB)', fontsize=8)
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle='--', alpha=0.4)

    fig2.suptitle(
        f'Late Knee Test: Selected Blocks (knee={n_bands}) | SNR={snr_db:.1f} dB\n'
        f'{os.path.basename(h5_path)}',
        fontsize=11,
    )
    fig2.tight_layout()

    out_path2 = os.path.join(out_dir, f"{stem1}_ev_selected_blocks.png")
    fig2.savefig(out_path2, dpi=150)
    plt.close(fig2)

    print(f"    plots -> {os.path.basename(out_path1)}, {os.path.basename(out_path2)}")


def main():
    """
    Generate late knee test data.

    Creates test samples with knees at positions 7 and 8 to evaluate model
    performance on these edge cases.

    SNR range: 6-20 dB
    JNR range: at least 3 dB above SNR, max 30 dB
    """
    N_IMAGES_PER_KNEE = 5  # 5 images per knee position per SNR level
    # Expanded SNR levels to cover the full range (6-20)
    SNR_LEVELS = [6, 8, 10, 12, 14, 16, 18, 20]

    test_dir = os.path.join('data', 'test_late_knee')
    os.makedirs(test_dir, exist_ok=True)

    print("\n" + "="*70)
    print("LATE KNEE TEST DATA GENERATION")
    print("="*70)
    print(f"\nObjective: Test model response to late knees at positions 7-8")
    print(f"Test knee positions: {LATE_KNEE_POSITIONS}")
    print(f"Images per knee per SNR: {N_IMAGES_PER_KNEE}")
    print(f"SNR levels: {SNR_LEVELS} dB")
    print(f"JNR range: SNR + {JNR_MIN_OFFSET_DB} dB to {JNR_MAX_DB} dB (dynamic per SNR level)")
    print(f"Output directory: {test_dir}")
    print(f"Seed range: Starting at 1000 (separate from training seeds 0-199)")

    # Start at seed=1000 to ensure no correlation with training data
    # Training uses: 0-99 (clean), 100-199 (contaminated)
    seed = 1000
    total_images = 0

    for n_bands in LATE_KNEE_POSITIONS:
        print(f"\n{'='*70}")
        print(f"Generating data with knee={n_bands}")
        print(f"{'='*70}")

        knee_dir = os.path.join(test_dir, f'knee_{n_bands}')
        os.makedirs(knee_dir, exist_ok=True)

        for snr_db in SNR_LEVELS:
            print(f"\n  SNR = {snr_db} dB:")

            for img_idx in range(N_IMAGES_PER_KNEE):
                out_path = os.path.join(
                    knee_dir,
                    f"test_knee{n_bands}_seed{seed}_snr{snr_db}.h5"
                )

                rfi_image, meta = generate_late_knee_rfi_image(
                    seed=seed,
                    snr_db=snr_db,
                    n_bands=n_bands
                )

                divide_cpi_and_save(
                    matrix=rfi_image,
                    meta=meta,
                    seed=seed,
                    snr_db=snr_db,
                    n_bands=n_bands,
                    output_path=out_path,
                )

                print(f"    seed={seed:03d}  -> {os.path.basename(out_path)}")
                plot_late_knee_profiles(
                    h5_path=out_path,
                    out_dir=knee_dir,
                    n_bands=n_bands
                )

                seed += 1
                total_images += 1

    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print(f"\nTotal test images generated: {total_images}")
    print(f"Output directory: {test_dir}/")
    print(f"\nDirectory structure:")
    for n_bands in LATE_KNEE_POSITIONS:
        knee_dir = os.path.join(test_dir, f'knee_{n_bands}')
        n_files = len([f for f in os.listdir(knee_dir) if f.endswith('.h5')]) if os.path.exists(knee_dir) else 0
        print(f"  knee_{n_bands}/: {n_files} HDF5 files + plots")

if __name__ == '__main__':
    main()
