#!/usr/bin/env python
"""
Create generalization test suite using REAL SAR data as background.

Injects all synthetic RFI patterns from evaluate_generalization_unified.py
into real SAR tiles, creating realistic test data with ground truth.

Usage:
    python analysis/scripts/create_real_background_test_suite.py \\
        --l0b-file /scratch2/bohuang/amazon/NISAR_L0_PR_RRSD_010_017_A_148S_20260110T101338_20260110T102252_X05009_N_J_001.h5 \\
        --test-indices 888222 896222 \\
        --output model/test_suite_real_background.npz \\
        --tile-size 128 200
"""
import numpy as np
import h5py
import argparse
import sys
from pathlib import Path

# Add root directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from generate_unet_segmentation_data import Raw, build_tone_remover, read_raw_tile, get_subswath_mask, amplitude_gap_mask

# Test case definitions (same as evaluate_generalization_unified.py)
TEST_CASES = [
    # IN-DISTRIBUTION (seen during training)
    {
        'name': 'Standard Gaussian Blobs',
        'category': 'IN-DIST',
        'description': '4-24 pulses × 15-90% range',
        'type': 'gaussian_blobs',
        'params': {'n_blobs': 4, 'pulse_range': (4, 24), 'range_frac': (0.15, 0.90)}
    },
    {
        'name': 'Multiple Overlapping Blobs',
        'category': 'IN-DIST',
        'description': '6-8 blobs with overlap',
        'type': 'gaussian_blobs',
        'params': {'n_blobs': 7, 'pulse_range': (8, 20), 'range_frac': (0.20, 0.70)}
    },
    {
        'name': 'Small Blob Count',
        'category': 'IN-DIST',
        'description': '1-2 isolated blobs',
        'type': 'gaussian_blobs',
        'params': {'n_blobs': 2, 'pulse_range': (6, 16), 'range_frac': (0.25, 0.60)}
    },
    {
        'name': 'Edge Blobs',
        'category': 'IN-DIST',
        'description': 'Blobs at boundaries',
        'type': 'edge_blobs',
        'params': {'n_blobs': 3}
    },
    {
        'name': 'Wide Horizontal Blob',
        'category': 'IN-DIST',
        'description': 'Large range extent (85%)',
        'type': 'gaussian_blobs',
        'params': {'n_blobs': 1, 'pulse_range': (12, 16), 'range_frac': (0.80, 0.85)}
    },

    # OUT-OF-DISTRIBUTION (NOT seen during training)
    {
        'name': 'Very Small Point Sources',
        'category': 'OOD',
        'description': '2x2 pixel spots',
        'type': 'point_sources',
        'params': {'n_points': 8, 'size': 2}
    },
    {
        'name': 'Diagonal Streaks',
        'category': 'OOD',
        'description': '45° diagonal lines',
        'type': 'diagonal_lines',
        'params': {'n_lines': 3, 'width': 3}
    },
    {
        'name': 'Full-Width Vertical Stripes',
        'category': 'OOD',
        'description': 'Narrowband spanning all pulses',
        'type': 'vertical_stripes',
        'params': {'n_stripes': 4, 'width': 3}
    },
    {
        'name': 'Full-Height Horizontal Bands',
        'category': 'OOD',
        'description': 'Wideband spanning all ranges',
        'type': 'horizontal_bands',
        'params': {'n_bands': 3, 'height': 6}
    },
    {
        'name': 'Very Large Uniform RFI',
        'category': 'OOD',
        'description': '>90% coverage',
        'type': 'large_uniform',
        'params': {'coverage': 0.65}
    },
    {
        'name': 'Thin Lines',
        'category': 'OOD',
        'description': '1-pixel vertical/horizontal',
        'type': 'thin_lines',
        'params': {'n_lines': 5}
    },
    {
        'name': 'L-Shaped Pattern',
        'category': 'OOD',
        'description': 'Geometric corner shape',
        'type': 'l_shape',
        'params': {'n_shapes': 2}
    },
    {
        'name': 'Scattered Random Pixels',
        'category': 'OOD',
        'description': 'Salt-and-pepper noise',
        'type': 'scattered',
        'params': {'density': 0.08}
    },
]

def create_gaussian_blob_mask(height, width, center_pulse, center_range,
                               pulse_size, range_size, mask_threshold=0.3, sigma_scale=3.0):
    """Create a 2D Gaussian blob (matches training data)"""
    y, x = np.ogrid[:height, :width]
    sigma_pulse = pulse_size / sigma_scale
    sigma_range = range_size / sigma_scale
    dist_sq = ((y - center_pulse) / sigma_pulse) ** 2 + ((x - center_range) / sigma_range) ** 2
    envelope = np.exp(-0.5 * dist_sq)
    mask = (envelope >= mask_threshold).astype(np.uint8)
    return mask, envelope

def generate_rfi_mask(case, height, width, valid_mask, seed=42):
    """Generate ground truth RFI mask for a test case"""
    np.random.seed(seed)
    mask = np.zeros((height, width), dtype=np.uint8)

    gap_left = np.where(valid_mask[height//2, :])[0][0]
    gap_right = width - np.where(valid_mask[height//2, :])[0][-1] - 1
    valid_width = width - gap_left - gap_right

    case_type = case['type']
    params = case['params']

    if case_type == 'gaussian_blobs':
        n_blobs = params['n_blobs']
        pulse_range = params['pulse_range']
        range_frac = params['range_frac']

        for _ in range(n_blobs):
            pulse_size = np.random.uniform(*pulse_range)
            rf = np.random.uniform(*range_frac)
            range_size = rf * valid_width
            center_pulse = np.random.uniform(0, height)
            center_range = np.random.uniform(gap_left, width - gap_right)

            blob, _ = create_gaussian_blob_mask(height, width, center_pulse, center_range,
                                            pulse_size, range_size)
            mask = mask | (blob & valid_mask)

    elif case_type == 'edge_blobs':
        positions = [
            (5, gap_left + 20),
            (height - 10, width - gap_right - 30),
            (height // 2, gap_left + 5),
        ]
        for i, (cp, cr) in enumerate(positions[:params['n_blobs']]):
            blob, _ = create_gaussian_blob_mask(height, width, cp, cr, 14, 0.4 * valid_width)
            mask = mask | (blob & valid_mask)

    elif case_type == 'point_sources':
        size = params['size']
        for _ in range(params['n_points']):
            y = np.random.randint(0, height - size)
            x = np.random.randint(gap_left, width - gap_right - size)
            mask[y:y+size, x:x+size] = 1
        mask = mask & valid_mask

    elif case_type == 'diagonal_lines':
        for i in range(params['n_lines']):
            start_x = gap_left + i * (valid_width // params['n_lines'])
            for offset in range(-params['width']//2, params['width']//2 + 1):
                for y in range(height):
                    x = start_x + y + offset
                    if gap_left <= x < width - gap_right:
                        mask[y, x] = 1

    elif case_type == 'vertical_stripes':
        for i in range(params['n_stripes']):
            x = gap_left + (i + 1) * (valid_width // (params['n_stripes'] + 1))
            w = params['width']
            mask[:, max(gap_left, x-w//2):min(width-gap_right, x+w//2+1)] = 1
        mask = mask & valid_mask

    elif case_type == 'horizontal_bands':
        for i in range(params['n_bands']):
            y = (i + 1) * (height // (params['n_bands'] + 1))
            h = params['height']
            mask[y-h//2:y+h//2+1, gap_left:width-gap_right] = 1

    elif case_type == 'large_uniform':
        coverage = params['coverage']
        h_start = int(height * (1 - coverage) / 2)
        h_end = int(height * (1 + coverage) / 2)
        w_start = gap_left + int(valid_width * (1 - coverage) / 2)
        w_end = width - gap_right - int(valid_width * (1 - coverage) / 2)
        mask[h_start:h_end, w_start:w_end] = 1

    elif case_type == 'thin_lines':
        for i in range(params['n_lines']):
            if i % 2 == 0:  # vertical
                x = gap_left + np.random.randint(0, valid_width)
                mask[:, x] = 1
            else:  # horizontal
                y = np.random.randint(0, height)
                mask[y, gap_left:width-gap_right] = 1
        mask = mask & valid_mask

    elif case_type == 'l_shape':
        for i in range(params['n_shapes']):
            y = np.random.randint(20, height - 40)
            x = gap_left + np.random.randint(20, valid_width - 40)
            arm_len = 30
            thickness = 8
            mask[y:y+arm_len, x:x+thickness] = 1
            mask[y:y+thickness, x:x+arm_len] = 1
        mask = mask & valid_mask

    elif case_type == 'scattered':
        scattered = np.random.rand(height, width) < params['density']
        mask = (scattered & valid_mask).astype(np.uint8)

    return mask

def inject_rfi_into_tile(clean_tile, rfi_mask, jsr_db=10.0):
    """
    Inject synthetic RFI into a clean SAR tile.

    Parameters
    ----------
    clean_tile : (P, K) complex64
        Clean SAR background
    rfi_mask : (P, K) uint8
        Binary mask indicating where to inject RFI
    jsr_db : float
        Jammer-to-Signal ratio in dB

    Returns
    -------
    (P, K) complex64
        Contaminated tile
    """
    tile_with_rfi = clean_tile.copy()

    # Calculate signal power (median over clean regions)
    signal_power = np.median(np.abs(clean_tile[rfi_mask == 0]))
    rfi_power = signal_power * 10 ** (jsr_db / 20.0)

    # Add RFI where mask is 1
    rfi_indices = np.where(rfi_mask)
    n_rfi_pixels = len(rfi_indices[0])

    rfi_noise = rfi_power * (np.random.randn(n_rfi_pixels) + 1j * np.random.randn(n_rfi_pixels))
    tile_with_rfi[rfi_indices] += rfi_noise.astype(np.complex64)

    return tile_with_rfi

def main():
    parser = argparse.ArgumentParser(description='Create generalization test suite with real SAR backgrounds')
    parser.add_argument('--l0b-file', type=str, required=True, help='Path to L0B HDF5 file')
    parser.add_argument('--test-indices', type=int, nargs='+', required=True,
                        help='Pulse indices to extract test tiles from')
    parser.add_argument('--output', type=str, required=True, help='Output .npz file')
    parser.add_argument('--tile-size', type=int, nargs=2, default=[256, 256],
                        help='Tile dimensions (height width) - default 256x256 to match training data')
    parser.add_argument('--frequency', type=str, default='A', help='Frequency band')
    parser.add_argument('--polarization', type=str, default='HH', help='Polarization')
    parser.add_argument('--range-start', type=int, default=1000, help='Starting range sample')
    parser.add_argument('--range-end', type=int, default=26000, help='Ending range sample')
    parser.add_argument('--jsr-range', type=float, nargs=2, default=[2.0, 30.0],
                        help='Jammer-to-signal ratio range in dB [min, max]')
    parser.add_argument('--remove-caltone', action='store_true', help='Remove calibration tone')
    args = parser.parse_args()

    height, width = args.tile_size
    freq = args.frequency
    pol = args.polarization

    print(f"Creating test suite with real SAR backgrounds")
    print(f"  L0B: {args.l0b_file}")
    print(f"  Test indices: {args.test_indices}")
    print(f"  Range: {args.range_start} to {args.range_end}")
    print(f"  Tile size: {height}x{width}")
    print(f"  JSR range: {args.jsr_range[0]:.1f} to {args.jsr_range[1]:.1f} dB")
    print(f"  Number of test cases: {len(TEST_CASES)}")

    # Open L0B file
    raw = Raw(hdf5file=args.l0b_file)
    raw.parsePolarizations()

    # Build tone remover if requested
    remover = None
    if args.remove_caltone:
        full_range = raw.getRawDataset(freq, pol).shape[1]
        remover, caltone_freq = build_tone_remover(raw, freq, pol, full_range)
        print(f"  Caltone removal: ON ({caltone_freq/1e6:.4f} MHz)")

    # Extract clean background tiles
    background_tiles = []
    valid_masks = []

    # Calculate range sampling positions within the specified range
    available_range = args.range_end - args.range_start
    if available_range < width:
        print(f"ERROR: Range window ({available_range} samples) is smaller than tile width ({width})")
        sys.exit(1)

    # Sample range positions uniformly across the valid range
    n_samples = len(args.test_indices)
    range_positions = np.linspace(args.range_start, args.range_end - width, n_samples).astype(int)

    print(f"\nExtracting {len(args.test_indices)} background tiles from range {args.range_start}-{args.range_end}...")
    for pulse_idx, range_start in zip(args.test_indices, range_positions):
        # Note: function signature is (p0, cpi_len, r0, cpi_width)
        tile = read_raw_tile(raw, freq, pol, pulse_idx, height, range_start, width, remover)
        valid = get_subswath_mask(raw, freq, pol, pulse_idx, height, range_start, width)
        valid = valid & amplitude_gap_mask(tile)

        background_tiles.append(tile)
        valid_masks.append(valid)
        print(f"  Pulse {pulse_idx}, Range {range_start}: valid fraction = {valid.sum() / valid.size:.3f}")

    # Generate test suite
    print(f"\nGenerating {len(TEST_CASES)} test patterns...")

    all_tiles = []
    all_masks = []
    all_valid = []
    all_names = []
    all_categories = []
    all_descriptions = []
    all_jsr_values = []

    # Sample JSR values uniformly across the range
    jsr_values = np.linspace(args.jsr_range[0], args.jsr_range[1], len(TEST_CASES))

    for i, case in enumerate(TEST_CASES):
        # Use different background for each test case (cycle through available tiles)
        bg_idx = i % len(background_tiles)
        clean_tile = background_tiles[bg_idx]
        valid_mask = valid_masks[bg_idx]

        # Generate RFI pattern
        rfi_mask = generate_rfi_mask(case, height, width, valid_mask, seed=42 + i)

        # Inject RFI into clean background with varying JSR
        jsr_db = jsr_values[i]
        contaminated_tile = inject_rfi_into_tile(clean_tile, rfi_mask, jsr_db)

        all_tiles.append(contaminated_tile)
        all_masks.append(rfi_mask)
        all_valid.append(valid_mask)
        all_names.append(case['name'])
        all_categories.append(case['category'])
        all_descriptions.append(case['description'])
        all_jsr_values.append(jsr_db)

        print(f"  [{i+1:2d}/{len(TEST_CASES)}] {case['name']:35s} ({case['category']:7s}) - "
              f"JSR: {jsr_db:4.1f} dB, RFI fraction: {rfi_mask.sum() / valid_mask.sum():.3f}")

    # Save to .npz
    print(f"\nSaving to {args.output}...")
    np.savez_compressed(
        args.output,
        tiles=np.array(all_tiles),
        masks=np.array(all_masks, dtype=bool),
        valid=np.array(all_valid, dtype=bool),
        names=np.array(all_names),
        categories=np.array(all_categories),
        descriptions=np.array(all_descriptions),
        jsr_values=np.array(all_jsr_values),
        l0b_file=args.l0b_file,
        test_indices=np.array(args.test_indices),
        range_positions=range_positions,
        range_start=args.range_start,
        range_end=args.range_end,
        frequency=freq,
        polarization=pol,
        jsr_range=np.array(args.jsr_range),
        tile_shape=np.array([height, width])
    )

    print(f"\nTest suite created successfully!")
    print(f"  {len(all_tiles)} tiles saved")
    print(f"  Total size: {sum(t.nbytes for t in all_tiles) / 1e6:.1f} MB")

if __name__ == '__main__':
    main()
