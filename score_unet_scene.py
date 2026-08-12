#!/usr/bin/env python
"""
score_unet_scene.py

============================================================================
MAIN SCRIPT FOR SCORING UNET ON REAL L0B SCENES
============================================================================

Run UNet semantic segmentation on a NISAR L0B granule, directly reading from
the L0B file like score_scene.py does. This tiles the raw data on the fly and
generates contamination maps showing where the UNet detects RFI.

This is the UNet equivalent of score_scene.py (which uses eigenvalue classifier).

Usage:
    # Basic usage (full scene)
    python score_unet_scene.py \\
        /path/to/NISAR_L0B.h5 \\
        --model model/best_model.pth \\
        --output-dir results/unet_scene

    # Vienna RFI region example
    python score_unet_scene.py \\
        /scratch2/bohuang/eu/NISAR_L0_PR_RRSD_016_012_A_148S_20260127T034309_20260127T034853_P05006_F_J_001.h5 \\
        --model model/best_model.pth \\
        --pulse-start 363793 --pulse-end 440011 \\
        --output-dir results/unet_vienna \\
        --freq A --pol HH

Outputs:
    contamination_map_<freq>_<pol>.png    - Spatial heatmap of RFI contamination
    contamination_hist_<freq>_<pol>.png   - Distribution of contamination
    sample_predictions_<freq>_<pol>.png   - Random sample tiles (12 tiles)
    zoom_tile_XXXX_<freq>_<pol>.png       - Detailed view of top contaminated tiles
    predictions_<freq>_<pol>.h5           - Per-tile contamination fractions
    results.json                          - Summary statistics
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import h5py
import torch

# Matplotlib backend for headless server
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# NISAR readers
from nisar.products.readers.Raw import Raw
from isce3.focus import ToneRemover

# Import UNet - NOT using SegUNet, using the training architecture
import torch.nn as nn
import torch.nn.functional as F

# Constants
EPS = 1e-12
CPI_LEN_DEFAULT = 16
CPI_WIDTH_DEFAULT = 250
PULSE_CHUNK_DEFAULT = 1600


# ---------------------------------------------------------------------------
# UNET ARCHITECTURE (from train_unet.py)
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """U-Net for binary RFI segmentation (matches training architecture)."""
    def __init__(self, in_channels=2, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)


def prepare_tile_for_unet(tile: np.ndarray, valid: np.ndarray = None) -> np.ndarray:
    """
    Convert complex tile to [real, imag] channels as used in training.

    Parameters
    ----------
    tile : (P, K) complex64
    valid : (P, K) bool, optional (not used, but kept for compatibility)

    Returns
    -------
    (2, P, K) float32 - [real, imag] normalized by 99th percentile
    """
    # Stack real and imaginary parts
    tile_real_imag = np.stack([tile.real, tile.imag], axis=0).astype(np.float32)

    # Normalize by 99th percentile of magnitude over valid samples
    magnitude = np.sqrt(tile_real_imag[0]**2 + tile_real_imag[1]**2)
    if valid is not None and valid.sum() > 0:
        scale = np.percentile(magnitude[valid], 99)
    else:
        scale = np.percentile(magnitude, 99)

    tile_real_imag = tile_real_imag / max(scale, 1e-6)

    return tile_real_imag


# ---------------------------------------------------------------------------
# NISAR UTILITY FUNCTIONS (embedded - no external module needed)
# ---------------------------------------------------------------------------

def read_raw_data_batch(raw: Raw, freq: str, pol: str, pulse_slice: slice, range_slice: slice):
    """Read a (pulse, range) window; ISCE3 handles BFPQLUT decoding."""
    dataset = raw.getRawDataset(freq, pol)
    p0 = pulse_slice.start if pulse_slice.start is not None else 0
    p1 = pulse_slice.stop if pulse_slice.stop is not None else dataset.shape[0]
    r0 = range_slice.start if range_slice.start is not None else 0
    r1 = range_slice.stop if range_slice.stop is not None else dataset.shape[1]
    return dataset[p0:p1, r0:r1]


def get_subswath_mask(raw: Raw, freq: str, pol: str,
                      pulse_indices: np.ndarray, range_indices: np.ndarray) -> np.ndarray:
    """Boolean valid-sample mask from the ISCE3 subswath boundaries."""
    tx_pol = pol[0]
    subswaths = raw.getSubSwaths(freq, tx_pol)
    swaths = subswaths[:, pulse_indices, :]
    num_pulses = len(pulse_indices)
    num_range_samples = len(range_indices)
    mask = np.zeros((num_pulses, num_range_samples), dtype=bool)
    r_offset = int(range_indices[0])
    if swaths is not None:
        for i in range(num_pulses):
            for start, end in swaths[:, i, :]:
                s = max(int(start) - r_offset, 0)
                e = min(int(end) - r_offset, num_range_samples)
                if e > s:
                    mask[i, s:e] = True
    return mask

# Caltone removal (same as score_scene.py)
CALTONE_WINDOW_SIZE = 64
CALTONE_DEFAULT_FREQ_HZ = 1214.883e6
CALTONE_LO_HZ = 1200e6
CALTONE_CLOCK_HZ = 240e6

try:
    from nisar.products.readers.Raw import caltone_frequency_from_raw
except ImportError:
    caltone_frequency_from_raw = None


def parse_caltone_freq_from_drt(raw: Raw, txrx_pol: str) -> float:
    """Local fallback for caltone frequency."""
    path = (f'{raw.TelemetryPath}/DRT/MISC/'
            f'CP_IFSW_CALTONE_PHASE_STEP_{txrx_pol[1]}')
    with h5py.File(raw.filename, mode='r', swmr=True) as f:
        try:
            ds = f[path]
        except KeyError:
            print(f'  caltone: missing "{path}"; using default '
                  f'{CALTONE_DEFAULT_FREQ_HZ} Hz')
            return CALTONE_DEFAULT_FREQ_HZ
        i_cal = np.median(ds[()]).astype(int)
        return (i_cal / 2 ** 32) * CALTONE_CLOCK_HZ + CALTONE_LO_HZ


def build_tone_remover(raw: Raw, freq: str, pol: str, num_rng_samples: int):
    """Construct a ToneRemover for one channel."""
    tx_pol = pol[0]
    fc, fs, _, _ = raw.getChirpParameters(freq, tx_pol)
    if caltone_frequency_from_raw is not None:
        caltone_freq = caltone_frequency_from_raw(raw, pol)
    else:
        caltone_freq = parse_caltone_freq_from_drt(raw, pol)
    remover = ToneRemover((caltone_freq - fc) / fs, num_rng_samples,
                          CALTONE_WINDOW_SIZE)
    return remover, caltone_freq


# ---------------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------------

def score_channel_unet(raw, freq, pol, model, args, device='cuda'):
    """
    Stream one channel, tile it, run UNet, and return predictions.

    Similar to score_channel in score_scene.py but runs UNet instead of
    eigenvalue classifier.

    Returns:
        rec (dict): per-tile predictions and geometry
    """
    tile_height = args.tile_height
    tile_width = args.tile_width

    dataset = raw.getRawDataset(freq, pol)
    total_pulses, total_range = dataset.shape

    # Default to full extent if not specified
    p_start = args.pulse_start if args.pulse_start is not None else 0
    p_end = args.pulse_end if args.pulse_end is not None else total_pulses
    p_end = min(p_end, total_pulses)

    r_start = args.range_start if args.range_start is not None else 0
    r_end = args.range_end if args.range_end is not None else total_range
    r_end = min(r_end, total_range)

    n_pt = (p_end - p_start) // tile_height
    n_rt = (r_end - r_start) // tile_width
    p_end = p_start + n_pt * tile_height
    r_end = r_start + n_rt * tile_width

    if n_pt <= 0 or n_rt <= 0:
        raise ValueError('Window is smaller than one tile')

    n_tiles = n_pt * n_rt
    print(f"\n[{freq}-{pol}]  pulses [{p_start}:{p_end}]  range [{r_start}:{r_end}]")
    print(f"  tile grid: {n_pt} x {n_rt} = {n_tiles} tiles "
          f"({tile_height}x{tile_width} each)")

    # Build caltone remover
    if args.remove_caltone:
        remover, caltone_freq = build_tone_remover(raw, freq, pol, total_range)
        print(f"  caltone removal ON  (f_caltone = {caltone_freq/1e6:.4f} MHz)")
    else:
        remover = None
        print("  caltone removal OFF")

    # Allocate storage
    probs_all = np.zeros((n_tiles, tile_height, tile_width), dtype=np.float32)
    contamination_fractions = np.zeros(n_tiles, dtype=np.float32)
    contaminated_samples = np.zeros(n_tiles, dtype=np.int32)
    tile_pulse = np.zeros(n_tiles, dtype=np.int32)
    tile_range = np.zeros(n_tiles, dtype=np.int32)

    chunk_tiles = max(1, args.pulse_chunk // tile_height)
    model.eval()

    with torch.no_grad():
        k = 0
        for chunk_start in range(0, n_pt, chunk_tiles):
            n_here = min(chunk_tiles, n_pt - chunk_start)
            cp0 = p_start + chunk_start * tile_height
            cp1 = cp0 + n_here * tile_height

            # Read raw data
            if remover is not None:
                raw_full = np.ascontiguousarray(
                    read_raw_data_batch(
                        raw, freq, pol, slice(cp0, cp1), slice(0, total_range)
                    )
                ).astype(np.complex64)
                for ip in range(raw_full.shape[0]):
                    raw_full[ip] = remover.remove_tone(raw_full[ip])
                raw_chunk = raw_full[:, r_start:r_end]
            else:
                raw_chunk = read_raw_data_batch(
                    raw, freq, pol, slice(cp0, cp1), slice(r_start, r_end)
                )

            # Get subswath mask if needed
            mask_chunk = (
                get_subswath_mask(raw, freq, pol,
                                 np.arange(cp0, cp1), np.arange(r_start, r_end))
                if args.compute_subswath_mask else None
            )

            # Process tiles in this chunk
            batch_tiles = []
            batch_valid = []
            batch_indices = []

            for lp in range(n_here):
                pt = chunk_start + lp
                lp0, lp1 = lp * tile_height, (lp + 1) * tile_height

                for rt in range(n_rt):
                    lr0, lr1 = rt * tile_width, (rt + 1) * tile_width

                    tile = np.ascontiguousarray(
                        raw_chunk[lp0:lp1, lr0:lr1]).astype(np.complex64)
                    valid = (np.ascontiguousarray(mask_chunk[lp0:lp1, lr0:lr1])
                            if mask_chunk is not None
                            else np.ones_like(tile, dtype=bool))

                    batch_tiles.append(tile)
                    batch_valid.append(valid)
                    batch_indices.append(k)

                    tile_pulse[k] = p_start + pt * tile_height
                    tile_range[k] = r_start + lr0
                    k += 1

            # Run batch inference
            if batch_tiles:
                inputs = []
                for tile, valid in zip(batch_tiles, batch_valid):
                    x = prepare_tile_for_unet(tile, valid)
                    inputs.append(x)

                inputs = torch.from_numpy(np.stack(inputs, axis=0)).to(device)
                logits = model(inputs)
                probs = torch.sigmoid(logits).cpu().numpy()[:, 0]  # (B, H, W)

                # Store predictions
                for i, idx in enumerate(batch_indices):
                    probs_all[idx] = probs[i]
                    valid = batch_valid[i]
                    if valid.sum() > 0:
                        flagged = (probs[i] >= args.threshold) & valid
                        contamination_fractions[idx] = flagged.sum() / valid.sum()
                        contaminated_samples[idx] = flagged.sum()

            print(f"    pulse tiles {chunk_start + n_here}/{n_pt}")

    return {
        'freq': freq, 'pol': pol, 'chan': f'{freq}-{pol}',
        'probs': probs_all,
        'contamination_fractions': contamination_fractions,
        'contaminated_samples': contaminated_samples,
        'tile_pulse': tile_pulse,
        'tile_range': tile_range,
        'n_pt': n_pt, 'n_rt': n_rt,
        'pulse_window': [p_start, p_end],
        'range_window': [r_start, r_end],
        'tile_height': tile_height,
        'tile_width': tile_width,
    }


def save_predictions_h5(rec, args, out_dir):
    """Save per-tile predictions to HDF5."""
    path = os.path.join(out_dir, f"predictions_{rec['freq']}_{rec['pol']}.h5")
    with h5py.File(path, 'w') as f:
        f.attrs['granule'] = os.path.basename(args.l0b_file)
        f.attrs['model'] = os.path.basename(args.model)
        f.attrs['labeled'] = False
        f.attrs['note'] = ('Real scene, no labels: contamination fractions are '
                          'model predictions, not ground truth')
        f.attrs['frequency'] = rec['freq']
        f.attrs['polarization'] = rec['pol']
        f.attrs['pulse_start'] = rec['pulse_window'][0]
        f.attrs['pulse_end'] = rec['pulse_window'][1]
        f.attrs['range_start'] = rec['range_window'][0]
        f.attrs['range_end'] = rec['range_window'][1]
        f.attrs['n_pulse_tiles'] = rec['n_pt']
        f.attrs['n_range_tiles'] = rec['n_rt']
        f.attrs['tile_height'] = rec['tile_height']
        f.attrs['tile_width'] = rec['tile_width']
        f.attrs['caltone_removed'] = bool(args.remove_caltone)
        f.attrs['gap_exclusion_used'] = bool(args.compute_subswath_mask)

        # Don't save full probability maps (huge), just summary stats
        f.create_dataset('contamination_fraction',
                        data=rec['contamination_fractions'])
        f.create_dataset('contaminated_samples',
                        data=rec['contaminated_samples'])
        f.create_dataset('tile_pulse', data=rec['tile_pulse'])
        f.create_dataset('tile_range', data=rec['tile_range'])

    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def plot_contamination_map(rec, out_dir):
    """Spatial heatmap of contamination fraction."""
    grid = rec['contamination_fractions'].reshape(rec['n_pt'], rec['n_rt'])

    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(grid, aspect='auto', cmap='YlOrRd', origin='upper',
                   vmin=0, vmax=0.5, interpolation='nearest',
                   extent=[0, rec['n_rt'], rec['n_pt'], 0])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Contamination fraction (RFI samples / valid samples)')

    ax.set_xlabel('Range tile index')
    ax.set_ylabel('Pulse tile index')

    mean_contam = rec['contamination_fractions'].mean()
    n_flagged = (rec['contamination_fractions'] > 0.01).sum()

    ax.set_title(f"UNet RFI Contamination Map -- {rec['chan']} (REAL DATA, NO LABELS)\n"
                 f"Mean contamination: {mean_contam:.1%}, "
                 f"{n_flagged}/{len(rec['contamination_fractions'])} tiles with RFI")

    fig.tight_layout()
    path = os.path.join(out_dir, f"contamination_map_{rec['freq']}_{rec['pol']}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_contamination_histogram(rec, out_dir):
    """Distribution of contamination fractions."""
    contamination = rec['contamination_fractions']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(contamination, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(contamination.mean(), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {contamination.mean():.1%}')
    ax1.axvline(np.median(contamination), color='orange', linestyle='--',
                linewidth=2, label=f'Median: {np.median(contamination):.1%}')
    ax1.set_xlabel('Contamination fraction')
    ax1.set_ylabel('Number of tiles')
    ax1.set_title(f'Distribution of tile contamination -- {rec["chan"]}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Cumulative
    sorted_contam = np.sort(contamination)
    cumulative = np.arange(1, len(sorted_contam) + 1) / len(sorted_contam)
    ax2.plot(sorted_contam, cumulative, linewidth=2, color='steelblue')
    ax2.axhline(0.5, color='orange', linestyle='--', alpha=0.5)
    ax2.axhline(0.9, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Contamination fraction')
    ax2.set_ylabel('Cumulative fraction of tiles')
    ax2.set_title('Cumulative distribution')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, f"contamination_hist_{rec['freq']}_{rec['pol']}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_zoom_contaminated_tiles(rec, raw, freq, pol, args, out_dir, indices):
    """
    Create detailed zoom-in plots for the most contaminated tiles.

    For each tile, shows:
    - Magnitude in dB
    - Binary RFI mask
    - Probability heatmap (continuous 0-1)
    - Overlay of mask on magnitude
    """
    print(f"  Creating zoom-in plots for top {len(indices)} contaminated tiles...")

    # Build remover if needed
    dataset = raw.getRawDataset(freq, pol)
    total_range = dataset.shape[1]
    if args.remove_caltone:
        remover, _ = build_tone_remover(raw, freq, pol, total_range)
    else:
        remover = None

    for rank, idx in enumerate(indices):
        p0 = rec['tile_pulse'][idx]
        r0 = rec['tile_range'][idx]
        p1 = p0 + rec['tile_height']
        r1 = r0 + rec['tile_width']

        # Read tile
        if remover is not None:
            raw_full = np.ascontiguousarray(
                read_raw_data_batch(raw, freq, pol, slice(p0, p1), slice(0, total_range))
            ).astype(np.complex64)
            for ip in range(raw_full.shape[0]):
                raw_full[ip] = remover.remove_tone(raw_full[ip])
            tile = raw_full[:, r0:r1]
        else:
            tile = read_raw_data_batch(raw, freq, pol, slice(p0, p1), slice(r0, r1))

        valid = (get_subswath_mask(raw, freq, pol, np.arange(p0, p1), np.arange(r0, r1))
                if args.compute_subswath_mask
                else np.ones_like(tile, dtype=bool))

        prob = rec['probs'][idx]
        contamination = rec['contamination_fractions'][idx]

        # Magnitude in dB
        mag_db = 20.0 * np.log10(np.abs(tile) + 1e-12)
        mag_db = np.where(valid, mag_db, np.nan)

        # Binary mask
        mask = (prob >= args.threshold) & valid

        # Create 2x2 plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        vmin, vmax = np.nanpercentile(mag_db[valid], [1, 99])

        # Magnitude
        im0 = axes[0, 0].imshow(mag_db, aspect='auto', cmap='gray',
                                vmin=vmin, vmax=vmax, interpolation='nearest')
        axes[0, 0].set_title(f'Magnitude (dB)\nTile {idx} | pulse={p0}, range={r0}',
                            fontsize=11)
        axes[0, 0].set_ylabel('Pulse index')
        axes[0, 0].set_xlabel('Range sample')
        fig.colorbar(im0, ax=axes[0, 0], label='dB')

        # Binary mask
        im1 = axes[0, 1].imshow(mask, aspect='auto', cmap='Reds',
                                vmin=0, vmax=1, interpolation='nearest')
        n_flagged = mask.sum()
        n_valid = valid.sum()
        axes[0, 1].set_title(f'Binary RFI Mask (threshold={args.threshold})\n'
                            f'{n_flagged}/{n_valid} samples flagged ({contamination:.1%})',
                            fontsize=11)
        axes[0, 1].set_ylabel('Pulse index')
        axes[0, 1].set_xlabel('Range sample')
        fig.colorbar(im1, ax=axes[0, 1], label='RFI detected')

        # Probability heatmap
        prob_masked = np.where(valid, prob, np.nan)
        im2 = axes[1, 0].imshow(prob_masked, aspect='auto', cmap='plasma',
                                vmin=0, vmax=1, interpolation='nearest')
        axes[1, 0].set_title('RFI Probability (continuous)\n'
                            f'Mean prob = {prob[valid].mean():.3f}, '
                            f'Max prob = {prob[valid].max():.3f}',
                            fontsize=11)
        axes[1, 0].set_ylabel('Pulse index')
        axes[1, 0].set_xlabel('Range sample')
        fig.colorbar(im2, ax=axes[1, 0], label='Probability')

        # Overlay
        axes[1, 1].imshow(mag_db, aspect='auto', cmap='gray',
                         vmin=vmin, vmax=vmax, interpolation='nearest', alpha=0.6)
        im3 = axes[1, 1].imshow(mask, aspect='auto', cmap='Reds',
                                vmin=0, vmax=1, interpolation='nearest', alpha=0.6)
        axes[1, 1].set_title(f'Overlay: Mask on Magnitude\n'
                            f'Rank #{rank+1} most contaminated tile',
                            fontsize=11)
        axes[1, 1].set_ylabel('Pulse index')
        axes[1, 1].set_xlabel('Range sample')

        fig.suptitle(f'Zoom-in: Tile {idx} -- {rec["chan"]} '
                    f'(contamination = {contamination:.1%})',
                    fontsize=14, fontweight='bold')
        fig.tight_layout()

        path = os.path.join(out_dir,
                           f"zoom_tile_{idx:04d}_{rec['freq']}_{rec['pol']}.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"  ✓ Saved {len(indices)} zoom-in plots")


def plot_sample_predictions(rec, raw, freq, pol, args, out_dir, n_samples=12, seed=42):
    """Grid of sample tiles with predicted masks."""
    rng = np.random.default_rng(seed)
    contamination = rec['contamination_fractions']

    # Sample from different contamination ranges
    n_per_range = max(1, n_samples // 3)
    indices = []

    low_idx = np.where(contamination < 0.05)[0]
    if len(low_idx) > 0:
        indices.extend(rng.choice(low_idx, size=min(n_per_range, len(low_idx)),
                                 replace=False))

    medium_idx = np.where((contamination >= 0.05) & (contamination < 0.2))[0]
    if len(medium_idx) > 0:
        indices.extend(rng.choice(medium_idx, size=min(n_per_range, len(medium_idx)),
                                 replace=False))

    high_idx = np.where(contamination >= 0.2)[0]
    if len(high_idx) > 0:
        indices.extend(rng.choice(high_idx, size=min(n_per_range, len(high_idx)),
                                 replace=False))

    if not indices:
        print("  Warning: No tiles for sample predictions")
        return

    indices = indices[:n_samples]

    # Also get the top contaminated tiles for zoom-in feature
    top_contaminated_indices = np.argsort(contamination)[::-1][:args.n_zoom]

    # Re-read tiles from L0B
    print(f"  Reading {len(indices)} sample tiles for visualization...")

    # Build remover if needed
    dataset = raw.getRawDataset(freq, pol)
    total_range = dataset.shape[1]
    if args.remove_caltone:
        remover, _ = build_tone_remover(raw, freq, pol, total_range)
    else:
        remover = None

    n_rows = len(indices)
    fig = plt.figure(figsize=(16, 2.5 * n_rows))
    gs = GridSpec(n_rows, 3, figure=fig, width_ratios=[1, 1, 1])

    for i, idx in enumerate(indices):
        p0 = rec['tile_pulse'][idx]
        r0 = rec['tile_range'][idx]
        p1 = p0 + rec['tile_height']
        r1 = r0 + rec['tile_width']

        # Read tile
        if remover is not None:
            raw_full = np.ascontiguousarray(
                read_raw_data_batch(raw, freq, pol, slice(p0, p1), slice(0, total_range))
            ).astype(np.complex64)
            for ip in range(raw_full.shape[0]):
                raw_full[ip] = remover.remove_tone(raw_full[ip])
            tile = raw_full[:, r0:r1]
        else:
            tile = read_raw_data_batch(raw, freq, pol, slice(p0, p1), slice(r0, r1))

        valid = (get_subswath_mask(raw, freq, pol, np.arange(p0, p1), np.arange(r0, r1))
                if args.compute_subswath_mask
                else np.ones_like(tile, dtype=bool))

        prob = rec['probs'][idx]

        # Magnitude in dB
        mag_db = 20.0 * np.log10(np.abs(tile) + 1e-12)
        mag_db = np.where(valid, mag_db, np.nan)

        # Mask
        mask = (prob >= args.threshold) & valid
        contam_frac = contamination[idx]

        # Plot magnitude
        ax1 = fig.add_subplot(gs[i, 0])
        vmin, vmax = np.nanpercentile(mag_db[valid], [1, 99])
        ax1.imshow(mag_db, aspect='auto', cmap='gray',
                  vmin=vmin, vmax=vmax, interpolation='nearest')
        ax1.set_title(f'Tile {idx}: Magnitude (dB)\np={p0}, r={r0}', fontsize=9)
        ax1.set_ylabel('Pulse')
        if i == n_rows - 1:
            ax1.set_xlabel('Range sample')

        # Plot mask
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.imshow(mask, aspect='auto', cmap='Reds',
                  vmin=0, vmax=1, interpolation='nearest')
        ax2.set_title(f'Predicted RFI Mask\nContamination: {contam_frac:.1%}',
                     fontsize=9)
        ax2.set_ylabel('Pulse')
        if i == n_rows - 1:
            ax2.set_xlabel('Range sample')

        # Plot overlay
        ax3 = fig.add_subplot(gs[i, 2])
        ax3.imshow(mag_db, aspect='auto', cmap='gray',
                  vmin=vmin, vmax=vmax, interpolation='nearest', alpha=0.7)
        ax3.imshow(mask, aspect='auto', cmap='Reds',
                  vmin=0, vmax=1, interpolation='nearest', alpha=0.5)
        ax3.set_title(f'Overlay\n{mask.sum()} / {valid.sum()} samples flagged',
                     fontsize=9)
        ax3.set_ylabel('Pulse')
        if i == n_rows - 1:
            ax3.set_xlabel('Range sample')

    fig.suptitle(f'UNet Predictions on Real Tiles -- {rec["chan"]} (seed={seed})',
                fontsize=12, y=0.995)
    fig.tight_layout()
    path = os.path.join(out_dir, f"sample_predictions_{rec['freq']}_{rec['pol']}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")

    # Zoom-in plots for top contaminated tiles
    if len(top_contaminated_indices) > 0:
        plot_zoom_contaminated_tiles(rec, raw, freq, pol, args, out_dir,
                                     top_contaminated_indices)


def report(recs, results):
    """Print summary statistics."""
    print(f"\n{'='*60}")
    print('SCENE SCORING  (unlabeled -- no accuracy can be computed)')
    print(f"{'='*60}")

    for rec in recs:
        contamination = rec['contamination_fractions']
        n = len(contamination)
        flagged = contamination > 0.01
        n_flag = int(flagged.sum())

        s = {
            'n_tiles': n,
            'flagged': n_flag,
            'flagged_fraction': float(n_flag / n),
            'mean_contamination': float(contamination.mean()),
            'median_contamination': float(np.median(contamination)),
            'max_contamination': float(contamination.max()),
        }
        results['channels'][rec['chan']] = s

        print(f"\n  {rec['chan']}: {n} tiles")
        print(f"    tiles with RFI   : {n_flag} ({100 * n_flag / n:.2f}%)")
        print(f"    mean contamination: {s['mean_contamination']:.2%}")
        print(f"    median contamination: {s['median_contamination']:.2%}")
        print(f"    max contamination: {s['max_contamination']:.2%}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Score a NISAR L0B scene with UNet semantic segmentation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('l0b_file', help='Input NISAR L0B HDF5 granule')
    parser.add_argument('--model', required=True, help='Trained UNet model (.pth)')

    parser.add_argument('--freq', choices=['A', 'B'], default=None)
    parser.add_argument('--pol', default=None)

    parser.add_argument('--pulse-start', type=int, default=None)
    parser.add_argument('--pulse-end', type=int, default=None)
    parser.add_argument('--range-start', type=int, default=None)
    parser.add_argument('--range-end', type=int, default=None)

    parser.add_argument('--tile-height', type=int, default=256,
                       help='Tile height in pulses')
    parser.add_argument('--tile-width', type=int, default=256,
                       help='Tile width in range samples')
    parser.add_argument('--pulse-chunk', type=int, default=PULSE_CHUNK_DEFAULT)

    parser.add_argument('--remove-caltone', dest='remove_caltone',
                       action='store_true', default=True)
    parser.add_argument('--no-remove-caltone', dest='remove_caltone',
                       action='store_false')
    parser.add_argument('--compute-subswath-mask', dest='compute_subswath_mask',
                       action='store_true', default=True)
    parser.add_argument('--no-compute-subswath-mask', dest='compute_subswath_mask',
                       action='store_false')

    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Probability threshold for binary classification')
    parser.add_argument('--n-samples', type=int, default=12,
                       help='Number of sample tiles to visualize')
    parser.add_argument('--sample-seed', type=int, default=42)
    parser.add_argument('--n-zoom', type=int, default=10,
                       help='Number of top contaminated tiles for zoom-in plots')

    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--output-dir', default='results/unet_scene')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print('UNet scene scoring (UNLABELED)')
    print(f"{'='*70}")
    print(f"  granule : {args.l0b_file}")
    print(f"  model   : {args.model}")
    print(f"  device  : {device}")

    pulse_str = (f"[{args.pulse_start if args.pulse_start is not None else 'full'}, "
                 f"{args.pulse_end if args.pulse_end is not None else 'full'})")
    print(f"  pulses  : {pulse_str}")

    # Load model (matching training architecture)
    model = UNet(in_channels=2, out_channels=1, features=[64, 128, 256, 512])
    checkpoint = torch.load(args.model, map_location=device)
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    print(f"  Model loaded: {model.n_parameters():,} parameters")

    # Open L0B
    raw = Raw(hdf5file=args.l0b_file)
    raw.parsePolarizations()

    freqs = [args.freq] if args.freq else list(raw.polarizations.keys())
    channels = []
    for freq in freqs:
        if freq not in raw.polarizations:
            continue
        pols = ([args.pol] if args.pol and args.pol in raw.polarizations[freq]
                else list(raw.polarizations[freq]))
        channels.extend((freq, p) for p in pols)

    if not channels:
        raise ValueError('No matching channels in this granule')
    print("  channels: " + ', '.join(f'{f}-{p}' for f, p in channels))

    # Score channels
    recs = []
    for f, p in channels:
        rec = score_channel_unet(raw, f, p, model, args, device=device)
        recs.append(rec)

    # Save and plot
    results = {
        'granule': os.path.basename(args.l0b_file),
        'model': args.model,
        'labeled': False,
        'pulse_window': recs[0]['pulse_window'],
        'range_window': recs[0]['range_window'],
        'tile_height': recs[0]['tile_height'],
        'tile_width': recs[0]['tile_width'],
        'channels': {},
    }

    for rec in recs:
        save_predictions_h5(rec, args, args.output_dir)
        plot_contamination_map(rec, args.output_dir)
        plot_contamination_histogram(rec, args.output_dir)
        plot_sample_predictions(rec, raw, rec['freq'], rec['pol'], args,
                               args.output_dir, args.n_samples, args.sample_seed)

    report(recs, results)

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved to {os.path.join(args.output_dir, 'results.json')}")


if __name__ == '__main__':
    main()
