#!/usr/bin/env python
"""
score_unet_scene_4channel.py

============================================================================
SCORING SCRIPT FOR 4-CHANNEL UNET ON REAL L0B SCENES
============================================================================

Run 4-channel UNet semantic segmentation on a NISAR L0B granule.
This is the 4-channel equivalent of score_unet_scene.py.

Key differences from 2-channel version:
- Uses build_input_channels() for 4-channel preprocessing
- Input: [mag_db, cos_phase, sin_phase, valid]
- SAME UNet architecture as 2-channel for fair comparison

Usage:
    python score_unet_scene_4channel.py \
        /path/to/NISAR_L0B.h5 \
        --model model/four_channel/best_model.pth \
        --pulse-start 363793 --pulse-end 440011 \
        --freq A --pol HH \
        --output-dir score/four_channel/unet_vienna

Outputs are identical to the 2-channel version for easy comparison.
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

# Import UNet architecture and 4-channel preprocessing
import torch.nn as nn
import torch.nn.functional as F
from unet import UNet
from input_transforms import build_input_channels

# Constants
EPS = 1e-12
CPI_LEN_DEFAULT = 16
CPI_WIDTH_DEFAULT = 250
PULSE_CHUNK_DEFAULT = 1600

# Use 4-channel preprocessing
prepare_tile_for_unet = build_input_channels


# ---------------------------------------------------------------------------
# NISAR UTILITY FUNCTIONS (same as 2-channel version)
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

# Caltone removal (same as 2-channel)
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
    Stream one channel, tile it, run 4-channel UNet, and return predictions.

    Key difference from 2-channel: uses build_input_channels() for preprocessing.
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

            # Run batch inference with 4-channel preprocessing
            if batch_tiles:
                inputs = []
                for tile, valid in zip(batch_tiles, batch_valid):
                    # KEY DIFFERENCE: use build_input_channels for 4-channel input
                    x = prepare_tile_for_unet(tile, valid, n_channels=4)
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


# ---------------------------------------------------------------------------
# SAVE & PLOT (reuse functions from 2-channel version)
# ---------------------------------------------------------------------------

# Import plotting functions from the 2-channel script
import sys
sys.path.insert(0, os.path.dirname(__file__))
from score_unet_scene_2channel import (
    save_predictions_h5,
    plot_contamination_map,
    plot_contamination_histogram,
    plot_zoom_contaminated_tiles,
    plot_sample_predictions,
    report,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Score a NISAR L0B scene with 4-channel UNet.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('l0b_file', help='Input NISAR L0B HDF5 granule')
    parser.add_argument('--model', required=True, help='Trained SegUNet model (.pth)')

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
    parser.add_argument('--output-dir', default='score/four_channel/unet_scene')

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print('4-CHANNEL UNet scene scoring (UNLABELED)')
    print(f"{'='*70}")
    print(f"  granule : {args.l0b_file}")
    print(f"  model   : {args.model}")
    print(f"  device  : {device}")

    pulse_str = (f"[{args.pulse_start if args.pulse_start is not None else 'full'}, "
                 f"{args.pulse_end if args.pulse_end is not None else 'full'})")
    print(f"  pulses  : {pulse_str}")

    # Load 4-channel UNet model (fixed architecture matching 2-channel)
    UNET_FEATURES = [64, 128, 256, 512]
    model = UNet(in_channels=4, out_channels=1, features=UNET_FEATURES)
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
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model loaded: {n_params:,} parameters (4-ch, features={UNET_FEATURES})")

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
        'model_type': '4-channel UNet',
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
