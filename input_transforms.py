#!/usr/bin/env python
"""
input_transforms.py

Input preprocessing functions for UNet models.

Two input representations:

1. prepare_tile_2channel() - [real, imag] for UNet (2-channel)
   Used by train_unet_2channel.py and score_unet_scene.py

2. build_input_channels() - [mag_db, cos/sin phase, valid] for SegUNet (4-channel)
   For future use with train_unet_4channel.py
"""

from typing import Optional, Tuple
import numpy as np

EPS = 1e-12

# For 4-channel input
MAG_SCALE_DB = 20.0
MAG_CLIP = (-2.0, 6.0)
FLOOR_QUANTILE = 0.5
N_CHANNELS_DEFAULT = 4


# ===========================================================================
# 2-CHANNEL INPUT (real, imag) - CURRENT TRAINED MODEL
# ===========================================================================

def prepare_tile_2channel(tile: np.ndarray, valid: np.ndarray = None) -> np.ndarray:
    """
    Convert complex tile to [real, imag] channels (2-channel input for UNet).

    This is what train_unet_2channel.py and the current trained model use.

    Parameters
    ----------
    tile : (P, K) complex64
        Raw complex tile
    valid : (P, K) bool, optional
        Validity mask (used for normalization)

    Returns
    -------
    (2, P, K) float32
        Channel 0: Real part (normalized)
        Channel 1: Imaginary part (normalized)
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


# ===========================================================================
# 4-CHANNEL INPUT (mag_db, cos/sin phase, valid) - FUTURE USE
# ===========================================================================

def _floor_relative_log_magnitude(tile: np.ndarray,
                                  valid: Optional[np.ndarray]) -> np.ndarray:
    """
    Per-tile floor relative log magnitude in scaled dB.

    The floor is estimated over VALID samples only, so an ADC gap (which is
    near zero power) cannot drag the floor estimate down and inflate the
    apparent contrast of everything else.
    """
    mag_db = 20.0 * np.log10(np.abs(tile) + EPS)

    if valid is not None and valid.any():
        floor = float(np.quantile(mag_db[valid], FLOOR_QUANTILE))
    else:
        floor = float(np.quantile(mag_db, FLOOR_QUANTILE))

    out = (mag_db - floor) / MAG_SCALE_DB
    return np.clip(out, MAG_CLIP[0], MAG_CLIP[1]).astype(np.float32)


def _adjacent_pulse_phase_diff(tile: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    cos and sin of arg(x[m] * conj(x[m-1])) at every range sample.

    Row 0 has no predecessor and is replicated from row 1, so the output
    keeps the tile shape. For a single-pulse tile the phase difference is
    undefined and both channels are returned as zero.
    """
    P = tile.shape[0]
    if P < 2:
        zeros = np.zeros(tile.shape, dtype=np.float32)
        return zeros, zeros.copy()

    prod = tile[1:] * np.conj(tile[:-1])
    phase = np.angle(prod)

    cos_d = np.empty(tile.shape, dtype=np.float32)
    sin_d = np.empty(tile.shape, dtype=np.float32)

    cos_d[1:] = np.cos(phase)
    sin_d[1:] = np.sin(phase)
    cos_d[0] = cos_d[1]
    sin_d[0] = sin_d[1]

    return cos_d, sin_d


def build_input_channels(tile: np.ndarray,
                         valid: Optional[np.ndarray] = None,
                         n_channels: int = N_CHANNELS_DEFAULT) -> np.ndarray:
    """
    Complex time-domain tile to multi-channel network input for SegUNet.

    Parameters
    ----------
    tile : (P, K) complex
        Raw pulse x range sample data. Caltone should already be removed.
    valid : (P, K) bool or None
        ADC gap / subswath validity. None = fully valid.
    n_channels : int, default=4
        Number of channels:
        - 2: real and imaginary parts (normalized)
        - 4: magnitude + phase + validity

    Returns
    -------
    (n_channels, P, K) float32
        For n_channels == 2:
            Channel 0: Real part (normalized)
            Channel 1: Imaginary part (normalized)
        For n_channels == 4:
            Channel 0: Floor-relative log magnitude (dB, scaled)
            Channel 1: cos of adjacent-pulse phase difference
            Channel 2: sin of adjacent-pulse phase difference
            Channel 3: Validity mask
    """
    if tile.ndim != 2:
        raise ValueError(f"tile must be 2-D (P, K), got shape {tile.shape}")
    if n_channels not in (2, 4):
        raise ValueError(f"n_channels must be 2 or 4, got {n_channels}")

    # 2 channels: real/imag representation
    if n_channels == 2:
        return prepare_tile_2channel(tile, valid)

    # 4 channels: magnitude + phase + validity
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != tile.shape:
            raise ValueError(
                f"valid mask shape {valid.shape} != tile shape {tile.shape}")

    mag = _floor_relative_log_magnitude(tile, valid)
    cos_d, sin_d = _adjacent_pulse_phase_diff(tile)

    if valid is None:
        valid_channel = np.ones(tile.shape, dtype=np.float32)
    else:
        valid_channel = valid.astype(np.float32)

    stacked = np.stack([mag, cos_d, sin_d, valid_channel], axis=0)

    # Zero the data-derived channels outside the valid window
    if valid is not None:
        stacked[:3] *= valid.astype(np.float32)[None]

    return np.ascontiguousarray(stacked)
