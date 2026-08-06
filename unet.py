#!/usr/bin/env python
"""
seg_unet.py

Baseline semantic segmentation model for per-sample RFI detection on raw
NISAR L0B tiles, operating entirely in the TIME DOMAIN (pulse x range
sample). No Fourier transform is taken anywhere in this file.

Scope
-----
This module holds three things and nothing else, so that training,
evaluation, and inference scripts can all import the SAME definitions and
cannot drift apart:

  1. build_input_channels() -- the complex tile to network input transform.
     Train/serve skew in this function is the single most likely source of a
     silent accuracy loss on real scenes, so it lives in exactly one place
     and is used by every consumer.
  2. SegUNet -- the network.
  3. masked_seg_loss() -- masked BCE + soft Dice, weighted so that false
     positives (over mitigation) cost more than false negatives.

Input representation
--------------------
A tile is a complex64 array of shape (P, K): P pulses (azimuth slow time) by
K raw range samples (fast time). P is a whole number of CPI blocks so that a
per-block knee label can be derived from the predicted mask, and K is padded
to a power-of-two friendly width by the generator.

Four channels are formed:

  0  Floor relative log magnitude, dB, scaled.
     In raw L0B the backdrop is uncompressed chirp returns, so it is speckly
     and roughly stationary in range. Wideband RFI raises the speckle level;
     narrowband RFI adds a constant modulus component, which REDUCES local
     amplitude fluctuation. The narrowband cue in the time domain is
     therefore partly a texture cue, not purely a brightness cue.

  1  cos of the adjacent-pulse phase difference at each range sample.
  2  sin of the same.
     These are the slow-time coherence channels, and they are the direct
     analog of what the SCM off-diagonal terms measure. An emitter with
     Doppler fd produces a near constant phase difference across range
     samples in the pulses where it dominates. Caveat: terrain returns in
     raw data are ALSO pulse-to-pulse correlated, which is why the SCM has
     structure at all, so this is not a clean RFI-only cue. Run the
     magnitude-only ablation (n_channels=1) before assuming these earn their
     place.

  3  Validity mask (ADC gap / subswath), 0 or 1.
     Supplied as an input AND used to mask the loss. The network should not
     have to infer where the echo window is.

Channels 1 and 2 are undefined for the first pulse; row 0 is replicated from
row 1 so all channels share the tile shape.

Architecture
------------
Symmetric 3-level UNet. At 16 x 250 an anisotropic pooling schedule was
needed, but a tile of a few hundred pulses by a few hundred range samples is
close to square, so symmetric pooling is correct here.

Deliberately small (roughly 0.5M parameters at base_channels=16). The first
dataset cut is a few thousand tiles; a ResNet34 style encoder would memorize
it. Add capacity only after the IoU vs JSR curve demonstrably plateaus.

Bilinear upsampling plus a 3x3 conv, NOT transposed convolution: transposed
conv produces checkerboard artifacts whose periodic structure looks exactly
like a real RFI signature, which is an expensive thing to debug.

GroupNorm rather than BatchNorm because tiles are large and batch sizes are
correspondingly small.

Usage
-----
    from seg_unet import SegUNet, build_input_channels, masked_seg_loss

    x = build_input_channels(tile_complex, valid_mask)   # (4, P, K) float32
    logits = model(x[None])                              # (1, 1, P, K)
    loss = masked_seg_loss(logits, target, valid_mask)

Running this file directly performs a shape and gradient smoke test and
prints the parameter count and receptive field.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

EPS = 1e-12

# Magnitude channel scaling. After subtracting the per-tile floor, the log
# magnitude is divided by MAG_SCALE_DB and clipped to MAG_CLIP. A 20 dB scale
# puts typical clutter near 0 and strong RFI near 1 to 2 without saturating.
MAG_SCALE_DB = 20.0
MAG_CLIP = (-2.0, 6.0)

# Quantile used as the per-tile noise floor proxy on the log magnitude. The
# median is robust while RFI occupies a minority of samples; lower it if very
# wide, very strong contamination is common in the training distribution.
FLOOR_QUANTILE = 0.5

# Default channel count. Set to 1 for the magnitude-only ablation, 3 to drop
# the validity channel, 4 for the full stack.
N_CHANNELS_DEFAULT = 4


# ---------------------------------------------------------------------------
# INPUT CONSTRUCTION
# ---------------------------------------------------------------------------

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
    Complex time-domain tile to network input.

    This is the ONLY place the transform is defined. Training, evaluation,
    and real-scene inference must all call it, or the model will silently see
    a different distribution at test time than it was trained on.

    Parameters
    ----------
    tile : (P, K) complex, raw pulse x range sample data. Caltone should
        already have been removed upstream, as in the detection path.
    valid : (P, K) bool or None. ADC gap / subswath validity. None is treated
        as fully valid.
    n_channels : 1 (magnitude only), 3 (magnitude + phase), or 4 (full).

    Returns
    -------
    (n_channels, P, K) float32
    """
    if tile.ndim != 2:
        raise ValueError(f"tile must be 2-D (P, K), got shape {tile.shape}")
    if n_channels not in (1, 3, 4):
        raise ValueError(f"n_channels must be 1, 3, or 4, got {n_channels}")

    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != tile.shape:
            raise ValueError(
                f"valid mask shape {valid.shape} != tile shape {tile.shape}")

    chans = [_floor_relative_log_magnitude(tile, valid)]

    if n_channels >= 3:
        cos_d, sin_d = _adjacent_pulse_phase_diff(tile)
        chans.extend([cos_d, sin_d])

    if n_channels >= 4:
        if valid is None:
            chans.append(np.ones(tile.shape, dtype=np.float32))
        else:
            chans.append(valid.astype(np.float32))

    stacked = np.stack(chans, axis=0)

    # Zero the data-derived channels outside the valid window so the network
    # is not fed the gap's numerical noise. The validity channel itself stays
    # intact, since that is what tells the network the gap is there.
    if valid is not None:
        n_data = min(n_channels, 3)
        stacked[:n_data] *= valid.astype(np.float32)[None]

    return np.ascontiguousarray(stacked)


# ---------------------------------------------------------------------------
# BUILDING BLOCKS
# ---------------------------------------------------------------------------

def _group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """GroupNorm with a group count that always divides num_channels."""
    groups = math.gcd(num_channels, max_groups)
    return nn.GroupNorm(max(groups, 1), num_channels)


class DoubleConv(nn.Module):
    """(conv 3x3 -> GroupNorm -> ReLU) x 2, shape preserving."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _group_norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _group_norm(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Max pool by 2 in both axes, then DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """
    Bilinear upsample, concatenate the skip connection, then DoubleConv.

    Transposed convolution is avoided on purpose: its checkerboard artifacts
    are periodic in both axes and are easily mistaken for a genuine
    interference signature in the predicted mask.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm = _group_norm(out_ch)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear',
                          align_corners=False)
        x = F.relu(self.norm(self.reduce(x)), inplace=True)
        return self.conv(torch.cat([x, skip], dim=1))


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------

class SegUNet(nn.Module):
    """
    Baseline 3-level symmetric UNet for per-sample RFI segmentation.

    Parameters
    ----------
    in_channels : must match the n_channels passed to build_input_channels.
    out_channels : 1 for the binary baseline. Set to 3 later for the
        clean / narrowband / wideband split, which is a head-only change;
        masked_seg_loss is binary, so a multi-class run needs its own loss.
    base_channels : width of the first encoder stage. Channels double at
        every level. 16 gives roughly 0.5M parameters.
    depth : number of downsampling levels. Tile dimensions must be divisible
        by 2 ** depth.

    Notes
    -----
    Input dimensions do not have to be divisible by 2 ** depth for the
    forward pass to run, because Up resizes to the skip tensor's shape, but
    keeping them divisible avoids repeated non-integer resampling. Pad the
    range width in the generator rather than here.
    """

    def __init__(self,
                 in_channels: int = N_CHANNELS_DEFAULT,
                 out_channels: int = 1,
                 base_channels: int = 16,
                 depth: int = 3):
        super().__init__()

        if depth < 1:
            raise ValueError("depth must be >= 1")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.depth = depth

        widths = [base_channels * (2 ** i) for i in range(depth + 1)]

        self.stem = DoubleConv(in_channels, widths[0])

        self.downs = nn.ModuleList([
            Down(widths[i], widths[i + 1]) for i in range(depth)
        ])

        self.ups = nn.ModuleList([
            Up(in_ch=widths[i + 1], skip_ch=widths[i], out_ch=widths[i])
            for i in reversed(range(depth))
        ])

        self.head = nn.Conv2d(widths[0], out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Start with a strong negative bias on the output so the initial
        # prediction is "clean everywhere". Most samples are clean, so this
        # avoids a large early loss spike and the collapse that can follow.
        nn.init.constant_(self.head.bias, -4.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, in_channels, P, K)

        Returns
        -------
        (B, out_channels, P, K) raw logits. Apply sigmoid outside, or use
        masked_seg_loss which expects logits.
        """
        skips = []
        h = self.stem(x)
        for down in self.downs:
            skips.append(h)
            h = down(h)

        for up, skip in zip(self.ups, reversed(skips)):
            h = up(h, skip)

        return self.head(h)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def receptive_field(self) -> int:
        """
        Theoretical receptive field of the encoder path, in input pixels,
        along one axis. Reported so the azimuth extent can be checked against
        the tile height: the whole reason for a multi-hundred-pulse tile is
        that the network can see an emitter persist, so the receptive field
        should be a large fraction of the tile height.
        """
        rf, jump = 1, 1
        for _ in range(2):                       # stem: two 3x3 convs
            rf += 2 * jump
        for _ in range(self.depth):
            jump *= 2                            # pool by 2
            for _ in range(2):                   # two 3x3 convs
                rf += 2 * jump
        return rf


# ---------------------------------------------------------------------------
# LOSS
# ---------------------------------------------------------------------------

def masked_seg_loss(logits: torch.Tensor,
                    target: torch.Tensor,
                    valid: Optional[torch.Tensor] = None,
                    pos_weight: float = 0.5,
                    dice_weight: float = 1.0,
                    bce_weight: float = 1.0,
                    dice_smooth: float = 1.0) -> torch.Tensor:
    """
    Masked binary cross entropy plus soft Dice.

    Parameters
    ----------
    logits : (B, 1, P, K) raw output of SegUNet.
    target : (B, 1, P, K) float in {0, 1}, the per-sample RFI mask.
    valid : (B, 1, P, K) float or bool, or None. Invalid samples (ADC gap)
        contribute nothing to either term.
    pos_weight : multiplier on the POSITIVE class term of the BCE. Values
        BELOW 1 make a false positive more costly than a false negative,
        which is the intended asymmetry: over mitigation destroys clean
        signal and is the worse failure for InSAR products. Tune this to pin
        the false positive rate on held-out clean tiles rather than to
        maximize IoU.
    dice_weight, bce_weight : relative term weights.
    dice_smooth : Dice numerator/denominator smoothing. Also what makes an
        all-clean tile (no positives at all) yield a finite, well behaved
        Dice term instead of a division by zero.

    Returns
    -------
    Scalar loss.

    Notes
    -----
    Normalization is by the COUNT OF VALID SAMPLES, not by the tensor size.
    Dividing by the tensor size would systematically down-weight tiles with
    large ADC gaps, which are exactly the tiles where the gap geometry
    matters most.
    """
    if logits.shape != target.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} != target shape "
            f"{tuple(target.shape)}")

    target = target.to(dtype=logits.dtype)

    if valid is None:
        mask = torch.ones_like(logits)
    else:
        mask = valid.to(dtype=logits.dtype)
        if mask.shape != logits.shape:
            raise ValueError(
                f"valid shape {tuple(mask.shape)} != logits shape "
                f"{tuple(logits.shape)}")

    n_valid = mask.sum().clamp(min=1.0)

    pw = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    bce_map = F.binary_cross_entropy_with_logits(
        logits, target, weight=None, pos_weight=pw, reduction='none')
    bce = (bce_map * mask).sum() / n_valid

    probs = torch.sigmoid(logits) * mask
    tgt = target * mask
    dims = tuple(range(1, probs.dim()))
    intersection = (probs * tgt).sum(dim=dims)
    denom = probs.sum(dim=dims) + tgt.sum(dim=dims)
    dice = 1.0 - ((2.0 * intersection + dice_smooth) / (denom + dice_smooth))
    dice = dice.mean()

    return bce_weight * bce + dice_weight * dice


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------

@torch.no_grad()
def mask_metrics(logits: torch.Tensor,
                 target: torch.Tensor,
                 valid: Optional[torch.Tensor] = None,
                 threshold: float = 0.5) -> dict:
    """
    Per-sample IoU, precision, recall, and false positive rate over valid
    samples. The false positive rate is reported separately because it is the
    quantity that governs over mitigation, and it should be pinned to a fixed
    target on held-out CLEAN tiles before any other metric is compared.
    """
    pred = (torch.sigmoid(logits) >= threshold).to(dtype=logits.dtype)
    tgt = target.to(dtype=logits.dtype)

    if valid is None:
        mask = torch.ones_like(logits)
    else:
        mask = valid.to(dtype=logits.dtype)

    pred = pred * mask
    tgt = tgt * mask

    tp = (pred * tgt).sum()
    fp = (pred * (1.0 - tgt) * mask).sum()
    fn = ((1.0 - pred) * tgt * mask).sum()
    tn = ((1.0 - pred) * (1.0 - tgt) * mask).sum()

    return {
        'iou': float(tp / (tp + fp + fn).clamp(min=1.0)),
        'precision': float(tp / (tp + fp).clamp(min=1.0)),
        'recall': float(tp / (tp + fn).clamp(min=1.0)),
        'fpr': float(fp / (fp + tn).clamp(min=1.0)),
        'n_positive': float(tgt.sum()),
        'n_valid': float(mask.sum()),
    }


@torch.no_grad()
def knee_from_mask(prob: np.ndarray,
                   cpi_len: int,
                   threshold: float = 0.5,
                   min_frac: float = 0.10) -> np.ndarray:
    """
    Derive a per-CPI-block knee from a predicted mask.

    This is the bridge metric: it converts a segmentation output into the
    same quantity the eigenvalue CNN predicts, so the two approaches can be
    scored against each other on the same test set. It is deliberately crude
    -- a pulse counts as contaminated if at least min_frac of its valid range
    samples are flagged, and the block knee is the number of such pulses,
    capped at cpi_len.

    A stronger version estimates the rank of the masked RFI component
    directly; keep that for the mitigation stage, where the subspace is
    needed anyway.

    Parameters
    ----------
    prob : (P, K) predicted probabilities for one tile.
    cpi_len : pulses per CPI block. P must be a whole number of blocks.

    Returns
    -------
    (P // cpi_len,) int array of per-block knees.
    """
    P, K = prob.shape
    if P % cpi_len != 0:
        raise ValueError(f"P ({P}) is not a whole number of cpi_len ({cpi_len}) blocks")

    flagged = (prob >= threshold)
    per_pulse = flagged.mean(axis=1) >= min_frac
    blocks = per_pulse.reshape(P // cpi_len, cpi_len)
    return np.minimum(blocks.sum(axis=1), cpi_len).astype(np.int32)


# ---------------------------------------------------------------------------
# SMOKE TEST
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    """Shape, gradient, and sanity checks. Not a substitute for training."""
    rng = np.random.default_rng(0)
    P, K, cpi_len = 256, 256, 16

    # Synthetic tile: complex Gaussian clutter, one constant modulus tone on
    # a run of pulses, and an ADC gap at the end of the range window.
    tile = (rng.standard_normal((P, K)) + 1j * rng.standard_normal((P, K))) / np.sqrt(2)
    tone = np.exp(2j * np.pi * (0.03 * np.arange(K)[None, :] + 0.11 * np.arange(P)[:, None]))
    rfi_rows = slice(64, 128)
    tile[rfi_rows] += 3.0 * tone[rfi_rows]

    valid = np.ones((P, K), dtype=bool)
    valid[:, -20:] = False
    tile[~valid] = 0.0

    target = np.zeros((P, K), dtype=np.float32)
    target[rfi_rows] = 1.0
    target[~valid] = 0.0

    for n_ch in (1, 3, 4):
        x = build_input_channels(tile, valid, n_channels=n_ch)
        assert x.shape == (n_ch, P, K), x.shape
        assert np.isfinite(x).all(), "non-finite value in input channels"

    x = build_input_channels(tile, valid, n_channels=4)
    xb = torch.from_numpy(x)[None]
    tb = torch.from_numpy(target)[None, None]
    vb = torch.from_numpy(valid.astype(np.float32))[None, None]

    model = SegUNet(in_channels=4, out_channels=1, base_channels=16, depth=3)
    logits = model(xb)
    assert logits.shape == (1, 1, P, K), logits.shape

    loss = masked_seg_loss(logits, tb, vb, pos_weight=0.5)
    loss.backward()

    n_grad = sum(1 for p in model.parameters()
                 if p.grad is not None and torch.isfinite(p.grad).all())
    n_param_tensors = sum(1 for p in model.parameters() if p.requires_grad)
    assert n_grad == n_param_tensors, "some parameters received no finite gradient"

    metrics = mask_metrics(logits.detach(), tb, vb)
    knees = knee_from_mask(torch.sigmoid(logits.detach())[0, 0].numpy(), cpi_len)

    print("seg_unet smoke test")
    print("-" * 52)
    print(f"  tile                : {P} pulses x {K} range samples")
    print(f"  input channels      : {x.shape[0]}")
    print(f"  parameters          : {model.n_parameters():,}")
    print(f"  receptive field     : {model.receptive_field()} px "
          f"({100.0 * model.receptive_field() / P:.0f}% of tile height)")
    print(f"  logits              : {tuple(logits.shape)}")
    print(f"  loss (untrained)    : {float(loss.detach()):.4f}")
    print(f"  initial mean prob   : {float(torch.sigmoid(logits.detach()).mean()):.5f}")
    print(f"  metrics (untrained) : iou={metrics['iou']:.3f} "
          f"prec={metrics['precision']:.3f} rec={metrics['recall']:.3f} "
          f"fpr={metrics['fpr']:.4f}")
    print(f"  knee blocks         : {knees.shape[0]} blocks, "
          f"range [{knees.min()}, {knees.max()}]")

    for depth in (2, 3, 4):
        m = SegUNet(in_channels=4, depth=depth, base_channels=16)
        with torch.no_grad():
            out = m(xb)
        assert out.shape == (1, 1, P, K), (depth, out.shape)
        print(f"  depth={depth}: {m.n_parameters():>9,} params, "
              f"RF {m.receptive_field():>4} px")

    print("-" * 52)
    print("all checks passed")


if __name__ == '__main__':
    _smoke_test()
