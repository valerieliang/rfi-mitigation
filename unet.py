#!/usr/bin/env python
"""
unet.py

U-Net model architectures for RFI semantic segmentation.

This module contains ONLY the model architectures - no input preprocessing
or loss functions. For input transforms, see input_transforms.py.

Two architectures are provided:

1. UNet (2-channel) - BatchNorm, for [real, imag] input
   - Used by train_unet_2channel.py
   - Current trained model (model/best_model.pth) uses this

2. SegUNet (4-channel) - GroupNorm, for [mag_dB, cos_phase, sin_phase, valid] input
   - For use with train_unet_4channel.py (when implemented)
   - Better for small batch sizes

Usage:
    from unet import UNet, SegUNet

    # 2-channel model
    model_2ch = UNet(in_channels=2, features=[64, 128, 256, 512])

    # 4-channel model
    model_4ch = SegUNet(in_channels=4, base_channels=16, depth=3)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# 2-CHANNEL UNET (BatchNorm, for real/imag input)
# ===========================================================================

class DoubleConvBN(nn.Module):
    """(Conv -> BatchNorm -> ReLU) x 2 for 2-channel UNet"""
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
    """
    U-Net for binary RFI segmentation (2-channel input: real, imag).

    This is the architecture used by the current trained model (model/best_model.pth).
    Uses BatchNorm and expects 2 input channels.

    Parameters
    ----------
    in_channels : int, default=2
        Number of input channels (real, imag)
    out_channels : int, default=1
        Number of output channels (binary segmentation)
    features : list, default=[64, 128, 256, 512]
        Channel counts at each encoder level
    """
    def __init__(self, in_channels=2, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:
            self.downs.append(DoubleConvBN(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConvBN(features[-1], features[-1] * 2)

        # Decoder
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConvBN(feature * 2, feature))

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

    def n_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# 4-CHANNEL SEGUNET (GroupNorm, for mag_dB/phase/valid input)
# ===========================================================================

def _group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """GroupNorm with a group count that always divides num_channels."""
    groups = math.gcd(num_channels, max_groups)
    return nn.GroupNorm(max(groups, 1), num_channels)


class DoubleConvGN(nn.Module):
    """(Conv 3x3 -> GroupNorm -> ReLU) x 2 for 4-channel SegUNet"""
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
    """Max pool by 2 in both axes, then DoubleConvGN."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConvGN(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Bilinear upsample, concatenate skip, then DoubleConvGN."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm = _group_norm(out_ch)
        self.conv = DoubleConvGN(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = F.relu(self.norm(self.reduce(x)), inplace=True)
        return self.conv(torch.cat([x, skip], dim=1))


class SegUNet(nn.Module):
    """
    Semantic segmentation UNet (4-channel input: mag_dB, cos/sin phase, valid).

    Uses GroupNorm instead of BatchNorm for better performance with small batch sizes.
    Expects 4 input channels from build_input_channels() in input_transforms.py.

    Parameters
    ----------
    in_channels : int, default=4
        Number of input channels (mag_dB, cos_phase, sin_phase, valid)
    out_channels : int, default=1
        Number of output channels (binary segmentation)
    base_channels : int, default=16
        Width of the first encoder stage (doubles at each level)
    depth : int, default=3
        Number of downsampling levels
    """
    def __init__(self,
                 in_channels: int = 4,
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

        self.stem = DoubleConvGN(in_channels, widths[0])

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
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Start with strong negative bias so initial prediction is "clean everywhere"
        nn.init.constant_(self.head.bias, -4.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, in_channels, H, W)

        Returns
        -------
        (B, out_channels, H, W) raw logits (apply sigmoid outside)
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
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def receptive_field(self) -> int:
        """Theoretical receptive field of the encoder path, in input pixels."""
        rf, jump = 1, 1
        for _ in range(2):  # stem: two 3x3 convs
            rf += 2 * jump
        for _ in range(self.depth):
            jump *= 2  # pool by 2
            for _ in range(2):  # two 3x3 convs
                rf += 2 * jump
        return rf
