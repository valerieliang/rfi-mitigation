#!/usr/bin/env python
"""
U-Net training for RFI semantic segmentation.

Optimized for GPU training with automatic RAM caching for speed.
Works with any HDF5 dataset(s) containing tiles/masks/valid.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


# ===========================================================================
# U-NET ARCHITECTURE
# ===========================================================================

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
    """U-Net for binary RFI segmentation."""
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


# ===========================================================================
# DATASET
# ===========================================================================

class RFIDataset(Dataset):
    """
    RFI dataset - automatically uses RAM caching if dataset fits in memory.
    """
    def __init__(self, h5_paths, use_valid_mask=True, cache_to_ram=True):
        self.h5_paths = h5_paths
        self.use_valid_mask = use_valid_mask

        # Determine dataset size
        self.file_sizes = []
        total_samples = 0
        for path in h5_paths:
            with h5py.File(path, 'r') as f:
                n = f['tiles'].shape[0]
                self.file_sizes.append(n)
                total_samples += n

        print(f"Dataset: {total_samples} samples from {len(h5_paths)} file(s)")
        for path, size in zip(h5_paths, self.file_sizes):
            print(f"  {Path(path).name}: {size} samples")

        # Auto-decide caching based on size
        estimated_gb = total_samples * 256 * 256 * 8 / 1024**3  # complex64 = 8 bytes
        if cache_to_ram and estimated_gb < 50:  # Cache if < 50GB
            print(f"\nCaching to RAM (~{estimated_gb:.1f} GB)...")
            self._load_to_ram()
            self.cached = True
        else:
            print(f"\nUsing on-disk loading (estimated {estimated_gb:.1f} GB)")
            self._build_index()
            self.cached = False

    def _load_to_ram(self):
        """Load all data into RAM."""
        all_tiles, all_masks, all_valid = [], [], []

        for path in self.h5_paths:
            with h5py.File(path, 'r') as f:
                all_tiles.append(f['tiles'][:])
                all_masks.append(f['masks'][:])
                all_valid.append(f['valid'][:])

        self.tiles = np.concatenate(all_tiles, axis=0)
        self.masks = np.concatenate(all_masks, axis=0)
        self.valid = np.concatenate(all_valid, axis=0)

        print(f"✓ Loaded {len(self.tiles)} samples to RAM")

    def _build_index(self):
        """Build file index for on-disk loading."""
        self.file_sample_map = []
        for file_idx, size in enumerate(self.file_sizes):
            for sample_idx in range(size):
                self.file_sample_map.append((file_idx, sample_idx))

    def __len__(self):
        return sum(self.file_sizes)

    def __getitem__(self, idx):
        if self.cached:
            tile = self.tiles[idx]
            mask = self.masks[idx].astype(np.float32)
            valid = self.valid[idx].astype(np.float32)
        else:
            file_idx, sample_idx = self.file_sample_map[idx]
            with h5py.File(self.h5_paths[file_idx], 'r') as f:
                tile = f['tiles'][sample_idx]
                mask = f['masks'][sample_idx].astype(np.float32)
                valid = f['valid'][sample_idx].astype(np.float32)

        # Convert complex to [real, imag]
        tile_real = np.stack([tile.real, tile.imag], axis=0).astype(np.float32)

        # Normalize by 99th percentile of valid magnitude
        magnitude = np.sqrt(tile_real[0]**2 + tile_real[1]**2)
        scale = np.percentile(magnitude[valid > 0.5], 99) if valid.sum() > 0 else 1.0
        tile_real = tile_real / max(scale, 1e-6)

        # To tensors
        tile_t = torch.from_numpy(tile_real)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        valid_t = torch.from_numpy(valid).unsqueeze(0)

        return (tile_t, mask_t, valid_t) if self.use_valid_mask else (tile_t, mask_t)


# ===========================================================================
# SPLITS
# ===========================================================================

def create_splits(dataset, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42):
    """Create balanced train/val/test splits across all files."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    rng = np.random.RandomState(seed)
    train_idx, val_idx, test_idx = [], [], []

    offset = 0
    for file_size in dataset.file_sizes:
        indices = np.arange(offset, offset + file_size)
        rng.shuffle(indices)

        n_train = int(train_frac * file_size)
        n_val = int(val_frac * file_size)

        train_idx.append(indices[:n_train])
        val_idx.append(indices[n_train:n_train + n_val])
        test_idx.append(indices[n_train + n_val:])

        offset += file_size

    # Concatenate and shuffle
    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    test_idx = np.concatenate(test_idx)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx


# ===========================================================================
# LOSS & METRICS
# ===========================================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target, valid_mask=None):
        pred = torch.sigmoid(pred)
        if valid_mask is not None:
            pred = pred * valid_mask
            target = target * valid_mask

        intersection = (pred.view(-1) * target.view(-1)).sum()
        union = pred.view(-1).sum() + target.view(-1).sum()

        return 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.dice = DiceLoss()

    def forward(self, pred, target, valid_mask=None):
        bce = self.bce(pred, target)
        if valid_mask is not None:
            bce = (bce * valid_mask).sum() / (valid_mask.sum() + 1e-6)
        else:
            bce = bce.mean()

        dice = self.dice(pred, target, valid_mask)
        return self.bce_weight * bce + self.dice_weight * dice


def compute_metrics(pred, target, valid_mask=None, threshold=0.5):
    """Compute IoU, precision, recall, F1."""
    pred = (torch.sigmoid(pred) > threshold).float()

    if valid_mask is not None:
        pred = pred * valid_mask
        target = target * valid_mask

    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    union = pred.sum() + target.sum() - tp

    iou = (tp + 1e-6) / (union + 1e-6)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    return {
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }


# ===========================================================================
# TRAINING
# ===========================================================================

def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    metrics = {'iou': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    for tiles, masks, valid in loader:
        tiles, masks, valid = tiles.to(device), masks.to(device), valid.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(tiles)
                loss = criterion(outputs, masks, valid)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(tiles)
            loss = criterion(outputs, masks, valid)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        batch_metrics = compute_metrics(outputs, masks, valid)
        for k in metrics:
            metrics[k] += batch_metrics[k]

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in metrics.items()}


def validate_epoch(model, loader, criterion, device, scaler=None):
    model.eval()
    total_loss = 0.0
    metrics = {'iou': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    with torch.no_grad():
        for tiles, masks, valid in loader:
            tiles, masks, valid = tiles.to(device), masks.to(device), valid.to(device)

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(tiles)
                    loss = criterion(outputs, masks, valid)
            else:
                outputs = model(tiles)
                loss = criterion(outputs, masks, valid)

            total_loss += loss.item()
            batch_metrics = compute_metrics(outputs, masks, valid)
            for k in metrics:
                metrics[k] += batch_metrics[k]

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in metrics.items()}


# ===========================================================================
# MAIN
# ===========================================================================

def main(args):
    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{args.run_name}_{timestamp}" if args.run_name else timestamp
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput: {output_dir}\n")

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    # Dataset
    dataset = RFIDataset(args.data, cache_to_ram=not args.no_cache)

    # Splits
    if args.load_splits and Path(args.load_splits).exists():
        print(f"\nLoading splits from {args.load_splits}")
        data = np.load(args.load_splits)
        train_idx, val_idx, test_idx = data['train'], data['val'], data['test']
    else:
        print(f"\nCreating splits: {args.train_frac:.0%}/{args.val_frac:.0%}/{args.test_frac:.0%}")
        train_idx, val_idx, test_idx = create_splits(
            dataset, args.train_frac, args.val_frac, args.test_frac, args.seed
        )
        np.savez(output_dir / 'splits.npz', train=train_idx, val=val_idx, test=test_idx)

    print(f"  Train: {len(train_idx)}")
    print(f"  Val:   {len(val_idx)}")
    print(f"  Test:  {len(test_idx)}\n")

    # Loaders
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)

    print(f"Batches per epoch: {len(train_loader)}\n")

    # Model
    print("Initializing U-Net...")
    model = UNet(in_channels=2, out_channels=1, features=args.features).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}\n")

    # Optimizer
    criterion = CombinedLoss() if args.loss == 'combined' else (
        DiceLoss() if args.loss == 'dice' else nn.BCEWithLogitsLoss()
    )
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Mixed precision scaler for H100
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() and not args.no_amp else None
    if scaler:
        print("Using mixed precision (AMP) for faster training\n")

    # Train
    print(f"Training for {args.epochs} epochs...\n")
    best_val_iou = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_f1': []}

    for epoch in range(args.epochs):
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device, scaler)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train: L={train_loss:.3f} IoU={train_metrics['iou']:.3f} | "
              f"Val: L={val_loss:.3f} IoU={val_metrics['iou']:.3f} F1={val_metrics['f1']:.3f}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_metrics['iou'])
        history['val_f1'].append(val_metrics['f1'])

        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            torch.save(model.state_dict(), output_dir / 'best_model.pth')

        if (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), output_dir / f'checkpoint_{epoch+1}.pth')

    # Test
    print(f"\n{'='*60}")
    test_loss, test_metrics = validate_epoch(model, test_loader, criterion, device, scaler)
    print(f"Test: Loss={test_loss:.3f} IoU={test_metrics['iou']:.3f} "
          f"P={test_metrics['precision']:.3f} R={test_metrics['recall']:.3f} F1={test_metrics['f1']:.3f}")

    # Save
    torch.save(model.state_dict(), output_dir / 'final_model.pth')
    np.savez(output_dir / 'history.npz', **history)
    np.savez(output_dir / 'test_results.npz', **test_metrics, loss=test_loss)

    print(f"\n✓ Done! Best val IoU: {best_val_iou:.3f} | Test IoU: {test_metrics['iou']:.3f}")
    print(f"  Saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', nargs='+', help='HDF5 dataset file(s)')
    parser.add_argument('--train-frac', type=float, default=0.7)
    parser.add_argument('--val-frac', type=float, default=0.15)
    parser.add_argument('--test-frac', type=float, default=0.15)
    parser.add_argument('--load-splits', default=None, help='Load existing splits.npz')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--loss', choices=['bce', 'dice', 'combined'], default='combined')
    parser.add_argument('--features', type=int, nargs='+', default=[64, 128, 256, 512])
    parser.add_argument('--no-cache', action='store_true', help='Disable RAM caching')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision (AMP)')
    parser.add_argument('--output-dir', default='results/unet')
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    main(args)
