#!/usr/bin/env python
"""
Train U-Net for RFI semantic segmentation with combined multi-polarization dataset.

Combines multiple polarizations (HH, HV) into a single dataset with proper
train/val/test splits that are saved for reproducibility.
"""

import os
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


# ---------------------------------------------------------------------------
# U-Net Architecture
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) x 2"""
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
    U-Net for binary segmentation of RFI.

    Input: (batch, 2, 256, 256) - complex radar as [real, imag] channels
    Output: (batch, 1, 256, 256) - logits for RFI probability per pixel
    """
    def __init__(self, in_channels=2, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (downsampling)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder (upsampling)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Final output
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Reverse skip connections for decoder
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Upsample
            skip = skip_connections[idx // 2]

            # Handle dimension mismatch if any
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

            x = torch.cat((skip, x), dim=1)  # Concatenate skip connection
            x = self.ups[idx + 1](x)  # DoubleConv

        return self.final_conv(x)


# ---------------------------------------------------------------------------
# Combined Multi-File Dataset
# ---------------------------------------------------------------------------

class CombinedRFIDataset(Dataset):
    """
    Combined dataset from multiple HDF5 files (e.g., HH and HV).

    Each file is loaded on-the-fly to avoid memory issues.
    """
    def __init__(self, h5_paths, use_valid_mask=True):
        """
        Parameters
        ----------
        h5_paths : list of str
            List of HDF5 file paths to combine
        """
        self.h5_paths = h5_paths
        self.use_valid_mask = use_valid_mask

        # Build index: (file_idx, sample_idx_within_file)
        self.file_sample_map = []
        self.file_sizes = []

        for file_idx, h5_path in enumerate(h5_paths):
            with h5py.File(h5_path, 'r') as f:
                n_samples = f['tiles'].shape[0]
                self.file_sizes.append(n_samples)
                for sample_idx in range(n_samples):
                    self.file_sample_map.append((file_idx, sample_idx))

        print(f"Combined dataset: {len(self.file_sample_map)} samples from {len(h5_paths)} files")
        for idx, (path, size) in enumerate(zip(h5_paths, self.file_sizes)):
            print(f"  File {idx} ({Path(path).name}): {size} samples")

    def __len__(self):
        return len(self.file_sample_map)

    def __getitem__(self, idx):
        file_idx, sample_idx = self.file_sample_map[idx]
        h5_path = self.h5_paths[file_idx]

        with h5py.File(h5_path, 'r') as f:
            # Load complex tile
            tile = f['tiles'][sample_idx]

            # Load binary mask
            mask = f['masks'][sample_idx].astype(np.float32)

            # Load validity mask
            valid = f['valid'][sample_idx].astype(np.float32)

        # Convert complex to [real, imag] channels
        tile_real = np.stack([tile.real, tile.imag], axis=0).astype(np.float32)

        # Normalize: magnitude-based per tile
        magnitude = np.sqrt(tile_real[0]**2 + tile_real[1]**2)
        scale = np.percentile(magnitude[valid > 0.5], 99) if valid.sum() > 0 else 1.0
        scale = max(scale, 1e-6)
        tile_real = tile_real / scale

        # Convert to torch tensors
        tile_t = torch.from_numpy(tile_real)  # (2, 256, 256)
        mask_t = torch.from_numpy(mask).unsqueeze(0)  # (1, 256, 256)
        valid_t = torch.from_numpy(valid).unsqueeze(0)  # (1, 256, 256)

        if self.use_valid_mask:
            return tile_t, mask_t, valid_t
        else:
            return tile_t, mask_t


# ---------------------------------------------------------------------------
# Train/Val/Test Split
# ---------------------------------------------------------------------------

def create_splits(dataset, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42):
    """
    Create train/val/test split indices with balanced representation from each file.

    Splits each file (e.g., HH, HV) separately to ensure each polarization is
    proportionally represented in train/val/test sets.

    Parameters
    ----------
    dataset : CombinedRFIDataset
    train_frac, val_frac, test_frac : float
        Split fractions (must sum to 1.0)
    seed : int

    Returns
    -------
    train_idx, val_idx, test_idx : np.ndarray
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Splits must sum to 1.0"

    rng = np.random.RandomState(seed)

    train_indices = []
    val_indices = []
    test_indices = []

    # Split each file separately to ensure balanced representation
    offset = 0
    for file_idx, file_size in enumerate(dataset.file_sizes):
        # Indices for this file
        file_indices = np.arange(offset, offset + file_size)
        rng.shuffle(file_indices)

        # Split this file's indices
        train_size = int(train_frac * file_size)
        val_size = int(val_frac * file_size)

        train_indices.append(file_indices[:train_size])
        val_indices.append(file_indices[train_size:train_size + val_size])
        test_indices.append(file_indices[train_size + val_size:])

        offset += file_size

    # Combine and shuffle across files
    train_idx = np.concatenate(train_indices)
    val_idx = np.concatenate(val_indices)
    test_idx = np.concatenate(test_indices)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx


def save_splits(output_dir, train_idx, val_idx, test_idx):
    """Save split indices to disk."""
    np.savez(
        output_dir / 'data_splits.npz',
        train=train_idx,
        val=val_idx,
        test=test_idx
    )
    print(f"Saved data splits to {output_dir / 'data_splits.npz'}")


def load_splits(split_file):
    """Load split indices from disk."""
    data = np.load(split_file)
    return data['train'], data['val'], data['test']


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target, valid_mask=None):
        pred = torch.sigmoid(pred)

        if valid_mask is not None:
            pred = pred * valid_mask
            target = target * valid_mask

        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """BCE + Dice loss."""
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.dice = DiceLoss()

    def forward(self, pred, target, valid_mask=None):
        # BCE loss
        bce_loss = self.bce(pred, target)
        if valid_mask is not None:
            bce_loss = (bce_loss * valid_mask).sum() / (valid_mask.sum() + 1e-6)
        else:
            bce_loss = bce_loss.mean()

        # Dice loss
        dice_loss = self.dice(pred, target, valid_mask)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_iou(pred, target, valid_mask=None, threshold=0.5):
    """Compute Intersection over Union (IoU)."""
    pred = (torch.sigmoid(pred) > threshold).float()

    if valid_mask is not None:
        pred = pred * valid_mask
        target = target * valid_mask

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()


def compute_metrics(pred, target, valid_mask=None, threshold=0.5):
    """Compute precision, recall, F1."""
    pred = (torch.sigmoid(pred) > threshold).float()

    if valid_mask is not None:
        pred = pred * valid_mask
        target = target * valid_mask

    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()

    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    return {
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
    }


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, criterion, optimizer, device, use_valid_mask=True):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    epoch_iou = 0.0

    for batch_idx, batch in enumerate(dataloader):
        if use_valid_mask:
            tiles, masks, valid = batch
            valid = valid.to(device)
        else:
            tiles, masks = batch
            valid = None

        tiles = tiles.to(device)
        masks = masks.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(tiles)
        loss = criterion(outputs, masks, valid)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        epoch_loss += loss.item()
        epoch_iou += compute_iou(outputs, masks, valid)

    n_batches = len(dataloader)
    return epoch_loss / n_batches, epoch_iou / n_batches


def validate_epoch(model, dataloader, criterion, device, use_valid_mask=True):
    """Validate for one epoch."""
    model.eval()
    epoch_loss = 0.0
    epoch_iou = 0.0
    epoch_metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if use_valid_mask:
                tiles, masks, valid = batch
                valid = valid.to(device)
            else:
                tiles, masks = batch
                valid = None

            tiles = tiles.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(tiles)
            loss = criterion(outputs, masks, valid)

            # Metrics
            epoch_loss += loss.item()
            epoch_iou += compute_iou(outputs, masks, valid)

            metrics = compute_metrics(outputs, masks, valid)
            for k in epoch_metrics:
                epoch_metrics[k] += metrics[k]

    n_batches = len(dataloader)
    for k in epoch_metrics:
        epoch_metrics[k] /= n_batches

    return epoch_loss / n_batches, epoch_iou / n_batches, epoch_metrics


# ---------------------------------------------------------------------------
# Main Training
# ---------------------------------------------------------------------------

def train_model(args):
    """Main training function."""

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{args.run_name}_{timestamp}" if args.run_name else timestamp
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Save config
    config = vars(args)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load combined dataset
    print(f"\nLoading datasets from:")
    for path in args.data_paths:
        print(f"  - {path}")

    full_dataset = CombinedRFIDataset(args.data_paths, use_valid_mask=True)
    print(f"Total samples: {len(full_dataset)}")

    # Create or load splits
    split_file = output_dir / 'data_splits.npz'
    if args.load_splits and os.path.exists(args.load_splits):
        print(f"\nLoading existing splits from: {args.load_splits}")
        train_idx, val_idx, test_idx = load_splits(args.load_splits)
    else:
        print(f"\nCreating new train/val/test splits:")
        print(f"  Train: {args.train_frac*100:.1f}%")
        print(f"  Val:   {args.val_frac*100:.1f}%")
        print(f"  Test:  {args.test_frac*100:.1f}%")

        train_idx, val_idx, test_idx = create_splits(
            full_dataset,
            args.train_frac, args.val_frac, args.test_frac,
            seed=args.seed
        )
        save_splits(output_dir, train_idx, val_idx, test_idx)

    print(f"  Train samples: {len(train_idx)}")
    print(f"  Val samples:   {len(val_idx)}")
    print(f"  Test samples:  {len(test_idx)}")

    # Create subsets
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Model
    print("\nInitializing U-Net...")
    model = UNet(in_channels=2, out_channels=1, features=args.features)
    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Loss and optimizer
    if args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'dice':
        criterion = DiceLoss()
    else:  # combined
        criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_iou = 0.0
    history = {
        'train_loss': [], 'train_iou': [],
        'val_loss': [], 'val_iou': [],
        'val_precision': [], 'val_recall': [], 'val_f1': []
    }

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train
        train_loss, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, device, use_valid_mask=True
        )

        # Validate
        val_loss, val_iou, val_metrics = validate_epoch(
            model, val_loader, criterion, device, use_valid_mask=True
        )

        # Scheduler step
        scheduler.step(val_loss)

        # Log
        print(f"  Train Loss: {train_loss:.4f}  Train IoU: {train_iou:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}  Val IoU:   {val_iou:.4f}")
        print(f"  Val Metrics - Precision: {val_metrics['precision']:.4f}, "
              f"Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")

        # Save history
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])

        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
            }, output_dir / 'best_model.pth')
            print(f"  ✓ Saved best model (IoU: {val_iou:.4f})")

        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_loss': val_loss,
            }, output_dir / f'checkpoint_epoch_{epoch+1}.pth')

    # Final evaluation on test set
    print("\n" + "="*50)
    print("Evaluating on test set...")
    test_loss, test_iou, test_metrics = validate_epoch(
        model, test_loader, criterion, device, use_valid_mask=True
    )

    print(f"Test Loss:   {test_loss:.4f}")
    print(f"Test IoU:    {test_iou:.4f}")
    print(f"Test Metrics - Precision: {test_metrics['precision']:.4f}, "
          f"Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")

    # Save final model and results
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_iou': val_iou,
        'val_loss': val_loss,
        'test_iou': test_iou,
        'test_loss': test_loss,
        'test_metrics': test_metrics,
    }, output_dir / 'final_model.pth')

    # Save history and test results
    np.savez(output_dir / 'history.npz', **history)
    np.savez(output_dir / 'test_results.npz',
             loss=test_loss, iou=test_iou, **test_metrics)

    print(f"\nTraining complete!")
    print(f"Best validation IoU: {best_val_iou:.4f}")
    print(f"Final test IoU:      {test_iou:.4f}")
    print(f"Models and results saved to: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    # Data
    parser.add_argument('data_paths', nargs='+',
                       help='Paths to training HDF5 files (e.g., HH.h5 HV.h5)')
    parser.add_argument('--train-frac', type=float, default=0.7,
                       help='Training split fraction (default: 0.7)')
    parser.add_argument('--val-frac', type=float, default=0.15,
                       help='Validation split fraction (default: 0.15)')
    parser.add_argument('--test-frac', type=float, default=0.15,
                       help='Test split fraction (default: 0.15)')
    parser.add_argument('--load-splits', default=None,
                       help='Load existing splits from .npz file')

    # Model
    parser.add_argument('--features', type=int, nargs='+', default=[64, 128, 256, 512],
                       help='U-Net feature dimensions per level')

    # Training
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--loss', choices=['bce', 'dice', 'combined'], default='combined',
                       help='Loss function')

    # System
    parser.add_argument('--num-workers', type=int, default=4,
                       help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Output
    parser.add_argument('--output-dir', default='results/unet',
                       help='Output directory')
    parser.add_argument('--run-name', default=None,
                       help='Run name (default: timestamp)')
    parser.add_argument('--save-every', type=int, default=10,
                       help='Save checkpoint every N epochs')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_model(args)
