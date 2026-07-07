"""
predict_nisar.py

Run trained RFI knee classifier on NISAR CPI tiles and save predictions.

This script:
1. Loads the trained model from models/multi_band/best_model.keras
2. Loads NISAR CPI tiles from the input HDF5 file
3. Extracts features (eigenvalues, global features) matching training format
4. Runs inference to predict knee positions
5. Saves predictions and statistics to output directory

Outputs:
    - predictions.npy: Predicted knee indices (N,)
    - probabilities.npy: Class probabilities (N, M+1)
    - prediction_map.npy: Spatial map (pulse_tiles, range_tiles)
    - metadata.json: Metadata including tile_indices, shapes, polarization
    - nisar_stats.json: Statistics (distribution, RFI rate, etc.)

Usage:
    python predict_nisar.py --nisar-h5 nisar_cpi_hv.h5
"""

import os
import sys
import json
import numpy as np
import h5py
import tensorflow as tf
from pathlib import Path

# Add parent directory to path to import train.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_linear import extract_features, M, N_CLASSES


def load_nisar_dataset(h5_path, max_tiles=None):
    """
    Load NISAR CPI tiles and extract features for evaluation.

    Args:
        h5_path (str): Path to processed NISAR HDF5 file
        max_tiles (int|None): Limit to first N tiles (for testing)

    Returns:
        eigen (np.ndarray): shape (N, M, 2) - eigenvalue features
        global_ (np.ndarray): shape (N, 5) - global features
        tile_names (list[str]): CPI tile keys (e.g., 'cpi_0_0')
        tile_indices (list[tuple]): (pulse_idx, range_idx) for each tile
        metadata (dict): CPI dimensions and grid info
    """

    print(f"Loading NISAR data from: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        # Get metadata
        n_pulse_tiles = f.attrs['n_pulse_tiles']
        n_range_tiles = f.attrs['n_range_tiles']
        cpi_height = f.attrs['cpi_height']
        cpi_width = f.attrs['cpi_width']
        polarization = f.attrs.get('polarization', 'UNKNOWN')

        print(f"  Polarization: {polarization}")
        print(f"  Pulse tiles: {n_pulse_tiles}")
        print(f"  Range tiles: {n_range_tiles}")
        print(f"  Total CPIs: {n_pulse_tiles * n_range_tiles:,}")
        print(f"  CPI size: {cpi_height} × {cpi_width}")

        # Get all CPI tile keys
        cpi_keys = [k for k in f.keys()
                    if k.startswith('cpi_')
                    and not k.endswith('_eigenvalues')
                    and not k.endswith('_eigenvalues_normalized')
                    and not k.endswith('_diagonal')]

        if max_tiles is not None:
            cpi_keys = cpi_keys[:max_tiles]
            print(f"  Limited to first {max_tiles} CPIs")

        print(f"\nExtracting features from {len(cpi_keys)} CPIs...")

        eigen_list = []
        global_list = []
        tile_names = []
        tile_indices = []

        for idx, key in enumerate(cpi_keys):
            # Extract CPI data
            cpi = f[key][:]

            # Extract features (same as training)
            eigen, glob = extract_features(cpi)

            eigen_list.append(eigen)
            global_list.append(glob)
            tile_names.append(key)

            # Parse tile indices from key (e.g., 'cpi_0_250')
            parts = key.split('_')
            pulse_idx = int(parts[1])
            range_idx = int(parts[2])
            tile_indices.append((pulse_idx, range_idx))

            # Progress
            if (idx + 1) % 10000 == 0 or (idx + 1) == len(cpi_keys):
                print(f"  [{100*(idx+1)/len(cpi_keys):5.1f}%] Processed {idx+1:,}/{len(cpi_keys):,} CPIs")

        metadata = {
            'polarization': polarization,
            'n_pulse_tiles': int(n_pulse_tiles),
            'n_range_tiles': int(n_range_tiles),
            'cpi_height': int(cpi_height),
            'cpi_width': int(cpi_width),
        }

    eigen = np.stack(eigen_list).astype(np.float32)
    global_ = np.stack(global_list).astype(np.float32)

    return eigen, global_, tile_names, tile_indices, metadata


def predict_nisar(model, eigen, global_, batch_size=512):
    """
    Run model inference on NISAR features.

    Args:
        model: Trained Keras model
        eigen (np.ndarray): shape (N, M, 2)
        global_ (np.ndarray): shape (N, 5)
        batch_size (int): Batch size for inference

    Returns:
        predictions (np.ndarray): shape (N,) - predicted knee indices
        probabilities (np.ndarray): shape (N, N_CLASSES) - class probabilities
    """

    print(f"\nRunning inference on {len(eigen)} CPIs...")

    # Predict in batches to avoid memory issues
    probs = model.predict([eigen, global_], batch_size=batch_size, verbose=1)
    predictions = np.argmax(probs, axis=-1)

    return predictions, probs


def analyze_predictions(predictions, tile_indices, metadata):
    """
    Analyze prediction statistics and create spatial distribution map.

    Args:
        predictions (np.ndarray): Predicted knee indices
        tile_indices (list[tuple]): (pulse_idx, range_idx) for each tile
        metadata (dict): CPI metadata

    Returns:
        stats (dict): Statistics dictionary
        pred_map (np.ndarray): 2D spatial map of predictions
    """
    cpi_height = metadata['cpi_height']
    cpi_width = metadata['cpi_width']

    print(f"\n{'='*70}")
    print("Prediction Statistics")
    print(f"{'='*70}")

    # Overall distribution
    unique, counts = np.unique(predictions, return_counts=True)
    total = len(predictions)

    print(f"\nOverall distribution:")
    for knee, count in zip(unique, counts):
        pct = 100 * count / total
        if knee == 0:
            print(f"  Clean (knee=0): {count:,} CPIs ({pct:.2f}%)")
        else:
            print(f"  Knee @ {knee}: {count:,} CPIs ({pct:.2f}%)")

    # RFI detection rate
    rfi_cpis = np.sum(predictions > 0)
    rfi_rate = rfi_cpis / total
    print(f"\nRFI Detection:")
    print(f"  RFI CPIs: {rfi_cpis:,} ({100*rfi_rate:.2f}%)")
    print(f"  Clean CPIs: {total - rfi_cpis:,} ({100*(1-rfi_rate):.2f}%)")

    # Spatial map - normalize absolute indices to 0-based tile grid
    pulse_indices = [pi for pi, ri in tile_indices]
    range_indices = [ri for pi, ri in tile_indices]
    min_pulse, max_pulse = min(pulse_indices), max(pulse_indices)
    min_range, max_range = min(range_indices), max(range_indices)

    # Calculate actual grid size from the data
    n_pulse_tiles_actual = (max_pulse - min_pulse) // cpi_height + 1
    n_range_tiles_actual = (max_range - min_range) // cpi_width + 1

    pred_map = np.full((n_pulse_tiles_actual, n_range_tiles_actual), -1, dtype=np.int32)
    for pred, (pi, ri) in zip(predictions, tile_indices):
        # Convert absolute indices to normalized tile indices
        pulse_tile_idx = (pi - min_pulse) // cpi_height
        range_tile_idx = (ri - min_range) // cpi_width
        pred_map[pulse_tile_idx, range_tile_idx] = pred

    stats = {
        'total_cpis': total,
        'rfi_cpis': int(rfi_cpis),
        'clean_cpis': int(total - rfi_cpis),
        'rfi_rate': float(rfi_rate),
        'distribution': {int(k): int(v) for k, v in zip(unique, counts)},
        'prediction_map_shape': pred_map.shape
    }

    return stats, pred_map


def save_predictions_h5(h5_path, tile_names, predictions, probabilities):
    """
    Save predictions back to the NISAR HDF5 file as new datasets.

    Args:
        h5_path (str): Path to NISAR HDF5 file
        tile_names (list[str]): CPI tile keys
        predictions (np.ndarray): Predicted knee indices
        probabilities (np.ndarray): Class probabilities
    """

    print(f"\nSaving predictions to: {h5_path}")

    with h5py.File(h5_path, 'a') as f:
        # Add predictions as attributes to each CPI dataset
        for tile_name, pred, prob in zip(tile_names, predictions, probabilities):
            if tile_name in f:
                dset = f[tile_name]
                dset.attrs['predicted_knee'] = int(pred)
                dset.attrs['prediction_confidence'] = float(prob[pred])

        # Also save full prediction arrays as datasets
        if 'predictions' in f:
            del f['predictions']
        if 'prediction_probabilities' in f:
            del f['prediction_probabilities']

        f.create_dataset('predictions', data=predictions)
        f.create_dataset('prediction_probabilities', data=probabilities)
        f.create_dataset('tile_names', data=np.array(tile_names, dtype='S'))

    print(f"  Saved predictions for {len(tile_names)} CPIs")


def main():
    """
    Main prediction pipeline for NISAR data.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Run trained model on NISAR data and save predictions')
    parser.add_argument('--nisar-h5', default='nisar_data/processed/nisar_hv_cpi_tiles.h5',
                        help='Path to processed NISAR HDF5 file')
    parser.add_argument('--model', default='models/multi_band/best_model.keras',
                        help='Path to trained model')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for results (auto-detects from polarization if not provided)')
    parser.add_argument('--max-tiles', type=int, default=None,
                        help='Limit to first N CPIs (for testing)')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size for inference')
    parser.add_argument('--save-to-h5', action='store_true',
                        help='Save predictions back to HDF5 file as attributes')

    args = parser.parse_args()

    # Check inputs
    if not Path(args.model).exists():
        print(f"ERROR: Model not found: {args.model}")
        print(f"Train a model first: python train.py")
        sys.exit(1)

    if not Path(args.nisar_h5).exists():
        print(f"ERROR: NISAR HDF5 not found: {args.nisar_h5}")
        print(f"Process NISAR data first: python preprocess_nisar/process_nisar_to_cpi.py")
        sys.exit(1)

    # Load model
    print(f"Loading model: {args.model}")
    model = tf.keras.models.load_model(args.model)
    print(f"  Model loaded successfully")

    # Load NISAR data
    eigen, global_, tile_names, tile_indices, metadata = load_nisar_dataset(
        args.nisar_h5,
        max_tiles=args.max_tiles
    )

    # Auto-set output directory if not provided
    if args.output_dir is None:
        polarization = metadata['polarization']
        args.output_dir = f'results/nisar_eval_{polarization}'
        print(f"\nAuto-detected output directory: {args.output_dir}")

    # Run inference
    predictions, probabilities = predict_nisar(
        model, eigen, global_,
        batch_size=args.batch_size
    )

    # Analyze results
    stats, pred_map = analyze_predictions(predictions, tile_indices, metadata)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Save statistics JSON
    stats_path = os.path.join(args.output_dir, 'nisar_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved statistics: {stats_path}")

    # Save predictions as numpy arrays
    np.save(os.path.join(args.output_dir, 'predictions.npy'), predictions)
    np.save(os.path.join(args.output_dir, 'probabilities.npy'), probabilities)
    np.save(os.path.join(args.output_dir, 'prediction_map.npy'), pred_map)
    print(f"Saved predictions: {args.output_dir}/predictions.npy")
    print(f"Saved probabilities: {args.output_dir}/probabilities.npy")
    print(f"Saved prediction map: {args.output_dir}/prediction_map.npy")

    # Save metadata (includes tile_indices for plotting)
    metadata_full = {
        **metadata,
        'tile_indices': tile_indices,
        'n_cpis': len(predictions),
    }
    metadata_path = os.path.join(args.output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata_full, f, indent=2)
    print(f"Saved metadata: {metadata_path}")

    # Optionally save back to HDF5
    if args.save_to_h5:
        save_predictions_h5(args.nisar_h5, tile_names, predictions, probabilities)

    print(f"\n{'='*70}")
    print("Prediction Complete!")
    print(f"{'='*70}")
    print(f"Results saved to: {args.output_dir}")
    print(f"  - nisar_stats.json")
    print(f"  - predictions.npy")
    print(f"  - probabilities.npy")
    print(f"  - prediction_map.npy")
    print(f"  - metadata.json")
    print(f"\nNext step: Generate plots with plot_nisar_predictions.py")


if __name__ == '__main__':
    main()
