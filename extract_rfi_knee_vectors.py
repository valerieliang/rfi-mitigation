#!/usr/bin/env python
"""
Extract RFI knee vectors from NISAR streaming results.

Converts prediction indices to row vectors of length 16:
- All 0s: no RFI detected (prediction = 0)
- [1,0,0,...,0]: RFI at knee 1 (prediction = 1)
- [1,1,0,...,0]: RFI at knee 2 (prediction = 2)
- [1,1,1,...,0]: RFI at knee 3 (prediction = 3)
- etc.
"""

import h5py
import numpy as np
import argparse
from pathlib import Path
import json


def prediction_to_knee_vector(prediction: int) -> np.ndarray:
    """
    Convert a prediction index (0-16) to a knee vector of length 16.

    Args:
        prediction: Integer 0-16 where:
            - 0 = clean (no RFI)
            - 1-16 = RFI detected at knee position 1-16

    Returns:
        Row vector of length 16:
            - All 0s if prediction = 0 (clean)
            - First k positions = 1, rest = 0 if prediction = k (RFI at knee k)
    """
    vector = np.zeros(16, dtype=np.int8)
    if prediction > 0:
        # Fill first 'prediction' positions with 1s
        vector[:prediction] = 1
    return vector


def extract_knee_vectors(h5_file_path: str, output_file: str = None,
                         save_format: str = 'npz') -> dict:
    """
    Extract RFI knee vectors from predictions in HDF5 file.

    Args:
        h5_file_path: Path to the HDF5 file containing predictions
        output_file: Optional output file path (if None, auto-generated)
        save_format: Format to save ('npz', 'h5', or 'none' for no save)

    Returns:
        Dictionary containing knee vectors and metadata
    """
    h5_path = Path(h5_file_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"File not found: {h5_file_path}")

    print(f"Reading predictions from: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        predictions = f['predictions'][:]
        confidence = f['confidence'][:]

        print(f"Predictions shape: {predictions.shape}")
        print(f"Prediction range: [{predictions.min()}, {predictions.max()}]")

        # Get unique predictions and counts
        unique, counts = np.unique(predictions, return_counts=True)
        print(f"\nPrediction distribution:")
        for pred, count in zip(unique, counts):
            pct = 100 * count / predictions.size
            status = "clean" if pred == 0 else f"knee {pred}"
            print(f"  {pred:2d} ({status:>7s}): {count:8d} ({pct:5.2f}%)")

    # Convert predictions to knee vectors
    print(f"\nConverting to knee vectors...")
    n_rows, n_cols = predictions.shape
    knee_vectors = np.zeros((n_rows, n_cols, 16), dtype=np.int8)

    for i in range(n_rows):
        for j in range(n_cols):
            knee_vectors[i, j, :] = prediction_to_knee_vector(predictions[i, j])

    print(f"Knee vectors shape: {knee_vectors.shape}")

    # Calculate statistics
    n_clean = np.sum(predictions == 0)
    n_rfi = np.sum(predictions > 0)
    rfi_rate = n_rfi / predictions.size

    result = {
        'knee_vectors': knee_vectors,
        'predictions': predictions,
        'confidence': confidence,
        'statistics': {
            'total_cpis': int(predictions.size),
            'clean_cpis': int(n_clean),
            'rfi_cpis': int(n_rfi),
            'rfi_rate': float(rfi_rate),
            'rfi_percent': float(rfi_rate * 100),
            'shape': predictions.shape,
            'knee_vector_shape': knee_vectors.shape
        }
    }

    # Save if requested
    if save_format != 'none':
        if output_file is None:
            # Auto-generate output filename
            output_file = h5_path.parent / f"{h5_path.stem}_knee_vectors"
        else:
            output_file = Path(output_file)

        if save_format == 'npz':
            output_path = output_file.with_suffix('.npz')
            print(f"\nSaving to: {output_path}")
            np.savez_compressed(
                output_path,
                knee_vectors=knee_vectors,
                predictions=predictions,
                confidence=confidence
            )
            print(f"Saved compressed NPZ file ({output_path.stat().st_size / 1e6:.2f} MB)")

        elif save_format == 'h5':
            output_path = output_file.with_suffix('.h5')
            print(f"\nSaving to: {output_path}")
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('knee_vectors', data=knee_vectors,
                               compression='gzip', compression_opts=9)
                f.create_dataset('predictions', data=predictions)
                f.create_dataset('confidence', data=confidence)

                # Save metadata
                f.attrs['total_cpis'] = result['statistics']['total_cpis']
                f.attrs['clean_cpis'] = result['statistics']['clean_cpis']
                f.attrs['rfi_cpis'] = result['statistics']['rfi_cpis']
                f.attrs['rfi_rate'] = result['statistics']['rfi_rate']

            print(f"Saved HDF5 file ({output_path.stat().st_size / 1e6:.2f} MB)")

        # Save JSON metadata
        json_path = output_file.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump({
                'source_file': str(h5_path),
                'output_file': str(output_path),
                'statistics': result['statistics']
            }, f, indent=2)
        print(f"Saved metadata to: {json_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Extract RFI knee vectors from NISAR streaming results'
    )
    parser.add_argument(
        'input_file',
        help='Path to HDF5 file containing predictions (e.g., hh_outputs.h5)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file path (without extension, auto-determined by format)',
        default=None
    )
    parser.add_argument(
        '-f', '--format',
        choices=['npz', 'h5', 'none'],
        default='npz',
        help='Output format (default: npz)'
    )
    parser.add_argument(
        '--show-examples',
        action='store_true',
        help='Show example knee vectors for different predictions'
    )

    args = parser.parse_args()

    if args.show_examples:
        print("Example knee vectors:")
        print("-" * 60)
        for pred in [0, 1, 2, 3, 8, 15, 16]:
            vec = prediction_to_knee_vector(pred)
            status = "clean" if pred == 0 else f"knee {pred}"
            print(f"Prediction {pred:2d} ({status:>7s}): {vec}")
        print("-" * 60)
        print()

    result = extract_knee_vectors(
        args.input_file,
        output_file=args.output,
        save_format=args.format
    )

    print("\n✓ Complete!")


if __name__ == '__main__':
    main()
