"""
inspect_streaming_output.py

Utility to inspect and analyze the output from process_nisar_streaming.py.

Provides:
  - Summary statistics
  - Sample CPI inspection
  - Eigenvalue distribution analysis
  - Prediction analysis (if available)
  - Export capabilities

Usage:
    # Basic inspection:
    python inspect_streaming_output.py output.h5

    # Detailed inspection with samples:
    python inspect_streaming_output.py output.h5 --samples 10

    # Export specific CPI to numpy:
    python inspect_streaming_output.py output.h5 --export-cpi cpi_0_0 --export-dir exports/

    # Export all predictions to CSV:
    python inspect_streaming_output.py output.h5 --export-predictions --export-dir exports/
"""

import os
import sys
import numpy as np
import h5py
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


def print_summary(h5_path):
    """Print summary statistics of the processed file."""
    print("="*70)
    print("FILE SUMMARY")
    print("="*70)

    with h5py.File(h5_path, 'r') as f:
        print(f"\nFile: {h5_path}")
        print(f"\nMetadata:")
        for key in sorted(f.attrs.keys()):
            print(f"  {key}: {f.attrs[key]}")

        # Count datasets
        cpi_keys = [k for k in f.keys()
                    if k.startswith('cpi_')
                    and not k.endswith(('_eigenvalues', '_eigenvalues_normalized',
                                       '_diagonal', '_probabilities'))]

        print(f"\nDataset counts:")
        print(f"  CPI tiles: {len(cpi_keys)}")

        # Check for predictions
        has_predictions = 'has_predictions' in f.attrs and f.attrs['has_predictions']
        print(f"  Has predictions: {has_predictions}")

        # Analyze predictions if available
        if has_predictions:
            predictions = []
            confidences = []
            entropies = []

            for key in cpi_keys:
                if key in f:
                    dset = f[key]
                    if 'predicted_knee' in dset.attrs:
                        predictions.append(dset.attrs['predicted_knee'])
                        confidences.append(dset.attrs.get('prediction_confidence', 0))
                        entropies.append(dset.attrs.get('prediction_entropy', 0))

            if predictions:
                predictions = np.array(predictions)
                confidences = np.array(confidences)
                entropies = np.array(entropies)

                print(f"\nPrediction Statistics:")
                print(f"  Total predictions: {len(predictions)}")

                # Distribution
                unique, counts = np.unique(predictions, return_counts=True)
                print(f"\n  Distribution:")
                for knee, count in zip(unique, counts):
                    pct = 100 * count / len(predictions)
                    if knee == 0:
                        print(f"    Clean (knee=0): {count:6d} ({pct:5.2f}%)")
                    else:
                        print(f"    Knee @ {knee:2d}:    {count:6d} ({pct:5.2f}%)")

                # RFI rate
                rfi_count = np.sum(predictions > 0)
                rfi_rate = rfi_count / len(predictions)
                print(f"\n  RFI detection:")
                print(f"    RFI CPIs:   {rfi_count:6d} ({100*rfi_rate:5.2f}%)")
                print(f"    Clean CPIs: {len(predictions)-rfi_count:6d} ({100*(1-rfi_rate):5.2f}%)")

                # Confidence statistics
                print(f"\n  Confidence:")
                print(f"    Mean: {np.mean(confidences):.4f}")
                print(f"    Std:  {np.std(confidences):.4f}")
                print(f"    Min:  {np.min(confidences):.4f}")
                print(f"    Max:  {np.max(confidences):.4f}")

                # Entropy statistics
                print(f"\n  Entropy:")
                print(f"    Mean: {np.mean(entropies):.4f}")
                print(f"    Std:  {np.std(entropies):.4f}")
                print(f"    Min:  {np.min(entropies):.4f}")
                print(f"    Max:  {np.max(entropies):.4f}")


def inspect_samples(h5_path, n_samples=5):
    """Inspect random CPI samples."""
    print("\n" + "="*70)
    print(f"SAMPLE INSPECTION ({n_samples} random CPIs)")
    print("="*70)

    with h5py.File(h5_path, 'r') as f:
        cpi_keys = [k for k in f.keys()
                    if k.startswith('cpi_')
                    and not k.endswith(('_eigenvalues', '_eigenvalues_normalized',
                                       '_diagonal', '_probabilities'))]

        if len(cpi_keys) == 0:
            print("No CPI tiles found!")
            return

        # Sample random CPIs
        np.random.seed(42)
        sample_keys = np.random.choice(cpi_keys, size=min(n_samples, len(cpi_keys)), replace=False)

        for idx, key in enumerate(sample_keys):
            print(f"\n{'─'*70}")
            print(f"Sample {idx+1}: {key}")
            print(f"{'─'*70}")

            # Load CPI data
            cpi = f[key][:]
            print(f"  Shape: {cpi.shape}")
            print(f"  Dtype: {cpi.dtype}")

            # Attributes
            dset = f[key]
            print(f"\n  Attributes:")
            for attr in sorted(dset.attrs.keys()):
                val = dset.attrs[attr]
                if isinstance(val, float):
                    print(f"    {attr}: {val:.6f}")
                else:
                    print(f"    {attr}: {val}")

            # Eigenvalues
            eigval_key = f"{key}_eigenvalues_normalized"
            if eigval_key in f:
                eigvals = f[eigval_key][:]
                print(f"\n  Normalized Eigenvalues (top 5):")
                for i, ev in enumerate(eigvals[:5]):
                    print(f"    λ_{i+1}: {ev:.6f}")

            # SCM diagonal
            diag_key = f"{key}_diagonal"
            if diag_key in f:
                diagonal = f[diag_key][:]
                print(f"\n  SCM Diagonal:")
                print(f"    Mean: {np.mean(diagonal):.6e}")
                print(f"    Std:  {np.std(diagonal):.6e}")
                print(f"    Min:  {np.min(diagonal):.6e}")
                print(f"    Max:  {np.max(diagonal):.6e}")

            # Prediction
            if 'predicted_knee' in dset.attrs:
                pred_knee = dset.attrs['predicted_knee']
                pred_conf = dset.attrs.get('prediction_confidence', 0)
                pred_entropy = dset.attrs.get('prediction_entropy', 0)

                print(f"\n  Prediction:")
                print(f"    Knee index: {pred_knee}")
                print(f"    Confidence: {pred_conf:.6f}")
                print(f"    Entropy:    {pred_entropy:.6f}")

                # Probabilities
                prob_key = f"{key}_probabilities"
                if prob_key in f:
                    probs = f[prob_key][:]
                    top5_idx = np.argsort(probs)[::-1][:5]
                    print(f"\n    Top 5 probabilities:")
                    for i in top5_idx:
                        print(f"      knee={i:2d}: {probs[i]:.6f}")


def export_cpi(h5_path, cpi_key, export_dir):
    """Export a specific CPI and its metadata to numpy files."""
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExporting {cpi_key} to {export_dir}...")

    with h5py.File(h5_path, 'r') as f:
        if cpi_key not in f:
            print(f"ERROR: CPI key '{cpi_key}' not found!")
            return

        # Export CPI data
        cpi = f[cpi_key][:]
        np.save(export_dir / f"{cpi_key}.npy", cpi)
        print(f"  Saved: {cpi_key}.npy")

        # Export eigenvalues
        eigval_key = f"{cpi_key}_eigenvalues"
        if eigval_key in f:
            eigvals = f[eigval_key][:]
            np.save(export_dir / f"{cpi_key}_eigenvalues.npy", eigvals)
            print(f"  Saved: {cpi_key}_eigenvalues.npy")

        eigval_norm_key = f"{cpi_key}_eigenvalues_normalized"
        if eigval_norm_key in f:
            eigvals_norm = f[eigval_norm_key][:]
            np.save(export_dir / f"{cpi_key}_eigenvalues_normalized.npy", eigvals_norm)
            print(f"  Saved: {cpi_key}_eigenvalues_normalized.npy")

        # Export diagonal
        diag_key = f"{cpi_key}_diagonal"
        if diag_key in f:
            diagonal = f[diag_key][:]
            np.save(export_dir / f"{cpi_key}_diagonal.npy", diagonal)
            print(f"  Saved: {cpi_key}_diagonal.npy")

        # Export metadata
        metadata = {}
        dset = f[cpi_key]
        for attr in dset.attrs.keys():
            metadata[attr] = float(dset.attrs[attr]) if isinstance(dset.attrs[attr], (float, np.floating)) else int(dset.attrs[attr])

        with open(export_dir / f"{cpi_key}_metadata.json", 'w') as fp:
            json.dump(metadata, fp, indent=2)
        print(f"  Saved: {cpi_key}_metadata.json")

        # Export probabilities if available
        prob_key = f"{cpi_key}_probabilities"
        if prob_key in f:
            probs = f[prob_key][:]
            np.save(export_dir / f"{cpi_key}_probabilities.npy", probs)
            print(f"  Saved: {cpi_key}_probabilities.npy")


def export_predictions(h5_path, export_dir):
    """Export all predictions to CSV and numpy arrays."""
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExporting predictions to {export_dir}...")

    with h5py.File(h5_path, 'r') as f:
        cpi_keys = [k for k in f.keys()
                    if k.startswith('cpi_')
                    and not k.endswith(('_eigenvalues', '_eigenvalues_normalized',
                                       '_diagonal', '_probabilities'))]

        # Collect data
        pulse_indices = []
        range_indices = []
        predictions = []
        confidences = []
        entropies = []
        max_eigvals_db = []

        for key in cpi_keys:
            dset = f[key]

            # Parse indices from key
            parts = key.split('_')
            pulse_idx = int(parts[1])
            range_idx = int(parts[2])

            pulse_indices.append(pulse_idx)
            range_indices.append(range_idx)
            max_eigvals_db.append(dset.attrs.get('max_eigval_db', 0))

            if 'predicted_knee' in dset.attrs:
                predictions.append(dset.attrs['predicted_knee'])
                confidences.append(dset.attrs.get('prediction_confidence', 0))
                entropies.append(dset.attrs.get('prediction_entropy', 0))
            else:
                predictions.append(-1)
                confidences.append(0)
                entropies.append(0)

        # Save as numpy arrays
        data = {
            'pulse_indices': np.array(pulse_indices),
            'range_indices': np.array(range_indices),
            'predictions': np.array(predictions),
            'confidences': np.array(confidences),
            'entropies': np.array(entropies),
            'max_eigvals_db': np.array(max_eigvals_db),
        }

        np.savez(
            export_dir / 'predictions_full.npz',
            **data
        )
        print(f"  Saved: predictions_full.npz")

        # Save as CSV
        import csv
        with open(export_dir / 'predictions.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['pulse_idx', 'range_idx', 'predicted_knee',
                           'confidence', 'entropy', 'max_eigval_db'])

            for i in range(len(pulse_indices)):
                writer.writerow([
                    pulse_indices[i],
                    range_indices[i],
                    predictions[i],
                    confidences[i],
                    entropies[i],
                    max_eigvals_db[i]
                ])

        print(f"  Saved: predictions.csv")
        print(f"  Total entries: {len(pulse_indices)}")


def main():
    parser = argparse.ArgumentParser(
        description='Inspect and analyze streaming NISAR processing output',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('h5_file', help='Input HDF5 file to inspect')
    parser.add_argument('--samples', type=int, default=5,
                        help='Number of random samples to inspect (default: 5)')
    parser.add_argument('--export-cpi', type=str, default=None,
                        help='Export specific CPI (e.g., cpi_0_0)')
    parser.add_argument('--export-predictions', action='store_true',
                        help='Export all predictions to CSV and numpy')
    parser.add_argument('--export-dir', type=str, default='exports',
                        help='Directory for exports (default: exports/)')

    args = parser.parse_args()

    # Check input
    h5_path = Path(args.h5_file)
    if not h5_path.exists():
        print(f"ERROR: File not found: {args.h5_file}")
        sys.exit(1)

    # Print summary
    print_summary(h5_path)

    # Inspect samples
    if args.samples > 0:
        inspect_samples(h5_path, n_samples=args.samples)

    # Export CPI
    if args.export_cpi:
        export_cpi(h5_path, args.export_cpi, args.export_dir)

    # Export predictions
    if args.export_predictions:
        export_predictions(h5_path, args.export_dir)

    print(f"\n{'='*70}")
    print("Inspection complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
