"""
verify_eigenvalue_computation.py

Verify that eigenvalue computation matches between:
1. generate_synthetic_data.py
2. process_nisar_streaming.py (used by inspect_cpi_tiles.py)

This script loads a synthetic CPI from the training data and compares
eigenvalues computed both ways.
"""

import os
import sys
import numpy as np
import h5py

# Import from generate_synthetic_data
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_synthetic_data import _compute_eigenvalues_normalized as compute_gen_synthetic

# Import from process_nisar_streaming
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'full_nisar_streaming'))
from process_nisar_streaming import compute_scm_and_eigenvalues


def test_eigenvalue_computation():
    """Test eigenvalue computation consistency."""

    # Find a synthetic data file
    data_path = 'data/multi_band/clean/image_0_snr6.h5'

    if not os.path.exists(data_path):
        print(f"ERROR: Test data not found at {data_path}")
        print("Please run generate_synthetic_data.py first to create synthetic data")
        return

    print("="*70)
    print("Eigenvalue Computation Verification")
    print("="*70)
    print(f"Loading test data from: {data_path}\n")

    with h5py.File(data_path, 'r') as f:
        # Load first CPI
        cpi_key = 'cpi_0_0'
        cpi = f[cpi_key][:]

        # Check if eigenvalues were pre-computed
        eigvals_key = f'{cpi_key}_eigenvalues'
        if eigvals_key in f:
            eigvals_stored = f[eigvals_key][:]
        else:
            eigvals_stored = None

        print(f"CPI shape: {cpi.shape}")
        print(f"CPI dtype: {cpi.dtype}")

        # Method 1: generate_synthetic_data.py approach
        print("\n" + "="*70)
        print("Method 1: generate_synthetic_data.py")
        print("="*70)
        eigvals_norm_1, max_eigval_db_1 = compute_gen_synthetic(cpi)

        # Compute unnormalized eigenvalues
        M, K = cpi.shape
        SCM_1 = (cpi @ cpi.conj().T) / K
        eigvals_unnorm_1 = np.linalg.eigvalsh(SCM_1)
        eigvals_unnorm_1 = np.sort(eigvals_unnorm_1)[::-1]  # Descending

        print(f"Max eigenvalue: {eigvals_unnorm_1[0]:.6f}")
        print(f"Max eigenvalue (dB): {max_eigval_db_1:.2f} dB")
        print(f"  Formula: 10 * log10({eigvals_unnorm_1[0]:.6f}) = {max_eigval_db_1:.2f}")
        print(f"\nFirst 5 unnormalized eigenvalues:")
        for i in range(5):
            print(f"  λ[{i+1}] = {eigvals_unnorm_1[i]:.6f}")
        print(f"\nFirst 5 normalized eigenvalues:")
        for i in range(5):
            print(f"  λ[{i+1}] / λ_max = {eigvals_norm_1[i]:.6f}")

        # Method 2: process_nisar_streaming.py approach
        print("\n" + "="*70)
        print("Method 2: process_nisar_streaming.py (used by inspect_cpi_tiles.py)")
        print("="*70)
        SCM_2, eigvals_unnorm_2, eigvals_norm_2, max_eigval_db_2 = compute_scm_and_eigenvalues(cpi)

        print(f"Max eigenvalue: {eigvals_unnorm_2[0]:.6f}")
        print(f"Max eigenvalue (dB): {max_eigval_db_2:.2f} dB")
        print(f"  Formula: 10 * log10({eigvals_unnorm_2[0]:.6f}) = {max_eigval_db_2:.2f}")
        print(f"\nFirst 5 unnormalized eigenvalues:")
        for i in range(5):
            print(f"  λ[{i+1}] = {eigvals_unnorm_2[i]:.6f}")
        print(f"\nFirst 5 normalized eigenvalues:")
        for i in range(5):
            print(f"  λ[{i+1}] / λ_max = {eigvals_norm_2[i]:.6f}")

        # Compare stored eigenvalues (if available)
        if eigvals_stored is not None:
            print("\n" + "="*70)
            print("Stored Eigenvalues (from HDF5 file)")
            print("="*70)
            print(f"First 5 stored eigenvalues:")
            for i in range(5):
                print(f"  λ[{i+1}] = {eigvals_stored[i]:.6f}")

        # Comparison
        print("\n" + "="*70)
        print("Comparison")
        print("="*70)

        # Compare normalized eigenvalues
        norm_diff = np.max(np.abs(eigvals_norm_1 - eigvals_norm_2))
        print(f"Max difference in normalized eigenvalues: {norm_diff:.2e}")

        # Compare unnormalized eigenvalues
        unnorm_diff = np.max(np.abs(eigvals_unnorm_1 - eigvals_unnorm_2))
        print(f"Max difference in unnormalized eigenvalues: {unnorm_diff:.2e}")

        # Compare max eigenvalue dB
        db_diff = abs(max_eigval_db_1 - max_eigval_db_2)
        print(f"Difference in max eigenvalue (dB): {db_diff:.2e} dB")

        # Compare with stored
        if eigvals_stored is not None:
            stored_diff = np.max(np.abs(eigvals_stored - eigvals_unnorm_1))
            print(f"Max difference vs stored eigenvalues: {stored_diff:.2e}")

        # Test passes if all differences are negligible
        tolerance = 1e-6
        if norm_diff < tolerance and unnorm_diff < tolerance and db_diff < tolerance:
            print("\n" + "="*70)
            print("✓ PASS: Both methods produce identical eigenvalues!")
            print("="*70)
        else:
            print("\n" + "="*70)
            print("✗ FAIL: Methods produce different eigenvalues!")
            print("="*70)

        # Explain dB computation
        print("\n" + "="*70)
        print("Understanding the dB Scale")
        print("="*70)
        print("\nThe eigenvalue dB scale represents POWER in dB:")
        print(f"  Power (dB) = 10 * log10(Power_linear)")
        print(f"\nFor this CPI:")
        print(f"  Max eigenvalue (linear): {eigvals_unnorm_1[0]:.2f}")
        print(f"  Max eigenvalue (dB):     {max_eigval_db_1:.2f} dB")
        print(f"\nThe SCM is computed as: SCM = (CPI @ CPI^H) / K")
        print(f"  where K = {K} (number of range bins)")
        print(f"\nSo the eigenvalues represent the POWER per range bin.")
        print(f"\nTypical clean signal eigenvalues range from ~10 to ~100,000")
        print(f"  corresponding to ~10 dB to ~50 dB")
        print(f"\nRFI-contaminated eigenvalues can be much larger (elevated modes)")


if __name__ == '__main__':
    test_eigenvalue_computation()
