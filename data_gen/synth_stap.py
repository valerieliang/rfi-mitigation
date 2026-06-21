"""
synth_stap.py
-------------
Low-level primitives for generating synthetic STAP image matrices.

Provides:
  complex_gaussian      -- draw a (K, M) complex Gaussian matrix at a target power
  generate_stap_matrix  -- assemble signal + noise (+ optional RFI) into one dict

Convention (shared with rfi_gen_stap.py and gen_stap_dataset.py)
-----------------------------------------------------------------
K : range bins  (rows)   -- snapshots used to form the SCM
M : pulses      (columns) -- SCM dimension; yields M eigenvalues

SCM = X.conj().T @ X / K,  shape (M, M)

K must be >= M for a full-rank SCM.  Default K=128, M=16 (8x overdetermined).

Power convention (radar standard):
  power_dB = 10 * log10( E[|x|^2] )

For x ~ CN(0, sigma^2):  sigma = 10 ** (power_dB / 20)
"""

from __future__ import annotations

import numpy as np
from rfi_gen_stap import _make_rfi_field


# ---------------------------------------------------------------------------
# Core primitive
# ---------------------------------------------------------------------------

def complex_gaussian(
    shape: tuple[int, int],
    power_db: float,
    seed: int | np.random.Generator,
) -> np.ndarray:
    """
    Return a (K, M) complex Gaussian matrix at the specified average power.

    Parameters
    ----------
    shape    : (K, M) -- (range bins, pulses).
    power_db : Target power in dB, defined as 10*log10(E[|x|^2]).
    seed     : Integer seed or an already-constructed np.random.Generator.
               Passing a Generator lets callers share a SeedSequence tree.

    Returns
    -------
    ndarray of shape (K, M), dtype complex128.
    """
    rng = seed if isinstance(seed, np.random.Generator) \
               else np.random.default_rng(seed)
    sigma   = 10.0 ** (power_db / 20.0)
    sigma_c = sigma / np.sqrt(2.0)
    real    = rng.normal(0.0, sigma_c, shape)
    imag    = rng.normal(0.0, sigma_c, shape)
    return (real + 1j * imag).astype(np.complex128)


# ---------------------------------------------------------------------------
# Matrix assembler
# ---------------------------------------------------------------------------

def generate_stap_matrix(
    K: int = 128,
    M: int = 16,
    noise_db: float = 3.0,
    signal_db: float = 9.0,
    noise_seed: int = 0,
    signal_seed: int = 1,
    rfi_seed: int | None = None,
    rfi_style: str | None = None,
    jnr_min_db: int = 10,
    jnr_max_db: int = 30,
) -> dict:
    """
    Generate a synthetic STAP image of shape (K, M).

    When rfi_style is given, an RFI component is generated and added on top
    of signal + noise.  When rfi_style is None, the image is clean.

    Parameters
    ----------
    K           : Range bins (rows).  Must be >= M.
    M           : Pulses (columns).  SCM is M x M; yields M eigenvalues.
    noise_db    : Noise power in dB.
    signal_db   : Signal power in dB.
    noise_seed  : RNG seed for the noise component.
    signal_seed : RNG seed for the signal component.  Must differ from noise_seed.
    rfi_seed    : RNG seed for the RFI component.  Must differ from the other two.
                  Required when rfi_style is not None.
    rfi_style   : "cw_tone", "wideband", or None (clean).
    jnr_min_db  : Lower bound of integer JNR draw (inclusive).
    jnr_max_db  : Upper bound of integer JNR draw (inclusive).

    Returns
    -------
    dict with keys:
      "clean"       : (K, M) complex128 -- signal + noise
      "data"        : (K, M) complex128 -- signal + noise + rfi (equals clean if no RFI)
      "signal"      : (K, M) complex128
      "noise"       : (K, M) complex128
      "rfi"         : (K, M) complex128 -- zero matrix when rfi_style is None
      "label"       : "clean" or "contaminated"
      "params"      : generation parameters
      "measured_db" : measured powers for each component
    """
    if noise_seed == signal_seed:
        raise ValueError("noise_seed and signal_seed must differ.")
    if rfi_style is not None:
        if rfi_seed is None:
            raise ValueError("rfi_seed is required when rfi_style is set.")
        if rfi_seed in (noise_seed, signal_seed):
            raise ValueError("rfi_seed must differ from noise_seed and signal_seed.")

    noise  = complex_gaussian((K, M), noise_db,  seed=noise_seed)
    signal = complex_gaussian((K, M), signal_db, seed=signal_seed)
    clean  = signal + noise

    if rfi_style is not None:
        rfi_rng    = np.random.default_rng(rfi_seed)
        rfi_result = _make_rfi_field(
            rfi_style, K, M, noise_db, jnr_min_db, jnr_max_db, rfi_rng
        )
        rfi   = rfi_result.field
        data  = clean + rfi
        label = "contaminated"
        jnr_db     = rfi_result.jnr_db
        rfi_power  = float(10.0 * np.log10(np.mean(np.abs(rfi) ** 2)))
    else:
        rfi        = np.zeros((K, M), dtype=np.complex128)
        data       = clean.copy()
        label      = "clean"
        jnr_db     = None
        rfi_power  = float("nan")

    def _pwr(x):
        return float(10.0 * np.log10(np.mean(np.abs(x) ** 2)))

    return {
        "clean":   clean,
        "data":    data,
        "signal":  signal,
        "noise":   noise,
        "rfi":     rfi,
        "label":   label,
        "params": {
            "K":          K,
            "M":          M,
            "noise_db":   noise_db,
            "signal_db":  signal_db,
            "noise_seed": noise_seed,
            "signal_seed":signal_seed,
            "rfi_seed":   rfi_seed,
            "rfi_style":  rfi_style,
            "jnr_db":     jnr_db,
            "jnr_min_db": jnr_min_db,
            "jnr_max_db": jnr_max_db,
        },
        "measured_db": {
            "signal": _pwr(signal),
            "noise":  _pwr(noise),
            "rfi":    rfi_power,
            "clean":  _pwr(clean),
            "data":   _pwr(data),
        },
    }


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def verify_independence(a: np.ndarray, b: np.ndarray) -> float:
    """
    Normalised cross-correlation magnitude between two matrices.
    Returns a value in [0, 1]; expect ~0 for independent components.
    """
    a = a.ravel(); a = a - a.mean()
    b = b.ravel(); b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.abs(np.dot(a.conj(), b)) / denom) if denom else 0.0


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    K, M = 128, 16

    for rfi_style, rfi_seed in [(None, None), ("cw_tone", 2), ("wideband", 2)]:
        result = generate_stap_matrix(
            K=K, M=M,
            noise_db=3.0, signal_db=9.0,
            noise_seed=0, signal_seed=1,
            rfi_seed=rfi_seed, rfi_style=rfi_style,
            jnr_min_db=15, jnr_max_db=25,
        )
        p = result["params"]
        m = result["measured_db"]

        print("=" * 55)
        print(f"label      : {result['label']}")
        print(f"rfi_style  : {p['rfi_style']}   jnr_db: {p['jnr_db']}")
        print(f"shape      : ({K}, {M})")
        print(f"measured powers (dB):")
        print(f"  signal = {m['signal']:+.3f}   noise = {m['noise']:+.3f}"
              f"   rfi = {m['rfi']:+.3f}   data = {m['data']:+.3f}")

        # Independence checks
        print(f"cross-corr signal/noise : "
              f"{verify_independence(result['signal'], result['noise']):.4f}")
        if result['label'] == "contaminated":
            print(f"cross-corr signal/rfi   : "
                  f"{verify_independence(result['signal'], result['rfi']):.4f}")
            print(f"cross-corr noise/rfi    : "
                  f"{verify_independence(result['noise'],  result['rfi']):.4f}")

        # SCM eigenvalues of the composite
        X      = result["data"]
        scm    = (X.conj().T @ X) / K
        evs    = np.sort(np.linalg.eigvalsh(scm))[::-1]
        evs_db = 10.0 * np.log10(np.maximum(evs, 1e-30))
        print(f"eigenvalues (dB, desc) : {np.round(evs_db, 1).tolist()}")

    print("=" * 55)
