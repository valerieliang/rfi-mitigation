"""
clean_data.py
-------------
Primitives for synthetic STAP CPI block generation.

Provides
--------
complex_gaussian   -- draw a (M, K) complex Gaussian matrix at a target power
compute_scm        -- form the M x M sample covariance from a (M, K) block
eigenvalues_db     -- return M eigenvalues of the SCM in descending dB order
STAPSample         -- dataclass holding one dataset item (all (M, K) fields)
generate_dataset   -- return a list of STAPSample objects
iter_dataset       -- lazy iterator version of generate_dataset

NISAR ST-EVD convention
-----------------------
M : pulses per CPI  (rows)    -- SCM dimension; yields M eigenvalues
K : range bins      (columns) -- snapshot count for the SCM estimate

CPI block S has shape (M, K).
SCM: R = S @ S.conj().T / K,  shape (M, M)

Matches the IGARSS 2023 paper formula:
    R_{i,j}[MxM] = S[MxK] . S^H[KxM]

Constraint: K >= 2*M for a non-degenerate SCM. Recommended K >= 4*M.

Power convention (radar standard):
    power_dB = 10 * log10( E[|x|^2] )
    sigma    = 10 ** (power_dB / 20)

Seed isolation
--------------
Each sample spawns three independent RNG child streams from a SeedSequence:
    child[0] -> signal
    child[1] -> noise
    child[2] -> RFI

Changing rfi_style never alters signal or noise for any sample index.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Iterator

from rfi_gen_stap import _make_rfi_field, _RFI_STYLES

# Child indices -- must never change; changing invalidates reproducibility.
_SIGNAL_CHILD = 0
_NOISE_CHILD  = 1
_RFI_CHILD    = 2


# ---------------------------------------------------------------------------
# Signal / noise primitives
# ---------------------------------------------------------------------------

def complex_gaussian(
    shape: tuple[int, int],
    power_db: float,
    seed: int | np.random.Generator,
) -> np.ndarray:
    """
    Return a (M, K) complex Gaussian matrix at the specified average power.

    Parameters
    ----------
    shape    : (M, K) -- (pulses, range bins).
    power_db : Target power in dB, 10*log10(E[|x|^2]).
    seed     : Integer seed or an already-constructed Generator.

    Returns
    -------
    ndarray of shape (M, K), dtype complex128.
    """
    rng = (seed if isinstance(seed, np.random.Generator)
           else np.random.default_rng(seed))
    sigma   = 10.0 ** (power_db / 20.0)
    sigma_c = sigma / np.sqrt(2.0)
    real    = rng.normal(0.0, sigma_c, shape)
    imag    = rng.normal(0.0, sigma_c, shape)
    return (real + 1j * imag).astype(np.complex128)


# ---------------------------------------------------------------------------
# SCM helpers
# ---------------------------------------------------------------------------

def compute_scm(S: np.ndarray) -> np.ndarray:
    """
    Compute the M x M sample covariance matrix from a (M, K) CPI block.

    R = S @ S.conj().T / K

    Parameters
    ----------
    S : (M, K) complex matrix -- rows are pulses, columns are range bins.

    Returns
    -------
    ndarray of shape (M, M), dtype complex128.
    """
    K = S.shape[1]
    return (S @ S.conj().T) / K


def eigenvalues_db(S: np.ndarray) -> np.ndarray:
    """
    Return the M eigenvalues of the SCM of S in descending dB order.

    Parameters
    ----------
    S : (M, K) complex matrix.

    Returns
    -------
    ndarray of shape (M,), real, descending.
    """
    evs = np.linalg.eigvalsh(compute_scm(S))
    return 10.0 * np.log10(np.maximum(np.sort(evs)[::-1], 1e-30))


# ---------------------------------------------------------------------------
# Sample container
# ---------------------------------------------------------------------------

@dataclass
class STAPSample:
    """
    One STAP dataset item. All matrix fields have shape (M, K).

    Fields
    ------
    index        : sample index within the dataset
    label        : "clean" or "contaminated"
    rfi_style    : RFI style string, or None for clean samples
    jnr_db       : integer JNR in dB (None if clean)
    clean        : (M, K) signal + noise
    contaminated : (M, K) signal + noise + rfi
    signal       : (M, K) signal component
    noise        : (M, K) noise component
    rfi          : (M, K) RFI component (zeros for clean samples)
    """
    index:        int
    label:        str
    rfi_style:    str | None
    jnr_db:       int | None
    clean:        np.ndarray
    contaminated: np.ndarray
    signal:       np.ndarray
    noise:        np.ndarray
    rfi:          np.ndarray

    def scm(self, which: str = "contaminated") -> np.ndarray:
        """M x M SCM for the given composite field."""
        return compute_scm(getattr(self, which))

    def eigenvalues_db(self, which: str = "contaminated") -> np.ndarray:
        """M eigenvalues of the SCM in descending dB order."""
        evs = np.linalg.eigvalsh(self.scm(which))
        return 10.0 * np.log10(np.maximum(np.sort(evs)[::-1], 1e-30))


# ---------------------------------------------------------------------------
# Dataset iterator (shared core)
# ---------------------------------------------------------------------------

def _iter_core(
    n_samples: int,
    M: int,
    K: int,
    noise_db: float,
    signal_db: float,
    jnr_min_db: int,
    jnr_max_db: int,
    contaminated_fraction: float,
    base_seed: int,
    rfi_style: str | None,
) -> Iterator[STAPSample]:
    rfi_style_names = list(_RFI_STYLES.keys())

    root_ss   = np.random.SeedSequence(base_seed)
    sample_ss = root_ss.spawn(n_samples)

    meta_rng     = np.random.default_rng(root_ss.spawn(1)[0])
    labels       = meta_rng.random(n_samples) < contaminated_fraction
    # Always draw styles unconditionally to keep meta_rng state identical
    # regardless of whether rfi_style is pinned.
    styles_drawn = meta_rng.choice(rfi_style_names, size=n_samples)

    for i in range(n_samples):
        children   = sample_ss[i].spawn(3)
        signal_rng = np.random.default_rng(children[_SIGNAL_CHILD])
        noise_rng  = np.random.default_rng(children[_NOISE_CHILD])
        rfi_rng    = np.random.default_rng(children[_RFI_CHILD])

        signal = complex_gaussian((M, K), signal_db, signal_rng)
        noise  = complex_gaussian((M, K), noise_db,  noise_rng)
        clean  = signal + noise

        is_contaminated = bool(labels[i])

        if is_contaminated:
            style      = rfi_style if rfi_style is not None else str(styles_drawn[i])
            rfi_result = _make_rfi_field(
                style, M, K, noise_db, jnr_min_db, jnr_max_db, rfi_rng
            )
            rfi_field    = rfi_result.field
            contaminated = clean + rfi_field
            jnr_db       = rfi_result.jnr_db
        else:
            style        = None
            rfi_field    = np.zeros((M, K), dtype=np.complex128)
            contaminated = clean.copy()
            jnr_db       = None

        yield STAPSample(
            index        = i,
            label        = "contaminated" if is_contaminated else "clean",
            rfi_style    = style,
            jnr_db       = jnr_db,
            clean        = clean,
            contaminated = contaminated,
            signal       = signal,
            noise        = noise,
            rfi          = rfi_field,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_dataset(
    n_samples: int,
    M: int = 16,
    K: int = 128,
    noise_db: float = 3.0,
    signal_db: float = 9.0,
    jnr_min_db: int = 10,
    jnr_max_db: int = 30,
    contaminated_fraction: float = 0.5,
    base_seed: int = 0,
    rfi_style: str | None = None,
) -> list[STAPSample]:
    """
    Generate a list of STAPSample objects (all held in memory).

    Parameters
    ----------
    n_samples             : Number of samples.
    M                     : Pulses per CPI (rows). SCM is M x M.
    K                     : Range bins per CPI (columns). Must be >= 2*M.
    noise_db              : Noise power in dB.
    signal_db             : Signal power in dB.
    jnr_min_db            : Lower bound of integer JNR draw (inclusive).
    jnr_max_db            : Upper bound of integer JNR draw (inclusive).
    contaminated_fraction : Fraction of samples that include RFI.
    base_seed             : Root RNG seed.
    rfi_style             : Pin to "cw_tone" or "wideband". None = random mix.
    """
    if K < 2 * M:
        raise ValueError(f"K={K} must be >= 2*M={2*M} for a non-degenerate SCM.")
    if rfi_style is not None and rfi_style not in _RFI_STYLES:
        raise ValueError(
            f"Unknown rfi_style {rfi_style!r}. Choose from: {list(_RFI_STYLES)}"
        )
    return list(_iter_core(
        n_samples, M, K, noise_db, signal_db,
        jnr_min_db, jnr_max_db, contaminated_fraction, base_seed, rfi_style,
    ))


def iter_dataset(
    n_samples: int,
    M: int = 16,
    K: int = 128,
    noise_db: float = 3.0,
    signal_db: float = 9.0,
    jnr_min_db: int = 10,
    jnr_max_db: int = 30,
    contaminated_fraction: float = 0.5,
    base_seed: int = 0,
    rfi_style: str | None = None,
) -> Iterator[STAPSample]:
    """
    Lazy iterator version of generate_dataset. Same parameters.
    Yields one STAPSample at a time; avoids holding all samples in memory.
    """
    if K < 2 * M:
        raise ValueError(f"K={K} must be >= 2*M={2*M} for a non-degenerate SCM.")
    if rfi_style is not None and rfi_style not in _RFI_STYLES:
        raise ValueError(
            f"Unknown rfi_style {rfi_style!r}. Choose from: {list(_RFI_STYLES)}"
        )
    yield from _iter_core(
        n_samples, M, K, noise_db, signal_db,
        jnr_min_db, jnr_max_db, contaminated_fraction, base_seed, rfi_style,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    M, K = 16, 128
    print("=" * 65)
    print(f"CPI shape: (M={M} pulses, K={K} range bins)")
    print(f"SCM: R = S @ S.conj().T / K,  shape ({M}, {M})")
    print()

    for style_arg in (None, "cw_tone", "wideband"):
        samples = generate_dataset(
            n_samples=10, M=M, K=K,
            noise_db=3.0, signal_db=9.0,
            jnr_min_db=10, jnr_max_db=30,
            contaminated_fraction=0.6,
            base_seed=42,
            rfi_style=style_arg,
        )
        styles_seen = {s.rfi_style for s in samples if s.rfi_style}
        shapes      = {s.contaminated.shape for s in samples}
        print(f"rfi_style={style_arg!r:10}  "
              f"styles={styles_seen}  shapes={shapes}")

    print()

    # Seed stability: signal/noise must be identical across rfi_style runs
    s_mixed = generate_dataset(10, M=M, K=K, base_seed=42, rfi_style=None)
    s_cw    = generate_dataset(10, M=M, K=K, base_seed=42, rfi_style="cw_tone")
    assert np.allclose(s_mixed[0].signal, s_cw[0].signal), \
        "Signal shifted when pinning rfi_style -- seed tree broken"
    assert np.allclose(s_mixed[0].noise, s_cw[0].noise), \
        "Noise shifted when pinning rfi_style -- seed tree broken"
    print("Seed stability check passed")

    s_again = generate_dataset(10, M=M, K=K, base_seed=42, rfi_style="cw_tone")
    assert np.allclose(s_cw[0].contaminated, s_again[0].contaminated)
    print("Reproducibility check passed")
    print("=" * 65)
