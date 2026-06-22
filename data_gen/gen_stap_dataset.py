"""
gen_stap_dataset.py
-------------------
Generate a dataset of synthetic STAP image matrices.

Each sample produces three independent (K, M) component matrices:
    signal  -- Gaussian complex signal at signal_db
    noise   -- Gaussian complex noise  at noise_db
    rfi     -- structured RFI at noise_db + jnr_db

And two composite images:
    clean         = signal + noise
    contaminated  = signal + noise + rfi

Convention
----------
K : range bins  (rows)   -- snapshots used to form the SCM
M : pulses      (columns) -- SCM dimension; yields M eigenvalues
SCM = X.conj().T @ X / K,  shape (M, M)

Seed isolation
--------------
All three components use disjoint RNG streams derived from a single
base_seed so they are always uncorrelated, across all sample indices:

    signal seed  = SeedSequence(base_seed).spawn(n_samples)[i].spawn(3)[0]
    noise  seed  = SeedSequence(base_seed).spawn(n_samples)[i].spawn(3)[1]
    rfi    seed  = SeedSequence(base_seed).spawn(n_samples)[i].spawn(3)[2]

numpy.random.SeedSequence guarantees that all child streams are
statistically independent of each other and of their parent.

rfi_style pinning
-----------------
Pass rfi_style="cw_tone" or rfi_style="wideband" to generate_dataset /
iter_dataset to lock all contaminated samples to one style.  The meta_rng
still draws a full styles array unconditionally so the seed tree is identical
whether or not a style is pinned -- changing rfi_style never shifts any other
sample's signal or noise.

Usage
-----
    from gen_stap_dataset import generate_dataset, STAPSample

    # Mixed styles
    samples = generate_dataset(n_samples=100, K=128, M=16)

    # Single style
    samples = generate_dataset(n_samples=100, K=128, M=16, rfi_style="cw_tone")
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Iterator

from rfi_gen_stap import RFIField, _make_rfi_field, _RFI_STYLES
from synth_stap import complex_gaussian

# Component IDs -- must never change; changing invalidates reproducibility.
_SIGNAL_CHILD = 0
_NOISE_CHILD  = 1
_RFI_CHILD    = 2


# ---------------------------------------------------------------------------
# Sample container
# ---------------------------------------------------------------------------

@dataclass
class STAPSample:
    """
    One STAP dataset item.

    Fields
    ------
    index        : sample index within the dataset
    label        : "clean" or "contaminated"
    rfi_style    : RFI style string, or None for clean samples
    jnr_db       : integer JNR drawn for this sample (None if clean)
    clean        : (K, M) signal + noise
    contaminated : (K, M) signal + noise + rfi
    signal       : (K, M) signal component
    noise        : (K, M) noise component
    rfi          : (K, M) RFI component (zero matrix for clean samples)
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
        X = getattr(self, which)
        K = X.shape[0]
        return (X.conj().T @ X) / K

    def eigenvalues_db(self, which: str = "contaminated") -> np.ndarray:
        evs = np.linalg.eigvalsh(self.scm(which))
        evs = np.sort(evs)[::-1]
        return 10.0 * np.log10(np.maximum(evs, 1e-30))


# ---------------------------------------------------------------------------
# Shared iteration core
# ---------------------------------------------------------------------------

def _make_rng_tree(n_samples: int, base_seed: int):
    """Return (sample_ss, labels, styles_drawn) for a dataset of n_samples."""
    root_ss      = np.random.SeedSequence(base_seed)
    sample_ss    = root_ss.spawn(n_samples)
    meta_rng     = np.random.default_rng(root_ss.spawn(1)[0])
    return root_ss, sample_ss, meta_rng


def _iter_core(
    n_samples: int,
    K: int,
    M: int,
    noise_db: float,
    signal_db: float,
    jnr_min_db: int,
    jnr_max_db: int,
    contaminated_fraction: float,
    base_seed: int,
    rfi_style: str | None,
) -> Iterator[STAPSample]:
    """Shared generator used by both generate_dataset and iter_dataset."""
    rfi_style_names = list(_RFI_STYLES.keys())

    root_ss   = np.random.SeedSequence(base_seed)
    sample_ss = root_ss.spawn(n_samples)

    meta_rng = np.random.default_rng(root_ss.spawn(1)[0])
    labels        = meta_rng.random(n_samples) < contaminated_fraction
    # Always draw styles unconditionally to keep meta_rng state stable
    # regardless of whether rfi_style is pinned.
    styles_drawn  = meta_rng.choice(rfi_style_names, size=n_samples)

    for i in range(n_samples):
        children   = sample_ss[i].spawn(3)
        signal_rng = np.random.default_rng(children[_SIGNAL_CHILD])
        noise_rng  = np.random.default_rng(children[_NOISE_CHILD])
        rfi_rng    = np.random.default_rng(children[_RFI_CHILD])

        signal = complex_gaussian((K, M), signal_db, signal_rng)
        noise  = complex_gaussian((K, M), noise_db,  noise_rng)
        clean  = signal + noise

        is_contaminated = bool(labels[i])

        if is_contaminated:
            # Use pinned style if given, otherwise the drawn style
            style      = rfi_style if rfi_style is not None else str(styles_drawn[i])
            rfi_result = _make_rfi_field(
                style, K, M, noise_db, jnr_min_db, jnr_max_db, rfi_rng
            )
            rfi_field    = rfi_result.field
            contaminated = clean + rfi_field
            jnr_db       = rfi_result.jnr_db
        else:
            style        = None
            rfi_field    = np.zeros((K, M), dtype=np.complex128)
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
    K: int = 128,
    M: int = 16,
    noise_db: float = 3.0,
    signal_db: float = 9.0,
    jnr_min_db: int = 10,
    jnr_max_db: int = 30,
    contaminated_fraction: float = 0.5,
    base_seed: int = 0,
    rfi_style: str | None = None,
) -> list[STAPSample]:
    """
    Generate a list of STAPSample objects.

    Parameters
    ----------
    n_samples             : Number of samples.
    K                     : Range bins (rows).  Must be >= M.
    M                     : Pulses (columns).  SCM is M x M.
    noise_db              : Noise power in dB.
    signal_db             : Signal power in dB.
    jnr_min_db            : Lower bound of integer JNR draw (inclusive).
    jnr_max_db            : Upper bound of integer JNR draw (inclusive).
    contaminated_fraction : Fraction of samples that include RFI.
    base_seed             : Root RNG seed.
    rfi_style             : Pin to "cw_tone" or "wideband".  None = random.
    """
    if K < M:
        raise ValueError(f"K={K} must be >= M={M} for a full-rank SCM.")
    if rfi_style is not None and rfi_style not in _RFI_STYLES:
        raise ValueError(
            f"Unknown rfi_style {rfi_style!r}. Choose from: {list(_RFI_STYLES)}"
        )
    return list(_iter_core(
        n_samples, K, M, noise_db, signal_db,
        jnr_min_db, jnr_max_db, contaminated_fraction, base_seed, rfi_style,
    ))


def iter_dataset(
    n_samples: int,
    K: int = 128,
    M: int = 16,
    noise_db: float = 3.0,
    signal_db: float = 9.0,
    jnr_min_db: int = 10,
    jnr_max_db: int = 30,
    contaminated_fraction: float = 0.5,
    base_seed: int = 0,
    rfi_style: str | None = None,
) -> Iterator[STAPSample]:
    """
    Lazy iterator version of generate_dataset.  Same parameters.
    Yields one STAPSample at a time.
    """
    if K < M:
        raise ValueError(f"K={K} must be >= M={M} for a full-rank SCM.")
    if rfi_style is not None and rfi_style not in _RFI_STYLES:
        raise ValueError(
            f"Unknown rfi_style {rfi_style!r}. Choose from: {list(_RFI_STYLES)}"
        )
    yield from _iter_core(
        n_samples, K, M, noise_db, signal_db,
        jnr_min_db, jnr_max_db, contaminated_fraction, base_seed, rfi_style,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    N_SAMPLES = 10
    K, M      = 128, 16

    print("=" * 65)
    print(f"Generating {N_SAMPLES} samples  (K={K}, M={M})")
    print()

    for style_arg, tag in [(None, "mixed"), ("cw_tone", "cw_tone"), ("wideband", "wideband")]:
        samples = generate_dataset(
            n_samples=N_SAMPLES, K=K, M=M,
            noise_db=3.0, signal_db=9.0,
            jnr_min_db=10, jnr_max_db=30,
            contaminated_fraction=0.6,
            base_seed=42,
            rfi_style=style_arg,
        )
        styles_seen = {s.rfi_style for s in samples if s.rfi_style}
        print(f"rfi_style={style_arg!r:10}  styles in output: {styles_seen}")

    print()

    # Verify signal/noise are identical across style runs (seed tree stability)
    s_mixed = generate_dataset(10, K=K, M=M, base_seed=42, rfi_style=None)
    s_cw    = generate_dataset(10, K=K, M=M, base_seed=42, rfi_style="cw_tone")
    assert np.allclose(s_mixed[0].signal, s_cw[0].signal), \
        "Signal changed when pinning rfi_style -- seed tree is broken"
    assert np.allclose(s_mixed[0].noise,  s_cw[0].noise), \
        "Noise changed when pinning rfi_style -- seed tree is broken"
    print("Seed stability check passed: signal/noise identical across rfi_style variants")

    # Reproducibility
    s_again = generate_dataset(10, K=K, M=M, base_seed=42, rfi_style="cw_tone")
    assert np.allclose(s_cw[0].contaminated, s_again[0].contaminated)
    print("Reproducibility check passed")
    print("=" * 65)