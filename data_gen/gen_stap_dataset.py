"""
gen_stap_dataset.py
-------------------
Generate a dataset of synthetic STAP image matrices.

Each sample produces three independent (K, M) component matrices:
    signal  -- Gaussian complex signal at signal_db
    noise   -- Gaussian complex noise  at noise_db
    rfi     -- structured RFI at noise_db + jnr_db  (style drawn randomly)

And two composite images:
    clean         = signal + noise        (no RFI, uncontaminated label)
    contaminated  = signal + noise + rfi  (RFI present, contaminated label)

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

Usage
-----
    from gen_stap_dataset import generate_dataset, STAPSample

    samples = generate_dataset(n_samples=100, K=128, M=16)
    for s in samples:
        print(s.label, s.rfi_style, s.jnr_db)
        # s.contaminated : (K, M) complex128  -- signal + noise + rfi
        # s.clean        : (K, M) complex128  -- signal + noise
        # s.signal       : (K, M) complex128
        # s.noise        : (K, M) complex128
        # s.rfi          : (K, M) complex128  (zeros when label == "clean")
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Iterator

from rfi_gen_stap import RFIField, _make_rfi_field, _RFI_STYLES
from synth_stap import complex_gaussian

# ---------------------------------------------------------------------------
# Component IDs used to partition the SeedSequence tree.
# These must never be changed; changing them invalidates reproducibility.
# ---------------------------------------------------------------------------
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
    label:        str           # "clean" or "contaminated"
    rfi_style:    str | None
    jnr_db:       int | None
    clean:        np.ndarray
    contaminated: np.ndarray
    signal:       np.ndarray
    noise:        np.ndarray
    rfi:          np.ndarray

    def scm(self, which: str = "contaminated") -> np.ndarray:
        """
        Compute the M x M sample covariance matrix for a given composite.

        Parameters
        ----------
        which : "contaminated", "clean", "signal", "noise", or "rfi"
        """
        X = getattr(self, which)
        K = X.shape[0]
        return (X.conj().T @ X) / K

    def eigenvalues_db(self, which: str = "contaminated") -> np.ndarray:
        """
        Return eigenvalues of the SCM in dB, sorted descending.
        """
        evs = np.linalg.eigvalsh(self.scm(which))
        evs = np.sort(evs)[::-1]
        return 10.0 * np.log10(np.maximum(evs, 1e-30))


# ---------------------------------------------------------------------------
# Main generator
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
) -> list[STAPSample]:
    """
    Generate a list of STAPSample objects.

    Each sample contains three fully independent (K, M) component matrices
    (signal, noise, rfi) plus two composite images (clean, contaminated).

    Independence is guaranteed by numpy.random.SeedSequence: each sample gets
    its own child sequence, which is further split into three grandchild
    sequences -- one per component.  No two components ever share an RNG state.

    Parameters
    ----------
    n_samples             : Number of samples to generate.
    K                     : Range bins per image (rows).  Must be >= M.
    M                     : Pulses per image (columns).  SCM is M x M.
    noise_db              : Noise power in dB.
    signal_db             : Signal power in dB.
    jnr_min_db            : Lower bound of integer JNR draw (inclusive).
    jnr_max_db            : Upper bound of integer JNR draw (inclusive).
    contaminated_fraction : Fraction of samples that include RFI.
                            0.0 = all clean, 1.0 = all contaminated.
    base_seed             : Root seed.  Changing this produces a completely
                            different but equally reproducible dataset.

    Returns
    -------
    list of STAPSample, length n_samples.
    """
    if K < M:
        raise ValueError(f"K={K} must be >= M={M} for a full-rank SCM.")
    if not (0.0 <= contaminated_fraction <= 1.0):
        raise ValueError("contaminated_fraction must be in [0.0, 1.0].")

    rfi_style_names = list(_RFI_STYLES.keys())

    # Root SeedSequence splits into one child per sample.
    root_ss   = np.random.SeedSequence(base_seed)
    sample_ss = root_ss.spawn(n_samples)

    # Separate RNG for dataset-level decisions (label assignment, style draw).
    # Uses a dedicated spawn so it never overlaps with component rngs.
    meta_rng  = np.random.default_rng(root_ss.spawn(1)[0])
    labels    = meta_rng.random(n_samples) < contaminated_fraction
    styles    = meta_rng.choice(rfi_style_names, size=n_samples)

    samples: list[STAPSample] = []

    for i in range(n_samples):
        # Three independent grandchild streams for this sample.
        children   = sample_ss[i].spawn(3)
        signal_rng = np.random.default_rng(children[_SIGNAL_CHILD])
        noise_rng  = np.random.default_rng(children[_NOISE_CHILD])
        rfi_rng    = np.random.default_rng(children[_RFI_CHILD])

        signal = complex_gaussian((K, M), signal_db, signal_rng)
        noise  = complex_gaussian((K, M), noise_db,  noise_rng)
        clean  = signal + noise

        is_contaminated = bool(labels[i])

        if is_contaminated:
            style      = str(styles[i])
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

        samples.append(STAPSample(
            index        = i,
            label        = "contaminated" if is_contaminated else "clean",
            rfi_style    = style,
            jnr_db       = jnr_db,
            clean        = clean,
            contaminated = contaminated,
            signal       = signal,
            noise        = noise,
            rfi          = rfi_field,
        ))

    return samples


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
) -> Iterator[STAPSample]:
    """
    Lazy iterator version of generate_dataset.
    Yields one STAPSample at a time; useful for large datasets where holding
    all samples in memory at once is not desirable.

    Parameters are identical to generate_dataset.
    """
    if K < M:
        raise ValueError(f"K={K} must be >= M={M} for a full-rank SCM.")

    rfi_style_names = list(_RFI_STYLES.keys())

    root_ss   = np.random.SeedSequence(base_seed)
    sample_ss = root_ss.spawn(n_samples)

    meta_rng  = np.random.default_rng(root_ss.spawn(1)[0])
    labels    = meta_rng.random(n_samples) < contaminated_fraction
    styles    = meta_rng.choice(rfi_style_names, size=n_samples)

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
            style      = str(styles[i])
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
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    N_SAMPLES = 10
    K, M      = 128, 16

    print("=" * 65)
    print(f"Generating {N_SAMPLES} samples  (K={K}, M={M})")
    print()

    samples = generate_dataset(
        n_samples=N_SAMPLES,
        K=K,
        M=M,
        noise_db=3.0,
        signal_db=9.0,
        jnr_min_db=10,
        jnr_max_db=30,
        contaminated_fraction=0.5,
        base_seed=42,
    )

    # -- Per-sample summary --------------------------------------------------
    print(f"{'idx':>3}  {'label':>13}  {'style':>10}  {'JNR':>6}  "
          f"{'pwr_signal':>10}  {'pwr_noise':>9}  {'pwr_rfi':>9}  "
          f"{'top_ev':>8}  {'2nd_ev':>8}")
    print("-" * 95)

    for s in samples:
        pwr = lambda X: 10.0 * np.log10(np.mean(np.abs(X) ** 2))
        pwr_sig = pwr(s.signal)
        pwr_noi = pwr(s.noise)
        pwr_rfi = pwr(s.rfi) if s.label == "contaminated" else float("nan")
        evs     = s.eigenvalues_db("contaminated")
        jnr_str = f"{s.jnr_db:>4} dB" if s.jnr_db is not None else "      --"
        sty_str = s.rfi_style if s.rfi_style else "--"
        print(
            f"{s.index:>3}  {s.label:>13}  {sty_str:>10}  {jnr_str}  "
            f"{pwr_sig:>+9.2f} dB  {pwr_noi:>+7.2f} dB  "
            f"{pwr_rfi:>+7.2f} dB  "
            f"{evs[0]:>+6.1f} dB  {evs[1]:>+6.1f} dB"
        )

    # -- Shape sanity check --------------------------------------------------
    print()
    s0 = samples[0]
    for attr in ("signal", "noise", "rfi", "clean", "contaminated"):
        mat = getattr(s0, attr)
        assert mat.shape == (K, M), \
            f"{attr} shape {mat.shape} != ({K}, {M})"
    print(f"Shape check passed: all components are ({K}, {M})")

    # -- Independence check between signal and noise -------------------------
    print()
    xcorrs = []
    for s in samples:
        n = s.noise.ravel();  n = n - n.mean()
        g = s.signal.ravel(); g = g - g.mean()
        denom = np.linalg.norm(n) * np.linalg.norm(g)
        xcorrs.append(float(np.abs(np.dot(n.conj(), g)) / denom) if denom else 0.0)
    print(f"Signal/noise cross-corr magnitude -- "
          f"mean: {np.mean(xcorrs):.4f}  max: {np.max(xcorrs):.4f}  (expect ~0)")

    # -- Reproducibility check -----------------------------------------------
    print()
    samples_b = generate_dataset(n_samples=N_SAMPLES, K=K, M=M, base_seed=42)
    assert np.allclose(samples[0].contaminated, samples_b[0].contaminated), \
        "Reproducibility check failed"
    print("Reproducibility check passed: same base_seed -> identical output")

    # -- Iterator smoke test -------------------------------------------------
    print()
    iter_sample = next(iter_dataset(n_samples=N_SAMPLES, K=K, M=M, base_seed=42))
    assert np.allclose(iter_sample.contaminated, samples[0].contaminated), \
        "iter_dataset and generate_dataset disagree on sample 0"
    print("iter_dataset check passed: matches generate_dataset on sample 0")
    print("=" * 65)
