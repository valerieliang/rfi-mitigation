"""
rfi_gen.base

Core interface and slow-time eigenvalue helpers shared by every RFI
style generator in this package.

The classifier this feeds (see model.py) predicts the "knee" index in a
per-CPI eigenvalue profile: the boundary between RFI eigenvalues (large)
and the signal / noise eigenvalues (the plateau). Each generator here
injects synthetic RFI into a slow-time data block so the resulting
eigenvalue profile has a knee of controllable character, spanning the
range from very obvious (a large drop) to none at all (a smooth decay).

Two knobs are kept deliberately separate:
  - RANK of the injected RFI sets WHERE the knee sits (its index). This
    is recovered as knee_truth, the effective rank of the RFI-only
    covariance. It is the training label.
  - INR (interference-to-noise ratio, inr_db) sets HOW FAR the RFI
    eigenvalues stand above the plateau, i.e. how OBVIOUS the knee is.
    This is reported as contrast_db.

A "very obvious" knee is high rank-contrast (large inr_db, sharp drop).
A "not at all" knee is low contrast (small inr_db) and / or a gradually
decaying spectrum (e.g. the chirp style), where no single drop stands
out.

Domain note:
  The slow-time eigenvalue method operates on a CPI block of M pulses by
  K range samples. For the physically faithful pipeline these are raw
  (L0B) pulses. For an initial prototype the same block can be taken
  from M consecutive azimuth lines of a focused RSLC scene; the
  eigenstructure is then approximate but adequate for developing the
  knee classifier. This module is agnostic: it operates on whatever
  complex (M, K) block it is given, or it synthesizes one.

Array conventions:
  S      : complex array of shape (M, K), slow-time by range.
  R      : M by M sample covariance, R = S @ S^H / K.
  w      : real eigenvalues of R, sorted descending.
  w_db   : 10 * log10(w).

No non-ASCII characters appear anywhere in this package.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

TINY = 1e-20


# ----------------------------------------------------------------------
# Slow-time covariance and eigenvalue helpers
# ----------------------------------------------------------------------
def slow_time_scm(S):
    """Sample covariance R = S @ S^H / K for a slow-time block (M, K)."""
    S = np.asarray(S)
    K = S.shape[1]
    return (S @ S.conj().T) / float(K)


def eig_profile(R):
    """Real eigenvalues of a Hermitian R, sorted descending and floored."""
    w = np.linalg.eigvalsh(R)        # ascending, real for Hermitian input
    w = np.clip(w.real, TINY, None)
    return w[::-1].copy()            # descending


def eig_db(w):
    """Convert linear eigenvalues to dB."""
    return 10.0 * np.log10(np.clip(np.asarray(w), TINY, None))


def slope_db(w_db):
    """
    First difference of the dB profile, padded to length M.

    Matches the second channel expected by model.py
    (slopes_dB_padded). Entry i is w_db[i+1] - w_db[i]; the final entry
    repeats the last slope so the array length equals M.
    """
    w_db = np.asarray(w_db)
    d = np.diff(w_db)
    if d.size == 0:
        return np.zeros_like(w_db)
    return np.concatenate([d, d[-1:]])


def effective_rank(w, rel_db=12.0, floor_ratio=1e-6):
    """
    Number of eigenvalues that genuinely carry RFI energy.

    Counts eigenvalues within rel_db of the top one (and above a tiny
    numerical floor). For an RFI-only covariance this equals the number
    of independent RFI degrees of freedom, which is exactly the knee
    index: index 0 means no RFI, index r means the boundary is after
    eigenvalue r.
    """
    w = np.asarray(w)
    top = float(w[0])
    thresh = max(top * 10.0 ** (-rel_db / 10.0), top * floor_ratio)
    return max(1, int(np.sum(w >= thresh)))


def knee_contrast_db(w, knee):
    """
    Height (dB) of the weakest RFI eigenvalue above the signal plateau.

    Large positive value -> obvious knee. Near zero -> faint / no knee.
    """
    w = np.asarray(w)
    knee = int(np.clip(knee, 1, len(w) - 1))
    w_db = eig_db(w)
    rfi_floor = w_db[knee - 1]
    plateau = float(np.median(w_db[knee:]))
    return float(rfi_floor - plateau)


def estimate_knee(w_db):
    """
    Heuristic knee detector: index of the steepest adjacent drop.

    This is a stand-in for the ST-EST slope baseline and is used only to
    report how a simple detector would score a profile; it is not the
    training label. Returns (knee_index, drop_db).
    """
    w_db = np.asarray(w_db)
    drops = w_db[:-1] - w_db[1:]
    if drops.size == 0:
        return 0, 0.0
    k = int(np.argmax(drops)) + 1
    return k, float(drops.max())


def cpi_features(R):
    """
    Per-CPI features for the eigen branch of model.py.

    Returns eig_db (M,), slope_db (M,), trace_db, condition_number.
    Threshold-block features (F factor, sigma_min, sigma_max, mu_min)
    are aggregated across many CPIs and so are produced downstream, not
    here.
    """
    w = eig_profile(R)
    w_db = eig_db(w)
    return {
        "eig_db": w_db,
        "slope_db": slope_db(w_db),
        "trace_db": 10.0 * np.log10(np.clip(np.trace(R).real, TINY, None)),
        "condition_number": float(w[0] / w[-1]),
    }


def synthetic_clean(M, K, rng):
    """
    A stand-in RFI-free background when no real block is supplied: white
    complex noise plus a few weak, smoothly varying signal components,
    giving the gentle decreasing profile characteristic of clean data.
    """
    noise = (rng.standard_normal((M, K))
             + 1j * rng.standard_normal((M, K))) * np.sqrt(0.5)
    sig = np.zeros((M, K), dtype=complex)
    for j, p in enumerate((0.6, 0.4, 0.25)):
        a = np.exp(1j * (2.0 * np.pi * (0.02 * (j + 1)) * np.arange(M)
                         + rng.uniform(0.0, 2.0 * np.pi)))
        g = rng.standard_normal(K) + 1j * rng.standard_normal(K)
        sig += p * np.outer(a, g) / np.sqrt(K)
    return (noise + sig).astype(np.complex64)


# ----------------------------------------------------------------------
# Realization container
# ----------------------------------------------------------------------
@dataclass
class RFIRealization:
    """One injected-RFI sample plus its ground truth and diagnostics."""
    style: str
    s_rfi: np.ndarray          # additive RFI field (M, K), complex
    clean: np.ndarray          # clean background (M, K), complex
    contaminated: np.ndarray   # clean + s_rfi (M, K), complex
    knee_truth: int            # training label: RFI eigenvalue count
    nominal_knee: int          # knee the style intended (for reference)
    inr_db: float              # top RFI eigenvalue above plateau (set)
    contrast_db: float         # measured weakest-RFI-above-plateau (dB)
    params: dict = field(default_factory=dict)

    def covariance(self):
        """Slow-time covariance of the contaminated block."""
        return slow_time_scm(self.contaminated)

    def profile_db(self):
        """Descending eigenvalue profile of the contaminated block (dB)."""
        return eig_db(eig_profile(self.covariance()))

    def features(self):
        """Per-CPI features (see cpi_features) of the contaminated block."""
        return cpi_features(self.covariance())


# ----------------------------------------------------------------------
# Generator base class
# ----------------------------------------------------------------------
class RFIGenerator(ABC):
    """
    Base class for all RFI styles.

    Subclasses implement _unit_field, returning a complex (M, K) RFI
    field of arbitrary scale; the base class normalizes it, measures its
    effective rank (the knee), and scales it so the top RFI eigenvalue
    sits inr_db above the background plateau.
    """

    style = "base"

    def __init__(self, inr_db=20.0, nominal_knee=None, **params):
        self.inr_db = float(inr_db)
        self._nominal_knee = nominal_knee
        self.params = params

    @abstractmethod
    def _unit_field(self, M, K, rng):
        """Return a complex (M, K) RFI field (scale is irrelevant)."""
        raise NotImplementedError

    def _synthetic_clean(self, M, K, rng):
        """Instance hook; delegates to the module-level synthetic_clean."""
        return synthetic_clean(M, K, rng)

    def _normalized_field(self, M, K, rng):
        s = np.asarray(self._unit_field(M, K, rng)).astype(np.complex64)
        n = np.linalg.norm(s)
        return s / n if n > 0 else s

    def generate(self, M=32, K=128, rng=None, clean=None):
        """
        Produce one RFIRealization.

        If clean is given (a complex (M, K) block from real data), the
        RFI is overlaid on it and M, K are taken from its shape.
        Otherwise a synthetic clean background is generated.
        """
        rng = np.random.default_rng() if rng is None else rng

        if clean is None:
            clean = self._synthetic_clean(M, K, rng)
        else:
            clean = np.asarray(clean).astype(np.complex64)
            M, K = clean.shape

        s_unit = self._normalized_field(M, K, rng)

        # Knee = effective rank of the RFI-only covariance.
        w_rfi = eig_profile(slow_time_scm(s_unit))
        knee_truth = effective_rank(w_rfi)
        nominal = (self._nominal_knee
                   if self._nominal_knee is not None else knee_truth)

        # Background plateau level (linear), used as the INR reference.
        w_clean = eig_profile(slow_time_scm(clean))
        plateau_lin = float(np.median(w_clean))

        # Scale RFI so its top eigenvalue is inr_db above the plateau.
        top_unit = float(w_rfi[0])
        target_top = plateau_lin * 10.0 ** (self.inr_db / 10.0)
        scale = np.sqrt(target_top / max(top_unit, TINY))
        s_rfi = (s_unit * scale).astype(np.complex64)

        contaminated = (clean + s_rfi).astype(np.complex64)
        w_cont = eig_profile(slow_time_scm(contaminated))
        contrast = knee_contrast_db(w_cont, knee_truth)

        return RFIRealization(
            style=self.style,
            s_rfi=s_rfi,
            clean=clean,
            contaminated=contaminated,
            knee_truth=knee_truth,
            nominal_knee=nominal,
            inr_db=self.inr_db,
            contrast_db=contrast,
            params=dict(self.params),
        )

    def apply(self, clean, rng=None):
        """Overlay RFI onto a real clean (M, K) block."""
        return self.generate(clean=clean, rng=rng)
