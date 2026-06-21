"""
rfi_gen_stap.py
---------------
RFI generators that produce (K, M) matrices matching the convention in
synth_stap.py.  All three components (signal, noise, RFI) share the same
shape and can be added directly:

    stap = signal + noise + rfi_result.field

Convention
----------
K : number of range bins  (rows)
M : number of pulses      (columns)  -- also the SCM dimension

The SCM is X.conj().T @ X / K, shape (M, M), yielding M eigenvalues.

RFI is generated across the FULL image (K, M) in a single call.  There is
no per-CPI block loop; the slow-time steering vector spans all M pulses and
the range weighting spans all K range bins at once.

JNR sampling
------------
JNR (Jammer-to-Noise Ratio) is drawn as a random INTEGER from an inclusive
range [jnr_min_db, jnr_max_db] at each call to .generate().

    rfi_power_db = noise_db + jnr_db

Two styles
----------
CWToneRFI   -- rank-1 CW tone.  One dominant eigenvalue; knee at index 1.
WidebandRFI -- rank-n_modes spread.  Knee at index n_modes.

Usage
-----
    from rfi_gen_stap import CWToneRFI, WidebandRFI

    gen = CWToneRFI(jnr_min_db=10, jnr_max_db=20, seed=42)
    rfi = gen.generate(K=128, M=16, noise_db=3.0)
    # rfi.field     : (K, M) complex128
    # rfi.jnr_db    : integer drawn from [10, 20]
    # rfi.style     : "cw_tone"
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RFIField:
    """Return value from any RFI generator's .generate() call."""
    field:        np.ndarray   # (K, M) complex128
    style:        str
    jnr_db:       int          # integer JNR drawn from [jnr_min_db, jnr_max_db]
    rfi_power_db: float        # absolute power = noise_db + jnr_db
    doppler:      float | None = None
    n_modes:      int   | None = None

    def measured_power_db(self) -> float:
        """Actual mean power of the field in dB, for validation."""
        return float(10.0 * np.log10(np.mean(np.abs(self.field) ** 2)))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _draw_jnr(jnr_min_db: int, jnr_max_db: int, rng: np.random.Generator) -> int:
    """Draw a uniformly random integer from [jnr_min_db, jnr_max_db] inclusive."""
    if jnr_min_db > jnr_max_db:
        raise ValueError(
            f"jnr_min_db ({jnr_min_db}) must be <= jnr_max_db ({jnr_max_db})"
        )
    # integers() upper bound is exclusive, so +1 makes the range inclusive
    return int(rng.integers(jnr_min_db, jnr_max_db + 1))


def _scale_to_power(matrix: np.ndarray, target_power_db: float) -> np.ndarray:
    """Rescale matrix so mean(|x|^2) == 10^(target_power_db / 10)."""
    current_power = np.mean(np.abs(matrix) ** 2)
    if current_power == 0.0:
        return matrix
    target_power = 10.0 ** (target_power_db / 10.0)
    return matrix * np.sqrt(target_power / current_power)


# ---------------------------------------------------------------------------
# CW Tone  (rank-1)
# ---------------------------------------------------------------------------

class CWToneRFI:
    """
    Single narrowband CW emitter.

    The slow-time steering vector a (shape M) is coherent across all pulses,
    so the RFI covariance is rank-1.  One eigenvalue dominates and the knee
    is at index 1.

    The range weighting g (shape K) is an independent random phase per range
    bin.  It does not affect the SCM rank; it only distributes power across
    rows so the full (K, M) image is populated at once.

    Parameters
    ----------
    jnr_min_db : int   -- lower bound of JNR range (inclusive).
    jnr_max_db : int   -- upper bound of JNR range (inclusive).
    doppler    : float | None
        Normalised slow-time frequency in cycles/pulse, range ~(-0.5, 0.5).
        If None, drawn uniformly from (-0.4, 0.4) at generate() time.
    seed       : int   -- master seed; call index is mixed in per generate().
    """

    style = "cw_tone"

    def __init__(
        self,
        jnr_min_db: int = 10,
        jnr_max_db: int = 30,
        doppler: float | None = None,
        seed: int = 0,
    ):
        self.jnr_min_db = int(jnr_min_db)
        self.jnr_max_db = int(jnr_max_db)
        self.doppler    = doppler
        self._seed      = seed
        self._call_idx  = 0

    def generate(self, K: int, M: int, noise_db: float) -> RFIField:
        """
        Generate a (K, M) CW-tone RFI matrix covering the full image.

        Parameters
        ----------
        K        : Number of range bins (rows).
        M        : Number of pulses (columns).
        noise_db : Noise floor in dB; JNR is added on top to get RFI power.
        """
        rng = np.random.default_rng([self._seed, self._call_idx])
        self._call_idx += 1

        jnr_db       = _draw_jnr(self.jnr_min_db, self.jnr_max_db, rng)
        rfi_power_db = noise_db + jnr_db

        doppler = self.doppler if self.doppler is not None \
                  else float(rng.uniform(-0.4, 0.4))

        # Slow-time steering vector: shape (M,)
        pulse_idx = np.arange(M)
        phi       = rng.uniform(0.0, 2.0 * np.pi)
        a = np.exp(1j * (2.0 * np.pi * doppler * pulse_idx + phi))

        # Range weighting: independent random phase per bin, shape (K,)
        g = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=K))

        # Outer product: (K,) x (M,) -> (K, M).  Rank-1 by construction.
        raw   = np.outer(g, a).astype(np.complex128)
        field = _scale_to_power(raw, rfi_power_db)

        return RFIField(
            field        = field,
            style        = self.style,
            jnr_db       = jnr_db,
            rfi_power_db = rfi_power_db,
            doppler      = doppler,
        )


# ---------------------------------------------------------------------------
# Wideband  (rank-n_modes)
# ---------------------------------------------------------------------------

class WidebandRFI:
    """
    Broadband multi-mode emitter.

    n_modes independent Doppler tones with exponentially decaying power
    (decay_db per mode).  The SCM has rank n_modes; knee is at index n_modes.

    Each mode gets an independent Gaussian range weighting of shape (K,),
    giving partial decorrelation across range bins.  The full (K, M) image
    is built in one shot with no per-CPI loop.

    Parameters
    ----------
    jnr_min_db     : int   -- lower bound of JNR range (inclusive).
    jnr_max_db     : int   -- upper bound of JNR range (inclusive).
    n_modes        : int   -- number of Doppler modes / SCM rank.
    decay_db       : float -- power decay per mode in dB (default 3 dB).
    doppler_spread : float -- total Doppler bandwidth in cycles/pulse.
    seed           : int   -- master seed.
    """

    style = "wideband"

    def __init__(
        self,
        jnr_min_db: int = 10,
        jnr_max_db: int = 30,
        n_modes: int = 6,
        decay_db: float = 3.0,
        doppler_spread: float = 0.9,
        seed: int = 0,
    ):
        self.jnr_min_db     = int(jnr_min_db)
        self.jnr_max_db     = int(jnr_max_db)
        self.n_modes        = int(n_modes)
        self.decay_db       = float(decay_db)
        self.doppler_spread = float(doppler_spread)
        self._seed          = seed
        self._call_idx      = 0

    def generate(self, K: int, M: int, noise_db: float) -> RFIField:
        """
        Generate a (K, M) wideband RFI matrix covering the full image.

        Parameters
        ----------
        K        : Number of range bins (rows).
        M        : Number of pulses (columns).
        noise_db : Noise floor in dB; JNR is added on top to get RFI power.
        """
        rng = np.random.default_rng([self._seed, self._call_idx])
        self._call_idx += 1

        jnr_db       = _draw_jnr(self.jnr_min_db, self.jnr_max_db, rng)
        rfi_power_db = noise_db + jnr_db

        pulse_idx = np.arange(M)
        dopplers  = np.linspace(
            -self.doppler_spread / 2.0,
             self.doppler_spread / 2.0,
             self.n_modes,
        )

        raw = np.zeros((K, M), dtype=np.complex128)
        for i, fd in enumerate(dopplers):
            amp = np.sqrt(10.0 ** (-self.decay_db * i / 10.0))

            # Slow-time steering vector: shape (M,)
            a = np.exp(1j * (2.0 * np.pi * fd * pulse_idx
                             + rng.uniform(0.0, 2.0 * np.pi)))

            # Independent Gaussian range weighting for this mode: shape (K,)
            g  = rng.standard_normal(K) + 1j * rng.standard_normal(K)
            g /= np.sqrt(2.0 * K)   # normalise so each mode contributes unit power

            # Outer product: (K,) x (M,) -> (K, M)
            raw += amp * np.outer(g, a)

        field = _scale_to_power(raw, rfi_power_db)

        return RFIField(
            field        = field,
            style        = self.style,
            jnr_db       = jnr_db,
            rfi_power_db = rfi_power_db,
            n_modes      = self.n_modes,
        )


# ---------------------------------------------------------------------------
# Style table and rng-driven field builder
# ---------------------------------------------------------------------------

# Mapping from style name -> (class, extra_kwargs).
# Add new styles here; all callers that need a style-dispatch use this table.
_RFI_STYLES: dict[str, tuple[type, dict]] = {
    "cw_tone":  (CWToneRFI,  {}),
    "wideband": (WidebandRFI, {"n_modes": 4}),
}


def _make_rfi_field(
    style: str,
    K: int,
    M: int,
    noise_db: float,
    jnr_min_db: int,
    jnr_max_db: int,
    rng: np.random.Generator,
) -> RFIField:
    """
    Generate one RFI field from an already-seeded Generator.

    This bypasses the class-level call_idx counter so the caller's
    SeedSequence tree has full control over all randomness.  Used by both
    synth_stap.generate_stap_matrix and gen_stap_dataset.generate_dataset.

    Parameters
    ----------
    style      : "cw_tone" or "wideband"
    K          : Range bins (rows).
    M          : Pulses (columns).
    noise_db   : Noise floor in dB; JNR added on top to get absolute RFI power.
    jnr_min_db : Lower bound of integer JNR draw (inclusive).
    jnr_max_db : Upper bound of integer JNR draw (inclusive).
    rng        : Pre-seeded Generator; all randomness is drawn from this.
    """
    if style not in _RFI_STYLES:
        raise ValueError(f"Unknown RFI style {style!r}. Choose from: {list(_RFI_STYLES)}")

    _, extra     = _RFI_STYLES[style]
    jnr_db       = _draw_jnr(jnr_min_db, jnr_max_db, rng)
    rfi_power_db = noise_db + jnr_db

    if style == "cw_tone":
        doppler   = float(rng.uniform(-0.4, 0.4))
        pulse_idx = np.arange(M)
        phi       = rng.uniform(0.0, 2.0 * np.pi)
        a         = np.exp(1j * (2.0 * np.pi * doppler * pulse_idx + phi))
        g         = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=K))
        raw       = np.outer(g, a).astype(np.complex128)
        field     = _scale_to_power(raw, rfi_power_db)
        return RFIField(
            field=field, style=style, jnr_db=jnr_db,
            rfi_power_db=rfi_power_db, doppler=doppler,
        )

    if style == "wideband":
        n_modes        = int(extra.get("n_modes", 4))
        decay_db       = float(extra.get("decay_db", 3.0))
        doppler_spread = float(extra.get("doppler_spread", 0.9))
        pulse_idx      = np.arange(M)
        dopplers       = np.linspace(
            -doppler_spread / 2.0, doppler_spread / 2.0, n_modes
        )
        raw = np.zeros((K, M), dtype=np.complex128)
        for i, fd in enumerate(dopplers):
            amp  = np.sqrt(10.0 ** (-decay_db * i / 10.0))
            a    = np.exp(1j * (2.0 * np.pi * fd * pulse_idx
                                + rng.uniform(0.0, 2.0 * np.pi)))
            g    = rng.standard_normal(K) + 1j * rng.standard_normal(K)
            g   /= np.sqrt(2.0 * K)
            raw += amp * np.outer(g, a)
        field = _scale_to_power(raw, rfi_power_db)
        return RFIField(
            field=field, style=style, jnr_db=jnr_db,
            rfi_power_db=rfi_power_db, n_modes=n_modes,
        )

    raise ValueError(f"Unknown RFI style: {style!r}")   # unreachable


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, type] = {
    "cw_tone":  CWToneRFI,
    "wideband": WidebandRFI,
}


def make_rfi_generator(style: str, **kwargs) -> CWToneRFI | WidebandRFI:
    """Instantiate an RFI generator by style name."""
    if style not in REGISTRY:
        raise ValueError(
            f"Unknown RFI style '{style}'. Choose from: {list(REGISTRY)}"
        )
    return REGISTRY[style](**kwargs)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from synth_stap import complex_gaussian

    K, M     = 128, 16
    noise_db = 3.0

    print("=" * 60)
    print(f"Image shape: (K={K} range bins, M={M} pulses)")
    print(f"SCM shape:   ({M}, {M})   ->  {M} eigenvalues")
    print()

    for cls, kwargs in [
        (CWToneRFI,  {"jnr_min_db": 10, "jnr_max_db": 20, "seed": 7}),
        (WidebandRFI, {"jnr_min_db": 10, "jnr_max_db": 20, "n_modes": 4, "seed": 7}),
    ]:
        gen    = cls(**kwargs)
        result = gen.generate(K=K, M=M, noise_db=noise_db)

        print(f"style       : {result.style}")
        print(f"  jnr_db    : {result.jnr_db} dB")
        print(f"  power     : expected {noise_db + result.jnr_db:+.1f} dB  |"
              f"  measured {result.measured_power_db():+.3f} dB")
        print(f"  shape     : {result.field.shape}")

        scm    = (result.field.conj().T @ result.field) / K
        evs    = np.sort(np.linalg.eigvalsh(scm))[::-1]
        evs_db = 10.0 * np.log10(np.maximum(evs, 1e-30))
        print(f"  top-5 eigenvalues (dB): {np.round(evs_db[:5], 1).tolist()}")
        print()

    # Full integration: signal + noise + RFI, all (K, M), direct addition
    print("-" * 60)
    print("Integration: stap = signal + noise + rfi")
    print()

    noise_mat  = complex_gaussian((K, M), power_db=3.0, seed=0)
    signal_mat = complex_gaussian((K, M), power_db=9.0, seed=1)

    for cls, kwargs, label in [
        (CWToneRFI,  {"jnr_min_db": 15, "jnr_max_db": 25, "seed": 99},
         "cw_tone  JNR in [15, 25]"),
        (WidebandRFI, {"jnr_min_db": 15, "jnr_max_db": 25, "n_modes": 4, "seed": 99},
         "wideband JNR in [15, 25]"),
    ]:
        gen        = cls(**kwargs)
        rfi_result = gen.generate(K=K, M=M, noise_db=3.0)

        stap = signal_mat + noise_mat + rfi_result.field
        assert stap.shape == (K, M), "shape mismatch -- convention error"

        scm    = (stap.conj().T @ stap) / K
        evs    = np.sort(np.linalg.eigvalsh(scm))[::-1]
        evs_db = 10.0 * np.log10(np.maximum(evs, 1e-30))

        print(f"{label}")
        print(f"  jnr drawn : {rfi_result.jnr_db} dB")
        print(f"  stap shape: {stap.shape}  (all components share this shape)")
        print(f"  eigenvalues (dB, descending): {np.round(evs_db, 1).tolist()}")
        print()

    print("=" * 60)
