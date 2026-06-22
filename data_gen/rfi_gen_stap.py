"""
rfi_gen_stap.py
---------------
RFI field generators producing (M, K) matrices that match the NISAR
ST-EVD convention used across this pipeline.

NISAR ST-EVD convention
-----------------------
M : pulses per CPI  (rows)    -- SCM dimension
K : range bins      (columns) -- snapshot count

CPI block S has shape (M, K).
SCM: R = S @ S.conj().T / K,  shape (M, M)

This matches the IGARSS 2023 paper formula:
    R_{i,j}[MxM] = S[MxK] . S^H[KxM]

All three components (signal, noise, RFI) share this shape and can be
added directly:
    stap = signal + noise + rfi_result.field

JNR sampling
------------
JNR (Jammer-to-Noise Ratio) is drawn as a random INTEGER from an
inclusive range [jnr_min_db, jnr_max_db] at each call to .generate().

    rfi_power_db = noise_db + jnr_db

Two styles
----------
CWToneRFI   -- rank-1 CW tone. Steering vector along pulse axis (rows).
               One dominant eigenvalue; knee at index 1.
WidebandRFI -- rank-n_modes. n_modes Doppler tones across pulse axis.
               Knee at index n_modes.

The field is generated for the full CPI in one shot (no per-block loop).
The range weighting spans all K columns; the steering vector spans all M
rows.
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
    field:        np.ndarray   # (M, K) complex128
    style:        str
    jnr_db:       int          # integer JNR drawn from [jnr_min_db, jnr_max_db]
    rfi_power_db: float        # absolute power = noise_db + jnr_db
    doppler:      float | None = None
    n_modes:      int   | None = None

    def measured_power_db(self) -> float:
        """Actual mean power of the field in dB."""
        return float(10.0 * np.log10(np.mean(np.abs(self.field) ** 2)))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _draw_jnr(
    jnr_min_db: int,
    jnr_max_db: int,
    rng: np.random.Generator,
) -> int:
    """Draw a uniformly random integer from [jnr_min_db, jnr_max_db] inclusive."""
    if jnr_min_db > jnr_max_db:
        raise ValueError(
            f"jnr_min_db ({jnr_min_db}) must be <= jnr_max_db ({jnr_max_db})"
        )
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

    The slow-time (pulse-domain) steering vector a has shape (M,).
    It is coherent across all M pulses, making the RFI covariance rank-1.
    One eigenvalue dominates; knee is at index 1.

    The range weighting g has shape (K,) -- independent random phase per
    range bin. It distributes power across columns without affecting rank.

    Outer product: np.outer(a, g) -> (M, K).

    Parameters
    ----------
    jnr_min_db : lower bound of JNR range (inclusive).
    jnr_max_db : upper bound of JNR range (inclusive).
    doppler    : normalised slow-time frequency in cycles/pulse, ~(-0.5, 0.5).
                 None = drawn uniformly from (-0.4, 0.4) at generate() time.
    seed       : master seed; call index is mixed in per generate().
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

    def generate(self, M: int, K: int, noise_db: float) -> RFIField:
        """
        Generate a (M, K) CW-tone RFI matrix.

        Parameters
        ----------
        M        : Pulses (rows).
        K        : Range bins (columns).
        noise_db : Noise floor in dB; JNR added on top.
        """
        rng = np.random.default_rng([self._seed, self._call_idx])
        self._call_idx += 1

        jnr_db       = _draw_jnr(self.jnr_min_db, self.jnr_max_db, rng)
        rfi_power_db = noise_db + jnr_db

        doppler = (self.doppler if self.doppler is not None
                   else float(rng.uniform(-0.4, 0.4)))

        # Slow-time steering vector along pulse axis: shape (M,)
        pulse_idx = np.arange(M)
        phi       = rng.uniform(0.0, 2.0 * np.pi)
        a = np.exp(1j * (2.0 * np.pi * doppler * pulse_idx + phi))

        # Range weighting: independent random phase per bin, shape (K,)
        g = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=K))

        # Outer product: (M,) x (K,) -> (M, K).  Rank-1 by construction.
        raw   = np.outer(a, g).astype(np.complex128)
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

    n_modes independent Doppler tones along the pulse axis (rows) with
    exponentially decaying power (decay_db per mode).  SCM rank = n_modes;
    knee at index n_modes.

    Each mode has an independent Gaussian range weighting of shape (K,),
    giving partial decorrelation across range bins.

    Outer product per mode: np.outer(a_i, g_i) -> (M, K).

    Parameters
    ----------
    jnr_min_db     : lower bound of JNR range (inclusive).
    jnr_max_db     : upper bound of JNR range (inclusive).
    n_modes        : number of Doppler modes / SCM rank.
    decay_db       : power decay per mode in dB (default 3 dB).
    doppler_spread : total Doppler bandwidth in cycles/pulse.
    seed           : master seed.
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

    def generate(self, M: int, K: int, noise_db: float) -> RFIField:
        """
        Generate a (M, K) wideband RFI matrix.

        Parameters
        ----------
        M        : Pulses (rows).
        K        : Range bins (columns).
        noise_db : Noise floor in dB; JNR added on top.
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

        raw = np.zeros((M, K), dtype=np.complex128)
        for i, fd in enumerate(dopplers):
            amp = np.sqrt(10.0 ** (-self.decay_db * i / 10.0))

            # Slow-time steering vector: shape (M,)
            a = np.exp(1j * (2.0 * np.pi * fd * pulse_idx
                             + rng.uniform(0.0, 2.0 * np.pi)))

            # Independent Gaussian range weighting: shape (K,)
            g  = rng.standard_normal(K) + 1j * rng.standard_normal(K)
            g /= np.sqrt(2.0 * K)

            # Outer product: (M,) x (K,) -> (M, K)
            raw += amp * np.outer(a, g)

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

_RFI_STYLES: dict[str, tuple[type, dict]] = {
    "cw_tone":  (CWToneRFI,  {}),
    "wideband": (WidebandRFI, {"n_modes": 4}),
}


def _make_rfi_field(
    style: str,
    M: int,
    K: int,
    noise_db: float,
    jnr_min_db: int,
    jnr_max_db: int,
    rng: np.random.Generator,
) -> RFIField:
    """
    Generate one (M, K) RFI field from an already-seeded Generator.

    Bypasses the class-level call_idx counter so the caller's SeedSequence
    tree controls all randomness. Used by synth_stap and gen_stap_dataset.

    Parameters
    ----------
    style      : "cw_tone" or "wideband"
    M          : Pulses (rows).
    K          : Range bins (columns).
    noise_db   : Noise floor in dB.
    jnr_min_db : Lower bound of integer JNR draw (inclusive).
    jnr_max_db : Upper bound of integer JNR draw (inclusive).
    rng        : Pre-seeded Generator.
    """
    if style not in _RFI_STYLES:
        raise ValueError(
            f"Unknown RFI style {style!r}. Choose from: {list(_RFI_STYLES)}"
        )

    _, extra     = _RFI_STYLES[style]
    jnr_db       = _draw_jnr(jnr_min_db, jnr_max_db, rng)
    rfi_power_db = noise_db + jnr_db

    if style == "cw_tone":
        doppler   = float(rng.uniform(-0.4, 0.4))
        pulse_idx = np.arange(M)
        phi       = rng.uniform(0.0, 2.0 * np.pi)
        a         = np.exp(1j * (2.0 * np.pi * doppler * pulse_idx + phi))
        g         = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=K))
        raw       = np.outer(a, g).astype(np.complex128)
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
        raw = np.zeros((M, K), dtype=np.complex128)
        for i, fd in enumerate(dopplers):
            amp  = np.sqrt(10.0 ** (-decay_db * i / 10.0))
            a    = np.exp(1j * (2.0 * np.pi * fd * pulse_idx
                                + rng.uniform(0.0, 2.0 * np.pi)))
            g    = rng.standard_normal(K) + 1j * rng.standard_normal(K)
            g   /= np.sqrt(2.0 * K)
            raw += amp * np.outer(a, g)
        field = _scale_to_power(raw, rfi_power_db)
        return RFIField(
            field=field, style=style, jnr_db=jnr_db,
            rfi_power_db=rfi_power_db, n_modes=n_modes,
        )

    raise ValueError(f"Unreachable: unknown style {style!r}")


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
    from synth_stap import compute_scm

    M, K     = 16, 128
    noise_db = 3.0

    print("=" * 60)
    print(f"CPI block shape: (M={M} pulses, K={K} range bins)")
    print(f"SCM: R = S @ S.conj().T / K,  shape ({M}, {M})")
    print()

    for cls, kwargs in [
        (CWToneRFI,  {"jnr_min_db": 10, "jnr_max_db": 20, "seed": 7}),
        (WidebandRFI, {"jnr_min_db": 10, "jnr_max_db": 20, "n_modes": 4, "seed": 7}),
    ]:
        gen    = cls(**kwargs)
        result = gen.generate(M=M, K=K, noise_db=noise_db)

        R      = compute_scm(result.field)
        evs    = np.sort(np.linalg.eigvalsh(R))[::-1]
        evs_db = 10.0 * np.log10(np.maximum(evs, 1e-30))

        print(f"style         : {result.style}")
        print(f"  field shape : {result.field.shape}  (M rows=pulses, K cols=range bins)")
        print(f"  SCM shape   : {R.shape}")
        print(f"  jnr_db      : {result.jnr_db} dB")
        print(f"  power       : expected {noise_db + result.jnr_db:+.1f} dB"
              f"  measured {result.measured_power_db():+.3f} dB")
        print(f"  top-5 evs   : {np.round(evs_db[:5], 1).tolist()}")
        print()

    print("=" * 60)
