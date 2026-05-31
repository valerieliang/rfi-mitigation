"""
rfi_gen.wideband

Wideband modulated emitter.

A wideband interferer is only partially coherent across pulses, so it
occupies a small band of slow-time frequencies rather than a single one.
Modeled here as several Doppler components with geometrically decaying
power, it produces a handful of eigenvalues that fall away gradually
instead of a single clean drop. The knee is real but soft: where you
draw it depends on the decay rate.

Tuning:
  n_modes      number of Doppler components in the band
  decay_db     power roll-off per component (larger -> sharper knee)
  doppler_spread  width of the occupied slow-time band
"""

import numpy as np

from .base import RFIGenerator


class Wideband(RFIGenerator):
    style = "wideband"

    def __init__(self, inr_db=20.0, n_modes=6, decay_db=3.0,
                 doppler_spread=0.9, **kw):
        super().__init__(inr_db=inr_db, nominal_knee=int(n_modes),
                         n_modes=int(n_modes), decay_db=float(decay_db),
                         doppler_spread=float(doppler_spread), **kw)
        self.n_modes = int(n_modes)
        self.decay_db = float(decay_db)
        self.doppler_spread = float(doppler_spread)

    def _unit_field(self, M, K, rng):
        m = np.arange(M)
        dopplers = np.linspace(-self.doppler_spread / 2.0,
                               self.doppler_spread / 2.0, self.n_modes)
        s = np.zeros((M, K), dtype=complex)
        for i, fd in enumerate(dopplers):
            amp = np.sqrt(10.0 ** (-self.decay_db * i / 10.0))
            a = np.exp(1j * (2.0 * np.pi * fd * m
                             + rng.uniform(0.0, 2.0 * np.pi)))
            # Independent range weighting per mode -> partial decorrelation.
            g = rng.standard_normal(K) + 1j * rng.standard_normal(K)
            s += amp * np.outer(a, g) / np.sqrt(K)
        return s


if __name__ == "__main__":
    r = Wideband().generate(M=32, K=128, rng=np.random.default_rng(0))
    print("style", r.style, "knee_truth", r.knee_truth,
          "nominal", r.nominal_knee, "contrast_db", round(r.contrast_db, 1))
    print("top eigenvalues (dB):", np.round(r.profile_db()[:10], 1))
