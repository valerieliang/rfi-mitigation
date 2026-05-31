"""
rfi_gen.multitone

Several narrowband emitters at distinct slow-time frequencies.

Each emitter contributes one near-orthogonal steering vector, so r
distinct tones give an RFI-only covariance of rank r and a knee at index
r. With equal powers (the default) the profile shows a flat top of r
large eigenvalues then a sharp drop. A positive power_taper_db turns
that flat top into a descending staircase, which softens the knee and is
useful for generating intermediate cases.

This style is the natural way to place the knee at a specific, known
index for training.
"""

import numpy as np

from .base import RFIGenerator


class MultiTone(RFIGenerator):
    style = "multitone"

    def __init__(self, inr_db=24.0, n_tones=3, doppler_spread=0.6,
                 power_taper_db=0.0, **kw):
        super().__init__(inr_db=inr_db, nominal_knee=int(n_tones),
                         n_tones=int(n_tones),
                         doppler_spread=float(doppler_spread),
                         power_taper_db=float(power_taper_db), **kw)
        self.n_tones = int(n_tones)
        self.doppler_spread = float(doppler_spread)
        self.power_taper_db = float(power_taper_db)

    def _unit_field(self, M, K, rng):
        m = np.arange(M)
        # Distinct, evenly spaced Dopplers so the steering vectors are
        # close to orthogonal (clean, separable eigenvalues).
        if self.n_tones == 1:
            dopplers = np.array([0.0])
        else:
            dopplers = np.linspace(-self.doppler_spread / 2.0,
                                   self.doppler_spread / 2.0, self.n_tones)
        s = np.zeros((M, K), dtype=complex)
        for i, fd in enumerate(dopplers):
            amp = np.sqrt(10.0 ** (-self.power_taper_db * i / 10.0))
            a = np.exp(1j * (2.0 * np.pi * fd * m
                             + rng.uniform(0.0, 2.0 * np.pi)))
            g = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=K))
            s += amp * np.outer(a, g)
        return s


if __name__ == "__main__":
    r = MultiTone(n_tones=4).generate(M=32, K=128, rng=np.random.default_rng(0))
    print("style", r.style, "knee_truth", r.knee_truth,
          "nominal", r.nominal_knee, "contrast_db", round(r.contrast_db, 1))
    print("top eigenvalues (dB):", np.round(r.profile_db()[:7], 1))
