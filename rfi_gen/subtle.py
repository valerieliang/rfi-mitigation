"""
rfi_gen.subtle

Faint, near-noise RFI.

Structurally this is one or two narrowband tones (low rank, so the knee
sits at a small index), but the defining feature is a very low INR: the
RFI eigenvalues barely rise above the signal plateau. The knee is
therefore present in the label but hard or impossible to see in the
profile, reproducing the low-sigma-ratio / Type A regime where a simple
slope threshold misses the boundary.

This is the "not at all obvious" end of the obvious-to-none range, driven
by contrast rather than by spectrum shape. Raise inr_db to make it
gradually emerge.
"""

import numpy as np

from .base import RFIGenerator


class SubtleNearNoise(RFIGenerator):
    style = "subtle"

    def __init__(self, inr_db=3.0, n_tones=1, doppler_spread=0.4, **kw):
        super().__init__(inr_db=inr_db, nominal_knee=int(n_tones),
                         n_tones=int(n_tones),
                         doppler_spread=float(doppler_spread), **kw)
        self.n_tones = int(n_tones)
        self.doppler_spread = float(doppler_spread)

    def _unit_field(self, M, K, rng):
        m = np.arange(M)
        if self.n_tones == 1:
            dopplers = np.array([rng.uniform(-0.25, 0.25)])
        else:
            dopplers = np.linspace(-self.doppler_spread / 2.0,
                                   self.doppler_spread / 2.0, self.n_tones)
        s = np.zeros((M, K), dtype=complex)
        for fd in dopplers:
            a = np.exp(1j * (2.0 * np.pi * fd * m
                             + rng.uniform(0.0, 2.0 * np.pi)))
            g = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=K))
            s += np.outer(a, g)
        return s


if __name__ == "__main__":
    r = SubtleNearNoise().generate(M=32, K=128, rng=np.random.default_rng(0))
    print("style", r.style, "knee_truth", r.knee_truth,
          "contrast_db", round(r.contrast_db, 1))
    print("top eigenvalues (dB):", np.round(r.profile_db()[:5], 1))
