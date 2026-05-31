"""
rfi_gen.chirp

Frequency-swept (chirp / LFM-like) emitter.

The interferer sweeps in frequency across the acquisition, so its
slow-time Doppler differs from one range sample to the next. The
covariance then integrates over a whole band of steering vectors and its
eigenvalues fill that band, decaying gradually with no single dominant
drop. This is the hardest case for any knee detector: the boundary
between RFI and signal is not well defined.

Tuning:
  sweep   width of the swept slow-time band (larger -> flatter, more
          eigenvalues, even less of a knee)
  f0      center Doppler of the sweep
"""

import numpy as np

from .base import RFIGenerator


class ChirpSweep(RFIGenerator):
    style = "chirp"

    def __init__(self, inr_db=20.0, sweep=0.8, f0=0.0, **kw):
        super().__init__(inr_db=inr_db, sweep=float(sweep), f0=float(f0), **kw)
        self.sweep = float(sweep)
        self.f0 = float(f0)

    def _unit_field(self, M, K, rng):
        m = np.arange(M)
        # Doppler varies linearly across range samples: a continuous band
        # of steering vectors, one per column.
        f = self.f0 + self.sweep * (np.arange(K) / float(K) - 0.5)
        phase = 2.0 * np.pi * np.outer(m, f)            # (M, K)
        amp = rng.standard_normal(K) + 1j * rng.standard_normal(K)
        return np.exp(1j * phase) * amp[None, :]


if __name__ == "__main__":
    r = ChirpSweep().generate(M=32, K=128, rng=np.random.default_rng(0))
    print("style", r.style, "knee_truth", r.knee_truth,
          "contrast_db", round(r.contrast_db, 1))
    print("top eigenvalues (dB):", np.round(r.profile_db()[:12], 1))
