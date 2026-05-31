"""
rfi_gen.cw_tone

Single narrowband continuous-wave (CW) emitter.

A CW interferer is coherent across all pulses of the CPI, so its
slow-time signature is a single steering vector: the RFI-only covariance
is rank 1. The eigenvalue profile therefore shows ONE dominant
eigenvalue and then an immediate drop to the signal plateau, i.e. the
sharpest possible knee, located at index 1.

Obviousness is controlled entirely by inr_db (default high). Lower it
toward 0 dB and even this rank-1 case becomes hard to see; for a
dedicated faint style see rfi_gen.subtle.
"""

import numpy as np

from .base import RFIGenerator


class CWTone(RFIGenerator):
    style = "cw_tone"

    def __init__(self, inr_db=28.0, doppler=0.13, **kw):
        # doppler is the normalized slow-time frequency of the tone,
        # in cycles per pulse, range about (-0.5, 0.5).
        super().__init__(inr_db=inr_db, nominal_knee=1, doppler=doppler, **kw)
        self.doppler = float(doppler)

    def _unit_field(self, M, K, rng):
        m = np.arange(M)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        a = np.exp(1j * (2.0 * np.pi * self.doppler * m + phi))
        # Range-domain waveform: a broadband complex pattern across range
        # samples. It does not change the rank, only the column weighting.
        g = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=K))
        return np.outer(a, g)


if __name__ == "__main__":
    r = CWTone().generate(M=32, K=128, rng=np.random.default_rng(0))
    print("style", r.style, "knee_truth", r.knee_truth,
          "contrast_db", round(r.contrast_db, 1))
    print("top eigenvalues (dB):", np.round(r.profile_db()[:5], 1))
