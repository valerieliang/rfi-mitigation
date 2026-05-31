"""
rfi_gen.pulsed

Transient / pulsed RFI present in only part of the CPI.

The interferer is active on a contiguous burst of pulses and silent
elsewhere. A coherence knob sets the character of the burst:
  coherence near 1  -> the burst is a clean tone; rank stays near 1 and
                       the knee is sharp despite the gating.
  coherence near 0  -> the burst is wideband noise; each active pulse is
                       an independent degree of freedom, so the rank
                       approaches the burst length and several mid-size
                       eigenvalues appear, smearing the knee.
Intermediate coherence gives one dominant eigenvalue plus a tail, an
ambiguous boundary that mimics real transient bursts.

Tuning:
  duty        fraction of the CPI occupied by the burst
  coherence   tone-like (1) to noise-like (0) content within the burst
  doppler     slow-time frequency of the coherent part
"""

import numpy as np

from .base import RFIGenerator


class PulsedBurst(RFIGenerator):
    style = "pulsed"

    def __init__(self, inr_db=24.0, duty=0.4, coherence=0.5,
                 doppler=0.2, **kw):
        super().__init__(inr_db=inr_db, duty=float(duty),
                         coherence=float(coherence), doppler=float(doppler),
                         **kw)
        self.duty = float(np.clip(duty, 0.05, 1.0))
        self.coherence = float(np.clip(coherence, 0.0, 1.0))
        self.doppler = float(doppler)

    def _unit_field(self, M, K, rng):
        p = max(1, int(round(self.duty * M)))
        start = int(rng.integers(0, M - p + 1))
        window = np.zeros(M)
        window[start:start + p] = 1.0

        m = np.arange(M)
        tone = np.exp(1j * (2.0 * np.pi * self.doppler * m
                            + rng.uniform(0.0, 2.0 * np.pi)))
        g = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=K))

        coherent = np.sqrt(self.coherence) * np.outer(tone * window, g)
        incoherent = np.sqrt(1.0 - self.coherence) * (
            rng.standard_normal((M, K)) + 1j * rng.standard_normal((M, K)))
        incoherent *= window[:, None]
        return coherent + incoherent


if __name__ == "__main__":
    r = PulsedBurst().generate(M=32, K=128, rng=np.random.default_rng(0))
    print("style", r.style, "knee_truth", r.knee_truth,
          "contrast_db", round(r.contrast_db, 1))
    print("top eigenvalues (dB):", np.round(r.profile_db()[:12], 1))
