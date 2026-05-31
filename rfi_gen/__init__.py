"""
rfi_gen

Synthetic RFI generators for slow-time eigenvalue / knee experiments.

Each style injects RFI into a complex slow-time block (M pulses by K
range samples) so the resulting eigenvalue profile has a knee of a
distinct character. Ordered roughly from the most obvious knee to none
at all:

  cw_tone    single CW tone, rank 1, sharpest possible knee at index 1
  multitone  several tones, knee placed at a chosen index r
  wideband   partially coherent band, a few decaying eigenvalues, soft
  pulsed     transient burst, smeared / ambiguous boundary
  subtle     low-INR tones, knee present in the label but barely visible
  chirp      frequency sweep, eigenvalues fill a band, no clear knee

Two independent knobs:
  inr_db        how far the RFI rises above the plateau (obviousness)
  style params  the shape of the RFI eigen-spectrum (rank, spread, ...)

Typical use:
  from rfi_gen import get_generator
  gen = get_generator("multitone", n_tones=3, inr_db=22)
  real = gen.apply(clean_block)          # overlay on a real clean block
  label = real.knee_truth                # training target for model.py
  profile = real.profile_db()            # (M,) eigenvalue profile in dB
"""

from .base import (
    RFIGenerator, RFIRealization,
    slow_time_scm, eig_profile, eig_db, slope_db, synthetic_clean,
    effective_rank, knee_contrast_db, estimate_knee, cpi_features,
)
from .cw_tone import CWTone
from .multitone import MultiTone
from .wideband import Wideband
from .pulsed import PulsedBurst
from .subtle import SubtleNearNoise
from .chirp import ChirpSweep

# Registry in rough obvious-to-none order.
STYLES = {
    "cw_tone": CWTone,
    "multitone": MultiTone,
    "wideband": Wideband,
    "pulsed": PulsedBurst,
    "subtle": SubtleNearNoise,
    "chirp": ChirpSweep,
}


def get_generator(name, **kwargs):
    """Instantiate a generator by style name with optional overrides."""
    key = name.lower()
    if key not in STYLES:
        raise KeyError("unknown style {!r}; choose from {}"
                       .format(name, sorted(STYLES)))
    return STYLES[key](**kwargs)


def all_generators(**common_kwargs):
    """One generator per style, sharing any common keyword arguments."""
    return {name: cls(**common_kwargs) for name, cls in STYLES.items()}


__all__ = [
    "RFIGenerator", "RFIRealization", "STYLES",
    "get_generator", "all_generators",
    "CWTone", "MultiTone", "Wideband", "PulsedBurst",
    "SubtleNearNoise", "ChirpSweep",
    "slow_time_scm", "eig_profile", "eig_db", "slope_db", "synthetic_clean",
    "effective_rank", "knee_contrast_db", "estimate_knee", "cpi_features",
]
