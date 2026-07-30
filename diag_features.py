"""
diag_features.py

Sorted SCM diagonal profile feature, shared by train_diag_profile.py and
test_only.py so training and evaluation cannot drift apart.

Pure numpy on purpose: test_only.py imports TensorFlow lazily inside main(), and
this module must not drag it in at import time.

Background
----------
generate_amazon_data.py places each injected RFI band in its OWN pulse row, drawn
without replacement, so

    label == number of distinct contaminated rows == number of RFI eigenvalues

The SCM diagonal is per-pulse-row power and keeps all M entries plus a validity
mask, which makes the label the COUNT of elevated entries in that vector. Any
single scalar summary destroys that:

  median/max  median sits in the clean group for label <= M/2 (always true when
              max_bands=6), so it reports the strongest band's JSR and saturates
              at label=1 -- a presence detector, not a counter.
  min/median  min and median are BOTH in the clean group, so it measures clean-row
              speckle spread and carries no RFI information at all.

The sorted profile keeps the step at index k, which is countable.
"""

import warnings

import numpy as np

# Local copies, matching train_only.py / test_only.py. Defined here rather than
# imported so this module stays free of a TensorFlow dependency.
EPS = 1e-12
DB_FLOOR = -100.0


def diag_profile_features(diag_lin, diag_valid_idx):
    """
    Sorted valid SCM diagonal profile, in dB relative to its own median.

    Returns (profile, valid_frac):
      profile    (N, M, 2) float32 -- [sorted_db, first_diff], descending
      valid_frac (N,)      float32 -- fraction of the M entries that were valid

    Normalization is by the MEDIAN of the valid entries, not the max. With
    max_bands=6 of M=16 rows the median always lies inside the clean group, so it
    is a robust estimate of uncontaminated pulse power: elevated entries come out
    POSITIVE in dB and floor entries near 0, putting the step the network must
    count at the 0 dB crossing.

    Invalid entries are excluded from both the sort and the median, then the tail
    is padded by repeating the last valid value. Padding with DB_FLOOR instead
    would manufacture a cliff at index n_valid that tracks validity rather than
    RFI -- exactly the spurious structure a conv latches onto. `valid_frac` goes
    to the global branch so the network can discount short profiles.
    """
    single = (np.asarray(diag_lin).ndim == 1)
    diag = np.atleast_2d(np.asarray(diag_lin, dtype=np.float64))
    valid = np.atleast_2d(np.asarray(diag_valid_idx, dtype=bool))

    n, m = diag.shape
    n_valid = valid.sum(axis=1)
    rows = np.arange(n)

    # Sort descending with invalid entries forced to the back.
    keyed = np.where(valid, diag, -np.inf)
    order = np.argsort(-keyed, axis=1, kind='stable')
    srt = np.take_along_axis(keyed, order, axis=1)

    # Repeat the last valid value across the invalid tail.
    last_valid = srt[rows, np.maximum(n_valid - 1, 0)]
    pad = np.arange(m)[None, :] >= n_valid[:, None]
    srt = np.where(pad, last_valid[:, None], srt)

    # Median over valid entries only. All-invalid rows warn and yield NaN; NaN is
    # the right answer there and the `dead` mask below handles it.
    masked = np.where(valid, diag, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        med = np.nanmedian(masked, axis=1)
    med = np.where(np.isfinite(med), med, 0.0)

    # Rows with nothing valid, or a non-positive median, carry no information:
    # emit a flat zero profile rather than pushing inf/nan into the graph.
    dead = (n_valid == 0) | ~(med > 0)

    ratio = np.maximum(srt, EPS) / np.maximum(med, EPS)[:, None]
    prof_db = 10.0 * np.log10(np.maximum(ratio, EPS))
    prof_db = np.maximum(prof_db, DB_FLOOR)
    prof_db = np.where(dead[:, None], 0.0, prof_db)

    slopes = np.diff(prof_db, axis=1)
    slopes = np.concatenate([slopes, np.zeros((n, 1))], axis=1)

    profile = np.stack([prof_db, slopes], axis=-1).astype(np.float32)
    valid_frac = (n_valid / float(m)).astype(np.float32)

    if single:
        return profile[0], valid_frac[0]
    return profile, valid_frac
