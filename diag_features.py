"""
diag_features.py

Sorted SCM diagonal profile feature, shared by train_diag_profile.py,
test_only.py and score_scene.py so training, evaluation and scene scoring cannot
drift apart.

Pure numpy on purpose: test_only.py imports TensorFlow lazily inside main(), and
this module must not drag it in at import time.

Background
----------
generate_amazon_data.py places each injected RFI band in its OWN pulse row, drawn
without replacement, so

    label == number of distinct contaminated rows == number of RFI eigenvalues

The SCM diagonal is per-pulse-row power and keeps all M entries plus a validity
mask, which makes the label the position of the step in the sorted profile. Any
single scalar summary destroys that:

  median/max  median sits in the clean group for label <= M/2 (always true when
              max_bands=6), so it reports the strongest band's JSR and saturates
              at label=1 -- a presence detector, not a counter.
  min/median  min and median are BOTH in the clean group, so it measures clean-row
              speckle spread and carries no RFI information at all.

`median/max` is nonetheless kept as the third GLOBAL scalar (N_GLOBAL_DIAG = 3),
alongside the profile rather than instead of it. The profile is max-normalized,
so max/median is not recoverable from it; the scalar restores that absolute
strength offset -- roughly the strongest band's JSR -- as context for the counting
decision the conv branch makes. It is a presence/strength cue, not the counter.

Normalization
-------------
Divide by the MAX of the valid entries in LINEAR, then convert to dB -- the same
convention train_only.features_from_eigenvalues uses for the eigenvalue profile
(ev / lam_max, then 10*log10). Entry 0 is therefore pinned at 0 dB and every other
entry is <= 0, exactly like the eigenvalue profile.

This replaced an earlier median normalization. The two differ by nothing but a
per-tile constant, since

    10log10(d/max) = 10log10(d/median) - 10log10(max/median)

so the SLOPE channel is identical either way and only the dB channel shifts. What
the change buys is invariance to `max/median` itself, which is roughly the
strongest band's JSR and which sat in a visibly different range for real HH scene
tiles than for synthetic training tiles -- the suspected cause of whole-scene
over-flagging in HH. The cost is that the absolute strength information carried by
that offset is no longer available to the network.

Note this makes the diagonal a KNEE-FINDING problem structurally identical to the
eigenvalue profile (locate the drop), rather than the thresholding problem median
normalization gave (count entries above 0 dB).

Scope
-----
Strictly PER-CPI. Every quantity here comes from one tile's own SCM diagonal;
nothing looks at neighbouring CPIs, absolute scene power, or anything outside
the tile.

The profile is a SINGLE channel (sorted dB) of length N_KEEP_DIAG = 12, the
largest valid entries, mirroring the eigenvalue profile's top-12-of-16. With
max_bands = 6 the kept 12 still hold every contaminated row plus six clean
rows of reference, so the step stays inside the window and nothing needed for
counting is discarded. Both branches are then the same length.

There is no first-difference channel: the conv stem has a width-5 kernel and
can learn a difference operator itself if one helps. `valid_frac` is still
returned but is NO LONGER a model input -- score_scene.py records it beside
the predictions as diagnostic metadata only.
"""

import numpy as np

# Local copies, matching train_only.py / test_only.py. Defined here rather than
# imported so this module stays free of a TensorFlow dependency.
EPS = 1e-12
DB_FLOOR = -100.0

# Channels in the diagonal profile tensor, and length of the global vector
# that accompanies it. Shared so train / test / score cannot disagree.
DIAG_CHANNELS = 1          # sorted dB only
N_GLOBAL_DIAG = 3          # [cond_db, eff_rank, diag_median_max_ratio]
N_KEEP_DIAG = 12           # largest valid diagonal entries kept, of M=16


def diag_profile_features(diag_lin, diag_valid_idx):
    """
    Sorted valid SCM diagonal profile, in dB relative to its own max.

    Returns (profile, valid_frac):
      profile    (N, N_KEEP_DIAG, 1) float32 -- the N_KEEP_DIAG largest valid
                                      entries, descending, entry 0 = 0 dB
      valid_frac (N,)      float32 -- fraction of the M entries that were
                                      valid. DIAGNOSTIC ONLY, not a model input.

    Invalid entries are excluded from both the sort and the max, then the tail
    is padded by repeating the last valid value BEFORE truncation to
    N_KEEP_DIAG, so a tile with fewer than N_KEEP_DIAG valid entries still
    yields a flat continuation rather than a hole. Padding with DB_FLOOR instead would
    manufacture a cliff at index n_valid that tracks validity rather than RFI --
    exactly the spurious structure a conv latches onto. `valid_frac` goes to the
    global branch so the network can discount short profiles.
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

    # Max over the valid entries is simply the first sorted entry, since the
    # invalid ones were pushed to the back before sorting. Rows with nothing
    # valid leave -inf here, which the `dead` mask below catches.
    mx = srt[:, 0]

    # Rows with nothing valid, or a non-positive max, carry no information:
    # emit a flat zero profile rather than pushing inf/nan into the graph.
    dead = (n_valid == 0) | ~(mx > 0)

    ratio = np.maximum(srt, EPS) / np.maximum(mx, EPS)[:, None]
    prof_db = 10.0 * np.log10(np.maximum(ratio, EPS))
    prof_db = np.maximum(prof_db, DB_FLOOR)
    prof_db = np.where(dead[:, None], 0.0, prof_db)

    # Keep the largest N_KEEP_DIAG, matching the eigenvalue branch's top-12-of-16.
    prof_db = prof_db[:, :N_KEEP_DIAG]

    profile = prof_db[:, :, None].astype(np.float32)
    valid_frac = (n_valid / float(m)).astype(np.float32)

    if single:
        return profile[0], valid_frac[0]
    return profile, valid_frac
