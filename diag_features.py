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

# --- new-globals variant (train_new_globals.py / model_new_globals.py) -------
#
# Two SCALAR diagonal statistics appended to the benchmark global vector, in
# place of the whole sorted profile on its own conv branch. See
# new_global_features() for the rationale.
N_GLOBAL_NEW = 5           # [cond_db, eff_rank, diag_median_max_ratio,
                           #  schur_horn_gap, participation_ratio]

# Prefix length used for the Schur-Horn gap. The gap curve falls to ~0 once
# k >= label, so a single k is a soft "is the label at most k" indicator rather
# than a counter; k=3 separates labels 1..6 with F=0.69 on data/amazon_contam
# (k=4 measured 0.70, inside the noise). Exposed so it can be swept.
GAP_K = 3

# The gap is a fraction of the trace, so it lives in [0, 1] with a p99 of about
# 0.19, roughly 100x smaller than cond_db. The global branch is a plain Dense
# with no input normalization and no stored train statistics, so the feature is
# reported in PERCENT of trace to put it in the same range as its neighbours.
# A fixed constant, not a fitted scaler -- nothing to serialize, nothing to
# drift between train, test and score.
GAP_SCALE = 100.0


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


# ---------------------------------------------------------------------------
# SCALAR DIAGONAL STATISTICS
#
# Both are computed on the FULL M=16 entries, not the top-12 kept for the
# profile. For the gap that is not optional: Schur-Horn guarantees
# non-negativity only for the complete vectors, and renormalizing a top-12
# truncation of each produces small negative values (measured down to -0.019).
# ---------------------------------------------------------------------------

def _masked_diag_pmf(diag_lin, diag_valid_idx):
    """
    Valid SCM diagonal entries as a probability mass function over pulse rows,
    plus a mask of rows carrying no usable power.

    Invalid entries are zeroed rather than dropped, so the vector keeps length
    M and the sort order below stays meaningful.
    """
    diag = np.atleast_2d(np.asarray(diag_lin, dtype=np.float64))
    valid = np.atleast_2d(np.asarray(diag_valid_idx, dtype=bool))

    d = np.where(valid & np.isfinite(diag), np.maximum(diag, 0.0), 0.0)
    tot = d.sum(axis=1)
    dead = ~(tot > 0)
    pmf = d / np.maximum(tot, EPS)[:, None]
    return pmf, dead


def participation_ratio(diag_lin, diag_valid_idx):
    """
    Effective number of pulse rows carrying the tile's power.

        N_eff = (sum_i d_i)^2 / sum_i d_i^2

    Equal power in all M rows gives M; one dominant row gives 1; k equal rows
    and the rest empty gives k. Numerator and denominator are both quadratic in
    d, so the statistic is scale-free BY CONSTRUCTION -- there is no max- or
    median-normalization to choose and therefore no per-tile offset to shift
    between synthetic and real scenes. That is the failure mode that motivated
    switching the sorted profile from median to max normalization; this feature
    cannot have it.

    It is the diagonal's counterpart to `eff_rank` in
    train_only.features_from_eigenvalues: eff_rank is exp(Shannon entropy) of
    the normalized eigenvalues (a Hill number of order 1), this is 1/sum(p^2)
    of the normalized diagonal (order 2).

    CAVEAT -- this is a CONCENTRATION measure, not a counter. It is dominated by
    the strongest rows, so at a fixed label it still moves with band strength:
    on data/amazon_contam at label=4 the median runs 6.95 (strongest band 3-10
    dB) down to 1.98 (20-31 dB). Treat it as context for the eigenvalue branch's
    count, not as something a count can be read off.

    Dead rows (no valid entry, or no positive power) return 0.0, outside the
    natural [1, M] range so they stay distinguishable downstream.
    """
    single = (np.asarray(diag_lin).ndim == 1)
    pmf, dead = _masked_diag_pmf(diag_lin, diag_valid_idx)

    neff = 1.0 / np.maximum((pmf ** 2).sum(axis=1), EPS)
    neff = np.where(dead, 0.0, neff).astype(np.float32)

    return float(neff[0]) if single else neff


def schur_horn_gap(eigvals_linear, diag_lin, diag_valid_idx, k=GAP_K,
                   scale=GAP_SCALE):
    """
    Majorization gap between the eigenvalue and diagonal spectra at prefix k.

        gap_k = sum_{i<k} lambda_(i)/trace  -  sum_{i<k} d_(i)/trace

    both sorted descending. The Schur-Horn theorem guarantees gap_k >= 0 for
    every k, with equality at k = M; verified here at min -6.3e-9 over 60k
    tiles x 16 prefixes, i.e. zero to floating point.

    What it measures
    ----------------
    Eigendecomposition finds the rotation of pulse space that concentrates power
    as tightly as possible; the diagonal is the power in the UN-rotated pulse
    basis. So the gap is how much concentration you gain by rotating away from
    the pulse basis -- a measure of interference coherence ACROSS pulses.

      RFI confined to individual pulse rows -> covariance already near-diagonal
      in the pulse basis -> nothing to gain -> gap ~ 0
      RFI spread coherently over all pulses -> flat diagonal but one eigenvector
      captures everything -> gap large

    This is the only diagonal statistic here that is not recoverable from either
    branch alone: eigenvalues are invariant under unitary rotation of pulse
    space and cannot know what the pulse basis is, while the diagonal discards
    all inter-pulse phase. The coherence lives only in the comparison.

    Note the null is NOT zero. On clean tiles the SCM's eigenvalues spread by
    Marchenko-Pastur while the diagonal (a plain mean over K range samples)
    stays flat, so gap@3 sits at +0.164 on clean data/amazon_contam tiles and
    falls to +0.002 once RFI is present. RFI REDUCES the gap here.

    Returned in percent of trace (see GAP_SCALE). Dead tiles return 0.0.
    """
    single = (np.asarray(eigvals_linear).ndim == 1)

    ev = np.atleast_2d(np.asarray(eigvals_linear, dtype=np.float64))
    ev = np.where(np.isfinite(ev), np.maximum(ev, 0.0), 0.0)
    ev_tot = ev.sum(axis=1)
    ev_dead = ~(ev_tot > 0)
    en = np.sort(ev, axis=1)[:, ::-1] / np.maximum(ev_tot, EPS)[:, None]

    pmf, diag_dead = _masked_diag_pmf(diag_lin, diag_valid_idx)
    dn = np.sort(pmf, axis=1)[:, ::-1]

    kk = int(np.clip(k, 1, min(en.shape[1], dn.shape[1])))
    gap = en[:, :kk].sum(axis=1) - dn[:, :kk].sum(axis=1)

    # Non-negativity is a theorem; clip guards against float error and against
    # the eigenvalue/diagonal masks disagreeing on a degenerate tile.
    gap = np.maximum(gap, 0.0) * scale
    gap = np.where(ev_dead | diag_dead, 0.0, gap).astype(np.float32)

    return float(gap[0]) if single else gap


def new_global_features(global_old, eigvals_linear, diag_lin, diag_valid_idx,
                        k=GAP_K, scale=GAP_SCALE):
    """
    The benchmark global vector with the two scalar diagonal statistics appended.

        [cond_db, eff_rank, diag_median_max_ratio, schur_horn_gap, N_eff]

    `global_old` is whatever train_only.features_from_eigenvalues returned, so
    the first three entries are byte-for-byte the benchmark's and any difference
    in results is attributable to the two added scalars.

    Why scalars instead of the sorted profile on its own conv branch
    ---------------------------------------------------------------
    Neither of these is domain-robust -- N_eff measures 1.8 on synthetic label-1
    tiles and about 9.9 on real scenes, the same shift the sorted profile has,
    because it summarizes the same structure. What changes is how much of that
    shift the model can fit. The diagonal conv branch in model_diag.py carries
    roughly 139k parameters, plus about 16k more from widening the fusion Dense,
    against roughly 553k for the eigenvalue branch -- about a fifth of the
    model's conv capacity sitting on a feature whose link to the label
    (label == number of elevated rows) is an artifact of the one-band-per-row
    injection in generate_amazon_data.py. Two scalars through the existing
    Dense(64) add 128 parameters and can express a monotone response, not an
    arbitrary positional pattern.

    Two further defects of the profile that scalars simply do not have: it forces
    a normalization choice (max vs median, the suspected cause of whole-scene
    over-flagging in HH), and its invalid tail has to be padded, which
    manufactures structure -- far-range amazon tiles with one valid row of
    sixteen come out flat at exactly 0.0 dB, indistinguishable from sixteen
    genuinely equal rows.

    This is a mitigation, NOT a fix. The label-to-diagonal relationship is still
    an injection-model artifact; only changing inject_rfi_bands removes it.
    """
    g = np.atleast_2d(np.asarray(global_old, dtype=np.float32))
    gap = np.atleast_1d(schur_horn_gap(
        eigvals_linear, diag_lin, diag_valid_idx, k=k, scale=scale))
    neff = np.atleast_1d(participation_ratio(diag_lin, diag_valid_idx))

    out = np.concatenate(
        [g[:, :N_GLOBAL_DIAG], gap[:, None], neff[:, None]], axis=1
    ).astype(np.float32)

    if out.shape[1] != N_GLOBAL_NEW:
        raise ValueError(
            f'expected {N_GLOBAL_NEW} global features, built {out.shape[1]}')

    return out[0] if np.asarray(global_old).ndim == 1 else out
