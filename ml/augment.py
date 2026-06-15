#!/usr/bin/env python3
"""
augment.py

Assemble a labeled training set for the knee classifier in model.py by
overlaying synthetic RFI (from the rfi_gen package) onto clean NISAR
slow-time blocks, organized into threshold blocks (TBs) so the adaptive
ST-EST global features can be computed exactly as in the IGARSS slides.

What it produces (saved to a .npz):
  eigen_input  (N, M, 2)  channel 0 = eigenvalue profile in dB
                          channel 1 = padded slope in dB
  global_input (N, 6)     standardized [F, sigma_min, sigma_max,
                          mu_min, trace_db, log10_condition_number]
  global_input_raw (N, 6) the same features before standardization
  labels       (N,)       knee index in [0, M]; 0 means no RFI
  plus per-sample bookkeeping (tb_id, split, style_id, inr) and the
  standardization stats (norm_mean, norm_std) and feature_names.

These plug straight into model.py:
  from model import build_model
  m = build_model(cpi_size=M, n_global_features=6)
  m.fit([eigen_train, global_train], y_train,
        validation_data=([eigen_val, global_val], y_val))

Threshold-block construction (mirrors the slides):
  A TB is n_cpi consecutive CPIs over one range block. Within a TB we
  track lambda_max and lambda_min per CPI (in dB), take their first
  differences across CPIs, and form:
    sigma_max = STD(diff lambda_max)
    sigma_min = STD(diff lambda_min)   (an estimate of the signal slope)
    mu_min    = MEAN(diff lambda_min)
    F         = sigma_max / sigma_min  (RFI likelihood; large -> RFI)
  F, the sigmas and mu_min are shared by every CPI in the TB; trace_db
  and the condition number are per CPI.

TB scenarios (to give the global features real discriminative signal):
  clean        no RFI in any CPI            -> labels 0, low F
  uniform      RFI in every CPI, INR jitter -> labels = knee, high F
  intermittent RFI in a random CPI subset   -> mixed labels, highest F

Data source:
  --qc-csv / --data-dir : read CLEAN granules (from check_nisar_clean.py)
                          and extract real clean blocks.
  --synthetic           : skip real data and use synthetic clean blocks
                          (lets the whole pipeline run before downloads).

Condition number is stored as log10 for numerical stability; this is the
only transform applied before standardization. No non-ASCII characters
are used in this file.

Dependencies: numpy (required), h5py (real mode only), matplotlib
(only for --figure).
"""

import argparse
import json
import os
import sys

import numpy as np

import rfi_gen
from rfi_gen import (
    get_generator, synthetic_clean, slow_time_scm, cpi_features,
)

TINY = 1e-20


# ----------------------------------------------------------------------
# Per-style random parameter draws (vary the knee index across the set)
# ----------------------------------------------------------------------
def draw_style_params(style, rng, M=32):
    """
    Random style parameters so the knee index varies across TBs.

    All rank-controlling parameters are capped so the realized knee
    index stays within [1, M//2].  The post-generation check in
    build_tb is a second safety net for generators (chirp, pulsed)
    whose rank is only approximately controlled by their parameters.
    """
    cap = knee_cap(M)
    if style == "cw_tone":
        return {"doppler": float(rng.uniform(-0.3, 0.3))}
    if style == "multitone":
        # n_tones directly sets the rank; clamp to cap.
        return {"n_tones": int(rng.integers(1, cap + 1)),
                "doppler_spread": float(rng.uniform(0.3, 0.8)),
                "power_taper_db": float(rng.choice([0.0, 0.0, 1.5]))}
    if style == "wideband":
        # n_modes approximates the rank; clamp to cap.
        return {"n_modes": int(rng.integers(4, min(9, cap + 1))),
                "decay_db": float(rng.uniform(1.5, 4.0)),
                "doppler_spread": float(rng.uniform(0.6, 1.0))}
    if style == "pulsed":
        # duty * M approximates the rank; cap duty at cap/M (~0.5).
        max_duty = cap / float(M)
        return {"duty": float(rng.uniform(0.2, max_duty)),
                "coherence": float(rng.uniform(0.1, 0.9)),
                "doppler": float(rng.uniform(-0.3, 0.3))}
    if style == "subtle":
        return {"n_tones": int(rng.integers(1, 3))}
    if style == "chirp":
        # Chirp rank grows with sweep; cap sweep at the value corresponding
        # to knee = cap via _chirp_sweep_for_knee.
        max_sweep = _chirp_sweep_for_knee(cap, M)
        return {"sweep": float(rng.uniform(0.3, max_sweep)),
                "f0": float(rng.uniform(-0.1, 0.1))}
    return {}


# ----------------------------------------------------------------------
# Targeted knee placement (for balanced generation)
# ----------------------------------------------------------------------
# Empirically calibrated parameter -> knee mappings (M = 32, K = 128):
#   multitone : knee == n_tones, exact, over 1..~28
#   wideband  : knee ~= n_modes, over 3..6
#   pulsed    : knee ~= round(duty * M), over ~6..26 at low coherence
#   chirp     : knee grows with sweep, ~14 (sweep 0.4) to 32 (sweep 1.0)
#   cw_tone   : knee 1 always
#   subtle    : knee == n_tones (1 or 2), faint by construction
#
# style_for_knee picks, for a desired knee index, one of the styles able
# to produce it (with params set to hit it) so that every knee value has
# real style variety rather than a single generator. The actual knee is
# still measured after generation; this only biases the draw.

def knee_cap(M):
    """
    Maximum allowed knee index: at most half of all eigenvalues may be
    RFI-dominated. Matches the per-CPI size caps used at inference time.
    """
    return M // 2


def _chirp_sweep_for_knee(target, M):
    """Invert the chirp sweep -> knee trend (roughly linear in [14, M//2])."""
    cap = knee_cap(M)
    lo_knee, hi_knee = 14, cap
    lo_sw, hi_sw = 0.4, 1.0
    if hi_knee <= lo_knee:
        return lo_sw
    t = (np.clip(target, lo_knee, hi_knee) - lo_knee) / float(hi_knee - lo_knee)
    return float(lo_sw + t * (hi_sw - lo_sw))


def style_for_knee(target, M, rng, allowed=None):
    """
    Return (style, params) likely to yield a knee at `target`.

    `target` is silently clamped to [1, M//2] so that no style can be
    parameterized to produce RFI across more than half the eigenvalues.
    Post-generation enforcement in build_tb drops any CPI that still
    exceeds the cap (can happen for chirp/pulsed whose rank is
    approximate).

    Several styles are offered per knee band so the balanced set keeps
    the RFI-type variety. If `allowed` is given, only those styles are
    considered (falling back to multitone, which can hit any index).
    """
    cap = knee_cap(M)
    # Clamp target to [1, cap] -- never request more than half the eigenvalues.
    target = int(np.clip(target, 1, cap))
    candidates = []

    if target == 1:
        candidates += [("cw_tone", {"doppler": float(rng.uniform(-0.3, 0.3))}),
                       ("subtle", {"n_tones": 1}),
                       ("multitone", {"n_tones": 1,
                                      "doppler_spread": 0.6})]
    elif target == 2:
        candidates += [("multitone", {"n_tones": 2, "doppler_spread": 0.7}),
                       ("subtle", {"n_tones": 2})]
    elif 3 <= target <= 6:
        candidates += [("multitone", {"n_tones": target,
                                      "doppler_spread": 0.8}),
                       ("wideband", {"n_modes": target, "decay_db": 2.0,
                                     "doppler_spread": 0.9})]
    elif 7 <= target <= min(12, cap):
        candidates += [("multitone", {"n_tones": target,
                                      "doppler_spread": 0.9}),
                       ("pulsed", {"duty": target / float(M),
                                   "coherence": float(rng.uniform(0.1, 0.3)),
                                   "doppler": float(rng.uniform(-0.3, 0.3))})]
    else:
        # target in [13, cap]: pulsed duty is capped at cap/M (~0.5).
        # chirp is excluded here because its realized rank depends on
        # sweep in a way that is not reliably invertible; it belongs
        # only in the random draw_style_params path where Layer-3
        # post-gen demotion handles any over-contaminated result.
        safe_duty = target / float(M)
        candidates += [("pulsed", {"duty": safe_duty,
                                   "coherence": float(rng.uniform(0.1, 0.3)),
                                   "doppler": float(rng.uniform(-0.3, 0.3))}),
                       ("multitone", {"n_tones": target,
                                      "doppler_spread": 0.95})]

    if allowed is not None:
        candidates = [c for c in candidates if c[0] in allowed]
    if not candidates:
        # multitone fallback: n_tones capped at cap, not M-1.
        candidates = [("multitone",
                       {"n_tones": int(np.clip(target, 1, cap)),
                        "doppler_spread": 0.9})]

    return candidates[int(rng.integers(len(candidates)))]


# ----------------------------------------------------------------------
# Clean-block sources
# ----------------------------------------------------------------------
def synthetic_strip(n_cpi, M, K, rng):
    """A list of n_cpi independent synthetic clean (M, K) blocks."""
    return [synthetic_clean(M, K, rng) for _ in range(n_cpi)]


def real_strip(h5_path, n_cpi, M, K, rng, use_l0=False):
    """
    A contiguous real strip sliced into n_cpi CPIs of (M, K).

    When use_l0 is True, reads from an L0B raw file via canvas_l0
    (pulse axis is the true slow-time axis). When False, reads from an
    RSLC file via canvas (azimuth lines used as a proxy for pulses).

    Returns a list of n_cpi blocks, or None if the scene is too small.
    """
    if use_l0:
        from rfi_gen import canvas_l0 as _canvas
    else:
        from rfi_gen import canvas as _canvas
    try:
        strip = _canvas.extract_block(h5_path, n_pulses=n_cpi * M,
                                      n_range=K, rng=rng)
    except Exception:
        return None
    return [strip[k * M:(k + 1) * M, :] for k in range(n_cpi)]


# ----------------------------------------------------------------------
# CPI / TB assembly
# ----------------------------------------------------------------------
def cpi_record(cov):
    """Extract per-CPI features and stats from a covariance matrix."""
    feats = cpi_features(cov)
    eig_db = feats["eig_db"]
    eigen = np.stack([eig_db, feats["slope_db"]], axis=-1)  # (M, 2)
    return {
        "eigen": eigen.astype(np.float32),
        "lam_max_db": float(eig_db[0]),
        "lam_min_db": float(eig_db[-1]),
        "trace_db": float(feats["trace_db"]),
        "log_cond": float(np.log10(max(feats["condition_number"], 1.0))),
    }


def build_tb(blocks, scenario, styles, inr_range, M, K, rng, tb_id,
             target_knee=None, inr_fixed=None):
    """
    Build all CPI records for one threshold block.

    blocks      : list of n_cpi clean (M, K) complex arrays
    scenario    : "clean" | "uniform" | "intermittent"
    target_knee : if set (balanced mode), choose a style + params aimed
                  at this knee index via style_for_knee, instead of a
                  uniform random style. Ignored for the clean scenario.
    inr_fixed   : if set, use this INR (with small jitter) instead of a
                  uniform draw over inr_range; lets balanced mode hold
                  obviousness roughly constant while sweeping the knee.

    Returns a list of per-CPI record dicts (global features filled in by
    the caller after the whole TB is known).
    """
    n_cpi = len(blocks)

    # Decide which CPIs carry RFI and with what generator.
    style = None
    gen = None
    base_inr = 0.0
    rfi_mask = np.zeros(n_cpi, dtype=bool)

    cap = knee_cap(M)

    if scenario != "clean":
        if target_knee is not None:
            style, params = style_for_knee(target_knee, M, rng, allowed=styles)
        else:
            style = str(rng.choice(styles))
            # Pass M so draw_style_params can cap rank-controlling params.
            params = draw_style_params(style, rng, M=M)
        base_inr = (float(inr_fixed) if inr_fixed is not None
                    else float(rng.uniform(*inr_range)))
        gen = get_generator(style, inr_db=base_inr, **params)
        if scenario == "uniform":
            rfi_mask[:] = True
        else:  # intermittent
            frac = float(rng.uniform(0.3, 0.7))
            rfi_mask = rng.random(n_cpi) < frac
            if not rfi_mask.any():
                rfi_mask[rng.integers(0, n_cpi)] = True

    records = []
    for k in range(n_cpi):
        clean = np.asarray(blocks[k]).astype(np.complex64)
        if rfi_mask[k]:
            gen.inr_db = float(base_inr + rng.uniform(-2.0, 2.0))
            real = gen.apply(clean, rng=rng)
            # Post-generation safety net: chirp and pulsed generators
            # control rank only approximately. If the realized knee
            # exceeds M//2, demote the record to clean rather than
            # mislabeling it as over-contaminated.
            if real.knee_truth > cap:
                cov = slow_time_scm(clean)
                rec = cpi_record(cov)
                rec["label"] = 0
                rec["style"] = "clean"
                rec["inr"] = float("nan")
            else:
                cov = real.covariance()
                rec = cpi_record(cov)
                rec["label"] = int(real.knee_truth)
                rec["style"] = style
                rec["inr"] = gen.inr_db
        else:
            cov = slow_time_scm(clean)
            rec = cpi_record(cov)
            rec["label"] = 0
            rec["style"] = "clean"
            rec["inr"] = float("nan")
        rec["tb_id"] = tb_id
        rec["cpi_index"] = k
        records.append(rec)

    # Threshold-block aggregates from the per-CPI lambda sequences.
    lam_max = np.array([r["lam_max_db"] for r in records])
    lam_min = np.array([r["lam_min_db"] for r in records])
    d_max = np.diff(lam_max)
    d_min = np.diff(lam_min)
    sigma_max = float(np.std(d_max)) if d_max.size else 0.0
    sigma_min = float(np.std(d_min)) if d_min.size else 0.0
    mu_min = float(np.mean(d_min)) if d_min.size else 0.0
    F = float(np.clip(sigma_max / max(sigma_min, 1e-3), 0.0, 100.0))

    for r in records:
        r["F"] = F
        r["sigma_min"] = sigma_min
        r["sigma_max"] = sigma_max
        r["mu_min"] = mu_min
    return records


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def pick_scenario(rng, clean_frac, intermittent_frac):
    if rng.random() < clean_frac:
        return "clean"
    if rng.random() < intermittent_frac:
        return "intermittent"
    return "uniform"


def _strip_provider(tb_jobs_iter, args, M, K, rng):
    """
    Yield clean CPI strips one TB at a time, transparently handling both
    synthetic and real sources. Returns (granule_name, blocks) or
    (granule_name, None) when a real scene was too small.
    """
    use_l0 = getattr(args, "l0", False)
    for job in tb_jobs_iter:
        if job is None:
            yield "synthetic", synthetic_strip(args.cpi_per_tb, M, K, rng)
        else:
            blocks = real_strip(os.path.join(args.data_dir, job),
                                args.cpi_per_tb, M, K, rng,
                                use_l0=use_l0)
            yield job, blocks


def build_balanced(tb_jobs, args, M, K, rng):
    """
    Quota-driven generation: aim for a roughly uniform number of CPI
    samples in each knee band (including a clean band), rather than
    accepting whatever the styles happen to produce.

    How it works
    ------------
    The knee axis [0, M] is split into bands (see _bands). Each band has
    a target count. For every TB we pick the most under-filled band and
    steer generation toward it: a clean band -> a clean TB; an RFI band
    -> a TB whose style and parameters are chosen (via style_for_knee)
    to land in that band. After the TB is built, each CPI is binned by
    its ACTUAL measured label; CPIs whose band is already full are
    dropped so no band overflows badly. This corrects for any residual
    mismatch between the targeted and realized knee.

    A configurable share of RFI TBs are made "intermittent" so the
    global F factor keeps a realistic, discriminative spread; their
    clean CPIs count toward the clean band and their RFI CPIs toward
    whatever band they realize.
    """
    bands = _bands(M)
    n_bands = len(bands)

    # Target per band. The clean band shares the same target as the rest
    # unless the user pins a clean fraction.
    total_target = args.target_per_band * n_bands
    target = {i: args.target_per_band for i in range(n_bands)}
    if args.clean_frac is not None:
        clean_idx = 0  # band 0 is the clean band by construction
        target[clean_idx] = int(round(args.clean_frac * total_target))

    counts = {i: 0 for i in range(n_bands)}
    overflow = max(1, int(round(args.cpi_per_tb * 0.5)))  # allowed slack

    def neediest_band():
        # Most under-filled band by remaining absolute count.
        deficits = [(target[i] - counts[i], i) for i in range(n_bands)]
        deficits.sort(reverse=True)
        return deficits[0][1] if deficits[0][0] > 0 else None

    def band_of(label):
        for i, (lo, hi) in enumerate(bands):
            if lo <= label <= hi:
                return i
        return n_bands - 1

    provider = _strip_provider(iter(tb_jobs), args, M, K, rng)
    all_records = []
    tb_id = 0
    made = 0
    max_tbs = args.max_tbs if args.max_tbs > 0 else len(tb_jobs)

    while made < max_tbs:
        nb = neediest_band()
        if nb is None:
            break  # every band met its target

        try:
            granule, blocks = next(provider)
        except StopIteration:
            # Real sources exhausted; top up with synthetic strips.
            granule, blocks = "synthetic", synthetic_strip(
                args.cpi_per_tb, M, K, rng)
        if blocks is None:
            continue  # scene too small

        if nb == 0:
            scenario = "clean"
            target_knee = None
        else:
            lo, hi = bands[nb]
            target_knee = int(rng.integers(lo, hi + 1))
            scenario = ("intermittent"
                        if rng.random() < args.intermittent_frac
                        else "uniform")

        recs = build_tb(blocks, scenario, args._styles, args._inr_range,
                        M, K, rng, tb_id, target_knee=target_knee,
                        inr_fixed=None)

        # Bin by actual label; keep only CPIs whose band still has room.
        kept = []
        for r in recs:
            b = band_of(r["label"])
            if counts[b] < target[b] + overflow:
                counts[b] += 1
                kept.append(r)
        for r in kept:
            r["granule"] = granule
        all_records.extend(kept)
        tb_id += 1
        made += 1

        if made % 25 == 0:
            filled = sum(1 for i in range(n_bands)
                         if counts[i] >= target[i])
            print("  ... {} TBs, {}/{} bands filled".format(
                made, filled, n_bands))

    return all_records, tb_id, bands, counts, target


def _bands(M):
    """
    Knee bands used by balanced generation. Band 0 is the clean band.

    All RFI bands are constrained to [1, knee_cap(M)] so every band
    is reachable by the generators. The old fixed bands (which extended
    to M) caused bands above M//2 to never fill.
    """
    cap = knee_cap(M)
    # Fine-grained low bands; merge into cap at the top.
    raw = [
        (0,  0),   # clean
        (1,  1),
        (2,  2),
        (3,  5),
        (6,  9),
        (10, 14),
    ]
    # Only add higher bands if cap allows them.
    if cap >= 15:
        raw.append((15, min(cap, 20)))
    if cap >= 21:
        raw.append((21, min(cap, 26)))
    if cap >= 27:
        raw.append((27, cap))

    # Trim any band whose lo > cap (can happen for small M).
    bands = [(lo, min(hi, cap)) for lo, hi in raw if lo <= cap]
    return bands


def main():
    ap = argparse.ArgumentParser(
        description="Build a knee-classifier training set by injecting "
                    "synthetic RFI into clean NISAR slow-time blocks.")
    ap.add_argument("--qc-csv", default=None,
                    help="QC summary CSV listing CLEAN granules.")
    ap.add_argument("--split-csv", default=None, metavar="scene_split.csv",
                    help="Scene-level split manifest produced by split_scenes.py. "
                         "When provided, only files tagged with --split are used. "
                         "Use this instead of separate train_dir/val_dir folders.")
    ap.add_argument("--split", default=None, choices=["train", "val", "test"],
                    help="Which split to build ('train', 'val', or 'test'). "
                         "Requires --split-csv.")
    ap.add_argument("--data-dir", default=".",
                    help="Directory holding the granule .h5 files.")
    ap.add_argument("--synthetic", action="store_true",
                    help="Use synthetic clean blocks instead of real data.")
    ap.add_argument("--out", default="training_set.npz",
                    help="Output .npz path.")
    ap.add_argument("--norm-stats", default=None, metavar="<train_set.npz>",
                    help="Load normalization stats (norm_mean, norm_std) from "
                         "a pre-computed training set instead of computing "
                         "from this dataset. Use this when building validation "
                         "sets: pass your train_set.npz here so val features "
                         "are standardized using train statistics (proper "
                         "cross-validation). If not set, stats are computed "
                         "from the current dataset.")
    ap.add_argument("--M", type=int, default=32, help="CPI size (pulses).")
    ap.add_argument("--k-range", type=int, default=128,
                    help="Range samples per CPI block.")
    ap.add_argument("--cpi-per-tb", type=int, default=20,
                    help="CPIs per threshold block.")
    ap.add_argument("--tbs-per-file", type=int, default=12,
                    help="Threshold blocks sampled per granule (real mode).")
    ap.add_argument("--n-tbs", type=int, default=200,
                    help="Total threshold blocks (synthetic mode).")
    ap.add_argument("--styles", default="all",
                    help="Comma list of styles, or 'all'.")
    ap.add_argument("--inr-min", type=float, default=2.0)
    ap.add_argument("--inr-max", type=float, default=30.0)
    ap.add_argument("--clean-frac", type=float, default=None,
                    help="Balanced mode: fraction of the total target that "
                         "should be clean (label 0); default gives the "
                         "clean band the same target as every other band. "
                         "Legacy mode: fraction of TBs that are fully clean "
                         "(defaults to 0.25 there).")
    ap.add_argument("--intermittent-frac", type=float, default=0.4,
                    help="Among RFI TBs, fraction that are intermittent.")
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="Validation fraction (split by TB to avoid leak).")
    # Balanced (quota-driven) generation.
    ap.add_argument("--balance", dest="balance", action="store_true",
                    default=True,
                    help="Quota-driven balanced generation (default ON).")
    ap.add_argument("--no-balance", dest="balance", action="store_false",
                    help="Disable balancing; use legacy random scenarios.")
    ap.add_argument("--target-per-band", type=int, default=300,
                    help="Balanced mode: target CPI samples per knee band.")
    ap.add_argument("--max-tbs", type=int, default=0,
                    help="Balanced mode: hard cap on TBs built "
                         "(0 = until every band meets target).")
    ap.add_argument("--l0", action="store_true",
                    help="Read raw L0B HDF5 files instead of RSLC granules. "
                         "Uses canvas_l0 in place of canvas, so --qc-csv "
                         "should point to the CSV produced by check_l0_clean.py. "
                         "The rest of the pipeline (RFI injection, SCM, TB "
                         "assembly, feature extraction) is unchanged.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--figure", action="store_true",
                    help="Write a diagnostic PNG next to the output.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    M, K = args.M, args.k_range
    inr_range = (args.inr_min, args.inr_max)

    if args.styles == "all":
        styles = list(rfi_gen.STYLES)
    else:
        styles = [s.strip() for s in args.styles.split(",") if s.strip()]
        bad = [s for s in styles if s not in rfi_gen.STYLES]
        if bad:
            sys.exit("unknown styles: {}".format(bad))

    # Stash resolved options for the balanced scheduler.
    args._styles = styles
    args._inr_range = inr_range

    # Decide the TB work pool as (granule_name_or_None) entries. In
    # balanced mode the scheduler stops on quota, so the pool is sized
    # generously and recycled if needed; in legacy mode it is the exact
    # list of TBs to build.
    if args.synthetic or not (args.qc_csv or args.split_csv):
        if not args.synthetic:
            print("No --qc-csv or --split-csv given; falling back to synthetic mode.")
        if args.balance:
            pool_size = args.max_tbs if args.max_tbs > 0 else 100000
            tb_jobs = [None] * pool_size
        else:
            tb_jobs = [None] * args.n_tbs
    else:
        # Select the appropriate canvas module depending on --l0.
        if args.l0:
            from rfi_gen import canvas_l0 as _canvas_mod
            print("L0 mode: reading raw L0B granules via canvas_l0.")
        else:
            from rfi_gen import canvas as _canvas_mod

        # Build the list of candidate filenames from whichever CSV was given.
        if args.split_csv:
            # scene_split.csv from split_scenes.py: columns filename, split, ...
            import csv as _csv
            with open(args.split_csv, newline="") as f:
                rows = list(_csv.DictReader(f))
            if args.split:
                rows = [r for r in rows if r.get("split") == args.split]
                print("Split '{}': {} scenes from {}".format(
                    args.split, len(rows), args.split_csv))
            else:
                print("No --split given; using all {} scenes from {}".format(
                    len(rows), args.split_csv))
            clean_files = [r["filename"] for r in rows]
        else:
            clean_files = _canvas_mod.read_clean_list(args.qc_csv)

        present = [f for f in clean_files
                   if os.path.exists(os.path.join(args.data_dir, f))]
        if not present:
            sys.exit("No granules found in {}. "
                     "Check --data-dir and --split-csv/--qc-csv."
                     .format(args.data_dir))
        print("Granules available: {}".format(len(present)))
        if args.balance:
            # Recycle granules round-robin into a generous pool; each TB
            # samples a random window so repeats still differ.
            pool_size = args.max_tbs if args.max_tbs > 0 else 100000
            tb_jobs = [present[i % len(present)] for i in range(pool_size)]
            rng.shuffle(tb_jobs)
        else:
            tb_jobs = []
            for f in present:
                tb_jobs.extend([f] * args.tbs_per_file)

    # Build threshold blocks.
    if args.balance:
        print("Balanced generation: target {} samples/band, {} bands."
              .format(args.target_per_band, len(_bands(M))))
        all_records, tb_id, bands, counts, target = build_balanced(
            tb_jobs, args, M, K, rng)
        print("Per-band fill (band -> count / target):")
        for i, (lo, hi) in enumerate(bands):
            tag = "clean" if i == 0 else "{}-{}".format(lo, hi)
            print("  {:>7}: {} / {}".format(tag, counts[i], target[i]))
    else:
        clean_frac = 0.25 if args.clean_frac is None else args.clean_frac
        all_records = []
        tb_id = 0
        for job in tb_jobs:
            scenario = pick_scenario(rng, clean_frac, args.intermittent_frac)
            if job is None:
                blocks = synthetic_strip(args.cpi_per_tb, M, K, rng)
            else:
                blocks = real_strip(os.path.join(args.data_dir, job),
                                    args.cpi_per_tb, M, K, rng,
                                    use_l0=args.l0)
                if blocks is None:
                    continue  # scene too small; skip
            recs = build_tb(blocks, scenario, styles, inr_range, M, K,
                            rng, tb_id)
            for r in recs:
                r["granule"] = job if job is not None else "synthetic"
            all_records.extend(recs)
            tb_id += 1

    if not all_records:
        sys.exit("No samples were built. Check inputs.")

    n = len(all_records)
    print("Built {} CPI samples across {} threshold blocks.".format(n, tb_id))

    # Assemble arrays.
    eigen_input = np.stack([r["eigen"] for r in all_records]).astype(np.float32)
    labels = np.array([r["label"] for r in all_records], dtype=np.int32)
    tb_ids = np.array([r["tb_id"] for r in all_records], dtype=np.int32)
    feat_names = ["F", "sigma_min", "sigma_max", "mu_min",
                  "trace_db", "log10_condition_number"]
    global_raw = np.array(
        [[r["F"], r["sigma_min"], r["sigma_max"], r["mu_min"],
          r["trace_db"], r["log_cond"]] for r in all_records],
        dtype=np.float32)

    style_list = sorted(set(r["style"] for r in all_records))
    style_to_id = {s: i for i, s in enumerate(style_list)}
    style_id = np.array([style_to_id[r["style"]] for r in all_records],
                        dtype=np.int32)
    inr = np.array([r["inr"] for r in all_records], dtype=np.float32)

    # Split by TB id to prevent leakage of shared global features.
    # When --split-csv is used the train/val/test separation is already
    # done at the scene level, so every sample here belongs to one split.
    # Force val_frac=0 in that case so the .npz is 100% split=0 and the
    # caller (train.py --train-data / --val-data / --test-data) handles
    # the three-way separation externally.
    uniq = np.unique(tb_ids)
    rng.shuffle(uniq)
    effective_val_frac = 0.0 if args.split_csv else args.val_frac
    n_val = max(0, int(round(effective_val_frac * len(uniq))))
    val_tbs = set(uniq[:n_val].tolist())
    is_val = np.array([1 if t in val_tbs else 0 for t in tb_ids],
                      dtype=np.int32)

    # Standardize global features. If --norm-stats is set, load pre-computed
    # stats (from the training set) for proper cross-validation. Otherwise
    # compute from the current dataset.
    train_mask = is_val == 0
    if args.norm_stats:
        # Load normalization parameters from a pre-computed set.
        d_norm = np.load(args.norm_stats, allow_pickle=True)
        mean = d_norm["norm_mean"]
        std = d_norm["norm_std"]
        print("Loaded normalization stats from: {}".format(args.norm_stats))
    else:
        # Compute from current training data.
        mean = global_raw[train_mask].mean(axis=0)
        std = global_raw[train_mask].std(axis=0)
        std[std < 1e-6] = 1.0
        print("Computed normalization stats from current training data.")
    
    global_input = ((global_raw - mean) / std).astype(np.float32)

    # Save.
    np.savez_compressed(
        args.out,
        eigen_input=eigen_input,
        global_input=global_input,
        global_input_raw=global_raw,
        labels=labels,
        tb_id=tb_ids,
        split=is_val,                 # 0 = train, 1 = val
        style_id=style_id,
        inr=inr,
        norm_mean=mean.astype(np.float32),
        norm_std=std.astype(np.float32),
        feature_names=np.array(feat_names),
        style_names=np.array(style_list),
        M=np.int32(M),
        n_knee_classes=np.int32(M + 1),
    )

    # Sidecar metadata + summary.
    n_train = int(train_mask.sum())
    n_val_s = int((~train_mask).sum())
    label_hist = {int(k): int(v) for k, v in
                  zip(*np.unique(labels, return_counts=True))}
    meta = {
        "n_samples": n, "n_tbs": int(tb_id),
        "n_train": n_train, "n_val": n_val_s,
        "M": M, "k_range": K, "cpi_per_tb": args.cpi_per_tb,
        "styles": styles, "inr_range": list(inr_range),
        "feature_names": feat_names,
        "norm_mean": mean.tolist(), "norm_std": std.tolist(),
        "label_histogram": label_hist,
        "mode": ("synthetic" if (args.synthetic or not args.qc_csv)
                 else ("real_l0" if args.l0 else "real_rslc")),
        "balanced": bool(args.balance),
        "target_per_band": (args.target_per_band if args.balance else None),
    }
    json_path = os.path.splitext(args.out)[0] + "_meta.json"
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved {} ({} train / {} val)".format(args.out, n_train, n_val_s))
    print("Wrote metadata {}".format(json_path))
    print("Label histogram (knee index -> count):")
    for kk in sorted(label_hist):
        print("  {:>2} : {}".format(kk, label_hist[kk]))

    if args.figure:
        write_figure(os.path.splitext(args.out)[0] + "_diagnostics.png",
                     eigen_input, global_raw, labels, feat_names)

    print("\nLoad and train with:")
    print("  d = np.load('{}', allow_pickle=True)".format(args.out))
    print("  tr = d['split'] == 0; va = d['split'] == 1")
    print("  from model import build_model")
    print("  m = build_model(cpi_size=int(d['M']), n_global_features=6)")
    print("  m.fit([d['eigen_input'][tr], d['global_input'][tr]], "
          "d['labels'][tr],")
    print("        validation_data=([d['eigen_input'][va], "
          "d['global_input'][va]], d['labels'][va]))")


def write_figure(path, eigen_input, global_raw, labels, feat_names):
    """Diagnostic figure: label histogram, F vs RFI, example profiles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 5))

    ax0 = fig.add_subplot(1, 3, 1)
    ax0.hist(labels, bins=np.arange(labels.max() + 2) - 0.5)
    ax0.set_title("knee label distribution")
    ax0.set_xlabel("knee index (0 = no RFI)")
    ax0.set_ylabel("samples")

    ax1 = fig.add_subplot(1, 3, 2)
    F = global_raw[:, feat_names.index("F")]
    clean = F[labels == 0]
    rfi = F[labels > 0]
    ax1.hist(clean, bins=30, alpha=0.6, label="clean (label 0)")
    ax1.hist(rfi, bins=30, alpha=0.6, label="RFI (label > 0)")
    ax1.set_title("F factor by RFI presence")
    ax1.set_xlabel("F = sigma_max / sigma_min")
    ax1.set_ylabel("samples")
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(1, 3, 3)
    M = eigen_input.shape[1]
    x = np.arange(1, M + 1)
    # one example each: clean, sharp (low knee), high knee
    def example(cond):
        idx = np.where(cond)[0]
        return idx[0] if idx.size else None
    for tag, cond in (("clean", labels == 0),
                      ("knee=1-2", (labels >= 1) & (labels <= 2)),
                      ("knee>=8", labels >= 8)):
        i = example(cond)
        if i is not None:
            ax2.plot(x, eigen_input[i, :, 0], marker=".", ms=3,
                     label="{} (label {})".format(tag, labels[i]))
    ax2.set_title("example eigenvalue profiles")
    ax2.set_xlabel("eigenvalue index")
    ax2.set_ylabel("eigenvalue (dB)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("Wrote diagnostics {}".format(path))


if __name__ == "__main__":
    main()