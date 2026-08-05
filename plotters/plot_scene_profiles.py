#!/usr/bin/env python
"""
plotters/plot_scene_profiles.py

Diagnostic eigenvalue-profile plots for a SCORED, UNLABELED scene, laid against
the LABELED training distribution it was scored with.

The question these plots answer: when a scene is under-detected, is the RFI
absent from the eigenvalue profile, or is it present but sitting outside the
JSR range the model was trained on?

The discriminating statistic is the STEP DEPTH at lambda_2 -- how far the
second eigenvalue falls below the first, after per-tile normalization by
lambda_max. A rank-1 interferer above a clutter floor produces a single deep
step at lambda_2 followed by the ordinary clutter ramp, so the step depth is a
direct proxy for interferer strength, and it is the quantity the knee=0 vs
knee=1 boundary effectively turns on.

Usage:
    python plotters/plot_scene_profiles.py \
        --predictions score/aus_sa/medhat/predictions_A_HV.h5 \
        --reference   data/australia_contam/australia_rfi_data_A_HH.h5 \
        --clean       data/australia_clean/australia_clean_data_A_HH.h5 \
        --out-dir     results/medhat_profiles
"""

import argparse
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_KEEP = 12          # leading eigenvalues used as model features
EPS = 1e-30
MIN_VALID_FRACTION = 0.8   # drop swath-edge / gap tiles


# ---------------------------------------------------------------------------
# LOADING
# ---------------------------------------------------------------------------

def normalized_db(eigvals_linear, n_keep=N_KEEP):
    """
    Per-tile eigenvalue profile in dB, normalized by lambda_max -- exactly the
    eigen-branch feature the model consumes.
    """
    ev = np.asarray(eigvals_linear, dtype=np.float64)[:, :n_keep]
    ev = np.maximum(ev, EPS)
    return 10.0 * np.log10(ev / np.maximum(ev[:, :1], EPS))


def load_scene(path):
    """Scored scene: predictions + eigenvalues, restricted to valid tiles."""
    with h5py.File(path, "r") as f:
        ev = f["eigenvalues"][:]
        knee = f["knee"][:]
        vf = f["valid_fraction"][:]
        tile_range = f["tile_range"][:]
        attrs = dict(f.attrs)
    ok = vf > MIN_VALID_FRACTION
    return dict(db=normalized_db(ev[ok]), knee=knee[ok],
                tile_range=tile_range[ok], attrs=attrs, n_total=len(ok),
                n_valid=int(ok.sum()))


def load_labeled(path):
    """Labeled training file: eigenvalues + knee labels."""
    with h5py.File(path, "r") as f:
        ev = f["eigenvalues"][:]
        labels = f["labels"][:] if "labels" in f else np.zeros(len(ev), np.int8)
        attrs = dict(f.attrs)
    return dict(db=normalized_db(ev), labels=labels, attrs=attrs)


def step_depth(db):
    """Depth of the lambda_2 step in dB (positive = deeper drop below lambda_1)."""
    return -db[:, 1]


# ---------------------------------------------------------------------------
# FIGURES
# ---------------------------------------------------------------------------

def fig_profiles(scene, ref, clean, scene_name, out_dir):
    """Median profiles, percentile envelope, per-predicted-class, and slopes."""
    idx = np.arange(1, N_KEEP + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) scene against the labeled training classes -------------------------
    ax = axes[0, 0]
    if clean is not None:
        ax.plot(idx, np.median(clean["db"], axis=0), "k--", lw=2,
                label="training clean scene")
    for k in range(0, 4):
        sel = ref["labels"] == k
        if sel.sum() < 50:
            continue
        ax.plot(idx, np.median(ref["db"][sel], axis=0), lw=1.6, alpha=0.85,
                label=f"training knee={k}  (n={sel.sum()})")
    ax.plot(idx, np.median(scene["db"], axis=0), "r-", lw=3,
            label=f"{scene_name} all valid  (n={scene['n_valid']})")
    ax.plot(idx, np.median(scene["db"][scene["knee"] == 0], axis=0), "r:", lw=2.5,
            label=f"{scene_name} predicted knee=0")
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("dB below $\\lambda_1$")
    ax.set_title("(a) Median profile vs the training distribution\n"
                 "compare the scene against the trained knee=0 and knee=1 curves")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.invert_yaxis()

    # (b) scene spread -------------------------------------------------------
    ax = axes[0, 1]
    pcts = [5, 25, 50, 75, 95]
    env = np.percentile(scene["db"], pcts, axis=0)
    ax.fill_between(idx, env[0], env[4], alpha=0.18, color="C3", label="p5-p95")
    ax.fill_between(idx, env[1], env[3], alpha=0.32, color="C3", label="p25-p75")
    ax.plot(idx, env[2], "r-", lw=2.5, label="median")
    if clean is not None:
        ax.plot(idx, np.median(clean["db"], axis=0), "k--", lw=2, label="training clean")
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("dB below $\\lambda_1$")
    ax.set_title(f"(b) {scene_name} profile spread\n"
                 "scene percentiles against the clean reference")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.invert_yaxis()

    # (c) split by predicted knee -------------------------------------------
    ax = axes[1, 0]
    if clean is not None:
        ax.plot(idx, np.median(clean["db"], axis=0), "k--", lw=2, label="training clean")
    for k in sorted(np.unique(scene["knee"])):
        sel = scene["knee"] == k
        if sel.sum() < 100:
            continue
        ax.plot(idx, np.median(scene["db"][sel], axis=0), lw=1.8,
                label=f"predicted knee={k}  (n={sel.sum()}, {100*sel.mean():.1f}%)")
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("dB below $\\lambda_1$")
    ax.set_title(f"(c) {scene_name} by predicted knee\n"
                 "profile of each predicted class within the scene")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.invert_yaxis()

    # (d) increments ---------------------------------------------------------
    ax = axes[1, 1]
    if clean is not None:
        ax.plot(idx[:-1], np.diff(np.median(clean["db"], axis=0)), "k--", lw=2,
                label="training clean")
    for k in [0, 1, 2]:
        sel = ref["labels"] == k
        if sel.sum() < 50:
            continue
        ax.plot(idx[:-1], np.diff(np.median(ref["db"][sel], axis=0)), lw=1.6,
                alpha=0.85, label=f"training knee={k}")
    ax.plot(idx[:-1], np.diff(np.median(scene["db"], axis=0)), "r-", lw=3,
            label=f"{scene_name} all valid")
    ax.set_xlabel("eigenvalue index $i$  (increment $\\lambda_{i+1}-\\lambda_i$)")
    ax.set_ylabel("dB")
    ax.set_title("(d) Profile increments\n"
                 "a deep first step then a flat tail indicates a dominant rank-1 term")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"{scene_name}: eigenvalue profiles against the training distribution",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(out_dir, "profiles.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_step_distribution(scene, ref, clean, scene_name, out_dir):
    """Where the scene's step depths fall relative to the trained classes."""
    s_scene = step_depth(scene["db"])
    s0 = step_depth(ref["db"][ref["labels"] == 0])
    s1 = step_depth(ref["db"][ref["labels"] == 1])

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    ax = axes[0]
    bins = np.linspace(0, 14, 160)
    ax.hist(s0, bins=bins, density=True, alpha=0.55, color="C0",
            label=f"training knee=0 (n={len(s0)})")
    ax.hist(s1, bins=bins, density=True, alpha=0.55, color="C1",
            label=f"training knee=1 (n={len(s1)})")
    ax.hist(s_scene, bins=bins, density=True, histtype="step", lw=2.5, color="C3",
            label=f"{scene_name} all valid (n={len(s_scene)})")

    # The trained classes are separable only when knee=0's upper tail sits below
    # knee=1's lower tail. When they invert, the band is a class OVERLAP region
    # in which no threshold on this statistic can separate RFI from clutter.
    p95_0, p1_1 = np.percentile(s0, 95), np.percentile(s1, 1)
    lo, hi = min(p95_0, p1_1), max(p95_0, p1_1)
    separable = p95_0 <= p1_1
    ax.axvspan(lo, hi, color="0.5", alpha=0.25)
    ax.axvline(lo, color="0.3", ls="--", lw=1)
    ax.axvline(hi, color="0.3", ls="--", lw=1)
    ax.text((lo + hi) / 2, ax.get_ylim()[1] * 0.92,
            ("decision gap" if separable else "CLASS OVERLAP") + f"\n{lo:.2f}-{hi:.2f} dB",
            ha="center", fontsize=8)

    below = float((s_scene < p1_1).mean())
    ax.set_xlabel("step depth at $\\lambda_2$  (dB below $\\lambda_1$)")
    ax.set_ylabel("density")
    ax.set_title(f"Step depth: {scene_name} vs the trained classes\n"
                 + (f"classes separable above {p95_0:.2f} dB; "
                    if separable else
                    f"trained classes OVERLAP over {lo:.2f}-{hi:.2f} dB; ")
                 + f"{100*below:.1f}% of scene tiles below the trained RFI p1")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 14)

    # cumulative view --------------------------------------------------------
    ax = axes[1]
    for arr, lbl, c in [(s0, "training knee=0", "C0"),
                        (s1, "training knee=1", "C1"),
                        (s_scene, f"{scene_name} all valid", "C3")]:
        xs = np.sort(arr)
        ax.plot(xs, np.linspace(0, 1, len(xs)), lw=2, color=c, label=lbl)
    if clean is not None:
        xs = np.sort(step_depth(clean["db"]))
        ax.plot(xs, np.linspace(0, 1, len(xs)), lw=2, color="k", ls="--",
                label="training clean scene")
    ax.axvspan(lo, hi, color="0.5", alpha=0.25)
    ax.set_xlabel("step depth at $\\lambda_2$ (dB)")
    ax.set_ylabel("cumulative fraction")
    ax.set_title("Cumulative distribution\n"
                 "how much of the scene falls inside each trained class range")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 14)

    fig.tight_layout()
    path = os.path.join(out_dir, "step_distribution.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_step_vs_range(scene, ref, scene_name, out_dir):
    """Spatial structure: step depth and detection rate against range block."""
    tr = scene["tile_range"]
    s = step_depth(scene["db"])
    urt = np.unique(tr)
    med = np.array([np.median(s[tr == r]) for r in urt])
    p90 = np.array([np.percentile(s[tr == r], 90) for r in urt])
    rate = np.array([(scene["knee"][tr == r] > 0).mean() for r in urt])
    boundary = np.percentile(step_depth(ref["db"][ref["labels"] == 1]), 1)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    x = np.arange(len(urt))
    ax.plot(x, med, lw=1.6, color="C3", label="median step depth")
    ax.plot(x, p90, lw=1.0, color="C3", alpha=0.45, label="p90 step depth")
    ax.axhline(boundary, color="0.3", ls="--", lw=1.5,
               label=f"trained knee=1 floor ({boundary:.2f} dB)")
    ax.set_xlabel("range block")
    ax.set_ylabel("step depth at $\\lambda_2$ (dB)")
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(x, 100 * rate, lw=1.2, color="C0", alpha=0.65, label="detection rate")
    ax2.set_ylabel("tiles with knee > 0 (%)", color="C0")
    ax2.tick_params(axis="y", labelcolor="C0")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax.set_title(f"{scene_name}: step depth and detection rate across the swath\n"
                 "coherent range-block structure indicates real emitters rather "
                 "than tile-level noise")
    fig.tight_layout()
    path = os.path.join(out_dir, "step_vs_range.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", required=True, help="scored scene predictions_*.h5")
    p.add_argument("--reference", required=True,
                   help="labeled training file with synthetic RFI (has 'labels')")
    p.add_argument("--clean", default=None, help="labeled clean training file (optional)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--name", default=None, help="scene name for titles")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    scene = load_scene(args.predictions)
    ref = load_labeled(args.reference)
    clean = load_labeled(args.clean) if args.clean else None
    name = args.name or os.path.basename(os.path.dirname(args.predictions))

    s = step_depth(scene["db"])
    s0 = step_depth(ref["db"][ref["labels"] == 0])
    s1 = step_depth(ref["db"][ref["labels"] == 1])
    jsr = (ref["attrs"].get("jsr_min_db"), ref["attrs"].get("jsr_max_db"))

    print(f"scene      : {name}  ({scene['n_valid']}/{scene['n_total']} valid tiles)")
    print(f"  knee>0   : {100*(scene['knee']>0).mean():.1f}%")
    print(f"  step@l2  : p5={np.percentile(s,5):.2f}  median={np.median(s):.2f}  "
          f"p95={np.percentile(s,95):.2f} dB")
    print(f"reference  : {os.path.basename(args.reference)}  JSR range {jsr} dB")
    print(f"  knee=0   : p95={np.percentile(s0,95):.2f} dB")
    print(f"  knee=1   : p1={np.percentile(s1,1):.2f}  p5={np.percentile(s1,5):.2f}  "
          f"median={np.median(s1):.2f} dB")
    print(f"  scene tiles below trained knee=1 p1: "
          f"{100*(s<np.percentile(s1,1)).mean():.1f}%")

    for path in [fig_profiles(scene, ref, clean, name, args.out_dir),
                 fig_step_distribution(scene, ref, clean, name, args.out_dir),
                 fig_step_vs_range(scene, ref, name, args.out_dir)]:
        print("wrote", path)


if __name__ == "__main__":
    main()
