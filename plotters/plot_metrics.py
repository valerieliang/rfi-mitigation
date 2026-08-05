#!/usr/bin/env python
"""
plotters/plot_metrics.py

Global cleanliness metrics -- extract AND plot in one place.

Two things this does, from one script:

  1. EXTRACT + PLOT (default): given one or more per-CPI feature files (the flat
     HDF5 layout written by generate_*_data.py / select_clean.py; see
     plotters/_common.py), compute the three global metrics
        - condition number (dB)          [from the eigenvalue profile]
        - effective rank                 [from the eigenvalue profile]
        - median / max SCM diagonal ratio [from the SCM diagonal]
     for each file, update the cross-run tracking CSVs under --metrics-dir (one
     CSV per metric, one row per run name), and THEN render the clean-vs-RFI
     overlap plots from those CSVs.

     These are the model's three global features: condition number and effective
     rank come from the eigenvalue profile, the median/max ratio from the SCM
     diagonal (matching features_from_eigenvalues in score_scene.py).

  2. PLOT ONLY (--plot-only): skip extraction and just (re)draw the overlap
     plots from the existing tracking CSVs in --metrics-dir. Use this when the
     CSVs are already up to date and you only want fresh figures.

Each input file already IS one frequency/polarization channel (freq/pol live in
its attrs), so no --freq/--pol selection is needed -- point --input at the file.
Only tiles whose SCM diagonal validity fraction meets --diag-valid-frac-thresh
and only the first --n-keep eigenvalues are used, matching the training pipeline.

If an input carries a `labels` dataset (knee: 0 = clean, k = injected RFI
bands), a second "<run> (rfi only)" row is written alongside the main row,
restricted to tiles labeled in [--rfi-label-min, --rfi-label-max], so the mixed
and RFI-only statistics for a run sit side by side in the tracking table.

Usage
-----
    # Extract metrics from two channels (record + plot in one call)
    python plotters/plot_metrics.py \\
        --input data/mountain_clean_caltone/mountain_clean_data_A_HH.h5 \\
        --run-name "mountains czech clean HH" \\
        --input data/mountain_clean_caltone/mountain_clean_data_A_HV.h5 \\
        --run-name "mountains czech clean HV" \\
        --metrics-dir metrics --output-dir metrics/plots

    # Re-draw the overlap plots from the existing CSVs only
    python plotters/plot_metrics.py --plot-only --metrics-dir metrics \\
        --output-dir metrics/plots --style both
"""

import argparse
import os
import re
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Allow running as either `python plotters/plot_metrics.py` or `-m`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (
    load_channel,
    describe_coverage,
    ensure_diagonal,
    compute_all_metrics,
    summarize,
    update_all_metrics_tables,
    N_KEEP_DEFAULT,
)


DEFAULT_DIAG_VALID_FRAC_THRESH = 0.8
DEFAULT_RFI_LABEL_MIN = 1
DEFAULT_RFI_LABEL_MAX = 6


# ---------------------------------------------------------------------------
# EXTRACTION
# ---------------------------------------------------------------------------

def channel_filtered_arrays(path, diag_valid_frac_thresh, n_keep, select,
                            rfi_label_min, rfi_label_max, clean_label):
    """
    Load one channel file, apply the diag-valid-frac filter, then keep the tile
    subset named by `select`:

      "all"   : every valid tile
      "rfi"   : tiles labeled in [rfi_label_min, rfi_label_max]   (needs labels)
      "clean" : tiles labeled == clean_label                       (needs labels)

    Returns (eig_kept, diag_lin, diag_valid_idx_or_None, labels_or_None, counts).
    diag_valid_idx is materialized to an all-True array when the file omits it,
    so callers (e.g. pooling across files) get a uniform shape.
    """
    ch = load_channel(path)
    describe_coverage(ch)
    # The median/max ratio is computed from the SCM diagonal, so make sure it is
    # available (recomputed from the L0B only if the file does not store it).
    ensure_diagonal(ch)

    dvf = (ch.diag_valid_frac if ch.diag_valid_frac is not None
           else np.ones(ch.n_tiles, dtype=np.float32))
    valid = dvf >= diag_valid_frac_thresh

    eig = ch.eigenvalues[valid][:, :n_keep]
    diag = ch.diagonal[valid]
    if ch.diag_valid_idx is not None:
        didx = ch.diag_valid_idx[valid]
    else:
        didx = np.ones(diag.shape, dtype=bool)
    labels = ch.labels[valid] if ch.labels is not None else None

    counts = {
        "n_total_in_file": int(ch.n_tiles),
        "n_passing_diag_valid_frac": int(valid.sum()),
    }

    if select == "all":
        return eig, diag, didx, labels, counts

    if labels is None:
        raise ValueError(f"--{select}-only needs a 'labels' dataset; {path} has none.")
    if select == "rfi":
        mask = (labels >= rfi_label_min) & (labels <= rfi_label_max)
        counts["n_rfi_labeled"] = int(mask.sum())
    elif select == "clean":
        mask = labels == clean_label
        counts["n_clean_labeled"] = int(mask.sum())
    else:
        raise ValueError(f"Unknown select '{select}'")
    counts["n_dropped"] = int((~mask).sum())
    if not mask.any():
        raise ValueError(f"--{select}-only found no matching tiles in {path}.")

    return eig[mask], diag[mask], didx[mask], labels[mask], counts


def summaries_from_arrays(eig, diag, didx, ratio_def):
    """Compute the three metric summaries from filtered per-tile arrays."""
    metrics = compute_all_metrics(eig, diag, didx, ratio_def)
    return [summarize(v, name) for name, v in metrics.items()]


def print_report(path, run_name, counts, summaries, report_name):
    print("=" * 70)
    print(f"Report: {report_name}")
    print(f"Input:  {path}")
    print(f"Run:    {run_name}")
    for k, v in counts.items():
        print(f"{k}: {v}")
    print("-" * 70)
    header = (f"{'metric':22s}{'count':>8s}{'mean':>10s}{'median':>10s}{'std':>10s}"
              f"{'min':>10s}{'max':>10s}{'p05':>10s}{'p95':>10s}{'iqr':>10s}")
    print(header)
    for s in summaries:
        if s.get("count", 0) == 0:
            print(f"{s['metric']:22s}{'0':>8s} (no valid CPI tiles)")
            continue
        print(f"{s['metric']:22s}{s['count']:>8d}{s['mean']:>10.3f}{s['median']:>10.3f}"
              f"{s['std']:>10.3f}{s['min']:>10.3f}{s['max']:>10.3f}"
              f"{s['p05']:>10.3f}{s['p95']:>10.3f}{s['iqr']:>10.3f}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# OVERLAP PLOTS (from the tracking CSVs)
# ---------------------------------------------------------------------------

METRIC_FILES = {
    "condition_number_db": "condition_number_db.csv",
    "effective_rank": "effective_rank.csv",
    "median_max_ratio": "median_max_ratio.csv",
}

METRIC_LABELS = {
    "condition_number_db": "Condition number (dB)",
    "effective_rank": "Effective rank",
    "median_max_ratio": "Median / max SCM diagonal ratio",
}

COLOR_CLEAN = "#2b6cb0"        # blue
COLOR_RFI = "#c53030"          # red
COLOR_LOWJSR_RFI = "#dd6b20"   # orange
COLOR_UNKNOWN = "#718096"      # gray


def classify_run(run_name):
    """
    Parse a run name into (domain, pol, class_) for grouping/coloring.

    pol is "HH" or "HV" (rows without an explicit HH/HV tag are dropped before
    plotting, so there is no "pooled" facet).
    """
    name = run_name.strip().lower()

    if "amazon" in name:
        domain = "amazon"
    elif "mountain" in name:
        domain = "mountains"
    else:
        domain = "other"

    pol_match = re.search(r"\b(hh|hv)\b", name)
    pol = pol_match.group(1).upper() if pol_match else None

    if "low-jsr" in name or "low jsr" in name:
        class_ = "low-jsr rfi"
    elif "rfi" in name or "contam" in name:
        class_ = "rfi"
    elif "clean" in name:
        class_ = "clean"
    else:
        class_ = "unknown"

    return domain, pol, class_


def class_color(class_):
    return {
        "clean": COLOR_CLEAN,
        "rfi": COLOR_RFI,
        "low-jsr rfi": COLOR_LOWJSR_RFI,
    }.get(class_, COLOR_UNKNOWN)


def load_metric_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["run"] = df["run"].astype(str).str.strip()
    for col in ("mean", "median", "std", "min", "max", "p05", "p95", "iqr", "count"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def apply_filters(df, patterns):
    """Keep only HH/HV-tagged rows, then any substring --filter patterns."""
    has_pol_tag = df["run"].str.contains(r"\b(?:hh|hv)\b", case=False, regex=True)
    df = df[has_pol_tag]

    if patterns:
        mask = pd.Series(False, index=df.index)
        for pat in patterns:
            mask |= df["run"].str.contains(re.escape(pat), case=False, regex=True)
        df = df[mask]

    return df.reset_index(drop=True)


def make_bxp_stat(row):
    """Build one matplotlib bxp() stats dict from a summary-stats row."""
    median = row["median"]
    iqr = row["iqr"] if pd.notna(row["iqr"]) else 0.0
    q1 = median - iqr / 2.0
    q3 = median + iqr / 2.0
    return {
        "label": row["run"],
        "med": median,
        "q1": q1,
        "q3": q3,
        "whislo": row["p05"],
        "whishi": row["p95"],
        "mean": row["mean"],
        "fliers": [],
    }


def plot_box_style(df, metric_key, output_dir):
    """One figure per metric, HH and HV side by side."""
    pols_present = [p for p in ("HH", "HV") if (df["pol"] == p).any()]
    n_panels = len(pols_present)
    if n_panels == 0:
        return None

    fig_height = max(2.0, 0.55 * len(df) / max(n_panels, 1) + 1.5)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, fig_height), squeeze=False)
    axes = axes[0]

    for ax, pol in zip(axes, pols_present):
        sub = df[df["pol"] == pol].copy()
        sub = sub.sort_values("median", ascending=True)

        stats = [make_bxp_stat(r) for _, r in sub.iterrows()]
        bxp_result = ax.bxp(stats, vert=False, showmeans=True, meanline=False,
                            patch_artist=True, widths=0.6)

        colors = [class_color(c) for c in sub["class_"]]
        for patch, color in zip(bxp_result["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_edgecolor(color)
        for whisker, color in zip(bxp_result["whiskers"], [c for c in colors for _ in range(2)]):
            whisker.set_color(color)
        for cap, color in zip(bxp_result["caps"], [c for c in colors for _ in range(2)]):
            cap.set_color(color)
        for median_line, color in zip(bxp_result["medians"], colors):
            median_line.set_color(color)
            median_line.set_linewidth(2.0)

        # Faint dotted extension out to the reported min/max.
        for i, (_, row) in enumerate(sub.iterrows()):
            y = i + 1
            color = class_color(row["class_"])
            ax.plot([row["min"], row["p05"]], [y, y], color=color, alpha=0.3,
                    linewidth=1.0, linestyle=":")
            ax.plot([row["p95"], row["max"]], [y, y], color=color, alpha=0.3,
                    linewidth=1.0, linestyle=":")

        ax.set_title(f"{pol}")
        ax.set_xlabel(METRIC_LABELS.get(metric_key, metric_key))
        ax.grid(axis="x", alpha=0.3)

    handles = [
        plt.Line2D([0], [0], color=COLOR_CLEAN, lw=6, alpha=0.5, label="clean"),
        plt.Line2D([0], [0], color=COLOR_RFI, lw=6, alpha=0.5, label="rfi (standard jsr)"),
        plt.Line2D([0], [0], color=COLOR_LOWJSR_RFI, lw=6, alpha=0.5, label="rfi (low jsr)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"{METRIC_LABELS.get(metric_key, metric_key)} -- box = p05/median/p95 "
        f"(dotted ends = min/max, diamond = mean)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))

    out_path = os.path.join(output_dir, f"{metric_key}_box.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_numberline_style(df, metric_key, output_dir):
    """One horizontal row per run: p05-p95 bar, min-max thin line, median | and mean diamond."""
    df = df.copy()
    df = df.sort_values(["pol", "median"], ascending=[True, True])

    fig_height = max(2.0, 0.45 * len(df) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_height))

    labels = []
    for i, (_, row) in enumerate(df.iterrows()):
        y = i
        color = class_color(row["class_"])
        ax.plot([row["min"], row["max"]], [y, y], color=color, alpha=0.25, linewidth=2)
        ax.plot([row["p05"], row["p95"]], [y, y], color=color, alpha=0.85, linewidth=6,
                solid_capstyle="butt")
        ax.plot(row["median"], y, marker="|", color="black", markersize=14, markeredgewidth=2)
        ax.plot(row["mean"], y, marker="D", color=color, markersize=5,
                markeredgecolor="black", markeredgewidth=0.5)
        labels.append(f"{row['run']}  [{row['pol']}]")

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(METRIC_LABELS.get(metric_key, metric_key))
    ax.grid(axis="x", alpha=0.3)
    ax.set_title(
        f"{METRIC_LABELS.get(metric_key, metric_key)} -- thick bar = p05-p95, "
        f"thin bar = min-max, | = median, diamond = mean",
        fontsize=10,
    )

    handles = [
        plt.Line2D([0], [0], color=COLOR_CLEAN, lw=6, alpha=0.85, label="clean"),
        plt.Line2D([0], [0], color=COLOR_RFI, lw=6, alpha=0.85, label="rfi (standard jsr)"),
        plt.Line2D([0], [0], color=COLOR_LOWJSR_RFI, lw=6, alpha=0.85, label="rfi (low jsr)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)

    fig.tight_layout()

    out_path = os.path.join(output_dir, f"{metric_key}_numberline.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_overlaps(metrics_dir, output_dir, style, metrics_to_plot, patterns):
    """Render the requested overlap plots from the tracking CSVs."""
    os.makedirs(output_dir, exist_ok=True)
    written = []
    for metric_key in metrics_to_plot:
        csv_path = os.path.join(metrics_dir, METRIC_FILES[metric_key])
        if not os.path.exists(csv_path):
            print(f"[warn] {csv_path} not found, skipping {metric_key}", file=sys.stderr)
            continue

        df = load_metric_csv(csv_path)
        df = apply_filters(df, patterns)
        if df.empty:
            print(f"[warn] no rows left for {metric_key} after filtering, skipping",
                  file=sys.stderr)
            continue

        domains, pols, classes = [], [], []
        for run in df["run"]:
            d, p, c = classify_run(run)
            domains.append(d)
            pols.append(p)
            classes.append(c)
        df["domain"] = domains
        df["pol"] = pols
        df["class_"] = classes
        df = df[df["pol"].notna()].reset_index(drop=True)
        if df.empty:
            print(f"[warn] no HH/HV-tagged rows for {metric_key}, skipping", file=sys.stderr)
            continue

        if style in ("box", "both"):
            out = plot_box_style(df, metric_key, output_dir)
            if out:
                written.append(out)
                print(f"Wrote {out}")
        if style in ("numberline", "both"):
            out = plot_numberline_style(df, metric_key, output_dir)
            if out:
                written.append(out)
                print(f"Wrote {out}")
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--input", action="append", default=None,
                        help="A per-CPI feature file to extract metrics from. Repeatable; "
                             "pairs positionally with --run-name.")
    parser.add_argument("--run-name", action="append", default=None,
                        help="Tracking-table row label for the matching --input (repeatable). "
                             "Defaults to the input filename if omitted.")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip extraction; just draw the overlap plots from the "
                             "existing tracking CSVs in --metrics-dir.")

    parser.add_argument("--metrics-dir", default="metrics",
                        help="Directory holding the per-metric tracking CSVs (default: metrics)")
    parser.add_argument("--output-dir", default="metrics/plots",
                        help="Directory to write overlap PNGs to (default: metrics/plots)")

    parser.add_argument("--n-keep", type=int, default=N_KEEP_DEFAULT,
                        help=f"Leading eigenvalues used per tile (default: {N_KEEP_DEFAULT})")
    parser.add_argument("--diag-valid-frac-thresh", type=float,
                        default=DEFAULT_DIAG_VALID_FRAC_THRESH,
                        help=f"Min SCM-diagonal valid fraction to keep a tile "
                             f"(default: {DEFAULT_DIAG_VALID_FRAC_THRESH})")
    parser.add_argument("--median-max-ratio-def",
                        choices=["median_over_max", "max_over_median"],
                        default="median_over_max",
                        help="Definition of the median/max ratio metric")
    parser.add_argument("--rfi-label-min", type=int, default=DEFAULT_RFI_LABEL_MIN,
                        help="Min knee label counted as RFI for the (rfi only) row")
    parser.add_argument("--rfi-label-max", type=int, default=DEFAULT_RFI_LABEL_MAX,
                        help="Max knee label counted as RFI for the (rfi only) row")
    parser.add_argument("--rfi-only", action="store_true",
                        help="Drop clean tiles and restrict the MAIN tracking row for every "
                             "--input to tiles labeled in [--rfi-label-min, --rfi-label-max]. "
                             "Requires a 'labels' dataset. Use for the amazon / contaminated "
                             "sets where only the RFI statistic is wanted.")
    parser.add_argument("--clean-only", action="store_true",
                        help="Inverse of --rfi-only: restrict every --input to clean tiles "
                             "(label == --clean-label). Requires a 'labels' dataset. Use to pull "
                             "the label-0 background out of contaminated files.")
    parser.add_argument("--clean-label", type=int, default=0,
                        help="Label value treated as clean for --clean-only (default: 0)")
    parser.add_argument("--pool", action="store_true",
                        help="Pool ALL --input files into a SINGLE tracking row (one --run-name), "
                             "concatenating their selected tiles before summarizing. Use e.g. to "
                             "build one 'amazon clean' row from the clean tiles of the train AND "
                             "test files together.")

    parser.add_argument("--style", choices=["box", "numberline", "both"], default="box",
                        help="Overlap plot style (default: box)")
    parser.add_argument("--filter", action="append", default=None,
                        help="Only plot runs whose name contains this substring "
                             "(case-insensitive; repeatable).")
    parser.add_argument("--metric", action="append", default=None,
                        choices=list(METRIC_FILES.keys()),
                        help="Restrict to specific metrics (default: all three)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Extract/record only; do not draw plots afterward.")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_to_plot = args.metric if args.metric else list(METRIC_FILES.keys())

    if not args.plot_only:
        inputs = args.input or []
        if not inputs:
            print("No --input given. Use --plot-only to draw from existing CSVs, or "
                  "pass one or more --input files.", file=sys.stderr)
            sys.exit(1)
        if args.rfi_only and args.clean_only:
            print("--rfi-only and --clean-only are mutually exclusive.", file=sys.stderr)
            sys.exit(1)

        select = "rfi" if args.rfi_only else ("clean" if args.clean_only else "all")
        report_name = {
            "rfi": f"RFI tiles only (labels {args.rfi_label_min}-{args.rfi_label_max})",
            "clean": f"Clean tiles only (label {args.clean_label})",
            "all": "All valid CPI tiles",
        }[select]

        run_names = args.run_name or []

        if args.pool:
            # All inputs collapse into one row; require exactly one run name.
            if len(run_names) != 1:
                print("--pool requires exactly one --run-name for the pooled row.",
                      file=sys.stderr)
                sys.exit(1)
            run_name = run_names[0]
            eigs, diags, didxs = [], [], []
            total = {"n_files": len(inputs), "n_tiles_pooled": 0}
            for path in inputs:
                eig, diag, didx, _labels, counts = channel_filtered_arrays(
                    path, args.diag_valid_frac_thresh, args.n_keep, select,
                    args.rfi_label_min, args.rfi_label_max, args.clean_label)
                eigs.append(eig)
                diags.append(diag)
                didxs.append(didx)
                total["n_tiles_pooled"] += eig.shape[0]
            eig = np.concatenate(eigs)
            diag = np.concatenate(diags)
            didx = np.concatenate(didxs)
            summaries = summaries_from_arrays(eig, diag, didx, args.median_max_ratio_def)
            print_report(f"{len(inputs)} files pooled", run_name, total, summaries,
                         f"{report_name} (pooled)")
            written = update_all_metrics_tables(args.metrics_dir, run_name, summaries)
            print(f"Updated tracking row '{run_name}' in: {', '.join(written)}")
            if args.no_plot:
                return
        else:
            if run_names and len(run_names) != len(inputs):
                print(f"Got {len(inputs)} --input but {len(run_names)} --run-name; they must "
                      f"pair 1:1 (or omit --run-name to default to filenames).", file=sys.stderr)
                sys.exit(1)

            for i, path in enumerate(inputs):
                run_name = (run_names[i] if run_names
                            else os.path.splitext(os.path.basename(path))[0])
                eig, diag, didx, labels, counts = channel_filtered_arrays(
                    path, args.diag_valid_frac_thresh, args.n_keep, select,
                    args.rfi_label_min, args.rfi_label_max, args.clean_label)
                summaries = summaries_from_arrays(eig, diag, didx, args.median_max_ratio_def)

                print_report(path, run_name, counts, summaries, report_name)
                written = update_all_metrics_tables(args.metrics_dir, run_name, summaries)
                print(f"Updated tracking row '{run_name}' in: {', '.join(written)}")

                # In the default "all" mode, also emit a secondary "(rfi only)" row
                # when the file carries labels, as before.
                if select == "all" and labels is not None:
                    rfi_mask = (labels >= args.rfi_label_min) & (labels <= args.rfi_label_max)
                    if rfi_mask.any():
                        rfi_run = f"{run_name} (rfi only)"
                        s = summaries_from_arrays(eig[rfi_mask], diag[rfi_mask],
                                                  didx[rfi_mask], args.median_max_ratio_def)
                        print_report(path, rfi_run,
                                     {"n_rfi_labeled": int(rfi_mask.sum())}, s,
                                     f"RFI tiles only (labels {args.rfi_label_min}-{args.rfi_label_max})")
                        written = update_all_metrics_tables(args.metrics_dir, rfi_run, s)
                        print(f"Updated tracking row '{rfi_run}' in: {', '.join(written)}")

            if args.no_plot:
                return

    written = plot_overlaps(args.metrics_dir, args.output_dir, args.style,
                            metrics_to_plot, args.filter)
    if not written:
        print("No plots written -- check --metrics-dir / --filter.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
