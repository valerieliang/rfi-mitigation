#!/usr/bin/env python
"""
plot_metric_overlaps.py

Draw box-plot or number-line visualizations of the clean vs RFI overlap in
condition_number_db.csv / effective_rank.csv / median_max_ratio.csv (the
tracking tables written by compare_global_metrics.py --run-name).

These CSVs hold SUMMARY statistics per run (mean, median, std, min, max,
p05, p95, iqr, count), not raw per-CPI values, so the "box plots" here are
reconstructed from those five numbers rather than drawn from raw data:
    whisker low  = p05
    box (q1)     = median - iqr/2
    box (q3)     = median + iqr/2
    box (median) = median
    whisker high = p95
    mean marker  = mean
    faint ends   = min / max (drawn as thin dotted extensions, not fliers,
                   since we have no real outlier list -- just the reported
                   extremes)

This is an approximation (assumes iqr is symmetric about the median), which
is fine for the purpose here: showing whether the clean-class range and the
RFI-class range for a given channel visually overlap.

Only rows whose run name contains an explicit "HH" or "HV" tag are plotted
by default, since a few legacy rows in these CSVs (e.g. "mountains czech
clean", "amazon train clean") are pre-polarization-split duplicates of the
HH-tagged rows and would otherwise double up the plot. Use --include-legacy
to include them anyway (plotted under a "pooled" polarization facet).

Usage
-----
    # Box-plot style, one PNG per metric, HH/HV side by side
    python plot_metric_overlaps.py --metrics-dir metrics --output-dir plots

    # Number-line style instead
    python plot_metric_overlaps.py --metrics-dir metrics --output-dir plots \\
        --style numberline

    # Only plot a subset of runs (substring match, case-insensitive)
    python plot_metric_overlaps.py --metrics-dir metrics --output-dir plots \\
        --filter "amazon" --filter "mountains czech"
"""

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METRIC_FILES = {
    "condition_number_db": "condition_number_db.csv",
    "effective_rank": "effective_rank.csv",
    "median_max_ratio": "median_max_ratio.csv",
}

METRIC_LABELS = {
    "condition_number_db": "Condition number (dB)",
    "effective_rank": "Effective rank",
    "median_max_ratio": "Median / max eigenvalue ratio",
}

# Colors are keyed on what the run name implies about its class, not the
# domain, so clean vs RFI overlap is the first thing the eye picks up.
COLOR_CLEAN = "#2b6cb0"        # blue
COLOR_RFI = "#c53030"          # red
COLOR_LOWJSR_RFI = "#dd6b20"   # orange
COLOR_UNKNOWN = "#718096"      # gray


def classify_run(run_name):
    """
    Parse a run name string into (domain, pol, class_) for grouping/coloring.

    domain  : "amazon", "mountains", or "other"
    pol     : "HH", "HV", or "pooled" (no explicit tag found)
    class_  : "clean", "low-jsr rfi", "rfi", or "unknown"
    """
    name = run_name.strip().lower()

    if "amazon" in name:
        domain = "amazon"
    elif "mountain" in name:
        domain = "mountains"
    else:
        domain = "other"

    pol_match = re.search(r"\b(hh|hv)\b", name)
    pol = pol_match.group(1).upper() if pol_match else "pooled"

    if "low-jsr" in name or "low jsr" in name:
        class_ = "low-jsr rfi"
    elif "rfi" in name:
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


def apply_filters(df, patterns, include_legacy):
    if not include_legacy:
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
    """One figure per metric, HH and HV side by side (or a single pooled
    panel if no HH/HV split is present in the filtered data)."""
    pols_present = [p for p in ("HH", "HV", "pooled") if (df["pol"] == p).any()]
    n_panels = len(pols_present)
    if n_panels == 0:
        return None

    fig_height = max(2.0, 0.55 * len(df) / max(n_panels, 1) + 1.5)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, fig_height), squeeze=False)
    axes = axes[0]

    for ax, pol in zip(axes, pols_present):
        sub = df[df["pol"] == pol].copy()
        # Sort by median so clean/RFI interleaving (i.e. overlap) is visible
        # top to bottom rather than grouped by insertion order.
        sub = sub.sort_values("median", ascending=True)

        stats = [make_bxp_stat(r) for _, r in sub.iterrows()]
        bxp_result = ax.bxp(
            stats,
            vert=False,
            showmeans=True,
            meanline=False,
            patch_artist=True,
            widths=0.6,
        )

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

        # Faint dotted extension out to the reported min/max, since bxp()
        # only knows about whislo/whishi (p05/p95) -- min/max are drawn
        # separately as a lighter-weight true range indicator.
        for i, (_, row) in enumerate(sub.iterrows()):
            y = i + 1
            color = class_color(row["class_"])
            ax.plot([row["min"], row["p05"]], [y, y], color=color, alpha=0.3,
                     linewidth=1.0, linestyle=":")
            ax.plot([row["p95"], row["max"]], [y, y], color=color, alpha=0.3,
                     linewidth=1.0, linestyle=":")

        ax.set_title(f"{pol}" if pol != "pooled" else "pooled (no pol tag)")
        ax.set_xlabel(METRIC_LABELS.get(metric_key, metric_key))
        ax.grid(axis="x", alpha=0.3)

    # Shared legend
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
    """Simpler number-line rendering: one horizontal row per run, a thick
    bar spanning p05-p95, a thin line spanning min-max, and a marker at the
    median. All runs for a metric share one axis (faceted by pol via
    vertical offset groups rather than separate subplots), which makes
    cross-pol comparison on the same run easier to see directly."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics-dir", default="metrics",
                        help="Directory containing condition_number_db.csv, "
                             "effective_rank.csv, median_max_ratio.csv (default: metrics)")
    parser.add_argument("--output-dir", default="plots",
                        help="Directory to write PNG files to (default: plots)")
    parser.add_argument("--style", choices=["box", "numberline", "both"], default="box",
                        help="Plot style (default: box)")
    parser.add_argument("--filter", action="append", default=None,
                        help="Only include runs whose name contains this substring "
                             "(case-insensitive). Can be passed multiple times; a run "
                             "matching ANY filter is included. Default: include all.")
    parser.add_argument("--include-legacy", action="store_true",
                        help="Also include pre-polarization-split rows that have no "
                             "explicit HH/HV tag in the run name (default: excluded, "
                             "since they duplicate the HH-tagged rows in these CSVs).")
    parser.add_argument("--metric", action="append", default=None,
                        choices=list(METRIC_FILES.keys()),
                        help="Restrict to one or more specific metrics "
                             "(default: all three)")
    args = parser.parse_args()

    metrics_to_plot = args.metric if args.metric else list(METRIC_FILES.keys())
    os.makedirs(args.output_dir, exist_ok=True)

    written = []
    for metric_key in metrics_to_plot:
        csv_path = os.path.join(args.metrics_dir, METRIC_FILES[metric_key])
        if not os.path.exists(csv_path):
            print(f"[warn] {csv_path} not found, skipping {metric_key}", file=sys.stderr)
            continue

        df = load_metric_csv(csv_path)
        df = apply_filters(df, args.filter, args.include_legacy)

        if df.empty:
            print(f"[warn] no rows left for {metric_key} after filtering, skipping", file=sys.stderr)
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

        if args.style in ("box", "both"):
            out = plot_box_style(df, metric_key, args.output_dir)
            if out:
                written.append(out)
                print(f"Wrote {out}")

        if args.style in ("numberline", "both"):
            out = plot_numberline_style(df, metric_key, args.output_dir)
            if out:
                written.append(out)
                print(f"Wrote {out}")

    if not written:
        print("No plots written -- check --metrics-dir / --filter.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
