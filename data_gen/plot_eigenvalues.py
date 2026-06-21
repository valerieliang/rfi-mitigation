"""
plot_eigenvalues.py
-------------------
Generate a dataset with gen_stap_dataset and plot the SCM eigenvalue profiles
for all contaminated samples, grouped by RFI style.

Each subplot shows one contaminated sample:
  - Blue line  : eigenvalue profile of the contaminated composite (descending dB)
  - Grey line  : eigenvalue profile of the clean reference (signal + noise only)
  - Red dashed : vertical line at the true knee index (= RFI subspace rank)
  - Annotation : JNR in dB

Usage
-----
    python plot_eigenvalues.py                   # default 20 samples, saves PNG
    python plot_eigenvalues.py --n 40 --seed 7   # custom count and seed
    python plot_eigenvalues.py --show            # display interactively instead
"""

from __future__ import annotations

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend; overridden by --show
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Pipeline import -- all three modules must be on the path
# ---------------------------------------------------------------------------
from gen_stap_dataset import generate_dataset, STAPSample
from rfi_gen_stap import _RFI_STYLES


# ---------------------------------------------------------------------------
# Knee index lookup
# ---------------------------------------------------------------------------

# Maps RFI style name -> expected SCM rank (= knee index)
_STYLE_KNEE: dict[str, int] = {
    "cw_tone":  1,
    "wideband": 4,    # matches n_modes=4 set in _RFI_STYLES
}


def _knee_index(sample: STAPSample) -> int:
    """Return the true knee index for a contaminated sample."""
    if sample.rfi_style is None:
        raise ValueError("Sample is clean -- no knee index.")
    return _STYLE_KNEE.get(sample.rfi_style, 1)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_eigenvalue_profiles(
    samples: list[STAPSample],
    output_path: str | None = "eigenvalue_profiles.png",
    show: bool = False,
    ncols: int = 4,
) -> None:
    """
    Plot eigenvalue profiles for all contaminated samples in `samples`.

    Parameters
    ----------
    samples     : Full dataset (clean + contaminated); clean samples are skipped.
    output_path : File path to save the figure.  None = do not save.
    show        : If True, call plt.show() instead of (or in addition to) saving.
    ncols       : Number of subplot columns.
    """
    contaminated = [s for s in samples if s.label == "contaminated"]
    if not contaminated:
        print("No contaminated samples found -- nothing to plot.")
        return

    M      = contaminated[0].contaminated.shape[1]
    x_axis = np.arange(M)        # eigenvalue indices 0 .. M-1

    # Group by style for colour coding
    styles      = sorted({s.rfi_style for s in contaminated})
    style_color = {"cw_tone": "#2166ac", "wideband": "#d6604d"}
    style_label = {"cw_tone": "CW tone (rank-1)", "wideband": "Wideband (rank-4)"}

    n        = len(contaminated)
    nrows    = (n + ncols - 1) // ncols
    fig_w    = ncols * 3.2
    fig_h    = nrows * 2.8

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(
        f"SCM Eigenvalue Profiles -- Contaminated Samples  (M={M} pulses)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.55, wspace=0.35)

    for ax_idx, s in enumerate(contaminated):
        ax  = fig.add_subplot(gs[ax_idx // ncols, ax_idx % ncols])
        col = style_color.get(s.rfi_style, "#555555")

        evs_cont  = s.eigenvalues_db("contaminated")
        evs_clean = s.eigenvalues_db("clean")
        knee      = _knee_index(s)

        # Clean reference
        ax.plot(x_axis, evs_clean, color="#aaaaaa", linewidth=1.2,
                linestyle="--", label="clean ref", zorder=2)

        # Contaminated profile
        ax.plot(x_axis, evs_cont, color=col, linewidth=1.8,
                marker="o", markersize=3.5, label="contaminated", zorder=3)

        # Knee marker
        ax.axvline(x=knee - 0.5, color="#cc0000", linewidth=1.4,
                   linestyle=":", zorder=4, label=f"knee @ {knee}")

        # JNR annotation
        ax.text(
            0.97, 0.97,
            f"JNR {s.jnr_db:+d} dB",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=7.5, color=col,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"),
        )

        ax.set_title(
            f"#{s.index}  {style_label.get(s.rfi_style, s.rfi_style)}",
            fontsize=8, pad=4,
        )
        ax.set_xlabel("Eigenvalue index", fontsize=7.5)
        ax.set_ylabel("Power (dB)",       fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.set_xlim(-0.5, M - 0.5)
        ax.grid(True, linewidth=0.4, alpha=0.5)

    # Hide unused axes
    for ax_idx in range(n, nrows * ncols):
        fig.add_subplot(gs[ax_idx // ncols, ax_idx % ncols]).set_visible(False)

    # Shared legend at figure level
    handles = [
        plt.Line2D([0], [0], color="#aaaaaa", linestyle="--", linewidth=1.2,
                   label="Clean reference"),
        plt.Line2D([0], [0], color=style_color["cw_tone"], linewidth=1.8,
                   marker="o", markersize=4, label="CW tone"),
        plt.Line2D([0], [0], color=style_color["wideband"], linewidth=1.8,
                   marker="o", markersize=4, label="Wideband"),
        plt.Line2D([0], [0], color="#cc0000", linestyle=":", linewidth=1.4,
                   label="True knee"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.03))

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    if show:
        matplotlib.use("TkAgg")
        plt.show()

    plt.close(fig)


def plot_style_summary(
    samples: list[STAPSample],
    output_path: str | None = "eigenvalue_summary.png",
    show: bool = False,
) -> None:
    """
    One panel per RFI style showing all contaminated profiles overlaid,
    coloured by JNR.  Gives a quick view of how JNR variation shifts
    the dominant eigenvalue(s).

    Parameters
    ----------
    samples     : Full dataset list.
    output_path : Save path.  None = do not save.
    show        : Show interactively.
    """
    contaminated = [s for s in samples if s.label == "contaminated"]
    if not contaminated:
        print("No contaminated samples -- nothing to plot.")
        return

    M      = contaminated[0].contaminated.shape[1]
    x_axis = np.arange(M)
    styles = sorted({s.rfi_style for s in contaminated})

    fig, axes = plt.subplots(1, len(styles), figsize=(5.5 * len(styles), 4.5),
                             sharey=False)
    if len(styles) == 1:
        axes = [axes]

    fig.suptitle(
        f"Eigenvalue Profiles by RFI Style  (M={M}, coloured by JNR)",
        fontsize=12, fontweight="bold",
    )

    for ax, style in zip(axes, styles):
        group = [s for s in contaminated if s.rfi_style == style]
        jnrs  = np.array([s.jnr_db for s in group], dtype=float)
        jnr_min, jnr_max = jnrs.min(), jnrs.max()

        cmap = plt.cm.plasma
        norm = plt.Normalize(vmin=jnr_min, vmax=jnr_max)

        # Clean reference band (min/max envelope across group)
        clean_stack = np.stack([s.eigenvalues_db("clean") for s in group])
        ax.fill_between(x_axis,
                        clean_stack.min(axis=0), clean_stack.max(axis=0),
                        color="#cccccc", alpha=0.45, label="Clean envelope")

        for s in group:
            evs = s.eigenvalues_db("contaminated")
            ax.plot(x_axis, evs, color=cmap(norm(s.jnr_db)),
                    linewidth=1.3, alpha=0.8)

        # Knee marker
        knee = _STYLE_KNEE.get(style, 1)
        ax.axvline(x=knee - 0.5, color="#cc0000", linewidth=1.6,
                   linestyle=":", label=f"Knee @ {knee}")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("JNR (dB)", fontsize=9)

        style_titles = {"cw_tone": "CW Tone (rank-1)", "wideband": "Wideband (rank-4)"}
        ax.set_title(style_titles.get(style, style), fontsize=10, fontweight="bold")
        ax.set_xlabel("Eigenvalue index", fontsize=9)
        ax.set_ylabel("Power (dB)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_xlim(-0.5, M - 0.5)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(fontsize=8)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    if show:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot STAP eigenvalue profiles.")
    p.add_argument("--n",       type=int,   default=20,   help="Total samples to generate")
    p.add_argument("--K",       type=int,   default=128,  help="Range bins")
    p.add_argument("--M",       type=int,   default=16,   help="Pulses (SCM dimension)")
    p.add_argument("--noise",   type=float, default=3.0,  help="Noise power (dB)")
    p.add_argument("--signal",  type=float, default=9.0,  help="Signal power (dB)")
    p.add_argument("--jnr-min", type=int,   default=10,   help="JNR lower bound (dB)")
    p.add_argument("--jnr-max", type=int,   default=30,   help="JNR upper bound (dB)")
    p.add_argument("--frac",    type=float, default=0.7,  help="Contaminated fraction")
    p.add_argument("--seed",    type=int,   default=42,   help="Base RNG seed")
    p.add_argument("--ncols",   type=int,   default=4,    help="Subplot columns")
    p.add_argument("--show",    action="store_true",      help="Show interactively")
    p.add_argument("--out-dir", type=str,   default=".",  help="Output directory")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    import os
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating {args.n} samples  (K={args.K}, M={args.M}, seed={args.seed}) ...")
    samples = generate_dataset(
        n_samples             = args.n,
        K                     = args.K,
        M                     = args.M,
        noise_db              = args.noise,
        signal_db             = args.signal,
        jnr_min_db            = args.jnr_min,
        jnr_max_db            = args.jnr_max,
        contaminated_fraction = args.frac,
        base_seed             = args.seed,
    )

    n_cont  = sum(1 for s in samples if s.label == "contaminated")
    n_clean = len(samples) - n_cont
    print(f"  {n_cont} contaminated, {n_clean} clean")

    per_sample_path = os.path.join(out_dir, "eigenvalue_profiles.png")
    summary_path    = os.path.join(out_dir, "eigenvalue_summary.png")

    print("Plotting per-sample profiles ...")
    plot_eigenvalue_profiles(
        samples,
        output_path = per_sample_path,
        show        = args.show,
        ncols       = args.ncols,
    )

    print("Plotting style summary ...")
    plot_style_summary(
        samples,
        output_path = summary_path,
        show        = args.show,
    )

    print("Done.")