"""
plot_eigenvalues.py
-------------------
Plot SCM eigenvalue profiles for contaminated STAP samples.

Two input modes:
  1. HDF5 file (--h5)   : load from a file produced by save_stap_dataset.py
                          and randomly sample N contaminated entries to plot.
  2. Generate on-the-fly: call generate_dataset with the usual knobs.

Each subplot shows one contaminated sample:
  - Coloured line : eigenvalue profile of the contaminated composite (dB, descending)
  - Grey dashed   : clean reference (signal + noise, same sample)
  - Red dotted    : vertical marker at the true knee index
  - Annotation    : JNR in dB, sample index inside the file

Two output figures per run:
  eigenvalue_profiles.png  -- one subplot per sample
  eigenvalue_summary.png   -- all profiles overlaid per style, coloured by JNR

Usage
-----
    # Plot 20 randomly sampled contaminated entries from an HDF5 file
    python plot_eigenvalues.py --h5 high_power_rfi_jnr.h5 --n-plot 20 --seed 0
    python plot_eigenvalues.py --h5 low_power_rfi_jnr.h5  --n-plot 20 --seed 0

    # Generate on-the-fly 
    python plot_eigenvalues.py --n 40 --jnr-min 10 --jnr-max 30 --seed 0
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gen_stap_dataset import generate_dataset, STAPSample


# ---------------------------------------------------------------------------
# Knee index lookup
# ---------------------------------------------------------------------------

_STYLE_KNEE: dict[str, int] = {
    "cw_tone":  1,
    "wideband": 4,
}

_STYLE_COLOR = {"cw_tone": "#2166ac", "wideband": "#d6604d"}
_STYLE_LABEL = {"cw_tone": "CW tone (rank-1)", "wideband": "Wideband (rank-4)"}


def _knee_index(rfi_style: str) -> int:
    return _STYLE_KNEE.get(rfi_style, 1)


def _eigenvalues_db(X: np.ndarray) -> np.ndarray:
    """Descending SCM eigenvalue profile in dB for a (K, M) matrix."""
    K   = X.shape[0]
    scm = (X.conj().T @ X) / K
    evs = np.sort(np.linalg.eigvalsh(scm))[::-1]
    return 10.0 * np.log10(np.maximum(evs, 1e-30))


# ---------------------------------------------------------------------------
# HDF5 loader
# ---------------------------------------------------------------------------

def load_samples_from_h5(
    path: str,
    n_plot: int,
    seed: int = 0,
) -> list[STAPSample]:
    """
    Load a random subset of contaminated samples from an HDF5 file.

    Parameters
    ----------
    path   : Path to .h5 file written by save_stap_dataset.py.
    n_plot : Number of contaminated samples to draw.
    seed   : RNG seed for the random selection (not for data generation).

    Returns
    -------
    List of STAPSample objects (label == "contaminated" for all).
    """
    with h5py.File(path, "r") as f:
        labels = f["labels/label"][:]                   # (N,) bytes
        cont_idx = np.where(labels == b"contaminated")[0]

        if len(cont_idx) == 0:
            raise ValueError(f"No contaminated samples found in {path}.")
        if n_plot > len(cont_idx):
            print(f"  Warning: only {len(cont_idx)} contaminated samples available, "
                  f"plotting all of them.")
            n_plot = len(cont_idx)

        rng      = np.random.default_rng(seed)
        chosen   = rng.choice(cont_idx, size=n_plot, replace=False)
        chosen   = np.sort(chosen)   # keep file order for readable indices

        styles_raw = f["labels/rfi_style"][:]
        jnr_raw    = f["labels/jnr_db"][:]

        samples: list[STAPSample] = []
        for file_idx in chosen:
            cont  = f["data/contaminated"][file_idx].astype(np.complex128)
            clean = f["data/clean"][file_idx].astype(np.complex128)
            sig   = f["data/signal"][file_idx].astype(np.complex128)
            noi   = f["data/noise"][file_idx].astype(np.complex128)
            rfi   = f["data/rfi"][file_idx].astype(np.complex128)

            style  = styles_raw[file_idx].decode()
            jnr_db = int(jnr_raw[file_idx])

            samples.append(STAPSample(
                index        = int(file_idx),
                label        = "contaminated",
                rfi_style    = style,
                jnr_db       = jnr_db,
                clean        = clean,
                contaminated = cont,
                signal       = sig,
                noise        = noi,
                rfi          = rfi,
            ))

    return samples


# ---------------------------------------------------------------------------
# Plot: one subplot per sample
# ---------------------------------------------------------------------------

def plot_eigenvalue_profiles(
    samples: list[STAPSample],
    title_tag: str = "",
    output_path: str | None = "eigenvalue_profiles.png",
    ncols: int = 4,
) -> None:
    contaminated = [s for s in samples if s.label == "contaminated"]
    if not contaminated:
        print("No contaminated samples -- nothing to plot.")
        return

    M      = contaminated[0].contaminated.shape[1]
    x_axis = np.arange(M)
    n      = len(contaminated)
    nrows  = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(ncols * 3.2, nrows * 2.8))
    tag = f"  [{title_tag}]" if title_tag else ""
    fig.suptitle(
        f"SCM Eigenvalue Profiles -- Contaminated Samples  (M={M}){tag}",
        fontsize=12, fontweight="bold", y=1.01,
    )
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.55, wspace=0.35)

    for ax_idx, s in enumerate(contaminated):
        ax  = fig.add_subplot(gs[ax_idx // ncols, ax_idx % ncols])
        col = _STYLE_COLOR.get(s.rfi_style, "#555555")

        evs_cont  = _eigenvalues_db(s.contaminated)
        evs_clean = _eigenvalues_db(s.clean)
        knee      = _knee_index(s.rfi_style)

        ax.plot(x_axis, evs_clean, color="#aaaaaa", linewidth=1.2,
                linestyle="--", zorder=2)
        ax.plot(x_axis, evs_cont, color=col, linewidth=1.8,
                marker="o", markersize=3.5, zorder=3)
        ax.axvline(x=knee - 0.5, color="#cc0000", linewidth=1.4,
                   linestyle=":", zorder=4)
        ax.text(0.97, 0.97, f"JNR {s.jnr_db:+d} dB",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color=col,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))
        ax.set_title(
            f"idx {s.index}  {_STYLE_LABEL.get(s.rfi_style, s.rfi_style)}",
            fontsize=7.5, pad=4,
        )
        ax.set_xlabel("Eigenvalue index", fontsize=7.5)
        ax.set_ylabel("Power (dB)",       fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.set_xlim(-0.5, M - 0.5)
        ax.grid(True, linewidth=0.4, alpha=0.5)

    for ax_idx in range(n, nrows * ncols):
        fig.add_subplot(gs[ax_idx // ncols, ax_idx % ncols]).set_visible(False)

    handles = [
        plt.Line2D([0], [0], color="#aaaaaa", linestyle="--", linewidth=1.2,
                   label="Clean reference"),
        plt.Line2D([0], [0], color=_STYLE_COLOR["cw_tone"], linewidth=1.8,
                   marker="o", markersize=4, label="CW tone"),
        plt.Line2D([0], [0], color=_STYLE_COLOR["wideband"], linewidth=1.8,
                   marker="o", markersize=4, label="Wideband"),
        plt.Line2D([0], [0], color="#cc0000", linestyle=":", linewidth=1.4,
                   label="True knee"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.03))

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot: all profiles overlaid per style, coloured by JNR
# ---------------------------------------------------------------------------

def plot_style_summary(
    samples: list[STAPSample],
    title_tag: str = "",
    output_path: str | None = "eigenvalue_summary.png",
) -> None:
    contaminated = [s for s in samples if s.label == "contaminated"]
    if not contaminated:
        print("No contaminated samples -- nothing to plot.")
        return

    M      = contaminated[0].contaminated.shape[1]
    x_axis = np.arange(M)
    styles = sorted({s.rfi_style for s in contaminated})

    fig, axes = plt.subplots(1, len(styles), figsize=(5.5 * len(styles), 4.5))
    if len(styles) == 1:
        axes = [axes]

    tag = f"  [{title_tag}]" if title_tag else ""
    fig.suptitle(
        f"Eigenvalue Profiles by RFI Style  (M={M}, coloured by JNR){tag}",
        fontsize=11, fontweight="bold",
    )

    for ax, style in zip(axes, styles):
        group = [s for s in contaminated if s.rfi_style == style]
        jnrs  = np.array([s.jnr_db for s in group], dtype=float)
        norm  = plt.Normalize(vmin=jnrs.min(), vmax=jnrs.max())
        cmap  = plt.cm.plasma

        clean_stack = np.stack([_eigenvalues_db(s.clean) for s in group])
        ax.fill_between(x_axis,
                        clean_stack.min(axis=0), clean_stack.max(axis=0),
                        color="#cccccc", alpha=0.45, label="Clean envelope")

        for s in group:
            ax.plot(x_axis, _eigenvalues_db(s.contaminated),
                    color=cmap(norm(s.jnr_db)), linewidth=1.3, alpha=0.85)

        knee = _STYLE_KNEE.get(style, 1)
        ax.axvline(x=knee - 0.5, color="#cc0000", linewidth=1.6,
                   linestyle=":", label=f"Knee @ {knee}")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, pad=0.02).set_label("JNR (dB)", fontsize=9)

        ax.set_title(_STYLE_LABEL.get(style, style), fontsize=10, fontweight="bold")
        ax.set_xlabel("Eigenvalue index", fontsize=9)
        ax.set_ylabel("Power (dB)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_xlim(-0.5, M - 0.5)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(fontsize=8)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot STAP eigenvalue profiles from HDF5 or generated on-the-fly.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # HDF5 mode
    p.add_argument("--h5",     type=str, default=None,
                   help="Path to HDF5 file from save_stap_dataset.py.  "
                        "When given, all --n/--jnr/--frac/--noise/--signal args are ignored.")
    p.add_argument("--n-plot", type=int, default=20,
                   help="Number of contaminated samples to randomly draw from the HDF5 file.")

    # Generate mode
    p.add_argument("--n",       type=int,   default=40,   help="[generate] Total samples")
    p.add_argument("--K",       type=int,   default=128,  help="[generate] Range bins")
    p.add_argument("--M",       type=int,   default=16,   help="[generate] Pulses")
    p.add_argument("--noise",   type=float, default=3.0,  help="[generate] Noise power (dB)")
    p.add_argument("--signal",  type=float, default=9.0,  help="[generate] Signal power (dB)")
    p.add_argument("--jnr-min", type=int,   default=10,   help="[generate] JNR lower bound (dB)")
    p.add_argument("--jnr-max", type=int,   default=30,   help="[generate] JNR upper bound (dB)")
    p.add_argument("--frac",    type=float, default=0.7,  help="[generate] Contaminated fraction")

    # Shared
    p.add_argument("--seed",    type=int,   default=0,    help="RNG seed (sample selection for HDF5 mode)")
    p.add_argument("--ncols",   type=int,   default=4,    help="Subplot columns")
    p.add_argument("--out-dir", type=str,   default=".",  help="Output directory for PNGs")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.h5:
        # ------------------------------------------------------------------
        # HDF5 mode: one run per file
        # ------------------------------------------------------------------
        h5_files = [f.strip() for f in args.h5.split(",") if f.strip()]

        for h5_path in h5_files:
            stem     = os.path.splitext(os.path.basename(h5_path))[0]
            title    = stem
            print(f"\n=== {h5_path} ===")
            print(f"  Loading {args.n_plot} contaminated samples (seed={args.seed}) ...")
            samples = load_samples_from_h5(h5_path, n_plot=args.n_plot, seed=args.seed)
            print(f"  Loaded {len(samples)} samples")

            jnrs = [s.jnr_db for s in samples]
            print(f"  JNR range in selection: {min(jnrs)} .. {max(jnrs)} dB")

            profiles_path = os.path.join(args.out_dir, f"{stem}_profiles.png")
            summary_path  = os.path.join(args.out_dir, f"{stem}_summary.png")

            print("  Plotting per-sample profiles ...")
            plot_eigenvalue_profiles(samples, title_tag=title,
                                     output_path=profiles_path, ncols=args.ncols)

            print("  Plotting style summary ...")
            plot_style_summary(samples, title_tag=title, output_path=summary_path)

    else:
        # ------------------------------------------------------------------
        # Generate mode
        # ------------------------------------------------------------------
        print(f"Generating {args.n} samples (seed={args.seed}) ...")
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
        n_cont = sum(1 for s in samples if s.label == "contaminated")
        print(f"  {n_cont} contaminated, {len(samples) - n_cont} clean")

        plot_eigenvalue_profiles(
            samples,
            output_path = os.path.join(args.out_dir, "eigenvalue_profiles.png"),
            ncols       = args.ncols,
        )
        plot_style_summary(
            samples,
            output_path = os.path.join(args.out_dir, "eigenvalue_summary.png"),
        )

    print("\nDone.")