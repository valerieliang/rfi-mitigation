#!/usr/bin/env python
"""
plotters/plot_profiles.py

Eigenvalue "mountain" + SCM-diagonal structure plots for the project's flat
per-CPI HDF5 files (see plotters/_common.py). Generalizes the old
process_mountains/ scripts: point it at any channel file (clean or contaminated,
amazon or mountains) and it reports the pulse/range coverage read from the file
and renders the diagnostic figures.

For each input channel it produces:

  * average eigenvalue shape   (mean +/- 1 std band, min/max envelope, median)
  * all eigenvalue profiles overlaid
  * SCM diagonal per CPI (all overlaid)
  * per-CPI grid            (eigenvalue profile + SCM-diagonal scatter, sampled)

When two polarizations of the same frequency are supplied together, it also
writes a combined HH+HV per-CPI grid.

When the file carries labels (knee: 0 = clean, k = injected RFI bands), it adds
the label-aware views rescued from the old data generator:

  * eigenvalue overlay colored by knee (with the knee drop-off marked)
  * per-block grid, one panel per randomly selected block, annotated with knee /
    JSR / baseline power

With --scm-heatmap it also draws a grid of SCM magnitude heatmaps for the same
selected blocks (bright rows/cols = injected bands). That needs the complex SCM,
which is not a stored per-tile field, so it is sourced from a `cpi` dataset when
present (files written with --save-cpi) or recomputed from the source L0B.

DATA SOURCE: everything is read from the H5. The only field ever recomputed from
the source L0B granule is the SCM `diagonal`, and only when the file does not
store it -- in which case the caltone is removed first (matching how the stored,
caltone-removed features were made). Use --l0b to profile a raw granule directly
when no preprocessed file exists yet.

Usage
-----
    # Profile one or more channel files (combined A HH+HV grid when both given)
    python plotters/plot_profiles.py \\
        data/mountain_clean_caltone/mountain_clean_data_A_HH.h5 \\
        data/mountain_clean_caltone/mountain_clean_data_A_HV.h5 \\
        --output-dir results/profiles

    # Add SCM magnitude heatmaps for the sampled blocks
    python plotters/plot_profiles.py rfi_data_A_HH.h5 --scm-heatmap

    # Profile a raw L0B granule directly (caltone removed on the fly)
    python plotters/plot_profiles.py --l0b scene.h5 --freq A --pol HH \\
        --pulse-start 435777 --pulse-end 441000 --output-dir results/scene
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (
    load_channel,
    channel_from_l0b,
    describe_coverage,
    ensure_diagonal,
    fetch_cpi_tiles,
    eigvals_to_db,
    EPS,
)


N_SHOW = 16   # show all 16 values in the plots

# Label-aware eigenvalue overlay y-limits (unnormalized dB, bottom pinned at 0).
EV_YLIM_BOTTOM_DB = 0.0
EV_YLIM_TOP_MARGIN_DB = 2.0
EV_YLIM_MIN_TOP_DB = 10.0


# ---------------------------------------------------------------------------
# BLOCK SAMPLER (for the label overlay / per-block grid / SCM heatmap)
# ---------------------------------------------------------------------------

def select_plot_blocks(plot_seed, tag, n_tiles, n_blocks):
    """Pick up to n_blocks distinct tile indices deterministically from a seed."""
    ss = np.random.SeedSequence([int(plot_seed), abs(hash(tag)) % (2 ** 31)])
    rng = np.random.default_rng(ss)
    n_pick = min(n_blocks, n_tiles)
    return sorted(int(i) for i in rng.choice(n_tiles, size=n_pick, replace=False))


# ---------------------------------------------------------------------------
# BASIC PROFILE PLOTS (eigenvalues + SCM diagonal)
# ---------------------------------------------------------------------------

def plot_average_shape(eig_db, out_path, freq, pol, n_show):
    """Mean eigenvalue profile with +/-1 std band, min/max envelope, and median."""
    eig = eig_db[:, :n_show]
    x = np.arange(n_show)
    mean = eig.mean(axis=0)
    std = eig.std(axis=0)
    med = np.median(eig, axis=0)
    lo, hi = eig.min(axis=0), eig.max(axis=0)

    y_max = np.ceil(np.max(eig_db[:, :12]))
    y_min = 0

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill_between(x, lo, hi, color="C0", alpha=0.12, label="min / max envelope")
    ax.fill_between(x, mean - std, mean + std, color="C0", alpha=0.30, label="mean +/- 1 std")
    ax.plot(x, mean, color="C0", lw=2.2, label="mean")
    ax.plot(x, med, color="C3", lw=1.4, ls="--", label="median")
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue (dB, power)")
    ax.set_title("Average eigenvalue shape: freq {} pol {} ({} CPIs, all {})"
                 .format(freq, pol, eig.shape[0], n_show))
    ax.set_xticks(x)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_overlaid(eig_db, out_path, freq, pol, n_show):
    """All CPI eigenvalue profiles overlaid on a single axes."""
    eig = eig_db[:, :n_show]
    x = np.arange(n_show)
    n = eig.shape[0]
    alpha = float(np.clip(30.0 / max(n, 1), 0.03, 0.5))

    y_max = np.ceil(np.max(eig_db[:, :12]))
    y_min = 0

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(n):
        ax.plot(x, eig[i], color="C0", alpha=alpha, lw=0.8)
    ax.plot(x, eig.mean(axis=0), color="k", lw=1.6, label="mean")
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue (dB, power)")
    ax.set_title("All eigenvalue profiles overlaid: freq {} pol {} ({} CPIs, all {})"
                 .format(freq, pol, n, n_show))
    ax.set_xticks(x)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_scm_diag(diag_db, out_path, freq, pol, n_show):
    """SCM diagonal values per CPI, all n_show positions overlaid."""
    d = diag_db[:, :n_show]
    x = np.arange(n_show)
    n = d.shape[0]
    alpha = float(np.clip(30.0 / max(n, 1), 0.03, 0.5))

    d_sorted_12 = np.sort(d, axis=1)[:, ::-1][:, :12]
    y_max = np.ceil(np.max(d_sorted_12))
    y_min = 0

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(n):
        ax.plot(x, d[i], color="C3", alpha=alpha, lw=0.8)
    ax.plot(x, d.mean(axis=0), color="k", lw=1.6, label="mean")
    ax.set_xlabel("Pulse index (SCM diagonal position)")
    ax.set_ylabel("SCM diagonal power (dB)")
    ax.set_title("SCM diagonal per CPI: freq {} pol {} ({} CPIs, all {})"
                 .format(freq, pol, n, n_show))
    ax.set_xticks(x)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_grid(eig_db, diag_db, pulse_idx, range_idx, out_path, freq, pol, n_show, max_grid):
    """Per-CPI grid: eigenvalue profile + SCM-diagonal scatter, evenly sampled."""
    eig = eig_db[:, :n_show]
    diag = diag_db[:, :n_show]
    x = np.arange(n_show)
    N = eig.shape[0]

    sel = (np.arange(N) if N <= max_grid
           else np.unique(np.linspace(0, N - 1, max_grid).round().astype(int)))
    n_total = len(sel)

    max_per_image = 5
    n_images = int(np.ceil(n_total / max_per_image))
    out_base, out_ext = os.path.splitext(out_path)
    output_files = []

    for img_idx in range(n_images):
        sel_chunk = sel[img_idx * max_per_image:(img_idx + 1) * max_per_image]
        n_in_chunk = len(sel_chunk)

        eig_y_max = np.ceil(np.max(eig_db[sel_chunk, :12]))
        eig_y_min = 0
        diag_chunk = diag_db[sel_chunk, :]
        diag_y_min = max(0, np.floor(np.min(diag_chunk)))
        diag_y_max = np.ceil(np.max(diag_chunk))

        fig, axes = plt.subplots(n_in_chunk, 2, figsize=(9.6, 1.9 * n_in_chunk),
                                 squeeze=False, sharex=True, sharey=False)
        for k, cpi_i in enumerate(sel_chunk):
            diag_vals = diag[cpi_i]
            diag_mean = np.mean(diag_vals)
            diag_std = np.std(diag_vals)
            diag_iqr = np.percentile(diag_vals, 75) - np.percentile(diag_vals, 25)

            ax_eig = axes[k][0]
            ax_eig.plot(x, eig[cpi_i], color="C0", lw=1.1)
            ax_eig.set_ylim(eig_y_min, eig_y_max)
            ax_eig.set_title("CPI {} Eigenvalues (p{}, r{})".format(
                cpi_i, pulse_idx[cpi_i], range_idx[cpi_i]), fontsize=7)
            ax_eig.tick_params(labelsize=6)
            ax_eig.grid(True, alpha=0.3)

            ax_diag = axes[k][1]
            ax_diag.scatter(x, diag[cpi_i], color="C3", s=15, alpha=0.7)
            ax_diag.set_ylim(diag_y_min, diag_y_max)
            ax_diag.set_title("CPI {} SCM Diag (p{}, r{}) | mean={:.1f} iqr={:.1f} std={:.1f}".format(
                cpi_i, pulse_idx[cpi_i], range_idx[cpi_i], diag_mean, diag_iqr, diag_std),
                fontsize=7)
            ax_diag.tick_params(labelsize=6)
            ax_diag.grid(True, alpha=0.3)

        subtitle = "" if N <= max_grid else " (showing {}/{}, evenly sampled)".format(n_total, N)
        part_info = " - Part {}/{}".format(img_idx + 1, n_images) if n_images > 1 else ""
        fig.suptitle("Eigenvalue & SCM diagonal grid: freq {} pol {}{}{}  [all {}]"
                     .format(freq, pol, subtitle, part_info, n_show), fontsize=11)
        fig.text(0.5, 0.01, "Index", ha="center", fontsize=9)
        fig.text(0.01, 0.5, "Power (dB)", va="center", rotation="vertical", fontsize=9)
        fig.tight_layout(rect=[0.02, 0.02, 1, 0.97])

        out_file = ("{}_part{:02d}{}".format(out_base, img_idx + 1, out_ext)
                    if n_images > 1 else out_path)
        fig.savefig(out_file, dpi=130)
        plt.close(fig)
        output_files.append(out_file)
    return output_files


def plot_grid_combined(eig_hh, eig_hv, diag_hh, diag_hv, pulse_idx, range_idx,
                       out_path, freq, n_show, max_grid,
                       eig_y_min, eig_y_max, diag_y_min, diag_y_max):
    """Combined per-CPI grid: HH eig | HH diag | HV eig | HV diag per row."""
    eig_hh_s, eig_hv_s = eig_hh[:, :n_show], eig_hv[:, :n_show]
    diag_hh_s, diag_hv_s = diag_hh[:, :n_show], diag_hv[:, :n_show]
    x = np.arange(n_show)
    N = eig_hh_s.shape[0]

    sel = (np.arange(N) if N <= max_grid
           else np.unique(np.linspace(0, N - 1, max_grid).round().astype(int)))
    n_total = len(sel)

    max_per_image = 5
    n_images = int(np.ceil(n_total / max_per_image))
    out_base, out_ext = os.path.splitext(out_path)
    output_files = []

    for img_idx in range(n_images):
        sel_chunk = sel[img_idx * max_per_image:(img_idx + 1) * max_per_image]
        n_in_chunk = len(sel_chunk)

        fig, axes = plt.subplots(n_in_chunk, 4, figsize=(9.6, 1.9 * n_in_chunk),
                                 squeeze=False, sharex=True)
        for k, cpi_i in enumerate(sel_chunk):
            for col, (data, ylim, kind, color) in enumerate((
                (eig_hh_s, (eig_y_min, eig_y_max), "HH Eig", "C0"),
                (diag_hh_s, (diag_y_min, diag_y_max), "HH Diag", "C3"),
                (eig_hv_s, (eig_y_min, eig_y_max), "HV Eig", "C0"),
                (diag_hv_s, (diag_y_min, diag_y_max), "HV Diag", "C3"),
            )):
                ax = axes[k][col]
                if "Diag" in kind:
                    ax.scatter(x, data[cpi_i], color=color, s=15, alpha=0.7)
                else:
                    ax.plot(x, data[cpi_i], color=color, lw=1.1)
                ax.set_ylim(*ylim)
                ax.set_title("CPI {} {} (p{}, r{})".format(
                    cpi_i, kind, pulse_idx[cpi_i], range_idx[cpi_i]), fontsize=7)
                ax.tick_params(labelsize=6)
                ax.grid(True, alpha=0.3)

        subtitle = "" if N <= max_grid else " (showing {}/{}, evenly sampled)".format(n_total, N)
        part_info = " - Part {}/{}".format(img_idx + 1, n_images) if n_images > 1 else ""
        fig.suptitle("Profile grid: freq {}{}{}  [all {}]"
                     .format(freq, subtitle, part_info, n_show), fontsize=11)
        fig.text(0.5, 0.01, "Index", ha="center", fontsize=9)
        fig.text(0.01, 0.5, "Power (dB)", va="center", rotation="vertical", fontsize=9)
        fig.tight_layout(rect=[0.02, 0.02, 1, 0.97])

        out_file = ("{}_part{:02d}{}".format(out_base, img_idx + 1, out_ext)
                    if n_images > 1 else out_path)
        fig.savefig(out_file, dpi=130)
        plt.close(fig)
        output_files.append(out_file)
    return output_files


# ---------------------------------------------------------------------------
# LABEL-AWARE PLOTS (only when the file carries labels / jsr) -- from the old
# data generator, now driven from the written file
# ---------------------------------------------------------------------------

def _jsr_range_str(jsr_row):
    """Format the realized JSR range for one tile's jsr_db row (NaN-padded)."""
    if jsr_row is None:
        return ""
    vals = jsr_row[np.isfinite(jsr_row)]
    if vals.size == 0:
        return ""
    return f" | JSR {vals.min():.0f}-{vals.max():.0f} dB"


def plot_label_overlay(ch, sel, out_dir, max_bands):
    """Eigenvalue overlay + per-block grid, colored by knee (labels)."""
    eig = ch.eigenvalues
    profiles_db = [10.0 * np.log10(np.maximum(eig[i], EPS)) for i in sel]
    knees = [int(ch.labels[i]) for i in sel]
    cpi_len = eig.shape[1]
    ev_index = np.arange(1, cpi_len + 1)

    global_max_db = float(np.max(np.concatenate(profiles_db)))
    top = max(global_max_db + EV_YLIM_TOP_MARGIN_DB, EV_YLIM_MIN_TOP_DB)
    ylim = [EV_YLIM_BOTTOM_DB, top]

    norm = mcolors.Normalize(vmin=0, vmax=max(max_bands, 1))
    cmap = cm.plasma
    freq, pol = ch.freq, ch.pol

    # --- overlay -----------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(11, 6))
    for prof, knee in zip(profiles_db, knees):
        ax1.plot(ev_index, prof, color=cmap(norm(knee)), alpha=0.75, linewidth=1.2)
        if knee > 0:
            ax1.axvline(x=knee + 0.5, color=cmap(norm(knee)), linestyle=":",
                        linewidth=1.5, alpha=0.5)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig1.colorbar(sm, ax=ax1).set_label("Number of RFI eigenvalues (injected bands)", fontsize=10)
    ax1.set_xlabel("Eigenvalue index (1-based, descending)", fontsize=11)
    ax1.set_ylabel("Eigenvalue (dB)", fontsize=11)
    ax1.set_ylim(ylim)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.set_title(f"Eigenvalue profiles -- {freq}-{pol}\n{len(sel)} sampled CPI blocks", fontsize=11)
    fig1.tight_layout()
    p1 = os.path.join(out_dir, f"{freq}_{pol}_ev_overlay.png")
    fig1.savefig(p1, dpi=150)
    plt.close(fig1)

    # --- per-block grid ----------------------------------------------------
    n = len(sel)
    n_cols = min(6, n)
    n_rows = int(np.ceil(n / n_cols))
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 3.1 * n_rows), squeeze=False)
    for ax, i, prof in zip(axes.flat, sel, profiles_db):
        knee = int(ch.labels[i])
        ax.plot(ev_index, prof, color=cmap(norm(knee)), linewidth=1.5)
        if knee > 0:
            ax.axvline(x=knee + 0.5, color="red", linestyle=":", linewidth=1.5, alpha=0.7)
        label = ("CLEAN" if knee == 0
                 else (f"{knee} RFI eigenvalue" if knee == 1 else f"{knee} RFI eigenvalues"))
        jsr_str = _jsr_range_str(ch.jsr_db[i] if ch.jsr_db is not None else None)
        power = float(ch.signal_power_db[i]) if ch.signal_power_db is not None else float("nan")
        vfrac = float(ch.valid_fraction[i]) if ch.valid_fraction is not None else 1.0
        p_abs = int(ch.tile_pulse[i]) if ch.tile_pulse is not None else i
        r_abs = int(ch.tile_range[i]) if ch.tile_range is not None else 0
        ax.set_title(f"p={p_abs} r={r_abs} [{label}]{jsr_str}\n"
                     f"baseline={power:.1f} dB | valid={vfrac*100:.0f}%", fontsize=7)
        ax.set_xlabel("EV index", fontsize=8)
        ax.set_ylabel("Eigenvalue (dB)", fontsize=8)
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig2.suptitle(f"Eigenvalue profiles per sampled block -- {freq}-{pol}", fontsize=12)
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    p2 = os.path.join(out_dir, f"{freq}_{pol}_ev_blocks.png")
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)

    return [p1, p2]


def plot_scm_heatmap(ch, sel, out_dir):
    """Grid of SCM magnitude heatmaps (20*log10|R_ij|) for the sampled blocks."""
    pairs = fetch_cpi_tiles(ch, sel)
    if not pairs:
        return None
    idxs = [i for i, _ in pairs]
    mags_db = [20.0 * np.log10(np.abs(scm) + EPS) for _, scm in pairs]
    finite = np.concatenate([m[np.isfinite(m)].ravel() for m in mags_db])
    vmin, vmax = np.percentile(finite, 5), np.percentile(finite, 100)

    n = len(idxs)
    n_cols = min(6, n)
    n_rows = int(np.ceil(n / n_cols))
    freq, pol = ch.freq, ch.pol
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.2 * n_rows), squeeze=False)

    im = None
    for ax, i, mag in zip(axes.flat, idxs, mags_db):
        im = ax.imshow(mag, cmap="inferno", vmin=vmin, vmax=vmax,
                       origin="upper", interpolation="nearest")
        if ch.labels is not None:
            knee = int(ch.labels[i])
            label = "CLEAN" if knee == 0 else f"RFI={knee}"
        else:
            label = ""
        rows = ""
        if ch.band_rows is not None:
            br = sorted({int(v) for v in ch.band_rows[i] if int(v) >= 0})
            rows = "" if not br else f"\nrows={br}"
        jsr_str = _jsr_range_str(ch.jsr_db[i] if ch.jsr_db is not None else None)
        p_abs = int(ch.tile_pulse[i]) if ch.tile_pulse is not None else i
        r_abs = int(ch.tile_range[i]) if ch.tile_range is not None else 0
        ax.set_title(f"p={p_abs} r={r_abs} [{label}]{rows}{jsr_str.strip(' |')}", fontsize=7)
        ax.set_xlabel("Pulse j", fontsize=8)
        ax.set_ylabel("Pulse i", fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axes.flat[n:]:
        ax.axis("off")
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85).set_label("|SCM| (dB)", fontsize=10)
    fig.suptitle(f"Gap-exclusion SCM magnitude -- {freq}-{pol}", fontsize=12)
    p = os.path.join(out_dir, f"{freq}_{pol}_scm.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# PER-CHANNEL DRIVER
# ---------------------------------------------------------------------------

def profile_channel(ch, args):
    """Render every figure for one channel; return its dB arrays for later reuse."""
    ensure_diagonal(ch)
    eig_db = eigvals_to_db(ch.eigenvalues)
    diag_db = eigvals_to_db(ch.diagonal)
    freq, pol, tag = ch.freq, ch.pol, ch.tag
    out_dir = args.output_dir
    pulse_idx = ch.tile_pulse if ch.tile_pulse is not None else np.arange(ch.n_tiles)
    range_idx = ch.tile_range if ch.tile_range is not None else np.zeros(ch.n_tiles, dtype=int)

    plot_average_shape(eig_db, os.path.join(out_dir, f"{tag}_avg_shape.png"), freq, pol, N_SHOW)
    plot_overlaid(eig_db, os.path.join(out_dir, f"{tag}_overlaid.png"), freq, pol, N_SHOW)
    plot_scm_diag(diag_db, os.path.join(out_dir, f"{tag}_scm_diag.png"), freq, pol, N_SHOW)
    grid_files = plot_grid(eig_db, diag_db, pulse_idx, range_idx,
                           os.path.join(out_dir, f"{tag}_grid.png"), freq, pol, N_SHOW, args.max_grid)
    print(f"  [{tag}] wrote avg_shape, overlaid, scm_diag, grid ({len(grid_files)} part(s))")

    if ch.labels is not None:
        max_bands = int(ch.attrs.get("max_bands", max(1, int(ch.labels.max()))))
        sel = select_plot_blocks(args.plot_seed, tag, ch.n_tiles, args.n_plot_blocks)
        label_files = plot_label_overlay(ch, sel, out_dir, max_bands)
        print(f"  [{tag}] wrote label-aware overlay + per-block grid "
              f"({len(label_files)} files, {len(sel)} blocks)")

    if args.scm_heatmap:
        sel = select_plot_blocks(args.plot_seed, tag, ch.n_tiles, args.n_plot_blocks)
        p = plot_scm_heatmap(ch, sel, out_dir)
        if p:
            print(f"  [{tag}] wrote SCM heatmap grid")

    return {"eig_db": eig_db, "diag_db": diag_db,
            "pulse_idx": pulse_idx, "range_idx": range_idx, "freq": freq, "pol": pol}


def maybe_combined_grid(results, args):
    """Write a combined HH+HV grid per frequency when both pols are present."""
    by_freq = {}
    for r in results:
        by_freq.setdefault(r["freq"], {})[r["pol"]] = r
    for freq, pol_map in by_freq.items():
        if "HH" not in pol_map or "HV" not in pol_map:
            continue
        hh, hv = pol_map["HH"], pol_map["HV"]
        n = min(hh["eig_db"].shape[0], hv["eig_db"].shape[0])
        eig_y_max = np.ceil(max(np.max(hh["eig_db"][:n, :12]), np.max(hv["eig_db"][:n, :12])))
        diag_hh_12 = np.sort(hh["diag_db"][:n], axis=1)[:, ::-1][:, :12]
        diag_hv_12 = np.sort(hv["diag_db"][:n], axis=1)[:, ::-1][:, :12]
        diag_y_max = np.ceil(max(np.max(diag_hh_12), np.max(diag_hv_12)))
        out = os.path.join(args.output_dir, f"profile_grid_{freq}.png")
        files = plot_grid_combined(
            hh["eig_db"][:n], hv["eig_db"][:n], hh["diag_db"][:n], hv["diag_db"][:n],
            hh["pulse_idx"][:n], hh["range_idx"][:n], out, freq, N_SHOW, args.max_grid,
            0, eig_y_max, 0, diag_y_max)
        print(f"  combined HH+HV grid for freq {freq}: {len(files)} part(s)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*",
                        help="One or more flat per-CPI HDF5 channel files.")
    parser.add_argument("--output-dir", default="results/profiles",
                        help="Directory for the output PNGs (default: results/profiles)")
    parser.add_argument("--max-grid", type=int, default=48,
                        help="Max CPIs drawn in the per-CPI grid (evenly sampled beyond this)")
    parser.add_argument("--plot-seed", type=int, default=99,
                        help="Seed for sampling blocks in the label overlay / SCM heatmap")
    parser.add_argument("--n-plot-blocks", type=int, default=12,
                        help="Number of blocks sampled for the label overlay / SCM heatmap")
    parser.add_argument("--scm-heatmap", action="store_true",
                        help="Also draw SCM magnitude heatmaps (needs a `cpi` dataset or the L0B)")

    # Direct raw-L0B mode (no preprocessed file yet).
    parser.add_argument("--l0b", default=None, help="Profile this raw NISAR L0B granule directly.")
    parser.add_argument("--freq", default=None, help="[--l0b] frequency, e.g. A")
    parser.add_argument("--pol", default=None, help="[--l0b] polarization, e.g. HH")
    parser.add_argument("--pulse-start", type=int, default=None, help="[--l0b] window start pulse")
    parser.add_argument("--pulse-end", type=int, default=None, help="[--l0b] window end pulse")
    parser.add_argument("--range-start", type=int, default=None, help="[--l0b] window start range")
    parser.add_argument("--range-end", type=int, default=None, help="[--l0b] window end range")
    parser.add_argument("--cpi-len", type=int, default=16, help="[--l0b] CPI pulse length")
    parser.add_argument("--cpi-width", type=int, default=250, help="[--l0b] CPI range width")
    parser.add_argument("--no-remove-caltone", dest="remove_caltone", action="store_false",
                        help="[--l0b] leave the instrument caltone in the raw data")
    parser.add_argument("--compute-subswath-mask", action="store_true",
                        help="[--l0b] use the gap-exclusion subswath mask")
    parser.set_defaults(remove_caltone=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    channels = []
    if args.l0b:
        if not (args.freq and args.pol):
            print("--l0b requires --freq and --pol.", file=sys.stderr)
            sys.exit(1)
        print(f"[+] Profiling raw L0B: {args.l0b}  freq {args.freq} pol {args.pol}")
        ch = channel_from_l0b(
            args.l0b, args.freq, args.pol,
            pulse_start=args.pulse_start, pulse_end=args.pulse_end,
            range_start=args.range_start, range_end=args.range_end,
            cpi_len=args.cpi_len, cpi_width=args.cpi_width,
            remove_caltone=args.remove_caltone, use_mask=args.compute_subswath_mask)
        describe_coverage(ch)
        channels.append(ch)
    else:
        if not args.inputs:
            print("No input files given (or use --l0b). See --help.", file=sys.stderr)
            sys.exit(1)
        for path in args.inputs:
            print(f"[+] Reading {path}")
            ch = load_channel(path)
            describe_coverage(ch)
            channels.append(ch)

    results = [profile_channel(ch, args) for ch in channels]
    maybe_combined_grid(results, args)
    print("\n[+] Done.")


if __name__ == "__main__":
    main()
