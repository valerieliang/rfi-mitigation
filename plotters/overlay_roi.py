#!/usr/bin/env python
"""
plotters/overlay_roi.py

Draw a pulse/range region-of-interest box over a focused scene image, so a
region picked by eye can be turned into the --pulse-start/--pulse-end and
--range-start/--range-end window that plot_profiles.py --l0b consumes.

The image is assumed to span the full pulse and range extent given, with the
FIRST pulse at the BOTTOM (ascending pass shown north-up, azimuth increasing
upward). Pass --pulse-down if the image instead runs earliest-pulse-first from
the top.

Usage:
    python plotters/overlay_roi.py scene.png \
        --pulse 681445 757408 --range 0 54253 \
        --box-pulse 694000 711000 --box-range 41800 54253 \
        --out roi_box.png
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", help="focused scene PNG/JPG to overlay")
    p.add_argument("--pulse", nargs=2, type=int, required=True,
                   metavar=("MIN", "MAX"), help="pulse extent of the whole image")
    p.add_argument("--range", nargs=2, type=int, required=True,
                   metavar=("MIN", "MAX"), help="range-sample extent of the whole image")
    p.add_argument("--box-pulse", nargs=2, type=int, required=True,
                   metavar=("MIN", "MAX"), help="pulse extent of the ROI box")
    p.add_argument("--box-range", nargs=2, type=int, required=True,
                   metavar=("MIN", "MAX"), help="range extent of the ROI box")
    p.add_argument("--hline", type=int, default=None,
                   help="draw a dashed horizontal line at this pulse")
    p.add_argument("--pulse-down", action="store_true",
                   help="image runs earliest pulse at the TOP instead of the bottom")
    p.add_argument("--out", default="roi_box.png")
    return p.parse_args()


def main():
    args = parse_args()
    im = mpimg.imread(args.image)
    h, w = im.shape[:2]

    p0, p1 = args.pulse
    r0, r1 = args.range

    def fx(r):
        return (r - r0) / (r1 - r0) * w

    def fy(p):
        frac = (p - p0) / (p1 - p0)
        return h * frac if args.pulse_down else h * (1.0 - frac)

    fig, ax = plt.subplots(figsize=(13, 13 * h / w))
    ax.imshow(im)

    bp0, bp1 = sorted(args.box_pulse)
    br0, br1 = sorted(args.box_range)
    y_top, y_bot = sorted((fy(bp0), fy(bp1)))
    ax.add_patch(plt.Rectangle((fx(br0), y_top), fx(br1) - fx(br0), y_bot - y_top,
                               fill=False, edgecolor="cyan", lw=2.5))
    ax.text(fx(br0), y_top - 8, f"pulse {bp0:,}-{bp1:,}  range {br0:,}-{br1:,}",
            color="cyan", fontsize=10, weight="bold")

    if args.hline is not None:
        ax.axhline(fy(args.hline), color="white", ls="--", lw=1.2)
        ax.text(6, fy(args.hline) - 6, f"pulse {args.hline:,}",
                color="white", fontsize=9)

    xt = np.linspace(r0, r1, 5).astype(int)
    yt = np.linspace(p0, p1, 5).astype(int)
    ax.set_xticks([fx(v) for v in xt])
    ax.set_xticklabels([f"{v:,}" for v in xt])
    ax.set_yticks([fy(v) for v in yt])
    ax.set_yticklabels([f"{v:,}" for v in yt])
    ax.set_xlabel("range sample")
    ax.set_ylabel("pulse")

    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
