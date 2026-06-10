"""
plot_motion_fraction.py — comparison plotter.

Python analogue of motion_fraction.m. Loads one or two
*_motion_fraction.txt files (the ones cycle.py / cycle.m write) and plots
MF vs time on identical axes.

Run:
    # one curve
    python plot_motion_fraction.py file_a.txt

    # two curves on separate figures (same as the original MATLAB script)
    python plot_motion_fraction.py file_a.txt file_b.txt

    # two curves OVERLAID on one figure (more useful for comparison)
    python plot_motion_fraction.py file_a.txt file_b.txt --overlay

Options:
    --n         number of rows to read (default: 401, like the MATLAB script)
    --ymax      y-axis upper limit (default: 0.12)
    --xmax      x-axis upper limit (default: 450)
    --labels    legend labels, space-separated and quoted if needed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_motion_fraction(path: Path, n: int | None = 401):
    """Read a *_motion_fraction.txt file and return (t, mf) arrays."""
    # The file has a header "Second\tMotionFraction"; np.loadtxt with skiprows=1
    # is robust to either tab or whitespace separators.
    data = np.loadtxt(path, skiprows=1)
    if n is not None:
        data = data[:n]
    return data[:, 0], data[:, 1]


def _style_axes(ax, xmax, ymax, fontsize=18):
    ax.set_xlabel("Time (s)", fontsize=fontsize + 6)
    ax.set_ylabel("MF", fontsize=fontsize + 6)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.tick_params(labelsize=fontsize)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path,
                    help="one or two *_motion_fraction.txt files")
    ap.add_argument("--n", type=int, default=401,
                    help="rows to read from each file (default 401)")
    ap.add_argument("--ymax", type=float, default=0.12)
    ap.add_argument("--xmax", type=float, default=450)
    ap.add_argument("--overlay", action="store_true",
                    help="put both curves on a single figure")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="legend labels, one per file")
    ap.add_argument("--save", type=Path, default=None,
                    help="optional path to save the (last) figure as PNG")
    args = ap.parse_args()

    if len(args.files) > 2:
        ap.error("Pass at most two files (matches the original MATLAB script).")

    colors = ["red", "blue"]
    default_labels = ["Surface with cone", "Surface without cone"]
    labels = args.labels if args.labels else default_labels[:len(args.files)]

    if args.overlay:
        fig, ax = plt.subplots(figsize=(9, 5))
        for path, color, label in zip(args.files, colors, labels):
            t, mf = load_motion_fraction(path, n=args.n)
            ax.plot(t, mf, color=color, linewidth=2, label=label)
        _style_axes(ax, args.xmax, args.ymax)
        ax.legend(fontsize=18)
        fig.tight_layout()
        if args.save:
            fig.savefig(args.save, dpi=150)
        plt.show()
    else:
        for path, color, label in zip(args.files, colors, labels):
            fig, ax = plt.subplots(figsize=(9, 5))
            t, mf = load_motion_fraction(path, n=args.n)
            ax.plot(t, mf, color=color, linewidth=2, label=label)
            _style_axes(ax, args.xmax, args.ymax)
            ax.legend(fontsize=18)
            fig.tight_layout()
            if args.save and path is args.files[-1]:
                fig.savefig(args.save, dpi=150)
        plt.show()


if __name__ == "__main__":
    main()
