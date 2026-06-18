"""
Script simple de visualisation.
Charge un ou deux fichiers *_motion_fraction.txt et les trace
avec des axes identiques, pour comparer deux conditions
expérimentales (ex : surface avec cône vs sans cône).
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
    data = np.loadtxt(path, skiprows=1) #ignore la ligne d'en-tête
    if n is not None:
        data = data[:n] #garde seulement les n première lignes
    return data[:, 0], data[:, 1] #colonne 0: tps, colonne 1: mf


def _style_axes(ax, xmax, ymax, fontsize=18): #standardise les axes pr que les deux figures soient directement comparables visuellement
    ax.set_xlabel("Time (s)", fontsize=fontsize + 6) #police 24pt
    ax.set_ylabel("MF", fontsize=fontsize + 6)
    ax.set_xlim(0, xmax) #fixe axe X de 0 à xmax
    ax.set_ylim(0, ymax) #fixe axe Y de 0 à ymax
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

    if args.overlay: #mode overlay trace les courbes sur le mm graphique, avec rouge et bleu pour les distinguer
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
    else: #sans le mode overlay: une figure séparée par fichier
        for path, color, label in zip(args.files, colors, labels):
            fig, ax = plt.subplots(figsize=(9, 5))
            t, mf = load_motion_fraction(path, n=args.n)
            ax.plot(t, mf, color=color, linewidth=2, label=label)
            _style_axes(ax, args.xmax, args.ymax)
            ax.legend(fontsize=18)
            fig.tight_layout()
            if args.save and path is args.files[-1]:
                fig.savefig(args.save, dpi=150) #sauvegarde seulement la dernière figure
        plt.show()


if __name__ == "__main__":
    main()
