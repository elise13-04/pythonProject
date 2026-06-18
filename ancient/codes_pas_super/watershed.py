"""
watershed_droplets.py
=====================
Segmente des gouttes avec fortes variations de taille et d'éclairage.
Utilise un seuillage local (Adaptive Thresholding) pour isoler les
gouttes quelle que soit l'illumination de l'arrière-plan.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from skimage import io, color, filters, morphology, segmentation, measure
from skimage.feature import peak_local_max

# ─────────────────────────── CONFIG (défauts) ────────────────────────────────
BLOCK_SIZE = 75  # Taille du voisinage pour le seuillage local (doit être impair)
OFFSET = 0.015  # Tolérance au dessus du fond local (évite de segmenter le bruit)
MIN_SIZE = 15  # Aire minimale (réduite car les gouttes à gauche sont très petites)
MAX_SIZE = 50_000  # Aire maximale


# ─────────────────────────────────────────────────────────────────────────────

def load_gray(path: Path) -> np.ndarray:
    img = io.imread(str(path))
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[..., :3]
        gray = color.rgb2gray(img)
    else:
        gray = img.astype(np.float64) / np.iinfo(img.dtype).max
    return gray


def segment_droplets(
        gray: np.ndarray,
        block_size: int = BLOCK_SIZE,
        offset: float = OFFSET,
        min_size: int = MIN_SIZE,
        max_size: int = MAX_SIZE,
) -> tuple[np.ndarray, list]:
    # ── 1. Seuillage Local Adaptatif ─────────────────────────────────────────
    # Calcule un seuil différent pour chaque région de l'image.
    # block_size doit être un peu plus grand que la plus grosse goutte.
    local_thresh = filters.threshold_local(gray, block_size=block_size, method='gaussian', offset=offset)
    binary = gray > local_thresh

    # ── 2. Nettoyage des petits artéfacts ────────────────────────────────────
    binary = morphology.remove_small_objects(binary, min_size=5)

    # ── 3. Remplissage des gouttes creuses ───────────────────────────────────
    # Les grosses gouttes au centre ont des reflets en anneau. On remplit le centre.
    binary = ndi.binary_fill_holes(binary)

    # ── 4. Distance transform & Graines (Watershed) ──────────────────────────
    dist = ndi.distance_transform_edt(binary)

    # On lisse très légèrement la carte de distance pour éviter qu'une grosse
    # goutte cabossée ne génère plusieurs centres.
    dist_smooth = filters.gaussian(dist, sigma=1.0)

    # On autorise des centres très proches (min_distance=3) pour bien séparer
    # les toutes petites gouttes à gauche.
    coords = peak_local_max(dist_smooth, min_distance=3, labels=binary)

    mask_seeds = np.zeros(dist.shape, dtype=bool)
    mask_seeds[tuple(coords.T)] = True
    markers, _ = ndi.label(mask_seeds)

    # ── 5. Watershed ─────────────────────────────────────────────────────────
    # Pas besoin de compacité ici, on laisse l'eau monter naturellement.
    labels = segmentation.watershed(
        -dist_smooth,
        markers,
        mask=binary,
        watershed_line=True  # Force la création d'une ligne de démarcation entre gouttes collées
    )

    # ── 6. Filtrage par aire ─────────────────────────────────────────────────
    props_all = measure.regionprops(labels)
    for rp in props_all:
        if rp.area < min_size or rp.area > max_size:
            labels[labels == rp.label] = 0

    labels, _, _ = segmentation.relabel_sequential(labels)
    props = measure.regionprops(labels)

    return labels, props


def make_figure(
        gray: np.ndarray,
        labels: np.ndarray,
        title_prefix: str = "",
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"{title_prefix}Raw image", fontsize=14)
    axes[0].axis("off")

    # Mode "inner" permet d'avoir des contours plus fins et précis
    boundary = segmentation.find_boundaries(labels, mode="inner")

    display_img = color.gray2rgb(gray) * 0.6
    display_img[boundary] = [0, 1, 0]

    axes[1].imshow(display_img)
    axes[1].set_title(f"{title_prefix}Contours (n={labels.max()})", fontsize=14)
    axes[1].axis("off")

    fig.tight_layout(pad=1.0)
    return fig


def main():
    ap = argparse.ArgumentParser(description="Watershed segmentation")
    ap.add_argument("image", type=Path, help="Chemin vers l'image")
    ap.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    ap.add_argument("--offset", type=float, default=OFFSET)
    ap.add_argument("--min-size", type=int, default=MIN_SIZE)
    ap.add_argument("--max-size", type=int, default=MAX_SIZE)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    # block_size doit être impair
    if args.block_size % 2 == 0:
        args.block_size += 1

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    out = args.output or args.image.parent / f"{args.image.stem}_watershed.png"

    print(f"Loading  : {args.image}")
    gray = load_gray(args.image)

    print("Segmenting…")
    labels, props = segment_droplets(
        gray,
        block_size=args.block_size,
        offset=args.offset,
        min_size=args.min_size,
        max_size=args.max_size,
    )
    print(f"Droplets detected : {len(props)}")

    fig = make_figure(gray, labels, title_prefix=f"{args.image.name}  —  ")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved    : {out}")


if __name__ == "__main__":
    main()