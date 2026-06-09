"""
condensation_detector.py
========================
Détection de gouttes de condensation sur une image unique,
avec calcul d'une ROI englobant les gouttes les plus significatives.

ROI logic :
  - Grosses gouttes détectées  → ROI sur toutes les grosses (cyan)
  - Aucune grosse              → ROI sur les N petites les plus grandes (orange)

Usage :
    python condensation_detector.py image.jpg
    python condensation_detector.py image.jpg --output resultat.jpg
    python condensation_detector.py image.jpg --roi-margin 20
    python condensation_detector.py image.jpg --no-labels
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import random

import cv2
import numpy as np

# ─── Constantes ────────────────────────────────────────────────────────────────

MIN_DIST_SMALL_LARGE = 1.3   # Facteur pour filtrer petites gouttes dans les grandes
DEFAULT_ROI_MARGIN   = 15    # Marge en pixels autour de la ROI
ROI_FALLBACK_TOP_N   = 5     # Nb de petites gouttes utilisées si pas de grosses


# ─── Couleur déterministe par ID ────────────────────────────────────────────────

def make_color(seed: int) -> Tuple[int, int, int]:
    """Génère une couleur BGR reproductible à partir d'un entier."""
    rng = random.Random(seed * 2654435761)
    h = rng.randint(0, 179)
    hsv = np.uint8([[[h, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# ─── Détection Multi-Pass ───────────────────────────────────────────────────────

def detect_droplets(
    gray: np.ndarray,
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    """
    Détecte les gouttes sur une image en niveaux de gris.

    Retourne :
        large_drops : liste de (x, y, r) pour les grosses gouttes (r >= 30)
        small_drops : liste de (x, y, r) pour les petites gouttes validées
    """
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred  = cv2.GaussianBlur(enhanced, (9, 9), 2)

    # ── Passe 1 : Grosses gouttes ──────────────────────────────────────────────
    large_drops: List[Tuple[int, int, int]] = []
    circles_large = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=100, param1=100, param2=34,
        minRadius=30, maxRadius=50,
    )
    if circles_large is not None:
        for x, y, r in np.round(circles_large[0]).astype(int):
            large_drops.append((int(x), int(y), int(r)))

    # ── Passe 2 : Petites gouttes ──────────────────────────────────────────────
    small_drops: List[Tuple[int, int, int]] = []
    circles_small = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=30, param1=100, param2=30,
        minRadius=15, maxRadius=29,
    )
    if circles_small is not None:
        for x, y, r in np.round(circles_small[0]).astype(int):
            is_inside_large = any(
                ((x - lx) ** 2 + (y - ly) ** 2) ** 0.5 < lr * MIN_DIST_SMALL_LARGE
                for lx, ly, lr in large_drops
            )
            if not is_inside_large:
                small_drops.append((int(x), int(y), int(r)))

    return large_drops, small_drops


# ─── Statistiques de rayons ────────────────────────────────────────────────────

def radius_statistics(
    large_drops: List[Tuple[int, int, int]],
    small_drops: List[Tuple[int, int, int]],
) -> dict:
    """
    Calcule les statistiques de rayons sur l'ensemble des gouttes détectées
    (grosses + petites).

    Retourne un dict avec :
        r_max        : rayon de la plus grande goutte (px)
        r_min        : rayon de la plus petite goutte (px)
        r_mean       : rayon moyen (px)
        drop_max     : (x, y, r) de la plus grande goutte
        drop_min     : (x, y, r) de la plus petite goutte
        n_large      : nombre de grosses gouttes
        n_small      : nombre de petites gouttes
        n_total      : nombre total de gouttes
    """
    all_drops = large_drops + small_drops
    if not all_drops:
        return {}

    radii     = [r for _, _, r in all_drops]
    drop_max  = max(all_drops, key=lambda d: d[2])
    drop_min  = min(all_drops, key=lambda d: d[2])

    return {
        "r_max":    drop_max[2],
        "r_min":    drop_min[2],
        "r_mean":   float(np.mean(radii)),
        "drop_max": drop_max,
        "drop_min": drop_min,
        "n_large":  len(large_drops),
        "n_small":  len(small_drops),
        "n_total":  len(all_drops),
    }


# ─── Calcul de la ROI ──────────────────────────────────────────────────────────

def compute_roi(
    large_drops: List[Tuple[int, int, int]],
    small_drops: List[Tuple[int, int, int]],
    img_shape: Tuple[int, int],
    margin: int = DEFAULT_ROI_MARGIN,
) -> Tuple[Optional[Tuple[int, int, int, int]], bool]:
    """
    Calcule la bounding box englobant les gouttes de référence (+ marge).

    Règle de sélection :
      - Si des grosses gouttes existent  → on les utilise toutes.
      - Sinon                            → on prend les ROI_FALLBACK_TOP_N
                                           petites gouttes les plus grandes
                                           (triées par rayon décroissant).

    Retourne :
        (roi, is_fallback)
        roi         : (x1, y1, x2, y2) clampé aux dimensions de l'image,
                      ou None si aucune goutte disponible.
        is_fallback : True si la ROI est basée sur des petites gouttes.
    """
    if large_drops:
        reference   = large_drops
        is_fallback = False
    elif small_drops:
        reference   = sorted(small_drops, key=lambda d: d[2], reverse=True)[:ROI_FALLBACK_TOP_N]
        is_fallback = True
    else:
        return None, False

    h, w = img_shape

    x1 = max(0,     min(cx - r for cx, cy, r in reference) - margin)
    y1 = max(0,     min(cy - r for cx, cy, r in reference) - margin)
    x2 = min(w - 1, max(cx + r for cx, cy, r in reference) + margin)
    y2 = min(h - 1, max(cy + r for cx, cy, r in reference) + margin)

    return (x1, y1, x2, y2), is_fallback


# ─── Statistiques dans la ROI ──────────────────────────────────────────────────

def roi_statistics(
    roi: Tuple[int, int, int, int],
    large_drops: List[Tuple[int, int, int]],
    small_drops: List[Tuple[int, int, int]],
    is_fallback: bool = False,
) -> dict:
    """
    Calcule, dans la ROI :
      - le nombre de grosses gouttes (ou top-N petites si fallback)
      - le nombre de petites gouttes dont le centre est dans la ROI
      - le nombre total de gouttes
      - la surface de la ROI (px²)
      - la surface cumulée des disques (px²) via masque binaire (sans doublons)
      - le pourcentage d'occupation
    """
    x1, y1, x2, y2 = roi
    roi_w    = x2 - x1
    roi_h    = y2 - y1
    roi_area = roi_w * roi_h

    mask = np.zeros((roi_h, roi_w), dtype=np.uint8)

    # Grosses gouttes (ou top-N petites en mode fallback) → toutes dans la ROI
    n_large = len(large_drops)
    for cx, cy, r in large_drops:
        cv2.circle(mask, (cx - x1, cy - y1), r, 255, -1)

    # Petites gouttes dont le centre est dans la ROI
    small_in_roi = [
        (cx, cy, r) for cx, cy, r in small_drops
        if x1 <= cx <= x2 and y1 <= cy <= y2
    ]
    n_small = len(small_in_roi)
    for cx, cy, r in small_in_roi:
        cv2.circle(mask, (cx - x1, cy - y1), r, 255, -1)

    covered_area = int(np.count_nonzero(mask))
    coverage_pct = 100.0 * covered_area / roi_area if roi_area > 0 else 0.0

    return {
        "n_large":      n_large,
        "n_small":      n_small,
        "n_total":      n_large + n_small,
        "roi_area_px":  roi_area,
        "drop_area_px": covered_area,
        "coverage_pct": coverage_pct,
        "is_fallback":  is_fallback,
    }


# ─── Annotation de l'image ─────────────────────────────────────────────────────

def annotate_image(
    image: np.ndarray,
    large_drops: List[Tuple[int, int, int]],
    small_drops: List[Tuple[int, int, int]],
    roi: Optional[Tuple[int, int, int, int]],
    stats: Optional[dict],
    rad_stats: Optional[dict],
    show_labels: bool = True,
) -> np.ndarray:
    """Dessine les gouttes, la ROI, les statistiques et les rayons extrêmes sur l'image (BGR)."""
    out = image.copy()

    all_drops = [(cx, cy, r) for cx, cy, r in large_drops] + \
                [(cx, cy, r) for cx, cy, r in small_drops]

    for idx, (cx, cy, r) in enumerate(all_drops):
        col = make_color(idx)
        cv2.circle(out, (cx, cy), max(r, 4), col, 2, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 2, col, -1, cv2.LINE_AA)
        if show_labels:
            cv2.putText(out, str(idx), (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # ── Mise en évidence des gouttes extrêmes ──────────────────────────────────
    if rad_stats:
        mx, my, mr = rad_stats["drop_max"]
        mn_x, mn_y, mn_r = rad_stats["drop_min"]

        # Grosse goutte → cercle blanc épais + label
        cv2.circle(out, (mx, my), mr + 4, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, f"MAX r={mr}px", (mx + mr + 6, my),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Petite goutte → cercle jaune épais + label
        cv2.circle(out, (mn_x, mn_y), mn_r + 4, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, f"MIN r={mn_r}px", (mn_x + mn_r + 6, mn_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    # ── Dessin de la ROI ───────────────────────────────────────────────────────
    if roi is not None:
        x1, y1, x2, y2 = roi
        is_fallback = stats.get("is_fallback", False) if stats else False

        roi_color = (0, 165, 255) if is_fallback else (0, 220, 220)

        overlay = out.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), roi_color, -1)
        cv2.addWeighted(overlay, 0.08, out, 0.92, 0, out)

        cv2.rectangle(out, (x1, y1), (x2, y2), roi_color, 2, cv2.LINE_AA)

        roi_label = f"ROI (top-{ROI_FALLBACK_TOP_N} petites)" if is_fallback else "ROI"
        cv2.putText(out, roi_label, (x1 + 4, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, roi_color, 2)

    # ── Panneau de statistiques ────────────────────────────────────────────────
    total = len(large_drops) + len(small_drops)
    lines = [f"Total gouttes : {total}"]

    # Rayons extrêmes
    if rad_stats:
        lines += [
            f"  Rayon MAX  : {rad_stats['r_max']} px  (pos {rad_stats['drop_max'][:2]})",
            f"  Rayon MIN  : {rad_stats['r_min']} px  (pos {rad_stats['drop_min'][:2]})",
            f"  Rayon moyen: {rad_stats['r_mean']:.1f} px",
        ]

    if stats:
        fb = stats["is_fallback"]
        label_large = f"Top-{ROI_FALLBACK_TOP_N} petites (ROI)" if fb else "Grosses (ROI)"
        lines += [
            f"  {label_large} : {stats['n_large']}",
            f"  Petites (ROI)  : {stats['n_small']}",
            f"  Total   (ROI)  : {stats['n_total']}",
            f"  Surface ROI    : {stats['roi_area_px']:,} px²",
            f"  Surface gouttes: {stats['drop_area_px']:,} px²",
            f"  Occupation     : {stats['coverage_pct']:.1f} %",
        ]

    y_txt = 28
    for line in lines:
        cv2.putText(out, line, (10, y_txt),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
        y_txt += 24

    return out


# ─── Pipeline principal ─────────────────────────────────────────────────────────

def run_pipeline(
    image_path: Path,
    output_path: Path,
    show_labels: bool,
    roi_margin: int,
) -> None:
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"Erreur : impossible de charger l'image '{image_path}'.")
        return

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # ── Détection ─────────────────────────────────────────────────────────────
    print("Détection des gouttes en cours...")
    large_drops, small_drops = detect_droplets(gray)
    print(f"  → {len(large_drops)} grosse(s) + {len(small_drops)} petite(s) "
          f"= {len(large_drops) + len(small_drops)} goutte(s) au total.")

    if not large_drops and not small_drops:
        print("Aucune goutte détectée. Vérifiez les paramètres HoughCircles.")
        return

    # ── Statistiques de rayons ─────────────────────────────────────────────────
    rad_stats = radius_statistics(large_drops, small_drops)
    print(f"\n── Rayons des gouttes ────────────────────────────────")
    print(f"  Rayon MAX   : {rad_stats['r_max']} px  "
          f"(centre : {rad_stats['drop_max'][:2]})")
    print(f"  Rayon MIN   : {rad_stats['r_min']} px  "
          f"(centre : {rad_stats['drop_min'][:2]})")
    print(f"  Rayon moyen : {rad_stats['r_mean']:.1f} px")
    print(f"─────────────────────────────────────────────────────")

    # ── ROI ───────────────────────────────────────────────────────────────────
    roi, is_fallback = compute_roi(large_drops, small_drops, (h, w), margin=roi_margin)

    stats = None
    if roi is not None:
        stats = roi_statistics(roi, large_drops, small_drops, is_fallback)
        x1, y1, x2, y2 = roi
        mode = f"top-{ROI_FALLBACK_TOP_N} petites (fallback)" if is_fallback else "grosses gouttes"
        print(f"\n── ROI ({mode}) ──────────────────────────────────")
        print(f"  Coordonnées  : ({x1}, {y1}) → ({x2}, {y2})")
        print(f"  Dimensions   : {x2 - x1} × {y2 - y1} px")
        label = f"Top-{ROI_FALLBACK_TOP_N} petites" if is_fallback else "Grosses"
        print(f"  {label}      : {stats['n_large']}")
        print(f"  Petites      : {stats['n_small']}")
        print(f"  Total        : {stats['n_total']}")
        print(f"  Surface ROI  : {stats['roi_area_px']:,} px²")
        print(f"  Surface gouttes (sans doublons) : {stats['drop_area_px']:,} px²")
        print(f"  Occupation   : {stats['coverage_pct']:.1f} %")
        print(f"─────────────────────────────────────────────────")
    else:
        print("Aucune goutte disponible → pas de ROI calculée.")

    # ── Annotation & sauvegarde ───────────────────────────────────────────────
    result = annotate_image(img_bgr, large_drops, small_drops,
                            roi, stats, rad_stats, show_labels=show_labels)
    cv2.imwrite(str(output_path), result)
    print(f"\nImage annotée sauvegardée : {output_path}")


# ─── Point d'entrée ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Détecte les gouttes de condensation et analyse une ROI."
    )
    parser.add_argument("image", type=Path, help="Chemin vers l'image source")
    parser.add_argument("--output", type=Path, default=None,
                        help="Chemin de l'image annotée (défaut : <nom>_detected.<ext>)")
    parser.add_argument("--roi-margin", type=int, default=DEFAULT_ROI_MARGIN,
                        help=f"Marge en px autour de la ROI (défaut : {DEFAULT_ROI_MARGIN})")
    parser.add_argument("--no-labels", action="store_true",
                        help="Désactive les numéros d'ID sur les gouttes")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.image.parent / f"{args.image.stem}_detected{args.image.suffix}"

    run_pipeline(args.image, args.output,
                 show_labels=not args.no_labels,
                 roi_margin=args.roi_margin)