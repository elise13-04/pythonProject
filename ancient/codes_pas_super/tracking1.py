"""
droplet_tracker.py
==================
Prend une vidéo de gouttelettes en entrée et produit une vidéo annotée avec :
  - Contours de chaque goutte détectée
  - ID numérique de chaque goutte
  - Trajectoires des gouttes en mouvement (traîne colorée)
  - Indicateurs visuels de fusion (MERGE) et séparation (SPLIT)
  - HUD : nombre de gouttes, frame courante, FPS de traitement

Usage:
    python droplet_tracker.py video.mp4
    python droplet_tracker.py video.mp4 --output result.mp4 --debug-frames

Paramètres ajustables dans le bloc CONFIG ou via CLI (--help).

Pipeline par frame :
    1. Extraction 1 frame/seconde
    2. Niveaux de gris + CLAHE (égalisation locale du contraste)
    3. Flou bilatéral (préserve les bords)
    4. Soustraction de fond MOG2
    5. Seuillage adaptatif + morphologie (opening/closing)
    6. Watershed pour séparer les gouttes jointives
    7. Filtrage par aire et circularité (anti faux-positifs)
    8. Tracking IoU + distance centroïde entre frames
    9. Détection fusions/séparations par comparaison des labels
   10. Dessin : contours, IDs, trajectoires, événements
"""

from __future__ import annotations

import argparse
import colorsys
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage import morphology, segmentation, measure, filters
from skimage.feature import peak_local_max


# ═══════════════════════════════ CONFIG ══════════════════════════════════════

# ── Extraction ───────────────────────────────────────────────────────────────
FRAMES_PER_SEC   = 1       # combien de frames extraire par seconde de vidéo
                            # 1 = une par seconde, 5 = cinq par seconde, etc.

# ── Pré-traitement ───────────────────────────────────────────────────────────
CLAHE_CLIP       = 3.0     # force de l'égalisation locale du contraste (1=doux, 8=fort)
CLAHE_GRID       = 8       # taille de la grille CLAHE (8x8 tuiles)
BILATERAL_D      = 9       # diamètre du filtre bilatéral (préserve les bords)
BILATERAL_SIGMA  = 75      # sigma couleur et espace du bilatéral

# ── Soustraction de fond MOG2 ─────────────────────────────────────────────────
MOG2_HISTORY     = 50      # nb de frames mémorisées pour le modèle de fond
MOG2_THRESHOLD   = 40      # sensibilité (bas = détecte plus, haut = moins de bruit)
MOG2_SHADOWS     = False   # True pour détecter les ombres (gris dans le masque)

# ── Segmentation ─────────────────────────────────────────────────────────────
OPEN_RADIUS      = 2       # érosion morphologique (supprime les points isolés)
CLOSE_RADIUS     = 3       # fermeture morphologique (comble les trous internes)
MIN_AREA         = 80      # aire minimale d'une goutte en pixels
MAX_AREA         = 40_000  # aire maximale (évite de détecter le fond entier)
MIN_CIRCULARITY  = 0.20    # circularité minimale : 4π×aire/périmètre² (1=cercle parfait)
                            # 0.20 accepte les formes allongées, 0.65 = quasi-circulaire
WATERSHED_SIGMA  = 2.0     # sigma du flou pour les graines watershed

# ── Tracking ─────────────────────────────────────────────────────────────────
MAX_DIST         = 80      # distance max (pixels) entre deux frames pour relier une goutte
IOU_WEIGHT       = 0.6     # poids de l'IoU vs distance dans le coût d'association (0–1)
MAX_LOST         = 3       # nb de frames consécutives perdues avant de supprimer une piste
MIN_HITS         = 2       # nb de frames consécutives vues avant d'afficher une piste
TRAIL_LENGTH     = 30      # nb de positions mémorisées pour la traîne

# ── Visualisation ─────────────────────────────────────────────────────────────
CONTOUR_COLOR    = (0,   255, 100)   # BGR : vert menthe pour les contours
TEXT_COLOR       = (255, 255,   0)   # BGR : jaune pour les IDs
TRAIL_FADE       = True    # True = la traîne s'estompe vers le passé
SHOW_BBOX        = False   # True = affiche le rectangle englobant
EVENT_DURATION   = 8       # nb de frames pendant lesquelles un événement reste affiché
OUTPUT_FPS       = 2       # FPS de la vidéo de sortie (doit correspondre à FRAMES_PER_SEC)

# ═════════════════════════════════════════════════════════════════════════════


# ─────────────────────────── Structures de données ───────────────────────────

def _make_color(track_id: int) -> Tuple[int, int, int]:
    """Génère une couleur BGR déterministe et bien saturée pour un ID donné."""
    hue = (track_id * 0.618033988749895) % 1.0   # suite de Fibonacci dorée
    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


@dataclass
class Droplet:
    """Une goutte détectée dans une frame."""
    label:       int
    centroid:    Tuple[float, float]   # (x, y) en pixels
    area:        float
    bbox:        Tuple[int, int, int, int]   # (r_min, c_min, r_max, c_max)
    contour_pts: np.ndarray            # points du contour OpenCV
    mask:        np.ndarray            # masque booléen dans le bbox


@dataclass
class Track:
    """Une piste de suivi pour une goutte à travers le temps."""
    track_id:   int
    color:      Tuple[int, int, int]
    trail:      deque = field(default_factory=lambda: deque(maxlen=TRAIL_LENGTH))
    hits:       int   = 0    # frames consécutives détectées
    lost:       int   = 0    # frames consécutives perdues
    last_drop:  Optional[Droplet] = None
    active:     bool  = True

    def update(self, drop: Droplet):
        self.trail.append((int(drop.centroid[0]), int(drop.centroid[1])))
        self.hits  += 1
        self.lost   = 0
        self.last_drop = drop

    def mark_lost(self):
        self.lost += 1
        if self.lost >= MAX_LOST:
            self.active = False


@dataclass
class Event:
    """Un événement de fusion ou séparation."""
    kind:     str    # "MERGE" ou "SPLIT"
    position: Tuple[int, int]
    frame_idx: int


# ──────────────────────────── Prétraitement ──────────────────────────────────

def preprocess(frame_bgr: np.ndarray,
               clahe: cv2.CLAHE) -> np.ndarray:
    """
    Gris → CLAHE → bilatéral → retourne image uint8 nettoyée.
    Le bilatéral lisse le bruit tout en préservant les bords des gouttes.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    eq   = clahe.apply(gray)
    bil  = cv2.bilateralFilter(eq, BILATERAL_D, BILATERAL_SIGMA, BILATERAL_SIGMA)
    return bil


# ──────────────────────────── Segmentation ───────────────────────────────────

def segment(preprocessed: np.ndarray,
            fg_mask: np.ndarray) -> List[Droplet]:
    """
    Segmente les gouttes dans une image prétraitée.

    Étapes :
      1. Seuillage adaptatif sur l'image complète
      2. AND avec le masque avant-plan MOG2
      3. Morphologie (opening + closing)
      4. Watershed pour séparer les gouttes jointives
      5. Filtrage aire + circularité
    """
    # ── Seuillage adaptatif ──────────────────────────────────────────────────
    # THRESH_BINARY_INV car fond sombre → on veut les zones claires (gouttes)
    thr = cv2.adaptiveThreshold(
        preprocessed, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31, C=-4
    )

    # ── Fusion avec le masque MOG2 ────────────────────────────────────────────
    # Le masque MOG2 vaut 255 sur les zones qui ont changé depuis le fond modélisé.
    # On prend l'union pour ne rater aucune goutte (même statique partiellement).
    fg_clean = cv2.dilate(fg_mask, np.ones((3, 3), np.uint8), iterations=1)
    combined = cv2.bitwise_or(thr, fg_clean)

    # ── Morphologie ──────────────────────────────────────────────────────────
    se_open  = morphology.disk(OPEN_RADIUS)
    se_close = morphology.disk(CLOSE_RADIUS)
    bw = combined > 0
    bw = morphology.opening(bw, se_open)
    bw = morphology.closing(bw, se_close)
    bw = morphology.remove_small_objects(bw, min_size=MIN_AREA)

    # ── Watershed ────────────────────────────────────────────────────────────
    dist  = ndi.distance_transform_edt(bw)
    sigma = WATERSHED_SIGMA
    dist_s = filters.gaussian(dist, sigma=sigma)
    min_d  = max(4, int(np.sqrt(MIN_AREA / np.pi) * 0.6))
    coords = peak_local_max(dist_s, min_distance=min_d, labels=bw)
    seeds  = np.zeros(dist.shape, dtype=bool)
    if coords.size:
        seeds[tuple(coords.T)] = True
    markers, _ = ndi.label(seeds)
    labels_ws   = segmentation.watershed(-dist_s, markers, mask=bw, compactness=0.05)

    # ── Extraction des régions + filtrage ────────────────────────────────────
    droplets: List[Droplet] = []
    props = measure.regionprops(labels_ws)
    for rp in props:
        area = rp.area
        if area < MIN_AREA or area > MAX_AREA:
            continue
        # circularité = 4π × aire / périmètre²
        perim = rp.perimeter
        if perim < 1:
            continue
        circ = (4 * np.pi * area) / (perim ** 2)
        if circ < MIN_CIRCULARITY:
            continue

        # Contour OpenCV à partir du masque de la région
        mask_full = (labels_ws == rp.label).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue

        cy, cx = rp.centroid
        r0, c0, r1, c1 = rp.bbox
        region_mask = rp.image   # masque booléen dans le bbox

        droplets.append(Droplet(
            label       = rp.label,
            centroid    = (float(cx), float(cy)),
            area        = float(area),
            bbox        = (r0, c0, r1, c1),
            contour_pts = cnts[0],
            mask        = region_mask,
        ))
    return droplets


# ──────────────────────────── Tracking ───────────────────────────────────────

def _iou(d1: Droplet, d2: Droplet, h: int, w: int) -> float:
    """IoU approximée via masques dans les bboxes complètes."""
    m1 = np.zeros((h, w), dtype=bool)
    m2 = np.zeros((h, w), dtype=bool)
    r0, c0, r1, c1 = d1.bbox
    m1[r0:r1, c0:c1] = d1.mask
    r0, c0, r1, c1 = d2.bbox
    m2[r0:r1, c0:c1] = d2.mask
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter) / float(union + 1e-9)


def associate(tracks: Dict[int, Track],
              detections: List[Droplet],
              frame_shape: Tuple[int, int]) -> Tuple[
                  Dict[int, int],   # track_id → det_index
                  List[int],        # det_index non associés (nouvelles gouttes)
                  List[int]]:       # track_id non associés (gouttes perdues)
    """
    Associe les pistes existantes aux nouvelles détections via
    l'algorithme hongrois (linear_sum_assignment) sur une matrice de coût
    combinant distance euclidienne + IoU.
    """
    active_ids  = [tid for tid, tr in tracks.items() if tr.active]
    if not active_ids or not detections:
        return {}, list(range(len(detections))), active_ids

    h, w = frame_shape
    n_tr = len(active_ids)
    n_dt = len(detections)
    cost = np.full((n_tr, n_dt), fill_value=1e9)

    for i, tid in enumerate(active_ids):
        tr = tracks[tid]
        if tr.last_drop is None:
            continue
        cx0, cy0 = tr.last_drop.centroid
        for j, det in enumerate(detections):
            cx1, cy1 = det.centroid
            dist_cost = np.hypot(cx1 - cx0, cy1 - cy0)
            if dist_cost > MAX_DIST:
                continue
            iou_score = _iou(tr.last_drop, det, h, w)
            cost[i, j] = (1 - IOU_WEIGHT) * dist_cost + IOU_WEIGHT * (1 - iou_score) * MAX_DIST

    row_ind, col_ind = linear_sum_assignment(cost)

    matched:   Dict[int, int] = {}
    unmatched_tr  = set(active_ids)
    unmatched_det = set(range(n_dt))

    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < 1e8:
            tid = active_ids[r]
            matched[tid] = c
            unmatched_tr.discard(tid)
            unmatched_det.discard(c)

    return matched, list(unmatched_det), list(unmatched_tr)


def detect_events(prev_drops: List[Droplet],
                  curr_drops: List[Droplet],
                  matched: Dict[int, int],
                  det_idx_to_new_id: Dict[int, int],
                  frame_idx: int) -> List[Event]:
    """
    Détecte les fusions et séparations :
      - MERGE : plusieurs pistes convergent vers une seule détection
      - SPLIT  : une piste donne naissance à plusieurs détections
    """
    events: List[Event] = []

    # combien de tracks pointent vers chaque détection ?
    det_count: Dict[int, List[int]] = defaultdict(list)
    for tid, di in matched.items():
        det_count[di].append(tid)

    # combien de détections reçoit chaque track ?
    # (une track ne peut recevoir qu'une détection, les splits viennent des
    #  nouvelles détections non assignées proches d'une track)
    # MERGE : N tracks → 1 det
    for di, tids in det_count.items():
        if len(tids) > 1 and di < len(curr_drops):
            cx, cy = curr_drops[di].centroid
            events.append(Event("MERGE", (int(cx), int(cy)), frame_idx))

    # SPLIT : 1 track perdue + N nouvelles détections à proximité
    # (heuristique simple : nouvelle détection proche d'une track perdue)
    return events


# ──────────────────────────── Dessin ─────────────────────────────────────────

def draw_frame(frame_bgr: np.ndarray,
               drops: List[Droplet],
               tracks: Dict[int, Track],
               det_to_track: Dict[int, int],
               events: List[Event],
               frame_idx: int,
               active_events: List[Event]) -> np.ndarray:
    """Dessine contours, IDs, trajectoires et événements sur la frame."""
    out = frame_bgr.copy()

    # ── Trajectoires (traînes) ────────────────────────────────────────────────
    for tr in tracks.values():
        if not tr.active or tr.hits < MIN_HITS:
            continue
        pts = list(tr.trail)
        if len(pts) < 2:
            continue
        for k in range(1, len(pts)):
            if TRAIL_FADE:
                alpha = k / len(pts)
                col   = tuple(int(c * alpha) for c in tr.color)
            else:
                col = tr.color
            cv2.line(out, pts[k - 1], pts[k], col, thickness=2,
                     lineType=cv2.LINE_AA)

    # ── Contours + IDs ────────────────────────────────────────────────────────
    for j, drop in enumerate(drops):
        tid = det_to_track.get(j)
        if tid is None:
            color_c = CONTOUR_COLOR
            label_s = "?"
        else:
            tr = tracks[tid]
            if tr.hits < MIN_HITS:
                continue    # piste trop jeune → on n'affiche pas encore
            color_c = tr.color
            label_s = str(tid)

        cv2.drawContours(out, [drop.contour_pts], -1, color_c, 2,
                         lineType=cv2.LINE_AA)

        cx, cy = int(drop.centroid[0]), int(drop.centroid[1])
        cv2.circle(out, (cx, cy), 3, color_c, -1)

        if SHOW_BBOX:
            r0, c0, r1, c1 = drop.bbox
            cv2.rectangle(out, (c0, r0), (c1, r1), color_c, 1)

        # ID au-dessus du centroïde
        cv2.putText(out, label_s, (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR,
                    1, cv2.LINE_AA)

    # ── Événements actifs ─────────────────────────────────────────────────────
    for ev in active_events:
        age   = frame_idx - ev.frame_idx
        alpha = max(0.0, 1.0 - age / EVENT_DURATION)
        cx, cy = ev.position
        e_color = (0, 80, 255) if ev.kind == "MERGE" else (255, 140, 0)
        e_color = tuple(int(c * alpha) for c in e_color)
        cv2.putText(out, ev.kind, (cx + 8, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, e_color,
                    2, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 12, e_color, 2, lineType=cv2.LINE_AA)

    # ── HUD (coin supérieur gauche) ───────────────────────────────────────────
    n_visible = sum(1 for tr in tracks.values() if tr.active and tr.hits >= MIN_HITS)
    hud_lines = [
        f"Frame : {frame_idx}",
        f"Gouttes: {n_visible}",
    ]
    for i, line in enumerate(hud_lines):
        cv2.putText(out, line, (10, 22 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200),
                    1, cv2.LINE_AA)

    return out


# ──────────────────────────── Pipeline principal ──────────────────────────────

def run(video_path: Path,
        output_path: Path,
        debug_frames: bool = False):

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir : {video_path}")

    src_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_nframes= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = src_nframes / src_fps

    print(f"Vidéo source  : {video_path.name}")
    print(f"Résolution    : {src_width}×{src_height}  |  {src_fps:.1f} FPS")
    print(f"Durée         : {duration_s:.1f} s  ({src_nframes} frames)")

    # ── Outils de traitement ──────────────────────────────────────────────────
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP,
        tileGridSize=(CLAHE_GRID, CLAHE_GRID)
    )
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=MOG2_THRESHOLD,
        detectShadows=MOG2_SHADOWS,
    )

    # ── Vidéo de sortie ───────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path), fourcc, OUTPUT_FPS,
        (src_width, src_height)
    )

    # ── Dossier debug ─────────────────────────────────────────────────────────
    dbg_dir = None
    if debug_frames:
        dbg_dir = output_path.parent / f"{output_path.stem}_debug"
        dbg_dir.mkdir(exist_ok=True)

    # ── État du tracker ───────────────────────────────────────────────────────
    tracks:        Dict[int, Track] = {}
    next_id:       int  = 1
    all_events:    List[Event] = []
    prev_drops:    List[Droplet] = []
    frame_idx:     int  = 0

    # Calcule les timestamps à extraire (1 frame tous les 1/FRAMES_PER_SEC secondes)
    step_sec = 1.0 / FRAMES_PER_SEC
    timestamps = np.arange(0, duration_s, step_sec)
    total = len(timestamps)
    print(f"Frames à traiter : {total}  (1 frame / {step_sec:.2f} s)")

    for k, t in enumerate(timestamps):
        # ── Lecture de la frame ───────────────────────────────────────────────
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue

        # ── Prétraitement ─────────────────────────────────────────────────────
        prep = preprocess(frame, clahe)

        # Le MOG2 est "entraîné" sur l'image prétraitée en niveaux de gris
        # On lui passe la version 3 canaux pour compatibilité
        prep3 = cv2.cvtColor(prep, cv2.COLOR_GRAY2BGR)
        fg_mask = bg_sub.apply(prep3)
        # fg_mask : 255 = avant-plan, 127 = ombre, 0 = fond
        fg_mask = (fg_mask == 255).astype(np.uint8) * 255

        # ── Segmentation ──────────────────────────────────────────────────────
        drops = segment(prep, fg_mask)

        # ── Association pistes / détections ───────────────────────────────────
        matched, new_det_idx, lost_ids = associate(
            tracks, drops, (src_height, src_width)
        )

        # Dictionnaire détection_index → track_id pour le dessin
        det_to_track: Dict[int, int] = {}

        # Mise à jour des pistes existantes
        for tid, di in matched.items():
            tracks[tid].update(drops[di])
            det_to_track[di] = tid

        # Pistes perdues
        for tid in lost_ids:
            tracks[tid].mark_lost()

        # Nouvelles pistes
        for di in new_det_idx:
            new_track = Track(
                track_id = next_id,
                color    = _make_color(next_id),
            )
            new_track.update(drops[di])
            tracks[next_id] = new_track
            det_to_track[di] = next_id
            next_id += 1

        # ── Détection d'événements ────────────────────────────────────────────
        new_events = detect_events(
            prev_drops, drops, matched, det_to_track, frame_idx
        )
        all_events.extend(new_events)

        # Événements encore à afficher (fenêtre glissante)
        active_ev = [ev for ev in all_events
                     if frame_idx - ev.frame_idx < EVENT_DURATION]

        # ── Dessin ────────────────────────────────────────────────────────────
        annotated = draw_frame(
            frame, drops, tracks, det_to_track,
            new_events, frame_idx, active_ev
        )

        writer.write(annotated)

        # ── Debug frames ──────────────────────────────────────────────────────
        if debug_frames and dbg_dir is not None:
            # Sauvegarde côte à côte : prétraitée | masque FG | annotée
            prep_bgr = cv2.cvtColor(prep, cv2.COLOR_GRAY2BGR)
            fg_bgr   = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
            panel    = np.hstack([prep_bgr, fg_bgr, annotated])
            cv2.imwrite(str(dbg_dir / f"frame_{frame_idx:04d}.png"), panel)

        prev_drops = drops
        frame_idx += 1

        if True:
            n_active = sum(1 for tr in tracks.values() if tr.active)
            print(f"  [{k+1:4d}/{total}]  t={t:6.1f}s  "
                  f"détections={len(drops):3d}  pistes actives={n_active:3d}")

    cap.release()
    writer.release()

    n_merged = sum(1 for ev in all_events if ev.kind == "MERGE")
    n_split  = sum(1 for ev in all_events if ev.kind == "SPLIT")
    print(f"\nTerminé.")
    print(f"  Pistes créées   : {next_id - 1}")
    print(f"  Fusions détect. : {n_merged}")
    print(f"  Séparations     : {n_split}")
    print(f"  Vidéo de sortie : {output_path}")
    if debug_frames:
        print(f"  Frames debug    : {dbg_dir}")


# ──────────────────────────── CLI ────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Droplet tracker — vidéo annotée avec contours et trajectoires."
    )
    ap.add_argument("video", type=Path,
                    help="stage_toronto\video.mp4")
    ap.add_argument("--output", type=Path, default=None,
                    help="Chemin de la vidéo de sortie (défaut: <stem>_tracked.mp4)")
    ap.add_argument("--fps-extract", type=float, default=FRAMES_PER_SEC,
                    help=f"Frames extraites par seconde (défaut: {FRAMES_PER_SEC})")
    ap.add_argument("--min-area", type=int, default=MIN_AREA,
                    help=f"Aire minimale d'une goutte en pixels (défaut: {MIN_AREA})")
    ap.add_argument("--max-dist", type=int, default=MAX_DIST,
                    help=f"Distance max de tracking entre frames (défaut: {MAX_DIST})")
    ap.add_argument("--min-circ", type=float, default=MIN_CIRCULARITY,
                    help=f"Circularité minimale 0-1 (défaut: {MIN_CIRCULARITY})")
    ap.add_argument("--debug-frames", action="store_true",
                    help="Sauvegarde les panneaux de debug par frame")
    args = ap.parse_args()

    out = args.output or args.video.parent / f"{args.video.stem}_tracked.mp4"
    run(args.video, out, debug_frames=args.debug_frames)


if __name__ == "__main__":
    main()