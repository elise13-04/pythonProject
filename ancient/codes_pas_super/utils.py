"""
utils.py — utilitaires partagés par tous les modules de tracking.

Contient :
  - load_frames()         : extrait les frames d'une vidéo N&B
  - save_video()          : écrit les frames annotées en fichier vidéo
  - make_color()          : couleur BGR déterministe par ID
  - draw_tracks()         : dessine cercles, IDs et traînes
  - hungarian_match()     : association optimale par algorithme hongrois
  - detect_droplets()     : détection robuste pour gouttes RÉFLECTIVES
                            (anneau lumineux, centre sombre)

Principe de detect_droplets :
  Un flou gaussien fort (sigma≈3) fusionne l'anneau lumineux et le centre
  sombre en un seul blob homogène que LoG détecte comme un maximum local.
  Un NMS (Non-Maximum Suppression) final élimine les doublons.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
from collections import deque
import colorsys

import cv2
import numpy as np
from skimage.feature import blob_log

TRAIL_LEN = 25


# ─────────────────────────── Chargement vidéo ────────────────────────────────

def load_frames(video_path: str | Path,
                fps_extract: float = 1.0,
                max_frames: int | None = None,
                upscale: float = 1.0) -> Tuple[List[np.ndarray], float, float]:
    """
    Charge une vidéo N&B et retourne (frames, src_fps, src_duration).

    fps_extract : frames à extraire par seconde source (1.0 = 1 frame/s)
    upscale     : facteur d'agrandissement avant traitement
    src_duration: durée totale de la vidéo source en secondes

    Pour que la vidéo de sortie ait la même durée que la source :
        frames, src_fps, src_dur = load_frames(...)
        save_video(annotated, out_path, src_duration=src_dur)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir : {video_path}")

    src_fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / src_fps

    step  = 1.0 / fps_extract
    times = np.arange(0.0, duration, step)
    if max_frames is not None:
        times = times[:max_frames]

    frames: List[np.ndarray] = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if upscale != 1.0:
            h, w = gray.shape
            gray = cv2.resize(gray, (int(w*upscale), int(h*upscale)),
                              interpolation=cv2.INTER_LINEAR)
        frames.append(gray)

    cap.release()
    return frames, src_fps, duration


# ─────────────────────────── Couleurs ────────────────────────────────────────

def make_color(track_id: int) -> Tuple[int, int, int]:
    """Couleur BGR déterministe et saturée pour un ID donné."""
    hue = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return (int(b*255), int(g*255), int(r*255))


# ─────────────────────────── Dessin ──────────────────────────────────────────

def draw_tracks(frame_gray: np.ndarray,
                detections: List[dict],
                trails: Dict[int, deque],
                colors: Dict[int, Tuple],
                label: str = "") -> np.ndarray:
    """
    Dessine sur une copie BGR :
      - Un cercle par goutte (rayon = rayon détecté)
      - L'ID au-dessus du centroïde
      - La traîne colorée (historique)
      - Un label de méthode en coin
    """
    out = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

    # Traînes
    for tid, pts in trails.items():
        col = colors.get(tid, (200, 200, 200))
        pt_list = list(pts)
        for k in range(1, len(pt_list)):
            alpha = k / len(pt_list)
            c = tuple(int(v * alpha) for v in col)
            cv2.line(out, pt_list[k-1], pt_list[k], c, 1, cv2.LINE_AA)

    # Cercles + IDs
    for det in detections:
        tid = det.get("id")
        cx, cy = int(det["x"]), int(det["y"])
        r  = max(int(det.get("r", 6)), 3)
        col = colors.get(tid, (0, 255, 100)) if tid is not None else (0, 255, 100)
        cv2.circle(out, (cx, cy), r, col, 1, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 2, col, -1)
        if tid is not None:
            cv2.putText(out, str(tid), (cx + 4, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0),
                        1, cv2.LINE_AA)

    if label:
        cv2.putText(out, label, (4, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180),
                    1, cv2.LINE_AA)
    cv2.putText(out, f"n={len(detections)}", (4, out.shape[0]-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1, cv2.LINE_AA)
    return out


# ─────────────────────────── Export vidéo ────────────────────────────────────

def save_video(frames_bgr: List[np.ndarray],
               output_path: str | Path,
               fps: float = 2.0,
               src_duration: float | None = None) -> None:
    """
    Sauvegarde en MP4.
    Si src_duration est fourni, fps_sortie = len(frames) / src_duration
    → la vidéo de sortie a exactement la même durée que la source.
    """
    if not frames_bgr:
        print("  [save_video] Aucune frame."); return

    out_fps = (len(frames_bgr) / src_duration
               if src_duration and src_duration > 0 else fps)
    out_fps = max(0.5, round(out_fps, 3))

    h, w = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(str(output_path),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             out_fps, (w, h))
    for f in frames_bgr:
        writer.write(f)
    writer.release()
    print(f"  Sauvegardé : {output_path}  "
          f"({len(frames_bgr)} frames @ {out_fps:.2f} fps)")


# ─────────────────────────── Association hongroise ───────────────────────────

def hungarian_match(prev: List[dict], curr: List[dict],
                    max_dist: float = 60.0) -> List[Tuple[int, int]]:
    """Association optimale par algorithme hongrois sur distance euclidienne."""
    from scipy.optimize import linear_sum_assignment
    if not prev or not curr:
        return []
    cost = np.full((len(prev), len(curr)), 1e9)
    for i, p in enumerate(prev):
        for j, c in enumerate(curr):
            d = np.hypot(p["x"]-c["x"], p["y"]-c["y"])
            if d < max_dist:
                cost[i, j] = d
    ri, ci = linear_sum_assignment(cost)
    return [(r, c) for r, c in zip(ri, ci) if cost[r, c] < 1e8]


# ─────────────────────────── Détection centrale ──────────────────────────────

# Paramètres de détection — ajustables selon la résolution de la vidéo
DETECT_CLAHE_CLIP  = 1.5   # force CLAHE (1=doux, 3=fort)
DETECT_CLAHE_GRID  = 4     # taille grille CLAHE
DETECT_BLUR_SIGMA  = 3.0   # flou gaussien — CLEF : fusionne anneau+centre en 1 blob
DETECT_MIN_SIGMA   = 3.0   # sigma LoG min → rayon min ≈ 4 px
DETECT_MAX_SIGMA   = 22.0  # sigma LoG max → rayon max ≈ 31 px
DETECT_NUM_SIGMA   = 10    # nb d'échelles LoG
DETECT_THRESHOLD   = 0.008 # seuil de réponse LoG (bas=plus de détections)
DETECT_OVERLAP     = 0.5   # IoU max entre deux blobs LoG avant suppression
DETECT_NMS_FACTOR  = 0.55  # seuil NMS : fusionne si dist < factor*(r1+r2)

_clahe = cv2.createCLAHE(clipLimit=DETECT_CLAHE_CLIP,
                          tileGridSize=(DETECT_CLAHE_GRID, DETECT_CLAHE_GRID))


def _nms_blobs(blobs: np.ndarray, factor: float = DETECT_NMS_FACTOR) -> np.ndarray:
    """
    Non-Maximum Suppression sur les blobs LoG.
    Si deux blobs sont à distance < factor*(r1+r2), supprime le plus petit.
    Empêche plusieurs cercles sur la même goutte.
    """
    if len(blobs) == 0:
        return blobs
    radii = blobs[:, 2] * np.sqrt(2)
    order = np.argsort(-radii)          # du plus grand au plus petit
    keep  = np.ones(len(blobs), dtype=bool)
    for i, idx in enumerate(order):
        if not keep[idx]:
            continue
        for jdx in order[i+1:]:
            if not keep[jdx]:
                continue
            dist = np.hypot(blobs[idx, 1] - blobs[jdx, 1],
                            blobs[idx, 0] - blobs[jdx, 0])
            if dist < factor * (radii[idx] + radii[jdx]):
                keep[jdx] = False
    return blobs[keep]


def detect_droplets(gray: np.ndarray) -> List[dict]:
    """
    Détecte les gouttes réflectives (anneau lumineux, centre sombre).
    Retourne une liste de dicts : x, y (centroïde), r (rayon en pixels).

    Pipeline :
      1. CLAHE léger  — améliore le contraste local sans écraser le fond
      2. Flou fort    — fusionne l'anneau lumineux et le centre sombre
                        en un blob homogène (c'est la clef du succès)
      3. LoG multi-échelle — détecte les blobs à toutes les tailles
      4. NMS          — élimine les cercles en double sur la même goutte
    """
    eq     = _clahe.apply(gray)
    blur   = cv2.GaussianBlur(eq, (0, 0), sigmaX=DETECT_BLUR_SIGMA)
    blur_f = blur.astype(np.float64) / 255.0

    blobs = blob_log(blur_f,
                     min_sigma=DETECT_MIN_SIGMA,
                     max_sigma=DETECT_MAX_SIGMA,
                     num_sigma=DETECT_NUM_SIGMA,
                     threshold=DETECT_THRESHOLD,
                     overlap=DETECT_OVERLAP)

    if len(blobs) == 0:
        return []

    blobs = _nms_blobs(blobs)

    return [{"x": float(b[1]),
             "y": float(b[0]),
             "r": max(float(b[2] * np.sqrt(2)), 2.0)}
            for b in blobs]