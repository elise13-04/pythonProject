"""
droplet_tracker.py
==================
Tracking des gouttes de condensation EN MOUVEMENT.

Pipeline :
  1. Détection multi-passes HoughCircles (blur léger, dp=1.0 pour plus de précision)
       - Passe 1 : grosses gouttes  (r = 30-75 px)
       - Passe 2 : petites gouttes  (r =  5-29 px)
  2. Filtre de mouvement par comparaison inter-frames
  3. laptrack : association temporelle globale

Usage :
    python droplet_tracker.py video.mp4
    python droplet_tracker.py video.mp4 --motion-thr 8    # plus de gouttes
    python droplet_tracker.py video.mp4 --motion-thr 15   # moins de gouttes
"""
from __future__ import annotations

import argparse
import colorsys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import laptrack

# ═════════════════════════════ CONFIG ════════════════════════════════════════

FPS_EXTRACT = 1.0

# Passe 1 — grosses gouttes
LARGE_BLUR    = (9, 9)      # flou léger → contours plus précis
LARGE_SIGMA   = 1
LARGE_DP      = 1.0         # résolution accumulateur max → centres plus précis
LARGE_MIN_DIST= 70
LARGE_P1      = 80
LARGE_P2      = 28
LARGE_MIN_R   = 30
LARGE_MAX_R   = 75

# Passe 2 — petites gouttes
SMALL_DP      = 1.0
SMALL_MIN_DIST= 30
SMALL_P1      = 80
SMALL_P2      = 22
SMALL_MIN_R   = 5
SMALL_MAX_R   = 29
INSIDE_THR    = 0.9         # exclut les petites dans une grosse

# Filtre mouvement
MOTION_THR    = 10.0        # ↑ = moins de gouttes, ↓ = plus

# laptrack
LAP_SEARCH_R  = 80
LAP_GAP_FRAMES= 2
LAP_GAP_R     = 120
MIN_TRACK_LEN = 2

# ═════════════════════════════════════════════════════════════════════════════


def load_frames(path: Path, fps: float) -> Tuple[List[np.ndarray], float, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    src_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / src_fps
    step     = 1.0 / fps
    frames   = []
    for t in np.arange(0.0, duration, step) + step / 2.0:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, f = cap.read()
        if not ok: continue
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f)
    cap.release()
    return frames, src_fps, duration


def detect(gray: np.ndarray) -> List[dict]:
    """Détection multi-passes avec flou léger et dp=1.0 pour la précision."""
    blurred = cv2.GaussianBlur(gray, LARGE_BLUR, LARGE_SIGMA)

    # Passe 1 : grosses
    large = []
    c = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT,
                          dp=LARGE_DP, minDist=LARGE_MIN_DIST,
                          param1=LARGE_P1, param2=LARGE_P2,
                          minRadius=LARGE_MIN_R, maxRadius=LARGE_MAX_R)
    if c is not None:
        for x, y, r in np.round(c[0]).astype(int):
            large.append({"x": float(x), "y": float(y), "r": float(r)})

    # Passe 2 : petites (hors intérieur des grosses)
    small = []
    c = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT,
                          dp=SMALL_DP, minDist=SMALL_MIN_DIST,
                          param1=SMALL_P1, param2=SMALL_P2,
                          minRadius=SMALL_MIN_R, maxRadius=SMALL_MAX_R)
    if c is not None:
        for x, y, r in np.round(c[0]).astype(int):
            inside = any(((x-d["x"])**2 + (y-d["y"])**2)**0.5 < d["r"] * INSIDE_THR
                         for d in large)
            if not inside:
                small.append({"x": float(x), "y": float(y), "r": float(r)})

    return large + small


def motion_score(curr, prev, nxt, cx, cy, r) -> float:
    H, W = curr.shape
    x0, x1 = max(0, cx-r), min(W, cx+r)
    y0, y1 = max(0, cy-r), min(H, cy+r)
    roi = curr[y0:y1, x0:x1].astype(np.float32)
    s = 0.0
    if prev is not None:
        s = max(s, np.abs(roi - prev[y0:y1, x0:x1].astype(np.float32)).mean())
    if nxt is not None:
        s = max(s, np.abs(roi - nxt[y0:y1, x0:x1].astype(np.float32)).mean())
    return s


def _color(tid: int) -> Tuple[int, int, int]:
    h = (tid * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.9, 1.0)
    return (int(b*255), int(g*255), int(r*255))


def run(video_path: Path, output_path: Path, motion_thr: float = MOTION_THR):

    print(f"Vidéo : {video_path.name}")
    frames, src_fps, duration = load_frames(video_path, FPS_EXTRACT)
    n = len(frames)
    print(f"  {n} frames  ({duration:.1f}s)\n")

    # ── Détection + filtre mouvement ──────────────────────────────────────────
    print(f"Détection (seuil mouvement={motion_thr})...")
    rows = []
    for fi, gray in enumerate(frames):
        prev = frames[fi-1] if fi > 0     else None
        nxt  = frames[fi+1] if fi < n-1   else None
        dets = detect(gray)
        for d in dets:
            if motion_score(gray, prev, nxt,
                            int(d["x"]), int(d["y"]), int(d["r"])) > motion_thr:
                rows.append({"frame": fi, **d})
        print(f"  [{fi+1}/{n}]  total={len(dets):3d}  "
              f"mobiles={sum(1 for d in dets if motion_score(gray,prev,nxt,int(d['x']),int(d['y']),int(d['r']))>motion_thr):3d}")

    print(f"\n  {len(rows)} détections mobiles\n")

    # ── laptrack ──────────────────────────────────────────────────────────────
    colors: Dict[int, Tuple] = {}
    track_df = pd.DataFrame(columns=["frame","x","y","r","track_id"])

    if rows:
        print("Tracking laptrack...")
        lt = laptrack.LapTrack(
            track_dist_metric       = "sqeuclidean",
            track_cost_cutoff       = LAP_SEARCH_R**2,
            gap_closing_dist_metric = "sqeuclidean",
            gap_closing_cost_cutoff = LAP_GAP_R**2,
            gap_closing_max_frame_count = LAP_GAP_FRAMES,
        )
        track_df, _, _ = lt.predict_dataframe(
            pd.DataFrame(rows),
            coordinate_cols=["x","y"], frame_col="frame",
            only_coordinate_cols=False)

        # Filtre pistes trop courtes
        valid = set(track_df.groupby("track_id").size()
                    .pipe(lambda s: s[s >= MIN_TRACK_LEN]).index)
        track_df = track_df[track_df["track_id"].isin(valid)]

        for tid in track_df["track_id"].unique():
            colors[int(tid)] = _color(int(tid))
        print(f"  {track_df['track_id'].nunique()} pistes\n")

    # ── Rendu ─────────────────────────────────────────────────────────────────
    print("Rendu vidéo...")
    h, w = frames[0].shape
    writer = cv2.VideoWriter(str(output_path),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             max(0.5, n/duration), (w, h))

    for fi, gray in enumerate(frames):
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        fd  = track_df[track_df["frame"] == fi] if len(track_df) else pd.DataFrame()

        # Cercles + IDs uniquement (pas de traînes)
        n_vis = 0
        for _, row in fd.iterrows():
            tid = int(row["track_id"])
            cx, cy = int(row["x"]), int(row["y"])
            r   = max(int(row["r"]), 3)
            col = colors.get(tid, (200, 200, 200))
            cv2.circle(out, (cx, cy), r,   col, 2, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), 4,   col, -1)
            cv2.putText(out, str(tid), (cx + r + 3, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0),
                        1, cv2.LINE_AA)
            n_vis += 1

        # HUD
        for i, line in enumerate([f"Frame  : {fi+1}/{n}", f"Gouttes: {n_vis}"]):
            cv2.putText(out, line, (12, 28 + i*26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220),
                        1, cv2.LINE_AA)
        writer.write(out)

    writer.release()
    print(f"Terminé  →  {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video",         type=Path)
    ap.add_argument("--output",      type=Path, default=None)
    ap.add_argument("--motion-thr",  type=float, default=MOTION_THR,
                    help=f"Seuil mouvement (défaut:{MOTION_THR})")
    args = ap.parse_args()
    out = args.output or args.video.parent / f"{args.video.stem}_tracked.mp4"
    run(args.video, out, args.motion_thr)

if __name__ == "__main__":
    main()