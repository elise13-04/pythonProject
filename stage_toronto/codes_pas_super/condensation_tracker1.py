"""
condensation_tracker1.py
=======================
Pipeline optimisé spécifiquement pour le tracking de gouttes de condensation
avec reflets spéculaires et éclairage hétérogène.

Approche :
  1. CLAHE (Égalisation adaptative) pour normaliser l'éclairage.
  2. Filtre Médian pour réduire le bruit (reflets parasites).
  3. HoughCircles paramétré pour trouver les contours extérieurs.
  4. LapTrack pour l'association temporelle et la détection des fusions (merges).

Usage :
    python condensation_tracker1.py video.mp4
"""

import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import deque, defaultdict

import cv2
import numpy as np
import pandas as pd

# Import depuis ton fichier utils.py existant
from utils import load_frames, save_video, make_color, TRAIL_LEN


def detect_droplets(frames: List[np.ndarray]) -> pd.DataFrame:
    """Détecte les gouttes image par image et retourne un DataFrame pour LapTrack."""
    print("Détection des gouttes en cours...")

    all_detections = []

    for frame_idx, frame in enumerate(frames):
        # 1. Prétraitement beaucoup plus doux
        # On utilise un flou gaussien pour lisser le fond sans détruire les contours des grosses gouttes
        blurred = cv2.GaussianBlur(frame, (9, 9), 2)

        # 2. Détection par Transformée de Hough (PARAMÈTRES STRICTS)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,  # Résolution de l'accumulateur
            minDist=70,  # Empêche la superposition des cercles
            param1=75,  # Sensibilité du détecteur de contours
            param2=32,  # Seuil valid cercles: ps pft mais ps defor.
            minRadius=5,  # Rayon minimal
            maxRadius=72  # Rayon maximal
        )

        if circles is not None:
            circles = (np.round(circles[0, :])
                       .astype("int"))
            for (x, y, r) in circles:
                all_detections.append({
                    "frame": frame_idx,
                    "x": float(x),
                    "y": float(y),
                    "r": float(r)
                })

        if (frame_idx + 1) % 10 == 0:
            print(f"  Frame {frame_idx + 1}/{len(frames)} traitée.")

    return pd.DataFrame(all_detections)

def run_pipeline(video_path: Path, output_path: Path, fps_extract: float):
    # Chargement
    frames, src_fps, src_dur = load_frames(video_path, fps_extract=fps_extract)
    if not frames:
        print("Erreur : Aucune frame chargée.")
        return

    # 1. Détection
    coords_df = detect_droplets(frames)

    if coords_df.empty:
        print("Aucune goutte détectée sur l'ensemble de la vidéo.")
        return

    # 2. Tracking avec LapTrack (gère les fusions !)
    print("\nTracking par LapTrack...")
    try:
        from laptrack import LapTrack
    except ImportError:
        print("Erreur : LapTrack n'est pas installé. Lancez 'pip install laptrack'.")
        return

    # Paramètres LapTrack (on favorise les liaisons courtes et on autorise les fusions)
    lt = LapTrack(
        track_dist_metric="sqeuclidean",
        track_cost_cutoff=400,  # (Distance max ^ 2) entre 2 frames
        gap_closing_dist_metric="sqeuclidean",
        gap_closing_cost_cutoff=400,
        gap_closing_max_frame_count=2, # Retrouve goutte perdue pdt 2 frames
        merging_dist_metric="sqeuclidean",
        merging_cost_cutoff=600,  # ok fusion si distance acceptable
    )

    track_df, split_df, merge_df = lt.predict_dataframe(
        coords_df,
        coordinate_cols=["x", "y"],
        frame_col="frame"
    )

    # 3. Dessin et Annotations
    print("\nGénération de la vidéo annotée...")
    results = []
    trails: Dict[int, deque] = defaultdict(lambda: deque(maxlen=TRAIL_LEN))
    colors: Dict[int, Tuple] = {}

    for pid in track_df["track_id"].unique():
        colors[int(pid)] = make_color(int(pid))

    for i, frame in enumerate(frames):
        out = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frame_data = track_df[track_df["frame"] == i]

        # Dessin des Traînes
        for _, row in frame_data.iterrows():
            pid = int(row["track_id"])
            trails[pid].append((int(row["x"]), int(row["y"])))

        for pid, trail in trails.items():
            if pid not in frame_data["track_id"].values:
                continue  # Ne dessine pas la traîne si la goutte a disparu
            col = colors.get(pid, (200, 200, 200))
            pts = list(trail)
            for k in range(1, len(pts)):
                alpha = k / len(pts)
                c = tuple(int(v * alpha) for v in col)
                cv2.line(out, pts[k - 1], pts[k], c, 1)

        # Dessin des Gouttes
        for _, row in frame_data.iterrows():
            pid = int(row["track_id"])
            cx, cy = int(row["x"]), int(row["y"])
            r = int(row.get("r", 5))
            col = colors.get(pid, (200, 200, 200))

            # Cercle global de la goutte
            cv2.circle(out, (cx, cy), max(r, 4), col, 2, cv2.LINE_AA)
            # ID de la goutte
            cv2.putText(out, str(pid), (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.putText(out, f"Gouttes : {len(frame_data)}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
        results.append(out)

    # Sauvegarde
    save_video(results, output_path, src_duration=src_dur)
    print(f"\nTerminé ! Vidéo sauvegardée sous : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path, help="Chemin vers la vidéo")
    parser.add_argument("--fps-extract", type=float, default=1.0)
    args = parser.parse_args()

    out_file = args.video.parent / f"{args.video.stem}_tracked.mp4"
    run_pipeline(args.video, out_file, args.fps_extract)