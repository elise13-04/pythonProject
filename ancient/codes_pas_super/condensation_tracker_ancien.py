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

MIN_DIST_SMALL_LARGE = 1.3

def detect_droplets(frames: List[np.ndarray]) -> pd.DataFrame:
    """Détecte les gouttes avec une approche Multi-Pass (Grosses puis Petites)."""
    print("Détection des gouttes en cours (Multi-Pass)...")

    all_detections = []

    clahe=cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

    for frame_idx, frame in enumerate(frames):
        enhanced= clahe.apply(frame)
        blurred = cv2.GaussianBlur(frame, (9, 9), 2)
        frame_dets = []

        # ---------------------------------------------------------
        # PASSE 1 : Les grosses gouttes (Tes paramètres)
        # ---------------------------------------------------------
        circles_large = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=170, # Assez grand pour éviter doublons sur les grosses
            param1=75,
            param2=34,  # Ton réglage strict
            minRadius=30,  # On cherche uniquement les moyennes/grosses
            maxRadius=50)

        large_centers = []
        if circles_large is not None:
            circles_large = np.round(circles_large[0, :]).astype("int")
            for (x, y, r) in circles_large:
                frame_dets.append({"frame": frame_idx, "x": float(x),
                                   "y": float(y), "r": float(r)})
                large_centers.append((x, y, r))

        # ---------------------------------------------------------
        # PASSE 2 : Les petites gouttes
        # ---------------------------------------------------------
        circles_small = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=30,  # TRÈS PETIT : autorise les petites gouttes à être proches
            param1=75,
            param2=30,  # Un peu plus bas car les petits cercles ont moins de pixels pour "voter"
            minRadius=15,  # Taille minimale
            maxRadius=29)  # Taille juste en dessous de la passe 1

        if circles_small is not None:
            circles_small = np.round(circles_small[0, :]).astype("int")
            for (x, y, r) in circles_small:
                # Vérification cruciale : cette petite goutte est-elle DANS une grosse ?
                is_inside_large = False
                for (lx, ly, lr) in large_centers:
                    # Distance entre le centre de la petite et de la grosse (Pythagore)
                    dist_centers = ((x - lx) ** 2 + (y - ly) ** 2) ** 0.5
                    # Si le centre de la petite est englobé dans le rayon de la grosse
                    if dist_centers < (lr * MIN_DIST_SMALL_LARGE):
                        is_inside_large = True
                        break

                if not is_inside_large:   # Si elle n'est pas dans une grosse, on la valide !
                    frame_dets.append({"frame": frame_idx, "x": float(x), "y": float(y), "r": float(r)})

        all_detections.extend(frame_dets)

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
    # Remplace le bloc LapTrack dans run_pipeline par ceci :

    coords_large = coords_df[coords_df["r"] >= 40].copy().reset_index(drop=True)
    coords_small = coords_df[coords_df["r"] < 40].copy().reset_index(drop=True)

    track_parts = []
    offset = 0

    # --- Grosses gouttes ---
    if len(coords_large) > 0:
        lt_large = LapTrack(
            track_dist_metric="sqeuclidean",
            track_cost_cutoff=18000,
            gap_closing_dist_metric="sqeuclidean",
            gap_closing_cost_cutoff=1000,
            gap_closing_max_frame_count=3,
            merging_dist_metric="sqeuclidean",
            merging_cost_cutoff=600,
        )
        track_large, _, merge_large = lt_large.predict_dataframe(
            coords_large, coordinate_cols=["x", "y"], frame_col="frame"
        )
        offset = track_large["track_id"].max() + 1
        track_parts.append(track_large)
    else:
        print("Aucune grosse goutte détectée.")
        merge_large = None

    # --- Petites gouttes ---
    if len(coords_small) > 0:
        lt_small = LapTrack(
            track_dist_metric="sqeuclidean",
            track_cost_cutoff=900,
            gap_closing_dist_metric="sqeuclidean",
            gap_closing_cost_cutoff=900,
            gap_closing_max_frame_count=2,
        )
        track_small, _, _ = lt_small.predict_dataframe(
            coords_small, coordinate_cols=["x", "y"], frame_col="frame"
        )
        track_small = track_small.copy()
        track_small["track_id"] += offset
        track_parts.append(track_small)
    else:
        print("Aucune petite goutte détectée.")

    # --- Fusion des résultats ---
    if not track_parts:
        print("Aucune goutte trackée. Abandon.")
        return

    track_df = pd.concat(track_parts, ignore_index=True)

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