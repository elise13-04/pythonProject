"""
cycle.py
Python analogue of cycle.m. Edit the CONFIG block, then run:
    python cycle.py

Outputs are written next to the input video, using ROI_LABEL in the filename:
    <stem>_<ROI_LABEL>_motion_metrics_per_second.csv
    <stem>_<ROI_LABEL>_non_shedding_periods.csv
    <stem>_<ROI_LABEL>_motion_fraction_plot.png
    <stem>_<ROI_LABEL>_motion_fraction.txt
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.morphology import disk, opening, dilation, remove_small_objects


# =============================== CONFIG ===================================
# Path to the input video. Resolved relative to this script's parent folder,
# so it works regardless of where you launch python from.
VIDEO_PATH = Path(__file__).resolve().parent.parent / "script_test.mp4"

# Region of interest, as fractions of frame width and height.
# Fractions make the same script work for any video resolution.
#
# Other examples
#   Side view, right half (matches cycle.m default)
#       ROI_X_FRAC = (0.5, 1.0); ROI_Y_FRAC = (0.0, 1.0); ROI_LABEL = "righthalf"
#   Side view, left half
#       ROI_X_FRAC = (0.0, 0.5); ROI_Y_FRAC = (0.0, 1.0); ROI_LABEL = "lefthalf"
#   Birdseye, centered crop covering the middle 60 percent
#       ROI_X_FRAC = (0.20, 0.80); ROI_Y_FRAC = (0.20, 0.80); ROI_LABEL = "topview_center"

#on prend tt l'image (région d'intérêt)
ROI_X_FRAC = (0.0, 1.0)
ROI_Y_FRAC = (0.0, 1.0)
ROI_LABEL  = "topview"

# Si c'est vrai, script sauvegarde des images pour chaque seconde analysée
DEBUG = True

# Detection parameters, tuned for top-down (birdseye) recordings of a
# droplet field. The MATLAB side-view defaults (MIN_AREA_FRACTION=8e-5,
# OPEN_RADIUS=2) were too sensitive here, because every static droplet's
# bright highlight twinkled from sensor noise and codec artifacts and got
# counted as motion. The values below suppress those twinkles while
# preserving connected, real shedding events.
BLUR_SIGMA        = 1.0 #ecart-type du flou
MAD_K             = 3.5 #sensibilité du seuil de detection
MIN_AREA_FRACTION = 5e-4   # taille min raise further (1e-3, 2e-3) if speckle remains
OPEN_RADIUS       = 3      # erodes single-droplet specks
DILATE_RADIUS     = 2
TEMPORAL_WINDOW   = 3
SMOOTH_REQUIRE    = 2
# ==========================================================================


@dataclass
class Params:
    blur_sigma: float = 1.0
    mad_k: float = 3.5
    min_area_fraction: float = 8e-5
    open_radius: int = 2
    dilate_radius: int = 2
    temporal_window: int = 3
    smooth_require: int = 2


def robust_mad(x: np.ndarray) -> float: #calcule Median Absolute Deviation mesure de dispersion
    x = x.astype(np.float64).ravel() #aplatit en 1D convert en float64
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med)) + np.finfo(float).eps


def robust_z(x: np.ndarray) -> np.ndarray: #cmb de mads chaque pixel s'écarte de la médiane
    x = x.astype(np.float64)
    med = np.median(x)
    return (x - med) / (robust_mad(x))


def detect_motion(curr_gray: np.ndarray, #deux frames consécutives
                  prev_gray: np.ndarray, #en niveaux de gris, floutées
                  p: Params,
                  se_open: np.ndarray,
                  se_dil: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    curr_z = robust_z(curr_gray) #normalise chaque frame par son propre z-score robuste
    prev_z = robust_z(prev_gray) #pour rendre la détection insensible aux changements globaux de lum
    d = np.abs(curr_z - prev_z) #différence absolue= carte de diff brute

    d_med = np.median(d)
    d_mad = 1.4826 * np.median(np.abs(d - d_med)) + np.finfo(float).eps
    thr = d_med + p.mad_k * d_mad
    bw = d > thr
    #masque booléen: Calcule un seuil adaptatif sur la carte de différence.
    # Les pixels dont la différence dépasse ce seuil sont marqués comme "en mouvement"
    if p.open_radius > 0:
        bw = opening(bw, se_open) #Erosion: Supp petits objets isolés sans trp affecter grds objets.
    if p.dilate_radius > 0:
        bw = dilation(bw, se_dil) #Dilatation : élargit légèrement les zones détectées pour combler
        # les trous à l'intérieur des objets en mouvement.
    min_area_px = max(1, int(round(p.min_area_fraction * bw.size)))
    if min_area_px > 1: #dernier filtre anti-bruit
        bw = remove_small_objects(bw, min_size=min_area_px)

    mf = bw.sum() / bw.size #Motion fraction: proportion de pixels en mouv
    return mf, d, bw


def process_video(video_path: Path,
                  roi_x_frac: tuple[float, float],
                  roi_y_frac: tuple[float, float],
                  roi_label: str,
                  params: Params,
                  debug: bool = False) -> dict:

    p = params
    cap = cv2.VideoCapture(str(video_path)) #ouvre vidéo avec OpenCV
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video, {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0 #frames (images) par seconde
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #nbr total de frames
    duration = n_frames / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#Convertit les fractions ROI en pixels réels,entre 0 et la taille de l'image garantit qu'on ne sort jamais du cadre.
    x1 = max(0, min(width,  int(round(roi_x_frac[0] * width))))
    x2 = max(0, min(width,  int(round(roi_x_frac[1] * width))))
    y1 = max(0, min(height, int(round(roi_y_frac[0] * height))))
    y2 = max(0, min(height, int(round(roi_y_frac[1] * height))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Empty ROI from fractions x={roi_x_frac}, y={roi_y_frac}")
    suffix = f"_{roi_label}"

    seconds = np.arange(0, int(np.floor(duration)) + 1) #liste des instants analysés
    n_s = len(seconds)
    sample_sec = np.zeros(n_s, dtype=int)
    motion_fraction = np.zeros(n_s, dtype=float)
    moving_raw = np.zeros(n_s, dtype=bool)

    se_open = disk(p.open_radius)
    se_dil = disk(p.dilate_radius)

    if debug:
        plt.ion()
        fig_dbg, axes = plt.subplots(2, 3, figsize=(12, 7))

    prev_gray: np.ndarray | None = None

    #NV
    debug_frame=[]

    for k, t in enumerate(seconds): #boucle principale
        safe_t = min(max(0.0, t + 0.5), max(0.0, duration - 1.0 / max(1.0, fps)))
        cap.set(cv2.CAP_PROP_POS_MSEC, safe_t * 1000.0) #met en millisecondes
        ok, frame = cap.read() #Pour chaque scd t, on se positionne à t + 0.5 s
        # (milieu de la seconde) pour avoir une frame représentative.
        if not ok:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                sample_sec[k] = t
                continue

        gray = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convertit en niveaux de gris
                if frame.ndim == 3 else frame)
        gray = gray[y1:y2, x1:x2] #decoupe la roi avec le slicing numpy

        gray_blurred = cv2.GaussianBlur(gray, (0, 0),         #applique flou gaussien
                                        sigmaX=p.blur_sigma, #OpenCV calcule la taille
                                        sigmaY=p.blur_sigma) #du noyau avec sigmaX

#NV
        if debug: #si debug=True, accumule toutes les données intermédiaires  dans une liste,
            debug_frame.append({  #(frame brute, grise, floutée, carte de différence, masque)
                "t":t,              #puis les sauvegarde toutes en images PNG après la boucle.
                "frame_rgb":cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).copy(),
                "gray":gray.copy(),              #Utile pour comprendre pourquoi la détection
                "gray_blurred":gray_blurred.copy(),         #fonctionne ou non.
                "d":None,
                "bw":None,
                "mf":None,})
        if prev_gray is None: #la 1ere frame ne peut pas être
            sample_sec[k] = t #comparée, donc on la sauvegarde
            prev_gray = gray_blurred #et on passe à la suivante
            if debug:
                plt.pause(0.001)
            continue
        mf, d, bw = detect_motion(gray_blurred, prev_gray, p, se_open, se_dil)
        sample_sec[k] = t #appelle la detection
        motion_fraction[k] = mf #enregistre si la seconde a du mouv (true/false)
        moving_raw[k] = mf > 0
        prev_gray = gray_blurred #met a jour la frame precedente
        if debug:
            debug_frame[-1]["d"]=d.copy()
            debug_frame[-1]["bw"]=bw.copy()
            debug_frame[-1]["mf"]=mf

#NV
    cap.release()
    if debug:
        plt.ioff()
        plt.close(fig_dbg)
        # --- Sauvegarde de toutes les frames debug ---
        save_dir = Path(__file__).resolve().parent / "debug_frame"
        save_dir.mkdir(exist_ok=True)

        print(f"\nSauvegarde de {len(debug_frame)} frames debug "
              f"dans {save_dir} ...")

        for i, f in enumerate(debug_frame):
            fig_save, axes_save = plt.subplots(2, 3, figsize=(13, 8))

            for ax in axes_save.ravel():
                ax.set_xticks([]); ax.set_yticks([])

            axes_save[0, 0].imshow(f["frame_rgb"])
            axes_save[0, 0].set_title(f"Raw frame (sec {f['t']})")
            axes_save[0, 1].imshow(f["gray"], cmap="gray")
            axes_save[0, 1].set_title("Gray + cropped ROI")
            axes_save[0, 2].imshow(f["gray_blurred"], cmap="gray")
            axes_save[0, 2].set_title("Blurred ROI")

            if f["d"] is not None:
                axes_save[1, 0].imshow(f["d"], cmap="magma")
                axes_save[1, 0].set_title("|Z_curr - Z_prev|")
                d_med = np.median(f["d"])
                d_mad = (1.4826 * np.median(np.abs(f["d"] - d_med))
                         + np.finfo(float).eps)
                axes_save[1, 1].imshow(f["d"] > (d_med + p.mad_k * d_mad),
                                       cmap="gray")
                axes_save[1, 1].set_title("Thresholded mask")
                axes_save[1, 2].imshow(f["bw"], cmap="gray")
                axes_save[1, 2].set_title(f"Clean mask  mf={f['mf']:.4f}")
            else:
                axes_save[1, 0].set_title("(first frame — no diff)")

            fig_save.suptitle(f"Frame {i + 1} / {len(debug_frame)}"
                              f"  —  sec {f['t']}", fontsize=12)
            fig_save.tight_layout()

            out_path = save_dir / f"frame_{i + 1:04d}_sec{int(f['t']):04d}.png"
            fig_save.savefig(out_path, dpi=120)
            plt.close(fig_save)

            if (i + 1) % 10 == 0 or (i + 1) == len(debug_frame):
                print(f"  {i + 1}/{len(debug_frame)} saved...")

        print(f"Finish. Images in : {save_dir}")
#lissage temporel
    sum_win = np.convolve(moving_raw.astype(float), #np.convolve avec np.ones(3)
                          np.ones(p.temporal_window), #calcule une somme glissante sur 3 secondes.
                          mode="same") #Si ≥ 2 des 3 secondes consécutives sont "en mouvement" la seconde centrale est confirmée comme "en mouvement"
    moving = sum_win >= p.smooth_require #Élimine les faux positifs isolés.

    intervals = []
    in_run, run_start = False, 0
    for k in range(n_s): #Détection des intervalles sans mouv
        if not moving[k] and not in_run: #detecte les plages continues de non mouv
            in_run = True #signifie qu'on compte une période calme
            run_start = sample_sec[k]
        is_last = (k == n_s - 1)
        if in_run and (moving[k] or is_last):
            run_end = sample_sec[k]
            if moving[k] and not is_last:
                run_end -= 1
            if run_end >= run_start: #qd le mouv reprend, on ferme l'intervalle
                intervals.append((run_start, run_end, run_end - run_start + 1))
            in_run = False

    folder = video_path.parent if video_path.parent.as_posix() else Path.cwd()
    stem = video_path.stem
#Exports: le script écrit 4 fichiers
    # CSV métriques : seconde, motion_fraction, moving_raw, moving_smooth
    out_metrics   = folder / f"{stem}{suffix}_motion_metrics_per_second.csv"
    # CSV intervalles calmes : start_sec, end_sec, duration_sec
    out_intervals = folder / f"{stem}{suffix}_non_shedding_periods.csv"
    # PNG : graphique de la motion fraction avec les zones en mouvement surlignées en bleu clair
    out_plot      = folder / f"{stem}{suffix}_motion_fraction_plot.png"
    # TXT : les mêmes données que le CSV métriques, en format tabulaire
    out_txt       = folder / f"{stem}{suffix}_motion_fraction.txt"

    pd.DataFrame({
        "second": sample_sec,
        "motion_fraction": motion_fraction,
        "moving_raw": moving_raw,
        "moving_smooth": moving,
    }).to_csv(out_metrics, index=False)

    if intervals:
        pd.DataFrame(intervals,
                     columns=["start_sec", "end_sec", "duration_sec"]
                     ).to_csv(out_intervals, index=False)
    else:
        pd.DataFrame(columns=["start_sec", "end_sec", "duration_sec"]
                     ).to_csv(out_intervals, index=False)

    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(sample_sec, motion_fraction, lw=1.2)
    yl = ax.get_ylim()
    in_move, s0 = False, np.nan
    for k in range(n_s):
        if moving[k] and not in_move:
            in_move, s0 = True, sample_sec[k]
        is_last = (k == n_s - 1)
        if in_move and (not moving[k] or is_last):
            e0 = sample_sec[k]
            if not moving[k]:
                e0 -= 1
            ax.axvspan(s0, e0, color=(0.85, 0.95, 1.0), alpha=0.25, lw=0)
            in_move = False
    ax.plot(sample_sec, motion_fraction, lw=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Motion fraction ({roi_label})")
    ax.set_title(f"Per-second motion estimate ({roi_label})")
    ax.grid(True); ax.set_ylim(yl)
    fig.tight_layout()
    fig.savefig(out_plot, dpi=150)
    plt.close(fig)

    with open(out_txt, "w") as f:
        f.write("Second\tMotionFraction\n")
        for s, m in zip(sample_sec, motion_fraction):
            f.write(f"{int(s)}\t{m:.6f}\n")

    print(f"Video, {video_path}")
    print(f"ROI '{roi_label}' (x={x1}..{x2} of {width}, y={y1}..{y2} of {height})")
    print(f"Per-second metrics   -> {out_metrics}")
    print(f"Non-shedding periods -> {out_intervals}")
    print(f"Plot                 -> {out_plot}")
    print(f"Results saved as TXT -> {out_txt}")

    return {
        "sample_sec": sample_sec,
        "motion_fraction": motion_fraction,
        "moving_raw": moving_raw,
        "moving_smooth": moving,
        "intervals": intervals,
        "fps": fps,
        "duration": duration,
        "roi": (x1, x2, y1, y2),
    }


def main():
    params = Params(
        blur_sigma=BLUR_SIGMA,
        mad_k=MAD_K,
        min_area_fraction=MIN_AREA_FRACTION,
        open_radius=OPEN_RADIUS,
        dilate_radius=DILATE_RADIUS,
        temporal_window=TEMPORAL_WINDOW,
        smooth_require=SMOOTH_REQUIRE,
    )
    process_video(
        video_path=VIDEO_PATH,
        roi_x_frac=ROI_X_FRAC,
        roi_y_frac=ROI_Y_FRAC,
        roi_label=ROI_LABEL,
        params=params,
        debug=DEBUG,
    )


if __name__ == "__main__":
    main()
