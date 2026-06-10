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
#
# Active setting, birdseye full frame
ROI_X_FRAC = (0.0, 1.0)
ROI_Y_FRAC = (0.0, 1.0)
ROI_LABEL  = "topview"

# Show per-frame diagnostic panels while running (slow).
DEBUG = True

# Detection parameters, tuned for top-down (birdseye) recordings of a
# droplet field. The MATLAB side-view defaults (MIN_AREA_FRACTION=8e-5,
# OPEN_RADIUS=2) were too sensitive here, because every static droplet's
# bright highlight twinkled from sensor noise and codec artifacts and got
# counted as motion. The values below suppress those twinkles while
# preserving connected, real shedding events.
BLUR_SIGMA        = 1.0
MAD_K             = 3.5
MIN_AREA_FRACTION = 5e-4   # raise further (1e-3, 2e-3) if speckle remains
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


def robust_mad(x: np.ndarray) -> float:
    x = x.astype(np.float64).ravel()
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med)) + np.finfo(float).eps


def robust_z(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    med = np.median(x)
    return (x - med) / (robust_mad(x))


def detect_motion(curr_gray: np.ndarray,
                  prev_gray: np.ndarray,
                  p: Params,
                  se_open: np.ndarray,
                  se_dil: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    curr_z = robust_z(curr_gray)
    prev_z = robust_z(prev_gray)
    d = np.abs(curr_z - prev_z)

    d_med = np.median(d)
    d_mad = 1.4826 * np.median(np.abs(d - d_med)) + np.finfo(float).eps
    thr = d_med + p.mad_k * d_mad
    bw = d > thr

    if p.open_radius > 0:
        bw = opening(bw, se_open)
    if p.dilate_radius > 0:
        bw = dilation(bw, se_dil)

    min_area_px = max(1, int(round(p.min_area_fraction * bw.size)))
    if min_area_px > 1:
        bw = remove_small_objects(bw, min_size=min_area_px)

    mf = bw.sum() / bw.size
    return mf, d, bw


def process_video(video_path: Path,
                  roi_x_frac: tuple[float, float],
                  roi_y_frac: tuple[float, float],
                  roi_label: str,
                  params: Params,
                  debug: bool = False) -> dict:

    p = params
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video, {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    x1 = max(0, min(width,  int(round(roi_x_frac[0] * width))))
    x2 = max(0, min(width,  int(round(roi_x_frac[1] * width))))
    y1 = max(0, min(height, int(round(roi_y_frac[0] * height))))
    y2 = max(0, min(height, int(round(roi_y_frac[1] * height))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Empty ROI from fractions x={roi_x_frac}, y={roi_y_frac}")
    suffix = f"_{roi_label}"

    seconds = np.arange(0, int(np.floor(duration)) + 1)
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

    for k, t in enumerate(seconds):
        safe_t = min(max(0.0, t + 0.5), max(0.0, duration - 1.0 / max(1.0, fps)))
        cap.set(cv2.CAP_PROP_POS_MSEC, safe_t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                sample_sec[k] = t
                continue

        gray = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if frame.ndim == 3 else frame)
        gray = gray[y1:y2, x1:x2]

        gray_blurred = cv2.GaussianBlur(gray, (0, 0),
                                        sigmaX=p.blur_sigma,
                                        sigmaY=p.blur_sigma)

        if debug:
            for ax in axes.ravel():
                ax.cla()
            axes[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title(f"Raw frame (sec {t})")
            axes[0, 1].imshow(gray, cmap="gray")
            axes[0, 1].set_title("Gray + cropped ROI")
            axes[0, 2].imshow(gray_blurred, cmap="gray")
            axes[0, 2].set_title("Blurred ROI")

        if prev_gray is None:
            sample_sec[k] = t
            prev_gray = gray_blurred
            if debug:
                plt.pause(0.001)
            continue

        mf, d, bw = detect_motion(gray_blurred, prev_gray, p, se_open, se_dil)
        sample_sec[k] = t
        motion_fraction[k] = mf
        moving_raw[k] = mf > 0
        prev_gray = gray_blurred

        if debug:
            axes[1, 0].imshow(d, cmap="magma")
            axes[1, 0].set_title("|Z_curr - Z_prev|")
            d_med = np.median(d)
            d_mad = 1.4826 * np.median(np.abs(d - d_med)) + np.finfo(float).eps
            axes[1, 1].imshow(d > (d_med + p.mad_k * d_mad), cmap="gray")
            axes[1, 1].set_title("Thresholded mask")
            axes[1, 2].imshow(bw, cmap="gray")
            axes[1, 2].set_title(f"Clean mask, mf={mf:.4f}")
            for ax in axes.ravel():
                ax.set_xticks([]); ax.set_yticks([])
            plt.pause(0.001)

    cap.release()
    if debug:
        plt.ioff()

    sum_win = np.convolve(moving_raw.astype(float),
                          np.ones(p.temporal_window),
                          mode="same")
    moving = sum_win >= p.smooth_require

    intervals = []
    in_run, run_start = False, 0
    for k in range(n_s):
        if not moving[k] and not in_run:
            in_run = True
            run_start = sample_sec[k]
        is_last = (k == n_s - 1)
        if in_run and (moving[k] or is_last):
            run_end = sample_sec[k]
            if moving[k] and not is_last:
                run_end -= 1
            if run_end >= run_start:
                intervals.append((run_start, run_end, run_end - run_start + 1))
            in_run = False

    folder = video_path.parent if video_path.parent.as_posix() else Path.cwd()
    stem = video_path.stem

    out_metrics   = folder / f"{stem}{suffix}_motion_metrics_per_second.csv"
    out_intervals = folder / f"{stem}{suffix}_non_shedding_periods.csv"
    out_plot      = folder / f"{stem}{suffix}_motion_fraction_plot.png"
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
