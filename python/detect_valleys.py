"""
detect_valleys.py — valley detection on a motion-fraction trace.

Python analogue of run_valleys_from_txt.m. Same method:
    - Savitzky-Golay smoothing
    - Adaptive baseline = moving median (default 20 s)
    - Threshold = baseline - K * MAD(residual)
    - Remove dips shorter than MinValleyDur
    - Merge dips separated by gaps shorter than MergeGap
    - Report START of each surviving valley

Run:
    python detect_valleys.py file_motion_fraction.txt
    python detect_valleys.py file.txt --mad-factor 0.5 --base-window 30

Outputs to stdout: a table of (Start_s, End_s, Duration_s).
Also pops up a diagnostic plot unless --no-plot is given.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


@dataclass
class ValleyParams:
    sg_order: int = 3            # Savitzky-Golay polynomial order
    sg_window_sec: float = 5.0   # Savitzky-Golay window (s)
    base_window_sec: float = 20  # moving-median baseline window (s)
    min_valley_dur_sec: float = 5
    merge_gap_sec: float = 3
    mad_factor: float = 0.3      # threshold = baseline - mad_factor * MAD


# --- helpers ---------------------------------------------------------------
def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def remove_short_segments(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Drop runs of True shorter than min_len samples."""
    out = mask.copy()
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return out
    breaks = np.concatenate([[0], np.flatnonzero(np.diff(idx) > 1) + 1, [idx.size]])
    for k in range(len(breaks) - 1):
        block = idx[breaks[k]:breaks[k + 1]]
        if block.size < min_len:
            out[block] = False
    return out


def merge_small_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill False gaps shorter than max_gap between True runs."""
    out = mask.copy()
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return out
    d = np.diff(idx)
    for i, di in enumerate(d):
        if di > 1 and (di - 1) <= max_gap:
            out[idx[i]:idx[i + 1] + 1] = True
    return out


def moving_median_shrink(x: np.ndarray, window: int) -> np.ndarray:
    """
    Centered moving median that 'shrinks' the window at the edges,
    matching MATLAB's movmedian(..., 'Endpoints', 'shrink').
    """
    return (pd.Series(x)
            .rolling(window=window, center=True, min_periods=1)
            .median()
            .to_numpy())


# --- main detector ---------------------------------------------------------
def detect_valleys(t: np.ndarray, mf: np.ndarray,
                   p: ValleyParams | None = None):
    p = p or ValleyParams()
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Could not infer a positive sampling period from t.")

    # 1. Savitzky-Golay smoothing
    sg_win = _odd(max(3, int(round(p.sg_window_sec / dt))))
    if sg_win <= p.sg_order:
        sg_win = _odd(p.sg_order + 2)
    mf_s = savgol_filter(mf, window_length=sg_win, polyorder=p.sg_order)

    # 2. Adaptive baseline (moving median)
    base_win = _odd(max(3, int(round(p.base_window_sec / dt))))
    baseline = moving_median_shrink(mf_s, base_win)

    # 3. MAD-based threshold (residual = baseline - smoothed signal)
    resid = baseline - mf_s
    med_r = np.median(resid)
    mad = 1.4826 * np.median(np.abs(resid - med_r)) + np.finfo(float).eps
    thr = baseline - p.mad_factor * mad
    valley_mask = mf_s <= thr

    # 4. Remove short dips & merge nearby ones
    min_len = max(1, int(round(p.min_valley_dur_sec / dt)))
    merge_gap = max(1, int(round(p.merge_gap_sec / dt)))
    valley_mask = remove_short_segments(valley_mask, min_len)
    valley_mask = merge_small_gaps(valley_mask, merge_gap)

    # 5. Extract intervals + start/center indices
    intervals, beg_idx, center_idx = [], [], []
    idx = np.flatnonzero(valley_mask)
    if idx.size:
        breaks = np.concatenate([[0],
                                 np.flatnonzero(np.diff(idx) > 1) + 1,
                                 [idx.size]])
        for k in range(len(breaks) - 1):
            block = idx[breaks[k]:breaks[k + 1]]
            s, e = block[0], block[-1]
            t_s, t_e = float(t[s]), float(t[e])
            dur = t_e - t_s
            if dur < p.min_valley_dur_sec:
                continue
            intervals.append((t_s, t_e, dur))
            beg_idx.append(s)
            center_idx.append(int(block[np.argmin(mf_s[block])]))

    return {
        "mf_smooth": mf_s,
        "baseline": baseline,
        "threshold": thr,
        "valley_mask": valley_mask,
        "intervals": intervals,            # list of (start_s, end_s, dur_s)
        "begin_idx": np.asarray(beg_idx, dtype=int),
        "center_idx": np.asarray(center_idx, dtype=int),
    }


# --- plotting --------------------------------------------------------------
def plot_valleys(t, mf, result, title="Valley detection (baseline - MAD method)"):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(t, mf, color=(0.8, 0.8, 0.8), label="MF raw")
    ax.plot(t, result["mf_smooth"], "b", lw=1.2, label="MF smoothed")
    ax.plot(t, result["baseline"], "r--", lw=1.2, label="Baseline")
    ax.plot(t, result["threshold"], "k-.", lw=1.0, label="Baseline - MAD")

    yl = ax.get_ylim()
    for s, e, _ in result["intervals"]:
        ax.axvspan(s, e, color=(0.9, 0.95, 1.0), alpha=0.35, lw=0)

    if result["begin_idx"].size:
        ax.plot(t[result["begin_idx"]], result["mf_smooth"][result["begin_idx"]],
                "g<", markerfacecolor="g", label="Valley start")
    if result["center_idx"].size:
        ax.plot(t[result["center_idx"]], result["mf_smooth"][result["center_idx"]],
                "kv", markerfacecolor="k", label="Valley center")

    ax.set_ylim(yl)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Motion fraction")
    ax.set_title(title)
    ax.grid(True); ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig


# --- CLI -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("txt", type=Path,
                    help="*_motion_fraction.txt produced by cycle.py / cycle.m")
    ap.add_argument("--sg-order", type=int, default=3)
    ap.add_argument("--sg-window", type=float, default=5.0,
                    help="Savitzky-Golay window length (seconds)")
    ap.add_argument("--base-window", type=float, default=20.0,
                    help="moving-median baseline window (seconds)")
    ap.add_argument("--min-dur", type=float, default=5.0,
                    help="minimum valley duration (seconds)")
    ap.add_argument("--merge-gap", type=float, default=3.0,
                    help="merge valleys separated by gaps shorter than this (s)")
    ap.add_argument("--mad-factor", type=float, default=0.3,
                    help="threshold = baseline - mad_factor * MAD (try 0.3-0.8)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--save-plot", type=Path, default=None)
    ap.add_argument("--save-csv", type=Path, default=None,
                    help="if given, save the valleys table as CSV")
    args = ap.parse_args()

    data = np.loadtxt(args.txt, skiprows=1)
    t, mf = data[:, 0], data[:, 1]

    p = ValleyParams(sg_order=args.sg_order,
                     sg_window_sec=args.sg_window,
                     base_window_sec=args.base_window,
                     min_valley_dur_sec=args.min_dur,
                     merge_gap_sec=args.merge_gap,
                     mad_factor=args.mad_factor)
    result = detect_valleys(t, mf, p)

    print("\nValleys detected with baseline - MAD method:")
    if not result["intervals"]:
        print("No valleys detected.")
    else:
        df = pd.DataFrame(result["intervals"],
                          columns=["Start_s", "End_s", "Duration_s"])
        print(df.to_string(index=False))
        if args.save_csv:
            df.to_csv(args.save_csv, index=False)
            print(f"\nSaved -> {args.save_csv}")

    if not args.no_plot:
        fig = plot_valleys(t, mf, result)
        if args.save_plot:
            fig.savefig(args.save_plot, dpi=150)
            print(f"Plot saved -> {args.save_plot}")
        plt.show()


if __name__ == "__main__":
    main()
