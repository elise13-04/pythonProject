"""Fine-resolution detection + tracking on a SHORT segment, to resolve a fast
coalescence that 1 fps misses. Here: vid_2026-06-15_13-08-29, sec 18-21
(real frames 1800-2100) at ~50 fps. Uses the fine-tuned train_tiff model + the same
merge-enabled LapTrack params and corrected merge-frame attribution as _track_all.
Run: .venv-cp3\\Scripts\\python.exe -X utf8 -u _track_segment.py
"""
import os, sys, glob, time
import cv2, numpy as np, pandas as pd
from collections import defaultdict
from scipy import ndimage
from cellpose import models

SRC = "tiff4"
START, STOP, STEP = 1800, 2101, 2      # sec 18-21 @ 100 fps -> every 2nd frame (~50 fps)
DIAM    = 110
OUT_FPS = 10                            # slow-mo playback to see the merge
TAG     = "seg_sec18-21"
OUT     = f"{SRC}_{TAG}_tracked.mp4"
DET_CSV = f"{SRC}_{TAG}_detections.csv"
MRG_CSV = f"{SRC}_{TAG}_merges.csv"
_mdls   = ([m for m in glob.glob("training/train_tiff/models/*") if not m.endswith("_train_losses.npy")]
           or [m for m in glob.glob("train_tiff/models/*") if not m.endswith("_train_losses.npy")])
MODEL   = max(_mdls, key=os.path.getmtime)

import importlib.util
_TDIR = next(d for d in ("stage_toronto", "ancient") if os.path.exists(os.path.join(d, "tracker.py")))
sys.path.insert(0, _TDIR)
spec = importlib.util.spec_from_file_location("tracker", os.path.join(_TDIR, "tracker.py"))
tracker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracker)
tracker.RADIUS_LARGE_LAP = 0
tracker.LAP_LARGE = dict(
    track_dist_metric="sqeuclidean",       track_cost_cutoff=10000,
    gap_closing_dist_metric="sqeuclidean", gap_closing_cost_cutoff=4000,
    gap_closing_max_frame_count=3,
    merging_dist_metric="sqeuclidean",     merging_cost_cutoff=5000,
)

def merge_idx_from(merge_df, track_df):
    idx = defaultdict(list)
    if merge_df is None or merge_df.empty:
        return idx
    for _, row in merge_df.iterrows():
        pa, ch = int(row["parent_track_id"]), int(row["child_track_id"])
        pf = track_df[track_df["track_id"] == pa]
        if pf.empty:
            continue
        fr = int(pf["frame"].max())
        cr = track_df[(track_df["track_id"] == ch) & (track_df["frame"] == fr)]
        if cr.empty:
            cr = pf[pf["frame"] == fr]
        idx[fr].append({"parent": pa, "child": ch,
                        "x": float(cr["x"].values[0]), "y": float(cr["y"].values[0])})
    return idx

sum_labels = getattr(ndimage, "sum_labels", ndimage.sum)
print("MODEL:", MODEL, "| segment", START, STOP - 1, "step", STEP, flush=True)
model = models.CellposeModel(gpu=False, pretrained_model=MODEL)

files = sorted(glob.glob(os.path.join(SRC, "*.tiff")))[START:STOP:STEP]
frames = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in files]
print(f"{len(frames)} frames", flush=True)

rows, t0 = [], time.time()
for fi, gray in enumerate(frames):
    masks = model.eval(gray, diameter=DIAM, channels=[0, 0],
                       flow_threshold=0.6, cellprob_threshold=-1.0)[0]
    nlab = int(masks.max())
    if nlab:
        ids = np.arange(1, nlab + 1)
        ones = np.ones_like(masks, dtype=np.float32)
        areas = sum_labels(ones, masks, ids)
        coms = ndimage.center_of_mass(ones, masks, ids)
        for (cy, cx), a in zip(coms, areas):
            if a > 0:
                rows.append({"frame": fi, "x": float(cx), "y": float(cy),
                             "r": float((a / np.pi) ** 0.5)})
    if (fi + 1) % 20 == 0 or fi + 1 == len(frames):
        print(f"  det {fi+1}/{len(frames)}  ({time.time()-t0:.0f}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv(DET_CSV, index=False)
track_df, merge_df = tracker.run_tracking(df)
midx = merge_idx_from(merge_df, track_df)
ntr = track_df["track_id"].nunique() if not track_df.empty else 0
print(f"{len(df)} det, {ntr} tracks, {len(merge_df)} MERGES", flush=True)
if len(merge_df):
    mrows = [{"frame": fr, "parent": ev["parent"], "child": ev["child"],
              "x": round(ev["x"]), "y": round(ev["y"])}
             for fr in sorted(midx) for ev in midx[fr]]
    pd.DataFrame(mrows).to_csv(MRG_CSV, index=False)
rendered = tracker.render(frames, track_df, midx)
tracker.save_video(rendered, OUT, fps=OUT_FPS)
print(f"done -> {OUT}  ({len(rendered)} frames @ {OUT_FPS} fps)", flush=True)
