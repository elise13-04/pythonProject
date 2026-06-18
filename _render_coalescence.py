"""Radius-aware coalescence detection + annotation, with IDENTITY RULE:
the merged droplet always inherits the ID of its LARGEST parent (the biggest of the
continuing track + the absorbed droplets, measured just before the merge).
Run: .venv-cp3\\Scripts\\python.exe -X utf8 -u _render_coalescence.py <vid_folder>
"""
import os, sys, glob
import cv2, pandas as pd
from collections import defaultdict

VID = sys.argv[1] if len(sys.argv) > 1 else "tiff4"
STEP, OUT_FPS = 100, 1
CONTAIN, MIN_CHILD_R, GROW, FLICKER = 1.0, 35, 1.05, 2  # FLICKER: reject if parent reappears within N frames
DET_CSV = next((c for c in (f"{VID}_tracked_detections.csv", f"{VID}_detections.csv")
                if os.path.exists(c)), f"{VID}_tracked_detections.csv")
OUT     = f"{VID}_coalescence.mp4"

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

def detect_and_relabel(track_df):
    """One ascending pass: detect absorptions (radius-aware) and relabel the merged
    droplet to its largest parent's id. Returns (relabeled track_df, merge_idx)."""
    tr = track_df.copy()
    merges = defaultdict(list)
    frames = sorted(tr["frame"].unique())
    # anti-flicker: position where each NEW track is born -> reject a parent that merely
    # flickered out and restarted nearby within FLICKER frames (detection blip, not a merge)
    born = defaultdict(list)
    for tid, g in tr.groupby("track_id"):
        g0 = g.sort_values("frame").iloc[0]
        born[int(g0.frame)].append((float(g0.x), float(g0.y)))
    def flickered(ex, ey, er, t):
        for f in range(t + 1, t + 1 + FLICKER):
            for bx, by in born.get(f, []):
                if (ex - bx) ** 2 + (ey - by) ** 2 <= er * er:
                    return True
        return False
    for i in range(len(frames) - 1):
        t, tn = frames[i], frames[i + 1]
        cur, nxt = tr[tr.frame == t], tr[tr.frame == tn]
        ending = set(cur.track_id) - set(nxt.track_id)
        if not ending:
            continue
        nxt_rows = list(nxt.itertuples())
        absorbed = defaultdict(list)                      # child_id -> [parent rows]
        for e in cur[cur.track_id.isin(ending)].itertuples():
            if flickered(e.x, e.y, e.r, t):               # parent reappears soon -> blip, not a merge
                continue
            best = None
            for c in nxt_rows:
                if c.r < MIN_CHILD_R or c.r < GROW * e.r:
                    continue
                d = ((e.x - c.x) ** 2 + (e.y - c.y) ** 2) ** 0.5
                if d <= c.r + CONTAIN * e.r and (best is None or d < best[0]):  # parent disk overlaps child disk
                    best = (d, c.track_id)
            if best is not None:
                absorbed[best[1]].append(e)
        for child_id, parents in absorbed.items():
            crow = nxt[nxt.track_id == child_id].iloc[0]
            child_prev = cur[cur.track_id == child_id]
            # constituents = continuing child (if it pre-existed) + absorbed parents
            consts = [(int(p.track_id), float(p.r)) for p in parents]
            if len(child_prev):
                consts.append((int(child_id), float(child_prev.iloc[0].r)))
            winner = max(consts, key=lambda kv: kv[1])[0]
            for p in parents:
                merges[tn].append({"parent": int(p.track_id), "child": int(winner),
                                   "x": float(crow.x), "y": float(crow.y)})
            if winner != child_id:                        # merged droplet takes largest parent's id
                tr.loc[(tr.track_id == child_id) & (tr.frame >= tn), "track_id"] = winner
    return tr, merges

df = pd.read_csv(DET_CSV)
track_df, _ = tracker.run_tracking(df)
track_df, midx = detect_and_relabel(track_df)
n_events = sum(len(v) for v in midx.values())
print(f"{VID}: {n_events} absorptions sur {len(midx)} frames", flush=True)
for fr in sorted(midx):
    winners = sorted(set(e["child"] for e in midx[fr]))
    print(f"  frame {fr}: {len(midx[fr])} absorbees -> goutte {winners} (id du plus gros parent)", flush=True)

files = sorted(glob.glob(os.path.join(VID, "*.tiff")))[::STEP]
frames = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in files]
rendered = tracker.render(frames, track_df, midx)

# --- overlay: origin trails of ALL absorbed parents, converging into the merged droplet ---
traj = {}
for tid, g in track_df.groupby("track_id"):
    traj[int(tid)] = {int(r.frame): (int(r.x), int(r.y)) for r in g.itertuples()}
edges = [(fm, e["parent"], e["child"]) for fm in midx for e in midx[fm] if e["parent"] != e["child"]]
for i in range(len(rendered)):
    img = rendered[i]
    for fm, p, w in edges:
        if i < fm or i not in traj.get(w, {}):
            continue
        ppath = [traj[p][f] for f in sorted(traj.get(p, {})) if f <= fm]
        if not ppath:
            continue
        col = tracker.make_color(p)
        for k in range(1, len(ppath)):                       # parent's origin trajectory
            cv2.line(img, ppath[k - 1], ppath[k], col, 1, cv2.LINE_AA)
        cv2.line(img, ppath[-1], traj[w][i], col, 1, cv2.LINE_AA)   # converge to merged droplet
        cv2.circle(img, ppath[0], 3, col, -1, cv2.LINE_AA)         # origin marker
print(f"{len(edges)} segments d'origine ajoutes", flush=True)

tracker.save_video(rendered, OUT, fps=OUT_FPS)
print(f"done -> {OUT}", flush=True)
