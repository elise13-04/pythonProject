"""
tracker_hough.py  ―  Détection HoughCircles + filtre de couverture sombre
=========================================================================
Pipeline :
  1. Normalisation du fond (division par Gaussienne 201px) → supprime le zigzag.
  2. HoughCircles en 3 passes (S/M/L) sur l'image normalisée.
  3. NMS inter-passes (les grands cercles ont priorité).
  4. Filtre de couverture sombre : rejette les cercles dont l'intérieur
     n'est PAS majoritairement sombre.  Élimine les grands faux cercles
     qui entourent plusieurs gouttes ou épousent le fond.
  5. Suivi LapTrack + CSV fusions + flash visuel sur les merges.

Dépendances :
    pip install opencv-python numpy pandas laptrack

Usage :
    python tracker_hough.py video.mp4
    python tracker_hough.py video.mp4 --fps-extract 5

Paramètres à ajuster (haut du fichier) :
    HOUGH_PASSES       → param2 pour contrôler sensibilité / faux positifs
    MIN_DARK_COVERAGE  → seuil de couverture sombre (0–1)
    NMS_OVERLAP        → aggressivité de la suppression des doublons

Sorties :
    <video>_hough.mp4
    <video>_hough_merges.csv
"""

import argparse, csv
from collections import defaultdict, deque
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
import pandas as pd

try:
    from laptrack import LapTrack
    HAS_LAPTRACK = True
except ImportError:
    HAS_LAPTRACK = False
    print("⚠  pip install laptrack  pour activer le suivi")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PARAMÈTRES — ajuster ici selon la vidéo
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Normalisation du fond
BG_KERNEL = 201          # doit être > diamètre max des gouttes (px, pleine résolution)

# HoughCircles — 3 passes sur IMAGE PLEINE RÉSOLUTION
# Colonnes : blur_k  dp   minDist  param1  param2  minR  maxR  label
# ↑ param2 = moins de cercles (moins de FP) ; ↓ param2 = plus de cercles
HOUGH_PASSES = [
    (  9, 1.2,   18,   60,   20,    5,   22,  "S"),
    ( 13, 1.2,   32,   62,   22,   18,   44,  "M"),
    ( 21, 1.3,   64,   65,   26,   36,   64,  "L"),
]

# Suppression des doublons inter-passes
NMS_OVERLAP = 0.45       # 0 = pas de suppression, 0.5 = standard

# Filtre de couverture sombre : un vrai cercle de goutte doit avoir
# son intérieur majoritairement sombre (corps de la goutte).
# Les grands faux cercles (plusieurs gouttes à l'intérieur) ont peu de surface sombre.
DARK_THRESHOLD      = 108   # valeur en dessous = "sombre" (sur image normalisée 0-255)
MIN_DARK_COVERAGE   = 0.32  # petits cercles  (< LARGE_R_THRESH px)
MIN_DARK_COVERAGE_L = 0.42  # grands cercles  (≥ LARGE_R_THRESH px)
LARGE_R_THRESH      = 36    # px pleine résolution

# LapTrack
RADIUS_LARGE_LAP = 36
LAP_LARGE = dict(
    track_dist_metric="sqeuclidean", track_cost_cutoff=20000,
    gap_closing_dist_metric="sqeuclidean", gap_closing_cost_cutoff=1800,
    gap_closing_max_frame_count=3,
    merging_dist_metric="sqeuclidean", merging_cost_cutoff=900,
)
LAP_SMALL = dict(
    track_dist_metric="sqeuclidean", track_cost_cutoff=1200,
    gap_closing_dist_metric="sqeuclidean", gap_closing_cost_cutoff=1200,
    gap_closing_max_frame_count=2,
)

TRAIL_LEN       = 30
MERGE_FLASH_LEN = 14
MERGE_LOG_ROWS  = 6
SIZE_CATS = [
    (64, "XL", (0,   0, 220)),
    (36, "L",  (0, 100, 255)),
    (18, "M",  (0, 200, 180)),
    ( 0, "S",  (150, 220, 0)),
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  UTILITAIRES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_color(seed: int) -> Tuple[int,int,int]:
    rng = np.random.default_rng(seed * 6364136223846793005 & 0xFFFFFFFF)
    hsv = np.uint8([[[rng.integers(0,180),rng.integers(150,255),rng.integers(180,255)]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0,0]
    return (int(bgr[0]),int(bgr[1]),int(bgr[2]))


def size_label(r: float) -> Tuple[str, Tuple]:
    for thr,lab,col in SIZE_CATS:
        if r >= thr: return lab, col
    return "S", (150,220,0)


def normalize_frame(gray: np.ndarray) -> np.ndarray:
    """Division par fond flou → élimine le gradient d'éclairage (zigzag)."""
    k = BG_KERNEL if BG_KERNEL%2==1 else BG_KERNEL+1
    bg = cv2.GaussianBlur(gray.astype(np.float32),(k,k),0)
    return np.clip(gray.astype(np.float32)/(bg+1e-6)*128, 0, 255).astype(np.uint8)


def dark_coverage(norm: np.ndarray, cx: float, cy: float, r: float) -> float:
    """
    Fraction de pixels SOMBRES à l'intérieur du cercle (rayon réduit à 75 %).
    Valeur haute → vrai cercle de goutte (corps sombre).
    Valeur basse → faux cercle autour de plusieurs gouttes (fond clair entre elles).
    """
    inner_r = max(1, int(r * 0.75))
    mask = np.zeros(norm.shape, np.uint8)
    cv2.circle(mask,(int(cx),int(cy)),inner_r,255,-1)
    inside = norm[mask>0]
    if len(inside) == 0:
        return 0.0
    return float((inside < DARK_THRESHOLD).sum()) / len(inside)


def nms(circles: List[Tuple], overlap: float = NMS_OVERLAP) -> List[Tuple]:
    """Non-Maximum Suppression. Grands cercles prioritaires."""
    kept: List[Tuple] = []
    for c in sorted(circles, key=lambda x: -x[2]):
        cx,cy,cr = c[:3]
        if not any(
            ((cx-kx)**2+(cy-ky)**2)**0.5 < (cr+kr)*overlap
            for kx,ky,kr in kept
        ):
            kept.append((cx,cy,cr))
    return kept


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DÉTECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_circles(gray: np.ndarray) -> List[Tuple[float,float,float]]:
    """
    Retourne [(x, y, r), …] en pixels (même résolution que gray).
    Applique : normalisation → Hough 3 passes → NMS → filtre couverture sombre.
    """
    norm = normalize_frame(gray)
    raw: List[Tuple] = []

    for blur_k,dp,minD,p1,p2,minR,maxR,_lab in HOUGH_PASSES:
        b = cv2.GaussianBlur(norm,(blur_k,blur_k),0)
        c = cv2.HoughCircles(b, cv2.HOUGH_GRADIENT, dp=dp, minDist=minD,
                              param1=p1, param2=p2,
                              minRadius=minR, maxRadius=maxR)
        if c is not None:
            raw.extend((float(x),float(y),float(r)) for x,y,r in c[0])

    deduped = nms(raw)

    # Filtre couverture sombre : rejette les faux grands cercles
    filtered = []
    for cx,cy,cr in deduped:
        cov = dark_coverage(norm, cx, cy, cr)
        min_cov = MIN_DARK_COVERAGE_L if cr >= LARGE_R_THRESH else MIN_DARK_COVERAGE
        if cov >= min_cov:
            filtered.append((cx, cy, cr))

    return filtered


def detect_all(frames: List[np.ndarray]) -> pd.DataFrame:
    print(f"Détection sur {len(frames)} frames…")
    rows = []
    for i,f in enumerate(frames):
        for x,y,r in detect_circles(f):
            rows.append({"frame":i,"x":x,"y":y,"r":r})
        if (i+1)%10==0 or i==len(frames)-1:
            n = sum(1 for row in rows if row["frame"]==i)
            print(f"  {i+1}/{len(frames)} — {n} gouttes")
    return pd.DataFrame(rows)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TRACKING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_tracking(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not HAS_LAPTRACK:
        df = df.copy(); df["track_id"] = range(len(df))
        return df, pd.DataFrame()

    large = df["r"] >= RADIUS_LARGE_LAP
    parts, merges, offset = [], [], 0

    for mask, params in [(large, LAP_LARGE), (~large, LAP_SMALL)]:
        sub = df[mask].copy().reset_index(drop=True)
        if len(sub)==0: continue
        lt = LapTrack(**params)
        tr, _, mg = lt.predict_dataframe(sub, coordinate_cols=["x","y"],
                                          frame_col="frame")
        tr = tr.copy(); tr["track_id"] += offset
        offset = int(tr["track_id"].max())+1
        parts.append(tr)
        if mg is not None and len(mg)>0:
            mg = mg.copy()
            for col in mg.columns:
                if "track_id" in col.lower():
                    mg[col] += offset - int(tr["track_id"].max()) - 1
            merges.append(mg)

    track_df = pd.concat(parts,  ignore_index=True) if parts  else pd.DataFrame()
    merge_df = pd.concat(merges, ignore_index=True) if merges else pd.DataFrame()
    return track_df, merge_df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CSV FUSIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def save_merges_csv(merge_df, track_df, stem, efps):
    if merge_df.empty: return
    rows=[]
    for _,row in merge_df.iterrows():
        fr=int(row.get("frame",row.get("child_frame",0)))
        pa=int(row.get("parent_track_id",-1))
        ch=int(row.get("child_track_id",-1))
        cr=track_df[(track_df["track_id"]==ch)&(track_df["frame"]==fr)]
        x=float(cr["x"].values[0]) if len(cr) else -1
        y=float(cr["y"].values[0]) if len(cr) else -1
        r=float(cr["r"].values[0]) if len(cr) and "r" in cr.columns else 0
        rows.append({"frame":fr,"time_s":f"{fr/max(efps,1e-6):.2f}",
                     "track_absorbe":pa,"track_disparu":ch,
                     "x":f"{x:.0f}","y":f"{y:.0f}","rayon_px":f"{r:.1f}"})
    p = stem.parent/f"{stem.stem}_merges.csv"
    with open(p,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  → {len(rows)} fusions → {p}")


def build_merge_idx(merge_df, track_df):
    idx=defaultdict(list)
    if merge_df.empty: return idx
    for _,row in merge_df.iterrows():
        fr=int(row.get("frame",row.get("child_frame",0)))
        pa=int(row.get("parent_track_id",-1))
        ch=int(row.get("child_track_id",-1))
        cr=track_df[(track_df["track_id"]==ch)&(track_df["frame"]==fr)]
        x=float(cr["x"].values[0]) if len(cr) else -1
        y=float(cr["y"].values[0]) if len(cr) else -1
        idx[fr].append({"parent":pa,"child":ch,"x":x,"y":y})
    return idx


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RENDU
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def draw_star(img, cx, cy, r, color, t=2):
    for deg in range(0,360,60):
        a=np.radians(deg)
        cv2.line(img,(cx,cy),(int(cx+r*np.cos(a)),int(cy+r*np.sin(a))),
                 color,t,cv2.LINE_AA)


def render(frames, track_df, merge_idx):
    print("Rendu…")
    results=[]
    trails   = defaultdict(lambda: deque(maxlen=TRAIL_LEN))
    colors   = {int(p): make_color(int(p)) for p in track_df["track_id"].unique()}
    actives  = []
    mlog     = deque(maxlen=MERGE_LOG_ROWS)
    n        = len(frames)

    for i,frame in enumerate(frames):
        out = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        fd  = track_df[track_df["frame"]==i]

        # Traînées
        for _,row in fd.iterrows():
            trails[int(row["track_id"])].append((int(row["x"]),int(row["y"])))
        for pid,trail in trails.items():
            if pid not in fd["track_id"].values: continue
            col=colors.get(pid,(180,180,180)); pts=list(trail)
            for k in range(1,len(pts)):
                a=k/len(pts)
                cv2.line(out,pts[k-1],pts[k],tuple(int(v*a) for v in col),1)

        # Cercles
        for _,row in fd.iterrows():
            pid=int(row["track_id"])
            cx,cy=int(row["x"]),int(row["y"])
            r=max(int(row.get("r",8)),4)
            col=colors.get(pid,(180,180,180))
            lab,_=size_label(float(row.get("r",0)))
            thick=3 if lab in("XL","L") else 2
            cv2.circle(out,(cx,cy),r,col,thick,cv2.LINE_AA)
            arm=max(3,r//5)
            cv2.line(out,(cx-arm,cy),(cx+arm,cy),col,1,cv2.LINE_AA)
            cv2.line(out,(cx,cy-arm),(cx,cy+arm),col,1,cv2.LINE_AA)
            cv2.putText(out,f"{pid}",(cx+r+2,cy-2),
                        cv2.FONT_HERSHEY_SIMPLEX,0.30,(255,255,255),1,cv2.LINE_AA)

        # Fusions
        if i in merge_idx:
            for ev in merge_idx[i]:
                pa,ch=ev["parent"],ev["child"]
                mx,my=int(ev["x"]),int(ev["y"])
                fc=tuple((int(colors.get(pa,(255,100,50))[k])+
                           int(colors.get(ch,(255,100,50))[k]))//2 for k in range(3))
                actives.append({"x":mx,"y":my,"col":fc,"ttl":MERGE_FLASH_LEN})
                mlog.appendleft(f"FUSION #{pa}+#{ch}→#{pa}  frame {i}")

        still=[]
        for ev in actives:
            a=ev["ttl"]/MERGE_FLASH_LEN; rs=int(22*a)+8
            col=tuple(int(c*a) for c in ev["col"])
            draw_star(out,ev["x"],ev["y"],rs,col,max(1,int(3*a)))
            cv2.circle(out,(ev["x"],ev["y"]),rs+6,col,max(1,int(2*a)),cv2.LINE_AA)
            ev["ttl"]-=1
            if ev["ttl"]>0: still.append(ev)
        actives[:]=still

        if mlog:
            ov=out.copy(); bh=14*(len(mlog)+1)
            cv2.rectangle(ov,(0,out.shape[0]-bh),(520,out.shape[0]),(30,30,30),-1)
            out=cv2.addWeighted(ov,0.55,out,0.45,0)
            for j,txt in enumerate(mlog):
                cv2.putText(out,txt,(6,out.shape[0]-bh+12+j*14),
                            cv2.FONT_HERSHEY_SIMPLEX,0.38,(60,230,255),1,cv2.LINE_AA)

        cv2.rectangle(out,(0,0),(270,22),(30,30,30),-1)
        cv2.putText(out,f"Frame {i+1}/{n}  Gouttes: {len(fd)}",
                    (5,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(220,220,220),1,cv2.LINE_AA)
        results.append(out)
        if (i+1)%20==0: print(f"  {i+1}/{n}")
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  VIDÉO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_frames(path, fps_extract=2.0):
    cap=cv2.VideoCapture(str(path))
    src_fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_dur=total/src_fps
    step=max(1,round(src_fps/fps_extract))
    frames,idx=[],0
    while True:
        ret,f=cap.read()
        if not ret: break
        if idx%step==0: frames.append(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY))
        idx+=1
    cap.release()
    print(f"{total} frames @ {src_fps:.0f} fps → {len(frames)} extraites")
    return frames, src_fps, src_dur


def save_video(frames, path, fps=10.0):
    if not frames: return
    h,w=frames[0].shape[:2]
    out=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
    for f in frames: out.write(f)
    out.release()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run(video_path: Path, fps_extract: float):
    frames, src_fps, src_dur = load_frames(video_path, fps_extract)
    efps = len(frames) / src_dur

    df = detect_all(frames)
    if df.empty: print("Aucune détection."); return
    print(f"{len(df)} détections totales.")

    track_df, merge_df = run_tracking(df)
    if not track_df.empty:
        print(f"{track_df['track_id'].nunique()} trajectoires, {len(merge_df)} fusions.")

    stem = video_path.parent / (video_path.stem + "_hough")
    if not merge_df.empty:
        save_merges_csv(merge_df, track_df, stem, efps)

    rendered = render(frames, track_df, build_merge_idx(merge_df, track_df))
    out_path = video_path.parent / f"{video_path.stem}_hough.mp4"
    save_video(rendered, out_path, fps=fps_extract)
    print(f"✓  {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Tracker de gouttes de condensation — HoughCircles + filtre couverture")
    p.add_argument("video", type=Path)
    p.add_argument("--fps-extract", type=float, default=2.0,
                   help="Frames/s à extraire (défaut 2)")
    args = p.parse_args()
    run(args.video, args.fps_extract)