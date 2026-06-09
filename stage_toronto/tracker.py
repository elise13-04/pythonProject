"""
tracker_hough.py  —  Tracking de gouttes de condensation
=========================================================
Nouveauté clé : masque du zigzag
  Le patron en zigzag central est détecté automatiquement comme une zone
  de luminosité locale élevée (GaussianBlur 101px > seuil 95) et exclu
  de la détection.  Aucune goutte fantôme dans le zigzag.

Pipeline complet :
  1. Normalisation (division par fond flou 201px)
  2. Masque zigzag (luminosité locale > 95 → zone exclue)
  3. HoughCircles 4 passes (maxR ≤ 34px, calibré sur données réelles)
  4. NMS inter-passes
  5. Filtre couverture sombre (rejette les cercles autour du fond)
  6. Vérification finale : le centre du cercle ne doit pas être dans le masque
  7. LapTrack + CSV fusions + flash visuel

Dépendances :
    pip install opencv-python numpy pandas laptrack

Usage :
    python tracker_hough.py video.mp4
    python tracker_hough.py video.mp4 --fps-extract 5

Réglages (section PARAMÈTRES ci-dessous) :
  ZIGZAG_BRIGHTNESS_THRESH  ↑ masque moins de surface  ↓ masque plus
  param2 dans HOUGH_PASSES  ↑ moins de détections      ↓ plus
  MIN_COV_*                 ↑ rejette plus de FP
"""

import argparse, csv
from collections import defaultdict, deque
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd

try:
    from laptrack import LapTrack
    HAS_LAPTRACK = True
except ImportError:
    HAS_LAPTRACK = False
    print("⚠  pip install laptrack  pour activer le suivi")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PARAMÈTRES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BG_KERNEL = 201          # normalisation fond (px)

# Masque zigzag
ZIGZAG_LOCAL_KERNEL   = 101   # taille du flou pour détecter la luminosité locale
ZIGZAG_BRIGHTNESS_THRESH = 95 # seuil : local_mean > valeur → zone zigzag
ZIGZAG_CLOSE_KERNEL   = 60    # fermeture morphologique pour remplir les trous
ZIGZAG_OPEN_KERNEL    = 20    # ouverture pour supprimer les petites taches

# 4 passes HoughCircles — PLEINE RÉSOLUTION
# (blur_k, dp, minDist, param1, param2, minR, maxR)
# maxR = 34px → calibré sur données réelles (p99 = 33px)
HOUGH_PASSES = [
    ( 5, 1.2,   6,  60,  16,   4,   9),
    ( 7, 1.2,  10,  60,  18,   7,  16),
    (11, 1.2,  16,  60,  22,  13,  24),
    (15, 1.2,  22,  60,  26,  20,  34),
]

NMS_OVERLAP       = 0.40
DARK_THRESHOLD    = 108
MIN_COV_SMALL     = 0.28   # r < LARGE_R_THRESH
MIN_COV_LARGE     = 0.40   # r ≥ LARGE_R_THRESH
LARGE_R_THRESH    = 20

# LapTrack
RADIUS_LARGE_LAP = 20
LAP_LARGE = dict(
    track_dist_metric="sqeuclidean", track_cost_cutoff=15000,
    gap_closing_dist_metric="sqeuclidean", gap_closing_cost_cutoff=1200,
    gap_closing_max_frame_count=3,
    merging_dist_metric="sqeuclidean", merging_cost_cutoff=700,
)
LAP_SMALL = dict(
    track_dist_metric="sqeuclidean", track_cost_cutoff=800,
    gap_closing_dist_metric="sqeuclidean", gap_closing_cost_cutoff=800,
    gap_closing_max_frame_count=2,
)

TRAIL_LEN       = 30
MERGE_FLASH_LEN = 14
MERGE_LOG_ROWS  = 6

SIZE_CATS = [
    (28, "L",  (0,  80, 220)),
    (16, "M",  (0, 190, 160)),
    ( 0, "S",  (140, 210,  0)),
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  UTILITAIRES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_color(seed: int) -> Tuple[int,int,int]:
    rng = np.random.default_rng(seed * 6364136223846793005 & 0xFFFFFFFF)
    hsv = np.uint8([[[rng.integers(0,180),rng.integers(150,255),rng.integers(180,255)]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0,0]
    return (int(bgr[0]),int(bgr[1]),int(bgr[2]))


def size_label(r: float) -> Tuple[str, Tuple]:
    for thr,lab,col in SIZE_CATS:
        if r >= thr: return lab, col
    return "S", (140,210,0)


def normalize_frame(gray: np.ndarray) -> np.ndarray:
    k = BG_KERNEL if BG_KERNEL%2==1 else BG_KERNEL+1
    bg = cv2.GaussianBlur(gray.astype(np.float32),(k,k),0)
    return np.clip(gray.astype(np.float32)/(bg+1e-6)*128, 0, 255).astype(np.uint8)


def make_zigzag_mask(gray: np.ndarray) -> np.ndarray:
    """
    Retourne un masque binaire uint8 :  1 = zone zigzag (exclure),  0 = zone gouttes.
    Méthode : luminosité locale (GaussianBlur 101px) > seuil → zigzag.
    Nettoyage morphologique pour éviter d'exclure les reflets des grosses gouttes.
    """
    lm = cv2.GaussianBlur(gray.astype(np.float32),
                           (ZIGZAG_LOCAL_KERNEL, ZIGZAG_LOCAL_KERNEL), 0)
    bright = (lm > ZIGZAG_BRIGHTNESS_THRESH).astype(np.uint8)
    k_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                     (ZIGZAG_CLOSE_KERNEL, ZIGZAG_CLOSE_KERNEL))
    k_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                     (ZIGZAG_OPEN_KERNEL,  ZIGZAG_OPEN_KERNEL))
    closed = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k_c)
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_o)


def dark_coverage(norm: np.ndarray, cx: float, cy: float, r: float) -> float:
    inner = max(1, int(r * 0.80))
    mask = np.zeros(norm.shape, np.uint8)
    cv2.circle(mask, (int(cx),int(cy)), inner, 255, -1)
    inside = norm[mask > 0]
    if len(inside) == 0: return 0.0
    return float((inside < DARK_THRESHOLD).sum()) / len(inside)


def nms(circles: List[Tuple], overlap: float = NMS_OVERLAP) -> List[Tuple]:
    kept: List[Tuple] = []
    for c in sorted(circles, key=lambda x: -x[2]):
        cx,cy,cr = c[:3]
        if not any(((cx-kx)**2+(cy-ky)**2)**0.5 < (cr+kr)*overlap
                   for kx,ky,kr in kept):
            kept.append((cx,cy,cr))
    return kept


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DÉTECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_circles(gray: np.ndarray) -> List[Tuple[float,float,float]]:
    """Retourne [(x, y, r), …] en pixels pleine résolution, zigzag exclu."""
    norm     = normalize_frame(gray)
    mask_zz  = make_zigzag_mask(gray)

    # Remplacer la zone zigzag par du gris uniforme → plus de gradient → Hough aveugle
    norm_m = norm.copy()
    norm_m[mask_zz == 1] = 128

    raw: List[Tuple] = []
    for blur_k,dp,minD,p1,p2,minR,maxR in HOUGH_PASSES:
        b = cv2.GaussianBlur(norm_m,(blur_k,blur_k),0)
        c = cv2.HoughCircles(b, cv2.HOUGH_GRADIENT, dp=dp, minDist=minD,
                              param1=p1, param2=p2,
                              minRadius=minR, maxRadius=maxR)
        if c is not None:
            raw.extend((float(x),float(y),float(r)) for x,y,r in c[0])

    deduped = nms(raw)

    return [
        (cx,cy,cr) for cx,cy,cr in deduped
        # Filtre 1 : couverture sombre
        if dark_coverage(norm, cx, cy, cr) >= (
            MIN_COV_LARGE if cr >= LARGE_R_THRESH else MIN_COV_SMALL)
        # Filtre 2 : le centre du cercle ne doit pas être dans le zigzag
        and mask_zz[min(int(cy), gray.shape[0]-1),
                    min(int(cx), gray.shape[1]-1)] == 0
    ]


def detect_all(frames: List[np.ndarray]) -> pd.DataFrame:
    print(f"Détection sur {len(frames)} frames…")
    rows = []
    for i,f in enumerate(frames):
        circles = detect_circles(f)
        for x,y,r in circles:
            rows.append({"frame":i,"x":x,"y":y,"r":r})
        if (i+1)%10==0 or i==len(frames)-1:
            print(f"  {i+1}/{len(frames)} — {len(circles)} gouttes")
    return pd.DataFrame(rows)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TRACKING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
        tr, _, mg = lt.predict_dataframe(sub, coordinate_cols=["x","y"], frame_col="frame")
        tr = tr.copy(); tr["track_id"] += offset
        offset = int(tr["track_id"].max())+1
        parts.append(tr)
        if mg is not None and len(mg)>0:
            mg = mg.copy()
            for col in mg.columns:
                if "track_id" in col.lower():
                    mg[col] += offset-int(tr["track_id"].max())-1
            merges.append(mg)

    track_df = pd.concat(parts,  ignore_index=True) if parts  else pd.DataFrame()
    merge_df = pd.concat(merges, ignore_index=True) if merges else pd.DataFrame()
    return track_df, merge_df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CSV FUSIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
    p=stem.parent/f"{stem.stem}_merges.csv"
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RENDU
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def draw_star(img,cx,cy,r,color,t=2):
    for deg in range(0,360,60):
        a=np.radians(deg)
        cv2.line(img,(cx,cy),(int(cx+r*np.cos(a)),int(cy+r*np.sin(a))),color,t,cv2.LINE_AA)


def render(frames, track_df, merge_idx):
    print("Rendu…")
    results=[]
    trails  = defaultdict(lambda: deque(maxlen=TRAIL_LEN))
    colors  = {int(p): make_color(int(p)) for p in track_df["track_id"].unique()}
    actives = []
    mlog    = deque(maxlen=MERGE_LOG_ROWS)
    n       = len(frames)

    for i,frame in enumerate(frames):
        out = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        fd  = track_df[track_df["frame"]==i]

        for _,row in fd.iterrows():
            trails[int(row["track_id"])].append((int(row["x"]),int(row["y"])))
        for pid,trail in trails.items():
            if pid not in fd["track_id"].values: continue
            col=colors.get(pid,(180,180,180)); pts=list(trail)
            for k in range(1,len(pts)):
                a=k/len(pts)
                cv2.line(out,pts[k-1],pts[k],tuple(int(v*a) for v in col),1)

        for _,row in fd.iterrows():
            pid=int(row["track_id"])
            cx,cy=int(row["x"]),int(row["y"])
            r=max(int(row.get("r",8)),4)
            col=colors.get(pid,(180,180,180))
            lab,_=size_label(float(row.get("r",0)))
            thick=2 if lab=="L" else 1
            cv2.circle(out,(cx,cy),r,col,thick,cv2.LINE_AA)
            arm=max(2,r//5)
            cv2.line(out,(cx-arm,cy),(cx+arm,cy),col,1,cv2.LINE_AA)
            cv2.line(out,(cx,cy-arm),(cx,cy+arm),col,1,cv2.LINE_AA)
            cv2.putText(out,f"{pid}[{lab}]",(cx+r+2,cy),
                        cv2.FONT_HERSHEY_SIMPLEX,0.28,(255,255,255),1,cv2.LINE_AA)

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  VIDÉO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
        description="Tracker gouttes — HoughCircles + masque zigzag + filtre couverture")
    p.add_argument("video", type=Path)
    p.add_argument("--fps-extract", type=float, default=2.0)
    args = p.parse_args()
    run(args.video, args.fps_extract)