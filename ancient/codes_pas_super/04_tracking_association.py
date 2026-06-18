"""
04_tracking_association.py — Tracking pour gouttes réflectives.

Méthodes :
  A. SORT        — Kalman + Hongrois + correction par flot optique
  B. trackpy     — Crocker-Grier sur image floutée
  C. Farneback   — flot dense → flèches de vitesse sur chaque goutte

Lucas-Kanade supprimé (suivait des coins, pas des gouttes).
"""
from __future__ import annotations
import argparse, warnings
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple

import cv2, numpy as np, pandas as pd
from utils import (load_frames, draw_tracks, save_video,
                   make_color, hungarian_match, TRAIL_LEN,
                   detect_droplets, _clahe, DETECT_BLUR_SIGMA)

MAX_DIST=65; MAX_AGE=4; MIN_HITS=2
TP_DIAMETER=11; TP_MIN_MASS=50; TP_SEARCH_R=35; TP_MEMORY=3


class _KF:
    _n=0
    def __init__(self,det):
        _KF._n+=1; self.id=_KF._n; self.hits=1; self.lost=0
        self.color=make_color(self.id); self.r=det.get("r",6)
        self.trail=deque(maxlen=TRAIL_LEN)
        self.trail.append((int(det["x"]),int(det["y"])))
        self.kf=cv2.KalmanFilter(4,2)
        self.kf.transitionMatrix=np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
        self.kf.measurementMatrix=np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        self.kf.processNoiseCov=np.eye(4,dtype=np.float32)*5e-3
        self.kf.measurementNoiseCov=np.eye(2,dtype=np.float32)*5e-2
        self.kf.errorCovPost=np.eye(4,dtype=np.float32)
        self.kf.statePost=np.array([det["x"],det["y"],0.,0.],np.float32).reshape(4,1)
    def predict(self):
        p=self.kf.predict(); return float(p[0].flat[0]),float(p[1].flat[0])
    def update(self,det):
        self.kf.correct(np.array([det["x"],det["y"]],np.float32).reshape(2,1))
        s=self.kf.statePost; self.trail.append((int(s[0].flat[0]),int(s[1].flat[0])))
        self.hits+=1; self.lost=0; self.r=det.get("r",self.r)
    def pos(self): s=self.kf.statePost; return float(s[0].flat[0]),float(s[1].flat[0])


def run_sort(frames):
    print("\n[SORT]"); _KF._n=0
    trackers=[]; results=[]; prev_gray=None
    for i,frame in enumerate(frames):
        dets=detect_droplets(frame)
        # Correction par flot optique médian
        fdx=fdy=0.0
        if prev_gray is not None:
            flow=cv2.calcOpticalFlowFarneback(prev_gray,frame,None,
                 pyr_scale=0.5,levels=2,winsize=12,iterations=3,poly_n=5,poly_sigma=1.1,flags=0)
            fdx=float(np.median(flow[...,0])); fdy=float(np.median(flow[...,1]))
        prev_gray=frame
        for t in trackers:
            t.predict()
            t.kf.statePost[0,0]+=fdx; t.kf.statePost[1,0]+=fdy
        if trackers and dets:
            preds=[{"x":t.pos()[0],"y":t.pos()[1],"id":t.id} for t in trackers]
            matches=hungarian_match(preds,dets,MAX_DIST)
            mt={m[0] for m in matches}; md={m[1] for m in matches}
            for ti,di in matches: trackers[ti].update(dets[di])
            for ti in range(len(trackers)):
                if ti not in mt: trackers[ti].lost+=1
            for di in range(len(dets)):
                if di not in md: trackers.append(_KF(dets[di]))
        elif dets:
            for d in dets: trackers.append(_KF(d))
        else:
            for t in trackers: t.lost+=1
        trackers=[t for t in trackers if t.lost<=MAX_AGE]
        active=[t for t in trackers if t.hits>=MIN_HITS]
        out=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
        for t in active:
            pts=list(t.trail)
            for k in range(1,len(pts)):
                alpha=k/len(pts); c=tuple(int(v*alpha) for v in t.color)
                cv2.line(out,pts[k-1],pts[k],c,1,cv2.LINE_AA)
        for t in active:
            cx,cy=int(t.pos()[0]),int(t.pos()[1])
            cv2.circle(out,(cx,cy),max(int(t.r),3),t.color,1,cv2.LINE_AA)
            cv2.putText(out,str(t.id),(cx+3,cy-3),cv2.FONT_HERSHEY_SIMPLEX,0.32,(255,255,0),1)
        cv2.putText(out,f"SORT n={len(active)}",(4,12),cv2.FONT_HERSHEY_SIMPLEX,0.38,(180,180,180),1)
        results.append(out)
        if (i+1)%10==0: print(f"  {i+1}/{len(frames)} — {len(active)} pistes")
    return results


def run_trackpy(frames):
    print("\n[trackpy]")
    warnings.filterwarnings("ignore")
    try: import trackpy as tp
    except ImportError: print("pip install trackpy"); return [cv2.cvtColor(f,cv2.COLOR_GRAY2BGR) for f in frames]
    records=[]
    for i,frame in enumerate(frames):
        eq=_clahe.apply(frame)
        blur=cv2.GaussianBlur(eq,(0,0),sigmaX=DETECT_BLUR_SIGMA)
        f=tp.locate(blur,diameter=TP_DIAMETER,minmass=TP_MIN_MASS,separation=TP_DIAMETER//2)
        if f is not None and len(f)>0: f["frame"]=i; records.append(f)
    if not records: print("  Aucune particule."); return [cv2.cvtColor(f,cv2.COLOR_GRAY2BGR) for f in frames]
    all_locs=pd.concat(records,ignore_index=True)
    try:
        traj=tp.link(all_locs,search_range=TP_SEARCH_R,memory=TP_MEMORY)
        traj=tp.filter_stubs(traj,threshold=2)
    except Exception as e: print(f"  Erreur: {e}"); return [cv2.cvtColor(f,cv2.COLOR_GRAY2BGR) for f in frames]
    trails=defaultdict(lambda: deque(maxlen=TRAIL_LEN)); colors={}
    for pid in traj["particle"].unique(): colors[int(pid)]=make_color(int(pid))
    results=[]
    for i,frame in enumerate(frames):
        out=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR); fd=traj[traj["frame"]==i]
        for _,row in fd.iterrows():
            pid=int(row["particle"]); trails[pid].append((int(row["x"]),int(row["y"])))
        for pid,trail in trails.items():
            col=colors.get(pid,(200,200,200)); pts=list(trail)
            for k in range(1,len(pts)):
                alpha=k/len(pts); c=tuple(int(v*alpha) for v in col)
                cv2.line(out,pts[k-1],pts[k],c,1)
        for _,row in fd.iterrows():
            pid=int(row["particle"]); cx,cy=int(row["x"]),int(row["y"])
            col=colors.get(pid,(200,200,200))
            cv2.circle(out,(cx,cy),5,col,1,cv2.LINE_AA)
            cv2.putText(out,str(pid),(cx+3,cy-3),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,255,0),1)
        cv2.putText(out,f"trackpy n={len(fd)}",(4,12),cv2.FONT_HERSHEY_SIMPLEX,0.38,(180,180,180),1)
        results.append(out)
        if (i+1)%10==0: print(f"  {i+1}/{len(frames)} — {len(fd)}")
    return results


def run_farneback_guided(frames):
    """Flot dense : flèches de vitesse colorées par intensité sur chaque goutte."""
    print("\n[Farneback guidé]")
    results=[]; prev=frames[0]
    fb=dict(pyr_scale=0.5,levels=2,winsize=10,iterations=3,poly_n=5,poly_sigma=1.1,flags=0)
    for i,frame in enumerate(frames):
        flow=cv2.calcOpticalFlowFarneback(prev,frame,None,**fb)
        dets=detect_droplets(frame); out=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
        max_mag=1e-9; vecs=[]
        for d in dets:
            cx,cy=int(d["x"]),int(d["y"])
            cy_c=np.clip(cy,0,flow.shape[0]-1); cx_c=np.clip(cx,0,flow.shape[1]-1)
            dx=float(flow[cy_c,cx_c,0]); dy=float(flow[cy_c,cx_c,1])
            mag=np.hypot(dx,dy); max_mag=max(max_mag,mag); vecs.append((cx,cy,dx,dy,mag,d["r"]))
        for (cx,cy,dx,dy,mag,r) in vecs:
            ratio=min(mag/max_mag,1.0); col=(int(255*(1-ratio)),50,int(255*ratio))
            cv2.circle(out,(cx,cy),max(int(r),3),col,1,cv2.LINE_AA)
            if mag>0.5:
                cv2.arrowedLine(out,(cx,cy),(int(cx+dx*3),int(cy+dy*3)),col,1,tipLength=0.4,line_type=cv2.LINE_AA)
        mm=float(np.mean([v[4] for v in vecs])) if vecs else 0
        cv2.putText(out,f"Farneback n={len(dets)} v={mm:.1f}px",(4,12),cv2.FONT_HERSHEY_SIMPLEX,0.35,(200,200,200),1)
        results.append(out); prev=frame
        if (i+1)%10==0: print(f"  {i+1}/{len(frames)} — {len(dets)} gouttes v_moy={mm:.1f}px")
    return results


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("video",type=Path)
    ap.add_argument("--method",choices=["sort","trackpy","farneback","all"],default="all")
    ap.add_argument("--output-dir",type=Path,default=Path(".")); ap.add_argument("--fps-extract",type=float,default=1.0)
    args=ap.parse_args()
    frames,_,src_dur=load_frames(args.video,fps_extract=args.fps_extract); print(f"Frames:{len(frames)}")
    runners={"sort":run_sort,"trackpy":run_trackpy,"farneback":run_farneback_guided}
    to_run=list(runners) if args.method=="all" else [args.method]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name in to_run:
        ann=runners[name](frames)
        save_video(ann,args.output_dir/f"04_tracking_{name}.mp4",src_duration=src_dur)

if __name__=="__main__": main()