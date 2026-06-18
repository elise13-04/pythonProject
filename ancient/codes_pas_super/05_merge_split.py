"""
05_merge_split.py — Fusions et séparations pour gouttes réflectives.

Méthodes :
  A. Topological — IoU sur bboxes estimées depuis les cercles (léger)
  B. laptrack    — LAP avec coûts de fusion/séparation explicites

Toutes deux utilisent detect_droplets (LoG+NMS) → un cercle par goutte.
"""
from __future__ import annotations
import argparse
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple

import cv2, numpy as np, pandas as pd
from utils import (load_frames, save_video, make_color, TRAIL_LEN, detect_droplets)

MAX_DIST=65; MAX_BLOBS=80; EVENT_FRAMES=8
MERGE_COLOR=(0,60,255); SPLIT_COLOR=(255,140,0)
LAP_R=55; LAP_GAP=2; LAP_COST=60


def _circle_iou(a,b):
    """IoU approximée entre deux cercles via intersection géométrique."""
    d=np.hypot(a["x"]-b["x"],a["y"]-b["y"]); r1=a["r"]; r2=b["r"]
    if d>=r1+r2: return 0.0
    if d<=abs(r1-r2): return min(r1,r2)**2/max(r1,r2)**2
    # Formule exacte d'intersection de deux disques
    a1=r1**2*np.arccos((d**2+r1**2-r2**2)/(2*d*r1+1e-9))
    a2=r2**2*np.arccos((d**2+r2**2-r1**2)/(2*d*r2+1e-9))
    a3=0.5*np.sqrt((-d+r1+r2)*(d+r1-r2)*(d-r1+r2)*(d+r1+r2)+1e-9)
    inter=a1+a2-a3; union=np.pi*(r1**2+r2**2)-inter
    return float(inter/(union+1e-9))


def _draw_events(out,events,fi):
    for ev in events:
        age=fi-ev["frame"]; alpha=max(0.0,1.0-age/EVENT_FRAMES)
        if alpha<=0: continue
        cx,cy=ev["pos"]; col=MERGE_COLOR if ev["kind"]=="MERGE" else SPLIT_COLOR
        col=tuple(int(c*alpha) for c in col)
        cv2.circle(out,(cx,cy),14,col,2,cv2.LINE_AA)
        cv2.putText(out,ev["kind"],(cx+16,cy+4),cv2.FONT_HERSHEY_SIMPLEX,0.42,col,1,cv2.LINE_AA)


def run_topological(frames):
    print("\n[Topological]")
    all_blobs=[]
    for i,f in enumerate(frames):
        drops=detect_droplets(f)
        drops=sorted(drops,key=lambda d:d["r"],reverse=True)[:MAX_BLOBS]
        all_blobs.append(drops)
        if (i+1)%10==0: print(f"  {i+1}/{len(frames)} — {len(drops)} gouttes")

    all_events=[]; trails=defaultdict(lambda: deque(maxlen=TRAIL_LEN))
    colors={}; next_id=1; results=[]

    for b in all_blobs[0]:
        b["id"]=next_id; colors[next_id]=make_color(next_id)
        trails[next_id].append((int(b["x"]),int(b["y"]))); next_id+=1

    out0=cv2.cvtColor(frames[0],cv2.COLOR_GRAY2BGR)
    for b in all_blobs[0]:
        col=colors.get(b["id"],(200,200,200))
        cv2.circle(out0,(int(b["x"]),int(b["y"])),max(int(b["r"]),3),col,1,cv2.LINE_AA)
    cv2.putText(out0,f"Topo n={len(all_blobs[0])}",(4,12),cv2.FONT_HERSHEY_SIMPLEX,0.38,(180,180,180),1)
    results.append(out0)

    for i in range(1,len(frames)):
        prev_b=all_blobs[i-1]; curr_b=all_blobs[i]; frame=frames[i]
        in_l=defaultdict(list); out_l=defaultdict(list)
        for ii,ba in enumerate(prev_b):
            for jj,bb in enumerate(curr_b):
                if np.hypot(ba["x"]-bb["x"],ba["y"]-bb["y"])>MAX_DIST: continue
                if _circle_iou(ba,bb)>0.05:
                    in_l[jj].append(ii); out_l[ii].append(jj)
        for jj,blob in enumerate(curr_b):
            parents=in_l[jj]
            if not parents: blob["id"]=next_id; colors[next_id]=make_color(next_id); next_id+=1
            elif len(parents)==1: blob["id"]=prev_b[parents[0]]["id"]
            else:
                big=max(parents,key=lambda p:prev_b[p]["r"])
                blob["id"]=prev_b[big]["id"]
                all_events.append({"kind":"MERGE","pos":(int(blob["x"]),int(blob["y"])),"frame":i})
        for ii,blob in enumerate(prev_b):
            if len(out_l[ii])>1:
                all_events.append({"kind":"SPLIT","pos":(int(blob["x"]),int(blob["y"])),"frame":i})
                for jj in out_l[ii][1:]:
                    curr_b[jj]["id"]=next_id; colors[next_id]=make_color(next_id); next_id+=1
        for blob in curr_b:
            tid=blob.get("id",0)
            if tid: trails[tid].append((int(blob["x"]),int(blob["y"])))
        out=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
        for tid,trail in trails.items():
            col=colors.get(tid,(200,200,200)); pts=list(trail)
            for k in range(1,len(pts)):
                alpha=k/len(pts); c=tuple(int(v*alpha) for v in col)
                cv2.line(out,pts[k-1],pts[k],c,1)
        for blob in curr_b:
            tid=blob.get("id"); col=colors.get(tid,(200,200,200)) if tid else (200,200,200)
            cv2.circle(out,(int(blob["x"]),int(blob["y"])),max(int(blob["r"]),3),col,1,cv2.LINE_AA)
            if tid: cv2.putText(out,str(tid),(int(blob["x"])+3,int(blob["y"])-3),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,255,0),1)
        _draw_events(out,[ev for ev in all_events if i-ev["frame"]<EVENT_FRAMES],i)
        cv2.putText(out,f"Topo n={len(curr_b)}",(4,12),cv2.FONT_HERSHEY_SIMPLEX,0.38,(180,180,180),1)
        results.append(out)

    print(f"  Fusions={sum(1 for e in all_events if e['kind']=='MERGE')}  Séparations={sum(1 for e in all_events if e['kind']=='SPLIT')}")
    return results


def run_laptrack(frames):
    print("\n[laptrack]")
    try: from laptrack import LapTrack
    except ImportError: print("pip install laptrack — fallback topological"); return run_topological(frames)

    rows=[]
    for i,f in enumerate(frames):
        drops=sorted(detect_droplets(f),key=lambda d:d["r"],reverse=True)[:MAX_BLOBS]
        for d in drops: rows.append({"frame":i,"x":d["x"],"y":d["y"],"r":d["r"]})
    if not rows: return [cv2.cvtColor(f,cv2.COLOR_GRAY2BGR) for f in frames]

    df=pd.DataFrame(rows)
    lt=LapTrack(track_dist_metric="sqeuclidean",track_cost_cutoff=LAP_R**2,
                gap_closing_dist_metric="sqeuclidean",gap_closing_cost_cutoff=LAP_R**2,
                gap_closing_max_frame_count=LAP_GAP,
                splitting_dist_metric="sqeuclidean",splitting_cost_cutoff=LAP_COST**2,
                merging_dist_metric="sqeuclidean",merging_cost_cutoff=LAP_COST**2)
    try: track_df,split_df,merge_df=lt.predict_dataframe(df,coordinate_cols=["x","y"],frame_col="frame",only_coordinate_cols=False)
    except Exception as e: print(f"  Erreur: {e} — fallback"); return run_topological(frames)

    trails=defaultdict(lambda: deque(maxlen=TRAIL_LEN)); colors={}
    all_events=[]
    for pid in track_df["track_id"].unique(): colors[int(pid)]=make_color(int(pid))
    for df2,kind in [(merge_df,"MERGE"),(split_df,"SPLIT")]:
        if df2 is not None and len(df2)>0:
            for _,row in df2.iterrows():
                all_events.append({"kind":kind,"pos":(int(row.get("x",0)),int(row.get("y",0))),"frame":int(row.get("frame",0))})

    results=[]
    for i,frame in enumerate(frames):
        out=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR); fd=track_df[track_df["frame"]==i]
        for _,row in fd.iterrows():
            pid=int(row["track_id"]); trails[pid].append((int(row["x"]),int(row["y"])))
        for pid,trail in trails.items():
            col=colors.get(pid,(200,200,200)); pts=list(trail)
            for k in range(1,len(pts)):
                alpha=k/len(pts); c=tuple(int(v*alpha) for v in col)
                cv2.line(out,pts[k-1],pts[k],c,1)
        for _,row in fd.iterrows():
            pid=int(row["track_id"]); cx,cy=int(row["x"]),int(row["y"]); r=int(row.get("r",5))
            col=colors.get(pid,(200,200,200))
            cv2.circle(out,(cx,cy),max(r,3),col,1,cv2.LINE_AA)
            cv2.putText(out,str(pid),(cx+3,cy-3),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,255,0),1)
        _draw_events(out,[ev for ev in all_events if i-ev["frame"]<EVENT_FRAMES],i)
        cv2.putText(out,f"laptrack n={len(fd)}",(4,12),cv2.FONT_HERSHEY_SIMPLEX,0.38,(180,180,180),1)
        results.append(out)
        if (i+1)%10==0: print(f"  {i+1}/{len(frames)} — {len(fd)}")
    print(f"  Fusions={sum(1 for e in all_events if e['kind']=='MERGE')}  Séparations={sum(1 for e in all_events if e['kind']=='SPLIT')}")
    return results


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("video",type=Path)
    ap.add_argument("--method",choices=["topological","laptrack","all"],default="all")
    ap.add_argument("--output-dir",type=Path,default=Path(".")); ap.add_argument("--fps-extract",type=float,default=1.0)
    args=ap.parse_args()
    frames,_,src_dur=load_frames(args.video,fps_extract=args.fps_extract); print(f"Frames:{len(frames)}")
    runners={"topological":run_topological,"laptrack":run_laptrack}
    to_run=list(runners) if args.method=="all" else [args.method]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name in to_run:
        ann=runners[name](frames)
        save_video(ann,args.output_dir/f"05_ms_{name}.mp4",src_duration=src_dur)

if __name__=="__main__": main()