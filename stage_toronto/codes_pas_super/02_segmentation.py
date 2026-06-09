"""
02_segmentation.py — Segmentation pour gouttes réflectives.

Méthodes :
  A. LoG+NMS   — détection directe, le cercle englobe toute la goutte
  B. Chan-Vese — travaille sur l'image floutée (pas le top-hat)

Felzenszwalb / LevelSets / Watershed supprimés (fragmentaient les anneaux).
"""
from __future__ import annotations
import argparse
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import measure
from skimage.segmentation import chan_vese

from utils import (load_frames, draw_tracks, save_video,
                   make_color, hungarian_match, TRAIL_LEN,
                   detect_droplets, _clahe, DETECT_BLUR_SIGMA)

MAX_DIST = 65


def _track(frames, all_drops, label):
    trails=defaultdict(lambda: deque(maxlen=TRAIL_LEN)); colors={}; prev=[]; next_id=1; results=[]
    for frame,curr in zip(frames,all_drops):
        matches=hungarian_match(prev,curr,MAX_DIST); id_map={ci:prev[pi].get("id",next_id) for pi,ci in matches}
        for ci,det in enumerate(curr):
            if ci not in id_map: id_map[ci]=next_id; colors[next_id]=make_color(next_id); next_id+=1
            det["id"]=id_map[ci]; trails[det["id"]].append((int(det["x"]),int(det["y"])))
        results.append(draw_tracks(frame,curr,trails,colors,label)); prev=curr
    return results


def run_log_nms(frames):
    """LoG+NMS — même pipeline que detect_droplets, un cercle par goutte."""
    print("\n[LoG+NMS]")
    all_drops=[detect_droplets(f) for f in frames]
    for i,d in enumerate(all_drops):
        if (i+1)%10==0: print(f"  {i+1}/{len(frames)} — {len(d)} gouttes")
    return _track(frames,all_drops,"LoG+NMS")


def run_chanvese_blur(frames):
    """
    Chan-Vese sur image floutée.
    Le flou fort transforme chaque goutte en région homogène grise,
    que Chan-Vese sépare du fond sombre.
    """
    print("\n[Chan-Vese blur] (lent)")
    all_drops=[]
    for i,frame in enumerate(frames):
        eq   =_clahe.apply(frame)
        blur =cv2.GaussianBlur(eq,(0,0),sigmaX=DETECT_BLUR_SIGMA)
        blur_f=blur.astype(np.float64)/255.0
        cv_mask=chan_vese(blur_f,mu=0.1,lambda1=1.0,lambda2=1.0,
                          tol=1e-3,max_num_iter=100,init_level_set="checkerboard")
        mask=cv_mask.astype(np.uint8)
        if blur_f[mask==1].mean()<blur_f[mask==0].mean(): mask=1-mask
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        mask_u8=cv2.morphologyEx(mask*255,cv2.MORPH_CLOSE,k)
        labeled,_=ndi.label(mask_u8>0)
        drops=[]
        for rp in measure.regionprops(labeled):
            if rp.area<10 or rp.area>8000: continue
            cy,cx=rp.centroid
            drops.append({"x":float(cx),"y":float(cy),
                          "r":float(np.sqrt(rp.area/np.pi))})
        all_drops.append(drops)
        if (i+1)%5==0: print(f"  {i+1}/{len(frames)} — {len(drops)} gouttes")
    return _track(frames,all_drops,"ChanVese")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("video",type=Path)
    ap.add_argument("--method",choices=["log","chanvese","all"],default="all")
    ap.add_argument("--output-dir",type=Path,default=Path("."))
    ap.add_argument("--fps-extract",type=float,default=1.0)
    args=ap.parse_args()
    frames,_,src_dur=load_frames(args.video,fps_extract=args.fps_extract)
    print(f"Frames:{len(frames)}")
    runners={"log":run_log_nms,"chanvese":run_chanvese_blur}
    to_run=list(runners) if args.method=="all" else [args.method]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name in to_run:
        ann=runners[name](frames)
        save_video(ann,args.output_dir/f"02_seg_{name}.mp4",src_duration=src_dur)

if __name__=="__main__": main()