"""
01_background_subtraction.py — MOG2 / KNN / ViBe pour gouttes réflectives.

La soustraction de fond identifie les zones EN MOUVEMENT.
Ce masque guide detect_droplets : on ne détecte les gouttes QUE dans les
zones qui ont changé depuis le fond modélisé.
Cela évite de tracker les milliers de gouttes statiques du fond.
"""
from __future__ import annotations
import argparse
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from utils import (load_frames, draw_tracks, save_video,
                   make_color, hungarian_match, TRAIL_LEN, detect_droplets)

MOG2_HISTORY = 50;  MOG2_THRESH = 30
KNN_HISTORY  = 50;  KNN_THRESH  = 300
FG_DILATE    = 20   # pixels — dilate le masque avant-plan pour capturer les bords
MAX_DIST     = 65
VIBE_N=20; VIBE_R=25; VIBE_MIN_MATCH=2; VIBE_PHI=16


def _guided_detect(gray, fg_mask):
    """
    Détecte les gouttes uniquement dans les zones couvertes par fg_mask.
    Si le masque couvre > 60% (modèle de fond instable), détecte partout.
    """
    if fg_mask.mean() / 255.0 > 0.60:
        return detect_droplets(gray)
    k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (FG_DILATE, FG_DILATE))
    fg  = cv2.dilate(fg_mask, k)
    all_d = detect_droplets(gray)
    return [d for d in all_d if fg[int(d["y"]), int(d["x"])] > 0]


def _track(frames, all_drops, label):
    trails = defaultdict(lambda: deque(maxlen=TRAIL_LEN))
    colors: Dict[int,Tuple] = {}
    prev: List[dict] = []; next_id = 1; results = []
    for frame, curr in zip(frames, all_drops):
        matches = hungarian_match(prev, curr, MAX_DIST)
        id_map = {ci: prev[pi].get("id", next_id) for pi, ci in matches}
        for ci, det in enumerate(curr):
            if ci not in id_map:
                id_map[ci] = next_id; colors[next_id] = make_color(next_id); next_id += 1
            det["id"] = id_map[ci]
            trails[det["id"]].append((int(det["x"]), int(det["y"])))
        results.append(draw_tracks(frame, curr, trails, colors, label)); prev = curr
    return results


def run_mog2(frames):
    print("\n[MOG2]")
    bg = cv2.createBackgroundSubtractorMOG2(MOG2_HISTORY, MOG2_THRESH, False)
    for f in frames[:10]: bg.apply(f, learningRate=0.1)
    all_drops = []
    for i, frame in enumerate(frames):
        fg = bg.apply(frame, learningRate=0.02)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        all_drops.append(_guided_detect(frame, fg))
        if (i+1) % 10 == 0: print(f"  {i+1}/{len(frames)} — {len(all_drops[-1])} gouttes")
    return _track(frames, all_drops, "MOG2")


def run_knn(frames):
    print("\n[KNN]")
    bg = cv2.createBackgroundSubtractorKNN(KNN_HISTORY, KNN_THRESH, False)
    for f in frames[:10]: bg.apply(f, learningRate=0.1)
    all_drops = []
    for i, frame in enumerate(frames):
        fg = bg.apply(frame, learningRate=0.02)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        all_drops.append(_guided_detect(frame, fg))
        if (i+1) % 10 == 0: print(f"  {i+1}/{len(frames)} — {len(all_drops[-1])} gouttes")
    return _track(frames, all_drops, "KNN")


class _ViBe:
    def __init__(self, f):
        h,w=f.shape; self.h,self.w=h,w; self.rng=np.random.default_rng(42)
        self.s=np.zeros((VIBE_N,h,w),np.uint8)
        for n in range(VIBE_N):
            self.s[n]=np.roll(np.roll(f,self.rng.integers(-1,2),0),self.rng.integers(-1,2),1)
    def apply(self,f):
        close=(np.abs(self.s.astype(np.int16)-f.astype(np.int16))<VIBE_R).sum(0)
        fg=(close<VIBE_MIN_MATCH).astype(np.uint8)*255
        bg=fg==0; upd=(self.rng.integers(0,VIBE_PHI,(self.h,self.w))==0)&bg
        if upd.any():
            idx=np.flatnonzero(upd); ni=self.rng.integers(0,VIBE_N,len(idx))
            r=idx//self.w; c=idx%self.w; self.s[ni,r,c]=f[r,c]
            dy=self.rng.integers(-1,2,len(idx)); dx=self.rng.integers(-1,2,len(idx))
            nr=np.clip(r+dy,0,self.h-1); nc=np.clip(c+dx,0,self.w-1)
            self.s[self.rng.integers(0,VIBE_N,len(idx)),nr,nc]=f[r,c]
        return fg


def run_vibe(frames):
    print("\n[ViBe]")
    vibe=_ViBe(frames[0]); all_drops=[]
    for i, frame in enumerate(frames):
        fg=vibe.apply(frame); all_drops.append(_guided_detect(frame,fg))
        if (i+1)%10==0: print(f"  {i+1}/{len(frames)} — {len(all_drops[-1])} gouttes")
    return _track(frames, all_drops, "ViBe")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("video",type=Path)
    ap.add_argument("--method",choices=["mog2","knn","vibe","all"],default="all")
    ap.add_argument("--output-dir",type=Path,default=Path("."))
    ap.add_argument("--fps-extract",type=float,default=1.0)
    args=ap.parse_args()
    frames,_,src_dur=load_frames(args.video,fps_extract=args.fps_extract)
    print(f"Frames:{len(frames)}  résolution:{frames[0].shape}")
    runners={"mog2":run_mog2,"knn":run_knn,"vibe":run_vibe}
    to_run=list(runners) if args.method=="all" else [args.method]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name in to_run:
        ann=runners[name](frames)
        save_video(ann,args.output_dir/f"01_bg_{name}.mp4",src_duration=src_dur)

if __name__=="__main__": main()