"""
03_blob_detection.py — Détection de blobs pour gouttes réflectives.

Toutes les méthodes utilisent la même base : image floutée (sigma=3)
qui transforme chaque goutte en blob homogène, puis LoG / DoG / DoH / SBD.
Un NMS final assure un seul cercle par goutte.

Hough Circles supprimé (incompatible avec les gouttes réflectives).
"""
from __future__ import annotations
import argparse
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from skimage.feature import blob_log, blob_dog, blob_doh

from utils import (load_frames, draw_tracks, save_video,
                   make_color, hungarian_match, TRAIL_LEN,
                   detect_droplets, _clahe, _nms_blobs,
                   DETECT_BLUR_SIGMA, DETECT_MIN_SIGMA, DETECT_MAX_SIGMA,
                   DETECT_NUM_SIGMA, DETECT_OVERLAP)

MAX_DIST     = 65
THRESH_LOG   = 0.008
THRESH_DOG   = 0.006
THRESH_DOH   = 0.001
SBD_MIN_AREA = 30; SBD_MAX_AREA = 8000
SBD_MIN_CIRC = 0.25; SBD_MIN_CONV = 0.5; SBD_MIN_IN = 0.2


def _preprocess(frame):
    eq=_clahe.apply(frame)
    blur=cv2.GaussianBlur(eq,(0,0),sigmaX=DETECT_BLUR_SIGMA)
    return blur.astype(np.float64)/255.0


def _to_dets(blobs):
    return [{"x":float(b[1]),"y":float(b[0]),"r":max(float(b[2]*np.sqrt(2)),2.0)} for b in blobs]


def _track(frames,all_dets,label):
    trails=defaultdict(lambda: deque(maxlen=TRAIL_LEN)); colors={}; prev=[]; next_id=1; results=[]
    for frame,curr in zip(frames,all_dets):
        matches=hungarian_match(prev,curr,MAX_DIST); id_map={ci:prev[pi].get("id",next_id) for pi,ci in matches}
        for ci,det in enumerate(curr):
            if ci not in id_map: id_map[ci]=next_id; colors[next_id]=make_color(next_id); next_id+=1
            det["id"]=id_map[ci]; trails[det["id"]].append((int(det["x"]),int(det["y"])))
        results.append(draw_tracks(frame,curr,trails,colors,label)); prev=curr
    return results


def run_log(frames):
    print("\n[LoG]")
    all_dets=[]
    for i,f in enumerate(frames):
        bf=_preprocess(f); blobs=blob_log(bf,min_sigma=DETECT_MIN_SIGMA,max_sigma=DETECT_MAX_SIGMA,
            num_sigma=DETECT_NUM_SIGMA,threshold=THRESH_LOG,overlap=DETECT_OVERLAP)
        blobs=_nms_blobs(blobs) if len(blobs) else blobs
        all_dets.append(_to_dets(blobs))
        if (i+1)%10==0: print(f"  {i+1}/{len(frames)} — {len(all_dets[-1])}")
    return _track(frames,all_dets,"LoG")


def run_dog(frames):
    print("\n[DoG]")
    all_dets=[]
    for i,f in enumerate(frames):
        bf=_preprocess(f); blobs=blob_dog(bf,min_sigma=DETECT_MIN_SIGMA,max_sigma=DETECT_MAX_SIGMA,
            threshold=THRESH_DOG,overlap=DETECT_OVERLAP)
        blobs=_nms_blobs(blobs) if len(blobs) else blobs
        all_dets.append(_to_dets(blobs))
        if (i+1)%10==0: print(f"  {i+1}/{len(frames)} — {len(all_dets[-1])}")
    return _track(frames,all_dets,"DoG")


def run_doh(frames):
    print("\n[DoH]")
    all_dets=[]
    for i,f in enumerate(frames):
        bf=_preprocess(f); blobs=blob_doh(bf,min_sigma=DETECT_MIN_SIGMA,max_sigma=DETECT_MAX_SIGMA,
            num_sigma=DETECT_NUM_SIGMA,threshold=THRESH_DOH,overlap=DETECT_OVERLAP)
        blobs=_nms_blobs(blobs) if len(blobs) else blobs
        all_dets.append(_to_dets(blobs))
        if (i+1)%10==0: print(f"  {i+1}/{len(frames)} — {len(all_dets[-1])}")
    return _track(frames,all_dets,"DoH")


def run_simple_blob(frames):
    print("\n[SimpleBlobDetector]")
    params=cv2.SimpleBlobDetector_Params()
    params.minThreshold=10; params.maxThreshold=220; params.thresholdStep=15
    params.filterByArea=True; params.minArea=SBD_MIN_AREA; params.maxArea=SBD_MAX_AREA
    params.filterByCircularity=True; params.minCircularity=SBD_MIN_CIRC
    params.filterByConvexity=True; params.minConvexity=SBD_MIN_CONV
    params.filterByInertia=True; params.minInertiaRatio=SBD_MIN_IN
    params.minRepeatability=2
    det=cv2.SimpleBlobDetector_create(params)
    all_dets=[]
    for i,f in enumerate(frames):
        eq=_clahe.apply(f)
        blur=cv2.GaussianBlur(eq,(0,0),sigmaX=DETECT_BLUR_SIGMA).astype(np.uint8)
        inv=255-blur
        kps=det.detect(inv)
        dets=[{"x":float(kp.pt[0]),"y":float(kp.pt[1]),"r":float(kp.size/2)} for kp in kps]
        all_dets.append(dets)
        if (i+1)%10==0: print(f"  {i+1}/{len(frames)} — {len(dets)}")
    return _track(frames,all_dets,"SBD")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("video",type=Path)
    ap.add_argument("--method",choices=["log","dog","doh","simpleblob","all"],default="all")
    ap.add_argument("--output-dir",type=Path,default=Path(".")); ap.add_argument("--fps-extract",type=float,default=1.0)
    args=ap.parse_args()
    frames,_,src_dur=load_frames(args.video,fps_extract=args.fps_extract)
    print(f"Frames:{len(frames)}")
    runners={"log":run_log,"dog":run_dog,"doh":run_doh,"simpleblob":run_simple_blob}
    to_run=list(runners) if args.method=="all" else [args.method]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name in to_run:
        ann=runners[name](frames)
        save_video(ann,args.output_dir/f"03_blob_{name}.mp4",src_duration=src_dur)

if __name__=="__main__": main()