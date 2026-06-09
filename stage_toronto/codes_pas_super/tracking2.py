"""
droplet_tracker.py (Version Ultra-Rapide)
"""

from __future__ import annotations

import argparse
import colorsys
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage import morphology, segmentation, measure, filters
from skimage.feature import peak_local_max

# ═══════════════════════════════ CONFIG ══════════════════════════════════════
FRAMES_PER_SEC = 1
CLAHE_CLIP = 3
CLAHE_GRID = 8

MOG2_HISTORY = 50
MOG2_THRESHOLD = 40
MOG2_SHADOWS = False

OPEN_RADIUS = 2
CLOSE_RADIUS = 3
MIN_AREA = 80
MAX_AREA = 40_000
MIN_CIRCULARITY = 0.2
WATERSHED_SIGMA = 2
MAX_DIST = 80
IOU_WEIGHT = 0.6
MAX_LOST = 3
MIN_HITS = 2
TRAIL_LENGTH = 30
CONTOUR_COLOR = (0, 255, 100)
TEXT_COLOR = (255, 255, 0)
TRAIL_FADE = True
SHOW_BBOX = False
EVENT_DURATION = 8
OUTPUT_FPS = 1


# ═════════════════════════════════════════════════════════════════════════════


def _make_color(track_id: int) -> Tuple[int, int, int]:
    hue = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


@dataclass
class Droplet:
    label: int
    centroid: Tuple[float, float]
    area: float
    bbox: Tuple[int, int, int, int]
    contour_pts: np.ndarray
    mask: np.ndarray


@dataclass
class Track:
    track_id: int
    color: Tuple[int, int, int]
    trail: deque = field(default_factory=lambda: deque(maxlen=TRAIL_LENGTH))
    hits: int = 0
    lost: int = 0
    last_drop: Optional[Droplet] = None
    active: bool = True

    def update(self, drop: Droplet):
        self.trail.append((int(drop.centroid[0]), int(drop.centroid[1])))
        self.hits += 1
        self.lost = 0
        self.last_drop = drop

    def mark_lost(self):
        self.lost += 1
        if self.lost >= MAX_LOST:
            self.active = False


@dataclass
class Event:
    kind: str
    position: Tuple[int, int]
    frame_idx: int


# ──────────────────────────── Prétraitement ──────────────────────────────────
def preprocess(frame_bgr: np.ndarray, clahe: cv2.CLAHE) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    eq = clahe.apply(gray)
    # OPTIMISATION 1 : Remplacement du Bilateral par un Gaussian Blur
    # C'est ~20x plus rapide et largement suffisant pour le seuillage adaptatif
    blur = cv2.GaussianBlur(eq, (7, 7), 2)
    return blur


# ──────────────────────────── Segmentation ───────────────────────────────────
def segment(preprocessed: np.ndarray, fg_mask: np.ndarray) -> List[Droplet]:
    thr = cv2.adaptiveThreshold(
        preprocessed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=31, C=-4
    )

    fg_clean = cv2.dilate(fg_mask, np.ones((3, 3), np.uint8), iterations=1)
    combined = cv2.bitwise_or(thr, fg_clean)

    # OPTIMISATION 2 : Morphologie OpenCV (Fait disparaitre tes erreurs et est 10x plus rapide)
    se_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_RADIUS * 2 + 1, OPEN_RADIUS * 2 + 1))
    se_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_RADIUS * 2 + 1, CLOSE_RADIUS * 2 + 1))

    bw = cv2.morphologyEx(combined, cv2.MORPH_OPEN, se_open)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, se_close)

    # Retire les tout petits pixels de bruit restants
    bw_bool = morphology.remove_small_objects(bw > 0, MIN_AREA)
    bw = bw_bool.astype(np.uint8) * 255

    # OPTIMISATION 3 : Distance Transform par OpenCV (Ultra-rapide)
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
    dist_s = filters.gaussian(dist, sigma=WATERSHED_SIGMA)

    min_d = max(4, int(np.sqrt(MIN_AREA / np.pi) * 0.6))
    coords = peak_local_max(dist_s, min_distance=min_d, labels=bw_bool)

    seeds = np.zeros(dist.shape, dtype=int)
    for i, (r, c) in enumerate(coords):
        seeds[r, c] = i + 1

    labels_ws = segmentation.watershed(-dist_s, seeds, mask=bw_bool, compactness=0.05)

    droplets: List[Droplet] = []
    props = measure.regionprops(labels_ws)
    for rp in props:
        area = rp.area
        if area < MIN_AREA or area > MAX_AREA:
            continue
        perim = rp.perimeter
        if perim < 1:
            continue
        circ = (4 * np.pi * area) / (perim ** 2)
        if circ < MIN_CIRCULARITY:
            continue

        mask_full = (labels_ws == rp.label).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue

        cy, cx = rp.centroid
        r0, c0, r1, c1 = rp.bbox
        region_mask = rp.image

        droplets.append(Droplet(
            label=rp.label,
            centroid=(float(cx), float(cy)),
            area=float(area),
            bbox=(r0, c0, r1, c1),
            contour_pts=cnts[0],
            mask=region_mask,
        ))
    return droplets


# ──────────────────────────── Tracking ───────────────────────────────────────

def _iou(d1: Droplet, d2: Droplet) -> float:
    """
    OPTIMISATION MAJEURE : Ne compare QUE la zone d'intersection des boîtes.
    Cela évite d'allouer des dizaines de milliers de matrices 1920x1080 par frame.
    """
    r0_1, c0_1, r1_1, c1_1 = d1.bbox
    r0_2, c0_2, r1_2, c1_2 = d2.bbox

    inter_r0 = max(r0_1, r0_2)
    inter_c0 = max(c0_1, c0_2)
    inter_r1 = min(r1_1, r1_2)
    inter_c1 = min(c1_1, c1_2)

    # Pas d'intersection des boîtes = pas de chevauchement
    if inter_r0 >= inter_r1 or inter_c0 >= inter_c1:
        return 0.0

        # Découpe des micro-masques
    sub_m1 = d1.mask[inter_r0 - r0_1: inter_r1 - r0_1, inter_c0 - c0_1: inter_c1 - c0_1]
    sub_m2 = d2.mask[inter_r0 - r0_2: inter_r1 - r0_2, inter_c0 - c0_2: inter_c1 - c0_2]

    inter = np.logical_and(sub_m1, sub_m2).sum()
    union = d1.area + d2.area - inter
    return float(inter) / float(union + 1e-9)


def associate(tracks: Dict[int, Track],
              detections: List[Droplet]) -> Tuple[Dict[int, int], List[int], List[int]]:
    active_ids = [tid for tid, tr in tracks.items() if tr.active]
    if not active_ids or not detections:
        return {}, list(range(len(detections))), active_ids

    n_tr = len(active_ids)
    n_dt = len(detections)
    cost = np.full((n_tr, n_dt), fill_value=1e9)

    for i, tid in enumerate(active_ids):
        tr = tracks[tid]
        if tr.last_drop is None:
            continue
        cx0, cy0 = tr.last_drop.centroid
        for j, det in enumerate(detections):
            cx1, cy1 = det.centroid
            dist_cost = np.hypot(cx1 - cx0, cy1 - cy0)
            if dist_cost > MAX_DIST:
                continue
            iou_score = _iou(tr.last_drop, det)
            cost[i, j] = (1 - IOU_WEIGHT) * dist_cost + IOU_WEIGHT * (1 - iou_score) * MAX_DIST

    row_ind, col_ind = linear_sum_assignment(cost)
    matched: Dict[int, int] = {}
    unmatched_tr = set(active_ids)
    unmatched_det = set(range(n_dt))

    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < 1e8:
            tid = active_ids[r]
            matched[tid] = c
            unmatched_tr.discard(tid)
            unmatched_det.discard(c)

    return matched, list(unmatched_det), list(unmatched_tr)


def detect_events(prev_drops, curr_drops, matched, det_idx_to_new_id, frame_idx):
    events = []
    det_count = defaultdict(list)
    for tid, di in matched.items():
        det_count[di].append(tid)

    for di, tids in det_count.items():
        if len(tids) > 1 and di < len(curr_drops):
            cx, cy = curr_drops[di].centroid
            events.append(Event("MERGE", (int(cx), int(cy)), frame_idx))
    return events


# ──────────────────────────── Dessin ─────────────────────────────────────────
def draw_frame(frame_bgr, drops, tracks, det_to_track, events, frame_idx, active_events):
    out = frame_bgr.copy()

    for tr in tracks.values():
        if not tr.active or tr.hits < MIN_HITS: continue
        pts = list(tr.trail)
        if len(pts) < 2: continue
        for k in range(1, len(pts)):
            if TRAIL_FADE:
                alpha = k / len(pts)
                col = tuple(int(c * alpha) for c in tr.color)
            else:
                col = tr.color
            cv2.line(out, pts[k - 1], pts[k], col, thickness=2, lineType=cv2.LINE_AA)

    for j, drop in enumerate(drops):
        tid = det_to_track.get(j)
        if tid is None:
            color_c = CONTOUR_COLOR
            label_s = "?"
        else:
            tr = tracks[tid]
            if tr.hits < MIN_HITS: continue
            color_c = tr.color
            label_s = str(tid)

        cv2.drawContours(out, [drop.contour_pts], -1, color_c, 2, lineType=cv2.LINE_AA)
        cx, cy = int(drop.centroid[0]), int(drop.centroid[1])
        cv2.circle(out, (cx, cy), 3, color_c, -1)
        if SHOW_BBOX:
            r0, c0, r1, c1 = drop.bbox
            cv2.rectangle(out, (c0, r0), (c1, r1), color_c, 1)
#        cv2.putText(out, label_s, (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)

    for ev in active_events:
        age = frame_idx - ev.frame_idx
        alpha = max(0.0, 1.0 - age / EVENT_DURATION)
        cx, cy = ev.position
        e_color = (0, 80, 255) if ev.kind == "MERGE" else (255, 140, 0)
        e_color = tuple(int(c * alpha) for c in e_color)
        cv2.putText(out, ev.kind, (cx + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, e_color, 2, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 12, e_color, 2, lineType=cv2.LINE_AA)

    n_visible = sum(1 for tr in tracks.values() if tr.active and tr.hits >= MIN_HITS)
    hud_lines = [f"Frame : {frame_idx}", f"Gouttes: {n_visible}"]
    for i, line in enumerate(hud_lines):
        cv2.putText(out, line, (10, 22 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    return out


# ──────────────────────────── Pipeline principal ──────────────────────────────
def run(video_path: Path, output_path: Path, debug_frames: bool = False):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir : {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = src_nframes / src_fps

    print(f"Vidéo source  : {video_path.name}")
    print(f"Résolution    : {src_width}×{src_height}  |  {src_fps:.1f} FPS")

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_GRID, CLAHE_GRID))
    bg_sub = cv2.createBackgroundSubtractorMOG2(history=MOG2_HISTORY, varThreshold=MOG2_THRESHOLD,
                                                detectShadows=MOG2_SHADOWS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, OUTPUT_FPS, (src_width, src_height))

    dbg_dir = None
    if debug_frames:
        dbg_dir = output_path.parent / f"{output_path.stem}_debug"
        dbg_dir.mkdir(exist_ok=True)

    tracks: Dict[int, Track] = {}
    next_id: int = 1
    all_events: List[Event] = []
    prev_drops: List[Droplet] = []
    frame_idx: int = 0

    step_sec = 1.0 / FRAMES_PER_SEC
    timestamps = np.arange(0, duration_s, step_sec)
    total = len(timestamps)

    for k, t in enumerate(timestamps):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok: continue

        prep = preprocess(frame, clahe)
        prep3 = cv2.cvtColor(prep, cv2.COLOR_GRAY2BGR)
        fg_mask = bg_sub.apply(prep3)
        fg_mask = (fg_mask == 255).astype(np.uint8) * 255

        drops = segment(prep, fg_mask)
        matched, new_det_idx, lost_ids = associate(tracks, drops)
        det_to_track: Dict[int, int] = {}

        for tid, di in matched.items():
            tracks[tid].update(drops[di])
            det_to_track[di] = tid

        for tid in lost_ids:
            tracks[tid].mark_lost()

        for di in new_det_idx:
            new_track = Track(track_id=next_id, color=_make_color(next_id))
            new_track.update(drops[di])
            tracks[next_id] = new_track
            det_to_track[di] = next_id
            next_id += 1

        new_events = detect_events(prev_drops, drops, matched, det_to_track, frame_idx)
        all_events.extend(new_events)
        active_ev = [ev for ev in all_events if frame_idx - ev.frame_idx < EVENT_DURATION]

        annotated = draw_frame(frame, drops, tracks, det_to_track, new_events, frame_idx, active_ev)
        writer.write(annotated)

        if debug_frames and dbg_dir is not None:
            prep_bgr = cv2.cvtColor(prep, cv2.COLOR_GRAY2BGR)
            fg_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
            panel = np.hstack([prep_bgr, fg_bgr, annotated])
            cv2.imwrite(str(dbg_dir / f"frame_{frame_idx:04d}.png"), panel)

        prev_drops = drops
        frame_idx += 1

        # Affichage à CHAQUE image pour que tu voies que ça tourne vite !
        n_active = sum(1 for tr in tracks.values() if tr.active)
        print(f"  [{k + 1:4d}/{total}]  t={t:6.1f}s  détections={len(drops):3d}  pistes actives={n_active:3d}")

    cap.release()
    writer.release()
    print(f"\nTerminé. Vidéo de sortie : {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--debug-frames", action="store_true")
    args = ap.parse_args()
    out = args.output or args.video.parent / f"{args.video.stem}_tracked.mp4"
    run(args.video, out, debug_frames=args.debug_frames)


if __name__ == "__main__":
    main()