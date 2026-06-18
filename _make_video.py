"""Step 1: assemble all TIFF frames in video_tiff1/ into a raw video at 100 fps.
No detection here - just stitching. Cellpose + tracking come later at a reduced rate.
Run: .venv-cp3\\Scripts\\python.exe -u _make_video.py
"""
import cv2, glob, os

SRC = "video_tiff1"
OUT = "video_tiff1.mp4"
FPS = 100.0

frames = sorted(glob.glob(os.path.join(SRC, "*.tiff")))
assert frames, f"no .tiff found in {SRC}/"
h, w = cv2.imread(frames[0]).shape[:2]
print(f"{len(frames)} frames, {w}x{h} -> {OUT} @ {FPS:.0f} fps")

vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
for i, fp in enumerate(frames):
    vw.write(cv2.imread(fp))
    if (i + 1) % 500 == 0:
        print(f"  {i + 1}/{len(frames)}")
vw.release()
print(f"done -> {OUT}  ({len(frames)/FPS:.1f} s of video, {os.path.getsize(OUT)/1e6:.0f} MB)")
