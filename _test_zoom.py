"""Zoomed side-by-side crops (original | cellpose masks) to judge per-droplet quality."""
import os
import cv2
import numpy as np
from cellpose import models

model = models.Cellpose(gpu=False, model_type="cyto3")

# (frame, x0, y0, size, diameter hint)
jobs = [
    ("Video1_3_mid.png",  700, 120, 400, 12),   # dense small droplets on electrodes
    ("Video1_1_late.png", 1050, 60, 400, 25),   # dense region, mixed small/medium
    ("Video1_1_late.png", 250, 60, 450, 45),    # left column: medium + the big drop
]

rng = np.random.default_rng(0)

for i, (name, x0, y0, sz, diam) in enumerate(jobs):
    img = cv2.imread(os.path.join("_frames", name))
    crop = img[y0:y0 + sz, x0:x0 + sz]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    masks, _, _, _ = model.eval(gray, diameter=diam, channels=[0, 0],
                                flow_threshold=0.6, cellprob_threshold=-1.0)
    n = int(masks.max())

    overlay = crop.copy()
    colors = rng.integers(60, 255, size=(n + 1, 3))
    fill = np.zeros_like(crop)
    for m in range(1, n + 1):
        fill[masks == m] = colors[m]
    overlay = cv2.addWeighted(crop, 0.55, fill, 0.45, 0)

    side = np.hstack([crop, overlay])
    side = cv2.resize(side, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST)
    out = os.path.join("_frames", "zoom%d_%s_d%d.png" % (i, name.split(".")[0], diam))
    cv2.imwrite(out, side)
    print(out, n, "droplets")
print("done")
