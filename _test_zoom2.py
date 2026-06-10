"""Cleaner zoom renders: native pixels, integer 2x nearest upscale, outline-only overlay."""
import os
import cv2
import numpy as np
from cellpose import models
from cellpose.utils import masks_to_outlines

model = models.Cellpose(gpu=False, model_type="cyto3")

jobs = [
    ("Video1_3_mid.png",  700, 120, 400, 12),
    ("Video1_1_late.png", 1050, 60, 400, 25),
    ("Video1_1_late.png", 250, 60, 450, 45),
]

for i, (name, x0, y0, sz, diam) in enumerate(jobs):
    img = cv2.imread(os.path.join("_frames", name))
    crop = img[y0:y0 + sz, x0:x0 + sz]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    masks, _, _, _ = model.eval(gray, diameter=diam, channels=[0, 0],
                                flow_threshold=0.6, cellprob_threshold=-1.0)
    n = int(masks.max())
    overlay = crop.copy()
    overlay[masks_to_outlines(masks)] = (0, 0, 255)
    side = np.hstack([crop, overlay])
    side = cv2.resize(side, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    out = os.path.join("_frames", "zoomB%d_%s_d%d.png" % (i, name.split(".")[0], diam))
    cv2.imwrite(out, side)
    print(out, n, "droplets")
print("done")
