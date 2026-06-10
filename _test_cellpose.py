"""Run pretrained Cellpose cyto3 on sample frames and draw mask outlines."""
import os
import cv2
import numpy as np
from cellpose import models
from cellpose.utils import masks_to_outlines

model = models.Cellpose(gpu=False, model_type="cyto3")

# (frame, expected droplet diameter in px) - rough per-frame guess; None lets cellpose estimate
jobs = [
    ("Video1_1_late.png", 25),
    ("Video1_3_mid.png", 12),
]

for name, diam in jobs:
    img = cv2.imread(os.path.join("_frames", name))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    masks, flows, styles, diams = model.eval(gray, diameter=diam, channels=[0, 0],
                                             flow_threshold=0.6, cellprob_threshold=-1.0)
    n = masks.max()
    outlines = masks_to_outlines(masks)
    out = img.copy()
    out[outlines] = (0, 0, 255)
    cv2.imwrite(os.path.join("_frames", name.replace(".png", "_cellpose.png")), out)
    print(name, n, "droplets, est diam", diams)
print("done")
