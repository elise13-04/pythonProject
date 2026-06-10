"""Two-scale Cellpose test: small pass (diam 25) + large pass (diam 90) on Video1_1_late."""
import os
import cv2
import numpy as np
from cellpose import models
from cellpose.utils import masks_to_outlines

model = models.Cellpose(gpu=False, model_type="cyto3")

name = "Video1_1_late.png"
img = cv2.imread(os.path.join("_frames", name))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

out = img.copy()
for diam, color in [(25, (0, 0, 255)), (90, (0, 255, 255))]:
    masks, _, _, _ = model.eval(gray, diameter=diam, channels=[0, 0],
                                flow_threshold=0.6, cellprob_threshold=-1.0)
    n = int(masks.max())
    outlines = masks_to_outlines(masks)
    out[outlines] = color
    print("diam", diam, ":", n, "objects")

cv2.imwrite(os.path.join("_frames", name.replace(".png", "_cellpose_multi.png")), out)
print("done")
