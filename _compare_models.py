"""Compare cellpose detection BEFORE (base cyto3) vs AFTER (our fine-tuned model)
on one frame, same diameter. Outputs a full side-by-side + a zoomed side-by-side.
Run: .venv-cp3\\Scripts\\python.exe -X utf8 -u _compare_models.py [vid] [frame_index]
"""
import os, sys, glob
import cv2, numpy as np
from cellpose import models
from cellpose.utils import masks_to_outlines

VID  = sys.argv[1] if len(sys.argv) > 1 else "tiff4"
IDX  = int(sys.argv[2]) if len(sys.argv) > 2 else 500
DIAM = 110
RED  = (0, 0, 255)

fs = sorted(glob.glob(os.path.join(VID, "*.tiff")))
img = cv2.imread(fs[IDX])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def run(model, label):
    masks = model.eval(gray, diameter=DIAM, channels=[0, 0],
                       flow_threshold=0.6, cellprob_threshold=-1.0)[0]
    out = img.copy()
    out[masks_to_outlines(masks)] = RED
    n = int(masks.max())
    cv2.putText(out, f"{label}: {n} gouttes", (25, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3, cv2.LINE_AA)
    return out, n

before, nb = run(models.CellposeModel(gpu=False, model_type="cyto3"), "AVANT (cyto3 base)")
_mdls = [m for m in glob.glob("training/train_tiff/models/*") if not m.endswith("_train_losses.npy")]
MODEL = max(_mdls, key=os.path.getmtime)
after, na = run(models.CellposeModel(gpu=False, pretrained_model=MODEL), "APRES (fine-tuned)")

TAG = f"{VID}_f{IDX}"
cv2.imwrite(f"compare_{TAG}_full.png", np.hstack([before, after]))
if len(sys.argv) > 6:                                 # optional zoom region: x0 y0 x1 y1
    x0, y0, x1, y1 = (int(sys.argv[i]) for i in range(3, 7))
else:
    x0, y0, x1, y1 = 500, 250, 1400, 950
zoom = np.hstack([before[y0:y1, x0:x1], after[y0:y1, x0:x1]])
cv2.imwrite(f"compare_{TAG}_zoom.png", zoom)
print(f"{VID} frame {IDX}: AVANT cyto3 = {nb} gouttes | APRES fine-tuned = {na} gouttes", flush=True)
print(f"-> compare_{TAG}_full.png, compare_{TAG}_zoom.png", flush=True)
