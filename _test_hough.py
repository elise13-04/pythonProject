"""Run the detect_circles pipeline from stage_toronto/tracker.py on sample frames."""
import sys, os
import cv2
import numpy as np

sys.path.insert(0, "stage_toronto")
import importlib.util
spec = importlib.util.spec_from_file_location("tracker", os.path.join("stage_toronto", "tracker.py"))
tracker = importlib.util.module_from_spec(spec)
# tracker.py runs argparse at import under __main__ guard only, so import is safe
spec.loader.exec_module(tracker)

frames = ["Video1_1_late.png", "Video1_3_mid.png", "Video1_5_late.png", "Video1_1_early.png"]
for name in frames:
    img = cv2.imread(os.path.join("_frames", name))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = tracker.detect_circles(gray)
    out = img.copy()
    for x, y, r in circles:
        cv2.circle(out, (int(x), int(y)), int(r), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join("_frames", name.replace(".png", "_hough.png")), out)
    print(name, len(circles), "circles")
print("done")
