import cv2, os

os.makedirs("_frames", exist_ok=True)
for vid in ["Video1_1.mp4", "Video1_2.mp4", "Video1_3.mp4", "Video1_4.mp4", "Video1_5.mp4"]:
    p = os.path.join("stage_toronto", vid)
    cap = cv2.VideoCapture(p)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(vid, "%dx%d" % (w, h), "%.1f fps" % fps, n, "frames", "%.1f s" % (n / fps if fps else 0))
    for frac, tag in [(0.05, "early"), (0.5, "mid"), (0.95, "late")]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * frac))
        ret, f = cap.read()
        if ret:
            cv2.imwrite(os.path.join("_frames", vid.replace(".mp4", "") + "_" + tag + ".png"), f)
    cap.release()
print("done")
