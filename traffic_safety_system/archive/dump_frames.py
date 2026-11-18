import cv2, sys, numpy as np
from video.motion import MotionCompensator
from lanes.lane_assignment import LaneAssigner

video, lanes_path = sys.argv[1], sys.argv[2]
PTS = {113: (3588, 2152), 123: (3395, 1999), 132: (3203, 1883)}   # frame -> contact point from your output

cap = cv2.VideoCapture(video)
ok, f0 = cap.read()
comp = MotionCompensator(f0)
lanes = LaneAssigner(lanes_path)
H, W = f0.shape[:2]
print(f"video {W}x{H}")

for idx, (x, y) in PTS.items():
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    comp.update(fr)
    img = fr.copy()
    for name, poly in lanes.lanes:
        p = comp.poly_to_cur(np.array(poly.exterior.coords)).astype(np.int32)
        cv2.polylines(img, [p], True, (0, 255, 255), 3)
        cv2.putText(img, name, tuple(p[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 3)
    cv2.circle(img, (x, y), 14, (0, 0, 255), -1)
    crop = img[max(y - 700, 0):min(y + 200, H), max(x - 800, 0):min(x + 300, W)]
    cv2.imwrite(f"data/outputs/dump_{idx}.png", crop)
    print(f"frame {idx}: dot {W - x}px from right edge, {H - y}px from bottom edge, shift={comp.max_shift_px():.0f}px")

