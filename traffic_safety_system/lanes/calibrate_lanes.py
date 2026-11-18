import cv2, json, sys, numpy as np
sys.path.insert(0, ".")
from video.display import fit

video_path = sys.argv[1] if len(sys.argv) > 1 else "data/videos/test1.mp4"
out_path = sys.argv[2] if len(sys.argv) > 2 else "configs/lanes.json"

cap = cv2.VideoCapture(video_path)
ok, frame = cap.read()
cap.release()
if not ok:
    raise RuntimeError(f"Cannot read frame from {video_path}")

H, W = frame.shape[:2]
_, scale = fit(frame)          # scale used only for display
print(f"Video {W}x{H}, display scale {scale:.2f}")

lanes, current = [], []

def redraw():
    img = frame.copy()
    thick = max(2, int(2 / scale))
    rad = max(4, int(5 / scale))
    font = 0.8 / scale
    for lane in lanes:
        pts = np.array(lane["polygon"])
        cv2.polylines(img, [pts], True, (0, 255, 0), thick)
        cv2.putText(img, lane["name"], tuple(lane["polygon"][0]),
                    cv2.FONT_HERSHEY_SIMPLEX, font, (0, 255, 0), thick)
    for p in current:
        cv2.circle(img, tuple(p), rad, (0, 0, 255), -1)
    if len(current) > 1:
        cv2.polylines(img, [np.array(current)], False, (0, 0, 255), thick)
    cv2.imshow("calibrate", fit(img)[0])

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        current.append([int(x / scale), int(y / scale)])   # back to video pixels
        redraw()

cv2.namedWindow("calibrate")
cv2.setMouseCallback("calibrate", on_mouse)
redraw()

print("Left click: add point | n: finish lane | u: undo point | s: save and quit | q: quit without saving")
while True:
    key = cv2.waitKey(0) & 0xFF
    if key == ord("n") and len(current) >= 3:
        lanes.append({"name": f"Lane {len(lanes) + 1}", "polygon": current.copy()})
        current.clear()
        redraw()
    elif key == ord("u") and current:
        current.pop()
        redraw()
    elif key == ord("s"):
        json.dump({"lanes": lanes}, open(out_path, "w"), indent=2)
        print(f"Saved {len(lanes)} lanes to {out_path}")
        break
    elif key == ord("q"):
        break

cv2.destroyAllWindows()

