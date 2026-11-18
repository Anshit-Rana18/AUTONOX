import cv2, json, sys, numpy as np
from shapely.geometry import Polygon
sys.path.insert(0, ".")
from video.display import fit

video_path = sys.argv[1]
out_path = sys.argv[2] if len(sys.argv) > 2 else "configs/speed_calibration.json"

cap = cv2.VideoCapture(video_path)
ok, frame = cap.read()
cap.release()
if not ok:
    raise RuntimeError(f"Cannot read frame from {video_path}")

H, W = frame.shape[:2]
_, scale = fit(frame)
pts = []

def redraw():
    img = frame.copy()
    thick, rad, font = max(2, int(2 / scale)), max(5, int(6 / scale)), 0.9 / scale
    for i, p in enumerate(pts):
        cv2.circle(img, tuple(p), rad, (0, 0, 255), -1)
        cv2.putText(img, str(i + 1), (p[0] + rad, p[1] - rad),
                    cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 255), thick)
    if len(pts) > 1:
        cv2.polylines(img, [np.array(pts)], len(pts) == 4, (0, 255, 255), thick)
    cv2.imshow("speed calibration", fit(img)[0])

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
        pts.append([int(x / scale), int(y / scale)])
        redraw()

cv2.namedWindow("speed calibration")
cv2.setMouseCallback("speed calibration", on_mouse)
redraw()
print("Click 4 ground points in this order: 1 near-left, 2 near-right, 3 far-right, 4 far-left.")
print("Points 1-2 are one edge ACROSS the road (width). Points 2-3 run ALONG the road (length).")
print("u: undo | s: save (after 4 points) | q: quit")

while True:
    key = cv2.waitKey(0) & 0xFF
    if key == ord("u") and pts:
        pts.pop(); redraw()
    elif key == ord("s") and len(pts) == 4:
        if not Polygon(pts).is_valid:
            print("The 4 points cross each other. Undo and click around the rectangle in order.")
            continue
        w = float(input("Real distance between points 1 and 2, ACROSS the road (metres): "))
        l = float(input("Real distance between points 2 and 3, ALONG the road (metres): "))
        json.dump({"image_points": pts, "width_m": w, "length_m": l,
                   "video": video_path, "frame_size": [W, H]}, open(out_path, "w"), indent=2)
        print(f"Saved {out_path}")
        break
    elif key == ord("q"):
        break
cv2.destroyAllWindows()

