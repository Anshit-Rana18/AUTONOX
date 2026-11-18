import cv2, sys

cap = cv2.VideoCapture(sys.argv[1])
if not cap.isOpened():
    raise SystemExit(f"Cannot open {sys.argv[1]}")
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(f"{w}x{h}  {fps:.1f} FPS  {n / fps:.1f} s  ({int(n)} frames)")
