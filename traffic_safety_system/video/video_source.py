import cv2

class VideoSource:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.start = 0

    def frames(self):
        idx = self.start
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield idx, idx / self.fps, frame   # frame index, video timestamp (s), image
            idx += 1

    def seek(self, frame_idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self.start = frame_idx

    def release(self):
        self.cap.release()

