import cv2, os
from collections import deque

class ClipRecorder:
    """Rolling pre-event buffer plus post-event frames, written as one MP4 per event.
    Frames are downscaled and JPEG-compressed in memory to keep RAM use small."""

    def __init__(self, fps, pre_s=3.0, post_s=3.0, width=960, quality=80):
        self.fps = fps
        self.pre = deque(maxlen=max(1, int(fps * pre_s)))
        self.post_n = max(1, int(fps * post_s))
        self.width, self.q = width, quality
        self.jobs, self.done = [], {}

    def _enc(self, frame):
        s = self.width / frame.shape[1]
        small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, self.q])[1]

    def push(self, idx, frame):
        item = (idx, self._enc(frame))
        self.pre.append(item)
        for j in self.jobs:
            j["frames"].append(item)
            j["post_left"] -= 1
        for j in [j for j in self.jobs if j["post_left"] <= 0]:
            self._write(j)
            self.jobs.remove(j)

    def start(self, event_id, path):
        """Call AFTER push() for the event frame, so that frame is the last pre-event frame."""
        self.jobs.append({"id": event_id, "path": path,
                          "frames": list(self.pre), "post_left": self.post_n})

    def _write(self, j):
        if not j["frames"]:
            return
        first = cv2.imdecode(j["frames"][0][1], cv2.IMREAD_COLOR)
        h, w = first.shape[:2]
        vw = cv2.VideoWriter(j["path"], cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        for _, buf in j["frames"]:
            vw.write(cv2.imdecode(buf, cv2.IMREAD_COLOR))
        vw.release()
        self.done[j["id"]] = {"file": os.path.basename(j["path"]),
                              "first_frame": j["frames"][0][0], "last_frame": j["frames"][-1][0],
                              "frames": len(j["frames"]), "width": w, "height": h}

    def flush(self):
        """Write clips still waiting for their post-event frames (they will be shorter)."""
        for j in self.jobs:
            self._write(j)
        self.jobs = []

