import json
import cv2, numpy as np
from collections import deque
from dataclasses import dataclass, asdict

@dataclass
class SpeedEvent:
    track_id: int
    speed_kmh: float
    limit_kmh: float
    ts: float
    frame: int

    def to_dict(self):
        return asdict(self)

class SpeedEstimator:
    def __init__(self, scfg):
        cal = json.load(open(scfg["calibration"]))
        src = np.float32(cal["image_points"])
        w, l = float(cal["width_m"]), float(cal["length_m"])
        self.H = cv2.getPerspectiveTransform(src, np.float32([[0, 0], [w, 0], [w, l], [0, l]]))
        self.zone = src.reshape(-1, 1, 2)
        self.limit = float(scfg.get("limit_kmh", 50))
        self.tol = float(scfg.get("tolerance_kmh", 0))
        self.window = float(scfg.get("window_s", 1.0))
        self.min_window = float(scfg.get("min_window_s", 0.5))
        self.over_s = float(scfg.get("over_s", 0.5))
        self.only_zone = bool(scfg.get("only_in_zone", True))
        self.max_kmh = float(scfg.get("max_plausible_kmh", 250))
        self.contact = float(scfg.get("contact_y_frac", 1.0))
        self.tracks, self.peak = {}, {}

    def update(self, tid, bbox, comp, idx, ts, valid=True):
        """Returns (speed_kmh or None, over_limit, SpeedEvent or None)."""
        st = self.tracks.setdefault(tid, {"pts": deque(), "speed": None, "speed_ts": -1e9,
                                          "over_since": None, "fired": False, "last": ts})
        st["last"] = ts
        event = None
        if valid:
            x1, y1, x2, y2 = bbox
            x, y = (x1 + x2) / 2, y1 + self.contact * (y2 - y1)
            if comp is not None:
                x, y = comp.to_ref(x, y)
            inside = cv2.pointPolygonTest(self.zone, (float(x), float(y)), False) >= 0
            if inside or not self.only_zone:
                X, Y = cv2.perspectiveTransform(np.float32([[[x, y]]]), self.H)[0, 0]
                pts = st["pts"]
                pts.append((ts, float(X), float(Y)))
                while pts and ts - pts[0][0] > self.window:
                    pts.popleft()
                if len(pts) >= 5 and pts[-1][0] - pts[0][0] >= self.min_window:
                    t = np.array([p[0] for p in pts]) - pts[0][0]
                    vx = np.polyfit(t, [p[1] for p in pts], 1)[0]
                    vy = np.polyfit(t, [p[2] for p in pts], 1)[0]
                    kmh = float(np.hypot(vx, vy) * 3.6)
                    if kmh <= self.max_kmh:
                        st["speed"], st["speed_ts"] = kmh, ts
                        self.peak[tid] = max(self.peak.get(tid, 0.0), kmh)
            else:
                st["pts"].clear()          # left the calibrated zone

        speed = st["speed"] if ts - st["speed_ts"] <= 0.5 else None
        over = speed is not None and speed > self.limit + self.tol
        if over:
            if st["over_since"] is None:
                st["over_since"] = ts
            if not st["fired"] and ts - st["over_since"] >= self.over_s:
                st["fired"] = True
                event = SpeedEvent(tid, round(speed, 1), self.limit, round(ts, 2), idx)
        else:
            st["over_since"] = None
        return speed, over, event

    def cleanup(self, ts):
        for t in [t for t, s in self.tracks.items() if ts - s["last"] > 3.0]:
            del self.tracks[t]

    def draw_zone(self, frame, comp=None, thickness=None):
        pts = self.zone.reshape(-1, 2)
        if comp is not None:
            pts = comp.poly_to_cur(pts)
        t = thickness if thickness else max(2, frame.shape[1] // 960)
        cv2.polylines(frame, [pts.astype(np.int32)], True, (255, 255, 0), t)

        


