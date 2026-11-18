"""
Autonox engine
--------------
A UI-friendly twin of run_final2.py. It reuses your existing modules unchanged
and exposes the frame loop as a generator, so Streamlit can display each frame.

run() yields (drawn_frame, clean_frame). The app saves the CLEAN frames and the
per-frame overlay data (engine.overlay_frames), so the look can be changed later.
"""
import base64
import copy
import time

import cv2
import numpy as np
import yaml

from video.video_source import VideoSource
from video.motion import make_compensator
from video.display import label_style
from video.overlay import opts_from_cfg, draw_frame, lane_polygons, zone_polygon
from detection.vehicle_detector import VehicleTracker
from lanes.lane_assignment import LaneAssigner
from events.lane_change_detector import LaneChangeDetector
from speed.speed_estimator import SpeedEstimator
from plates.plate_detector import PlateDetector
from plates.plate_quality import quality_score
from plates.plate_reader import PlateReader
from plates.identity import group_tracks


def load_cfg(path="configs/default.yaml"):
    return yaml.safe_load(open(path))


def load_plate_models(cfg):
    """Heavy objects with no per-run state: safe to cache between runs."""
    return PlateDetector(cfg), PlateReader(cfg)


def _thumb_b64(img, h=64):
    if img is None or img.size == 0:
        return ""
    s = h / img.shape[0]
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode() if ok else ""


class AutonoxEngine:
    def __init__(self, cfg, source, lanes_path=None, speed_path=None,
                 do_lane=True, do_speed=True, do_plates=True, plate_models=None):
        self.cfg = copy.deepcopy(cfg)
        self.source = source
        self.lanes_path = lanes_path
        self.speed_path = speed_path
        self.do_lane, self.do_speed, self.do_plates = do_lane, do_speed, do_plates
        self.plate_models = plate_models

        if do_lane and not lanes_path:
            raise ValueError("Lane detection needs a lane calibration file.")
        if do_speed and not speed_path:
            raise ValueError("Speed detection needs a speed calibration file.")

        v = self.cfg["video"]
        v["source"] = source
        if lanes_path:
            v["lanes"] = lanes_path
        if speed_path:
            self.cfg["speed"]["calibration"] = speed_path
        else:
            v["motion_mask"] = "none"      # no speed zone to mask out

        # live results
        self.events = []                   # rows for the UI
        self.seen, self.best, self.state, self.rel = {}, {}, {}, {}
        self.gmap = {}
        self.overlay_frames = []           # one plain-data dict per frame (for re-rendering)
        self.n_frames = self.total_frames = 0
        self.ts = 0.0
        self.vehicles_now = 0
        self.speeds_now = []
        self.peak_seen = 0.0
        self.t_start = time.time()
        self.fps_proc = 0.0
        self.width = self.height = 0
        self.src_fps = 30.0
        self._speed_row = {}               # track id -> event row (one speed event per track)

    # ------------------------------------------------------------------ results
    def plate_for(self, tid):
        g = self.gmap.get(tid)
        if g:
            return g["status"], g["plate"]
        st = self.state.get(tid)
        if st and st.get("plate"):
            return st["status"], st["plate"]
        return "UNREADABLE", None

    def stats(self):
        confirmed = sum(1 for s in self.state.values() if s.get("status") == "CONFIRMED")
        n_lane = sum(e["kind"] == "lane" for e in self.events)
        n_speed = sum(e["kind"] == "speed" for e in self.events)
        avg = float(np.mean(self.speeds_now)) if self.speeds_now else None
        prog = (self.n_frames / self.total_frames) if self.total_frames > 0 else None
        return {"frame": self.n_frames, "ts": self.ts, "vehicles_now": self.vehicles_now,
                "unique": len(self.seen), "lane_events": n_lane, "speed_events": n_speed,
                "plates": confirmed, "avg_speed": avg, "peak_speed": self.peak_seen,
                "fps": self.fps_proc, "progress": prog,
                "limit": getattr(self, "limit", None)}

    def rows(self):
        out = []
        for e in self.events:
            status, plate = self.plate_for(e["tid"]) if self.do_plates else ("OFF", None)
            out.append({**e, "plate_status": status, "plate": plate})
        return out

    def _dedupe(self):
        """Same rules as run_final2.py: repeats of one lane change by one vehicle within 3 s
        become one event, and each vehicle keeps only its fastest speed event."""
        owner = {t: min(g["tracks"]) for t, g in self.gmap.items()}
        kept, last, top = [], {}, {}
        for e in self.events:
            k = owner.get(e["tid"], e["tid"])
            if e["kind"] == "lane":
                key = (k, e.get("from"), e.get("to"))
                if key in last and e["ts"] - last[key] < 3.0:
                    continue
                last[key] = e["ts"]
                kept.append(e)
            else:
                cur = top.get(k)
                if cur is None or e["value"] > cur["value"]:
                    top[k] = e
        self.events = sorted(kept + list(top.values()), key=lambda e: e["id"])

    # --------------------------------------------------------------------- loop
    def run(self):
        cfg = self.cfg
        pcfg, ecfg, scfg = cfg["plates"], cfg.get("events", {}), cfg["speed"]
        opts = opts_from_cfg(cfg)

        src = VideoSource(self.source)
        self.src_fps, self.width, self.height = src.fps, src.width, src.height

        # same window and frame limit as run_final2.py
        start_s, end_s = ecfg.get("start_s", 0), ecfg.get("end_s", 10**9)
        max_frames = pcfg.get("max_frames", 10**9)
        first = int(start_s * src.fps) if start_s else 0
        if first:
            src.seek(first)
        n = src.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        n = int(n) if n and n > 0 else 0
        last = min(x for x in (n or 10**12, int(min(end_s, 10**9) * src.fps), max_frames))
        self.total_frames = max(0, last - first) if last < 10**12 else 0

        tracker = VehicleTracker(cfg)
        lanes = LaneAssigner(self.lanes_path) if self.do_lane else None
        lcd = LaneChangeDetector(
            confirm_frames=int(src.fps * ecfg.get("confirm_s", 0.5)),
            stale_frames=int(src.fps * 3),
            rapid_s=ecfg.get("rapid_s", 1.5),
            init_frames=max(2, int(src.fps * ecfg.get("init_s", 0.1)))) if self.do_lane else None
        sest = SpeedEstimator(scfg) if self.do_speed else None
        self.limit = sest.limit if sest else None

        detector = reader = None
        if self.do_plates:
            detector, reader = self.plate_models or load_plate_models(cfg)

        comp = None
        if self.do_lane or self.do_speed:
            ref = cv2.VideoCapture(self.source)   # motion reference = frame 0 (as in run_final2)
            ok, f0 = ref.read()
            ref.release()
            if not ok:
                raise RuntimeError("Could not read the first frame for motion reference.")
            comp = make_compensator(cfg, f0)

        margin_frac = ecfg.get("margin_frac", 0.12)
        edge = int(src.width * 0.01)
        style = label_style(src.width, src.height)        # the page shows frames in about 900 x 640
        flagged = {}
        self.t_start = time.time()

        try:
            for idx, ts, frame in src.frames():
                if ts > end_s or idx >= max_frames:
                    break
                clean = frame.copy()
                self.n_frames, self.ts = idx - first + 1, ts
                if comp is not None:
                    comp.update(clean)
                tracks = tracker.update(clean)
                self.vehicles_now = len(tracks)
                self.speeds_now = []
                data = {"lanes": lane_polygons(lanes, comp) if lanes is not None else [],
                        "zone": zone_polygon(sest, comp) if sest is not None else None,
                        "veh": []}

                for t in tracks:
                    tid = t["track_id"]
                    self.seen.setdefault(tid, [ts, ts])[1] = ts
                    x1, y1 = max(t["bbox"][0], 0), max(t["bbox"][1], 0)
                    x2, y2 = min(t["bbox"][2], src.width), min(t["bbox"][3], src.height)
                    vw, vh = x2 - x1, y2 - y1
                    if vw <= 0 or vh <= 0:
                        continue
                    cut = x1 <= edge or x2 >= src.width - edge or y2 >= src.height - edge

                    # ---- lane + lane-change event ----
                    name, why, inside = None, ("edge" if cut else "ok"), True
                    if lanes is not None:
                        name, why = (None, "edge") if cut else lanes.assign_ex(
                            (x1, y1, x2, y2), comp, margin_frac=margin_frac)
                        # "outside lanes" for display only: contact point in no lane polygon at all
                        inside = lanes.assign_ex((x1, y1, x2, y2), comp, margin_frac=0.0)[1] != "outside"
                        lane = int(name.split()[-1]) if name else None
                        ev = lcd.update(tid, lane, idx, ts)
                        if ev:
                            self.events.append({
                                "id": f"LC-{sum(e['kind']=='lane' for e in self.events)+1:04d}",
                                "kind": "lane", "tid": tid, "ts": ev.end_ts,
                                "from": ev.from_lane, "to": ev.to_lane,
                                "detail": f"Lane {ev.from_lane} → {ev.to_lane}"
                                          + ("  ·  multi-lane" if ev.lanes_crossed > 1 else "")
                                          + ("  ·  rapid" if ev.rapid else ""),
                                "value": None,
                                "thumb": _thumb_b64(clean[y1:y2, x1:x2])})
                            flagged[tid] = idx + int(src.fps * 2)

                    # ---- speed + overspeed event ----
                    spd, over = None, False
                    if sest is not None:
                        spd, over, sev = sest.update(tid, (x1, y1, x2, y2), comp, idx, ts, valid=not cut)
                        if spd is not None:
                            self.speeds_now.append(spd)
                            self.peak_seen = max(self.peak_seen, spd)
                        if sev:
                            row = self._speed_row.get(tid)
                            if row is None:
                                row = {"id": f"SP-{len(self._speed_row)+1:04d}", "kind": "speed",
                                       "tid": tid, "ts": sev.ts, "value": sev.speed_kmh,
                                       "detail": "",
                                       "thumb": _thumb_b64(clean[y1:y2, x1:x2])}
                                self._speed_row[tid] = row
                                self.events.append(row)
                            elif sev.speed_kmh > row["value"]:
                                row["value"], row["ts"] = sev.speed_kmh, sev.ts
                            row["detail"] = f"{row['value']:.0f} km/h  ·  limit {sev.limit_kmh:.0f}"

                    # ---- plate pipeline ----
                    plate = None
                    if detector is not None:
                        st = self.state.get(tid, {"status": "", "plate": None})
                        if (vh >= pcfg["min_vehicle_h"] and idx % pcfg["every_n"] == 0
                                and st["status"] != "CONFIRMED"):
                            vehicle = clean[y1:y2, x1:x2]
                            good = []
                            for d in detector.detect(vehicle):
                                bx1, by1, bx2, by2 = d["bbox"]
                                pw, ph = bx2 - bx1, by2 - by1
                                if (pw >= pcfg["min_plate_w"] and d["conf"] >= pcfg["min_conf"]
                                        and pcfg["min_aspect"] <= pw / max(ph, 1) <= pcfg["max_aspect"]):
                                    good.append(d)
                            if good:
                                p = max(good, key=lambda d: d["conf"])
                                px1, py1, px2, py2 = p["bbox"]
                                crop = vehicle[max(py1, 0):py2, max(px1, 0):px2].copy()
                                if crop.size > 0:
                                    score, _ = quality_score(crop)
                                    self.rel[tid] = (px1 / vw, py1 / vh, px2 / vw, py2 / vh)
                                    lst = self.best.setdefault(tid, [])
                                    if len(lst) < pcfg["keep_top"] or score > lst[-1]["score"]:
                                        lst.append({"score": score, "crop": crop, "frame": idx,
                                                    "w": crop.shape[1], "read": reader.read(crop),
                                                    "vehicle": vehicle.copy(), "ts": ts})
                                        lst.sort(key=lambda b: b["score"], reverse=True)
                                        del lst[pcfg["keep_top"]:]
                                        s, pl = reader.decide([b["read"] for b in lst])
                                        self.state[tid] = {"status": s, "plate": pl}

                        st = self.state.get(tid)
                        if tid in self.rel and st and st["plate"]:
                            fx1, fy1, fx2, fy2 = self.rel[tid]
                            plate = {"box": [x1 + int(fx1 * vw), y1 + int(fy1 * vh),
                                             x1 + int(fx2 * vw), y1 + int(fy2 * vh)],
                                     "text": st["plate"], "ok": st["status"] == "CONFIRMED"}

                    data["veh"].append({
                        "tid": tid, "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "inside": bool(inside), "lane": name, "why": why,
                        "hot": flagged.get(tid, -1) >= idx,
                        "spd": None if spd is None else round(float(spd), 1),
                        "over": bool(over), "plate": plate})

                draw_frame(frame, data, opts, style)
                self.overlay_frames.append(data)

                if lcd is not None:
                    lcd.cleanup(idx)
                if sest is not None:
                    sest.cleanup(ts)

                el = time.time() - self.t_start
                self.fps_proc = self.n_frames / el if el > 0 else 0.0
                yield frame, clean

            # finished normally: merge tracks that are the same vehicle (same as run_final2)
            if detector is not None:
                try:
                    groups = group_tracks(self.best, self.state, self.seen, reader,
                                          pcfg.get("merge_window_s", 6.0), pcfg.get("merge_max_dist", 1))
                    self.gmap.update({t: g for g in groups for t in g["tracks"]})
                except Exception as ex:           # keep unmerged results
                    print(f"Track merging failed ({ex}); keeping unmerged plates")
            self._dedupe()
        finally:
            src.release()


