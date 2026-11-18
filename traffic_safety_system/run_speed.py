import cv2, yaml, json
from video.display import show, label_style, put_text
from video.overlay import opts_from_cfg, draw_frame, zone_polygon, WHITE
from video.video_source import VideoSource
from video.motion import make_compensator
from detection.vehicle_detector import VehicleTracker
from speed.speed_estimator import SpeedEstimator

cfg = yaml.safe_load(open("configs/default.yaml"))
ecfg, scfg = cfg.get("events", {}), cfg["speed"]
opts = opts_from_cfg(cfg)
src = VideoSource(cfg["video"]["source"])
tracker = VehicleTracker(cfg)
est = SpeedEstimator(scfg)
writer = cv2.VideoWriter("data/outputs/speed.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                         src.fps, (src.width, src.height))

# The motion reference must be frame 0, where the calibration was drawn.
ref = cv2.VideoCapture(cfg["video"]["source"])
_, f0 = ref.read()
ref.release()
comp = make_compensator(cfg, f0)

start_s, end_s = ecfg.get("start_s", 0), ecfg.get("end_s", 10**9)
if start_s:
    src.seek(int(start_s * src.fps))

edge = int(src.width * 0.01)
style = label_style(src.width, src.height, 1280, 720)     # the OpenCV window is 1280 x 720
fired, evs = 0, []

for idx, ts, frame in src.frames():
    if ts > end_s:
        break
    comp.update(frame)
    tracks = tracker.update(frame)
    data = {"lanes": [], "zone": zone_polygon(est, comp), "veh": []}
    n_over = 0

    for t in tracks:
        tid = t["track_id"]
        x1, y1, x2, y2 = t["bbox"]
        cut = x1 <= edge or x2 >= src.width - edge or y2 >= src.height - edge
        spd, over, ev = est.update(tid, t["bbox"], comp, idx, ts, valid=not cut)
        if ev:
            fired += 1
            evs.append(ev.to_dict())
            print(f"[{ts:6.2f}s] OVERSPEED: track {tid} {ev.speed_kmh:.0f} km/h (limit {ev.limit_kmh:.0f})")
        n_over += over
        # No lanes in this script, so every vehicle counts as "inside" (nothing is hidden).
        data["veh"].append({"tid": tid, "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "inside": True, "lane": None, "why": None, "hot": False,
                            "spd": None if spd is None else round(float(spd), 1),
                            "over": bool(over), "plate": None})

    draw_frame(frame, data, opts, style)
    est.cleanup(ts)
    put_text(frame, f"Speed limit {est.limit:.0f} km/h   Overspeed now: {n_over}   Flagged: {fired}",
             20, int(30 * style["k"]), WHITE, style["fs"] * 1.2, style["th"])
    writer.write(frame)
    show("speed", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

json.dump({str(t): round(v, 1) for t, v in est.peak.items()},
          open("data/outputs/speed_peaks.json", "w"), indent=2)
json.dump(evs, open("data/outputs/speed_events.json", "w"), indent=2)
print(f"{fired} overspeed events. Peak speeds per track in data/outputs/speed_peaks.json")
src.release(); writer.release()
try:
    cv2.destroyAllWindows()
except cv2.error:
    pass

