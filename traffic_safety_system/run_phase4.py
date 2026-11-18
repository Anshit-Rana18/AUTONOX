import cv2, yaml, json
from video.display import show
from video.video_source import VideoSource
from video.motion import MotionCompensator
from detection.vehicle_detector import VehicleTracker
from lanes.lane_assignment import LaneAssigner
from events.lane_change_detector import LaneChangeDetector

cfg = yaml.safe_load(open("configs/default.yaml"))
ecfg = cfg.get("events", {})
src = VideoSource(cfg["video"]["source"])
tracker = VehicleTracker(cfg)
lanes = LaneAssigner(cfg["video"]["lanes"])
detector = LaneChangeDetector(
    confirm_frames=int(src.fps * ecfg.get("confirm_s", 0.5)),
    stale_frames=int(src.fps * 3),
    rapid_s=ecfg.get("rapid_s", 1.5),
    init_frames=max(2, int(src.fps * ecfg.get("init_s", 0.1))))
writer = cv2.VideoWriter("data/outputs/phase4.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                         src.fps, (src.width, src.height))

# The motion reference must be frame 0, where the lanes were drawn.
ref = cv2.VideoCapture(cfg["video"]["source"])
_, f0 = ref.read()
ref.release()
comp = MotionCompensator(f0)

start_s, end_s = ecfg.get("start_s", 0), ecfg.get("end_s", 10**9)
if start_s:
    src.seek(int(start_s * src.fps))

margin_frac = ecfg.get("margin_frac", 0.12)
edge = int(src.width * 0.01)
events, highlight = [], {}
hold = int(src.fps * 2)

for idx, ts, frame in src.frames():
    if ts > end_s:
        break
    comp.update(frame)
    tracks = tracker.update(frame)
    if cfg["video"].get("draw_lanes", False):
        frame = lanes.draw(frame, comp)

    for t in tracks:
        tid = t["track_id"]
        x1, y1, x2, y2 = t["bbox"]
        cut = x1 <= edge or x2 >= src.width - edge or y2 >= src.height - edge
        name, why = (None, "edge") if cut else lanes.assign_ex((x1, y1, x2, y2), comp, margin_frac=margin_frac)
        lane = int(name.split()[-1]) if name else None
        ev = detector.update(tid, lane, idx, ts)
        if ev:
            events.append(ev.to_dict())
            highlight[tid] = idx + hold
            print(f"[{ts:6.2f}s] EVENT #{len(events)}: track {tid} "
                  f"L{ev.from_lane}->L{ev.to_lane} "
                  f"({ev.duration_s}s{', RAPID' if ev.rapid else ''}"
                  f"{', MULTI' if ev.lanes_crossed > 1 else ''})")

        flagged = highlight.get(tid, -1) >= idx
        color = (0, 165, 255) if flagged else ((0, 255, 0) if lane else (0, 0, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if flagged else 2)
        label = f"#{tid} {name or 'no lane'}" + (" CHANGED" if flagged else "")
        cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    detector.cleanup(idx)
    cv2.putText(frame, f"Lane changes: {len(events)}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    writer.write(frame)
    show("phase4", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

json.dump(events, open("data/outputs/events.json", "w"), indent=2)
print(f"{len(events)} events saved to data/outputs/events.json")
src.release(); writer.release()
try:
    cv2.destroyAllWindows()
except cv2.error:
    pass

