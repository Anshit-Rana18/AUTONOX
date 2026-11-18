import cv2, yaml, time
from video.video_source import VideoSource
from detection.vehicle_detector import VehicleTracker
from lanes.lane_assignment import LaneAssigner

cfg = yaml.safe_load(open("configs/default.yaml"))
src = VideoSource(cfg["video"]["source"])
tracker = VehicleTracker(cfg)
lanes = LaneAssigner("configs/SD_lanes.json")
writer = cv2.VideoWriter("data/outputs/phase3.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                         src.fps, (src.width, src.height))

for idx, ts, frame in src.frames():
    if cfg["video"].get("draw_lanes", False):
        frame = lanes.draw(frame)
    for t in tracker.update(frame):
        x1, y1, x2, y2 = t["bbox"]
        lane = lanes.assign(t["bbox"])
        color = (0, 255, 0) if lane else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'#{t["track_id"]} {lane or "no lane"}',
                    (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.circle(frame, ((x1 + x2) // 2, y2), 4, color, -1)   # the contact point
    writer.write(frame)
    cv2.imshow("phase3", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

src.release(); writer.release(); cv2.destroyAllWindows()

