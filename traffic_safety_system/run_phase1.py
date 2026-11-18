import cv2, yaml, time
from video.video_source import VideoSource
from detection.vehicle_detector import VehicleTracker

cfg = yaml.safe_load(open("configs/default.yaml"))
src = VideoSource(cfg["video"]["source"])
tracker = VehicleTracker(cfg)
writer = cv2.VideoWriter(cfg["video"]["output"], cv2.VideoWriter_fourcc(*"mp4v"),
                         src.fps, (src.width, src.height))

t0 = time.time()
for idx, ts, frame in src.frames():
    for t in tracker.update(frame):
        x1, y1, x2, y2 = t["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'#{t["track_id"]} {t["cls"]} {t["conf"]:.2f}',
                    (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    writer.write(frame)
    cv2.imshow("phase1", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print(f"Processed {idx+1} frames in {time.time()-t0:.1f}s")
src.release(); writer.release(); cv2.destroyAllWindows()

