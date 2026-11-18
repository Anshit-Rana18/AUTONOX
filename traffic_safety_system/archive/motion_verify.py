import cv2, sys
from video.video_source import VideoSource
from video.motion import MotionCompensator
from lanes.lane_assignment import LaneAssigner

video, lanes_path = sys.argv[1], sys.argv[2]
src = VideoSource(video)
lanes = LaneAssigner(lanes_path)
comp = None
targets = {int(t * src.fps) for t in (20, 40, 60)}

for idx, ts, frame in src.frames():
    if comp is None:
        comp = MotionCompensator(frame)
        continue
    if idx in targets:
        comp.update(frame)
        raw = lanes.draw(frame)
        fixed = lanes.draw(frame, comp)
        both = cv2.hconcat([cv2.resize(raw, None, fx=0.5, fy=0.5),
                            cv2.resize(fixed, None, fx=0.5, fy=0.5)])
        cv2.putText(both, "NO correction", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(both, "WITH correction", (both.shape[1] // 2 + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(f"data/outputs/motion_{int(ts)}s.png", both)
        print(f"saved {int(ts)}s  shift={comp.max_shift_px():.0f}px ok={comp.ok}")


