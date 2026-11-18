import sys
sys.path.insert(0, ".")
from video.video_source import VideoSource
from video.motion import MotionCompensator

src = VideoSource(sys.argv[1])
comp = None
for idx, ts, frame in src.frames():
    if comp is None:
        comp = MotionCompensator(frame)
        continue
    comp.update(frame)
    if idx % max(1, int(src.fps / 2)) == 0:
        print(f"t={ts:5.1f}s  shift={comp.max_shift_px():6.1f}px  inliers={comp.inliers}  ok={comp.ok}")

