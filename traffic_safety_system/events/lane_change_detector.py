from dataclasses import dataclass, asdict

@dataclass
class LaneChangeEvent:
    track_id: int
    from_lane: int
    to_lane: int
    lanes_crossed: int
    start_frame: int      # last frame seen in the old lane
    end_frame: int        # frame where the new lane was confirmed
    start_ts: float
    end_ts: float
    duration_s: float     # old lane -> change confirmed (includes the confirm time)
    transit_s: float      # old lane -> first frame seen in the new lane
    rapid: bool           # True when transit_s < rapid_s

    def to_dict(self):
        return asdict(self)

class LaneChangeDetector:
    def __init__(self, confirm_frames=8, stale_frames=90, rapid_s=1.5, init_frames=3):
        self.confirm = confirm_frames      # frames in a NEW lane to confirm a change
        self.init = init_frames            # frames in the FIRST lane to learn where a track started
        self.stale = stale_frames          # forget tracks unseen for this many frames
        self.rapid_s = rapid_s
        self.tracks = {}

    def update(self, track_id, lane, frame_idx, ts):
        """lane: int or None. Returns a LaneChangeEvent when one is confirmed."""
        s = self.tracks.setdefault(track_id, {
            "stable": None, "cand": None, "cand_n": 0, "cand_ts": ts,
            "last_stable_frame": frame_idx, "last_stable_ts": ts,
            "last_seen": frame_idx})
        s["last_seen"] = frame_idx
        if lane is None:
            return None

        if s["stable"] is None:                     # learning the starting lane
            if lane == s["cand"]:
                s["cand_n"] += 1
            else:
                s["cand"], s["cand_n"] = lane, 1
            if s["cand_n"] >= self.init:
                s["stable"], s["cand"], s["cand_n"] = lane, None, 0
                s["last_stable_frame"], s["last_stable_ts"] = frame_idx, ts
            return None

        if lane == s["stable"]:                     # still home: forget any candidate
            s["cand"], s["cand_n"] = None, 0
            s["last_stable_frame"], s["last_stable_ts"] = frame_idx, ts
            return None

        if lane == s["cand"]:                       # a different lane: count toward confirmation
            s["cand_n"] += 1
        else:
            s["cand"], s["cand_n"], s["cand_ts"] = lane, 1, ts   # remember when it first appeared

        if s["cand_n"] >= self.confirm:
            dur = ts - s["last_stable_ts"]
            transit = s["cand_ts"] - s["last_stable_ts"]
            ev = LaneChangeEvent(
                track_id=track_id, from_lane=s["stable"], to_lane=lane,
                lanes_crossed=abs(lane - s["stable"]),
                start_frame=s["last_stable_frame"], end_frame=frame_idx,
                start_ts=round(s["last_stable_ts"], 2), end_ts=round(ts, 2),
                duration_s=round(dur, 2), transit_s=round(transit, 2),
                rapid=transit < self.rapid_s)
            s["stable"], s["cand"], s["cand_n"] = lane, None, 0
            s["last_stable_frame"], s["last_stable_ts"] = frame_idx, ts
            return ev
        return None

    def cleanup(self, frame_idx):
        dead = [t for t, s in self.tracks.items() if frame_idx - s["last_seen"] > self.stale]
        for t in dead:
            del self.tracks[t]

