"""Shared drawing: one look for the live view, the re-rendered video and the run scripts.

Per frame, the pipeline builds a plain-data dict:
  {"lanes": [[[x, y], ...], ...], "zone": [[x, y], ...] or None,
   "veh": [{"tid", "bbox", "inside", "lane", "why", "hot", "spd", "over",
            "plate": {"box", "text", "ok"} or None}, ...]}
draw_frame() turns that dict plus the display switches into pixels.
"""
import json
import cv2
import numpy as np

from video.display import label_style

FONT = cv2.FONT_HERSHEY_SIMPLEX

DEFAULT_OPTS = {"hide_lanes": True, "hide_speed_zone": False, "hide_outside": True,
                "hide_plates": False, "show_reason": False}
OPT_ORDER = ("hide_lanes", "hide_speed_zone", "hide_outside", "hide_plates")

# BGR colours
WHITE, GREEN, AMBER, ORANGE = (245, 245, 245), (0, 200, 0), (0, 200, 255), (0, 165, 255)
MAGENTA, PINK, GREY, ZONE = (255, 0, 255), (200, 130, 255), (150, 150, 150), (255, 255, 0)
LANE_TXT, SPEED_OK, SPEED_OVER = (255, 200, 120), (0, 255, 255), (0, 140, 255)
LANE_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]


def opts_from_cfg(cfg):
    o = dict(DEFAULT_OPTS)
    o.update(cfg.get("overlay") or {})
    return o


def lane_polygons(lanes, comp=None):
    """Lane polygons in the current frame's coordinates (follows camera motion)."""
    out = []
    for _, poly in lanes.lanes:
        pts = np.array(poly.exterior.coords)
        if comp is not None:
            pts = comp.poly_to_cur(pts)
        out.append(np.round(pts).astype(int).tolist())
    return out


def zone_polygon(est, comp=None):
    pts = est.zone.reshape(-1, 2)
    if comp is not None:
        pts = comp.poly_to_cur(pts)
    return np.round(pts).astype(int).tolist()


def _tint(frame, polys, alpha=0.2):
    if not polys:
        return
    over = frame.copy()
    for i, p in enumerate(polys):
        cv2.fillPoly(over, [np.array(p, np.int32)], LANE_COLORS[i % len(LANE_COLORS)])
    cv2.addWeighted(over, alpha, frame, 1 - alpha, 0, dst=frame)


def _corners(frame, x1, y1, x2, y2, color, t):
    """Corner brackets: far less clutter than full rectangles in dense traffic."""
    L = max(6, int(0.22 * min(x2 - x1, y2 - y1)))
    for px, py, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(frame, (px, py), (px + dx * L, py), color, t, cv2.LINE_AA)
        cv2.line(frame, (px, py), (px, py + dy * L), color, t, cv2.LINE_AA)


def _pill(frame, x, y, segs, fs, th):
    """Dark translucent label with coloured text segments. (x, y) = left edge, bottom edge."""
    pad = max(2, int(fs * 5))
    sizes = [cv2.getTextSize(t, FONT, fs, th)[0] for t, _ in segs]
    w = sum(s[0] for s in sizes) + 2 * pad
    h = max(s[1] for s in sizes) + 2 * pad
    H, W = frame.shape[:2]
    x0 = int(max(0, min(x, W - w)))
    y0 = int(max(0, min(y - h, H - h)))
    sub = frame[y0:y0 + h, x0:x0 + w]
    if sub.shape[0] == h and sub.shape[1] == w:
        frame[y0:y0 + h, x0:x0 + w] = cv2.addWeighted(sub, 0.35, np.zeros_like(sub), 0.65, 0)
    cx = x0 + pad
    for (t, c), s in zip(segs, sizes):
        cv2.putText(frame, t, (cx, y0 + h - pad), FONT, fs, c, th, cv2.LINE_AA)
        cx += s[0]


def draw_vehicle(frame, v, opts, st):
    x1, y1, x2, y2 = v["bbox"]
    over, hot = v.get("over"), v.get("hot")
    inside = v.get("inside", True)
    col = MAGENTA if over else ORANGE if hot else GREEN if inside else GREY
    if over or hot:
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, st["bt"] + 1, cv2.LINE_AA)
    else:
        _corners(frame, x1, y1, x2, y2, col, st["bt"])
    segs = [(f"#{v['tid']}", WHITE)]
    if v.get("lane"):
        segs.append((" L" + v["lane"].split()[-1], LANE_TXT))
    elif opts.get("show_reason") and v.get("why"):
        segs.append((f" {v['why']}", GREY))
    if hot:
        segs.append((" CHANGED", ORANGE))
    if v.get("spd") is not None:
        segs.append((f"  {v['spd']:.0f} km/h", SPEED_OVER if over else SPEED_OK))
    if over:
        segs.append(("  OVERSPEED", PINK))
    _pill(frame, x1, y1 - 2, segs, st["fs"], st["th"])


def draw_plate(frame, p, st):
    bx1, by1, bx2, by2 = p["box"]
    col = GREEN if p["ok"] else AMBER
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), col, max(1, st["bt"] - 1), cv2.LINE_AA)
    text = p["text"] if p["ok"] else f'{p["text"]}? (review)'
    _pill(frame, bx1, by1 - 2, [(text, col)], st["fs"] * 0.85, st["th"])   # plates: smaller text


def draw_frame(frame, data, opts, st):
    """Draws one frame's overlay in place, honouring the hide switches."""
    if not opts.get("hide_lanes", True):
        _tint(frame, data.get("lanes"))
    if data.get("zone") and not opts.get("hide_speed_zone", False):
        cv2.polylines(frame, [np.array(data["zone"], np.int32)], True, ZONE, st["bt"], cv2.LINE_AA)
    hide_out = opts.get("hide_outside", True)
    shown = [v for v in data.get("veh", []) if not (hide_out and v.get("inside") is False)]
    for v in shown:
        draw_vehicle(frame, v, opts, st)
    if not opts.get("hide_plates", False):
        for v in shown:
            if v.get("plate"):
                draw_plate(frame, v["plate"], st)


def render_video(clean_path, data_path, out_path, opts, box_w=900, box_h=640):
    """Draws the saved overlay data onto the clean recording. No detection is run."""
    d = json.load(open(data_path, encoding="utf-8"))
    frames = d["frames"]
    cap = cv2.VideoCapture(clean_path)
    fps = d.get("fps") or cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    st = label_style(w, h, box_w, box_h)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i < len(frames):
            draw_frame(fr, frames[i], opts, st)
        writer.write(fr)
        i += 1
    cap.release()
    writer.release()
    return out_path


