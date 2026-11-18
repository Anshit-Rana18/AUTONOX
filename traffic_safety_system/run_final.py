import cv2, yaml, os, json, shutil
from video.display import show
from video.video_source import VideoSource
from video.motion import MotionCompensator
from detection.vehicle_detector import VehicleTracker
from lanes.lane_assignment import LaneAssigner
from events.lane_change_detector import LaneChangeDetector
from plates.plate_detector import PlateDetector
from plates.plate_quality import quality_score
from plates.plate_reader import PlateReader
from plates.identity import group_tracks
from evidence.evidence_builder import build_evidence
from evidence.clip_recorder import ClipRecorder
from evidence.integrity import video_fingerprint, finalize_folder

cfg = yaml.safe_load(open("configs/default.yaml"))
pcfg, ecfg = cfg["plates"], cfg.get("events", {})
vcfg = cfg.get("evidence", {})
SHOW = pcfg.get("show", True)
EVT_DIR = "data/evidence/events"

src = VideoSource(cfg["video"]["source"])
tracker = VehicleTracker(cfg)
lanes = LaneAssigner(cfg["video"]["lanes"])
lcd = LaneChangeDetector(
    confirm_frames=int(src.fps * ecfg.get("confirm_s", 0.5)),
    stale_frames=int(src.fps * 3),
    rapid_s=ecfg.get("rapid_s", 1.5),
    init_frames=max(2, int(src.fps * ecfg.get("init_s", 0.1))))
detector = PlateDetector(cfg)
reader = PlateReader(cfg)
recorder = ClipRecorder(src.fps, vcfg.get("pre_s", 3.0), vcfg.get("post_s", 3.0),
                        vcfg.get("clip_width", 960))
os.makedirs(EVT_DIR, exist_ok=True)
writer = cv2.VideoWriter("data/outputs/final.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                         src.fps, (src.width, src.height))

print("Fingerprinting source video...")
VIDEO_FP = video_fingerprint(cfg["video"]["source"])

# The motion reference must be frame 0, where the lanes were drawn.
ref = cv2.VideoCapture(cfg["video"]["source"])
_, f0 = ref.read()
ref.release()
comp = MotionCompensator(f0)

start_s, end_s = ecfg.get("start_s", 0), ecfg.get("end_s", 10**9)
if start_s:
    src.seek(int(start_s * src.fps))
print(f"Video {src.width}x{src.height} @ {src.fps:.0f} FPS | window {start_s}-{end_s}s | "
      f"max_frames {pcfg.get('max_frames')} | lanes {cfg['video']['lanes']}")

margin_frac = ecfg.get("margin_frac", 0.12)
edge = int(src.width * 0.01)

best, state, rel, seen = {}, {}, {}, {}   # per-track plate data
events, pending, flagged = [], [], {}     # lane-change events
gmap = {}                                 # track_id -> merged vehicle group (filled at the end)
popup_delay = int(src.fps * ecfg.get("popup_delay_s", 1.0))
GREEN, AMBER, ORANGE = (0, 200, 0), (0, 200, 255), (0, 165, 255)


def make_screen(e):
    ev, tid = e["ev"], e["tid"]
    g = gmap.get(tid)
    if g:                                  # final, merged result
        st = {"status": g["status"], "plate": g["plate"]}
        vehicle, crop = g["top"]["vehicle"], g["top"]["crop"]
        alts = sorted({r["plate"] for r in g["reads"]} - {g["plate"]})
    else:                                  # live result for this track
        lst = best.get(tid, [])
        st = state.get(tid, {"status": "UNREADABLE", "plate": None})
        vehicle = lst[0]["vehicle"] if lst else e["vehicle"]
        crop = lst[0]["crop"] if lst else None
        alts = sorted({b["read"]["plate"] for b in lst} - {st["plate"]})
    etype = f"Lane change L{ev.from_lane}->L{ev.to_lane}"
    etype += (" RAPID" if ev.rapid else "") + (" MULTI-LANE" if ev.lanes_crossed > 1 else "")
    return build_evidence(EVT_DIR, e["id"], cfg["camera_id"], ev.end_ts,
                          etype, tid, vehicle, crop, st["status"], st["plate"], alts,
                          extra=ev.to_dict())


try:
    for idx, ts, frame in src.frames():
        if ts > end_s or idx >= pcfg.get("max_frames", 10**9):
            break
        clean = frame.copy()                 # crops come from the untinted frame
        comp.update(clean)
        tracks = tracker.update(clean)
        if cfg["video"].get("draw_lanes", False):
            frame = lanes.draw(frame, comp)
        new_events = []

        for t in tracks:
            tid = t["track_id"]
            seen.setdefault(tid, [ts, ts])[1] = ts
            x1, y1 = max(t["bbox"][0], 0), max(t["bbox"][1], 0)
            x2, y2 = min(t["bbox"][2], src.width), min(t["bbox"][3], src.height)
            vw, vh = x2 - x1, y2 - y1
            if vw <= 0 or vh <= 0:
                continue

            # ---- lane and lane-change event ----
            cut = x1 <= edge or x2 >= src.width - edge or y2 >= src.height - edge
            name, why = (None, "edge") if cut else lanes.assign_ex(
                (x1, y1, x2, y2), comp, margin_frac=margin_frac)
            lane = int(name.split()[-1]) if name else None
            ev = lcd.update(tid, lane, idx, ts)
            if ev:
                e = {"id": f"EVT-{len(events) + 1:04d}", "tid": tid, "ev": ev,
                     "due": idx + popup_delay, "shown": False,
                     "vehicle": clean[y1:y2, x1:x2].copy()}
                events.append(e); pending.append(e); new_events.append(e)
                flagged[tid] = idx + int(src.fps * 2)
                print(f"[{ts:6.2f}s] {e['id']}: track {tid} L{ev.from_lane}->L{ev.to_lane}")

            hot = flagged.get(tid, -1) >= idx
            col = ORANGE if hot else ((0, 255, 0) if lane else (0, 0, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3 if hot else 2)
            cv2.putText(frame, f"#{tid} {name or why}" + (" CHANGED" if hot else ""),
                        (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

            # ---- plate pipeline (stops once a track is CONFIRMED) ----
            st = state.get(tid, {"status": "", "plate": None})
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
                        rel[tid] = (px1 / vw, py1 / vh, px2 / vw, py2 / vh)
                        lst = best.setdefault(tid, [])
                        if len(lst) < pcfg["keep_top"] or score > lst[-1]["score"]:
                            lst.append({"score": score, "crop": crop, "frame": idx,
                                        "w": crop.shape[1], "read": reader.read(crop),
                                        "vehicle": vehicle.copy(), "ts": ts})
                            lst.sort(key=lambda b: b["score"], reverse=True)
                            del lst[pcfg["keep_top"]:]
                            s, plate = reader.decide([b["read"] for b in lst])
                            state[tid] = {"status": s, "plate": plate}

            st = state.get(tid)
            if tid in rel and st and st["plate"]:
                fx1, fy1, fx2, fy2 = rel[tid]
                bx1, by1 = x1 + int(fx1 * vw), y1 + int(fy1 * vh)
                bx2, by2 = x1 + int(fx2 * vw), y1 + int(fy2 * vh)
                ok = st["status"] == "CONFIRMED"
                c = GREEN if ok else AMBER
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), c, 2)
                cv2.putText(frame, st["plate"] if ok else f'{st["plate"]}? (review)',
                            (bx1, max(by1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

        lcd.cleanup(idx)

        # ---- popup for events whose waiting time is over ----
        for e in pending:
            if not e["shown"] and idx >= e["due"]:
                path = make_screen(e)
                e["shown"] = True
                if SHOW:
                    show("Lane-change evidence", cv2.imread(path))

        cv2.putText(frame, f"Lane changes: {len(events)}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # ---- evidence: clip buffer, then context image and clip start for new events ----
        recorder.push(idx, frame)
        for e in new_events:
            folder = os.path.join(EVT_DIR, e["id"])
            os.makedirs(folder, exist_ok=True)
            ctx = frame.copy()
            fs = max(0.8, src.width / 1920)
            bar = int(60 * fs)
            cv2.rectangle(ctx, (0, src.height - bar), (src.width, src.height), (0, 0, 0), -1)
            cv2.putText(ctx, f"{e['id']}  {cfg['camera_id']}  video time {ts:.2f}s  frame {idx}",
                        (20, src.height - int(bar * 0.3)), cv2.FONT_HERSHEY_SIMPLEX,
                        fs, (255, 255, 255), max(1, int(2 * fs)))
            cv2.imwrite(os.path.join(folder, "context.jpg"), ctx,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            e["ctx_frame"], e["ctx_ts"] = idx, ts
            recorder.start(e["id"], os.path.join(folder, "clip.mp4"))

        writer.write(frame)
        if SHOW:
            show("final", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if idx % 30 == 0:
            print(f"frame {idx}, events: {len(events)}")
except KeyboardInterrupt:
    print("\nStopped by user, saving results so far...")
finally:
    recorder.flush()
    try:
        groups = group_tracks(best, state, seen, reader,
                              pcfg.get("merge_window_s", 6.0), pcfg.get("merge_max_dist", 1))
    except Exception as ex:
        print(f"Track merging failed ({ex}); keeping events unmerged")
        groups = []
    gmap.update({t: g for g in groups for t in g["tracks"]})
    owner = {t: i for i, g in enumerate(groups) for t in g["tracks"]}

    kept, dropped, last = [], [], {}
    for e in events:                       # same vehicle, same change, within 3 s -> one event
        key = (owner.get(e["tid"], f"t{e['tid']}"), e["ev"].from_lane, e["ev"].to_lane)
        if key in last and e["ev"].end_ts - last[key] < 3.0:
            dropped.append(e)
            continue
        last[key] = e["ev"].end_ts
        kept.append(e)
    for e in dropped:                      # remove evidence of collapsed duplicates
        shutil.rmtree(os.path.join(EVT_DIR, e["id"]), ignore_errors=True)
    events = kept

    out = []
    for e in events:                       # rebuild screens with the final plate data
        make_screen(e)
        folder = os.path.join(EVT_DIR, e["id"])
        clip = recorder.done.get(e["id"])
        finalize_folder(folder, {
            "source_video": VIDEO_FP["file"],
            "source_video_sha256": VIDEO_FP["sha256"],
            "source_fps": src.fps,
            "source_resolution": f"{src.width}x{src.height}",
            "event_frame": e.get("ctx_frame"),
            "context_image": "context.jpg",
            "clip": clip,
            "notes": [
                "Machine-read plate text. CONFIRMED means reads agreed; it is not human-verified.",
                "screen.png shows an enlarged plate view. Unmodified originals: vehicle.png, plate_original.png.",
                "context.jpg is the annotated full frame (JPEG quality 95).",
                "clip.mp4 is annotated, downscaled and re-encoded; the source video is the primary record.",
            ]})
        g = gmap.get(e["tid"])
        st = g or state.get(e["tid"], {})
        out.append({**e["ev"].to_dict(), "event_id": e["id"],
                    "plate_status": st.get("status", "UNREADABLE"),
                    "plate": st.get("plate")})
    json.dump(out, open("data/outputs/final_events.json", "w"), indent=2)
    print(f"\n{len(events)} events. Evidence in {EVT_DIR}/")
    src.release(); writer.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

