import cv2, yaml, os, json
from video.video_source import VideoSource
from detection.vehicle_detector import VehicleTracker
from plates.plate_detector import PlateDetector
from video.display import show
from plates.plate_quality import quality_score
from plates.plate_reader import PlateReader
from plates.identity import group_tracks
from evidence.evidence_builder import build_evidence

cfg = yaml.safe_load(open("configs/default.yaml"))
pcfg = cfg["plates"]
SHOW = pcfg.get("show", True)

src = VideoSource(cfg["video"]["source"])
tracker = VehicleTracker(cfg)
detector = PlateDetector(cfg)
reader = PlateReader(cfg)
os.makedirs("data/evidence/plates", exist_ok=True)
writer = cv2.VideoWriter("data/outputs/phase6b.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                         src.fps, (src.width, src.height))

best = {}    # track_id -> top crops, each with its OCR read
state = {}   # track_id -> {"status": ..., "plate": ...}
rel = {}     # track_id -> plate box as fractions of the vehicle box
seen = {}
GREEN, AMBER = (0, 200, 0), (0, 200, 255)

try:
    for idx, ts, frame in src.frames():
        if idx >= pcfg.get("max_frames", 10**9):
            break
        for t in tracker.update(frame):
            tid = t["track_id"]
            seen.setdefault(tid, [ts, ts])[1] = ts
            x1, y1 = max(t["bbox"][0], 0), max(t["bbox"][1], 0)
            x2, y2 = min(t["bbox"][2], src.width), min(t["bbox"][3], src.height)
            vw, vh = x2 - x1, y2 - y1
            if vw <= 0 or vh <= 0:
                continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"#{tid}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            st = state.get(tid, {"status": "", "plate": None})
            if (vh >= pcfg["min_vehicle_h"] and idx % pcfg["every_n"] == 0 
                    and not (st["status"] == "CONFIRMED"
                    and len(best.get(tid, [])) >= pcfg["keep_top"])):
                vehicle = frame[y1:y2, x1:x2]
                good = []
                for d in detector.detect(vehicle):
                    bx1, by1, bx2, by2 = d["bbox"]
                    pw, ph = bx2 - bx1, by2 - by1
                    aspect = pw / max(ph, 1)
                    if (pw >= pcfg["min_plate_w"] and d["conf"] >= pcfg["min_conf"]
                            and pcfg["min_aspect"] <= aspect <= pcfg["max_aspect"]):
                        good.append(d)
                if good:
                    p = max(good, key=lambda d: d["conf"])
                    px1, py1, px2, py2 = p["bbox"]
                    crop = vehicle[max(py1, 0):py2, max(px1, 0):px2].copy()
                    if crop.size > 0:
                        score, sharp = quality_score(crop)
                        rel[tid] = (px1 / vw, py1 / vh, px2 / vw, py2 / vh)
                        lst = best.setdefault(tid, [])
                        if len(lst) < pcfg["keep_top"] or score > lst[-1]["score"]:
                            lst.append({"score": score, "crop": crop, "frame": idx,
                                        "w": crop.shape[1], "read": reader.read(crop),
                                        "vehicle": vehicle.copy(), "ts": ts})
                            lst.sort(key=lambda b: b["score"], reverse=True)
                            del lst[pcfg["keep_top"]:]
                            status, plate = reader.decide([b["read"] for b in lst])
                            state[tid] = {"status": status, "plate": plate}

            st = state.get(tid)
            if tid in rel and st and st["plate"]:
                fx1, fy1, fx2, fy2 = rel[tid]
                bx1, by1 = x1 + int(fx1 * vw), y1 + int(fy1 * vh)
                bx2, by2 = x1 + int(fx2 * vw), y1 + int(fy2 * vh)
                ok = st["status"] == "CONFIRMED"
                col = GREEN if ok else AMBER
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), col, 2)
                label = st["plate"] if ok else f'{st["plate"]}? (review)'
                cv2.putText(frame, label, (bx1, max(by1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        writer.write(frame)
        if SHOW:
            show("phase6b", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if idx % 30 == 0:
            print(f"frame {idx}, tracks with plate crops: {len(best)}")
except KeyboardInterrupt:
    print("\nStopped by user, saving results so far...")
finally:
    results = {}
    print(f"\n{'track':>6} {'status':>11} {'plate':>9}")
    for tid in sorted(best):
        for rank, b in enumerate(best[tid], 1):
            cv2.imwrite(f"data/evidence/plates/track{tid}_r{rank}_f{b['frame']}.png", b["crop"])
        s = state.get(tid, {"status": "UNREADABLE", "plate": None})
        results[tid] = {**s, "reads": [b["read"] for b in best[tid]]}
        print(f"{tid:>6} {s['status']:>11} {s['plate'] or '-':>9}")

    groups = group_tracks(best, state, seen, reader,
                          pcfg.get("merge_window_s", 6.0), pcfg.get("merge_max_dist", 1))
    print(f"\n{len(best)} tracks -> {len(groups)} vehicles")
    for n, g in enumerate(sorted(groups, key=lambda g: g["first_ts"]), 1):
        alts = sorted({r["plate"] for r in g["reads"]} - {g["plate"]})
        print(f"vehicle {n:>2}  tracks {g['tracks']}  {g['status']:11} {g['plate'] or '-'}")
        build_evidence("data/evidence/events", f"VEH-{n:03d}", cfg["camera_id"],
                       g["top"]["ts"], "TEST: plate capture (stand-in trigger)",
                       "+".join(map(str, g["tracks"])), g["top"]["vehicle"], g["top"]["crop"],
                       g["status"], g["plate"], alts)
    
    json.dump(results, open("data/outputs/plate_results.json", "w"), indent=2)
    src.release(); writer.release(); cv2.destroyAllWindows()


