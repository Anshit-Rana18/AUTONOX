import cv2, yaml, os
from video.video_source import VideoSource
from detection.vehicle_detector import VehicleTracker
from plates.plate_detector import PlateDetector
from plates.plate_quality import quality_score

cfg = yaml.safe_load(open("configs/default.yaml"))
pcfg = cfg["plates"]
src = VideoSource(cfg["video"]["source"])
tracker = VehicleTracker(cfg)
plates = PlateDetector(cfg)
os.makedirs("data/evidence/plates", exist_ok=True)
writer = cv2.VideoWriter("data/outputs/phase6.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                         src.fps, (src.width, src.height))

best = {}   # track_id -> {"score", "crop", "frame", "w", "sharp", "conf"}

for idx, ts, frame in src.frames():
    if idx % 2 == 1:
        continue    
    for t in tracker.update(frame):
        tid = t["track_id"]
        x1, y1, x2, y2 = t["bbox"]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, src.width), min(y2, src.height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"#{tid}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        if (y2 - y1) < pcfg["min_vehicle_h"] or idx % pcfg["every_n"] != 0:
            continue

        lst = best.get(tid, [])
        if len(lst) >= pcfg["keep_top"] and lst[-1]["w"] >= 80:
            continue

        vehicle = frame[y1:y2, x1:x2]
        found = plates.detect(vehicle)
        good = []
        for d in found:
            bx1, by1, bx2, by2 = d["bbox"]
            pw, ph = bx2 - bx1, by2 - by1
            aspect = pw / max(ph, 1)
            if (pw >= pcfg["min_plate_w"] and d["conf"] >= pcfg["min_conf"]
                    and pcfg["min_aspect"] <= aspect <= pcfg["max_aspect"]):
                good.append(d)
        if not good:
            continue
        p = max(good, key=lambda d: d["conf"])
        px1, py1, px2, py2 = p["bbox"]
        crop = vehicle[max(py1, 0):py2, max(px1, 0):px2].copy()
        if crop.size == 0:
            continue

        score, sharp = quality_score(crop)
        cv2.rectangle(frame, (x1 + px1, y1 + py1), (x1 + px2, y1 + py2), (0, 165, 255), 2)
        cv2.putText(frame, f"plate {crop.shape[1]}px", (x1 + px1, max(y1 + py1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

        lst = best.setdefault(tid, [])
        lst.append({"score": score, "crop": crop, "frame": idx,
                    "w": crop.shape[1], "sharp": sharp, "conf": p["conf"]})
        lst.sort(key=lambda b: b["score"], reverse=True)
        del lst[pcfg["keep_top"]:]

    writer.write(frame)
    # cv2.imshow("phase6", frame)
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break

print(f"\n{'track':>6} {'rank':>5} {'plate_w(px)':>12} {'sharp':>8} {'det_conf':>9} {'frame':>7}")
for tid, lst in sorted(best.items()):
    for rank, b in enumerate(lst, 1):
        cv2.imwrite(f"data/evidence/plates/track{tid}_r{rank}_f{b['frame']}.png", b["crop"])
        print(f"{tid:>6} {rank:>5} {b['w']:>12} {b['sharp']:>8.1f} {b['conf']:>9.2f} {b['frame']:>7}")
print(f"\nPlates found for {len(best)} tracks. Crops saved in data/evidence/plates/")

src.release(); writer.release(); cv2.destroyAllWindows()

