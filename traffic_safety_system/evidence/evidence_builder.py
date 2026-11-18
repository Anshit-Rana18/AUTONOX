import cv2, json, os, numpy as np

GREEN, AMBER, RED, GREY = (0, 200, 0), (0, 190, 255), (0, 0, 220), (90, 90, 90)
PLATE_W = 420
MAX_VEHICLE_W = 520

def _resize_w(img, w):
    s = w / img.shape[1]
    return cv2.resize(img, (w, max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_CUBIC)

def build_evidence(out_dir, event_id, camera_id, ts, event_type, track_id,
                   vehicle, plate_crop, status, plate_text, alt_reads=(), extra=None,
                   header="TRAFFIC EVENT", detail_lines=None, header_color=(255, 255, 255)):
    folder = os.path.join(out_dir, event_id)
    os.makedirs(folder, exist_ok=True)
    lines = list(detail_lines or [])

    # Originals are saved untouched; only the on-screen plate view is enlarged.
    cv2.imwrite(os.path.join(folder, "vehicle.png"), vehicle)
    if plate_crop is not None:
        cv2.imwrite(os.path.join(folder, "plate_original.png"), plate_crop)

    v = vehicle if vehicle.shape[1] <= MAX_VEHICLE_W else _resize_w(vehicle, MAX_VEHICLE_W)
    if plate_crop is not None:
        p = _resize_w(plate_crop, PLATE_W)
    else:
        p = np.full((110, PLATE_W, 3), 40, np.uint8)
        cv2.putText(p, "NO PLATE CROP", (110, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREY, 2)

    pad, top, row_h = 16, 56, max(v.shape[0], p.shape[0] + 40)
    W = pad + v.shape[1] + pad + PLATE_W + pad
    H = top + row_h + 190 + 28 * len(lines)
    canvas = np.full((H, W, 3), 30, np.uint8)

    cv2.putText(canvas, f"{header}  {event_id}", (pad, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, header_color, 2)
    canvas[top:top + v.shape[0], pad:pad + v.shape[1]] = v
    cv2.putText(canvas, "Vehicle", (pad, top + row_h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    x0 = pad + v.shape[1] + pad
    canvas[top:top + p.shape[0], x0:x0 + PLATE_W] = p
    cv2.putText(canvas, "Plate close-up (enlarged for display)", (x0, top + p.shape[0] + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    col = {"CONFIRMED": GREEN, "UNVERIFIED": AMBER}.get(status, RED)
    if status == "CONFIRMED":
        plate_line = f"Plate: {plate_text}   [CONFIRMED]"
    elif status == "UNVERIFIED":
        plate_line = f"Plate candidate: {plate_text}?   [NEEDS HUMAN CHECK]"
    else:
        plate_line = "Plate: unreadable"
    y = top + row_h + 52
    cv2.putText(canvas, plate_line, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    if alt_reads:
        cv2.putText(canvas, "Other reads: " + ", ".join(alt_reads), (pad, y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    info = f"Event: {event_type}   Track: #{track_id}   Camera: {camera_id}   Video time: {ts:.2f}s"
    cv2.putText(canvas, info, (pad, y + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(canvas, "Review status: PENDING   (not a confirmed violation)",
                (pad, y + 92), cv2.FONT_HERSHEY_SIMPLEX, 0.55, AMBER, 1)
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (pad, y + 126 + 28 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, header_color, 2)

    cv2.imwrite(os.path.join(folder, "screen.png"), canvas)
    record = {"event_id": event_id, "camera_id": camera_id, "track_id": track_id,
              "event_type": event_type, "video_time_s": round(ts, 2),
              "plate_status": status, "plate_text": plate_text,
              "other_reads": list(alt_reads), "review_status": "pending",
              **(extra or {})}
    json.dump(record, open(os.path.join(folder, "record.json"), "w"), indent=2)
    return os.path.join(folder, "screen.png")

