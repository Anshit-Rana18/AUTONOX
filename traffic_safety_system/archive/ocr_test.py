import cv2, glob, os, csv, re, yaml
from collections import Counter, defaultdict
from plates.plate_ocr import PlateOCR, validate

ocr = PlateOCR()
cfg = yaml.safe_load(open("configs/default.yaml"))
files = sorted(glob.glob("data/evidence/plates/*.png"))
rows, by_track = [], defaultdict(list)

for f in files:
    m = re.search(r"track(\d+)_r(\d+)", os.path.basename(f))
    tid = int(m.group(1))
    crop = cv2.imread(f)
    raw, conf = ocr.read(crop)
    fixed, valid = validate(raw, cfg["plates"]["ocr"])
    state_ok = valid
    rows.append([tid, os.path.basename(f), raw, fixed, valid, state_ok, round(conf, 2), ""])
    if valid and state_ok:
        by_track[tid].append(fixed)
    print(f"track {tid:>4} {os.path.basename(f):28} raw={raw:14} fixed={fixed:12} "
          f"valid={valid} state={state_ok} conf={conf:.2f}")

print("\n--- Per-track result ---")
for tid in sorted({r[0] for r in rows}):
    reads = by_track.get(tid, [])
    if not reads:
        print(f"track {tid:>4}: UNREADABLE (no valid-format read)")
        continue
    text, n = Counter(reads).most_common(1)[0]
    status = "CONFIRMED" if n >= 2 else "UNCERTAIN"
    print(f"track {tid:>4}: {status:10} {text}   (reads: {reads})")

with open("data/outputs/ocr_results.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["track", "file", "raw", "fixed", "valid", "state_ok", "conf", "ground_truth"])
    w.writerows(rows)
print("\nSaved data/outputs/ocr_results.csv")