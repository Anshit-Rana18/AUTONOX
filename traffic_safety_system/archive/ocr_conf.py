import cv2, glob, os, re
from collections import defaultdict
from fast_plate_ocr import LicensePlateRecognizer

TRUTH = {1: "NA13NRU", 7: "AP05JEO", 11: "BG65USJ"}   # add more as you read them

m = LicensePlateRecognizer("cct-s-v2-global-model")
rows = defaultdict(list)

for f in sorted(glob.glob("data/evidence/plates/*.png")):
    tid = int(re.search(r"track(\d+)_", os.path.basename(f)).group(1))
    p = m.run(cv2.imread(f), return_confidence=True)[0]
    plate = re.sub(r"[^A-Z0-9]", "", p.plate.upper())
    probs = [float(x) for x in p.char_probs[:len(plate)]]   # assumes padding is at the end
    minp = min(probs) if probs else 0.0
    rows[tid].append((plate, minp))

    note = ""
    if tid in TRUTH:
        if plate == TRUTH[tid]:
            note = "  OK"
        else:
            note = f"  WRONG (truth {TRUTH[tid]})"
    print(f"track {tid:>4} {os.path.basename(f):26} {plate:9} "
          f"min_char_p={minp:.4f} region={p.region} ({p.region_prob:.2f}){note}")
    if note.startswith("  WRONG"):
        print("      per-char:", " ".join(f"{c}:{x:.3f}" for c, x in zip(plate, probs)))

print("\n--- Per track ---")
for tid in sorted(rows):
    plates = [r[0] for r in rows[tid]]
    agree = len(plates) >= 2 and len(set(plates)) == 1
    lowest = min(r[1] for r in rows[tid])
    print(f"track {tid:>4}: {'AGREE   ' if agree else 'DISAGREE'} "
          f"reads={plates} lowest_min_p={lowest:.4f}")

