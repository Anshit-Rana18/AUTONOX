import cv2, glob, os, re, yaml
from collections import defaultdict
from plates.plate_reader import PlateReader

cfg = yaml.safe_load(open("configs/default.yaml"))
reader = PlateReader(cfg)

TRUTH = {1: "NA13NRU", 3: "MW51VSU",  7: "AP05JEO", 8: "KH05ZZK", 11: "BG65USJ", 12: "FJ14ZHY", 18: "WR02FKD", 19: "LM13VCV", 26: "EY61NBG", 51: "AV08HVF", 56: "HN14CDV", 63: "NA54KGJ", 80: "AK64DMV", 120: "EF10DZT"}   # paste your full TRUTH dictionary from ocr_compare.py here

reads = defaultdict(list)
for f in sorted(glob.glob("data/evidence/plates/*.png")):
    tid = int(re.search(r"track(\d+)_", os.path.basename(f)).group(1))
    reads[tid].append(reader.read(cv2.imread(f)))

ok = bad = 0
for tid in sorted(reads):
    status, plate = reader.decide(reads[tid])
    t = TRUTH.get(tid)
    note = ""
    if status == "CONFIRMED" and t:
        if plate == t:
            ok += 1; note = "correct"
        else:
            bad += 1; note = f"WRONG (truth {t})"
    elif t:
        note = f"(truth {t}, read {'matches' if plate == t else 'differs'})"
    print(f"track {tid:>4} {status:11} {plate or '-':9} {note}")

print(f"\nConfirmed and correct: {ok}   Confirmed but WRONG: {bad}")


