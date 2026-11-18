import cv2, glob, os, re
from collections import Counter, defaultdict
from plates.plate_ocr import PlateOCR, fix_uk
from fast_plate_ocr import LicensePlateRecognizer

TRUTH = {1: "NA13NRU", 3: "MW51VSU",  7: "AP05JEO", 8: "KH05ZZK", 11: "BG65USJ", 12: "FJ14ZHY", 18: "WR02FKD", 19: "LM13VCV", 26: "EY61NBG", 51: "AV08HVF", 63: "NA54KGJ", 80: "AK64DMV", 120: "EF10DZT"}   # edit if you read them differently

easy = PlateOCR()
fpo = LicensePlateRecognizer("cct-s-v2-global-model")

def fpo_read(crop):
    r = fpo.run(crop)
    r = r[0] if isinstance(r, (list, tuple)) else r
    return re.sub(r"[^A-Z0-9]", "", str(getattr(r, "plate", r)).upper())

def vote(reads):
    reads = [r for r in reads if len(r) == 7]
    if not reads:
        return ""
    return "".join(Counter(r[i] for r in reads).most_common(1)[0][0] for i in range(7))

engines = {
    "easy_plain": lambda c: easy.read_plain(c)[0],
    "easy_prep":  lambda c: easy.read(c)[0],
    "fastplate":  fpo_read,
}
per_track = {name: defaultdict(list) for name in engines}

for f in sorted(glob.glob("data/evidence/plates/*.png")):
    tid = int(re.search(r"track(\d+)_", os.path.basename(f)).group(1))
    crop = cv2.imread(f)
    line = f"track {tid:>4} {os.path.basename(f):26}"
    for name, fn in engines.items():
        raw = fn(crop)
        fixed, valid = fix_uk(raw)
        per_track[name][tid].append(fixed if valid else "")
        line += f" | {name}: {raw or '-':10}->{fixed or '-':8}"
    if tid in TRUTH:
        line += f"  [truth {TRUTH[tid]}]"
    print(line)

print("\n--- Voted result per track (per-position majority over crops) ---")
for name in engines:
    print(f"\n[{name}]")
    hits = 0
    for tid in sorted(per_track[name]):
        v = vote(per_track[name][tid])
        mark = ""
        if tid in TRUTH:
            ok = v == TRUTH[tid]
            hits += ok
            mark = "  OK" if ok else f"  WRONG (truth {TRUTH[tid]})"
        print(f"  track {tid:>4}: {v or 'UNREADABLE'}{mark}")
    print(f"  exact matches on known plates: {hits}/{len(TRUTH)}")

