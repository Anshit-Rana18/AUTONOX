import json

TRUTH = {"NA13NRU", "MW51VSU", "AP05JEO", "KH05ZZK", "BG65USJ", "FJ14ZHY",
         "WR02FKD", "LM13VCV", "EY61NBG", "AV08HVF", "NA54KGJ", "AK64DMV", "EF10DZT"}

res = json.load(open("data/outputs/plate_results.json"))
wrong = 0
for tid, r in sorted(res.items(), key=lambda kv: int(kv[0])):
    if r["status"] != "CONFIRMED":
        continue
    p = r["plate"]
    if p in TRUTH:
        print(f"track {tid:>4} {p}  matches truth")
    else:
        wrong += 1
        near = min(TRUTH, key=lambda t: sum(a != b for a, b in zip(p, t)) + abs(len(p) - len(t)))
        print(f"track {tid:>4} {p}  NOT in truth list (closest: {near}) -> check the crop")
print(f"\nConfirmed but not in truth list: {wrong}")

