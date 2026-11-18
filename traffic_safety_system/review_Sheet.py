import cv2, json, glob, numpy as np
import sys

res = json.load(open("data/outputs/plate_results.json"))
only = {int(a) for a in sys.argv[1:]}
W, COLS = 360, 3
tiles = []
for tid, r in sorted(res.items(), key=lambda kv: int(kv[0])):
    files = sorted(glob.glob(f"data/evidence/plates/track{tid}_r1_*.png"))
    if not files or not r["plate"]:
        continue
    img = cv2.imread(files[0])
    img = cv2.resize(img, (W, max(1, int(img.shape[0] * W / img.shape[1]))),
                     interpolation=cv2.INTER_CUBIC)
    tile = np.full((130, W, 3), 30, np.uint8)
    h = min(img.shape[0], 80)
    tile[0:h, :] = img[:h]
    cv2.putText(tile, f"#{tid} {r['status'][:4]}  {r['plate']}", (6, 112),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    tiles.append(tile)

if not tiles:
    raise SystemExit("No tiles: run run_phase6b.py first and check data/evidence/plates/")

while len(tiles) % COLS:
    tiles.append(np.full((130, W, 3), 30, np.uint8))
rows = [np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)]
cv2.imwrite("data/outputs/review_sheet.png", np.vstack(rows))
print(f"Saved data/outputs/review_sheet.png with {len(tiles)} tiles")

