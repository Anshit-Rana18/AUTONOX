import cv2, easyocr, torch, os, glob

reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
os.makedirs("data/outputs/ocr_debug", exist_ok=True)
files = sorted(glob.glob("data/evidence/plates/track11_r1*.png") +
               glob.glob("data/evidence/plates/track7_r1*.png") +
               glob.glob("data/evidence/plates/track1_r1*.png"))

def variants(crop):
    h, w = crop.shape[:2]
    up = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    yield "original", crop
    yield "up3_color", up
    yield "up3_gray", gray
    yield "up3_gray_clahe", cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield "up3_otsu", otsu

for f in files:
    name = os.path.basename(f)[:-4]
    crop = cv2.imread(f)
    print(f"\n=== {name}  size={crop.shape[1]}x{crop.shape[0]} ===")
    for vname, img in variants(crop):
        cv2.imwrite(f"data/outputs/ocr_debug/{name}_{vname}.png", img)
        res = reader.readtext(img, detail=1, paragraph=False)
        text = " | ".join(f"{r[1]} ({r[2]:.2f})" for r in res)
        print(f"{vname:16} -> {text or '(nothing)'}")

