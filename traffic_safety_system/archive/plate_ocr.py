import re
import cv2
import easyocr
import torch

ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Characters OCR commonly confuses, used only where the plate format
# tells us whether a position must be a digit or a letter.
TO_DIGIT = {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1",
            "Z": "2", "S": "5", "B": "8", "G": "6"}
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"}

# Indian state / UT series codes. Check against your region; may be incomplete.
STATE_CODES = {"AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "GA", "GJ",
               "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN",
               "MP", "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TR",
               "TS", "TG", "UK", "UP", "WB"}

PLATE_RE = re.compile(r"[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}")

UK_RE = re.compile(r"[A-Z]{2}\d{2}[A-Z]{3}")
UK_TO_DIGIT = {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1",
               "Z": "2", "S": "5", "B": "8", "G": "6", "J": "3"}
UK_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"}

class PlateOCR:
    def __init__(self):
        self.reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

    def preprocess(self, crop):
        h, w = crop.shape[:2]
        scale = max(2.0, 300.0 / w)
        big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def read(self, crop):
        """Returns (raw_text, mean_confidence)."""
        img = self.preprocess(crop)
        res = self.reader.readtext(img, allowlist=ALLOW, detail=1, paragraph=False)
        if not res:
            return "", 0.0
        H = img.shape[0]

        def key(r):
            ys = [p[1] for p in r[0]]
            row = 0 if (min(ys) + max(ys)) / 2 < H / 2 else 1   # two-line plates
            return (row, r[0][0][0])

        res.sort(key=key)
        text = "".join(r[1] for r in res)
        conf = sum(r[2] for r in res) / len(res)
        return text, float(conf)

    def read_plain(self, crop):
        """No resizing or contrast changes. Returns (text, mean_conf)."""
        res = self.reader.readtext(crop, allowlist=ALLOW, detail=1, paragraph=False)
        if not res:
            return "", 0.0
        res.sort(key=lambda r: r[0][0][0])
        return "".join(r[1] for r in res), float(sum(r[2] for r in res) / len(res))


def fix_format(text):
    """Apply the Indian plate layout (SS NN XX.. NNNN) to correct look-alike
    characters by position. Returns (fixed_text, format_valid, state_ok)."""
    t = re.sub(r"[^A-Z0-9]", "", text.upper())
    n = len(t)
    if n not in (9, 10, 11):
        return t, False, False
    kinds = "LL" + "DD" + "L" * (n - 8) + "DDDD"
    out = []
    for ch, k in zip(t, kinds):
        out.append(TO_DIGIT.get(ch, ch) if k == "D" else TO_LETTER.get(ch, ch))
    s = "".join(out)
    return s, bool(PLATE_RE.fullmatch(s)), s[:2] in STATE_CODES

def validate(text, cfg):
    """cfg is the plates.ocr section of the config.
    Returns (cleaned_text, valid)."""
    t = re.sub(r"[^A-Z0-9]", "", text.upper())
    mode = cfg.get("format", "generic")
    if mode == "india":
        fixed, valid, state_ok = fix_format(text)
        return fixed, valid and state_ok
    lo, hi = cfg.get("min_len", 5), cfg.get("max_len", 8)
    ok = lo <= len(t) <= hi
    if cfg.get("regex"):
        ok = ok and bool(re.fullmatch(cfg["regex"], t))
    if cfg.get("need_digit_and_letter", True):
        ok = ok and any(c.isdigit() for c in t) and any(c.isalpha() for c in t)
    return t, ok

def fix_uk(text):
    """Positional correction for the layout LL DD LLL. Returns (text, valid)."""
    t = re.sub(r"[^A-Z0-9]", "", text.upper())
    if len(t) != 7:
        return t, False
    kinds = "LLDDLLL"
    s = "".join((UK_TO_DIGIT.get(c, c) if k == "D" else UK_TO_LETTER.get(c, c))
                for c, k in zip(t, kinds))
    return s, bool(UK_RE.fullmatch(s))