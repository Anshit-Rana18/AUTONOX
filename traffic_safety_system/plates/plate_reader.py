import re
from fast_plate_ocr import LicensePlateRecognizer

class PlateReader:
    def __init__(self, cfg):
        pc = cfg["plates"]
        o = dict(pc["ocr"])
        o.update(pc.get("ocr_profiles", {}).get(pc.get("ocr_profile"), {}))
        self.model = LicensePlateRecognizer(o.get("model", "cct-s-v2-global-model"))
        self.regex = re.compile(o["regex"]) if o.get("regex") else None
        self.region = o.get("expected_region")
        self.min_region_p = o.get("min_region_p", 0.9)
        self.agree_p = o.get("agree_min_char_p", 0.80)
        self.min_agree = o.get("min_agree", 2)

    def read(self, crop):
        p = self.model.run(crop, return_confidence=True)[0]
        plate = re.sub(r"[^A-Z0-9]", "", p.plate.upper())
        probs = [float(x) for x in p.char_probs[:len(plate)]]
        return {"plate": plate, "min_p": min(probs) if probs else 0.0,
                "region": p.region, "region_p": float(p.region_prob or 0)}

    def valid(self, r):
        if not r["plate"]:
            return False
        if self.regex and not self.regex.fullmatch(r["plate"]):
            return False
        if self.region and (r["region"] != self.region or r["region_p"] < self.min_region_p):
            return False
        return True

    def decide(self, reads):
        valid = [r for r in reads if self.valid(r)]
        if not valid:
            return "UNREADABLE", None
        counts = {}
        for r in valid:
            c = counts.setdefault(r["plate"], {"n": 0, "strong": 0, "best": 0.0})
            c["n"] += 1
            c["best"] = max(c["best"], r["min_p"])
            if r["min_p"] >= self.agree_p:
                c["strong"] += 1
        top, c = max(counts.items(),
                     key=lambda kv: (kv[1]["strong"], kv[1]["n"], kv[1]["best"]))
        dissent = len(valid) - c["n"]       # valid reads that name a different plate      
        if c["strong"] >= self.min_agree and dissent == 0:
            return "CONFIRMED", top
        return "UNVERIFIED", top
