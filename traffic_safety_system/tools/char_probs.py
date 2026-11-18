import cv2, sys
from fast_plate_ocr import LicensePlateRecognizer

m = LicensePlateRecognizer("cct-s-v2-global-model")
p = m.run(cv2.imread(sys.argv[1]), return_confidence=True)[0]
print(p.plate)
print(" ".join(f"{c}:{x:.3f}" for c, x in zip(p.plate, p.char_probs)))

