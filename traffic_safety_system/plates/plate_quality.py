import cv2

def sharpness(img):
    """Variance of the Laplacian: higher means sharper, low means blurry."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def quality_score(crop):
    """Heuristic: wide AND sharp plates score higher. Not a legal standard,
    just a way to pick the best observation of a plate."""
    h, w = crop.shape[:2]
    if w < 10 or h < 5:
        return 0.0, 0.0
    s = sharpness(crop)
    return w * (s / (s + 100.0)), s
