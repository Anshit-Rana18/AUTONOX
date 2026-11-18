import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX


def fit(img, max_w=1280, max_h=720):
    """Returns (display_image, scale). Only shrinks, never enlarges."""
    h, w = img.shape[:2]
    s = min(1.0, max_w / w, max_h / h)
    if s < 1.0:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img, s


def show(name, img, max_w=1280, max_h=720):
    cv2.imshow(name, fit(img, max_w, max_h)[0])


def draw_scale(w, h, box_w=900, box_h=640):
    """How much a w x h frame is shrunk to fit the display box (never below 1)."""
    return max(1.0, w / box_w, h / box_h)


def label_style(w, h, box_w=900, box_h=640):
    """Text size and line thickness that look the same on screen for any video size.
    box_w, box_h: the area the frame is shown in (web page about 900 x 640,
    OpenCV window 1280 x 720)."""
    k = draw_scale(w, h, box_w, box_h)
    return {"k": k, "fs": 0.42 * k, "th": max(1, round(0.8 * k)), "bt": max(1, round(1.2 * k))}


def put_text(frame, text, x, y, color, scale, thick):
    """Text with a dark outline, readable on any background (used for HUD lines)."""
    cv2.putText(frame, text, (x, y), FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)

