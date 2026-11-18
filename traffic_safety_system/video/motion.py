import json

import cv2, numpy as np

class MotionCompensator:
    """Estimates how the current frame is shifted relative to the reference frame.
    H maps current-frame pixels to reference-frame pixels.
    mask: optional uint8 image (255 = use, 0 = ignore) so features are taken
    only from static parts of the scene."""
    def __init__(self, ref_frame, nfeatures=4000, min_inliers=15, max_shift=0.30, mask=None):
        self.mask = mask                       # must be set before it is used below
        self.orb = cv2.ORB_create(nfeatures)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
        self.kp_ref, self.des_ref = self.orb.detectAndCompute(gray, self.mask)
        self.h, self.w = gray.shape
        self.min_inliers, self.max_shift = min_inliers, max_shift
        self.H = np.eye(3)
        self.ok, self.inliers = True, 0

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, self.mask)
        self.ok = False
        if des is None or self.des_ref is None:
            return self.H
        m = sorted(self.bf.match(des, self.des_ref), key=lambda x: x.distance)[:400]
        if len(m) < self.min_inliers:
            return self.H
        src = np.float32([kp[x.queryIdx].pt for x in m]).reshape(-1, 1, 2)
        dst = np.float32([self.kp_ref[x.trainIdx].pt for x in m]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None or int(mask.sum()) < self.min_inliers:
            return self.H
        if self.max_shift_px(H) > self.max_shift * self.w:   # reject wild estimates
            return self.H
        self.H, self.ok, self.inliers = H, True, int(mask.sum())
        return self.H

    def max_shift_px(self, H=None):
        H = self.H if H is None else H
        c = np.float32([[0, 0], [self.w, 0], [self.w, self.h], [0, self.h]]).reshape(-1, 1, 2)
        moved = cv2.perspectiveTransform(c, H).reshape(-1, 2)
        return float(np.abs(moved - c.reshape(-1, 2)).max())

    def to_ref(self, x, y):
        p = cv2.perspectiveTransform(np.float32([[[x, y]]]), self.H)[0, 0]
        return float(p[0]), float(p[1])

    def poly_to_cur(self, pts):
        inv = np.linalg.inv(self.H)
        return cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), inv).reshape(-1, 2)

def make_compensator(cfg, ref_frame):
    """Builds the same MotionCompensator for every script."""
    h, w = ref_frame.shape[:2]
    mask = None
    if cfg["video"].get("motion_mask", "none") == "speed_zone":
        cal = json.load(open(cfg["speed"]["calibration"]))
        mask = np.full((h, w), 255, np.uint8)
        cv2.fillPoly(mask, [np.array(cal["image_points"], np.int32)], 0)
    return MotionCompensator(ref_frame, mask=mask)
