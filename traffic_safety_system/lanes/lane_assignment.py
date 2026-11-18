import json
import cv2, numpy as np
from shapely.geometry import Point, Polygon, LineString

class LaneAssigner:
    def __init__(self, lanes_path="configs/lanes.json"):
        data = json.load(open(lanes_path))
        self.lanes = [(l["name"], Polygon(l["polygon"])) for l in data["lanes"]]

    def _span(self, poly, x, y):
        """Left and right edge of the polygon on the horizontal line through y, around x."""
        minx, _, maxx, _ = poly.bounds
        inter = poly.intersection(LineString([(minx - 1, y), (maxx + 1, y)]))
        for s in (list(inter.geoms) if hasattr(inter, "geoms") else [inter]):
            if not s.is_empty and s.bounds[0] <= x <= s.bounds[2]:
                return s.bounds[0], s.bounds[2]
        return None

    def assign_ex(self, bbox, comp=None, margin_frac=0.0):
        """Returns (lane_name or None, reason). reason: ok | outside | dead"""
        x1, y1, x2, y2 = bbox
        x, y = (x1 + x2) / 2, y2
        if comp is not None:
            x, y = comp.to_ref(x, y)
        p = Point(x, y)
        for name, poly in self.lanes:
            if poly.contains(p):
                if margin_frac:
                    span = self._span(poly, x, y)
                    if span:
                        dead = margin_frac * (span[1] - span[0])
                        if x - span[0] < dead or span[1] - x < dead:
                            return None, "dead"
                return name, "ok"
        return None, "outside"

    def assign(self, bbox, comp=None, margin_frac=0.0):
        return self.assign_ex(bbox, comp, margin_frac)[0]

    def draw(self, frame, comp=None):
        overlay = frame.copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
        for i, (name, poly) in enumerate(self.lanes):
            pts = np.array(poly.exterior.coords)
            if comp is not None:
                pts = comp.poly_to_cur(pts)
            cv2.fillPoly(overlay, [pts.astype(np.int32)], colors[i % len(colors)])
        return cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)

    