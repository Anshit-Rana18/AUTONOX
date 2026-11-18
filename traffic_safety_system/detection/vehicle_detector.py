from ultralytics import YOLO

class VehicleTracker:
    def __init__(self, cfg):
        self.model = YOLO(cfg["detector"]["model"])
        self.conf = cfg["detector"]["conf"]
        self.classes = cfg["detector"]["classes"]
        self.imgsz = cfg["detector"]["imgsz"]
        self.tracker = cfg["tracker"]["type"]

    def update(self, frame):
        """Returns a list of dicts: track_id, cls, conf, bbox (x1,y1,x2,y2)."""
        r = self.model.track(frame, persist=True, tracker=self.tracker,
                             conf=self.conf, classes=self.classes,
                             imgsz=self.imgsz, agnostic_nms=True, verbose=False)[0]
        out = []
        if r.boxes is not None and r.boxes.id is not None:
            for box, tid, c, cf in zip(r.boxes.xyxy.cpu().numpy(),
                                       r.boxes.id.int().cpu().tolist(),
                                       r.boxes.cls.int().cpu().tolist(),
                                       r.boxes.conf.cpu().tolist()):
                out.append({"track_id": tid, "cls": self.model.names[c],
                            "conf": cf, "bbox": tuple(map(int, box))})
        return out


