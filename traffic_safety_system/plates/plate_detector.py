from ultralytics import YOLO

class PlateDetector:
    def __init__(self, cfg):
        p = cfg["plates"]
        self.model = YOLO(p["model"])
        self.conf = p["conf"]
        self.imgsz = p["imgsz"]

    def detect(self, vehicle_img):
        """Returns plates found in a vehicle crop: list of {bbox, conf}.
        bbox is relative to the crop, not the full frame."""
        r = self.model.predict(vehicle_img, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        out = []
        if r.boxes is not None:
            for box, cf in zip(r.boxes.xyxy.cpu().numpy(),
                               r.boxes.conf.cpu().tolist()):
                out.append({"bbox": tuple(map(int, box)), "conf": cf})
        return out
 