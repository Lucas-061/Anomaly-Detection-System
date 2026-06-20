from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    confidence: float
    label: str = "person"


class PersonDetector:
    """Person detector with YOLO when available and OpenCV HOG as fallback."""

    def __init__(self, model_path: str = "models/yolov8n.pt", confidence: float = 0.45):
        self.confidence = confidence
        self.backend = "hog"
        self.model = None
        self.hog = None

        try:
            from ultralytics import YOLO

            model_file = self._resolve_model_path(model_path)
            self.model = YOLO(str(model_file))
            self.backend = "yolo"
        except Exception:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def _resolve_model_path(self, model_path: str) -> Path | str:
        candidates = [
            Path(model_path),
            Path("yolov8n.pt"),
            Path("models") / "yolov8n.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return "yolov8n.pt"

    def detect(self, frame) -> list[Detection]:
        if self.backend == "yolo":
            return self._detect_yolo(frame)
        return self._detect_hog(frame)

    def _detect_yolo(self, frame) -> list[Detection]:
        detections: list[Detection] = []
        results = self.model.predict(frame, conf=self.confidence, verbose=False)

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                if class_id != 0 or conf < self.confidence:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(Detection((x1, y1, x2, y2), conf))

        return detections

    def _detect_hog(self, frame) -> list[Detection]:
        resized = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
        scale_x = frame.shape[1] / resized.shape[1]
        scale_y = frame.shape[0] / resized.shape[0]

        rects, weights = self.hog.detectMultiScale(
            resized,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )

        detections: list[Detection] = []
        for (x, y, w, h), weight in zip(rects, weights):
            conf = float(weight)
            if conf < 0.4:
                continue
            x1 = int(x * scale_x)
            y1 = int(y * scale_y)
            x2 = int((x + w) * scale_x)
            y2 = int((y + h) * scale_y)
            detections.append(Detection((x1, y1, x2, y2), min(conf, 1.0)))

        return detections
