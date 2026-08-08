from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class Detection:    #保存一次人体检测结果，包括人体框、置信度和类别标签
    bbox: tuple[int, int, int, int]
    confidence: float
    label: str = "person"


class PersonDetector:    #人体检测器，优先使用 YOLOv8，加载失败时使用 OpenCV HOG 备用检测
    """Person detector with YOLO when available and OpenCV HOG as fallback."""

    def __init__(self, model_path: str = "models/yolov8n.pt", confidence: float = 0.45):
        self.confidence = confidence
        self.backend = "hog"
        self.model = None
        self.hog = None

        try:
            from ultralytics import YOLO    #导入 YOLO

            model_file = self._resolve_model_path(model_path)
            self.model = YOLO(str(model_file))    #加载模型
            self.backend = "yolo"
        except Exception:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def _resolve_model_path(self, model_path: str) -> Path | str:    #查找 YOLOv8 模型路径，优先使用 models/yolov8n.pt
        candidates = [
            Path(model_path),
            Path("yolov8n.pt"),
            Path("models") / "yolov8n.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return "yolov8n.pt"

    def detect(self, frame) -> list[Detection]:    #对输入视频帧进行人体检测，并返回检测到的人体列表
        if self.backend == "yolo":    #检测入口
            return self._detect_yolo(frame)
        return self._detect_hog(frame)

    def _detect_yolo(self, frame) -> list[Detection]:    #使用 YOLOv8 检测人体，只保留 person 类别，YOLOv8 实际检测函数
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

    def _detect_hog(self, frame) -> list[Detection]:    #当 YOLO 不可用时，使用 OpenCV HOG 进行备用人体检测，效果很差
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
