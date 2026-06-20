import cv2
import numpy as np


class VirtualFence:
    def __init__(self):
        self.points: list[tuple[int, int]] = []

    def mouse_callback(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.points.clear()

    @property
    def ready(self) -> bool:
        return len(self.points) >= 3

    def contains(self, point: tuple[int, int]) -> bool:
        if not self.ready:
            return False
        polygon = np.array(self.points, np.int32)
        return cv2.pointPolygonTest(polygon, point, False) >= 0

    def draw(self, frame) -> None:
        for point in self.points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)

        if len(self.points) >= 2:
            polygon = np.array(self.points, np.int32)
            cv2.polylines(frame, [polygon], isClosed=self.ready, color=(255, 0, 0), thickness=2)

