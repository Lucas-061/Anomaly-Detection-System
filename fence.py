import cv2
import numpy as np


class VirtualFence:    #虚拟围栏类，保存围栏点、绘制围栏并判断点是否在围栏内部
    def __init__(self):
        self.points: list[tuple[int, int]] = []

    def mouse_callback(self, event, x, y, flags, param) -> None:    #旧版 OpenCV 鼠标回调函数，目前 PyQt6 中主要使用 handle_video_click 替代，在ui_main.py
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.points.clear()

    @property
    def ready(self) -> bool:    #判断围栏点数量是否足够形成多边形区域，顺时针画点
        return len(self.points) >= 3

    def contains(self, point: tuple[int, int]) -> bool:    #判断某个点，例如人体中心点，是否位于虚拟围栏内部
        if not self.ready:
            return False
        polygon = np.array(self.points, np.int32)
        return cv2.pointPolygonTest(polygon, point, False) >= 0    #虚拟围栏绘制和点是否在围栏内判断

    def draw(self, frame) -> None:    #在视频帧上绘制虚拟围栏点和多边形边界
        for point in self.points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)

        if len(self.points) >= 2:
            polygon = np.array(self.points, np.int32)
            cv2.polylines(frame, [polygon], isClosed=self.ready, color=(255, 0, 0), thickness=2)

