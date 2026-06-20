from pathlib import Path
import csv
import sys
import time

import cv2
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from alarm import AlarmManager, SAVE_ALARM_TYPES
from behavior import BehaviorAnalyzer
from detector import PersonDetector
from fence import VirtualFence
from tracker import CentroidTracker
from video_clip import AlarmClipRecorder


ALARM_NAMES = {
    "intrusion": "违规闯入",
    "cross_fence": "翻越围栏",
    "long_stay": "长时间滞留",
    "fall_down": "摔倒",
    "running": "奔跑",
    "climbing": "攀爬",
}

ALARM_NAMES_EN = {
    "intrusion": "Intrusion",
    "cross_fence": "Cross Fence",
    "long_stay": "Long Stay",
    "fall_down": "Fall Down",
    "running": "Running",
    "climbing": "Climbing",
}

MODE_NAMES_EN = {
    "摄像头识别": "Camera Recognition",
    "文件夹视频识别": "Video File Recognition",
}

WARNING_COLORS = {
    "一级预警": "#c19a00",
    "二级预警": "#d06f00",
    "三级预警": "#c50f1f",
}

WARNING_NAMES_EN = {
    "一级预警": "Level 1",
    "二级预警": "Level 2",
    "三级预警": "Level 3",
}

WARNING_STATUS_STYLES = {
    "一级预警": "color: #6f5200; background: #fff4ce; border: 1px solid #f1d36b;",
    "二级预警": "color: #7a3b00; background: #fde7d3; border: 1px solid #f3b36b;",
    "三级预警": "color: #a80000; background: #fde7e9; border: 1px solid #f1aeb5;",
    "异常": "color: #a80000; background: #fde7e9; border: 1px solid #f1aeb5;",
}

CRITICAL_WARNING_LEVEL = "三级预警"
CRITICAL_WARNING_REARM_FRAMES = 60

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v"}
TRAIN_VIDEO_DIR = Path(__file__).resolve().parent / "TrainVedio"
ALARM_LOG_FILE = Path(__file__).resolve().parent / "records" / "alarm_log.csv"


class VideoLabel(QLabel):
    frame_clicked = pyqtSignal(int, int, object)

    def __init__(self):
        super().__init__("请选择识别方式")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(960, 620)
        self.setObjectName("videoLabel")
        self.setWordWrap(True)
        self.setMouseTracking(True)
        self.frame_size: tuple[int, int] | None = None
        self.pixmap_rect = None

    def show_message(self, text: str) -> None:
        self.clear()
        self.setText(text)
        self.frame_size = None
        self.pixmap_rect = None

    def show_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb.shape
        self.frame_size = (width, height)
        bytes_per_line = channel * width
        image = QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        offset_x = int((self.width() - pixmap.width()) / 2)
        offset_y = int((self.height() - pixmap.height()) / 2)
        self.pixmap_rect = (offset_x, offset_y, pixmap.width(), pixmap.height())
        self.setText("")
        self.setPixmap(pixmap)

    def mousePressEvent(self, event) -> None:
        frame_point = self.to_frame_point(int(event.position().x()), int(event.position().y()))
        if frame_point is not None:
            self.frame_clicked.emit(frame_point[0], frame_point[1], event.button())
        super().mousePressEvent(event)

    def to_frame_point(self, label_x: int, label_y: int) -> tuple[int, int] | None:
        if self.frame_size is None or self.pixmap_rect is None:
            return None

        offset_x, offset_y, pixmap_width, pixmap_height = self.pixmap_rect
        if not (offset_x <= label_x <= offset_x + pixmap_width and offset_y <= label_y <= offset_y + pixmap_height):
            return None

        frame_width, frame_height = self.frame_size
        frame_x = int((label_x - offset_x) * frame_width / pixmap_width)
        frame_y = int((label_y - offset_y) * frame_height / pixmap_height)
        frame_x = max(0, min(frame_width - 1, frame_x))
        frame_y = max(0, min(frame_height - 1, frame_y))
        return frame_x, frame_y


class AlarmLogDialog(QMainWindow):
    def __init__(self, log_file: Path, parent=None):
        super().__init__(parent)
        self.log_file = log_file
        self.setWindowTitle("报警记录")
        self.resize(1350, 620)

        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels(["时间", "视频来源", "人员ID", "报警类型", "报警名称", "级别", "截图路径", "视频片段"])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.verticalHeader().setDefaultSectionSize(34)
        self.table.verticalHeader().setMinimumSectionSize(30)
        self.table.verticalHeader().setFixedWidth(52)
        self.table.horizontalHeader().setMinimumSectionSize(72)
        self.setCentralWidget(self.table)
        self.load_records()

        self.setStyleSheet(
            """
            QMainWindow {
                background: #f3f3f3;
            }
            QTableWidget {
                background: #ffffff;
                alternate-background-color: #fafafa;
                border: 1px solid #d1d1d1;
                gridline-color: #eeeeee;
                color: #242424;
                font-size: 15px;
            }
            QHeaderView::section {
                background: #f5f5f5;
                color: #1b1b1b;
                border: 0;
                border-right: 1px solid #e5e5e5;
                border-bottom: 1px solid #d1d1d1;
                padding: 7px;
                font-weight: 600;
            }
            """
        )

    def load_records(self) -> None:
        self.table.setRowCount(0)
        if not self.log_file.exists():
            return

        with self.log_file.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = [
                row.get("time", ""),
                row.get("source", ""),
                row.get("track_id", ""),
                row.get("alarm_type", ""),
                row.get("alarm_name", ""),
                row.get("level", ""),
                row.get("screenshot", ""),
                row.get("video_clip", ""),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                level_color = WARNING_COLORS.get(row.get("level", ""))
                if level_color:
                    item.setForeground(QColor(level_color))
                self.table.setItem(row_index, column, item)

        self.apply_table_layout()

    def apply_table_layout(self) -> None:
        header = self.table.horizontalHeader()
        for column in range(self.table.columnCount()):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)

        widths = [190, 220, 90, 125, 110, 80, 310, 360]
        for column, width in enumerate(widths):
            self.table.setColumnWidth(column, width)

        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于摄像头的异常行为识别系统")
        self.resize(1500, 850)

        self.detector = PersonDetector(model_path="models/yolov8n.pt", confidence=0.45)
        self.alarm_manager = AlarmManager()
        self.clip_recorder = AlarmClipRecorder()
        self.analyzer = BehaviorAnalyzer(stay_seconds=8.0)
        self.tracker = CentroidTracker()
        self.fence = VirtualFence()

        self.capture = None
        self.current_source = None
        self.current_mode = "idle"
        self.prepared_frame = None
        self.pending_video_source: str | None = None
        self.pending_video_mode: str | None = None
        self.last_motion_status_by_track: dict[int, tuple[str, str]] = {}
        self.motion_status_repeat_count: dict[int, int] = {}
        self.saved_critical_alarm_keys: set[tuple[int, str]] = set()
        self.critical_alarm_absent_frames: dict[tuple[int, str], int] = {}
        self.last_frame_at = time.time()
        self.fps = 0.0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.storage_error_timer = QTimer(self)
        self.storage_error_timer.timeout.connect(self.show_storage_errors)
        self.storage_error_timer.start(1000)

        self.video_label = VideoLabel()
        self.video_label.frame_clicked.connect(self.handle_video_click)
        self.status_label = QLabel("待机")
        self.status_label.setObjectName("statusLabel")
        self.mode_label = QLabel("未选择识别方式")
        self.mode_label.setObjectName("modeLabel")
        self.data_list = QListWidget()
        self.data_list.setObjectName("dataList")

        self.choose_button = QPushButton("识别方式选择")
        self.choose_button.setObjectName("primaryButton")
        self.choose_button.clicked.connect(self.show_mode_menu)

        self.return_button = QPushButton("退出识别")
        self.return_button.setObjectName("dangerButton")
        self.return_button.clicked.connect(self.return_home)
        self.return_button.setEnabled(False)

        self.playback_button = QPushButton("播放")
        self.playback_button.setObjectName("secondaryButton")
        self.playback_button.clicked.connect(self.play_prepared_video)
        self.playback_button.setEnabled(False)

        self.alarm_log_button = QPushButton("报警记录")
        self.alarm_log_button.setObjectName("secondaryButton")
        self.alarm_log_button.clicked.connect(self.show_alarm_log)

        self.alarm_log_window = None

        self.setup_ui()
        self.return_home()

    def setup_ui(self) -> None:
        central = QWidget()
        central.setObjectName("page")
        root = QHBoxLayout(central)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(24)

        video_title = QLabel("异常行为识别监控")
        video_title.setObjectName("videoTitle")
        video_caption = QLabel("实时画面")
        video_caption.setObjectName("videoCaption")

        video_header = QHBoxLayout()
        video_header.setContentsMargins(0, 0, 0, 0)
        video_header.addWidget(video_title)
        video_header.addStretch(1)
        video_header.addWidget(video_caption)
        video_header.addWidget(self.status_label)

        video_panel = QFrame()
        video_panel.setObjectName("videoPanel")
        video_panel_layout = QVBoxLayout(video_panel)
        video_panel_layout.setContentsMargins(20, 18, 20, 20)
        video_panel_layout.setSpacing(14)
        video_panel_layout.addLayout(video_header)
        video_panel_layout.addWidget(self.video_label, stretch=1)

        root.addWidget(video_panel, stretch=1)

        side = QVBoxLayout()
        side.setSpacing(18)

        side_panel = QFrame()
        side_panel.setObjectName("sidePanel")
        side_panel_layout = QVBoxLayout(side_panel)
        side_panel_layout.setContentsMargins(18, 18, 18, 18)
        side_panel_layout.setSpacing(18)

        data_title = QLabel("数据显示框")
        data_title.setObjectName("panelTitle")
        data_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        data_panel = QVBoxLayout()
        data_panel.setSpacing(10)
        data_panel.setContentsMargins(16, 16, 16, 16)
        data_panel.addWidget(data_title)
        data_panel.addWidget(self.mode_label)
        data_panel.addWidget(self.data_list)

        data_widget = QFrame()
        data_widget.setObjectName("dataPanel")
        data_widget.setLayout(data_panel)
        data_widget.setMinimumHeight(520)

        button_title = QLabel("操作面板")
        button_title.setObjectName("sectionTitle")

        actions = QVBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(12)
        actions.addWidget(button_title)
        actions.addWidget(self.choose_button)
        actions.addWidget(self.return_button)
        actions.addWidget(self.playback_button)
        actions.addWidget(self.alarm_log_button)

        side_panel_layout.addWidget(data_widget, stretch=1)
        side_panel_layout.addLayout(actions)
        side.addWidget(side_panel)

        root.addLayout(side)
        self.setCentralWidget(central)

        self.setStyleSheet(
            """
            QWidget#page {
                background: #f3f3f3;
            }
            QFrame#videoPanel, QFrame#sidePanel {
                background: #ffffff;
                border: 1px solid #e5e5e5;
                border-radius: 8px;
            }
            QFrame#sidePanel {
                min-width: 360px;
                max-width: 390px;
            }
            QLabel#videoTitle {
                color: #1b1b1b;
                font-family: "Segoe UI", "Microsoft YaHei";
                font-size: 24px;
                font-weight: 600;
            }
            QLabel#videoCaption {
                color: #616161;
                font-family: "Segoe UI", "Microsoft YaHei";
                font-size: 14px;
                font-weight: 500;
                padding-right: 8px;
            }
            QLabel#statusLabel {
                border-radius: 12px;
                padding: 4px 12px;
                font-size: 14px;
                font-weight: 600;
            }
            QLabel#videoLabel {
                border: 1px solid #d1d1d1;
                border-radius: 6px;
                background: #fafafa;
                color: #242424;
                font-family: "Segoe UI", "Microsoft YaHei";
                font-size: 40px;
                font-weight: 600;
            }
            QLabel#panelTitle {
                color: #1b1b1b;
                min-height: 34px;
                font-family: "Segoe UI", "Microsoft YaHei";
                font-size: 23px;
                font-weight: 600;
            }
            QLabel#modeLabel {
                color: #424242;
                background: #f8f8f8;
                border: 1px solid #e5e5e5;
                border-radius: 6px;
                padding: 8px 10px;
                font-size: 14px;
            }
            QLabel#sectionTitle {
                color: #616161;
                font-size: 15px;
                font-weight: 600;
                padding-top: 4px;
            }
            QFrame#dataPanel {
                background: #ffffff;
                border: 1px solid #e5e5e5;
                border-radius: 8px;
            }
            QListWidget#dataList {
                border: 1px solid #e5e5e5;
                border-radius: 6px;
                background: #fafafa;
                color: #242424;
                font-size: 15px;
                padding: 8px;
                outline: 0;
            }
            QListWidget#dataList::item {
                min-height: 26px;
                padding: 4px 6px;
                border-bottom: 1px solid #eeeeee;
            }
            QListWidget#dataList::item:selected {
                background: #e5f1fb;
                color: #1b1b1b;
            }
            QPushButton {
                border: 0;
                border-radius: 5px;
                color: #ffffff;
                min-height: 52px;
                font-family: "Segoe UI", "Microsoft YaHei";
                font-size: 20px;
                font-weight: 600;
            }
            QPushButton#primaryButton {
                background: #0f6cbd;
            }
            QPushButton#primaryButton:hover {
                background: #115ea3;
            }
            QPushButton#primaryButton:pressed {
                background: #0f548c;
            }
            QPushButton#dangerButton {
                background: #c50f1f;
            }
            QPushButton#dangerButton:hover {
                background: #a80000;
            }
            QPushButton#dangerButton:pressed {
                background: #8f0000;
            }
            QPushButton#secondaryButton {
                background: #ffffff;
                color: #242424;
                border: 1px solid #d1d1d1;
            }
            QPushButton#secondaryButton:hover {
                background: #f5f5f5;
                border: 1px solid #c7c7c7;
            }
            QPushButton#secondaryButton:pressed {
                background: #ededed;
            }
            QPushButton:disabled {
                background: #f0f0f0;
                color: #a6a6a6;
                border: 1px solid #e0e0e0;
            }
            QMenu {
                background: white;
                border: 1px solid #d1d1d1;
                border-radius: 6px;
                font-size: 16px;
            }
            QMenu::item {
                padding: 10px 28px;
            }
            QMenu::item:selected {
                background: #e5f1fb;
                color: #0f6cbd;
            }
            """
        )

    def show_mode_menu(self) -> None:
        if self.timer.isActive():
            return

        menu = QMenu(self)
        camera_action = QAction("摄像头识别", self)
        folder_action = QAction("文件夹视频识别", self)
        camera_action.triggered.connect(self.start_camera)
        folder_action.triggered.connect(self.choose_folder_video)
        menu.addAction(camera_action)
        menu.addAction(folder_action)
        menu.exec(self.choose_button.mapToGlobal(self.choose_button.rect().bottomLeft()))

    def start_camera(self) -> None:
        self.start_source(0, "摄像头识别")

    def show_alarm_log(self) -> None:
        if not ALARM_LOG_FILE.exists():
            QMessageBox.information(self, "报警记录", "当前还没有报警记录。")
            return
        self.alarm_log_window = AlarmLogDialog(ALARM_LOG_FILE, self)
        self.alarm_log_window.show()

    def choose_folder_video(self) -> None:
        TRAIN_VIDEO_DIR.mkdir(exist_ok=True)
        file_filter = "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.m4v);;所有文件 (*)"
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择待识别视频",
            str(TRAIN_VIDEO_DIR),
            file_filter,
        )
        if not file_path:
            return

        video_path = Path(file_path)
        self.prepare_video_source(str(video_path), f"文件夹视频识别：{video_path.name}")

    def prepare_video_source(self, source: str, mode_name: str) -> None:
        self.stop_capture()
        self.current_source = source
        self.current_mode = mode_name
        self.pending_video_source = source
        self.pending_video_mode = mode_name
        self.tracker = CentroidTracker()
        self.fence = VirtualFence()
        self.analyzer = BehaviorAnalyzer(stay_seconds=8.0)
        self.reset_motion_display_state()
        self.fps = 0.0

        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            capture.release()
            self.video_label.show_message("视频源打开失败")
            self.set_status("异常")
            self.mode_label.setText("视频源打开失败")
            self.data_list.clear()
            self.add_data_item(str(source))
            self.return_button.setEnabled(True)
            self.playback_button.setEnabled(False)
            return

        ok, frame = capture.read()
        capture.release()
        if not ok:
            self.video_label.show_message("视频文件读取失败")
            self.set_status("异常")
            self.mode_label.setText("视频文件读取失败")
            self.data_list.clear()
            self.add_data_item(str(source))
            self.return_button.setEnabled(True)
            self.playback_button.setEnabled(False)
            return

        frame = self.resize_frame(frame)
        self.prepared_frame = frame
        self.data_list.clear()
        self.add_data_item(mode_name)
        self.add_data_item("已选择视频，请先绘制虚拟围栏，再点击播放")
        self.add_data_item("左键添加围栏点，右键清空围栏")
        self.set_status("待播放")
        self.mode_label.setText(mode_name)
        self.choose_button.setEnabled(False)
        self.return_button.setEnabled(True)
        self.playback_button.setEnabled(True)
        self.show_prepared_frame()

    def play_prepared_video(self) -> None:
        if self.pending_video_source is None or self.pending_video_mode is None:
            return
        self.start_source(self.pending_video_source, self.pending_video_mode, reset_fence=False)

    def start_source(self, source, mode_name: str, reset_fence: bool = True) -> None:
        self.stop_capture()
        self.current_source = source
        self.current_mode = mode_name
        self.tracker = CentroidTracker()
        if reset_fence:
            self.fence = VirtualFence()
        self.analyzer = BehaviorAnalyzer(stay_seconds=8.0)
        self.reset_motion_display_state()
        self.fps = 0.0
        self.last_frame_at = time.time()

        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            self.capture = None
            self.video_label.show_message("视频源打开失败")
            self.set_status("异常")
            self.mode_label.setText("视频源打开失败")
            self.data_list.clear()
            self.add_data_item(str(source))
            self.return_button.setEnabled(True)
            return

        source_fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.clip_recorder.reset(fps=source_fps)
        self.data_list.clear()
        self.add_data_item(mode_name)
        self.add_data_item(f"检测后端：{self.detector.backend}")
        self.set_status("识别中")
        self.mode_label.setText(mode_name)
        self.choose_button.setEnabled(False)
        self.return_button.setEnabled(True)
        self.playback_button.setEnabled(False)
        self.timer.start(1)

    def handle_video_click(self, x: int, y: int, button) -> None:
        if self.capture is None and self.prepared_frame is None:
            return

        if button == Qt.MouseButton.LeftButton:
            self.fence.points.append((x, y))
            self.add_data_item(f"添加围栏点：({x}, {y})")
        elif button == Qt.MouseButton.RightButton:
            self.fence.points.clear()
            self.add_data_item("已清空虚拟围栏")
        self.data_list.scrollToBottom()
        if self.capture is None and self.prepared_frame is not None:
            self.show_prepared_frame()

    def update_frame(self) -> None:
        if self.capture is None:
            return

        ok, frame = self.capture.read()
        if not ok:
            self.finish_current_video()
            return

        if frame.shape[1] > 960:
            frame = self.resize_frame(frame)

        now = time.time()
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(now - self.last_frame_at, 1e-6))
        self.last_frame_at = now

        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections, frame)
        active_alarm_texts: list[str] = []

        self.fence.draw(frame)
        pending_alarm_events = []
        current_critical_alarm_keys = set()

        for track in tracks:
            inside_fence = self.fence.contains(track.center)
            alarms = self.analyzer.analyze(track, inside_fence)
            self.draw_track(frame, track, alarms)
            for alarm_type, level in alarms:
                text = f"ID {track.track_id}：{ALARM_NAMES[alarm_type]}（{level}）"
                active_alarm_texts.append(text)
                pending_alarm_events.append((track.track_id, alarm_type, level, text))
                if level == CRITICAL_WARNING_LEVEL and alarm_type in SAVE_ALARM_TYPES:
                    current_critical_alarm_keys.add((track.track_id, alarm_type))
                self.set_status(level)

        self.draw_overlay(frame, len(tracks), active_alarm_texts)
        self.update_critical_alarm_rearm(current_critical_alarm_keys)

        for track_id, alarm_type, level, text in pending_alarm_events:
            video_clip = ""
            should_save_alarm = self.should_save_alarm_file(track_id, alarm_type, level)
            if should_save_alarm:
                video_clip = self.clip_recorder.start_clip(alarm_type, track_id)
                screenshot = self.alarm_manager.trigger(
                    frame,
                    track_id,
                    alarm_type,
                    level,
                    source=str(self.current_source) if self.current_source is not None else "",
                    video_clip=video_clip,
                )
            else:
                screenshot = None
            if screenshot:
                text = f"{text} 已保存截图和片段"
            if self.should_display_motion_status(track_id, alarm_type, level):
                self.add_data_item(text, alarm_type=alarm_type, level=level)
                self.data_list.scrollToBottom()
            self.show_storage_errors()

        self.video_label.show_frame(frame)
        self.clip_recorder.add_frame(frame)
        self.show_storage_errors()

    def resize_frame(self, frame):
        if frame.shape[1] > 960:
            scale = 960 / frame.shape[1]
            return cv2.resize(frame, (960, int(frame.shape[0] * scale)))
        return frame

    def show_prepared_frame(self) -> None:
        if self.prepared_frame is None:
            return
        frame = self.prepared_frame.copy()
        self.fence.draw(frame)
        self.draw_overlay(frame, 0, [])
        self.video_label.show_frame(frame)

    def add_data_item(self, text: str, alarm_type: str | None = None, level: str | None = None) -> None:
        item = QListWidgetItem(text)
        color = WARNING_COLORS.get(level or "")
        if color:
            item.setForeground(QColor(color))
            font = item.font()
            font.setBold(True)
            item.setFont(font)
        elif alarm_type == "cross_fence":
            item.setForeground(QColor("#c50f1f"))
        self.data_list.addItem(item)

    def show_storage_errors(self) -> None:
        errors = self.alarm_manager.collect_errors() + self.clip_recorder.collect_errors()
        for error in errors:
            self.add_data_item(error, level="三级预警")
        if errors:
            self.data_list.scrollToBottom()

    def should_display_motion_status(self, track_id: int, alarm_type: str, level: str) -> bool:
        status_key = (alarm_type, level)
        if self.last_motion_status_by_track.get(track_id) != status_key:
            self.last_motion_status_by_track[track_id] = status_key
            self.motion_status_repeat_count[track_id] = 1
            return True

        repeat_count = self.motion_status_repeat_count.get(track_id, 0) + 1
        self.motion_status_repeat_count[track_id] = repeat_count
        return repeat_count <= 3

    def reset_motion_display_state(self) -> None:
        self.last_motion_status_by_track.clear()
        self.motion_status_repeat_count.clear()
        self.saved_critical_alarm_keys.clear()
        self.critical_alarm_absent_frames.clear()

    def should_save_alarm_file(self, track_id: int, alarm_type: str, level: str) -> bool:
        if alarm_type not in SAVE_ALARM_TYPES:
            return False

        if level != CRITICAL_WARNING_LEVEL:
            return True

        key = (track_id, alarm_type)
        if key in self.saved_critical_alarm_keys:
            return False

        self.saved_critical_alarm_keys.add(key)
        self.critical_alarm_absent_frames[key] = 0
        return True

    def update_critical_alarm_rearm(self, current_keys: set[tuple[int, str]]) -> None:
        for key in current_keys:
            self.critical_alarm_absent_frames[key] = 0

        for key in list(self.saved_critical_alarm_keys):
            if key in current_keys:
                continue

            absent_frames = self.critical_alarm_absent_frames.get(key, 0) + 1
            if absent_frames >= CRITICAL_WARNING_REARM_FRAMES:
                self.saved_critical_alarm_keys.discard(key)
                self.critical_alarm_absent_frames.pop(key, None)
            else:
                self.critical_alarm_absent_frames[key] = absent_frames

    def draw_track(self, frame, track, alarms: list[tuple[str, str]]) -> None:
        x1, y1, x2, y2 = track.bbox
        color = (0, 0, 255) if alarms else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, track.center, 4, color, -1)
        label = f"ID {track.track_id}"
        if alarms:
            label += " " + ",".join(ALARM_NAMES_EN[name] for name, _ in alarms)
        cv2.putText(frame, label, (x1, max(y1 - 8, 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        for start, end in zip(track.history[-20:-1], track.history[-19:]):
            cv2.line(frame, start, end, (0, 255, 255), 2)

    def draw_overlay(self, frame, track_count: int, alarms: list[str]) -> None:
        mode_text = self.current_mode
        for chinese, english in MODE_NAMES_EN.items():
            if mode_text.startswith(chinese):
                if "：" in mode_text:
                    mode_text = f"{english}: {mode_text.split('：', 1)[1]}"
                else:
                    mode_text = english
                break

        lines = [
            mode_text,
            f"FPS: {self.fps:.1f}  Persons: {track_count}",
            f"Fence points: {len(self.fence.points)}  Left:add  Right:clear",
        ]
        lines.extend(self.to_english_alarm_text(text) for text in alarms[:3])

        for index, text in enumerate(lines):
            y = 28 + index * 28
            color = (0, 0, 255) if index >= 2 else (255, 255, 255)
            cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 3)
            cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 1)

    def to_english_alarm_text(self, text: str) -> str:
        result = text
        for alarm_id, chinese_name in ALARM_NAMES.items():
            result = result.replace(chinese_name, ALARM_NAMES_EN[alarm_id])
        for chinese_name, english_name in WARNING_NAMES_EN.items():
            result = result.replace(chinese_name, english_name)
        result = result.replace("（", " (").replace("）", ")").replace("：", ": ")
        return result

    def finish_current_video(self) -> None:
        self.clip_recorder.finish_all()
        self.stop_capture()
        self.video_label.show_message("识别完成，可再次点击播放重新识别")
        self.set_status("已完成")
        self.mode_label.setText("识别完成")
        self.add_data_item("识别完成，可再次点击播放重新识别，或点击退出识别返回主页面")
        self.data_list.scrollToBottom()
        self.choose_button.setEnabled(False)
        self.return_button.setEnabled(True)
        self.playback_button.setEnabled(self.pending_video_source is not None)

    def return_home(self) -> None:
        self.stop_capture()
        self.current_mode = "idle"
        self.current_source = None
        self.prepared_frame = None
        self.pending_video_source = None
        self.pending_video_mode = None
        self.reset_motion_display_state()
        self.fence = VirtualFence()
        self.video_label.show_message("请选择识别方式")
        self.set_status("待机")
        self.mode_label.setText("未选择识别方式")
        self.data_list.clear()
        self.add_data_item("等待选择识别方式")
        self.choose_button.setEnabled(True)
        self.return_button.setEnabled(False)
        self.playback_button.setEnabled(False)

    def stop_capture(self) -> None:
        self.timer.stop()
        self.clip_recorder.finish_all()
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)
        warning_style = WARNING_STATUS_STYLES.get(text)
        if warning_style:
            self.status_label.setStyleSheet(warning_style)
        else:
            self.status_label.setStyleSheet(
                "color: #0f6cbd; background: #eef6fc; border: 1px solid #cfe4f5;"
            )

    def closeEvent(self, event) -> None:
        self.stop_capture()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei UI", 10))
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
