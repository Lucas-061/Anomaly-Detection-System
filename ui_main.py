from pathlib import Path
import csv
import sys
import time

import cv2
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
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


ALARM_NAMES = {    #预警项目
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
    "一级预警": "#e7d06a",
    "二级预警": "#f2a65a",
    "三级预警": "#ff6b6b",
}

WARNING_NAMES_EN = {
    "一级预警": "Level 1",
    "二级预警": "Level 2",
    "三级预警": "Level 3",
}

WARNING_STATUS_STYLES = {    #数据显示框
    "一级预警": "color: #e7d06a; background: #3f3822; border: 1px solid #7d6c2c;",
    "二级预警": "color: #f2a65a; background: #46321f; border: 1px solid #8a5b2c;",
    "三级预警": "color: #ff6b6b; background: #4a2525; border: 1px solid #9a4646;",
    "异常": "color: #ff6b6b; background: #4a2525; border: 1px solid #9a4646;",
}

CRITICAL_WARNING_LEVEL = "三级预警"
CRITICAL_WARNING_REARM_FRAMES = 60

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v"}
TRAIN_VIDEO_DIR = Path(__file__).resolve().parent / "TrainVedio"
RECORDS_DIR = Path(__file__).resolve().parent / "records"
ALARM_LOG_FILE = Path(__file__).resolve().parent / "records" / "alarm_log.csv"


class VideoLabel(QLabel):    #视频显示控件，用于在 PyQt6 界面中显示 OpenCV 视频帧，并接收鼠标点击绘制围栏
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

    def show_message(self, text: str) -> None:    #在视频显示框中显示提示文字
        self.clear()
        self.setText(text)
        self.frame_size = None
        self.pixmap_rect = None

    def show_frame(self, frame) -> None:    #将 OpenCV 的 BGR 图像帧转换为 PyQt6 可显示的 QPixmap，并显示到视频画面框
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

    def mousePressEvent(self, event) -> None:    #捕获用户在视频画面上的鼠标点击事件
        frame_point = self.to_frame_point(int(event.position().x()), int(event.position().y()))
        if frame_point is not None:
            self.frame_clicked.emit(frame_point[0], frame_point[1], event.button())
        super().mousePressEvent(event)

    def to_frame_point(self, label_x: int, label_y: int) -> tuple[int, int] | None:    #将 PyQt6 控件坐标转换为真实视频帧坐标，保证围栏点画在正确的视频位
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


class AlarmLogDialog(QMainWindow):    #报警记录查看窗口，用表格形式显示 alarm_log.csv 中的历史报警信息
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
                background: #1a1a1a;
                font-family: "Consolas", "SimSun", "Microsoft YaHei UI";
            }
            QTableWidget {
                background: #303030;
                alternate-background-color: #3a3a3a;
                border: 1px solid #4a4a4a;
                gridline-color: #4b4b4b;
                color: #eeeeee;
                font-family: "Consolas", "SimSun", "Microsoft YaHei UI";
                font-size: 15px;
                selection-background-color: #4b5964;
                selection-color: #ffffff;
            }
            QHeaderView::section {
                background: #3a3a3a;
                color: #edf0f2;
                font-family: "SimSun", "Microsoft YaHei UI";
                border: 0;
                border-right: 1px solid #555555;
                border-bottom: 1px solid #555555;
                padding: 9px;
                font-weight: 600;
            }
            QScrollBar:vertical, QScrollBar:horizontal {
                background: #2a2a2a;
                width: 12px;
                height: 12px;
                margin: 0;
            }
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
                background: #696969;
                min-height: 26px;
                min-width: 26px;
                border-radius: 0;
            }
            QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
                background: #858585;
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                width: 0;
                height: 0;
            }
            """
        )

    def load_records(self) -> None:    #读取 records/alarm_log.csv，将报警记录加载到表格中
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

    def apply_table_layout(self) -> None:    #设置报警记录表格的列宽和显示方式
        header = self.table.horizontalHeader()
        for column in range(self.table.columnCount()):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)

        widths = [190, 220, 90, 125, 110, 80, 310, 360]
        for column, width in enumerate(widths):
            self.table.setColumnWidth(column, width)

        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)


class MainWindow(QMainWindow):    #系统主窗口，负责界面显示、按钮交互、视频读取、识别调度和报警处理
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
        self.session_screenshots: list[str] = []
        self.session_video_clips: list[str] = []
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
        self.data_list.setAlternatingRowColors(True)

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

    def setup_ui(self) -> None:    #搭建主界面布局，包括视频显示区、数据显示框、操作按钮和整体样式
        central = QWidget()
        central.setObjectName("page")
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(14)

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
        video_panel_layout.setContentsMargins(12, 12, 12, 12)
        video_panel_layout.setSpacing(12)
        video_panel_layout.addLayout(video_header)
        video_panel_layout.addWidget(self.video_label, stretch=1)

        root.addWidget(video_panel, stretch=1)

        side = QVBoxLayout()
        side.setSpacing(12)

        side_panel = QFrame()
        side_panel.setObjectName("sidePanel")
        side_panel_layout = QVBoxLayout(side_panel)
        side_panel_layout.setContentsMargins(12, 12, 12, 12)
        side_panel_layout.setSpacing(12)

        data_title = QLabel("数据显示框")
        data_title.setObjectName("panelTitle")
        data_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        data_panel = QVBoxLayout()
        data_panel.setSpacing(10)
        data_panel.setContentsMargins(12, 12, 12, 12)
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
            QWidget {
                font-family: "SimSun", "Microsoft YaHei UI";
            }
            QWidget#page {
                background: #1a1a1a;
            }
            QFrame#videoPanel, QFrame#sidePanel {
                background: #232323;
                border: 1px solid #3a3a3a;
                border-radius: 0;
            }
            QFrame#sidePanel {
                min-width: 370px;
                max-width: 410px;
            }
            QLabel#videoTitle {
                color: #e8edf2;
                font-family: "SimSun", "Microsoft YaHei UI";
                font-size: 22px;
                font-weight: 600;
                padding: 4px 0 8px 0;
            }
            QLabel#videoCaption {
                color: #9ca7b2;
                font-family: "SimSun", "Microsoft YaHei UI";
                font-size: 14px;
                font-weight: 500;
                padding-right: 8px;
            }
            QLabel#statusLabel {
                border-radius: 0;
                padding: 6px 12px;
                font-size: 14px;
                font-weight: 600;
            }
            QLabel#videoLabel {
                border: 1px solid #444444;
                border-radius: 0;
                background: #111111;
                color: #b8c1cb;
                font-family: "SimSun", "Microsoft YaHei UI";
                font-size: 36px;
                font-weight: 600;
            }
            QLabel#panelTitle {
                color: #f0f3f6;
                min-height: 36px;
                font-family: "SimSun", "Microsoft YaHei UI";
                font-size: 20px;
                font-weight: 600;
                background: #333333;
                border: 1px solid #454545;
                padding: 4px 8px;
            }
            QLabel#modeLabel {
                color: #e2e2e2;
                background: #333333;
                border: 1px solid #454545;
                border-left: 4px solid #1aa0e8;
                border-radius: 0;
                padding: 10px 12px;
                font-size: 14px;
            }
            QLabel#sectionTitle {
                color: #dfe5eb;
                font-size: 16px;
                font-weight: 600;
                padding: 12px 0 2px 0;
            }
            QFrame#dataPanel {
                background: #2a2a2a;
                border: 1px solid #444444;
                border-radius: 0;
            }
            QListWidget#dataList {
                border: 1px solid #4a4a4a;
                border-radius: 0;
                background: #2f2f2f;
                alternate-background-color: #383838;
                color: #eeeeee;
                font-family: "Consolas", "SimSun", "Microsoft YaHei UI";
                font-size: 15px;
                padding: 6px;
                outline: 0;
            }
            QListWidget#dataList::item {
                min-height: 30px;
                padding: 6px 8px;
                border-bottom: 1px solid #4b4b4b;
            }
            QListWidget#dataList::item:selected {
                background: #4b5964;
                color: #ffffff;
            }
            QPushButton {
                border: 1px solid #505050;
                border-radius: 0;
                color: #e8e8e8;
                min-height: 52px;
                font-family: "SimSun", "Microsoft YaHei UI";
                font-size: 18px;
                font-weight: 600;
                padding: 5px 16px;
            }
            QPushButton#primaryButton {
                background: #343434;
                border-color: #555555;
            }
            QPushButton#primaryButton:hover {
                background: #3f3f3f;
                border-color: #1aa0e8;
            }
            QPushButton#primaryButton:pressed {
                background: #292929;
                border-color: #1380bb;
            }
            QPushButton#dangerButton {
                background: #343434;
                border-color: #555555;
                color: #ffb3b3;
            }
            QPushButton#dangerButton:hover {
                background: #3f3f3f;
                border-color: #ff6b6b;
            }
            QPushButton#dangerButton:pressed {
                background: #292929;
                border-color: #c94f4f;
            }
            QPushButton#secondaryButton {
                background: #343434;
                color: #e8e8e8;
                border: 1px solid #555555;
            }
            QPushButton#secondaryButton:hover {
                background: #3f3f3f;
                border: 1px solid #7a7a7a;
            }
            QPushButton#secondaryButton:pressed {
                background: #292929;
            }
            QPushButton:disabled {
                background: #282828;
                color: #777777;
                border: 1px solid #3a3a3a;
            }
            QMenu {
                background: #dedede;
                color: #202020;
                border: 1px solid #777777;
                border-radius: 0;
                font-size: 16px;
            }
            QMenu::item {
                padding: 11px 28px;
                background: transparent;
                border: 1px solid transparent;
            }
            QMenu::item:selected {
                background: #cfcfcf;
                color: #101010;
                border: 1px solid #1aa0e8;
            }
            QScrollBar:vertical {
                background: #2a2a2a;
                width: 12px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #696969;
                min-height: 26px;
                border-radius: 0;
            }
            QScrollBar::handle:vertical:hover {
                background: #858585;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
            """
        )

    def show_mode_menu(self) -> None:    #点击“识别方式选择”后弹出菜单，让用户选择摄像头识别或文件夹视频识别
        if self.timer.isActive():
            return

        menu = QMenu(self)
        camera_action = QAction("摄像头识别", self)    #摄像头识别
        folder_action = QAction("文件夹视频识别", self)
        camera_action.triggered.connect(self.start_camera)
        folder_action.triggered.connect(self.choose_folder_video)
        menu.addAction(camera_action)
        menu.addAction(folder_action)
        menu.exec(self.choose_button.mapToGlobal(self.choose_button.rect().bottomLeft()))

    def start_camera(self) -> None:    #启动摄像头识别模式
        self.start_source(0, "摄像头识别")

    def show_alarm_log(self) -> None:    #打开报警记录窗口，显示历史报警 CSV 数据
        if not ALARM_LOG_FILE.exists():
            self.show_styled_message("报警记录", "当前还没有报警记录。", QMessageBox.Icon.Information)
            return
        self.alarm_log_window = AlarmLogDialog(ALARM_LOG_FILE, self)
        self.alarm_log_window.show()

    def choose_folder_video(self) -> None:    #打开文件选择窗口，自选视频文件
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

    def prepare_video_source(self, source: str, mode_name: str) -> None:    #选择视频后先读取第一帧并显示，等待用户绘制虚拟围栏和点击播放，不立即开始识别
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

    def play_prepared_video(self) -> None:    #用户点击“播放”后，开始识别之前选择好的视频
        if self.pending_video_source is None or self.pending_video_mode is None:
            return
        self.start_source(self.pending_video_source, self.pending_video_mode, reset_fence=False)

    def start_source(self, source, mode_name: str, reset_fence: bool = True) -> None:    #打开摄像头或视频文件，初始化跟踪器、行为分析器、报警片段缓存，并启动定时器开始逐帧识别
        self.stop_capture()
        self.current_source = source
        self.current_mode = mode_name
        self.session_screenshots.clear()
        self.session_video_clips.clear()
        self.tracker = CentroidTracker()
        if reset_fence:
            self.fence = VirtualFence()
        self.analyzer = BehaviorAnalyzer(stay_seconds=8.0)
        self.reset_motion_display_state()
        self.fps = 0.0
        self.last_frame_at = time.time()

        self.capture = cv2.VideoCapture(source)    #打开摄像头或本地视频文件，cv2.VideoCapture(0)表示打开摄像头
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

    def handle_video_click(self, x: int, y: int, button) -> None:     #处理视频画面上的鼠标点击。左键添加围栏点，右键清空围栏
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

    def update_frame(self) -> None:    #系统识别主循环函数。每次读取一帧视频，完成 YOLO 检测、目标跟踪、围栏判断、行为识别、报警保存和界面刷新
        if self.capture is None:
            return

        ok, frame = self.capture.read()    # openCV逐帧读取摄像头画，根指令
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
                if video_clip:
                    self.session_video_clips.append(video_clip)
                screenshot = self.alarm_manager.trigger(
                    frame,
                    track_id,
                    alarm_type,
                    level,
                    source=str(self.current_source) if self.current_source is not None else "",
                    video_clip=video_clip,
                )
                if screenshot:
                    self.session_screenshots.append(screenshot)
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

    def resize_frame(self, frame):    #当视频宽度过大时，将画面缩放到合适宽度，降低识别和显示压力
        if frame.shape[1] > 960:
            scale = 960 / frame.shape[1]
            return cv2.resize(frame, (960, int(frame.shape[0] * scale)))
        return frame

    def show_prepared_frame(self) -> None:    #在视频未播放前显示第一帧，并叠加用户绘制的虚拟围栏
        if self.prepared_frame is None:
            return
        frame = self.prepared_frame.copy()
        self.fence.draw(frame)
        self.draw_overlay(frame, 0, [])
        self.video_label.show_frame(frame)

    def add_data_item(self, text: str, alarm_type: str | None = None, level: str | None = None) -> None:    #向右侧数据显示框添加信息，并根据预警等级设置文字颜色
        item = QListWidgetItem(text)
        color = WARNING_COLORS.get(level or "")
        if color:
            item.setForeground(QColor(color))
            font = item.font()
            font.setBold(True)
            item.setFont(font)
        elif alarm_type == "cross_fence":
            item.setForeground(QColor("#ff6b6b"))
        self.data_list.addItem(item)

    def show_storage_errors(self) -> None:    #收集截图、CSV、视频片段保存过程中的错误，并显示到右侧数据显示框
        errors = self.alarm_manager.collect_errors() + self.clip_recorder.collect_errors()
        for error in errors:
            self.add_data_item(error, level="三级预警")
        if errors:
            self.data_list.scrollToBottom()

    def should_display_motion_status(self, track_id: int, alarm_type: str, level: str) -> bool:    #限制右侧数据显示框重复刷屏。同一目标同一状态最多显示 3 条
        status_key = (alarm_type, level)
        if self.last_motion_status_by_track.get(track_id) != status_key:
            self.last_motion_status_by_track[track_id] = status_key
            self.motion_status_repeat_count[track_id] = 1
            return True

        repeat_count = self.motion_status_repeat_count.get(track_id, 0) + 1
        self.motion_status_repeat_count[track_id] = repeat_count
        return repeat_count <= 3

    def reset_motion_display_state(self) -> None:    #重置右侧数据显示框的运动状态去重记录
        self.last_motion_status_by_track.clear()
        self.motion_status_repeat_count.clear()
        self.saved_critical_alarm_keys.clear()
        self.critical_alarm_absent_frames.clear()

    def should_save_alarm_file(self, track_id: int, alarm_type: str, level: str) -> bool:    #判断某次报警是否需要保存截图和视频片段
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

    def update_critical_alarm_rearm(self, current_keys: set[tuple[int, str]]) -> None:    #当三级预警状态消失一定帧数后，重新允许下一次同类三级预警保存，防止长期误屏蔽
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

    def draw_track(self, frame, track, alarms: list[tuple[str, str]]) -> None:    #在视频帧上绘制人体框、人员 ID、报警名称和运动轨
        x1, y1, x2, y2 = track.bbox
        color = (0, 0, 255) if alarms else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)    #画检测框、ID、轨迹、报警文字
        cv2.circle(frame, track.center, 4, color, -1)
        label = f"ID {track.track_id}"
        if alarms:
            label += " " + ",".join(ALARM_NAMES_EN[name] for name, _ in alarms)
        cv2.putText(frame, label, (x1, max(y1 - 8, 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        for start, end in zip(track.history[-20:-1], track.history[-19:]):
            cv2.line(frame, start, end, (0, 255, 255), 2)

    def draw_overlay(self, frame, track_count: int, alarms: list[str]) -> None:    #在视频左上角绘制当前模式、FPS、人数、围栏点数和报警提示
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

    def to_english_alarm_text(self, text: str) -> str:    #将中文报警文本转换成英文，用于 OpenCV 画面显示，避免中文乱码
        result = text
        for alarm_id, chinese_name in ALARM_NAMES.items():
            result = result.replace(chinese_name, ALARM_NAMES_EN[alarm_id])
        for chinese_name, english_name in WARNING_NAMES_EN.items():
            result = result.replace(chinese_name, english_name)
        result = result.replace("（", " (").replace("）", ")").replace("：", ": ")
        return result

    def finish_current_video(self) -> None:    #识别结束后停止识别，保存未完成的报警片段，并提示用户可重新播放或退出识别
        self.clip_recorder.finish_all(wait=True)
        self.stop_capture(finish_clips=False)
        self.ask_keep_session_records()
        self.video_label.show_message("识别完成，可再次点击播放重新识别")
        self.set_status("已完成")
        self.mode_label.setText("识别完成")
        self.add_data_item("识别完成，可再次点击播放重新识别，或点击退出识别返回主页面")
        self.data_list.scrollToBottom()
        self.choose_button.setEnabled(False)
        self.return_button.setEnabled(True)
        self.playback_button.setEnabled(self.pending_video_source is not None)

    def ask_keep_session_records(self) -> None:
        if not self.session_screenshots and not self.session_video_clips:
            return

        keep_records = self.ask_keep_records_dialog()
        if not keep_records:
            removed_files, removed_rows = self.discard_session_records()
            self.add_data_item(f"已删除本次记录：文件 {removed_files} 个，CSV 记录 {removed_rows} 条")
        else:
            self.add_data_item("已保留本次报警截图、视频片段和报警记录")

        self.session_screenshots.clear()
        self.session_video_clips.clear()

    def ask_keep_records_dialog(self) -> bool:
        dialog = QDialog(self)
        dialog.setWindowTitle("保留本次记录")
        dialog.setModal(True)

        root = QVBoxLayout(dialog)
        root.setContentsMargins(18, 18, 18, 16)
        root.setSpacing(16)

        content = QHBoxLayout()
        content.setSpacing(14)

        icon = QLabel("?")
        icon.setFixedSize(44, 44)
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setObjectName("dialogQuestionIcon")

        message = QLabel("本次识别已产生报警截图、视频片段或报警记录，是否保留本次记录？")
        message.setWordWrap(False)
        message.setObjectName("dialogMessage")

        content.addWidget(icon)
        content.addWidget(message)
        root.addLayout(content)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        keep_button = QPushButton("保留")
        discard_button = QPushButton("不保留")
        keep_button.clicked.connect(dialog.accept)
        discard_button.clicked.connect(dialog.reject)
        buttons.addWidget(keep_button)
        buttons.addWidget(discard_button)
        buttons.addStretch(1)
        root.addLayout(buttons)

        dialog.setStyleSheet(
            """
            QDialog {
                background: #242424;
                color: #eeeeee;
                font-family: "SimSun", "Microsoft YaHei UI";
            }
            QLabel#dialogQuestionIcon {
                background: #1497dd;
                color: #ffffff;
                border-radius: 22px;
                font-size: 28px;
            }
            QLabel#dialogMessage {
                color: #eeeeee;
                font-size: 16px;
                padding: 2px 0;
            }
            QPushButton {
                background: #343434;
                color: #e8e8e8;
                border: 1px solid #555555;
                min-width: 96px;
                min-height: 34px;
                font-size: 15px;
            }
            QPushButton:hover {
                background: #3f3f3f;
                border: 1px solid #1aa0e8;
            }
            QPushButton:pressed {
                background: #292929;
                border: 1px solid #1380bb;
            }
            """
        )
        return dialog.exec() == QDialog.DialogCode.Accepted

    def show_styled_message(self, title: str, text: str, icon: QMessageBox.Icon) -> None:
        box = self.create_styled_message_box(title, text, icon)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.button(QMessageBox.StandardButton.Ok).setText("确定")
        box.exec()

    def create_styled_message_box(self, title: str, text: str, icon: QMessageBox.Icon) -> QMessageBox:
        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setText(text)
        box.setIcon(icon)
        box.setStyleSheet(
            """
            QMessageBox {
                background: #242424;
                color: #eeeeee;
                font-family: "SimSun", "Microsoft YaHei UI";
                font-size: 15px;
            }
            QMessageBox QLabel {
                color: #eeeeee;
                background: transparent;
                min-width: 280px;
                max-width: 360px;
                padding: 6px 4px;
                line-height: 1.4;
                qproperty-alignment: AlignLeft;
            }
            QMessageBox QPushButton {
                background: #343434;
                color: #e8e8e8;
                border: 1px solid #555555;
                min-width: 78px;
                min-height: 32px;
                padding: 4px 12px;
                font-family: "SimSun", "Microsoft YaHei UI";
                font-size: 14px;
            }
            QMessageBox QPushButton:hover {
                background: #3f3f3f;
                border: 1px solid #1aa0e8;
            }
            QMessageBox QPushButton:pressed {
                background: #292929;
                border: 1px solid #1380bb;
            }
            """
        )
        return box

    def discard_session_records(self) -> tuple[int, int]:
        removed_files = 0
        paths = [Path(path) for path in self.session_screenshots + self.session_video_clips if path]
        for path in paths:
            try:
                if not self.is_safe_session_record_path(path):
                    self.add_data_item(f"已阻止非记录目录删除：{path}", level="三级预警")
                    continue
                if path.exists() and path.is_file():
                    path.unlink()
                    removed_files += 1
            except Exception as exc:
                self.add_data_item(f"本次记录删除失败：{path}，原因：{exc}", level="三级预警")

        removed_rows = self.alarm_manager.remove_records_by_screenshots(set(self.session_screenshots))
        return removed_files, removed_rows

    def is_safe_session_record_path(self, path: Path) -> bool:
        try:
            resolved_path = path.resolve()
            resolved_records = RECORDS_DIR.resolve()
        except Exception:
            return False
        return resolved_path.is_relative_to(resolved_records)

    def return_home(self) -> None:    #停止当前识别任务，清空状态，返回初始主页面
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

    def stop_capture(self, finish_clips: bool = True) -> None:    #停止定时器，释放摄像头或视频文件资源，并保存未完成的报警片段
        self.timer.stop()
        if finish_clips:
            self.clip_recorder.finish_all()
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def set_status(self, text: str) -> None:    #设置顶部状态标签文字，并根据预警等级切换颜色
        self.status_label.setText(text)
        warning_levels = list(WARNING_STATUS_STYLES)
        if text == "识别中":
            self.status_label.setStyleSheet(
                "color: #d9ffe8; background: #214433; border: 1px solid #4ec982;"
            )
        elif len(warning_levels) > 2 and text == warning_levels[2]:
            self.status_label.setStyleSheet(
                "color: #ffdada; background: #4a2525; border: 1px solid #ff6b6b;"
            )
        elif len(warning_levels) > 1 and text == warning_levels[1]:
            self.status_label.setStyleSheet(
                "color: #ffe5c8; background: #46321f; border: 1px solid #f2a65a;"
            )
        elif warning_levels and text == warning_levels[0]:
            self.status_label.setStyleSheet(
                "color: #fff6bd; background: #3f3822; border: 1px solid #e7d06a;"
            )
        else:
            self.status_label.setStyleSheet(
                "color: #cce8ff; background: #173a52; border: 1px solid #2b8cc4;"
            )

    def closeEvent(self, event) -> None:    #点击右上角关闭按钮时，释放摄像头和视频资源。
        self.stop_capture()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setFont(QFont("SimSun", 10))
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
