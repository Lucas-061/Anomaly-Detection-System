import csv
from datetime import datetime
from pathlib import Path
import cv2


ALARM_NAMES = {
    "intrusion": "违规闯入",
    "cross_fence": "翻越围栏",
    "fall_down": "摔倒",
}

SAVE_ALARM_TYPES = set(ALARM_NAMES)


class AlarmManager:    #报警保存管理器，负责保存报警截图和写入 CSV 报警记录
    def __init__(self, output_dir: str = "records"):
        self.output_dir = Path(output_dir)
        self.screenshot_dir = self.output_dir / "screenshots"
        self.log_file = self.output_dir / "alarm_log.csv"
        self.errors: list[str] = []

        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_log_header()

    def trigger(    #当指定报警发生时，保存当前帧截图，并向 alarm_log.csv 写入报警记录
        self,
        frame,
        track_id: int,
        alarm_type: str,
        level: str,
        source: str | None = None,
        video_clip: str | None = None,
    ) -> str | None:
        if alarm_type not in SAVE_ALARM_TYPES:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alarm_dir = self.screenshot_dir / alarm_type
        alarm_dir.mkdir(parents=True, exist_ok=True)

        screenshot = alarm_dir / f"{timestamp}_id{track_id}_{alarm_type}.jpg"
        try:
            ok = cv2.imwrite(str(screenshot), frame)    #报警截图保存
        except Exception as exc:
            self.errors.append(f"报警截图保存失败：{screenshot}，原因：{exc}")
            return None

        if not ok:
            self.errors.append(f"报警截图保存失败：{screenshot}")
            return None

        try:
            with self.log_file.open("a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    source or "",
                    track_id,
                    alarm_type,
                    ALARM_NAMES[alarm_type],
                    level,
                    str(screenshot),
                    video_clip or "",
                ])
        except Exception as exc:
            self.errors.append(f"报警记录写入失败：{self.log_file}，原因：{exc}")

        return str(screenshot)

    def collect_errors(self) -> list[str]:    #返回并清空报警截图或 CSV 写入过程中产生的错误信息
        errors = self.errors[:]
        self.errors.clear()
        return errors

    def remove_records_by_screenshots(self, screenshot_paths: set[str]) -> int:
        if not screenshot_paths or not self.log_file.exists():
            return 0

        try:
            with self.log_file.open("r", newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                rows = list(reader)
                fieldnames = reader.fieldnames or []
        except Exception as exc:
            self.errors.append(f"报警记录读取失败：{self.log_file}，原因：{exc}")
            return 0

        if not fieldnames:
            return 0

        kept_rows = [row for row in rows if row.get("screenshot", "") not in screenshot_paths]
        removed_count = len(rows) - len(kept_rows)
        if removed_count == 0:
            return 0

        try:
            with self.log_file.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(kept_rows)
        except Exception as exc:
            self.errors.append(f"报警记录删除失败：{self.log_file}，原因：{exc}")
            return 0

        return removed_count

    def _ensure_log_header(self) -> None:    #确保报警日志 CSV 存在且表头为最新格式
        self.output_dir.mkdir(parents=True, exist_ok=True)
        header = ["time", "source", "track_id", "alarm_type", "alarm_name", "level", "screenshot", "video_clip"]
        if self.log_file.exists():
            with self.log_file.open("r", newline="", encoding="utf-8") as file:
                first_line = file.readline().strip()
            if first_line == ",".join(header):
                return

        try:
            with self.log_file.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(header)
        except Exception as exc:
            self.errors.append(f"报警日志初始化失败：{self.log_file}，原因：{exc}")
