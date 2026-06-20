from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from math import ceil

import cv2


@dataclass
class ActiveClip:
    path: Path
    frames_left: int
    frames: list = field(default_factory=list)


class AlarmClipRecorder:
    def __init__(
        self,
        output_dir: str = "records/videos",
        pre_seconds: float = 3.0,
        post_seconds: float = 5.0,
        max_fps: float = 20.0,
        max_active_clips: int = 3,
    ):
        self.output_dir = Path(output_dir)
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.max_fps = max_fps
        self.max_active_clips = max_active_clips
        self.fps = 25.0
        self.frame_sample_step = 1
        self.frame_index = 0
        self.frame_size: tuple[int, int] | None = None
        self.pre_buffer = deque()
        self.active_clips: list[ActiveClip] = []
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.errors: list[str] = []
        self.error_lock = Lock()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def reset(self, fps: float | None = None, frame_size: tuple[int, int] | None = None) -> None:
        self.finish_all()
        source_fps = fps if fps and fps > 1 else self.max_fps
        self.frame_sample_step = max(1, ceil(source_fps / self.max_fps))
        self.fps = max(1.0, source_fps / self.frame_sample_step)
        self.frame_index = 0
        self.frame_size = frame_size
        self.pre_buffer.clear()
        self.active_clips.clear()

    def add_frame(self, frame) -> None:
        if frame is None:
            return

        self.frame_index += 1
        if self.frame_index % self.frame_sample_step != 0:
            return

        frame_copy = frame.copy()
        if self.frame_size is None:
            height, width = frame_copy.shape[:2]
            self.frame_size = (width, height)

        self.pre_buffer.append(frame_copy)
        max_pre_frames = max(1, int(self.fps * self.pre_seconds))
        while len(self.pre_buffer) > max_pre_frames:
            self.pre_buffer.popleft()

        remaining_clips: list[ActiveClip] = []
        for clip in self.active_clips:
            clip.frames.append(frame_copy.copy())
            clip.frames_left -= 1
            if clip.frames_left <= 0:
                self._write_clip(clip)
            else:
                remaining_clips.append(clip)
        self.active_clips = remaining_clips

    def start_clip(self, alarm_type: str, track_id: int) -> str:
        if len(self.active_clips) >= self.max_active_clips:
            self._add_error(f"报警片段数量达到上限，已跳过保存：ID {track_id} {alarm_type}")
            return ""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alarm_dir = self.output_dir / alarm_type
        alarm_dir.mkdir(parents=True, exist_ok=True)
        path = alarm_dir / f"{timestamp}_id{track_id}_{alarm_type}.mp4"

        clip = ActiveClip(
            path=path,
            frames_left=max(1, int(self.fps * self.post_seconds)),
            frames=[frame.copy() for frame in self.pre_buffer],
        )
        self.active_clips.append(clip)
        return str(path)

    def finish_all(self) -> None:
        for clip in self.active_clips:
            self._write_clip(clip)
        self.active_clips.clear()

    def _write_clip(self, clip: ActiveClip) -> None:
        frames = [frame.copy() for frame in clip.frames]
        path = clip.path
        fps = self.fps
        self.executor.submit(self._write_clip_sync, path, frames, fps)

    def _write_clip_sync(self, path: Path, frames: list, fps: float) -> None:
        try:
            self._write_clip_file(path, frames, fps)
        except Exception as exc:
            self._add_error(f"报警视频片段保存失败：{path}，原因：{exc}")

    def _write_clip_file(self, path: Path, frames: list, fps: float) -> None:
        if not frames:
            return

        first = frames[0]
        height, width = first.shape[:2]
        size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, size)
        if not writer.isOpened():
            self._add_error(f"报警视频片段保存失败：无法创建文件 {path}")
            return

        for frame in frames:
            if (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size)
            writer.write(frame)
        writer.release()

    def collect_errors(self) -> list[str]:
        with self.error_lock:
            errors = self.errors[:]
            self.errors.clear()
        return errors

    def _add_error(self, message: str) -> None:
        with self.error_lock:
            self.errors.append(message)
