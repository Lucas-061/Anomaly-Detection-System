import time

from tracker import Track


class BehaviorAnalyzer:
    def __init__(
        self,
        stay_seconds: float = 8.0,
        run_speed: float = 260.0,
        fall_ratio: float = 1.15,
        climb_pixels: int = 18,
    ):
        self.stay_seconds = stay_seconds
        self.run_speed = run_speed
        self.fall_ratio = fall_ratio
        self.climb_pixels = climb_pixels

    def analyze(self, track: Track, inside_fence: bool) -> list[tuple[str, str]]:
        now = time.time()
        alarms: list[tuple[str, str]] = []

        track.previous_inside_fence = track.inside_fence
        track.inside_fence = inside_fence

        if inside_fence and not track.previous_inside_fence:
            track.enter_time = now
            alarms.append(("intrusion", "二级预警"))
            if len(track.history) >= 2:
                alarms.append(("cross_fence", "三级预警"))
        elif not inside_fence:
            track.enter_time = None

        if inside_fence and track.enter_time is not None:
            stay_time = now - track.enter_time
            if stay_time >= self.stay_seconds:
                alarms.append(("long_stay", "一级预警"))

        if self._is_falling(track):
            alarms.append(("fall_down", "三级预警"))

        if self._is_running(track):
            alarms.append(("running", "一级预警"))

        if self._is_climbing(track):
            alarms.append(("climbing", "二级预警"))

        return alarms

    def _is_falling(self, track: Track) -> bool:
        x1, y1, x2, y2 = track.bbox
        width = max(x2 - x1, 1)
        height = max(y2 - y1, 1)
        ratio = width / height

        if ratio >= self.fall_ratio:
            track.fall_frames += 1
        else:
            track.fall_frames = 0

        return track.fall_frames >= 4

    def _is_running(self, track: Track) -> bool:
        if track.speed >= self.run_speed:
            track.run_frames += 1
        else:
            track.run_frames = 0

        return track.run_frames >= 3

    def _is_climbing(self, track: Track) -> bool:
        if len(track.history) < 8:
            return False

        recent = track.history[-8:]
        old_y = recent[0][1]
        new_y = recent[-1][1]
        moved_up = old_y - new_y

        if moved_up >= self.climb_pixels:
            track.climb_frames += 1
        else:
            track.climb_frames = 0

        return track.climb_frames >= 4
