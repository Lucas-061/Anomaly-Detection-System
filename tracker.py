from dataclasses import dataclass, field
from math import hypot
import time

import cv2
import numpy as np

from detector import Detection


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


@dataclass
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    center: tuple[int, int]
    last_center: tuple[int, int]
    last_seen: float
    created_at: float
    history: list[tuple[int, int]] = field(default_factory=list)
    missing_frames: int = 0
    inside_fence: bool = False
    previous_inside_fence: bool = False
    enter_time: float | None = None
    speed: float = 0.0
    fall_frames: int = 0
    climb_frames: int = 0
    run_frames: int = 0
    feature: np.ndarray | None = None


class CentroidTracker:
    def __init__(self, max_distance: float = 120.0, max_missing: int = 24):
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.next_id = 1
        self.tracks: dict[int, Track] = {}

    def update(self, detections: list[Detection], frame=None) -> list[Track]:
        now = time.time()
        unmatched_track_ids = set(self.tracks.keys())
        unmatched_detections = set(range(len(detections)))
        features = [self._extract_feature(frame, detection.bbox) for detection in detections]

        pairs = []
        for track_id, track in self.tracks.items():
            for det_index, detection in enumerate(detections):
                score = self._match_score(track, detection, features[det_index])
                if score is not None:
                    pairs.append((score, track_id, det_index))

        for _, track_id, det_index in sorted(pairs, key=lambda item: item[0]):
            if track_id not in unmatched_track_ids or det_index not in unmatched_detections:
                continue
            self._update_track(self.tracks[track_id], detections[det_index], features[det_index], now)
            unmatched_track_ids.remove(track_id)
            unmatched_detections.remove(det_index)

        for track_id in list(unmatched_track_ids):
            track = self.tracks[track_id]
            track.missing_frames += 1
            if track.missing_frames > self.max_missing:
                del self.tracks[track_id]

        for det_index in unmatched_detections:
            self._create_track(detections[det_index], features[det_index], now)

        return list(self.tracks.values())

    def _create_track(self, detection: Detection, feature: np.ndarray | None, now: float) -> None:
        center = bbox_center(detection.bbox)
        track = Track(
            track_id=self.next_id,
            bbox=detection.bbox,
            confidence=detection.confidence,
            center=center,
            last_center=center,
            history=[center],
            last_seen=now,
            created_at=now,
            feature=feature,
        )
        self.tracks[self.next_id] = track
        self.next_id += 1

    def _update_track(self, track: Track, detection: Detection, feature: np.ndarray | None, now: float) -> None:
        dt = max(now - track.last_seen, 1e-6)
        new_center = bbox_center(detection.bbox)

        track.last_center = track.center
        track.center = new_center
        track.bbox = detection.bbox
        track.confidence = detection.confidence
        track.speed = hypot(new_center[0] - track.last_center[0], new_center[1] - track.last_center[1]) / dt
        track.last_seen = now
        track.missing_frames = 0
        if feature is not None:
            if track.feature is None:
                track.feature = feature
            else:
                track.feature = cv2.normalize(0.75 * track.feature + 0.25 * feature, None).flatten()
        track.history.append(new_center)
        if len(track.history) > 60:
            track.history = track.history[-60:]

    def _match_score(self, track: Track, detection: Detection, feature: np.ndarray | None) -> float | None:
        center = bbox_center(detection.bbox)
        distance = hypot(track.center[0] - center[0], track.center[1] - center[1])
        if distance > self.max_distance * (1 + min(track.missing_frames, 8) * 0.12):
            return None

        distance_score = distance / self.max_distance
        size_score = self._size_difference(track.bbox, detection.bbox)
        feature_distance = self._feature_distance(track.feature, feature)

        if feature_distance is not None and feature_distance > 0.65 and distance > self.max_distance * 0.45:
            return None

        if feature_distance is None:
            return 0.75 * distance_score + 0.25 * size_score
        return 0.45 * distance_score + 0.20 * size_score + 0.35 * feature_distance

    def _size_difference(self, old_bbox: tuple[int, int, int, int], new_bbox: tuple[int, int, int, int]) -> float:
        old_x1, old_y1, old_x2, old_y2 = old_bbox
        new_x1, new_y1, new_x2, new_y2 = new_bbox
        old_area = max((old_x2 - old_x1) * (old_y2 - old_y1), 1)
        new_area = max((new_x2 - new_x1) * (new_y2 - new_y1), 1)
        return min(abs(old_area - new_area) / max(old_area, new_area), 1.0)

    def _feature_distance(self, old_feature: np.ndarray | None, new_feature: np.ndarray | None) -> float | None:
        if old_feature is None or new_feature is None:
            return None
        similarity = cv2.compareHist(old_feature.astype("float32"), new_feature.astype("float32"), cv2.HISTCMP_CORREL)
        similarity = max(min(float(similarity), 1.0), -1.0)
        return (1.0 - similarity) / 2.0

    def _extract_feature(self, frame, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        if frame is None:
            return None

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            return None

        person = frame[y1:y2, x1:x2]
        if person.size == 0:
            return None

        hsv = cv2.cvtColor(person, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
        return cv2.normalize(hist, hist).flatten()
