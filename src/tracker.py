"""Lightweight Kalman-filter and Hungarian-assignment multi-object tracker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Detection:
    """A detector result represented as ``[x1, y1, x2, y2]`` pixels."""

    bbox: np.ndarray
    score: float
    class_id: int

    def __post_init__(self) -> None:
        self.bbox = np.asarray(self.bbox, dtype=float)
        if self.bbox.shape != (4,) or not np.isfinite(self.bbox).all():
            raise ValueError("bbox must contain four finite coordinates")
        if self.bbox[2] <= self.bbox[0] or self.bbox[3] <= self.bbox[1]:
            raise ValueError("bbox must have positive width and height")
        if not 0.0 <= self.score <= 1.0 or not np.isfinite(self.score):
            raise ValueError("score must be between 0 and 1")
        if self.class_id < 0:
            raise ValueError("class_id must be non-negative")


@dataclass
class Track:
    """Public state for one active track."""

    track_id: int
    bbox: np.ndarray
    score: float
    class_id: int
    hits: int = 1
    age: int = 1
    time_since_update: int = 0

class KalmanBox:
    """Constant-velocity Kalman filter over box center and size."""

    def __init__(self, bbox: np.ndarray) -> None:
        x1, y1, x2, y2 = bbox.astype(float)
        width = max(x2 - x1, 1.0)
        height = max(y2 - y1, 1.0)
        self.state = np.array(
            [(x1 + x2) / 2, (y1 + y2) / 2, width, height, 0, 0, 0, 0],
            dtype=float,
        )
        self.covariance = np.eye(8, dtype=float)
        self.covariance[4:, 4:] *= 100.0
        self.covariance[:4, :4] *= 10.0
        self.transition = np.eye(8, dtype=float)
        self.transition[0, 4] = 1.0
        self.transition[1, 5] = 1.0
        self.transition[2, 6] = 1.0
        self.transition[3, 7] = 1.0
        self.measurement = np.zeros((4, 8), dtype=float)
        self.measurement[:, :4] = np.eye(4)
        self.process_noise = np.eye(8, dtype=float) * 0.05
        self.process_noise[4:, 4:] *= 2.0
        self.measurement_noise = np.eye(4, dtype=float) * 1.0

    def predict(self) -> np.ndarray:
        self.state = self.transition @ self.state
        self.covariance = (
            self.transition @ self.covariance @ self.transition.T + self.process_noise
        )
        return self.to_bbox()

    def update(self, bbox: np.ndarray) -> None:
        measurement = self._to_measurement(bbox)
        innovation = measurement - self.measurement @ self.state
        innovation_covariance = (
            self.measurement @ self.covariance @ self.measurement.T
            + self.measurement_noise
        )
        gain = self.covariance @ self.measurement.T @ np.linalg.inv(innovation_covariance)
        self.state = self.state + gain @ innovation
        identity = np.eye(8)
        self.covariance = (identity - gain @ self.measurement) @ self.covariance

    def to_bbox(self) -> np.ndarray:
        center_x, center_y, width, height = self.state[:4]
        width = max(float(width), 1.0)
        height = max(float(height), 1.0)
        return np.array(
            [center_x - width / 2, center_y - height / 2,
             center_x + width / 2, center_y + height / 2],
            dtype=float,
        )

    @staticmethod
    def _to_measurement(bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = bbox.astype(float)
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2,
                         max(x2 - x1, 1.0), max(y2 - y1, 1.0)])


class MultiObjectTracker:
    """Track boxes with Kalman prediction and one-to-one IoU assignment."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 15,
        min_hits: int = 3,
    ) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")
        if max_age < 0 or min_hits < 1:
            raise ValueError("max_age must be non-negative and min_hits must be positive")
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self._tracks: list[_TrackState] = []
        self._next_id = 1

    def update(self, detections: Iterable[Detection]) -> list[Track]:
        """Advance the tracker one frame and return confirmed active tracks."""
        detection_list = list(detections)
        predicted = [track.filter.predict() for track in self._tracks]
        matches, unmatched_tracks, unmatched_detections = self._associate(
            predicted, detection_list
        )

        for track_index, detection_index in matches:
            track = self._tracks[track_index]
            detection = detection_list[detection_index]
            track.filter.update(detection.bbox)
            track.bbox = detection.bbox.astype(float).copy()
            track.score = detection.score
            track.class_id = detection.class_id
            track.hits += 1
            track.time_since_update = 0
            track.age += 1

        for track_index in unmatched_tracks:
            track = self._tracks[track_index]
            track.bbox = predicted[track_index]
            track.time_since_update += 1
            track.age += 1

        for detection_index in unmatched_detections:
            detection = detection_list[detection_index]
            self._tracks.append(
                _TrackState(
                    track_id=self._next_id,
                    filter=KalmanBox(detection.bbox),
                    bbox=detection.bbox.astype(float).copy(),
                    score=detection.score,
                    class_id=detection.class_id,
                )
            )
            self._next_id += 1

        self._tracks = [
            track for track in self._tracks if track.time_since_update <= self.max_age
        ]
        return [
            track.as_public()
            for track in self._tracks
            if track.hits >= self.min_hits and track.time_since_update == 0
        ]

    def _associate(
        self,
        predicted: list[np.ndarray],
        detections: list[Detection],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not predicted or not detections:
            return [], list(range(len(predicted))), list(range(len(detections)))

        iou_matrix = np.array(
            [[
                iou(track_box, detection.bbox)
                if self._tracks[track_index].class_id == detection.class_id
                else 0.0
                for detection in detections
            ]
             for track_index, track_box in enumerate(predicted)],
            dtype=float,
        )
        row_indices, column_indices = linear_sum_assignment(1.0 - iou_matrix)
        matches: list[tuple[int, int]] = []
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for track_index, detection_index in zip(row_indices, column_indices):
            if iou_matrix[track_index, detection_index] >= self.iou_threshold:
                matches.append((int(track_index), int(detection_index)))
                matched_tracks.add(int(track_index))
                matched_detections.add(int(detection_index))
        return (
            matches,
            [index for index in range(len(predicted)) if index not in matched_tracks],
            [index for index in range(len(detections)) if index not in matched_detections],
        )


@dataclass
class _TrackState:
    track_id: int
    filter: KalmanBox
    bbox: np.ndarray
    score: float
    class_id: int
    hits: int = 1
    age: int = 1
    time_since_update: int = 0

    def as_public(self) -> Track:
        return Track(
            track_id=self.track_id,
            bbox=self.bbox.copy(),
            score=self.score,
            class_id=self.class_id,
            hits=self.hits,
            age=self.age,
            time_since_update=self.time_since_update,
        )


def iou(first: np.ndarray, second: np.ndarray) -> float:
    """Return intersection-over-union for two ``[x1, y1, x2, y2]`` boxes."""
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    second_area = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0
