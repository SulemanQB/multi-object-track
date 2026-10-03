"""Basic behavioral tests for the tracker."""

import numpy as np
import pytest

from src.main import validate_arguments
from src.tracker import Detection, MultiObjectTracker, iou
from src.visualization import draw_tracks


def detection(x1: float, y1: float, x2: float, y2: float, class_id: int = 0) -> Detection:
    return Detection(np.array([x1, y1, x2, y2], dtype=float), 0.9, class_id)


def test_iou_is_correct_for_overlapping_boxes() -> None:
    assert np.isclose(iou(np.array([0, 0, 10, 10]), np.array([5, 5, 15, 15])), 25 / 175)


def test_matching_preserves_id_when_object_moves() -> None:
    tracker = MultiObjectTracker(iou_threshold=0.1, max_age=2, min_hits=1)
    first = tracker.update([detection(0, 0, 10, 10)])
    second = tracker.update([detection(1, 0, 11, 10)])
    assert first[0].track_id == second[0].track_id


def test_short_missed_detection_is_deleted_after_max_age() -> None:
    tracker = MultiObjectTracker(iou_threshold=0.1, max_age=1, min_hits=1)
    track = tracker.update([detection(0, 0, 10, 10)])[0]
    assert tracker.update([]) == []
    assert tracker.update([]) == []
    assert track.track_id == 1


def test_hungarian_matching_does_not_duplicate_detection() -> None:
    tracker = MultiObjectTracker(iou_threshold=0.1, max_age=2, min_hits=1)
    tracker.update([detection(0, 0, 10, 10), detection(20, 0, 30, 10)])
    tracks = tracker.update([detection(1, 0, 11, 10)])
    assert len(tracks) == 1
    assert len({track.track_id for track in tracks}) == 1


def test_min_hits_confirms_after_configured_detections() -> None:
    tracker = MultiObjectTracker(iou_threshold=0.1, max_age=2, min_hits=3)
    assert tracker.update([detection(0, 0, 10, 10)]) == []
    assert tracker.update([detection(1, 0, 11, 10)]) == []
    assert len(tracker.update([detection(2, 0, 12, 10)])) == 1


def test_class_mismatch_does_not_reuse_track_id() -> None:
    tracker = MultiObjectTracker(iou_threshold=0.1, max_age=2, min_hits=1)
    person = tracker.update([detection(0, 0, 10, 10, class_id=0)])[0]
    car = tracker.update([detection(1, 0, 11, 10, class_id=2)])[0]
    assert car.track_id != person.track_id


def test_detection_rejects_invalid_boxes() -> None:
    with pytest.raises(ValueError, match="positive width"):
        detection(10, 0, 0, 10)


def test_visualization_falls_back_for_unknown_class_id() -> None:
    tracker = MultiObjectTracker(iou_threshold=0.1, max_age=1, min_hits=1)
    track = tracker.update([detection(0, 0, 10, 10, class_id=9)])[0]
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    rendered = draw_tracks(frame, [track], ["person"], 1, 10.0)
    assert rendered.shape == frame.shape


def test_cli_validation_rejects_invalid_tracker_settings() -> None:
    with pytest.raises(ValueError, match="iou-threshold"):
        validate_arguments(
            type("Args", (), {
                "confidence": 0.35,
                "iou_threshold": 1.1,
                "max_age": 15,
                "min_hits": 3,
            })()
        )
