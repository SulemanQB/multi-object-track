"""Drawing helpers for annotated tracking frames."""

from __future__ import annotations

import cv2
import numpy as np

if __package__:
    from .tracker import Track
else:
    from tracker import Track


def color_for_id(track_id: int) -> tuple[int, int, int]:
    """Return a deterministic BGR color for a track id."""
    seed = (track_id * 2654435761) & 0xFFFFFFFF
    return (int(seed & 255), int((seed >> 8) & 255), int((seed >> 16) & 255))


def draw_tracks(
    frame: np.ndarray,
    tracks: list[Track],
    class_names: dict[int, str] | list[str],
    frame_number: int,
    fps: float,
) -> np.ndarray:
    """Draw boxes, labels, and a compact status overlay in place."""
    for track in tracks:
        x1, y1, x2, y2 = np.round(track.bbox).astype(int)
        color = color_for_id(track.track_id)
        if isinstance(class_names, dict):
            class_name = class_names.get(track.class_id, str(track.class_id))
        else:
            class_name = (
                class_names[track.class_id]
                if 0 <= track.class_id < len(class_names)
                else str(track.class_id)
            )
        label = f"ID: {track.track_id} | {class_name} | {track.score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        label_top = max(0, y1 - text_height - baseline - 5)
        cv2.rectangle(frame, (x1, label_top), (x1 + text_width + 6, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

    status = f"Frame: {frame_number}  FPS: {fps:.1f}  Active tracks: {len(tracks)}"
    cv2.rectangle(frame, (0, 0), (420, 28), (20, 20, 20), -1)
    cv2.putText(frame, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return frame
