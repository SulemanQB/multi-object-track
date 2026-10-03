"""Small command-line and video I/O utilities."""

from __future__ import annotations

from pathlib import Path

import cv2


def open_video(path: str) -> cv2.VideoCapture:
    """Open a local video and raise a useful error when it cannot be read."""
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video was not found: {input_path}")
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open input video: {input_path}")
    return capture


def make_writer(output_path: str, capture: cv2.VideoCapture) -> cv2.VideoWriter:
    """Create an MP4 writer using the input video's dimensions and frame rate."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    fps = source_fps if source_fps and source_fps > 0 else 30.0
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not create output video: {destination}")
    return writer
