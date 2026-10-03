"""Run YOLO detection and Kalman/Hungarian multi-object tracking."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

if __package__:
    from .detector import YoloDetector, ensure_model_path
    from .tracker import MultiObjectTracker
    from .utils import make_writer, open_video
    from .visualization import draw_tracks
else:
    from detector import YoloDetector, ensure_model_path
    from tracker import MultiObjectTracker
    from utils import make_writer, open_video
    from visualization import draw_tracks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to a local input video")
    parser.add_argument("--output", default="outputs/tracked.mp4", help="Output MP4 path")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model name or path")
    parser.add_argument("--confidence", type=float, default=0.35, help="Detector confidence threshold")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="Minimum IoU for matching")
    parser.add_argument("--max-age", type=int, default=15, help="Missed frames before deletion")
    parser.add_argument("--min-hits", type=int, default=3, help="Matches needed to confirm a track")
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate numeric runtime settings before loading model weights."""
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("--confidence must be between 0 and 1")
    if not 0.0 <= args.iou_threshold <= 1.0:
        raise ValueError("--iou-threshold must be between 0 and 1")
    if args.max_age < 0:
        raise ValueError("--max-age must be non-negative")
    if args.min_hits < 1:
        raise ValueError("--min-hits must be at least 1")


def run(args: argparse.Namespace) -> tuple[int, float, int]:
    validate_arguments(args)
    detector = YoloDetector(ensure_model_path(args.model), args.confidence)
    tracker = MultiObjectTracker(args.iou_threshold, args.max_age, args.min_hits)
    capture = open_video(args.input)
    writer = None
    frame_count = 0
    total_processing_seconds = 0.0
    maximum_active_tracks = 0
    try:
        writer = make_writer(args.output, capture)
        while True:
            success, frame = capture.read()
            if not success:
                break
            started = time.perf_counter()
            detections = detector.detect(frame)
            tracks = tracker.update(detections)
            elapsed = time.perf_counter() - started
            total_processing_seconds += elapsed
            frame_count += 1
            measured_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            maximum_active_tracks = max(maximum_active_tracks, len(tracks))
            writer.write(draw_tracks(frame, tracks, detector.names, frame_count, measured_fps))
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    average_fps = frame_count / total_processing_seconds if total_processing_seconds else 0.0
    print(f"Processed {frame_count} frames at {average_fps:.2f} FPS")
    print(f"Maximum active tracks: {maximum_active_tracks}")
    print(f"Saved annotated video: {Path(args.output)}")
    return frame_count, average_fps, maximum_active_tracks


def main() -> None:
    try:
        run(build_parser().parse_args())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
