"""Ultralytics YOLO detection adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

if __package__:
    from .tracker import Detection
else:
    from tracker import Detection


class YoloDetector:
    """Run a pretrained Ultralytics YOLO model and normalize its output."""

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.35) -> None:
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.names = self.model.names

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return detections for one BGR OpenCV frame."""
        results = self.model.predict(frame, conf=self.confidence, verbose=False)
        if not results or results[0].boxes is None:
            return []
        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        return [
            Detection(bbox=box, score=float(score), class_id=int(class_id))
            for box, score, class_id in zip(xyxy, scores, class_ids)
        ]

    def class_name(self, class_id: int) -> str:
        """Resolve a model class id to a display label."""
        if isinstance(self.names, dict):
            return str(self.names.get(class_id, class_id))
        return str(self.names[class_id]) if 0 <= class_id < len(self.names) else str(class_id)


def ensure_model_path(model_path: str) -> str:
    """Give a clear error for a missing explicit local model path."""
    if "/" in model_path or "\\" in model_path:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"YOLO model was not found: {path}")
    return model_path
