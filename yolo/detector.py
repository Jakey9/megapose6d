"""Modular YOLO detector with a stable interface for MegaPose integration.

Supports any Ultralytics-compatible .pt weight file. Swap general COCO weights
for a custom-trained model by changing the model_path argument.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from ultralytics import YOLO


class YoloDetector:
    """YOLO-based 2D object detector with a swappable model backend.

    Args:
        model_path: Path to an Ultralytics .pt weight file.
            Default "yolov8n.pt" auto-downloads on first use.
        conf: Minimum confidence threshold for detections.
        target_label: If set, only return detections whose class name
            matches this string (case-insensitive). Useful for filtering
            general COCO models to a specific class like "cup".
        device: Inference device ("cuda", "cpu", or device index).
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf: float = 0.5,
        target_label: Optional[str] = None,
        device: str = "cuda",
    ):
        self.model = YOLO(model_path)
        self.conf = conf
        self.target_label = target_label.lower() if target_label else None
        self.device = device

    def detect(self, rgb: np.ndarray) -> list[dict] | None:
        """Run detection on a single RGB image.

        Args:
            rgb: [H, W, 3] uint8 BGR or RGB image (Ultralytics handles both).

        Returns:
            List of detections, each a dict with keys:
                bbox: np.ndarray [x1, y1, x2, y2] in pixel coords
                label: str class name
                conf: float confidence score
            Returns None if no detections pass the filter.
        """
        results = self.model.predict(
            rgb, conf=self.conf, device=self.device, verbose=False
        )
        if not results or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes
        names = results[0].names

        detections = []
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            label = names[cls_id]
            conf = float(boxes.conf[i].item())
            bbox = boxes.xyxy[i].cpu().numpy().astype(np.float32)

            if self.target_label and label.lower() != self.target_label:
                continue

            detections.append({"bbox": bbox, "label": label, "conf": conf})

        return detections if detections else None

    def detect_best(
        self, rgb: np.ndarray
    ) -> tuple[np.ndarray, str, float] | None:
        """Return the highest-confidence detection.

        Args:
            rgb: [H, W, 3] uint8 image.

        Returns:
            Tuple of (bbox_xyxy, label, confidence) or None.
        """
        detections = self.detect(rgb)
        if not detections:
            return None
        best = max(detections, key=lambda d: d["conf"])
        return best["bbox"], best["label"], best["conf"]
