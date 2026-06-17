"""Few-shot object localization using DINOv2 visual similarity.

Locates objects in scene images by matching dense patch features against
a set of reference images. No class labels or training required — purely
visual similarity driven.

Usage:
    from DINO.detector import DinoDetector

    detector = DinoDetector(
        reference_dir="DINO/datasets/yellowCube",
        label="yellowCube",
    )
    result = detector.detect_best(scene_rgb)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from DINO.models import load_dinov2


class DinoDetector:
    """Few-shot object detector using DINOv2 patch feature matching.

    Extracts dense patch tokens from reference images at init time, then
    compares them against scene patch tokens to produce a similarity heatmap.
    A bounding box is fitted around the highest-similarity region.

    Args:
        reference_dir: Directory containing 3-10 reference images of the object.
        label: Object label string (returned with detections).
        model_name: DINOv2 model variant.
            Options: "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14".
        similarity_threshold: Minimum cosine similarity to consider a patch
            as belonging to the object.
        min_blob_area: Minimum number of patches in a connected component
            to be considered a valid detection.
        device: Torch device for inference.
    """

    PATCH_SIZE = 14
    SCENE_RESIZE = 518  # 37x37 patches at patch_size=14

    def __init__(
        self,
        reference_dir: Union[str, Path],
        label: str,
        model_name: str = "dinov2_vits14",
        similarity_threshold: float = 0.5,
        min_blob_area: int = 4,
        device: str = "cuda",
    ):
        self.label = label
        self.similarity_threshold = similarity_threshold
        self.min_blob_area = min_blob_area
        self.device = device

        self.model = load_dinov2(model_name, pretrained=True)
        self.model.eval().to(self.device)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.ref_descriptor = self._build_reference_descriptor(
            Path(reference_dir)
        )

    def _build_reference_descriptor(self, ref_dir: Path) -> torch.Tensor:
        """Extract and average patch features from all reference images.

        Returns:
            Tensor of shape [feat_dim] — the averaged CLS+patch descriptor
            used for per-patch similarity scoring.
        """
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = sorted(
            p for p in ref_dir.iterdir()
            if p.suffix.lower() in extensions
        )
        if len(image_paths) < 1:
            raise ValueError(
                f"No reference images found in {ref_dir}. "
                f"Need at least 1 image (3-10 recommended)."
            )

        all_patch_features = []

        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize to square input for consistent patch count
            img_resized = cv2.resize(img_rgb, (self.SCENE_RESIZE, self.SCENE_RESIZE))
            tensor = self.transform(img_resized).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model.forward_features(tensor)
                patch_tokens = features["x_norm_patchtokens"]  # [1, N, D]

            all_patch_features.append(patch_tokens.squeeze(0))  # [N, D]

        if not all_patch_features:
            raise ValueError(f"Could not load any images from {ref_dir}")

        # Average all patch features across all reference images into a single
        # descriptor vector. This captures the object's appearance compactly.
        stacked = torch.cat(all_patch_features, dim=0)  # [total_patches, D]
        descriptor = stacked.mean(dim=0)  # [D]
        descriptor = F.normalize(descriptor, dim=0)

        return descriptor

    def _extract_scene_features(
        self, rgb: np.ndarray
    ) -> tuple[torch.Tensor, int, int, float, float]:
        """Extract patch tokens from a scene image.

        Returns:
            patch_tokens: [N_patches, D] normalized features
            grid_h: number of patches vertically
            grid_w: number of patches horizontally
            scale_y: pixel-to-patch vertical scale factor
            scale_x: pixel-to-patch horizontal scale factor
        """
        h_orig, w_orig = rgb.shape[:2]

        # Resize so the shorter side is SCENE_RESIZE, maintaining aspect ratio
        # Actually for simplicity and consistency, resize to fixed square
        img_resized = cv2.resize(rgb, (self.SCENE_RESIZE, self.SCENE_RESIZE))
        tensor = self.transform(img_resized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model.forward_features(tensor)
            patch_tokens = features["x_norm_patchtokens"]  # [1, N, D]

        patch_tokens = patch_tokens.squeeze(0)  # [N, D]
        patch_tokens = F.normalize(patch_tokens, dim=1)

        grid_size = self.SCENE_RESIZE // self.PATCH_SIZE  # 37
        grid_h = grid_size
        grid_w = grid_size

        scale_y = h_orig / self.SCENE_RESIZE
        scale_x = w_orig / self.SCENE_RESIZE

        return patch_tokens, grid_h, grid_w, scale_y, scale_x

    def detect(self, rgb: np.ndarray) -> list[dict] | None:
        """Run detection on a single RGB image.

        Args:
            rgb: [H, W, 3] uint8 RGB or BGR image.

        Returns:
            List of detections, each a dict with keys:
                bbox: np.ndarray [x1, y1, x2, y2] in pixel coords
                label: str object label
                conf: float confidence (mean similarity)
            Returns None if no detections pass the threshold.
        """
        patch_tokens, grid_h, grid_w, scale_y, scale_x = (
            self._extract_scene_features(rgb)
        )

        # Cosine similarity between each scene patch and the reference descriptor
        similarity = (patch_tokens @ self.ref_descriptor).cpu().numpy()  # [N]
        sim_map = similarity.reshape(grid_h, grid_w)

        # Threshold to binary mask
        mask = (sim_map >= self.similarity_threshold).astype(np.uint8)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        detections = []
        for i in range(1, num_labels):  # skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_blob_area:
                continue

            # Bounding box in patch coordinates
            px = stats[i, cv2.CC_STAT_LEFT]
            py = stats[i, cv2.CC_STAT_TOP]
            pw = stats[i, cv2.CC_STAT_WIDTH]
            ph = stats[i, cv2.CC_STAT_HEIGHT]

            # Mean similarity within this component as confidence
            component_mask = (labels == i)
            conf = float(sim_map[component_mask].mean())

            # Convert patch coords to pixel coords
            x1 = px * self.PATCH_SIZE * scale_x
            y1 = py * self.PATCH_SIZE * scale_y
            x2 = (px + pw) * self.PATCH_SIZE * scale_x
            y2 = (py + ph) * self.PATCH_SIZE * scale_y

            bbox = np.array([x1, y1, x2, y2], dtype=np.float32)

            detections.append({
                "bbox": bbox,
                "label": self.label,
                "conf": conf,
            })

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
