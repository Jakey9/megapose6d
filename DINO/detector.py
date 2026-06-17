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

    Uses multi-descriptor nearest-neighbor matching with adaptive thresholding
    to robustly localize objects regardless of scene content.

    Args:
        reference_dir: Directory containing 3-10 reference images of the object.
            Images should be tight crops with the object filling the frame.
        label: Object label string (returned with detections).
        model_name: DINOv2 model variant.
            Options: "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14".
        n_ref_patches: Number of top reference patches to keep as descriptors.
            More patches = better recall but slower matching.
        adaptive_std_factor: Number of standard deviations above mean for
            adaptive threshold. Higher = stricter detection.
        min_blob_area: Minimum number of patches in a connected component
            to be considered a valid detection.
        max_blob_ratio: Maximum ratio of detected blob area to total grid area.
            Rejects detections that cover too much of the frame.
        device: Torch device for inference.
    """

    PATCH_SIZE = 14
    SCENE_RESIZE = 518  # 37x37 patches at patch_size=14

    def __init__(
        self,
        reference_dir: Union[str, Path],
        label: str,
        model_name: str = "dinov2_vits14",
        n_ref_patches: int = 30,
        adaptive_std_factor: float = 2.0,
        min_blob_area: int = 4,
        max_blob_ratio: float = 0.5,
        device: str = "cuda",
    ):
        self.label = label
        self.n_ref_patches = n_ref_patches
        self.adaptive_std_factor = adaptive_std_factor
        self.min_blob_area = min_blob_area
        self.max_blob_ratio = max_blob_ratio
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

        self.ref_patches = self._build_reference_patches(Path(reference_dir))

    def _build_reference_patches(self, ref_dir: Path) -> torch.Tensor:
        """Extract the most distinctive patch features from reference images.

        Instead of averaging into a single descriptor (which loses discriminative
        power), we keep the top-K most mutually-consistent patches across all
        reference images. Each represents a distinctive local appearance of the
        object (a yellow face, an edge, a corner, etc.).

        Returns:
            Tensor of shape [K, feat_dim] — top-K reference patch descriptors.
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

        all_patches = []

        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_resized = cv2.resize(img_rgb, (self.SCENE_RESIZE, self.SCENE_RESIZE))
            tensor = self.transform(img_resized).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model.forward_features(tensor)
                patch_tokens = features["x_norm_patchtokens"]  # [1, N, D]

            patches = patch_tokens.squeeze(0)  # [N, D]
            patches = F.normalize(patches, dim=1)
            all_patches.append(patches)

        if not all_patches:
            raise ValueError(f"Could not load any images from {ref_dir}")

        # Select the most discriminative patches via cross-image consistency:
        # Patches representing the object will be similar across images,
        # while background patches will differ.
        if len(all_patches) >= 2:
            ref_patches = self._select_consistent_patches(all_patches)
        else:
            # Single image: use center patches (assume object is centered)
            patches = all_patches[0]
            grid_size = self.SCENE_RESIZE // self.PATCH_SIZE
            ref_patches = self._select_center_patches(patches, grid_size)

        return ref_patches

    def _select_consistent_patches(
        self, all_patches: list
    ) -> torch.Tensor:
        """Select patches that are consistent across multiple reference images.

        For each patch in each image, compute its max-similarity to patches in
        other images. Patches with high cross-image similarity are likely object
        patches (the object is consistent, backgrounds vary).
        """
        n_images = len(all_patches)
        scored_patches = []

        for i, patches_i in enumerate(all_patches):
            # For each patch in image i, find its best match in other images
            cross_sims = []
            for j, patches_j in enumerate(all_patches):
                if i == j:
                    continue
                # [N_i, N_j] similarity matrix
                sim_matrix = patches_i @ patches_j.T
                best_match_sim = sim_matrix.max(dim=1).values  # [N_i]
                cross_sims.append(best_match_sim)

            # Average best-match similarity across other images
            avg_cross_sim = torch.stack(cross_sims, dim=0).mean(dim=0)  # [N_i]

            for idx in range(patches_i.shape[0]):
                scored_patches.append((avg_cross_sim[idx].item(), patches_i[idx]))

        # Sort by cross-image consistency and keep top K
        scored_patches.sort(key=lambda x: x[0], reverse=True)
        top_k = min(self.n_ref_patches, len(scored_patches))
        selected = torch.stack([p[1] for p in scored_patches[:top_k]], dim=0)

        return selected  # [K, D]

    def _select_center_patches(
        self, patches: torch.Tensor, grid_size: int
    ) -> torch.Tensor:
        """Select patches from the center region (single-image fallback)."""
        center_start = grid_size // 4
        center_end = grid_size - center_start
        indices = []
        for r in range(center_start, center_end):
            for c in range(center_start, center_end):
                indices.append(r * grid_size + c)

        center_patches = patches[indices]
        # Take top K by norm (most "activated" patches)
        norms = center_patches.norm(dim=1)
        top_k = min(self.n_ref_patches, len(indices))
        _, top_idx = norms.topk(top_k)
        return center_patches[top_idx]

    def _extract_scene_features(
        self, rgb: np.ndarray
    ) -> tuple:
        """Extract patch tokens from a scene image.

        Returns:
            patch_tokens: [N_patches, D] normalized features
            grid_h: number of patches vertically
            grid_w: number of patches horizontally
            scale_y: pixel-to-patch vertical scale factor
            scale_x: pixel-to-patch horizontal scale factor
        """
        h_orig, w_orig = rgb.shape[:2]

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

    def detect(self, rgb: np.ndarray) -> list:
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

        # Per-patch max similarity against reference patch set (nearest-neighbor)
        # Each scene patch is scored by how well it matches its BEST reference patch
        sim_matrix = patch_tokens @ self.ref_patches.T  # [N_scene, K_ref]
        similarity = sim_matrix.max(dim=1).values.cpu().numpy()  # [N_scene]
        sim_map = similarity.reshape(grid_h, grid_w)

        # Adaptive threshold: mean + factor * std
        # This finds patches significantly more similar than the background baseline
        sim_mean = sim_map.mean()
        sim_std = sim_map.std()
        threshold = sim_mean + self.adaptive_std_factor * sim_std

        mask = (sim_map >= threshold).astype(np.uint8)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        total_area = grid_h * grid_w
        detections = []

        for i in range(1, num_labels):  # skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_blob_area:
                continue

            # Reject blobs that cover too much of the frame
            if area / total_area > self.max_blob_ratio:
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
    ) -> tuple:
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
