# DINOv2 Few-Shot Object Localization

A training-free object detector that locates objects using visual similarity matching. Instead of requiring annotated training data or class labels, it uses 3-10 reference images of the target object and finds it in scene images by comparing DINOv2 patch features.

## How It Works

1. **Reference encoding** — Reference images are passed through DINOv2 to extract dense patch tokens. The most distinctive and cross-image-consistent patches are selected as the reference descriptor set (top-K patches).
2. **Scene matching** — The scene image produces a grid of patch features. Each scene patch is scored by its max cosine similarity to the reference patch set (nearest-neighbor matching).
3. **Adaptive thresholding** — The similarity map is thresholded adaptively (mean + N*std) to isolate patches significantly more similar than the background.
4. **Bounding box extraction** — Connected components are found in the thresholded mask. Valid blobs (not too small, not too large) are converted to pixel-space bounding boxes.

```
Reference Images ──> DINOv2 ──> Cross-image consistency filter ──> Top-K Patches [K, D]
                                                                          │
Scene Image ──────> DINOv2 ──> Patch Features [N, D] ──> Max-Similarity (NN match)
                                                                          │
                                                        Adaptive Threshold (mean + factor*std)
                                                                          │
                                                        Connected Components ──> BBox
```

## Installation

### Prerequisites

The following packages are required (all should already be available in the MegaPose conda environment):

- Python >= 3.9
- PyTorch >= 1.13
- torchvision
- OpenCV (`opencv-python`)
- NumPy

### Install DINOv2 Weights

No separate installation or `torch.hub` clone is needed. This module includes a
self-contained ViT implementation (`DINO/models.py`) that is Python 3.9+
compatible. Pretrained weights are downloaded automatically from Facebook AI
on first use:

```bash
# Weights download automatically on first run (~86MB for vits14)
# Cached at ~/.cache/torch/hub/checkpoints/
python -c "from DINO.models import load_dinov2; load_dinov2('dinov2_vits14')"
```

If you are behind a firewall, manually download the weights:

```bash
mkdir -p ~/.cache/torch/hub/checkpoints
wget -P ~/.cache/torch/hub/checkpoints/ \
    https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
```

### Verify Installation

```bash
cd /path/to/megapose6d
python -c "from DINO.detector import DinoDetector; print('OK')"
```

## Collecting Reference Images

Use the built-in collector script to capture reference images from RealSense:

```bash
python DINO/collect.py --name yellowCube
```

The collector provides:

- **Center crop guide** — green rectangle shows what will be saved; area outside is dimmed
- **Blur rejection** — refuses to save blurry frames (sharpness shown in HUD)
- **Clean saves** — no HUD text in saved images (only on display)
- **Auto center-crop** — crops to center 60% by default (`--crop-ratio`)

### Collector Arguments


| Argument          | Default      | Description                                     |
| ----------------- | ------------ | ----------------------------------------------- |
| `--name`          | `yellowCube` | Object name (saves to `DINO/datasets/<name>/`)  |
| `--crop-ratio`    | `0.6`        | Center crop fraction on save (1.0 = full frame) |
| `--min-sharpness` | `50.0`       | Minimum Laplacian variance to accept            |
| `--no-crop`       | disabled     | Save full frame without cropping                |
| `--width`         | `640`        | RealSense width                                 |
| `--height`        | `480`        | RealSense height                                |


### Guidelines for Good Reference Images

- The object MUST fill 70%+ of the saved image (tight crops)
- **Vary the background** between shots (different surfaces, hold in hand, etc.)
- Capture from 3-5 different angles (front, side, top, angled)
- No text overlays or HUD elements in saved images
- Good, even lighting — avoid harsh shadows or overexposure
- 5-10 images is optimal (diminishing returns past 10)

## Usage

### Standalone Detector

```python
from DINO.detector import DinoDetector
import cv2

detector = DinoDetector(
    reference_dir="DINO/datasets/yellowCube",
    label="yellowCube",
    model_name="dinov2_vits14",
    n_ref_patches=30,           # Number of reference patch descriptors
    adaptive_std_factor=2.0,    # Threshold = mean + 2*std (higher = stricter)
    min_blob_area=4,            # Min patches for valid detection
    max_blob_ratio=0.5,         # Reject detections > 50% of frame
    device="cuda",
)

scene = cv2.imread("scene.jpg")
scene_rgb = cv2.cvtColor(scene, cv2.COLOR_BGR2RGB)

# Get best detection
result = detector.detect_best(scene_rgb)
if result is not None:
    bbox, label, confidence = result
    print(f"Found {label} at {bbox} with confidence {confidence:.3f}")

# Or get all detections
detections = detector.detect(scene_rgb)
if detections:
    for det in detections:
        print(f"  {det['label']}: bbox={det['bbox']}, conf={det['conf']:.3f}")
```

### Live Tracker (RealSense + MegaPose)

Run the full 6D pose tracking pipeline:

```bash
cd /path/to/megapose6d

python -m megapose.scripts.run_live_tracker_dino \
    --object-label cube \
    --mesh-dir local_data/live_objects \
    --reference-dir DINO/datasets/yellowCube \
    --dino-model dinov2_vits14 \
    --adaptive-std-factor 2.0 \
    --verify-interval 10 \
    --drift-threshold 0.10
```

### Tracking Architecture

The tracker operates in two modes:

- **DETECT (slow path)** — Triggered on first frame or when tracking is lost. Runs DINO detection followed by full MegaPose coarse+refiner pipeline (~1-3s).
- **TRACK (fast path)** — Subsequent frames use MegaPose refiner only (~30-50ms) for real-time tracking.

Four layers of failure detection trigger re-detection:


| Check               | Frequency      | Cost  | What it catches                           |
| ------------------- | -------------- | ----- | ----------------------------------------- |
| Frame-to-frame jump | Every frame    | Free  | Sudden pose snaps                         |
| Cumulative drift    | Every frame    | ~0    | Slow creep from anchor pose               |
| DINO verification   | Every N frames | ~20ms | Pose detached from object; object removed |
| Scoring model       | Optional       | ~30ms | Render mismatch (disabled by default)     |


#### Command Line Arguments

**Detection parameters:**


| Argument                | Default                    | Description                                   |
| ----------------------- | -------------------------- | --------------------------------------------- |
| `--object-label`        | (required)                 | Object label matching mesh directory name     |
| `--mesh-dir`            | `local_data/live_objects`  | Directory containing object meshes            |
| `--reference-dir`       | `DINO/datasets/yellowCube` | Directory with reference images               |
| `--dino-model`          | `dinov2_vits14`            | DINOv2 variant (`vits14`, `vitb14`, `vitl14`) |
| `--n-ref-patches`       | `30`                       | Number of reference patch descriptors to keep |
| `--adaptive-std-factor` | `2.0`                      | Std devs above mean for adaptive threshold    |
| `--min-blob-area`       | `4`                        | Min connected patches for valid detection     |
| `--max-blob-ratio`      | `0.5`                      | Max frame fraction a detection can cover      |


**Tracking and verification:**


| Argument                      | Default  | Description                                               |
| ----------------------------- | -------- | --------------------------------------------------------- |
| `--verify-interval`           | `10`     | DINO verification every N frames (0=disabled)             |
| `--drift-threshold`           | `0.10`   | Max cumulative drift in meters before re-detect           |
| `--pose-delta-threshold`      | `0.15`   | Max per-frame pose jump before flagging                   |
| `--max-track-failures`        | `5`      | Failures needed to trigger full re-detection              |
| `--continuous-detection`      | disabled | Run DINO every frame (expensive, for fast-moving objects) |
| `--periodic-scoring-interval` | `0`      | Run MegaPose scorer every N frames (0=disabled)           |


**MegaPose and hardware:**


| Argument                      | Default                             | Description                          |
| ----------------------------- | ----------------------------------- | ------------------------------------ |
| `--megapose-model`            | `megapose-1.0-RGB-multi-hypothesis` | MegaPose model name                  |
| `--use-depth`                 | disabled                            | Use depth stream for pose estimation |
| `--detect-refiner-iterations` | `5`                                 | Refiner iterations during detection  |
| `--detect-hypotheses`         | `5`                                 | Pose hypotheses during detection     |
| `--track-refiner-iterations`  | `1`                                 | Refiner iterations during tracking   |
| `--realsense-width`           | `640`                               | Camera resolution width              |
| `--realsense-height`          | `480`                               | Camera resolution height             |
| `--no-display`                | disabled                            | Headless mode (no OpenCV window)     |


## Model Variants


| Model           | Parameters | Feature Dim | Speed          | Accuracy |
| --------------- | ---------- | ----------- | -------------- | -------- |
| `dinov2_vits14` | 21M        | 384         | Fast (~20ms)   | Good     |
| `dinov2_vitb14` | 86M        | 768         | Medium (~40ms) | Better   |
| `dinov2_vitl14` | 300M       | 1024        | Slow (~80ms)   | Best     |


For real-time tracking, `dinov2_vits14` is recommended. Use `dinov2_vitb14` if detection robustness is more important than speed.

## Tuning Tips

- **Detects the entire frame**: Your reference images have too much background. Recapture with the object filling 70%+ of the frame using `python DINO/collect.py`. Raise `--adaptive-std-factor` to 2.5-3.0.
- **Object not detected**: Lower `--adaptive-std-factor` (e.g., 1.5). Add more reference images with varied viewpoints. Lower `--min-blob-area`.
- **False positives on similar objects**: Increase `--n-ref-patches` to 50 for more discriminative matching. Raise `--adaptive-std-factor`.
- **Pose drifts without re-detecting**: Lower `--drift-threshold` (e.g., 0.05). Lower `--verify-interval` (e.g., 5) for more frequent checks.
- **Too many re-detections (stuttering)**: Raise `--verify-interval` to 20-30. Raise `--drift-threshold`. Raise `--max-track-failures`.
- **Jittery bounding box**: Enable `--continuous-detection` for smoother bbox updates, or rely on MegaPose refiner only.
- **Small objects**: Lower `--min-blob-area` to 2-3 so small blobs are not discarded.

## Directory Structure

```
DINO/
├── __init__.py          # Package exports
├── detector.py          # DinoDetector class (multi-descriptor NN matching)
├── models.py            # Self-contained DINOv2 ViT (Python 3.9 compatible)
├── collect.py           # Reference image collection script (RealSense)
├── README.md            # This file
└── datasets/
    └── yellowCube/      # Example reference images
        ├── img_000.jpg
        ├── img_001.jpg
        └── ...
```

## Integration with MegaPose

The `DinoDetector` implements the same interface as `YoloDetector`:

```python
# Both return the same format:
# detect(rgb) -> list[{"bbox": np.ndarray, "label": str, "conf": float}] | None
# detect_best(rgb) -> tuple[np.ndarray, str, float] | None

# Drop-in replacement in any tracking script:
# from yolo.detector import YoloDetector   # before
from DINO.detector import DinoDetector      # after
```

This makes it easy to swap between YOLO (trained, class-based) and DINO (few-shot, similarity-based) detection depending on your use case.