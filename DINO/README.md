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
- Center crop guide overlay (object should fill the green box)
- Blur rejection (refuses to save blurry frames)
- Clean saves (no HUD text in saved images)
- Auto center-crop on save (configurable via `--crop-ratio`)

**Critical guidelines for reference images:**
- The object MUST fill 70%+ of the frame (tight crops)
- Vary the background between shots (different surfaces, hold in hand)
- Capture from 3-5 different angles
- No text overlays or HUD elements in saved images
- Good, even lighting — avoid harsh shadows
- 5-10 images is optimal

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
    --object-label yellowCube \
    --mesh-dir local_data/live_objects \
    --reference-dir DINO/datasets/yellowCube \
    --dino-model dinov2_vits14 \
    --adaptive-std-factor 2.0
```

#### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--object-label` | (required) | Object label matching mesh directory name |
| `--mesh-dir` | `local_data/live_objects` | Directory containing object meshes |
| `--reference-dir` | `DINO/datasets/yellowCube` | Directory with reference images |
| `--dino-model` | `dinov2_vits14` | DINOv2 variant (`vits14`, `vitb14`, `vitl14`) |
| `--n-ref-patches` | `30` | Number of reference patch descriptors to keep |
| `--adaptive-std-factor` | `2.0` | Std devs above mean for adaptive threshold |
| `--min-blob-area` | `4` | Min connected patches for valid detection |
| `--max-blob-ratio` | `0.5` | Max frame fraction a detection can cover |
| `--megapose-model` | `megapose-1.0-RGB-multi-hypothesis` | MegaPose model name |
| `--use-depth` | disabled | Use depth stream for pose estimation |
| `--continuous-detection` | disabled | Run DINO every frame (slower, better for moving objects) |
| `--max-track-failures` | `10` | Failures before re-running detection |
| `--pose-delta-threshold` | `0.15` | Max pose jump before flagging as lost |
| `--no-display` | disabled | Headless mode (no OpenCV window) |

## Model Variants

| Model | Parameters | Feature Dim | Speed | Accuracy |
|-------|-----------|-------------|-------|----------|
| `dinov2_vits14` | 21M | 384 | Fast (~20ms) | Good |
| `dinov2_vitb14` | 86M | 768 | Medium (~40ms) | Better |
| `dinov2_vitl14` | 300M | 1024 | Slow (~80ms) | Best |

For real-time tracking, `dinov2_vits14` is recommended. Use `dinov2_vitb14` if detection robustness is more important than speed.

## Tuning Tips

- **Detects the entire frame**: Your reference images have too much background. Recapture with the object filling 70%+ of the frame. Raise `--adaptive-std-factor` to 2.5-3.0.
- **Object not detected**: Lower `--adaptive-std-factor` (e.g., 1.5). Add more reference images with varied viewpoints. Lower `--min-blob-area`.
- **False positives on similar objects**: Increase `--n-ref-patches` to 50 for more discriminative matching. Raise `--adaptive-std-factor`.
- **Jittery bounding box**: Enable `--continuous-detection` for smoother tracking, or rely on MegaPose refiner only.
- **Small objects**: Lower `--min-blob-area` to 2-3 so small blobs are not discarded.

## Directory Structure

```
DINO/
├── __init__.py          # Package exports
├── detector.py          # DinoDetector class (multi-descriptor NN matching)
├── models.py            # Self-contained DINOv2 ViT (Python 3.9 compatible)
├── collect.py           # Reference image collection script
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
