# DINOv2 Few-Shot Object Localization

A training-free object detector that locates objects using visual similarity matching. Instead of requiring annotated training data or class labels, it uses 3-10 reference images of the target object and finds it in scene images by comparing DINOv2 patch features.

## How It Works

1. **Reference encoding** - Reference images are resized and passed through DINOv2 to extract dense patch tokens. These are averaged into a single compact descriptor vector.
2. **Scene matching** - Scene images are processed the same way, producing a grid of patch features. Cosine similarity is computed between each scene patch and the reference descriptor.
3. **Bounding box extraction** - The similarity heatmap is thresholded and connected components are found. The largest qualifying blob is converted to a pixel-space bounding box.

```
Reference Images ──> DINOv2 ──> Average Descriptor
                                       │
Scene Image ──────> DINOv2 ──> Patch Features ──> Cosine Similarity Map
                                                         │
                                          Threshold + Connected Components
                                                         │
                                                   Bounding Box
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

## Usage

### Prepare Reference Images

Place 3-10 images of your target object in a directory:

```
DINO/datasets/<object_name>/
├── img_01.jpg
├── img_02.jpg
├── img_03.png
└── ...
```

Guidelines for reference images:
- Capture the object from different angles
- Use a clean, uncluttered background if possible
- Ensure good lighting and focus
- The object should fill most of the frame (tight crops work best)
- More images = more robust matching, but diminishing returns past 10

### Standalone Detector Usage

```python
from DINO.detector import DinoDetector
import cv2

# Initialize with reference images
detector = DinoDetector(
    reference_dir="DINO/datasets/yellowCube",
    label="yellowCube",
    model_name="dinov2_vits14",       # "dinov2_vitb14" for higher accuracy
    similarity_threshold=0.5,          # Lower = more sensitive, higher = fewer false positives
    min_blob_area=4,                   # Minimum patch count for valid detection
    device="cuda",
)

# Detect in a scene image
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
    --similarity-threshold 0.5
```

#### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--object-label` | (required) | Object label matching mesh directory name |
| `--mesh-dir` | `local_data/live_objects` | Directory containing object meshes |
| `--reference-dir` | `DINO/datasets/yellowCube` | Directory with reference images |
| `--dino-model` | `dinov2_vits14` | DINOv2 variant (`vits14`, `vitb14`, `vitl14`) |
| `--similarity-threshold` | `0.5` | Cosine similarity cutoff for detection |
| `--min-blob-area` | `4` | Min connected patches for valid detection |
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

- **Object not detected**: Lower `--similarity-threshold` (e.g., 0.4). Add more reference images with varied viewpoints.
- **False positives**: Raise `--similarity-threshold` (e.g., 0.6). Increase `--min-blob-area`.
- **Jittery bounding box**: Enable `--continuous-detection` for smoother tracking, or rely on MegaPose refiner only.
- **Small objects**: Lower `--min-blob-area` to 2-3 so small blobs are not discarded.

## Directory Structure

```
DINO/
├── __init__.py          # Package exports
├── detector.py          # DinoDetector class
├── README.md            # This file
└── datasets/
    └── yellowCube/      # Example reference images
        ├── cube_3.jpeg
        ├── cube_6.jpeg
        ├── cube_8.jpeg
        ├── cube_9.jpeg
        └── cube_12.jpeg
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
