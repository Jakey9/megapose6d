# YOLO Custom Model Training

Train a custom YOLO model for your specific objects, then drop the weights
into `yolo/models/` for use with the live tracker.

## Dataset Structure

Place your dataset under `yolo/training/datasets/<dataset_name>/`:

```
yolo/training/datasets/my_object/
├── dataset.yaml          # Dataset config (see below)
├── images/
│   ├── train/            # Training images
│   │   ├── img_001.jpg
│   │   └── ...
│   └── val/              # Validation images
│       ├── img_100.jpg
│       └── ...
└── labels/
    ├── train/            # YOLO-format label files (one per image)
    │   ├── img_001.txt
    │   └── ...
    └── val/
        ├── img_100.txt
        └── ...
```

### Label Format

Each `.txt` file has one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```
All values are normalized to [0, 1] relative to image dimensions.

### dataset.yaml Example

```yaml
path: yolo/training/datasets/my_object
train: images/train
val: images/val

names:
  0: my-object-label
```

The class name here should match the mesh directory name under
`local_data/live_objects/` so the tracker can map detections to meshes.

## Training

```bash
# From megapose6d/ root directory
yolo train \
    data=yolo/training/datasets/my_object/dataset.yaml \
    model=yolov8n.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    project=yolo/training/runs \
    name=my_object
```

Trained weights will be saved to:
`yolo/training/runs/my_object/weights/best.pt`

## Export to Models Directory

```bash
cp yolo/training/runs/my_object/weights/best.pt yolo/models/my_object.pt
```

## Use with Live Tracker

```bash
python -m megapose.scripts.run_live_tracker \
    --object-label my-object-label \
    --yolo-model yolo/models/my_object.pt
```

## Tips

- Start with at least 100-200 annotated images for decent results.
- Use data augmentation (Ultralytics applies it by default).
- For single-object detection, a small model (yolov8n) trains fast and runs fast.
- Validate that your class name in `dataset.yaml` matches `--object-label`.
