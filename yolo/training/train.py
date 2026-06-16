"""YOLO training wrapper script.

Trains a YOLO model on a dataset in yolo/training/datasets/ and copies the
best weights to yolo/models/ for immediate use with the live tracker.

Usage:
    python yolo/training/train.py --dataset cube --epochs 100
    python yolo/training/train.py --dataset cube --epochs 50 --batch 8 --imgsz 640
"""

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = REPO_ROOT / "yolo" / "training" / "datasets"
RUNS_DIR = REPO_ROOT / "yolo" / "training" / "runs"
MODELS_DIR = REPO_ROOT / "yolo" / "models"


def main():
    parser = argparse.ArgumentParser(description="Train YOLO and export to yolo/models/")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (subdirectory in yolo/training/datasets/)",
    )
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--device", type=str, default="0", help="Device: '0' for GPU, 'cpu' for CPU"
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Skip copying best.pt to yolo/models/",
    )
    args = parser.parse_args()

    dataset_dir = DATASETS_DIR / args.dataset
    dataset_yaml = dataset_dir / "dataset.yaml"

    if not dataset_dir.exists():
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        print(f"[INFO] Available datasets: {[d.name for d in DATASETS_DIR.iterdir() if d.is_dir()]}")
        sys.exit(1)

    if not dataset_yaml.exists():
        print(f"[ERROR] dataset.yaml not found in {dataset_dir}")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    print(f"[INFO] Training YOLO on dataset '{args.dataset}'")
    print(f"[INFO]   Base model: {args.model}")
    print(f"[INFO]   Epochs: {args.epochs}")
    print(f"[INFO]   Image size: {args.imgsz}")
    print(f"[INFO]   Batch size: {args.batch}")
    print(f"[INFO]   Device: {args.device}")
    print(f"[INFO]   Dataset YAML: {dataset_yaml}")
    print()

    model = YOLO(args.model)
    results = model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(RUNS_DIR),
        name=args.dataset,
        exist_ok=True,
    )

    best_weights = RUNS_DIR / args.dataset / "weights" / "best.pt"
    if not best_weights.exists():
        print(f"[WARN] best.pt not found at {best_weights}")
        last_weights = RUNS_DIR / args.dataset / "weights" / "last.pt"
        if last_weights.exists():
            best_weights = last_weights
            print(f"[INFO] Using last.pt instead")
        else:
            print("[ERROR] No weights found after training.")
            sys.exit(1)

    if not args.no_copy:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = MODELS_DIR / f"{args.dataset}.pt"
        shutil.copy2(best_weights, output_path)
        print(f"\n[OK] Weights copied to: {output_path}")
        print(f"[INFO] Use with tracker:")
        print(f"  python -m megapose.scripts.run_live_tracker \\")
        print(f"      --object-label {args.dataset} \\")
        print(f"      --yolo-model yolo/models/{args.dataset}.pt \\")
        print(f"      --yolo-label {args.dataset}")
    else:
        print(f"\n[OK] Training complete. Weights at: {best_weights}")


if __name__ == "__main__":
    main()
