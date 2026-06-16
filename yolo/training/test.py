"""Test YOLO detection on a live RealSense stream.

Verifies that your trained YOLO model detects objects correctly before
running the full MegaPose pipeline. Shows bounding boxes and confidence
scores on a live camera feed.

Usage:
    python yolo/training/test.py --yolo-model yolo/models/cube.pt --conf 0.3
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from yolo.detector import YoloDetector  # noqa: E402


def init_realsense(width: int, height: int, fps: int):
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)

    profile = pipeline.start(config)

    for _ in range(30):
        pipeline.wait_for_frames()

    return pipeline, profile


def main():
    parser = argparse.ArgumentParser(
        description="Test YOLO detection on live RealSense stream"
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLO .pt weights",
    )
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument(
        "--yolo-label",
        type=str,
        default=None,
        help="Filter to this class only (default: show all)",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    # Resolve YOLO model path
    yolo_model_path = Path(args.yolo_model)
    if not yolo_model_path.is_absolute() and not yolo_model_path.exists():
        yolo_model_path = REPO_ROOT / yolo_model_path
    if not yolo_model_path.exists() and args.yolo_model != "yolov8n.pt":
        print(f"[ERROR] Model not found: {yolo_model_path}")
        sys.exit(1)
    yolo_model_path = str(yolo_model_path)

    print(f"[INFO] Loading YOLO model: {yolo_model_path}")
    yolo = YoloDetector(
        model_path=yolo_model_path,
        conf=args.conf,
        target_label=args.yolo_label,
        device="cpu",
    )
    print("[INFO] YOLO model loaded.")

    print(f"[INFO] Starting RealSense at {args.width}x{args.height} @ {args.fps}fps")
    pipeline, profile = init_realsense(args.width, args.height, args.fps)
    print("[INFO] RealSense ready. Press 'q' to quit.")

    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    ]
    frame_count = 0
    fps_smooth = 0.0

    try:
        while True:
            t0 = time.time()

            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            rgb = np.asanyarray(color_frame.get_data())
            detections = yolo.detect(rgb)

            vis = rgb[:, :, ::-1].copy()  # RGB -> BGR

            if detections:
                for i, det in enumerate(detections):
                    bbox = det["bbox"].astype(int)
                    label = det["label"]
                    conf = det["conf"]
                    color = colors[i % len(colors)]

                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

                    text = f"{label} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        vis, (x1, y1 - th - 6), (x1 + tw, y1), color, -1
                    )
                    cv2.putText(
                        vis, text, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA,
                    )

                det_text = f"{len(detections)} detection(s)"
            else:
                det_text = "No detections"

            # FPS
            elapsed = time.time() - t0
            fps_instant = 1.0 / max(elapsed, 1e-6)
            fps_smooth = 0.9 * fps_smooth + 0.1 * fps_instant

            cv2.putText(
                vis, f"{det_text} | {fps_smooth:.1f} FPS | conf>{args.conf}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
            )

            filter_text = f"filter: {args.yolo_label}" if args.yolo_label else "filter: none (all classes)"
            cv2.putText(
                vis, filter_text,
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
            )

            cv2.imshow("YOLO Detection Check", vis)
            frame_count += 1

            if frame_count % 60 == 0 and detections:
                print(
                    f"[INFO] Frame {frame_count}: {len(detections)} detection(s) - "
                    f"{', '.join(f'{d[\"label\"]} ({d[\"conf\"]:.2f})' for d in detections)}"
                )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"\n[INFO] Done. Processed {frame_count} frames.")


if __name__ == "__main__":
    main()
