"""RealSense data collection for YOLO training.

Displays a live RealSense stream. Press 's' to save the current frame,
press 'q' to quit. Saved images can be annotated and used for YOLO training.

Usage:
    python yolo/training/collect.py --name cube
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


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
        description="Collect RealSense images for YOLO training"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="capture",
        help="Subdirectory name for saved images (default: capture)",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    output_dir = REPO_ROOT / "yolo" / "training" / "datasets" / "raw_data" / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = list(output_dir.glob("img_*.jpg"))
    save_count = len(existing)

    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Existing images: {save_count}")
    print(f"[INFO] Starting RealSense at {args.width}x{args.height} @ {args.fps}fps")

    pipeline, profile = init_realsense(args.width, args.height, args.fps)

    print("[INFO] Ready. Press 's' to save, 'q' to quit.")

    try:
        while True:
            import pyrealsense2 as rs

            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            rgb = np.asanyarray(color_frame.get_data())
            vis = rgb[:, :, ::-1].copy()  # RGB -> BGR

            cv2.putText(
                vis,
                f"Saved: {save_count} | Press 's' to save, 'q' to quit",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Data Collection", vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                filename = f"img_{save_count:03d}.jpg"
                filepath = output_dir / filename
                cv2.imwrite(str(filepath), vis)
                save_count += 1
                print(f"[SAVED] {filepath}")

            elif key == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"\n[INFO] Done. Total saved: {save_count} images in {output_dir}")


if __name__ == "__main__":
    main()
