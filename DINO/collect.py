"""RealSense data collection for DINOv2 few-shot reference images.

Displays a live RealSense stream with a center crop guide overlay.
Press 's' to save the current frame (clean, without overlay text).
Press 'q' to quit.

Saved images are written WITHOUT any HUD text so they can be used
directly as DINOv2 reference images.

Best practices for reference images:
  - Hold the object close so it fills 70%+ of the frame
  - Vary the background between shots (different surfaces, held in hand)
  - Capture from multiple angles (front, side, top, angled)
  - Ensure good, even lighting — avoid harsh shadows
  - Aim for 5-10 images total

Usage:
    python DINO/collect.py --name yellowCube
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent


def init_realsense(width: int, height: int, fps: int):
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)

    profile = pipeline.start(config)

    for _ in range(30):
        pipeline.wait_for_frames()

    return pipeline, profile


def draw_center_guide(img: np.ndarray, ratio: float = 0.6) -> np.ndarray:
    """Draw a center crop rectangle to help frame the object."""
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    rw, rh = int(w * ratio) // 2, int(h * ratio) // 2

    overlay = img.copy()
    # Dim the outside region
    mask = np.zeros_like(img)
    cv2.rectangle(mask, (cx - rw, cy - rh), (cx + rw, cy + rh), (255, 255, 255), -1)
    overlay = np.where(mask > 0, img, (img * 0.4).astype(np.uint8))

    # Draw guide rectangle
    cv2.rectangle(overlay, (cx - rw, cy - rh), (cx + rw, cy + rh), (0, 255, 0), 2)
    return overlay


def compute_sharpness(img: np.ndarray) -> float:
    """Laplacian variance as a sharpness metric."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def main():
    parser = argparse.ArgumentParser(
        description="Collect reference images for DINOv2 few-shot detection"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yellowCube",
        help="Object name — images saved to DINO/datasets/<name>/",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--crop-ratio",
        type=float,
        default=0.6,
        help="Center crop ratio applied when saving (0.0-1.0). "
             "Crops to this fraction of the frame to remove background.",
    )
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=50.0,
        help="Minimum Laplacian variance to accept an image (blur rejection).",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Save full frame without center cropping.",
    )
    args = parser.parse_args()

    output_dir = REPO_ROOT / "datasets" / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = list(output_dir.glob("img_*.jpg"))
    save_count = len(existing)

    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Existing images: {save_count}")
    print(f"[INFO] Starting RealSense at {args.width}x{args.height} @ {args.fps}fps")
    print(f"[INFO] Center crop ratio: {args.crop_ratio} ({'disabled' if args.no_crop else 'active'})")
    print(f"[INFO] Min sharpness: {args.min_sharpness}")
    print()
    print("[TIPS] Hold object close — fill the green guide box")
    print("[TIPS] Vary background between shots")
    print("[TIPS] Capture 5-10 images from different angles")
    print()
    print("[INFO] Press 's' to save, 'q' to quit.")

    pipeline, profile = init_realsense(args.width, args.height, args.fps)

    try:
        while True:
            import pyrealsense2 as rs

            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            rgb = np.asanyarray(color_frame.get_data())
            bgr = rgb[:, :, ::-1].copy()

            # Display frame with guide overlay (text is display-only, never saved)
            vis = draw_center_guide(bgr, args.crop_ratio)
            sharpness = compute_sharpness(bgr)

            # HUD text only on display copy
            status_color = (0, 255, 0) if sharpness >= args.min_sharpness else (0, 0, 255)
            cv2.putText(
                vis,
                f"Saved: {save_count} | Sharpness: {sharpness:.0f} | 's'=save 'q'=quit",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                status_color,
                1,
                cv2.LINE_AA,
            )
            if sharpness < args.min_sharpness:
                cv2.putText(
                    vis,
                    "TOO BLURRY - hold steady",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow("DINO Reference Collection", vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                # Reject blurry frames
                if sharpness < args.min_sharpness:
                    print(f"[REJECT] Too blurry (sharpness={sharpness:.0f} < {args.min_sharpness})")
                    continue

                # Save the CLEAN frame (no text overlay)
                save_img = bgr.copy()

                # Apply center crop to focus on the object
                if not args.no_crop:
                    h, w = save_img.shape[:2]
                    new_h = int(h * args.crop_ratio)
                    new_w = int(w * args.crop_ratio)
                    top = (h - new_h) // 2
                    left = (w - new_w) // 2
                    save_img = save_img[top:top + new_h, left:left + new_w]

                filename = f"img_{save_count:03d}.jpg"
                filepath = output_dir / filename
                cv2.imwrite(str(filepath), save_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                save_count += 1
                print(f"[SAVED] {filepath} ({save_img.shape[1]}x{save_img.shape[0]}, sharpness={sharpness:.0f})")

            elif key == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"\n[INFO] Done. Total saved: {save_count} images in {output_dir}")
        if save_count < 3:
            print("[WARN] Recommend at least 3-5 reference images for robust detection.")


if __name__ == "__main__":
    main()
