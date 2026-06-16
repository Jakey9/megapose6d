"""RealSense camera sanity check.

Verifies that pyrealsense2 is installed and the RealSense camera is working.
Displays live RGB and depth streams with camera intrinsics overlay.

Usage:
    python -m megapose.scripts.check_realsense
"""

import sys

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    print("[ERROR] pyrealsense2 not installed. Run: pip install pyrealsense2")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("[ERROR] opencv-python not installed. Run: pip install opencv-python")
    sys.exit(1)


def main():
    width, height, fps = 640, 480, 30

    # print(f"[INFO] pyrealsense2 version: {rs.__version__}")
    print(f"[INFO] Configuring streams: {width}x{height} @ {fps} FPS")

    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("[ERROR] No RealSense device found. Check USB connection.")
        sys.exit(1)

    dev = devices[0]
    print(f"[INFO] Device found: {dev.get_info(rs.camera_info.name)}")
    print(f"[INFO] Serial number: {dev.get_info(rs.camera_info.serial_number)}")
    print(f"[INFO] Firmware: {dev.get_info(rs.camera_info.firmware_version)}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    print("[INFO] Starting pipeline ...")
    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"[ERROR] Failed to start pipeline: {e}")
        sys.exit(1)

    align = rs.align(rs.stream.color)

    color_profile = profile.get_stream(rs.stream.color)
    intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
    K = np.array(
        [
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    print(f"[INFO] Color intrinsics:")
    print(f"       Resolution: {intrinsics.width}x{intrinsics.height}")
    print(f"       fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}")
    print(f"       ppx={intrinsics.ppx:.2f}, ppy={intrinsics.ppy:.2f}")
    print(f"       Distortion model: {intrinsics.model}")
    print(f"[INFO] K matrix:")
    print(f"       {K[0]}")
    print(f"       {K[1]}")
    print(f"       {K[2]}")
    print()
    print("[INFO] Streams running. Press 'q' to quit.")
    print("[INFO] Showing: RGB (left) | Depth colormap (right)")

    # Allow auto-exposure to settle
    for _ in range(30):
        pipeline.wait_for_frames()

    frame_count = 0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            rgb = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_m = depth_raw.astype(np.float32) / 1000.0

            # RGB display (convert to BGR for OpenCV)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Depth colormap
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_raw, alpha=0.03), cv2.COLORMAP_JET
            )

            # Center pixel depth
            cy, cx = height // 2, width // 2
            center_depth = depth_m[cy, cx]

            # Overlay info on RGB
            cv2.putText(
                bgr,
                f"RGB {width}x{height} @ {fps}fps",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                bgr,
                f"fx={intrinsics.fx:.1f} fy={intrinsics.fy:.1f}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            # Crosshair at center
            cv2.drawMarker(
                bgr, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 1
            )

            # Overlay info on depth
            cv2.putText(
                depth_colormap,
                f"Depth (center: {center_depth:.3f}m)",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                depth_colormap,
                f"Range: {depth_m[depth_m > 0].min():.2f}m - {depth_m.max():.2f}m"
                if depth_m.max() > 0
                else "No depth data",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            cv2.drawMarker(
                depth_colormap, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 1
            )

            # Side by side
            combined = np.hstack([bgr, depth_colormap])
            cv2.imshow("RealSense Sanity Check", combined)

            frame_count += 1
            if frame_count % 150 == 0:
                print(
                    f"[INFO] Frame {frame_count}: "
                    f"center_depth={center_depth:.3f}m, "
                    f"rgb_mean={rgb.mean():.1f}"
                )

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"\n[INFO] Done. Processed {frame_count} frames.")
        print("[OK] RealSense camera is working correctly.")


if __name__ == "__main__":
    main()
