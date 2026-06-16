"""Standalone RealSense + YOLO + MegaPose fast tracker.

Runs a tight detection-and-tracking loop using:
  - Intel RealSense for RGB (+ optional depth) capture
  - YOLO for 2D bounding box detection (initial + recovery)
  - MegaPose for 6D pose estimation and fast frame-to-frame tracking

Architecture:
  - First frame / tracking lost: YOLO detect -> full MegaPose pipeline (~1-3s)
  - Subsequent frames: MegaPose refiner only (~30-50ms) for 20-30 FPS tracking

Usage:
    python -m megapose.scripts.run_live_tracker \
        --object-label puzzle-half-trapezoid \
        --mesh-dir local_data/live_objects \
        --yolo-model yolo/models/yolov8n.pt
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from megapose.config import LOCAL_DATA_DIR
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.inference.types import ObservationTensor
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.logging import get_logger, set_logging_level
from megapose.utils.tensor_collection import PandasTensorCollection

logger = get_logger(__name__)

# Add repo root to path so we can import from yolo/
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from yolo.detector import YoloDetector  # noqa: E402


def build_object_dataset(mesh_dir: Path) -> RigidObjectDataset:
    """Scan mesh directory and build a RigidObjectDataset."""
    rigid_objects = []
    mesh_units = "mm"
    for object_dir in sorted(mesh_dir.iterdir()):
        if not object_dir.is_dir():
            continue
        label = object_dir.name
        mesh_path = None
        for fn in object_dir.glob("*"):
            if fn.suffix in {".obj", ".ply"}:
                mesh_path = fn
                break
        if mesh_path is None:
            logger.warning(f"No mesh found in {object_dir}, skipping.")
            continue
        rigid_objects.append(
            RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units)
        )
        logger.info(f"Loaded mesh for '{label}': {mesh_path}")

    if not rigid_objects:
        raise RuntimeError(f"No meshes found in {mesh_dir}")

    return RigidObjectDataset(rigid_objects)


def build_detections(label: str, bbox: np.ndarray) -> PandasTensorCollection:
    """Build a DetectionsType from a single bounding box."""
    infos = pd.DataFrame(
        dict(label=[label], batch_im_id=[0], instance_id=[0])
    )
    bboxes = torch.as_tensor(bbox[np.newaxis, :]).float()
    return PandasTensorCollection(infos=infos, bboxes=bboxes)


def build_pose_input(
    label: str, pose_4x4: np.ndarray
) -> PandasTensorCollection:
    """Build a PoseEstimatesType from a prior pose for refiner input."""
    infos = pd.DataFrame(
        dict(label=[label], batch_im_id=[0], instance_id=[0])
    )
    poses = torch.as_tensor(pose_4x4[np.newaxis, :, :]).float()
    return PandasTensorCollection(infos=infos, poses=poses)


def init_realsense(width: int, height: int, fps: int):
    """Initialize RealSense pipeline and return (pipeline, align, profile)."""
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    # Allow auto-exposure to settle
    for _ in range(30):
        pipeline.wait_for_frames()

    return pipeline, align, profile


def grab_frame(pipeline, align, use_depth: bool):
    """Grab one aligned frame from RealSense.

    Returns:
        rgb: [H, W, 3] uint8
        depth: [H, W] float32 in meters, or None
        K: [3, 3] float32 intrinsics
    """
    import pyrealsense2 as rs

    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)

    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame() if use_depth else None

    rgb = np.asanyarray(color_frame.get_data())

    depth = None
    if depth_frame is not None:
        depth = (
            np.asanyarray(depth_frame.get_data()).astype(np.float32) / 1000.0
        )

    intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
    K = np.array(
        [
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    return rgb, depth, K


def draw_pose_axes(img, K, pose_4x4, axis_length=0.05):
    """Draw 3D coordinate axes projected onto the image."""
    R = pose_4x4[:3, :3]
    t = pose_4x4[:3, 3]

    origin = t
    axes = np.float32(
        [[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]]
    )
    axes_world = (R @ axes.T).T + origin

    def project(pt3d):
        p = K @ pt3d
        return int(p[0] / p[2]), int(p[1] / p[2])

    center = project(origin)
    x_end = project(axes_world[0])
    y_end = project(axes_world[1])
    z_end = project(axes_world[2])

    cv2.line(img, center, x_end, (255, 0, 0), 2)  # X = red
    cv2.line(img, center, y_end, (0, 255, 0), 2)  # Y = green
    cv2.line(img, center, z_end, (0, 0, 255), 2)  # Z = blue

    return img


def draw_bbox(img, bbox, label, color=(0, 255, 0)):
    """Draw bounding box with label on image."""
    x1, y1, x2, y2 = bbox.astype(int)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        img,
        label,
        (x1, max(y1 - 8, 0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )
    return img


def compute_pose_delta(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    """Compute a scalar distance between two 4x4 poses (translation + rotation)."""
    t_delta = np.linalg.norm(pose_a[:3, 3] - pose_b[:3, 3])
    R_delta = pose_a[:3, :3] @ pose_b[:3, :3].T
    cos_angle = (np.trace(R_delta) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_delta = np.abs(np.arccos(cos_angle))
    return t_delta + 0.1 * angle_delta


def main():
    set_logging_level("info")

    parser = argparse.ArgumentParser(
        description="Standalone RealSense + YOLO + MegaPose fast tracker"
    )
    parser.add_argument(
        "--object-label",
        type=str,
        required=True,
        help="Object label (must match mesh directory name)",
    )
    parser.add_argument(
        "--mesh-dir",
        type=str,
        default="local_data/live_objects",
        help="Path to mesh directory (relative to megapose6d root or absolute)",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLO .pt weights",
    )
    parser.add_argument("--yolo-conf", type=float, default=0.5)
    parser.add_argument(
        "--yolo-label",
        type=str,
        default=None,
        help="YOLO class to filter for (default: None = accept any detection)",
    )
    parser.add_argument(
        "--megapose-model",
        type=str,
        default="megapose-1.0-RGB-multi-hypothesis",
    )
    parser.add_argument("--use-depth", action="store_true")
    parser.add_argument("--track-refiner-iterations", type=int, default=1)
    parser.add_argument("--detect-refiner-iterations", type=int, default=5)
    parser.add_argument("--detect-hypotheses", type=int, default=5)
    parser.add_argument("--max-track-failures", type=int, default=10)
    parser.add_argument(
        "--pose-delta-threshold",
        type=float,
        default=0.15,
        help="Max pose delta per frame before flagging as lost",
    )
    parser.add_argument(
        "--periodic-scoring-interval",
        type=int,
        default=0,
        help="Run scoring model every N frames (0=disabled, uses pose-delta only)",
    )
    parser.add_argument("--realsense-width", type=int, default=640)
    parser.add_argument("--realsense-height", type=int, default=480)
    parser.add_argument("--realsense-fps", type=int, default=30)
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV window (headless mode)",
    )
    args = parser.parse_args()

    # Resolve mesh directory
    mesh_dir = Path(args.mesh_dir)
    if not mesh_dir.is_absolute():
        mesh_dir = REPO_ROOT / mesh_dir
    if not mesh_dir.exists():
        logger.error(f"Mesh directory not found: {mesh_dir}")
        sys.exit(1)

    object_label = args.object_label
    model_name = args.megapose_model

    # --- Load models ---
    logger.info(f"Building object dataset from {mesh_dir} ...")
    object_dataset = build_object_dataset(mesh_dir)

    available_labels = [obj.label for obj in object_dataset.list_objects]
    if object_label not in available_labels:
        logger.error(
            f"Object label '{object_label}' not found in dataset. "
            f"Available: {available_labels}"
        )
        sys.exit(1)

    logger.info(f"Loading MegaPose model '{model_name}' ...")
    model_info = NAMED_MODELS[model_name]
    pose_estimator = load_named_model(model_name, object_dataset).cuda()
    logger.info("MegaPose model loaded.")

    logger.info(f"Loading YOLO model from '{args.yolo_model}' (CPU) ...")
    yolo = YoloDetector(
        model_path=args.yolo_model,
        conf=args.yolo_conf,
        target_label=args.yolo_label,
        device="cpu",
    )
    logger.info("YOLO model loaded (running on CPU to save GPU memory).")

    # --- Initialize RealSense ---
    logger.info("Initializing RealSense camera ...")
    pipeline, align, profile = init_realsense(
        args.realsense_width, args.realsense_height, args.realsense_fps
    )
    logger.info("RealSense initialized.")

    # --- Tracking state ---
    last_pose: np.ndarray | None = None
    last_bbox: np.ndarray | None = None
    consecutive_failures = 0
    frame_count = 0
    fps_smooth = 0.0

    logger.info("Starting tracking loop. Press 'q' to quit.")

    try:
        while True:
            t_frame_start = time.time()

            rgb, depth, K = grab_frame(pipeline, align, args.use_depth)
            if not args.use_depth:
                depth = None

            need_detect = (
                last_pose is None
                or consecutive_failures >= args.max_track_failures
            )

            if need_detect:
                # --- SLOW PATH: YOLO + full pipeline ---
                result = yolo.detect_best(rgb)
                if result is None:
                    vis = rgb[:, :, ::-1].copy()  # RGB -> BGR for OpenCV
                    cv2.putText(
                        vis,
                        "No detection",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )
                    if not args.no_display:
                        cv2.imshow("MegaPose Tracker", vis)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    continue

                bbox, label, conf = result
                last_bbox = bbox
                logger.info(
                    f"YOLO detection: label={label}, conf={conf:.2f}, "
                    f"bbox={bbox.astype(int).tolist()}"
                )

                observation = ObservationTensor.from_numpy(rgb, depth, K).cuda()
                detections = build_detections(object_label, bbox).cuda()

                with torch.no_grad():
                    output, _ = pose_estimator.run_inference_pipeline(
                        observation,
                        detections=detections,
                        n_refiner_iterations=args.detect_refiner_iterations,
                        n_pose_hypotheses=args.detect_hypotheses,
                    )

                if len(output.poses) == 0:
                    logger.warning("Full pipeline returned no poses.")
                    continue

                last_pose = output.poses[0].cpu().numpy()
                consecutive_failures = 0
                mode_str = "DETECT"

            else:
                # --- FAST PATH: refiner only ---
                observation = ObservationTensor.from_numpy(rgb, depth, K).cuda()
                data_TCO = build_pose_input(object_label, last_pose).cuda()

                with torch.no_grad():
                    preds, _ = pose_estimator.forward_refiner(
                        observation,
                        data_TCO,
                        n_iterations=args.track_refiner_iterations,
                    )

                iter_key = f"iteration={args.track_refiner_iterations}"
                refined = preds[iter_key]

                if len(refined.poses) == 0:
                    consecutive_failures += 1
                    mode_str = "TRACK (fail)"
                else:
                    new_pose = refined.poses[0].cpu().numpy()

                    # Validate via pose delta
                    delta = compute_pose_delta(last_pose, new_pose)
                    if delta > args.pose_delta_threshold:
                        consecutive_failures += 1
                        mode_str = f"TRACK (delta={delta:.3f})"
                    else:
                        # Optional periodic scoring
                        if (
                            args.periodic_scoring_interval > 0
                            and frame_count % args.periodic_scoring_interval == 0
                        ):
                            with torch.no_grad():
                                scored, _ = pose_estimator.forward_scoring_model(
                                    observation, refined
                                )
                            score = scored.infos["pose_score"].iloc[0]
                            if score < 0.3:
                                consecutive_failures += 1
                                mode_str = f"TRACK (score={score:.2f})"
                            else:
                                last_pose = new_pose
                                consecutive_failures = 0
                                mode_str = f"TRACK (score={score:.2f})"
                        else:
                            last_pose = new_pose
                            consecutive_failures = 0
                            mode_str = "TRACK"

            frame_count += 1

            # --- Visualization ---
            t_frame_end = time.time()
            frame_time = t_frame_end - t_frame_start
            fps_instant = 1.0 / max(frame_time, 1e-6)
            fps_smooth = 0.9 * fps_smooth + 0.1 * fps_instant

            if not args.no_display:
                vis = rgb[:, :, ::-1].copy()  # RGB -> BGR

                if last_bbox is not None:
                    draw_bbox(vis, last_bbox, object_label)

                if last_pose is not None:
                    draw_pose_axes(vis, K, last_pose)

                info = (
                    f"{mode_str} | {fps_smooth:.1f} FPS | "
                    f"failures={consecutive_failures}/{args.max_track_failures}"
                )
                cv2.putText(
                    vis,
                    info,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                # Show translation
                if last_pose is not None:
                    t = last_pose[:3, 3]
                    pos_str = f"T=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]m"
                    cv2.putText(
                        vis,
                        pos_str,
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

                cv2.imshow("MegaPose Tracker", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                if frame_count % 30 == 0:
                    logger.info(
                        f"Frame {frame_count}: {mode_str} | "
                        f"{fps_smooth:.1f} FPS"
                    )

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
