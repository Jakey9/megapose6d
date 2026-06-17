"""Standalone RealSense + DINOv2 + MegaPose fast tracker.

Runs a tight detection-and-tracking loop using:
  - Intel RealSense for RGB (+ optional depth) capture
  - DINOv2 few-shot detector for 2D bounding box via visual similarity
  - MegaPose for 6D pose estimation and fast frame-to-frame tracking

Architecture:
  - First frame / tracking lost: DINO detect -> full MegaPose pipeline (~1-3s)
  - Subsequent frames: MegaPose refiner only (~30-50ms) for 20-30 FPS tracking

Usage:
    python -m megapose.scripts.run_live_tracker_dino \
        --object-label yellowCube \
        --mesh-dir local_data/live_objects \
        --reference-dir DINO/datasets/yellowCube
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import pyrealsense2 as rs

from megapose.config import LOCAL_DATA_DIR
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.inference.types import ObservationTensor
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.logging import get_logger, set_logging_level
from megapose.utils.tensor_collection import PandasTensorCollection

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from DINO.detector import DinoDetector  # noqa: E402


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
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

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

    cv2.line(img, center, x_end, (255, 0, 0), 2)
    cv2.line(img, center, y_end, (0, 255, 0), 2)
    cv2.line(img, center, z_end, (0, 0, 255), 2)

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


def project_pose_center(K: np.ndarray, pose_4x4: np.ndarray) -> np.ndarray:
    """Project the pose origin (object center) onto the image plane.

    Returns:
        [x, y] pixel coordinates of the projected object center.
    """
    t = pose_4x4[:3, 3]
    p = K @ t
    if abs(p[2]) < 1e-6:
        return np.array([-1.0, -1.0])
    return np.array([p[0] / p[2], p[1] / p[2]])


def point_in_bbox(point: np.ndarray, bbox: np.ndarray, margin: float = 0.3) -> bool:
    """Check if a 2D point is inside a bbox with margin expansion.

    Args:
        point: [x, y] pixel coordinates.
        bbox: [x1, y1, x2, y2] bounding box.
        margin: Expand the bbox by this fraction on each side for tolerance.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    x1_exp = x1 - margin * w
    y1_exp = y1 - margin * h
    x2_exp = x2 + margin * w
    y2_exp = y2 + margin * h
    return x1_exp <= point[0] <= x2_exp and y1_exp <= point[1] <= y2_exp


def main():
    set_logging_level("info")

    parser = argparse.ArgumentParser(
        description="Standalone RealSense + DINOv2 + MegaPose fast tracker"
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
        "--reference-dir",
        type=str,
        default="DINO/datasets/yellowCube",
        help="Directory containing reference images of the target object",
    )
    parser.add_argument(
        "--dino-model",
        type=str,
        default="dinov2_vits14",
        help="DINOv2 model variant (dinov2_vits14, dinov2_vitb14, dinov2_vitl14)",
    )
    parser.add_argument(
        "--n-ref-patches",
        type=int,
        default=30,
        help="Number of top reference patches to keep as descriptors",
    )
    parser.add_argument(
        "--adaptive-std-factor",
        type=float,
        default=2.0,
        help="Std deviations above mean for adaptive threshold (higher = stricter)",
    )
    parser.add_argument(
        "--min-blob-area",
        type=int,
        default=4,
        help="Minimum patch count for a valid detection blob",
    )
    parser.add_argument(
        "--max-blob-ratio",
        type=float,
        default=0.5,
        help="Max fraction of frame a detection can cover (rejects full-frame blobs)",
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
    parser.add_argument("--max-track-failures", type=int, default=5)
    parser.add_argument(
        "--pose-delta-threshold",
        type=float,
        default=0.15,
        help="Max pose delta per frame before flagging as lost",
    )
    parser.add_argument(
        "--verify-interval",
        type=int,
        default=10,
        help="Run DINO verification every N frames to catch drift/removal (0=disabled)",
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=0.10,
        help="Max cumulative translation drift (meters) from anchor before re-detect",
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
    parser.add_argument(
        "--continuous-detection",
        action="store_true",
        help="Run DINO every frame to update bounding box (slower but tracks moving objects better)",
    )
    args = parser.parse_args()

    # Resolve mesh directory
    mesh_dir = Path(args.mesh_dir)
    if not mesh_dir.is_absolute():
        mesh_dir = REPO_ROOT / mesh_dir
    if not mesh_dir.exists():
        logger.error(f"Mesh directory not found: {mesh_dir}")
        sys.exit(1)

    # Resolve reference directory
    reference_dir = Path(args.reference_dir)
    if not reference_dir.is_absolute():
        reference_dir = REPO_ROOT / reference_dir
    if not reference_dir.exists():
        logger.error(f"Reference image directory not found: {reference_dir}")
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
    pose_estimator = load_named_model(model_name, object_dataset).cuda()
    logger.info("MegaPose model loaded.")

    logger.info(
        f"Loading DINOv2 detector (model={args.dino_model}, "
        f"references={reference_dir}) ..."
    )
    dino = DinoDetector(
        reference_dir=str(reference_dir),
        label=object_label,
        model_name=args.dino_model,
        n_ref_patches=args.n_ref_patches,
        adaptive_std_factor=args.adaptive_std_factor,
        min_blob_area=args.min_blob_area,
        max_blob_ratio=args.max_blob_ratio,
        device="cuda",
    )
    logger.info("DINOv2 detector loaded and reference features cached.")

    # --- Initialize RealSense ---
    logger.info("Initializing RealSense camera ...")
    pipeline, align, profile = init_realsense(
        args.realsense_width, args.realsense_height, args.realsense_fps
    )
    logger.info("RealSense initialized.")

    # --- Tracking state ---
    last_pose: np.ndarray | None = None
    anchor_pose: np.ndarray | None = None  # Pose at last confident detection
    last_bbox: np.ndarray | None = None
    consecutive_failures = 0
    frame_count = 0
    fps_smooth = 0.0

    logger.info("Starting tracking loop. Press 'q' to quit.")
    logger.info(
        f"Verification: every {args.verify_interval} frames | "
        f"drift threshold: {args.drift_threshold}m"
    )

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
                # --- SLOW PATH: DINO + full MegaPose pipeline ---
                result = dino.detect_best(rgb)
                if result is None:
                    vis = rgb[:, :, ::-1].copy()
                    cv2.putText(
                        vis,
                        "No detection - object not visible",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    if not args.no_display:
                        cv2.imshow("MegaPose Tracker (DINO)", vis)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    frame_count += 1
                    continue

                bbox, label, conf = result
                last_bbox = bbox
                logger.info(
                    f"DINO detection: conf={conf:.3f}, "
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
                    frame_count += 1
                    continue

                last_pose = output.poses[0].cpu().numpy()
                anchor_pose = last_pose.copy()
                consecutive_failures = 0
                mode_str = "DETECT"

            else:
                # --- FAST PATH: refiner only ---
                if args.continuous_detection:
                    result = dino.detect_best(rgb)
                    if result is not None:
                        last_bbox = result[0]

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
                    mode_str = "TRACK (no pose)"
                else:
                    new_pose = refined.poses[0].cpu().numpy()

                    # Check 1: Frame-to-frame jump
                    delta = compute_pose_delta(last_pose, new_pose)
                    if delta > args.pose_delta_threshold:
                        consecutive_failures += 1
                        mode_str = f"TRACK (jump={delta:.3f})"
                    else:
                        last_pose = new_pose
                        consecutive_failures = 0
                        mode_str = "TRACK"

                # Check 2: Cumulative drift from anchor pose
                if (
                    consecutive_failures == 0
                    and anchor_pose is not None
                    and last_pose is not None
                ):
                    drift = np.linalg.norm(
                        last_pose[:3, 3] - anchor_pose[:3, 3]
                    )
                    if drift > args.drift_threshold:
                        consecutive_failures += 2
                        mode_str = f"TRACK (drift={drift:.3f}m)"
                        logger.info(
                            f"Cumulative drift {drift:.3f}m exceeds "
                            f"threshold {args.drift_threshold}m"
                        )

                # Check 3: Periodic DINO verification (lightweight)
                # Runs DINO every N frames to confirm pose is on the object
                if (
                    args.verify_interval > 0
                    and frame_count % args.verify_interval == 0
                    and consecutive_failures == 0
                    and last_pose is not None
                ):
                    verify_result = dino.detect_best(rgb)

                    if verify_result is None:
                        # Object not visible at all — immediate failure
                        consecutive_failures = args.max_track_failures
                        mode_str = "VERIFY (object gone)"
                        logger.info(
                            "DINO verification: object not found in frame"
                        )
                    else:
                        verify_bbox = verify_result[0]
                        last_bbox = verify_bbox  # Update bbox from DINO

                        # Check if pose center projects inside DINO bbox
                        pose_center_px = project_pose_center(K, last_pose)
                        if not point_in_bbox(pose_center_px, verify_bbox):
                            consecutive_failures += 3
                            mode_str = "VERIFY (pose outside bbox)"
                            logger.info(
                                f"Pose center {pose_center_px.astype(int).tolist()} "
                                f"outside DINO bbox {verify_bbox.astype(int).tolist()}"
                            )

                # Check 4: Periodic scoring (optional, heavier)
                if (
                    args.periodic_scoring_interval > 0
                    and frame_count % args.periodic_scoring_interval == 0
                    and consecutive_failures == 0
                    and last_pose is not None
                ):
                    pose_tc = build_pose_input(object_label, last_pose).cuda()
                    with torch.no_grad():
                        scored, _ = pose_estimator.forward_scoring_model(
                            observation, pose_tc
                        )
                    score = scored.infos["pose_score"].iloc[0]
                    if score < 0.3:
                        consecutive_failures += 2
                        mode_str = f"SCORE (low={score:.2f})"

            frame_count += 1

            # --- Print pose to terminal ---
            if last_pose is not None and frame_count % 10 == 0:
                t = last_pose[:3, 3]
                R = last_pose[:3, :3]
                logger.info(
                    f"[{mode_str}] "
                    f"T=[{t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}]m | "
                    f"R=[{R[0,0]:+.3f} {R[0,1]:+.3f} {R[0,2]:+.3f}; "
                    f"{R[1,0]:+.3f} {R[1,1]:+.3f} {R[1,2]:+.3f}; "
                    f"{R[2,0]:+.3f} {R[2,1]:+.3f} {R[2,2]:+.3f}]"
                )

            # --- Visualization ---
            t_frame_end = time.time()
            frame_time = t_frame_end - t_frame_start
            fps_instant = 1.0 / max(frame_time, 1e-6)
            fps_smooth = 0.9 * fps_smooth + 0.1 * fps_instant

            if not args.no_display:
                vis = rgb[:, :, ::-1].copy()

                if last_bbox is not None:
                    draw_bbox(vis, last_bbox, object_label)

                if last_pose is not None:
                    draw_pose_axes(vis, K, last_pose)

                    # Draw pose projected center as a dot
                    pose_px = project_pose_center(K, last_pose)
                    cx, cy = int(pose_px[0]), int(pose_px[1])
                    cv2.circle(vis, (cx, cy), 5, (255, 0, 255), -1)

                status_color = (
                    (0, 255, 0) if consecutive_failures == 0 else (0, 0, 255)
                )
                info = (
                    f"{mode_str} | {fps_smooth:.1f} FPS | "
                    f"fails={consecutive_failures}/{args.max_track_failures}"
                )
                cv2.putText(
                    vis,
                    info,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    status_color,
                    2,
                    cv2.LINE_AA,
                )

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

                cv2.imshow("MegaPose Tracker (DINO)", vis)
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
