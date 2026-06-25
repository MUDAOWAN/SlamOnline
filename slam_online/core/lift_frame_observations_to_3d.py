#!/usr/bin/env python3
"""
Lift per-frame 2D object masks into per-frame 3D Gaussian observations.

Input:
  frame_observations.json from online_grounded_sam2_worker.py
  same-run MonoGS Gaussian PLY
  queue task records with pose_c2w and intrinsics

Output:
  frame_3d_observations.json/csv with lightweight metadata
  frame_3d_hits.npz with hit Gaussian point indices

This script does not perform final object clustering. It creates the
observation-level 3D evidence used by later object memory association.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from gaussian_ply import read_binary_little_endian_gaussian_ply


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "frame_3d_observations"


def find_first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def resolve_monogs_point_cloud(monogs_run_dir: Path) -> Path:
    candidates = [
        monogs_run_dir / "point_cloud" / "final" / "point_cloud.ply",
        monogs_run_dir / "point_cloud" / "final_after_opt" / "point_cloud.ply",
    ]
    found = find_first_existing(candidates)
    if found is not None:
        return found

    ply_paths = sorted(monogs_run_dir.glob("**/point_cloud.ply"))
    if not ply_paths:
        raise ValueError(f"No point_cloud.ply found under MonoGS run dir: {monogs_run_dir}")
    return ply_paths[-1]


def load_task_pose_c2w(task: dict[str, Any]) -> np.ndarray | None:
    pose = task.get("pose_c2w")
    if pose is None:
        return None
    pose_np = np.asarray(pose, dtype=np.float64)
    if pose_np.shape == (16,):
        pose_np = pose_np.reshape(4, 4)
    if pose_np.shape != (4, 4):
        return None
    return pose_np


def project_points(
    points_world: np.ndarray,
    camera_to_world: np.ndarray,
    width: int,
    height: int,
    intrinsics: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    world_to_camera = np.linalg.inv(camera_to_world)
    points_h = np.concatenate([points_world, np.ones((len(points_world), 1), dtype=np.float32)], axis=1)
    points_cam = (world_to_camera @ points_h.T).T[:, :3]

    z = points_cam[:, 2]
    in_front = z > 1e-6
    valid_depth_indices = np.nonzero(in_front)[0]
    if len(valid_depth_indices) == 0:
        return valid_depth_indices, np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)

    z_valid = z[valid_depth_indices]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = intrinsics["fx"] * points_cam[valid_depth_indices, 0] / z_valid + intrinsics["cx"]
        v = intrinsics["fy"] * points_cam[valid_depth_indices, 1] / z_valid + intrinsics["cy"]

    finite = np.isfinite(u) & np.isfinite(v)
    finite_indices = valid_depth_indices[finite]
    if len(finite_indices) == 0:
        return finite_indices, np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)

    u_i = np.rint(u[finite]).astype(np.int32)
    v_i = np.rint(v[finite]).astype(np.int32)
    in_image = (u_i >= 0) & (u_i < width) & (v_i >= 0) & (v_i < height)
    return finite_indices[in_image], u_i[in_image], v_i[in_image]


def load_mask_path(mask_path: Path, target_hw: tuple[int, int]) -> np.ndarray | None:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    height, width = target_hw
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def next_available_dir(base_dir: Path, name: str) -> Path:
    candidate = base_dir / name
    if not candidate.exists():
        return candidate
    suffix = 1
    while True:
        candidate = base_dir / f"{name}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def make_output_dir(output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return next_available_dir(output_root, f"lift_{timestamp}")


def load_queue_tasks(queue_root: Path, states: list[str]) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for state in states:
        state_dir = queue_root / state
        if not state_dir.exists():
            continue
        for path in sorted(state_dir.glob("*.json")):
            wrapper = read_json(path)
            task = wrapper.get("task")
            if not isinstance(task, dict):
                continue
            frame_stem = str(task.get("frame_stem") or f"frame_{int(task['frame_id']):06d}")
            task = dict(task)
            task["_queue_path"] = str(path)
            task["_queue_state"] = state
            tasks[frame_stem] = task
    return tasks


def normalize_states(value: str) -> list[str]:
    states = [item.strip() for item in value.split(",") if item.strip()]
    return states or ["done"]


def get_frame_size(task: dict[str, Any], frame: dict[str, Any], mask_path: Path) -> tuple[int, int] | None:
    width = task.get("width")
    height = task.get("height")
    if width is not None and height is not None:
        return int(width), int(height)

    image_path = Path(str(task.get("image_path") or frame.get("image_path") or ""))
    if image_path.exists():
        image = cv2.imread(str(image_path))
        if image is not None:
            h, w = image.shape[:2]
            return int(w), int(h)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is not None:
        h, w = mask.shape[:2]
        return int(w), int(h)
    return None


def load_depth_meters(task: dict[str, Any], target_hw: tuple[int, int]) -> np.ndarray | None:
    depth_path = Path(str(task.get("depth_path") or ""))
    if not depth_path.exists():
        return None
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    height, width = target_hw
    if depth.shape[:2] != (height, width):
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
    depth = depth.astype(np.float32)
    depth_scale = float(task.get("depth_scale") or 1.0)
    if depth_scale > 0:
        depth = depth / depth_scale
    return depth


def camera_depths_for_indices(points: np.ndarray, pose_c2w: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if len(indices) == 0:
        return np.empty((0,), dtype=np.float32)
    world_to_camera = np.linalg.inv(pose_c2w)
    selected = points[indices]
    selected_h = np.concatenate([selected, np.ones((len(selected), 1), dtype=np.float32)], axis=1)
    return (world_to_camera @ selected_h.T).T[:, 2].astype(np.float32)


def split_indices_by_voxel_components(
    points: np.ndarray,
    indices: np.ndarray,
    voxel_size: float,
    min_points: int,
    min_ratio: float,
    max_components: int,
) -> list[np.ndarray]:
    if len(indices) == 0:
        return []
    if voxel_size <= 0.0:
        return [indices]

    hit_points = points[indices]
    voxel_coords = np.floor(hit_points / voxel_size).astype(np.int32)
    voxel_to_point_positions: dict[tuple[int, int, int], list[int]] = {}
    for point_pos, coord in enumerate(voxel_coords):
        key = (int(coord[0]), int(coord[1]), int(coord[2]))
        voxel_to_point_positions.setdefault(key, []).append(point_pos)

    remaining = set(voxel_to_point_positions.keys())
    components: list[np.ndarray] = []
    neighbor_offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        voxels = [seed]
        while stack:
            vx, vy, vz = stack.pop()
            for dx, dy, dz in neighbor_offsets:
                neighbor = (vx + dx, vy + dy, vz + dz)
                if neighbor not in remaining:
                    continue
                remaining.remove(neighbor)
                stack.append(neighbor)
                voxels.append(neighbor)

        point_positions = []
        for voxel in voxels:
            point_positions.extend(voxel_to_point_positions[voxel])
        component_indices = indices[np.asarray(point_positions, dtype=np.int64)]
        components.append(component_indices.astype(np.int32))

    components.sort(key=len, reverse=True)
    min_count = max(int(min_points), int(np.ceil(len(indices) * min_ratio)))
    components = [component for component in components if len(component) >= min_count]
    if max_components > 0:
        components = components[:max_components]
    return components


def bbox_from_points(points: np.ndarray, percentile_low: float = 0.0, percentile_high: float = 100.0) -> dict[str, Any]:
    if percentile_low > 0.0 or percentile_high < 100.0:
        bbox_min = np.percentile(points, percentile_low, axis=0)
        bbox_max = np.percentile(points, percentile_high, axis=0)
        inside = np.all((points >= bbox_min) & (points <= bbox_max), axis=1)
        center_points = points[inside] if np.any(inside) else points
    else:
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        center_points = points
    bbox_size = bbox_max - bbox_min
    center = center_points.mean(axis=0)
    return {
        "center_3d": [round(float(x), 6) for x in center.tolist()],
        "bbox_min": [round(float(x), 6) for x in bbox_min.tolist()],
        "bbox_max": [round(float(x), 6) for x in bbox_max.tolist()],
        "bbox_size": [round(float(x), 6) for x in bbox_size.tolist()],
        "bbox_volume": round(float(np.prod(np.maximum(bbox_size, 0.0))), 9),
        "bbox_percentile_low": float(percentile_low),
        "bbox_percentile_high": float(percentile_high),
    }


def make_rejected_observation(
    observation: dict[str, Any],
    frame: dict[str, Any],
    reason: str,
    detail: str | None = None,
) -> dict[str, Any]:
    item = {
        "observation_id": observation.get("observation_id"),
        "frame_id": frame.get("frame_id"),
        "frame_stem": frame.get("frame_stem"),
        "label": observation.get("label"),
        "status": "rejected_3d",
        "reject_reason": reason,
        "detail": detail,
        "mask_path": observation.get("mask_path"),
        "max_box_score": observation.get("max_box_score"),
        "max_mask_score": observation.get("max_mask_score"),
        "mask_area_ratio": observation.get("mask_area_ratio"),
    }
    return item


def lift_observation(
    observation: dict[str, Any],
    frame: dict[str, Any],
    task: dict[str, Any],
    points: np.ndarray,
    visible_indices: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    width: int,
    height: int,
    obs_index: int,
    min_hit_points: int,
    min_hit_ratio: float,
    pose_c2w: np.ndarray,
    depth_meters: np.ndarray | None,
    depth_abs_tolerance: float,
    depth_rel_tolerance: float,
    bbox_percentile_low: float,
    bbox_percentile_high: float,
    split_components: bool,
    component_voxel_size: float,
    component_min_points: int,
    component_min_ratio: float,
    max_components_per_observation: int,
) -> list[tuple[dict[str, Any], np.ndarray | None]]:
    mask_path = Path(str(observation.get("mask_path") or ""))
    if not mask_path.exists():
        return [(make_rejected_observation(observation, frame, "missing_mask", str(mask_path)), None)]

    mask = load_mask_path(mask_path, (height, width))
    if mask is None:
        return [(make_rejected_observation(observation, frame, "failed_read_mask", str(mask_path)), None)]

    visible_count = int(len(visible_indices))
    if visible_count == 0:
        return [(make_rejected_observation(observation, frame, "no_visible_points"), None)]

    hit_mask = mask[v, u] > 127
    hit_indices = visible_indices[hit_mask].astype(np.int32)
    raw_hit_count = int(len(hit_indices))
    depth_consistent_count = None
    if depth_meters is not None and raw_hit_count > 0 and (depth_abs_tolerance > 0.0 or depth_rel_tolerance > 0.0):
        hit_u = u[hit_mask]
        hit_v = v[hit_mask]
        observed_depth = depth_meters[hit_v, hit_u]
        projected_depth = camera_depths_for_indices(points, pose_c2w, hit_indices)
        valid_depth = np.isfinite(observed_depth) & (observed_depth > 0.0) & np.isfinite(projected_depth)
        depth_delta = np.abs(projected_depth - observed_depth)
        allowed_delta = np.zeros_like(depth_delta, dtype=np.float32)
        if depth_abs_tolerance > 0.0:
            allowed_delta = np.maximum(allowed_delta, depth_abs_tolerance)
        if depth_rel_tolerance > 0.0:
            allowed_delta = np.maximum(allowed_delta, observed_depth * depth_rel_tolerance)
        keep_depth = valid_depth & (depth_delta <= allowed_delta)
        hit_indices = hit_indices[keep_depth].astype(np.int32)
        depth_consistent_count = int(len(hit_indices))
    hit_count = int(len(hit_indices))
    hit_ratio = float(hit_count / max(visible_count, 1))
    if hit_count < min_hit_points:
        rejected = make_rejected_observation(
            observation,
            frame,
            "too_few_hit_points",
            f"hit_point_count={hit_count} min_hit_points={min_hit_points}",
        )
        rejected["visible_point_count"] = visible_count
        rejected["raw_hit_point_count"] = raw_hit_count
        rejected["depth_consistent_hit_point_count"] = depth_consistent_count
        rejected["hit_point_count"] = hit_count
        rejected["hit_ratio"] = round(hit_ratio, 8)
        return [(rejected, hit_indices)]
    if hit_ratio < min_hit_ratio:
        rejected = make_rejected_observation(
            observation,
            frame,
            "hit_ratio_too_low",
            f"hit_ratio={hit_ratio:.8f} min_hit_ratio={min_hit_ratio}",
        )
        rejected["visible_point_count"] = visible_count
        rejected["raw_hit_point_count"] = raw_hit_count
        rejected["depth_consistent_hit_point_count"] = depth_consistent_count
        rejected["hit_point_count"] = hit_count
        rejected["hit_ratio"] = round(hit_ratio, 8)
        return [(rejected, hit_indices)]

    if split_components:
        component_indices = split_indices_by_voxel_components(
            points=points,
            indices=hit_indices,
            voxel_size=component_voxel_size,
            min_points=component_min_points,
            min_ratio=component_min_ratio,
            max_components=max_components_per_observation,
        )
        if not component_indices:
            rejected = make_rejected_observation(
                observation,
                frame,
                "no_valid_component",
                f"hit_point_count={hit_count} component_min_points={component_min_points} component_min_ratio={component_min_ratio}",
            )
            rejected["visible_point_count"] = visible_count
            rejected["raw_hit_point_count"] = raw_hit_count
            rejected["depth_consistent_hit_point_count"] = depth_consistent_count
            rejected["hit_point_count"] = hit_count
            rejected["hit_ratio"] = round(hit_ratio, 8)
            return [(rejected, hit_indices)]
    else:
        component_indices = [hit_indices]

    visible_key = f"{str(frame.get('frame_stem'))}_visible"
    lifted_items: list[tuple[dict[str, Any], np.ndarray | None]] = []
    component_count = len(component_indices)
    for component_idx, indices_for_component in enumerate(component_indices):
        current_obs_index = obs_index + component_idx
        hit_points = points[indices_for_component]
        geometry = bbox_from_points(hit_points, bbox_percentile_low, bbox_percentile_high)
        hit_key = f"obs_{current_obs_index:06d}_hit"
        base_observation_id = str(observation.get("observation_id"))
        observation_id = (
            f"{base_observation_id}:comp{component_idx:03d}"
            if split_components and component_count > 1
            else base_observation_id
        )
        lifted = {
            "observation_id": observation_id,
            "source_observation_id": observation.get("observation_id"),
            "observation_index": current_obs_index,
            "component_index": component_idx if split_components else None,
            "component_count": component_count if split_components else 1,
            "frame_id": int(frame["frame_id"]),
            "frame_stem": str(frame["frame_stem"]),
            "label": observation.get("label"),
            "label_slug": observation.get("label_slug"),
            "status": "accepted_3d",
            "mask_path": str(mask_path),
            "detections_json": observation.get("detections_json"),
            "hit_key": hit_key,
            "visible_key": visible_key,
            "hit_point_count": int(len(indices_for_component)),
            "raw_hit_point_count": raw_hit_count,
            "depth_consistent_hit_point_count": depth_consistent_count,
            "visible_point_count": visible_count,
            "hit_ratio": round(float(len(indices_for_component) / max(visible_count, 1)), 8),
            "max_box_score": observation.get("max_box_score"),
            "max_mask_score": observation.get("max_mask_score"),
            "mask_area_ratio": observation.get("mask_area_ratio"),
            "detection_count": observation.get("detection_count"),
            "kept_detection_count": observation.get("kept_detection_count"),
            "queue_task_path": task.get("_queue_path"),
            "queue_state": task.get("_queue_state"),
            "split_components": split_components,
            "component_voxel_size": component_voxel_size if split_components else None,
            **geometry,
        }
        lifted_items.append((lifted, indices_for_component.astype(np.int32)))
    return lifted_items


def write_observations_csv(path: Path, observations: list[dict[str, Any]]) -> None:
    fieldnames = [
        "observation_index",
        "observation_id",
        "frame_id",
        "frame_stem",
        "label",
        "status",
        "hit_point_count",
        "visible_point_count",
        "hit_ratio",
        "center_3d",
        "bbox_min",
        "bbox_max",
        "bbox_size",
        "bbox_volume",
        "max_box_score",
        "max_mask_score",
        "mask_area_ratio",
        "hit_key",
        "visible_key",
        "mask_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for obs in observations:
            writer.writerow(
                {
                    key: json.dumps(obs.get(key), ensure_ascii=False)
                    if isinstance(obs.get(key), list)
                    else obs.get(key)
                    for key in fieldnames
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Lift 2D frame observations into 3D Gaussian observations")
    parser.add_argument("--frame_observations_json", type=Path, required=True)
    parser.add_argument("--point_cloud", type=Path, default=None)
    parser.add_argument("--monogs_run_dir", type=Path, default=None)
    parser.add_argument("--queue_root", type=Path, required=True)
    parser.add_argument("--queue_states", type=str, default="done,processing,pending,failed")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--min_hit_points", type=int, default=30)
    parser.add_argument("--min_hit_ratio", type=float, default=0.0)
    parser.add_argument("--bbox_percentile_low", type=float, default=2.0)
    parser.add_argument("--bbox_percentile_high", type=float, default=98.0)
    parser.add_argument("--depth_abs_tolerance", type=float, default=0.05, help="Meters; 0 disables absolute depth consistency")
    parser.add_argument("--depth_rel_tolerance", type=float, default=0.03, help="Relative depth tolerance; 0 disables relative depth consistency")
    parser.add_argument(
        "--split_components",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Split disconnected 3D hit components before writing observations",
    )
    parser.add_argument("--component_voxel_size", type=float, default=0.08, help="Voxel size in meters for 3D component splitting")
    parser.add_argument("--component_min_points", type=int, default=120, help="Minimum points kept per split component")
    parser.add_argument("--component_min_ratio", type=float, default=0.08, help="Minimum component size ratio relative to source hit points")
    parser.add_argument("--max_components_per_observation", type=int, default=3, help="Maximum split components emitted for one 2D observation; <=0 keeps all")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--max_observations", type=int, default=None)
    args = parser.parse_args()
    if not (0.0 <= args.bbox_percentile_low < args.bbox_percentile_high <= 100.0):
        raise ValueError("--bbox_percentile_low/high must satisfy 0 <= low < high <= 100")

    if args.monogs_run_dir is not None:
        point_cloud_path = resolve_monogs_point_cloud(args.monogs_run_dir)
    elif args.point_cloud is not None:
        point_cloud_path = args.point_cloud
    else:
        raise ValueError("Pass --point_cloud or --monogs_run_dir")

    data = read_json(args.frame_observations_json)
    frames = data.get("frames") if isinstance(data, dict) else None
    if not isinstance(frames, list):
        raise ValueError(f"Expected frames list in {args.frame_observations_json}")
    if args.max_frames is not None and args.max_frames > 0:
        frames = frames[: args.max_frames]

    queue_tasks = load_queue_tasks(args.queue_root, normalize_states(args.queue_states))
    points, _gaussian_vertices, _gaussian_header = read_binary_little_endian_gaussian_ply(point_cloud_path)

    output_dir = make_output_dir(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_results: list[dict[str, Any]] = []
    lifted_observations: list[dict[str, Any]] = []
    rejected_observations: list[dict[str, Any]] = []
    npz_arrays: dict[str, np.ndarray] = {}
    processed_frames = 0
    skipped_frames = 0
    observation_index = 0

    for frame in frames:
        frame_stem = str(frame.get("frame_stem") or f"frame_{int(frame['frame_id']):06d}")
        task = queue_tasks.get(frame_stem)
        if not isinstance(task, dict):
            skipped_frames += 1
            frame_results.append(
                {
                    "frame_id": frame.get("frame_id"),
                    "frame_stem": frame_stem,
                    "status": "missing_queue_task",
                }
            )
            continue

        pose = load_task_pose_c2w(task)
        intrinsics = task.get("intrinsics")
        accepted_2d = frame.get("accepted_observations") or []
        if pose is None:
            skipped_frames += 1
            frame_results.append({"frame_id": frame.get("frame_id"), "frame_stem": frame_stem, "status": "missing_pose"})
            continue
        if not isinstance(intrinsics, dict):
            skipped_frames += 1
            frame_results.append({"frame_id": frame.get("frame_id"), "frame_stem": frame_stem, "status": "missing_intrinsics"})
            continue
        if not accepted_2d:
            processed_frames += 1
            frame_results.append(
                {
                    "frame_id": frame.get("frame_id"),
                    "frame_stem": frame_stem,
                    "status": "processed",
                    "accepted_2d_count": 0,
                    "accepted_3d_count": 0,
                    "rejected_3d_count": 0,
                }
            )
            continue

        first_mask = Path(str(accepted_2d[0].get("mask_path") or ""))
        frame_size = get_frame_size(task, frame, first_mask)
        if frame_size is None:
            skipped_frames += 1
            frame_results.append({"frame_id": frame.get("frame_id"), "frame_stem": frame_stem, "status": "missing_frame_size"})
            continue
        width, height = frame_size
        depth_meters = load_depth_meters(task, (height, width))

        valid_indices, u, v = project_points(
            points_world=points,
            camera_to_world=pose,
            width=width,
            height=height,
            intrinsics={
                "fx": float(intrinsics["fx"]),
                "fy": float(intrinsics["fy"]),
                "cx": float(intrinsics["cx"]),
                "cy": float(intrinsics["cy"]),
            },
        )
        visible_key = f"{frame_stem}_visible"
        npz_arrays[visible_key] = valid_indices.astype(np.int32)

        frame_accepted_3d = 0
        frame_rejected_3d = 0
        frame_observation_ids = []
        for observation in accepted_2d:
            if args.max_observations is not None and args.max_observations > 0 and observation_index >= args.max_observations:
                break

            lifted_results = lift_observation(
                observation=observation,
                frame=frame,
                task=task,
                points=points,
                visible_indices=valid_indices,
                u=u,
                v=v,
                width=width,
                height=height,
                obs_index=observation_index,
                min_hit_points=args.min_hit_points,
                min_hit_ratio=args.min_hit_ratio,
                pose_c2w=pose,
                depth_meters=depth_meters,
                depth_abs_tolerance=args.depth_abs_tolerance,
                depth_rel_tolerance=args.depth_rel_tolerance,
                bbox_percentile_low=args.bbox_percentile_low,
                bbox_percentile_high=args.bbox_percentile_high,
                split_components=args.split_components,
                component_voxel_size=args.component_voxel_size,
                component_min_points=args.component_min_points,
                component_min_ratio=args.component_min_ratio,
                max_components_per_observation=args.max_components_per_observation,
            )
            for lifted, hit_indices in lifted_results:
                if args.max_observations is not None and args.max_observations > 0 and observation_index >= args.max_observations:
                    break

                if lifted["status"] == "accepted_3d":
                    lifted["observation_index"] = observation_index
                    lifted["hit_key"] = f"obs_{observation_index:06d}_hit"
                    if hit_indices is not None:
                        npz_arrays[lifted["hit_key"]] = hit_indices.astype(np.int32)
                    lifted_observations.append(lifted)
                    frame_accepted_3d += 1
                    frame_observation_ids.append(lifted["observation_id"])
                else:
                    lifted["observation_index"] = observation_index
                    rejected_observations.append(lifted)
                    frame_rejected_3d += 1
                observation_index += 1

        processed_frames += 1
        frame_results.append(
            {
                "frame_id": frame.get("frame_id"),
                "frame_stem": frame_stem,
                "status": "processed",
                "width": width,
                "height": height,
                "visible_key": visible_key,
                "visible_point_count": int(len(valid_indices)),
                "accepted_2d_count": len(accepted_2d),
                "accepted_3d_count": frame_accepted_3d,
                "rejected_3d_count": frame_rejected_3d,
                "accepted_3d_observation_ids": frame_observation_ids,
                "queue_task_path": task.get("_queue_path"),
            }
        )

        if args.max_observations is not None and args.max_observations > 0 and observation_index >= args.max_observations:
            break

    hits_path = output_dir / "frame_3d_hits.npz"
    np.savez_compressed(hits_path, **npz_arrays)

    summary = {
        "frame_count": len(frames),
        "processed_frame_count": processed_frames,
        "skipped_frame_count": skipped_frames,
        "accepted_3d_observation_count": len(lifted_observations),
        "rejected_3d_observation_count": len(rejected_observations),
        "npz_array_count": len(npz_arrays),
        "min_hit_points": args.min_hit_points,
        "min_hit_ratio": args.min_hit_ratio,
        "bbox_percentile_low": args.bbox_percentile_low,
        "bbox_percentile_high": args.bbox_percentile_high,
        "depth_abs_tolerance": args.depth_abs_tolerance,
        "depth_rel_tolerance": args.depth_rel_tolerance,
        "split_components": args.split_components,
        "component_voxel_size": args.component_voxel_size,
        "component_min_points": args.component_min_points,
        "component_min_ratio": args.component_min_ratio,
        "max_components_per_observation": args.max_components_per_observation,
    }
    payload = {
        "format": "frame_3d_observations_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metadata": {
            "frame_observations_json": str(args.frame_observations_json),
            "point_cloud": str(point_cloud_path),
            "queue_root": str(args.queue_root),
            "queue_states": normalize_states(args.queue_states),
            "hits_npz": str(hits_path),
            "bbox_percentile_low": args.bbox_percentile_low,
            "bbox_percentile_high": args.bbox_percentile_high,
            "depth_abs_tolerance": args.depth_abs_tolerance,
            "depth_rel_tolerance": args.depth_rel_tolerance,
            "split_components": args.split_components,
            "component_voxel_size": args.component_voxel_size,
            "component_min_points": args.component_min_points,
            "component_min_ratio": args.component_min_ratio,
            "max_components_per_observation": args.max_components_per_observation,
        },
        "summary": summary,
        "frames": frame_results,
        "observations": lifted_observations,
        "rejected_observations": rejected_observations,
    }
    write_json(output_dir / "frame_3d_observations.json", payload)
    write_observations_csv(output_dir / "frame_3d_observations.csv", lifted_observations)

    summary_lines = [
        "Frame 3D observation lifting summary",
        "",
        f"frames: {summary['frame_count']}",
        f"processed_frames: {processed_frames}",
        f"skipped_frames: {skipped_frames}",
        f"accepted_3d_observations: {len(lifted_observations)}",
        f"rejected_3d_observations: {len(rejected_observations)}",
        f"bbox_percentile: {args.bbox_percentile_low}-{args.bbox_percentile_high}",
        f"depth_abs_tolerance: {args.depth_abs_tolerance}",
        f"depth_rel_tolerance: {args.depth_rel_tolerance}",
        f"split_components: {args.split_components}",
        f"component_voxel_size: {args.component_voxel_size}",
        f"component_min_points: {args.component_min_points}",
        f"component_min_ratio: {args.component_min_ratio}",
        f"max_components_per_observation: {args.max_components_per_observation}",
        f"hits_npz: {hits_path}",
        f"output: {output_dir}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[point_cloud] {point_cloud_path}")
    print(f"[frames] {len(frames)}")
    print(f"[processed_frames] {processed_frames}")
    print(f"[skipped_frames] {skipped_frames}")
    print(f"[accepted_3d_observations] {len(lifted_observations)}")
    print(f"[rejected_3d_observations] {len(rejected_observations)}")
    print(f"[hits_npz] {hits_path}")
    print(f"[output] {output_dir}")


if __name__ == "__main__":
    main()
