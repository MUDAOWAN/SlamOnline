#!/usr/bin/env python3
"""
Visualize object_memory_update.py outputs as Gaussian-compatible PLY files.

Inputs:
  - original MonoGS Gaussian point_cloud.ply
  - object_memory.json
  - object_memory_points.npz

Outputs:
  - object_memory_colored_original_background.ply: original scene colors with object points colored
  - object_memory_colored_dim_background.ply: optional dim-gray scene with object points colored
  - object_memory_objects_only.ply: optional only object points
  - per_object_ply/*.ply: optional one original-scene+bbox PLY per object
  - object_color_legend.svg/json/csv and summary.txt

By default, this script visualizes only object points with at least 5000
Gaussian hits. It does not append bbox/center marker Gaussians, but center and
bbox metadata remain in the JSON/CSV/SVG legend and summary.

Use --bbox_only to leave the original scene colors untouched and append only
colored 3D bbox/center markers for the selected objects.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from gaussian_ply import (
    read_binary_little_endian_gaussian_ply,
    rgb_to_sh_dc,
    write_binary_little_endian_gaussian_ply,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "object_memory_visualization"


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
    return next_available_dir(output_root, f"object_memory_viz_{timestamp}")


def parse_statuses(value: str) -> set[str]:
    statuses = {part.strip() for part in value.split(",") if part.strip()}
    if not statuses:
        raise ValueError("At least one status is required")
    return statuses


def parse_csv_set(value: str | None) -> set[str]:
    if value is None or not value.strip():
        return set()
    return {part.strip().lower().replace("_", " ") for part in value.split(",") if part.strip()}


def parse_csv_raw_set(value: str | None) -> set[str]:
    if value is None or not value.strip():
        return set()
    return {part.strip() for part in value.split(",") if part.strip()}


def stable_color(index: int) -> tuple[int, int, int]:
    """Generate visually separated deterministic colors without extra deps."""
    hue = (index * 0.618033988749895) % 1.0
    saturation = 0.68
    value = 0.92
    h = hue * 6.0
    c = value * saturation
    x = c * (1.0 - abs(h % 2.0 - 1.0))
    m = value - c
    if h < 1:
        r, g, b = c, x, 0
    elif h < 2:
        r, g, b = x, c, 0
    elif h < 3:
        r, g, b = 0, c, x
    elif h < 4:
        r, g, b = 0, x, c
    elif h < 5:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return (
        int(round((r + m) * 255)),
        int(round((g + m) * 255)),
        int(round((b + m) * 255)),
    )


def object_label(obj: dict[str, Any]) -> str:
    label = str(obj.get("canonical_label") or "").strip().lower().replace("_", " ")
    return label or "object"


def label_color_map(objects: list[dict[str, Any]]) -> dict[str, tuple[int, int, int]]:
    labels = sorted({object_label(obj) for obj in objects})
    return {label: stable_color(idx) for idx, label in enumerate(labels)}


def color_vertices(vertices: np.ndarray, color_rgb: tuple[int, int, int]) -> None:
    rgb = tuple(channel / 255.0 for channel in color_rgb)
    dc = rgb_to_sh_dc(rgb)
    vertices["f_dc_0"] = dc[0]
    vertices["f_dc_1"] = dc[1]
    vertices["f_dc_2"] = dc[2]


def color_vertices_at(vertices: np.ndarray, indices: np.ndarray, color_rgb: tuple[int, int, int]) -> None:
    if len(indices) == 0:
        return
    rgb = tuple(channel / 255.0 for channel in color_rgb)
    dc = rgb_to_sh_dc(rgb)
    vertices["f_dc_0"][indices] = dc[0]
    vertices["f_dc_1"][indices] = dc[1]
    vertices["f_dc_2"][indices] = dc[2]


def update_vertex_count_header(header_lines: list[str], vertex_count: int) -> list[str]:
    out = []
    replaced = False
    for line in header_lines:
        if line.startswith("element vertex "):
            out.append(f"element vertex {vertex_count}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        raise ValueError("PLY header has no element vertex line")
    return out


def safe_filename(value: str) -> str:
    value = value.strip().lower().replace(" ", "_").replace("/", "_")
    value = re.sub(r"[^a-z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "object"


def bbox_corners(bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    x0, y0, z0 = bbox_min
    x1, y1, z1 = bbox_max
    return np.asarray(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float32,
    )


BBOX_EDGE_INDICES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def sample_bbox_edges(bbox_min: np.ndarray, bbox_max: np.ndarray, samples_per_edge: int) -> np.ndarray:
    return sample_box_edges(bbox_corners(bbox_min, bbox_max), samples_per_edge)


def sample_box_edges(corners: np.ndarray, samples_per_edge: int) -> np.ndarray:
    samples = []
    alphas = np.linspace(0.0, 1.0, max(2, samples_per_edge), dtype=np.float32)
    for start_idx, end_idx in BBOX_EDGE_INDICES:
        start = corners[start_idx]
        end = corners[end_idx]
        samples.append(start[None, :] * (1.0 - alphas[:, None]) + end[None, :] * alphas[:, None])
    return np.concatenate(samples, axis=0)


def robust_obb_corners(points: np.ndarray, low_percentile: float, high_percentile: float) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < 8:
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        return bbox_corners(bbox_min, bbox_max), (bbox_min + bbox_max) * 0.5

    center = np.median(points, axis=0)
    centered = points - center[None, :]
    cov = np.cov(centered, rowvar=False)
    if not np.all(np.isfinite(cov)):
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        return bbox_corners(bbox_min, bbox_max), (bbox_min + bbox_max) * 0.5

    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    axes = vectors[:, order].astype(np.float32)
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1.0

    local = centered @ axes
    local_min = np.percentile(local, low_percentile, axis=0)
    local_max = np.percentile(local, high_percentile, axis=0)
    x0, y0, z0 = local_min
    x1, y1, z1 = local_max
    local_corners = np.asarray(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float32,
    )
    corners = center[None, :] + local_corners @ axes.T
    local_center = (local_min + local_max) * 0.5
    world_center = center + local_center @ axes.T
    return corners.astype(np.float32), world_center.astype(np.float32)


def expand_marker_points(points: np.ndarray, radius: float) -> np.ndarray:
    if len(points) == 0 or radius <= 0.0:
        return points
    offsets = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [radius, 0.0, 0.0],
            [-radius, 0.0, 0.0],
            [0.0, radius, 0.0],
            [0.0, -radius, 0.0],
            [0.0, 0.0, radius],
            [0.0, 0.0, -radius],
        ],
        dtype=np.float32,
    )
    return (points[:, None, :] + offsets[None, :, :]).reshape(-1, 3)


def make_debug_gaussians(
    template_vertex: np.void,
    xyz: np.ndarray,
    color_rgb: tuple[int, int, int],
    scale: float,
    opacity: float,
) -> np.ndarray:
    debug_vertices = np.empty(len(xyz), dtype=template_vertex.dtype)
    debug_vertices[:] = template_vertex
    debug_vertices["x"] = xyz[:, 0]
    debug_vertices["y"] = xyz[:, 1]
    debug_vertices["z"] = xyz[:, 2]

    if "nx" in debug_vertices.dtype.names:
        debug_vertices["nx"] = 0.0
    if "ny" in debug_vertices.dtype.names:
        debug_vertices["ny"] = 0.0
    if "nz" in debug_vertices.dtype.names:
        debug_vertices["nz"] = 0.0
    if {"scale_0", "scale_1", "scale_2"}.issubset(debug_vertices.dtype.names):
        log_scale = np.float32(math.log(max(scale, 1e-8)))
        debug_vertices["scale_0"] = log_scale
        debug_vertices["scale_1"] = log_scale
        debug_vertices["scale_2"] = log_scale
    if {"rot_0", "rot_1", "rot_2", "rot_3"}.issubset(debug_vertices.dtype.names):
        debug_vertices["rot_0"] = 1.0
        debug_vertices["rot_1"] = 0.0
        debug_vertices["rot_2"] = 0.0
        debug_vertices["rot_3"] = 0.0
    if "opacity" in debug_vertices.dtype.names:
        debug_vertices["opacity"] = np.float32(opacity)
    color_vertices(debug_vertices, color_rgb)
    return debug_vertices


def resolve_points_npz(object_memory_json: Path, data: dict[str, Any], explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    path = metadata.get("object_memory_points_npz")
    if path:
        return Path(str(path))
    return object_memory_json.parent / "object_memory_points.npz"


def filter_objects(
    objects: list[dict[str, Any]],
    statuses: set[str],
    final_statuses: set[str],
    min_points: int,
    labels: set[str],
    object_ids: set[str],
    max_bbox_volume: float | None,
) -> list[dict[str, Any]]:
    selected = []
    for obj in objects:
        if str(obj.get("status")) not in statuses:
            continue
        final_status = str(obj.get("final_status") or "kept")
        if final_statuses and final_status not in final_statuses:
            continue
        if int(obj.get("point_count") or 0) < min_points:
            continue
        obj_id = str(obj.get("object_id") or "")
        canonical_label = str(obj.get("canonical_label") or "").strip().lower().replace("_", " ")
        aliases = {str(alias).strip().lower().replace("_", " ") for alias in obj.get("aliases", [])}
        if labels and canonical_label not in labels and not (aliases & labels):
            continue
        if object_ids and obj_id not in object_ids:
            continue
        if max_bbox_volume is not None and float(obj.get("bbox_volume") or 0.0) > max_bbox_volume:
            continue
        selected.append(obj)
    return selected


def write_legend_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "object_id",
        "status",
        "canonical_label",
        "aliases",
        "rgb",
        "support_frame_count",
        "support_observation_count",
        "point_count",
        "center_3d",
        "bbox_size",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "object_id": row["object_id"],
                    "status": row["status"],
                    "canonical_label": row["canonical_label"],
                    "aliases": ",".join(row["aliases"]),
                    "rgb": ",".join(str(v) for v in row["rgb"]),
                    "support_frame_count": row["support_frame_count"],
                    "support_observation_count": row["support_observation_count"],
                    "point_count": row["point_count"],
                    "center_3d": ",".join(str(v) for v in row["center_3d"]),
                    "bbox_size": ",".join(str(v) for v in row["bbox_size"]),
                }
            )


def write_legend_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    swatch_size = 18
    row_height = 30
    top = 24
    left = 18
    text_x = left + swatch_size + 12
    width = 980
    height = max(80, top * 2 + row_height * len(rows))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f7f4"/>',
        '<text x="18" y="18" font-family="DejaVu Sans, Arial, sans-serif" font-size="14" fill="#222">Object color legend</text>',
    ]
    for idx, row in enumerate(rows):
        y = top + idx * row_height
        r, g, b = row["rgb"]
        label = html.escape(str(row["canonical_label"]))
        object_id = html.escape(str(row["object_id"]))
        status = html.escape(str(row["status"]))
        aliases = html.escape(", ".join(str(x) for x in row.get("aliases", [])))
        points = html.escape(str(row["point_count"]))
        center = html.escape(", ".join(str(x) for x in row.get("center_3d", [])))
        text = f"{object_id} | {label} | {status} | points={points} | center=[{center}] | aliases=[{aliases}]"
        lines.extend(
            [
                f'<rect x="{left}" y="{y}" width="{swatch_size}" height="{swatch_size}" rx="3" fill="rgb({r},{g},{b})"/>',
                f'<text x="{text_x}" y="{y + 14}" font-family="DejaVu Sans, Arial, sans-serif" font-size="13" fill="#222">{text}</text>',
            ]
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_simple_legend_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    label_rows: dict[str, tuple[int, int, int]] = {}
    for row in rows:
        label = str(row.get("canonical_label") or "object").strip().lower().replace("_", " ")
        if label and label not in label_rows:
            rgb = row.get("rgb")
            if isinstance(rgb, (list, tuple)) and len(rgb) == 3:
                label_rows[label] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))

    swatch_size = 20
    row_height = 34
    padding_x = 20
    padding_y = 22
    text_x = padding_x + swatch_size + 12
    labels = sorted(label_rows)
    text_width = max((len(label) for label in labels), default=6) * 8
    width = max(260, min(640, text_x + text_width + padding_x))
    height = max(64, padding_y * 2 + row_height * len(labels))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    for idx, label in enumerate(labels):
        y = padding_y + idx * row_height
        r, g, b = label_rows[label]
        escaped_label = html.escape(label)
        lines.extend(
            [
                f'<rect x="{padding_x}" y="{y}" width="{swatch_size}" height="{swatch_size}" rx="3" fill="rgb({r},{g},{b})"/>',
                f'<text x="{text_x}" y="{y + 15}" font-family="DejaVu Sans, Arial, sans-serif" font-size="15" fill="#222">{escaped_label}</text>',
            ]
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize object memory as colored Gaussian PLY")
    parser.add_argument("--point_cloud", type=Path, required=True)
    parser.add_argument("--object_memory_json", type=Path, required=True)
    parser.add_argument("--object_points_npz", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--statuses", type=str, default="confirmed")
    parser.add_argument("--final_statuses", type=str, default="", help="Comma-separated final_status values, e.g. kept")
    parser.add_argument("--min_points", type=int, default=1)
    parser.add_argument("--labels", type=str, default="", help="Comma-separated canonical labels or aliases to visualize, e.g. chair,office chair")
    parser.add_argument("--object_ids", type=str, default="", help="Comma-separated object ids to visualize, e.g. obj_000002,obj_000013")
    parser.add_argument("--max_bbox_volume", type=float, default=None, help="Skip objects with bbox volume above this value")
    parser.add_argument("--dim_background", action="store_true", help="Also write a gray-background PLY")
    parser.add_argument("--background_rgb", type=str, default="135,135,135")
    parser.add_argument("--write_objects_only", action="store_true", help="Also write an objects-only PLY")
    parser.add_argument("--with_bbox_markers", action="store_true", help="Append bbox and center marker Gaussians")
    parser.add_argument(
        "--bbox_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Do not recolor object points; append only bbox/center markers on the original scene",
    )
    parser.add_argument("--write_per_object_plys", action="store_true", help="Write one original-scene+bbox PLY for each selected object")
    parser.add_argument("--per_object_dir_name", type=str, default="per_object_ply")
    parser.add_argument("--bbox_line_samples", type=int, default=96)
    parser.add_argument("--bbox_mode", choices=("aabb", "obb"), default="obb")
    parser.add_argument("--obb_percentile_low", type=float, default=2.0)
    parser.add_argument("--obb_percentile_high", type=float, default=98.0)
    parser.add_argument("--bbox_scale", type=float, default=0.01)
    parser.add_argument("--bbox_opacity", type=float, default=4.0)
    parser.add_argument("--bbox_marker_radius", type=float, default=0.005, help="Expand bbox line samples into small 3D crosses for more visible boxes")
    args = parser.parse_args()

    statuses = parse_statuses(args.statuses)
    final_statuses = parse_csv_raw_set(args.final_statuses)
    labels = parse_csv_set(args.labels)
    object_ids = {part.strip() for part in args.object_ids.split(",") if part.strip()}
    background_rgb = tuple(int(x.strip()) for x in args.background_rgb.split(","))
    if len(background_rgb) != 3:
        raise ValueError("--background_rgb must be r,g,b")
    if args.bbox_only:
        args.with_bbox_markers = True
    if args.write_per_object_plys:
        args.with_bbox_markers = True

    output_dir = make_output_dir(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_object_dir = output_dir / args.per_object_dir_name
    if args.write_per_object_plys:
        per_object_dir.mkdir(parents=True, exist_ok=True)

    memory_data = read_json(args.object_memory_json)
    objects = memory_data.get("objects") if isinstance(memory_data, dict) else None
    if not isinstance(objects, list):
        raise ValueError(f"Expected objects list in {args.object_memory_json}")
    selected_objects = filter_objects(
        objects=objects,
        statuses=statuses,
        final_statuses=final_statuses,
        min_points=args.min_points,
        labels=labels,
        object_ids=object_ids,
        max_bbox_volume=args.max_bbox_volume,
    )

    points_npz_path = resolve_points_npz(args.object_memory_json, memory_data, args.object_points_npz)
    object_points = np.load(points_npz_path)
    _points, gaussian_vertices, gaussian_header = read_binary_little_endian_gaussian_ply(args.point_cloud)

    print(f"[point_cloud] {args.point_cloud}")
    print(f"[object_memory] {args.object_memory_json}")
    print(f"[object_points] {points_npz_path}")
    print(f"[objects_total] {len(objects)}")
    print(f"[objects_selected] {len(selected_objects)}")
    print(f"[output] {output_dir}")

    colored_original_scene = gaussian_vertices.copy()
    colored_dim_scene = gaussian_vertices.copy() if args.dim_background else None
    if colored_dim_scene is not None:
        color_vertices(colored_dim_scene, background_rgb)  # type: ignore[arg-type]

    object_only_blocks: list[np.ndarray] = []
    marker_blocks: list[np.ndarray] = []
    per_object_outputs: list[dict[str, Any]] = []
    template_vertex = gaussian_vertices[0]
    legend_rows: list[dict[str, Any]] = []
    colors_by_label = label_color_map(selected_objects)

    for obj in selected_objects:
        object_id = str(obj["object_id"])
        point_key = str(obj.get("object_point_key") or f"{object_id}_points")
        if point_key not in object_points:
            print(f"[warn] missing points for {object_id}: {point_key}")
            continue

        indices = np.asarray(object_points[point_key], dtype=np.int64)
        indices = indices[(indices >= 0) & (indices < len(gaussian_vertices))]
        if len(indices) == 0:
            print(f"[warn] empty points for {object_id}: {point_key}")
            continue

        color = colors_by_label[object_label(obj)]
        if not args.bbox_only:
            color_vertices_at(colored_original_scene, indices, color)
            if colored_dim_scene is not None:
                color_vertices_at(colored_dim_scene, indices, color)

        if args.write_objects_only and not args.bbox_only:
            selected_vertices = gaussian_vertices[indices].copy()
            color_vertices(selected_vertices, color)
            object_only_blocks.append(selected_vertices)

        per_object_marker_blocks: list[np.ndarray] = []
        if args.with_bbox_markers:
            if args.bbox_mode == "obb":
                object_xyz = np.column_stack(
                    (
                        gaussian_vertices["x"][indices],
                        gaussian_vertices["y"][indices],
                        gaussian_vertices["z"][indices],
                    )
                ).astype(np.float32)
                corners, center_point = robust_obb_corners(
                    object_xyz,
                    low_percentile=args.obb_percentile_low,
                    high_percentile=args.obb_percentile_high,
                )
                edge_points = sample_box_edges(corners, args.bbox_line_samples)
            else:
                bbox_min = np.asarray(obj["bbox_min"], dtype=np.float32)
                bbox_max = np.asarray(obj["bbox_max"], dtype=np.float32)
                edge_points = sample_bbox_edges(bbox_min, bbox_max, args.bbox_line_samples)
                center_point = np.asarray(obj["center_3d"], dtype=np.float32)
            edge_points = expand_marker_points(edge_points, args.bbox_marker_radius)
            edge_vertices = make_debug_gaussians(
                    template_vertex=template_vertex,
                    xyz=edge_points,
                    color_rgb=color,
                    scale=args.bbox_scale,
                    opacity=args.bbox_opacity,
                )
            marker_blocks.append(edge_vertices)
            per_object_marker_blocks.append(edge_vertices)
            if args.write_objects_only:
                object_only_blocks.append(edge_vertices)

            center = center_point[None, :]
            center = expand_marker_points(center, args.bbox_marker_radius * 2.0)
            center_vertices = make_debug_gaussians(
                    template_vertex=template_vertex,
                    xyz=center,
                    color_rgb=(255, 255, 255),
                    scale=args.bbox_scale * 1.8,
                    opacity=args.bbox_opacity,
            )
            marker_blocks.append(center_vertices)
            per_object_marker_blocks.append(center_vertices)
            if args.write_objects_only:
                object_only_blocks.append(center_vertices)

        per_object_path = None
        if args.write_per_object_plys:
            per_object_vertices = np.concatenate([gaussian_vertices.copy(), *per_object_marker_blocks], axis=0)
            per_object_header = update_vertex_count_header(gaussian_header, len(per_object_vertices))
            filename = (
                f"{object_id}_{safe_filename(str(obj.get('canonical_label') or 'object'))}_"
                f"{safe_filename(str(obj.get('status') or 'status'))}.ply"
            )
            per_object_path = per_object_dir / filename
            write_binary_little_endian_gaussian_ply(per_object_path, per_object_vertices, per_object_header)
            per_object_outputs.append(
                {
                    "object_id": object_id,
                    "canonical_label": obj.get("canonical_label"),
                    "status": obj.get("status"),
                    "path": str(per_object_path),
                }
            )

        legend_rows.append(
            {
                "object_id": object_id,
                "status": obj.get("status"),
                "canonical_label": obj.get("canonical_label"),
                "aliases": obj.get("aliases", []),
                "rgb": color,
                "support_frame_count": obj.get("support_frame_count"),
                "support_observation_count": obj.get("support_observation_count"),
                "point_count": int(len(indices)),
                "center_3d": obj.get("center_3d"),
                "bbox_size": obj.get("bbox_size"),
                "object_point_key": point_key,
                "per_object_ply": str(per_object_path) if per_object_path else None,
            }
        )

    scene_blocks = [colored_original_scene, *marker_blocks]
    colored_original_vertices = np.concatenate(scene_blocks, axis=0)
    colored_original_header = update_vertex_count_header(gaussian_header, len(colored_original_vertices))
    original_background_path = output_dir / "object_memory_colored_original_background.ply"
    write_binary_little_endian_gaussian_ply(original_background_path, colored_original_vertices, colored_original_header)

    dim_background_path = None
    if colored_dim_scene is not None:
        dim_blocks = [colored_dim_scene, *marker_blocks]
        colored_dim_vertices = np.concatenate(dim_blocks, axis=0)
        colored_dim_header = update_vertex_count_header(gaussian_header, len(colored_dim_vertices))
        dim_background_path = output_dir / "object_memory_colored_dim_background.ply"
        write_binary_little_endian_gaussian_ply(dim_background_path, colored_dim_vertices, colored_dim_header)

    objects_only_path = None
    if args.write_objects_only:
        if object_only_blocks:
            object_only_vertices = np.concatenate(object_only_blocks, axis=0)
        else:
            object_only_vertices = gaussian_vertices[:0].copy()
        objects_only_header = update_vertex_count_header(gaussian_header, len(object_only_vertices))
        objects_only_path = output_dir / "object_memory_objects_only.ply"
        write_binary_little_endian_gaussian_ply(objects_only_path, object_only_vertices, objects_only_header)

    write_json(output_dir / "object_color_legend.json", {"objects": legend_rows})
    if args.write_per_object_plys:
        write_json(output_dir / "per_object_plys.json", {"objects": per_object_outputs})
    write_legend_csv(output_dir / "object_color_legend.csv", legend_rows)
    write_legend_svg(output_dir / "object_color_legend.svg", legend_rows)
    write_simple_legend_svg(output_dir / "object_color_name_legend.svg", legend_rows)

    lines = [
        "Object memory visualization summary",
        "",
        f"point_cloud: {args.point_cloud}",
        f"object_memory_json: {args.object_memory_json}",
        f"object_points_npz: {points_npz_path}",
        f"statuses: {sorted(statuses)}",
        f"labels: {sorted(labels)}",
        f"object_ids: {sorted(object_ids)}",
        f"objects_total: {len(objects)}",
        f"objects_selected: {len(legend_rows)}",
        f"min_points: {args.min_points}",
        f"max_bbox_volume: {args.max_bbox_volume}",
        f"with_bbox_markers: {args.with_bbox_markers}",
        f"bbox_only: {args.bbox_only}",
        f"bbox_mode: {args.bbox_mode}",
        f"obb_percentile: {args.obb_percentile_low}-{args.obb_percentile_high}",
        f"bbox_line_samples: {args.bbox_line_samples}",
        f"bbox_scale: {args.bbox_scale}",
        f"bbox_marker_radius: {args.bbox_marker_radius}",
        "color_by: canonical_label",
        f"write_per_object_plys: {args.write_per_object_plys}",
        f"per_object_dir: {per_object_dir if args.write_per_object_plys else None}",
        f"colored_original_background_ply: {original_background_path}",
        f"colored_dim_background_ply: {dim_background_path}",
        f"objects_only_ply: {objects_only_path}",
        f"legend_svg: {output_dir / 'object_color_legend.svg'}",
        f"simple_legend_svg: {output_dir / 'object_color_name_legend.svg'}",
        "",
        "objects:",
    ]
    for row in legend_rows:
        lines.append(
            f"- {row['object_id']} {row['canonical_label']} status={row['status']} "
            f"points={row['point_count']} center={row['center_3d']} "
            f"rgb={row['rgb']} aliases={row['aliases']}"
        )
    lines.extend(["", f"output: {output_dir}"])
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[colored_original_background_ply] {original_background_path}")
    if dim_background_path is not None:
        print(f"[colored_dim_background_ply] {dim_background_path}")
    if objects_only_path is not None:
        print(f"[objects_only_ply] {objects_only_path}")
    if args.write_per_object_plys:
        print(f"[per_object_ply_dir] {per_object_dir}")
        print(f"[per_object_ply_count] {len(per_object_outputs)}")
    print(f"[legend_svg] {output_dir / 'object_color_legend.svg'}")
    print(f"[simple_legend_svg] {output_dir / 'object_color_name_legend.svg'}")
    print(f"[legend_csv] {output_dir / 'object_color_legend.csv'}")
    print(f"[done] outputs at {output_dir}")


if __name__ == "__main__":
    main()
