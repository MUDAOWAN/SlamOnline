#!/usr/bin/env python3
"""
Export visible scene objects and metric spatial relations for LLM reasoning.

The exporter consumes a refined object memory and writes compact JSON/CSV/MD
files. It keeps the geometry numeric and explicit so downstream prompts can
reason from measured scene facts instead of image appearance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "scene_objects"


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
    return next_available_dir(output_root, f"scene_{timestamp}")


def normalize_label(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ")


def safe_name(value: str) -> str:
    return "_".join(part for part in normalize_label(value).split() if part) or "object"


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def finite_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def vector3(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        return [0.0, 0.0, 0.0]
    return [round(finite_float(item), 6) for item in value]


def parse_csv_raw_set(value: str | None) -> set[str]:
    if value is None or not value.strip():
        return set()
    return {part.strip() for part in value.split(",") if part.strip()}


def object_volume(obj: dict[str, Any]) -> float:
    volume = finite_float(obj.get("bbox_volume"))
    if volume > 0.0:
        return volume
    sx, sy, sz = vector3(obj.get("bbox_size"))
    return sx * sy * sz


def select_objects(
    objects: list[dict[str, Any]],
    statuses: set[str],
    final_statuses: set[str],
    min_points: int,
) -> list[dict[str, Any]]:
    selected = []
    for obj in objects:
        if statuses and str(obj.get("status")) not in statuses:
            continue
        if final_statuses and str(obj.get("final_status") or "kept") not in final_statuses:
            continue
        if finite_int(obj.get("point_count")) < min_points:
            continue
        selected.append(obj)
    return selected


def assign_display_names(objects: list[dict[str, Any]]) -> dict[str, str]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for obj in objects:
        by_label[normalize_label(obj.get("canonical_label"))].append(obj)

    display_names: dict[str, str] = {}
    for label, items in sorted(by_label.items()):
        ordered = sorted(items, key=lambda item: str(item.get("object_id") or ""))
        for index, obj in enumerate(ordered, start=1):
            display_names[str(obj.get("object_id"))] = f"{safe_name(label)}_{index}"
    return display_names


def aabb_corners(bbox_min: list[float], bbox_max: list[float]) -> list[list[float]]:
    x0, y0, z0 = bbox_min
    x1, y1, z1 = bbox_max
    return [
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ]


def object_record(obj: dict[str, Any], display_name: str) -> dict[str, Any]:
    bbox_min = vector3(obj.get("bbox_min"))
    bbox_max = vector3(obj.get("bbox_max"))
    center = vector3(obj.get("center_3d"))
    bbox_size = vector3(obj.get("bbox_size"))
    return {
        "object_id": str(obj.get("object_id")),
        "display_name": display_name,
        "canonical_label": normalize_label(obj.get("canonical_label")),
        "aliases": [normalize_label(alias) for alias in obj.get("aliases", [])],
        "status": obj.get("status"),
        "final_status": obj.get("final_status") or "kept",
        "center_3d": center,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "bbox_size": bbox_size,
        "bbox_volume": round(object_volume(obj), 9),
        "bbox_corners": aabb_corners(bbox_min, bbox_max),
        "support_frame_count": finite_int(obj.get("support_frame_count")),
        "support_observation_count": finite_int(obj.get("support_observation_count")),
        "point_count": finite_int(obj.get("point_count")),
        "source_object_ids": obj.get("source_object_ids") or [str(obj.get("object_id"))],
    }


def center_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    ca = a["center_3d"]
    cb = b["center_3d"]
    return math.sqrt(sum((ca[i] - cb[i]) ** 2 for i in range(3)))


def aabb_overlap(a: dict[str, Any], b: dict[str, Any]) -> tuple[float, float]:
    mins = [max(a["bbox_min"][i], b["bbox_min"][i]) for i in range(3)]
    maxs = [min(a["bbox_max"][i], b["bbox_max"][i]) for i in range(3)]
    size = [max(0.0, maxs[i] - mins[i]) for i in range(3)]
    inter = size[0] * size[1] * size[2]
    va = max(float(a["bbox_volume"]), 1e-9)
    vb = max(float(b["bbox_volume"]), 1e-9)
    union = max(va + vb - inter, 1e-9)
    return inter / union, inter / max(min(va, vb), 1e-9)


def relation_records(
    objects: list[dict[str, Any]],
    near_distance: float,
    horizontal_clearance: float,
    vertical_clearance: float,
    top_k_nearest: int,
) -> list[dict[str, Any]]:
    pair_rows: list[dict[str, Any]] = []
    nearest_by_object: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for idx, a in enumerate(objects):
        for b in objects[idx + 1 :]:
            distance = center_distance(a, b)
            dx = b["center_3d"][0] - a["center_3d"][0]
            dy = b["center_3d"][1] - a["center_3d"][1]
            dz = b["center_3d"][2] - a["center_3d"][2]
            overlap_iou, overlap_smaller = aabb_overlap(a, b)
            labels = []
            if distance <= near_distance:
                labels.append("near")
            if overlap_iou > 0.0:
                labels.append("aabb_overlap")
            if abs(dx) >= horizontal_clearance:
                labels.append("x_greater" if dx > 0 else "x_less")
            if abs(dy) >= horizontal_clearance:
                labels.append("y_greater" if dy > 0 else "y_less")
            if abs(dz) >= vertical_clearance:
                labels.append("higher" if dz > 0 else "lower")
            if a["canonical_label"] == b["canonical_label"]:
                labels.append("same_label")

            row = {
                "object_id_a": a["object_id"],
                "object_id_b": b["object_id"],
                "display_name_a": a["display_name"],
                "display_name_b": b["display_name"],
                "label_a": a["canonical_label"],
                "label_b": b["canonical_label"],
                "distance_3d": round(distance, 6),
                "delta_xyz_b_minus_a": [round(dx, 6), round(dy, 6), round(dz, 6)],
                "aabb_iou": round(overlap_iou, 6),
                "aabb_intersection_over_smaller": round(overlap_smaller, 6),
                "relations": labels,
            }
            pair_rows.append(row)
            nearest_by_object[a["object_id"]].append((distance, b["object_id"]))
            nearest_by_object[b["object_id"]].append((distance, a["object_id"]))

    keep_pairs: set[tuple[str, str]] = set()
    for object_id, neighbors in nearest_by_object.items():
        for _, other_id in sorted(neighbors)[:top_k_nearest]:
            keep_pairs.add(tuple(sorted((object_id, other_id))))

    compact_rows = []
    for row in pair_rows:
        key = tuple(sorted((row["object_id_a"], row["object_id_b"])))
        if key in keep_pairs or "near" in row["relations"] or "aabb_overlap" in row["relations"]:
            compact_rows.append(row)
    return sorted(compact_rows, key=lambda item: (item["distance_3d"], item["object_id_a"], item["object_id_b"]))


def write_objects_csv(path: Path, objects: list[dict[str, Any]]) -> None:
    fieldnames = [
        "object_id",
        "display_name",
        "canonical_label",
        "aliases",
        "center_x",
        "center_y",
        "center_z",
        "size_x",
        "size_y",
        "size_z",
        "bbox_volume",
        "support_frame_count",
        "support_observation_count",
        "point_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for obj in objects:
            writer.writerow(
                {
                    "object_id": obj["object_id"],
                    "display_name": obj["display_name"],
                    "canonical_label": obj["canonical_label"],
                    "aliases": ",".join(obj["aliases"]),
                    "center_x": obj["center_3d"][0],
                    "center_y": obj["center_3d"][1],
                    "center_z": obj["center_3d"][2],
                    "size_x": obj["bbox_size"][0],
                    "size_y": obj["bbox_size"][1],
                    "size_z": obj["bbox_size"][2],
                    "bbox_volume": obj["bbox_volume"],
                    "support_frame_count": obj["support_frame_count"],
                    "support_observation_count": obj["support_observation_count"],
                    "point_count": obj["point_count"],
                }
            )


def write_relations_csv(path: Path, relations: list[dict[str, Any]]) -> None:
    fieldnames = [
        "display_name_a",
        "display_name_b",
        "label_a",
        "label_b",
        "distance_3d",
        "delta_x",
        "delta_y",
        "delta_z",
        "aabb_iou",
        "aabb_intersection_over_smaller",
        "relations",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rel in relations:
            dx, dy, dz = rel["delta_xyz_b_minus_a"]
            writer.writerow(
                {
                    "display_name_a": rel["display_name_a"],
                    "display_name_b": rel["display_name_b"],
                    "label_a": rel["label_a"],
                    "label_b": rel["label_b"],
                    "distance_3d": rel["distance_3d"],
                    "delta_x": dx,
                    "delta_y": dy,
                    "delta_z": dz,
                    "aabb_iou": rel["aabb_iou"],
                    "aabb_intersection_over_smaller": rel["aabb_intersection_over_smaller"],
                    "relations": ",".join(rel["relations"]),
                }
            )


def write_markdown(path: Path, objects: list[dict[str, Any]], relations: list[dict[str, Any]]) -> None:
    label_counts = Counter(obj["canonical_label"] for obj in objects)
    lines = [
        "# Scene Objects For LLM Reasoning",
        "",
        "Coordinate frame: metric 3D coordinates from the reconstructed scene. Use numeric deltas and distances directly; x/y are horizontal scene axes and z is vertical.",
        "",
        f"Objects: {len(objects)}",
        f"Labels: {dict(label_counts.most_common())}",
        "",
        "## Objects",
        "",
        "| name | label | center_xyz | size_xyz | evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for obj in objects:
        lines.append(
            f"| {obj['display_name']} | {obj['canonical_label']} | {obj['center_3d']} | "
            f"{obj['bbox_size']} | frames={obj['support_frame_count']}, points={obj['point_count']} |"
        )
    lines.extend(["", "## Closest / Overlapping Relations", ""])
    for rel in relations[:80]:
        relation_text = ", ".join(rel["relations"]) if rel["relations"] else "metric_relation"
        lines.append(
            f"- {rel['display_name_a']} -> {rel['display_name_b']}: "
            f"distance={rel['distance_3d']}m, delta={rel['delta_xyz_b_minus_a']}, {relation_text}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export scene objects and spatial relations")
    parser.add_argument("--object_memory_json", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--statuses", type=str, default="confirmed")
    parser.add_argument("--final_statuses", type=str, default="kept")
    parser.add_argument("--min_points", type=int, default=1)
    parser.add_argument("--near_distance", type=float, default=1.25)
    parser.add_argument("--horizontal_clearance", type=float, default=0.25)
    parser.add_argument("--vertical_clearance", type=float, default=0.20)
    parser.add_argument("--top_k_nearest", type=int, default=4)
    args = parser.parse_args()

    data = read_json(args.object_memory_json)
    raw_objects = data.get("objects") if isinstance(data, dict) else None
    if not isinstance(raw_objects, list):
        raise ValueError(f"Expected objects list in {args.object_memory_json}")

    statuses = parse_csv_raw_set(args.statuses)
    final_statuses = parse_csv_raw_set(args.final_statuses)
    selected = select_objects(raw_objects, statuses=statuses, final_statuses=final_statuses, min_points=args.min_points)
    display_names = assign_display_names(selected)
    objects = [object_record(obj, display_names[str(obj.get("object_id"))]) for obj in selected]
    objects = sorted(objects, key=lambda obj: (obj["canonical_label"], obj["display_name"], obj["object_id"]))
    relations = relation_records(
        objects,
        near_distance=args.near_distance,
        horizontal_clearance=args.horizontal_clearance,
        vertical_clearance=args.vertical_clearance,
        top_k_nearest=args.top_k_nearest,
    )

    output_dir = make_output_dir(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_counts = Counter(obj["canonical_label"] for obj in objects)
    payload = {
        "format": "scene_objects_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metadata": {
            "source_object_memory_json": str(args.object_memory_json),
            "statuses": sorted(statuses),
            "final_statuses": sorted(final_statuses),
            "near_distance": args.near_distance,
            "horizontal_clearance": args.horizontal_clearance,
            "vertical_clearance": args.vertical_clearance,
            "top_k_nearest": args.top_k_nearest,
            "coordinate_frame_note": "x/y are horizontal scene axes; z is vertical.",
        },
        "summary": {
            "object_count": len(objects),
            "label_counts": dict(label_counts.most_common()),
            "relation_count": len(relations),
        },
        "objects": objects,
        "relations": relations,
    }
    write_json(output_dir / "scene_objects.json", payload)
    write_objects_csv(output_dir / "scene_objects.csv", objects)
    write_json(output_dir / "scene_relations.json", {"relations": relations})
    write_relations_csv(output_dir / "scene_relations.csv", relations)
    write_markdown(output_dir / "scene_summary_for_llm.md", objects, relations)

    lines = [
        "Scene object export summary",
        "",
        f"object_memory_json: {args.object_memory_json}",
        f"objects: {len(objects)}",
        f"label_counts: {dict(label_counts.most_common())}",
        f"relations: {len(relations)}",
        f"scene_objects_json: {output_dir / 'scene_objects.json'}",
        f"scene_objects_csv: {output_dir / 'scene_objects.csv'}",
        f"scene_relations_json: {output_dir / 'scene_relations.json'}",
        f"scene_summary_for_llm: {output_dir / 'scene_summary_for_llm.md'}",
        f"output: {output_dir}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[objects] {len(objects)}")
    print(f"[label_counts] {dict(label_counts.most_common())}")
    print(f"[relations] {len(relations)}")
    print(f"[output] {output_dir}")


if __name__ == "__main__":
    main()
