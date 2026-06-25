#!/usr/bin/env python3
"""
Merge partial/duplicate object-memory nodes after conservative refine.

P3 is intentionally conservative:

  - it starts from objects whose final_status is kept
  - it merges same-label partial/whole or strong duplicate pairs
  - it can use label-similarity relations for cautious cross-label merges
  - it writes a new final object memory and final object point arrays

The script does not modify the source files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "object_memory_global_merge"


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
    return next_available_dir(output_root, f"merge_{timestamp}")


def normalize_label(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ")


def parse_csv_raw_set(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def finite_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (3,):
        return np.zeros((3,), dtype=np.float64)
    arr[~np.isfinite(arr)] = 0.0
    return arr


def load_pair_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def label_pair_key(label_a: str, label_b: str) -> str:
    a, b = sorted((normalize_label(label_a), normalize_label(label_b)))
    return f"{a}|{b}"


def load_label_similarity(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    data = read_json(path)
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise ValueError(f"Expected items list in label similarity JSON: {path}")

    pairs: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        label_a = normalize_label(item.get("label_a"))
        label_b = normalize_label(item.get("label_b"))
        if not label_a or not label_b or label_a == label_b:
            continue
        pairs[label_pair_key(label_a, label_b)] = {
            "label_a": label_a,
            "label_b": label_b,
            "relation": str(item.get("relation") or "unknown"),
            "merge_policy": str(item.get("merge_policy") or "unknown"),
            "similarity": finite_float(item.get("similarity")),
            "reason": str(item.get("reason") or ""),
        }
    return pairs


def semantic_pair_for_objects(
    obj_a: dict[str, Any],
    obj_b: dict[str, Any],
    label_similarity: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    label_a = normalize_label(obj_a.get("canonical_label"))
    label_b = normalize_label(obj_b.get("canonical_label"))
    return label_similarity.get(label_pair_key(label_a, label_b))


def support_score(obj: dict[str, Any]) -> float:
    frames = finite_int(obj.get("support_frame_count"))
    observations = finite_int(obj.get("support_observation_count"))
    points = finite_int(obj.get("point_count"))
    box_score = finite_float(obj.get("mean_box_score"))
    return frames * 10.0 + observations * 2.0 + min(points / 1000.0, 20.0) + box_score


def label_family(label: str) -> str | None:
    label = normalize_label(label)
    if "chair" in label or label == "stool":
        return "chair"
    if "table" in label or "desk" in label:
        return "table"
    if label in {"couch", "sofa", "loveseat"}:
        return "sofa"
    return None


def is_chair_like(label: str) -> bool:
    return label_family(label) == "chair"


def bbox_volume(obj: dict[str, Any]) -> float:
    return finite_float(obj.get("bbox_volume"))


def resolve_points_npz(object_memory_json: Path, data: dict[str, Any], explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    path = metadata.get("object_memory_points_npz")
    if path:
        return Path(str(path))
    return object_memory_json.parent / "object_memory_points.npz"


def choose_target_for_same_label(
    obj_a: dict[str, Any],
    obj_b: dict[str, Any],
    relation: str,
) -> tuple[str, str]:
    id_a = str(obj_a.get("object_id"))
    id_b = str(obj_b.get("object_id"))
    if relation == "possible_partial_whole_same_label":
        if bbox_volume(obj_a) != bbox_volume(obj_b):
            return (id_a, id_b) if bbox_volume(obj_a) > bbox_volume(obj_b) else (id_b, id_a)
    score_a = support_score(obj_a)
    score_b = support_score(obj_b)
    if score_a != score_b:
        return (id_a, id_b) if score_a > score_b else (id_b, id_a)
    return (id_a, id_b) if bbox_volume(obj_a) >= bbox_volume(obj_b) else (id_b, id_a)


def more_specific_label(label_a: str, label_b: str) -> str | None:
    label_a = normalize_label(label_a)
    label_b = normalize_label(label_b)
    if not label_a or not label_b or label_a == label_b:
        return None
    words_a = set(label_a.split())
    words_b = set(label_b.split())
    if words_a < words_b:
        return label_b
    if words_b < words_a:
        return label_a
    if len(words_a) != len(words_b):
        return label_a if len(words_a) > len(words_b) else label_b
    return None


def choose_target_for_merge(
    obj_a: dict[str, Any],
    obj_b: dict[str, Any],
    source_relation: str,
    semantic_pair: dict[str, Any] | None,
) -> tuple[str, str]:
    label_a = normalize_label(obj_a.get("canonical_label"))
    label_b = normalize_label(obj_b.get("canonical_label"))
    if label_a == label_b:
        return choose_target_for_same_label(obj_a, obj_b, source_relation)

    id_a = str(obj_a.get("object_id"))
    id_b = str(obj_b.get("object_id"))
    score_a = support_score(obj_a)
    score_b = support_score(obj_b)
    semantic_relation = str((semantic_pair or {}).get("relation") or "unknown")
    if semantic_relation == "parent_child":
        specific = more_specific_label(label_a, label_b)
        if specific == label_a and score_a >= score_b * 0.5:
            return id_a, id_b
        if specific == label_b and score_b >= score_a * 0.5:
            return id_b, id_a

    if score_a != score_b:
        return (id_a, id_b) if score_a > score_b else (id_b, id_a)
    return (id_a, id_b) if bbox_volume(obj_a) >= bbox_volume(obj_b) else (id_b, id_a)


def should_merge_same_label(
    row: dict[str, str],
    obj_a: dict[str, Any],
    obj_b: dict[str, Any],
    table_min_point_containment: float,
    table_min_bbox_containment: float,
    generic_min_point_containment: float,
    generic_min_bbox_containment: float,
    duplicate_min_point_iou: float,
    chair_min_point_iou: float,
    chair_min_bbox_iou: float,
    chair_max_center_distance: float,
) -> tuple[bool, str]:
    relation = str(row.get("relation") or "")
    if relation not in {"possible_partial_whole_same_label", "possible_duplicate_same_label"}:
        return False, "not_same_label_merge_relation"

    label_a = normalize_label(obj_a.get("canonical_label"))
    label_b = normalize_label(obj_b.get("canonical_label"))
    if label_a != label_b:
        return False, "labels_differ"

    point_containment = finite_float(row.get("point_intersection_over_smaller"))
    bbox_containment = finite_float(row.get("bbox_intersection_over_smaller"))
    point_iou = finite_float(row.get("point_iou"))
    bbox_iou = finite_float(row.get("bbox_iou"))
    center_distance = finite_float(row.get("center_distance"))

    if is_chair_like(label_a):
        if point_iou >= chair_min_point_iou and bbox_iou >= chair_min_bbox_iou and center_distance <= chair_max_center_distance:
            return True, "chair_like_near_exact_duplicate"
        return False, "chair_like_preserve_instances"

    if label_family(label_a) == "table":
        if relation == "possible_partial_whole_same_label":
            ok = point_containment >= table_min_point_containment and bbox_containment >= table_min_bbox_containment
            return ok, "table_partial_whole" if ok else "table_overlap_too_weak"
        ok = point_iou >= duplicate_min_point_iou or point_containment >= table_min_point_containment
        return ok, "table_duplicate" if ok else "table_duplicate_too_weak"

    if relation == "possible_partial_whole_same_label":
        ok = point_containment >= generic_min_point_containment and bbox_containment >= generic_min_bbox_containment
        return ok, "generic_partial_whole" if ok else "generic_overlap_too_weak"

    ok = point_iou >= duplicate_min_point_iou or (
        point_containment >= generic_min_point_containment and bbox_containment >= generic_min_bbox_containment
    )
    return ok, "generic_duplicate" if ok else "generic_duplicate_too_weak"


def should_merge_by_label_similarity(
    row: dict[str, str],
    obj_a: dict[str, Any],
    obj_b: dict[str, Any],
    semantic_pair: dict[str, Any] | None,
    near_synonym_min_point_iou: float,
    near_synonym_min_bbox_iou: float,
    near_synonym_partial_min_point_containment: float,
    near_synonym_partial_min_bbox_containment: float,
    near_synonym_partial_max_center_distance: float,
    parent_child_min_point_iou: float,
    parent_child_min_bbox_iou: float,
    parent_child_partial_min_point_containment: float,
    parent_child_partial_min_bbox_containment: float,
    parent_child_partial_max_center_distance: float,
) -> tuple[bool, str]:
    label_a = normalize_label(obj_a.get("canonical_label"))
    label_b = normalize_label(obj_b.get("canonical_label"))
    if label_a == label_b:
        return False, "same_label_uses_same_label_policy"
    if semantic_pair is None:
        return False, "semantic_relation_unknown"
    if str(row.get("relation") or "") != "label_conflict_same_region":
        return False, "not_cross_label_same_region_relation"

    semantic_relation = str(semantic_pair.get("relation") or "unknown")
    merge_policy = str(semantic_pair.get("merge_policy") or "unknown")
    if semantic_relation == "related_but_distinct" or merge_policy == "hold":
        return False, f"semantic_{semantic_relation}_review_only"
    if semantic_relation not in {"near_synonym", "parent_child"}:
        return False, f"semantic_{semantic_relation}_not_mergeable"

    point_iou = finite_float(row.get("point_iou"))
    bbox_iou = finite_float(row.get("bbox_iou"))
    point_containment = finite_float(row.get("point_intersection_over_smaller"))
    bbox_containment = finite_float(row.get("bbox_intersection_over_smaller"))
    center_distance = finite_float(row.get("center_distance"))

    if semantic_relation == "near_synonym":
        duplicate_ok = point_iou >= near_synonym_min_point_iou and bbox_iou >= near_synonym_min_bbox_iou
        partial_ok = (
            point_containment >= near_synonym_partial_min_point_containment
            and bbox_containment >= near_synonym_partial_min_bbox_containment
            and center_distance <= near_synonym_partial_max_center_distance
        )
        if duplicate_ok:
            return True, "semantic_near_synonym_duplicate"
        if partial_ok:
            return True, "semantic_near_synonym_partial_whole"
        return False, "semantic_near_synonym_overlap_too_weak"

    duplicate_ok = point_iou >= parent_child_min_point_iou and bbox_iou >= parent_child_min_bbox_iou
    partial_ok = (
        point_containment >= parent_child_partial_min_point_containment
        and bbox_containment >= parent_child_partial_min_bbox_containment
        and center_distance <= parent_child_partial_max_center_distance
    )
    if duplicate_ok:
        return True, "semantic_parent_child_duplicate"
    if partial_ok:
        return True, "semantic_parent_child_partial_whole"
    return False, "semantic_parent_child_overlap_too_weak"


class UnionFind:
    def __init__(self, ids: list[str]) -> None:
        self.parent = {object_id: object_id for object_id in ids}

    def find(self, object_id: str) -> str:
        parent = self.parent[object_id]
        if parent != object_id:
            self.parent[object_id] = self.find(parent)
        return self.parent[object_id]

    def union_to(self, absorbed_id: str, target_id: str) -> None:
        absorbed_root = self.find(absorbed_id)
        target_root = self.find(target_id)
        if absorbed_root != target_root:
            self.parent[absorbed_root] = target_root


def bbox_from_objects(items: list[dict[str, Any]]) -> dict[str, Any]:
    mins = np.asarray([as_array(obj.get("bbox_min")) for obj in items], dtype=np.float64)
    maxs = np.asarray([as_array(obj.get("bbox_max")) for obj in items], dtype=np.float64)
    bbox_min = mins.min(axis=0)
    bbox_max = maxs.max(axis=0)
    bbox_size = np.maximum(bbox_max - bbox_min, 0.0)
    center = (bbox_min + bbox_max) * 0.5
    return {
        "center_3d": [round(float(x), 6) for x in center.tolist()],
        "bbox_min": [round(float(x), 6) for x in bbox_min.tolist()],
        "bbox_max": [round(float(x), 6) for x in bbox_max.tolist()],
        "bbox_size": [round(float(x), 6) for x in bbox_size.tolist()],
        "bbox_volume": round(float(np.prod(bbox_size)), 9),
    }


def merge_objects(
    root_id: str,
    member_ids: list[str],
    objects_by_id: dict[str, dict[str, Any]],
    source_points: dict[str, np.ndarray],
) -> tuple[dict[str, Any], np.ndarray]:
    root = dict(objects_by_id[root_id])
    members = [objects_by_id[object_id] for object_id in member_ids]
    label_counts: Counter[str] = Counter()
    observation_indices: list[int] = []
    observation_ids: list[Any] = []
    observed_frames: set[int] = set()
    aliases: set[str] = set()
    point_arrays = []

    for obj in members:
        label_counts.update({normalize_label(k): finite_int(v) for k, v in dict(obj.get("label_counts", {})).items()})
        aliases.update(normalize_label(alias) for alias in obj.get("aliases", []))
        observation_indices.extend(finite_int(item) for item in obj.get("observation_indices", []))
        observation_ids.extend(obj.get("observation_ids", []))
        observed_frames.update(finite_int(item) for item in obj.get("observed_frames", []))
        point_key = str(obj.get("object_point_key") or f"{obj.get('object_id')}_points")
        if point_key in source_points:
            point_arrays.append(np.asarray(source_points[point_key], dtype=np.int32))

    points = np.unique(np.concatenate(point_arrays)) if point_arrays else np.empty((0,), dtype=np.int32)
    geometry = bbox_from_objects(members)
    root.update(geometry)
    root["object_id"] = root_id
    root["global_status"] = "final"
    root["source_object_ids"] = sorted(member_ids)
    root["absorbed_object_ids"] = sorted(object_id for object_id in member_ids if object_id != root_id)
    root["label_counts"] = dict(label_counts.most_common())
    root_label = normalize_label(root.get("canonical_label"))
    root["canonical_label"] = root_label if root_label in root["label_counts"] else next(
        iter(root["label_counts"]), root_label
    )
    root["aliases"] = sorted(alias for alias in aliases if alias)
    root["observation_indices"] = sorted(set(observation_indices))
    root["observation_ids"] = observation_ids
    root["observed_frames"] = sorted(observed_frames)
    root["first_seen_frame"] = min(observed_frames) if observed_frames else root.get("first_seen_frame")
    root["last_seen_frame"] = max(observed_frames) if observed_frames else root.get("last_seen_frame")
    root["support_frame_count"] = len(observed_frames)
    root["support_observation_count"] = len(observation_ids)
    root["point_count"] = int(len(points))
    root["object_point_key"] = f"{root_id}_points"
    root["final_status"] = "kept"
    return root, points.astype(np.int32)


def write_merge_report(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "action",
        "absorbed_object_id",
        "target_object_id",
        "label",
        "reason",
        "source_relation",
        "semantic_relation",
        "semantic_merge_policy",
        "semantic_similarity",
        "semantic_reason",
        "bbox_iou",
        "bbox_intersection_over_smaller",
        "point_intersection_over_smaller",
        "point_iou",
        "center_distance",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge same-label partial/duplicate object-memory nodes")
    parser.add_argument("--object_memory_json", type=Path, required=True)
    parser.add_argument("--object_pair_issues_csv", type=Path, required=True)
    parser.add_argument("--label_similarity_json", type=Path, default=None)
    parser.add_argument("--object_points_npz", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--include_final_statuses", type=str, default="kept")
    parser.add_argument("--table_min_point_containment", type=float, default=0.60)
    parser.add_argument("--table_min_bbox_containment", type=float, default=0.70)
    parser.add_argument("--generic_min_point_containment", type=float, default=0.80)
    parser.add_argument("--generic_min_bbox_containment", type=float, default=0.80)
    parser.add_argument("--duplicate_min_point_iou", type=float, default=0.35)
    parser.add_argument("--chair_min_point_iou", type=float, default=0.85)
    parser.add_argument("--chair_min_bbox_iou", type=float, default=0.80)
    parser.add_argument("--chair_max_center_distance", type=float, default=0.10)
    parser.add_argument("--near_synonym_min_point_iou", type=float, default=0.50)
    parser.add_argument("--near_synonym_min_bbox_iou", type=float, default=0.20)
    parser.add_argument("--near_synonym_partial_min_point_containment", type=float, default=0.70)
    parser.add_argument("--near_synonym_partial_min_bbox_containment", type=float, default=0.70)
    parser.add_argument("--near_synonym_partial_max_center_distance", type=float, default=0.45)
    parser.add_argument("--parent_child_min_point_iou", type=float, default=0.65)
    parser.add_argument("--parent_child_min_bbox_iou", type=float, default=0.35)
    parser.add_argument("--parent_child_partial_min_point_containment", type=float, default=0.85)
    parser.add_argument("--parent_child_partial_min_bbox_containment", type=float, default=0.80)
    parser.add_argument("--parent_child_partial_max_center_distance", type=float, default=0.35)
    args = parser.parse_args()

    data = read_json(args.object_memory_json)
    objects = data.get("objects") if isinstance(data, dict) else None
    if not isinstance(objects, list):
        raise ValueError(f"Expected objects list in {args.object_memory_json}")

    include_final_statuses = parse_csv_raw_set(args.include_final_statuses)
    selected = [
        obj
        for obj in objects
        if str(obj.get("final_status") or "kept") in include_final_statuses
    ]
    objects_by_id = {str(obj.get("object_id")): obj for obj in selected}
    pair_rows = load_pair_rows(args.object_pair_issues_csv)
    label_similarity = load_label_similarity(args.label_similarity_json)

    source_points_path = resolve_points_npz(args.object_memory_json, data, args.object_points_npz)
    source_npz = np.load(source_points_path)
    source_points = {key: np.asarray(source_npz[key], dtype=np.int32) for key in source_npz.files}

    uf = UnionFind(sorted(objects_by_id))
    merge_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    for row in pair_rows:
        object_id_a = str(row.get("object_id_a") or "")
        object_id_b = str(row.get("object_id_b") or "")
        if object_id_a not in objects_by_id or object_id_b not in objects_by_id:
            continue
        obj_a = objects_by_id[object_id_a]
        obj_b = objects_by_id[object_id_b]
        semantic_pair = semantic_pair_for_objects(obj_a, obj_b, label_similarity)
        if normalize_label(obj_a.get("canonical_label")) != normalize_label(obj_b.get("canonical_label")):
            ok, reason = should_merge_by_label_similarity(
                row=row,
                obj_a=obj_a,
                obj_b=obj_b,
                semantic_pair=semantic_pair,
                near_synonym_min_point_iou=args.near_synonym_min_point_iou,
                near_synonym_min_bbox_iou=args.near_synonym_min_bbox_iou,
                near_synonym_partial_min_point_containment=args.near_synonym_partial_min_point_containment,
                near_synonym_partial_min_bbox_containment=args.near_synonym_partial_min_bbox_containment,
                near_synonym_partial_max_center_distance=args.near_synonym_partial_max_center_distance,
                parent_child_min_point_iou=args.parent_child_min_point_iou,
                parent_child_min_bbox_iou=args.parent_child_min_bbox_iou,
                parent_child_partial_min_point_containment=args.parent_child_partial_min_point_containment,
                parent_child_partial_min_bbox_containment=args.parent_child_partial_min_bbox_containment,
                parent_child_partial_max_center_distance=args.parent_child_partial_max_center_distance,
            )
        else:
            ok, reason = should_merge_same_label(
                row=row,
                obj_a=obj_a,
                obj_b=obj_b,
                table_min_point_containment=args.table_min_point_containment,
                table_min_bbox_containment=args.table_min_bbox_containment,
                generic_min_point_containment=args.generic_min_point_containment,
                generic_min_bbox_containment=args.generic_min_bbox_containment,
                duplicate_min_point_iou=args.duplicate_min_point_iou,
                chair_min_point_iou=args.chair_min_point_iou,
                chair_min_bbox_iou=args.chair_min_bbox_iou,
                chair_max_center_distance=args.chair_max_center_distance,
            )
        if not ok:
            if reason.endswith("preserve_instances") or reason.startswith("semantic_"):
                skipped_rows.append(
                    {
                        "object_id_a": object_id_a,
                        "object_id_b": object_id_b,
                        "label": label_pair_key(
                            normalize_label(obj_a.get("canonical_label")),
                            normalize_label(obj_b.get("canonical_label")),
                        ),
                        "reason": reason,
                        "source_relation": row.get("relation"),
                        "semantic_relation": (semantic_pair or {}).get("relation"),
                        "semantic_merge_policy": (semantic_pair or {}).get("merge_policy"),
                        "semantic_similarity": (semantic_pair or {}).get("similarity"),
                        "point_iou": row.get("point_iou"),
                        "bbox_iou": row.get("bbox_iou"),
                    }
                )
            continue
        target_id, absorbed_id = choose_target_for_merge(obj_a, obj_b, str(row.get("relation") or ""), semantic_pair)
        uf.union_to(absorbed_id=absorbed_id, target_id=target_id)
        merge_rows.append(
            {
                "action": "merge_into_target",
                "absorbed_object_id": absorbed_id,
                "target_object_id": target_id,
                "label": label_pair_key(
                    normalize_label(obj_a.get("canonical_label")),
                    normalize_label(obj_b.get("canonical_label")),
                ),
                "reason": reason,
                "source_relation": row.get("relation"),
                "semantic_relation": (semantic_pair or {}).get("relation"),
                "semantic_merge_policy": (semantic_pair or {}).get("merge_policy"),
                "semantic_similarity": (semantic_pair or {}).get("similarity"),
                "semantic_reason": (semantic_pair or {}).get("reason"),
                "bbox_iou": row.get("bbox_iou"),
                "bbox_intersection_over_smaller": row.get("bbox_intersection_over_smaller"),
                "point_intersection_over_smaller": row.get("point_intersection_over_smaller"),
                "point_iou": row.get("point_iou"),
                "center_distance": row.get("center_distance"),
            }
        )

    groups: dict[str, list[str]] = {}
    for object_id in sorted(objects_by_id):
        groups.setdefault(uf.find(object_id), []).append(object_id)

    final_objects: list[dict[str, Any]] = []
    final_arrays: dict[str, np.ndarray] = {}
    for root_id, member_ids in sorted(groups.items()):
        final_obj, point_indices = merge_objects(root_id, sorted(member_ids), objects_by_id, source_points)
        final_objects.append(final_obj)
        final_arrays[str(final_obj["object_point_key"])] = point_indices

    output_dir = make_output_dir(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    points_path = output_dir / "final_object_memory_points.npz"
    np.savez_compressed(points_path, **final_arrays)

    label_counts = Counter(normalize_label(obj.get("canonical_label")) for obj in final_objects)
    merged_object_count = sum(max(0, len(obj.get("source_object_ids", [])) - 1) for obj in final_objects)
    summary = {
        "source_object_count": len(objects),
        "selected_object_count": len(selected),
        "final_object_count": len(final_objects),
        "merged_object_count": merged_object_count,
        "merge_action_count": len(merge_rows),
        "final_label_counts": dict(label_counts.most_common()),
        "include_final_statuses": sorted(include_final_statuses),
        "source_object_memory_json": str(args.object_memory_json),
        "source_object_pair_issues_csv": str(args.object_pair_issues_csv),
        "source_label_similarity_json": str(args.label_similarity_json) if args.label_similarity_json else None,
        "label_similarity_pair_count": len(label_similarity),
        "source_object_points_npz": str(source_points_path),
        "final_object_points_npz": str(points_path),
    }
    payload = {
        "format": "final_object_memory_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metadata": {
            "source_object_memory_json": str(args.object_memory_json),
            "source_object_pair_issues_csv": str(args.object_pair_issues_csv),
            "source_label_similarity_json": str(args.label_similarity_json) if args.label_similarity_json else None,
            "source_object_points_npz": str(source_points_path),
            "object_memory_points_npz": str(points_path),
            "merge_pass": "label_similarity_partial_duplicate_v1",
        },
        "summary": summary,
        "objects": final_objects,
    }
    write_json(output_dir / "final_object_memory.json", payload)
    write_merge_report(output_dir / "merge_report.csv", merge_rows)
    write_json(output_dir / "skipped_merge_pairs.json", {"pairs": skipped_rows})

    lines = [
        "Object memory global merge summary",
        "",
        f"source_object_memory_json: {args.object_memory_json}",
        f"object_pair_issues_csv: {args.object_pair_issues_csv}",
        f"label_similarity_json: {args.label_similarity_json}",
        f"label_similarity_pairs: {len(label_similarity)}",
        f"source_objects: {len(objects)}",
        f"selected_objects: {len(selected)}",
        f"final_objects: {len(final_objects)}",
        f"merged_objects: {merged_object_count}",
        f"merge_actions: {len(merge_rows)}",
        f"final_label_counts: {dict(label_counts.most_common())}",
        f"final_object_memory_json: {output_dir / 'final_object_memory.json'}",
        f"final_object_points_npz: {points_path}",
        f"output: {output_dir}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[source_objects] {len(objects)}")
    print(f"[selected_objects] {len(selected)}")
    print(f"[final_objects] {len(final_objects)}")
    print(f"[merged_objects] {merged_object_count}")
    print(f"[merge_actions] {len(merge_rows)}")
    print(f"[final_label_counts] {dict(label_counts.most_common())}")
    print(f"[points_npz] {points_path}")
    print(f"[output] {output_dir}")


if __name__ == "__main__":
    main()
