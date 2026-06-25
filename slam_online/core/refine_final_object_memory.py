#!/usr/bin/env python3
"""
Final display refinement for post-merge object memory.

This pass does not change merge membership or point arrays. It annotates final
objects with display-oriented final_status values, so downstream visualization
can show a cleaner scene-level object set while retaining every object for
review.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "object_memory_final_refine"


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
    return next_available_dir(output_root, f"final_refine_{timestamp}")


def normalize_label(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ")


def label_pair_key(label_a: str, label_b: str) -> str:
    a, b = sorted((normalize_label(label_a), normalize_label(label_b)))
    return f"{a}|{b}"


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def finite_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_csv_raw_set(value: str | None) -> set[str]:
    if value is None or not value.strip():
        return set()
    return {part.strip() for part in value.split(",") if part.strip()}


def load_pair_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
            "relation": str(item.get("relation") or "unknown"),
            "merge_policy": str(item.get("merge_policy") or "unknown"),
            "similarity": finite_float(item.get("similarity")),
            "reason": str(item.get("reason") or ""),
        }
    return pairs


def support_score(obj: dict[str, Any]) -> float:
    frames = finite_int(obj.get("support_frame_count"))
    observations = finite_int(obj.get("support_observation_count"))
    points = finite_int(obj.get("point_count"))
    return frames * 10.0 + observations * 2.0 + min(points / 1000.0, 20.0)


def bbox_size(obj: dict[str, Any]) -> tuple[float, float, float]:
    values = obj.get("bbox_size")
    if not isinstance(values, list) or len(values) != 3:
        return (0.0, 0.0, 0.0)
    return tuple(max(0.0, finite_float(value)) for value in values)  # type: ignore[return-value]


def bbox_volume(obj: dict[str, Any]) -> float:
    volume = finite_float(obj.get("bbox_volume"))
    if volume > 0.0:
        return volume
    sx, sy, sz = bbox_size(obj)
    return sx * sy * sz


def object_labels(obj: dict[str, Any]) -> set[str]:
    labels = {normalize_label(obj.get("canonical_label"))}
    aliases = obj.get("aliases")
    if isinstance(aliases, list):
        labels.update(normalize_label(alias) for alias in aliases)
    return {label for label in labels if label}


def labels_share_evidence(obj_a: dict[str, Any], obj_b: dict[str, Any]) -> bool:
    return bool(object_labels(obj_a) & object_labels(obj_b))


def dimension_ratios(part_obj: dict[str, Any], whole_obj: dict[str, Any]) -> tuple[float, float, float]:
    part = bbox_size(part_obj)
    whole = bbox_size(whole_obj)
    ratios = []
    for part_size, whole_size in zip(part, whole):
        ratios.append(part_size / max(whole_size, 1e-6))
    return tuple(ratios)  # type: ignore[return-value]


def source_to_final_ids(objects: list[dict[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for obj in objects:
        final_id = str(obj.get("object_id") or "")
        if not final_id:
            continue
        out[final_id] = final_id
        source_ids = obj.get("source_object_ids")
        if not isinstance(source_ids, list):
            source_ids = [final_id]
        for source_id in source_ids:
            out[str(source_id)] = final_id
    return out


def better_object(obj_a: dict[str, Any], obj_b: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    score_a = support_score(obj_a)
    score_b = support_score(obj_b)
    if score_a != score_b:
        return (obj_a, obj_b) if score_a > score_b else (obj_b, obj_a)
    points_a = finite_int(obj_a.get("point_count"))
    points_b = finite_int(obj_b.get("point_count"))
    if points_a != points_b:
        return (obj_a, obj_b) if points_a > points_b else (obj_b, obj_a)
    volume_a = finite_float(obj_a.get("bbox_volume"))
    volume_b = finite_float(obj_b.get("bbox_volume"))
    return (obj_a, obj_b) if volume_a >= volume_b else (obj_b, obj_a)


def add_hidden_candidate(
    candidates: dict[str, dict[str, Any]],
    hidden_obj: dict[str, Any],
    kept_obj: dict[str, Any] | None,
    reason: str,
    rank_score: float,
    row: dict[str, str] | None = None,
    semantic_pair: dict[str, Any] | None = None,
) -> None:
    object_id = str(hidden_obj.get("object_id") or "")
    if not object_id:
        return
    item = {
        "object_id": object_id,
        "label": normalize_label(hidden_obj.get("canonical_label")),
        "status": hidden_obj.get("status"),
        "support_frame_count": hidden_obj.get("support_frame_count"),
        "support_observation_count": hidden_obj.get("support_observation_count"),
        "point_count": hidden_obj.get("point_count"),
        "reason": reason,
        "kept_object_id": str(kept_obj.get("object_id")) if kept_obj else None,
        "kept_label": normalize_label(kept_obj.get("canonical_label")) if kept_obj else None,
        "source_relation": row.get("relation") if row else None,
        "bbox_iou": row.get("bbox_iou") if row else None,
        "bbox_intersection_over_smaller": row.get("bbox_intersection_over_smaller") if row else None,
        "point_intersection_over_smaller": row.get("point_intersection_over_smaller") if row else None,
        "point_iou": row.get("point_iou") if row else None,
        "center_distance": row.get("center_distance") if row else None,
        "semantic_relation": semantic_pair.get("relation") if semantic_pair else None,
        "semantic_merge_policy": semantic_pair.get("merge_policy") if semantic_pair else None,
        "_rank_score": rank_score,
    }
    old = candidates.get(object_id)
    if old is None or finite_float(item["_rank_score"]) > finite_float(old.get("_rank_score")):
        candidates[object_id] = item


def should_hide_overlap(
    row: dict[str, str],
    obj_a: dict[str, Any],
    obj_b: dict[str, Any],
    semantic_pair: dict[str, Any] | None,
    partial_min_point_containment: float,
    partial_min_bbox_containment: float,
    partial_min_bbox_iou: float,
    duplicate_min_point_iou: float,
    duplicate_min_bbox_iou: float,
    semantic_duplicate_min_point_iou: float,
    semantic_duplicate_min_bbox_iou: float,
    semantic_duplicate_max_center_distance: float,
    part_min_point_containment: float,
    part_min_bbox_containment: float,
    part_min_bbox_iou: float,
    part_max_point_ratio: float,
    part_max_volume_ratio: float,
    part_max_min_dimension_ratio: float,
    part_max_axis_dimension_ratio: float,
    subobject_min_point_containment: float,
    subobject_max_point_ratio: float,
    subobject_max_center_distance: float,
    part_max_center_distance: float,
    weak_support_ratio: float,
    max_cross_label_partial_frames: int,
) -> tuple[dict[str, Any], dict[str, Any], str] | None:
    relation = str(row.get("relation") or "")
    point_containment = finite_float(row.get("point_intersection_over_smaller"))
    bbox_containment = finite_float(row.get("bbox_intersection_over_smaller"))
    point_iou = finite_float(row.get("point_iou"))
    bbox_iou = finite_float(row.get("bbox_iou"))
    center_distance = finite_float(row.get("center_distance"))

    if str(obj_a.get("status")) != "confirmed" or str(obj_b.get("status")) != "confirmed":
        return None

    stronger, weaker = better_object(obj_a, obj_b)
    stronger_score = max(support_score(stronger), 1.0)
    weaker_score = support_score(weaker)
    score_ratio = weaker_score / stronger_score
    same_label = normalize_label(obj_a.get("canonical_label")) == normalize_label(obj_b.get("canonical_label"))
    semantic_relation = str((semantic_pair or {}).get("relation") or "unknown")
    semantically_close = semantic_relation in {"near_synonym", "parent_child"} or labels_share_evidence(obj_a, obj_b)

    duplicate_like = point_iou >= duplicate_min_point_iou and bbox_iou >= duplicate_min_bbox_iou
    if duplicate_like and score_ratio <= weak_support_ratio:
        return weaker, stronger, "hidden_as_duplicate_lower_support"

    semantic_duplicate_like = (
        semantically_close
        and point_iou >= semantic_duplicate_min_point_iou
        and bbox_iou >= semantic_duplicate_min_bbox_iou
        and center_distance <= semantic_duplicate_max_center_distance
    )
    if semantic_duplicate_like and score_ratio <= max(weak_support_ratio, 0.90):
        return weaker, stronger, "hidden_as_semantic_duplicate"

    partial_like = (
        point_containment >= partial_min_point_containment
        and bbox_containment >= partial_min_bbox_containment
        and bbox_iou >= partial_min_bbox_iou
    )
    if partial_like:
        if same_label and relation in {"possible_partial_whole_same_label", "possible_duplicate_same_label"}:
            return weaker, stronger, "hidden_as_same_label_partial"

        if semantic_relation in {"near_synonym", "parent_child"} and score_ratio <= weak_support_ratio:
            return weaker, stronger, f"hidden_as_{semantic_relation}_partial"

        weak_frames = finite_int(weaker.get("support_frame_count"))
        if weak_frames <= max_cross_label_partial_frames and score_ratio <= weak_support_ratio:
            return weaker, stronger, "hidden_as_low_support_cross_label_partial"

    point_ratio = finite_int(weaker.get("point_count")) / max(float(finite_int(stronger.get("point_count"))), 1.0)
    volume_ratio = bbox_volume(weaker) / max(bbox_volume(stronger), 1e-6)
    dim_ratios = dimension_ratios(weaker, stronger)
    min_dim_ratio = min(dim_ratios)
    axis_part_like = min_dim_ratio <= part_max_min_dimension_ratio or any(
        ratio <= part_max_axis_dimension_ratio for ratio in dim_ratios
    )
    compact_part_like = point_ratio <= part_max_point_ratio or volume_ratio <= part_max_volume_ratio
    thin_part_like = (
        semantically_close
        and point_containment >= part_min_point_containment
        and bbox_containment >= part_min_bbox_containment
        and bbox_iou >= part_min_bbox_iou
        and center_distance <= part_max_center_distance
        and compact_part_like
        and axis_part_like
    )
    low_point_subobject_like = (
        semantically_close
        and point_containment >= subobject_min_point_containment
        and point_ratio <= subobject_max_point_ratio
        and center_distance <= subobject_max_center_distance
    )
    geometric_part_like = thin_part_like or low_point_subobject_like
    if geometric_part_like:
        return weaker, stronger, "hidden_as_geometric_part"

    return None


def write_hidden_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "object_id",
        "label",
        "status",
        "support_frame_count",
        "support_observation_count",
        "point_count",
        "reason",
        "kept_object_id",
        "kept_label",
        "source_relation",
        "semantic_relation",
        "semantic_merge_policy",
        "bbox_iou",
        "bbox_intersection_over_smaller",
        "point_intersection_over_smaller",
        "point_iou",
        "center_distance",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Mark final objects hidden for cleaner display")
    parser.add_argument("--object_memory_json", type=Path, required=True)
    parser.add_argument("--object_pair_issues_csv", type=Path, required=True)
    parser.add_argument("--label_similarity_json", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--hide_non_confirmed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min_confirmed_frames", type=int, default=3)
    parser.add_argument("--partial_min_point_containment", type=float, default=0.85)
    parser.add_argument("--partial_min_bbox_containment", type=float, default=0.70)
    parser.add_argument("--partial_min_bbox_iou", type=float, default=0.10)
    parser.add_argument("--duplicate_min_point_iou", type=float, default=0.80)
    parser.add_argument("--duplicate_min_bbox_iou", type=float, default=0.45)
    parser.add_argument("--semantic_duplicate_min_point_iou", type=float, default=0.55)
    parser.add_argument("--semantic_duplicate_min_bbox_iou", type=float, default=0.70)
    parser.add_argument("--semantic_duplicate_max_center_distance", type=float, default=0.15)
    parser.add_argument("--part_min_point_containment", type=float, default=0.55)
    parser.add_argument("--part_min_bbox_containment", type=float, default=0.25)
    parser.add_argument("--part_min_bbox_iou", type=float, default=0.08)
    parser.add_argument("--part_max_point_ratio", type=float, default=0.35)
    parser.add_argument("--part_max_volume_ratio", type=float, default=0.55)
    parser.add_argument("--part_max_min_dimension_ratio", type=float, default=0.35)
    parser.add_argument("--part_max_axis_dimension_ratio", type=float, default=0.45)
    parser.add_argument("--subobject_min_point_containment", type=float, default=0.80)
    parser.add_argument("--subobject_max_point_ratio", type=float, default=0.25)
    parser.add_argument("--subobject_max_center_distance", type=float, default=0.45)
    parser.add_argument("--part_max_center_distance", type=float, default=0.65)
    parser.add_argument("--weak_support_ratio", type=float, default=0.55)
    parser.add_argument("--max_cross_label_partial_frames", type=int, default=8)
    parser.add_argument("--keep_object_ids", type=str, default="")
    args = parser.parse_args()

    data = read_json(args.object_memory_json)
    objects = data.get("objects") if isinstance(data, dict) else None
    if not isinstance(objects, list):
        raise ValueError(f"Expected objects list in {args.object_memory_json}")

    keep_object_ids = parse_csv_raw_set(args.keep_object_ids)
    objects = [dict(obj) for obj in objects]
    objects_by_id = {str(obj.get("object_id")): obj for obj in objects}
    source_to_final = source_to_final_ids(objects)
    label_similarity = load_label_similarity(args.label_similarity_json)
    hidden_candidates: dict[str, dict[str, Any]] = {}

    if args.hide_non_confirmed:
        for obj in objects:
            if str(obj.get("status")) != "confirmed":
                add_hidden_candidate(
                    candidates=hidden_candidates,
                    hidden_obj=obj,
                    kept_obj=None,
                    reason="hidden_non_confirmed",
                    rank_score=10_000.0,
                )

    if args.min_confirmed_frames > 0:
        for obj in objects:
            if str(obj.get("status")) == "confirmed" and finite_int(obj.get("support_frame_count")) < args.min_confirmed_frames:
                add_hidden_candidate(
                    candidates=hidden_candidates,
                    hidden_obj=obj,
                    kept_obj=None,
                    reason="hidden_low_confirmed_support",
                    rank_score=9_000.0 - support_score(obj),
                )

    for row in load_pair_rows(args.object_pair_issues_csv):
        final_id_a = source_to_final.get(str(row.get("object_id_a") or ""))
        final_id_b = source_to_final.get(str(row.get("object_id_b") or ""))
        if not final_id_a or not final_id_b or final_id_a == final_id_b:
            continue
        obj_a = objects_by_id.get(final_id_a)
        obj_b = objects_by_id.get(final_id_b)
        if obj_a is None or obj_b is None:
            continue
        semantic_pair = label_similarity.get(label_pair_key(obj_a.get("canonical_label"), obj_b.get("canonical_label")))
        decision = should_hide_overlap(
            row=row,
            obj_a=obj_a,
            obj_b=obj_b,
            semantic_pair=semantic_pair,
            partial_min_point_containment=args.partial_min_point_containment,
            partial_min_bbox_containment=args.partial_min_bbox_containment,
            partial_min_bbox_iou=args.partial_min_bbox_iou,
            duplicate_min_point_iou=args.duplicate_min_point_iou,
            duplicate_min_bbox_iou=args.duplicate_min_bbox_iou,
            semantic_duplicate_min_point_iou=args.semantic_duplicate_min_point_iou,
            semantic_duplicate_min_bbox_iou=args.semantic_duplicate_min_bbox_iou,
            semantic_duplicate_max_center_distance=args.semantic_duplicate_max_center_distance,
            part_min_point_containment=args.part_min_point_containment,
            part_min_bbox_containment=args.part_min_bbox_containment,
            part_min_bbox_iou=args.part_min_bbox_iou,
            part_max_point_ratio=args.part_max_point_ratio,
            part_max_volume_ratio=args.part_max_volume_ratio,
            part_max_min_dimension_ratio=args.part_max_min_dimension_ratio,
            part_max_axis_dimension_ratio=args.part_max_axis_dimension_ratio,
            subobject_min_point_containment=args.subobject_min_point_containment,
            subobject_max_point_ratio=args.subobject_max_point_ratio,
            subobject_max_center_distance=args.subobject_max_center_distance,
            part_max_center_distance=args.part_max_center_distance,
            weak_support_ratio=args.weak_support_ratio,
            max_cross_label_partial_frames=args.max_cross_label_partial_frames,
        )
        if decision is None:
            continue
        hidden_obj, kept_obj, reason = decision
        add_hidden_candidate(
            candidates=hidden_candidates,
            hidden_obj=hidden_obj,
            kept_obj=kept_obj,
            reason=reason,
            rank_score=support_score(kept_obj) + finite_float(row.get("point_intersection_over_smaller")) * 100.0,
            row=row,
            semantic_pair=semantic_pair,
        )

    for object_id in keep_object_ids:
        hidden_candidates.pop(object_id, None)

    hidden_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in sorted(hidden_candidates.values(), key=lambda item: str(item.get("object_id")))
    ]
    hidden_ids = {str(row["object_id"]) for row in hidden_rows}

    refined_objects: list[dict[str, Any]] = []
    for obj in objects:
        obj = dict(obj)
        object_id = str(obj.get("object_id") or "")
        obj["final_refine_status"] = "hidden" if object_id in hidden_ids else "visible"
        obj["final_status"] = "hidden_by_final_refine" if object_id in hidden_ids else "kept"
        obj["final_refine_reason"] = None
        obj["final_refine_kept_object_id"] = None
        if object_id in hidden_candidates:
            hidden = hidden_candidates[object_id]
            obj["final_refine_reason"] = hidden.get("reason")
            obj["final_refine_kept_object_id"] = hidden.get("kept_object_id")
        refined_objects.append(obj)

    visible_objects = [obj for obj in refined_objects if obj.get("final_status") == "kept"]
    visible_confirmed_objects = [obj for obj in visible_objects if str(obj.get("status")) == "confirmed"]
    final_counts = Counter(str(obj.get("final_status")) for obj in refined_objects)
    visible_label_counts = Counter(normalize_label(obj.get("canonical_label")) for obj in visible_confirmed_objects)
    hidden_reason_counts = Counter(str(row.get("reason")) for row in hidden_rows)

    output_dir = make_output_dir(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = dict(data.get("metadata") if isinstance(data.get("metadata"), dict) else {})
    metadata.update(
        {
            "source_object_memory_json": str(args.object_memory_json),
            "source_object_pair_issues_csv": str(args.object_pair_issues_csv),
            "source_label_similarity_json": str(args.label_similarity_json) if args.label_similarity_json else None,
            "final_refine_pass": "display_precision_v1",
        }
    )
    summary = dict(data.get("summary") if isinstance(data.get("summary"), dict) else {})
    summary.update(
        {
            "final_refine_status_counts": dict(final_counts.most_common()),
            "visible_confirmed_object_count": len(visible_confirmed_objects),
            "hidden_object_count": len(hidden_rows),
            "hidden_reason_counts": dict(hidden_reason_counts.most_common()),
            "visible_confirmed_label_counts": dict(visible_label_counts.most_common()),
            "final_refine_parameters": {
                "hide_non_confirmed": args.hide_non_confirmed,
                "min_confirmed_frames": args.min_confirmed_frames,
                "partial_min_point_containment": args.partial_min_point_containment,
                "partial_min_bbox_containment": args.partial_min_bbox_containment,
                "partial_min_bbox_iou": args.partial_min_bbox_iou,
                "duplicate_min_point_iou": args.duplicate_min_point_iou,
                "duplicate_min_bbox_iou": args.duplicate_min_bbox_iou,
                "semantic_duplicate_min_point_iou": args.semantic_duplicate_min_point_iou,
                "semantic_duplicate_min_bbox_iou": args.semantic_duplicate_min_bbox_iou,
                "semantic_duplicate_max_center_distance": args.semantic_duplicate_max_center_distance,
                "part_min_point_containment": args.part_min_point_containment,
                "part_min_bbox_containment": args.part_min_bbox_containment,
                "part_min_bbox_iou": args.part_min_bbox_iou,
                "part_max_point_ratio": args.part_max_point_ratio,
                "part_max_volume_ratio": args.part_max_volume_ratio,
                "part_max_min_dimension_ratio": args.part_max_min_dimension_ratio,
                "part_max_axis_dimension_ratio": args.part_max_axis_dimension_ratio,
                "subobject_min_point_containment": args.subobject_min_point_containment,
                "subobject_max_point_ratio": args.subobject_max_point_ratio,
                "subobject_max_center_distance": args.subobject_max_center_distance,
                "part_max_center_distance": args.part_max_center_distance,
                "weak_support_ratio": args.weak_support_ratio,
                "max_cross_label_partial_frames": args.max_cross_label_partial_frames,
                "keep_object_ids": sorted(keep_object_ids),
            },
        }
    )
    payload = dict(data)
    payload["format"] = "final_refined_object_memory_v1"
    payload["created_at"] = datetime.now().isoformat(timespec="seconds")
    payload["metadata"] = metadata
    payload["summary"] = summary
    payload["objects"] = refined_objects

    write_json(output_dir / "final_refined_object_memory.json", payload)
    write_json(output_dir / "hidden_final_objects.json", {"objects": hidden_rows})
    write_hidden_csv(output_dir / "hidden_final_objects.csv", hidden_rows)

    lines = [
        "Final object memory display refine summary",
        "",
        f"object_memory_json: {args.object_memory_json}",
        f"object_pair_issues_csv: {args.object_pair_issues_csv}",
        f"label_similarity_json: {args.label_similarity_json}",
        f"objects_total: {len(refined_objects)}",
        f"visible_confirmed_objects: {len(visible_confirmed_objects)}",
        f"hidden_objects: {len(hidden_rows)}",
        f"hidden_reason_counts: {dict(hidden_reason_counts.most_common())}",
        f"visible_confirmed_label_counts: {dict(visible_label_counts.most_common())}",
        f"final_refined_object_memory_json: {output_dir / 'final_refined_object_memory.json'}",
        f"hidden_final_objects_json: {output_dir / 'hidden_final_objects.json'}",
        f"output: {output_dir}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[objects_total] {len(refined_objects)}")
    print(f"[visible_confirmed_objects] {len(visible_confirmed_objects)}")
    print(f"[hidden_objects] {len(hidden_rows)}")
    print(f"[hidden_reason_counts] {dict(hidden_reason_counts.most_common())}")
    print(f"[visible_confirmed_label_counts] {dict(visible_label_counts.most_common())}")
    print(f"[output] {output_dir}")


if __name__ == "__main__":
    main()
