#!/usr/bin/env python3
"""
Audit object_memory_update.py outputs before global refinement.

This script does not change object memory. It produces object-level and pairwise
diagnostics to guide the next cleanup step:

  - low evidence objects
  - label conflicts in the same 3D region
  - possible partial/whole duplicates
  - unusually large or sparse boxes for a label
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "object_memory_quality_audit"


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
    return next_available_dir(output_root, f"audit_{timestamp}")


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


def normalize_label(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ")


def as_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (3,):
        return np.zeros((3,), dtype=np.float64)
    arr[~np.isfinite(arr)] = 0.0
    return arr


def quantile_summary(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"count": 0, "min": None, "p25": None, "p50": None, "p75": None, "p90": None, "max": None, "mean": None}
    q = np.percentile(arr, [25, 50, 75, 90])
    return {
        "count": int(len(arr)),
        "min": round(float(arr.min()), 6),
        "p25": round(float(q[0]), 6),
        "p50": round(float(q[1]), 6),
        "p75": round(float(q[2]), 6),
        "p90": round(float(q[3]), 6),
        "max": round(float(arr.max()), 6),
        "mean": round(float(arr.mean()), 6),
    }


def bbox_metrics(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> dict[str, float]:
    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    inter_size = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = float(np.prod(inter_size))
    a_size = np.maximum(a_max - a_min, 0.0)
    b_size = np.maximum(b_max - b_min, 0.0)
    a_vol = float(np.prod(a_size))
    b_vol = float(np.prod(b_size))
    union = a_vol + b_vol - inter_vol
    smaller = max(min(a_vol, b_vol), 1e-12)
    larger = max(max(a_vol, b_vol), 1e-12)
    return {
        "bbox_iou": inter_vol / union if union > 0.0 else 0.0,
        "bbox_intersection_over_smaller": inter_vol / smaller,
        "bbox_smaller_over_larger": smaller / larger,
        "bbox_intersection_volume": inter_vol,
    }


def point_overlap(a_points: np.ndarray | None, b_points: np.ndarray | None) -> dict[str, float | int | None]:
    if a_points is None or b_points is None or len(a_points) == 0 or len(b_points) == 0:
        return {
            "point_intersection_count": None,
            "point_intersection_over_smaller": None,
            "point_iou": None,
        }
    a_unique = np.unique(a_points.astype(np.int64))
    b_unique = np.unique(b_points.astype(np.int64))
    inter = np.intersect1d(a_unique, b_unique, assume_unique=True)
    smaller = max(min(len(a_unique), len(b_unique)), 1)
    union = max(len(a_unique) + len(b_unique) - len(inter), 1)
    return {
        "point_intersection_count": int(len(inter)),
        "point_intersection_over_smaller": round(float(len(inter) / smaller), 6),
        "point_iou": round(float(len(inter) / union), 6),
    }


def resolve_points_npz(object_memory_json: Path, data: dict[str, Any], explicit_path: Path | None) -> Path | None:
    if explicit_path is not None:
        return explicit_path
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    path = metadata.get("object_memory_points_npz")
    if path:
        return Path(str(path))
    fallback = object_memory_json.parent / "object_memory_points.npz"
    return fallback if fallback.exists() else None


def load_object_points(path: Path | None) -> dict[str, np.ndarray]:
    if path is None or not path.exists():
        return {}
    npz = np.load(path)
    return {key: np.asarray(npz[key], dtype=np.int64) for key in npz.files}


def label_stats(objects: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for obj in objects:
        grouped[normalize_label(obj.get("canonical_label"))].append(obj)
    stats = {}
    for label, items in grouped.items():
        volumes = [finite_float(obj.get("bbox_volume")) for obj in items]
        point_counts = [finite_float(obj.get("point_count")) for obj in items]
        frame_counts = [finite_float(obj.get("support_frame_count")) for obj in items]
        stats[label] = {
            "object_count": len(items),
            "bbox_volume": quantile_summary(volumes),
            "point_count": quantile_summary(point_counts),
            "support_frame_count": quantile_summary(frame_counts),
        }
    return stats


def object_quality_row(
    obj: dict[str, Any],
    label_baseline: dict[str, dict[str, Any]],
    watch_labels: set[str],
    min_support_frames: int,
    min_support_observations: int,
    low_box_score: float,
    low_mask_score: float,
    large_volume_multiplier: float,
) -> dict[str, Any]:
    label = normalize_label(obj.get("canonical_label"))
    support_frames = finite_int(obj.get("support_frame_count"))
    support_observations = finite_int(obj.get("support_observation_count"))
    point_count = finite_int(obj.get("point_count"))
    bbox_volume = finite_float(obj.get("bbox_volume"))
    bbox_size = as_array(obj.get("bbox_size"))
    bbox_diag = float(np.linalg.norm(np.maximum(bbox_size, 0.0)))
    nonzero_dims = bbox_size[bbox_size > 1e-9]
    aspect_ratio = float(nonzero_dims.max() / nonzero_dims.min()) if len(nonzero_dims) else 0.0
    mean_box_score = finite_float(obj.get("mean_box_score"))
    mean_mask_score = finite_float(obj.get("mean_mask_score"))
    label_counts = obj.get("label_counts") if isinstance(obj.get("label_counts"), dict) else {}
    top_label_count = max((finite_int(value) for value in label_counts.values()), default=0)
    label_purity = float(top_label_count / max(support_observations, 1))
    baseline_volume_p75 = label_baseline.get(label, {}).get("bbox_volume", {}).get("p75")
    volume_ratio_to_label_p75 = None
    if baseline_volume_p75 not in (None, 0):
        volume_ratio_to_label_p75 = float(bbox_volume / max(float(baseline_volume_p75), 1e-12))

    flags = []
    if label in watch_labels:
        flags.append("watch_label")
    if support_frames < min_support_frames or support_observations < min_support_observations:
        flags.append("low_support")
    if mean_box_score > 0.0 and mean_box_score < low_box_score:
        flags.append("low_box_score")
    if mean_mask_score > 0.0 and mean_mask_score < low_mask_score:
        flags.append("low_mask_score")
    if label_purity < 0.7:
        flags.append("mixed_label_evidence")
    if volume_ratio_to_label_p75 is not None and volume_ratio_to_label_p75 >= large_volume_multiplier:
        flags.append("large_for_label")
    if aspect_ratio >= 8.0:
        flags.append("elongated_bbox")
    if point_count < 500:
        flags.append("low_point_count")

    severity = 0
    severity += 3 if "watch_label" in flags else 0
    severity += 2 if "low_support" in flags else 0
    severity += 2 if "large_for_label" in flags else 0
    severity += 2 if "mixed_label_evidence" in flags else 0
    severity += 1 if "low_box_score" in flags else 0
    severity += 1 if "low_mask_score" in flags else 0
    severity += 1 if "elongated_bbox" in flags else 0
    severity += 1 if "low_point_count" in flags else 0

    return {
        "object_id": obj.get("object_id"),
        "status": obj.get("status"),
        "canonical_label": label,
        "aliases": obj.get("aliases", []),
        "flags": flags,
        "severity": severity,
        "support_frame_count": support_frames,
        "support_observation_count": support_observations,
        "point_count": point_count,
        "bbox_volume": round(bbox_volume, 9),
        "bbox_size": [round(float(x), 6) for x in bbox_size.tolist()],
        "bbox_diag": round(bbox_diag, 6),
        "bbox_aspect_ratio": round(aspect_ratio, 6),
        "mean_box_score": round(mean_box_score, 6),
        "mean_mask_score": round(mean_mask_score, 6),
        "mean_hit_ratio": round(finite_float(obj.get("mean_hit_ratio")), 8),
        "label_purity": round(label_purity, 6),
        "volume_ratio_to_label_p75": round(volume_ratio_to_label_p75, 6) if volume_ratio_to_label_p75 is not None else None,
        "center_3d": obj.get("center_3d"),
        "first_seen_frame": obj.get("first_seen_frame"),
        "last_seen_frame": obj.get("last_seen_frame"),
        "object_point_key": obj.get("object_point_key"),
    }


def pair_relation(
    obj_a: dict[str, Any],
    obj_b: dict[str, Any],
    points_by_key: dict[str, np.ndarray],
    min_bbox_iou: float,
    min_containment: float,
    min_point_containment: float,
) -> dict[str, Any] | None:
    label_a = normalize_label(obj_a.get("canonical_label"))
    label_b = normalize_label(obj_b.get("canonical_label"))
    a_min = as_array(obj_a.get("bbox_min"))
    a_max = as_array(obj_a.get("bbox_max"))
    b_min = as_array(obj_b.get("bbox_min"))
    b_max = as_array(obj_b.get("bbox_max"))
    bbox = bbox_metrics(a_min, a_max, b_min, b_max)

    a_key = str(obj_a.get("object_point_key") or "")
    b_key = str(obj_b.get("object_point_key") or "")
    points = point_overlap(points_by_key.get(a_key), points_by_key.get(b_key))
    point_containment = points.get("point_intersection_over_smaller")
    point_containment_value = float(point_containment) if point_containment is not None else 0.0

    same_label = label_a == label_b
    high_bbox_overlap = bbox["bbox_iou"] >= min_bbox_iou or bbox["bbox_intersection_over_smaller"] >= min_containment
    high_point_overlap = point_containment is not None and point_containment_value >= min_point_containment

    relation = None
    reason = None
    if same_label and (high_bbox_overlap or high_point_overlap):
        if bbox["bbox_intersection_over_smaller"] >= min_containment and bbox["bbox_smaller_over_larger"] <= 0.65:
            relation = "possible_partial_whole_same_label"
            reason = "same label with one bbox mostly contained in the other"
        else:
            relation = "possible_duplicate_same_label"
            reason = "same label with strong spatial overlap"
    elif not same_label and (high_bbox_overlap or high_point_overlap):
        relation = "label_conflict_same_region"
        reason = "different labels share strong 3D evidence"

    if relation is None:
        return None

    center_a = as_array(obj_a.get("center_3d"))
    center_b = as_array(obj_b.get("center_3d"))
    return {
        "relation": relation,
        "reason": reason,
        "object_id_a": obj_a.get("object_id"),
        "label_a": label_a,
        "status_a": obj_a.get("status"),
        "support_frames_a": obj_a.get("support_frame_count"),
        "observations_a": obj_a.get("support_observation_count"),
        "bbox_volume_a": obj_a.get("bbox_volume"),
        "object_id_b": obj_b.get("object_id"),
        "label_b": label_b,
        "status_b": obj_b.get("status"),
        "support_frames_b": obj_b.get("support_frame_count"),
        "observations_b": obj_b.get("support_observation_count"),
        "bbox_volume_b": obj_b.get("bbox_volume"),
        "center_distance": round(float(np.linalg.norm(center_a - center_b)), 6),
        **{key: round(float(value), 6) for key, value in bbox.items()},
        **points,
    }


def write_object_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "object_id",
        "status",
        "canonical_label",
        "flags",
        "severity",
        "support_frame_count",
        "support_observation_count",
        "point_count",
        "bbox_volume",
        "bbox_size",
        "bbox_diag",
        "bbox_aspect_ratio",
        "mean_box_score",
        "mean_mask_score",
        "mean_hit_ratio",
        "label_purity",
        "volume_ratio_to_label_p75",
        "center_3d",
        "first_seen_frame",
        "last_seen_frame",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), ensure_ascii=False)
                    if isinstance(row.get(key), (list, dict))
                    else row.get(key)
                    for key in fieldnames
                }
            )


def write_pair_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "relation",
        "reason",
        "object_id_a",
        "label_a",
        "object_id_b",
        "label_b",
        "center_distance",
        "bbox_iou",
        "bbox_intersection_over_smaller",
        "bbox_smaller_over_larger",
        "point_intersection_count",
        "point_intersection_over_smaller",
        "point_iou",
        "bbox_volume_a",
        "bbox_volume_b",
        "support_frames_a",
        "support_frames_b",
        "observations_a",
        "observations_b",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def parse_csv_set(value: str) -> set[str]:
    return {normalize_label(part) for part in value.split(",") if part.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit object memory quality before global refinement")
    parser.add_argument("--object_memory_json", type=Path, required=True)
    parser.add_argument("--object_points_npz", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--statuses", type=str, default="confirmed,candidate")
    parser.add_argument("--watch_labels", type=str, default="", help="Comma-separated labels to always flag for inspection")
    parser.add_argument("--min_support_frames", type=int, default=2)
    parser.add_argument("--min_support_observations", type=int, default=3)
    parser.add_argument("--low_box_score", type=float, default=0.35)
    parser.add_argument("--low_mask_score", type=float, default=0.0)
    parser.add_argument("--large_volume_multiplier", type=float, default=2.5)
    parser.add_argument("--min_bbox_iou", type=float, default=0.25)
    parser.add_argument("--min_bbox_containment", type=float, default=0.75)
    parser.add_argument("--min_point_containment", type=float, default=0.35)
    parser.add_argument("--top_k", type=int, default=30)
    args = parser.parse_args()

    data = read_json(args.object_memory_json)
    objects = data.get("objects") if isinstance(data, dict) else None
    if not isinstance(objects, list):
        raise ValueError(f"Expected objects list in {args.object_memory_json}")

    statuses = parse_csv_set(args.statuses)
    watch_labels = parse_csv_set(args.watch_labels)
    selected = [obj for obj in objects if normalize_label(obj.get("status")) in statuses]
    baselines = label_stats(selected)
    points_path = resolve_points_npz(args.object_memory_json, data, args.object_points_npz)
    points_by_key = load_object_points(points_path)

    output_dir = make_output_dir(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    object_rows = [
        object_quality_row(
            obj=obj,
            label_baseline=baselines,
            watch_labels=watch_labels,
            min_support_frames=args.min_support_frames,
            min_support_observations=args.min_support_observations,
            low_box_score=args.low_box_score,
            low_mask_score=args.low_mask_score,
            large_volume_multiplier=args.large_volume_multiplier,
        )
        for obj in selected
    ]
    object_rows.sort(key=lambda row: (-int(row["severity"]), str(row["canonical_label"]), str(row["object_id"])))
    flagged_objects = [row for row in object_rows if int(row["severity"]) > 0]

    pair_rows = []
    for i, obj_a in enumerate(selected):
        for obj_b in selected[i + 1 :]:
            relation = pair_relation(
                obj_a=obj_a,
                obj_b=obj_b,
                points_by_key=points_by_key,
                min_bbox_iou=args.min_bbox_iou,
                min_containment=args.min_bbox_containment,
                min_point_containment=args.min_point_containment,
            )
            if relation is not None:
                pair_rows.append(relation)
    pair_rows.sort(
        key=lambda row: (
            str(row["relation"]),
            -finite_float(row.get("point_intersection_over_smaller")),
            -finite_float(row.get("bbox_intersection_over_smaller")),
            str(row["object_id_a"]),
            str(row["object_id_b"]),
        )
    )

    label_counts = Counter(row["canonical_label"] for row in object_rows)
    flag_counts = Counter(flag for row in object_rows for flag in row["flags"])
    relation_counts = Counter(row["relation"] for row in pair_rows)
    summary = {
        "object_memory_json": str(args.object_memory_json),
        "object_points_npz": str(points_path) if points_path else None,
        "objects_total": len(objects),
        "objects_selected": len(selected),
        "flagged_object_count": len(flagged_objects),
        "pair_issue_count": len(pair_rows),
        "label_counts": dict(label_counts.most_common()),
        "flag_counts": dict(flag_counts.most_common()),
        "relation_counts": dict(relation_counts.most_common()),
        "label_baselines": baselines,
        "parameters": {
            "statuses": sorted(statuses),
            "watch_labels": sorted(watch_labels),
            "min_support_frames": args.min_support_frames,
            "min_support_observations": args.min_support_observations,
            "low_box_score": args.low_box_score,
            "low_mask_score": args.low_mask_score,
            "large_volume_multiplier": args.large_volume_multiplier,
            "min_bbox_iou": args.min_bbox_iou,
            "min_bbox_containment": args.min_bbox_containment,
            "min_point_containment": args.min_point_containment,
        },
    }

    payload = {
        "format": "object_memory_quality_audit_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "objects": object_rows,
        "flagged_objects": flagged_objects,
        "pairs": pair_rows,
    }
    write_json(output_dir / "object_memory_quality_audit.json", payload)
    write_object_rows_csv(output_dir / "object_quality.csv", object_rows)
    write_object_rows_csv(output_dir / "flagged_objects.csv", flagged_objects)
    write_pair_rows_csv(output_dir / "object_pair_issues.csv", pair_rows)

    top_flagged = flagged_objects[: args.top_k]
    top_pairs = pair_rows[: args.top_k]
    lines = [
        "Object memory quality audit summary",
        "",
        f"object_memory_json: {args.object_memory_json}",
        f"object_points_npz: {points_path}",
        f"objects_total: {len(objects)}",
        f"objects_selected: {len(selected)}",
        f"flagged_objects: {len(flagged_objects)}",
        f"pair_issues: {len(pair_rows)}",
        f"flag_counts: {dict(flag_counts.most_common())}",
        f"relation_counts: {dict(relation_counts.most_common())}",
        "",
        "Top flagged objects:",
    ]
    for row in top_flagged:
        lines.append(
            f"  {row['object_id']} {row['canonical_label']} severity={row['severity']} "
            f"flags={','.join(row['flags'])} frames={row['support_frame_count']} "
            f"obs={row['support_observation_count']} volume={row['bbox_volume']}"
        )
    lines.append("")
    lines.append("Top pair issues:")
    for row in top_pairs:
        lines.append(
            f"  {row['relation']}: {row['object_id_a']}({row['label_a']}) <-> "
            f"{row['object_id_b']}({row['label_b']}) "
            f"bbox_contain={row['bbox_intersection_over_smaller']} "
            f"point_contain={row['point_intersection_over_smaller']}"
        )
    lines.append("")
    lines.append(f"output: {output_dir}")
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[objects_total] {len(objects)}")
    print(f"[objects_selected] {len(selected)}")
    print(f"[flagged_objects] {len(flagged_objects)}")
    print(f"[pair_issues] {len(pair_rows)}")
    print(f"[flag_counts] {dict(flag_counts.most_common())}")
    print(f"[relation_counts] {dict(relation_counts.most_common())}")
    print(f"[output] {output_dir}")


if __name__ == "__main__":
    main()
