#!/usr/bin/env python3
"""
Conservative global refinement for object memory label conflicts.

This pass does not delete objects. It adds final_status/refine metadata so that
downstream visualization can hide objects that are likely wrong labels for an
already stronger object in the same 3D region.

By default this is label-agnostic: it does not need a dataset-specific list of
wrong labels. It suppresses the weaker side only when two different labels share
near-duplicate 3D evidence and the labels are not treated as compatible aliases
or fine-grained variants.
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
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "object_memory_global_refine"


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
    return next_available_dir(output_root, f"refine_{timestamp}")


def normalize_label(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ")


def parse_csv_set(value: str) -> set[str]:
    return {normalize_label(part) for part in value.split(",") if part.strip()}


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


def load_pair_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def support_score(obj: dict[str, Any]) -> float:
    frames = finite_int(obj.get("support_frame_count"))
    observations = finite_int(obj.get("support_observation_count"))
    points = finite_int(obj.get("point_count"))
    box_score = finite_float(obj.get("mean_box_score"))
    return frames * 10.0 + observations * 2.0 + min(points / 1000.0, 20.0) + box_score


def label_tokens(label: str) -> set[str]:
    tokens = set()
    for token in normalize_label(label).split():
        if not token:
            continue
        tokens.add(token)
        if token.endswith("s") and len(token) > 3:
            tokens.add(token[:-1])
    return tokens


def label_family(label: str) -> str | None:
    label = normalize_label(label)
    if "chair" in label or label == "stool":
        return "chair"
    if label in {"couch", "sofa", "loveseat", "arm sofa"}:
        return "sofa"
    if "table" in label or "desk" in label:
        return "table"
    if "computer" in label or "laptop" in label or "monitor" in label:
        return "computer"
    if label in {"person", "woman", "man", "human"}:
        return "person"
    return None


def labels_compatible(label_a: str, label_b: str) -> bool:
    a = normalize_label(label_a)
    b = normalize_label(label_b)
    if a == b:
        return True
    if a in b or b in a:
        return True
    if label_tokens(a) & label_tokens(b):
        return True
    family_a = label_family(a)
    family_b = label_family(b)
    return family_a is not None and family_a == family_b


def choose_suppression_side(
    row: dict[str, str],
    objects_by_id: dict[str, dict[str, Any]],
    suppress_labels: set[str],
    min_point_containment: float,
    min_bbox_containment: float,
    min_point_iou: float,
    min_bbox_iou: float,
    min_bbox_iou_with_containment: float,
    max_center_distance: float,
    min_support_ratio: float,
    max_suppressed_frames_without_ratio: int,
    min_support_score_margin: float,
    auto_strong_support_ratio: float,
    protect_labels: set[str],
) -> tuple[str, str, str] | None:
    if row.get("relation") != "label_conflict_same_region":
        return None

    point_containment = finite_float(row.get("point_intersection_over_smaller"), default=0.0)
    bbox_containment = finite_float(row.get("bbox_intersection_over_smaller"), default=0.0)
    point_iou = finite_float(row.get("point_iou"), default=0.0)
    bbox_iou = finite_float(row.get("bbox_iou"), default=0.0)
    center_distance = finite_float(row.get("center_distance"), default=1e9)
    duplicate_like = point_iou >= min_point_iou and (
        bbox_iou >= min_bbox_iou
        or center_distance <= max_center_distance
        or point_containment >= min_point_containment
        or bbox_containment >= min_bbox_containment
    )
    if not duplicate_like:
        return None

    object_id_a = str(row.get("object_id_a") or "")
    object_id_b = str(row.get("object_id_b") or "")
    obj_a = objects_by_id.get(object_id_a)
    obj_b = objects_by_id.get(object_id_b)
    if obj_a is None or obj_b is None:
        return None

    label_a = normalize_label(obj_a.get("canonical_label"))
    label_b = normalize_label(obj_b.get("canonical_label"))
    if labels_compatible(label_a, label_b):
        return None

    if suppress_labels:
        a_suppressible = label_a in suppress_labels
        b_suppressible = label_b in suppress_labels
        if a_suppressible == b_suppressible:
            return None
        suppressed = obj_a if a_suppressible else obj_b
        keeper = obj_b if a_suppressible else obj_a
    else:
        score_a = support_score(obj_a)
        score_b = support_score(obj_b)
        if abs(score_a - score_b) < min_support_score_margin:
            return None
        suppressed = obj_a if score_a < score_b else obj_b
        keeper = obj_b if score_a < score_b else obj_a
        if normalize_label(suppressed.get("canonical_label")) in protect_labels:
            return None
    suppressed_id = str(suppressed.get("object_id"))
    keeper_id = str(keeper.get("object_id"))

    suppressed_frames = finite_int(suppressed.get("support_frame_count"))
    keeper_frames = finite_int(keeper.get("support_frame_count"))
    suppressed_observations = finite_int(suppressed.get("support_observation_count"))
    keeper_observations = finite_int(keeper.get("support_observation_count"))
    weak_suppressed_ok = suppressed_frames <= max_suppressed_frames_without_ratio
    ratio_ok = (
        keeper_frames >= max(1.0, suppressed_frames * min_support_ratio)
        or keeper_observations >= max(1.0, suppressed_observations * min_support_ratio)
    )
    strong_ratio_ok = (
        keeper_frames >= max(1.0, suppressed_frames * auto_strong_support_ratio)
        or keeper_observations >= max(1.0, suppressed_observations * auto_strong_support_ratio)
    )
    if not suppress_labels and not weak_suppressed_ok and not strong_ratio_ok:
        return None
    if not ratio_ok and not weak_suppressed_ok:
        return None

    reason = "weaker_auto_label_conflict_same_region" if not suppress_labels else "weaker_suppress_label_conflict_same_region"
    return suppressed_id, keeper_id, reason


def write_report_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "action",
        "suppressed_object_id",
        "suppressed_label",
        "suppressed_status",
        "suppressed_support_frames",
        "suppressed_support_observations",
        "kept_object_id",
        "kept_label",
        "kept_status",
        "kept_support_frames",
        "kept_support_observations",
        "reason",
        "source_relation",
        "bbox_iou",
        "bbox_intersection_over_smaller",
        "point_intersection_over_smaller",
        "point_iou",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Conservatively mark weak label-conflict objects in object memory")
    parser.add_argument("--object_memory_json", type=Path, required=True)
    parser.add_argument("--object_pair_issues_csv", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--suppress_labels", type=str, default="", help="Optional comma-separated labels eligible for suppression")
    parser.add_argument("--min_point_containment", type=float, default=0.90)
    parser.add_argument("--min_bbox_containment", type=float, default=0.95)
    parser.add_argument("--min_point_iou", type=float, default=0.30)
    parser.add_argument("--min_bbox_iou", type=float, default=0.10)
    parser.add_argument("--min_bbox_iou_with_containment", type=float, default=0.10)
    parser.add_argument("--max_center_distance", type=float, default=0.60)
    parser.add_argument("--min_support_ratio", type=float, default=2.0)
    parser.add_argument("--max_suppressed_frames_without_ratio", type=int, default=3)
    parser.add_argument("--min_support_score_margin", type=float, default=10.0)
    parser.add_argument("--auto_strong_support_ratio", type=float, default=4.0)
    parser.add_argument(
        "--protect_labels",
        type=str,
        default="",
        help="Comma-separated labels never automatically suppressed in label-agnostic mode",
    )
    args = parser.parse_args()

    data = read_json(args.object_memory_json)
    objects = data.get("objects") if isinstance(data, dict) else None
    if not isinstance(objects, list):
        raise ValueError(f"Expected objects list in {args.object_memory_json}")

    suppress_labels = parse_csv_set(args.suppress_labels)
    protect_labels = parse_csv_set(args.protect_labels)
    objects_by_id = {str(obj.get("object_id")): obj for obj in objects}
    pair_rows = load_pair_rows(args.object_pair_issues_csv)

    candidates: dict[str, dict[str, Any]] = {}
    for row in pair_rows:
        decision = choose_suppression_side(
            row=row,
            objects_by_id=objects_by_id,
            suppress_labels=suppress_labels,
            min_point_containment=args.min_point_containment,
            min_bbox_containment=args.min_bbox_containment,
            min_point_iou=args.min_point_iou,
            min_bbox_iou=args.min_bbox_iou,
            min_bbox_iou_with_containment=args.min_bbox_iou_with_containment,
            max_center_distance=args.max_center_distance,
            min_support_ratio=args.min_support_ratio,
            max_suppressed_frames_without_ratio=args.max_suppressed_frames_without_ratio,
            min_support_score_margin=args.min_support_score_margin,
            auto_strong_support_ratio=args.auto_strong_support_ratio,
            protect_labels=protect_labels,
        )
        if decision is None:
            continue
        suppressed_id, keeper_id, reason = decision
        suppressed = objects_by_id[suppressed_id]
        keeper = objects_by_id[keeper_id]
        candidate = {
            "action": "mark_rejected_by_refine",
            "suppressed_object_id": suppressed_id,
            "suppressed_label": normalize_label(suppressed.get("canonical_label")),
            "suppressed_status": suppressed.get("status"),
            "suppressed_support_frames": suppressed.get("support_frame_count"),
            "suppressed_support_observations": suppressed.get("support_observation_count"),
            "kept_object_id": keeper_id,
            "kept_label": normalize_label(keeper.get("canonical_label")),
            "kept_status": keeper.get("status"),
            "kept_support_frames": keeper.get("support_frame_count"),
            "kept_support_observations": keeper.get("support_observation_count"),
            "reason": reason,
            "source_relation": row.get("relation"),
            "bbox_iou": row.get("bbox_iou"),
            "bbox_intersection_over_smaller": row.get("bbox_intersection_over_smaller"),
            "point_intersection_over_smaller": row.get("point_intersection_over_smaller"),
            "point_iou": row.get("point_iou"),
            "_rank_score": support_score(keeper) + finite_float(row.get("point_intersection_over_smaller")) * 100.0,
        }
        old = candidates.get(suppressed_id)
        if old is None or finite_float(candidate["_rank_score"]) > finite_float(old.get("_rank_score")):
            candidates[suppressed_id] = candidate

    report_rows = []
    suppressed_ids = set(candidates)
    for obj in objects:
        object_id = str(obj.get("object_id"))
        obj["final_status"] = "kept"
        obj["refine_reason"] = None
        obj["refine_suppressed_by"] = None
        obj["refine_notes"] = []
        if object_id in suppressed_ids:
            candidate = candidates[object_id]
            obj["final_status"] = "rejected_by_refine"
            obj["refine_reason"] = candidate["reason"]
            obj["refine_suppressed_by"] = candidate["kept_object_id"]
            obj["refine_notes"] = [
                {
                    "type": "label_conflict_same_region",
                    "kept_object_id": candidate["kept_object_id"],
                    "kept_label": candidate["kept_label"],
                    "point_intersection_over_smaller": finite_float(candidate["point_intersection_over_smaller"]),
                    "bbox_intersection_over_smaller": finite_float(candidate["bbox_intersection_over_smaller"]),
                }
            ]
            report = {key: value for key, value in candidate.items() if not key.startswith("_")}
            report_rows.append(report)

    final_counts = Counter(str(obj.get("final_status")) for obj in objects)
    suppressed_label_counts = Counter(row["suppressed_label"] for row in report_rows)
    kept_label_counts = Counter(row["kept_label"] for row in report_rows)
    output_dir = make_output_dir(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    metadata = dict(metadata)
    metadata.update(
        {
            "source_object_memory_json": str(args.object_memory_json),
            "source_object_pair_issues_csv": str(args.object_pair_issues_csv),
            "refine_pass": "auto_label_conflict_v2" if not suppress_labels else "conservative_label_conflict_v1",
            "suppress_labels": sorted(suppress_labels),
        }
    )
    data["metadata"] = metadata
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    summary = dict(summary)
    summary.update(
        {
            "final_status_counts": dict(final_counts.most_common()),
            "refine_suppressed_count": len(report_rows),
            "refine_suppressed_label_counts": dict(suppressed_label_counts.most_common()),
            "refine_kept_label_counts_for_suppression": dict(kept_label_counts.most_common()),
            "refine_parameters": {
                "suppress_labels": sorted(suppress_labels),
                "min_point_containment": args.min_point_containment,
                "min_bbox_containment": args.min_bbox_containment,
                "min_point_iou": args.min_point_iou,
                "min_bbox_iou": args.min_bbox_iou,
                "min_bbox_iou_with_containment": args.min_bbox_iou_with_containment,
                "max_center_distance": args.max_center_distance,
                "min_support_ratio": args.min_support_ratio,
                "max_suppressed_frames_without_ratio": args.max_suppressed_frames_without_ratio,
                "min_support_score_margin": args.min_support_score_margin,
                "auto_strong_support_ratio": args.auto_strong_support_ratio,
                "protect_labels": sorted(protect_labels),
            },
        }
    )
    data["summary"] = summary

    write_json(output_dir / "refined_object_memory.json", data)
    write_report_csv(output_dir / "refine_report.csv", report_rows)
    lines = [
        "Object memory conservative refine summary",
        "",
        f"object_memory_json: {args.object_memory_json}",
        f"object_pair_issues_csv: {args.object_pair_issues_csv}",
        f"objects_total: {len(objects)}",
        f"refine_pass: {metadata['refine_pass']}",
        f"final_status_counts: {dict(final_counts.most_common())}",
        f"refine_suppressed_count: {len(report_rows)}",
        f"suppressed_label_counts: {dict(suppressed_label_counts.most_common())}",
        f"kept_label_counts_for_suppression: {dict(kept_label_counts.most_common())}",
        f"output: {output_dir}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[objects_total] {len(objects)}")
    print(f"[final_status_counts] {dict(final_counts.most_common())}")
    print(f"[refine_suppressed_count] {len(report_rows)}")
    print(f"[suppressed_label_counts] {dict(suppressed_label_counts.most_common())}")
    print(f"[output] {output_dir}")


if __name__ == "__main__":
    main()
