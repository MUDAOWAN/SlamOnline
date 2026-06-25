#!/usr/bin/env python3
"""
Audit per-frame 3D Gaussian observations before object memory association.

The goal is to inspect whether lifted 2D masks produce reasonable 3D evidence:
hit point counts, hit ratios, bbox sizes, bbox volumes, label frequency, and
frame-level observation density.
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
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "frame_3d_observation_audit"


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


def list_float(values: Any, length: int = 3) -> list[float]:
    if not isinstance(values, list):
        return [0.0] * length
    out = [finite_float(item) for item in values[:length]]
    while len(out) < length:
        out.append(0.0)
    return out


def quantile_summary(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    q = np.percentile(arr, [5, 10, 25, 50, 75, 90, 95])
    return {
        "count": int(len(arr)),
        "min": round(float(arr.min()), 6),
        "p05": round(float(q[0]), 6),
        "p10": round(float(q[1]), 6),
        "p25": round(float(q[2]), 6),
        "p50": round(float(q[3]), 6),
        "p75": round(float(q[4]), 6),
        "p90": round(float(q[5]), 6),
        "p95": round(float(q[6]), 6),
        "max": round(float(arr.max()), 6),
        "mean": round(float(arr.mean()), 6),
    }


def compact_observation(obs: dict[str, Any]) -> dict[str, Any]:
    return {
        "observation_id": obs.get("observation_id"),
        "observation_index": obs.get("observation_index"),
        "frame_id": obs.get("frame_id"),
        "frame_stem": obs.get("frame_stem"),
        "label": obs.get("label"),
        "hit_point_count": obs.get("hit_point_count"),
        "visible_point_count": obs.get("visible_point_count"),
        "hit_ratio": obs.get("hit_ratio"),
        "bbox_volume": obs.get("bbox_volume"),
        "bbox_size": obs.get("bbox_size"),
        "center_3d": obs.get("center_3d"),
        "max_box_score": obs.get("max_box_score"),
        "max_mask_score": obs.get("max_mask_score"),
        "mask_area_ratio": obs.get("mask_area_ratio"),
        "mask_path": obs.get("mask_path"),
        "detections_json": obs.get("detections_json"),
        "hit_key": obs.get("hit_key"),
    }


def make_label_stats(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for obs in observations:
        grouped[str(obs.get("label") or "unknown")].append(obs)

    rows = []
    for label, items in grouped.items():
        hit_counts = [finite_float(obs.get("hit_point_count")) for obs in items]
        hit_ratios = [finite_float(obs.get("hit_ratio")) for obs in items]
        volumes = [finite_float(obs.get("bbox_volume")) for obs in items]
        mask_areas = [finite_float(obs.get("mask_area_ratio")) for obs in items]
        box_scores = [finite_float(obs.get("max_box_score")) for obs in items]
        bbox_sizes = np.asarray([list_float(obs.get("bbox_size")) for obs in items], dtype=np.float64)
        frame_ids = sorted({finite_int(obs.get("frame_id")) for obs in items})
        rows.append(
            {
                "label": label,
                "count": len(items),
                "frame_count": len(frame_ids),
                "first_frame_id": frame_ids[0] if frame_ids else None,
                "last_frame_id": frame_ids[-1] if frame_ids else None,
                "hit_point_count": quantile_summary(hit_counts),
                "hit_ratio": quantile_summary(hit_ratios),
                "bbox_volume": quantile_summary(volumes),
                "mask_area_ratio": quantile_summary(mask_areas),
                "max_box_score": quantile_summary(box_scores),
                "bbox_size_mean": [round(float(x), 6) for x in bbox_sizes.mean(axis=0).tolist()] if len(bbox_sizes) else [0.0, 0.0, 0.0],
                "bbox_size_max": [round(float(x), 6) for x in bbox_sizes.max(axis=0).tolist()] if len(bbox_sizes) else [0.0, 0.0, 0.0],
            }
        )
    return sorted(rows, key=lambda item: (-int(item["count"]), str(item["label"])))


def write_label_stats_csv(path: Path, label_stats: list[dict[str, Any]]) -> None:
    fieldnames = [
        "label",
        "count",
        "frame_count",
        "hit_point_median",
        "hit_point_p10",
        "hit_point_p90",
        "hit_ratio_median",
        "bbox_volume_median",
        "bbox_volume_p90",
        "bbox_volume_max",
        "bbox_size_mean",
        "bbox_size_max",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in label_stats:
            writer.writerow(
                {
                    "label": item["label"],
                    "count": item["count"],
                    "frame_count": item["frame_count"],
                    "hit_point_median": item["hit_point_count"]["p50"],
                    "hit_point_p10": item["hit_point_count"]["p10"],
                    "hit_point_p90": item["hit_point_count"]["p90"],
                    "hit_ratio_median": item["hit_ratio"]["p50"],
                    "bbox_volume_median": item["bbox_volume"]["p50"],
                    "bbox_volume_p90": item["bbox_volume"]["p90"],
                    "bbox_volume_max": item["bbox_volume"]["max"],
                    "bbox_size_mean": json.dumps(item["bbox_size_mean"], ensure_ascii=False),
                    "bbox_size_max": json.dumps(item["bbox_size_max"], ensure_ascii=False),
                }
            )


def write_observation_examples_csv(path: Path, observations: list[dict[str, Any]]) -> None:
    fieldnames = [
        "observation_id",
        "observation_index",
        "frame_id",
        "frame_stem",
        "label",
        "hit_point_count",
        "hit_ratio",
        "bbox_volume",
        "bbox_size",
        "center_3d",
        "max_box_score",
        "mask_area_ratio",
        "mask_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for obs in observations:
            compact = compact_observation(obs)
            writer.writerow(
                {
                    key: json.dumps(compact.get(key), ensure_ascii=False)
                    if isinstance(compact.get(key), list)
                    else compact.get(key)
                    for key in fieldnames
                }
            )


def frame_density(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for frame in frames:
        rows.append(
            {
                "frame_id": frame.get("frame_id"),
                "frame_stem": frame.get("frame_stem"),
                "status": frame.get("status"),
                "accepted_2d_count": frame.get("accepted_2d_count"),
                "accepted_3d_count": frame.get("accepted_3d_count"),
                "rejected_3d_count": frame.get("rejected_3d_count"),
                "visible_point_count": frame.get("visible_point_count"),
            }
        )
    return sorted(rows, key=lambda item: (-finite_int(item.get("accepted_3d_count")), finite_int(item.get("frame_id"))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit lifted 3D observations before object memory association")
    parser.add_argument("--frame_3d_observations_json", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    data = read_json(args.frame_3d_observations_json)
    observations = data.get("observations") if isinstance(data, dict) else None
    frames = data.get("frames") if isinstance(data, dict) else None
    rejected = data.get("rejected_observations") if isinstance(data, dict) else []
    if not isinstance(observations, list):
        raise ValueError(f"Expected observations list in {args.frame_3d_observations_json}")
    if not isinstance(frames, list):
        frames = []
    if not isinstance(rejected, list):
        rejected = []

    output_dir = make_output_dir(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    hit_counts = [finite_float(obs.get("hit_point_count")) for obs in observations]
    hit_ratios = [finite_float(obs.get("hit_ratio")) for obs in observations]
    volumes = [finite_float(obs.get("bbox_volume")) for obs in observations]
    mask_areas = [finite_float(obs.get("mask_area_ratio")) for obs in observations]
    box_scores = [finite_float(obs.get("max_box_score")) for obs in observations]
    mask_scores = [finite_float(obs.get("max_mask_score")) for obs in observations]
    bbox_sizes = np.asarray([list_float(obs.get("bbox_size")) for obs in observations], dtype=np.float64)
    labels = Counter(str(obs.get("label") or "unknown") for obs in observations)

    top_k = max(1, int(args.top_k))
    largest_volume = sorted(observations, key=lambda obs: finite_float(obs.get("bbox_volume")), reverse=True)[:top_k]
    smallest_hit_count = sorted(observations, key=lambda obs: (finite_float(obs.get("hit_point_count")), str(obs.get("observation_id"))))[:top_k]
    lowest_hit_ratio = sorted(observations, key=lambda obs: (finite_float(obs.get("hit_ratio")), str(obs.get("observation_id"))))[:top_k]
    largest_bbox_axis = sorted(
        observations,
        key=lambda obs: max(list_float(obs.get("bbox_size"))),
        reverse=True,
    )[:top_k]
    frame_rows = frame_density(frames)
    label_stats = make_label_stats(observations)

    summary = {
        "script": "audit_frame_3d_observations.py",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(args.frame_3d_observations_json),
        "output": str(output_dir),
        "observation_count": len(observations),
        "rejected_observation_count": len(rejected),
        "frame_count": len(frames),
        "label_count": len(labels),
        "top_labels": labels.most_common(top_k),
        "hit_point_count": quantile_summary(hit_counts),
        "hit_ratio": quantile_summary(hit_ratios),
        "bbox_volume": quantile_summary(volumes),
        "mask_area_ratio": quantile_summary(mask_areas),
        "max_box_score": quantile_summary(box_scores),
        "max_mask_score": quantile_summary(mask_scores),
        "bbox_size_x": quantile_summary(bbox_sizes[:, 0] if len(bbox_sizes) else []),
        "bbox_size_y": quantile_summary(bbox_sizes[:, 1] if len(bbox_sizes) else []),
        "bbox_size_z": quantile_summary(bbox_sizes[:, 2] if len(bbox_sizes) else []),
        "top_largest_bbox_volume": [compact_observation(obs) for obs in largest_volume],
        "top_smallest_hit_count": [compact_observation(obs) for obs in smallest_hit_count],
        "top_lowest_hit_ratio": [compact_observation(obs) for obs in lowest_hit_ratio],
        "top_largest_bbox_axis": [compact_observation(obs) for obs in largest_bbox_axis],
        "top_dense_frames": frame_rows[:top_k],
        "label_stats": label_stats,
    }

    write_json(output_dir / "audit_3d_observations.json", summary)
    write_label_stats_csv(output_dir / "label_stats.csv", label_stats)
    write_observation_examples_csv(output_dir / "largest_bbox_volume.csv", largest_volume)
    write_observation_examples_csv(output_dir / "smallest_hit_count.csv", smallest_hit_count)
    write_observation_examples_csv(output_dir / "lowest_hit_ratio.csv", lowest_hit_ratio)
    write_observation_examples_csv(output_dir / "largest_bbox_axis.csv", largest_bbox_axis)

    lines = [
        "Frame 3D observation audit summary",
        "",
        f"observations: {len(observations)}",
        f"rejected_observations: {len(rejected)}",
        f"frames: {len(frames)}",
        f"labels: {len(labels)}",
        "",
        "hit_point_count:",
        json.dumps(summary["hit_point_count"], ensure_ascii=False),
        "hit_ratio:",
        json.dumps(summary["hit_ratio"], ensure_ascii=False),
        "bbox_volume:",
        json.dumps(summary["bbox_volume"], ensure_ascii=False),
        "",
        "top labels:",
    ]
    for label, count in labels.most_common(top_k):
        lines.append(f"- {label}: {count}")
    lines.extend(["", f"output: {output_dir}"])
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[observations] {len(observations)}")
    print(f"[frames] {len(frames)}")
    print(f"[labels] {len(labels)}")
    print(f"[hit_point_count] {summary['hit_point_count']}")
    print(f"[bbox_volume] {summary['bbox_volume']}")
    print(f"[top_labels] {labels.most_common(min(top_k, 10))}")
    print(f"[output] {output_dir}")


if __name__ == "__main__":
    main()
