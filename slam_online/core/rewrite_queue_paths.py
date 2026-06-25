#!/usr/bin/env python3
"""Rewrite queue task image/depth paths for a local dataset root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_STATES = "pending,processing,done,failed"


def parse_states(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def depth_stem_from_frame_stem(frame_stem: str) -> str:
    if frame_stem.startswith("frame_"):
        return "depth_" + frame_stem.removeprefix("frame_")
    return frame_stem.replace("frame", "depth", 1)


def resolve_relative_path(
    dataset_root: Path,
    task: dict[str, Any],
    relative_key: str,
    fallback_dir: str,
    suffix: str,
    stem: str,
) -> tuple[Path, str]:
    relative = task.get(relative_key)
    if relative:
        rel_path = Path(str(relative))
    else:
        rel_path = Path(fallback_dir) / f"{stem}{suffix}"
    return dataset_root / rel_path, rel_path.as_posix()


def rewrite_task_paths(path: Path, dataset_root: Path, dry_run: bool) -> tuple[bool, bool, bool]:
    wrapper = read_json(path)
    task = wrapper.get("task")
    if not isinstance(task, dict):
        return False, False, False

    frame_stem = str(task.get("frame_stem") or f"frame_{int(task.get('frame_id', 0)):06d}")
    depth_stem = depth_stem_from_frame_stem(frame_stem)
    image_path, image_relative = resolve_relative_path(
        dataset_root=dataset_root,
        task=task,
        relative_key="image_relative_path",
        fallback_dir="results",
        suffix=".jpg",
        stem=frame_stem,
    )
    depth_path, depth_relative = resolve_relative_path(
        dataset_root=dataset_root,
        task=task,
        relative_key="depth_relative_path",
        fallback_dir="results",
        suffix=".png",
        stem=depth_stem,
    )

    changed = (
        task.get("image_path") != str(image_path)
        or task.get("depth_path") != str(depth_path)
        or task.get("image_relative_path") != image_relative
        or task.get("depth_relative_path") != depth_relative
    )
    if not changed:
        return False, image_path.exists(), depth_path.exists()

    task.setdefault("source_dataset_root", task.get("dataset_root"))
    task["dataset_root"] = str(dataset_root)
    task["image_path"] = str(image_path)
    task["image_relative_path"] = image_relative
    task["image_exists"] = image_path.exists()
    task["depth_path"] = str(depth_path)
    task["depth_relative_path"] = depth_relative
    task["depth_exists"] = depth_path.exists()

    if not dry_run:
        write_json(path, wrapper)
    return True, image_path.exists(), depth_path.exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite online semantic queue paths for this machine.")
    parser.add_argument("--queue_root", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--states", type=str, default=DEFAULT_STATES)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    total = 0
    changed = 0
    missing_images = 0
    missing_depths = 0
    for state in parse_states(args.states):
        state_dir = args.queue_root / state
        if not state_dir.exists():
            continue
        for path in sorted(state_dir.glob("*.json")):
            total += 1
            did_change, image_exists, depth_exists = rewrite_task_paths(path, dataset_root, dry_run=args.dry_run)
            changed += int(did_change)
            missing_images += int(not image_exists)
            missing_depths += int(not depth_exists)

    action = "would rewrite" if args.dry_run else "rewrote"
    print(f"[queue_root] {args.queue_root}")
    print(f"[dataset_root] {dataset_root}")
    print(f"[tasks] {total}")
    print(f"[{action}] {changed}")
    print(f"[missing_images] {missing_images}")
    print(f"[missing_depths] {missing_depths}")


if __name__ == "__main__":
    main()
