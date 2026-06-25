#!/usr/bin/env python3
"""
Minimal online adapter that emits MonoGS keyframe semantic tasks into the queue.

This is a safe stand-in for a future MonoGS frontend callback:
  current prototype: manifest task -> queue/pending JSON
  future online hook: keyframe metadata -> queue/pending JSON

It does not modify or run MonoGS. It only simulates online keyframe arrival from
an existing phase-1 manifest.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path(
    "/home/sky/czh/datasets/table/online_semantic_manifest/manifest_20260602_105441/manifest.json"
)
DEFAULT_QUEUE_ROOT = Path("/home/sky/czh/datasets/table/online_semantic_queue")


def parse_csv_ints(value: str | None) -> list[int]:
    if value is None or not value.strip():
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(path)


def ensure_queue_dirs(queue_root: Path) -> None:
    for name in ("pending", "processing", "done", "failed", "adapter_runs"):
        (queue_root / name).mkdir(parents=True, exist_ok=True)


def task_filename(task: dict[str, Any]) -> str:
    task_index = int(task.get("task_index", task.get("trajectory_index", 0)))
    frame_id = int(task["frame_id"])
    return f"task_{task_index:06d}_frame_{frame_id:06d}.json"


def queue_status(queue_root: Path) -> dict[str, int]:
    ensure_queue_dirs(queue_root)
    return {
        name: len(list((queue_root / name).glob("*.json")))
        for name in ("pending", "processing", "done", "failed")
    }


def select_tasks(
    tasks: list[dict[str, Any]],
    start_index: int,
    task_indices: str | None,
    frame_ids: str | None,
    max_tasks: int | None,
) -> list[dict[str, Any]]:
    if task_indices:
        selected = []
        for idx in parse_csv_ints(task_indices):
            if idx < 0 or idx >= len(tasks):
                raise ValueError(f"Task index out of range: {idx}; valid range is 0..{len(tasks) - 1}")
            selected.append(tasks[idx])
    elif frame_ids:
        wanted = set(parse_csv_ints(frame_ids))
        selected = [task for task in tasks if int(task["frame_id"]) in wanted]
        missing = sorted(wanted - {int(task["frame_id"]) for task in selected})
        if missing:
            raise ValueError(f"Frame ids not found in manifest: {missing}")
    else:
        if start_index < 0 or start_index >= len(tasks):
            raise ValueError(f"start_index out of range: {start_index}; valid range is 0..{len(tasks) - 1}")
        selected = tasks[start_index:]

    if max_tasks is not None and max_tasks > 0:
        selected = selected[:max_tasks]
    return selected


def emit_task(
    task: dict[str, Any],
    manifest_path: Path,
    manifest_metadata: dict[str, Any],
    queue_root: Path,
    overwrite: bool,
) -> tuple[str, Path]:
    filename = task_filename(task)
    pending_path = queue_root / "pending" / filename
    existing_paths = [
        pending_path,
        queue_root / "processing" / filename,
        queue_root / "done" / filename,
        queue_root / "failed" / filename,
    ]
    if not overwrite and any(path.exists() for path in existing_paths):
        return "skipped_existing", pending_path
    if overwrite:
        for path in existing_paths:
            if path.exists():
                path.unlink()

    wrapper = {
        "format": "online_semantic_task_v1",
        "queue_status": "pending",
        "queued_at": datetime.now().isoformat(timespec="seconds"),
        "source": "online_monogs_adapter",
        "source_mode": "manifest_replay_keyframe_simulation",
        "source_manifest": str(manifest_path),
        "manifest_metadata": manifest_metadata,
        "event": {
            "type": "keyframe",
            "emitted_at": datetime.now().isoformat(timespec="seconds"),
            "frame_id": int(task["frame_id"]),
            "frame_stem": task.get("frame_stem"),
            "is_init": bool(task.get("is_init", False)),
            "is_keyframe": bool(task.get("is_keyframe", True)),
        },
        "task": task,
    }
    write_json_atomic(pending_path, wrapper)
    return "enqueued", pending_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay MonoGS keyframe metadata into semantic queue")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--queue_root", type=Path, default=DEFAULT_QUEUE_ROOT)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--task_indices", type=str, default=None)
    parser.add_argument("--frame_ids", type=str, default=None)
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--interval_sec", type=float, default=0.0, help="Sleep between emitted tasks to simulate online keyframe arrival")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ensure_queue_dirs(args.queue_root)
    manifest = load_json(args.manifest)
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError(f"Manifest has no task list: {args.manifest}")

    selected = select_tasks(
        tasks,
        start_index=args.start_index,
        task_indices=args.task_indices,
        frame_ids=args.frame_ids,
        max_tasks=args.max_tasks,
    )
    if not selected:
        raise ValueError("No tasks selected")

    run_log = {
        "script": "Online/core/online_monogs_adapter.py",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(args.manifest),
        "queue_root": str(args.queue_root),
        "start_index": args.start_index,
        "task_indices": args.task_indices,
        "frame_ids": args.frame_ids,
        "max_tasks": args.max_tasks,
        "interval_sec": args.interval_sec,
        "overwrite": args.overwrite,
        "selected_task_count": len(selected),
        "emitted": [],
    }

    counts = {"enqueued": 0, "skipped_existing": 0}
    for task_idx, task in enumerate(selected):
        status, path = emit_task(
            task=task,
            manifest_path=args.manifest,
            manifest_metadata=manifest.get("metadata", {}),
            queue_root=args.queue_root,
            overwrite=args.overwrite,
        )
        counts[status] = counts.get(status, 0) + 1
        event = {
            "status": status,
            "path": str(path),
            "frame_id": int(task["frame_id"]),
            "frame_stem": task.get("frame_stem"),
            "task_index": int(task.get("task_index", task_idx)),
        }
        run_log["emitted"].append(event)
        print(f"[{status}] frame={event['frame_stem']} path={path}")
        if args.interval_sec > 0 and task_idx != len(selected) - 1:
            time.sleep(args.interval_sec)

    run_log["counts"] = counts
    run_log["queue_status"] = queue_status(args.queue_root)
    run_path = args.queue_root / "adapter_runs" / f"adapter_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    write_json_atomic(run_path, run_log)

    print(json.dumps({**counts, **run_log["queue_status"], "run_log": str(run_path)}, indent=2))


if __name__ == "__main__":
    main()
