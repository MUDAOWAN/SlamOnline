#!/usr/bin/env python3
"""
File-backed semantic task queue for the Online/core adapter.

The queue is intentionally simple and inspectable:
  queue_root/
    queue_config.json
    pending/*.json
    processing/*.json
    done/*.json
    failed/*.json

This is a safe phase for online wiring because it does not modify MonoGS and it
does not run any semantic model. Later a MonoGS adapter can enqueue keyframes as
they are created, while workers consume the same pending JSON tasks.
"""

from __future__ import annotations

import argparse
import json
import shutil
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
    for name in ("pending", "processing", "done", "failed"):
        (queue_root / name).mkdir(parents=True, exist_ok=True)


def init_queue(queue_root: Path, reset_processing: bool = False) -> None:
    ensure_queue_dirs(queue_root)
    config_path = queue_root / "queue_config.json"
    if not config_path.exists():
        write_json_atomic(
            config_path,
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "queue_root": str(queue_root),
                "layout": ["pending", "processing", "done", "failed"],
                "task_format": "online_semantic_task_v1",
            },
        )

    if reset_processing:
        for task_path in sorted((queue_root / "processing").glob("*.json")):
            task_path.replace(queue_root / "pending" / task_path.name)


def task_filename(task: dict[str, Any]) -> str:
    task_index = int(task.get("task_index", task.get("trajectory_index", 0)))
    frame_id = int(task["frame_id"])
    return f"task_{task_index:06d}_frame_{frame_id:06d}.json"


def select_manifest_tasks(
    tasks: list[dict[str, Any]],
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
        selected = list(tasks)

    if max_tasks is not None and max_tasks > 0:
        selected = selected[:max_tasks]
    return selected


def enqueue_manifest(
    manifest_path: Path,
    queue_root: Path,
    task_indices: str | None,
    frame_ids: str | None,
    max_tasks: int | None,
    overwrite: bool,
) -> dict[str, int]:
    init_queue(queue_root)
    manifest = load_json(manifest_path)
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError(f"Manifest has no task list: {manifest_path}")

    selected = select_manifest_tasks(tasks, task_indices, frame_ids, max_tasks)
    counts = {"selected": len(selected), "enqueued": 0, "skipped_existing": 0}
    for task in selected:
        wrapped = {
            "format": "online_semantic_task_v1",
            "queue_status": "pending",
            "queued_at": datetime.now().isoformat(timespec="seconds"),
            "source_manifest": str(manifest_path),
            "manifest_metadata": manifest.get("metadata", {}),
            "task": task,
        }
        pending_path = queue_root / "pending" / task_filename(task)
        existing_paths = [
            pending_path,
            queue_root / "processing" / pending_path.name,
            queue_root / "done" / pending_path.name,
            queue_root / "failed" / pending_path.name,
        ]
        if not overwrite and any(path.exists() for path in existing_paths):
            counts["skipped_existing"] += 1
            continue
        if overwrite:
            for path in existing_paths:
                if path.exists():
                    path.unlink()
        write_json_atomic(pending_path, wrapped)
        counts["enqueued"] += 1
    return counts


def queue_status(queue_root: Path) -> dict[str, int]:
    ensure_queue_dirs(queue_root)
    return {
        name: len(list((queue_root / name).glob("*.json")))
        for name in ("pending", "processing", "done", "failed")
    }


def clear_queue(queue_root: Path, states: list[str]) -> dict[str, int]:
    allowed = {"pending", "processing", "done", "failed"}
    invalid = sorted(set(states) - allowed)
    if invalid:
        raise ValueError(f"Invalid states for clear: {invalid}")
    counts = {}
    for state in states:
        state_dir = queue_root / state
        count = 0
        if state_dir.exists():
            for task_path in state_dir.glob("*.json"):
                task_path.unlink()
                count += 1
        counts[state] = count
    return counts


def retry_failed(queue_root: Path, max_tasks: int | None = None) -> dict[str, int]:
    ensure_queue_dirs(queue_root)
    failed_paths = sorted((queue_root / "failed").glob("*.json"))
    if max_tasks is not None and max_tasks > 0:
        failed_paths = failed_paths[:max_tasks]

    moved = 0
    skipped_existing = 0
    for failed_path in failed_paths:
        pending_path = queue_root / "pending" / failed_path.name
        if pending_path.exists():
            skipped_existing += 1
            continue
        wrapper = load_json(failed_path)
        wrapper["queue_status"] = "pending"
        wrapper["retried_at"] = datetime.now().isoformat(timespec="seconds")
        wrapper.pop("worker_result", None)
        wrapper.pop("finished_at", None)
        write_json_atomic(failed_path, wrapper)
        failed_path.replace(pending_path)
        moved += 1
    return {"retried_failed": moved, "skipped_existing": skipped_existing}


def retry_processing(queue_root: Path, max_tasks: int | None = None) -> dict[str, int]:
    ensure_queue_dirs(queue_root)
    processing_paths = sorted((queue_root / "processing").glob("*.json"))
    if max_tasks is not None and max_tasks > 0:
        processing_paths = processing_paths[:max_tasks]

    moved = 0
    skipped_existing = 0
    for processing_path in processing_paths:
        pending_path = queue_root / "pending" / processing_path.name
        if pending_path.exists():
            skipped_existing += 1
            continue
        wrapper = load_json(processing_path)
        wrapper["queue_status"] = "pending"
        wrapper["retried_at"] = datetime.now().isoformat(timespec="seconds")
        wrapper.pop("worker_result", None)
        wrapper.pop("processing_started_at", None)
        write_json_atomic(processing_path, wrapper)
        processing_path.replace(pending_path)
        moved += 1
    return {"retried_processing": moved, "skipped_existing": skipped_existing}


def retry_done(queue_root: Path, max_tasks: int | None = None) -> dict[str, int]:
    ensure_queue_dirs(queue_root)
    done_paths = sorted((queue_root / "done").glob("*.json"))
    if max_tasks is not None and max_tasks > 0:
        done_paths = done_paths[:max_tasks]

    moved = 0
    skipped_existing = 0
    for done_path in done_paths:
        pending_path = queue_root / "pending" / done_path.name
        if pending_path.exists():
            skipped_existing += 1
            continue
        wrapper = load_json(done_path)
        wrapper["queue_status"] = "pending"
        wrapper["retried_at"] = datetime.now().isoformat(timespec="seconds")
        wrapper["retry_reason"] = "retry_done"
        wrapper.pop("worker_result", None)
        wrapper.pop("finished_at", None)
        write_json_atomic(done_path, wrapper)
        done_path.replace(pending_path)
        moved += 1
    return {"retried_done": moved, "skipped_existing": skipped_existing}


def export_pending_manifest(queue_root: Path, output: Path) -> dict[str, int]:
    ensure_queue_dirs(queue_root)
    task_wrappers = [load_json(path) for path in sorted((queue_root / "pending").glob("*.json"))]
    tasks = [wrapper["task"] for wrapper in task_wrappers]
    manifest = {
        "metadata": {
            "script": "Online/core/online_semantic_queue.py",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "queue_root": str(queue_root),
            "source": "pending_queue_export",
            "selected_task_count": len(tasks),
        },
        "tasks": tasks,
    }
    write_json_atomic(output, manifest)
    return {"exported": len(tasks)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage Online/core semantic task queue")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create queue directories")
    init_parser.add_argument("--queue_root", type=Path, default=DEFAULT_QUEUE_ROOT)
    init_parser.add_argument("--reset_processing", action="store_true")

    enqueue_parser = subparsers.add_parser("enqueue-manifest", help="Enqueue tasks from a manifest")
    enqueue_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    enqueue_parser.add_argument("--queue_root", type=Path, default=DEFAULT_QUEUE_ROOT)
    enqueue_parser.add_argument("--task_indices", type=str, default=None)
    enqueue_parser.add_argument("--frame_ids", type=str, default=None)
    enqueue_parser.add_argument("--max_tasks", type=int, default=None)
    enqueue_parser.add_argument("--overwrite", action="store_true")

    status_parser = subparsers.add_parser("status", help="Print queue counts")
    status_parser.add_argument("--queue_root", type=Path, default=DEFAULT_QUEUE_ROOT)

    clear_parser = subparsers.add_parser("clear", help="Delete task files in selected states")
    clear_parser.add_argument("--queue_root", type=Path, default=DEFAULT_QUEUE_ROOT)
    clear_parser.add_argument("--states", type=str, default="pending", help="Comma-separated states")

    retry_failed_parser = subparsers.add_parser("retry-failed", help="Move failed tasks back to pending")
    retry_failed_parser.add_argument("--queue_root", type=Path, default=DEFAULT_QUEUE_ROOT)
    retry_failed_parser.add_argument("--max_tasks", type=int, default=None)

    retry_processing_parser = subparsers.add_parser("retry-processing", help="Move processing tasks back to pending")
    retry_processing_parser.add_argument("--queue_root", type=Path, default=DEFAULT_QUEUE_ROOT)
    retry_processing_parser.add_argument("--max_tasks", type=int, default=None)

    retry_done_parser = subparsers.add_parser("retry-done", help="Move done tasks back to pending for prompt regeneration")
    retry_done_parser.add_argument("--queue_root", type=Path, default=DEFAULT_QUEUE_ROOT)
    retry_done_parser.add_argument("--max_tasks", type=int, default=None)

    export_parser = subparsers.add_parser("export-pending-manifest", help="Export pending queue tasks as a manifest")
    export_parser.add_argument("--queue_root", type=Path, default=DEFAULT_QUEUE_ROOT)
    export_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "init":
        init_queue(args.queue_root, reset_processing=args.reset_processing)
        print(json.dumps(queue_status(args.queue_root), indent=2))
    elif args.command == "enqueue-manifest":
        counts = enqueue_manifest(
            manifest_path=args.manifest,
            queue_root=args.queue_root,
            task_indices=args.task_indices,
            frame_ids=args.frame_ids,
            max_tasks=args.max_tasks,
            overwrite=args.overwrite,
        )
        print(json.dumps({**counts, **queue_status(args.queue_root)}, indent=2))
    elif args.command == "status":
        print(json.dumps(queue_status(args.queue_root), indent=2))
    elif args.command == "clear":
        states = [state.strip() for state in args.states.split(",") if state.strip()]
        counts = clear_queue(args.queue_root, states)
        print(json.dumps({**counts, **queue_status(args.queue_root)}, indent=2))
    elif args.command == "retry-failed":
        counts = retry_failed(args.queue_root, max_tasks=args.max_tasks)
        print(json.dumps({**counts, **queue_status(args.queue_root)}, indent=2))
    elif args.command == "retry-processing":
        counts = retry_processing(args.queue_root, max_tasks=args.max_tasks)
        print(json.dumps({**counts, **queue_status(args.queue_root)}, indent=2))
    elif args.command == "retry-done":
        counts = retry_done(args.queue_root, max_tasks=args.max_tasks)
        print(json.dumps({**counts, **queue_status(args.queue_root)}, indent=2))
    elif args.command == "export-pending-manifest":
        counts = export_pending_manifest(args.queue_root, args.output)
        print(json.dumps({**counts, **queue_status(args.queue_root)}, indent=2))
    else:
        raise ValueError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
