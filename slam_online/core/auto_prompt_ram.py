#!/usr/bin/env python3
"""
Generate automatic image tags from RGB images with RAM/RAM++.

This is the first step toward online 3D object memory:

  queue keyframe image -> RAM++ raw tags

The script is intentionally independent from the Grounded-SAM2 worker. It can
run on one image for smoke tests, or read existing queue task JSON files without
moving their queue state.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "auto_prompts_ram"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "third_party"
    / "recognize-anything"
    / "pretrained"
    / "ram_plus_swin_large_14m.pth"
)

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
    return next_available_dir(output_root, f"auto_prompts_{timestamp}")


def write_json(path: Path, data: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def parse_csv(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def split_tag_string(value: str) -> list[str]:
    pieces = re.split(r"\s*\|\s*|\s*,\s*|\s*;\s*|\n+", value)
    return [piece.strip() for piece in pieces if piece.strip()]


def normalize_tag(tag: str) -> str:
    tag = tag.strip().lower().replace("_", " ")
    tag = re.sub(r"\s+", " ", tag)
    tag = re.sub(r"^(a|an|the)\s+", "", tag)
    return tag


def unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def extract_english_tags(raw: Any) -> tuple[list[str], Any]:
    """Parse RAM/RAM++ inference output while keeping raw output for debugging."""
    if isinstance(raw, str):
        return split_tag_string(raw), raw

    if isinstance(raw, dict):
        for key in ("tags", "tag", "english_tags", "image_tags", "res"):
            value = raw.get(key)
            if isinstance(value, str):
                return split_tag_string(value), raw
            if isinstance(value, list):
                return [str(item) for item in value], raw

    if isinstance(raw, (list, tuple)):
        if not raw:
            return [], raw
        first = raw[0]
        if isinstance(first, str):
            return split_tag_string(first), raw
        if isinstance(first, list):
            return [str(item) for item in first], raw
        if isinstance(first, dict):
            return extract_english_tags(first)

    return [str(raw)], raw


def normalize_raw_tags(
    tags: list[str],
    max_tags: int,
    min_chars: int,
) -> list[str]:
    normalized = [normalize_tag(tag) for tag in tags]
    cleaned_tags = []
    for tag in normalized:
        if len(tag) < min_chars:
            continue
        if tag.isdigit():
            continue
        cleaned_tags.append(tag)
    cleaned_tags = unique_preserve_order(cleaned_tags)
    if max_tags > 0:
        cleaned_tags = cleaned_tags[:max_tags]
    return cleaned_tags


def import_ram_backend():
    try:
        from ram import get_transform
        from ram.models import ram, ram_plus
    except ImportError as exc:
        raise ImportError(
            "Failed to import RAM/RAM++. Install Recognize Anything and its "
            "dependencies in the Semantic2D environment. If the traceback mentions "
            "a missing package such as scipy, install that package and retry. "
            "Original error: "
            f"{exc}"
        ) from exc

    inference_fn = None
    try:
        from ram import inference_ram as maybe_fn

        inference_fn = maybe_fn
    except Exception:
        pass

    if not callable(inference_fn):
        try:
            from ram.inference import inference_ram as maybe_fn

            inference_fn = maybe_fn
        except Exception:
            pass

    if not callable(inference_fn):
        if hasattr(inference_fn, "inference_ram"):
            inference_fn = inference_fn.inference_ram
        elif hasattr(inference_fn, "inference"):
            inference_fn = inference_fn.inference

    if not callable(inference_fn):
        raise ImportError("Could not locate a callable RAM inference function.")

    return get_transform, ram, ram_plus, inference_fn


class RamAutoPromptModel:
    def __init__(
        self,
        checkpoint: Path,
        model_variant: str,
        image_size: int,
        vit: str,
        device: torch.device,
    ) -> None:
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"RAM/RAM++ checkpoint not found: {checkpoint}\n"
                "Download the checkpoint from the official Recognize Anything README "
                "and pass it with --checkpoint."
            )

        get_transform, ram, ram_plus, inference_fn = import_ram_backend()
        model_ctor = ram_plus if model_variant == "ram_plus" else ram
        self.model = model_ctor(pretrained=str(checkpoint), image_size=image_size, vit=vit)
        self.model.eval()
        self.model.to(device)
        self.transform = get_transform(image_size=image_size)
        self.inference_fn = inference_fn
        self.device = device
        self.checkpoint = checkpoint
        self.model_variant = model_variant
        self.image_size = image_size
        self.vit = vit

    @torch.no_grad()
    def predict_tags(self, image_path: Path) -> tuple[list[str], Any]:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        raw = self.inference_fn(tensor, self.model)
        tags, raw_kept = extract_english_tags(raw)
        return tags, raw_kept


def collect_image_inputs(
    image_paths: list[Path],
    queue_root: Path | None,
    queue_states: list[str],
    max_images: int | None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    for image_path in image_paths:
        items.append(
            {
                "source": "image",
                "image_path": str(image_path.expanduser()),
                "frame_stem": image_path.stem,
                "frame_id": None,
                "queue_record": None,
            }
        )

    if queue_root is not None:
        for state in queue_states:
            for record_path in sorted((queue_root / state).glob("*.json")):
                wrapper = load_json(record_path)
                task = wrapper.get("task", {})
                image_path = task.get("image_path")
                if not image_path:
                    continue
                items.append(
                    {
                        "source": f"queue/{state}",
                        "queue_record": str(record_path),
                        "image_path": str(image_path),
                        "frame_stem": str(
                            task.get("frame_stem")
                            or f"frame_{int(task.get('frame_id', len(items))):06d}"
                        ),
                        "frame_id": task.get("frame_id"),
                    }
                )

    if max_images is not None and max_images > 0:
        items = items[:max_images]
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RAM/RAM++ image tags")
    parser.add_argument("--image", type=Path, action="append", default=[], help="Image path. Can be passed multiple times.")
    parser.add_argument("--queue_root", type=Path, default=None, help="Read queue task JSON files without moving them.")
    parser.add_argument("--queue_states", type=str, default="done", help="Comma-separated queue states to read, e.g. done,pending.")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model_variant", choices=("ram_plus", "ram"), default="ram_plus")
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--vit", type=str, default="swin_l")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument(
        "--max_tags",
        dest="max_tags",
        type=int,
        default=20,
        help="Maximum RAM tags to keep after syntactic normalization.",
    )
    parser.add_argument("--min_chars", type=int, default=2)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(args.device)

    queue_states = parse_csv(args.queue_states)
    image_inputs = collect_image_inputs(args.image, args.queue_root, queue_states, args.max_images)
    if not image_inputs:
        raise ValueError("No input images. Pass --image or --queue_root.")

    output_dir = make_output_dir(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = RamAutoPromptModel(
        checkpoint=args.checkpoint.expanduser(),
        model_variant=args.model_variant,
        image_size=args.image_size,
        vit=args.vit,
        device=device,
    )

    print(f"[backend] {args.model_variant}")
    print(f"[checkpoint] {args.checkpoint}")
    print(f"[output] {output_dir}")
    print(f"[images] {len(image_inputs)}")

    results: list[dict[str, Any]] = []
    global_tag_counts: dict[str, int] = {}
    for item in tqdm(image_inputs, desc="ram auto prompts"):
        start = time.perf_counter()
        image_path = Path(str(item["image_path"])).expanduser()
        result: dict[str, Any] = {
            **item,
            "image_path": str(image_path),
            "status": "pending",
            "raw_tags": [],
            "elapsed_sec": 0.0,
        }
        try:
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            raw_tags, raw_output = model.predict_tags(image_path)
            raw_tags = normalize_raw_tags(
                raw_tags,
                max_tags=args.max_tags,
                min_chars=args.min_chars,
            )
            result["status"] = "processed"
            result["raw_tags"] = raw_tags
            result["raw_output"] = raw_output
            for tag in raw_tags:
                global_tag_counts[tag] = global_tag_counts.get(tag, 0) + 1
        except Exception as exc:
            result["status"] = "failed"
            result["error"] = repr(exc)
        result["elapsed_sec"] = round(time.perf_counter() - start, 6)
        results.append(result)

        stem = str(item.get("frame_stem") or image_path.stem)
        write_json(output_dir / f"{stem}_auto_prompts.json", result)

    scene_tags = [
        tag
        for tag, _count in sorted(
            global_tag_counts.items(), key=lambda pair: (-pair[1], pair[0])
        )
    ]
    if args.max_tags > 0:
        scene_tags = scene_tags[: args.max_tags]

    run_config = {
        "script": "auto_prompt_ram.py",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "backend": args.model_variant,
        "checkpoint": str(args.checkpoint),
        "image_size": args.image_size,
        "vit": args.vit,
        "device": str(device),
        "queue_root": str(args.queue_root) if args.queue_root else None,
        "queue_states": queue_states,
        "output": str(output_dir),
        "max_tags": args.max_tags,
        "tag_policy": "RAM++ only outputs normalized raw_tags. Semantic filtering into object_prompts is done downstream by auto_prompt_filter.py.",
        "results": results,
        "scene_tag_counts": global_tag_counts,
        "scene_tags": scene_tags,
    }
    write_json(output_dir / "auto_prompts.json", run_config)

    summary_lines = [
        "RAM/RAM++ auto prompt summary",
        "",
        f"images: {len(results)}",
        f"processed: {sum(1 for item in results if item['status'] == 'processed')}",
        f"failed: {sum(1 for item in results if item['status'] == 'failed')}",
        "",
        "scene_tags:",
    ]
    for tag in scene_tags:
        summary_lines.append(f"- {tag}: {global_tag_counts[tag]}")
    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[scene_tags] {scene_tags}")
    print(f"[done] outputs at {output_dir}")


if __name__ == "__main__":
    main()
