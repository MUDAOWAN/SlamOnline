#!/usr/bin/env python3
"""
Filter RAM/RAM++ tags into object prompts for Grounded-SAM2.

This is the first semantic gate in the online object-memory route:

  RAM++ raw tags -> LLM/rule prompt classifier -> object prompts

The LLM classifier decides whether a tag is suitable as an independent 3D
object prompt. Results are cached so repeated tags do not require repeated API
calls. API keys are read from environment variables and are never written to
outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "auto_prompt_filter"
DEFAULT_CACHE = DEFAULT_OUTPUT_ROOT / "prompt_classifier_cache.json"
DEFAULT_LLM_CONFIG = PROJECT_ROOT / "configs" / "llm_prompt_filter.json"

PROMPT_POLICY_VERSION = "instance_object_v2"

KEEP_CATEGORIES = {"physical_object"}
HOLD_CATEGORIES = {
    "scene_or_place",
    "structural_surface",
    "attribute",
    "material",
    "action",
    "generic_category",
    "uncertain",
}
VALID_CATEGORIES = KEEP_CATEGORIES | HOLD_CATEGORIES


RULE_SCENE_TERMS = {
    "bathroom",
    "bedroom",
    "corridor",
    "hallway",
    "home",
    "house",
    "indoor",
    "indoors",
    "interior",
    "kitchen",
    "living room",
    "office",
    "room",
    "scene",
}
RULE_ATTRIBUTE_TERMS = {
    "black",
    "blue",
    "bright",
    "brown",
    "closed",
    "dark",
    "gray",
    "green",
    "large",
    "modern",
    "open",
    "red",
    "small",
    "white",
    "wooden",
    "yellow",
}
RULE_MATERIAL_TERMS = {
    "ceramic",
    "fabric",
    "glass",
    "leather",
    "metal",
    "plastic",
    "steel",
    "wood",
}
RULE_GENERIC_TERMS = {
    "appliance",
    "decoration",
    "equipment",
    "furniture",
    "item",
    "object",
    "stuff",
    "thing",
}


def normalize_tag(tag: str) -> str:
    tag = tag.strip().lower().replace("_", " ")
    tag = re.sub(r"\s+", " ", tag)
    tag = re.sub(r"^(a|an|the)\s+", "", tag)
    return tag


def unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        normalized = normalize_tag(str(item))
        if not normalized or normalized in seen:
            continue
        out.append(normalized)
        seen.add(normalized)
    return out


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
    return next_available_dir(output_root, f"prompt_filter_{timestamp}")


def read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_llm_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(
            f"LLM config not found: {path}. Copy configs/llm_prompt_filter.example.json "
            "to configs/llm_prompt_filter.json and fill in your local API settings."
        )
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in LLM config: {path}")
    return data


def load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = read_json(path)
    return data if isinstance(data, dict) else {}


def save_cache(path: Path, cache: dict[str, Any]) -> None:
    write_json(path, cache)


def extract_prompt_records(data: Any) -> list[dict[str, Any]]:
    """Support one frame RAM tag JSON or scene-level auto_prompts.json."""
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        records = []
        for item in data["results"]:
            if isinstance(item, dict):
                prompts = item.get("raw_tags") or item.get("candidate_tags") or item.get("prompts") or []
                if prompts:
                    records.append(
                        {
                            "frame_stem": str(item.get("frame_stem") or "scene"),
                            "frame_id": item.get("frame_id"),
                            "image_path": item.get("image_path"),
                            "source_record": item,
                            "prompts": unique_preserve_order([str(x) for x in prompts]),
                        }
                    )
        return records

    if isinstance(data, dict):
        prompts = (
            data.get("raw_tags")
            or data.get("scene_tags")
            or data.get("candidate_tags")
            or data.get("prompts")
            or data.get("scene_candidate_tags")
            or data.get("scene_prompts")
            or []
        )
        if not isinstance(prompts, list):
            raise ValueError("Expected raw_tags/scene_tags/candidate_tags/prompts/scene_candidate_tags/scene_prompts to be a list")
        return [
            {
                "frame_stem": str(data.get("frame_stem") or "scene"),
                "frame_id": data.get("frame_id"),
                "image_path": data.get("image_path"),
                "source_record": data,
                "prompts": unique_preserve_order([str(x) for x in prompts]),
            }
        ]

    if isinstance(data, list):
        return [
            {
                "frame_stem": "scene",
                "frame_id": None,
                "image_path": None,
                "source_record": data,
                "prompts": unique_preserve_order([str(x) for x in data]),
            }
        ]

    raise ValueError("Unsupported auto prompt JSON format")


def rule_classify(tag: str) -> dict[str, Any]:
    tag = normalize_tag(tag)
    if tag in RULE_SCENE_TERMS:
        category = "scene_or_place"
        reason = "Rule match: place/scene term."
    elif tag in RULE_ATTRIBUTE_TERMS:
        category = "attribute"
        reason = "Rule match: visual attribute term."
    elif tag in RULE_MATERIAL_TERMS:
        category = "material"
        reason = "Rule match: material term."
    elif tag in RULE_GENERIC_TERMS:
        category = "generic_category"
        reason = "Rule match: broad parent category, not a concrete object instance."
    else:
        category = "uncertain"
        reason = "No rule match; hold unless the LLM confirms an independent object instance."

    is_object_prompt = category in KEEP_CATEGORIES
    return {
        "tag": tag,
        "category": category,
        "is_object_prompt": is_object_prompt,
        "action": "keep" if is_object_prompt else "hold",
        "normalized_label": tag if is_object_prompt else None,
        "reason": reason,
        "classifier": "rule",
    }


def build_llm_messages(tags: list[str]) -> list[dict[str, str]]:
    categories = ", ".join(sorted(VALID_CATEGORIES))
    return [
        {
            "role": "system",
            "content": (
                "You are a prompt classifier for an open-vocabulary 3D object "
                "mapping system. Decide whether each image tag is suitable as an "
                "independent, countable physical object prompt for object detection, "
                "segmentation, and 3D object memory. Be conservative: keep only "
                "discrete object instances, not background regions or room structure. "
                "Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                "Classify each tag into exactly one category:\n"
                f"{categories}\n\n"
                "Definitions:\n"
                "- physical_object: a discrete, countable object or clearly bounded fixture with its own instance mask and compact 3D extent.\n"
                "- scene_or_place: a room, place, background, environment, or whole-scene label.\n"
                "- structural_surface: a continuous architectural or room-boundary surface, ground plane, background region, or built-in area that is not an independent object instance.\n"
                "- attribute: color, size, style, state, or other property.\n"
                "- material: substance such as wood, glass, metal, fabric.\n"
                "- action: verb/action/event.\n"
                "- generic_category: broad parent class too vague for final object memory, e.g. furniture, equipment, object.\n"
                "- uncertain: unclear whether the tag is an independent countable object.\n\n"
                "Decision tests:\n"
                "- Keep only tags that refer to one or more separable object instances that a person could count individually.\n"
                "- Hold tags that describe continuous scene surfaces, room envelope parts, background regions, global layout, substances, properties, or broad groups.\n"
                "- Hold uncertain tags; do not keep a tag just because a detector might draw a large mask for it.\n\n"
                "Use action='keep' only for physical_object. "
                "Use action='hold' for all other categories. "
                "Do not invent new tags. Keep reasons short.\n\n"
                "Return JSON with this schema:\n"
                "{\n"
                '  "items": [\n'
                '    {"tag": "...", "category": "...", "is_object_prompt": true, '
                '"action": "keep", "normalized_label": "...", "reason": "..."}\n'
                "  ]\n"
                "}\n\n"
                f"Tags: {json.dumps(tags, ensure_ascii=False)}"
            ),
        },
    ]


def call_openai_chat(
    tags: list[str],
    model: str,
    api_key: str,
    base_url: str,
    timeout_sec: int,
    headers: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    body = {
        "model": model,
        "messages": build_llm_messages(tags),
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        url=base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) SplatGraph/0.1",
            **(headers or {}),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API HTTP error {exc.code}: {details}") from exc

    content = payload["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    items = parsed.get("items")
    if not isinstance(items, list):
        raise ValueError(f"LLM response missing items list: {content}")
    return items


def normalize_classification(item: dict[str, Any], fallback_tag: str) -> dict[str, Any]:
    tag = normalize_tag(str(item.get("tag") or fallback_tag))
    category = str(item.get("category") or "uncertain").strip()
    if category not in VALID_CATEGORIES:
        category = "uncertain"
    is_object_prompt = bool(item.get("is_object_prompt", category in KEEP_CATEGORIES))
    action = str(item.get("action") or ("keep" if is_object_prompt else "hold"))
    if action not in {"keep", "hold"}:
        action = "keep" if is_object_prompt else "hold"
    if category not in KEEP_CATEGORIES:
        is_object_prompt = False
        action = "hold"
    normalized_label = item.get("normalized_label")
    if normalized_label is not None:
        normalized_label = normalize_tag(str(normalized_label))
    elif is_object_prompt:
        normalized_label = tag
    return {
        "tag": tag,
        "category": category,
        "is_object_prompt": is_object_prompt,
        "action": action,
        "normalized_label": normalized_label,
        "reason": str(item.get("reason") or ""),
        "classifier": str(item.get("classifier") or "llm"),
    }


def classify_tags(
    tags: list[str],
    classifier: str,
    cache: dict[str, Any],
    model: str,
    api_key: str | None,
    base_url: str,
    timeout_sec: int,
    headers: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    tags = unique_preserve_order(tags)
    cache_key_prefix = f"{classifier}:{model}:{PROMPT_POLICY_VERSION}:"
    results_by_tag: dict[str, dict[str, Any]] = {}
    missing = []

    for tag in tags:
        key = cache_key_prefix + tag
        if key in cache:
            results_by_tag[tag] = normalize_classification(cache[key], tag)
            results_by_tag[tag]["from_cache"] = True
        else:
            missing.append(tag)

    if missing:
        if classifier == "rule":
            new_items = [rule_classify(tag) for tag in missing]
        elif classifier == "llm":
            if not api_key:
                raise RuntimeError("LLM classifier requires an API key. Set OPENAI_API_KEY or pass --api_key_env.")
            raw_items = call_openai_chat(missing, model, api_key, base_url, timeout_sec, headers=headers)
            by_returned_tag = {
                normalize_tag(str(item.get("tag", ""))): item
                for item in raw_items
                if isinstance(item, dict)
            }
            new_items = []
            for tag in missing:
                raw_item = by_returned_tag.get(tag)
                if raw_item is None:
                    raw_item = rule_classify(tag)
                    raw_item["classifier"] = "rule_fallback_missing_llm_item"
                new_items.append(normalize_classification(raw_item, tag))
        else:
            raise ValueError(f"Unsupported classifier: {classifier}")

        for item in new_items:
            tag = item["tag"]
            item["from_cache"] = False
            results_by_tag[tag] = item
            cache[cache_key_prefix + tag] = item

    return [results_by_tag[tag] for tag in tags]


def write_filter_outputs(output_dir: Path, record: dict[str, Any], classifications: list[dict[str, Any]]) -> dict[str, Any]:
    frame_stem = str(record["frame_stem"])
    object_prompts = [
        str(item["normalized_label"] or item["tag"])
        for item in classifications
        if item["action"] == "keep" and item["is_object_prompt"]
    ]
    object_prompts = unique_preserve_order(object_prompts)
    held_out_prompts = [item["tag"] for item in classifications if item["action"] != "keep"]

    result = {
        "frame_stem": frame_stem,
        "frame_id": record.get("frame_id"),
        "image_path": record.get("image_path"),
        "input_prompts": record["prompts"],
        "object_prompts": object_prompts,
        "held_out_prompts": held_out_prompts,
        "classified_tags": {item["tag"]: item for item in classifications},
    }

    json_path = output_dir / f"{frame_stem}_prompt_filter.json"
    prompts_path = output_dir / f"{frame_stem}_object_prompts.txt"
    held_path = output_dir / f"{frame_stem}_held_out_prompts.txt"
    write_json(json_path, result)
    prompts_path.write_text(",".join(object_prompts) + "\n", encoding="utf-8")
    held_path.write_text(",".join(held_out_prompts) + "\n", encoding="utf-8")

    result["outputs"] = {
        "prompt_filter_json": str(json_path),
        "object_prompts_txt": str(prompts_path),
        "held_out_prompts_txt": str(held_path),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter RAM/RAM++ tags into object prompts")
    parser.add_argument("--auto_prompt_json", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--llm_config", type=Path, default=None, help="Local JSON config with model/base_url/api_key. Do not commit it.")
    parser.add_argument("--classifier", choices=("llm", "rule"), default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--timeout_sec", type=int, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    args = parser.parse_args()

    config = load_llm_config(args.llm_config)
    classifier = args.classifier or str(config.get("classifier") or "llm")
    if classifier not in {"llm", "rule"}:
        raise ValueError(f"Unsupported classifier in config: {classifier}")
    model = args.model or str(config.get("model") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini")
    base_url = args.base_url or str(config.get("base_url") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1")
    timeout_sec = int(args.timeout_sec if args.timeout_sec is not None else config.get("timeout_sec", 60))
    cache_path = args.cache or Path(str(config.get("cache") or DEFAULT_CACHE))
    api_key = str(config.get("api_key") or os.environ.get(args.api_key_env) or "")
    config_headers = config.get("headers") if isinstance(config.get("headers"), dict) else {}
    headers = {str(k): str(v) for k, v in config_headers.items()}

    auto_prompt_data = read_json(args.auto_prompt_json)
    records = extract_prompt_records(auto_prompt_data)
    output_dir = make_output_dir(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = load_cache(cache_path)

    print(f"[classifier] {classifier}")
    print(f"[model] {model}")
    print(f"[input] {args.auto_prompt_json}")
    print(f"[output] {output_dir}")
    print(f"[records] {len(records)}")

    start = time.perf_counter()
    results = []
    for record in records:
        classifications = classify_tags(
            record["prompts"],
            classifier=classifier,
            cache=cache,
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout_sec=timeout_sec,
            headers=headers,
        )
        results.append(write_filter_outputs(output_dir, record, classifications))

    save_cache(cache_path, cache)

    run_config = {
        "script": "auto_prompt_filter.py",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(args.auto_prompt_json),
        "output": str(output_dir),
        "classifier": classifier,
        "model": model,
        "prompt_policy_version": PROMPT_POLICY_VERSION,
        "base_url": base_url,
        "llm_config": str(args.llm_config) if args.llm_config else None,
        "api_key_source": "config" if config.get("api_key") else args.api_key_env,
        "headers": sorted(headers.keys()),
        "cache": str(cache_path),
        "elapsed_sec": round(time.perf_counter() - start, 6),
        "results": results,
    }
    write_json(output_dir / "prompt_filter.json", run_config)

    summary = ["Auto prompt filter summary", ""]
    for result in results:
        summary.append(f"{result['frame_stem']}:")
        summary.append(f"  keep: {', '.join(result['object_prompts']) if result['object_prompts'] else '(none)'}")
        summary.append(f"  hold: {', '.join(result['held_out_prompts']) if result['held_out_prompts'] else '(none)'}")
    summary.append("")
    summary.append(f"output: {output_dir}")
    (output_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    for result in results:
        print(f"[{result['frame_stem']}] keep={result['object_prompts']} hold={result['held_out_prompts']}")
    print(f"[done] outputs at {output_dir}")


if __name__ == "__main__":
    main()
