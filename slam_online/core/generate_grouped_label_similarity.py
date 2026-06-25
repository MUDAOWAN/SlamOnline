#!/usr/bin/env python3
"""
Generate label similarity for object memory with a grouped LLM workflow.

Instead of asking the model to score every possible label pair, this script:

1. Collects labels from frame_3d_observations.json.
2. Asks an LLM to group semantically related labels.
3. Scores only label pairs inside each group.

The final label_similarity.json keeps the same "pairs" format consumed by:

  object_memory_update.py --label_similarity_json
"""

from __future__ import annotations

import argparse
import csv
import itertools
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
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "label_similarity_grouped"
DEFAULT_LLM_CONFIG = PROJECT_ROOT / "configs" / "llm_prompt_filter.json"
DEFAULT_CACHE = DEFAULT_OUTPUT_ROOT / "grouped_label_similarity_cache.json"

VALID_GROUP_TYPES = {
    "alias_family",
    "object_family",
    "part_whole",
    "related_structure",
    "other_related",
}
VALID_RELATIONS = {
    "same_object_alias",
    "near_synonym",
    "parent_child",
    "part_of",
    "related_but_distinct",
    "different",
}
VALID_POLICIES = {"allow", "cautious", "hold"}


def normalize_label(label: str) -> str:
    label = label.strip().lower().replace("_", " ")
    label = re.sub(r"\s+", " ", label)
    label = re.sub(r"^(a|an|the)\s+", "", label)
    return label


def pair_key(label_a: str, label_b: str) -> str:
    left, right = sorted([normalize_label(label_a), normalize_label(label_b)])
    return f"{left}|{right}"


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
    return next_available_dir(output_root, f"grouped_similarity_{timestamp}")


def load_llm_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(
            f"LLM config not found: {path}. Copy configs/llm_prompt_filter.example.json "
            "to configs/llm_prompt_filter.json and fill in local API settings."
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


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
    return parsed


def extract_labels(data: Any, min_count: int) -> tuple[list[str], dict[str, int]]:
    if not isinstance(data, dict) or not isinstance(data.get("observations"), list):
        raise ValueError("Expected frame_3d_observations.json with an observations list")

    counts: dict[str, int] = {}
    for obs in data["observations"]:
        if not isinstance(obs, dict):
            continue
        label = normalize_label(str(obs.get("label") or ""))
        if not label:
            continue
        counts[label] = counts.get(label, 0) + 1

    labels = sorted(label for label, count in counts.items() if count >= min_count)
    return labels, counts


def call_openai_compatible_chat(
    messages: list[dict[str, str]],
    model: str,
    api_key: str,
    base_url: str,
    timeout_sec: int,
    headers: dict[str, str],
    retries: int,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    request_headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) SplatGraph/0.1",
        **headers,
    }

    last_error: Exception | None = None
    for attempt in range(max(1, retries + 1)):
        request = urllib.request.Request(
            url=base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=request_headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                payload = json.loads(response.read().decode("utf-8"))
            content = payload["choices"][0]["message"]["content"]
            return extract_json_object(content)
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"OpenAI-compatible API HTTP error {exc.code}: {details}")
        except Exception as exc:  # noqa: BLE001 - preserve original API/parsing context.
            last_error = exc
        if attempt < retries:
            time.sleep(1.0 + attempt)
    assert last_error is not None
    raise last_error


def build_group_messages(labels: list[str], label_counts: dict[str, int]) -> list[dict[str, str]]:
    label_payload = [{"label": label, "count": label_counts.get(label, 0)} for label in labels]
    return [
        {
            "role": "system",
            "content": (
                "You group open-vocabulary object labels for a 3D object memory system. "
                "Your grouping is only used to decide which pairs deserve later scoring; "
                "it is not a merge decision. Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                "Group labels into small semantic families.\n\n"
                "Rules:\n"
                "- Put labels together only if they may be aliases, near synonyms, parent/child labels, "
                "part/whole labels, or closely related scene structures.\n"
                "- Prefer precise groups. Do not create broad groups like all furniture unless labels "
                "really need pairwise comparison.\n"
                "- Leave unrelated labels ungrouped.\n"
                "- Use only provided labels. Do not invent labels.\n"
                "- Counts are observation counts, not semantic importance.\n\n"
                "Valid group_type values:\n"
                "alias_family, object_family, part_whole, related_structure, other_related.\n\n"
                "Return JSON with this schema:\n"
                "{\n"
                '  "groups": [\n'
                '    {"group_name": "...", "group_type": "object_family", '
                '"labels": ["..."], "reason": "..."}\n'
                "  ],\n"
                '  "ungrouped": ["..."]\n'
                "}\n\n"
                f"Labels: {json.dumps(label_payload, ensure_ascii=False)}"
            ),
        },
    ]


def build_pair_messages(pairs: list[tuple[str, str]], groups: list[dict[str, Any]]) -> list[dict[str, str]]:
    group_by_pair: dict[str, dict[str, str]] = {}
    for group in groups:
        labels = sorted(set(group["labels"]))
        for left, right in itertools.combinations(labels, 2):
            group_by_pair[pair_key(left, right)] = {
                "group_id": str(group["group_id"]),
                "group_name": str(group["group_name"]),
                "group_type": str(group["group_type"]),
            }

    pair_payload = []
    for left, right in pairs:
        pair_payload.append(
            {
                "label_a": left,
                "label_b": right,
                **group_by_pair.get(pair_key(left, right), {}),
            }
        )

    return [
        {
            "role": "system",
            "content": (
                "You score candidate object-label pairs for a 3D object memory system. "
                "High similarity means labels may refer to the same physical object if "
                "3D geometric evidence also agrees. Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                "For each candidate pair, classify relation into exactly one of:\n"
                "same_object_alias, near_synonym, parent_child, part_of, related_but_distinct, different.\n\n"
                "Definitions:\n"
                "- same_object_alias: common names for the same object type, e.g. couch/sofa.\n"
                "- near_synonym: often interchangeable in indoor scenes, but not always exact.\n"
                "- parent_child: one label is broader than the other, e.g. chair/armchair.\n"
                "- part_of: one is a component or attached sub-object of the other.\n"
                "- related_but_distinct: related but should usually remain separate objects.\n"
                "- different: not semantically related enough to help object merging.\n\n"
                "Assign similarity in [0, 1]. Suggested ranges:\n"
                "- same_object_alias: 0.90-1.00\n"
                "- near_synonym: 0.75-0.90\n"
                "- parent_child: 0.55-0.75\n"
                "- part_of: 0.45-0.65\n"
                "- related_but_distinct: 0.25-0.45\n"
                "- different: 0.00-0.25\n\n"
                "Use merge_policy as allow, cautious, or hold. Keep reasons short. "
                "Use only the provided labels.\n\n"
                "Return JSON with this schema:\n"
                "{\n"
                '  "items": [\n'
                '    {"label_a": "...", "label_b": "...", "relation": "...", '
                '"similarity": 0.0, "merge_policy": "allow", "reason": "..."}\n'
                "  ]\n"
                "}\n\n"
                f"Candidate pairs: {json.dumps(pair_payload, ensure_ascii=False)}"
            ),
        },
    ]


def normalize_groups(raw: dict[str, Any], labels: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    valid_labels = set(labels)
    groups: list[dict[str, Any]] = []
    seen_in_groups: set[str] = set()

    raw_groups = raw.get("groups", [])
    if not isinstance(raw_groups, list):
        raw_groups = []

    for group in raw_groups:
        if not isinstance(group, dict):
            continue
        group_labels: list[str] = []
        raw_labels = group.get("labels", [])
        if not isinstance(raw_labels, list):
            continue
        for label in raw_labels:
            normalized = normalize_label(str(label))
            if normalized in valid_labels and normalized not in group_labels:
                group_labels.append(normalized)
        if len(group_labels) < 2:
            continue

        group_type = str(group.get("group_type") or "object_family").strip()
        if group_type not in VALID_GROUP_TYPES:
            group_type = "other_related"

        groups.append(
            {
                "group_id": f"group_{len(groups):03d}",
                "group_name": str(group.get("group_name") or f"group_{len(groups):03d}"),
                "group_type": group_type,
                "labels": sorted(group_labels),
                "reason": str(group.get("reason") or ""),
            }
        )
        seen_in_groups.update(group_labels)

    ungrouped: list[str] = []
    raw_ungrouped = raw.get("ungrouped", [])
    if isinstance(raw_ungrouped, list):
        for label in raw_ungrouped:
            normalized = normalize_label(str(label))
            if normalized in valid_labels and normalized not in ungrouped:
                ungrouped.append(normalized)

    for label in labels:
        if label not in seen_in_groups and label not in ungrouped:
            ungrouped.append(label)

    return groups, sorted(ungrouped)


def dry_run_groups(labels: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    groups: list[dict[str, Any]] = []
    used: set[str] = set()

    for idx, label in enumerate(labels):
        if label in used:
            continue
        label_tokens = set(label.split())
        related = [label]
        for other in labels[idx + 1 :]:
            other_tokens = set(other.split())
            if label in other or other in label or (label_tokens & other_tokens):
                related.append(other)
        if len(related) >= 2:
            for item in related:
                used.add(item)
            groups.append(
                {
                    "group_id": f"group_{len(groups):03d}",
                    "group_name": f"dry_run_related_{len(groups):03d}",
                    "group_type": "other_related",
                    "labels": sorted(related),
                    "reason": "dry_run lexical overlap group",
                }
            )

    ungrouped = sorted(label for label in labels if label not in used)
    return groups, ungrouped


def pairs_from_groups(groups: list[dict[str, Any]], max_group_size: int) -> list[tuple[str, str]]:
    seen: set[str] = set()
    pairs: list[tuple[str, str]] = []

    for group in groups:
        labels = sorted(set(group["labels"]))
        if max_group_size > 0 and len(labels) > max_group_size:
            labels = labels[:max_group_size]
        for left, right in itertools.combinations(labels, 2):
            key = pair_key(left, right)
            if key in seen:
                continue
            seen.add(key)
            pairs.append((left, right))
    return pairs


def default_score_for_relation(relation: str) -> tuple[float, str]:
    if relation == "same_object_alias":
        return 0.95, "allow"
    if relation == "near_synonym":
        return 0.85, "allow"
    if relation == "parent_child":
        return 0.65, "cautious"
    if relation == "part_of":
        return 0.55, "cautious"
    if relation == "related_but_distinct":
        return 0.35, "hold"
    return 0.2, "hold"


def normalize_similarity_item(item: dict[str, Any], fallback_pair: tuple[str, str]) -> dict[str, Any]:
    label_a = normalize_label(str(item.get("label_a") or fallback_pair[0]))
    label_b = normalize_label(str(item.get("label_b") or fallback_pair[1]))

    relation = str(item.get("relation") or "different").strip()
    if relation not in VALID_RELATIONS:
        relation = "different"

    default_similarity, default_policy = default_score_for_relation(relation)
    try:
        similarity = float(item.get("similarity", default_similarity))
    except (TypeError, ValueError):
        similarity = default_similarity
    similarity = max(0.0, min(1.0, similarity))

    merge_policy = str(item.get("merge_policy") or default_policy).strip()
    if merge_policy not in VALID_POLICIES:
        merge_policy = default_policy

    return {
        "label_a": label_a,
        "label_b": label_b,
        "pair_key": pair_key(label_a, label_b),
        "relation": relation,
        "similarity": round(similarity, 6),
        "merge_policy": merge_policy,
        "reason": str(item.get("reason") or ""),
    }


def dry_run_similarity_item(pair: tuple[str, str]) -> dict[str, Any]:
    left, right = pair
    left_tokens = set(left.split())
    right_tokens = set(right.split())

    if left == right:
        relation = "same_object_alias"
    elif left in right or right in left:
        relation = "parent_child"
    elif left_tokens & right_tokens:
        relation = "related_but_distinct"
    else:
        relation = "different"

    similarity, policy = default_score_for_relation(relation)
    return normalize_similarity_item(
        {
            "label_a": left,
            "label_b": right,
            "relation": relation,
            "similarity": similarity,
            "merge_policy": policy,
            "reason": "dry_run lexical placeholder",
        },
        pair,
    )


def get_groups(
    labels: list[str],
    label_counts: dict[str, int],
    cache: dict[str, Any],
    model: str,
    api_key: str,
    base_url: str,
    timeout_sec: int,
    headers: dict[str, str],
    retries: int,
    dry_run: bool,
    reuse_groups_json: Path | None,
) -> tuple[list[dict[str, Any]], list[str], str]:
    if reuse_groups_json is not None:
        raw = read_json(reuse_groups_json)
        groups, ungrouped = normalize_groups(raw, labels)
        return groups, ungrouped, "reused"

    cache_key = f"label_groups:{model}:{','.join(labels)}"
    if cache_key in cache:
        groups, ungrouped = normalize_groups(cache[cache_key], labels)
        return groups, ungrouped, "cache"

    if dry_run:
        groups, ungrouped = dry_run_groups(labels)
        cache[cache_key] = {"groups": groups, "ungrouped": ungrouped}
        return groups, ungrouped, "dry_run"

    if not api_key:
        raise RuntimeError("Grouped label generation requires an API key in config or OPENAI_API_KEY.")

    raw = call_openai_compatible_chat(
        messages=build_group_messages(labels, label_counts),
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout_sec=timeout_sec,
        headers=headers,
        retries=retries,
    )
    groups, ungrouped = normalize_groups(raw, labels)
    cache[cache_key] = {"groups": groups, "ungrouped": ungrouped}
    return groups, ungrouped, "llm"


def score_pairs(
    pairs: list[tuple[str, str]],
    groups: list[dict[str, Any]],
    cache: dict[str, Any],
    model: str,
    api_key: str,
    base_url: str,
    timeout_sec: int,
    headers: dict[str, str],
    retries: int,
    batch_size: int,
    dry_run: bool,
) -> list[dict[str, Any]]:
    cache_prefix = f"grouped_label_similarity:{model}:"
    results: dict[str, dict[str, Any]] = {}
    missing: list[tuple[str, str]] = []

    for pair in pairs:
        key = pair_key(*pair)
        cache_key = cache_prefix + key
        if cache_key in cache:
            item = normalize_similarity_item(cache[cache_key], pair)
            item["from_cache"] = True
            results[key] = item
        else:
            missing.append(pair)

    if dry_run:
        for pair in missing:
            item = dry_run_similarity_item(pair)
            item["from_cache"] = False
            results[item["pair_key"]] = item
        return [results[pair_key(*pair)] for pair in pairs]

    if missing and not api_key:
        raise RuntimeError("Pair scoring requires an API key in config or OPENAI_API_KEY.")

    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        raw = call_openai_compatible_chat(
            messages=build_pair_messages(batch, groups),
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout_sec=timeout_sec,
            headers=headers,
            retries=retries,
        )
        raw_items = raw.get("items")
        if not isinstance(raw_items, list):
            raise ValueError(f"Pair scoring response missing items list: {raw}")

        returned: dict[str, dict[str, Any]] = {}
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            left = raw_item.get("label_a")
            right = raw_item.get("label_b")
            if left is None or right is None:
                continue
            returned[pair_key(str(left), str(right))] = raw_item

        for pair in batch:
            key = pair_key(*pair)
            raw_item = returned.get(key)
            if raw_item is None:
                raw_item = {
                    "label_a": pair[0],
                    "label_b": pair[1],
                    "relation": "related_but_distinct",
                    "similarity": 0.35,
                    "merge_policy": "hold",
                    "reason": "missing from model response",
                }
            item = normalize_similarity_item(raw_item, pair)
            item["from_cache"] = False
            results[key] = item
            cache[cache_prefix + key] = item
        time.sleep(0.1)

    return [results[pair_key(*pair)] for pair in pairs]


def write_items_csv(path: Path, items: list[dict[str, Any]], label_counts: dict[str, int]) -> None:
    fieldnames = [
        "label_a",
        "label_b",
        "count_a",
        "count_b",
        "relation",
        "similarity",
        "merge_policy",
        "reason",
        "from_cache",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "label_a": item["label_a"],
                    "label_b": item["label_b"],
                    "count_a": label_counts.get(item["label_a"], 0),
                    "count_b": label_counts.get(item["label_b"], 0),
                    "relation": item["relation"],
                    "similarity": item["similarity"],
                    "merge_policy": item["merge_policy"],
                    "reason": item["reason"],
                    "from_cache": item.get("from_cache", False),
                }
            )


def write_groups_csv(path: Path, groups: list[dict[str, Any]], label_counts: dict[str, int]) -> None:
    fieldnames = ["group_id", "group_name", "group_type", "label", "label_count", "reason"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for group in groups:
            for label in group["labels"]:
                writer.writerow(
                    {
                        "group_id": group["group_id"],
                        "group_name": group["group_name"],
                        "group_type": group["group_type"],
                        "label": label,
                        "label_count": label_counts.get(label, 0),
                        "reason": group["reason"],
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate grouped LLM label similarity for object memory")
    parser.add_argument("--frame_3d_observations_json", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--llm_config", type=Path, default=DEFAULT_LLM_CONFIG)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--timeout_sec", type=int, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=80)
    parser.add_argument("--min_label_count", type=int, default=1)
    parser.add_argument("--min_output_similarity", type=float, default=0.0)
    parser.add_argument("--max_group_size", type=int, default=0)
    parser.add_argument("--reuse_groups_json", type=Path, default=None)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    config = load_llm_config(args.llm_config)
    model = args.model or str(config.get("model") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini")
    base_url = args.base_url or str(config.get("base_url") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1")
    timeout_sec = int(args.timeout_sec if args.timeout_sec is not None else config.get("timeout_sec", 60))
    api_key = str(config.get("api_key") or os.environ.get(args.api_key_env) or "")
    config_headers = config.get("headers") if isinstance(config.get("headers"), dict) else {}
    headers = {str(key): str(value) for key, value in config_headers.items()}
    cache_path = args.cache or Path(str(config.get("grouped_label_similarity_cache") or DEFAULT_CACHE))

    data = read_json(args.frame_3d_observations_json)
    labels, label_counts = extract_labels(data, args.min_label_count)
    all_pair_count = len(labels) * (len(labels) - 1) // 2
    cache = load_cache(cache_path)

    output_dir = make_output_dir(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[labels] {len(labels)}")
    print(f"[all_pairs_if_naive] {all_pair_count}")
    print(f"[model] {model}")
    print(f"[dry_run] {args.dry_run}")
    print(f"[output] {output_dir}")

    groups, ungrouped, group_source = get_groups(
        labels=labels,
        label_counts=label_counts,
        cache=cache,
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout_sec=timeout_sec,
        headers=headers,
        retries=max(0, int(args.retries)),
        dry_run=args.dry_run,
        reuse_groups_json=args.reuse_groups_json,
    )
    candidate_pairs = pairs_from_groups(groups, args.max_group_size)

    print(f"[group_source] {group_source}")
    print(f"[groups] {len(groups)}")
    print(f"[ungrouped] {len(ungrouped)}")
    print(f"[candidate_pairs] {len(candidate_pairs)}")

    items = score_pairs(
        pairs=candidate_pairs,
        groups=groups,
        cache=cache,
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout_sec=timeout_sec,
        headers=headers,
        retries=max(0, int(args.retries)),
        batch_size=max(1, int(args.batch_size)),
        dry_run=args.dry_run,
    )
    write_json(cache_path, cache)

    filtered_items = [item for item in items if float(item["similarity"]) >= args.min_output_similarity]
    pairs_map = {item["pair_key"]: item["similarity"] for item in filtered_items}
    relation_counts: dict[str, int] = {}
    for item in filtered_items:
        relation_counts[item["relation"]] = relation_counts.get(item["relation"], 0) + 1

    payload = {
        "format": "grouped_label_similarity_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metadata": {
            "frame_3d_observations_json": str(args.frame_3d_observations_json),
            "model": model,
            "base_url": base_url,
            "llm_config": str(args.llm_config) if args.llm_config else None,
            "headers": sorted(headers.keys()),
            "cache": str(cache_path),
            "dry_run": args.dry_run,
            "group_source": group_source,
            "min_label_count": args.min_label_count,
            "min_output_similarity": args.min_output_similarity,
            "max_group_size": args.max_group_size,
            "all_pair_count": all_pair_count,
            "candidate_pair_count": len(candidate_pairs),
        },
        "labels": labels,
        "label_counts": label_counts,
        "groups": groups,
        "ungrouped": ungrouped,
        "items": filtered_items,
        "pairs": pairs_map,
    }

    semantic_groups = {
        "format": "semantic_label_groups_v1",
        "created_at": payload["created_at"],
        "metadata": {
            "frame_3d_observations_json": str(args.frame_3d_observations_json),
            "model": model,
            "dry_run": args.dry_run,
            "group_source": group_source,
        },
        "labels": labels,
        "label_counts": label_counts,
        "groups": groups,
        "ungrouped": ungrouped,
    }

    write_json(output_dir / "label_similarity.json", payload)
    write_json(output_dir / "semantic_groups.json", semantic_groups)
    write_items_csv(output_dir / "label_similarity.csv", filtered_items, label_counts)
    write_groups_csv(output_dir / "semantic_groups.csv", groups, label_counts)

    top_items = sorted(filtered_items, key=lambda item: (-float(item["similarity"]), item["label_a"], item["label_b"]))[:30]
    lines = [
        "Grouped label similarity summary",
        "",
        f"labels: {len(labels)}",
        f"all_pairs_if_naive: {all_pair_count}",
        f"groups: {len(groups)}",
        f"ungrouped: {len(ungrouped)}",
        f"candidate_pairs: {len(candidate_pairs)}",
        f"written_pairs: {len(filtered_items)}",
        f"dry_run: {args.dry_run}",
        f"group_source: {group_source}",
        f"relation_counts: {dict(sorted(relation_counts.items()))}",
        "",
        "groups:",
    ]
    for group in groups:
        labels_text = ", ".join(group["labels"])
        lines.append(f"- {group['group_id']} {group['group_name']} [{group['group_type']}]: {labels_text}")
    if ungrouped:
        lines.extend(["", f"ungrouped: {', '.join(ungrouped)}"])
    lines.append("")
    lines.append("top similarities:")
    for item in top_items:
        lines.append(
            f"- {item['label_a']} | {item['label_b']}: {item['similarity']} "
            f"{item['relation']} ({item['merge_policy']})"
        )
    lines.extend(["", f"output: {output_dir}"])
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[written_pairs] {len(filtered_items)}")
    print(f"[relation_counts] {dict(sorted(relation_counts.items()))}")
    print(f"[label_similarity_json] {output_dir / 'label_similarity.json'}")
    print(f"[done] outputs at {output_dir}")


if __name__ == "__main__":
    main()
