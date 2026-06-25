#!/usr/bin/env python3
"""
Generate detector-first prompt masks with Grounding DINO + SAM 2.

This worker consumes the same queue task format as online_semantic_worker.py,
but avoids SAM+CLIP's "always pick the most relevant region" failure mode:

  prompt -> Grounding DINO boxes -> SAM 2 masks -> queue-compatible mask paths

If Grounding DINO returns no boxes for a prompt, the worker writes an empty mask.
That makes absent prompts such as "door" produce no 3D votes instead of a large
false-positive mask.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUEUE_ROOT = PROJECT_ROOT / "output" / "queue"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "online_grounded_sam2_2d"
DEFAULT_PROMPTS = "couch,pillow,potted plant,lamp,door"
DEFAULT_GROUNDING_MODEL_ID = "IDEA-Research/grounding-dino-base"
DEFAULT_SAM2_MODEL_ID = "facebook/sam2-hiera-base-plus"


def slugify(text: str) -> str:
    return text.strip().lower().replace(" ", "_").replace("/", "_")


def parse_prompts(value: str) -> list[str]:
    prompts = [prompt.strip() for prompt in value.split(",") if prompt.strip()]
    if not prompts:
        raise ValueError("At least one prompt is required")
    return prompts


def unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        out.append(normalized)
        seen.add(normalized)
    return out


def split_prompt_text(value: str) -> list[str]:
    pieces = re.split(r"\s*\|\s*|\s*,\s*|\s*;\s*|\n+", value)
    return [piece.strip() for piece in pieces if piece.strip()]


def load_prompts_file(path: Path, frame_stem: str | None = None) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")

    if path.suffix.lower() == ".json":
        data = load_json(path)
        if isinstance(data, dict):
            if isinstance(data.get("object_prompts"), list):
                return unique_preserve_order([str(item) for item in data["object_prompts"]])
            if isinstance(data.get("prompts"), list):
                return unique_preserve_order([str(item) for item in data["prompts"]])
            if isinstance(data.get("raw_tags"), list):
                return unique_preserve_order([str(item) for item in data["raw_tags"]])
            if isinstance(data.get("scene_tags"), list):
                return unique_preserve_order([str(item) for item in data["scene_tags"]])
            if isinstance(data.get("scene_prompts"), list):
                return unique_preserve_order([str(item) for item in data["scene_prompts"]])
            if isinstance(data.get("results"), list):
                if frame_stem:
                    for item in data["results"]:
                        if str(item.get("frame_stem")) != frame_stem:
                            continue
                        for key in ("object_prompts", "prompts", "raw_tags", "candidate_tags"):
                            if isinstance(item.get(key), list):
                                return unique_preserve_order([str(prompt) for prompt in item[key]])
                prompts: list[str] = []
                for item in data["results"]:
                    if not isinstance(item, dict):
                        continue
                    for key in ("object_prompts", "prompts", "raw_tags", "candidate_tags"):
                        if isinstance(item.get(key), list):
                            prompts.extend(str(prompt) for prompt in item[key])
                            break
                if prompts:
                    return unique_preserve_order(prompts)
        if isinstance(data, list):
            return unique_preserve_order([str(item) for item in data])
        raise ValueError(f"Could not read prompts from JSON: {path}")

    return unique_preserve_order(split_prompt_text(path.read_text(encoding="utf-8")))


def load_per_frame_prompts(prompts_dir: Path, frame_stem: str) -> list[str]:
    candidates = [
        prompts_dir / f"{frame_stem}_object_prompts.txt",
        prompts_dir / f"{frame_stem}_prompt_filter.json",
        prompts_dir / f"{frame_stem}_auto_prompts.json",
    ]
    for path in candidates:
        if path.exists():
            return load_prompts_file(path, frame_stem=frame_stem)
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No per-frame prompts found for {frame_stem}. Searched: {searched}")


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
    return next_available_dir(output_root, f"worker_{timestamp}")


def make_single_output_dir(output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return next_available_dir(output_root, f"single_{timestamp}")


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(path)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def ensure_queue_dirs(queue_root: Path) -> None:
    for name in ("pending", "processing", "done", "failed"):
        (queue_root / name).mkdir(parents=True, exist_ok=True)


def load_queue_tasks(queue_root: Path, max_tasks: int | None) -> list[tuple[Path, dict[str, Any]]]:
    ensure_queue_dirs(queue_root)
    pending_paths = sorted((queue_root / "pending").glob("*.json"))
    if max_tasks is not None and max_tasks > 0:
        pending_paths = pending_paths[:max_tasks]

    selected: list[tuple[Path, dict[str, Any]]] = []
    for pending_path in pending_paths:
        processing_path = queue_root / "processing" / pending_path.name
        pending_path.replace(processing_path)
        wrapper = load_json(processing_path)
        wrapper["queue_status"] = "processing"
        wrapper["processing_started_at"] = datetime.now().isoformat(timespec="seconds")
        write_json_atomic(processing_path, wrapper)
        selected.append((processing_path, wrapper))
    return selected


def finish_queue_task(
    queue_root: Path,
    processing_path: Path,
    wrapper: dict[str, Any],
    result: dict[str, Any],
) -> None:
    success = result.get("status") == "processed"
    final_state = "done" if success else "failed"
    wrapper["queue_status"] = final_state
    wrapper["finished_at"] = datetime.now().isoformat(timespec="seconds")
    wrapper["worker_result"] = result
    write_json_atomic(processing_path, wrapper)
    processing_path.replace(queue_root / final_state / processing_path.name)


def make_image_task(image_path: Path) -> dict[str, Any]:
    image_path = image_path.expanduser().resolve()
    frame_stem = image_path.stem
    match = re.search(r"(\d+)$", frame_stem)
    frame_id = int(match.group(1)) if match else 0
    return {
        "task_index": 0,
        "trajectory_index": 0,
        "monogs_frame_idx": 0,
        "frame_id": frame_id,
        "frame_stem": frame_stem,
        "is_init": False,
        "is_keyframe": False,
        "image_path": str(image_path),
        "image_relative_path": None,
        "image_exists": image_path.exists(),
        "depth_path": None,
        "depth_relative_path": None,
        "depth_exists": False,
        "timestamp": 0.0,
        "depth_timestamp": 0.0,
        "pose_c2w": None,
        "intrinsics": None,
        "width": None,
        "height": None,
        "depth_scale": 1.0,
    }


def import_grounding_dino():
    try:
        import torch.utils._pytree as torch_pytree

        if not hasattr(torch_pytree, "register_pytree_node") and hasattr(torch_pytree, "_register_pytree_node"):
            torch_pytree.register_pytree_node = torch_pytree._register_pytree_node
    except Exception:
        pass

    try:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError as exc:
        raise ImportError(
            "Missing Grounding DINO dependencies. Install transformers/accelerate first; "
            "see README.md Grounded SAM 2 section."
        ) from exc
    return AutoModelForZeroShotObjectDetection, AutoProcessor


def import_sam2_predictor():
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise ImportError(
            "Missing SAM 2 dependency. Install facebookresearch/sam2 first; "
            "see README.md Grounded SAM 2 section."
        ) from exc
    return SAM2ImagePredictor


class GroundingDinoDetector:
    def __init__(self, model_id: str, device: torch.device) -> None:
        AutoModelForZeroShotObjectDetection, AutoProcessor = import_grounding_dino()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.model.eval()
        self.device = device
        self.model_id = model_id

    @torch.no_grad()
    def detect(
        self,
        image_rgb_pil: Image.Image,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
        max_detections: int,
    ) -> list[dict[str, Any]]:
        text = prompt.strip()
        if not text.endswith("."):
            text += "."

        inputs = self.processor(images=image_rgb_pil, text=text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        target_size = torch.tensor([image_rgb_pil.size[::-1]], device=self.device)
        postprocess = self.processor.post_process_grounded_object_detection
        postprocess_kwargs = {
            "input_ids": inputs.get("input_ids"),
            "text_threshold": text_threshold,
            "target_sizes": target_size,
        }
        if "box_threshold" in inspect.signature(postprocess).parameters:
            postprocess_kwargs["box_threshold"] = box_threshold
        else:
            postprocess_kwargs["threshold"] = box_threshold
        post = postprocess(outputs, **postprocess_kwargs)[0]

        boxes = post.get("boxes", torch.empty((0, 4), device=self.device)).detach().cpu().numpy()
        scores = post.get("scores", torch.empty((0,), device=self.device)).detach().cpu().numpy()
        labels = post.get("labels", [""] * len(boxes))

        detections = []
        for det_idx, (box, score) in enumerate(zip(boxes, scores)):
            x0, y0, x1, y1 = [float(x) for x in box.tolist()]
            detections.append(
                {
                    "det_index": int(det_idx),
                    "prompt": prompt,
                    "label": str(labels[det_idx]) if det_idx < len(labels) else prompt,
                    "box_xyxy": [round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)],
                    "box_score": round(float(score), 6),
                }
            )
        detections.sort(key=lambda item: item["box_score"], reverse=True)
        return detections[:max_detections]


class Sam2BoxSegmenter:
    def __init__(self, model_id: str, device: torch.device) -> None:
        SAM2ImagePredictor = import_sam2_predictor()
        try:
            self.predictor = SAM2ImagePredictor.from_pretrained(model_id, device=str(device))
        except TypeError:
            self.predictor = SAM2ImagePredictor.from_pretrained(model_id)
            if hasattr(self.predictor, "model"):
                self.predictor.model.to(device)
        self.model_id = model_id
        self.device = device

    def segment_boxes(self, image_rgb: np.ndarray, boxes_xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(boxes_xyxy) == 0:
            h, w = image_rgb.shape[:2]
            return np.zeros((0, h, w), dtype=bool), np.zeros((0,), dtype=np.float32)

        self.predictor.set_image(image_rgb)
        masks, scores, _logits = self.predictor.predict(
            box=boxes_xyxy.astype(np.float32),
            multimask_output=False,
        )
        masks_np = np.asarray(masks)
        if masks_np.ndim == 4:
            masks_np = masks_np[:, 0]
        elif masks_np.ndim == 2:
            masks_np = masks_np[None, ...]

        scores_np = np.asarray(scores, dtype=np.float32).reshape(-1)
        if len(scores_np) != len(masks_np):
            scores_np = np.ones((len(masks_np),), dtype=np.float32)
        return masks_np.astype(bool), scores_np


def draw_overlay(
    image_bgr: np.ndarray,
    prompt_detections: list[dict[str, Any]],
    combined_mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    overlay = image_bgr.copy()
    mask_color = np.zeros_like(overlay)
    mask_color[combined_mask > 0] = color
    overlay = cv2.addWeighted(overlay, 1.0, mask_color, alpha, 0.0)
    for det in prompt_detections:
        x0, y0, x1, y1 = [int(round(x)) for x in det["box_xyxy"]]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)
        cv2.putText(
            overlay,
            f"{det['prompt']} {det['box_score']:.2f}",
            (max(x0, 0), max(y0 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return overlay


def detection_instance_id(frame_stem: str, slug: str, det_idx: int) -> str:
    return f"{frame_stem}:{slug}:det{det_idx:03d}"


def write_detection_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))


def process_task(
    task: dict[str, Any],
    output_root: Path,
    prompts: list[str],
    detector: GroundingDinoDetector,
    segmenter: Sam2BoxSegmenter,
    box_threshold: float,
    text_threshold: float,
    min_mask_score: float,
    max_detections_per_prompt: int,
    min_mask_area_ratio: float,
    max_mask_area_ratio: float,
    write_overlay: bool,
    overlay_alpha: float,
    write_instance_observations: bool,
) -> dict[str, Any]:
    start = time.perf_counter()
    frame_stem = str(task.get("frame_stem") or f"frame_{int(task['frame_id']):06d}")
    image_path = Path(str(task["image_path"])) if task.get("image_path") else None
    frame_out = output_root / frame_stem
    frame_out.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "frame_id": int(task["frame_id"]),
        "frame_stem": frame_stem,
        "image_path": str(image_path) if image_path else None,
        "status": "pending",
        "backend": "grounding_dino_sam2",
        "prompts": prompts,
        "outputs": {},
        "elapsed_sec": 0.0,
    }
    if image_path is None or not image_path.exists():
        result["status"] = "missing_image"
        result["elapsed_sec"] = round(time.perf_counter() - start, 6)
        return result

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        result["status"] = "failed_read_image"
        result["elapsed_sec"] = round(time.perf_counter() - start, 6)
        return result

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb_pil = Image.fromarray(image_rgb)
    height, width = image_rgb.shape[:2]

    prompt_outputs: dict[str, dict[str, Any]] = {}
    all_prompt_detections: dict[str, list[dict[str, Any]]] = {}
    prompt_validation: dict[str, dict[str, Any]] = {}
    accepted_observations: list[dict[str, Any]] = []
    rejected_observations: list[dict[str, Any]] = []
    for prompt_idx, prompt in enumerate(prompts):
        detections = detector.detect(
            image_rgb_pil=image_rgb_pil,
            prompt=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            max_detections=max_detections_per_prompt,
        )
        boxes = np.asarray([det["box_xyxy"] for det in detections], dtype=np.float32)
        masks, mask_scores = segmenter.segment_boxes(image_rgb, boxes)

        kept_detections = []
        combined = np.zeros((height, width), dtype=np.uint8)
        for det_idx, det in enumerate(detections):
            mask_score = float(mask_scores[det_idx]) if det_idx < len(mask_scores) else 1.0
            mask = masks[det_idx] if det_idx < len(masks) else np.zeros((height, width), dtype=bool)
            area_ratio = float(mask.mean()) if mask.size else 0.0
            det["mask_score"] = round(mask_score, 6)
            det["mask_area_ratio"] = round(area_ratio, 6)
            if mask_score < min_mask_score:
                det["kept"] = False
                det["reject_reason"] = "low_mask_score"
                continue
            if area_ratio < min_mask_area_ratio:
                det["kept"] = False
                det["reject_reason"] = "mask_area_too_small"
                continue
            if area_ratio > max_mask_area_ratio:
                det["kept"] = False
                det["reject_reason"] = "mask_area_too_large"
                continue
            det["kept"] = True
            det["instance_index"] = int(len(kept_detections))
            combined[mask] = 255
            kept_detections.append(det)

        slug = slugify(prompt)
        mask_path = frame_out / f"{slug}_mask.png"
        cv2.imwrite(str(mask_path), combined)

        detections_path = frame_out / f"{slug}_detections.json"
        detections_path.write_text(json.dumps(detections, indent=2), encoding="utf-8")

        outputs: dict[str, Any] = {
            "mask": str(mask_path),
            "detections_json": str(detections_path),
            "detection_count": len(detections),
            "kept_detection_count": len(kept_detections),
            "mask_area_ratio": round(float((combined > 0).mean()), 6),
        }
        if not detections:
            validation_status = "rejected"
            reject_reason = "no_detection"
        elif not kept_detections:
            validation_status = "rejected"
            reject_reasons = sorted(
                {
                    str(det.get("reject_reason", "not_kept"))
                    for det in detections
                    if not det.get("kept", False)
                }
            )
            reject_reason = ",".join(reject_reasons) if reject_reasons else "no_kept_detection"
        else:
            validation_status = "accepted"
            reject_reason = None
        outputs["validation_status"] = validation_status
        outputs["reject_reason"] = reject_reason
        if write_overlay:
            color = (
                int(70 + (prompt_idx * 67) % 180),
                int(110 + (prompt_idx * 43) % 140),
                int(90 + (prompt_idx * 89) % 160),
            )
            overlay = draw_overlay(image_bgr, kept_detections, combined, color, overlay_alpha)
            overlay_path = frame_out / f"{slug}_overlay.jpg"
            cv2.imwrite(str(overlay_path), overlay)
            outputs["overlay"] = str(overlay_path)

        prompt_outputs[prompt] = outputs
        all_prompt_detections[prompt] = detections
        prompt_validation[prompt] = {
            "status": validation_status,
            "reject_reason": reject_reason,
            "detection_count": len(detections),
            "kept_detection_count": len(kept_detections),
            "mask_area_ratio": outputs["mask_area_ratio"],
            "instance_observation_count": len(kept_detections) if write_instance_observations else (1 if validation_status == "accepted" else 0),
            "max_box_score": max((float(det["box_score"]) for det in detections), default=0.0),
            "max_mask_score": max((float(det.get("mask_score", 0.0)) for det in detections), default=0.0),
        }
        base_observation = {
            "frame_id": int(task["frame_id"]),
            "frame_stem": frame_stem,
            "image_path": str(image_path),
            "label": prompt,
            "label_slug": slug,
            "prompt": prompt,
            "prompt_mask_path": str(mask_path),
            "overlay_path": outputs.get("overlay"),
            "detections_json": str(detections_path),
            "detection_count": len(detections),
            "kept_detection_count": len(kept_detections),
            "max_box_score": prompt_validation[prompt]["max_box_score"],
            "max_mask_score": prompt_validation[prompt]["max_mask_score"],
        }
        if validation_status == "accepted" and write_instance_observations:
            for kept_idx, det in enumerate(kept_detections):
                det_idx = int(det.get("det_index", kept_idx))
                det_mask = masks[det_idx] if det_idx < len(masks) else np.zeros((height, width), dtype=bool)
                instance_slug = f"{slug}_det{kept_idx:03d}"
                instance_mask_path = frame_out / f"{instance_slug}_mask.png"
                write_detection_mask(instance_mask_path, det_mask)
                instance_overlay_path = None
                if write_overlay:
                    color = (
                        int(70 + ((prompt_idx + kept_idx) * 67) % 180),
                        int(110 + ((prompt_idx + kept_idx) * 43) % 140),
                        int(90 + ((prompt_idx + kept_idx) * 89) % 160),
                    )
                    instance_combined = np.zeros((height, width), dtype=np.uint8)
                    instance_combined[det_mask] = 255
                    instance_overlay = draw_overlay(image_bgr, [det], instance_combined, color, overlay_alpha)
                    instance_overlay_path = frame_out / f"{instance_slug}_overlay.jpg"
                    cv2.imwrite(str(instance_overlay_path), instance_overlay)
                accepted_observations.append(
                    {
                        **base_observation,
                        "observation_id": detection_instance_id(frame_stem, slug, kept_idx),
                        "label_slug": instance_slug,
                        "status": "accepted",
                        "reject_reason": None,
                        "mask_path": str(instance_mask_path),
                        "overlay_path": str(instance_overlay_path) if instance_overlay_path else outputs.get("overlay"),
                        "mask_area_ratio": float(det.get("mask_area_ratio", 0.0)),
                        "max_box_score": float(det.get("box_score", 0.0)),
                        "max_mask_score": float(det.get("mask_score", 0.0)),
                        "box_xyxy": det.get("box_xyxy"),
                        "det_index": det_idx,
                        "instance_index": kept_idx,
                        "is_instance_observation": True,
                        "kept_detections": [det],
                    }
                )
        elif validation_status == "accepted":
            accepted_observations.append(
                {
                    **base_observation,
                    "observation_id": f"{frame_stem}:{slug}",
                    "status": validation_status,
                    "reject_reason": reject_reason,
                    "mask_path": str(mask_path),
                    "mask_area_ratio": outputs["mask_area_ratio"],
                    "is_instance_observation": False,
                    "kept_detections": kept_detections,
                }
            )
        else:
            rejected_observations.append(
                {
                    **base_observation,
                    "observation_id": f"{frame_stem}:{slug}",
                    "status": validation_status,
                    "reject_reason": reject_reason,
                    "mask_path": str(mask_path),
                    "mask_area_ratio": outputs["mask_area_ratio"],
                    "is_instance_observation": False,
                    "kept_detections": kept_detections,
                }
            )

    (frame_out / "detections.json").write_text(json.dumps(all_prompt_detections, indent=2), encoding="utf-8")
    (frame_out / "prompt_validation.json").write_text(json.dumps(prompt_validation, indent=2), encoding="utf-8")
    frame_observation = {
        "frame_id": int(task["frame_id"]),
        "frame_stem": frame_stem,
        "image_path": str(image_path),
        "frame_output_dir": str(frame_out),
        "prompt_count": len(prompts),
        "accepted_count": len(accepted_observations),
        "rejected_count": len(rejected_observations),
        "observation_mode": "instance" if write_instance_observations else "prompt_union",
        "accepted_observations": accepted_observations,
        "rejected_observations": rejected_observations,
        "prompt_validation_json": str(frame_out / "prompt_validation.json"),
        "detections_json": str(frame_out / "detections.json"),
    }
    write_json(frame_out / "frame_observation.json", frame_observation)
    result["outputs"] = prompt_outputs
    result["detections_json"] = str(frame_out / "detections.json")
    result["prompt_validation_json"] = str(frame_out / "prompt_validation.json")
    result["prompt_validation"] = prompt_validation
    result["frame_observation_json"] = str(frame_out / "frame_observation.json")
    result["frame_observation"] = frame_observation
    result["accepted_observation_count"] = len(accepted_observations)
    result["rejected_observation_count"] = len(rejected_observations)
    result["status"] = "processed"
    result["elapsed_sec"] = round(time.perf_counter() - start, 6)
    return result


def collect_frame_observations(task_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observations = []
    for result in task_results:
        frame_observation = result.get("frame_observation")
        if isinstance(frame_observation, dict):
            observations.append(frame_observation)
    return sorted(observations, key=lambda item: (int(item.get("frame_id", 0)), str(item.get("frame_stem", ""))))


def write_frame_observation_outputs(output_dir: Path, task_results: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    frame_observations = collect_frame_observations(task_results)
    total_prompt_count = sum(int(item.get("prompt_count", 0)) for item in frame_observations)
    accepted_count = sum(int(item.get("accepted_count", 0)) for item in frame_observations)
    rejected_count = sum(int(item.get("rejected_count", 0)) for item in frame_observations)
    instance_count = sum(
        1
        for item in frame_observations
        for obs in item.get("accepted_observations", [])
        if obs.get("is_instance_observation")
    )
    summary = {
        "frame_count": len(frame_observations),
        "total_prompt_count": total_prompt_count,
        "accepted_observation_count": accepted_count,
        "rejected_observation_count": rejected_count,
        "accepted_instance_observation_count": instance_count,
        "accept_rate": round(accepted_count / total_prompt_count, 6) if total_prompt_count else 0.0,
    }
    payload = {
        "format": "grounded_sam2_frame_observations_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metadata": metadata,
        "summary": summary,
        "frames": frame_observations,
    }
    json_path = output_dir / "frame_observations.json"
    csv_path = output_dir / "frame_observations.csv"
    write_json(json_path, payload)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_id",
                "frame_stem",
                "image_path",
                "prompt_count",
                "accepted_count",
                "rejected_count",
                "observation_mode",
                "accepted_instance_count",
                "accepted_labels",
                "rejected_labels",
            ],
        )
        writer.writeheader()
        for item in frame_observations:
            writer.writerow(
                {
                    "frame_id": item.get("frame_id"),
                    "frame_stem": item.get("frame_stem"),
                    "image_path": item.get("image_path"),
                    "prompt_count": item.get("prompt_count"),
                    "accepted_count": item.get("accepted_count"),
                    "rejected_count": item.get("rejected_count"),
                    "observation_mode": item.get("observation_mode"),
                    "accepted_instance_count": sum(
                        1 for obs in item.get("accepted_observations", []) if obs.get("is_instance_observation")
                    ),
                    "accepted_labels": ",".join(obs["label"] for obs in item.get("accepted_observations", [])),
                    "rejected_labels": ",".join(obs["label"] for obs in item.get("rejected_observations", [])),
                }
            )

    return {
        **summary,
        "frame_observations_json": str(json_path),
        "frame_observations_csv": str(csv_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 2D masks with Grounding DINO + SAM 2")
    parser.add_argument("--image", type=Path, default=None, help="Single-image debug mode. Does not read or mutate the queue.")
    parser.add_argument("--queue_root", type=Path, default=DEFAULT_QUEUE_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--prompts", type=str, default=DEFAULT_PROMPTS)
    parser.add_argument("--prompts_file", type=Path, default=None, help="Read prompts from txt/json, including auto_prompt_ram.py outputs.")
    parser.add_argument("--per_frame_prompts_dir", type=Path, default=None, help="Queue mode only: read <frame_stem>_object_prompts.txt from this directory for each task.")
    parser.add_argument("--grounding_model_id", type=str, default=DEFAULT_GROUNDING_MODEL_ID)
    parser.add_argument("--sam2_model_id", type=str, default=DEFAULT_SAM2_MODEL_ID)
    parser.add_argument("--box_threshold", type=float, default=0.35)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--min_mask_score", type=float, default=0.0)
    parser.add_argument("--min_mask_area_ratio", type=float, default=0.0)
    parser.add_argument("--max_mask_area_ratio", type=float, default=0.55)
    parser.add_argument("--max_detections_per_prompt", type=int, default=5)
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_overlay", action="store_true")
    parser.add_argument("--overlay_alpha", type=float, default=0.45)
    parser.add_argument(
        "--prompt_union_observations",
        action="store_true",
        help="Use legacy prompt-level union masks as observations instead of one observation per kept detection",
    )
    args = parser.parse_args()

    single_task = make_image_task(args.image) if args.image is not None else None
    frame_stem = single_task["frame_stem"] if single_task is not None else None
    if args.prompts_file is not None:
        prompts = load_prompts_file(args.prompts_file, frame_stem=frame_stem)
        if not prompts:
            raise ValueError(f"No prompts found in {args.prompts_file}")
    else:
        prompts = parse_prompts(args.prompts)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(args.device)

    output_dir = make_single_output_dir(args.output_root) if single_task is not None else make_output_dir(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = GroundingDinoDetector(args.grounding_model_id, device)
    segmenter = Sam2BoxSegmenter(args.sam2_model_id, device)

    if single_task is not None:
        print(f"[image] {single_task['image_path']}")
        print(f"[output] {output_dir}")
        print(f"[prompts] {prompts}")
        print(f"[grounding_model_id] {args.grounding_model_id}")
        print(f"[sam2_model_id] {args.sam2_model_id}")
        result = process_task(
            task=single_task,
            output_root=output_dir,
            prompts=prompts,
            detector=detector,
            segmenter=segmenter,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            min_mask_score=args.min_mask_score,
            max_detections_per_prompt=args.max_detections_per_prompt,
            min_mask_area_ratio=args.min_mask_area_ratio,
            max_mask_area_ratio=args.max_mask_area_ratio,
            write_overlay=not args.no_overlay,
            overlay_alpha=args.overlay_alpha,
            write_instance_observations=not args.prompt_union_observations,
        )
        counts = {result["status"]: 1}
        observation_outputs = write_frame_observation_outputs(
            output_dir,
            [result],
            metadata={
                "mode": "single_image",
                "image": single_task["image_path"],
                "prompts_file": str(args.prompts_file) if args.prompts_file else None,
                "grounding_model_id": args.grounding_model_id,
                "sam2_model_id": args.sam2_model_id,
                "box_threshold": args.box_threshold,
                "text_threshold": args.text_threshold,
                "min_mask_score": args.min_mask_score,
                "min_mask_area_ratio": args.min_mask_area_ratio,
                "max_mask_area_ratio": args.max_mask_area_ratio,
                "max_detections_per_prompt": args.max_detections_per_prompt,
                "observation_mode": "prompt_union" if args.prompt_union_observations else "instance",
            },
        )
        run_config = {
            "script": "online_grounded_sam2_worker.py",
            "mode": "single_image",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "image": single_task["image_path"],
            "output": str(output_dir),
            "backend": "grounding_dino_sam2",
            "grounding_model_id": args.grounding_model_id,
            "sam2_model_id": args.sam2_model_id,
            "prompts": prompts,
            "prompts_file": str(args.prompts_file) if args.prompts_file else None,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "min_mask_score": args.min_mask_score,
            "min_mask_area_ratio": args.min_mask_area_ratio,
            "max_mask_area_ratio": args.max_mask_area_ratio,
            "max_detections_per_prompt": args.max_detections_per_prompt,
            "observation_mode": "prompt_union" if args.prompt_union_observations else "instance",
            "frame_observations": observation_outputs,
            "status_counts": counts,
            "tasks": [result],
        }
        with (output_dir / "run_config.json").open("w") as f:
            json.dump(run_config, f, indent=2)

        summary = ["Grounding DINO + SAM 2 single-image summary", ""]
        validation = result.get("prompt_validation", {})
        for prompt in prompts:
            item = validation.get(prompt, {})
            status = item.get("status", "unknown")
            reason = item.get("reject_reason")
            suffix = f" ({reason})" if reason else ""
            summary.append(
                f"{prompt}: {status}{suffix}, kept={item.get('kept_detection_count', 0)}, "
                f"mask_area={item.get('mask_area_ratio', 0.0)}"
            )
        summary.append("")
        summary.append(f"output: {output_dir}")
        (output_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
        print(f"[status_counts] {counts}")
        print(f"[done] outputs at {output_dir}")
        return

    queue_items = load_queue_tasks(args.queue_root, args.max_tasks)
    if not queue_items:
        raise ValueError(f"No pending queue tasks in {args.queue_root}")

    print(f"[queue_root] {args.queue_root}")
    print(f"[output] {output_dir}")
    if args.per_frame_prompts_dir is not None:
        print(f"[per_frame_prompts_dir] {args.per_frame_prompts_dir}")
    else:
        print(f"[prompts] {prompts}")
    print(f"[grounding_model_id] {args.grounding_model_id}")
    print(f"[sam2_model_id] {args.sam2_model_id}")
    print(f"[tasks] {len(queue_items)}")

    counts: dict[str, int] = {}
    task_results = []
    for processing_path, wrapper in tqdm(queue_items, desc="grounded sam2 worker"):
        try:
            task = wrapper["task"]
            task_prompts = prompts
            if args.per_frame_prompts_dir is not None:
                frame_stem = str(task.get("frame_stem") or f"frame_{int(task['frame_id']):06d}")
                task_prompts = load_per_frame_prompts(args.per_frame_prompts_dir, frame_stem)
                if not task_prompts:
                    raise ValueError(f"No object prompts for {frame_stem}")
            result = process_task(
                task=task,
                output_root=output_dir,
                prompts=task_prompts,
                detector=detector,
                segmenter=segmenter,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                min_mask_score=args.min_mask_score,
                max_detections_per_prompt=args.max_detections_per_prompt,
                min_mask_area_ratio=args.min_mask_area_ratio,
                max_mask_area_ratio=args.max_mask_area_ratio,
                write_overlay=not args.no_overlay,
                overlay_alpha=args.overlay_alpha,
                write_instance_observations=not args.prompt_union_observations,
            )
        except Exception as exc:
            result = {
                "frame_id": wrapper.get("task", {}).get("frame_id"),
                "frame_stem": wrapper.get("task", {}).get("frame_stem"),
                "status": "failed",
                "backend": "grounding_dino_sam2",
                "error": repr(exc),
                "outputs": {},
                "elapsed_sec": 0.0,
            }
        finish_queue_task(args.queue_root, processing_path, wrapper, result)
        task_results.append(result)
        counts[result["status"]] = counts.get(result["status"], 0) + 1

    run_config = {
        "script": "online_grounded_sam2_worker.py",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "queue_root": str(args.queue_root),
        "output": str(output_dir),
        "backend": "grounding_dino_sam2",
        "grounding_model_id": args.grounding_model_id,
        "sam2_model_id": args.sam2_model_id,
        "prompts": prompts,
        "per_frame_prompts_dir": str(args.per_frame_prompts_dir) if args.per_frame_prompts_dir else None,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "min_mask_score": args.min_mask_score,
        "min_mask_area_ratio": args.min_mask_area_ratio,
        "max_mask_area_ratio": args.max_mask_area_ratio,
        "max_detections_per_prompt": args.max_detections_per_prompt,
        "observation_mode": "prompt_union" if args.prompt_union_observations else "instance",
        "status_counts": counts,
        "tasks": task_results,
    }
    observation_outputs = write_frame_observation_outputs(
        output_dir,
        task_results,
        metadata={
            "mode": "queue",
            "queue_root": str(args.queue_root),
            "prompts": prompts,
            "per_frame_prompts_dir": str(args.per_frame_prompts_dir) if args.per_frame_prompts_dir else None,
            "grounding_model_id": args.grounding_model_id,
            "sam2_model_id": args.sam2_model_id,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "min_mask_score": args.min_mask_score,
            "min_mask_area_ratio": args.min_mask_area_ratio,
            "max_mask_area_ratio": args.max_mask_area_ratio,
            "max_detections_per_prompt": args.max_detections_per_prompt,
            "observation_mode": "prompt_union" if args.prompt_union_observations else "instance",
        },
    )
    run_config["frame_observations"] = observation_outputs
    with (output_dir / "run_config.json").open("w") as f:
        json.dump(run_config, f, indent=2)

    summary = ["Grounding DINO + SAM 2 worker summary", ""]
    for status, count in sorted(counts.items()):
        summary.append(f"{status}: {count}")
    summary.extend(
        [
            "",
            f"frame observations: {observation_outputs['frame_count']}",
            f"accepted observations: {observation_outputs['accepted_observation_count']}",
            f"accepted instance observations: {observation_outputs['accepted_instance_observation_count']}",
            f"rejected observations: {observation_outputs['rejected_observation_count']}",
            f"frame_observations_json: {observation_outputs['frame_observations_json']}",
        ]
    )
    summary.append("")
    summary.append(f"output: {output_dir}")
    (output_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"[status_counts] {counts}")
    print(f"[done] outputs at {output_dir}")


if __name__ == "__main__":
    main()
