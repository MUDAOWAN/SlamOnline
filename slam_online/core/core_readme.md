# SlamOnline Core Pipeline

This document records the current main pipeline under `slam_online/core`.
The example commands use Replica `office3` with this scene output root:

```bash
PROJECT=/home/sky/czh/SplatGraph/SlamOnline
SCENE_OUT=$PROJECT/output/replica_office3_third
QUEUE=$SCENE_OUT/queue
PLY=$SCENE_OUT/results/datasets_Replica/2026-06-15-15-14-44/point_cloud/final/point_cloud.ply
LLM_CONFIG=$PROJECT/configs/llm_prompt_filter.json
```

Recommended short output folders:

```text
01_ram/
02_prompt/
03_sam2_inst/
04_lift3d/
05_memory/
06_audit/
07_merge/
08_final/
09_viz/
10_scene/
```

Each script still creates a timestamped run folder inside its `--output_root`,
for example `01_ram/auto_prompts_YYYYMMDD_HHMMSS`.

## Overview

| Step | Script | Function | Main output |
|---:|---|---|---|
| 0 | MonoGS | Build Gaussian map and write keyframe queue | `results/.../point_cloud.ply`, `queue/` |
| 1 | `online_semantic_queue.py` | Inspect or reset queue state | queue counts |
| 2 | `auto_prompt_ram.py` | Generate RAM++ raw tags from queue images | `auto_prompts.json` |
| 3 | `auto_prompt_filter.py` | Filter raw tags into per-frame object prompts | `*_object_prompts.txt` |
| 4 | `online_grounded_sam2_worker.py` | Consume queue and produce 2D instance masks | `frame_observations.json` |
| 5 | `lift_frame_observations_to_3d.py` | Lift 2D observations to 3D Gaussian observations | `frame_3d_observations.json`, `frame_3d_hits.npz` |
| 6 | `audit_frame_3d_observations.py` | Audit lifted 3D observations | `label_stats.csv`, audit CSVs |
| 7 | `generate_grouped_label_similarity.py` | Generate grouped label relation file | `label_similarity.json` |
| 8 | `object_memory_update.py` | Associate 3D observations into object memory | `object_memory.json`, `object_memory_points.npz` |
| 9 | `audit_object_memory_quality.py` | Audit object-level quality and pair issues | `object_pair_issues.csv` |
| 10 | `refine_object_memory_conflicts.py` | Mark weak conflicting objects before merge | `refined_object_memory.json` |
| 11 | `merge_object_memory_global.py` | Merge duplicate or partial objects | `final_object_memory.json`, `final_object_memory_points.npz` |
| 12 | `refine_final_object_memory.py` | Hide remaining display-level duplicates or parts | `final_refined_object_memory.json` |
| 13 | `visualize_object_memory.py` | Write PLY visualization | `object_memory_colored_original_background.ply` |
| 14 | `export_scene_objects.py` | Export objects and spatial relations for LLM use | `scene_objects.json`, `scene_relations.json` |

## Queue Notes

The queue layout is:

```text
queue/
  pending/
  processing/
  done/
  failed/
```

`online_grounded_sam2_worker.py` consumes queue files:

```text
pending -> processing -> done or failed
```

`auto_prompt_ram.py` and `lift_frame_observations_to_3d.py` read queue files
but do not move them.

Check queue state:

```bash
cd $PROJECT
python slam_online/core/online_semantic_queue.py status \
  --queue_root "$QUEUE"
```

If you need to rerun the 2D worker on an already consumed queue, move `done`
items back to `pending` first:

```bash
python slam_online/core/online_semantic_queue.py retry-done \
  --queue_root "$QUEUE"
```

## Step 0: MonoGS

Run MonoGS from the MonoGS directory:

```bash
cd $PROJECT/third_party/MonoGS
conda activate SlamOnline

python slam.py \
  --config configs/rgbd/replica/office3_third_online_semantic_save.yaml
```

Expected outputs:

```text
$SCENE_OUT/results/.../point_cloud/final/point_cloud.ply
$SCENE_OUT/queue/
```

## Step 1: RAM++ Tags

RAM reads queue task JSON files and writes raw image tags.

```bash
cd $PROJECT
conda activate Semantic2D

python slam_online/core/auto_prompt_ram.py \
  --queue_root "$QUEUE" \
  --queue_states pending,done \
  --output_root "$SCENE_OUT/01_ram" \
  --checkpoint "$PROJECT/third_party/recognize-anything/pretrained/ram_plus_swin_large_14m.pth" \
  --model_variant ram_plus \
  --device cuda \
  --max_tags 20
```

Use the latest output directory as:

```bash
RAM_DIR=$SCENE_OUT/01_ram/auto_prompts_YYYYMMDD_HHMMSS
```

Main output:

```text
$RAM_DIR/auto_prompts.json
```

## Step 2: Prompt Filter

Filter RAM++ tags into object prompts.

```bash
cd $PROJECT

python slam_online/core/auto_prompt_filter.py \
  --auto_prompt_json "$RAM_DIR/auto_prompts.json" \
  --output_root "$SCENE_OUT/02_prompt" \
  --llm_config "$LLM_CONFIG" \
  --cache "$SCENE_OUT/02_prompt/prompt_classifier_cache.json"
```

Use the latest output directory as:

```bash
PROMPT_DIR=$SCENE_OUT/02_prompt/prompt_filter_YYYYMMDD_HHMMSS
```

Main outputs:

```text
$PROMPT_DIR/prompt_filter.json
$PROMPT_DIR/frame_xxxxxx_object_prompts.txt
```

## Step 3: GroundingDINO + SAM2 Instance Masks

This step consumes queue `pending` items. It should use per-frame prompts.

```bash
cd $PROJECT
conda activate Semantic2D

python slam_online/core/online_grounded_sam2_worker.py \
  --queue_root "$QUEUE" \
  --output_root "$SCENE_OUT/03_sam2_inst" \
  --per_frame_prompts_dir "$PROMPT_DIR" \
  --grounding_model_id IDEA-Research/grounding-dino-base \
  --sam2_model_id facebook/sam2-hiera-base-plus \
  --box_threshold 0.35 \
  --text_threshold 0.25 \
  --min_mask_area_ratio 0.002 \
  --max_mask_area_ratio 0.35 \
  --max_detections_per_prompt 5 \
  --device cuda
```

Use the latest output directory as:

```bash
SAM2_DIR=$SCENE_OUT/03_sam2_inst/worker_YYYYMMDD_HHMMSS
```

Main output:

```text
$SAM2_DIR/frame_observations.json
```

Do not use `--prompt_union_observations` for the main pipeline.

## Step 4: Lift 2D Observations To 3D

The default settings already use the current main strategy:

```text
bbox percentile: 2-98
depth consistency: abs=0.05, rel=0.03
component split: enabled
```

Run:

```bash
cd $PROJECT
conda activate SlamOnline

python slam_online/core/lift_frame_observations_to_3d.py \
  --frame_observations_json "$SAM2_DIR/frame_observations.json" \
  --point_cloud "$PLY" \
  --queue_root "$QUEUE" \
  --queue_states done \
  --output_root "$SCENE_OUT/04_lift3d"
```

Use the latest output directory as:

```bash
LIFT_DIR=$SCENE_OUT/04_lift3d/lift_YYYYMMDD_HHMMSS
```

Main outputs:

```text
$LIFT_DIR/frame_3d_observations.json
$LIFT_DIR/frame_3d_hits.npz
```

## Step 5: Audit 3D Observations

```bash
cd $PROJECT
conda activate SlamOnline

python slam_online/core/audit_frame_3d_observations.py \
  --frame_3d_observations_json "$LIFT_DIR/frame_3d_observations.json" \
  --output_root "$SCENE_OUT/06_audit/frame_3d"
```

Use the latest output directory as:

```bash
FRAME_AUDIT_DIR=$SCENE_OUT/06_audit/frame_3d/audit_YYYYMMDD_HHMMSS
```

Main outputs:

```text
$FRAME_AUDIT_DIR/audit_3d_observations.json
$FRAME_AUDIT_DIR/label_stats.csv
```

## Step 6: Generate Label Similarity

```bash
cd $PROJECT

python slam_online/core/generate_grouped_label_similarity.py \
  --frame_3d_observations_json "$LIFT_DIR/frame_3d_observations.json" \
  --output_root "$SCENE_OUT/06_audit/label_similarity" \
  --llm_config "$LLM_CONFIG" \
  --cache "$SCENE_OUT/06_audit/label_similarity/grouped_label_similarity_cache.json"
```

Use the latest output directory as:

```bash
LABEL_SIM_DIR=$SCENE_OUT/06_audit/label_similarity/grouped_similarity_YYYYMMDD_HHMMSS
```

Main output:

```text
$LABEL_SIM_DIR/label_similarity.json
```

## Step 7: Build Object Memory

```bash
cd $PROJECT
conda activate SlamOnline

python slam_online/core/object_memory_update.py \
  --frame_3d_observations_json "$LIFT_DIR/frame_3d_observations.json" \
  --hits_npz "$LIFT_DIR/frame_3d_hits.npz" \
  --label_similarity_json "$LABEL_SIM_DIR/label_similarity.json" \
  --output_root "$SCENE_OUT/05_memory" \
  --match_threshold 0.6 \
  --min_semantic_similarity 0.35 \
  --confirmed_min_frames 2 \
  --confirmed_min_observations 3
```

Use the latest output directory as:

```bash
MEM_DIR=$SCENE_OUT/05_memory/memory_YYYYMMDD_HHMMSS
```

Main outputs:

```text
$MEM_DIR/object_memory.json
$MEM_DIR/object_memory_points.npz
```

## Step 8: Audit Object Memory

```bash
cd $PROJECT
conda activate SlamOnline

python slam_online/core/audit_object_memory_quality.py \
  --object_memory_json "$MEM_DIR/object_memory.json" \
  --object_points_npz "$MEM_DIR/object_memory_points.npz" \
  --output_root "$SCENE_OUT/06_audit/object_memory" \
  --statuses confirmed,candidate
```

Use the latest output directory as:

```bash
MEM_AUDIT_DIR=$SCENE_OUT/06_audit/object_memory/audit_YYYYMMDD_HHMMSS
```

Main output:

```text
$MEM_AUDIT_DIR/object_pair_issues.csv
```

## Step 9: Conservative Refine

```bash
cd $PROJECT
conda activate SlamOnline

python slam_online/core/refine_object_memory_conflicts.py \
  --object_memory_json "$MEM_DIR/object_memory.json" \
  --object_pair_issues_csv "$MEM_AUDIT_DIR/object_pair_issues.csv" \
  --output_root "$SCENE_OUT/07_merge/refine"
```

Use the latest output directory as:

```bash
REFINE_DIR=$SCENE_OUT/07_merge/refine/refine_YYYYMMDD_HHMMSS
```

Main output:

```text
$REFINE_DIR/refined_object_memory.json
```

## Step 10: Global Merge

```bash
cd $PROJECT
conda activate SlamOnline

python slam_online/core/merge_object_memory_global.py \
  --object_memory_json "$REFINE_DIR/refined_object_memory.json" \
  --object_pair_issues_csv "$MEM_AUDIT_DIR/object_pair_issues.csv" \
  --object_points_npz "$MEM_DIR/object_memory_points.npz" \
  --label_similarity_json "$LABEL_SIM_DIR/label_similarity.json" \
  --output_root "$SCENE_OUT/07_merge/merge" \
  --include_final_statuses kept
```

Use the latest output directory as:

```bash
MERGE_DIR=$SCENE_OUT/07_merge/merge/merge_YYYYMMDD_HHMMSS
```

Main outputs:

```text
$MERGE_DIR/final_object_memory.json
$MERGE_DIR/final_object_memory_points.npz
```

## Step 11: Final Display Refine

```bash
cd $PROJECT
conda activate SlamOnline

python slam_online/core/refine_final_object_memory.py \
  --object_memory_json "$MERGE_DIR/final_object_memory.json" \
  --object_pair_issues_csv "$MEM_AUDIT_DIR/object_pair_issues.csv" \
  --label_similarity_json "$LABEL_SIM_DIR/label_similarity.json" \
  --output_root "$SCENE_OUT/08_final"
```

Use the latest output directory as:

```bash
FINAL_DIR=$SCENE_OUT/08_final/final_refine_YYYYMMDD_HHMMSS
```

Main output:

```text
$FINAL_DIR/final_refined_object_memory.json
```

## Step 12: Visualization

The visualization defaults already use the current main display strategy:

```text
statuses: confirmed
min_points: 1
bbox_only: enabled
bbox_mode: obb
bbox_line_samples: 96
bbox_scale: 0.01
bbox_marker_radius: 0.005
```

Run:

```bash
cd $PROJECT
conda activate SlamOnline

python slam_online/core/visualize_object_memory.py \
  --point_cloud "$PLY" \
  --object_memory_json "$FINAL_DIR/final_refined_object_memory.json" \
  --object_points_npz "$MERGE_DIR/final_object_memory_points.npz" \
  --output_root "$SCENE_OUT/09_viz" \
  --final_statuses kept \
  --write_per_object_plys
```

Use the latest output directory as:

```bash
VIZ_DIR=$SCENE_OUT/09_viz/object_memory_viz_YYYYMMDD_HHMMSS
```

Main output:

```text
$VIZ_DIR/object_memory_colored_original_background.ply
```

## Step 13: Export Scene Objects

```bash
cd $PROJECT
conda activate SlamOnline

python slam_online/core/export_scene_objects.py \
  --object_memory_json "$FINAL_DIR/final_refined_object_memory.json" \
  --output_root "$SCENE_OUT/10_scene" \
  --statuses confirmed \
  --final_statuses kept
```

Use the latest output directory as:

```bash
SCENE_DIR=$SCENE_OUT/10_scene/scene_YYYYMMDD_HHMMSS
```

Main outputs:

```text
$SCENE_DIR/scene_objects.json
$SCENE_DIR/scene_objects.csv
$SCENE_DIR/scene_relations.json
$SCENE_DIR/scene_relations.csv
$SCENE_DIR/scene_summary_for_llm.md
```

## Current Main Scripts

Keep these scripts in `slam_online/core`:

```text
auto_prompt_ram.py
auto_prompt_filter.py
online_semantic_queue.py
online_monogs_adapter.py
online_grounded_sam2_worker.py
lift_frame_observations_to_3d.py
audit_frame_3d_observations.py
generate_grouped_label_similarity.py
object_memory_update.py
audit_object_memory_quality.py
refine_object_memory_conflicts.py
merge_object_memory_global.py
refine_final_object_memory.py
visualize_object_memory.py
export_scene_objects.py
gaussian_ply.py
```

The old `slam_online/core/pass` holding area has been removed. Keep active
project-owned scripts directly under `slam_online/core`.
