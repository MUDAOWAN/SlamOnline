# SlamOnline Project Notes

Last updated: 2026-06-25

## Project Position

SlamOnline is the new main folder for the online semantic 3D reconstruction branch of SplatGraph. Its goal is:

**Online RGB-D SLAM + 3D Gaussian reconstruction + asynchronous open-vocabulary semantic lifting + object-centric scene representation.**

The original `/home/sky/czh/SplatGraph` remains the historical workspace and source of migrated code. Future development should happen under `/home/sky/czh/SplatGraph/SlamOnline` and run in the `SlamOnline` conda environment unless noted otherwise.

## Relationship To SplatGraph

SplatGraph is the broader project name: a 3D Gaussian-based semantic object graph for metric spatial reasoning. SlamOnline extracts the current online route into a cleaner GitHub-style project root. It does not copy datasets, checkpoints, experiment outputs, caches, logs, or the full historical `prompt.md`.

Code migrated from SplatGraph keeps its original behavior unless explicitly noted in this document.

## Current Workflow

```text
RGB-D dataset
  -> third_party/MonoGS/slam.py
  -> same-run Gaussian PLY under point_cloud/final/point_cloud.ply
  -> MonoGS frontend emits queue/pending/*.json keyframe tasks
  -> auto_prompt_ram.py generates raw visual tags
  -> auto_prompt_filter.py filters per-frame object prompts
  -> online_grounded_sam2_worker.py writes Grounding DINO + SAM 2 instance masks
  -> lift_frame_observations_to_3d.py lifts 2D observations onto Gaussians
  -> object_memory_update.py builds object memory
  -> refine/merge/audit/visualize/export scripts produce final scene objects
```

Historical verified outputs before the current object-memory pipeline:

- MonoGS same-run smoke: `/tmp/splatgraph_monogs_same_run/results/datasets_small/2026-06-03-10-22-35`
- Replica room0 400-frame run: `/tmp/splatgraph_replica_room0/results/czh_datasets/2026-06-03-14-51-32`

## Completed Work

- MonoGS can save a same-run Gaussian PLY and emit semantic queue tasks from frontend keyframes.
- `online_semantic_queue.py` supports status, clear, retry-failed, and retry-processing.
- `auto_prompt_ram.py` and `auto_prompt_filter.py` generate object prompts from queued frames.
- `online_grounded_sam2_worker.py` consumes queue tasks and writes detector-first Grounding DINO + SAM 2 frame observations.
- `lift_frame_observations_to_3d.py` projects 2D observations onto the same-run Gaussian PLY.
- `object_memory_update.py`, refinement, merge, audit, visualization, and export scripts form the current object-memory path.
- The current command sequence is maintained in `slam_online/core/core_readme.md`.

## Core Code

| Path | Purpose |
|---|---|
| `slam_online/core/online_semantic_queue.py` | File-backed semantic task queue. |
| `slam_online/core/auto_prompt_ram.py` | RAM++ raw tag generation from queued frames. |
| `slam_online/core/auto_prompt_filter.py` | Per-frame object prompt filtering with local LLM config support. |
| `slam_online/core/online_grounded_sam2_worker.py` | Current detector-first Grounding DINO + SAM 2 prompt mask worker. |
| `slam_online/core/lift_frame_observations_to_3d.py` | 2D observation to 3D Gaussian lifting. |
| `slam_online/core/object_memory_update.py` | Associate 3D observations into object memory. |
| `slam_online/core/refine_object_memory_conflicts.py` | Mark weak conflicting object entries before merge. |
| `slam_online/core/merge_object_memory_global.py` | Merge duplicate or partial object entries. |
| `slam_online/core/refine_final_object_memory.py` | Hide remaining display-level duplicates or parts. |
| `slam_online/core/visualize_object_memory.py` | Write colored PLY object-memory visualizations. |
| `slam_online/core/export_scene_objects.py` | Export scene objects and spatial relations for LLM use. |
| `slam_online/core/gaussian_ply.py` | Shared Gaussian PLY read/write helpers. |
| `third_party/MonoGS/` | Runnable MonoGS runtime, copied with caches/results/builds removed. |
The old LangSplatV2/SAM+CLIP route has been removed from this project. The
current active semantic backend is Grounding DINO + SAM 2.

## Object-First Boundary

Current SlamOnline is not yet full persistent instance tracking.

Why current code is still an evolving object-memory pipeline:

- It starts from frame-level object prompts and 2D instance observations.
- It lifts observations to the Gaussian map and associates them into object memory.
- It refines and merges object entries, but long-horizon identity tracking is still experimental.
- It does not yet maintain a mature object graph with fused language features and relations.

Difference from true segment-first instance tracking:

- True tracking starts with class-agnostic or open-vocabulary 2D segments, not fixed text prompts.
- It associates segments across views and over time.
- It maintains persistent 3D instance ids as the map grows.
- It fuses per-object language descriptors and view evidence before arbitrary prompt query.

This boundary does not block the current MonoGS + Grounded-SAM2 + semantic
lifting flow. It only means current object outputs should be treated as
refined object-memory entries, not final persistent scene-graph nodes.

Future modules needed for true instance tracking:

- Class-agnostic 2D mask proposal pipeline per keyframe.
- 2D-to-3D segment association and object id matching.
- Persistent object state store with track history.
- Per-object language feature fusion.
- Caption generation and relation extraction.
- Object graph query layer for LLM spatial reasoning.

## Async Grounded-SAM2 Strategy

The current near-term online target is not hard real-time semantic fusion on
every incoming frame. The target is an asynchronous pipeline that is realistic
for a single 12GB GPU and still reflects robot-style online perception:

```text
MonoGS process
  -> reads RGB-D stream / sampled dataset sequentially
  -> builds 3D Gaussian map
  -> emits keyframe tasks to file-backed queue

GroundingDINO + SAM2 process
  -> consumes queue tasks asynchronously and conservatively, e.g. 1-2 keyframes per batch
  -> writes detector-first 2D masks, bboxes, scores, and diagnostics

After MonoGS finishes the scene, or at coarse checkpoints
  -> lift_frame_observations_to_3d.py projects finished 2D observations to the same-run Gaussian PLY
  -> object_memory_update.py and refinement scripts update object memory
```

This is a real asynchronous queue design, not a fake synchronized script:

- MonoGS does not need to call GroundingDINO/SAM2 directly.
- The 2D semantic worker can run in a separate conda environment named `Semantic2D`.
- Both processes communicate through `output/queue/{pending,processing,done,failed}` and explicit mask paths stored in queue result JSON files.
- On a single 12GB GPU, the first reliable mode is staggered execution: run MonoGS first, then run the Grounded-SAM2 worker, then run 3D lifting and object-memory update.
- The next mode is concurrent low-frequency consumption: MonoGS writes queue while the worker consumes only 1-2 keyframes at a time.
- Final 2D-to-3D lifting and object-memory update are expected to be much faster than Grounded-SAM2 inference, so they can run after the scene or periodically.

Scientific claim to test:

```text
Detector-first semantics can run asynchronously with online Gaussian SLAM,
reducing post-scene semantic latency while keeping mapping stable on limited GPU memory.
```

Implementation steps:

1. Run staggered mode: MonoGS complete -> Grounded-SAM2 consumes queue -> 3D lifting -> object memory.
2. Run concurrent mode on 12GB GPU with small batches, reduced model sizes, and explicit monitoring of GPU memory/FPS.
3. If concurrent mode is unstable, keep the asynchronous file-backed design but schedule the worker after MonoGS or on a second GPU.
4. Next major upgrade: automatic object discovery -> generated prompts/candidates -> Grounded-SAM2 masks -> object memory -> scene graph.

## Directory Structure

```text
SlamOnline/
  README.md
  requirement.txt
  slam_online.md
  docs/
    research-directions.md
    summary_prompt.md
  envs/
    SlamOnline.reference.yml
  slam_online/
    __init__.py
    core/
      online_semantic_queue.py
      auto_prompt_ram.py
      auto_prompt_filter.py
      online_grounded_sam2_worker.py
      lift_frame_observations_to_3d.py
      object_memory_update.py
      refine_object_memory_conflicts.py
      merge_object_memory_global.py
      refine_final_object_memory.py
      visualize_object_memory.py
      export_scene_objects.py
      gaussian_ply.py
      ...
  third_party/
    MonoGS/
```

## Migration Sources

| Source | Target | Notes |
|---|---|---|
| `Online/core/*.py` | `slam_online/core/` | Online queue, Grounded-SAM2 worker, prompt, lifting, object-memory, and export utilities. |
| `core/*.py` selected scripts | `slam_online/core/` | Shared helpers and selected object-memory utilities. |
| `MonoGS/` | `third_party/MonoGS/` | Runtime copied excluding results, wandb, media, build, cache, logs, `.git`. |
| `docs/research-directions.md` | `docs/research-directions.md` | Research context. |
| `prompt.md` | `docs/summary_prompt.md` | Summary only; full history not copied. |

## Not Copied

- Datasets under `/home/sky/czh/datasets`.
- Experiment outputs under `/tmp`, `MonoGS/results`, `runs`, `outputs`, `wandb`.
- Checkpoints and model weights.
- `.git`, `__pycache__`, build directories, egg-info generated metadata, logs, media assets.
- Large external reference projects under `Online/OVO`, `Online/concept-graphs`, `Online/SNI-SLAM`, `Online/SemGauss-SLAM`, etc.
- Full `prompt.md` conversation history.

## Environment

Recommended unified conda environment:

```bash
cd /home/sky/czh/SplatGraph/SlamOnline
conda create -n SlamOnline python=3.9 pip=22.3.1
conda activate SlamOnline
conda install -c pytorch -c nvidia pytorch=2.1.0 torchvision=0.16.0 torchaudio=2.1.0 pytorch-cuda=12.1
pip install -r requirement.txt
export LD_LIBRARY_PATH="$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib")'):${LD_LIBRARY_PATH}"
cd third_party/MonoGS/submodules/simple-knn
python setup.py develop
cd ../diff-gaussian-rasterization
python setup.py develop
```

Alternatively, create the PyTorch/CUDA base from the reference file, then install pip dependencies:

```bash
cd /home/sky/czh/SplatGraph/SlamOnline
conda env create -f envs/SlamOnline.reference.yml
conda activate SlamOnline
pip install -r requirement.txt
export LD_LIBRARY_PATH="$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib")'):${LD_LIBRARY_PATH}"
cd third_party/MonoGS/submodules/simple-knn
python setup.py develop
cd ../diff-gaussian-rasterization
python setup.py develop
```

Potential conflict:

- Original MonoGS env used Python 3.9, PyTorch 2.1.0, CUDA 12.1.
- The recommended unified environment follows MonoGS because MonoGS runtime and CUDA extensions are more version-sensitive here.
- NumPy is pinned to `1.26.4`; NumPy 2.x can break MonoGS/3DGS CUDA extension builds and some packages compiled against NumPy 1.x.
- `setuptools` is pinned to `68.2.2` with `wheel` and `ninja` for old-style PyTorch CUDA extension builds such as `simple-knn` and `diff-gaussian-rasterization`.
- Grounding DINO through HuggingFace `transformers` can run in the base env, but SAM 2 official packages require Python 3.10 and newer PyTorch than the current MonoGS base. Run `online_grounded_sam2_worker.py` in a separate `Semantic2D` env while sharing the same file-backed queue and output paths.

## Current Run Commands

Use `slam_online/core/core_readme.md` as the authoritative command manual for
the current core pipeline. The expected command order is:

1. MonoGS writes the Gaussian PLY and keyframe queue.
2. `online_semantic_queue.py` inspects or resets queue state.
3. `auto_prompt_ram.py` generates RAM++ raw tags.
4. `auto_prompt_filter.py` filters tags into per-frame object prompts.
5. `online_grounded_sam2_worker.py` writes 2D frame observations.
6. `lift_frame_observations_to_3d.py` lifts observations onto Gaussians.
7. `audit_frame_3d_observations.py` checks lifted observations.
8. `generate_grouped_label_similarity.py` builds label grouping hints.
9. `object_memory_update.py` creates object memory.
10. `audit_object_memory_quality.py` checks object-level quality.
11. `refine_object_memory_conflicts.py`, `merge_object_memory_global.py`, and
    `refine_final_object_memory.py` clean duplicate or partial entries.
12. `visualize_object_memory.py` writes PLY visualizations.
13. `export_scene_objects.py` writes scene objects and spatial relations.

## Default Paths To Review

Some migrated scripts still keep historical defaults:

- `slam_online/core/online_semantic_queue.py`: table dataset queue defaults.
- `slam_online/core/online_grounded_sam2_worker.py`: generic queue/output/model defaults.
- `slam_online/core/auto_prompt_ram.py`: checkpoint and queue/output defaults.
- `slam_online/core/auto_prompt_filter.py`: local LLM config path defaults.
- `slam_online/core/lift_frame_observations_to_3d.py`: point cloud, queue, and output defaults.
- `slam_online/core/object_memory_update.py`: object-memory thresholds and output defaults.
- `third_party/MonoGS/configs/rgbd/replica/new_room0_online_semantic_save.yaml`: `/home/sky/czh/SplatGraph/SlamOnline/datasets/new_room0`, `/home/sky/czh/SplatGraph/SlamOnline/output/results`, and `/home/sky/czh/SplatGraph/SlamOnline/output/queue`.
- `third_party/MonoGS/configs/rgbd/tum/d455_online_semantic_save.yaml`: D455/table paths.

For real runs, pass paths explicitly or edit configs. Future cleanup should move these defaults into a central config or CLI-only workflow.

## Known Issues And Risks

- Single 12GB GPU can OOM if MonoGS and Grounded-SAM2 run concurrently. Prefer staggered runs, smaller batches, lower-resolution inputs, or separate GPUs.
- The old SAM+CLIP route has been removed; use the current Grounded-SAM2 route.
- Unified `SlamOnline` env is recommended but not yet verified after migration.
- Prompt-first outputs can duplicate overlapping concepts, for example `chair/sofa` or `table/desk`.
- Current object sidecar has no persistent ids or true multi-view object tracking.
- Some scripts retain historical default paths; explicit CLI paths are safer.

## Future Work

- Improve prompt-first stability with prompt diagnostics, per-prompt visible, `min_votes`, adaptive thresholding, and prompt grouping.
- Implement segment/object-first candidate generation from class-agnostic masks.
- Add persistent instance ids and track history.
- Fuse per-object CLIP/Lang features.
- Add captions, relation extraction, and metric object graph.
- Build LLM-readable scene graph for spatial reasoning.

## Development Convention

Use `/home/sky/czh/SplatGraph/SlamOnline` as the project root. Put new project-owned scripts under `slam_online/core/` or a clearly named package module. Keep third-party code under `third_party/` and avoid editing it unless the change is necessary and documented.
