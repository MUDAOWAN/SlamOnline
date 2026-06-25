# Prompt History Summary

Source history: `/home/sky/czh/SplatGraph/prompt.md`.

This file intentionally does not copy the full historical conversation. The relevant, code-verified project state for SlamOnline is:

- The online chain has been run end to end: MonoGS same-run PLY, queue tasks, LangSplatV2/SAM+CLIP worker, online fusion, object bbox, and `object_candidates.json`.
- MonoGS and SAM+CLIP are run asynchronously on a file-backed queue because a 12GB GPU can OOM when MonoGS and SAM ViT-H run concurrently.
- Replica room0 400-frame sampling produced a usable Gaussian PLY at `/tmp/splatgraph_replica_room0/results/czh_datasets/2026-06-03-14-51-32/point_cloud/final/point_cloud.ply`.
- Prompt-first fusion works, but overlapping prompts such as `chair/sofa` or `table/desk` can create duplicate or mixed candidates.
- The next research direction is object/instance-first: stabilize 3D object candidates, attach per-object CLIP/Lang features, view evidence, bbox, captions, and later support arbitrary prompt query over object features.
- The current object candidate sidecar is v0.1: it normalizes prompt-first bbox outputs into an object-shaped schema, but it is not yet a true multi-view instance tracker.

When this summary conflicts with code or generated run files, trust code and run metadata first.
