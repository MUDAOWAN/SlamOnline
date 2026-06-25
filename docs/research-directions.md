# Research Directions: Multimodal Spatial Reasoning

Last updated: 2026-06-01

This document is a project-facing research map for SplatGraph. It is based on the user-provided taxonomy figure and text for "Multimodal Spatial Reasoning", plus the current repository context in `AGENTS.md` and `core/`.

It is not a complete literature review. Paper names from the taxonomy are kept as anchors, but individual paper details should be treated as "to be checked" unless they are explicitly described at a high level here.

## 1. Overview

Multimodal spatial reasoning studies how AI systems understand and reason about space from multiple signals: images, video, depth, point clouds, 3D scenes, language, audio, robot state, and action history. Typical questions include where an object is, which object is closer, whether one object supports another, how an agent should move, and how a scene may change over time.

For SplatGraph, the central research problem is narrower and more metric: convert RGB-D scenes into a 3D Gaussian-based semantic object graph, then let an LLM reason from explicit 3D evidence instead of guessing from image priors. The current route is:

```text
RGB-D / MonoGS point cloud + trajectory
  -> SAM+CLIP / LangSplatV2 2D prompt masks
  -> 2D-to-3D semantic voting on Gaussian points
  -> object instances with 3D centers and bboxes
  -> metric object graph
  -> LLM spatial reasoning
```

This makes the project most closely related to 3D visual grounding, 3D scene reasoning and QA, graph-based 3D reasoning, and tool-assisted MLLM spatial reasoning.

## 2. Taxonomy from the Figure

The user-provided taxonomy groups multimodal spatial reasoning into four large directions:

- General MLLM: test-time scaling, post-training, model design, and explainability for spatially stronger multimodal language models.
- 3D Vision: 3D visual grounding, 3D scene reasoning and QA, and 3D generation.
- Embodied AI: navigation, embodied QA, grasping, vision-language action, and world models.
- Novel Modalities: video-based and audio-based spatial reasoning.

When OCR and the provided text differ, this document follows the provided text.

## 3. Direction Summaries

### 3.1 General MLLM / Test-Time Scaling

Test-time scaling means improving spatial reasoning at inference time without changing model weights. In multimodal spatial reasoning, this can include decomposing a question into steps, asking the model to verbalize spatial evidence, using multiple prompts or self-checking, calling geometry tools, retrieving scene facts, or scoring candidate answers.

Prompt engineering methods such as Spatial-MM, VSI-Bench, and VoT are listed in the taxonomy. At the concept level, this direction asks whether better instructions, intermediate reasoning formats, visual chain-of-thought, or benchmark-style prompts can make a general MLLM more reliable on spatial tasks. Specific paper details are待进一步查阅.

Tool use methods such as SpatialScore and SpatialPIN are especially relevant to SplatGraph. A tool-using model can call external modules for projection, 3D distance, bbox overlap, support/contact heuristics, graph search, retrieval, or answer verification. This matches the project's philosophy: LLMs should consume computed geometric evidence rather than inventing metric relations.

Research outlook: high. Test-time and tool-use methods are practical because they can wrap existing MLLMs. They are useful for robotics, AR assistants, spatial search, home automation, and scene QA systems.

Project value: high. SplatGraph can expose `object_graph.json` as a tool output, then test whether MLLM answers improve when given explicit distances, surfaces, bboxes, and relation labels.

Relatedness: high.

### 3.2 General MLLM / Post-Training

Post-training modifies a base MLLM after pretraining. SFT uses curated instruction-answer examples, while RL uses rewards or preferences to encourage desired behavior. For spatial reasoning, post-training can teach the model to follow coordinate conventions, compare 3D distances, parse object relations, use depth, or avoid plausible but unsupported answers.

The taxonomy lists Multi-SpatialMLLM and SpatialVLM under SFT, and Video-R1 and Spatial-R1 under RL. These names indicate a direction where spatial tasks are not expected to emerge perfectly from generic vision-language pretraining. Specific mechanisms and datasets are待进一步查阅.

Research outlook: high, but more resource intensive than test-time methods. It matters if a system needs strong spatial reasoning without heavy external tooling, or if it must learn task-specific answer styles.

Project value: medium. SplatGraph can eventually generate its own small supervised dataset from object graphs, such as "which object is closer" or "is A above B". Full model fine-tuning is not the current priority, but the project can design graph-derived QA pairs and evaluation metrics that would support future SFT or RL.

Relatedness: medium.

### 3.3 General MLLM / Model Design

Model design changes the architecture or input representation to better encode spatial structure. Spatial MLLMs may add depth encoders, point-cloud encoders, multi-view fusion, object tokens, scene graph tokens, coordinate-aware attention, or modules that preserve metric information.

The taxonomy lists Spatial-MLLM, SpatialRGPT, and Spatial-ORMLLM. At a high level, this direction is about making spatial information first-class rather than forcing a 2D image encoder to infer it implicitly. Specific details for these works are待进一步查阅.

For SplatGraph, the most relevant representation question is how to feed 3D Gaussian-derived objects and relations to an LLM. Options include plain JSON, relation triples, graph text, table-like metrics, serialized object nodes, or a learned graph encoder in the future.

Research outlook: high. Better spatial model design is important for AR, robotics, digital twins, CAD assistants, and embodied agents.

Project value: high for representation design, medium for training a new model. The current project can adopt graph/text serialization ideas now, while architectural training is a later research extension.

Relatedness: high.

### 3.4 General MLLM / Explainability

Explainability asks why a model produced a spatial answer. In spatial reasoning, explanations can cite object evidence, bounding boxes, relative positions, paths, visibility, attention regions, tool outputs, or causal chains such as "A is above B because A's bottom z is higher than B's top z and their XY footprints overlap."

The taxonomy lists Beyond Semantics, ADAPTVIS, and RelatiViT. Specific paper details are待进一步查阅. The common research goal is to move beyond final-answer accuracy and make spatial decisions inspectable.

Research outlook: medium to high. Explainability is important for safety-critical robotics, debugging, human trust, and scientific reporting.

Project value: high. SplatGraph already produces interpretable artifacts: 2D masks, colored Gaussian PLYs, `objects.json`, and soon graph metrics. These can become explanation evidence in experiments and paper figures.

Relatedness: high.

### 3.5 3D Vision / 3D Visual Grounding

3D visual grounding maps natural language to a concrete object, region, location, or relation in a 3D scene. A query like "the bottle next to the kettle" should return the correct 3D object instance or bbox, not just a phrase-level answer.

The taxonomy splits this into 3D input, multi-view input, and hybrid 2D+3D input. 3D input methods such as LLM-Grounder and Grounded 3D-LLM can directly use point clouds, meshes, bboxes, or scene-level 3D features. Multi-view methods such as VLM-Grounder and 3DAxisPrompt use multiple RGB views and camera geometry to infer 3D grounding. Hybrid methods such as SeeGround and ReasonGrounder combine 2D semantic strength with 3D geometric consistency. Specific paper details are待进一步查阅.

The tradeoff is important. 3D input gives metric structure but may lose texture and open-vocabulary semantics. Multi-view input preserves rich image semantics but needs pose consistency and aggregation. Hybrid 2D+3D is often attractive because modern 2D foundation models are strong, while 3D geometry gives metric grounding.

Research outlook: high. 3D grounding is central to robot instruction following, AR scene search, spatial assistants, digital twins, and interactive scene editing.

Project value: very high. SplatGraph's current pipeline is a hybrid grounding system: 2D open-vocabulary SAM/CLIP masks are lifted into a 3D Gaussian point cloud, then clustered into object instances with bboxes. This is one of the project's closest research families.

Relatedness: high.

### 3.6 3D Vision / 3D Scene Reasoning and QA

3D scene reasoning and QA asks questions over 3D scenes: object identity, location, distance, support, containment, relative height, path, affordance, or relational comparison. It differs from plain VQA because the answer should depend on 3D geometry and metric evidence.

Training-required methods such as LLaVA-3D and 3DGraphLLM usually learn from 3D scene-language data, scene graphs, or 3D QA annotations. Training-free methods such as SpatialPIN and Agent3D-Zero try to use existing models with tools, agents, prompting, or structured scene representations. Specific implementation details are待进一步查阅.

Graph-based methods such as 3DGraphLLM are especially relevant because scene graphs compress complex 3D data into object nodes and relation edges. This matches SplatGraph's plan: build object nodes from Gaussian semantic clusters, compute metric relations, then use the graph for LLM-readable spatial reasoning.

Research outlook: very high. 3D QA can support spatial search, robotics, inspection, smart homes, education, accessibility, and AR assistants.

Project value: very high. This is probably the closest category to the planned third stage: `objects.json` -> metric edges -> `object_graph.json` -> QA.

Relatedness: high.

### 3.7 3D Vision / 3D Generation

3D generation creates scenes, layouts, objects, or editable 3D assets. It connects to spatial reasoning because generated scenes must obey spatial constraints: object size, placement, support, accessibility, collision, and functional layout.

Layout generation methods such as LayoutGPT and Layout-your-3D focus on arranging objects or rooms according to language, constraints, and common spatial priors. Programmatic 3D generation methods such as 3D-GPT and CAD-Recode represent scenes or objects through code-like programs, CAD steps, or structured commands. Specific paper details are待进一步查阅.

Research outlook: high for content creation, simulation, interior design, robotics training environments, and CAD automation.

Project value: low to medium for the current stage. It is not needed for 2D-to-3D semantic grounding or object graph construction, but could become a future extension: use SplatGraph scene graphs as constraints for scene editing, reconstruction QA, or simulation generation.

Relatedness: low.

### 3.8 Embodied AI / Vision-Language Navigation

Vision-language navigation asks an agent to move through an environment according to language goals, such as "go to the chair near the table" or "find the bottle on the counter." It requires scene understanding, interpreting user intent, planning paths, and updating beliefs from new observations.

The taxonomy groups this into scene understanding, intention interpretation, and planning/navigation. Scene understanding identifies objects and layout. Intention interpretation maps vague language into actionable spatial goals. Planning and navigation choose movements while respecting obstacles, uncertainty, and changing observations. Listed works include Spartun3D, GSA-VLN, AutoSpatial, LL3DA, NavVLM, and NavCoT; details are待进一步查阅.

Research outlook: high for service robots, warehouse robots, AR navigation, and assistive technology.

Project value: medium as a future extension. SplatGraph's object graph could become a spatial memory for an agent, but current code does not yet include robot state, actions, path planning, or online updates.

Relatedness: medium.

### 3.9 Embodied AI / Embodied Question Answering

Embodied QA asks an agent to answer questions by actively exploring, observing, and remembering. Unlike ordinary VQA or static 3D QA, the system may need to move to reveal hidden objects, revisit locations, or combine observations over time.

The taxonomy lists OpenEQA and EMBOSR. OpenEQA is a representative embodied QA direction where agents are evaluated on open-vocabulary questions grounded in environments. EMBOSR details are待进一步查阅.

Research outlook: high for home robots, AR assistants, inspection, and long-horizon autonomous systems.

Project value: medium. SplatGraph can provide a static scene memory and metric object graph, which is one ingredient for embodied QA. Active exploration and action history are later-stage additions.

Relatedness: medium.

### 3.10 Embodied AI / Embodied Grasping

Embodied grasping uses perception, language, and geometry to select and execute grasps. Spatial reasoning matters because a robot must identify the target object, estimate its 3D pose and free space, avoid collisions, and understand task constraints such as "grab the handle" or "pick the bottle without knocking over the kettle."

The taxonomy lists ThinkGrasp and FreeGrasp; details are待进一步查阅.

Research outlook: high for robotics and manipulation.

Project value: low for the current phase. The project's object bboxes and semantic labels could help future grasp target selection, but it currently lacks robot kinematics, grasp poses, contact models, and manipulation policies.

Relatedness: low.

### 3.11 Embodied AI / Vision-Language Action

Vision-language-action models connect visual and language understanding to low-level or high-level actions. They must translate instructions into movement, manipulation, or interaction policies.

The taxonomy lists 3D-VLA, pi0.5, and Chat-VLA2. Specific model details are待进一步查阅. The spatial skills needed between perception and action include object localization, affordance estimation, relation reasoning, obstacle awareness, memory, and temporal prediction.

Research outlook: very high for general-purpose robots and interactive assistants.

Project value: low to medium. SplatGraph could be a perception and spatial memory module for a future VLA stack, but action control is outside the current repository.

Relatedness: low.

### 3.12 Embodied AI / Embodied World Model

An embodied world model predicts how the environment changes as an agent moves or acts. It needs spatial memory, object permanence, object relations, dynamics, uncertainty, and action-conditioned prediction.

The taxonomy lists TesserAct and EVA; details are待进一步查阅.

Research outlook: very high for long-term autonomous agents, robotics simulation, planning, and continual scene understanding.

Project value: medium as a long-term goal. SplatGraph's future 3.5 plan for cross-time scene change understanding is a small step toward world-model-like spatial memory, but current work is static-scene graph construction.

Relatedness: medium.

### 3.13 Novel Modalities / Video-Based Spatial Reasoning

Video adds temporal evidence that static images lack: object motion, occlusion removal, interaction, stability, causality, and time-varying relations. It can answer questions like "what moved", "what was behind the cup", or "did the bottle fall after contact."

The taxonomy lists VideoLLaMA2, VideoINSTA, Video-R1, and SpaceR. Specific paper details are待进一步查阅.

Research outlook: high. Video spatial reasoning is important for surveillance, robotics, sports, smart homes, AR, and physical reasoning.

Project value: medium. Current SplatGraph already uses many frames, but mainly as static multi-view evidence for semantic voting. A future dynamic version could compare object graphs over time and reason about object movement or scene changes.

Relatedness: medium.

### 3.14 Novel Modalities / Audio-Based Spatial Reasoning

Audio-based spatial reasoning uses sound to infer location, direction, distance, room structure, events, or hidden activity. Spatial audio can complement vision when objects are occluded, outside the camera view, or visually ambiguous.

The taxonomy lists STARSS23, SpatialSoundQA, ACORN, and SAVVY. Specific paper details are待进一步查阅.

Research outlook: medium to high for robotics, AR/VR, hearing assistance, meeting rooms, smart homes, and multimodal monitoring.

Project value: low for now. The current project has RGB-D, point clouds, trajectories, masks, and Gaussian geometry, but no audio pipeline. It is a remote future extension.

Relatedness: low.

## 4. Most Relevant Directions for This Project

1. 3D Scene Reasoning and QA: This is the closest match to the planned SplatGraph stage. The project is already moving from `objects.json` to metric object graph construction, which can support questions about distance, support, height, and object relations.

2. 3D Visual Grounding: The current pipeline is a hybrid 2D+3D grounding route. It uses open-vocabulary 2D masks, camera poses, and Gaussian points to produce 3D object instances.

3. General MLLM Tool Use / Test-Time Scaling: The project can expose geometry calculations as tools or structured evidence, then test whether MLLMs answer spatial questions more reliably.

4. Model Design for Spatial MLLMs: The current practical question is not training a new model, but designing the right LLM-readable representation: JSON object graph, relation triples, metric tables, and evidence summaries.

5. Explainability: Existing debug artifacts can become explanation evidence: 2D overlays, per-prompt Gaussian PLYs, instance bbox PLYs, and graph edge metrics.

6. Post-Training: Medium-term. Once graph-derived QA pairs exist, they can become data for SFT, reward design, or benchmark construction.

## 5. Possible Project Extensions

- 3D scene QA: generate deterministic QA from `object_graph.json`, then compare rule-based answers and LLM answers.
- Language-to-3D grounding: accept text queries such as "the kettle closest to the water bottle" and return object ids or bboxes.
- 3D scene graph reasoning: compute edges such as near, above, below, support, contact, and overlap from object bboxes.
- Agent-based spatial reasoning: wrap graph queries and geometry checks as tools for a planning or reasoning agent.
- Embodied navigation: use the object graph as a map-level semantic memory for navigation goals.
- Video-based dynamic spatial reasoning: compare object graphs across time for movement, disappearance, and relation change.
- Audio-based spatial reasoning: long-term extension for sound source localization or multimodal scene memory.
- VLA / grasping integration: long-term extension where object graph nodes become target candidates for manipulation.

## 6. Notes for Future Codex Sessions

- Read this document before doing research-positioning, related-work, or spatial-reasoning design tasks.
- Treat the taxonomy as a direction map, not verified paper summaries.
- Do not invent details for named papers. If a task needs a specific paper, look it up separately and cite or record the source.
- Keep project judgments tied to verifiable repository state: `AGENTS.md`, `core/debug_semantic_2d.py`, `core/semantic_vote_gaussians.py`, `core/extract_semantic_object_bboxes.py`, and actual run outputs such as `objects.json`.
- The highest-priority near-term technical route is object graph construction from grounded Gaussian object instances.
- This document is for project planning and paper motivation, not a strict survey.
