# SlamOnline

SlamOnline is the cleaned project root for the current SplatGraph online pipeline:

```text
RGB-D dataset
  -> MonoGS same-run Gaussian reconstruction
  -> file-backed semantic queue
  -> Grounding DINO + SAM 2 worker
  -> 2D observations lifted to 3D Gaussian observations
  -> object memory, refinement, visualization, and scene export
```

The original SplatGraph repository remains the source/history workspace. New work should use this directory as the main project root and the `SlamOnline` conda environment.

## Semantic Backend

SlamOnline currently uses the detector-first Grounding DINO + SAM 2 backend:

| Worker | Backend | Use case |
|---|---|---|
| `slam_online/core/online_grounded_sam2_worker.py` | Grounding DINO box detection + SAM 2 mask refinement | Current main route. If a prompt has no detection, it writes an empty mask instead of forcing a false-positive region. |

The old LangSplatV2/SAM+CLIP prompt-scoring route has been removed from this
project. The active 2D semantic backend is Grounding DINO + SAM 2.

## Environment Deployment

These steps create the `SlamOnline` environment for MonoGS, Open3D, and the
local 3DGS CUDA extensions. Grounding DINO + SAM 2 should run in the separate
`Semantic2D` environment described below.

### 1. Create The Conda Environment

```bash
cd /home/sky/czh/SplatGraph/SlamOnline

conda create -n SlamOnline python=3.9 pip=22.3.1
conda activate SlamOnline

conda install -c pytorch -c nvidia \
  pytorch=2.1.0 torchvision=0.16.0 torchaudio=2.1.0 pytorch-cuda=12.1
```

### 2. Install Python Dependencies

Use the pinned `numpy` and `setuptools` versions. NumPy 2.x and very new setuptools can break the old-style MonoGS CUDA extension build.

```bash
cd /home/sky/czh/SplatGraph/SlamOnline
conda activate SlamOnline

python -m pip install "setuptools==68.2.2" wheel ninja packaging --force-reinstall
python -m pip install "numpy==1.26.4" --force-reinstall
python -m pip install --default-timeout 120 --retries 10 -r requirement.txt
```

If PyPI is unstable, use a mirror:

```bash
python -m pip install \
  --default-timeout 120 \
  --retries 10 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -r requirement.txt
```

### 3. Expose PyTorch Runtime Libraries

`simple_knn._C` may fail with `ImportError: libc10.so` unless PyTorch's `torch/lib` directory is on `LD_LIBRARY_PATH`.

For the current shell:

```bash
export LD_LIBRARY_PATH="$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib")'):${LD_LIBRARY_PATH}"
```

To make this automatic whenever `conda activate SlamOnline` runs:

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/slamonline_torch_lib.sh" <<'EOF'
export SLAMONLINE_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib")'):${LD_LIBRARY_PATH:-}"
EOF

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/slamonline_torch_lib.sh" <<'EOF'
export LD_LIBRARY_PATH="${SLAMONLINE_OLD_LD_LIBRARY_PATH:-}"
unset SLAMONLINE_OLD_LD_LIBRARY_PATH
EOF
```

Reactivate the environment after creating these hooks:

```bash
conda deactivate
conda activate SlamOnline
```

Known symptom and fix:

```text
ImportError: libc10.so: cannot open shared object file: No such file or directory
```

This means the CUDA extension was built, but the runtime linker cannot find PyTorch shared libraries. Run the `export LD_LIBRARY_PATH=...` command above, then create the conda activate/deactivate hooks so the fix persists across terminals.

### 4. Build MonoGS CUDA Extensions

Use `setup.py develop` for these old extension packages.

```bash
cd /home/sky/czh/SplatGraph/SlamOnline/third_party/MonoGS/submodules/simple-knn
python setup.py develop

cd /home/sky/czh/SplatGraph/SlamOnline/third_party/MonoGS/submodules/diff-gaussian-rasterization
python setup.py develop
```

### 5. Verify The Environment

```bash
cd /home/sky/czh/SplatGraph/SlamOnline
conda activate SlamOnline

python -c "import numpy, torch; print('numpy', numpy.__version__); print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import cv2, open3d; print('basic imports ok')"
python -c "import simple_knn._C; print('simple_knn ok')"
python -c "import diff_gaussian_rasterization; print('rasterizer ok')"
```

Expected key output:

```text
numpy 1.26.4
torch 2.1.0 12.1 True
basic imports ok
simple_knn ok
rasterizer ok
```

Verified local result on this machine:

```text
simple_knn ok
rasterizer ok
```

### 6. Queue Retry Helpers

If a worker batch failed, move failed tasks back to pending:

```bash
python slam_online/core/online_semantic_queue.py retry-failed \
  --queue_root /home/sky/czh/SplatGraph/SlamOnline/output/queue
```

If the worker already completed but you changed the prompt list, move done tasks back to pending and rerun the worker with the new prompts:

```bash
python slam_online/core/online_semantic_queue.py retry-done \
  --queue_root /home/sky/czh/SplatGraph/SlamOnline/output/queue
```

### 7. Semantic2D: Grounding DINO + SAM 2 Backend

The detector-first worker is:

```text
slam_online/core/online_grounded_sam2_worker.py
```

It writes queue-compatible result JSON files and per-prompt masks, so
the downstream 3D lifting and object-memory scripts can consume the worker
outputs. See `slam_online/core/core_readme.md` for the current command sequence.

Install ordinary HuggingFace dependencies in the current `SlamOnline` env. Do
not install the newest `transformers` blindly: recent releases expect newer
PyTorch pytree APIs than torch 2.1 provides.

```bash
cd /home/sky/czh/SplatGraph/SlamOnline
conda activate SlamOnline
python -m pip install \
  "transformers==4.40.2" \
  accelerate \
  safetensors \
  "huggingface-hub==0.23.5"
```

SAM 2 has a stricter official environment than the current MonoGS-oriented
SlamOnline base. The SAM 2 repo states `python>=3.10`, `torch>=2.5.1`, and
`torchvision>=0.20.1`; Grounded-SAM-2 uses Python 3.10 and torch >= 2.3.1 in
its demo environment. Our base SlamOnline env is Python 3.9 / torch 2.1.0 to
keep MonoGS CUDA extensions stable. Do not force SAM 2 into the base env.

Create a separate `Semantic2D` conda environment for Grounding DINO + SAM 2.
It reads and writes the same file-backed queue/output paths, so it can run
alongside or after MonoGS without sharing Python packages with MonoGS:

```bash
conda create -n Semantic2D python=3.10 -y
conda activate Semantic2D

python -m pip install \
  --default-timeout 120 \
  --retries 10 \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install \
  --default-timeout 120 \
  --retries 10 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -r /home/sky/czh/SplatGraph/SlamOnline/requirements-grounded-sam2.txt
```

If you installed the environment before RAM++ dependencies were added, update it
in place:

```bash
conda activate Semantic2D
python -m pip install scipy fairscale openai
```

If GitHub access is unstable, configure git proxy before installing SAM 2:

```bash
git config --global http.proxy socks5h://127.0.0.1:7897
git config --global https.proxy socks5h://127.0.0.1:7897
```

RAM++ is optional but recommended for automatic prompt discovery. Keep it in the
same `Semantic2D` environment. Place the RAM++ checkpoint here, or pass another
path with `--checkpoint`:

```text
/home/sky/czh/SplatGraph/SlamOnline/third_party/recognize-anything/pretrained/ram_plus_swin_large_14m.pth
```

Recommended HuggingFace download command with resume support:

```bash
cd /home/sky/czh/SplatGraph/SlamOnline
conda activate Semantic2D

mkdir -p /home/sky/czh/SplatGraph/SlamOnline/third_party/recognize-anything/pretrained

python - <<'PY'
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="xinyu1205/recognize-anything-plus-model",
    filename="ram_plus_swin_large_14m.pth",
    local_dir="/home/sky/czh/SplatGraph/SlamOnline/third_party/recognize-anything/pretrained",
    local_dir_use_symlinks=False,
    resume_download=True,
)
print("download ok:", path)
PY
```

If the terminal does not use your VPN automatically, export the proxy first:

```bash
export HTTP_PROXY=socks5h://127.0.0.1:7897
export HTTPS_PROXY=socks5h://127.0.0.1:7897
export ALL_PROXY=socks5h://127.0.0.1:7897

python -m pip install socksio "httpx[socks]"
```

`wget` fallback with an explicit SOCKS proxy:

```bash
wget -c \
  -e use_proxy=yes \
  -e https_proxy=socks5h://127.0.0.1:7897 \
  -O /home/sky/czh/SplatGraph/SlamOnline/third_party/recognize-anything/pretrained/ram_plus_swin_large_14m.pth \
  "https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth"
```

Check the downloaded checkpoint:

```bash
ls -lh /home/sky/czh/SplatGraph/SlamOnline/third_party/recognize-anything/pretrained/ram_plus_swin_large_14m.pth
```

Expected size is about `2.9G` to `3.0G`.

Single-image auto-prompt smoke test:

```bash
cd /home/sky/czh/SplatGraph/SlamOnline
conda activate Semantic2D

python slam_online/core/auto_prompt_ram.py \
  --image /home/sky/czh/SplatGraph/SlamOnline/datasets/new_room0/images/frame_000265.jpg \
  --output_root /home/sky/czh/SplatGraph/SlamOnline/output/auto_prompts_ram \
  --max_tags 20
```

Queue-based auto-prompt test on the current Replica room0 third run:

```bash
cd /home/sky/czh/SplatGraph/SlamOnline
conda activate Semantic2D

python slam_online/core/auto_prompt_ram.py \
  --queue_root /home/sky/czh/SplatGraph/SlamOnline/output/replica_room0_third/queue \
  --queue_states done \
  --output_root /home/sky/czh/SplatGraph/SlamOnline/output/replica_room0_third/auto_prompts_ram \
  --max_images 5 \
  --max_tags 20
```

Expected output: a timestamped `auto_prompts_YYYYMMDD_HHMMSS/` directory with
per-frame `*_auto_prompts.json`, scene-level `auto_prompts.json`, and
`summary.txt`. RAM++ outputs normalized `raw_tags` only; it does not decide
which tags are valid object prompts. The LLM prompt filter below converts
`raw_tags` into `object_prompts`.

Filter RAM++ tags with an LLM prompt classifier before running Grounded-SAM2:

```bash
cd /home/sky/czh/SplatGraph/SlamOnline
conda activate Semantic2D

cp configs/llm_prompt_filter.example.json configs/llm_prompt_filter.json
# Edit configs/llm_prompt_filter.json and fill in model/base_url/api_key.

python slam_online/core/auto_prompt_filter.py \
  --auto_prompt_json /home/sky/czh/SplatGraph/SlamOnline/output/auto_prompts_ram/auto_prompts_20260610_105916/frame_000265_auto_prompts.json \
  --output_root /home/sky/czh/SplatGraph/SlamOnline/output/auto_prompt_filter \
  --llm_config /home/sky/czh/SplatGraph/SlamOnline/configs/llm_prompt_filter.json
```

Expected output: a timestamped `prompt_filter_YYYYMMDD_HHMMSS/` directory with
`frame_000265_prompt_filter.json`, `frame_000265_object_prompts.txt`,
`frame_000265_held_out_prompts.txt`, and `summary.txt`. The object prompts text
file is the next input to Grounded-SAM2.

`configs/llm_prompt_filter.json` is ignored by git. Do not paste or commit API
keys into tracked files.

Validate RAM++ prompts on one image with Grounding DINO + SAM 2 without touching
the queue:

```bash
cd /home/sky/czh/SplatGraph/SlamOnline
conda activate Semantic2D

python slam_online/core/online_grounded_sam2_worker.py \
  --image /home/sky/czh/SplatGraph/SlamOnline/datasets/new_room0/images/frame_000265.jpg \
  --prompts_file /home/sky/czh/SplatGraph/SlamOnline/output/auto_prompt_filter/prompt_filter_YYYYMMDD_HHMMSS/frame_000265_object_prompts.txt \
  --output_root /home/sky/czh/SplatGraph/SlamOnline/output/grounded_sam2_single \
  --grounding_model_id IDEA-Research/grounding-dino-base \
  --sam2_model_id facebook/sam2-hiera-base-plus \
  --box_threshold 0.35 \
  --text_threshold 0.25 \
  --min_mask_area_ratio 0.002 \
  --max_mask_area_ratio 0.35 \
  --max_detections_per_prompt 5
```

Expected output: `single_YYYYMMDD_HHMMSS/frame_000265/` with per-prompt
`*_mask.png`, `*_overlay.jpg`, `*_detections.json`, plus
`prompt_validation.json` and `summary.txt`.

First smoke test, after re-queueing tasks:

```bash
cd /home/sky/czh/SplatGraph/SlamOnline
conda activate Semantic2D

python slam_online/core/online_semantic_queue.py retry-done \
  --queue_root /home/sky/czh/SplatGraph/SlamOnline/output/queue

python slam_online/core/online_grounded_sam2_worker.py \
  --queue_root /home/sky/czh/SplatGraph/SlamOnline/output/queue \
  --output_root /home/sky/czh/SplatGraph/SlamOnline/output/online_grounded_sam2_2d \
  --prompts "couch,pillow,potted plant,lamp,door" \
  --grounding_model_id IDEA-Research/grounding-dino-base \
  --sam2_model_id facebook/sam2-hiera-base-plus \
  --box_threshold 0.35 \
  --text_threshold 0.25 \
  --max_tasks 1
```

Expected behavior for a frame with no visible door: `door_mask.png` should be
black and `door_detections.json` should contain no kept detections.

## Run Notes

Start with [slam_online.md](slam_online.md) for the full workflow, current run commands, known limits, and object-first roadmap.

### Shell Placeholder Note

Do not paste angle-bracket placeholders such as `<FUSION_RUN>` directly into zsh/bash commands. In zsh, `<FUSION_RUN>` is parsed as input redirection and fails with:

```text
zsh: no such file or directory: FUSION_RUN
```

Use the real path printed by the previous command, or assign it to a shell variable:

```bash
SCENE_OUT=/home/sky/czh/SplatGraph/SlamOnline/output/replica_office3_third
LIFT_DIR=$SCENE_OUT/04_lift3d/lift_YYYYMMDD_HHMMSS

python slam_online/core/object_memory_update.py \
  --frame_3d_observations_json "$LIFT_DIR/frame_3d_observations.json" \
  --output_root "$SCENE_OUT/05_memory"
```
