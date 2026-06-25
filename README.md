# SlamOnline

SlamOnline 是一个面向在线语义 3D 重建的整理版项目根目录，当前主线是：

```text
RGB-D 数据
  -> MonoGS 在线 3D Gaussian 重建
  -> 文件队列保存关键帧任务
  -> Grounding DINO + SAM 2 生成 2D 语义观测
  -> 将 2D 观测提升到 3D Gaussian
  -> 构建、清理、可视化并导出 object memory
```

当前项目已经移除旧的 LangSplatV2/SAM+CLIP 路线，主要语义后端为
Grounding DINO + SAM 2。

## 项目结构

```text
SlamOnline/
  README.md                         # GitHub 项目首页和快速开始
  requirement.txt                   # SlamOnline 主环境依赖
  requirements-grounded-sam2.txt    # Semantic2D 语义环境依赖
  slam_online.md                    # 项目架构、迁移状态和路线说明
  docs/
    environment_setup.md            # 详细环境安装、排错和历史命令备份
  envs/
    SlamOnline.reference.yml        # 参考 conda 环境
  slam_online/core/
    core_readme.md                  # 当前核心流水线的详细运行手册
  third_party/MonoGS/               # MonoGS 第三方代码
```

## 环境配置

本项目建议使用两个 conda 环境：

- `SlamOnline`：运行 MonoGS、3D Gaussian 相关处理、3D lifting 和 object memory。
- `Semantic2D`：运行 Grounding DINO + SAM 2、RAM++ 和 2D 语义观测生成。

### 1. SlamOnline 主环境

```bash
cd /home/sky/czh/SplatGraph/SlamOnline

conda create -n SlamOnline python=3.9 pip=22.3.1
conda activate SlamOnline

conda install -c pytorch -c nvidia \
  pytorch=2.1.0 torchvision=0.16.0 torchaudio=2.1.0 pytorch-cuda=12.1

python -m pip install "setuptools==68.2.2" wheel ninja packaging --force-reinstall
python -m pip install "numpy==1.26.4" --force-reinstall
python -m pip install -r requirement.txt
```

构建 MonoGS 使用的 CUDA 扩展：

```bash
cd third_party/MonoGS/submodules/simple-knn
python setup.py develop

cd ../diff-gaussian-rasterization
python setup.py develop
```

如果运行时报 `libc10.so` 找不到，可以先在当前 shell 中执行：

```bash
export LD_LIBRARY_PATH="$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib")'):${LD_LIBRARY_PATH}"
```

### 2. Semantic2D 语义环境

SAM 2 对 Python 和 PyTorch 版本要求更高，因此建议单独创建环境：

```bash
conda create -n Semantic2D python=3.10 -y
conda activate Semantic2D

python -m pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install -r /home/sky/czh/SplatGraph/SlamOnline/requirements-grounded-sam2.txt
```

RAM++ checkpoint、SAM 2 下载、代理和常见安装问题见
[docs/environment_setup.md](docs/environment_setup.md)。

## 快速运行

### 1. 运行 MonoGS 并生成队列

```bash
cd /home/sky/czh/SplatGraph/SlamOnline/third_party/MonoGS
conda activate SlamOnline

python slam.py --config configs/rgbd/replica/office3_third_online_semantic_save.yaml
```

输出通常包括：

```text
output/.../point_cloud/final/point_cloud.ply
output/.../queue/
```

### 2. 查看队列状态

```bash
cd /home/sky/czh/SplatGraph/SlamOnline
conda activate SlamOnline

python slam_online/core/online_semantic_queue.py status \
  --queue_root /path/to/scene_output/queue
```

### 3. 运行当前核心流水线

当前完整流程以 [slam_online/core/core_readme.md](slam_online/core/core_readme.md)
为准，主要步骤包括：

1. `auto_prompt_ram.py`：从队列图像生成 RAM++ 标签。
2. `auto_prompt_filter.py`：把标签过滤成每帧 object prompts。
3. `online_grounded_sam2_worker.py`：生成 2D frame observations。
4. `lift_frame_observations_to_3d.py`：把 2D 观测提升到 3D Gaussian。
5. `object_memory_update.py`：构建 object memory。
6. `refine_object_memory_conflicts.py`、`merge_object_memory_global.py`、`refine_final_object_memory.py`：清理重复或局部对象。
7. `visualize_object_memory.py`、`export_scene_objects.py`：输出可视化和场景对象 JSON。

## 配置说明

`configs/llm_prompt_filter.example.json` 是可上传的示例配置。

真实运行时可以复制为本地配置：

```bash
cp configs/llm_prompt_filter.example.json configs/llm_prompt_filter.json
```

然后填写自己的 `base_url`、`model` 和 `api_key`。
`configs/llm_prompt_filter.json` 属于本地私密配置，不建议上传到 GitHub。

## 重要文档

- [slam_online/core/core_readme.md](slam_online/core/core_readme.md)：当前核心代码的详细使用说明。
- [slam_online.md](slam_online.md)：项目定位、架构、迁移来源、已知限制和后续方向。
- [docs/environment_setup.md](docs/environment_setup.md)：详细环境配置、下载说明、排错和历史命令备份。
- [docs/research-directions.md](docs/research-directions.md)：研究方向记录。

## 不建议上传的内容

以下内容建议仅本地保留，或通过 `.gitignore` 排除：

- `datasets/`
- `output/`
- `paper/`
- 模型权重和 checkpoint，例如 `*.pth`、`*.pt`、`*.ckpt`
- 日志、缓存、构建产物和实验结果
- `configs/llm_prompt_filter.json`
- `HANDOFF.md`

## 当前状态

- MonoGS 第三方代码保留在 `third_party/MonoGS/`。
- `third_party/LangSplatV2/` 已移除。
- `slam_online/core/pass/` 已移除。
- 当前核心代码集中在 `slam_online/core/`。

