# BEHAVIOR 数据集制备本地部署总结

日期：2026-06-22  
分支：`data`  
本地资源假设：RTX 3090 24GB，约 100GB 可用空间。

## 结论

你的本地机器可以做 BEHAVIOR 数据制备的 smoke test、小 subset 恢复、小 subset H5 -> WDS 转换；不适合完整 BEHAVIOR raw episode 全量生成。

更推荐的路线是：

1. 从 Hugging Face 下载 PointWorld 已发布的 BEHAVIOR generated H5 小 subset，比如一个 task package。
2. 恢复成 `behavior/flows/task-*/episode_*.hdf5`。
3. 跑 integrity check。
4. 生成 manifest。
5. 转成 WDS。
6. 切回 `main` 或 `siglip` 分支训练。

不推荐在 100GB 空间下完整跑 raw BEHAVIOR -> generated H5，因为 README 明确说 full pipeline 是 multi-terabyte 级别；如果保留中间文件，可能超过 10TB。BEHAVIOR raw list 里有 7757 个 episode，全量不是本地单卡小盘任务。

## 两条路线

### 路线 A：下载已生成 H5，再转 WDS

这是你本地最现实的路线。

需要环境：

- 普通 Python 环境，不需要 OmniGibson Docker。
- `conda` / `pip`
- `huggingface-cli`
- `zstd`
- `tar`
- 足够磁盘空间存放你下载的 subset、恢复后的 H5、WDS 输出。

安装 Python 环境：

```bash
conda create -n pointworld-data python=3.10
conda activate pointworld-data
git submodule update --init --recursive
python -m pip install -r requirements.txt
python -m pip install --no-deps urdfpy==0.0.22
python -m pip install -r environments/requirements_urdfpy_runtime.txt
pip install -e third_party/co-tracker
pip install -e third_party/vggt
```

`zstd`：

```bash
conda install -c conda-forge zstd
```

下载一个 BEHAVIOR task subset 示例：

```bash
huggingface-cli download nvidia/PointWorld-BEHAVIOR \
  --repo-type dataset \
  --include "recover_dataset_from_parts.sh" \
  --include "behavior/flows/task-0000/*" \
  --local-dir /path/to/PointWorld-BEHAVIOR-subset
```

恢复：

```bash
bash /path/to/PointWorld-BEHAVIOR-subset/recover_dataset_from_parts.sh \
  --packages /path/to/PointWorld-BEHAVIOR-subset \
  --out /path/to/pointworld_behavior_subset_restored \
  --include-prefix behavior/flows/task-0000 \
  --threads 0
```

恢复后应有：

```text
/path/to/pointworld_behavior_subset_restored/behavior/flows/task-*/episode_*.hdf5
```

转 WDS：

```bash
python data_integrity_check.py \
  --input_dir /path/to/pointworld_behavior_subset_restored/behavior/flows \
  --domain behavior \
  --num_mp_workers 4

python make_wds_manifest.py \
  --input_dir /path/to/pointworld_behavior_subset_restored/behavior/flows \
  --domain behavior \
  --output_manifest /path/to/pointworld_behavior_subset_restored/behavior/flows/wds_manifest_seed42_test0.1.json

python convert_wds.py \
  --input_dir /path/to/pointworld_behavior_subset_restored/behavior/flows \
  --output_dir /path/to/pointworld_behavior_subset_restored/behavior/wds \
  --domain behavior \
  --manifest /path/to/pointworld_behavior_subset_restored/behavior/flows/wds_manifest_seed42_test0.1.json
```

100GB 空间建议：

- 只下载一个或少数几个 `task-*` package。
- 下载包、恢复 H5、WDS 三者会同时占空间；恢复完成并确认 WDS 可用后，可以删除下载包或中间恢复目录节省空间。
- 不要下载完整 `nvidia/PointWorld-BEHAVIOR`。

## 路线 B：从 BEHAVIOR raw episode 重新生成 H5

这是复现实验生成流程，而不是简单准备训练数据。它需要 OmniGibson / Isaac / BEHAVIOR-1K runtime，推荐 Docker。

需要环境：

- Linux + NVIDIA driver
- Docker
- NVIDIA Container Toolkit
- 可运行 `docker run --gpus all`
- Hugging Face 网络访问；如果 raw dataset 需要权限，设置 `HF_TOKEN`
- 约 40GB 的 OmniGibson / BEHAVIOR assets 目录
- 额外 cache 目录 `POINTWORLD_CACHE_DIR`
- 输出目录 `BEHAVIOR_ROOT`
- 推荐远大于 100GB 的空间；100GB 只适合处理极少量 episode。

构建 Docker image：

```bash
docker build -f docker/dockerfile_behavior -t pointworld-behavior:v3.7.2 .
```

初始化 OmniGibson / BEHAVIOR assets，一次性约 40GB：

```bash
export OG_DATA_ROOT=/path/to/your/og_data
mkdir -p "$OG_DATA_ROOT"

docker run --rm --gpus all --ipc=host \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e OMNI_KIT_ACCEPT_EULA=YES \
  -e OMNIGIBSON_DATA_PATH=/data \
  -v "$OG_DATA_ROOT":/data \
  pointworld-behavior:v3.7.2 \
  bash -lc 'cd /BEHAVIOR-1K && ./setup.sh --bddl --omnigibson --dataset --accept-nvidia-eula --accept-dataset-tos --confirm-no-conda'
```

它会生成：

```text
$OG_DATA_ROOT/behavior-1k-assets
$OG_DATA_ROOT/omnigibson-robot-assets
$OG_DATA_ROOT/2025-challenge-task-instances
$OG_DATA_ROOT/omnigibson.key
```

运行生成脚本：

```bash
export POINTWORLD_REPO=/path/to/PointWorld
export OG_DATA_ROOT=/path/to/your/og_data
export POINTWORLD_CACHE_DIR=/path/to/local/cache/dir
export BEHAVIOR_ROOT=/path/to/processed/behavior/outputs
mkdir -p "$POINTWORLD_CACHE_DIR" "$BEHAVIOR_ROOT/flows"

docker run --rm --gpus all --ipc=host --ulimit core=0 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e OMNIGIBSON_HEADLESS=1 \
  -e OMNIGIBSON_DATA_PATH=/data \
  -e OMNIGIBSON_DATASET_PATH=/data/behavior-1k-assets \
  -e OMNIGIBSON_ASSET_PATH=/data/omnigibson-robot-assets \
  -e OMNIGIBSON_KEY_PATH=/data/omnigibson.key \
  -e POINTWORLD_CACHE_DIR=/cache/pointworld \
  -v "$POINTWORLD_REPO":/workspace/point-world \
  -v "$OG_DATA_ROOT":/data \
  -v "$POINTWORLD_CACHE_DIR":/cache/pointworld \
  -v "$BEHAVIOR_ROOT":/workspace/behavior \
  pointworld-behavior:v3.7.2 \
  bash -lc "cd /workspace/point-world && \
  python simulation/behavior_3d_flows.py \
    --headless \
    --input_list simulation/behavior_paths.txt \
    --output_root /workspace/behavior/flows \
    --rank 0 \
    --world_size 1"
```

本地 3090 + 100GB 的注意事项：

- 第一次运行会有 OmniGibson / Isaac extension sync、shader compilation、scene material setup，可能等好几分钟才开始处理。
- `simulation/behavior_paths.txt` 全量是 7757 个 raw episode；不要直接全量跑。
- 先复制一个很小的 input list，例如只放 1 到 5 行：

```bash
head -n 1 simulation/behavior_paths.txt > simulation/behavior_paths_local_smoke.txt
```

然后把命令里的 `--input_list` 改为：

```bash
--input_list simulation/behavior_paths_local_smoke.txt
```

- `POINTWORLD_CACHE_DIR` 会缓存 Hugging Face raw episode。100GB 空间下要定期清理 cache 和失败的中间输出。
- `BEHAVIOR_ROOT/flows` 会产生 generated H5；后续还要 WDS 输出，空间会继续增长。
- 24GB 3090 对默认 180x320、单 worker、headless smoke 运行应更现实；不建议同机多 worker。

## 第三方 checkpoint 是否必须

BEHAVIOR raw -> generated H5 主要依赖 OmniGibson / BEHAVIOR-1K runtime，不依赖 DROID 那套 FoundationStereo / CoTracker / VGGT 视觉标注 pipeline。

README 的第三方 checkpoint 部分主要是完整数据生成工具链一起列出，尤其 DROID 会用到：

- CoTracker3
- VGGT-1B
- FoundationStereo

如果你只走 BEHAVIOR Docker 生成，可以先不下载这些大 checkpoint。  
如果你走路线 A，也不需要这些 checkpoint。

## 推荐你本地的最小部署方案

我建议先做路线 A：

1. `conda create -n pointworld-data python=3.10`
2. `pip install -r requirements.txt`
3. 安装 `urdfpy` runtime 兼容依赖
4. `conda install -c conda-forge zstd`
5. 下载 `task-0000` 已生成 H5 subset
6. 恢复
7. integrity check
8. manifest
9. convert WDS

这样不需要 Docker / OmniGibson，不会被 Isaac shader 编译和 40GB assets 卡住，也更适合 100GB 空间。

等路线 A 跑通后，再考虑路线 B 的 1 episode smoke test。

## 本地目录规划建议

在 100GB 下尽量把路径分开，方便删除：

```text
/data/pointworld_behavior_downloads      # HF packaged parts，可删
/data/pointworld_behavior_restored       # restored generated H5
/data/pointworld_behavior_wds            # WDS train/test shards
/data/pointworld_cache                   # raw streaming cache，仅路线 B 需要
/data/og_data                            # OmniGibson assets，仅路线 B 需要
```

如果只做路线 A，小 subset 下可以不建 `pointworld_cache` 和 `og_data`。

## 风险点

- 完整 BEHAVIOR package 和完整 raw generation 都不是 100GB 级别任务。
- Docker image + OG assets + cache + output H5 + WDS 可能很快超过 100GB。
- Hugging Face raw dataset 如果 gated/private，需要 `HF_TOKEN`。
- Docker 路线要求 NVIDIA Container Toolkit 正常工作。
- WDS 转换需要 manifest；如果只恢复 subset，不要用 full paper manifest，应该自己生成 subset manifest。
