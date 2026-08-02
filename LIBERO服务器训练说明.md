# LIBERO 数据在服务器上的训练与评估

本文对应 GitHub `libero` 分支。该分支支持 LIBERO Panda 的四视角场景点流、
由 qpos + URDF 在线生成机器人点流、移动物体选择性保密集采样、12,000 点上限，
以及所有有效场景点完全等权的训练 loss。5 mm selector 仅用于 moved/static 指标，
不会调整 loss 权重。

## 1. 获取代码

```bash
git clone git@github.com:cowboy446/PointWorld.git
cd PointWorld
git fetch origin libero
git switch libero
git pull --ff-only origin libero
```

若仓库已经存在：

```bash
cd /path/to/PointWorld
git fetch origin
git switch libero
git pull --ff-only origin libero
```

## 2. 准备环境和模型权重

按照仓库 `README.md` 创建 `pointwm` 环境，并准备 DINOv3 ViT-L16 权重：

```bash
conda activate pointwm
export POINTWORLD_PYTHON="$(which python)"
```

DINOv3 checkpoint 必须位于仓库原有代码所要求的位置。若服务器 CUDA、PyTorch 或
flash-attn 版本不同，先按 `README.md` 完成环境验证。

## 3. 上传并检查数据

把生成的 `pointworld_wds` 整个目录上传到服务器，目录必须保留为：

```text
pointworld_wds/
├── metadata_rank0.json
├── split_manifest.json
├── train/
│   └── libero-train-demo_*.tar
└── test/
    └── libero-test-demo_*.tar
```

配置绝对路径：

```bash
export DATA_DIR=/absolute/path/to/pointworld_wds
export STATS_DIR=$PWD/stats/libero_scene3_20260802
```

脚本会拒绝相对 `DATA_DIR`，从而避免 WebDataset 在错误工作目录下解析路径。

## 4. 计算 normalization statistics

默认固定 seed 512026、2 个视角、scene cap 12000、robot points 1024，并读取 256
个训练 clips：

```bash
DATA_DIR="$DATA_DIR" STATS_DIR="$STATS_DIR" \
  ./scripts/prepare_libero_stats.sh
```

使用全部训练 clips 计算统计量：

```bash
MAX_SAMPLES=0 DATA_DIR="$DATA_DIR" STATS_DIR="$STATS_DIR" \
  ./scripts/prepare_libero_stats.sh
```

输出为 `$STATS_DIR/norm_stats.json`。

## 5. 先跑一个训练 smoke test

```bash
DATA_DIR="$DATA_DIR" STATS_DIR="$STATS_DIR" \
EXP_NAME=libero-scene3-smoke MAX_TRAIN_STEPS=10 \
EVAL_FREQ=5 SAVE_FREQ=5 NUM_EVAL_BATCHES=2 \
  ./scripts/train_libero.sh
```

成功后 checkpoint 位于：

```text
train_logs/libero-scene3-smoke/model-last.pt
```

## 6. 全量训练

单卡默认命令：

```bash
DATA_DIR="$DATA_DIR" STATS_DIR="$STATS_DIR" \
EXP_NAME=libero-scene3-uniform-cap12k \
NUM_EPOCHS=200 MAX_TRAIN_STEPS=-1 \
  ./scripts/train_libero.sh
```

默认值是 batch size 1、2 个相机、12,000 scene points、1,024 robot points、
PTv3-small 和 predictor dim 128。可根据显存覆盖，例如：

```bash
BATCH_SIZE=2 NUM_WORKERS=8 EVAL_NUM_WORKERS=4 \
DATA_DIR="$DATA_DIR" STATS_DIR="$STATS_DIR" \
  ./scripts/train_libero.sh
```

## 7. 测试集评估

评估完整 test split：

```bash
DATA_DIR="$DATA_DIR" STATS_DIR="$STATS_DIR" \
EXP_NAME=libero-scene3-uniform-cap12k EVAL_NUM_BATCHES=0 \
  ./scripts/eval_libero.sh
```

也可以显式指定 checkpoint：

```bash
MODEL_PATH=/absolute/path/to/model-last.pt \
DATA_DIR="$DATA_DIR" STATS_DIR="$STATS_DIR" \
  ./scripts/eval_libero.sh
```

评估结果写入 `eval_logs/<timestamp>/metrics.json`。

## 8. 推理可视化

显示 5 条测试 clip，端口为 8080：

```bash
EVAL_VIZ_NUM=5 EVAL_SKIP_VIZ=false VIEWER_PORT=8080 \
DATA_DIR="$DATA_DIR" STATS_DIR="$STATS_DIR" \
EXP_NAME=libero-scene3-uniform-cap12k \
  ./scripts/eval_libero.sh
```

浏览器访问 `http://服务器地址:8080`。如果服务器没有直接开放端口，可在本地使用
SSH 转发：

```bash
ssh -L 8080:localhost:8080 user@server
```

## 9. 常用可覆盖参数

三个 shell 脚本都采用环境变量配置，不需要修改脚本本身。重要变量包括：

- `POINTWORLD_PYTHON`：Python 解释器，默认 `python`。
- `DATA_DIR`：WDS 绝对路径，必填。
- `STATS_DIR`：normalization statistics 目录。
- `EXP_NAME`、`NUM_EPOCHS`、`MAX_TRAIN_STEPS`。
- `LOG_DIR`：训练日志和 checkpoint 根目录。
- `BATCH_SIZE`、`NUM_WORKERS`、`EVAL_NUM_WORKERS`。
- `MAX_SCENE_POINTS`、`MAX_ROBOT_POINTS`、`NUM_CAMERAS`。
- `EVAL_FREQ`、`SAVE_FREQ`、`NUM_EVAL_BATCHES`。
- `EVAL_NUM_BATCHES`、`EVAL_VIZ_NUM`、`VIEWER_PORT`。

修改点数或相机数后，建议用相同配置重新计算 normalization statistics。
