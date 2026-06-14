# PointWorld 部署与最小测试指南（从零开始）

## 概述
本文件为你从零开始把 `PointWorld` 项目部署并运行最小烟雾测试的中文步骤清单。整体分为四大块：

1. 环境准备（OS / Python / CUDA / Conda）
2. 第三方模型依赖（DINOv3 submodule）
3. 数据准备（切换到 `data` 分支，恢复 Hugging Face 数据包并转换为 WDS）
4. 主分支部署与最小测试（切回 `main`，创建 conda 环境，安装依赖，下载 checkpoints，运行 smoke eval）

## 前置条件
- 有网络访问以下载数据与 checkpoints（Hugging Face）。
- 有合适的 GPU 环境或至少能在 CPU 上做快速 smoke 测试（推荐 Linux x86_64，Python 3.10）。
- 本仓库已 clone 到本地（当前仓库根目录即为本工程目录）。

仓库当前状态检查（在仓库根目录运行）：

```bash
pwd
git status --porcelain=2 --branch
git remote -v
git branch -a
```

## 一、环境准备（conda）
目标：创建 `pointwm` 环境并安装主依赖。

```bash
# 在仓库根目录
# 创建并激活 conda 环境（来自 main 分支的环境描述）
conda env create -n pointwm -f environments/train_eval.yml
conda activate pointwm

# 基础工具
python -m pip install huggingface_hub==0.26.2
python -m pip install timm==1.0.19 --no-deps
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -m pip install networkx==3.4.2 --no-deps
```

可选：如果需要可视化功能，运行：

```bash
conda env update -n pointwm -f environments/train_eval_viz.yml --prune
python -m pip install huggingface_hub==0.26.2
python -m pip install timm==1.0.19 --no-deps
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -m pip install networkx==3.4.2 --no-deps
```

注意：`flash-attn`、`torch` 等可能需要与系统 CUDA / cuDNN 兼容的二进制包，按你的平台选择合适的 `pip`/`conda` wheel（如果遇到编译问题，可先跑 CPU-level smoke 测试）。

## 二、第三方模型（DINOv3）
PointWorld 需要 DINOv3 checkpoints（受限下载）。流程：

```bash
# 初始化子模块
git submodule update --init --recursive
mkdir -p third_party/dinov3/checkpoints
# 然后把 DINOv3 提供的权重放到上面目录（需要你已取得 DINOv3 下载链接/许可）
wget -O third_party/dinov3/checkpoints/<dinov3_vitl16_pretrain_*.pth> "<URL_FROM_DINOV3_ACCESS_EMAIL>"
```

如果暂时无法拿到 DINOv3，可以先跳过，使用小型/合成数据做 smoke test（但某些评估或模型加载会失败）。

## 三、数据准备（使用 `data` 分支）
重要：使用 `data` 分支的 pipeline 恢复与转换 H5 包为 WDS，然后 `main` 分支的代码消费 WDS。流程：

1. 切换并更新 `data` 分支：

```bash
git fetch --all
git checkout data
git pull --ff-only
```

2. 下载你需要的数据包（Hugging Face 上的 DROID / BEHAVIOR 包）。把下载的包放在任意目录，然后用仓库中提供的 `recover_dataset_from_parts.sh` 恢复（参见 data 分支 README）。示例（伪命令）:

```bash
# 假设你把 hf 包下载到 ~/downloads/PointWorld-DROID
# 进入数据分支提供的脚本路径并运行恢复脚本（data 分支说明里有具体示例）
# 示例：
./recover_dataset_from_parts.sh ~/downloads/PointWorld-DROID /path/to/pointworld_droid_restored
```

3. 运行数据完整性检查与 H5 -> WDS 转换（data 分支的工具）：

```bash
python data_integrity_check.py --input /path/to/pointworld_droid_restored
python convert_wds.py --input /path/to/pointworld_droid_restored --output /path/to/droid/wds
```

4. 确认最终的 WDS 目录结构：

```
/path/to/droid/wds
/path/to/behavior/wds
```

提示：为了做小规模 smoke test，你可以只恢复并转换少量 shards（README 提到可用子集用于调试）。

完成数据准备后，切回 `main`：

```bash
git checkout main
git pull --ff-only
```

## 四、主分支部署与最小烟雾测试
目标：在 `main` 下创建环境、下载 checkpoints 并运行一次快速 `eval.py`（少量 batch）来验证 pipeline 可运行。

1. 创建本地跟踪 `data` 分支（可选，如果你在 step 三用了远端 data）：

```bash
# 如果还没创建本地 data 分支
git checkout -b data origin/data
# 用完后回到 main
git checkout main
```

2. 下载预训练 checkpoints（将需要的 checkpoint 放到 `pretrained_checkpoints/`）：

```bash
# 使用 huggingface cli（需登录或使用 token）
huggingface-cli download nvidia/PointWorld_models --local-dir pretrained_checkpoints --include "small-droid/model-best.pt"
# 或者通过 huggingface_hub API 下载特定文件
```

3. 运行快速 smoke eval（示例）：

```bash
MODEL_PATH=pretrained_checkpoints/small-droid/model-best.pt
python eval.py \
  --model_path "${MODEL_PATH}" \
  --domains=droid \
  --data_dirs=/path/to/droid/wds \
  --batch_size=1 \
  --eval_num_batches=10
```

- 如果没有 WDS，可以先用一个非常小的手工 WDS 子集或 synthetic 数据替代来验证脚本能跑通（日志/错误信息会提示缺少哪些字段）。

4. 可视化（可选）

```bash
python eval.py --model_path "${MODEL_PATH}" --domains=droid --data_dirs=/path/to/droid/wds --batch_size=1 --eval_num_batches=100 --eval_viz_num=8 --viewer_port=8080
# 然后浏览器访问 http://localhost:8080
```

## 调试与常见问题
- 若遇到 GPU/torch 兼容问题，先在 CPU 上运行 `--eval_num_batches=1` 做快速检查。
- 若缺少某些 pip wheel（如 `flash-attn`）导致安装失败，可先跳过安装，做代码级 smoke test，再逐步解决编译/兼容问题。
- 如果 `convert_wds.py` 报错，检查恢复的 H5 是否完整，并运行 `data_integrity_check.py` 查找问题。

## 建议的 agent（任务划分）
我建议创建三个“agent”角色（可以把执行脚本或任务说明放在 `copilot-log/agents/`）：

- `env-agent`：负责环境创建（conda 环境、pip 包、CUDA 检查）。
- `data-agent`：负责数据包下载、恢复、完整性检查、H5->WDS 转换。
- `test-agent`：负责下载 checkpoints、运行 smoke eval、收集日志和可视化。

如果你同意，我会在 `copilot-log/agents/` 下创建这三个简短的任务说明文件，并开始执行 `env-agent` 的第一步（创建并激活 conda 环境）。

## 下一步（建议）
- 我现在可以立即：
  1. 在仓库里创建 `copilot-log/agents/` 三个 agent 描述文件；
  2. 创建并激活 `pointwm`（会在本机 conda 环境中安装依赖）；
  3. 或者从你这里得到优先级（你想先准备数据还是先建环境）。

请选择要我执行的下一步（例如“创建 agent 文件并建环境”或“先准备数据”）。
