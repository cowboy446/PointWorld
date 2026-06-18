# AGENTS.md

本文件是未来 agent / Codex 接手 PointWorld 仓库时的入口说明。先读这里，再按任务需要阅读更细的文档和代码。

## 当前仓库语境

- 仓库路径：`/home/zhangrong/zhangrong-workspace/robot-wm/point-wm/PointWorld`
- 当前主要分支：`main`
- 远端：`origin git@github.com:cowboy446/PointWorld.git`
- 本项目的 `main` 分支主要负责 PointWorld 的训练、评估和可视化；数据转换 / 构建流程主要不在这个分支里。
- 已有中文代码分析文档：`POINTWORLD_MAIN_ANALYSIS.md`
- 本次归档补充的 agent 上下文：`POINTWORLD_AGENT_CONTEXT.md`
- ablation 相关设计文档：`ABLATION_PLAN.md`。在 `ablation` 分支还应阅读 `ABLATION_IMPLEMENTATION.md`。

## 接手任务时先看什么

1. 先读 `POINTWORLD_AGENT_CONTEXT.md`，里面有模型结构、关键维度、confidence / filtered eval、近期实验结果和已知坑。
2. 如果任务涉及模型、训练或评估细节，再读 `POINTWORLD_MAIN_ANALYSIS.md`。
3. 如果任务涉及最新 scaling 图和指标，优先看：
   - `eval_logs/eval_2/scaling_comparison.png`
   - `eval_logs/eval_2/scaling_comparison_eval_2_summary.md`
   - `eval_logs/eval_2/scaling_comparison_eval_2_summary.csv`
4. 如果任务涉及 DROID subset 评估脚本，看 `eval_subset.sh`。

## 工程习惯

- 使用 `rg` / `rg --files` 搜索代码。
- 修改文件时优先用 `apply_patch`，不要用 shell 重定向直接写文件。
- 不要重置或回滚用户已有改动。这个仓库可能有未提交的实验输出或用户手动改动。
- 代码改动后尽量运行最小可行验证；如果涉及图表，生成后用图片查看确认。
- 需要联网或 push 到 git 远端时通常需要权限升级。

## PointWorld 关键入口

- 训练入口：`train.py`
- 评估入口：`eval.py`
- 参数定义：`arguments.py`
- 主模型：`pointworld/base.py`
- DINOv3 scene feature：`scene_featurizer.py`
- PTv3：`ptv3/ptv3.py`
- 损失：`pointworld/losses.py`
- metrics：`pointworld/metrics.py`、`evaluation/metrics.py`
- trainer：`training/trainer.py`
- tester：`evaluation/tester.py`
- confidence annotation：`evaluation/annotation.py`
- 数据组件：`dataset_components/`

## 重要事实速记

- 时间设置：`CONTEXT_HORIZON=1`，`PRED_HORIZON=10`，所以 `T=11`。
- DINOv3 ViT-L16 输出层特征维度为 `1024`；使用 `[4, 11, 17, 23]` 四层拼接，所以进入 `feat_proj` 前是 `4096`。
- `feat_proj` 在 `scene_featurizer.py`，通常把 `4096 -> predictor_dim`，默认 `256`。
- PTv3 稀疏输入概念上来自 `(B, Ns + T*Nr, C)` feature 和 `(B, Ns + T*Nr, 3)` coord，但实际送入的是 `exists` mask 后的 `(N_valid, C)` / `(N_valid, 3)`。
- `N_valid = scene_exists[:,0].sum() + robot_exists.reshape(B, T*Nr).sum()`。
- PTv3 最终 `point.feat` 是 `(N_valid, C)`；如果 scatter 回 dense，可理解成 `(B, Ns + T*Nr, C)`。
- 当前模型会提取 robot features 并 max-pool 成 robot latent，用于 FiLM 条件化 scene features；但不预测 robot flow。
- 主要评估指标看 `full_eval/test/filtered_l2_moved/mean`，越低越好。
- `total_loss` 包含 uncertainty / log-variance 项，不适合作为不同 checkpoint 间的唯一 headline 指标。

## Notion 记录

之前已经把 `POINTWORLD_MAIN_ANALYSIS.md` 上传到 Notion 的 `PointWorld` 笔记内部：

- Notion parent page ID：`37c09403-5b24-8042-9a9a-dc857ed08ef7`
- 创建的页面标题：`PointWorld main 分支代码分析`
- URL：`https://app.notion.com/p/382094035b24817b9589ece461b034df`

如果未来要继续沉淀到 Notion，优先把新总结作为该 PointWorld 笔记的子页面或补充页面。
