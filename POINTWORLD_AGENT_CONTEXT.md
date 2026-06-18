# PointWorld Agent Context

日期：2026-06-18

这份文档用于归档当前对话窗口中形成的项目知识，方便未来 agent 继续工作。更完整的逐文件代码分析见 `POINTWORLD_MAIN_ANALYSIS.md`。

## 最近完成的工作

- 阅读并总结了 PointWorld `main` 分支的模型、训练、评估、loss、metrics 和 wandb 记录含义。
- 将分析文档写入 `POINTWORLD_MAIN_ANALYSIS.md`，并上传到 Notion 的 `PointWorld` 笔记下。
- 新增并推送了 `eval_subset.sh`，用于 DROID WDS subset 的 confidence annotation 和 filtered evaluation。
- 根据 `eval_logs/eval_1` 和 `eval_logs/eval_2` 的 `metrics.json` 生成 scaling 对比图与 summary。
- `eval_logs/eval_2` 当前使用同一个 global test set，包含 `831` 个 sequences / clips，`9141` frames。

## 当前重要文件

- `AGENTS.md`：未来 agent 的入口规范。
- `POINTWORLD_MAIN_ANALYSIS.md`：中文长文档，系统说明 PointWorld main 分支。
- `POINTWORLD_AGENT_CONTEXT.md`：本文件，归档近期上下文和实验结论。
- `eval_subset.sh`：DROID subset 评估示例脚本。
- `eval_logs/eval_2/scaling_comparison.png`：最新简洁版 scaling 图。
- `eval_logs/eval_2/scaling_comparison.svg`：同图 SVG 版本。
- `eval_logs/eval_2/scaling_comparison_eval_2_summary.md`：最新 eval_2 指标总结。
- `eval_logs/eval_2/scaling_comparison_eval_2_summary.csv`：最新 eval_2 指标表。

## 模型结构速记

PointWorld 是 action-conditioned 3D world model。输入是第 0 帧 RGB-D / scene points，以及未来 robot point flow 表示的动作轨迹；输出是未来 scene point flow / displacement。

关键时间约定：

- `CONTEXT_HORIZON = 1`
- `PRED_HORIZON = 10`
- `T = 11`
- 第 0 帧是 context，后 10 帧是预测 horizon。

常用维度：

- `B`：batch size
- `T`：时间长度，通常 11
- `Ns`：scene points 数量，padding 后 batch 内统一
- `Nr`：robot points 数量，padding 后 batch 内统一
- `C`：内部通道数，即 `--predictor_dim`，常用 `256`

输入到 PTv3 前的概念 dense 形式：

- 坐标：`(B, Ns + T*Nr, 3)`
- 特征：`(B, Ns + T*Nr, C)`

实际 sparse 形式：

- 坐标：`(N_valid, 3)`
- 特征：`(N_valid, C)`
- `N_valid = scene_exists[:, 0].sum() + robot_exists.reshape(B, T*Nr).sum()`

PTv3 输出：

- 实际输出 `point.feat` 是 `(N_valid, C)`。
- 如果 scatter 回 dense，可理解为 `(B, Ns + T*Nr, C)`。
- 当前 PointWorld 会取 robot 的 feature，并 max-pool 成 robot latent，再通过 FiLM 条件化 scene features；但最终只预测 scene flow，不预测 robot flow。

## DINOv3 / Scene Feature

代码位置：`scene_featurizer.py`

- 使用 DINOv3 ViT-L/16。
- 单层 token feature dim 是 `1024`。
- 当前取层：`[4, 11, 17, 23]`。
- 拼接后 `in_dim = 4096`。
- `feat_proj = nn.Linear(in_dim, channels)`，默认 `4096 -> 256`。
- `feat_proj` 只在 scene feature extractor 里，不在 PTv3 内部。

## point.feat 在哪里来

PTv3 中 `Point(data_dict)` 会从 `data_dict["feat"]` 初始化 `point.feat`。随后依次被 Embedding、SerializedAttention、Block、GridPooling / GridUnpooling 更新。最终 PointWorld 从 `point.feat` 中按 scene / robot mask 拆出：

- scene features：用于预测 scene displacement 和 uncertainty。
- robot features：用于 robot latent max-pooling 和 FiLM 条件化。

## Loss 和 Metrics

主要 loss：

- weighted Huber
- heteroscedastic uncertainty NLL
- release 里 `total_loss == dynamics_loss`

主要 target：

- `gt_scene_flows_relative = gt_scene_flows - gt_scene_flows[:, 0:1]`
- target 会被 clip 到约 `[-0.5, 0.5]`
- loss 只应关注非 context 的预测时间步和 supervised mask。

主要评估指标：

- `full_eval/test/filtered_l2_moved/mean`：最常用 headline metric，越低越好。
- `full_eval/test/l2_moved/mean`：unfiltered moved-point L2。
- `full_eval/test/filtered_l2/mean`：filtered all supervised points。
- `full_eval/test/l2/mean`：unfiltered all supervised points。
- `full_eval/test/total_loss`：包含 uncertainty / log-var 项，不建议作为跨 checkpoint 的唯一结论指标。

## Confidence / Filtered Evaluation

Expert confidence 文件用于 filtered metrics，不是模型输入的一部分。

- 官方发布的 confidence 文件覆盖 DROID 官方 evaluation split，不覆盖任意 flow shard。
- 如果评估自定义 DROID subset，需要确保 subset 的 WDS manifest 来自 confidence 文件覆盖的 clips，或者自己生成 confidence annotation。
- 生成方式：运行 `eval.py --run_confidence_annotation=true`，会写入类似 `${DATA_DIR}/test/expert_confidence-seed=42.h5`。
- `CONFIDENCE_MODEL_PATH` 用于生成 confidence annotation。
- `EVAL_MODEL_PATH` 用于实际评估；二者可以相同，但概念上不同。
- `confidence_thres=0.8` 更接近保留比例 / quantile 逻辑，不是简单的绝对阈值。
- unfiltered eval 不依赖 confidence mask，适合 smoke test；filtered eval 更接近论文报告方式。

模型本身不知道哪些点是“低置信度”。低置信度筛选来自评估阶段的 confidence annotation / uncertainty 统计，用来决定哪些预测参与 filtered metrics。

## 评估非确定性

同一个 checkpoint 重复跑 eval 可能有轻微差异，原因包括：

- CUDA / FlashAttention / spconv / torch_scatter / grid_sample 的非确定性。
- AMP / 多 GPU 浮点归约顺序。
- WebDataset 多 worker 读取顺序和 batching。

若要 debug 稳定性，可临时使用：

- 单 GPU，例如 `export CUDA_VISIBLE_DEVICES=0`
- `BATCH_SIZE=1`
- `NUM_WORKERS=0`
- `EVAL_NUM_WORKERS=0`
- 可尝试 `--deterministic_data=true --deterministic_train=true`

## eval_2 最新结果

目录：`eval_logs/eval_2`

同一个 global test set：

- `full_eval/test/_meta/total_sequences = 831`
- `full_eval/test/_meta/total_frames = 9141`

五个结果：

| setup | model | filtered_l2_moved | filtered_l2 | l2_moved unfiltered | keep fraction | total loss |
|---|---|---:|---:|---:|---:|---:|
| ours 1190 small | small 132M | 0.066500 | 0.019646 | 0.085168 | 0.593 | 8.272 |
| ours 1190 base | base 411M | 0.064218 | 0.015705 | 0.083940 | 0.618 | 14.277 |
| ours 8152 small | small 132M | 0.041487 | 0.010908 | 0.070113 | 0.628 | 46.869 |
| ours 8152 base | base 411M | 0.037804 | 0.010174 | 0.069563 | 0.626 | 42.001 |
| official pretrained small | small 132M | 0.050066 | 0.013324 | 0.063596 | 0.618 | -0.276 |

按 `filtered_l2_moved/mean` 排名：

1. ours 8152 base：`0.037804`
2. ours 8152 small：`0.041487`
3. official pretrained small：`0.050066`
4. ours 1190 base：`0.064218`
5. ours 1190 small：`0.066500`

相对变化：

- ours small，1190 -> 8152 clips：`37.61%` error reduction
- ours base，1190 -> 8152 clips：`41.13%` error reduction
- ours 1190 clips，small -> base：`3.43%` error reduction
- ours 8152 clips，small -> base：`8.88%` error reduction
- official small 相比 ours 1190 small：`24.71%` lower error
- ours 8152 small 相比 official small：`17.14%` lower error

解释：

- 对我们自己的 checkpoints，scaling 结论成立：更多数据显著改善，base 在两个数据规模下都优于 small。
- 官方 small 位于 ours 1190-small 和 ours 8152-small 之间。
- 官方 small 的 unfiltered moved L2 是 `0.063596`，低于 ours 8152 small/base 的 `0.070113` / `0.069563`；但 filtered moved L2 上弱于两个 8152 自训模型。这说明官方 small 在全体 moved 点上更稳，而自训 8152 模型在 high-confidence filtered 区域更好。

## eval 图生成规范

用户偏好简洁图，不要过度花哨。`eval_logs/eval_1/scaling_comparison.png` 是参考风格：

- 2x2 panel
- 左上：主指标 heatmap
- 右上：data scaling line plot
- 左下：model scaling bar chart
- 右下：filtered vs unfiltered moved L2 bar chart

当前 `eval_logs/eval_2/scaling_comparison.png` 已按这个风格重画。

## 常用命令

查看 eval_2 最新 summary：

```bash
sed -n '1,220p' eval_logs/eval_2/scaling_comparison_eval_2_summary.md
```

列出 eval_2 metrics：

```bash
find eval_logs/eval_2 -maxdepth 2 -name metrics.json -print | sort
```

运行 subset eval 示例：

```bash
bash eval_subset.sh
```

## 未来建议

- 如果继续比较 scaling，尽量让所有 checkpoint 使用同一个 `DATA_DIR`、同一个 test split、同一个 confidence H5。
- 如果要比较官方模型与自训模型，filtered 和 unfiltered 都要报告，因为二者可能给出不同侧面的结论。
- 若要把图用于论文 / 汇报，可以保留当前简洁版，同时另存一版只含主指标 heatmap + line plot 的更干净图。
- 若要同步文档到 Notion，优先更新已有 `PointWorld main 分支代码分析` 页面，或在 `PointWorld` 父页面下新增“评估与 scaling 实验记录”。
