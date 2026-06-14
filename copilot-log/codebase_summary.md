# PointWorld 项目代码梳理（main 分支）

本文档总结了 `PointWorld` main 分支的核心实现：模型结构、数据接口、训练/评估流程、输入/输出和可视化要点，并指向关键源码位置，便于后续调试与改进。

**主要入口**
- 训练：[train.py](train.py)
- 评估：[eval.py](eval.py)
- 全局参数解析：[arguments.py](arguments.py)

**关键模块/文件**
- 模型骨干与封装：[pointworld/base.py](pointworld/base.py)
- PTV3 Transformer：[ptv3/ptv3.py](ptv3/ptv3.py) 与 [ptv3/module.py](ptv3/module.py)
- 场景特征编码（DINOv3）：[scene_featurizer.py](scene_featurizer.py)
- 数据加载与预处理：dataset_components/* （主要是 [dataset_components/dataloader.py](dataset_components/dataloader.py)、[dataset_components/decoders.py](dataset_components/decoders.py)、[dataset_components/collate.py](dataset_components/collate.py)、[dataset_components/pipeline.py](dataset_components/pipeline.py)）
- 损失与度量：[pointworld/losses.py](pointworld/losses.py)、[pointworld/metrics.py](pointworld/metrics.py)
- 评估可视化：[visualization/prediction_viz/visualizer.py](visualization/prediction_viz/visualizer.py) 与相关工具
- 训练管理：[training/trainer.py](training/trainer.py)
- 评估驱动：[evaluation/tester.py](evaluation/tester.py)


## 一、模型结构（总体概览）
- 顶层模型为 `BaseModel`（定义在 [pointworld/base.py](pointworld/base.py)）：负责整合场景编码、机器人特征投影、时间嵌入、PTv3 预测器与不确定性头。
  - 场景编码器：`SceneFeatureEncoder`（[scene_featurizer.py](scene_featurizer.py)）
    - 使用 DINOv3 ViT（子模块在 `third_party/dinov3`）对每个相机视角提取 patch-level 特征，随后将多摄像头特征按点投影并平均，最后线性投影到 `channels`（与预测器输入维度对齐）。
  - 机器人特征：通过 `MLP` 投影到预测器维度，并加上时间嵌入与类型 embedding。
  - 预测器（DynamicsPredictor）：
    - Predictor backbone 使用 `PointTransformerV3`（在 `ptv3/ptv3.py` 中由 `build_ptv3()` 构建，参数通过 `ptv3_arch.yaml` 配置）。
    - 输出两个头：一个 dynamics head（预测相对场景位移，维度为 3），一个 `log_var` 头用于异方差不确定性建模。
    - 通过拼接场景点与时间序列机器人点作为 transformer 输入，得到每个场景点的表征，再由头部预测未来 T-1 步的场景流与 log variance。
  - 在 `forward()` 中，模型会：
    - 对输入进行归一化（调用 `norm_stats` 中的统计量）；
    - 调用 encoder 与 predictor 得到 `pred`（相对位移）与 `log_var`；
    - 反归一化得到最终 `scene_flows`（绝对流量），并计算 `confidence`（由 `log_var` 转换而来）。


## 二、输入 / 输出 数据接口（Data contract）
- 数据通过 WebDataset（WDS）读取并经 `dataset_components` 解码/转换后进入模型，最后由 `dataset_components/collate.py` 聚合为 batch。

- 重要的 batch 字段（训练/评估的最小必需键）：
  - `scene_flows` : (B, T, Ns, 3) — 场景点的世界坐标或相对流（time-first tensor）
  - `gt_scene_flows` / `gt_scene_flows_relative` : 真实目标（用于 loss/metric）
  - `scene_features` : (B, T, Ns, Ds) — 点的原始特征
  - `scene_exists` : (B, T, Ns) — 点是否存在的 mask
  - `robot_flows` : (B, T, Nr, 3) — 机器人点轨迹
  - `robot_features` : (B, T, Nr, Fr) — 机器人点特征
  - `robot_exists` : (B, T, Nr) — 机器人点存在 mask
  - `scene_context_mask`, `scene_supervised_mask`, `scene_moved_mask`, `scene_static_mask`, `scene_selector_gt` — 用于选择 supervision/评价的布尔/选择器张量
  - `__domain__`, `__key__` 等元信息字段
  - `point_weights`, `weights_norm` — collate 生成的权重，用于加权 loss

- 模型前向返回（outputs）常见字段：
  - `scene_relative_norm` : 归一化后的相对预测（B,T,Ns,3）
  - `scene_relative` : 反归一化的相对预测
  - `scene_flows` : 最终场景流（B,T,Ns,3）等于 scene_coord0 + pred
  - `pred` : dynamics head 的原始输出
  - `log_var` : 不确定性 head（用于计算置信度与 NLL reweight）
  - `confidence` : 0..1 的置信度标量（基于 `log_var` 计算）

- Loss 由 `pointworld/losses.py` 计算：
  - 每点采用 Huber 损失并由异方差 log-var 加权（NLL 风格：error/var + log_var），再用 `point_weights` 聚合得到总 loss。
  - 评估阶段额外计算过滤后的指标（filtered_l2）需要 expert confidence 文件（由 `data` 分支处理后放入 WDS）。


## 三、训练流程（Trainer）
- 入口：`train.py` 实例化 `Trainer(args)`（[training/trainer.py](training/trainer.py)）。
- 关键步骤：
  1. 解析 args，并根据 checkpoint contract 调整 args（`pointworld/checkpoint_contract.py`）。
  2. 创建 dataloaders：调用 `dataset_components/dataloader.build_dataloader()`。
  3. 构建 `BaseModel` 并移动到设备；若多卡则封装为 DDP。
  4. 创建 AdamW 优化器；使用 AMP（自动混合精度）和 `GradScaler`。
  5. 训练循环：
     - 获取 batch，前向、计算 loss、scale、反向、clip grad、step optimizer；
     - 周期性执行 eval（`eval_step`）与保存 checkpoint；
     - NaN 检测与跳批逻辑（若发现 NaN 会记录并尝试保存调试检查点）。
- Checkpoint：通过 `training/checkpointing.py` 管理（保存 `model`、`optimizer`、`meta`，并记录 `model_contract` / 数据 contract 以方便 eval 时自动配置）。


## 四、评估流程（Tester / eval.py）
- `evaluation/tester.py` 继承 `Trainer`，但以 `inference_only=True` 初始化。
- 加载 checkpoint（或指定 `--eval_exp_name`），并从 checkpoint contract 中恢复训练时使用的 domains 与数据契约。
- 根据需要执行：
  - 信心注释生成（`--run_confidence_annotation`）
  - 带置信度过滤的评估（需要 confidence mask 文件）
  - 可视化若 `--eval_viz_num>0`：会实例化 `PredictionVisualizer` 并逐样本渲染（打开本地 HTTP viewer）
- 指标聚合：按 domain/batch 聚合后输出 JSON（存入 eval_logs 目录）。


## 五、可视化（如何运行与实现）
- 可视化实现集中在 `visualization/prediction_viz`：
  - `PredictionVisualizer`（主要实现文件：[visualization/prediction_viz/visualizer.py](visualization/prediction_viz/visualizer.py)）负责把样本和预测转为 Viser-compatible 表示并启动/更新 3D viewer。
  - `evaluation/tester.py` 中的 `_visualize_eval_samples()` 会：
    - 构建单个样本字典（调用 `build_sample_from_dictionary`），传入 ground-truth 与预测；
    - 调用 visualizer.visualize 并在交互式 TTY 下让用户逐样本查看（或启动 live session，在浏览器访问 `http://localhost:<port>`）。
- 运行示例：
  ```bash
  python eval.py \
    --model_path pretrained_checkpoints/small-droid/model-best.pt \
    --domains=droid \
    --data_dirs=/path/to/droid/wds \
    --batch_size=1 \
    --eval_num_batches=100 \
    --eval_viz_num=8 \
    --viewer_port=8080
  ```


## 六、如何做最小烟雾测试（建议步骤）
1. 准备小 WDS 子集（`data` 分支的恢复脚本可只恢复少量 shards），或手工构造一个最小样本 WDS，确保包含上文列出的必需键。
2. 使用 release `small-droid` checkpoint（放在 `pretrained_checkpoints/`）运行 `eval.py --eval_num_batches=1`，确认能成功加载模型并前向通过。
3. 若只做模型加载测试，可在 `Trainer` 的 `inference_only=True` 模式下直接从 checkpoint 推理少量样本以验证 I/O 与输出字段。


## 七、关键实现位置快速索引（便于修改/扩展）
- 参数解析：[arguments.py](arguments.py)
- 数据解码与 sample 构建：[dataset_components/decoders.py](dataset_components/decoders.py)、[dataset_components/pipeline.py](dataset_components/pipeline.py)
- Collate 与批处理：[dataset_components/collate.py](dataset_components/collate.py)
- 模型主线：[pointworld/base.py](pointworld/base.py)
- PTV3 变换器：[ptv3/ptv3.py](ptv3/ptv3.py)（大部分实现）
- 场景视觉 encoder（DINOv3）：[scene_featurizer.py](scene_featurizer.py)
- 训练主流程：[training/trainer.py](training/trainer.py)
- 评估/可视化：[evaluation/tester.py](evaluation/tester.py)、[visualization/prediction_viz/visualizer.py](visualization/prediction_viz/visualizer.py)


## 八、下一步建议（可选）
- 我可以：
  - 生成一个最小 WDS 子集示例脚本（合成少量字段）以便在不下载官方数据的情况下做 smoke 测试；
  - 在 `pointwm-min` 环境中尝试加载一个小 checkpoint（如 `small-droid`）并执行 1 个 batch 的 `eval.py`（前提是你已经下载 checkpoint）；
  - 或继续按你之前要求把 `data` 分支的子集恢复并转换为 WDS。

---
文档来自对仓库关键文件的静态扫描与源码阅读。如需我把本摘要展开为更详细的 API 参考（函数签名、字段尺寸示例、JSON 或 WDS 元数据示例），我可以继续补充。