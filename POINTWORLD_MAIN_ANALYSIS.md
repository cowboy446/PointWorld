# PointWorld main 分支代码分析

日期：2026-06-17

这份文档总结的是当前本地 PointWorld `main` 工作树。检查时本地 HEAD 是
`2cc8f32 change conda env name`，比 `origin/main` 领先 1 个提交。

## 使用的资料

Notion 背景笔记：

- `PointWorld`：项目总览和高层架构笔记。
- `PointWorld数据集处理`：DROID shard -> H5 -> WDS 的处理记录，以及 smoke-test 训练命令。
- `PointWorld论文精读：Scaling 3D World Models for In-The-Wild Robotic Manipulation`：论文总结和复现注意点。
- `PointWorld模型测试`：简短模型测试记录。

阅读过的代码路径：

- 入口：`train.py`、`eval.py`
- CLI / 配置：`arguments.py`、`ptv3/ptv3_arch.yaml`
- 训练：`training/trainer.py`、`training/checkpointing.py`
- 模型：`pointworld/base.py`、`scene_featurizer.py`、`pointworld/embeddings.py`
- 损失和指标：`pointworld/losses.py`、`pointworld/metrics.py`
- 数据：`dataset_components/dataloader.py`、`dataset_components/decoders.py`、
  `dataset_components/pipeline.py`、`dataset_components/transforms.py`、
  `dataset_components/collate.py`、`dataset_components/robot.py`
- 评估：`evaluation/tester.py`、`evaluation/annotation.py`、`evaluation/metrics.py`、
  `evaluation/meta.py`

## 项目整体功能

PointWorld 是一个用于机器人操作的 action-conditioned 3D world model。它接收部分可观测的 RGB-D 场景，以及以 robot point flow 表示的未来机器人动作，然后预测完整场景未来的 3D 点位置 / 点流。

`main` 分支不是数据构建分支。它假设数据已经在 `data` 分支或外部流程里转换成本地 WebDataset shards，然后负责：

- 加载 WDS 数据并做运行时 augmentation。
- 用冻结的 DINOv3 加 raw point features 提取 scene features。
- 构造 robot gripper point flow 并打包 robot features。
- 用 PointTransformerV3 做 dynamics prediction。
- 计算 Huber + heteroscedastic uncertainty loss。
- 记录训练过程指标和完整评估指标。
- 可选地做 DROID confidence-mask filtered evaluation。
- 可选地做预测结果可视化。

## 核心时间和维度约定

release 常量里设置：

- `CONTEXT_HORIZON = 1`
- `PRED_HORIZON = 10`
- 模型总时间步 `T = 11`

第一个时间步 `t=0` 是观测到的 context。模型会给所有 `T` 个槽位输出相对 scene displacement，但第 0 帧输出被强制置零，loss 和 metrics 只作用在非 context 的预测时间步上。

常用符号：

- `B`：batch size。
- `T`：总序列长度，通常是 11。
- `Ns`：batch 内 padding 后的 scene point 数量。collate 之前会受 `--max_scene_points` 限制，默认上限 12000。
- `Nr`：batch 内 padding 后的 robot point 数量。默认 presample 上限是 `--max_robot_points=500`。
- `Ds`：`gather_features` 之后的 raw scene feature 维度。
- `Fr`：`gather_features` 之后的 raw robot feature 维度。
- `C`：模型内部 channel 维度，即 `--predictor_dim`，默认 256。

主要 batch 张量：

| Key | Shape | 含义 |
| --- | --- | --- |
| `scene_flows` | `(B,T,Ns,3)` | context masking 之后的 scene point 位置。对 `t>0`，scene 输入会被覆盖成前一帧 context 值，所以模型看到的 scene 是静态的。 |
| `gt_scene_flows` | `(B,T,Ns,3)` | context masking 前的真实 scene point 位置。 |
| `gt_scene_flows_relative` | `(B,T,Ns,3)` | `gt_scene_flows - gt_scene_flows[:,0:1]`，并被 clip 到 +/-0.5，是归一化前的 prediction target。 |
| `robot_flows` | `(B,T,Nr,3)` | 未来动作轨迹上的 robot surface / gripper point 位置。 |
| `scene_features` | `(B,1,Ns,Ds)` | collate / padding 前只保留第 0 帧 raw scene features，模型实际消费 `[:,0]`。 |
| `robot_features` | `(B,T,Nr,Fr)` | 拼接后的 robot point features。 |
| `scene_exists` | `(B,T,Ns)` | True 表示真实 scene point，False 表示 padding。 |
| `robot_exists` | `(B,T,Nr)` | True 表示真实 robot point，False 表示 padding。 |
| `scene_context_mask` | `(B,T,Ns,1)` | context 时间步为 True，prediction horizon 为 False。 |
| `scene_supervised_mask` | `(B,T,Ns,1)` | DROID 中表示 visibility 和 depth-valid 的监督 mask；仿真中默认全 True。 |
| `scene_moved_mask` | `(B,T,Ns,1)` | movement selector label 大于 0.5 的点。 |
| `scene_static_mask` | `(B,T,Ns,1)` | moved mask 的补集。 |
| `scene_selector_gt` | `(B,T,Ns)` | 由每步 GT movement 得到的 soft movement weight label。 |
| `point_weights` | `(B,T,Ns)` | 归一化后的 loss 权重；prediction / supervised / existing 范围外为 0。 |

模型输出：

| Key | Shape | 含义 |
| --- | --- | --- |
| `pred` | `(B,T,Ns,3)` | 归一化空间里的相对 displacement prediction。 |
| `log_var` | `(B,T,Ns,1)` | 用于 uncertainty / NLL weighting 的预测 log variance。 |
| `scene_relative_norm` | `(B,T,Ns,3)` | 和 `pred` 相同的归一化 prediction。 |
| `scene_relative` | `(B,T,Ns,3)` | unnormalize 后的相对 displacement。 |
| `scene_flows` | `(B,T,Ns,3)` | 预测的 scene 绝对位置：`scene_coord0 + scene_relative`。 |
| `confidence` | `(B,T,Ns)` | 从 clamp 后的 variance 推导出来，用于 confidence-aware metrics / annotation。 |

## 数据加载流程

`train.py` 本身很薄：解析 args，实例化 `Trainer`，调用 `trainer.train()`，结束时关闭 wandb，并在需要时 teardown DDP。

`Trainer.setup_dataloader` 通过 `build_dataloader` 构造 train / test loaders。

重要数据路径：

1. `gather_shard_paths` 在 `<data_dir>/<split>` 下找 `.tar` WDS shards。相对路径的 `data_dir` 会被解析到 `dataset_components/` 下，所以 Notion 里的训练命令提醒使用绝对 `--data_dirs` 是正确的。
2. `build_dataset` 构造 `webdataset.WebDataset`。
3. `decode_data` 解码 `.npy`、`.jpg`、`.pyd`、`.pkl`；DROID wrist pose 会转换到 TCP，BEHAVIOR gripper pose 会转换到类似 OpenCV 的坐标系。
4. `build_flow_sample` 统一构造 scene flow 和 robot flow sample。
5. `sample_cameras` 采样 train / eval 使用的 camera。
6. `canonicalize_gripper_keys_and_flags` 规范化 right / left gripper keys。
7. `sample_transform_pipeline` 应用几何和颜色变换，并产出最终字段。
8. `custom_collate_fn` padding 变长点数，创建 existence masks 和 `point_weights`。

训练时数据变换：

- Center shift。
- bounds filtering 到 `[-3,3]^3`。
- camera 分辨率断言 `(H,W)=(180,320)`。
- 按 `--grid_size` 做 voxel grid sampling，默认 0.015。
- 围绕 robot-relevant region 做 sphere crop。
- scene point 数量上限裁剪。
- 随机绕 z 轴旋转。
- 随机 scale，范围 `[0.90,1.10]`。
- 以 0.5 概率做 x / y 反射。
- color auto-contrast、translation、jitter。
- 颜色归一化到 `[0,1]`。
- 在 context masking 前复制 GT scene flow。
- 应用固定 context mask，context horizon 为 1。

评估时数据变换是确定性的：grid sampling、确定性点数上限裁剪、center shift、颜色归一化、GT copy、固定 context masking。

## Scene 和 Robot Features

release 模式中默认 CLI feature lists 是固定的：

- `robot_features`: `robot_flows`, `robot_colors`, `robot_normals`,
  `gripper_open`, `robot_velocity`, `robot_acceleration`
- `scene_features`: `scene_flows`, `scene_colors`, `scene_normals`,
  `gripper_open`, `dist2robot`

Robot feature 构造：

- `robot_flows`：默认只采样 gripper-only robot surface points，即 `RELEASE_GRIPPER_ONLY=True`。
- DROID 使用 7 个 Panda arm joints，加 Robotiq finger joint mapping。
- BEHAVIOR 使用保存的 `joint_names`、robot sampler 的 joint ordering，以及移动底盘 `base_pose`。
- feature gathering 中 robot colors 固定为 magenta。
- velocity 和 acceleration 是 robot point positions 在时间维度上的有限差分。
- `gripper_open` 会重复到所有 robot points 上。

Scene feature 构造：

- 为了节省内存，raw scene features 只保留 timestep 0。
- `dist2robot` 对每个初始 scene point，存储它到每个 timestep 最近 robot point 的距离，所以这部分 feature 维度是 `T`。
- `gripper_open` 会广播到每个 scene point。

对于单臂 DROID，通常期望维度是：

- Scene raw feature dim `Ds = 3 scene xyz + 3 color + 3 normal + 1 gripper_open + 11 dist2robot = 21`。
- Robot raw feature dim `Fr = 3 xyz + 3 color + 3 normal + 1 gripper_open + 3 velocity + 3 acceleration = 16`。

对于双臂 BEHAVIOR mixed runs，gripper-open slots 可能扩展为左右两侧，因此 feature 维度会由 checkpoint / data contract 决定。

## 模型架构

`BaseModel` 由三部分组成：

1. `SceneFeatureEncoder`
2. Robot feature projection + temporal embedding
3. `DynamicsPredictor`

### SceneFeatureEncoder

`SceneFeatureEncoder` 从两类信息里构造 scene point embeddings：

- 冻结的 DINOv3 ViT-L/16 2D features。
- WDS pipeline 中已有的 raw scene features。

DINO 路径：

- 使用 `dinov3_vitl16`。
- 需要本地 DINOv3 submodule 和 `third_party/dinov3/checkpoints` 下的权重。
- 抽取 intermediate layers `[4,11,17,23]`。
- 拼成每个 patch token 4096 维的特征。
- 将 scene 3D points 投影到所有采样 camera views。
- 用 RGB-D depth consistency 判断可见性。
- 用双线性 `grid_sample` 从 ViT patch features 中采样点特征。
- 对所有有效 camera features 求平均。
- 投影到 `C` channels。

raw scene 路径：

- 按 domain stats 归一化 `scene_features`。
- 将 `Ds -> C`。
- DINO 分支和 raw 分支分别 LayerNorm。
- 拼接后从 `2C -> C`。

输出：`scene_feat0`，shape 为 `(B,Ns,C)`。

### Robot Embedding

`robot_features` 先按 domain stats 归一化，然后通过 MLP 投影：

- Raw: `(B,T,Nr,Fr)`
- MLP output: `(B,T,Nr,C)`

模型还会加上：

- 基于 `torch.linspace(0,1,T)` 的 sin / cos temporal embedding。
- 一个 learned `robot_type_emb`。

输出：`robot_feat`，shape 为 `(B,T,Nr,C)`。

### DynamicsPredictor

`DynamicsPredictor.forward` 会把初始 scene points 和所有未来 robot points 拼成一个 point set：

- Coordinates: `(B, Ns + T*Nr, 3)`
- Features: `(B, Ns + T*Nr, C)`
- Existence mask: `(B, Ns + T*Nr)`
- `is_robot`: flatten 后的布尔标记，用来区分 scene points 和 robot points。

有效点会送入 PointTransformerV3。PTv3 size 从 `ptv3/ptv3_arch.yaml` 读取：

- `small`：更浅，`channels_max=128`。
- `base`：默认配置，`channels_max=256`。
- `large`：要求 `channels_eq=256`。

PTv3 使用 `z`、`z-trans`、`hilbert`、`hilbert-trans` 四种序列化 order，默认 drop path 为 0.3，patch size 来自 `--ptv3_patch_size`，默认 256。

PTv3 之后：

- scene output features scatter 回 `(B,Ns,C)`。
- 输入 scene features 走一条 FiLM-modulated skip connection。
- 有效 robot output features 会在每个 batch item 内跨所有 robot points / timesteps 做 max pooling，扩展到 `(B,Ns,C)`，再经过 FiLM modulation。
- 最终 scene representation 是：
  `scene_ptv3_output + scene_skip_film + robot_global_film`。

Prediction heads：

- `dynamics_head`: `C -> 128 -> 128 -> 3*(T-1)`。
- `log_var_head`: `C -> 128 -> 128 -> 1*(T-1)`。

两个 head 都只输出未来时间步；timestep 0 会 prepend zeros。因此：

- `pred_norm`: `(B,T,Ns,3)`，在 `t=0` 处为 0。
- `log_var`: `(B,T,Ns,1)`，在后续 clamp / prepare 前，`t=0` 处为 0。

最后 `BaseModel` 会把 `pred_norm` unnormalize 成相对 displacement，再加到 `scene_coord0` 上，得到预测的绝对 scene positions。

## 归一化

`pointworld/norm_stats.py` 只加载一个 canonical 文件：
`<norm_stats_path>/norm_stats.json`。

它要求文件里有：

- `statistics`：各 domain 的 raw `robot_features` 和 `scene_features` 统计量。
- `per_timestep_statistics`：各 domain、各 timestep 的 `gt_scene_flows_relative` 统计量。

模型注册的 buffers：

- `norm_stats_per_step_mean`: `(D,T,3)`
- `norm_stats_per_step_var`: `(D,T,3)`
- `robot_norm_mean`: `(D,Fr)`
- `robot_norm_var`: `(D,Fr)`
- `scene_norm_mean`: `(D,Ds)`
- `scene_norm_var`: `(D,Ds)`

运行时会把 `data_dict["__domain__"]` 映射为 domain indices，所以 mixed-domain batch 中每个样本可以用自己的归一化统计。

## 损失函数

release 代码里唯一的训练 loss 是带 uncertainty 的 dynamics loss。

Target：

- `gt_target = data_dict["gt_scene_flows_relative"]`
- `gt_target_norm = model.normalize(gt_target)`

Prediction：

- `output_norm = outputs["scene_relative_norm"]`

有效监督预测 mask：

```text
pred_exists_supervised =
    (~scene_context_mask) & scene_exists & scene_supervised_mask
```

Point weights：

- 在 `custom_collate_fn` 中构造。
- 起点是 `scene_selector_gt ** RELEASE_WEIGHT_GAMMA`。
- 在 `pred_exists_supervised` 外置零。
- 用总 weight sum 做归一化，分母 clamp 到至少 1。
- 默认 gamma 是 1.0。

逐元素 loss：

1. 在归一化 prediction 和归一化 target 之间计算 Huber loss：
   `Huber(output_norm, gt_target_norm, delta=args.huber_delta)`，shape 是
   `(B,T,Ns,3)`。
2. 将预测的 log variance clamp 到 `[log(1e-6), log(1e2)]`。
3. 如果 domain 包含 `"behavior"`，会替换仿真样本的 log variance：
   - 如果同一个 batch 里也有真实数据，用真实样本的 mean variance；
   - 如果是 simulation-only batch，用常量 `SIM_VAR_CONST = 1e-3`。
4. 使用类似 heteroscedastic NLL 的 per-dimension loss：

```text
0.5 * (huber_error / exp(log_var_clamped)
       + UNCERTAINTY_LOGVAR_WEIGHT * log_var_clamped)
```

5. 对 xyz 维度求平均，得到 `(B,T,Ns)`。
6. 乘上 `point_weights` 后求和。

因此当前 release 代码中，记录出来的 `total_loss` 和 `dynamics_loss` 是同一个标量。

## Metrics 和 Wandb Logging

wandb 只在 rank 0 初始化：

- Project: `point-world`
- Name: `--exp_name` 或 wandb 自动生成的名字。
- Config: 原始 parsed CLI args。
- `wandb.watch(self.model, log="all")` 会记录 gradients / parameters。

初始化阶段 logs：

- `model/total_params`：模型参数量。
- `model/robot_features_dim`：推断出的 `Fr`。
- `model/scene_features_dim`：推断出的 `Ds`。
- `model/data_keys`：setup 时观察到的 batch keys。

训练循环 logs：

- `train/total_loss`：当前优化步之后的 weighted dynamics loss。
- `train/dynamics_loss`：release 中和 total loss 相同。
- `train/l2/mean`：所有 supervised prediction points 上的绝对位置 L2 均值。
- `train/l2_moved/mean`：moved supervised points 上的 L2。
- `train/l2_static/mean`：static supervised points 上的 L2。
- `train/pred/mean`、`train/pred/max`、`train/pred/min`：归一化 prediction tensor 的值域统计，不是物理空间 displacement magnitude。
- `train/pred_moved/*`、`train/pred_static/*`：按 moved / static mask 分组后的 normalized prediction 统计。
- `train/uncertainty/mean`：log-var preparation 后的 mean predicted variance。
- `train/confidence/*`：由 clamp 后 variance 推导出的 confidence 统计。
- `train/l2_conf/mean`、`train/l2_moved_conf/mean`、
  `train/l2_static_conf/mean`：只保留高 confidence fraction 后的 L2。保留比例是 `--confidence_thres`，默认 0.8。
- `train/weights/mean`、`train/weights/sum`：loss weights 分布。归一化后 sum 通常接近 1。
- `train/moved_percentage`：supervised points 中 moved points 的比例。
- `train/static_percentage`：supervised points 中 static points 的比例。
- `train/supervised_percentage`：supervised prediction elements 占总 `B*T*Ns` 的比例。
- `train/<domain>/...`：对 `args.domains` 中每个 domain 分别计算的同一套 metrics。
- `grad_norm`：unscale 和 clip 后的 gradient norm。
- `batch_count`：全局 batch 计数，每步增加 `world_size`。
- `curr_run_batch_count`：当前 run 的 batch 计数，也每步增加 `world_size`。
- `sample_count`：已处理样本数，每步增加 `batch_size * world_size`。
- `epoch_count`：`sample_count / train_total_samples`。

周期评估 logs：

- `train_eval/<metric>`：用 persistent train iterator 采样得到的 evaluation-mode metrics。
- `test/<metric>`：如果 test loader 存在，用 persistent test iterator 采样得到的 evaluation-mode metrics。

`Trainer.full_eval` 的完整评估 logs：

- `full_eval/train/<metric>`
- `full_eval/test/<metric>`

独立 `eval.py` 会把 metrics 写到：

```text
eval_logs/<timestamp>-<eval_exp_name>-split=test-seed=<seed>/metrics.json
```

metric names 位于 `full_eval/test/...` 下。

## 训练流程

`Trainer.__init__`：

1. 如果 `--distributed=true`，初始化 distributed process group。
2. 用 `args.seed + rank` 设置随机种子。
3. 初始化 wandb 和 experiment directory。
4. 如果提供 checkpoint，加载 checkpoint metadata，并把 checkpoint model contract 应用到 args。
5. 构造 dataloaders，并推断 `robot_features_dim`、`scene_features_dim`。
6. 构造 `BaseModel`。
7. 如果需要，包一层 DDP。
8. 对所有 trainable parameters 创建 AdamW optimizer。
9. 如果提供 checkpoint，加载模型权重、optimizer 和 counters。
10. 记录模型和数据维度。

`Trainer.train`：

1. 使用 AMP `GradScaler`。
2. 当 `epoch_count < num_epochs` 时循环。
3. 每个 train step 前，如果 `eval_freq > 0` 且到达 cadence，则做周期评估。
4. 如果 `save_freq > 0` 且到达 cadence，则保存 checkpoint。
5. 将 batch 移到 device。
6. 在 autocast 下 forward `BaseModel`。
7. 如果 NaN output handler 标记该 batch，则跳过。
8. 计算 `model.loss_fn`。
9. 检查 total loss 是否 NaN。
10. 对 scaled loss 做 backward。
11. unscale optimizer gradients。
12. 用 `--grad_clip_max_norm` 做 grad norm clipping。
13. 处理 NaN grad norm。
14. 检查模型参数是否 NaN。
15. optimizer step 和 scaler update。
16. rank 0 将 metrics 记录到 wandb。
17. 更新 counters 和估算 epoch。
18. 达到目标 epoch 时保存 final checkpoint 并退出。

Optimizer：

- AdamW
- `lr = --base_lr`，默认 `1e-4`
- `weight_decay = --weight_decay`，默认 `0.01`

## 评估流程

`eval.py` 构造 `Tester(args)` 并调用 `run_evaluation`。

`Tester` 继承 `Trainer`，但做了这些评估专用处理：

- 设置 `WANDB_MODE=disabled`。
- 强制 non-DDP evaluation。
- 要求提供 `--model_path` 或 `--eval_exp_name`。
- 先加载 checkpoint metadata。
- 应用 checkpoint model contract。
- 要求请求评估的 domains 必须包含在 checkpoint training domains 中。
- 为了模型和 norm stats 兼容，会把 `args.domains` 设回 checkpoint train domains，同时在 eval loader 中 whitelist 用户请求的 eval domains。
- 如果不允许缺失 confidence masks，会把 seed 覆盖为 42。

对每个 release eval split，目前只有 `test`：

1. 可选地可视化样本。
2. 如果 `--run_confidence_annotation=true`，先跑一遍模型，把 low-confidence voxels 写入 `expert_confidence-seed=<seed>.h5`。
3. 否则，对于真实 domain，要求已有 confidence H5，或者在允许缺失时跳过 filtered metrics。
4. 构造确定性 eval loader。
5. 如果真实 domain 有 confidence H5，则注入 `scene_filter_mask`。
6. forward 模型，计算 loss dict，累积 weighted metrics 和 metadata。
7. 保存 JSON metrics report。

DROID filtered metrics：

- `scene_filter_mask` 通过 `scene_flows - __shift_amount__` 还原 world voxel coordinates 后构造。
- H5 中存储的 low-confidence voxels 会被剔除。
- loss 代码会输出：
  - `filtered_l2/mean`
  - `filtered_l2_moved/mean`
  - `filtered_l2_static/mean`
- README 中强调的 DROID 主指标是 `full_eval/test/filtered_l2_moved/mean`。

BEHAVIOR evaluation 不需要 confidence annotation，因为 simulation data 被视为 noiseless。

## 重要 CLI 参数的实际意义

- `--domains`：数据 / 模型 domains，例如 `droid` 或 `droid,behavior`。
- `--data_dirs`：和 `--domains` 一一对应的 WDS roots。
- `--norm_stats_path`：包含 canonical `norm_stats.json` 的目录。
- `--ptv3_size`：backbone size，取值为 `small|base|large`。
- `--predictor_dim`：模型内部 channel 数 `C`。
- `--ptv3_patch_size`：PTv3 attention grouping 使用的 patch size。
- `--grid_size`：数据 downsampling 和 confidence-mask voxelization 使用的 voxel size。
- `--max_scene_points`：sampling / augmentation 后的 scene point 数上限。
- `--max_robot_points`：robot sampler 的 point 数上限。
- `--train_min_num_cameras`、`--train_max_num_cameras`：训练时随机 camera 数范围。
- `--eval_min_num_cameras`、`--eval_max_num_cameras`：评估时 camera 数范围。
- `--huber_delta`：归一化 target 空间里的 Huber loss delta。
- `--confidence_thres`：confidence-aware metrics / annotation 中保留的高 confidence 点比例，不是绝对 confidence cutoff。
- `--eval_freq`：训练循环中周期评估 cadence，单位是 batch。
- `--save_freq`：checkpoint 保存 cadence，单位是 batch。
- `--num_eval_batches`：训练时周期评估采样 batch 数。
- `--eval_num_batches`：独立 eval 的 batch 数；`-1` 表示全量。
- `--run_confidence_annotation`：生成 confidence H5，而不是做普通 filtered evaluation。
- `--allow_missing_confidence_mask`：如果真实 domain 的 H5 不存在，则跳过 filtered metrics。

## 主要注意点

- `main` 分支消费的是 WDS shards，不是 Hugging Face 上的 raw package shards。
- 相对 `--data_dirs` 会被解析到 `dataset_components` 下，本地实验建议使用绝对路径。
- 即使 DINOv3 是 frozen，DINOv3 权重仍然是必需的。
- `T=11`，不是 10，因为包含 context frame。
- `scene_features` 只从 timestep 0 构造，同时包含 `dist2robot` 这类 trajectory-level features。
- 模型预测的是相对 displacement，然后再转换成绝对 scene point positions。
- `pred/*` metrics 是 normalized prediction values，而 `l2/*` metrics 是物理空间里的 point-position errors。
- `confidence_thres` 的行为是按 quantile 保留一个比例的高置信点。
- DROID filtered eval 中，seed 和 grid size 必须和 confidence H5 文件匹配。
- 本地 worktree 比 `origin/main` 领先 1 个提交；本总结反映的是当前本地文件。

