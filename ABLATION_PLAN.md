# PointWorld Ablation Branch Plan

日期：2026-06-18

本文件记录即将创建的 `ablation` 分支设计思路。`main` 分支保持 release 默认行为；`ablation` 分支用于验证 robot / scene 输入 feature embedding 是否都必要。

## 目标

在不重新构建数据集的前提下，增加以下开关：

- robot 和 scene raw feature 是否使用 `gripper_open` feature。
- scene DINOv3 使用哪些 intermediate layers。
- scene 是否完全不使用 DINOv3 feature。

## 设计原则

- 默认配置必须等价于当前 release 行为：
  - robot features: `robot_flows,robot_colors,robot_normals,gripper_open,robot_velocity,robot_acceleration`
  - scene raw features: `scene_flows,scene_colors,scene_normals,gripper_open,dist2robot`
  - DINOv3 layers: `4,11,17,23`
- 不要求重新生成 WebDataset。数据管线仍可产出完整 raw features；模型内部按配置选择实际消费的 feature slices。
- 如果禁用某个 raw feature，同步切片 normalization mean / var，避免维度不匹配。
- DINOv3 关闭时不加载 DINOv3 submodule / checkpoint，方便在没有 DINO 权重的机器上做几何 feature ablation。
- `predictor_dim` / PTv3 channel 仍保持用户指定值，例如 256。变化的是进入 projection MLP 的 raw feature 维度，而不是 transformer channel。

## 预期分支实现

计划在 `ablation` 分支增加：

- `--robot_use_gripper_open_feature=true|false`：历史命名保留，但语义是全局控制 robot 与 scene raw feature 是否消费 `gripper_open`。
- `--scene_dino_layers=4,11,17,23`
- `--scene_use_dino=true|false`

内部处理：

- `arguments.py` 解析并校验 DINO layer 列表。
- `pointworld/base.py` 根据实际启用的 feature names 切片 `robot_features` / `scene_features` 和两路 norm stats，再建立对应输入维度的 projection。
- `scene_featurizer.py` 根据 `scene_use_dino` 和 `scene_dino_layers` 决定是否加载 DINOv3，以及 DINO feature projection 的输入维度。
- `SceneFeatureEncoder` 在 DINO 关闭时只使用 raw scene feature projection；DINO 开启时保持 raw+DINO concat 后投影。

## 测试限制

当前没有下载完整数据集，不能直接用 train / eval 验证端到端数值正确性。分支完成后至少要做：

- `python -m compileall arguments.py scene_featurizer.py pointworld/base.py`
- `python - <<'PY' ... get_args(skip_command_line=True) ... PY` 的默认参数 smoke check
- 静态阅读确认默认行为兼容 release。
