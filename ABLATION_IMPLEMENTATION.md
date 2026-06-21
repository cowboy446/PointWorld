# Ablation Branch Implementation Notes

日期：2026-06-18

分支：`ablation`

本分支用于验证 PointWorld 是否需要当前完整的 robot / scene feature embedding。默认参数保持 release 行为；只有显式设置 ablation 开关时，模型结构和实际消费的输入 feature 才会变化。

## 新增参数

### `--robot_use_gripper_open_feature`

默认：`true`

- `true`：保持 release 行为，robot raw feature 和 scene raw feature 都包含 `gripper_open`。
- `false`：数据管线仍可产生完整 `robot_features` / `scene_features`，但模型在 projection 前切掉两路 `gripper_open` 对应维度；两路 normalization stats 也同步切片。

设计理由：

- 不要求重新构建 WDS。
- 不改变 `--robot_features` release 默认列表。
- 不改变 `--scene_features` release 默认列表。
- 对 DROID 单臂，默认 robot feature dim 约为 `16`，禁用后模型消费维度约为 `15`。
- 对 DROID 单臂，默认 scene raw feature dim 约为 `21`，禁用后模型消费维度约为 `20`。
- 对双臂，`gripper_open` 可能是 2 维；实现通过 `total_dim - known_non_gripper_dim` 分别推断 robot / scene gripper 维度。

### `--scene_use_2d_backbone`

默认：`true`

- `true`：使用配置的 2D backbone feature，与 raw scene feature concat 后投影到 `predictor_dim`。
- `false`：完全不加载 2D visual backbone，只用 raw scene features 投影到 `predictor_dim`。
- `--scene_use_dino` 是旧名兼容 alias。

### `--scene_dino_layers`

默认：`4,11,17,23`

- 可设置为任意 DINOv3 ViT-L/16 intermediate layer 列表，例如 `--scene_dino_layers=23` 或 `--scene_dino_layers=11,23`。
- 如果 `--scene_use_2d_backbone=true` 且使用 DINO，该列表不能为空。
- 如果要不用 2D backbone，推荐显式写：`--scene_use_2d_backbone=false --scene_dino_layers=none`。

## 维度设计

`predictor_dim` 不随 ablation 自动改变。无论是否禁用 DINO 或 gripper feature，PTv3 的输入 channel 仍是 `args.predictor_dim`，通常是 `256`。

变化的是 projection 前的 raw input dim：

- robot：`robot_proj` 的输入维度随 `--robot_use_gripper_open_feature` 改变。
- scene raw：`scene_raw_feat_proj` 的输入维度随 `--robot_use_gripper_open_feature` 改变。
- scene DINO：`SceneEncoder2D.feat_proj` 的输入维度是 `1024 * len(scene_dino_layers)`。
- scene no-DINO：`SceneFeatureEncoder` 只使用 `scene_raw_feat_proj`，不创建 `SceneEncoder2D`。

## 改动文件

- `arguments.py`
  - 新增 `--scene_use_2d_backbone`，并保留 `--scene_use_dino` 作为兼容 alias
  - 新增 `--scene_dino_layers`
  - 新增 `--robot_use_gripper_open_feature`
  - 新增 DINO layer list parser / validation
- `pointworld/checkpoint_contract.py`
  - 将三个新结构开关写入 checkpoint contract
  - legacy checkpoint 默认回填 release 行为
- `scene_featurizer.py`
  - DINO layers 从 args 读取
  - 支持 `scene_use_2d_backbone=false` 时跳过 2D backbone 加载
  - no-DINO 时只返回 raw scene projection
- `pointworld/base.py`
  - 增加 robot / scene feature dim / index helper
  - `robot_proj` 按模型实际消费维度构建
  - `scene_raw_feat_proj` 按模型实际消费维度构建
  - forward 中按 runtime tensor dim 动态切片 robot 和 scene raw feature
  - normalization stats 按相同 robot / scene feature indices 切片

## 默认兼容性

默认参数：

```bash
--scene_use_2d_backbone=true
--scene_dino_layers=4,11,17,23
--robot_use_gripper_open_feature=true
```

这对应原 release 行为：

- DINOv3 使用 4 层拼接：`4 * 1024 = 4096`
- `feat_proj: 4096 -> predictor_dim`
- robot feature 仍包含 `gripper_open`
- scene raw feature 与 DINO feature 仍 concat 后投影
- scene raw feature 仍包含 `gripper_open`

## 验证情况

当前机器没有完整数据集，不能做 train / eval 端到端验证。

已经完成：

```bash
python -m compileall arguments.py scene_featurizer.py pointworld/base.py pointworld/checkpoint_contract.py
```

通过。

默认参数 smoke check：

```bash
python - <<'PY'
from arguments import parse_args
args = parse_args(skip_command_line=True)
print(args.scene_use_2d_backbone, args.scene_dino_layers, args.robot_use_gripper_open_feature)
PY
```

输出：

```text
True [4, 11, 17, 23] True
```

Ablation 参数 smoke check：

```bash
python - <<'PY'
import sys
from arguments import parse_args
old = sys.argv
try:
    sys.argv = ['prog', '--scene_use_2d_backbone=false', '--scene_dino_layers=none', '--robot_use_gripper_open_feature=false']
    args = parse_args()
    print(args.scene_use_2d_backbone, args.scene_dino_layers, args.robot_use_gripper_open_feature)
finally:
    sys.argv = old
PY
```

输出：

```text
False [] False
```

Robot feature selector helper check in `pointwm` conda env:

```bash
conda run -n pointwm python -c "import torch; print(torch.__version__); from pointworld.base import _robot_feature_indices, _robot_selected_dim; features=['robot_flows','robot_colors','robot_normals','gripper_open','robot_velocity','robot_acceleration']; print('selected keep gripper, dim16:', _robot_selected_dim(features, 16, True)); print('selected drop gripper, dim16:', _robot_selected_dim(features, 16, False)); print('indices drop gripper, dim16:', _robot_feature_indices(features, 16, False).tolist()); print('indices drop gripper, dim15:', _robot_feature_indices(features, 15, False).tolist())"
```

输出：

```text
2.5.1+cu124
selected keep gripper, dim16: 16
selected drop gripper, dim16: 15
indices drop gripper, dim16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15]
indices drop gripper, dim15: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
```

2 维 gripper-open 推断 check:

```bash
conda run -n pointwm python -c "from pointworld.base import _robot_feature_indices, _robot_selected_dim; features=['robot_flows','robot_colors','robot_normals','gripper_open','robot_velocity','robot_acceleration']; print('selected drop gripper, dim17:', _robot_selected_dim(features, 17, False)); print('indices drop gripper, dim17:', _robot_feature_indices(features, 17, False).tolist())"
```

输出：

```text
selected drop gripper, dim17: 15
indices drop gripper, dim17: [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16]
```

Scene feature selector helper check in `pointwm` conda env:

```bash
conda run -n pointwm python -c "from pointworld.base import _scene_feature_indices, _scene_selected_dim; features=['scene_flows','scene_colors','scene_normals','gripper_open','dist2robot']; print('selected keep gripper, dim21:', _scene_selected_dim(features, 21, True)); print('selected drop gripper, dim21:', _scene_selected_dim(features, 21, False)); print('indices drop gripper, dim21:', _scene_feature_indices(features, 21, False).tolist()); print('indices drop gripper, dim20:', _scene_feature_indices(features, 20, False).tolist()); print('selected drop gripper, dim22:', _scene_selected_dim(features, 22, False)); print('indices drop gripper, dim22:', _scene_feature_indices(features, 22, False).tolist())"
```

输出：

```text
selected keep gripper, dim21: 21
selected drop gripper, dim21: 20
indices drop gripper, dim21: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
indices drop gripper, dim20: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
selected drop gripper, dim22: 20
indices drop gripper, dim22: [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
```

Scene raw projection dim check:

```bash
conda run -n pointwm python -c "from types import SimpleNamespace; from scene_featurizer import SceneFeatureEncoder; args=SimpleNamespace(scene_use_2d_backbone=False, scene_2d_backbone='dinov3', scene_dino_layers=[], disable_compile=True); enc=SceneFeatureEncoder(args, 128, {'scene_features_dim':20}, 0, lambda x:x); print(tuple(enc.scene_raw_feat_proj.weight.shape), type(enc.scene_proj).__name__, enc.backbone_names)"
```

输出：

```text
(128, 20) Identity []
```

未完成：

- 未运行 train / eval，因为用户当前没有下载完整数据集。

## 建议实验矩阵

在数据集和 DINO 权重可用后，建议至少跑：

1. baseline：默认参数。
2. no gripper raw feature：`--robot_use_gripper_open_feature=false`。
3. DINO last layer only：`--scene_dino_layers=23`。
4. DINO two layers：`--scene_dino_layers=11,23`。
5. no 2D backbone：`--scene_use_2d_backbone=false --scene_dino_layers=none`。
6. no 2D backbone + no robot gripper：组合 ablation。

每组都应报告：

- `full_eval/test/filtered_l2_moved/mean`
- `full_eval/test/l2_moved/mean`
- `full_eval/test/filtered_l2/mean`
- `full_eval/test/l2/mean`
- confidence keep fraction

filtered 和 unfiltered 都要看，因为这两类指标可能给出不同侧面的结论。
