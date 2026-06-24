# Gaussian 分支总结

## 范围

这个分支给 PointWorld 增加了一个可选的第 0 帧 3D Gaussian Splatting 监督路径。原有的 scene flow dynamics loss 仍然保留；当 `--enable_gaussian_splatting=true` 时，每个第 0 帧场景点会额外预测经典 3DGS 属性，并且只用第 0 帧的多视角输入图像做可微渲染监督。

实现遵循 Kerbl 等人在 2023 年 3DGS 论文中的标准表示：高斯中心、用四元数和尺度表示的协方差、球谐颜色和 opacity。本阶段颜色只使用 0 阶球谐函数。整体形式也参考了 pixelSplat 和 MVSplat 这类前馈式 3DGS 工作：网络一次前向预测高斯 primitives，再通过 photometric splatting loss 训练。

参考资料：

- Kerbl et al., 3D Gaussian Splatting for Real-Time Radiance Field Rendering: https://arxiv.org/abs/2308.04079
- 3DGS 项目页和代码链接：https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- pixelSplat: https://davidcharatan.com/pixelsplat/
- MVSplat: https://donydchen.github.io/mvsplat/

## 主要代码改动

- `pointworld/base.py`
  - 在现有 PointTransformer dynamics predictor 后增加 Gaussian head。
  - 每个场景点输出：
    - `delta_mu`：从第 0 帧场景点出发的有界 3D 中心偏移。
    - `sh0`：0 阶球谐 RGB 系数。
    - `q`：归一化四元数。
    - `s`：正的各向异性尺度。
    - `o`：sigmoid opacity。
  - 运行时高斯中心为 `means = scene_coord0 + delta_mu`。

- `pointworld/gaussian_renderer.py`
  - 接入 graphdeco/Inria 原始 `diff-gaussian-rasterization` CUDA 渲染器。
  - 保留原 PyTorch 渲染器，作为显式 `--gaussian_renderer_backend=torch` fallback/debug 路径。
  - 使用 batch 中所有采样到的 `cam*` 第 0 帧视角。
  - 使用现有 world-to-camera extrinsics 和 intrinsics 投影高斯中心。
  - 渲染与原图同尺寸的 RGB 图像，并生成第 0 帧初始点云投影 mask 供可视化或可选 masked loss 使用。
  - mask 会把每个初始场景点的投影像素及其周围 5x5 区域设为 1，避免监督区域过于离散。
  - 默认对整张图计算 photometric loss；如果 `--gaussian_use_projection_mask=true`，则只在初始点云投影 5x5 区域上计算 loss。

- `pointworld/losses.py`
  - 在现有 dynamics loss 上可选加入 Gaussian image loss。
  - image loss 对 batch 中所有有效采样视角取平均。
  - loss 形式为 `(1 - lambda) * L1 + lambda * D-SSIM`，默认 `lambda = 0.2`。
  - 只使用第 0 帧 `*_initial_rgb` 图像；第 1 到第 11 帧不做图像监督。

- `training/trainer.py`
  - 开启 Gaussian 监督后，在训练和训练过程 eval 中保存 Gaussian render preview。
  - 保存文件包括 `*_pred.png`、`*_target.png`、`*_mask.png`。
  - 训练输出目录为 `<train_log_dir>/<exp_name>/gaussian_renders/...`。

- `evaluation/tester.py`
  - 在独立 `eval.py` 中保存 eval render，目录为 `eval_logs/<eval_run>/gaussian_renders/...`。

- `arguments.py`
  - 新增启用 Gaussian 监督、loss 权重、CUDA renderer backend、near/far plane、初始化和渲染保存相关参数。

- `train_gaussian.sh`
  - 新增仿照 `train.sh` 的训练脚本，并显式打开 Gaussian CUDA backend。

- `install_gaussian_cuda.sh`
  - 初始化 graphdeco rasterizer submodule。
  - 如果系统已有 `nvcc`，直接使用本机 CUDA 编译器，避免离线服务器访问 NVIDIA conda channel。
  - 如果系统没有 `nvcc`，在 `pointwm` conda 环境中安装 CUDA 12.4 nvcc/CCCL headers。
  - 用 `--no-build-isolation` 编译安装 `diff_gaussian_rasterization`。
  - 自动运行一个 CUDA forward/backward smoke test。

- `third_party/diff-gaussian-rasterization`
  - 指向原始 3DGS 论文使用的经典 graphdeco/Inria CUDA rasterizer。

## 渲染器实现细节

默认渲染器现在是 `diff_gaussian`，也就是原始 3DGS release 中的 graphdeco/Inria CUDA rasterizer。它消耗预测出来的 Gaussian `means`、从 0 阶球谐转换得到的 RGB、opacity、各向异性 scale 和 quaternion rotation，并返回可微 RGB render。

旧的局部 PyTorch renderer 只保留为 fallback/debug backend。当前默认训练不加 mask，直接监督整张第 0 帧 RGB 图；如果打开 `--gaussian_use_projection_mask=true`，loss mask 会由第 0 帧初始点云投影生成，而不是由 Gaussian radii 生成。每个投影点默认覆盖以该像素为中心的 5x5 区域。

安装 CUDA backend：

```bash
bash install_gaussian_cuda.sh
```

## 监督范围

对每个采样到的相机视角：

1. 第 0 帧场景点加上预测的 `delta_mu`，得到 Gaussian 中心。
2. Gaussian 中心投影到该相机视角。
3. graphdeco CUDA rasterizer 使用预测的 `sh0/q/s/o` 渲染 RGB。
4. 默认情况下，整张图像都参与 L1 和 D-SSIM。
5. 如果 `--gaussian_use_projection_mask=true`，则只使用第 0 帧初始点云投影到该视角后形成的 5x5 局部区域作为 loss mask。

保存的 `*_mask.png` 始终表示初始点云投影 mask，默认每个投影点覆盖 5x5 区域，方便检查相机参数、坐标系和点云覆盖情况；它不一定代表当前训练 loss 的实际监督范围。

## 命令行参数

- `--enable_gaussian_splatting`：启用 Gaussian head、render loss 和 render 保存。
- `--gaussian_loss_weight`：Gaussian image loss 加到 dynamics loss 前的权重。
- `--gaussian_ssim_weight`：D-SSIM 混合权重，默认 `0.2`。
- `--gaussian_use_projection_mask`：是否只在第 0 帧初始点云投影 mask 上监督；默认 `false`，即整图监督。
- `--gaussian_renderer_backend`：渲染 backend，可选 `diff_gaussian`（默认）、`torch`、`auto`。
- `--gaussian_znear`：graphdeco CUDA rasterizer near plane。
- `--gaussian_zfar`：graphdeco CUDA rasterizer far plane。
- `--gaussian_min_render_depth`：进入 CUDA rasterizer 前的最小相机空间深度，默认 `0.05`；太近的点会被跳过，避免屏幕半径爆炸。
- `--gaussian_max_screen_radius`：每个视角中 Gaussian 的最大屏幕空间半径，默认 `64` 像素；渲染前会按 `scale <= max_screen_radius * depth / focal` 动态限制 scale。
- `--gaussian_patch_radius`：投影 mask 半径，默认 `2` 表示每个投影点标记 5x5 区域；同时也供 PyTorch fallback renderer 使用。
- `--gaussian_init_scale`：初始 world-space Gaussian scale。
- `--gaussian_min_scale`：softplus 后额外加上的正尺度下界。
- `--gaussian_max_scale`：渲染前的 world-space scale 上界，默认 `0.05`；用于防止少数高斯尺度发散后让 CUDA rasterizer 分配异常大的 tile buffer。
- `--gaussian_init_opacity`：logit 参数化之前的初始 opacity。
- `--gaussian_delta_mu_max`：`tanh` 有界中心偏移的最大绝对值，单位为 world units。
- `--gaussian_train_save_freq`：每 N 个训练 step 保存一次训练 render；`<=0` 表示关闭训练 step 保存。
- `--gaussian_eval_save`：在训练过程 eval 和 `eval.py` 中保存 render。
- `--gaussian_save_max_images`：每次保存事件最多保存多少张视角图。

示例：

```bash
bash train_gaussian.sh
```

## 输出目录

训练：

```text
train_logs/<exp_name>/gaussian_renders/train/step_00000300/
train_logs/<exp_name>/gaussian_renders/train_eval/step_00000300/
train_logs/<exp_name>/gaussian_renders/test/step_00000300/
```

评估：

```text
eval_logs/<timestamp>-<eval_exp_name>-split=test-seed=<seed>/gaussian_renders/eval_test/step_00000000/
```

每个保存视角包含：

- `<sample>_<cam>_pred.png`
- `<sample>_<cam>_target.png`
- `<sample>_<cam>_mask.png`

## 验证结果

已完成：

```bash
conda run -n pointwm python -m py_compile arguments.py pointworld/base.py pointworld/losses.py pointworld/gaussian_renderer.py training/trainer.py evaluation/tester.py
bash install_gaussian_cuda.sh
```

在 `pointwm` 环境中完成的额外 CUDA 检查：

- 直接调用 graphdeco rasterizer forward/backward smoke test：
  - 输出 shape 为 `(3, 64, 64)`。
  - radii 非零，示例为 `[7, 7]`。
  - Gaussian means 和 colors 有非零梯度。
- PointWorld `gaussian_image_loss(..., backend="diff_gaussian")` smoke test：
  - loss 为 `0.3992305696`。
  - `means`、`sh0`、`s`、`o` 都有非零梯度。
- 整图/投影 mask 两种监督模式 smoke test：
  - `--gaussian_use_projection_mask=false` 时 `mask_fraction = 1.0`。
  - `--gaussian_use_projection_mask=true` 时会在每个初始点云投影像素附近标记 5x5 区域，`mask_fraction` 随点数、重叠和边界裁剪变化。
- render 保存 smoke test：
  - 已在 `/tmp/pointworld_gaussian_cuda_test/gaussian_renders/smoke/step_00000001/` 写出 `pred.png`、`target.png`、`mask.png`。

## 显存保护

`diff-gaussian-rasterization` 会根据每个 Gaussian 在当前视角下的屏幕空间半径分配 tile/binning buffer。如果某些点离相机过近，或者预测尺度过大，即使 GPU 还有大量空闲显存，也可能出现异常大的分配请求，例如几十万 GiB 以上。这不是普通 batch size 显存不足，而是 rasterizer 的 screen-space radius 爆炸。

当前实现有三层保护：

- `--gaussian_max_scale=0.05`：限制 world-space scale。
- `--gaussian_min_render_depth=0.05`：过滤太靠近相机的点。
- `--gaussian_max_screen_radius=64`：按当前视角 depth/focal 动态限制 scale，让投影半径不超过指定像素。
- 进入 graphdeco CUDA rasterizer 前，只保留投影落在当前图像范围内、参数全为 finite 的 Gaussians；如果某个视角仍触发 rasterizer OOM，会跳过该视角，避免整个 eval/train step 崩溃。

未完成：

- 没有启动完整 WDS train/eval 大任务，因为这会直接开始本地大规模训练。
