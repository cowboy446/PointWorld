# PointWorld SigLIP2 Integration

This branch adds SigLIP2 as an optional 2-D visual backbone for scene-point
features. It keeps the DINOv3 path intact and adds a combined mode for direct
ablation:

- `--scene_2d_backbone=dinov3`
- `--scene_2d_backbone=siglip`
- `--scene_2d_backbone=dinov3+siglip`

`--scene_use_2d_backbone=false` disables all 2-D visual backbones and uses only
raw scene features. `--scene_use_dino=false` remains a deprecated compatibility
alias.

## Why SigLIP2

SigLIP2 is the newer Google SigLIP family release. The public model card and
Transformers documentation describe it as improving semantic understanding,
localization, and dense-feature behavior over SigLIP. PointWorld needs dense
scene-point features, so the default checkpoint here is:

```bash
google/siglip2-base-patch16-256
```

This is a conservative default: ViT-B keeps memory lower than large/So400m/g
variants while giving a patch-16 dense grid that matches the current DINOv3
sampling assumptions.

## Files Added Or Changed

- `third_party/siglip/siglip2_features.py`
  - Thin Hugging Face Transformers wrapper.
  - Loads a frozen `AutoModel` and `AutoImageProcessor`.
  - Exposes patch tokens, patch grids, and same-size dense feature maps.
- `tools/visualize_siglip_feature_map.py`
  - Input: one image.
  - Output: same-size `(H, W, C)` `.npy` feature map.
  - Also writes a PCA RGB preview PNG.
- `scene_featurizer.py`
  - DINOv3 and SigLIP2 now share the same 3-D point projection, visibility, and
    camera aggregation path.
  - The selected 2-D backbone outputs are layer-normalized, concatenated with
    raw scene features, and projected back to `predictor_dim`.
- `arguments.py`
  - Adds SigLIP2 model/layer arguments and the 2-D backbone selector.
- `pointworld/checkpoint_contract.py`
  - Adds the new visual-backbone fields to checkpoint compatibility metadata.

## Setup

Install the updated dependencies:

```bash
pip install -r environments/requirements.txt
```

The SigLIP2 checkpoint downloads through the normal Hugging Face cache on first
use. To prefetch:

```bash
bash third_party/siglip/download_siglip2.sh
```

DINOv3 still uses the existing local `third_party/dinov3` checkout and local
checkpoint file.

## Visualize One Image

```bash
python tools/visualize_siglip_feature_map.py \
  --image /path/to/image.png \
  --out script_outputs/siglip/image_features.npy \
  --preview script_outputs/siglip/image_features_pca.png
```

The `.npy` file has shape `(H, W, C)`, where `C` is the SigLIP2 vision hidden
size. The preview is only for quick inspection; training uses the tensor
features directly.

## Training Modes

DINOv3 only, matching previous behavior:

```bash
WANDB_MODE=disabled python train.py \
  --domains droid \
  --scene_use_2d_backbone=true \
  --scene_2d_backbone=dinov3 \
  --scene_dino_layers=4,11,17,23 \
  ...
```

SigLIP2 only:

```bash
WANDB_MODE=disabled python train.py \
  --domains droid \
  --scene_use_2d_backbone=true \
  --scene_2d_backbone=siglip \
  --scene_siglip_model third_party/siglip/checkpoints/google-siglip2-base-patch16-256 \
  ...
```

DINOv3 plus SigLIP2:

```bash
WANDB_MODE=disabled python train.py \
  --domains droid \
  --scene_use_2d_backbone=true \
  --scene_2d_backbone=dinov3+siglip \
  --scene_dino_layers=4,11,17,23 \
  --scene_siglip_model third_party/siglip/checkpoints/google-siglip2-base-patch16-256 \
  ...
```

Raw scene features only:

```bash
python train.py ... --scene_use_2d_backbone=false --scene_dino_layers=none
```

## Logging Actual Training RGB Inputs

To inspect the RGB images actually sampled by the dataloader and consumed by
the DINOv3/SigLIP2 scene backbone, enable:

```bash
WANDB_MODE=online python train.py \
  ... \
  --log_scene_rgb_to_wandb=true \
  --scene_rgb_log_freq=100 \
  --scene_rgb_log_max_images=4
```

At each logging event, rank 0 saves PNGs under:

```text
<log_dir>/<exp_name>/scene_rgb_inputs/
```

and logs them to W&B under:

```text
debug/scene_rgb_inputs
debug/scene_rgb_input_count
debug/scene_rgb_input_dir
```

These are the post-camera-sampling `cam*_initial_rgb` tensors in the training
batch. In other words, they are exactly the selected camera views that the 2-D
backbone sees for that step, not all cameras stored in the original sample.

## Implementation Notes

For PointWorld batches, camera RGB tensors remain at their dataset resolution.
Both DINOv3 and SigLIP2 consume the original view image, produce a patch grid,
and sample that grid at projected 3-D scene point locations. Pixels outside the
valid patch-token region are masked out before multi-camera averaging.

In `dinov3+siglip` mode, each backbone first projects its native feature
dimension to `predictor_dim`; the fused input is:

```text
[DINOv3 projected scene feature, SigLIP2 projected scene feature, raw scene feature]
```

then a final linear layer maps the concatenated tensor back to `predictor_dim`
before PTv3.

SigLIP2 in Transformers has both NaFlex and FixRes variants. NaFlex checkpoints
use patchified inputs plus `pixel_attention_mask` and `spatial_shapes`; FixRes
checkpoints are backwards compatible with SigLIP and accept image tensors
directly. The adapter detects the frozen vision tower's forward signature and
supports both forms. The default `google/siglip2-base-patch16-256` checkpoint
currently loads through the FixRes/SigLIP-compatible image-tensor path.

## Verification Log

Commands run on this branch:

```bash
/home/zhangrong/miniconda3/envs/pointwm/bin/python -m py_compile \
  arguments.py scene_featurizer.py pointworld/base.py \
  pointworld/checkpoint_contract.py training/trainer.py \
  third_party/siglip/siglip2_features.py tools/visualize_siglip_feature_map.py
```

```bash
/home/zhangrong/miniconda3/envs/pointwm/bin/python \
  tools/visualize_siglip_feature_map.py \
  --image /tmp/pointworld_siglip_test.png \
  --out /tmp/pointworld_siglip_features.npy \
  --preview /tmp/pointworld_siglip_features.png \
  --model /tmp/pointworld_tiny_siglip2 \
  --device cpu
```

Result: saved a same-size feature map with shape `(48, 64, 32)` using a local
tiny SigLIP2 checkpoint.

Synthetic `BaseModel.forward()` plus `loss_fn()` sanity checks also passed for:

- `--scene_2d_backbone=siglip`
- `--scene_2d_backbone=dinov3+siglip`

Both produced finite losses with DROID normalization stats. The combined-mode
check used the local DINOv3 checkpoint and a tiny local SigLIP2 checkpoint to
avoid a long network download during code validation.

After the real checkpoint downloaded, these commands also passed with
`third_party/siglip/checkpoints/google-siglip2-base-patch16-256`:

```bash
/home/zhangrong/miniconda3/envs/pointwm/bin/python \
  tools/smoke_pointworld_scene_backbone.py \
  --scene_2d_backbone=siglip \
  --scene_siglip_model third_party/siglip/checkpoints/google-siglip2-base-patch16-256
```

```bash
/home/zhangrong/miniconda3/envs/pointwm/bin/python \
  tools/smoke_pointworld_scene_backbone.py \
  --scene_2d_backbone=dinov3+siglip \
  --scene_siglip_model third_party/siglip/checkpoints/google-siglip2-base-patch16-256 \
  --scene_dino_layers 23 --ns 4 --nr 2
```

```bash
/home/zhangrong/miniconda3/envs/pointwm/bin/python \
  tools/smoke_pointworld_scene_backbone.py \
  --scene_2d_backbone=dinov3 \
  --scene_dino_layers 23 --ns 4 --nr 2
```

The real SigLIP2 visualization command also passed:

```bash
/home/zhangrong/miniconda3/envs/pointwm/bin/python \
  tools/visualize_siglip_feature_map.py \
  --image /tmp/pointworld_siglip_test.png \
  --out /tmp/pointworld_siglip2_real_features.npy \
  --preview /tmp/pointworld_siglip2_real_features.png \
  --model third_party/siglip/checkpoints/google-siglip2-base-patch16-256 \
  --device cuda
```

Result: saved a same-size real SigLIP2 feature map with shape `(48, 64, 768)`.

Scope note: the validation above intentionally uses synthetic batches so it can
run without a local WebDataset. It verifies the same `BaseModel.forward()` and
`loss_fn()` path used by `train.py`, including PTv3, normalization stats, robot
feature projection, scene 2-D backbone fusion, and metric collection. A full
WebDataset training run is still the next experiment-level check once the
desired data path and run budget are chosen.
