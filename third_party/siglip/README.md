# SigLIP2 Integration

PointWorld uses Hugging Face Transformers to load Google's SigLIP2 checkpoints
instead of vendoring the upstream model repository. The local files in this
directory are only thin adapters for PointWorld feature extraction.

## Current Choice

- Default checkpoint: `google/siglip2-base-patch16-256`
- Reason: SigLIP2 is the newer SigLIP family release and explicitly improves
  localization and dense-feature quality, which is the part PointWorld needs for
  projected scene-point features.
- Loader: `transformers.AutoModel` plus `transformers.AutoImageProcessor`

The first run downloads weights into the normal Hugging Face cache. To run
offline, pre-download the model on a machine with access:

```bash
bash third_party/siglip/download_siglip2.sh
```

By default the helper writes under
`third_party/siglip/checkpoints/google-siglip2-base-patch16-256`, which is
ignored by git. You can also pass a different model id and destination:

```bash
bash third_party/siglip/download_siglip2.sh \
  google/siglip2-base-patch16-384 \
  /data/checkpoints/google-siglip2-base-patch16-384
```

Use that local path with `--scene_siglip_model /data/checkpoints/...` if the
training machine should avoid network access.

## Feature API

`siglip2_features.py` exposes `Siglip2FeatureExtractor`, which returns patch
tokens from the frozen vision tower:

- input image tensor: `(B, 3, H, W)` in `[0, 1]`
- patch tokens: `(B, floor(H / patch_size) * floor(W / patch_size), hidden)`
- feature grid: `(B, hidden, floor(H / patch_size), floor(W / patch_size))`
- dense feature map: `(B, hidden, H, W)` after bilinear upsampling

PointWorld samples the patch-token grid at projected scene-point locations,
matching the DINOv3 sampling path.

The adapter supports both SigLIP2 variants exposed by Transformers:

- NaFlex-style checkpoints that take already patchified inputs plus
  `pixel_attention_mask` and `spatial_shapes`.
- FixRes checkpoints, including the default `google/siglip2-base-patch16-256`,
  that are backwards compatible with SigLIP and take `(B, 3, H, W)` image
  tensors directly.
