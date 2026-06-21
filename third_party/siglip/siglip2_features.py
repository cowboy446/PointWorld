# SPDX-License-Identifier: Apache-2.0
"""Frozen SigLIP2 vision feature extraction helpers.

This adapter deliberately depends on Hugging Face Transformers rather than
vendoring the full SigLIP2 implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Siglip2FeatureConfig:
    model_name: str = "google/siglip2-base-patch16-256"
    layer: int = -1


class Siglip2FeatureExtractor(torch.nn.Module):
    """Frozen SigLIP2 vision tower wrapper returning patch-level features."""

    def __init__(self, config: Siglip2FeatureConfig, device: torch.device | str = "cpu"):
        super().__init__()
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ImportError(
                "SigLIP2 support requires `transformers`. Install the updated "
                "`environments/requirements.txt` or run `pip install transformers`."
            ) from exc

        self.config = config
        self.model = AutoModel.from_pretrained(config.model_name)
        self.image_processor = AutoImageProcessor.from_pretrained(config.model_name)
        self.model.eval()
        self.model.to(device)
        for param in self.model.parameters():
            param.requires_grad_(False)

        vision_config = getattr(self.model.config, "vision_config", self.model.config)
        self.hidden_size = int(getattr(vision_config, "hidden_size"))
        self.patch_size = int(getattr(vision_config, "patch_size", 16))
        vision_model = getattr(self.model, "vision_model", None)
        if vision_model is None:
            raise RuntimeError(f"Model {config.model_name!r} does not expose a vision_model.")
        forward_params = inspect.signature(vision_model.forward).parameters
        self._uses_patchified_input = "attention_mask" in forward_params and "spatial_shapes" in forward_params

        mean = getattr(self.image_processor, "image_mean", (0.5, 0.5, 0.5))
        std = getattr(self.image_processor, "image_std", (0.5, 0.5, 0.5))
        self.register_buffer("_image_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_image_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize `(B,3,H,W)` images in `[0,1]` using checkpoint stats."""
        mean = self._image_mean.to(device=images.device, dtype=images.dtype)
        std = self._image_std.to(device=images.device, dtype=images.dtype)
        return (images - mean) / std

    def patchify_images(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Patchify normalized `(B,3,H,W)` images for SigLIP2 NaFlex input."""
        patch = self.patch_size
        patch_h = images.shape[-2] // patch
        patch_w = images.shape[-1] // patch
        if patch_h < 1 or patch_w < 1:
            raise ValueError(
                f"Image size {tuple(images.shape[-2:])} is smaller than SigLIP2 patch size {patch}."
            )
        cropped = images[..., : patch_h * patch, : patch_w * patch]
        channels_last = cropped.permute(0, 2, 3, 1).contiguous()
        patches = channels_last.reshape(
            images.shape[0],
            patch_h,
            patch,
            patch_w,
            patch,
            images.shape[1],
        )
        patches = patches.permute(0, 1, 3, 2, 4, 5).reshape(images.shape[0], patch_h * patch_w, -1)
        attention_mask = torch.ones(
            images.shape[0],
            patch_h * patch_w,
            device=images.device,
            dtype=torch.bool,
        )
        spatial_shapes = torch.tensor(
            [[patch_h, patch_w]] * images.shape[0],
            device=images.device,
            dtype=torch.long,
        )
        return patches, attention_mask, spatial_shapes

    def _vision_forward_patchified(
        self,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ):
        vision_model = getattr(self.model, "vision_model", None)
        if vision_model is None:
            raise RuntimeError(f"Model {self.config.model_name!r} does not expose a vision_model.")
        kwargs = {
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
            "spatial_shapes": spatial_shapes,
            "output_hidden_states": True,
            "return_dict": True,
        }
        try:
            return vision_model(**kwargs, interpolate_pos_encoding=True)
        except TypeError:
            return vision_model(**kwargs)

    def _vision_forward_image(self, pixel_values: torch.Tensor):
        vision_model = getattr(self.model, "vision_model", None)
        if vision_model is None:
            raise RuntimeError(f"Model {self.config.model_name!r} does not expose a vision_model.")
        return vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
            interpolate_pos_encoding=True,
        )

    @torch.no_grad()
    def extract_patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Return patch tokens `(B, N, C)` for images already in `[0,1]`."""
        normalized = self.normalize_images(images)
        if self._uses_patchified_input:
            pixel_values, attention_mask, spatial_shapes = self.patchify_images(normalized)
            outputs = self._vision_forward_patchified(pixel_values, attention_mask, spatial_shapes)
        else:
            outputs = self._vision_forward_image(normalized)
        if self.config.layer == -1:
            tokens = outputs.last_hidden_state
        else:
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("SigLIP2 did not return hidden_states.")
            tokens = hidden_states[self.config.layer]

        patch_h = images.shape[-2] // self.patch_size
        patch_w = images.shape[-1] // self.patch_size
        n_patches = patch_h * patch_w
        if tokens.shape[1] != n_patches:
            tokens = tokens[:, -n_patches:, :]
        return tokens

    @torch.no_grad()
    def extract_feature_grid(self, images: torch.Tensor) -> torch.Tensor:
        """Return patch feature grid `(B, C, patch_h, patch_w)`."""
        tokens = self.extract_patch_tokens(images)
        patch_h = images.shape[-2] // self.patch_size
        patch_w = images.shape[-1] // self.patch_size
        return tokens.reshape(images.shape[0], patch_h, patch_w, -1).permute(0, 3, 1, 2).contiguous()

    @torch.no_grad()
    def extract_dense_feature_map(self, images: torch.Tensor) -> torch.Tensor:
        """Return bilinearly upsampled feature map `(B, C, H, W)`."""
        grid = self.extract_feature_grid(images)
        return F.interpolate(grid, size=images.shape[-2:], mode="bilinear", align_corners=False)
