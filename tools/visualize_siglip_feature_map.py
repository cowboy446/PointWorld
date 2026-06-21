#!/usr/bin/env python3
"""Export a same-size SigLIP2 feature map and an RGB PCA preview for one image."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from third_party.siglip.siglip2_features import Siglip2FeatureConfig, Siglip2FeatureExtractor


def _load_image(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _pca_rgb(feature_map: torch.Tensor) -> np.ndarray:
    """Create an `(H,W,3)` uint8 PCA preview from `(C,H,W)` features."""
    feat = feature_map.permute(1, 2, 0).reshape(-1, feature_map.shape[0]).float()
    feat = feat - feat.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(feat, full_matrices=False)
    rgb = feat @ vh[:3].T
    rgb = rgb.reshape(feature_map.shape[1], feature_map.shape[2], 3)
    rgb = rgb - rgb.amin(dim=(0, 1), keepdim=True)
    rgb = rgb / rgb.amax(dim=(0, 1), keepdim=True).clamp(min=1e-6)
    return (rgb.cpu().numpy() * 255.0).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path, help="Input RGB image.")
    parser.add_argument("--out", required=True, type=Path, help="Output .npy path for `(H,W,C)` features.")
    parser.add_argument("--preview", type=Path, default=None, help="Optional PCA RGB preview PNG path.")
    parser.add_argument("--model", default="google/siglip2-base-patch16-256", help="Hugging Face SigLIP2 checkpoint.")
    parser.add_argument("--layer", type=int, default=-1, help="Vision hidden-state layer, -1 for final output.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    image = _load_image(args.image).to(device)
    extractor = Siglip2FeatureExtractor(
        Siglip2FeatureConfig(model_name=args.model, layer=args.layer),
        device=device,
    )
    dense = extractor.extract_dense_feature_map(image)[0].cpu()
    hwc = dense.permute(1, 2, 0).contiguous().numpy().astype(np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, hwc)

    preview_path = args.preview
    if preview_path is None:
        preview_path = args.out.with_suffix(".pca.png")
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_pca_rgb(dense)).save(preview_path)
    print(f"Saved feature map: {args.out} shape={hwc.shape}")
    print(f"Saved PCA preview: {preview_path}")


if __name__ == "__main__":
    main()
