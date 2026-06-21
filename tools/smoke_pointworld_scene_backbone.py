#!/usr/bin/env python3
"""Synthetic BaseModel forward/loss smoke test for scene 2-D backbone modes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arguments import parse_args as parse_pointworld_args
from pointworld.base import BaseModel, CONTEXT_HORIZON, PRED_HORIZON


def _make_batch(args, ns: int, nr: int, image_hw: tuple[int, int]) -> dict:
    device = torch.device(args.device)
    batch_size = 1
    timesteps = CONTEXT_HORIZON + PRED_HORIZON
    height, width = image_hw
    coords0 = torch.stack(
        [
            torch.linspace(8, max(8, width - 16), ns, device=device),
            torch.linspace(10, max(10, height - 14), ns, device=device),
            torch.ones(ns, device=device),
        ],
        dim=-1,
    ).view(1, 1, ns, 3)
    scene_flows = coords0.repeat(batch_size, timesteps, 1, 1)
    gt_scene_flows = scene_flows.clone()
    robot_flows = torch.rand(batch_size, timesteps, nr, 3, device=device)
    robot_flows[..., :2] = robot_flows[..., :2] * 40 + 8
    robot_flows[..., 2] = 1.0

    batch = {
        "__domain__": ["droid"],
        "scene_flows": scene_flows,
        "gt_scene_flows": gt_scene_flows,
        "gt_scene_flows_relative": gt_scene_flows - gt_scene_flows[:, :1],
        "scene_exists": torch.ones(batch_size, timesteps, ns, dtype=torch.bool, device=device),
        "robot_flows": robot_flows,
        "robot_features": torch.randn(batch_size, timesteps, nr, 16, device=device),
        "robot_exists": torch.ones(batch_size, timesteps, nr, dtype=torch.bool, device=device),
        "scene_features": torch.randn(batch_size, timesteps, ns, 31, device=device),
        "point_weights": torch.ones(batch_size, timesteps, ns, device=device),
        "scene_moved_mask": torch.zeros(batch_size, timesteps, ns, 1, dtype=torch.bool, device=device),
        "scene_static_mask": torch.ones(batch_size, timesteps, ns, 1, dtype=torch.bool, device=device),
        "scene_context_mask": torch.zeros(batch_size, timesteps, ns, 1, dtype=torch.bool, device=device),
        "scene_supervised_mask": torch.ones(batch_size, timesteps, ns, 1, dtype=torch.bool, device=device),
        "cam0_initial_rgb": torch.randint(
            0, 255, (batch_size, height, width, 3), dtype=torch.uint8, device=device
        ),
        "cam0_initial_depth": torch.ones(batch_size, height, width, device=device),
        "cam0_intrinsic": torch.eye(3, device=device).unsqueeze(0),
        "cam0_extrinsic": torch.eye(4, device=device).unsqueeze(0),
        "cam0_exists": torch.ones(batch_size, dtype=torch.bool, device=device),
    }
    batch["scene_context_mask"][:, 0] = True
    batch["point_weights"][:, 0] = 0.0
    return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_2d_backbone", default="siglip", choices=["dinov3", "siglip", "dinov3+siglip"])
    parser.add_argument("--scene_siglip_model", default="third_party/siglip/checkpoints/google-siglip2-base-patch16-256")
    parser.add_argument("--scene_dino_layers", default="23")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--predictor_dim", type=int, default=32)
    parser.add_argument("--ptv3_patch_size", type=int, default=16)
    parser.add_argument("--ns", type=int, default=8)
    parser.add_argument("--nr", type=int, default=4)
    parser.add_argument("--image_height", type=int, default=64)
    parser.add_argument("--image_width", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    args = parse_pointworld_args(skip_command_line=True)
    args.device = cli.device
    args.domains = ["droid"]
    args.norm_stats_path = "stats/droid"
    args.scene_use_2d_backbone = True
    args.scene_use_dino = args.scene_use_2d_backbone
    args.scene_2d_backbone = cli.scene_2d_backbone
    args.scene_dino_layers = [int(v) for v in cli.scene_dino_layers.split(",") if v]
    args.scene_siglip_model = cli.scene_siglip_model
    args.scene_siglip_layer = -1
    args.disable_compile = True
    args.distributed = False
    args.ptv3_size = "small"
    args.ptv3_patch_size = cli.ptv3_patch_size
    args.predictor_dim = cli.predictor_dim
    args.seed = 42

    device = torch.device(args.device)
    batch = _make_batch(args, cli.ns, cli.nr, (cli.image_height, cli.image_width))
    model = BaseModel(args, {"robot_features_dim": 16, "scene_features_dim": 31}, rank=0).to(device)
    model.train()
    outputs = model(batch, training=True)
    loss, logs = model.loss_fn(outputs, batch, training=True)
    print(
        f"mode={cli.scene_2d_backbone} pred_shape={tuple(outputs['scene_flows'].shape)} "
        f"loss={float(loss.detach().cpu()):.6f} finite={torch.isfinite(loss).item()} "
        f"metrics={len(logs)}"
    )


if __name__ == "__main__":
    main()
