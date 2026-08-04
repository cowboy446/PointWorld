#!/usr/bin/env python3
"""Compare deterministic PointWorld batches from indexed H5 and legacy WDS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from arguments import parse_args as pointworld_args
from dataset_components.dataloader import build_dataloader


def _args(data_dir: Path, seed: int):
    args = pointworld_args(skip_command_line=True)
    args.domains = ["libero"]
    args.data_dirs = [str(data_dir.resolve())]
    args.seed = seed
    args.deterministic_data = True
    args.deterministic_train = True
    args.max_scene_points = 12000
    args.max_robot_points = 1024
    args.train_min_num_cameras = args.train_max_num_cameras = 2
    args.eval_min_num_cameras = args.eval_max_num_cameras = 2
    args.batch_size = 1
    args.num_workers = 0
    args.eval_num_workers = 0
    return args


def _collect(data_dir: Path, seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    loader, info = build_dataloader(_args(data_dir, seed), "test")
    result = {}
    for batch in loader:
        key = batch["__key__"][0]
        result[key] = batch
    return result, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("indexed_h5", type=Path)
    parser.add_argument("wds", type=Path)
    parser.add_argument("--seed", type=int, default=512026)
    args = parser.parse_args()
    h5_samples, h5_info = _collect(args.indexed_h5, args.seed)
    wds_samples, wds_info = _collect(args.wds, args.seed)
    if set(h5_samples) != set(wds_samples):
        raise AssertionError(
            f"Sample keys differ: H5={sorted(h5_samples)}, "
            f"WDS={sorted(wds_samples)}"
        )

    fields = (
        "scene_flows", "scene_features", "scene_visibility",
        "scene_depth_valid_mask", "scene_body_ids", "scene_entity_ids",
        "scene_dense_preserve_mask", "robot_flows", "robot_features",
        "gt_scene_flows", "gt_scene_flows_relative", "scene_context_mask",
        "right_gripper_pose", "right_gripper_open", "joint_positions",
        "gripper_positions",
    )
    maximum_absolute_difference = {}
    compared = 0
    for key in sorted(h5_samples):
        h5_batch = h5_samples[key]
        wds_batch = wds_samples[key]
        for field in fields:
            if field not in h5_batch or field not in wds_batch:
                raise AssertionError(f"{key}: missing comparison field {field}")
            actual = h5_batch[field].cpu()
            expected = wds_batch[field].cpu()
            if actual.shape != expected.shape:
                raise AssertionError(
                    f"{key}/{field}: shape {tuple(actual.shape)} != "
                    f"{tuple(expected.shape)}"
                )
            if actual.dtype.is_floating_point:
                difference = float((actual - expected).abs().max().item())
                if not torch.allclose(actual, expected, rtol=1e-6, atol=1e-6):
                    raise AssertionError(
                        f"{key}/{field}: max abs difference={difference}"
                    )
            else:
                difference = 0.0
                if not torch.equal(actual, expected):
                    raise AssertionError(f"{key}/{field}: values differ")
            maximum_absolute_difference[field] = max(
                maximum_absolute_difference.get(field, 0.0), difference
            )
            compared += 1
    print(json.dumps(
        {
            "samples": len(h5_samples),
            "field_comparisons": compared,
            "maximum_absolute_difference": maximum_absolute_difference,
            "h5_info": h5_info,
            "wds_info": wds_info,
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
