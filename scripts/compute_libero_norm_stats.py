#!/usr/bin/env python3
"""Compute PointWorld normalization statistics from LIBERO train samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from arguments import parse_args as pointworld_default_args
from dataset_components.dataloader import build_dataloader


class Moments:
    def __init__(self):
        self.count = 0
        self.total = None
        self.square_total = None

    def update(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1, values.shape[-1])
        if self.total is None:
            self.total = np.zeros(values.shape[-1], dtype=np.float64)
            self.square_total = np.zeros(values.shape[-1], dtype=np.float64)
        self.count += len(values)
        self.total += values.sum(axis=0)
        self.square_total += np.square(values).sum(axis=0)

    def result(self):
        mean = self.total / self.count
        variance = np.maximum(self.square_total / self.count - np.square(mean), 1e-12)
        return {"mean": mean.tolist(), "variance": variance.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-scene-points", type=int, default=12000)
    parser.add_argument("--max-robot-points", type=int, default=500)
    parser.add_argument("--num-cameras", type=int, default=2)
    cli = parser.parse_args()

    args = pointworld_default_args(skip_command_line=True)
    args.seed = cli.seed
    args.domains = ["libero"]
    args.data_dirs = [str(cli.data_dir.resolve())]
    args.deterministic_data = True
    args.deterministic_train = True
    args.eval_min_num_cameras = cli.num_cameras
    args.eval_max_num_cameras = cli.num_cameras
    args.max_scene_points = cli.max_scene_points
    args.max_robot_points = cli.max_robot_points
    args.batch_size = 1
    args.num_workers = 0
    args.eval_num_workers = 0
    loader, _ = build_dataloader(
        args, "test", override_splits="train",
    )
    robot = Moments()
    scene = Moments()
    per_step = [Moments() for _ in range(11)]
    count = 0
    for sample in loader:
        robot.update(sample["robot_features"][0].numpy())
        scene.update(sample["scene_features"][0].numpy())
        relative = sample["gt_scene_flows_relative"][0].numpy()
        if relative.shape[0] != 11:
            raise ValueError(f"Expected 11 frames, got {relative.shape}")
        for timestep in range(11):
            per_step[timestep].update(relative[timestep])
        count += 1
        if count % 25 == 0:
            print(f"stats samples={count}")
        if cli.max_samples > 0 and count >= cli.max_samples:
            break
    if count == 0:
        raise RuntimeError("No train samples were decoded")
    payload = {
        "statistics": {
            "libero": {
                "robot_features": robot.result(),
                "scene_features": scene.result(),
            }
        },
        "per_timestep_statistics": {
            "libero": {
                "gt_scene_flows_relative": {
                    f"timestep_{index}": moments.result()
                    for index, moments in enumerate(per_step)
                }
            }
        },
        "metadata": {
            "source_data_dir": str(cli.data_dir.resolve()),
            "num_samples": count,
            "mode": "deterministic-test-pipeline-on-train-split",
            "seed": cli.seed,
        },
    }
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    output = cli.output_dir / "norm_stats.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output.resolve()} from {count} samples")


if __name__ == "__main__":
    main()
