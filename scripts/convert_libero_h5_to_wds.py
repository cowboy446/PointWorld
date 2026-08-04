#!/usr/bin/env python3
"""Convert a consolidated LIBERO clip HDF5 to trajectory-split WebDataset."""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import tarfile
from pathlib import Path

import h5py
import numpy as np


CLIP_RE = re.compile(
    r"^(?P<demo>demo_\d+)__(?P<start>\d{6}):(?P<end>\d{6})$"
)


def _npy(value):
    stream = io.BytesIO()
    np.save(stream, value)
    return stream.getvalue()


def _add(archive, name, payload):
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mtime = 0
    archive.addfile(info, io.BytesIO(payload))


def _clip_info(key: str):
    match = CLIP_RE.fullmatch(key)
    if match is None:
        raise ValueError(f"Invalid consolidated clip key: {key}")
    return match.group("demo"), int(match.group("start")), int(match.group("end"))


def split_demos(demo_names, train_fraction=0.8, seed=0):
    names = sorted(set(demo_names), key=lambda name: int(name.split("_")[-1]))
    if len(names) < 2:
        raise ValueError("At least two source demonstrations are required for a split")
    rng = np.random.default_rng(seed)
    shuffled = [names[index] for index in rng.permutation(len(names))]
    num_train = int(round(len(names) * train_fraction))
    num_train = min(max(num_train, 1), len(names) - 1)
    return set(shuffled[:num_train]), set(shuffled[num_train:])


def _write_sample(archive, src, clip_key):
    clip = src[clip_key]
    demo_name, start, end = _clip_info(clip_key)
    camera_keys = sorted(key for key in clip if key.startswith("camera_"))
    sample_key = f"{demo_name}-{start:06d}-{end:06d}"
    for camera_key in camera_keys:
        camera = clip[camera_key]
        for key in (
            "scene_flows", "scene_colors", "scene_normals", "scene_visibility",
            "scene_depth_valid_mask", "initial_depth", "intrinsic",
        ):
            _add(
                archive, f"{sample_key}.{camera_key}_{key}.npy",
                _npy(camera[key][()]),
            )
        _, point_count = camera["scene_flows"].shape[:2]
        for key in (
            "scene_body_ids", "scene_geom_ids", "scene_entity_ids",
            "scene_dense_preserve_mask",
        ):
            values = camera[key][()]
            if values.shape != (point_count,):
                raise ValueError(
                    f"{clip_key}/{camera_key}/{key} has shape {values.shape}; "
                    f"expected {(point_count,)}"
                )
            # Point identity and the clip-level dense-preserve decision are
            # time-invariant.  Keep the natural (N,) representation on disk;
            # PointWorld expands it lazily after decoding.  The previous
            # (T,N) representation repeated the same values for all 11 frames
            # and nearly doubled every WDS shard.
            _add(
                archive, f"{sample_key}.{camera_key}_{key}.npy",
                _npy(values),
            )
        # HDF5 keeps the natural camera-to-LIBERO-base pose used to build the
        # point cloud. PointWorld's projector expects world/base-to-camera.
        _add(
            archive, f"{sample_key}.{camera_key}_extrinsic.npy",
            _npy(np.linalg.inv(camera["extrinsic"][()]).astype(np.float32)),
        )
        _add(
            archive, f"{sample_key}.{camera_key}_initial_rgb.jpg",
            np.asarray(camera["initial_rgb"][0], dtype=np.uint8).tobytes(),
        )
    for key in (
        "gripper_open", "gripper_pose", "joint_positions", "gripper_positions",
    ):
        _add(archive, f"{sample_key}.{key}.npy", _npy(clip[key][()]))
    _add(
        archive, f"{sample_key}.robot_root_transform.npy",
        _npy(src["_robot_root_transform"][()]),
    )
    _add(archive, f"{sample_key}.__domain__", b"libero")
    metadata = {
        "source_h5": str(Path(src.filename).resolve()),
        "clip_key": clip_key,
        "source_demo": demo_name,
        "source_frame_range": [start, end],
        "camera_names": camera_keys,
        "camera_resolution": list(clip[camera_keys[0]]["initial_depth"].shape),
        "sampling_kinds": json.loads(clip.attrs.get("sampling_kinds", "[]")),
        "contact_pattern": str(clip.attrs.get("sampling_contact_pattern", "")),
        "robot_model": str(src.attrs["robot_model"]),
        "robot_urdf": str(src.attrs["robot_urdf"]),
        "dense_preserve_definition": str(src.attrs["dense_preserve_definition"]),
        "dense_motion_threshold_m": float(clip.attrs["dense_motion_threshold_m"]),
        "body_names": [str(value) for value in clip["body_names"].asstr()[()]],
        "body_entity_ids": clip["body_entity_ids"][()].astype(int).tolist(),
        "body_dense_preserve_mask": (
            clip["body_dense_preserve_mask"][()].astype(bool).tolist()
        ),
        "body_physical_eligible_mask": (
            clip["body_physical_eligible_mask"][()].astype(bool).tolist()
        ),
        "entity_max_point_displacement_m": (
            clip["entity_max_point_displacement_m"][()].astype(float).tolist()
        ),
    }
    _add(
        archive, f"{sample_key}.libero_metadata.json",
        json.dumps(metadata, sort_keys=True).encode(),
    )
    return sample_key


def convert_split(h5_path: Path, output_dir: Path, train_fraction=0.8, seed=0):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train").mkdir(exist_ok=True)
    (output_dir / "test").mkdir(exist_ok=True)
    with h5py.File(h5_path, "r") as src:
        clip_keys = sorted(
            key for key in src
            if isinstance(src[key], h5py.Group) and CLIP_RE.fullmatch(key)
        )
        demos = [_clip_info(key)[0] for key in clip_keys]
        train_demos, test_demos = split_demos(demos, train_fraction, seed)
        split_keys = {
            "train": [key for key in clip_keys if _clip_info(key)[0] in train_demos],
            "test": [key for key in clip_keys if _clip_info(key)[0] in test_demos],
        }
        sample_keys = {}
        for split, keys in split_keys.items():
            sample_keys[split] = []
            split_demos_ordered = sorted(
                {_clip_info(key)[0] for key in keys},
                key=lambda name: int(name.split("_")[-1]),
            )
            for demo_name in split_demos_ordered:
                demo_keys = [
                    key for key in keys if _clip_info(key)[0] == demo_name
                ]
                tar_path = (
                    output_dir / split / f"libero-{split}-{demo_name}.tar"
                )
                temporary = tar_path.with_name(tar_path.name + ".tmp")
                if temporary.exists():
                    temporary.unlink()
                with tarfile.open(temporary, "w") as archive:
                    sample_keys[split].extend(
                        _write_sample(archive, src, key) for key in demo_keys
                    )
                os.replace(temporary, tar_path)
            print(
                f"Wrote {output_dir / split}: shards={len(split_demos_ordered)} "
                f"samples={len(keys)}"
            )
    split_manifest = {
        "schema": "libero_pointworld_trajectory_split.v2",
        "static_scene_point_fields": [
            "scene_body_ids",
            "scene_geom_ids",
            "scene_entity_ids",
            "scene_dense_preserve_mask",
        ],
        "static_scene_point_field_shape": "N",
        "source_h5": str(h5_path.resolve()),
        "seed": seed,
        "train_fraction": train_fraction,
        "train_demos": sorted(train_demos, key=lambda x: int(x.split("_")[-1])),
        "test_demos": sorted(test_demos, key=lambda x: int(x.split("_")[-1])),
        "train_samples": len(sample_keys["train"]),
        "test_samples": len(sample_keys["test"]),
    }
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pointworld_metadata = {
        "train": {"processed_count": len(sample_keys["train"])},
        "test": {"processed_count": len(sample_keys["test"])},
    }
    (output_dir / "metadata_rank0.json").write_text(
        json.dumps(pointworld_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return split_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not 0.0 < args.train_fraction < 1.0:
        parser.error("--train-fraction must be between 0 and 1")
    manifest = convert_split(
        args.h5.resolve(), args.output_dir.resolve(),
        train_fraction=args.train_fraction, seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
