"""Map-style loader for compact LIBERO PointWorld HDF5 shards."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import h5py
import numpy as np
from torch.utils.data import Dataset

from dataset_components.cameras import sample_cameras
from dataset_components.decoders import build_flow_sample
from dataset_components.pipeline import apply_release_pipeline_to_sample
from dataset_components.robot import canonicalize_gripper_keys_and_flags


def _sample_key(clip_key: str) -> str:
    demo, frame_range = clip_key.split("__", 1)
    start, end = frame_range.split(":", 1)
    return f"{demo}-{start}-{end}"


class IndexedH5Reader:
    """Read decoded PointWorld samples by integer index from H5 shards.

    File handles are opened lazily and are deliberately removed when the
    object is pickled, giving every DataLoader worker its own read-only handle.
    """

    def __init__(self, paths: str | Path | Iterable[str | Path]):
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self.paths = [str(Path(path).resolve()) for path in paths]
        if not self.paths:
            raise ValueError("IndexedH5Reader requires at least one H5 shard")
        self.index: list[tuple[int, str]] = []
        self._handles: dict[int, h5py.File] = {}
        for file_index, path in enumerate(self.paths):
            with h5py.File(path, "r") as handle:
                contract = str(handle.attrs.get("pointworld_contract", ""))
                if contract != "libero-indexed-training-v1":
                    raise ValueError(
                        f"Unsupported H5 contract {contract!r} in {path}"
                    )
                keys = handle["_index/clip_keys"].asstr()[()].tolist()
            self.index.extend((file_index, str(key)) for key in keys)

    def __len__(self):
        return len(self.index)

    def _handle(self, file_index: int) -> h5py.File:
        handle = self._handles.get(file_index)
        if handle is None:
            handle = h5py.File(self.paths[file_index], "r", swmr=True)
            self._handles[file_index] = handle
        return handle

    def __getitem__(self, item: int) -> dict:
        file_index, clip_key = self.index[item]
        source = self._handle(file_index)
        clip = source[clip_key]
        sample = {"__key__": _sample_key(clip_key), "__domain__": "libero"}
        for name in (
            "joint_positions", "gripper_positions", "gripper_pose",
            "gripper_open",
        ):
            sample[name] = clip[name][()]
        sample["robot_root_transform"] = source["_robot_root_transform"][()]

        body_entity_ids = clip["body_entity_ids"][()]
        body_dense = clip["body_dense_preserve_mask"][()]
        for camera_name in sorted(
            name for name in clip if name.startswith("camera_")
        ):
            camera = clip[camera_name]
            flows = camera["scene_flows"][()].astype(np.float32)
            frame_count, point_count = flows.shape[:2]
            body_ids = camera["scene_body_ids"][()].astype(np.int32)
            prefix = f"{camera_name}_"
            sample[prefix + "scene_flows"] = flows
            sample[prefix + "scene_normals"] = (
                camera["scene_normals"][()].astype(np.float32) / 127.0
            )
            sample[prefix + "scene_colors"] = np.repeat(
                camera["scene_colors0"][()][None], frame_count, axis=0
            )
            all_valid = np.ones((frame_count, point_count), dtype=bool)
            sample[prefix + "scene_visibility"] = all_valid
            sample[prefix + "scene_depth_valid_mask"] = all_valid.copy()
            sample[prefix + "scene_body_ids"] = np.repeat(
                body_ids[None], frame_count, axis=0
            )
            sample[prefix + "scene_entity_ids"] = np.repeat(
                body_entity_ids[body_ids][None], frame_count, axis=0
            )
            sample[prefix + "scene_dense_preserve_mask"] = np.repeat(
                body_dense[body_ids][None], frame_count, axis=0
            )
            sample[prefix + "initial_depth"] = (
                camera["initial_depth"][()].astype(np.float32) / 1000.0
            )
            sample[prefix + "intrinsic"] = camera["intrinsic"][()]
            sample[prefix + "extrinsic"] = np.linalg.inv(
                camera["extrinsic"][()]
            ).astype(np.float32)
            jpeg = np.asarray(camera["initial_rgb"][0], dtype=np.uint8)
            bgr = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(
                    f"Failed to decode {clip_key}/{camera_name}/initial_rgb"
                )
            sample[prefix + "initial_rgb"] = np.ascontiguousarray(
                bgr[..., ::-1]
            )
        return sample

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handles"] = {}
        return state

    def __del__(self):
        self.close()


class LiberoIndexedH5Dataset(Dataset):
    """Apply the unchanged PointWorld release pipeline to indexed H5 clips."""

    def __init__(
        self,
        paths,
        *,
        mode,
        args,
        robot_sampler,
        rank=0,
        has_bimanual_robot=False,
    ):
        if mode not in {"train", "test"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.reader = IndexedH5Reader(paths)
        self.mode = mode
        self.args = args
        self.robot_sampler = robot_sampler
        self.rank = rank
        self.has_bimanual_robot = has_bimanual_robot

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, item):
        deterministic = self.mode != "train" or self.args.deterministic_train
        sample = self.reader[item]
        sample = build_flow_sample(
            sample,
            domain="libero",
            robot_sampler=self.robot_sampler,
            max_robot_points=self.args.max_robot_points,
            deterministic=deterministic,
            seed=self.args.seed + self.rank,
            force_single_arm=False,
        )
        if self.mode == "train":
            min_cameras = self.args.train_min_num_cameras
            max_cameras = self.args.train_max_num_cameras
        else:
            min_cameras = self.args.eval_min_num_cameras
            max_cameras = self.args.eval_max_num_cameras
        sample = sample_cameras(
            sample,
            min_num_cameras=min_cameras,
            max_num_cameras=max_cameras,
            deterministic=deterministic,
            seed=self.args.seed + self.rank,
        )
        sample = canonicalize_gripper_keys_and_flags(sample)
        return apply_release_pipeline_to_sample(
            sample,
            domain="libero",
            mode=self.mode,
            args=self.args,
            has_bimanual_robot=self.has_bimanual_robot,
            rank=self.rank,
        )
