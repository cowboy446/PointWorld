import numpy as np
import pytest

from dataset_components.decoders import _get_droid_scene_data


STATIC_FIELDS = (
    "scene_body_ids",
    "scene_geom_ids",
    "scene_entity_ids",
    "scene_dense_preserve_mask",
)


def _sample(frame_count=11, point_count=7):
    prefix = "camera_test"
    sample = {
        f"{prefix}_scene_flows": np.zeros(
            (frame_count, point_count, 3), dtype=np.float32
        ),
        f"{prefix}_scene_colors": np.zeros(
            (frame_count, point_count, 3), dtype=np.uint8
        ),
        f"{prefix}_scene_normals": np.zeros(
            (frame_count, point_count, 3), dtype=np.float32
        ),
        f"{prefix}_scene_visibility": np.ones(
            (frame_count, point_count), dtype=bool
        ),
        f"{prefix}_scene_depth_valid_mask": np.ones(
            (frame_count, point_count), dtype=bool
        ),
        f"{prefix}_initial_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
        f"{prefix}_initial_depth": np.zeros((4, 4), dtype=np.float32),
        f"{prefix}_intrinsic": np.eye(3, dtype=np.float32),
        f"{prefix}_extrinsic": np.eye(4, dtype=np.float32),
    }
    return sample


@pytest.mark.parametrize("storage_shape", ["compact", "legacy"])
def test_static_scene_labels_accept_compact_and_legacy_shapes(storage_shape):
    frame_count, point_count = 11, 7
    sample = _sample(frame_count, point_count)
    values = np.arange(point_count, dtype=np.int32)
    for field in STATIC_FIELDS:
        stored = values
        if storage_shape == "legacy":
            stored = np.repeat(values[None], frame_count, axis=0)
        sample[f"camera_test_{field}"] = stored

    decoded = _get_droid_scene_data(sample)["camera_test"]

    for field in STATIC_FIELDS:
        assert decoded[field].shape == (frame_count, point_count)
        np.testing.assert_array_equal(decoded[field][0], values)
        np.testing.assert_array_equal(decoded[field][-1], values)


def test_static_scene_labels_reject_invalid_shape():
    sample = _sample()
    for field in STATIC_FIELDS:
        sample[f"camera_test_{field}"] = np.arange(7, dtype=np.int32)
    sample["camera_test_scene_body_ids"] = np.zeros((7, 1), dtype=np.int32)

    with pytest.raises(ValueError, match="compact shape"):
        _get_droid_scene_data(sample)
