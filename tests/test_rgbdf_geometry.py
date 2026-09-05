from __future__ import annotations

import numpy as np
import pytest

from core.finetune.preprocessing.depth import decode_metric_depth, encode_metric_depth
from core.finetune.preprocessing.flow_encoding import encode_flow_hsv, fit_flow_scale
from core.finetune.preprocessing.rigid_flow import compute_rigid_flow


def test_metric_depth_encoding_preserves_invalid_and_round_trips() -> None:
    depth = np.array([[[0, 1, 2, 649, 1300, 1500]]], dtype=np.uint16)
    encoded, valid = encode_metric_depth(depth, max_depth_mm=1300)
    decoded = decode_metric_depth(encoded, valid, max_depth_mm=1300)

    assert encoded.shape == (1, 3, 1, 6)
    assert encoded.dtype == np.uint8
    assert np.array_equal(encoded[:, 0], encoded[:, 1])
    assert encoded[0, 0, 0, 0] == 0
    assert encoded[0, 0, 0, -1] == 255
    assert np.max(np.abs(decoded[valid] - np.clip(depth, 1, 1300)[valid])) <= 3


def test_rigid_flow_identity_is_zero() -> None:
    depth = np.ones((2, 3, 4), dtype=np.float32)
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, :3], 2, axis=0)
    flow, valid = compute_rigid_flow(depth, intrinsics, extrinsics)

    assert np.all(flow == 0)
    assert not valid[0].any()
    assert valid[1].all()


def test_rigid_flow_known_camera_translation() -> None:
    depth = np.ones((2, 5, 7), dtype=np.float32)
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
    intrinsics[:, 0, 0] = 2
    intrinsics[:, 1, 1] = 2
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, :3], 2, axis=0)
    extrinsics[1, 0, 3] = 0.5

    flow, valid = compute_rigid_flow(depth, intrinsics, extrinsics, occlusion_check=False)

    assert np.max(np.abs(flow[1, valid[1], 0] - 1.0)) < 1e-4
    assert np.max(np.abs(flow[1, valid[1], 1])) < 1e-4


def test_flow_scale_and_white_zero_encoding() -> None:
    flow = np.array([[[[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]]]], dtype=np.float32)
    valid = np.array([[[False, True, True]]])
    scale = fit_flow_scale(flow, valid, percentile=100)
    encoded = encode_flow_hsv(flow, valid, magnitude_scale=scale)

    assert scale == pytest.approx(2.0)
    assert encoded.shape == (1, 3, 1, 3)
    assert np.array_equal(encoded[0, :, 0, 0], [255, 255, 255])
    assert not np.array_equal(encoded[0, :, 0, 1], [255, 255, 255])


@pytest.mark.parametrize("scale", [0.0, np.nan, np.inf])
def test_flow_encoding_rejects_invalid_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="magnitude scale"):
        encode_flow_hsv(
            np.zeros((1, 1, 1, 2), dtype=np.float32),
            np.ones((1, 1, 1), dtype=bool),
            magnitude_scale=scale,
        )
