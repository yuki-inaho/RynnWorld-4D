import hashlib
import json

import numpy as np

from core.finetune.preprocessing.materialize import materialize_rgbdf, pad_rgbdf_inputs
from tests.test_colmap_rgbdf_source import make_source


def test_cpu_materializer_is_deterministic_and_private_path_free(tmp_path):
    source = make_source(tmp_path / "private-source")
    camera_path = source / "scenes" / "scene_000000" / "cameras.npz"
    with np.load(camera_path) as values:
        camera = {key: values[key] for key in values.files}
    camera["extrinsics_w2c"][:, 0, 3] = np.arange(120, dtype=np.float32) * 0.001
    np.savez(camera_path, **camera)
    output = tmp_path / "artifacts"
    expected = {"train": 2, "val": 0, "smoke": 0}

    first = materialize_rgbdf(
        source,
        output,
        expected_clips=expected,
        min_valid_flow_ratio=0.01,
        expected_source_size=(8, 6),
        target_height=6,
        target_width=8,
    )
    artifact = next((output / "intermediate" / "train").glob("*.npz"))
    first_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    second = materialize_rgbdf(
        source,
        output,
        expected_clips=expected,
        min_valid_flow_ratio=0.01,
        expected_source_size=(8, 6),
        target_height=6,
        target_width=8,
    )

    assert first["clip_counts"] == expected
    assert second["flow_scale"] == first["flow_scale"]
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == first_hash
    with np.load(artifact) as arrays:
        assert arrays["rgb"].shape == (65, 6, 8, 3)
        assert arrays["depth_rgb"].shape == (65, 3, 6, 8)
        assert arrays["flow_rgb"].shape == (65, 3, 6, 8)
        assert np.isfinite(arrays["flow_rgb"]).all()
    rendered = json.dumps(first)
    assert str(tmp_path) not in rendered
    assert "private-source" not in rendered


def test_rgbdf_padding_preserves_source_and_uses_modality_specific_borders():
    rgb = np.full((1, 2, 3, 3), 7, dtype=np.uint8)
    depth = np.full((1, 3, 2, 3), 11, dtype=np.uint8)
    flow = np.full((1, 3, 2, 3), 13, dtype=np.uint8)
    valid = np.ones((1, 2, 3), dtype=bool)

    rgb_out, depth_out, flow_out, depth_valid, flow_valid = pad_rgbdf_inputs(
        rgb,
        depth,
        flow,
        valid,
        valid,
        target_height=4,
        target_width=5,
    )

    assert rgb_out.shape == (1, 4, 5, 3)
    assert np.all(rgb_out == 7)
    assert np.all(depth_out[:, :, 1:3, 1:4] == 11)
    assert np.all(flow_out[:, :, 1:3, 1:4] == 13)
    assert np.all(depth_out[:, :, 0] == 0)
    assert np.all(flow_out[:, :, 0] == 255)
    assert not depth_valid[:, 0].any()
    assert not flow_valid[:, 0].any()
