from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core.finetune.datasets.colmap_rgbdf_source import ColmapRGBDFSource


def make_source(root: Path, *, frame_count: int = 120) -> Path:
    metadata = {
        "format": "colmap_rgbd_v1",
        "schema_version": 1,
        "scene_count": 1,
        "frame_count": frame_count,
        "image": {"channels": 3, "dtype": "uint8", "width": 8, "height": 6},
        "depth": {
            "dtype": "uint16",
            "unit": "millimeters",
            "invalid_value": 0,
            "max_depth_mm": 1300,
            "width": 8,
            "height": 6,
        },
        "camera": {
            "extrinsics": "opencv_world_to_camera",
            "intrinsics": "pixel_units",
        },
    }
    (root / "splits").mkdir(parents=True)
    scene = root / "scenes" / "scene_000000"
    rgb = scene / "rgb"
    depth = scene / "depth"
    rgb.mkdir(parents=True)
    depth.mkdir()
    (root / "dataset.json").write_text(json.dumps(metadata), encoding="utf-8")

    frame_ids = np.arange(frame_count, dtype=np.int64)
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], frame_count, axis=0)
    intrinsics[:, 0, 0] = 100
    intrinsics[:, 1, 1] = 100
    intrinsics[:, 0, 2] = 3.5
    intrinsics[:, 1, 2] = 2.5
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, :3], frame_count, axis=0)
    np.savez(
        scene / "cameras.npz",
        frame_ids=frame_ids,
        intrinsics=intrinsics,
        extrinsics_w2c=extrinsics,
        quality_flags=np.ones(frame_count, dtype=bool),
    )
    sequences = np.stack([frame_ids[:-3], frame_ids[1:-2], frame_ids[2:-1], frame_ids[3:]], axis=1)
    np.savez(
        scene / "sequences.npz",
        sequences=sequences,
        lengths=np.full(len(sequences), 4, dtype=np.int64),
    )

    for frame_id in frame_ids:
        Image.fromarray(np.full((6, 8, 3), frame_id % 255, dtype=np.uint8)).save(
            rgb / f"frame_{frame_id:06d}.png"
        )
        Image.fromarray(np.full((6, 8), 1000, dtype=np.uint16)).save(
            depth / f"frame_{frame_id:06d}.png"
        )

    entries = "\n".join(f"scene_000000/sequence_{index:06d}" for index in range(70))
    (root / "splits" / "train.txt").write_text(entries + "\n", encoding="utf-8")
    (root / "splits" / "val.txt").write_text("scene_000000/sequence_000080\n", encoding="utf-8")
    (root / "splits" / "smoke.txt").write_text("scene_000000/sequence_000100\n", encoding="utf-8")
    return root


def test_source_validates_and_builds_split_safe_clips(tmp_path: Path) -> None:
    source = ColmapRGBDFSource(make_source(tmp_path / "source"))
    clips = source.clip_windows("train", length=65, stride=8)

    assert [clip.frame_ids[0] for clip in clips] == [0, 8]
    assert all(len(clip.frame_ids) == 65 for clip in clips)
    assert all(np.all(np.diff(clip.frame_ids) == 1) for clip in clips)


def test_source_rejects_unsupported_schema_without_leaking_root(tmp_path: Path) -> None:
    root = make_source(tmp_path / "private-scene")
    metadata = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    metadata["schema_version"] = 99
    (root / "dataset.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="schema is unsupported") as error:
        ColmapRGBDFSource(root)
    assert str(root) not in str(error.value)


def test_source_rejects_split_leakage(tmp_path: Path) -> None:
    root = make_source(tmp_path / "source")
    (root / "splits" / "val.txt").write_text(
        "scene_000000/sequence_000000\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="split frame leakage"):
        ColmapRGBDFSource(root)


def test_source_rejects_missing_image(tmp_path: Path) -> None:
    root = make_source(tmp_path / "source")
    (root / "scenes" / "scene_000000" / "rgb" / "frame_000010.png").unlink()
    with pytest.raises(ValueError, match="RGB or depth frame is missing"):
        ColmapRGBDFSource(root, enforce_split_disjointness=False)


def test_source_rejects_nonfinite_camera(tmp_path: Path) -> None:
    root = make_source(tmp_path / "source")
    path = root / "scenes" / "scene_000000" / "cameras.npz"
    with np.load(path) as values:
        payload = {key: values[key] for key in values.files}
    payload["intrinsics"][0, 0, 0] = np.nan
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="camera metadata must be finite"):
        ColmapRGBDFSource(root, enforce_split_disjointness=False)


def test_source_rejects_depth_unit_and_camera_shape(tmp_path: Path) -> None:
    root = make_source(tmp_path / "source")
    metadata = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    metadata["depth"]["unit"] = "meters"
    (root / "dataset.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="encoding or camera convention"):
        ColmapRGBDFSource(root)

    root = make_source(tmp_path / "second-source")
    camera_path = root / "scenes" / "scene_000000" / "cameras.npz"
    with np.load(camera_path) as values:
        payload = {key: values[key] for key in values.files}
    payload["extrinsics_w2c"] = payload["extrinsics_w2c"][:, :, :3]
    np.savez(camera_path, **payload)
    with pytest.raises(ValueError, match="incompatible shapes"):
        ColmapRGBDFSource(root, enforce_split_disjointness=False)


def test_source_rejects_duplicate_split_sequence(tmp_path: Path) -> None:
    root = make_source(tmp_path / "source")
    duplicate = "scene_000000/sequence_000080\n"
    (root / "splits" / "val.txt").write_text(duplicate * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate sequence"):
        ColmapRGBDFSource(root, enforce_split_disjointness=False)
