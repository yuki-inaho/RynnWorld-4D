"""Validated access to the private ``colmap_rgbd_v1`` source format.

Errors deliberately describe the violated contract without embedding the
private dataset root, scene names, or source frame paths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from core.finetune.preprocessing.clips import (
    anonymous_clip_id,
    build_clip_windows,
    contiguous_runs,
)


@dataclass(frozen=True)
class RGBDFClip:
    clip_id: str
    split: str
    scene_index: int
    frame_ids: tuple[int, ...]


@dataclass(frozen=True)
class _Scene:
    directory: Path
    frame_ids: np.ndarray
    intrinsics: np.ndarray
    extrinsics_w2c: np.ndarray
    quality_flags: np.ndarray
    sequences: np.ndarray
    lengths: np.ndarray
    row_by_frame_id: dict[int, int]


class ColmapRGBDFSource:
    """Fail-fast reader and split/clip planner for ``colmap_rgbd_v1``."""

    SPLITS = ("train", "val", "smoke")

    def __init__(
        self,
        root: str | Path,
        *,
        enforce_split_disjointness: bool = True,
        validate_frame_files: bool = True,
    ) -> None:
        self._root = Path(root).expanduser()
        self._validate_frame_files = validate_frame_files
        self._metadata = self._load_metadata()
        self._scene_names, self._scenes = self._load_scenes()
        self._split_sequences = {split: self._load_split(split) for split in self.SPLITS}
        self._validate_counts()
        if enforce_split_disjointness:
            self._validate_split_disjointness()

    @staticmethod
    def _error(reason: str) -> ValueError:
        return ValueError(f"COLMAP RGB-DF source is invalid: {reason}")

    def _load_metadata(self) -> dict:
        if not self._root.is_dir():
            raise self._error("dataset root is missing")
        try:
            with (self._root / "dataset.json").open(encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, json.JSONDecodeError):
            raise self._error("dataset metadata cannot be read") from None

        if not isinstance(metadata, dict):
            raise self._error("dataset metadata must be an object")
        if metadata.get("format") != "colmap_rgbd_v1" or metadata.get("schema_version") != 1:
            raise self._error("schema is unsupported")

        image = metadata.get("image", {})
        depth = metadata.get("depth", {})
        camera = metadata.get("camera", {})
        expected = (
            image.get("channels") == 3
            and image.get("dtype") == "uint8"
            and depth.get("dtype") == "uint16"
            and depth.get("unit") == "millimeters"
            and depth.get("invalid_value") == 0
            and depth.get("max_depth_mm") == 1300
            and camera.get("extrinsics") == "opencv_world_to_camera"
            and camera.get("intrinsics") == "pixel_units"
        )
        if not expected:
            raise self._error("encoding or camera convention is unsupported")
        try:
            width, height = int(image["width"]), int(image["height"])
        except (KeyError, TypeError, ValueError):
            raise self._error("image dimensions are invalid") from None
        if (
            width <= 0
            or height <= 0
            or depth.get("width") != width
            or depth.get("height") != height
        ):
            raise self._error("RGB and depth dimensions do not match")
        self.image_size = (width, height)
        return metadata

    def _load_scenes(self) -> tuple[dict[str, int], list[_Scene]]:
        scenes_root = self._root / "scenes"
        if not scenes_root.is_dir():
            raise self._error("scene directory is missing")
        directories = sorted(path for path in scenes_root.iterdir() if path.is_dir())
        if not directories:
            raise self._error("no scenes are available")

        names: dict[str, int] = {}
        scenes: list[_Scene] = []
        for scene_index, directory in enumerate(directories):
            names[directory.name] = scene_index
            scenes.append(self._load_scene(directory))
        return names, scenes

    def _load_scene(self, directory: Path) -> _Scene:
        try:
            with np.load(directory / "cameras.npz") as values:
                frame_ids = np.asarray(values["frame_ids"], dtype=np.int64)
                intrinsics = np.asarray(values["intrinsics"], dtype=np.float32)
                extrinsics = np.asarray(values["extrinsics_w2c"], dtype=np.float32)
                quality = np.asarray(
                    values.get("quality_flags", np.ones(frame_ids.shape, dtype=bool)),
                    dtype=bool,
                )
            with np.load(directory / "sequences.npz") as values:
                sequences = np.asarray(values["sequences"], dtype=np.int64)
                lengths = np.asarray(values["lengths"], dtype=np.int64)
        except (OSError, KeyError, ValueError):
            raise self._error("camera or sequence metadata is invalid") from None

        count = frame_ids.size
        if frame_ids.ndim != 1 or count < 2 or np.unique(frame_ids).size != count:
            raise self._error("camera frame IDs are invalid")
        if intrinsics.shape != (count, 3, 3) or extrinsics.shape != (count, 3, 4):
            raise self._error("camera metadata has incompatible shapes")
        if quality.shape != (count,) or not np.isfinite(intrinsics).all() or not np.isfinite(extrinsics).all():
            raise self._error("camera metadata must be finite")
        if np.any(intrinsics[:, 0, 0] <= 0) or np.any(intrinsics[:, 1, 1] <= 0):
            raise self._error("camera focal lengths must be positive")
        if sequences.ndim != 2 or lengths.shape != (sequences.shape[0],):
            raise self._error("sequence metadata has incompatible shapes")
        if np.any(lengths < 2) or np.any(lengths > sequences.shape[1]):
            raise self._error("sequence lengths are invalid")

        known = set(frame_ids.tolist())
        for row, length in zip(sequences, lengths, strict=True):
            selected = row[: int(length)]
            if (
                np.any(selected < 0)
                or np.unique(selected).size != selected.size
                or not set(selected.tolist()) <= known
            ):
                raise self._error("sequence references an unknown frame")

        if self._validate_frame_files:
            self._validate_scene_frames(directory, frame_ids)

        return _Scene(
            directory=directory,
            frame_ids=frame_ids,
            intrinsics=intrinsics,
            extrinsics_w2c=extrinsics,
            quality_flags=quality,
            sequences=sequences,
            lengths=lengths,
            row_by_frame_id={int(value): index for index, value in enumerate(frame_ids)},
        )

    def _validate_scene_frames(self, directory: Path, frame_ids: np.ndarray) -> None:
        width, height = self.image_size
        for frame_id in frame_ids:
            rgb_path = directory / "rgb" / f"frame_{int(frame_id):06d}.png"
            depth_path = directory / "depth" / f"frame_{int(frame_id):06d}.png"
            if not rgb_path.is_file() or not depth_path.is_file():
                raise self._error("RGB or depth frame is missing")
            try:
                with Image.open(rgb_path) as image:
                    rgb_valid = image.size == (width, height) and image.mode == "RGB"
                with Image.open(depth_path) as image:
                    depth_array = np.asarray(image)
            except OSError:
                raise self._error("RGB or depth frame cannot be read") from None
            if not rgb_valid:
                raise self._error("RGB frame encoding is invalid")
            if depth_array.shape != (height, width) or depth_array.dtype != np.uint16:
                raise self._error("depth frame encoding is invalid")

    def _load_split(self, split: str) -> tuple[tuple[int, tuple[int, ...]], ...]:
        try:
            lines = (self._root / "splits" / f"{split}.txt").read_text(encoding="utf-8").splitlines()
        except OSError:
            raise self._error("split definition cannot be read") from None
        selected: list[tuple[int, tuple[int, ...]]] = []
        seen_entries: set[tuple[int, int]] = set()
        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split("/")
            if len(parts) != 2 or not parts[1].startswith("sequence_"):
                raise self._error("split entry has an unsupported format")
            scene_index = self._scene_names.get(parts[0])
            try:
                sequence_index = int(parts[1].removeprefix("sequence_"))
            except ValueError:
                raise self._error("split sequence identifier is invalid") from None
            if scene_index is None:
                raise self._error("split references an unknown scene")
            scene = self._scenes[scene_index]
            if not 0 <= sequence_index < scene.sequences.shape[0]:
                raise self._error("split sequence identifier is out of range")
            entry = (scene_index, sequence_index)
            if entry in seen_entries:
                raise self._error("split contains a duplicate sequence")
            seen_entries.add(entry)
            frame_ids = tuple(
                int(value)
                for value in scene.sequences[sequence_index, : scene.lengths[sequence_index]]
            )
            rows = [scene.row_by_frame_id[value] for value in frame_ids]
            if np.all(scene.quality_flags[rows]):
                selected.append((scene_index, frame_ids))
        if not selected:
            raise self._error("split has no usable sequences")
        return tuple(selected)

    def _validate_counts(self) -> None:
        if self._metadata.get("scene_count") != len(self._scenes):
            raise self._error("scene count does not match metadata")
        total_frames = sum(len(scene.frame_ids) for scene in self._scenes)
        if self._metadata.get("frame_count") != total_frames:
            raise self._error("frame count does not match metadata")
        declared = self._metadata.get("splits")
        if declared is not None:
            for split in self.SPLITS:
                if declared.get(split) != len(self._split_sequences[split]):
                    raise self._error("split count does not match metadata")

    def split_frame_keys(self, split: str) -> frozenset[tuple[int, int]]:
        if split not in self.SPLITS:
            raise self._error("unsupported split")
        return frozenset(
            (scene_index, frame_id)
            for scene_index, frame_ids in self._split_sequences[split]
            for frame_id in frame_ids
        )

    def _validate_split_disjointness(self) -> None:
        keys = {split: self.split_frame_keys(split) for split in self.SPLITS}
        for index, left in enumerate(self.SPLITS):
            for right in self.SPLITS[index + 1 :]:
                if keys[left] & keys[right]:
                    raise self._error("split frame leakage detected")

    def contiguous_runs(self, split: str) -> tuple[tuple[int, tuple[int, ...]], ...]:
        by_scene: dict[int, set[int]] = {}
        for scene_index, frame_id in self.split_frame_keys(split):
            by_scene.setdefault(scene_index, set()).add(frame_id)
        return contiguous_runs(by_scene)

    def clip_windows(self, split: str, *, length: int = 65, stride: int = 8) -> tuple[RGBDFClip, ...]:
        if length < 2 or stride < 1:
            raise self._error("clip length or stride is invalid")
        return tuple(
            RGBDFClip(
                anonymous_clip_id(split, scene_index, frame_ids),
                split,
                scene_index,
                frame_ids,
            )
            for scene_index, frame_ids in build_clip_windows(
                self.contiguous_runs(split), length=length, stride=stride
            )
        )

    def read_clip(self, clip: RGBDFClip) -> dict[str, np.ndarray]:
        scene = self._scenes[clip.scene_index]
        rows = [scene.row_by_frame_id[value] for value in clip.frame_ids]
        rgb_frames = []
        depth_frames = []
        for frame_id in clip.frame_ids:
            with Image.open(scene.directory / "rgb" / f"frame_{frame_id:06d}.png") as image:
                rgb_frames.append(np.asarray(image, dtype=np.uint8).copy())
            with Image.open(scene.directory / "depth" / f"frame_{frame_id:06d}.png") as image:
                depth_frames.append(np.asarray(image, dtype=np.uint16).copy())
        return {
            "rgb": np.stack(rgb_frames),
            "depth_mm": np.stack(depth_frames),
            "intrinsics": scene.intrinsics[rows].copy(),
            "extrinsics_w2c": scene.extrinsics_w2c[rows].copy(),
        }
