"""Pure split-safe clip planning helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping


def contiguous_runs(
    frame_ids_by_scene: Mapping[int, set[int]],
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    runs: list[tuple[int, tuple[int, ...]]] = []
    for scene_index in sorted(frame_ids_by_scene):
        current: list[int] = []
        for frame_id in sorted(frame_ids_by_scene[scene_index]):
            if current and frame_id != current[-1] + 1:
                runs.append((scene_index, tuple(current)))
                current = []
            current.append(frame_id)
        if current:
            runs.append((scene_index, tuple(current)))
    return tuple(runs)


def build_clip_windows(
    runs: Iterable[tuple[int, tuple[int, ...]]],
    *,
    length: int,
    stride: int,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    if length < 2 or stride < 1:
        raise ValueError("clip length must be >= 2 and stride must be >= 1")
    return tuple(
        (scene_index, run[start : start + length])
        for scene_index, run in runs
        for start in range(0, len(run) - length + 1, stride)
    )


def anonymous_clip_id(split: str, scene_index: int, frame_ids: tuple[int, ...]) -> str:
    payload = (
        f"rynnworld4d_colmap_rgbdf_latents_v1:{split}:"
        f"{scene_index}:{frame_ids[0]}:{frame_ids[-1]}"
    )
    return f"clip_{hashlib.sha256(payload.encode()).hexdigest()[:20]}"
