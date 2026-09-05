"""Deterministic CPU materialization for COLMAP RGB/depth/rigid-flow clips."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from core.finetune.datasets.colmap_rgbdf_source import ColmapRGBDFSource, RGBDFClip
from core.finetune.preprocessing.depth import encode_metric_depth
from core.finetune.preprocessing.flow_encoding import encode_flow_hsv
from core.finetune.preprocessing.rigid_flow import compute_rigid_flow

EXPECTED_CLIPS = {"train": 93, "val": 8, "smoke": 2}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _flow_for_clip(source: ColmapRGBDFSource, clip: RGBDFClip) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    raw = source.read_clip(clip)
    flow, valid = compute_rigid_flow(
        raw["depth_mm"].astype(np.float32) / 1000.0,
        raw["intrinsics"],
        raw["extrinsics_w2c"],
    )
    return raw, flow, valid


def _fit_train_scale(source: ColmapRGBDFSource, clips: tuple[RGBDFClip, ...], *, percentile: float) -> tuple[float, int]:
    samples: list[np.ndarray] = []
    for clip in clips:
        _, flow, valid = _flow_for_clip(source, clip)
        magnitudes = np.linalg.norm(flow, axis=-1)[valid]
        if magnitudes.size:
            stride = max(1, magnitudes.size // 20_000)
            samples.append(magnitudes[::stride][:20_000])
    if not samples:
        raise ValueError("training rigid flow has no valid samples")
    values = np.concatenate(samples)
    if not np.isfinite(values).all():
        raise ValueError("training rigid flow has non-finite magnitudes")
    scale = float(np.percentile(values, percentile))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("training rigid flow magnitude scale is not positive")
    return scale, int(values.size)


def _photometric_residual(rgb: np.ndarray, flow: np.ndarray, valid: np.ndarray) -> float:
    _, height, width, _ = rgb.shape
    pixel_x, pixel_y = np.meshgrid(np.arange(width), np.arange(height))
    totals: list[np.ndarray] = []
    for target in range(1, len(rgb)):
        mask = valid[target]
        if not mask.any():
            continue
        target_x = np.rint(pixel_x + flow[target, ..., 0]).astype(np.int64).clip(0, width - 1)
        target_y = np.rint(pixel_y + flow[target, ..., 1]).astype(np.int64).clip(0, height - 1)
        source_values = rgb[target - 1][mask].astype(np.float32)
        target_values = rgb[target, target_y[mask], target_x[mask]].astype(np.float32)
        totals.append(np.abs(source_values - target_values).mean(axis=-1) / 255.0)
    return float(np.concatenate(totals).mean()) if totals else float("nan")


def materialize_rgbdf(
    source_root: str | Path,
    output_root: str | Path,
    *,
    expected_clips: dict[str, int] = EXPECTED_CLIPS,
    length: int = 65,
    stride: int = 8,
    flow_percentile: float = 99.0,
    min_valid_flow_ratio: float = 0.05,
    max_photometric_residual: float = 0.75,
    expected_source_size: tuple[int, int] = (640, 480),
) -> dict:
    source = ColmapRGBDFSource(source_root)
    if source.image_size != expected_source_size:
        raise ValueError("COLMAP RGB-DF source resolution does not match the declared contract")
    output = Path(output_root)
    clips = {split: source.clip_windows(split, length=length, stride=stride) for split in source.SPLITS}
    counts = {split: len(values) for split, values in clips.items()}
    if counts != expected_clips:
        raise ValueError(f"clip counts do not match the declared contract: {counts}")
    flow_scale, scale_samples = _fit_train_scale(source, clips["train"], percentile=flow_percentile)

    manifests: dict[str, list[dict[str, str]]] = {split: [] for split in source.SPLITS}
    reports: list[dict[str, object]] = []
    for split in source.SPLITS:
        for clip in clips[split]:
            raw, flow, flow_valid = _flow_for_clip(source, clip)
            depth_rgb, depth_valid = encode_metric_depth(raw["depth_mm"])
            flow_rgb = encode_flow_hsv(flow, flow_valid, magnitude_scale=flow_scale)
            valid_ratio = float(flow_valid[1:].mean())
            residual = _photometric_residual(raw["rgb"], flow, flow_valid)
            if not math_is_finite(residual) or valid_ratio < min_valid_flow_ratio or residual > max_photometric_residual:
                raise ValueError(f"RGB-DF quality gate failed for {clip.clip_id}")
            relative = Path("intermediate") / split / f"{clip.clip_id}.npz"
            final_path = output / relative
            final_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = final_path.with_name(f".{final_path.stem}.tmp.npz")
            np.savez_compressed(
                temporary,
                rgb=raw["rgb"],
                depth_rgb=depth_rgb,
                flow_rgb=flow_rgb,
                depth_valid=depth_valid,
                flow_valid=flow_valid,
            )
            os.replace(temporary, final_path)
            digest = _sha256(final_path)
            manifests[split].append({"clip_id": clip.clip_id, "intermediate": relative.as_posix(), "sha256": digest})
            reports.append(
                {
                    "clip_id": clip.clip_id,
                    "split": split,
                    "valid_flow_ratio": valid_ratio,
                    "photometric_residual": residual,
                }
            )

    for split, items in manifests.items():
        _atomic_json(
            output / f"intermediate_{split}.json",
            {
                "format": "rynnworld4d_colmap_rgbdf_intermediate_v1",
                "schema_version": 1,
                "split": split,
                "items": items,
            },
        )
    report = {
        "format": "rynnworld4d_colmap_rgbdf_validation_v1",
        "schema_version": 1,
        "allow_nan": False,
        "clip_counts": counts,
        "source_frame_counts": {split: len(source.split_frame_keys(split)) for split in source.SPLITS},
        "run_counts": {split: len(source.contiguous_runs(split)) for split in source.SPLITS},
        "flow_encoding": "hsv_white_zero_v1",
        "flow_percentile": flow_percentile,
        "flow_scale": flow_scale,
        "flow_scale_sample_count": scale_samples,
        "source_size": {"height": source.image_size[1], "width": source.image_size[0]},
        "training_canvas": {"height": source.image_size[1], "width": source.image_size[0]},
        "padding": "none_v1",
        "clips": reports,
    }
    _atomic_json(output / "reports" / "data_validation.json", report)
    return report


def math_is_finite(value: float) -> bool:
    return bool(np.isfinite(value))


def audit_source_metadata(source_root: str | Path, *, length: int = 65, stride: int = 8) -> dict:
    source = ColmapRGBDFSource(source_root, validate_frame_files=False)
    return {
        "clip_counts": {split: len(source.clip_windows(split, length=length, stride=stride)) for split in source.SPLITS},
        "source_frame_counts": {split: len(source.split_frame_keys(split)) for split in source.SPLITS},
        "run_counts": {split: len(source.contiguous_runs(split)) for split in source.SPLITS},
    }
