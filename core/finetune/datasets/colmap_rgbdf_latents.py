"""Fail-fast loader for materialized COLMAP RGB/depth/flow latents."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import torch
from safetensors import SafetensorError
from safetensors.torch import load_file
from torch.utils.data import Dataset


class ColmapRGBDFLatentDataset(Dataset):
    FORMAT = "rynnworld4d_colmap_rgbdf_latents_v1"

    def __init__(self, manifest_path: str | Path, *, expected_split: str) -> None:
        self._manifest_path = Path(manifest_path)
        self._root = self._manifest_path.parent
        try:
            manifest_bytes = self._manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
        except (OSError, json.JSONDecodeError):
            raise ValueError("RGB-DF latent manifest cannot be read") from None
        self.fingerprint = hashlib.sha256(manifest_bytes).hexdigest()

        if not isinstance(manifest, dict):
            raise TypeError("RGB-DF latent manifest must be an object")
        if manifest.get("format") != self.FORMAT or manifest.get("schema_version") != 1:
            raise ValueError("RGB-DF latent manifest schema is unsupported")
        if expected_split not in {"train", "val", "smoke"}:
            raise ValueError("expected split is unsupported")
        if manifest.get("split") != expected_split:
            raise ValueError("RGB-DF latent manifest split does not match")

        shape = manifest.get("latent_shape")
        if not (
            isinstance(shape, list)
            and len(shape) == 4
            and all(isinstance(value, int) and value > 0 for value in shape)
        ):
            raise ValueError("RGB-DF latent manifest shape is invalid")
        self._expected_shape = tuple(shape)
        text_path = self._resolve_relative(manifest.get("text_embeddings"))
        if not text_path.is_file():
            raise ValueError("RGB-DF text embedding artifact is missing")
        if self._sha256(text_path) != self._required_hash(manifest.get("text_embeddings_sha256")):
            raise ValueError("RGB-DF text embedding artifact hash does not match")
        try:
            text_values = load_file(text_path, device="cpu")
        except (OSError, SafetensorError):
            raise ValueError("RGB-DF text embedding artifact cannot be read") from None
        if set(text_values) != {"text_embeds"}:
            raise ValueError("RGB-DF text embedding keys are invalid")
        self._text_embedding = text_values["text_embeds"]
        if (
            self._text_embedding.dtype != torch.float32
            or self._text_embedding.ndim != 3
            or self._text_embedding.shape[0] != 1
            or not torch.isfinite(self._text_embedding).all()
        ):
            raise ValueError("RGB-DF text embedding is invalid")

        items = manifest.get("items")
        if not isinstance(items, list) or not items:
            raise ValueError("RGB-DF latent manifest has no items")
        self._items: list[tuple[str, Path, Path]] = []
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, dict) or set(item) != {
                "clip_id",
                "flow_depth_sha256",
                "rgb_latents",
                "rgb_sha256",
                "flow_depth_latents",
            }:
                raise ValueError("RGB-DF latent manifest item fields are invalid")
            clip_id = item["clip_id"]
            if not isinstance(clip_id, str) or not clip_id.startswith("clip_") or clip_id in seen:
                raise ValueError("RGB-DF latent manifest clip ID is invalid")
            seen.add(clip_id)
            rgb_path = self._resolve_relative(item["rgb_latents"])
            flow_depth_path = self._resolve_relative(item["flow_depth_latents"])
            if not rgb_path.is_file() or not flow_depth_path.is_file():
                raise ValueError("RGB-DF latent artifact is missing")
            if self._sha256(rgb_path) != self._required_hash(item["rgb_sha256"]):
                raise ValueError(f"RGB latent artifact hash for {clip_id} does not match")
            if self._sha256(flow_depth_path) != self._required_hash(item["flow_depth_sha256"]):
                raise ValueError(f"depth/flow latent artifact hash for {clip_id} does not match")
            self._items.append((clip_id, rgb_path, flow_depth_path))

    @staticmethod
    def _required_hash(value: Any) -> str:
        if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("RGB-DF latent artifact hash is invalid")
        return value

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def _resolve_relative(self, value: Any) -> Path:
        if not isinstance(value, str):
            raise TypeError("RGB-DF latent artifact path must be relative")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("RGB-DF latent artifact path must be relative and contained")
        root = self._root.resolve()
        resolved = root.joinpath(*path.parts).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("RGB-DF latent artifact path must be relative and contained")
        return resolved

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        clip_id, rgb_path, flow_depth_path = self._items[index]
        try:
            rgb = load_file(rgb_path, device="cpu")
            flow_depth = load_file(flow_depth_path, device="cpu")
        except (OSError, SafetensorError):
            raise ValueError(f"latent artifact for {clip_id} cannot be read") from None
        if set(rgb) != {"video_latents"}:
            raise ValueError(f"RGB latent keys for {clip_id} are invalid")
        if set(flow_depth) != {"depth_latents", "flow_latents"}:
            raise ValueError(f"depth/flow latent keys for {clip_id} are invalid")

        video = rgb["video_latents"]
        depth = flow_depth["depth_latents"]
        flow = flow_depth["flow_latents"]
        text = self._text_embedding
        if tuple(video.shape) != self._expected_shape or depth.shape != video.shape or flow.shape != video.shape:
            raise ValueError(f"latent shape for {clip_id} is invalid")
        if video.dtype != torch.float32 or depth.dtype != torch.float32 or flow.dtype != torch.float32:
            raise ValueError(f"latent dtype for {clip_id} is invalid")
        if not all(torch.isfinite(value).all() for value in (video, depth, flow, text)):
            raise ValueError(f"latent values for {clip_id} must be finite")
        text = text.squeeze(0)
        return {
            "sample_id": clip_id,
            "encoded_video": video,
            "encoded_depth": depth,
            "encoded_flow": flow,
            "img_latent": video[:, :1],
            "depth_latent": depth[:, :1],
            "flow_latent": flow[:, :1],
            "null_embedding": text,
            "text_embedding": text,
        }
