"""Wan VAE encoding for materialized RGB/depth/flow clips."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from safetensors.torch import save_file

LatentEncoder = Callable[[np.ndarray], torch.Tensor]


def _contained(root: Path, value: object) -> Path:
    if not isinstance(value, str):
        raise TypeError("intermediate artifact path must be relative")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("intermediate artifact path must be relative and contained")
    resolved_root = root.resolve()
    resolved = resolved_root.joinpath(*path.parts).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError("intermediate artifact path must be relative and contained")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_safetensors(tensors: dict[str, torch.Tensor], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    save_file(tensors, temporary)
    os.replace(temporary, path)
    return _sha256(path)


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def encode_intermediate_manifests(
    artifact_root: str | Path,
    *,
    encode_video: LatentEncoder,
    text_embedding: torch.Tensor,
    splits: Iterable[str] = ("train", "val", "smoke"),
    expected_shape: tuple[int, int, int, int] = (48, 17, 30, 40),
) -> dict[str, int]:
    root = Path(artifact_root)
    text = text_embedding.detach().cpu().float().contiguous()
    if text.ndim != 3 or text.shape[0] != 1 or not torch.isfinite(text).all():
        raise ValueError("empty-prompt embedding is invalid")
    text_relative = Path("embeddings") / "empty_prompt.safetensors"
    text_sha256 = _atomic_safetensors({"text_embeds": text}, root / text_relative)

    counts: dict[str, int] = {}
    for split in splits:
        manifest_path = root / f"intermediate_{split}.json"
        try:
            source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise ValueError(f"intermediate manifest cannot be read: {split}") from None
        if (
            source_manifest.get("format") != "rynnworld4d_colmap_rgbdf_intermediate_v1"
            or source_manifest.get("schema_version") != 1
            or source_manifest.get("split") != split
            or not isinstance(source_manifest.get("items"), list)
        ):
            raise ValueError(f"intermediate manifest schema is invalid: {split}")
        output_items: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in source_manifest["items"]:
            if not isinstance(item, dict) or set(item) != {"clip_id", "intermediate", "sha256"}:
                raise ValueError(f"intermediate manifest item is invalid: {split}")
            clip_id = item["clip_id"]
            if not isinstance(clip_id, str) or not clip_id.startswith("clip_") or clip_id in seen:
                raise ValueError(f"intermediate clip ID is invalid: {split}")
            seen.add(clip_id)
            intermediate_path = _contained(root, item["intermediate"])
            if _sha256(intermediate_path) != item["sha256"]:
                raise ValueError(f"intermediate artifact hash mismatch: {clip_id}")
            try:
                with np.load(intermediate_path) as arrays:
                    if set(arrays.files) != {"rgb", "depth_rgb", "flow_rgb", "depth_valid", "flow_valid"}:
                        raise ValueError
                    rgb = arrays["rgb"].copy()
                    depth_rgb = arrays["depth_rgb"].copy()
                    flow_rgb = arrays["flow_rgb"].copy()
            except (OSError, ValueError):
                raise ValueError(f"intermediate artifact is invalid: {clip_id}") from None
            video_latent = encode_video(rgb)
            depth_latent = encode_video(depth_rgb)
            flow_latent = encode_video(flow_rgb)
            latents = (video_latent, depth_latent, flow_latent)
            if any(tuple(value.shape) != expected_shape for value in latents):
                raise ValueError(f"encoded latent shape is invalid: {clip_id}")
            if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in latents):
                raise ValueError(f"encoded latent values are invalid: {clip_id}")

            rgb_relative = Path("latents") / split / f"{clip_id}_rgb.safetensors"
            flow_depth_relative = Path("latents") / split / f"{clip_id}_flow_depth.safetensors"
            rgb_sha256 = _atomic_safetensors(
                {"video_latents": video_latent.contiguous()},
                root / rgb_relative,
            )
            flow_depth_sha256 = _atomic_safetensors(
                {
                    "depth_latents": depth_latent.contiguous(),
                    "flow_latents": flow_latent.contiguous(),
                },
                root / flow_depth_relative,
            )
            output_items.append(
                {
                    "clip_id": clip_id,
                    "flow_depth_sha256": flow_depth_sha256,
                    "rgb_latents": rgb_relative.as_posix(),
                    "rgb_sha256": rgb_sha256,
                    "flow_depth_latents": flow_depth_relative.as_posix(),
                }
            )

        _atomic_json(
            {
                "format": "rynnworld4d_colmap_rgbdf_latents_v1",
                "schema_version": 1,
                "split": split,
                "latent_shape": list(expected_shape),
                "text_embeddings": text_relative.as_posix(),
                "text_embeddings_sha256": text_sha256,
                "items": output_items,
            },
            root / f"{split}.json",
        )
        counts[split] = len(output_items)
    return counts


class WanLatentEncoder:
    def __init__(self, model_path: str | Path, *, device: str = "cuda") -> None:
        from diffusers import AutoencoderKLWan
        from transformers import T5TokenizerFast, UMT5EncoderModel

        self.device = torch.device(device)
        self.model_path = str(model_path)
        tokenizer = T5TokenizerFast.from_pretrained(self.model_path, subfolder="tokenizer")
        text_encoder = UMT5EncoderModel.from_pretrained(
            self.model_path,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        inputs = tokenizer(
            [""],
            padding="max_length",
            max_length=226,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            hidden = text_encoder(
                inputs.input_ids.to(self.device),
                inputs.attention_mask.to(self.device),
            ).last_hidden_state
        sequence_length = int(inputs.attention_mask[0].sum())
        hidden = hidden[:, :sequence_length]
        padding = hidden.new_zeros((1, 226 - sequence_length, hidden.shape[-1]))
        self.text_embedding = torch.cat((hidden, padding), dim=1).detach().cpu().float()
        del text_encoder, tokenizer
        torch.cuda.empty_cache()

        self.vae = AutoencoderKLWan.from_pretrained(
            self.model_path,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.vae.enable_slicing()
        # diffusers 0.35.2 tiled_encode bypasses Wan's patchify step when
        # patch_size=2, so a 3-channel video reaches the 12-channel encoder
        # convolution and fails. The regular causal encoder performs patchify.
        self.vae.disable_tiling()
        self.vae.requires_grad_(False)

    def __call__(self, frames: np.ndarray) -> torch.Tensor:
        values = np.asarray(frames)
        if values.dtype != np.uint8 or values.ndim != 4:
            raise ValueError("Wan VAE input must be a uint8 frame array")
        if values.shape[-1] == 3:
            tensor = torch.from_numpy(values).permute(3, 0, 1, 2)
        elif values.shape[1] == 3:
            tensor = torch.from_numpy(values).permute(1, 0, 2, 3)
        else:
            raise ValueError("Wan VAE input must have three channels")
        if tuple(tensor.shape[-2:]) != (480, 640):
            raise ValueError("Wan VAE input must use the 640x480 training canvas")
        tensor = tensor.unsqueeze(0).to(self.device, dtype=self.vae.dtype)
        tensor = tensor.div(127.5).sub(1.0)
        with torch.no_grad():
            latent = self.vae.encode(tensor).latent_dist.mode()
            mean = torch.tensor(self.vae.config.latents_mean, device=self.device, dtype=latent.dtype).view(1, -1, 1, 1, 1)
            std = torch.tensor(self.vae.config.latents_std, device=self.device, dtype=latent.dtype).view(1, -1, 1, 1, 1)
            latent = (latent - mean) / std
        return latent.squeeze(0).detach().cpu().float().contiguous()
