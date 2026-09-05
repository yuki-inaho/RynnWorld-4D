"""Validation and privacy-safe serialization for Hydra training configs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_training_config(config: DictConfig, *, require_artifacts: bool) -> None:
    OmegaConf.resolve(config)
    _require(config.data.format == "rynnworld4d_colmap_rgbdf_latents_v1", "data format is unsupported")
    _require(config.data.source_frames >= 2, "source frame count must be >= 2")
    _require(
        (int(config.data.source_height), int(config.data.source_width)) == (480, 640),
        "COLMAP source resolution must be 640x480",
    )
    _require(
        (int(config.data.height), int(config.data.width)) == (480, 640),
        "training resolution must be 640x480",
    )
    expected_latent_frames = (int(config.data.source_frames) - 1) // 4 + 1
    _require((int(config.data.source_frames) - 1) % 4 == 0, "source frame count must satisfy 4n+1")
    _require(config.data.latent_frames == expected_latent_frames, "source-to-latent frame count is inconsistent")
    _require(config.data.prompt == "" and config.data.allow_empty_prompt is True, "empty prompt must be explicit")
    _require(config.trainer.micro_batch_size >= 1, "micro batch size must be >= 1")
    _require(config.trainer.micro_batch_size <= 2, "32 GB Blackwell profile supports micro batch size at most 2")
    _require(config.trainer.gradient_accumulation_steps >= 1, "gradient accumulation must be >= 1")
    _require(config.trainer.epochs >= 1, "epochs must be >= 1")
    _require(config.checkpoint.k >= 1, "top-K checkpoint count must be >= 1")
    _require(config.checkpoint.monitor == "val/loss_total", "checkpoint monitor must be val/loss_total")
    _require(config.checkpoint.mode == "min", "checkpoint mode must be min")
    _require(config.checkpoint.save_last is False, "save_last must be false for top-K-only mode")
    _require(config.logging.name in {"tensorboard", "disabled"}, "logging backend is unsupported")
    _require(config.optimizer.name in {"amuse", "adamw"}, "optimizer is unsupported")
    _require(config.distributed.num_processes == 1, "managed RGB-DF training supports one GPU process")
    if config.optimizer.name == "amuse":
        _require(config.optimizer.scheduler is None, "AMUSE requires its internal warmup without an external scheduler")
        _require(config.distributed.custom_optimizer is True, "AMUSE requires a custom-optimizer distributed profile")
    else:
        _require(bool(config.optimizer.scheduler), "AdamW requires an external scheduler")

    deepspeed_path = Path(str(config.distributed.config_path))
    _require(deepspeed_path.is_file(), "DeepSpeed config is missing")
    try:
        deepspeed = json.loads(deepspeed_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ValueError("DeepSpeed config cannot be read") from None
    if config.optimizer.name == "amuse":
        _require("optimizer" not in deepspeed and "scheduler" not in deepspeed, "AMUSE DeepSpeed config cannot own optimizer or scheduler")
        _require(deepspeed.get("zero_allow_untested_optimizer") is True, "AMUSE DeepSpeed profile must explicitly allow its custom optimizer")
    for key in ("muon_lr", "aux_lr") if config.optimizer.name == "amuse" else ("learning_rate",):
        _require(math.isfinite(float(config.optimizer[key])) and float(config.optimizer[key]) > 0, f"{key} must be finite and positive")

    if require_artifacts:
        _require(config.optimizer.name == "amuse", "managed RGB-DF checkpoints currently require AMUSE")
        for field in ("source_root", "latent_root", "train_manifest", "val_manifest"):
            _require(Path(str(config.data[field])).exists(), f"required data artifact is missing: data.{field}")
        for field in ("base_path", "stage3_path"):
            _require(Path(str(config.model[field])).exists(), f"required model artifact is missing: model.{field}")
        if config.resume_from_checkpoint is not None:
            _require(
                Path(str(config.resume_from_checkpoint)).is_dir(),
                "resume_from_checkpoint must be an existing explicit checkpoint directory",
            )


def redacted_config(config: DictConfig) -> dict[str, Any]:
    value = OmegaConf.to_container(config, resolve=True)

    def redact(item: Any, key: str = "") -> Any:
        if isinstance(item, dict):
            return {name: redact(child, name) for name, child in item.items()}
        if isinstance(item, list):
            return [redact(child, key) for child in item]
        if isinstance(item, str) and (Path(item).is_absolute() or any(token in key.lower() for token in ("root", "path", "manifest", "dir"))):
            return "<redacted>"
        return item

    return redact(value)
