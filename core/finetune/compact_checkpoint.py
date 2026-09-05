"""Compact, exact-resume checkpoints for single-rank ZeRO-3 AMUSE training."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from core.finetune.optim import (
    find_amuse_optimizer,
    find_zero3_optimizer,
    sync_zero3_master_to_model,
)

FORMAT = "rynnworld4d_compact_amuse_zero3_v1"
FILES = {"metadata.json", "rng_state.pt", "training_state.pt"}


@dataclass(frozen=True)
class CompactResumeState:
    global_step: int
    first_epoch: int
    metric: float


def _logical_shape(parameter: nn.Parameter) -> tuple[int, ...]:
    return tuple(int(value) for value in getattr(parameter, "ds_shape", parameter.shape))


def _model_structure(model: nn.Module) -> tuple[str, dict[int, str], set[str]]:
    module = getattr(model, "module", model)
    parameter_names: dict[int, str] = {}
    trainable_names: set[str] = set()
    description: list[dict[str, Any]] = []
    for name, parameter in module.named_parameters():
        parameter_names[id(parameter)] = name
        if parameter.requires_grad:
            trainable_names.add(name)
        description.append(
            {
                "dtype": str(parameter.dtype),
                "name": name,
                "requires_grad": bool(parameter.requires_grad),
                "shape": list(_logical_shape(parameter)),
            }
        )
    encoded = json.dumps(description, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), parameter_names, trainable_names


def _trainable_layout(
    zero3: object,
    parameter_names: dict[int, str],
    expected_names: set[str],
) -> list[list[dict[str, Any]]]:
    layout: list[list[dict[str, Any]]] = []
    seen: set[str] = set()
    for subgroup in zero3.fp16_groups:
        entries: list[dict[str, Any]] = []
        for parameter in subgroup:
            name = parameter_names.get(id(parameter))
            if name is None or name in seen or not parameter.requires_grad:
                raise RuntimeError("ZeRO-3 trainable parameter layout is inconsistent")
            seen.add(name)
            entries.append(
                {
                    "dtype": str(parameter.dtype),
                    "name": name,
                    "numel": int(getattr(parameter, "ds_numel", parameter.numel())),
                    "shape": list(_logical_shape(parameter)),
                }
            )
        layout.append(entries)
    if seen != expected_names:
        raise RuntimeError("ZeRO-3 trainable parameter layout is incomplete")
    return layout


def _rng_state(generator: torch.Generator | None) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
        "generator": generator.get_state() if generator is not None else None,
    }


def _restore_rng(state: dict[str, Any], generator: torch.Generator | None) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda", "generator"}
    if set(state) != required:
        raise ValueError("compact checkpoint RNG state is invalid")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    if generator is not None:
        if state["generator"] is None:
            raise ValueError("compact checkpoint generator state is missing")
        generator.set_state(state["generator"])


def save_compact_amuse_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: object,
    epoch: int,
    global_step: int,
    metric: float,
    config_fingerprint: str,
    grouping_fingerprint: str,
    pretrained_fingerprint: str,
    data_fingerprints: dict[str, str],
    generator: torch.Generator | None,
) -> None:
    path = Path(path)
    if any(path.iterdir()):
        raise ValueError("compact checkpoint target must be empty")
    if epoch < 0 or global_step < 1 or not math.isfinite(float(metric)):
        raise ValueError("compact checkpoint progress metadata is invalid")
    raw = find_amuse_optimizer(optimizer)
    zero3 = find_zero3_optimizer(optimizer)
    if zero3 is None or int(zero3.partition_count) != 1:
        raise RuntimeError("compact AMUSE checkpoints require single-rank ZeRO-3")
    if raw.train_mode:
        raise RuntimeError("compact AMUSE checkpoints must be saved with evaluation x weights")
    fingerprints = [config_fingerprint, grouping_fingerprint, pretrained_fingerprint, *data_fingerprints.values()]
    if (
        any(len(value) != 64 for value in fingerprints)
        or set(data_fingerprints) != {"train", "val"}
    ):
        raise ValueError("compact checkpoint fingerprints are invalid")

    model_fingerprint, parameter_names, trainable_names = _model_structure(model)
    layout = _trainable_layout(zero3, parameter_names, trainable_names)
    metadata = {
        "data_fingerprints": data_fingerprints,
        "config_fingerprint": config_fingerprint,
        "epoch": epoch,
        "files": sorted(FILES),
        "format": FORMAT,
        "global_step": global_step,
        "grouping_fingerprint": grouping_fingerprint,
        "metric": float(metric),
        "model_structure_fingerprint": model_fingerprint,
        "next_epoch": epoch + 1,
        "pretrained_fingerprint": pretrained_fingerprint,
        "trainable_layout": layout,
        "world_size": 1,
    }
    torch.save({"format": FORMAT, "zero3_optimizer": zero3.state_dict()}, path / "training_state.pt")
    torch.save(_rng_state(generator), path / "rng_state.pt")
    (path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_compact_amuse_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: object,
    config_fingerprint: str,
    grouping_fingerprint: str,
    pretrained_fingerprint: str,
    data_fingerprints: dict[str, str],
    generator: torch.Generator | None,
) -> CompactResumeState:
    path = Path(path)
    if not path.is_dir() or {item.name for item in path.iterdir()} != FILES:
        raise ValueError("compact checkpoint files are missing or unexpected")
    try:
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ValueError("compact checkpoint metadata cannot be read") from None
    required = {
        "data_fingerprints",
        "config_fingerprint",
        "epoch",
        "files",
        "format",
        "global_step",
        "grouping_fingerprint",
        "metric",
        "model_structure_fingerprint",
        "next_epoch",
        "pretrained_fingerprint",
        "trainable_layout",
        "world_size",
    }
    if set(metadata) != required or metadata.get("format") != FORMAT or metadata.get("world_size") != 1:
        raise ValueError("compact checkpoint metadata schema is invalid")
    if metadata["files"] != sorted(FILES):
        raise ValueError("compact checkpoint artifact declaration is invalid")
    if metadata["grouping_fingerprint"] != grouping_fingerprint:
        raise ValueError("compact checkpoint AMUSE grouping fingerprint does not match")
    if metadata["config_fingerprint"] != config_fingerprint:
        raise ValueError("compact checkpoint training config fingerprint does not match")
    if metadata["pretrained_fingerprint"] != pretrained_fingerprint:
        raise ValueError("compact checkpoint pretrained model fingerprint does not match")
    if metadata["data_fingerprints"] != data_fingerprints:
        raise ValueError("compact checkpoint data fingerprints do not match")
    model_fingerprint, parameter_names, trainable_names = _model_structure(model)
    if metadata["model_structure_fingerprint"] != model_fingerprint:
        raise ValueError("compact checkpoint model structure fingerprint does not match")

    zero3 = find_zero3_optimizer(optimizer)
    raw = find_amuse_optimizer(optimizer)
    if zero3 is None or int(zero3.partition_count) != 1:
        raise RuntimeError("compact AMUSE checkpoints require single-rank ZeRO-3")
    if metadata["trainable_layout"] != _trainable_layout(zero3, parameter_names, trainable_names):
        raise ValueError("compact checkpoint trainable parameter layout does not match")
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("format") != FORMAT or set(state) != {"format", "zero3_optimizer"}:
        raise ValueError("compact checkpoint training state is invalid")
    raw.train_mode = False
    zero3.load_state_dict([state["zero3_optimizer"]], load_optimizer_states=True)
    sync_zero3_master_to_model(zero3)
    rng = torch.load(path / "rng_state.pt", map_location="cpu", weights_only=False)
    if not isinstance(rng, dict):
        raise TypeError("compact checkpoint RNG state is invalid")
    _restore_rng(rng, generator)

    epoch = metadata["epoch"]
    next_epoch = metadata["next_epoch"]
    global_step = metadata["global_step"]
    metric = float(metadata["metric"])
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 0
        or next_epoch != epoch + 1
        or isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < 1
        or not math.isfinite(metric)
    ):
        raise ValueError("compact checkpoint progress metadata is invalid")
    return CompactResumeState(global_step=global_step, first_epoch=next_epoch, metric=metric)
