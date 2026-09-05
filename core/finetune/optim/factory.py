"""Fail-closed AMUSE grouping and construction."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from torch import nn

from core.finetune.optim.amuse import AMUSE
from core.finetune.optim.zero3_amuse import Zero3CompatibleAMUSE

DEFAULT_FALLBACK_PATTERNS = (
    "*.patch_embedding.weight",
    "*.proj_out.weight",
    "*joint_out*.weight",
)


@dataclass(frozen=True)
class AmuseParameterGrouping:
    muon_names: tuple[str, ...]
    fallback_names: tuple[str, ...]
    frozen_names: tuple[str, ...]
    fingerprint: str
    muon_numel: int
    fallback_numel: int


@dataclass(frozen=True)
class AmuseBuildResult:
    optimizer: AMUSE
    grouping: AmuseParameterGrouping
    warmup_steps: int


class AmuseInternalScheduler:
    """Scheduler-shaped adapter; AMUSE updates group learning rates itself."""

    def __init__(self, optimizer: AMUSE) -> None:
        self.optimizer = optimizer

    def step(self) -> None:
        return None

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, str]:
        return {"type": "amuse_internal"}

    def load_state_dict(self, state_dict: dict[str, str]) -> None:
        if state_dict != {"type": "amuse_internal"}:
            raise ValueError("AMUSE internal scheduler state is invalid")


def _logical_shape(parameter: nn.Parameter) -> tuple[int, ...]:
    shape = getattr(parameter, "ds_shape", parameter.shape)
    return tuple(int(value) for value in shape)


def _logical_numel(parameter: nn.Parameter) -> int:
    return int(getattr(parameter, "ds_numel", parameter.numel()))


def _matches_any(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def classify_amuse_parameters(
    model: nn.Module,
    *,
    fallback_patterns: Sequence[str] = DEFAULT_FALLBACK_PATTERNS,
) -> AmuseParameterGrouping:
    patterns = tuple(str(pattern) for pattern in fallback_patterns)
    if any(not pattern for pattern in patterns):
        raise ValueError("AMUSE fallback patterns must be non-empty strings")

    named_parameters = dict(model.named_parameters())
    named_modules = dict(model.named_modules())
    muon_names: list[str] = []
    fallback_names: list[str] = []
    frozen_names: list[str] = []
    matrix_modules = (nn.Linear, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)
    for name, parameter in sorted(named_parameters.items()):
        if not parameter.requires_grad:
            frozen_names.append(name)
            continue
        module_name, separator, attribute = name.rpartition(".")
        owner = named_modules.get(module_name) if separator else None
        shape = _logical_shape(parameter)
        use_muon = (
            attribute == "weight"
            and isinstance(owner, matrix_modules)
            and len(shape) in {2, 4, 5}
            and not _matches_any(name, patterns)
        )
        (muon_names if use_muon else fallback_names).append(name)

    trainable_names = {name for name, parameter in named_parameters.items() if parameter.requires_grad}
    grouped_names = set(muon_names) | set(fallback_names)
    if grouped_names != trainable_names or set(muon_names) & set(fallback_names):
        raise ValueError("AMUSE parameter partition must be complete and disjoint")
    if not muon_names or not fallback_names:
        raise ValueError("AMUSE requires non-empty Muon and auxiliary parameter groups")

    ordered_muon = tuple(sorted(muon_names, key=lambda name: _logical_shape(named_parameters[name]), reverse=True))
    ordered_fallback = tuple(fallback_names)

    def describe(name: str) -> dict[str, object]:
        parameter = named_parameters[name]
        return {"dtype": str(parameter.dtype), "name": name, "shape": list(_logical_shape(parameter))}

    encoded = json.dumps(
        {
            "format_version": 1,
            "fallback": [describe(name) for name in ordered_fallback],
            "muon": [describe(name) for name in ordered_muon],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return AmuseParameterGrouping(
        muon_names=ordered_muon,
        fallback_names=ordered_fallback,
        frozen_names=tuple(frozen_names),
        fingerprint=hashlib.sha256(encoded).hexdigest(),
        muon_numel=sum(_logical_numel(named_parameters[name]) for name in ordered_muon),
        fallback_numel=sum(_logical_numel(named_parameters[name]) for name in ordered_fallback),
    )


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _tier(name: str) -> str:
    if "joint_out" in name:
        return "joint_out"
    if "joint_" in name or "modality_embed" in name:
        return "joint_other"
    return "base"


def build_amuse_optimizer(
    model: nn.Module,
    *,
    total_optimizer_steps: int,
    muon_lr: float,
    aux_lr: float,
    joint_out_multiplier: float = 1.0,
    joint_other_multiplier: float = 1.0,
    beta1: float = 0.4,
    beta2: float = 0.999,
    eps: float = 1.0e-10,
    momentum: float = 0.95,
    rho: float = 0.3,
    r: float = 0.0,
    weight_lr_power: float = 2.0,
    warmup_ratio: float = 0.05,
    min_warmup_steps: int = 2,
    weight_decay: float = 0.01,
    weight_decay_at_y: float = 0.0,
    fallback_patterns: Sequence[str] = DEFAULT_FALLBACK_PATTERNS,
) -> AmuseBuildResult:
    if isinstance(total_optimizer_steps, bool) or not isinstance(total_optimizer_steps, int) or total_optimizer_steps < 1:
        raise ValueError("total_optimizer_steps must be a positive integer")
    if isinstance(min_warmup_steps, bool) or not isinstance(min_warmup_steps, int) or min_warmup_steps < 2:
        raise ValueError("min_warmup_steps must be an integer >= 2")
    values = {
        "muon_lr": muon_lr,
        "aux_lr": aux_lr,
        "joint_out_multiplier": joint_out_multiplier,
        "joint_other_multiplier": joint_other_multiplier,
        "beta1": beta1,
        "beta2": beta2,
        "eps": eps,
        "momentum": momentum,
        "rho": rho,
        "r": r,
        "weight_lr_power": weight_lr_power,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "weight_decay_at_y": weight_decay_at_y,
    }
    values = {name: _finite(name, value) for name, value in values.items()}
    if values["muon_lr"] <= 0 or values["aux_lr"] <= 0:
        raise ValueError("AMUSE learning rates must be > 0")
    if values["joint_out_multiplier"] <= 0 or values["joint_other_multiplier"] <= 0:
        raise ValueError("AMUSE learning-rate multipliers must be > 0")
    if not 0 < values["beta1"] < 1 or not 0 <= values["beta2"] < 1:
        raise ValueError("AMUSE beta values are invalid")
    if values["eps"] <= 0 or not 0 <= values["momentum"] < 1:
        raise ValueError("AMUSE epsilon or momentum is invalid")
    if not 0 <= values["rho"] <= 1 or not 0 < values["warmup_ratio"] <= 1:
        raise ValueError("AMUSE rho or warmup ratio is invalid")
    if values["weight_decay"] < 0 or values["weight_decay_at_y"] < 0:
        raise ValueError("AMUSE weight decay values must be >= 0")

    grouping = classify_amuse_parameters(model, fallback_patterns=fallback_patterns)
    named = dict(model.named_parameters())
    warmup_steps = max(min_warmup_steps, math.ceil(total_optimizer_steps * values["warmup_ratio"]))
    names_by_kind = {"muon": grouping.muon_names, "aux": grouping.fallback_names}
    multipliers = {
        "base": 1.0,
        "joint_other": values["joint_other_multiplier"],
        "joint_out": values["joint_out_multiplier"],
    }
    groups: list[dict[str, Any]] = []
    for kind, names in names_by_kind.items():
        for tier, multiplier in multipliers.items():
            selected = [name for name in names if _tier(name) == tier]
            if not selected:
                continue
            chunks = [[name] for name in selected] if kind == "muon" else [selected]
            for chunk_index, chunk in enumerate(chunks):
                group: dict[str, Any] = {
                    "name": f"amuse_{kind}_{tier}_{chunk_index:04d}",
                    "params": [named[name] for name in chunk],
                    "param_names": chunk,
                    "use_muon": kind == "muon",
                    "lr": values[f"{kind}_lr"] * multiplier,
                    "weight_decay": values["weight_decay"],
                }
                if kind == "muon":
                    group.update(
                        momentum=values["momentum"],
                        aux_update_type="adamw",
                        logical_shape=_logical_shape(named[chunk[0]]),
                    )
                else:
                    group.update(update_type="adamw", beta2=values["beta2"], eps=values["eps"])
                groups.append(group)
    optimizer = Zero3CompatibleAMUSE(
        groups,
        weight_decay_at_y=values["weight_decay_at_y"],
        beta1=values["beta1"],
        weight_lr_power=values["weight_lr_power"],
        warmup_steps=warmup_steps,
        rho=values["rho"],
        r=values["r"],
    )
    return AmuseBuildResult(optimizer=optimizer, grouping=grouping, warmup_steps=warmup_steps)


def find_amuse_optimizer(optimizer: object) -> AMUSE:
    current = optimizer
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        if isinstance(current, AMUSE):
            return current
        seen.add(id(current))
        current = getattr(current, "optimizer", None)
    raise RuntimeError("AMUSE optimizer is not reachable through the wrapper chain")


def find_zero3_optimizer(optimizer: object) -> object | None:
    current = optimizer
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        if hasattr(current, "fp32_partitioned_groups_flat") and hasattr(current, "sub_group_to_group_id"):
            return current
        seen.add(id(current))
        current = getattr(current, "optimizer", None)
    return None


def attach_zero3_amuse(optimizer: object) -> AMUSE:
    raw = find_amuse_optimizer(optimizer)
    zero3 = find_zero3_optimizer(optimizer)
    if zero3 is None:
        return raw
    raw._zero3_partition_owner = zero3
    allowed = set(zero3.fp32_partitioned_groups_flat)
    for parameter in list(raw.state):
        if parameter not in allowed:
            del raw.state[parameter]
    return raw


def sync_zero3_master_to_model(zero3: object) -> None:
    """Copy AMUSE's x/y FP32 partitions into the model-visible partitions."""

    if int(zero3.partition_count) != 1:
        raise RuntimeError("AMUSE mode switching currently requires one ZeRO-3 data-parallel rank")
    fp32_partitions = zero3.fp32_partitioned_groups_flat
    fp16_partitions = zero3.fp16_partitioned_groups_flat
    if len(fp32_partitions) != len(fp16_partitions):
        raise RuntimeError("ZeRO-3 FP32/model partition counts do not match")
    for subgroup_id, (fp32_partition, model_partition) in enumerate(
        zip(fp32_partitions, fp16_partitions, strict=True)
    ):
        if model_partition is None:
            raise RuntimeError("AMUSE mode switching does not support swapped ZeRO-3 model partitions")
        model_partition.data.copy_(fp32_partition.data)
        zero3._unflatten_partitioned_parameters(subgroup_id)

    # Persistent parameters stay gathered between forwards. Their visible
    # tensors do not alias the partition buffer after DeepSpeed's post-step
    # gather, so refresh them explicitly on the supported single-rank path.
    for parameter in zero3.persistent_parameters:
        if parameter.numel() == parameter.ds_numel:
            parameter.data.copy_(parameter.ds_tensor.data.view(parameter.ds_shape))


def set_amuse_mode(optimizer: object, *, training: bool) -> AMUSE:
    raw = find_amuse_optimizer(optimizer)
    zero3 = find_zero3_optimizer(optimizer)
    if zero3 is None or any(group["params"] for group in raw.param_groups):
        raw.train() if training else raw.eval()
        return raw

    partitions_by_group: dict[int, list[nn.Parameter]] = {}
    for subgroup, group_index in zip(
        zero3.fp32_partitioned_groups_flat,
        zero3.sub_group_to_group_id,
        strict=True,
    ):
        partitions_by_group.setdefault(int(group_index), []).append(subgroup)
    previous = [group["params"] for group in raw.param_groups]
    try:
        for index, group in enumerate(raw.param_groups):
            group["params"] = partitions_by_group.get(index, [])
        raw.train() if training else raw.eval()
    finally:
        for group, parameters in zip(raw.param_groups, previous, strict=True):
            group["params"] = parameters
    sync_zero3_master_to_model(zero3)
    return raw
