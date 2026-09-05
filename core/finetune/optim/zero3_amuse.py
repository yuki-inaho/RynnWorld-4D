"""Single-process ZeRO-3 adapter that restores Muon matrix shapes per step."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager

import torch

from core.finetune.optim.amuse import AMUSE


class Zero3CompatibleAMUSE(AMUSE):
    """AMUSE with shape restoration for DeepSpeed's flat FP32 partitions.

    DeepSpeed ZeRO-3 temporarily replaces every optimizer group's parameters
    with a flat FP32 partition. Muon needs the original 2-D/4-D matrix shape.
    This adapter is deliberately limited to one process and one matrix per
    Muon group, making the reshape exact rather than inferred.
    """

    @contextmanager
    def _matrix_shapes(self) -> Iterator[None]:
        restored: list[tuple[torch.nn.Parameter, torch.Tensor | None]] = []
        for group in self.param_groups:
            if not group.get("use_muon"):
                continue
            parameters = group["params"]
            if not parameters:
                continue
            if len(parameters) != 1:
                shapes = [tuple(parameter.shape) for parameter in parameters]
                raise RuntimeError(
                    f"ZeRO-3 AMUSE requires exactly one matrix per Muon group; "
                    f"received count={len(parameters)}, shapes={shapes}"
                )
            parameter = parameters[0]
            shape = tuple(group.get("logical_shape", ()))
            if not shape or math.prod(shape) != parameter.numel():
                raise RuntimeError("ZeRO-3 AMUSE flat partition does not match its logical matrix shape")
            if parameter.ndim != 1:
                continue
            gradient = parameter.grad
            parameter.data = parameter.data.view(shape)
            if gradient is not None:
                gradient.data = gradient.data.view(shape)
            for value in self.state[parameter].values():
                if isinstance(value, torch.Tensor) and value.numel() == parameter.numel() and value.ndim == 1:
                    value.data = value.data.view(shape)
            restored.append((parameter, gradient))
        try:
            yield
        finally:
            for parameter, gradient in restored:
                parameter.data = parameter.data.reshape(-1)
                if gradient is not None:
                    gradient.data = gradient.data.reshape(-1)
                for value in self.state[parameter].values():
                    if isinstance(value, torch.Tensor) and value.numel() == parameter.numel():
                        value.data = value.data.reshape(-1)

    @torch.no_grad()
    def step(self, closure=None):
        owner = getattr(self, "_zero3_partition_owner", None)
        active_groups = [group for group in self.param_groups if group["params"]]
        if owner is None or len(active_groups) == len(self.param_groups):
            with self._matrix_shapes():
                return super().step(closure)
        if len(active_groups) != 1:
            raise RuntimeError(
                f"ZeRO-3 AMUSE expected exactly one active subgroup per step, got {len(active_groups)}"
            )

        # ZeRO-3 invokes the base optimizer once per subgroup and empties all
        # other parameter groups. Upstream AMUSE advances every group counter
        # on every call, so temporarily expose only the subgroup being updated.
        all_groups = self.param_groups
        self.param_groups = active_groups
        try:
            with self._matrix_shapes():
                return super().step(closure)
        finally:
            self.param_groups = all_groups

    @contextmanager
    def _zero3_parameters(self) -> Iterator[None]:
        owner = getattr(self, "_zero3_partition_owner", None)
        if owner is None or any(group["params"] for group in self.param_groups):
            yield
            return
        parameters_by_group: dict[int, list[torch.nn.Parameter]] = {}
        for parameter, group_index in zip(
            owner.fp32_partitioned_groups_flat,
            owner.sub_group_to_group_id,
            strict=True,
        ):
            parameters_by_group.setdefault(int(group_index), []).append(parameter)
        previous = [group["params"] for group in self.param_groups]
        try:
            for index, group in enumerate(self.param_groups):
                group["params"] = parameters_by_group.get(index, [])
            yield
        finally:
            for group, parameters in zip(self.param_groups, previous, strict=True):
                group["params"] = parameters

    def state_dict(self):
        with self._zero3_parameters():
            return super().state_dict()

    def load_state_dict(self, state_dict):
        with self._zero3_parameters():
            return super().load_state_dict(state_dict)
