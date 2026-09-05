"""Rank-zero, scalar-only TensorBoard logging."""

from __future__ import annotations

import math
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from typing_extensions import Self

SCALAR_TAGS = frozenset(
    {
        "train/loss_total",
        "train/loss_rgb",
        "train/loss_depth",
        "train/loss_flow",
        "train/grad_norm",
        "val/loss_total",
        "val/loss_rgb",
        "val/loss_depth",
        "val/loss_flow",
        "optimizer/lr",
        "optimizer/lr_muon",
        "optimizer/lr_aux",
        "system/gpu_peak_allocated_gib",
        "system/gpu_peak_reserved_gib",
        "system/samples_per_second",
    }
)


class ScalarTensorBoardLogger:
    def __init__(
        self,
        log_dir: str | Path,
        *,
        is_main_process: bool,
        flush_secs: int = 10,
    ) -> None:
        self._writer = (
            SummaryWriter(log_dir=str(log_dir), flush_secs=flush_secs)
            if is_main_process
            else None
        )

    def log(self, tag: str, value: float, step: int) -> None:
        if tag not in SCALAR_TAGS:
            raise ValueError(f"TensorBoard scalar tag is not allowed: {tag}")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"TensorBoard scalar must be finite: {tag}")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("TensorBoard step must be a non-negative integer")
        if self._writer is not None:
            self._writer.add_scalar(tag, numeric, step)

    def log_many(self, values: dict[str, float], step: int) -> None:
        for tag, value in values.items():
            self.log(tag, value, step)

    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
