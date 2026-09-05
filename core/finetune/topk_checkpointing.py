"""Atomic metric-based top-K checkpoint management."""

from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckpointRecord:
    metric: float
    epoch: int
    step: int
    directory: str


class TopKCheckpointManager:
    def __init__(self, root: str | Path, *, k: int, monitor: str, mode: str) -> None:
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("top-K checkpoint count must be an integer >= 1")
        if mode not in {"min", "max"}:
            raise ValueError("top-K checkpoint mode must be min or max")
        if monitor != "val/loss_total":
            raise ValueError("top-K checkpoint monitor must be val/loss_total")
        self.root = Path(root)
        self.k = k
        self.monitor = monitor
        self.mode = mode
        self.leaderboard_path = self.root / "leaderboard.json"
        self.records = self._load()

    def _sort_key(self, record: CheckpointRecord) -> tuple[float, int, int, str]:
        metric = record.metric if self.mode == "min" else -record.metric
        return metric, record.epoch, record.step, record.directory

    def _load(self) -> list[CheckpointRecord]:
        if not self.leaderboard_path.exists():
            return []
        try:
            payload = json.loads(self.leaderboard_path.read_text(encoding="utf-8"))
            if (
                payload.get("format") != "rynnworld4d_topk_v1"
                or payload.get("monitor") != self.monitor
                or payload.get("mode") != self.mode
                or payload.get("k") != self.k
            ):
                raise ValueError
            records = [CheckpointRecord(**item) for item in payload["records"]]
        except (OSError, TypeError, KeyError, ValueError, json.JSONDecodeError):
            raise ValueError("top-K checkpoint leaderboard is invalid") from None
        if any(not math.isfinite(record.metric) for record in records):
            raise ValueError("top-K checkpoint leaderboard metric must be finite")
        if records != sorted(records, key=self._sort_key) or len(records) > self.k:
            raise ValueError("top-K checkpoint leaderboard ordering is invalid")
        if any(not (self.root / record.directory).is_dir() for record in records):
            raise ValueError("top-K checkpoint leaderboard references a missing checkpoint")
        return records

    def _write_leaderboard(self, records: list[CheckpointRecord]) -> None:
        payload = {
            "format": "rynnworld4d_topk_v1",
            "monitor": self.monitor,
            "mode": self.mode,
            "k": self.k,
            "records": [asdict(record) for record in records],
        }
        temporary = self.root / ".leaderboard.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, self.leaderboard_path)

    def qualifies(self, metric: float, *, epoch: int, step: int) -> bool:
        candidate = CheckpointRecord(float(metric), epoch, step, f"best-e{epoch:04d}-s{step:08d}")
        if not math.isfinite(candidate.metric):
            raise ValueError("top-K checkpoint metric must be finite")
        return len(self.records) < self.k or self._sort_key(candidate) < self._sort_key(self.records[-1])

    def consider(
        self,
        metric: float,
        *,
        epoch: int,
        step: int,
        save: Callable[[Path], None],
    ) -> Path | None:
        if not self.qualifies(metric, epoch=epoch, step=step):
            return None
        self.root.mkdir(parents=True, exist_ok=True)
        directory = f"best-e{epoch:04d}-s{step:08d}"
        final_path = self.root / directory
        if final_path.exists():
            raise ValueError("top-K checkpoint target already exists")
        temporary = self.root / f".{directory}.tmp"
        if temporary.exists():
            raise ValueError("top-K checkpoint temporary target already exists")
        temporary.mkdir()
        try:
            save(temporary)
            if not any(temporary.iterdir()):
                raise ValueError("top-K checkpoint save produced no artifacts")
            os.replace(temporary, final_path)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

        candidate = CheckpointRecord(float(metric), epoch, step, directory)
        kept = sorted([*self.records, candidate], key=self._sort_key)[: self.k]
        removed = [record for record in [*self.records, candidate] if record not in kept]
        self._write_leaderboard(kept)
        self.records = kept
        for record in removed:
            path = self.root / record.directory
            if path.exists():
                shutil.rmtree(path)
        return final_path
