import json

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from core.finetune.tensorboard import ScalarTensorBoardLogger
from core.finetune.topk_checkpointing import TopKCheckpointManager


def test_scalar_tensorboard_writes_only_finite_allowlisted_values(tmp_path):
    with ScalarTensorBoardLogger(tmp_path, is_main_process=True) as logger:
        logger.log_many(
            {
                "train/loss_total": 1.25,
                "val/loss_total": 0.75,
                "optimizer/lr": 1e-6,
                "system/gpu_peak_allocated_gib": 12.5,
                "system/samples_per_second": 0.1,
            },
            step=4,
        )
        logger.flush()

    accumulator = EventAccumulator(str(tmp_path)).Reload()
    assert set(accumulator.Tags()["scalars"]) == {
        "train/loss_total",
        "val/loss_total",
        "optimizer/lr",
        "system/gpu_peak_allocated_gib",
        "system/samples_per_second",
    }
    assert accumulator.Scalars("val/loss_total")[0].step == 4


def test_scalar_tensorboard_rejects_unknown_or_nonfinite(tmp_path):
    logger = ScalarTensorBoardLogger(tmp_path, is_main_process=False)
    with pytest.raises(ValueError, match="not allowed"):
        logger.log("private/source_path", 1.0, 0)
    with pytest.raises(ValueError, match="finite"):
        logger.log("train/loss_total", float("nan"), 0)


def _save_marker(path):
    (path / "state.txt").write_text("complete", encoding="utf-8")


@pytest.mark.parametrize(
    ("mode", "metrics", "expected"),
    [
        ("min", [0.5, 0.2, 0.4, 0.3, 0.1], [0.1, 0.2, 0.3]),
        ("max", [0.5, 0.2, 0.4, 0.3, 0.6], [0.6, 0.5, 0.4]),
    ],
)
def test_topk_keeps_only_best_metrics_and_reloads(tmp_path, mode, metrics, expected):
    manager = TopKCheckpointManager(tmp_path, k=3, monitor="val/loss_total", mode=mode)
    for step, metric in enumerate(metrics, start=1):
        manager.consider(metric, epoch=0, step=step, save=_save_marker)
        assert len(list(tmp_path.glob("best-*"))) <= 3

    assert [record.metric for record in manager.records] == expected
    reloaded = TopKCheckpointManager(tmp_path, k=3, monitor="val/loss_total", mode=mode)
    assert reloaded.records == manager.records


def test_topk_tie_is_deterministic_and_nonfinite_fails(tmp_path):
    manager = TopKCheckpointManager(tmp_path, k=1, monitor="val/loss_total", mode="min")
    first = manager.consider(0.5, epoch=0, step=1, save=_save_marker)
    assert manager.consider(0.5, epoch=0, step=2, save=_save_marker) is None
    assert manager.records[0].directory == first.name
    with pytest.raises(ValueError, match="finite"):
        manager.consider(float("nan"), epoch=0, step=3, save=_save_marker)


def test_topk_failed_save_preserves_existing_best(tmp_path):
    manager = TopKCheckpointManager(tmp_path, k=1, monitor="val/loss_total", mode="min")
    first = manager.consider(0.5, epoch=0, step=1, save=_save_marker)

    def fail(_path):
        raise RuntimeError("save failed")

    with pytest.raises(RuntimeError, match="save failed"):
        manager.consider(0.1, epoch=0, step=2, save=fail)
    assert first.is_dir()
    payload = json.loads((tmp_path / "leaderboard.json").read_text())
    assert payload["records"][0]["step"] == 1
