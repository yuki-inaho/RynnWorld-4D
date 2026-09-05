"""One-step GPU compatibility smoke for custom AMUSE under DeepSpeed."""

from __future__ import annotations

import json
import os
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from core.finetune.compact_checkpoint import (
    load_compact_amuse_checkpoint,
    save_compact_amuse_checkpoint,
)
from core.finetune.optim import (
    attach_zero3_amuse,
    build_amuse_optimizer,
    find_zero3_optimizer,
    set_amuse_mode,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(16, 16)
        self.output = nn.Linear(16, 4)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.output(torch.tanh(self.hidden(values)))


def main() -> None:
    torch.manual_seed(42)
    accelerator = Accelerator(mixed_precision="bf16")
    model = TinyModel()
    result = build_amuse_optimizer(
        model,
        total_optimizer_steps=2,
        muon_lr=1e-1,
        aux_lr=1e-2,
        fallback_patterns=("*output.weight",),
    )
    loader = DataLoader(
        TensorDataset(torch.randn(8, 16), torch.randn(8, 4)),
        batch_size=2,
    )
    model, optimizer, loader = accelerator.prepare(model, result.optimizer, loader)
    raw = attach_zero3_amuse(optimizer)
    if accelerator.is_main_process:
        print(
            json.dumps(
                {
                    "groups": [
                        {
                            "name": group.get("name"),
                            "param_names": group.get("param_names"),
                            "shapes": [list(parameter.shape) for parameter in group["params"]],
                            "use_muon": group.get("use_muon"),
                        }
                        for group in raw.param_groups
                    ]
                }
            )
        )
    raw.train()
    values = targets = loss = None
    for values, targets in loader:
        values = values.to(dtype=torch.bfloat16)
        targets = targets.to(dtype=torch.bfloat16)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(values), targets)
        accelerator.backward(loss)
        optimizer.step()
        if not torch.isfinite(loss):
            raise RuntimeError("AMUSE DeepSpeed smoke loss is non-finite")
        if raw.param_groups[0]["k"] == 2:
            break
    if values is None or loss is None or raw.param_groups[0]["k"] != 2:
        raise RuntimeError("AMUSE DeepSpeed smoke did not complete two optimizer steps")
    with torch.no_grad():
        prediction_y = model(values).float()
    zero3 = find_zero3_optimizer(optimizer)
    if zero3 is None:
        raise RuntimeError("AMUSE DeepSpeed smoke could not find the ZeRO-3 optimizer")
    partitions_y = [partition.detach().clone() for partition in zero3.fp32_partitioned_groups_flat]
    model_partitions_y = [partition.detach().clone() for partition in zero3.fp16_partitioned_groups_flat]
    visible_parameters_y = [parameter.detach().clone() for parameter in model.module.parameters()]
    set_amuse_mode(optimizer, training=False)
    if raw.train_mode:
        raise RuntimeError("AMUSE did not enter evaluation-weight mode")
    with torch.no_grad():
        prediction_x = model(values).float()
    partitions_x = [partition.detach().clone() for partition in zero3.fp32_partitioned_groups_flat]
    model_partitions_x = [partition.detach().clone() for partition in zero3.fp16_partitioned_groups_flat]
    visible_parameters_x = [parameter.detach().clone() for parameter in model.module.parameters()]
    partition_xy_diff = max(
        float((weight_y - weight_x).abs().max())
        for weight_y, weight_x in zip(partitions_y, partitions_x, strict=True)
    )
    output_xy_diff = float((prediction_y - prediction_x).abs().max())
    model_partition_xy_diff = max(
        float((weight_y - weight_x).abs().max())
        for weight_y, weight_x in zip(model_partitions_y, model_partitions_x, strict=True)
    )
    visible_parameter_xy_diff = max(
        float((weight_y - weight_x).abs().max())
        for weight_y, weight_x in zip(visible_parameters_y, visible_parameters_x, strict=True)
    )
    if accelerator.is_main_process:
        print(
            json.dumps(
                {
                    "diagnostic_group_steps": [group["k"] for group in raw.param_groups],
                    "diagnostic_xy_fp32": partition_xy_diff,
                    "diagnostic_xy_model_partition": model_partition_xy_diff,
                    "diagnostic_xy_visible_parameter": visible_parameter_xy_diff,
                    "diagnostic_xy_output": output_xy_diff,
                }
            )
        )
    if partition_xy_diff == 0.0:
        raise RuntimeError("AMUSE evaluation weights did not differ from its training weights after two steps")
    if output_xy_diff == 0.0:
        raise RuntimeError("AMUSE evaluation weights were not synchronized into the ZeRO-3 model")
    checkpoint_generator = torch.Generator(device=accelerator.device).manual_seed(123)
    random.seed(123)
    with tempfile.TemporaryDirectory(prefix="rynnworld4d-amuse-smoke-") as directory:
        save_compact_amuse_checkpoint(
            Path(directory),
            model=model,
            optimizer=optimizer,
            epoch=0,
            global_step=2,
            metric=float(loss),
            config_fingerprint="c" * 64,
            grouping_fingerprint=result.grouping.fingerprint,
            pretrained_fingerprint="d" * 64,
            data_fingerprints={"train": "a" * 64, "val": "b" * 64},
            generator=checkpoint_generator,
        )
        expected_random = random.random()
        expected_generator = torch.rand(
            4,
            device=accelerator.device,
            generator=checkpoint_generator,
        )
        random.seed(999)
        checkpoint_generator.manual_seed(999)
        set_amuse_mode(optimizer, training=True)
        optimizer.zero_grad(set_to_none=True)
        mutation_loss = torch.nn.functional.mse_loss(model(values), targets)
        accelerator.backward(mutation_loss)
        optimizer.step()
        set_amuse_mode(optimizer, training=False)
        with torch.no_grad():
            prediction_x_mutated = model(values).float()
        if torch.equal(prediction_x, prediction_x_mutated):
            raise RuntimeError("compact checkpoint smoke mutation did not change model weights")
        resume = load_compact_amuse_checkpoint(
            directory,
            model=model,
            optimizer=optimizer,
            config_fingerprint="c" * 64,
            grouping_fingerprint=result.grouping.fingerprint,
            pretrained_fingerprint="d" * 64,
            data_fingerprints={"train": "a" * 64, "val": "b" * 64},
            generator=checkpoint_generator,
        )
        if resume.global_step != 2 or resume.first_epoch != 1:
            raise RuntimeError("compact checkpoint restored incorrect progress")
        with torch.no_grad():
            prediction_x_resumed = model(values).float()
        if not torch.equal(prediction_x, prediction_x_resumed):
            raise RuntimeError("compact checkpoint did not restore AMUSE x weights exactly")
        if random.random() != expected_random:
            raise RuntimeError("compact checkpoint did not restore Python RNG state")
        actual_generator = torch.rand(
            4,
            device=accelerator.device,
            generator=checkpoint_generator,
        )
        if not torch.equal(expected_generator, actual_generator):
            raise RuntimeError("compact checkpoint did not restore explicit generator state")
    set_amuse_mode(optimizer, training=True)
    if not raw.train_mode:
        raise RuntimeError("AMUSE did not restore training-weight mode")
    with torch.no_grad():
        prediction_y_restored = model(values).float()
    if not torch.allclose(prediction_y, prediction_y_restored, atol=2e-2, rtol=2e-2):
        raise RuntimeError("AMUSE training weights did not survive the ZeRO-3 x/y round trip")
    state = raw.state_dict()
    if not state["state"]:
        raise RuntimeError("AMUSE ZeRO-3 optimizer state is empty")
    if accelerator.is_main_process:
        evidence = {
            "amuse_deepspeed": "pass",
            "compact_resume": "pass",
            "format": "rynnworld4d_amuse_deepspeed_smoke_v1",
            "gpu": torch.cuda.get_device_name(0),
            "loss": float(loss),
            "optimizer_steps": 2,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "xy_model_partition_max_abs_diff": model_partition_xy_diff,
            "xy_output_max_abs_diff": output_xy_diff,
            "xy_partition_max_abs_diff": partition_xy_diff,
            "y_roundtrip_max_abs_diff": float((prediction_y - prediction_y_restored).abs().max()),
        }
        evidence_root = Path("outputs/readiness")
        evidence_root.mkdir(parents=True, exist_ok=True)
        temporary = evidence_root / ".amuse-deepspeed-smoke.json.tmp"
        temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, evidence_root / "amuse-deepspeed-smoke.json")
        print(json.dumps(evidence))
    accelerator.end_training()


if __name__ == "__main__":
    main()
