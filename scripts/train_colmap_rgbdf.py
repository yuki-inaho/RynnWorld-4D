"""Hydra-managed COLMAP RGB-DF training launcher."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from core.finetune.config import redacted_config, validate_training_config


def _string(value: object) -> str:
    return str(value).lower() if isinstance(value, bool) else str(value)


def training_command(config: DictConfig, output_dir: str | Path) -> list[str]:
    is_amuse = config.optimizer.name == "amuse"
    learning_rate = config.optimizer.aux_lr if is_amuse else config.optimizer.learning_rate
    joint_out_multiplier = config.optimizer.joint_out_multiplier if is_amuse else 1.0
    joint_other_multiplier = config.optimizer.joint_other_multiplier if is_amuse else 1.0
    command = [
        "accelerate",
        "launch",
        "--use_deepspeed",
        "--num_processes",
        str(config.distributed.num_processes),
        "--num_machines",
        "1",
        "--gpu_ids",
        "0",
        "--mixed_precision",
        str(config.trainer.mixed_precision),
        "--dynamo_backend",
        "no",
        "--deepspeed_config_file",
        str(config.distributed.config_path),
        "--zero3_init_flag",
        "true",
        "--zero3_save_16bit_model",
        "false",
        "finetune_rynnworld4d.py",
    ]
    arguments = {
        "model_path": config.model.base_path,
        "model_name": config.model.name,
        "model_type": config.model.name,
        "training_type": config.model.training_type,
        "output_dir": output_dir,
        "report_to": "none",
        "train_resolution": f"{config.data.source_frames}x{config.data.height}x{config.data.width}",
        "train_epochs": config.trainer.epochs,
        "seed": config.seed,
        "batch_size": config.trainer.micro_batch_size,
        "gradient_accumulation_steps": config.trainer.gradient_accumulation_steps,
        "mixed_precision": config.trainer.mixed_precision,
        "num_workers": config.trainer.num_workers,
        "pin_memory": config.trainer.pin_memory,
        "gradient_checkpointing": config.trainer.gradient_checkpointing,
        "optimizer": config.optimizer.name,
        "learning_rate": learning_rate,
        "beta1": config.optimizer.beta1,
        "beta2": config.optimizer.beta2,
        "epsilon": config.optimizer.epsilon,
        "weight_decay": config.optimizer.weight_decay,
        "lr_scheduler": "constant" if is_amuse else config.optimizer.scheduler,
        "lr_warmup_steps": 0 if is_amuse else config.optimizer.warmup_steps,
        "checkpointing_steps": 999999999,
        "checkpointing_limit": config.checkpoint.k,
        "save_final_checkpoint": config.trainer.save_final_checkpoint,
        "do_validation": False,
        "cache_dir": ".cache/rynnworld4d",
        "prompt": config.data.prompt,
        "dataset_format": "colmap_rgbdf",
        "managed_training": True,
        "managed_tensorboard": config.logging.name == "tensorboard",
        "metric_validation": True,
        "validation_every_epochs": config.trainer.validation_every_epochs,
        "topk": config.checkpoint.k,
        "checkpoint_monitor": config.checkpoint.monitor,
        "checkpoint_mode": config.checkpoint.mode,
        "save_last": config.checkpoint.save_last,
        "is_concat": True,
        "fusion_mode": config.model.fusion_mode,
        "share_ffn": config.model.share_ffn,
        "joint_start_layer": config.model.joint_start_layer,
        "joint_end_layer": config.model.joint_end_layer,
        "joint_every_n_layers": config.model.joint_every_n_layers,
        "joint_frame_wise": config.model.joint_frame_wise,
        "joint_use_rope": config.model.joint_use_rope,
        "joint_unidirectional": config.model.joint_unidirectional,
        "joint_video_decay": config.model.joint_video_decay,
        "joint_video_decay_steps": config.model.joint_video_decay_steps,
        "joint_out_lr": learning_rate * joint_out_multiplier,
        "joint_other_lr_multiplier": joint_other_multiplier,
        "loss_weight_flow": config.model.loss_weight_flow,
        "use_ema": False,
        "freeze_non_joint": config.model.freeze_non_joint,
        "branch_dropout_prob": config.model.branch_dropout_prob,
        "periodic_inference_steps": config.trainer.periodic_inference_steps,
        "load_stage2_model_weights": config.model.stage3_path,
    }
    if is_amuse:
        arguments.update(
            {
                "amuse_muon_lr": config.optimizer.muon_lr,
                "amuse_aux_lr": config.optimizer.aux_lr,
                "amuse_beta1": config.optimizer.beta1,
                "amuse_momentum": config.optimizer.momentum,
                "amuse_rho": config.optimizer.rho,
                "amuse_r": config.optimizer.r,
                "amuse_weight_lr_power": config.optimizer.weight_lr_power,
                "amuse_warmup_ratio": config.optimizer.warmup_ratio,
                "amuse_min_warmup_steps": config.optimizer.min_warmup_steps,
                "amuse_weight_decay_at_y": config.optimizer.weight_decay_at_y,
            }
        )
    if config.trainer.max_steps is not None:
        arguments["train_steps"] = config.trainer.max_steps
    if config.resume_from_checkpoint is not None:
        arguments["resume_from_checkpoint"] = config.resume_from_checkpoint
    for name, value in arguments.items():
        command.extend((f"--{name}", _string(value)))
    return command


@hydra.main(version_base=None, config_path="../configs/training", config_name="config")
def main(config: DictConfig) -> None:
    require_artifacts = config.mode != "config"
    validate_training_config(config, require_artifacts=require_artifacts)
    if config.mode == "config":
        return
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.redacted.json").write_text(
        json.dumps(redacted_config(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["RYNNWORLD4D_TRAIN_MANIFEST"] = str(config.data.train_manifest)
    environment["RYNNWORLD4D_VAL_MANIFEST"] = str(config.data.val_manifest)
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    subprocess.run(
        [sys.executable, "-m", "scripts.train_ready"],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
    )
    subprocess.run(training_command(config, output_dir), check=True, env=environment)


if __name__ == "__main__":
    main()
