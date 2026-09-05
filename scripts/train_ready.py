"""Model-free, privacy-safe readiness gate for managed RGB-DF training."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from core.finetune.config import validate_training_config
from core.finetune.datasets.colmap_rgbdf_latents import ColmapRGBDFLatentDataset
from core.finetune.datasets.colmap_rgbdf_source import ColmapRGBDFSource

FORMAT = "rynnworld4d_train_readiness_v1"
EXPECTED_CLIPS = {"train": 93, "val": 8, "smoke": 2}
EXPECTED_LATENT_SHAPE = (48, 17, 30, 40)
AMUSE_SHA256 = "84fd3fbbc99e1718cf1c821ceff3369439f48e6fbd8ecc3a2b83afa5d82eea1f"
LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
MIN_FREE_BYTES = 45_000_000_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(report: dict[str, object]) -> None:
    destination = Path(os.environ.get("RYNNWORLD4D_READINESS_REPORT", "outputs/readiness/train-ready.json"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def _smoke_evidence_is_valid(root: Path) -> bool:
    try:
        payload = json.loads((root / "outputs/readiness/amuse-deepspeed-smoke.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("format") == "rynnworld4d_amuse_deepspeed_smoke_v1"
        and payload.get("amuse_deepspeed") == "pass"
        and payload.get("compact_resume") == "pass"
        and payload.get("optimizer_steps") == 2
        and float(payload.get("xy_output_max_abs_diff", 0.0)) > 0.0
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    checks = {
        "amuse_provenance": False,
        "data_quality_report": False,
        "disk_capacity": False,
        "environment": False,
        "gpu_32gb": False,
        "hydra_config": False,
        "latent_artifacts": False,
        "pretrained_artifacts": False,
        "source_contract": False,
        "zero3_amuse_smoke": False,
    }
    details: dict[str, object] = {"allow_nan": False}
    source_value = os.environ.get("RYNNWORLD4D_COLMAP_ROOT")
    latent_value = os.environ.get("RYNNWORLD4D_LATENT_ROOT")
    checks["environment"] = bool(source_value and latent_value)

    config = None
    try:
        with initialize_config_dir(version_base=None, config_dir=str(root / "configs/training")):
            config = compose(config_name="config")
        config.data.source_root = source_value or "__missing_source__"
        config.data.latent_root = latent_value or "__missing_latents__"
        OmegaConf.resolve(config)
        validate_training_config(config, require_artifacts=False)
        checks["hydra_config"] = True
    except Exception:  # noqa: BLE001 - readiness boundary redacts all underlying exceptions
        details["hydra_config_status"] = "failed"
        config = None

    if config is not None:
        if source_value:
            try:
                source = ColmapRGBDFSource(source_value)
                counts = {
                    split: len(
                        source.clip_windows(
                            split,
                            length=int(config.data.source_frames),
                            stride=int(config.data.clip_stride),
                        )
                    )
                    for split in source.SPLITS
                }
                expected_size = (int(config.data.source_width), int(config.data.source_height))
                checks["source_contract"] = counts == EXPECTED_CLIPS and source.image_size == expected_size
                details["source_clip_counts"] = counts
            except Exception:  # noqa: BLE001 - readiness boundary redacts all underlying exceptions
                details["source_contract_status"] = "failed"

        if latent_value:
            try:
                latent_counts: dict[str, int] = {}
                for split in ("train", "val", "smoke"):
                    manifest = Path(str(config.data.latent_root)) / f"{split}.json"
                    dataset = ColmapRGBDFLatentDataset(manifest, expected_split=split)
                    latent_counts[split] = len(dataset)
                    for index in range(len(dataset)):
                        sample = dataset[index]
                        if tuple(sample["encoded_video"].shape) != EXPECTED_LATENT_SHAPE:
                            raise ValueError("latent shape mismatch")
                checks["latent_artifacts"] = latent_counts == EXPECTED_CLIPS
                details["latent_clip_counts"] = latent_counts
            except Exception:  # noqa: BLE001 - readiness boundary redacts all underlying exceptions
                details["latent_artifacts_status"] = "failed"

            try:
                quality = json.loads(
                    (Path(str(config.data.latent_root)) / "reports/data_validation.json").read_text(encoding="utf-8")
                )
                checks["data_quality_report"] = (
                    quality.get("format") == "rynnworld4d_colmap_rgbdf_validation_v1"
                    and quality.get("allow_nan") is False
                    and quality.get("clip_counts") == EXPECTED_CLIPS
                    and quality.get("source_size") == {"height": 480, "width": 640}
                    and quality.get("training_canvas") == {"height": 480, "width": 640}
                    and quality.get("padding") == "none_v1"
                )
            except (OSError, json.JSONDecodeError):
                details["data_quality_report_status"] = "failed"

        stage3 = Path(str(config.model.stage3_path)) / "pytorch_model/mp_rank_00_model_states.pt"
        base = Path(str(config.model.base_path))
        checks["pretrained_artifacts"] = stage3.is_file() and all(
            (base / name).is_dir() for name in ("scheduler", "text_encoder", "tokenizer", "transformer", "vae")
        )

    checks["amuse_provenance"] = (
        _sha256(root / "core/finetune/optim/amuse.py") == AMUSE_SHA256
        and _sha256(root / "third_party/amuse/LICENSE") == LICENSE_SHA256
    )
    checks["zero3_amuse_smoke"] = _smoke_evidence_is_valid(root)
    free_bytes = shutil.disk_usage(root).free
    checks["disk_capacity"] = free_bytes >= MIN_FREE_BYTES
    details["free_disk_gib"] = round(free_bytes / 1024**3, 2)
    if torch.cuda.is_available():
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        checks["gpu_32gb"] = total_bytes >= 32_000_000_000
        details["gpu_memory_gib"] = round(total_bytes / 1024**3, 2)

    failures = sorted(name for name, passed in checks.items() if not passed)
    report = {
        "allow_nan": False,
        "checks": checks,
        "details": details,
        "failures": failures,
        "format": FORMAT,
        "ready": not failures,
    }
    _write_report(report)
    print(json.dumps(report, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
