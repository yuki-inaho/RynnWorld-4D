import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError

from core.finetune.config import redacted_config, validate_training_config
from scripts.train_colmap_rgbdf import training_command

CONFIG_DIR = Path(__file__).parents[1] / "configs" / "training"


def compose_config(overrides=None):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name="config", overrides=overrides or [])


def test_default_config_is_amuse_tensorboard_topk(monkeypatch, tmp_path):
    source, latents = tmp_path / "source", tmp_path / "latents"
    source.mkdir()
    latents.mkdir()
    monkeypatch.setenv("RYNNWORLD4D_COLMAP_ROOT", str(source))
    monkeypatch.setenv("RYNNWORLD4D_LATENT_ROOT", str(latents))
    config = compose_config()
    OmegaConf.resolve(config)
    assert config.optimizer.name == "amuse"
    assert config.optimizer.scheduler is None
    assert config.logging.name == "tensorboard"
    assert config.checkpoint.k == 3
    assert config.checkpoint.monitor == "val/loss_total"
    assert config.checkpoint.save_last is False
    assert config.trainer.micro_batch_size == 2
    assert config.data.source_frames == 65
    assert (config.data.source_width, config.data.source_height) == (630, 476)
    assert (config.data.width, config.data.height) == (640, 480)
    assert config.data.latent_frames == 17


def test_missing_environment_variable_is_named(monkeypatch):
    monkeypatch.delenv("RYNNWORLD4D_COLMAP_ROOT", raising=False)
    monkeypatch.delenv("RYNNWORLD4D_LATENT_ROOT", raising=False)
    config = compose_config()
    with pytest.raises(InterpolationResolutionError, match="RYNNWORLD4D_COLMAP_ROOT"):
        OmegaConf.resolve(config)


def test_overrides_compose_without_model_allocation(monkeypatch, tmp_path):
    monkeypatch.setenv("RYNNWORLD4D_COLMAP_ROOT", str(tmp_path / "source"))
    monkeypatch.setenv("RYNNWORLD4D_LATENT_ROOT", str(tmp_path / "latents"))
    config = compose_config(["optimizer=adamw", "trainer=smoke", "logging=disabled"])
    assert config.optimizer.name == "adamw"
    assert config.trainer.max_steps == 1
    assert config.logging.name == "disabled"
    command = " ".join(training_command(config, "outputs/training/test"))
    assert "--managed_tensorboard false" in command
    assert "--learning_rate 1e-06" in command
    assert "--lr_scheduler cosine_with_warmup" in command
    assert "--amuse_muon_lr" not in command


def test_validator_rejects_invalid_amuse_scheduler(monkeypatch, tmp_path):
    monkeypatch.setenv("RYNNWORLD4D_COLMAP_ROOT", str(tmp_path / "source"))
    monkeypatch.setenv("RYNNWORLD4D_LATENT_ROOT", str(tmp_path / "latents"))
    config = compose_config(["optimizer.scheduler=cosine"])
    with pytest.raises(ValueError, match="internal warmup"):
        validate_training_config(config, require_artifacts=False)


def test_validator_rejects_oom_risk_micro_batch(monkeypatch, tmp_path):
    monkeypatch.setenv("RYNNWORLD4D_COLMAP_ROOT", str(tmp_path / "source"))
    monkeypatch.setenv("RYNNWORLD4D_LATENT_ROOT", str(tmp_path / "latents"))
    config = compose_config(["trainer.micro_batch_size=3"])
    with pytest.raises(ValueError, match="at most 2"):
        validate_training_config(config, require_artifacts=False)


def test_redacted_config_contains_no_private_absolute_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("RYNNWORLD4D_COLMAP_ROOT", str(tmp_path / "private_source"))
    monkeypatch.setenv("RYNNWORLD4D_LATENT_ROOT", str(tmp_path / "private_latents"))
    rendered = json.dumps(redacted_config(compose_config()))
    assert str(tmp_path) not in rendered
    assert "private_source" not in rendered


def test_training_command_uses_environment_for_private_manifests(monkeypatch, tmp_path):
    monkeypatch.setenv("RYNNWORLD4D_COLMAP_ROOT", str(tmp_path / "private_source"))
    monkeypatch.setenv("RYNNWORLD4D_LATENT_ROOT", str(tmp_path / "private_latents"))
    command = training_command(compose_config(), "outputs/training/test")
    rendered = " ".join(command)
    assert str(tmp_path) not in rendered
    assert "--train_manifest" not in command
    assert "--optimizer amuse" in rendered
