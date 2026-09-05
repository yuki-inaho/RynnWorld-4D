import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file

from core.finetune.datasets.colmap_rgbdf_latents import ColmapRGBDFLatentDataset


def make_manifest(tmp_path, *, shape=(4, 3, 2, 2), split="train"):
    latent_dir = tmp_path / "latents"
    latent_dir.mkdir()
    video = torch.zeros(shape, dtype=torch.float32)
    text = torch.zeros((1, 5, 6), dtype=torch.float32)
    save_file({"video_latents": video}, latent_dir / "rgb.safetensors")
    save_file(
        {"depth_latents": video + 1, "flow_latents": video + 2},
        latent_dir / "flow_depth.safetensors",
    )
    save_file({"text_embeds": text}, latent_dir / "text.safetensors")
    rgb_sha256 = hashlib.sha256((latent_dir / "rgb.safetensors").read_bytes()).hexdigest()
    flow_depth_sha256 = hashlib.sha256((latent_dir / "flow_depth.safetensors").read_bytes()).hexdigest()
    text_sha256 = hashlib.sha256((latent_dir / "text.safetensors").read_bytes()).hexdigest()
    manifest = {
        "format": "rynnworld4d_colmap_rgbdf_latents_v1",
        "schema_version": 1,
        "split": split,
        "latent_shape": list(shape),
        "text_embeddings": "latents/text.safetensors",
        "text_embeddings_sha256": text_sha256,
        "items": [
            {
                "clip_id": "clip_0123456789abcdefabcd",
                "flow_depth_sha256": flow_depth_sha256,
                "rgb_latents": "latents/rgb.safetensors",
                "rgb_sha256": rgb_sha256,
                "flow_depth_latents": "latents/flow_depth.safetensors",
            }
        ],
    }
    path = tmp_path / f"{split}.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def test_loads_finite_relative_latents(tmp_path):
    path, _ = make_manifest(tmp_path)
    dataset = ColmapRGBDFLatentDataset(path, expected_split="train")
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["sample_id"].startswith("clip_")
    assert sample["encoded_video"].shape == (4, 3, 2, 2)
    assert sample["img_latent"].shape == (4, 1, 2, 2)
    assert torch.equal(sample["null_embedding"], sample["text_embedding"])
    assert dataset.fingerprint == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("bad_path", ["/private/data.safetensors", "../escape.safetensors"])
def test_rejects_noncontained_paths(tmp_path, bad_path):
    path, manifest = make_manifest(tmp_path)
    manifest["items"][0]["rgb_latents"] = bad_path
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="relative and contained"):
        ColmapRGBDFLatentDataset(path, expected_split="train")


def test_rejects_symlink_escape(tmp_path):
    path, manifest = make_manifest(tmp_path)
    rgb = tmp_path / manifest["items"][0]["rgb_latents"]
    outside = tmp_path.parent / f"{tmp_path.name}-outside.safetensors"
    outside.write_bytes(rgb.read_bytes())
    rgb.unlink()
    rgb.symlink_to(outside)

    with pytest.raises(ValueError, match="relative and contained"):
        ColmapRGBDFLatentDataset(path, expected_split="train")


def test_rejects_split_mismatch(tmp_path):
    path, _ = make_manifest(tmp_path, split="val")
    with pytest.raises(ValueError, match="split does not match"):
        ColmapRGBDFLatentDataset(path, expected_split="train")


def test_corrupt_item_fails_instead_of_substitution(tmp_path):
    path, manifest = make_manifest(tmp_path)
    rgb = tmp_path / manifest["items"][0]["rgb_latents"]
    rgb.write_bytes(b"broken")
    with pytest.raises(ValueError, match="hash"):
        ColmapRGBDFLatentDataset(path, expected_split="train")


def test_nonfinite_latent_fails(tmp_path):
    path, manifest = make_manifest(tmp_path)
    rgb = tmp_path / manifest["items"][0]["rgb_latents"]
    video = torch.zeros((4, 3, 2, 2), dtype=torch.float32)
    video[0, 0, 0, 0] = torch.nan
    save_file({"video_latents": video}, rgb)
    manifest["items"][0]["rgb_sha256"] = hashlib.sha256(rgb.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")
    dataset = ColmapRGBDFLatentDataset(path, expected_split="train")
    with pytest.raises(ValueError, match="must be finite"):
        dataset[0]
