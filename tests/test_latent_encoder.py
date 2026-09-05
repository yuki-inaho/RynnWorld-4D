import hashlib
import json

import numpy as np
import torch

from core.finetune.datasets.colmap_rgbdf_latents import ColmapRGBDFLatentDataset
from core.finetune.preprocessing.latent_encoder import encode_intermediate_manifests


def test_intermediate_encoder_writes_loadable_relative_manifest(tmp_path):
    intermediate = tmp_path / "intermediate" / "train" / "clip_0123456789abcdefabcd.npz"
    intermediate.parent.mkdir(parents=True)
    rgb = np.zeros((5, 6, 8, 3), dtype=np.uint8)
    channels = np.zeros((5, 3, 6, 8), dtype=np.uint8)
    valid = np.ones((5, 6, 8), dtype=bool)
    np.savez_compressed(
        intermediate,
        rgb=rgb,
        depth_rgb=channels,
        flow_rgb=channels,
        depth_valid=valid,
        flow_valid=valid,
    )
    manifest = {
        "format": "rynnworld4d_colmap_rgbdf_intermediate_v1",
        "schema_version": 1,
        "split": "train",
        "items": [
            {
                "clip_id": "clip_0123456789abcdefabcd",
                "intermediate": "intermediate/train/clip_0123456789abcdefabcd.npz",
                "sha256": hashlib.sha256(intermediate.read_bytes()).hexdigest(),
            }
        ],
    }
    (tmp_path / "intermediate_train.json").write_text(json.dumps(manifest), encoding="utf-8")

    def fake_encoder(_frames):
        return torch.zeros((4, 3, 2, 2), dtype=torch.float32)

    counts = encode_intermediate_manifests(
        tmp_path,
        encode_video=fake_encoder,
        text_embedding=torch.zeros((1, 5, 6)),
        splits=("train",),
        expected_shape=(4, 3, 2, 2),
    )
    dataset = ColmapRGBDFLatentDataset(tmp_path / "train.json", expected_split="train")
    assert counts == {"train": 1}
    assert len(dataset) == 1
    assert dataset[0]["encoded_video"].shape == (4, 3, 2, 2)
