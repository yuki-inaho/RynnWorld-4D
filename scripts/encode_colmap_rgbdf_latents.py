#!/usr/bin/env python3
"""Encode lossless COLMAP RGB-DF intermediates with the released Wan VAE."""

from __future__ import annotations

import json
import os

from core.finetune.preprocessing.latent_encoder import (
    WanLatentEncoder,
    encode_intermediate_manifests,
)


def required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required environment variable is unset: {name}")
    return value


def main() -> None:
    artifact_root = required_environment("RYNNWORLD4D_LATENT_ROOT")
    model_path = os.environ.get("RYNNWORLD4D_BASE_MODEL", "pretrained/Wan2.2-TI2V-5B-Diffusers")
    encoder = WanLatentEncoder(model_path)
    counts = encode_intermediate_manifests(
        artifact_root,
        encode_video=encoder,
        text_embedding=encoder.text_embedding,
    )
    print(json.dumps({"encoded_clips": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
