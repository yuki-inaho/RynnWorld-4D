#!/usr/bin/env python3
"""Validate or materialize the private COLMAP RGB-D source."""

from __future__ import annotations

import argparse
import json
import os

from core.finetune.preprocessing.materialize import (
    audit_source_metadata,
    materialize_rgbdf,
)


def required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required environment variable is unset: {name}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("audit-metadata", "materialize"))
    args = parser.parse_args()
    source_root = required_environment("RYNNWORLD4D_COLMAP_ROOT")
    if args.mode == "audit-metadata":
        print(json.dumps(audit_source_metadata(source_root), sort_keys=True))
        return
    latent_root = required_environment("RYNNWORLD4D_LATENT_ROOT")
    report = materialize_rgbdf(source_root, latent_root)
    print(json.dumps({"allow_nan": report["allow_nan"], "clip_counts": report["clip_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
