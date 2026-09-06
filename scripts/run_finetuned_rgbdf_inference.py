"""Run inference-sft.py with a managed compact AMUSE checkpoint overlay."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch


ROOT = Path(__file__).resolve().parents[1]
FORMAT = "rynnworld4d_compact_amuse_zero3_v1"
sys.path.insert(0, str(ROOT))


def _load_inference_module():
    script = ROOT / "inference-sft.py"
    spec = importlib.util.spec_from_file_location("rynnworld4d_inference_sft", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load inference-sft.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_released_weights(checkpoint_path: str, transformer: torch.nn.Module) -> int:
    """Load the released Stage-3 model directly, without a duplicate /tmp copy."""
    path = Path(checkpoint_path) / "pytorch_model" / "mp_rank_00_model_states.pt"
    if not path.is_file():
        raise FileNotFoundError(f"released model state is missing: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    source = checkpoint.get("module")
    if not isinstance(source, dict):
        raise ValueError("released checkpoint has no module state")
    destination = transformer.state_dict()
    loaded = 0
    for source_name, value in source.items():
        name = source_name.replace("module.", "").replace(".base_layer.", ".")
        target = destination.get(name)
        if target is not None and tuple(target.shape) == tuple(value.shape):
            target.copy_(value)
            loaded += 1
    if loaded < 2_000:
        raise ValueError(f"released checkpoint coverage is unexpectedly low: {loaded}")
    del destination, source, checkpoint
    gc.collect()
    print(f"Loaded {loaded} released Stage-3 model tensors without a temporary copy.")
    return loaded


def load_compact_overlay(checkpoint_path: str, transformer: torch.nn.Module) -> int:
    """Restore single-rank ZeRO-3 FP32 flat groups into ordinary model parameters."""
    path = Path(checkpoint_path)
    expected_files = {"metadata.json", "rng_state.pt", "training_state.pt"}
    if not path.is_dir() or {item.name for item in path.iterdir()} != expected_files:
        raise ValueError("compact checkpoint files are missing or unexpected")
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("format") != FORMAT or metadata.get("world_size") != 1:
        raise ValueError("unsupported compact checkpoint metadata")
    layout = metadata.get("trainable_layout")
    if not isinstance(layout, list):
        raise ValueError("compact checkpoint trainable layout is missing")

    state = torch.load(
        path / "training_state.pt",
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(state, dict) or state.get("format") != FORMAT:
        raise ValueError("compact checkpoint training state is invalid")
    zero3 = state.get("zero3_optimizer")
    if not isinstance(zero3, dict) or int(zero3.get("partition_count", 0)) != 1:
        raise ValueError("compact checkpoint is not a single-rank ZeRO-3 state")
    flat_groups = zero3.get("fp32_flat_groups")
    if not isinstance(flat_groups, list) or len(flat_groups) != len(layout):
        raise ValueError("compact checkpoint flat groups do not match metadata")

    parameters = dict(transformer.named_parameters())
    seen: set[str] = set()
    with torch.no_grad():
        for entries, flat in zip(layout, flat_groups, strict=True):
            if not isinstance(entries, list) or not torch.is_tensor(flat):
                raise ValueError("compact checkpoint group is invalid")
            offset = 0
            for entry in entries:
                if set(entry) != {"dtype", "name", "numel", "shape"}:
                    raise ValueError("compact checkpoint layout entry is invalid")
                name = entry["name"]
                numel = int(entry["numel"])
                shape = tuple(int(value) for value in entry["shape"])
                parameter = parameters.get(name)
                if parameter is None or name in seen:
                    raise ValueError(f"compact parameter is missing or duplicated: {name}")
                if tuple(parameter.shape) != shape or parameter.numel() != numel:
                    raise ValueError(f"compact parameter shape mismatch: {name}")
                value = flat[offset : offset + numel].reshape(shape)
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
                offset += numel
                seen.add(name)
            if offset != flat.numel():
                raise ValueError("compact flat group has trailing or missing values")
    if len(seen) != 290:
        raise ValueError(f"unexpected compact parameter count: {len(seen)}")

    decay_buffers = 0
    for block in transformer.blocks:
        if hasattr(block, "joint_gate_video_decay"):
            block.joint_gate_video_decay.zero_()
            decay_buffers += 1
    del flat_groups, zero3, state
    gc.collect()
    print(
        f"Loaded {len(seen)} fine-tuned tensors from compact step "
        f"{metadata.get('global_step')}; zeroed {decay_buffers} final decay buffers."
    )
    return len(seen)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--compact_checkpoint", required=True)
    args, inference_args = parser.parse_known_args()
    module = _load_inference_module()

    # The joint transformer is slightly larger than 32 GiB when moved as one
    # unit. Reuse the CLI's CPU-offload branch but install leaf-module hooks so
    # only the currently executing block resides on the GPU.
    def enable_sequential_offload(self, *unused_args, **unused_kwargs):
        return self.enable_sequential_cpu_offload()

    module.RynnWorld4DInferencePipeline.enable_model_cpu_offload = (
        enable_sequential_offload
    )

    def load_combined(checkpoint_path: str, transformer: torch.nn.Module) -> None:
        load_released_weights(checkpoint_path, transformer)
        load_compact_overlay(args.compact_checkpoint, transformer)

    module.load_sft_checkpoint = load_combined
    sys.argv = [str(ROOT / "inference-sft.py"), *inference_args]
    module.main()


if __name__ == "__main__":
    main()
