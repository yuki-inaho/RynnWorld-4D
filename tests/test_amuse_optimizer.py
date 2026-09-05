import copy
import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn

from core.finetune.optim import (
    build_amuse_optimizer,
    classify_amuse_parameters,
    find_amuse_optimizer,
)


class TinyAmuseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Parameter(torch.zeros(1, 4))
        self.hidden = nn.Linear(4, 4)
        self.joint_out = nn.Linear(4, 2)

    def forward(self, inputs):
        return self.joint_out(torch.tanh(self.hidden(inputs) + self.token))


def build(model):
    return build_amuse_optimizer(
        model,
        total_optimizer_steps=20,
        muon_lr=1e-3,
        aux_lr=1e-4,
        warmup_ratio=0.1,
        fallback_patterns=("*joint_out.weight",),
    )


def step(model, optimizer, inputs, targets):
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()
    return loss.detach()


def assert_nested_close(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0.0, atol=1e-7)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            assert_nested_close(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            assert_nested_close(left_item, right_item)
    else:
        assert left == right


def test_vendored_amuse_matches_pinned_upstream_blob():
    root = Path(__file__).parents[1]
    source = (root / "core/finetune/optim/amuse.py").read_bytes()
    git_blob = hashlib.sha1(f"blob {len(source)}\0".encode() + source, usedforsecurity=False).hexdigest()
    assert git_blob == "144361bf100d0a3a07172fb007a6fb27ff58f046"
    assert "48922743b32f33f919ab54edde3dbad0d0ce2dc7" in (root / "third_party/amuse/UPSTREAM.md").read_text()
    assert (root / "third_party/amuse/LICENSE").read_text().startswith("                                 Apache License")


def test_parameter_partition_is_complete_disjoint_and_tiered():
    model = TinyAmuseModel()
    model.joint_out.bias.requires_grad_(False)
    grouping = classify_amuse_parameters(model, fallback_patterns=("*joint_out.weight",))
    assert grouping.muon_names == ("hidden.weight",)
    assert set(grouping.fallback_names) == {"token", "hidden.bias", "joint_out.weight"}
    assert grouping.frozen_names == ("joint_out.bias",)
    assert len(grouping.fingerprint) == 64
    result = build(model)
    assert {group["name"] for group in result.optimizer.param_groups} == {
        "amuse_muon_base_0000",
        "amuse_aux_base_0000",
        "amuse_aux_joint_out_0000",
    }


def test_amuse_requires_train_mode_and_round_trips_eval_weights():
    torch.manual_seed(7)
    model = TinyAmuseModel()
    optimizer = build(model).optimizer
    inputs, targets = torch.randn(3, 4), torch.randn(3, 2)
    with pytest.raises(Exception, match="train mode"):
        step(model, optimizer, inputs, targets)
    optimizer.train()
    assert torch.isfinite(step(model, optimizer, inputs, targets))
    assert torch.isfinite(step(model, optimizer, inputs * 0.75, targets))
    training = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer.eval()
    evaluation = {name: value.detach().clone() for name, value in model.named_parameters()}
    assert any(not torch.equal(training[name], evaluation[name]) for name in training)
    optimizer.train()
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, training[name])


def test_amuse_eval_checkpoint_resume_matches_next_step():
    torch.manual_seed(11)
    model = TinyAmuseModel()
    optimizer = build(model).optimizer
    optimizer.train()
    inputs, targets = torch.randn(3, 4), torch.randn(3, 2)
    step(model, optimizer, inputs, targets)
    optimizer.eval()
    model_state, optimizer_state = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
    optimizer.train()
    step(model, optimizer, inputs * 1.5, targets)

    resumed = TinyAmuseModel()
    resumed_optimizer = build(resumed).optimizer
    resumed.load_state_dict(model_state)
    resumed_optimizer.load_state_dict(optimizer_state)
    resumed_optimizer.train()
    step(resumed, resumed_optimizer, inputs * 1.5, targets)
    assert_nested_close(model.state_dict(), resumed.state_dict())
    assert_nested_close(optimizer.state_dict(), resumed_optimizer.state_dict())


def test_wrapper_lookup_is_explicit_and_unknown_optimizer_has_no_fallback():
    optimizer = build(TinyAmuseModel()).optimizer
    wrapper = type("Wrapper", (), {"optimizer": optimizer})()
    assert find_amuse_optimizer(wrapper) is optimizer
    with pytest.raises(RuntimeError, match="not reachable"):
        find_amuse_optimizer(torch.optim.AdamW(TinyAmuseModel().parameters()))
