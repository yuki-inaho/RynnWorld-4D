import pytest
import torch

from core.finetune.utils.optimizer_utils import get_optimizer


def test_unknown_optimizer_does_not_fallback_to_adamw():
    parameter = torch.nn.Parameter(torch.ones(2))
    with pytest.raises(ValueError, match="Unsupported choice"):
        get_optimizer([parameter], optimizer_name="amues")


def test_explicit_adamw_still_works():
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = get_optimizer([parameter], optimizer_name="adamw")
    assert isinstance(optimizer, torch.optim.AdamW)
