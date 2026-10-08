"""The fixed-budget option must bypass residual/KKT checkpoint selection."""
from types import SimpleNamespace

import pytest
import torch

from cfm_project import aggregate_benchmarks as core


@pytest.mark.parametrize("early_stopping, expected", [(True, 10), (False, 12)])
def test_stage_a_respects_declared_budget(tmp_path, monkeypatch, early_stopping, expected):
    data = SimpleNamespace(target=torch.zeros(1), source=tmp_path)
    (tmp_path / "train.npz").write_bytes(b"test provenance")
    endpoints = torch.zeros((2, 1))
    monkeypatch.setattr(core, "paired_endpoints", lambda data: (endpoints, endpoints, {}))

    def initialize(*args):
        model = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(model.weight)
        return model, dict(post_warmup_gradient_balance=1., warmup_steps=0)

    def objective(model, *args):
        return model.weight.square().sum(), torch.zeros(1)

    monkeypatch.setattr(core, "initialize_path", initialize)
    monkeypatch.setattr(core, "base_and_residual", objective)
    config = core.default_config()
    config.update(outer_steps=12, inner_steps=1, stage_a_early_stopping=early_stopping)
    _, _, summary = core.fit_path(data, "mfm", 3, config, tmp_path)
    assert summary["outer_steps"] == expected
    assert summary["stable_stopping"] == early_stopping
    assert summary["final"]["total_gradient_norm"] == 0.
    if not early_stopping:
        assert "fixed outer-step budget" in summary["stopping"]
