from __future__ import annotations

import pytest
import torch

from cfm_project.metrics import (
    balanced_empirical_w2_distance,
)


def test_weighted_wasserstein_uses_declared_nonuniform_masses() -> None:
    x = torch.tensor([[0.0], [2.0]])
    y = torch.tensor([[0.0], [2.0]])
    x_weights = torch.tensor([0.75, 0.25])
    y_weights = torch.tensor([0.25, 0.75])

    w2 = balanced_empirical_w2_distance(
        x=x,
        y=y,
        x_weights=x_weights,
        y_weights=y_weights,
        method="pot_emd2",
    )

    assert w2 == pytest.approx(2.0**0.5)


def test_weighted_wasserstein_accepts_unequal_support_sizes() -> None:
    x = torch.tensor([[0.0], [1.0], [2.0]])
    y = torch.tensor([[0.0], [2.0]])

    assert balanced_empirical_w2_distance(
        x=x,
        y=y,
        method="pot_emd2",
    ) == pytest.approx((1.0 / 3.0) ** 0.5)
