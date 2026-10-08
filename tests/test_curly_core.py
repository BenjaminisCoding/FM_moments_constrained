from __future__ import annotations

import torch

from cfm_project.curly_core import (
    curly_learned_coupling,
    curly_mean_path,
    curly_path_and_velocity,
    knn_reference_velocity,
)
from cfm_project.models import PathCorrection


def test_curly_mean_path_preserves_endpoints() -> None:
    torch.manual_seed(3)
    x0 = torch.randn(8, 3)
    x1 = torch.randn(8, 3)
    model = PathCorrection(state_dim=3, hidden_dims=[12], activation="silu")

    t0 = torch.zeros(8, 1)
    t1 = torch.ones(8, 1)

    assert torch.allclose(curly_mean_path(t0, x0, x1, model), x0)
    assert torch.allclose(curly_mean_path(t1, x0, x1, model), x1)


def test_curly_path_and_velocity_matches_linear_when_alpha_zero() -> None:
    x0 = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    x1 = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
    t = torch.tensor([[0.25], [0.75]])
    model = PathCorrection(state_dim=2, hidden_dims=[8], activation="silu")

    xt, ut, t_out = curly_path_and_velocity(
        t=t,
        x0=x0,
        x1=x1,
        geopath_net=model,
        path_alpha=0.0,
        sigma=0.0,
        create_graph=False,
    )

    assert torch.allclose(t_out, t)
    assert torch.allclose(xt, (1.0 - t) * x0 + t * x1)
    assert torch.allclose(ut, x1 - x0)


def test_knn_reference_velocity_uses_nearest_velocity_for_k_one() -> None:
    reference_x = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    reference_v = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]])
    query = torch.tensor([[0.1, 0.0], [1.9, 0.1]])

    velocity = knn_reference_velocity(query, reference_x, reference_v, k=1)

    assert torch.allclose(velocity, torch.tensor([[1.0, 0.0], [0.0, 2.0]]))


def test_curly_learned_coupling_returns_batch_pairing() -> None:
    torch.manual_seed(5)
    x0 = torch.randn(4, 2)
    x1 = torch.randn(4, 2)
    model = PathCorrection(state_dim=2, hidden_dims=[8], activation="silu")
    reference_x = torch.cat([x0, x1], dim=0)
    reference_v = torch.ones_like(reference_x)

    paired_x0, paired_x1, cost = curly_learned_coupling(
        x0=x0,
        x1=x1,
        geopath_net=model,
        reference_x=reference_x,
        reference_v=reference_v,
        k=2,
        path_alpha=1.0,
        sigma=0.0,
        velocity_scale=1.0,
        num_times=1,
        chunk_size=2,
        generator=torch.Generator().manual_seed(11),
    )

    assert paired_x0.shape == x0.shape
    assert paired_x1.shape == x1.shape
    assert cost >= 0.0
