import importlib.util

import pytest
import torch

from cfm_project.constraints import update_lagrange_multipliers
from cfm_project.data import EmpiricalCouplingProblem
from cfm_project.mfm_core import (
    build_metric_backend,
    land_geopath_loss,
    mfm_path_and_velocity,
    resolve_mfm_backend,
)
from cfm_project.models import PathCorrection
from cfm_project.training import (
    _build_metric_reference_pool,
    _init_lambdas,
    _metric_constrained_geopath_objective,
    _metric_geopath_objective,
)


def test_mfm_alpha0_path_and_velocity_match_linear_cfm() -> None:
    torch.manual_seed(101)
    x0 = torch.randn(64, 2)
    x1 = torch.randn(64, 2)
    t = torch.rand(64, 1)

    mu_t, u_t, _ = mfm_path_and_velocity(
        t=t,
        x0=x0,
        x1=x1,
        geopath_net=None,
        alpha=0.0,
        create_graph=False,
    )

    linear = (1.0 - t) * x0 + t * x1
    base_velocity = x1 - x0
    assert torch.allclose(mu_t, linear)
    assert torch.allclose(u_t, base_velocity)


def test_land_loss_is_finite_on_random_inputs() -> None:
    torch.manual_seed(102)
    x_t = torch.randn(80, 2)
    u_t = torch.randn(80, 2)
    manifold_samples = torch.randn(240, 2)

    loss = land_geopath_loss(
        x_t=x_t,
        u_t=u_t,
        manifold_samples=manifold_samples,
        gamma=0.125,
        rho=1e-3,
    )

    assert torch.isfinite(loss).item()
    assert float(loss.item()) >= 0.0


def test_backend_resolution_auto_and_explicit_torchcfm_behavior() -> None:
    selected = resolve_mfm_backend("auto")
    assert selected in {"native", "torchcfm"}

    torchcfm_available = importlib.util.find_spec("torchcfm") is not None
    if torchcfm_available:
        assert resolve_mfm_backend("torchcfm") == "torchcfm"
    else:
        with pytest.raises(RuntimeError):
            _ = resolve_mfm_backend("torchcfm")


def test_build_metric_backend_auto_constructs_matcher() -> None:
    backend = build_metric_backend(
        requested_backend="auto",
        geopath_net=None,
        sigma=0.1,
        alpha=0.0,
    )
    assert backend.name in {"native", "torchcfm"}


def test_reference_pool_policy_endpoints_only_vs_anchors_all() -> None:
    problem = EmpiricalCouplingProblem(
        x0_pool=torch.zeros(20, 2),
        x1_pool=torch.ones(20, 2) * 10.0,
        label="test",
    )
    generator = torch.Generator(device=torch.device("cpu"))
    generator.manual_seed(7)

    def target_sampler(t: float, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        _ = generator
        marker = float(round(t * 100.0))
        return torch.full((n, 2), marker, dtype=torch.float32)

    endpoints_only = _build_metric_reference_pool(
        problem=problem,
        target_sampler=target_sampler,
        times=[0.25, 0.5, 0.75],
        n_samples_per_time=4,
        generator=generator,
        reference_pool_policy="endpoints_only",
    )
    anchors_all = _build_metric_reference_pool(
        problem=problem,
        target_sampler=target_sampler,
        times=[0.25, 0.5, 0.75],
        n_samples_per_time=4,
        generator=generator,
        reference_pool_policy="anchors_all",
    )

    assert endpoints_only.shape[0] == 8
    assert anchors_all.shape[0] == 20
    assert set(torch.unique(endpoints_only[:, 0]).tolist()) == {0.0, 100.0}
    assert set(torch.unique(anchors_all[:, 0]).tolist()) == {0.0, 25.0, 50.0, 75.0, 100.0}


def test_metric_constrained_al_objective_finite_and_lambda_clipping() -> None:
    torch.manual_seed(11)
    x0 = torch.randn(48, 2)
    x1 = torch.randn(48, 2)
    model = PathCorrection(state_dim=2, hidden_dims=(16, 16))
    times = [0.25, 0.5, 0.75]
    targets = {t: torch.zeros(6) for t in times}
    lambdas = _init_lambdas(times=times, targets=targets)
    manifold_samples = torch.randn(128, 2)

    loss, residuals, _ = _metric_constrained_geopath_objective(
        mode="metric_constrained_al",
        geopath_model=model,
        alpha_mfm=1.0,
        x0=x0,
        x1=x1,
        manifold_samples=manifold_samples,
        times=times,
        targets=targets,
        lambdas=lambdas,
        rho=1.0,
        land_gamma=0.125,
        land_rho=1e-3,
        moment_eta=1.0,
    )
    assert torch.isfinite(loss).item()

    updated = update_lagrange_multipliers(
        lambdas=lambdas,
        residuals=residuals,
        rho=1.0,
        clip_value=0.05,
    )
    all_vals = torch.cat([v.reshape(-1) for v in updated.values()])
    assert torch.max(torch.abs(all_vals)).item() <= 0.05 + 1e-8


def test_metric_soft_objective_monotonic_in_eta_and_eta0_matches_land_only() -> None:
    torch.manual_seed(12)
    x0 = torch.randn(64, 2)
    x1 = torch.randn(64, 2)
    model = PathCorrection(state_dim=2, hidden_dims=(12, 12))
    times = [0.25, 0.5, 0.75]
    targets = {t: torch.zeros(6) for t in times}
    manifold_samples = torch.randn(200, 2)

    torch.manual_seed(999)
    land_only, _ = _metric_geopath_objective(
        geopath_model=model,
        alpha_mfm=1.0,
        x0=x0,
        x1=x1,
        manifold_samples=manifold_samples,
        land_gamma=0.125,
        land_rho=1e-3,
    )
    torch.manual_seed(999)
    eta0, _, _ = _metric_constrained_geopath_objective(
        mode="metric_constrained_soft",
        geopath_model=model,
        alpha_mfm=1.0,
        x0=x0,
        x1=x1,
        manifold_samples=manifold_samples,
        times=times,
        targets=targets,
        lambdas={},
        rho=1.0,
        land_gamma=0.125,
        land_rho=1e-3,
        moment_eta=0.0,
    )
    torch.manual_seed(999)
    eta1, _, _ = _metric_constrained_geopath_objective(
        mode="metric_constrained_soft",
        geopath_model=model,
        alpha_mfm=1.0,
        x0=x0,
        x1=x1,
        manifold_samples=manifold_samples,
        times=times,
        targets=targets,
        lambdas={},
        rho=1.0,
        land_gamma=0.125,
        land_rho=1e-3,
        moment_eta=1.0,
    )

    assert torch.allclose(eta0, land_only, atol=1e-6, rtol=1e-6)
    assert float(eta1.item()) >= float(eta0.item()) - 1e-8
