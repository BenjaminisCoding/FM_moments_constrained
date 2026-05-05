import torch
import pytest

from cfm_project.data import GaussianOTProblem
from cfm_project.metrics import (
    balanced_empirical_w2_distance,
    empirical_w1_distance,
    empirical_w2_distance,
    gaussian_w2_distance,
    intermediate_empirical_w2_metrics,
    intermediate_wasserstein_metrics,
    interpolant_empirical_w2_metrics,
)
from cfm_project.paths import linear_path


def test_gaussian_w2_distance_is_zero_for_identical_gaussians() -> None:
    mean = torch.tensor([0.5, -1.0])
    cov = torch.tensor([[1.2, 0.3], [0.3, 0.8]])

    value = gaussian_w2_distance(mean, cov, mean, cov)

    assert value <= 1e-3


def test_gaussian_w2_distance_matches_mean_shift_when_covariances_match() -> None:
    mean_a = torch.tensor([0.0, 0.0])
    mean_b = torch.tensor([3.0, 4.0])
    cov = torch.eye(2)

    value = gaussian_w2_distance(mean_a, cov, mean_b, cov)

    assert abs(value - 5.0) <= 1e-6


def test_intermediate_wasserstein_metrics_small_for_static_matching_problem() -> None:
    problem = GaussianOTProblem(
        mean0=torch.zeros(2),
        cov0=torch.eye(2),
        mean1=torch.zeros(2),
        cov1=torch.eye(2),
        kappa=0.0,
    )
    generator = torch.Generator()
    generator.manual_seed(123)

    def zero_velocity(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del t
        return torch.zeros_like(x)

    metrics = intermediate_wasserstein_metrics(
        velocity_fn=zero_velocity,
        problem=problem,
        times=[0.25, 0.5, 0.75],
        n_samples=4096,
        n_steps=32,
        generator=generator,
    )

    assert metrics["intermediate_w2_gaussian_avg"] < 0.15
    per_time = metrics["intermediate_w2_gaussian"]
    assert isinstance(per_time, dict)
    assert set(per_time.keys()) == {"0.25", "0.50", "0.75"}


def test_empirical_w2_distance_zero_for_identical_point_clouds() -> None:
    torch.manual_seed(77)
    x = torch.randn(64, 2)

    value = empirical_w2_distance(x, x.clone())

    assert value <= 1e-8


def test_empirical_w2_distance_matches_shift_for_paired_clouds() -> None:
    torch.manual_seed(78)
    x = torch.randn(96, 2)
    shift = torch.tensor([3.0, 4.0])
    y = x + shift

    value = empirical_w2_distance(x, y)

    assert abs(value - 5.0) <= 1e-6


def test_empirical_w1_distance_matches_shift_for_paired_clouds() -> None:
    torch.manual_seed(79)
    x = torch.randn(96, 2)
    shift = torch.tensor([3.0, 4.0])
    y = x + shift

    value = empirical_w1_distance(x, y)

    assert abs(value - 5.0) <= 1e-6


def test_balanced_empirical_w2_distance_supports_weighted_masses() -> None:
    x = torch.tensor([[0.0], [10.0]])
    y = torch.tensor([[1.0], [9.0]])
    weights = torch.tensor([0.75, 0.25])

    value = balanced_empirical_w2_distance(
        x=x,
        y=y,
        x_weights=weights,
        y_weights=weights,
    )

    assert abs(value - 1.0) <= 1e-6


def test_balanced_empirical_w2_distance_supports_rectangular_sizes() -> None:
    x = torch.tensor([[0.0], [2.0]])
    y = torch.tensor([[1.0]])
    x_weights = torch.tensor([0.5, 0.5])

    value = balanced_empirical_w2_distance(
        x=x,
        y=y,
        x_weights=x_weights,
        y_weights=torch.tensor([1.0]),
    )

    assert abs(value - 1.0) <= 1e-6


def test_balanced_empirical_w2_distance_pot_matches_exact_lp() -> None:
    pytest.importorskip("ot")
    x = torch.tensor([[0.0], [10.0]])
    y = torch.tensor([[1.0], [9.0]])
    weights = torch.tensor([0.75, 0.25])

    value_exact = balanced_empirical_w2_distance(
        x=x,
        y=y,
        x_weights=weights,
        y_weights=weights,
        method="exact_lp",
    )
    value_pot = balanced_empirical_w2_distance(
        x=x,
        y=y,
        x_weights=weights,
        y_weights=weights,
        method="pot_emd2",
        num_itermax=200000,
    )

    assert abs(value_exact - value_pot) <= 1e-8


def test_intermediate_empirical_w2_metrics_small_for_static_matching_problem() -> None:
    problem = GaussianOTProblem(
        mean0=torch.zeros(2),
        cov0=torch.eye(2),
        mean1=torch.zeros(2),
        cov1=torch.eye(2),
        kappa=0.0,
    )
    generator = torch.Generator()
    generator.manual_seed(124)

    def zero_velocity(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del t
        return torch.zeros_like(x)

    metrics = intermediate_empirical_w2_metrics(
        velocity_fn=zero_velocity,
        problem=problem,
        times=[0.25, 0.5, 0.75],
        n_samples=128,
        n_steps=24,
        generator=generator,
    )

    assert metrics["intermediate_empirical_w2_avg"] < 0.8
    per_time = metrics["intermediate_empirical_w2"]
    assert isinstance(per_time, dict)
    assert set(per_time.keys()) == {"0.25", "0.50", "0.75"}


def test_interpolant_empirical_w2_metrics_keys_and_finite_values() -> None:
    torch.manual_seed(125)
    x0 = torch.randn(96, 2)
    x1 = torch.randn(96, 2)
    times = [0.25, 0.5, 0.75]

    def target_sampler(t: float, n_samples: int, generator: torch.Generator | None) -> torch.Tensor:
        del generator
        t_batch = torch.full((x0.shape[0], 1), float(t), device=x0.device, dtype=x0.dtype)
        target = linear_path(t_batch, x0, x1)
        return target[:n_samples]

    metrics = interpolant_empirical_w2_metrics(
        x0=x0,
        x1=x1,
        times=times,
        target_sampler=target_sampler,
        g_model=None,
        generator=None,
    )

    assert set(metrics.keys()) == {
        "linear_empirical_w2",
        "linear_empirical_w2_avg",
        "learned_empirical_w2",
        "learned_empirical_w2_avg",
        "delta_avg_learned_minus_linear",
    }
    linear_per_time = metrics["linear_empirical_w2"]
    learned_per_time = metrics["learned_empirical_w2"]
    assert isinstance(linear_per_time, dict)
    assert isinstance(learned_per_time, dict)
    assert set(linear_per_time.keys()) == {"0.25", "0.50", "0.75"}
    assert set(learned_per_time.keys()) == {"0.25", "0.50", "0.75"}
    assert torch.isfinite(torch.tensor(metrics["linear_empirical_w2_avg"])).item()
    assert torch.isfinite(torch.tensor(metrics["learned_empirical_w2_avg"])).item()


def test_interpolant_empirical_w2_metrics_include_holdout_fields_when_requested() -> None:
    torch.manual_seed(126)
    x0 = torch.randn(80, 2)
    x1 = torch.randn(80, 2)
    times = [0.25, 0.75]
    holdout_time = 0.5

    def target_sampler(t: float, n_samples: int, generator: torch.Generator | None) -> torch.Tensor:
        del generator
        t_batch = torch.full((x0.shape[0], 1), float(t), device=x0.device, dtype=x0.dtype)
        target = linear_path(t_batch, x0, x1)
        return target[:n_samples]

    metrics = interpolant_empirical_w2_metrics(
        x0=x0,
        x1=x1,
        times=times,
        target_sampler=target_sampler,
        g_model=None,
        mode="metric_alpha0",
        mfm_alpha=0.0,
        holdout_time=holdout_time,
    )

    assert set(metrics["linear_empirical_w2"].keys()) == {"0.25", "0.75"}
    assert set(metrics["learned_empirical_w2"].keys()) == {"0.25", "0.75"}
    assert "linear_holdout_empirical_w2" in metrics
    assert "learned_holdout_empirical_w2" in metrics
    assert "delta_holdout_learned_minus_linear" in metrics
    assert abs(float(metrics["delta_holdout_learned_minus_linear"])) <= 1e-6
