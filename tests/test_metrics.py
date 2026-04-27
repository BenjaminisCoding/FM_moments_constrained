import torch

from cfm_project.data import GaussianOTProblem
from cfm_project.metrics import (
    empirical_w2_distance,
    gaussian_w2_distance,
    intermediate_empirical_w2_metrics,
    intermediate_wasserstein_metrics,
)


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
