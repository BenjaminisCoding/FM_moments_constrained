from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from cfm_project.constraints import moment_features_2d
from cfm_project.data import (
    GaussianOTProblem,
    analytic_bridge_cov,
    analytic_bridge_mean,
    matrix_sqrt_psd,
    sample_gaussian,
)


@torch.no_grad()
def covariance(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=0)
    centered = x - mean
    return centered.T @ centered / x.shape[0]


@torch.no_grad()
def euler_integrate_velocity(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    n_steps: int = 100,
) -> torch.Tensor:
    x = x0.clone()
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t_val = torch.full((x.shape[0], 1), float(step) * dt, device=x.device, dtype=x.dtype)
        x = x + dt * velocity_fn(t_val, x)
    return x


@torch.no_grad()
def euler_velocity_snapshots(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    times: list[float],
    n_steps: int,
) -> dict[float, torch.Tensor]:
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    unique_times = sorted({float(t) for t in times})
    for t in unique_times:
        if t < 0.0 or t > 1.0:
            raise ValueError(f"Snapshot times must be in [0, 1], got {t}")

    step_to_times: dict[int, list[float]] = {}
    for t in unique_times:
        step = int(round(t * n_steps))
        step = max(0, min(n_steps, step))
        step_to_times.setdefault(step, []).append(t)

    snapshots: dict[float, torch.Tensor] = {}
    x = x0.clone()
    dt = 1.0 / n_steps
    for step in range(n_steps + 1):
        for t in step_to_times.get(step, []):
            snapshots[float(t)] = x.clone()
        if step == n_steps:
            break
        t_val = torch.full((x.shape[0], 1), float(step) * dt, device=x.device, dtype=x.dtype)
        x = x + dt * velocity_fn(t_val, x)
    return snapshots


@torch.no_grad()
def gaussian_w2_distance(
    mean_a: torch.Tensor,
    cov_a: torch.Tensor,
    mean_b: torch.Tensor,
    cov_b: torch.Tensor,
) -> float:
    diff = mean_a - mean_b
    mean_term = torch.dot(diff, diff)
    cov_b_sqrt = matrix_sqrt_psd(cov_b)
    middle = cov_b_sqrt @ cov_a @ cov_b_sqrt
    middle_sqrt = matrix_sqrt_psd(middle)
    trace_term = torch.trace(cov_a + cov_b - 2.0 * middle_sqrt)
    w2_sq = torch.clamp(mean_term + trace_term, min=0.0)
    return float(torch.sqrt(w2_sq).item())


@torch.no_grad()
def empirical_w2_distance(
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}")
    if x.ndim != 2:
        raise ValueError(f"Expected x and y to have shape (N, d), got {x.shape}")
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    cost = ((x_np[:, None, :] - y_np[None, :, :]) ** 2).sum(axis=-1)
    row_ind, col_ind = linear_sum_assignment(cost)
    w2_sq = float(cost[row_ind, col_ind].mean())
    return float(np.sqrt(max(w2_sq, 0.0)))


@torch.no_grad()
def transport_quality_metrics(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    problem: GaussianOTProblem,
    n_samples: int,
    n_steps: int,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    x0 = sample_gaussian(problem.mean0, problem.cov0, n_samples=n_samples, generator=generator)
    x1_hat = euler_integrate_velocity(velocity_fn, x0, n_steps=n_steps)
    mean_hat = x1_hat.mean(dim=0)
    cov_hat = covariance(x1_hat)

    mean_err = torch.linalg.norm(mean_hat - problem.mean1).item()
    cov_err = torch.linalg.norm(cov_hat - problem.cov1, ord="fro").item()
    return {
        "transport_mean_error_l2": float(mean_err),
        "transport_cov_error_fro": float(cov_err),
        "transport_score": float(mean_err + cov_err),
    }


@torch.no_grad()
def intermediate_wasserstein_metrics(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    problem: GaussianOTProblem,
    times: list[float],
    n_samples: int,
    n_steps: int,
    generator: torch.Generator | None = None,
) -> dict[str, float | dict[str, float]]:
    x0 = sample_gaussian(problem.mean0, problem.cov0, n_samples=n_samples, generator=generator)
    snapshots = euler_velocity_snapshots(
        velocity_fn=velocity_fn,
        x0=x0,
        times=times,
        n_steps=n_steps,
    )

    per_time: dict[str, float] = {}
    for t in sorted({float(ti) for ti in times}):
        xt = snapshots[float(t)]
        mean_hat = xt.mean(dim=0)
        cov_hat = covariance(xt)
        mean_target = analytic_bridge_mean(t, problem)
        cov_target = analytic_bridge_cov(t, problem)
        per_time[f"{t:.2f}"] = gaussian_w2_distance(
            mean_a=mean_hat,
            cov_a=cov_hat,
            mean_b=mean_target,
            cov_b=cov_target,
        )

    avg_value = float(sum(per_time.values()) / len(per_time)) if per_time else 0.0
    return {
        "intermediate_w2_gaussian": per_time,
        "intermediate_w2_gaussian_avg": avg_value,
    }


@torch.no_grad()
def intermediate_empirical_w2_metrics(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    problem: GaussianOTProblem,
    times: list[float],
    n_samples: int,
    n_steps: int,
    target_sampler: Callable[[float, int, torch.Generator | None], torch.Tensor] | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, float | dict[str, float]]:
    x0 = sample_gaussian(problem.mean0, problem.cov0, n_samples=n_samples, generator=generator)
    snapshots = euler_velocity_snapshots(
        velocity_fn=velocity_fn,
        x0=x0,
        times=times,
        n_steps=n_steps,
    )

    per_time: dict[str, float] = {}
    for t in sorted({float(ti) for ti in times}):
        xt = snapshots[float(t)]
        if target_sampler is None:
            target_mean = analytic_bridge_mean(t, problem)
            target_cov = analytic_bridge_cov(t, problem)
            target_samples = sample_gaussian(
                target_mean,
                target_cov,
                n_samples=n_samples,
                generator=generator,
            )
        else:
            target_samples = target_sampler(t, n_samples, generator)
            if target_samples.shape != xt.shape:
                raise ValueError(
                    f"target_sampler returned shape {target_samples.shape}, expected {xt.shape}"
                )
        per_time[f"{t:.2f}"] = empirical_w2_distance(xt, target_samples)

    avg_value = float(sum(per_time.values()) / len(per_time)) if per_time else 0.0
    return {
        "intermediate_empirical_w2": per_time,
        "intermediate_empirical_w2_avg": avg_value,
    }


@torch.no_grad()
def path_energy_proxy(
    velocity_targets: torch.Tensor,
) -> float:
    return float(torch.mean(torch.sum(velocity_targets**2, dim=1)).item())


@torch.no_grad()
def feature_residual_norm(
    x: torch.Tensor,
    target_feature: torch.Tensor,
) -> float:
    residual = moment_features_2d(x) - target_feature
    return float(torch.linalg.norm(residual).item())
