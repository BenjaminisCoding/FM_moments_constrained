from __future__ import annotations

from typing import Callable

import torch


def moment_features(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"Expected x with shape (N, d), got {tuple(x.shape)}")
    if x.shape[0] <= 0:
        raise ValueError("moment_features requires at least one sample.")
    mean = x.mean(dim=0)
    centered = x - mean
    cov = centered.T @ centered / x.shape[0]
    return torch.cat([mean, cov.reshape(-1)], dim=0)


def moment_features_2d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"Expected x with shape (N, 2), got {tuple(x.shape)}")
    return moment_features(x)


def residual_from_samples(
    x: torch.Tensor,
    target_feature: torch.Tensor,
) -> torch.Tensor:
    return moment_features(x) - target_feature


def constraint_residuals(
    path_fn: Callable[[float], torch.Tensor],
    times: list[float],
    targets: dict[float, torch.Tensor],
) -> dict[float, torch.Tensor]:
    residuals: dict[float, torch.Tensor] = {}
    for t in times:
        xt = path_fn(float(t))
        residuals[float(t)] = residual_from_samples(xt, targets[float(t)])
    return residuals


def residual_norms(residuals: dict[float, torch.Tensor]) -> dict[float, float]:
    return {float(t): float(torch.linalg.norm(res).item()) for t, res in residuals.items()}


def augmented_lagrangian_terms(
    residuals: dict[float, torch.Tensor],
    lambdas: dict[float, torch.Tensor],
    rho: float,
) -> tuple[torch.Tensor, dict[float, float]]:
    total = torch.zeros((), device=next(iter(residuals.values())).device)
    per_time: dict[float, float] = {}
    for t, res in residuals.items():
        lam = lambdas[float(t)]
        term = torch.dot(lam, res) + 0.5 * rho * torch.dot(res, res)
        total = total + term
        per_time[float(t)] = float(term.detach().item())
    return total, per_time


def update_lagrange_multipliers(
    lambdas: dict[float, torch.Tensor],
    residuals: dict[float, torch.Tensor],
    rho: float,
    clip_value: float | None = None,
) -> dict[float, torch.Tensor]:
    updated: dict[float, torch.Tensor] = {}
    for t, lam in lambdas.items():
        new_lam = lam + rho * residuals[float(t)].detach()
        if clip_value is not None:
            new_lam = torch.clamp(new_lam, -clip_value, clip_value)
        updated[float(t)] = new_lam
    return updated
