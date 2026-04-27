from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def matrix_sqrt_psd(matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    evals, evecs = torch.linalg.eigh(matrix)
    evals = torch.clamp(evals, min=eps)
    sqrt_diag = torch.diag(torch.sqrt(evals))
    return evecs @ sqrt_diag @ evecs.T


def matrix_inv_sqrt_psd(matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    evals, evecs = torch.linalg.eigh(matrix)
    evals = torch.clamp(evals, min=eps)
    inv_sqrt_diag = torch.diag(1.0 / torch.sqrt(evals))
    return evecs @ inv_sqrt_diag @ evecs.T


def gaussian_ot_map_matrix(cov0: torch.Tensor, cov1: torch.Tensor) -> torch.Tensor:
    cov0_sqrt = matrix_sqrt_psd(cov0)
    cov0_inv_sqrt = matrix_inv_sqrt_psd(cov0)
    middle = cov0_sqrt @ cov1 @ cov0_sqrt
    middle_sqrt = matrix_sqrt_psd(middle)
    return cov0_inv_sqrt @ middle_sqrt @ cov0_inv_sqrt


@dataclass
class GaussianOTProblem:
    mean0: torch.Tensor
    cov0: torch.Tensor
    mean1: torch.Tensor
    cov1: torch.Tensor
    kappa: float

    @property
    def dim(self) -> int:
        return int(self.mean0.shape[0])

    def to(self, device: torch.device, dtype: torch.dtype) -> "GaussianOTProblem":
        return GaussianOTProblem(
            mean0=self.mean0.to(device=device, dtype=dtype),
            cov0=self.cov0.to(device=device, dtype=dtype),
            mean1=self.mean1.to(device=device, dtype=dtype),
            cov1=self.cov1.to(device=device, dtype=dtype),
            kappa=float(self.kappa),
        )


def sample_gaussian(
    mean: torch.Tensor,
    cov: torch.Tensor,
    n_samples: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    chol = torch.linalg.cholesky(cov)
    eps = torch.randn(
        n_samples,
        mean.shape[0],
        device=mean.device,
        dtype=mean.dtype,
        generator=generator,
    )
    return mean + eps @ chol.T


def exact_discrete_ot_pairs(
    x0: torch.Tensor,
    x1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if x0.shape != x1.shape:
        raise ValueError(f"x0 and x1 must have same shape, got {x0.shape} and {x1.shape}")
    x0_np = x0.detach().cpu().numpy()
    x1_np = x1.detach().cpu().numpy()
    cost = ((x0_np[:, None, :] - x1_np[None, :, :]) ** 2).sum(axis=-1)
    row_ind, col_ind = linear_sum_assignment(cost)
    total_cost = float(cost[row_ind, col_ind].sum())
    device = x0.device
    paired_x0 = x0[torch.as_tensor(row_ind, device=device)]
    paired_x1 = x1[torch.as_tensor(col_ind, device=device)]
    return paired_x0, paired_x1, total_cost


def random_discrete_pairs(
    x0: torch.Tensor,
    x1: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if x0.shape != x1.shape:
        raise ValueError(f"x0 and x1 must have same shape, got {x0.shape} and {x1.shape}")
    perm = torch.randperm(x1.shape[0], device=x1.device, generator=generator)
    paired_x1 = x1[perm]
    total_cost = float(torch.sum((x0 - paired_x1) ** 2).item())
    return x0, paired_x1, total_cost


def nonlinear_scale(t: float | torch.Tensor, kappa: float) -> float | torch.Tensor:
    return 1.0 + kappa * t * (1.0 - t)


def analytic_bridge_mean(t: float, problem: GaussianOTProblem) -> torch.Tensor:
    return (1.0 - t) * problem.mean0 + t * problem.mean1


def analytic_bridge_cov(t: float, problem: GaussianOTProblem) -> torch.Tensor:
    dim = problem.dim
    eye = torch.eye(dim, device=problem.mean0.device, dtype=problem.mean0.dtype)
    a_map = gaussian_ot_map_matrix(problem.cov0, problem.cov1)
    bt = (1.0 - t) * eye + t * a_map
    scale = nonlinear_scale(t, problem.kappa)
    return (scale**2) * bt @ problem.cov0 @ bt.T


def gaussian_moment_feature_vector(mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    return torch.cat([mean, cov.reshape(-1)], dim=0)


def analytic_target_moment_features(
    times: Sequence[float],
    problem: GaussianOTProblem,
) -> dict[float, torch.Tensor]:
    targets: dict[float, torch.Tensor] = {}
    for t in times:
        mean_t = analytic_bridge_mean(t, problem)
        cov_t = analytic_bridge_cov(t, problem)
        targets[float(t)] = gaussian_moment_feature_vector(mean_t, cov_t)
    return targets


def sample_exact_ot_batch(
    problem: GaussianOTProblem,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    return sample_coupled_batch(problem=problem, batch_size=batch_size, coupling="ot", generator=generator)


def sample_random_batch(
    problem: GaussianOTProblem,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    return sample_coupled_batch(problem=problem, batch_size=batch_size, coupling="random", generator=generator)


def sample_coupled_batch(
    problem: GaussianOTProblem,
    batch_size: int,
    coupling: str,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    x0 = sample_gaussian(problem.mean0, problem.cov0, n_samples=batch_size, generator=generator)
    x1 = sample_gaussian(problem.mean1, problem.cov1, n_samples=batch_size, generator=generator)
    coupling_name = coupling.lower()
    if coupling_name == "ot":
        return exact_discrete_ot_pairs(x0, x1)
    if coupling_name == "random":
        return random_discrete_pairs(x0, x1, generator=generator)
    raise ValueError(f"Unsupported coupling '{coupling}'. Expected one of: ot, random.")


def to_problem_from_config(
    mean0: Sequence[float],
    cov0: Sequence[Sequence[float]],
    mean1: Sequence[float],
    cov1: Sequence[Sequence[float]],
    kappa: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> GaussianOTProblem:
    return GaussianOTProblem(
        mean0=torch.tensor(mean0, dtype=dtype, device=device),
        cov0=torch.tensor(cov0, dtype=dtype, device=device),
        mean1=torch.tensor(mean1, dtype=dtype, device=device),
        cov1=torch.tensor(cov1, dtype=dtype, device=device),
        kappa=float(kappa),
    )
