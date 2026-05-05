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


@dataclass
class EmpiricalCouplingProblem:
    x0_pool: torch.Tensor
    x1_pool: torch.Tensor
    label: str = "empirical"
    global_ot_src_idx: torch.Tensor | None = None
    global_ot_tgt_idx: torch.Tensor | None = None
    global_ot_mass: torch.Tensor | None = None
    global_ot_total_cost: float | None = None

    @property
    def dim(self) -> int:
        if self.x0_pool.ndim != 2 or self.x1_pool.ndim != 2:
            raise ValueError(
                f"Expected empirical pools with shape (N, d), got {self.x0_pool.shape} and {self.x1_pool.shape}"
            )
        if self.x0_pool.shape[1] != self.x1_pool.shape[1]:
            raise ValueError(
                f"Empirical pool dimensions mismatch: {self.x0_pool.shape} vs {self.x1_pool.shape}"
            )
        return int(self.x0_pool.shape[1])

    @property
    def has_global_ot_support(self) -> bool:
        return (
            self.global_ot_src_idx is not None
            and self.global_ot_tgt_idx is not None
            and self.global_ot_mass is not None
            and self.global_ot_src_idx.numel() > 0
            and self.global_ot_tgt_idx.numel() > 0
            and self.global_ot_mass.numel() > 0
        )

    def to(self, device: torch.device, dtype: torch.dtype) -> "EmpiricalCouplingProblem":
        src_idx = None
        if self.global_ot_src_idx is not None:
            src_idx = self.global_ot_src_idx.to(device=device, dtype=torch.long)
        tgt_idx = None
        if self.global_ot_tgt_idx is not None:
            tgt_idx = self.global_ot_tgt_idx.to(device=device, dtype=torch.long)
        mass = None
        if self.global_ot_mass is not None:
            mass = self.global_ot_mass.to(device=device, dtype=dtype)
        return EmpiricalCouplingProblem(
            x0_pool=self.x0_pool.to(device=device, dtype=dtype),
            x1_pool=self.x1_pool.to(device=device, dtype=dtype),
            label=self.label,
            global_ot_src_idx=src_idx,
            global_ot_tgt_idx=tgt_idx,
            global_ot_mass=mass,
            global_ot_total_cost=self.global_ot_total_cost,
        )


CouplingProblem = GaussianOTProblem | EmpiricalCouplingProblem


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


def moment_feature_vector_from_samples(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"Expected x with shape (N, d), got {tuple(x.shape)}")
    mean = x.mean(dim=0)
    centered = x - mean
    cov = centered.T @ centered / x.shape[0]
    return gaussian_moment_feature_vector(mean, cov)


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
    problem: CouplingProblem,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    return sample_coupled_batch(problem=problem, batch_size=batch_size, coupling="ot", generator=generator)


def sample_random_batch(
    problem: CouplingProblem,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    return sample_coupled_batch(problem=problem, batch_size=batch_size, coupling="random", generator=generator)


def _sample_from_empirical_pool(
    pool: torch.Tensor,
    n_samples: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if pool.ndim != 2:
        raise ValueError(f"Expected empirical pool with shape (N, d), got {tuple(pool.shape)}")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if pool.shape[0] <= 0:
        raise ValueError("Empirical pool is empty.")
    idx = torch.randint(
        low=0,
        high=pool.shape[0],
        size=(n_samples,),
        device=pool.device,
        generator=generator,
    )
    return pool[idx]


def sample_coupled_batch(
    problem: CouplingProblem,
    batch_size: int,
    coupling: str,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    coupling_name = coupling.lower()
    if coupling_name == "ot_global":
        if not isinstance(problem, EmpiricalCouplingProblem):
            raise ValueError("coupling='ot_global' is only supported for empirical problems.")
        if not problem.has_global_ot_support:
            raise ValueError(
                "coupling='ot_global' requested but EmpiricalCouplingProblem has no global OT support. "
                "Prepare the dataset with a precomputed global OT plan."
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if problem.global_ot_mass is None or problem.global_ot_src_idx is None or problem.global_ot_tgt_idx is None:
            raise ValueError("Global OT support tensors are missing.")
        support_probs = problem.global_ot_mass
        support_probs = support_probs / torch.clamp(support_probs.sum(), min=torch.finfo(support_probs.dtype).eps)
        support_choice = torch.multinomial(
            support_probs,
            num_samples=batch_size,
            replacement=True,
            generator=generator,
        )
        src_idx = problem.global_ot_src_idx[support_choice]
        tgt_idx = problem.global_ot_tgt_idx[support_choice]
        x0 = problem.x0_pool[src_idx]
        x1 = problem.x1_pool[tgt_idx]
        total_cost = float(torch.sum((x0 - x1) ** 2).item())
        return x0, x1, total_cost

    if isinstance(problem, GaussianOTProblem):
        x0 = sample_gaussian(problem.mean0, problem.cov0, n_samples=batch_size, generator=generator)
        x1 = sample_gaussian(problem.mean1, problem.cov1, n_samples=batch_size, generator=generator)
    elif isinstance(problem, EmpiricalCouplingProblem):
        x0 = _sample_from_empirical_pool(problem.x0_pool, n_samples=batch_size, generator=generator)
        x1 = _sample_from_empirical_pool(problem.x1_pool, n_samples=batch_size, generator=generator)
    else:
        raise TypeError(f"Unsupported problem type: {type(problem)}")
    if coupling_name == "ot":
        return exact_discrete_ot_pairs(x0, x1)
    if coupling_name == "random":
        return random_discrete_pairs(x0, x1, generator=generator)
    raise ValueError(f"Unsupported coupling '{coupling}'. Expected one of: ot, random, ot_global.")


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
