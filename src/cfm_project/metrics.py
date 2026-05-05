from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from cfm_project.constraints import moment_features
from cfm_project.data import (
    GaussianOTProblem,
    analytic_bridge_cov,
    analytic_bridge_mean,
    matrix_sqrt_psd,
    sample_gaussian,
)
from cfm_project.mfm_core import mfm_mean_path
from cfm_project.models import PathCorrection
from cfm_project.ot_utils import (
    balanced_empirical_w2_distance_exact,
    balanced_empirical_w2_distance_pot,
)
from cfm_project.paths import corrected_path, linear_path


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
def balanced_empirical_w2_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    x_weights: torch.Tensor | None = None,
    y_weights: torch.Tensor | None = None,
    method: str = "exact_lp",
    num_itermax: int | None = None,
    max_variables: int | None = None,
    support_tol: float = 1e-12,
) -> float:
    method_norm = str(method).strip().lower()
    if method_norm == "exact_lp":
        return balanced_empirical_w2_distance_exact(
            x=x,
            y=y,
            src_weights=x_weights,
            tgt_weights=y_weights,
            support_tol=float(support_tol),
            max_variables=max_variables,
        )
    if method_norm == "pot_emd2":
        return balanced_empirical_w2_distance_pot(
            x=x,
            y=y,
            src_weights=x_weights,
            tgt_weights=y_weights,
            num_itermax=num_itermax,
        )
    raise ValueError(
        f"Unsupported balanced_empirical_w2_distance method '{method}'. "
        "Expected one of: exact_lp, pot_emd2."
    )


@torch.no_grad()
def empirical_w1_distance(
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}")
    if x.ndim != 2:
        raise ValueError(f"Expected x and y to have shape (N, d), got {x.shape}")
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    cost = np.linalg.norm(x_np[:, None, :] - y_np[None, :, :], axis=-1)
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(cost[row_ind, col_ind].mean())


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


INTERPOLANT_METRIC_MODES = {
    "metric",
    "metric_alpha0",
    "metric_constrained_al",
    "metric_constrained_soft",
}


def _interpolant_samples_for_mode(
    *,
    mode: str,
    t_batch: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
    g_model: PathCorrection | None,
    mfm_alpha: float,
) -> torch.Tensor:
    mode_name = str(mode).strip().lower()
    if mode_name in {"baseline", "metric_alpha0"}:
        return linear_path(t_batch, x0, x1)
    if mode_name in INTERPOLANT_METRIC_MODES:
        return mfm_mean_path(
            t=t_batch,
            x0=x0,
            x1=x1,
            geopath_net=g_model,
            alpha=float(mfm_alpha),
        )
    if mode_name == "constrained" and g_model is None:
        # Backward compatibility for tests/callers that use constrained mode
        # without a learned path correction model.
        return linear_path(t_batch, x0, x1)
    if g_model is None:
        raise ValueError(f"g_model is required for mode '{mode}'.")
    return corrected_path(t_batch, x0, x1, g_model)


@torch.no_grad()
def interpolant_snapshot_sets(
    x0: torch.Tensor,
    x1: torch.Tensor,
    times: list[float],
    target_sampler: Callable[[float, int, torch.Generator | None], torch.Tensor],
    g_model: PathCorrection | None = None,
    mode: str = "constrained",
    mfm_alpha: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[dict[float, torch.Tensor], dict[float, torch.Tensor], dict[float, torch.Tensor]]:
    if x0.shape != x1.shape:
        raise ValueError(f"x0 and x1 must have same shape, got {x0.shape} and {x1.shape}")
    unique_times = sorted({float(t) for t in times})
    linear_by_time: dict[float, torch.Tensor] = {}
    learned_by_time: dict[float, torch.Tensor] = {}
    target_by_time: dict[float, torch.Tensor] = {}
    for t in unique_times:
        t_batch = torch.full((x0.shape[0], 1), t, device=x0.device, dtype=x0.dtype)
        linear_xt = linear_path(t_batch, x0, x1)
        learned_xt = _interpolant_samples_for_mode(
            mode=mode,
            t_batch=t_batch,
            x0=x0,
            x1=x1,
            g_model=g_model,
            mfm_alpha=float(mfm_alpha),
        )
        target_xt = target_sampler(float(t), x0.shape[0], generator)
        if target_xt.shape != linear_xt.shape:
            raise ValueError(
                f"target_sampler returned shape {target_xt.shape} for t={t}, expected {linear_xt.shape}"
            )
        linear_by_time[float(t)] = linear_xt
        learned_by_time[float(t)] = learned_xt
        target_by_time[float(t)] = target_xt
    return linear_by_time, learned_by_time, target_by_time


@torch.no_grad()
def interpolant_empirical_w2_metrics(
    x0: torch.Tensor,
    x1: torch.Tensor,
    times: list[float],
    target_sampler: Callable[[float, int, torch.Generator | None], torch.Tensor],
    g_model: PathCorrection | None = None,
    mode: str = "constrained",
    mfm_alpha: float = 1.0,
    holdout_time: float | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, float | dict[str, float]]:
    eval_times = sorted({float(t) for t in times} | ({float(holdout_time)} if holdout_time is not None else set()))
    linear_by_time, learned_by_time, target_by_time = interpolant_snapshot_sets(
        x0=x0,
        x1=x1,
        times=eval_times,
        target_sampler=target_sampler,
        g_model=g_model,
        mode=mode,
        mfm_alpha=float(mfm_alpha),
        generator=generator,
    )
    linear_w2: dict[str, float] = {}
    learned_w2: dict[str, float] = {}
    for t in sorted(linear_by_time.keys()):
        target_samples = target_by_time[t]
        linear_w2[f"{t:.2f}"] = empirical_w2_distance(linear_by_time[t], target_samples)
        learned_w2[f"{t:.2f}"] = empirical_w2_distance(learned_by_time[t], target_samples)
    non_holdout_keys = [f"{float(t):.2f}" for t in sorted({float(ti) for ti in times})]
    linear_non_holdout = {k: linear_w2[k] for k in non_holdout_keys}
    learned_non_holdout = {k: learned_w2[k] for k in non_holdout_keys}
    linear_avg = (
        float(sum(linear_non_holdout.values()) / len(linear_non_holdout))
        if linear_non_holdout
        else 0.0
    )
    learned_avg = (
        float(sum(learned_non_holdout.values()) / len(learned_non_holdout))
        if learned_non_holdout
        else 0.0
    )
    out: dict[str, float | dict[str, float]] = {
        "linear_empirical_w2": linear_non_holdout,
        "linear_empirical_w2_avg": linear_avg,
        "learned_empirical_w2": learned_non_holdout,
        "learned_empirical_w2_avg": learned_avg,
        "delta_avg_learned_minus_linear": float(learned_avg - linear_avg),
    }
    if holdout_time is not None:
        holdout_key = f"{float(holdout_time):.2f}"
        if holdout_key not in linear_w2 or holdout_key not in learned_w2:
            raise KeyError(
                "Holdout time not found in interpolant snapshots: "
                f"holdout={holdout_key}, available={sorted(linear_w2.keys())}"
            )
        linear_holdout = float(linear_w2[holdout_key])
        learned_holdout = float(learned_w2[holdout_key])
        out["linear_holdout_empirical_w2"] = linear_holdout
        out["learned_holdout_empirical_w2"] = learned_holdout
        out["delta_holdout_learned_minus_linear"] = float(learned_holdout - linear_holdout)
    return out


def _lookup_target_samples_by_time(
    target_samples_by_time: dict[float, torch.Tensor],
    t: float,
    tol: float = 1e-8,
) -> torch.Tensor:
    if float(t) in target_samples_by_time:
        return target_samples_by_time[float(t)]
    for key, value in target_samples_by_time.items():
        if abs(float(key) - float(t)) <= tol:
            return value
    raise KeyError(
        f"Missing target samples for t={float(t):.6f}. "
        f"Available keys={sorted(float(k) for k in target_samples_by_time.keys())}"
    )


@torch.no_grad()
def interpolant_full_ot_w2_metrics(
    x0_pool: torch.Tensor,
    x1_pool: torch.Tensor,
    plan_src_idx: torch.Tensor,
    plan_tgt_idx: torch.Tensor,
    plan_mass: torch.Tensor,
    times: list[float],
    target_samples_by_time: dict[float, torch.Tensor],
    g_model: PathCorrection | None = None,
    mode: str = "constrained",
    mfm_alpha: float = 1.0,
    holdout_time: float | None = None,
    method: str = "exact_lp",
    num_itermax: int | None = None,
    max_variables: int | None = None,
    support_tol: float = 1e-12,
) -> dict[str, float | dict[str, float]]:
    if plan_src_idx.ndim != 1 or plan_tgt_idx.ndim != 1 or plan_mass.ndim != 1:
        raise ValueError("Global OT support tensors must be 1D.")
    if not (plan_src_idx.shape[0] == plan_tgt_idx.shape[0] == plan_mass.shape[0]):
        raise ValueError(
            "Global OT support tensors must have the same length, got "
            f"{plan_src_idx.shape}, {plan_tgt_idx.shape}, {plan_mass.shape}."
        )
    if plan_mass.shape[0] <= 0:
        raise ValueError("Global OT support is empty.")

    mass = plan_mass.to(device=x0_pool.device, dtype=x0_pool.dtype)
    mass = mass / torch.clamp(mass.sum(), min=torch.finfo(mass.dtype).eps)
    x0_support = x0_pool[plan_src_idx.to(device=x0_pool.device, dtype=torch.long)]
    x1_support = x1_pool[plan_tgt_idx.to(device=x1_pool.device, dtype=torch.long)]

    eval_times = sorted({float(t) for t in times} | ({float(holdout_time)} if holdout_time is not None else set()))
    linear_w2: dict[str, float] = {}
    learned_w2: dict[str, float] = {}
    for t in eval_times:
        t_batch = torch.full((x0_support.shape[0], 1), float(t), device=x0_support.device, dtype=x0_support.dtype)
        linear_xt = linear_path(t_batch, x0_support, x1_support)
        learned_xt = _interpolant_samples_for_mode(
            mode=mode,
            t_batch=t_batch,
            x0=x0_support,
            x1=x1_support,
            g_model=g_model,
            mfm_alpha=float(mfm_alpha),
        )
        target_xt = _lookup_target_samples_by_time(target_samples_by_time=target_samples_by_time, t=float(t))
        linear_w2[f"{float(t):.2f}"] = balanced_empirical_w2_distance(
            linear_xt,
            target_xt,
            x_weights=mass,
            y_weights=None,
            method=method,
            num_itermax=num_itermax,
            max_variables=max_variables,
            support_tol=support_tol,
        )
        learned_w2[f"{float(t):.2f}"] = balanced_empirical_w2_distance(
            learned_xt,
            target_xt,
            x_weights=mass,
            y_weights=None,
            method=method,
            num_itermax=num_itermax,
            max_variables=max_variables,
            support_tol=support_tol,
        )

    non_holdout_keys = [f"{float(t):.2f}" for t in sorted({float(ti) for ti in times})]
    linear_non_holdout = {k: linear_w2[k] for k in non_holdout_keys}
    learned_non_holdout = {k: learned_w2[k] for k in non_holdout_keys}
    linear_avg = (
        float(sum(linear_non_holdout.values()) / len(linear_non_holdout))
        if linear_non_holdout
        else 0.0
    )
    learned_avg = (
        float(sum(learned_non_holdout.values()) / len(learned_non_holdout))
        if learned_non_holdout
        else 0.0
    )
    out: dict[str, float | dict[str, float]] = {
        "linear_full_ot_w2": linear_non_holdout,
        "linear_full_ot_w2_avg": linear_avg,
        "learned_full_ot_w2": learned_non_holdout,
        "learned_full_ot_w2_avg": learned_avg,
        "delta_avg_learned_minus_linear_full_ot": float(learned_avg - linear_avg),
    }
    if holdout_time is not None:
        holdout_key = f"{float(holdout_time):.2f}"
        if holdout_key not in linear_w2 or holdout_key not in learned_w2:
            raise KeyError(
                "Holdout time not found in full-OT interpolant snapshots: "
                f"holdout={holdout_key}, available={sorted(linear_w2.keys())}"
            )
        linear_holdout = float(linear_w2[holdout_key])
        learned_holdout = float(learned_w2[holdout_key])
        out["linear_holdout_full_ot_w2"] = linear_holdout
        out["learned_holdout_full_ot_w2"] = learned_holdout
        out["delta_holdout_learned_minus_linear_full_ot"] = float(learned_holdout - linear_holdout)
    return out


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
    residual = moment_features(x) - target_feature
    return float(torch.linalg.norm(residual).item())
