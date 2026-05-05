from __future__ import annotations

import random
from typing import Any, Callable

import numpy as np
import torch

from cfm_project.constraints import (
    augmented_lagrangian_terms,
    constraint_residuals,
    residual_norms,
    update_lagrange_multipliers,
)
from cfm_project.data import (
    CouplingProblem,
    EmpiricalCouplingProblem,
    GaussianOTProblem,
    analytic_bridge_cov,
    analytic_bridge_mean,
    gaussian_moment_feature_vector,
    moment_feature_vector_from_samples,
    sample_coupled_batch,
    sample_gaussian,
)
from cfm_project.mfm_core import (
    MetricBackend,
    build_metric_backend,
    land_geopath_loss,
    mfm_mean_path,
    mfm_path_and_velocity,
)
from cfm_project.metrics import (
    balanced_empirical_w2_distance,
    empirical_w1_distance,
    empirical_w2_distance,
    euler_velocity_snapshots,
    intermediate_empirical_w2_metrics,
    intermediate_wasserstein_metrics,
    interpolant_empirical_w2_metrics,
    interpolant_full_ot_w2_metrics,
    interpolant_snapshot_sets,
    path_energy_proxy,
    transport_quality_metrics,
)
from cfm_project.models import PathCorrection, VelocityField
from cfm_project.paths import corrected_path, path_and_velocity, vector_time_derivative

METRIC_BASE_MODES = {"metric", "metric_alpha0"}
METRIC_CONSTRAINED_MODES = {"metric_constrained_al", "metric_constrained_soft"}
METRIC_MODES = METRIC_BASE_MODES | METRIC_CONSTRAINED_MODES
METRIC_AL_MODES = {"metric_constrained_al"}
METRIC_SOFT_MODES = {"metric_constrained_soft"}
CONSTRAINED_BETA_SCHEDULES = {"constant", "piecewise", "linear"}


def _metric_moment_style(mode: str) -> str:
    if mode in METRIC_AL_MODES:
        return "al"
    if mode in METRIC_SOFT_MODES:
        return "soft"
    return "none"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def is_stage_a_only_profile(train_cfg: dict[str, Any]) -> bool:
    stage_a_steps = int(train_cfg["stage_a_steps"])
    stage_b_steps = int(train_cfg["stage_b_steps"])
    stage_c_steps = int(train_cfg["stage_c_steps"])
    return stage_a_steps > 0 and stage_b_steps == 0 and stage_c_steps == 0


def _uniform_time(batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.rand((batch_size, 1), device=device, dtype=dtype)


def _endpoint_moment_feature(problem: CouplingProblem, t: float) -> torch.Tensor:
    if isinstance(problem, GaussianOTProblem):
        mean_t = analytic_bridge_mean(float(t), problem)
        cov_t = analytic_bridge_cov(float(t), problem)
        return gaussian_moment_feature_vector(mean_t, cov_t)
    if isinstance(problem, EmpiricalCouplingProblem):
        pool = problem.x0_pool if float(t) <= 0.0 else problem.x1_pool
        return moment_feature_vector_from_samples(pool)
    raise TypeError(f"Unsupported problem type for endpoint moments: {type(problem)}")


def _lookup_target_feature(
    targets: dict[float, torch.Tensor],
    t: float,
    tol: float = 1e-8,
) -> torch.Tensor:
    if float(t) in targets:
        return targets[float(t)]
    for key, value in targets.items():
        if abs(float(key) - float(t)) <= tol:
            return value
    raise KeyError(f"Missing target moments for t={t:.6f}")


def _anchor_moment_feature(
    problem: CouplingProblem,
    targets: dict[float, torch.Tensor],
    t: float,
) -> torch.Tensor:
    if float(t) <= 0.0:
        return _endpoint_moment_feature(problem, t=0.0)
    if float(t) >= 1.0:
        return _endpoint_moment_feature(problem, t=1.0)
    return _lookup_target_feature(targets=targets, t=float(t))


def _build_constrained_beta_schedule(
    problem: CouplingProblem,
    targets: dict[float, torch.Tensor],
    constraint_times: list[float],
    beta0: float,
    beta_schedule: str,
    drift_p: float,
    drift_eps: float,
    min_scale: float,
    max_scale: float,
) -> dict[str, Any]:
    schedule_name = str(beta_schedule).strip().lower()
    if schedule_name not in CONSTRAINED_BETA_SCHEDULES:
        raise ValueError(
            f"Unsupported train.beta_schedule '{beta_schedule}'. "
            "Expected one of: constant, piecewise, linear."
        )
    if drift_p < 0.0:
        raise ValueError(f"train.beta_drift_p must be non-negative, got {drift_p}")
    if drift_eps <= 0.0:
        raise ValueError(f"train.beta_drift_eps must be positive, got {drift_eps}")
    if min_scale <= 0.0:
        raise ValueError(f"train.beta_min_scale must be positive, got {min_scale}")
    if max_scale <= 0.0:
        raise ValueError(f"train.beta_max_scale must be positive, got {max_scale}")
    if min_scale > max_scale:
        raise ValueError(
            "train.beta_min_scale must be <= train.beta_max_scale, "
            f"got {min_scale} > {max_scale}"
        )

    anchors = sorted({0.0, 1.0, *[float(t) for t in constraint_times]})
    if len(anchors) < 2:
        raise ValueError("Constrained beta schedule requires at least two anchor times.")

    n_intervals = len(anchors) - 1
    if schedule_name == "constant":
        interval_betas = [float(beta0) for _ in range(n_intervals)]
        return {
            "name": "constant",
            "base_beta": float(beta0),
            "anchors": anchors,
            "interval_drifts": [0.0 for _ in range(n_intervals)],
            "drift_mean": 0.0,
            "interval_scales": [1.0 for _ in range(n_intervals)],
            "interval_betas": interval_betas,
            "anchor_betas": [float(beta0) for _ in anchors],
            "drift_p": float(drift_p),
            "drift_eps": float(drift_eps),
            "min_scale": float(min_scale),
            "max_scale": float(max_scale),
        }

    anchor_features = [_anchor_moment_feature(problem=problem, targets=targets, t=t) for t in anchors]
    interval_drifts: list[float] = []
    for idx in range(n_intervals):
        t0 = float(anchors[idx])
        t1 = float(anchors[idx + 1])
        dt = float(t1 - t0)
        if dt <= 0.0:
            raise ValueError(f"Anchor times must be strictly increasing, got dt={dt} at idx={idx}")
        diff = anchor_features[idx + 1] - anchor_features[idx]
        drift = float(torch.linalg.norm(diff).item() / dt)
        interval_drifts.append(drift)

    drift_mean = float(np.mean(interval_drifts)) if interval_drifts else 0.0
    interval_scales: list[float] = []
    if drift_mean <= float(drift_eps):
        interval_scales = [1.0 for _ in interval_drifts]
    else:
        for drift in interval_drifts:
            raw_scale = (drift_mean / (float(drift) + float(drift_eps))) ** float(drift_p)
            clipped_scale = float(np.clip(raw_scale, float(min_scale), float(max_scale)))
            interval_scales.append(clipped_scale)
    interval_betas = [float(beta0) * scale for scale in interval_scales]

    anchor_betas = [interval_betas[0]]
    for idx in range(1, len(anchors) - 1):
        left_beta = interval_betas[idx - 1]
        right_beta = interval_betas[idx]
        anchor_betas.append(0.5 * (left_beta + right_beta))
    anchor_betas.append(interval_betas[-1])

    return {
        "name": schedule_name,
        "base_beta": float(beta0),
        "anchors": anchors,
        "interval_drifts": interval_drifts,
        "drift_mean": float(drift_mean),
        "interval_scales": interval_scales,
        "interval_betas": interval_betas,
        "anchor_betas": anchor_betas,
        "drift_p": float(drift_p),
        "drift_eps": float(drift_eps),
        "min_scale": float(min_scale),
        "max_scale": float(max_scale),
    }


def _beta_weights_at_times(
    t: torch.Tensor,
    beta0: float,
    beta_schedule: dict[str, Any] | None,
) -> torch.Tensor:
    t_flat = t.reshape(-1)
    if beta_schedule is None:
        return torch.full_like(t_flat, float(beta0))
    schedule_name = str(beta_schedule.get("name", "constant")).strip().lower()
    if schedule_name == "constant":
        return torch.full_like(t_flat, float(beta0))

    anchors = torch.tensor(beta_schedule["anchors"], device=t.device, dtype=t.dtype)
    n_intervals = int(anchors.shape[0] - 1)
    if n_intervals <= 0:
        raise ValueError("Constrained beta schedule has no intervals.")
    if n_intervals == 1:
        interval_idx = torch.zeros_like(t_flat, dtype=torch.long)
    else:
        boundaries = anchors[1:-1]
        interval_idx = torch.bucketize(t_flat, boundaries)
        interval_idx = torch.clamp(interval_idx, min=0, max=n_intervals - 1)

    if schedule_name == "piecewise":
        interval_betas = torch.tensor(beta_schedule["interval_betas"], device=t.device, dtype=t.dtype)
        return interval_betas[interval_idx]

    if schedule_name == "linear":
        anchor_betas = torch.tensor(beta_schedule["anchor_betas"], device=t.device, dtype=t.dtype)
        t0 = anchors[interval_idx]
        t1 = anchors[interval_idx + 1]
        b0 = anchor_betas[interval_idx]
        b1 = anchor_betas[interval_idx + 1]
        denom = torch.clamp(t1 - t0, min=torch.finfo(t.dtype).eps)
        weight = (t_flat - t0) / denom
        return b0 + weight * (b1 - b0)

    raise ValueError(
        f"Unsupported constrained beta schedule '{schedule_name}'. "
        "Expected one of: constant, piecewise, linear."
    )


def _constraint_residuals_for_mode(
    mode: str,
    x0: torch.Tensor,
    x1: torch.Tensor,
    times: list[float],
    targets: dict[float, torch.Tensor],
    g_model: PathCorrection | None,
    mfm_alpha: float = 1.0,
) -> dict[float, torch.Tensor]:
    def path_fn(t_value: float) -> torch.Tensor:
        t_batch = torch.full((x0.shape[0], 1), t_value, device=x0.device, dtype=x0.dtype)
        return _path_samples_for_mode(
            mode=mode,
            x0=x0,
            x1=x1,
            t_batch=t_batch,
            g_model=g_model,
            mfm_alpha=mfm_alpha,
        )

    return constraint_residuals(path_fn=path_fn, times=times, targets=targets)


def _path_samples_for_mode(
    mode: str,
    x0: torch.Tensor,
    x1: torch.Tensor,
    t_batch: torch.Tensor,
    g_model: PathCorrection | None,
    mfm_alpha: float = 1.0,
) -> torch.Tensor:
    if mode == "baseline":
        return (1.0 - t_batch) * x0 + t_batch * x1
    if mode in METRIC_MODES:
        return mfm_mean_path(
            t=t_batch,
            x0=x0,
            x1=x1,
            geopath_net=g_model,
            alpha=float(mfm_alpha),
        )
    if g_model is None:
        raise ValueError("g_model is required in constrained mode.")
    return corrected_path(t_batch, x0, x1, g_model)


def _pseudo_constraint_residuals_for_mode(
    mode: str,
    x0: torch.Tensor,
    x1: torch.Tensor,
    times: list[float],
    pseudo_targets: dict[float, torch.Tensor],
    pseudo_posterior: Callable[[torch.Tensor], torch.Tensor],
    g_model: PathCorrection | None,
    mfm_alpha: float = 1.0,
) -> dict[float, torch.Tensor]:
    residuals: dict[float, torch.Tensor] = {}
    for t in times:
        t_value = float(t)
        t_batch = torch.full((x0.shape[0], 1), t_value, device=x0.device, dtype=x0.dtype)
        xt = _path_samples_for_mode(
            mode=mode,
            x0=x0,
            x1=x1,
            t_batch=t_batch,
            g_model=g_model,
            mfm_alpha=mfm_alpha,
        )
        probs = pseudo_posterior(xt)
        if probs.ndim != 2:
            raise ValueError(f"pseudo_posterior must return shape (N, K), got {tuple(probs.shape)}")
        target = pseudo_targets[float(t_value)]
        target_local = target
        if target_local.device != probs.device or target_local.dtype != probs.dtype:
            target_local = target_local.to(device=probs.device, dtype=probs.dtype)
        residuals[float(t_value)] = probs.mean(dim=0) - target_local
    return residuals


def _residual_squared_mean(residuals: dict[float, torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
    residual_sq_terms = [torch.dot(res, res) for res in residuals.values()]
    if residual_sq_terms:
        return torch.mean(torch.stack(residual_sq_terms))
    return torch.zeros((), device=ref.device, dtype=ref.dtype)


def _constrained_objective(
    g_model: PathCorrection,
    x0: torch.Tensor,
    x1: torch.Tensor,
    times: list[float],
    targets: dict[float, torch.Tensor],
    lambdas: dict[float, torch.Tensor],
    rho: float,
    alpha: float,
    beta: float,
    beta_schedule: dict[str, Any] | None = None,
    pseudo_targets: dict[float, torch.Tensor] | None = None,
    pseudo_lambdas: dict[float, torch.Tensor] | None = None,
    pseudo_rho: float | None = None,
    pseudo_eta: float = 0.0,
    pseudo_posterior: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[
    torch.Tensor,
    dict[float, torch.Tensor],
    dict[float, torch.Tensor] | None,
    dict[str, float],
]:
    t_rand = _uniform_time(x0.shape[0], x0.device, x0.dtype)
    _, u_target, t_req = path_and_velocity(
        mode="constrained",
        t=t_rand,
        x0=x0,
        x1=x1,
        g_model=g_model,
        create_graph=True,
    )
    base_velocity = x1 - x0
    energy = torch.mean(torch.sum((u_target - base_velocity) ** 2, dim=1))
    du_dt = vector_time_derivative(u_target, t_req, create_graph=True)
    smoothness_per_sample = torch.sum(du_dt**2, dim=1)
    smoothness = torch.mean(smoothness_per_sample)
    beta_t = _beta_weights_at_times(t=t_req.detach(), beta0=float(beta), beta_schedule=beta_schedule)
    weighted_smoothness = torch.mean(beta_t * smoothness_per_sample)
    regularizer = alpha * energy + weighted_smoothness

    residuals = _constraint_residuals_for_mode(
        mode="constrained",
        x0=x0,
        x1=x1,
        times=times,
        targets=targets,
        g_model=g_model,
    )
    al_term, per_time = augmented_lagrangian_terms(residuals=residuals, lambdas=lambdas, rho=rho)
    total = regularizer + al_term
    pseudo_residuals: dict[float, torch.Tensor] | None = None
    pseudo_term = torch.zeros((), device=x0.device, dtype=x0.dtype)
    pseudo_active = (
        float(pseudo_eta) > 0.0
        and pseudo_targets is not None
        and pseudo_posterior is not None
        and pseudo_lambdas is not None
        and pseudo_rho is not None
    )
    if pseudo_active:
        pseudo_residuals = _pseudo_constraint_residuals_for_mode(
            mode="constrained",
            x0=x0,
            x1=x1,
            times=times,
            pseudo_targets=pseudo_targets,
            pseudo_posterior=pseudo_posterior,
            g_model=g_model,
        )
        pseudo_term, pseudo_per_time = augmented_lagrangian_terms(
            residuals=pseudo_residuals,
            lambdas=pseudo_lambdas,
            rho=float(pseudo_rho),
        )
        total = total + float(pseudo_eta) * pseudo_term
    else:
        pseudo_per_time = {}

    stats = {
        "regularizer": float(regularizer.detach().item()),
        "energy_term": float(energy.detach().item()),
        "smoothness_term": float(smoothness.detach().item()),
        "weighted_smoothness_term": float(weighted_smoothness.detach().item()),
        "beta_t_mean": float(beta_t.detach().mean().item()),
        "al_term": float(al_term.detach().item()),
        "pseudo_term": float(pseudo_term.detach().item()),
    }
    for t, value in per_time.items():
        stats[f"al_t_{t:.2f}"] = float(value)
    for t, value in pseudo_per_time.items():
        stats[f"pseudo_al_t_{t:.2f}"] = float(value)
    return total, residuals, pseudo_residuals, stats


def _metric_geopath_objective(
    geopath_model: PathCorrection,
    alpha_mfm: float,
    x0: torch.Tensor,
    x1: torch.Tensor,
    manifold_samples: torch.Tensor,
    land_gamma: float,
    land_rho: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    mu_t, u_t, _ = mfm_path_and_velocity(
        t=None,
        x0=x0,
        x1=x1,
        geopath_net=geopath_model,
        alpha=float(alpha_mfm),
        create_graph=True,
    )
    loss = land_geopath_loss(
        x_t=mu_t,
        u_t=u_t,
        manifold_samples=manifold_samples,
        gamma=float(land_gamma),
        rho=float(land_rho),
    )
    stats = {"land_loss": float(loss.detach().item())}
    return loss, stats


def _metric_constrained_geopath_objective(
    mode: str,
    geopath_model: PathCorrection,
    alpha_mfm: float,
    x0: torch.Tensor,
    x1: torch.Tensor,
    manifold_samples: torch.Tensor,
    times: list[float],
    targets: dict[float, torch.Tensor],
    lambdas: dict[float, torch.Tensor],
    rho: float,
    land_gamma: float,
    land_rho: float,
    moment_eta: float,
    pseudo_targets: dict[float, torch.Tensor] | None = None,
    pseudo_lambdas: dict[float, torch.Tensor] | None = None,
    pseudo_rho: float | None = None,
    pseudo_eta: float = 0.0,
    pseudo_posterior: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[
    torch.Tensor,
    dict[float, torch.Tensor],
    dict[float, torch.Tensor] | None,
    dict[str, float],
]:
    if mode not in METRIC_CONSTRAINED_MODES:
        raise ValueError(f"Unsupported metric-constrained mode: {mode}")
    if moment_eta < 0.0:
        raise ValueError(f"moment_eta must be non-negative, got {moment_eta}")

    mu_t, u_t, _ = mfm_path_and_velocity(
        t=None,
        x0=x0,
        x1=x1,
        geopath_net=geopath_model,
        alpha=float(alpha_mfm),
        create_graph=True,
    )
    land_loss = land_geopath_loss(
        x_t=mu_t,
        u_t=u_t,
        manifold_samples=manifold_samples,
        gamma=float(land_gamma),
        rho=float(land_rho),
    )
    residuals = _constraint_residuals_for_mode(
        mode=mode,
        x0=x0,
        x1=x1,
        times=times,
        targets=targets,
        g_model=geopath_model,
        mfm_alpha=float(alpha_mfm),
    )

    residual_sq_mean = _residual_squared_mean(residuals=residuals, ref=x0)

    stats: dict[str, float] = {
        "land_loss": float(land_loss.detach().item()),
        "moment_sq_mean": float(residual_sq_mean.detach().item()),
    }
    pseudo_residuals: dict[float, torch.Tensor] | None = None
    pseudo_sq_mean = torch.zeros((), device=x0.device, dtype=x0.dtype)
    pseudo_term = torch.zeros((), device=x0.device, dtype=x0.dtype)
    pseudo_active = (
        float(pseudo_eta) > 0.0 and pseudo_targets is not None and pseudo_posterior is not None
    )
    if pseudo_active:
        pseudo_residuals = _pseudo_constraint_residuals_for_mode(
            mode=mode,
            x0=x0,
            x1=x1,
            times=times,
            pseudo_targets=pseudo_targets,
            pseudo_posterior=pseudo_posterior,
            g_model=geopath_model,
            mfm_alpha=float(alpha_mfm),
        )
        pseudo_sq_mean = _residual_squared_mean(residuals=pseudo_residuals, ref=x0)
        stats["pseudo_sq_mean"] = float(pseudo_sq_mean.detach().item())

    if mode in METRIC_AL_MODES:
        al_term, per_time = augmented_lagrangian_terms(
            residuals=residuals,
            lambdas=lambdas,
            rho=float(rho),
        )
        total = land_loss + float(moment_eta) * al_term
        stats["moment_term"] = float(al_term.detach().item())
        for t, value in per_time.items():
            stats[f"al_t_{t:.2f}"] = float(value)
        if pseudo_active:
            if pseudo_lambdas is None or pseudo_rho is None:
                raise ValueError(
                    "Pseudo AL term requires pseudo_lambdas and pseudo_rho in metric_constrained_al mode."
                )
            pseudo_term, pseudo_per_time = augmented_lagrangian_terms(
                residuals=pseudo_residuals,
                lambdas=pseudo_lambdas,
                rho=float(pseudo_rho),
            )
            total = total + float(pseudo_eta) * pseudo_term
            for t, value in pseudo_per_time.items():
                stats[f"pseudo_al_t_{t:.2f}"] = float(value)
    else:
        total = land_loss + float(moment_eta) * residual_sq_mean
        stats["moment_term"] = float(residual_sq_mean.detach().item())
        if pseudo_active:
            pseudo_term = pseudo_sq_mean
            total = total + float(pseudo_eta) * pseudo_term

    stats["pseudo_term"] = float(pseudo_term.detach().item())

    return total, residuals, pseudo_residuals, stats


def _cfm_loss(
    mode: str,
    v_model: VelocityField,
    g_model: PathCorrection | None,
    x0: torch.Tensor,
    x1: torch.Tensor,
    mfm_backend: MetricBackend | None = None,
) -> tuple[torch.Tensor, float]:
    t_rand = _uniform_time(x0.shape[0], x0.device, x0.dtype)
    if mode in METRIC_MODES:
        if mfm_backend is None:
            raise ValueError("mfm_backend is required for metric modes.")
        t_rand, xt, u_target = mfm_backend.sample_location_and_conditional_flow(
            x0=x0,
            x1=x1,
            t=t_rand,
            create_graph=False,
        )
    else:
        xt, u_target, _ = path_and_velocity(
            mode=mode,
            t=t_rand,
            x0=x0,
            x1=x1,
            g_model=g_model,
            create_graph=False,
        )
    pred = v_model(t_rand, xt)
    loss = torch.mean(torch.sum((pred - u_target) ** 2, dim=1))
    return loss, path_energy_proxy(u_target.detach())


def _init_lambdas(
    times: list[float],
    targets: dict[float, torch.Tensor],
) -> dict[float, torch.Tensor]:
    lambdas: dict[float, torch.Tensor] = {}
    for t in times:
        lambdas[float(t)] = torch.zeros_like(targets[float(t)])
    return lambdas


def _eval_constraint_norms(
    mode: str,
    problem: CouplingProblem,
    coupling: str,
    batch_size: int,
    times: list[float],
    targets: dict[float, torch.Tensor],
    g_model: PathCorrection | None,
    mfm_alpha: float,
    generator: torch.Generator,
) -> dict[float, float]:
    x0, x1, _ = sample_coupled_batch(
        problem,
        batch_size=batch_size,
        coupling=coupling,
        generator=generator,
    )
    residuals = _constraint_residuals_for_mode(
        mode=mode,
        x0=x0,
        x1=x1,
        times=times,
        targets=targets,
        g_model=g_model,
        mfm_alpha=mfm_alpha,
    )
    return residual_norms(residuals)


def _eval_pseudo_constraint_norms(
    mode: str,
    problem: CouplingProblem,
    coupling: str,
    batch_size: int,
    times: list[float],
    pseudo_targets: dict[float, torch.Tensor],
    pseudo_posterior: Callable[[torch.Tensor], torch.Tensor],
    g_model: PathCorrection | None,
    mfm_alpha: float,
    generator: torch.Generator,
) -> dict[float, float]:
    x0, x1, _ = sample_coupled_batch(
        problem,
        batch_size=batch_size,
        coupling=coupling,
        generator=generator,
    )
    residuals = _pseudo_constraint_residuals_for_mode(
        mode=mode,
        x0=x0,
        x1=x1,
        times=times,
        pseudo_targets=pseudo_targets,
        pseudo_posterior=pseudo_posterior,
        g_model=g_model,
        mfm_alpha=mfm_alpha,
    )
    return residual_norms(residuals)


def _eval_cfm_loss(
    mode: str,
    problem: CouplingProblem,
    coupling: str,
    v_model: VelocityField,
    g_model: PathCorrection | None,
    mfm_backend: MetricBackend | None,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[float, float]:
    x0, x1, _ = sample_coupled_batch(
        problem,
        batch_size=batch_size,
        coupling=coupling,
        generator=generator,
    )
    if mode in METRIC_MODES:
        if mfm_backend is None:
            raise ValueError("mfm_backend is required for metric modes.")
        t, xt, u_target = mfm_backend.sample_location_and_conditional_flow(
            x0=x0,
            x1=x1,
            t=_uniform_time(batch_size, x0.device, x0.dtype),
            create_graph=False,
        )
    else:
        t = _uniform_time(batch_size, x0.device, x0.dtype)
        xt, u_target, _ = path_and_velocity(
            mode=mode,
            t=t,
            x0=x0,
            x1=x1,
            g_model=g_model,
            create_graph=False,
        )
    with torch.no_grad():
        pred = v_model(t, xt)
        loss = torch.mean(torch.sum((pred - u_target.detach()) ** 2, dim=1))
    return float(loss.item()), path_energy_proxy(u_target.detach())


def _sample_from_pool(
    pool: torch.Tensor,
    n_samples: int,
    generator: torch.Generator,
) -> torch.Tensor:
    idx = torch.randint(
        low=0,
        high=pool.shape[0],
        size=(n_samples,),
        device=pool.device,
        generator=generator,
    )
    return pool[idx]


def _build_metric_reference_pool(
    problem: CouplingProblem,
    target_sampler: Callable[[float, int, torch.Generator | None], torch.Tensor] | None,
    times: list[float],
    n_samples_per_time: int,
    generator: torch.Generator,
    reference_pool_policy: str,
) -> torch.Tensor:
    if n_samples_per_time <= 0:
        raise ValueError(f"n_samples_per_time must be positive, got {n_samples_per_time}")
    policy = str(reference_pool_policy).strip().lower()
    if policy == "endpoints_only":
        anchor_times = [0.0, 1.0]
    elif policy == "anchors_all":
        anchor_times = sorted({0.0, 1.0, *[float(t) for t in times]})
    else:
        raise ValueError(
            "Unsupported mfm.reference_pool_policy "
            f"'{reference_pool_policy}'. Expected one of: endpoints_only, anchors_all."
        )
    chunks: list[torch.Tensor] = []
    for t in anchor_times:
        if target_sampler is not None:
            chunk = target_sampler(float(t), n_samples_per_time, generator)
            chunks.append(chunk)
            continue
        if isinstance(problem, GaussianOTProblem):
            mean_t = analytic_bridge_mean(float(t), problem)
            cov_t = analytic_bridge_cov(float(t), problem)
            chunk = sample_gaussian(mean_t, cov_t, n_samples=n_samples_per_time, generator=generator)
            chunks.append(chunk)
            continue
        if isinstance(problem, EmpiricalCouplingProblem):
            if t <= 0.0:
                chunks.append(_sample_from_pool(problem.x0_pool, n_samples_per_time, generator))
            elif t >= 1.0:
                chunks.append(_sample_from_pool(problem.x1_pool, n_samples_per_time, generator))
            else:
                n0 = n_samples_per_time // 2
                n1 = n_samples_per_time - n0
                x0_chunk = _sample_from_pool(problem.x0_pool, n0, generator)
                x1_chunk = _sample_from_pool(problem.x1_pool, n1, generator)
                chunks.append(torch.cat([x0_chunk, x1_chunk], dim=0))
            continue
        raise TypeError(f"Unsupported problem type for metric reference pool: {type(problem)}")
    return torch.cat(chunks, dim=0)


def _to_cpu_snapshot_dict(samples_by_time: dict[float, torch.Tensor]) -> dict[float, torch.Tensor]:
    return {float(t): tensor.detach().cpu() for t, tensor in samples_by_time.items()}


def _lookup_target_pool_by_time(
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
        f"Missing full target pool for t={float(t):.6f}. "
        f"Available keys={sorted(float(v) for v in target_samples_by_time.keys())}"
    )


def _eval_full_ot_rollout_metrics(
    problem: CouplingProblem,
    v_model: VelocityField,
    times: list[float],
    n_steps: int,
    target_samples_by_time: dict[float, torch.Tensor],
    holdout_time: float | None = None,
    method: str = "exact_lp",
    num_itermax: int | None = None,
    max_variables: int | None = None,
    support_tol: float = 1e-12,
) -> tuple[dict[str, float | dict[str, float]], dict[str, Any]]:
    if not isinstance(problem, EmpiricalCouplingProblem):
        raise ValueError("Full-OT rollout metrics require an EmpiricalCouplingProblem.")
    eval_times_set = {float(t) for t in times} | {1.0}
    if holdout_time is not None:
        eval_times_set.add(float(holdout_time))
    eval_times = sorted(eval_times_set)

    x0_eval = problem.x0_pool
    generated_by_time = euler_velocity_snapshots(
        velocity_fn=v_model,
        x0=x0_eval,
        times=eval_times,
        n_steps=n_steps,
    )
    target_by_time: dict[float, torch.Tensor] = {}
    full_ot_w2_by_time: dict[str, float] = {}
    for t in eval_times:
        t_value = float(t)
        generated = generated_by_time[t_value]
        target = _lookup_target_pool_by_time(target_samples_by_time=target_samples_by_time, t=t_value)
        target_by_time[t_value] = target
        full_ot_w2_by_time[f"{t_value:.2f}"] = balanced_empirical_w2_distance(
            generated,
            target,
            x_weights=None,
            y_weights=None,
            method=method,
            num_itermax=num_itermax,
            max_variables=max_variables,
            support_tol=support_tol,
        )

    intermediate = {f"{float(t):.2f}": full_ot_w2_by_time[f"{float(t):.2f}"] for t in sorted(set(times))}
    intermediate_avg = float(sum(intermediate.values()) / len(intermediate)) if intermediate else 0.0
    endpoint_w2 = float(full_ot_w2_by_time["1.00"])
    holdout_key = None if holdout_time is None else f"{float(holdout_time):.2f}"
    metrics = {
        "intermediate_full_ot_w2": intermediate,
        "intermediate_full_ot_w2_avg": intermediate_avg,
        "transport_endpoint_full_ot_w2": endpoint_w2,
        "holdout_full_ot_w2": None if holdout_key is None else float(full_ot_w2_by_time[holdout_key]),
    }
    artifacts = {
        "generated_by_time": _to_cpu_snapshot_dict(generated_by_time),
        "target_by_time": _to_cpu_snapshot_dict(target_by_time),
        "full_ot_w2_by_time": {float(t): float(full_ot_w2_by_time[f"{float(t):.2f}"]) for t in eval_times},
    }
    return metrics, artifacts


def _eval_empirical_rollout_metrics(
    problem: CouplingProblem,
    coupling: str,
    v_model: VelocityField,
    times: list[float],
    n_samples: int,
    n_steps: int,
    target_sampler: Callable[[float, int, torch.Generator | None], torch.Tensor],
    generator: torch.Generator,
    holdout_time: float | None = None,
) -> tuple[dict[str, float | dict[str, float]], dict[str, Any]]:
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    eval_times_set = {float(t) for t in times} | {1.0}
    if holdout_time is not None:
        eval_times_set.add(float(holdout_time))
    eval_times = sorted(eval_times_set)
    x0_eval, _, _ = sample_coupled_batch(
        problem=problem,
        batch_size=n_samples,
        coupling=coupling,
        generator=generator,
    )
    generated_by_time = euler_velocity_snapshots(
        velocity_fn=v_model,
        x0=x0_eval,
        times=eval_times,
        n_steps=n_steps,
    )
    target_by_time: dict[float, torch.Tensor] = {}
    empirical_w2_by_time: dict[str, float] = {}
    empirical_w1_by_time: dict[str, float] = {}
    for t in eval_times:
        t_value = float(t)
        generated = generated_by_time[t_value]
        target = target_sampler(t_value, n_samples, generator)
        if target.shape != generated.shape:
            raise ValueError(
                f"target_sampler returned shape {target.shape} for t={t_value:.2f}, "
                f"expected {generated.shape}"
        )
        target_by_time[t_value] = target
        empirical_w2_by_time[f"{t_value:.2f}"] = empirical_w2_distance(generated, target)
        empirical_w1_by_time[f"{t_value:.2f}"] = empirical_w1_distance(generated, target)

    intermediate = {f"{float(t):.2f}": empirical_w2_by_time[f"{float(t):.2f}"] for t in sorted(set(times))}
    intermediate_w1 = {f"{float(t):.2f}": empirical_w1_by_time[f"{float(t):.2f}"] for t in sorted(set(times))}
    intermediate_avg = float(sum(intermediate.values()) / len(intermediate)) if intermediate else 0.0
    intermediate_w1_avg = (
        float(sum(intermediate_w1.values()) / len(intermediate_w1)) if intermediate_w1 else 0.0
    )
    endpoint_w2 = float(empirical_w2_by_time["1.00"])
    endpoint_w1 = float(empirical_w1_by_time["1.00"])
    holdout_key = None if holdout_time is None else f"{float(holdout_time):.2f}"
    metrics = {
        "intermediate_empirical_w2": intermediate,
        "intermediate_empirical_w2_avg": intermediate_avg,
        "intermediate_empirical_w1": intermediate_w1,
        "intermediate_empirical_w1_avg": intermediate_w1_avg,
        "transport_endpoint_empirical_w2": endpoint_w2,
        "transport_endpoint_empirical_w1": endpoint_w1,
        "transport_score": endpoint_w2,
        "holdout_empirical_w2": None if holdout_key is None else float(empirical_w2_by_time[holdout_key]),
        "holdout_empirical_w1": None if holdout_key is None else float(empirical_w1_by_time[holdout_key]),
    }
    artifacts = {
        "generated_by_time": _to_cpu_snapshot_dict(generated_by_time),
        "target_by_time": _to_cpu_snapshot_dict(target_by_time),
        "empirical_w2_by_time": {float(t): float(empirical_w2_by_time[f"{float(t):.2f}"]) for t in eval_times},
        "empirical_w1_by_time": {float(t): float(empirical_w1_by_time[f"{float(t):.2f}"]) for t in eval_times},
    }
    return metrics, artifacts


def train_experiment(
    cfg: dict[str, Any],
    problem: CouplingProblem,
    targets: dict[float, torch.Tensor],
    pseudo_targets: dict[float, torch.Tensor] | None = None,
    pseudo_posterior: Callable[[torch.Tensor], torch.Tensor] | None = None,
    target_sampler: Callable[[float, int, torch.Generator | None], torch.Tensor] | None = None,
    target_samples_by_time: dict[float, torch.Tensor] | None = None,
    data_family: str = "gaussian",
) -> dict[str, Any]:
    device = torch.device(cfg["device"])
    dtype = torch.float32
    mode = str(cfg["experiment"]["mode"])
    if mode not in {"baseline", "constrained", *METRIC_MODES}:
        raise ValueError(
            f"Unsupported experiment mode '{mode}'. "
            "Expected one of: baseline, constrained, metric, metric_alpha0, "
            "metric_constrained_al, metric_constrained_soft."
        )

    set_seed(int(cfg["seed"]))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(cfg["seed"]))

    model_cfg = cfg["model"]
    state_dim = int(cfg["data"]["dim"])
    v_model = VelocityField(
        state_dim=state_dim,
        hidden_dims=model_cfg["velocity_hidden_dims"],
        activation=model_cfg["activation"],
    ).to(device=device, dtype=dtype)

    mfm_cfg = cfg.get("mfm", {})
    mfm_alpha = 0.0 if mode == "metric_alpha0" else float(mfm_cfg.get("alpha", 1.0))
    mfm_sigma = float(mfm_cfg.get("sigma", 0.1))
    mfm_requested_backend = str(mfm_cfg.get("backend", "auto"))
    mfm_land_gamma = float(mfm_cfg.get("land_gamma", 0.125))
    mfm_land_rho = float(mfm_cfg.get("land_rho", 1e-3))
    mfm_land_samples = int(mfm_cfg.get("land_metric_samples", 256))
    mfm_reference_pool_policy = str(mfm_cfg.get("reference_pool_policy", "endpoints_only"))
    mfm_moment_eta = float(mfm_cfg.get("moment_eta", 1.0))

    g_model: PathCorrection | None = None
    if mode == "constrained" or (mode in METRIC_MODES and mfm_alpha != 0.0):
        g_model = PathCorrection(
            state_dim=state_dim,
            hidden_dims=model_cfg["path_hidden_dims"],
            activation=model_cfg["activation"],
        ).to(device=device, dtype=dtype)

    train_cfg = cfg["train"]
    times = [float(t) for t in cfg["data"]["constraint_times"]]
    raw_interpolant_eval_times = cfg.get("data", {}).get("interpolant_eval_times", None)
    if raw_interpolant_eval_times is None:
        interpolant_eval_times_override: list[float] | None = None
    else:
        interpolant_eval_times_override = sorted({float(t) for t in raw_interpolant_eval_times})
        if not interpolant_eval_times_override:
            raise ValueError("data.interpolant_eval_times must be non-empty when provided.")
    coupling = str(cfg["data"].get("coupling", "ot")).lower()
    batch_size = int(train_cfg["batch_size"])
    eval_batch_size = int(train_cfg["eval_batch_size"])
    stage_a_only = is_stage_a_only_profile(train_cfg)
    stage_steps = {
        "stage_a_steps": int(train_cfg["stage_a_steps"]),
        "stage_b_steps": int(train_cfg["stage_b_steps"]),
        "stage_c_steps": int(train_cfg["stage_c_steps"]),
    }
    eval_full_ot_metrics = bool(train_cfg.get("eval_full_ot_metrics", False))
    eval_full_ot_method = str(train_cfg.get("eval_full_ot_method", "pot_emd2")).strip().lower()
    if eval_full_ot_method not in {"exact_lp", "pot_emd2"}:
        raise ValueError(
            f"Unsupported train.eval_full_ot_method '{eval_full_ot_method}'. "
            "Expected one of: exact_lp, pot_emd2."
        )
    eval_full_ot_num_itermax_raw = train_cfg.get("eval_full_ot_num_itermax", None)
    eval_full_ot_num_itermax = (
        None if eval_full_ot_num_itermax_raw is None else int(eval_full_ot_num_itermax_raw)
    )
    if eval_full_ot_num_itermax is not None and eval_full_ot_num_itermax <= 0:
        raise ValueError(
            "train.eval_full_ot_num_itermax must be positive when provided, "
            f"got {eval_full_ot_num_itermax}."
        )
    eval_full_ot_max_variables_raw = train_cfg.get("eval_full_ot_max_variables", None)
    eval_full_ot_max_variables = (
        None if eval_full_ot_max_variables_raw is None else int(eval_full_ot_max_variables_raw)
    )
    eval_full_ot_support_tol = float(train_cfg.get("eval_full_ot_support_tol", 1e-12))
    pseudo_eta = float(train_cfg.get("pseudo_eta", 0.0))
    if pseudo_eta < 0.0:
        raise ValueError(f"train.pseudo_eta must be non-negative, got {pseudo_eta}")
    pseudo_rho = float(train_cfg.get("pseudo_rho", train_cfg.get("rho", 1.0)))
    if pseudo_rho <= 0.0:
        raise ValueError(f"train.pseudo_rho must be positive, got {pseudo_rho}")
    pseudo_lambda_clip = float(
        train_cfg.get("pseudo_lambda_clip", train_cfg.get("lambda_clip", 100.0))
    )
    if pseudo_lambda_clip <= 0.0:
        raise ValueError(
            f"train.pseudo_lambda_clip must be positive, got {pseudo_lambda_clip}"
        )

    if stage_a_only and mode not in {"constrained", *METRIC_MODES}:
        raise ValueError(
            "Stage-A-only profile requires one of: constrained, metric, metric_alpha0, "
            "metric_constrained_al, metric_constrained_soft."
        )
    if mode in METRIC_MODES and stage_steps["stage_c_steps"] > 0:
        raise ValueError("Metric modes currently require stage_c_steps=0.")
    if mfm_moment_eta < 0.0:
        raise ValueError(f"mfm.moment_eta must be non-negative, got {mfm_moment_eta}")
    if data_family in {"bridge_sde", "single_cell"} and target_sampler is None:
        raise ValueError(f"{data_family} data requires a target_sampler for evaluation metrics.")
    if pseudo_eta > 0.0:
        if data_family != "single_cell":
            raise ValueError(
                "Pseudo constraints are supported for data.family=single_cell only in v1."
            )
        if pseudo_targets is None or pseudo_posterior is None:
            raise ValueError(
                "train.pseudo_eta>0 requires pseudo_targets and pseudo_posterior from single-cell prep."
            )
        missing = [float(t) for t in times if float(t) not in pseudo_targets]
        if missing:
            raise ValueError(
                "Pseudo targets are missing constrained times: "
                + ", ".join(f"{float(t):.6f}" for t in missing)
            )

    mfm_backend: MetricBackend | None = None
    metric_reference_pool: torch.Tensor | None = None
    if mode in METRIC_MODES:
        mfm_backend = build_metric_backend(
            requested_backend=mfm_requested_backend,
            geopath_net=g_model,
            sigma=mfm_sigma,
            alpha=mfm_alpha,
        )
        if g_model is not None and mfm_alpha != 0.0:
            metric_reference_pool = _build_metric_reference_pool(
                problem=problem,
                target_sampler=target_sampler,
                times=times,
                n_samples_per_time=mfm_land_samples,
                generator=generator,
                reference_pool_policy=mfm_reference_pool_policy,
            )

    history: list[dict[str, float | str | int]] = []
    global_step = 0
    lambdas: dict[float, torch.Tensor] = {}
    pseudo_lambdas: dict[float, torch.Tensor] = {}
    constrained_beta_schedule: dict[str, Any] | None = None
    pseudo_constraints_active = bool(
        pseudo_eta > 0.0
        and pseudo_targets is not None
        and pseudo_posterior is not None
        and mode in {"constrained", *METRIC_CONSTRAINED_MODES}
    )

    if mode in {"constrained", *METRIC_AL_MODES}:
        lambdas = _init_lambdas(times=times, targets=targets)
    if pseudo_constraints_active and mode in {"constrained", *METRIC_AL_MODES}:
        pseudo_lambdas = _init_lambdas(times=times, targets=pseudo_targets)
    if mode == "constrained":
        constrained_beta_schedule = _build_constrained_beta_schedule(
            problem=problem,
            targets=targets,
            constraint_times=times,
            beta0=float(train_cfg["beta"]),
            beta_schedule=str(train_cfg.get("beta_schedule", "constant")),
            drift_p=float(train_cfg.get("beta_drift_p", 1.0)),
            drift_eps=float(train_cfg.get("beta_drift_eps", 1e-6)),
            min_scale=float(train_cfg.get("beta_min_scale", 0.3)),
            max_scale=float(train_cfg.get("beta_max_scale", 3.0)),
        )

    if mode == "constrained" and g_model is not None:
        optimizer_g = torch.optim.Adam(g_model.parameters(), lr=float(train_cfg["lr_g"]))
        for step in range(int(train_cfg["stage_a_steps"])):
            x0, x1, _ = sample_coupled_batch(
                problem,
                batch_size=batch_size,
                coupling=coupling,
                generator=generator,
            )
            optimizer_g.zero_grad(set_to_none=True)
            loss_g, residuals, pseudo_residuals, stats = _constrained_objective(
                g_model=g_model,
                x0=x0,
                x1=x1,
                times=times,
                targets=targets,
                lambdas=lambdas,
                rho=float(train_cfg["rho"]),
                alpha=float(train_cfg["alpha"]),
                beta=float(train_cfg["beta"]),
                beta_schedule=constrained_beta_schedule,
                pseudo_targets=pseudo_targets,
                pseudo_lambdas=(pseudo_lambdas if pseudo_lambdas else None),
                pseudo_rho=float(pseudo_rho),
                pseudo_eta=float(pseudo_eta),
                pseudo_posterior=pseudo_posterior,
            )
            loss_g.backward()
            optimizer_g.step()
            lambdas = update_lagrange_multipliers(
                lambdas=lambdas,
                residuals=residuals,
                rho=float(train_cfg["rho"]),
                clip_value=float(train_cfg["lambda_clip"]),
            )
            if pseudo_constraints_active and pseudo_residuals is not None and pseudo_lambdas:
                pseudo_lambdas = update_lagrange_multipliers(
                    lambdas=pseudo_lambdas,
                    residuals=pseudo_residuals,
                    rho=float(pseudo_rho),
                    clip_value=float(pseudo_lambda_clip),
                )
                pseudo_avg_residual_norm = float(
                    np.mean(list(residual_norms(pseudo_residuals).values()))
                )
            else:
                pseudo_avg_residual_norm = None
            history.append(
                {
                    "stage": "stage_a",
                    "step": step,
                    "global_step": global_step,
                    "loss": float(loss_g.detach().item()),
                    "avg_residual_norm": float(np.mean(list(residual_norms(residuals).values()))),
                    "pseudo_avg_residual_norm": pseudo_avg_residual_norm,
                    "regularizer": stats["regularizer"],
                    "pseudo_term": stats["pseudo_term"],
                }
            )
            global_step += 1
    elif mode in METRIC_MODES and g_model is not None and mfm_alpha != 0.0:
        if metric_reference_pool is None:
            raise ValueError("metric_reference_pool must be initialized for metric mode.")
        optimizer_g = torch.optim.Adam(g_model.parameters(), lr=float(train_cfg["lr_g"]))
        for step in range(int(train_cfg["stage_a_steps"])):
            x0, x1, _ = sample_coupled_batch(
                problem,
                batch_size=batch_size,
                coupling=coupling,
                generator=generator,
            )
            optimizer_g.zero_grad(set_to_none=True)
            pseudo_residuals: dict[float, torch.Tensor] | None = None
            if mode in METRIC_CONSTRAINED_MODES:
                loss_g, residuals, pseudo_residuals, stats = _metric_constrained_geopath_objective(
                    mode=mode,
                    geopath_model=g_model,
                    alpha_mfm=float(mfm_alpha),
                    x0=x0,
                    x1=x1,
                    manifold_samples=metric_reference_pool,
                    times=times,
                    targets=targets,
                    lambdas=lambdas,
                    rho=float(train_cfg["rho"]),
                    land_gamma=mfm_land_gamma,
                    land_rho=mfm_land_rho,
                    moment_eta=float(mfm_moment_eta),
                    pseudo_targets=pseudo_targets,
                    pseudo_lambdas=(pseudo_lambdas if pseudo_lambdas else None),
                    pseudo_rho=float(pseudo_rho),
                    pseudo_eta=float(pseudo_eta),
                    pseudo_posterior=pseudo_posterior,
                )
            else:
                loss_g, stats = _metric_geopath_objective(
                    geopath_model=g_model,
                    alpha_mfm=float(mfm_alpha),
                    x0=x0,
                    x1=x1,
                    manifold_samples=metric_reference_pool,
                    land_gamma=mfm_land_gamma,
                    land_rho=mfm_land_rho,
                )
            loss_g.backward()
            optimizer_g.step()
            if mode in METRIC_AL_MODES:
                lambdas = update_lagrange_multipliers(
                    lambdas=lambdas,
                    residuals=residuals,
                    rho=float(train_cfg["rho"]),
                    clip_value=float(train_cfg["lambda_clip"]),
                )
                if pseudo_constraints_active and pseudo_residuals is not None and pseudo_lambdas:
                    pseudo_lambdas = update_lagrange_multipliers(
                        lambdas=pseudo_lambdas,
                        residuals=pseudo_residuals,
                        rho=float(pseudo_rho),
                        clip_value=float(pseudo_lambda_clip),
                    )
            history_row: dict[str, float | int | str] = {
                "stage": "stage_a",
                "step": step,
                "global_step": global_step,
                "loss": float(loss_g.detach().item()),
                "land_loss": stats["land_loss"],
            }
            if "moment_term" in stats:
                history_row["moment_term"] = float(stats["moment_term"])
            if "moment_sq_mean" in stats:
                history_row["moment_sq_mean"] = float(stats["moment_sq_mean"])
            if "pseudo_term" in stats:
                history_row["pseudo_term"] = float(stats["pseudo_term"])
            if "pseudo_sq_mean" in stats:
                history_row["pseudo_sq_mean"] = float(stats["pseudo_sq_mean"])
            if mode in METRIC_CONSTRAINED_MODES:
                history_row["avg_residual_norm"] = float(
                    np.mean(list(residual_norms(residuals).values()))
                )
                if pseudo_residuals is not None:
                    history_row["pseudo_avg_residual_norm"] = float(
                        np.mean(list(residual_norms(pseudo_residuals).values()))
                    )
            history.append(history_row)
            global_step += 1

    optimizer_v = torch.optim.Adam(v_model.parameters(), lr=float(train_cfg["lr_v"]))
    if mode in {"constrained", *METRIC_MODES} and g_model is not None:
        for param in g_model.parameters():
            param.requires_grad_(False)

    for step in range(int(train_cfg["stage_b_steps"])):
        x0, x1, _ = sample_coupled_batch(
            problem,
            batch_size=batch_size,
            coupling=coupling,
            generator=generator,
        )
        optimizer_v.zero_grad(set_to_none=True)
        loss_v, energy_proxy = _cfm_loss(
            mode=mode,
            v_model=v_model,
            g_model=g_model,
            x0=x0,
            x1=x1,
            mfm_backend=mfm_backend,
        )
        loss_v.backward()
        optimizer_v.step()
        history.append(
            {
                "stage": "stage_b",
                "step": step,
                "global_step": global_step,
                "loss": float(loss_v.detach().item()),
                "path_energy_proxy": float(energy_proxy),
            }
        )
        global_step += 1

    if mode == "constrained" and g_model is not None:
        for param in g_model.parameters():
            param.requires_grad_(True)
        optimizer_joint = torch.optim.Adam(
            [
                {"params": v_model.parameters(), "lr": float(train_cfg["lr_v"])},
                {"params": g_model.parameters(), "lr": float(train_cfg["lr_g"])},
            ]
        )
        for step in range(int(train_cfg["stage_c_steps"])):
            x0, x1, _ = sample_coupled_batch(
                problem,
                batch_size=batch_size,
                coupling=coupling,
                generator=generator,
            )
            optimizer_joint.zero_grad(set_to_none=True)
            cfm, energy_proxy = _cfm_loss(
                mode="constrained",
                v_model=v_model,
                g_model=g_model,
                x0=x0,
                x1=x1,
            )
            lg, residuals, pseudo_residuals, stats = _constrained_objective(
                g_model=g_model,
                x0=x0,
                x1=x1,
                times=times,
                targets=targets,
                lambdas=lambdas,
                rho=float(train_cfg["rho"]),
                alpha=float(train_cfg["alpha"]),
                beta=float(train_cfg["beta"]),
                beta_schedule=constrained_beta_schedule,
                pseudo_targets=pseudo_targets,
                pseudo_lambdas=(pseudo_lambdas if pseudo_lambdas else None),
                pseudo_rho=float(pseudo_rho),
                pseudo_eta=float(pseudo_eta),
                pseudo_posterior=pseudo_posterior,
            )
            joint = cfm + float(train_cfg["eta_joint"]) * lg
            joint.backward()
            optimizer_joint.step()
            lambdas = update_lagrange_multipliers(
                lambdas=lambdas,
                residuals=residuals,
                rho=float(train_cfg["rho"]),
                clip_value=float(train_cfg["lambda_clip"]),
            )
            if pseudo_constraints_active and pseudo_residuals is not None and pseudo_lambdas:
                pseudo_lambdas = update_lagrange_multipliers(
                    lambdas=pseudo_lambdas,
                    residuals=pseudo_residuals,
                    rho=float(pseudo_rho),
                    clip_value=float(pseudo_lambda_clip),
                )
                pseudo_avg_residual_norm = float(
                    np.mean(list(residual_norms(pseudo_residuals).values()))
                )
            else:
                pseudo_avg_residual_norm = None
            history.append(
                {
                    "stage": "stage_c",
                    "step": step,
                    "global_step": global_step,
                    "loss": float(joint.detach().item()),
                    "cfm_loss": float(cfm.detach().item()),
                    "path_energy_proxy": float(energy_proxy),
                    "avg_residual_norm": float(np.mean(list(residual_norms(residuals).values()))),
                    "pseudo_avg_residual_norm": pseudo_avg_residual_norm,
                    "pseudo_term": stats["pseudo_term"],
                }
            )
            global_step += 1

    v_model.eval()
    if g_model is not None:
        g_model.eval()

    eval_residuals = _eval_constraint_norms(
        mode=mode,
        problem=problem,
        coupling=coupling,
        batch_size=eval_batch_size,
        times=times,
        targets=targets,
        g_model=g_model,
        mfm_alpha=mfm_alpha,
        generator=generator,
    )
    eval_pseudo_residuals: dict[float, float] | None = None
    if pseudo_constraints_active and pseudo_targets is not None and pseudo_posterior is not None:
        eval_pseudo_residuals = _eval_pseudo_constraint_norms(
            mode=mode,
            problem=problem,
            coupling=coupling,
            batch_size=eval_batch_size,
            times=times,
            pseudo_targets=pseudo_targets,
            pseudo_posterior=pseudo_posterior,
            g_model=g_model,
            mfm_alpha=mfm_alpha,
            generator=generator,
        )

    transport: dict[str, float | dict[str, float] | None] = {
        "transport_mean_error_l2": None,
        "transport_cov_error_fro": None,
        "transport_score": None,
        "transport_endpoint_empirical_w2": None,
        "transport_endpoint_empirical_w1": None,
        "intermediate_w2_gaussian": None,
        "intermediate_w2_gaussian_avg": None,
        "intermediate_empirical_w2": None,
        "intermediate_empirical_w2_avg": None,
        "intermediate_empirical_w1": None,
        "intermediate_empirical_w1_avg": None,
        "intermediate_full_ot_w2": None,
        "intermediate_full_ot_w2_avg": None,
        "transport_endpoint_full_ot_w2": None,
        "holdout_full_ot_w2": None,
        "holdout_empirical_w2": None,
        "holdout_empirical_w1": None,
    }
    cfm_val: float | None = None
    eval_path_energy: float | None = None
    interpolant_artifacts: dict[str, Any] | None = None
    interpolant_eval: dict[str, float | dict[str, float]] | None = None
    rollout_artifacts: dict[str, Any] | None = None

    if not stage_a_only:
        cfm_val, eval_path_energy = _eval_cfm_loss(
            mode=mode,
            problem=problem,
            coupling=coupling,
            v_model=v_model,
            g_model=g_model,
            mfm_backend=mfm_backend,
            batch_size=eval_batch_size,
            generator=generator,
        )
        if data_family in {"bridge_sde", "single_cell"}:
            if target_sampler is None:
                raise ValueError(f"{data_family} non-Stage-A-only evaluation requires target_sampler.")
            holdout_time_raw = cfg.get("experiment", {}).get("holdout_time", None)
            holdout_time = None if holdout_time_raw is None else float(holdout_time_raw)
            empirical_metrics, rollout_artifacts = _eval_empirical_rollout_metrics(
                problem=problem,
                coupling=coupling,
                v_model=v_model,
                times=times,
                n_samples=int(train_cfg.get("eval_intermediate_ot_samples", 256)),
                n_steps=int(train_cfg["eval_transport_steps"]),
                target_sampler=target_sampler,
                generator=generator,
                holdout_time=holdout_time,
            )
            transport.update(empirical_metrics)
            if data_family == "single_cell" and eval_full_ot_metrics:
                if target_samples_by_time is None:
                    raise ValueError(
                        "train.eval_full_ot_metrics=true requires target_samples_by_time for single-cell data."
                    )
                full_ot_metrics, full_ot_artifacts = _eval_full_ot_rollout_metrics(
                    problem=problem,
                    v_model=v_model,
                    times=times,
                    n_steps=int(train_cfg["eval_transport_steps"]),
                    target_samples_by_time=target_samples_by_time,
                    holdout_time=holdout_time,
                    method=eval_full_ot_method,
                    num_itermax=eval_full_ot_num_itermax,
                    max_variables=eval_full_ot_max_variables,
                    support_tol=eval_full_ot_support_tol,
                )
                transport.update(full_ot_metrics)
                if rollout_artifacts is None:
                    rollout_artifacts = {}
                rollout_artifacts["full_ot"] = full_ot_artifacts
        elif isinstance(problem, GaussianOTProblem):
            transport_metrics = transport_quality_metrics(
                velocity_fn=v_model,
                problem=problem,
                n_samples=int(train_cfg["eval_transport_samples"]),
                n_steps=int(train_cfg["eval_transport_steps"]),
                generator=generator,
            )
            intermediate_w2 = intermediate_wasserstein_metrics(
                velocity_fn=v_model,
                problem=problem,
                times=times,
                n_samples=int(train_cfg["eval_transport_samples"]),
                n_steps=int(train_cfg["eval_transport_steps"]),
                generator=generator,
            )
            empirical_w2: dict[str, float | dict[str, float]] = {}
            if bool(train_cfg.get("eval_intermediate_empirical_w2", True)):
                empirical_w2 = intermediate_empirical_w2_metrics(
                    velocity_fn=v_model,
                    problem=problem,
                    times=times,
                    n_samples=int(train_cfg.get("eval_intermediate_ot_samples", 256)),
                    n_steps=int(train_cfg["eval_transport_steps"]),
                    target_sampler=target_sampler,
                    generator=generator,
                )
            transport.update(transport_metrics)
            transport.update(intermediate_w2)
            transport.update(empirical_w2)
    else:
        target_sampler_fn = target_sampler
        if target_sampler_fn is None:
            if isinstance(problem, GaussianOTProblem):
                def _gaussian_target_sampler(
                    t: float,
                    n_samples: int,
                    generator: torch.Generator | None = None,
                ) -> torch.Tensor:
                    mean_t = analytic_bridge_mean(float(t), problem)
                    cov_t = analytic_bridge_cov(float(t), problem)
                    return sample_gaussian(mean_t, cov_t, n_samples=n_samples, generator=generator)

                target_sampler_fn = _gaussian_target_sampler
            else:
                raise ValueError("Stage-A-only interpolant evaluation requires target_sampler.")
        holdout_time_raw = cfg.get("experiment", {}).get("holdout_time", None)
        holdout_time = None if holdout_time_raw is None else float(holdout_time_raw)
        if interpolant_eval_times_override is not None:
            interpolant_eval_times = list(interpolant_eval_times_override)
        else:
            interpolant_eval_times = sorted(
                {float(t) for t in times} | ({float(holdout_time)} if holdout_time is not None else set())
            )
        x0_eval, x1_eval, _ = sample_coupled_batch(
            problem,
            batch_size=int(train_cfg.get("eval_intermediate_ot_samples", 256)),
            coupling=coupling,
            generator=generator,
        )
        interpolant_eval = interpolant_empirical_w2_metrics(
            x0=x0_eval,
            x1=x1_eval,
            times=interpolant_eval_times,
            target_sampler=target_sampler_fn,
            g_model=g_model,
            mode=mode,
            mfm_alpha=float(mfm_alpha),
            holdout_time=holdout_time,
            generator=generator,
        )
        if data_family == "single_cell" and eval_full_ot_metrics:
            if target_samples_by_time is None:
                raise ValueError(
                    "train.eval_full_ot_metrics=true requires target_samples_by_time for single-cell data."
                )
            if not isinstance(problem, EmpiricalCouplingProblem) or not problem.has_global_ot_support:
                raise ValueError(
                    "train.eval_full_ot_metrics=true for single-cell requires coupling='ot_global' "
                    "with a cached global OT plan."
                )
            if problem.global_ot_src_idx is None or problem.global_ot_tgt_idx is None or problem.global_ot_mass is None:
                raise ValueError("Global OT support tensors are missing for full-OT interpolant evaluation.")
            full_ot_interp = interpolant_full_ot_w2_metrics(
                x0_pool=problem.x0_pool,
                x1_pool=problem.x1_pool,
                plan_src_idx=problem.global_ot_src_idx,
                plan_tgt_idx=problem.global_ot_tgt_idx,
                plan_mass=problem.global_ot_mass,
                times=interpolant_eval_times,
                target_samples_by_time=target_samples_by_time,
                g_model=g_model,
                mode=mode,
                mfm_alpha=float(mfm_alpha),
                holdout_time=holdout_time,
                method=eval_full_ot_method,
                num_itermax=eval_full_ot_num_itermax,
                max_variables=eval_full_ot_max_variables,
                support_tol=eval_full_ot_support_tol,
            )
            interpolant_eval.update(full_ot_interp)
        linear_by_time, learned_by_time, target_by_time = interpolant_snapshot_sets(
            x0=x0_eval,
            x1=x1_eval,
            times=interpolant_eval_times,
            target_sampler=target_sampler_fn,
            g_model=g_model,
            mode=mode,
            mfm_alpha=float(mfm_alpha),
            generator=generator,
        )
        interpolant_artifacts = {
            "x0": x0_eval.detach().cpu(),
            "x1": x1_eval.detach().cpu(),
            "linear_by_time": _to_cpu_snapshot_dict(linear_by_time),
            "learned_by_time": _to_cpu_snapshot_dict(learned_by_time),
            "target_by_time": _to_cpu_snapshot_dict(target_by_time),
        }

    summary: dict[str, Any] = {
        "mode": mode,
        "coupling": coupling,
        "data_family": data_family,
        "stage_steps": stage_steps,
        "stage_a_only": bool(stage_a_only),
        "stage_c_enabled": bool(stage_steps["stage_c_steps"] > 0),
        "eval_full_ot_metrics": bool(eval_full_ot_metrics),
        "eval_full_ot_method": str(eval_full_ot_method),
        "eval_full_ot_num_itermax": eval_full_ot_num_itermax,
        "eval_full_ot_max_variables": eval_full_ot_max_variables,
        "eval_full_ot_support_tol": float(eval_full_ot_support_tol),
        "constraint_times": [float(t) for t in times],
        "interpolant_eval_times": (
            None if interpolant_eval_times_override is None else [float(t) for t in interpolant_eval_times_override]
        ),
        "cfm_val_loss": cfm_val,
        "path_energy_proxy": eval_path_energy,
        "constraint_residual_norms": {f"{k:.2f}": v for k, v in eval_residuals.items()},
        "constraint_residual_avg": float(np.mean(list(eval_residuals.values()))),
        "pseudo_constraint_residual_norms": (
            None
            if eval_pseudo_residuals is None
            else {f"{k:.2f}": v for k, v in eval_pseudo_residuals.items()}
        ),
        "pseudo_constraint_residual_avg": (
            None
            if eval_pseudo_residuals is None
            else float(np.mean(list(eval_pseudo_residuals.values())))
        ),
        "pseudo_constraints_active": bool(pseudo_constraints_active),
        "pseudo_eta": float(pseudo_eta),
        "pseudo_rho": float(pseudo_rho),
        "pseudo_lambda_clip": float(pseudo_lambda_clip),
        "seed": int(cfg["seed"]),
    }
    if "protocol" in cfg.get("experiment", {}):
        summary["protocol"] = str(cfg["experiment"].get("protocol"))
    if "holdout_index" in cfg.get("experiment", {}):
        summary["holdout_index"] = cfg["experiment"].get("holdout_index")
    if "holdout_time" in cfg.get("experiment", {}):
        summary["holdout_time"] = cfg["experiment"].get("holdout_time")
    if isinstance(problem, EmpiricalCouplingProblem):
        summary["global_ot_support_size"] = (
            None if problem.global_ot_mass is None else int(problem.global_ot_mass.numel())
        )
        summary["global_ot_total_cost"] = (
            None if problem.global_ot_total_cost is None else float(problem.global_ot_total_cost)
        )
    if mode == "constrained" and constrained_beta_schedule is not None:
        summary["beta_schedule"] = str(constrained_beta_schedule["name"])
        summary["beta_schedule_base"] = float(constrained_beta_schedule["base_beta"])
        summary["beta_schedule_anchor_times"] = [float(t) for t in constrained_beta_schedule["anchors"]]
        summary["beta_schedule_interval_drifts"] = [
            float(v) for v in constrained_beta_schedule["interval_drifts"]
        ]
        summary["beta_schedule_interval_values"] = [
            float(v) for v in constrained_beta_schedule["interval_betas"]
        ]
        summary["beta_schedule_anchor_values"] = [
            float(v) for v in constrained_beta_schedule["anchor_betas"]
        ]
        summary["beta_schedule_drift_mean"] = float(constrained_beta_schedule["drift_mean"])
        summary["beta_schedule_drift_p"] = float(constrained_beta_schedule["drift_p"])
        summary["beta_schedule_drift_eps"] = float(constrained_beta_schedule["drift_eps"])
        summary["beta_schedule_min_scale"] = float(constrained_beta_schedule["min_scale"])
        summary["beta_schedule_max_scale"] = float(constrained_beta_schedule["max_scale"])
    if mode in METRIC_MODES:
        summary["mfm_backend"] = None if mfm_backend is None else mfm_backend.name
        summary["mfm_backend_impl"] = None if mfm_backend is None else mfm_backend.impl
        summary["mfm_alpha"] = float(mfm_alpha)
        summary["mfm_sigma"] = float(mfm_sigma)
        summary["mfm_land_gamma"] = float(mfm_land_gamma)
        summary["mfm_land_rho"] = float(mfm_land_rho)
        summary["mfm_reference_pool_policy"] = str(mfm_reference_pool_policy)
        summary["mfm_moment_style"] = _metric_moment_style(mode)
        summary["mfm_moment_eta"] = float(mfm_moment_eta)
    summary.update(transport)
    if interpolant_eval is not None:
        summary["interpolant_eval"] = interpolant_eval

    checkpoint = {
        "velocity_state_dict": v_model.state_dict(),
        "path_state_dict": None if g_model is None else g_model.state_dict(),
        "mode": mode,
        "config": cfg,
        "summary": summary,
    }
    return {
        "summary": summary,
        "history": history,
        "checkpoint": checkpoint,
        "velocity_model": v_model,
        "path_model": g_model,
        "interpolant_artifacts": interpolant_artifacts,
        "rollout_artifacts": rollout_artifacts,
    }
