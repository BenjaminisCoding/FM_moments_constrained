from __future__ import annotations

import random
import time
from typing import Any, Callable

import numpy as np
import torch

from cfm_project.constraints import (
    augmented_lagrangian_block_terms,
    augmented_lagrangian_terms,
    block_residual_norms,
    constraint_residual_blocks,
    constraint_residuals,
    normalize_moment_block_normalization,
    normalize_moment_feature_blocks,
    normalize_residual_blocks,
    residual_norms,
    select_moment_feature_blocks,
    split_moment_feature_vector,
    update_lagrange_multiplier_blocks,
    update_lagrange_multipliers,
)
from cfm_project.curly_core import (
    CurlyReferencePool,
    curly_drift_alignment_loss,
    curly_learned_coupling,
    curly_mean_path,
    curly_path_and_velocity,
    knn_reference_velocity,
    normalize_reference_pool_policy,
    normalize_stage_b_coupling,
    select_reference_times,
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
    RBFMetric,
    build_metric_backend,
    fit_rbf_metric,
    land_geopath_loss,
    mfm_mean_path,
    mfm_path_and_velocity,
    rbf_geopath_loss,
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
CURLY_BASE_MODES = {"curly"}
CURLY_CONSTRAINED_MODES = {"curly_constrained_al"}
CURLY_MODES = CURLY_BASE_MODES | CURLY_CONSTRAINED_MODES
CURLY_AL_MODES = {"curly_constrained_al"}
CONSTRAINED_BETA_SCHEDULES = {"constant", "piecewise", "linear"}


def _metric_moment_style(mode: str) -> str:
    if mode in METRIC_AL_MODES:
        return "al"
    if mode in METRIC_SOFT_MODES:
        return "soft"
    return "none"


def _curly_moment_style(mode: str) -> str:
    if mode in CURLY_AL_MODES:
        return "al"
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


def _uniform_time(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    return torch.rand((batch_size, 1), device=device, dtype=dtype, generator=generator)


def _endpoint_moment_feature(
    problem: CouplingProblem,
    t: float,
    moment_feature_blocks: tuple[str, ...],
    moment_feature_params: object | None = None,
) -> torch.Tensor:
    if isinstance(problem, GaussianOTProblem):
        mean_t = analytic_bridge_mean(float(t), problem)
        cov_t = analytic_bridge_cov(float(t), problem)
        return gaussian_moment_feature_vector(
            mean_t,
            cov_t,
            feature_blocks=moment_feature_blocks,
            feature_params=moment_feature_params,
        )
    if isinstance(problem, EmpiricalCouplingProblem):
        pool = problem.x0_pool if float(t) <= 0.0 else problem.x1_pool
        return moment_feature_vector_from_samples(
            pool,
            feature_blocks=moment_feature_blocks,
            feature_params=moment_feature_params,
        )
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
    moment_feature_blocks: tuple[str, ...],
    moment_feature_params: object | None = None,
) -> torch.Tensor:
    if float(t) <= 0.0:
        return _endpoint_moment_feature(
            problem,
            t=0.0,
            moment_feature_blocks=moment_feature_blocks,
            moment_feature_params=moment_feature_params,
        )
    if float(t) >= 1.0:
        return _endpoint_moment_feature(
            problem,
            t=1.0,
            moment_feature_blocks=moment_feature_blocks,
            moment_feature_params=moment_feature_params,
        )
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
    moment_feature_blocks: tuple[str, ...] | None = None,
    moment_feature_params: object | None = None,
) -> dict[str, Any]:
    active_blocks = normalize_moment_feature_blocks(moment_feature_blocks)
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

    anchor_features = [
        _anchor_moment_feature(
            problem=problem,
            targets=targets,
            t=t,
            moment_feature_blocks=active_blocks,
            moment_feature_params=moment_feature_params,
        )
        for t in anchors
    ]
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


def _beta_schedule_has_nonzero_weight(
    beta0: float,
    beta_schedule: dict[str, Any] | None,
) -> bool:
    if abs(float(beta0)) > 0.0:
        return True
    if beta_schedule is None:
        return False
    for key in ("interval_betas", "anchor_betas"):
        values = beta_schedule.get(key, [])
        if any(abs(float(value)) > 0.0 for value in values):
            return True
    return False


def _constraint_residuals_for_mode(
    mode: str,
    x0: torch.Tensor,
    x1: torch.Tensor,
    times: list[float],
    targets: dict[float, torch.Tensor],
    g_model: PathCorrection | None,
    mfm_alpha: float = 1.0,
    curly_path_alpha: float = 1.0,
    moment_feature_blocks: tuple[str, ...] | None = None,
    moment_feature_params: object | None = None,
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
            curly_path_alpha=curly_path_alpha,
        )

    return constraint_residuals(
        path_fn=path_fn,
        times=times,
        targets=targets,
        feature_blocks=moment_feature_blocks,
        feature_params=moment_feature_params,
    )


def _constraint_residual_blocks_for_mode(
    mode: str,
    x0: torch.Tensor,
    x1: torch.Tensor,
    times: list[float],
    targets: dict[float, torch.Tensor],
    g_model: PathCorrection | None,
    mfm_alpha: float = 1.0,
    curly_path_alpha: float = 1.0,
    moment_feature_blocks: tuple[str, ...] | None = None,
    moment_feature_params: object | None = None,
) -> dict[float, dict[str, torch.Tensor]]:
    def path_fn(t_value: float) -> torch.Tensor:
        t_batch = torch.full((x0.shape[0], 1), t_value, device=x0.device, dtype=x0.dtype)
        return _path_samples_for_mode(
            mode=mode,
            x0=x0,
            x1=x1,
            t_batch=t_batch,
            g_model=g_model,
            mfm_alpha=mfm_alpha,
            curly_path_alpha=curly_path_alpha,
        )

    return constraint_residual_blocks(
        path_fn=path_fn,
        times=times,
        targets=targets,
        feature_blocks=moment_feature_blocks,
        feature_params=moment_feature_params,
    )


def _path_samples_for_mode(
    mode: str,
    x0: torch.Tensor,
    x1: torch.Tensor,
    t_batch: torch.Tensor,
    g_model: PathCorrection | None,
    mfm_alpha: float = 1.0,
    curly_path_alpha: float = 1.0,
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
    if mode in CURLY_MODES:
        return curly_mean_path(
            t=t_batch,
            x0=x0,
            x1=x1,
            geopath_net=g_model,
            path_alpha=float(curly_path_alpha),
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
    curly_path_alpha: float = 1.0,
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
            curly_path_alpha=curly_path_alpha,
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


def _avg_flat_residual_norm(residuals: dict[float, torch.Tensor]) -> float:
    if not residuals:
        return 0.0
    return float(np.mean(list(residual_norms(residuals).values())))


def _avg_block_residual_norm(residuals: dict[float, dict[str, torch.Tensor]]) -> float:
    values: list[float] = []
    for block_values in block_residual_norms(residuals).values():
        values.extend(float(v) for v in block_values.values())
    return float(np.mean(values)) if values else 0.0


def _block_residual_squared_mean(
    residuals: dict[float, dict[str, torch.Tensor]],
    ref: torch.Tensor,
) -> torch.Tensor:
    terms: list[torch.Tensor] = []
    for block_residuals in residuals.values():
        for res in block_residuals.values():
            terms.append(torch.dot(res, res))
    if terms:
        return torch.stack(terms).mean()
    return torch.zeros((), device=ref.device, dtype=ref.dtype)


def _block_error_summary(
    residuals: dict[float, dict[str, torch.Tensor]],
) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for t, block_residuals in residuals.items():
        time_key = f"{float(t):.2f}"
        for block, res in block_residuals.items():
            l2 = float(torch.linalg.norm(res).detach().item())
            rmse = float(torch.sqrt(torch.mean(res.detach() ** 2)).item())
            out.setdefault(block, {})[time_key] = {
                "l2": l2,
                "rmse": rmse,
            }
    return out


def _block_l2_summary(
    residuals: dict[float, dict[str, torch.Tensor]],
) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for t, block_residuals in residuals.items():
        time_key = f"{float(t):.2f}"
        for block, res in block_residuals.items():
            out.setdefault(block, {})[time_key] = {
                "l2": float(torch.linalg.norm(res).detach().item()),
            }
    return out


def _avg_block_metric(
    summary: dict[str, dict[str, dict[str, float]]],
    metric: str,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for block, by_time in summary.items():
        values = [float(metrics[metric]) for metrics in by_time.values() if metric in metrics]
        if values:
            out[block] = float(np.mean(values))
    return out


def _eval_interpolant_moment_errors(
    mode: str,
    x0: torch.Tensor,
    x1: torch.Tensor,
    times: list[float],
    targets: dict[float, torch.Tensor],
    g_model: PathCorrection | None,
    mfm_alpha: float,
    curly_path_alpha: float,
    moment_feature_blocks: tuple[str, ...],
    moment_block_normalization: str,
    moment_feature_params: object | None = None,
) -> dict[str, Any]:
    raw_blocks = _constraint_residual_blocks_for_mode(
        mode=mode,
        x0=x0,
        x1=x1,
        times=times,
        targets=targets,
        g_model=g_model,
        mfm_alpha=float(mfm_alpha),
        curly_path_alpha=float(curly_path_alpha),
        moment_feature_blocks=moment_feature_blocks,
        moment_feature_params=moment_feature_params,
    )
    normalized_blocks = normalize_residual_blocks(
        residuals=raw_blocks,
        dim=int(x0.shape[1]),
        normalization=moment_block_normalization,
    )
    raw_summary = _block_error_summary(raw_blocks)
    normalized_summary = _block_l2_summary(normalized_blocks)
    return {
        "raw": raw_summary,
        "normalized": normalized_summary,
        "raw_l2_avg_by_block": _avg_block_metric(raw_summary, "l2"),
        "raw_rmse_avg_by_block": _avg_block_metric(raw_summary, "rmse"),
        "normalized_l2_avg_by_block": _avg_block_metric(normalized_summary, "l2"),
    }


def _constrained_objective(
    g_model: PathCorrection,
    x0: torch.Tensor,
    x1: torch.Tensor,
    times: list[float],
    targets: dict[float, torch.Tensor],
    lambdas: dict[float, torch.Tensor] | dict[float, dict[str, torch.Tensor]],
    rho: float,
    alpha: float,
    beta: float,
    moment_eta: float = 1.0,
    beta_schedule: dict[str, Any] | None = None,
    moment_feature_blocks: tuple[str, ...] | None = None,
    moment_block_normalization: str = "none",
    moment_feature_params: object | None = None,
    pseudo_targets: dict[float, torch.Tensor] | None = None,
    pseudo_lambdas: dict[float, torch.Tensor] | None = None,
    pseudo_rho: float | None = None,
    pseudo_eta: float = 0.0,
    pseudo_posterior: Callable[[torch.Tensor], torch.Tensor] | None = None,
    time_generator: torch.Generator | None = None,
) -> tuple[
    torch.Tensor,
    dict[float, torch.Tensor] | dict[float, dict[str, torch.Tensor]],
    dict[float, torch.Tensor] | None,
    dict[str, float],
]:
    t_rand = _uniform_time(x0.shape[0], x0.device, x0.dtype, generator=time_generator)
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
    beta_active = _beta_schedule_has_nonzero_weight(
        beta0=float(beta),
        beta_schedule=beta_schedule,
    )
    if beta_active:
        du_dt = vector_time_derivative(u_target, t_req, create_graph=True)
        smoothness_per_sample = torch.sum(du_dt**2, dim=1)
        smoothness = torch.mean(smoothness_per_sample)
        beta_t = _beta_weights_at_times(t=t_req.detach(), beta0=float(beta), beta_schedule=beta_schedule)
        weighted_smoothness = torch.mean(beta_t * smoothness_per_sample)
    else:
        smoothness = torch.zeros((), device=x0.device, dtype=x0.dtype)
        weighted_smoothness = torch.zeros((), device=x0.device, dtype=x0.dtype)
        beta_t = torch.zeros(x0.shape[0], device=x0.device, dtype=x0.dtype)
    regularizer = alpha * energy + weighted_smoothness

    normalization = normalize_moment_block_normalization(moment_block_normalization)
    active_blocks = normalize_moment_feature_blocks(moment_feature_blocks)
    if normalization == "none":
        residuals = _constraint_residuals_for_mode(
            mode="constrained",
            x0=x0,
            x1=x1,
            times=times,
            targets=targets,
            g_model=g_model,
            moment_feature_blocks=active_blocks,
            moment_feature_params=moment_feature_params,
        )
        al_term, per_time = augmented_lagrangian_terms(
            residuals=residuals,
            lambdas=lambdas,  # type: ignore[arg-type]
            rho=rho,
        )
        per_block: dict[str, dict[float, float]] = {}
        raw_avg_residual_norm = _avg_flat_residual_norm(residuals)
        normalized_avg_residual_norm = raw_avg_residual_norm
    else:
        raw_residuals = _constraint_residual_blocks_for_mode(
            mode="constrained",
            x0=x0,
            x1=x1,
            times=times,
            targets=targets,
            g_model=g_model,
            moment_feature_blocks=active_blocks,
            moment_feature_params=moment_feature_params,
        )
        residuals = normalize_residual_blocks(
            residuals=raw_residuals,
            dim=int(x0.shape[1]),
            normalization=normalization,
        )
        al_term, per_time, per_block = augmented_lagrangian_block_terms(
            residuals=residuals,
            lambdas=lambdas,  # type: ignore[arg-type]
            rho=rho,
        )
        raw_avg_residual_norm = _avg_block_residual_norm(raw_residuals)
        normalized_avg_residual_norm = _avg_block_residual_norm(residuals)
    total = regularizer + float(moment_eta) * al_term
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
        "moment_eta": float(moment_eta),
        "pseudo_term": float(pseudo_term.detach().item()),
        "raw_avg_residual_norm": float(raw_avg_residual_norm),
        "normalized_avg_residual_norm": float(normalized_avg_residual_norm),
    }
    for t, value in per_time.items():
        stats[f"al_t_{t:.2f}"] = float(value)
    for block, block_values in per_block.items():
        for t, value in block_values.items():
            stats[f"al_{block}_t_{t:.2f}"] = float(value)
    for t, value in pseudo_per_time.items():
        stats[f"pseudo_al_t_{t:.2f}"] = float(value)
    return total, residuals, pseudo_residuals, stats


def _metric_geopath_objective(
    geopath_model: PathCorrection,
    alpha_mfm: float,
    x0: torch.Tensor,
    x1: torch.Tensor,
    manifold_samples: torch.Tensor,
    geopath_metric: str,
    land_gamma: float,
    land_rho: float,
    rbf_metric: RBFMetric | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    mu_t, u_t, _ = mfm_path_and_velocity(
        t=None,
        x0=x0,
        x1=x1,
        geopath_net=geopath_model,
        alpha=float(alpha_mfm),
        create_graph=True,
    )
    if geopath_metric == "land":
        loss = land_geopath_loss(
            x_t=mu_t,
            u_t=u_t,
            manifold_samples=manifold_samples,
            gamma=float(land_gamma),
            rho=float(land_rho),
        )
    elif geopath_metric == "rbf":
        if rbf_metric is None:
            raise ValueError("mfm.geopath_metric=rbf requires a fitted RBF metric.")
        loss = rbf_geopath_loss(x_t=mu_t, u_t=u_t, metric=rbf_metric)
    else:
        raise ValueError(f"Unsupported mfm.geopath_metric '{geopath_metric}'.")
    stats = {
        "land_loss": float(loss.detach().item()),
        "geopath_metric_loss": float(loss.detach().item()),
    }
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
    lambdas: dict[float, torch.Tensor] | dict[float, dict[str, torch.Tensor]],
    rho: float,
    geopath_metric: str,
    land_gamma: float,
    land_rho: float,
    rbf_metric: RBFMetric | None,
    moment_eta: float,
    moment_feature_blocks: tuple[str, ...] | None = None,
    moment_block_normalization: str = "none",
    moment_feature_params: object | None = None,
    pseudo_targets: dict[float, torch.Tensor] | None = None,
    pseudo_lambdas: dict[float, torch.Tensor] | None = None,
    pseudo_rho: float | None = None,
    pseudo_eta: float = 0.0,
    pseudo_posterior: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[
    torch.Tensor,
    dict[float, torch.Tensor] | dict[float, dict[str, torch.Tensor]],
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
    if geopath_metric == "land":
        land_loss = land_geopath_loss(
            x_t=mu_t,
            u_t=u_t,
            manifold_samples=manifold_samples,
            gamma=float(land_gamma),
            rho=float(land_rho),
        )
    elif geopath_metric == "rbf":
        if rbf_metric is None:
            raise ValueError("mfm.geopath_metric=rbf requires a fitted RBF metric.")
        land_loss = rbf_geopath_loss(x_t=mu_t, u_t=u_t, metric=rbf_metric)
    else:
        raise ValueError(f"Unsupported mfm.geopath_metric '{geopath_metric}'.")
    normalization = normalize_moment_block_normalization(moment_block_normalization)
    active_blocks = normalize_moment_feature_blocks(moment_feature_blocks)
    if normalization == "none":
        residuals = _constraint_residuals_for_mode(
            mode=mode,
            x0=x0,
            x1=x1,
            times=times,
            targets=targets,
            g_model=geopath_model,
            mfm_alpha=float(alpha_mfm),
            moment_feature_blocks=active_blocks,
            moment_feature_params=moment_feature_params,
        )
        residual_sq_mean = _residual_squared_mean(residuals=residuals, ref=x0)
        per_block: dict[str, dict[float, float]] = {}
        raw_avg_residual_norm = _avg_flat_residual_norm(residuals)
        normalized_avg_residual_norm = raw_avg_residual_norm
    else:
        raw_residuals = _constraint_residual_blocks_for_mode(
            mode=mode,
            x0=x0,
            x1=x1,
            times=times,
            targets=targets,
            g_model=geopath_model,
            mfm_alpha=float(alpha_mfm),
            moment_feature_blocks=active_blocks,
            moment_feature_params=moment_feature_params,
        )
        residuals = normalize_residual_blocks(
            residuals=raw_residuals,
            dim=int(x0.shape[1]),
            normalization=normalization,
        )
        residual_sq_mean = _block_residual_squared_mean(residuals=residuals, ref=x0)
        raw_avg_residual_norm = _avg_block_residual_norm(raw_residuals)
        normalized_avg_residual_norm = _avg_block_residual_norm(residuals)

    stats: dict[str, float] = {
        "land_loss": float(land_loss.detach().item()),
        "geopath_metric_loss": float(land_loss.detach().item()),
        "moment_sq_mean": float(residual_sq_mean.detach().item()),
        "raw_avg_residual_norm": float(raw_avg_residual_norm),
        "normalized_avg_residual_norm": float(normalized_avg_residual_norm),
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
        if normalization == "none":
            al_term, per_time = augmented_lagrangian_terms(
                residuals=residuals,  # type: ignore[arg-type]
                lambdas=lambdas,  # type: ignore[arg-type]
                rho=float(rho),
            )
            per_block = {}
        else:
            al_term, per_time, per_block = augmented_lagrangian_block_terms(
                residuals=residuals,  # type: ignore[arg-type]
                lambdas=lambdas,  # type: ignore[arg-type]
                rho=float(rho),
            )
        total = land_loss + float(moment_eta) * al_term
        stats["moment_term"] = float(al_term.detach().item())
        for t, value in per_time.items():
            stats[f"al_t_{t:.2f}"] = float(value)
        for block, block_values in per_block.items():
            for t, value in block_values.items():
                stats[f"al_{block}_t_{t:.2f}"] = float(value)
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


def _curly_geopath_objective(
    mode: str,
    geopath_model: PathCorrection,
    x0: torch.Tensor,
    x1: torch.Tensor,
    reference_pool: CurlyReferencePool,
    times: list[float],
    targets: dict[float, torch.Tensor],
    lambdas: dict[float, torch.Tensor] | dict[float, dict[str, torch.Tensor]],
    rho: float,
    path_alpha: float,
    sigma: float,
    velocity_scale: float,
    reference_k: int,
    cosine_weight: float,
    l2_weight: float,
    l2_mu_dot_scale: float,
    moment_eta: float,
    moment_feature_blocks: tuple[str, ...] | None = None,
    moment_block_normalization: str = "none",
    moment_feature_params: object | None = None,
    pseudo_targets: dict[float, torch.Tensor] | None = None,
    pseudo_lambdas: dict[float, torch.Tensor] | None = None,
    pseudo_rho: float | None = None,
    pseudo_eta: float = 0.0,
    pseudo_posterior: Callable[[torch.Tensor], torch.Tensor] | None = None,
    time_generator: torch.Generator | None = None,
) -> tuple[
    torch.Tensor,
    dict[float, torch.Tensor] | dict[float, dict[str, torch.Tensor]] | None,
    dict[float, torch.Tensor] | None,
    dict[str, float],
]:
    if mode not in CURLY_MODES:
        raise ValueError(f"Unsupported Curly mode: {mode}")
    if moment_eta < 0.0:
        raise ValueError(f"curly.moment_eta must be non-negative, got {moment_eta}")

    t_rand = _uniform_time(x0.shape[0], x0.device, x0.dtype, generator=time_generator)
    xt, mu_dot, _ = curly_path_and_velocity(
        t=t_rand,
        x0=x0,
        x1=x1,
        geopath_net=geopath_model,
        path_alpha=float(path_alpha),
        sigma=float(sigma),
        create_graph=True,
    )
    reference_velocity = knn_reference_velocity(
        x=xt,
        reference_x=reference_pool.positions,
        reference_v=reference_pool.velocities,
        k=int(reference_k),
    )
    drift_loss, stats = curly_drift_alignment_loss(
        path_velocity=mu_dot,
        reference_velocity=reference_velocity,
        velocity_scale=float(velocity_scale),
        cosine_weight=float(cosine_weight),
        l2_weight=float(l2_weight),
        l2_mu_dot_scale=float(l2_mu_dot_scale),
    )
    total = drift_loss
    residuals: dict[float, torch.Tensor] | dict[float, dict[str, torch.Tensor]] | None = None
    pseudo_residuals: dict[float, torch.Tensor] | None = None
    pseudo_term = torch.zeros((), device=x0.device, dtype=x0.dtype)

    if mode in CURLY_AL_MODES:
        normalization = normalize_moment_block_normalization(moment_block_normalization)
        active_blocks = normalize_moment_feature_blocks(moment_feature_blocks)
        if normalization == "none":
            residuals = _constraint_residuals_for_mode(
                mode=mode,
                x0=x0,
                x1=x1,
                times=times,
                targets=targets,
                g_model=geopath_model,
                curly_path_alpha=float(path_alpha),
                moment_feature_blocks=active_blocks,
                moment_feature_params=moment_feature_params,
            )
            al_term, per_time = augmented_lagrangian_terms(
                residuals=residuals,
                lambdas=lambdas,  # type: ignore[arg-type]
                rho=float(rho),
            )
            per_block: dict[str, dict[float, float]] = {}
            raw_avg_residual_norm = _avg_flat_residual_norm(residuals)
            normalized_avg_residual_norm = raw_avg_residual_norm
        else:
            raw_residuals = _constraint_residual_blocks_for_mode(
                mode=mode,
                x0=x0,
                x1=x1,
                times=times,
                targets=targets,
                g_model=geopath_model,
                curly_path_alpha=float(path_alpha),
                moment_feature_blocks=active_blocks,
                moment_feature_params=moment_feature_params,
            )
            residuals = normalize_residual_blocks(
                residuals=raw_residuals,
                dim=int(x0.shape[1]),
                normalization=normalization,
            )
            al_term, per_time, per_block = augmented_lagrangian_block_terms(
                residuals=residuals,
                lambdas=lambdas,  # type: ignore[arg-type]
                rho=float(rho),
            )
            raw_avg_residual_norm = _avg_block_residual_norm(raw_residuals)
            normalized_avg_residual_norm = _avg_block_residual_norm(residuals)
        total = total + float(moment_eta) * al_term
        stats["moment_term"] = float(al_term.detach().item())
        stats["curly_moment_eta"] = float(moment_eta)
        stats["raw_avg_residual_norm"] = float(raw_avg_residual_norm)
        stats["normalized_avg_residual_norm"] = float(normalized_avg_residual_norm)
        for t, value in per_time.items():
            stats[f"al_t_{t:.2f}"] = float(value)
        for block, block_values in per_block.items():
            for t, value in block_values.items():
                stats[f"al_{block}_t_{t:.2f}"] = float(value)

        pseudo_active = (
            float(pseudo_eta) > 0.0
            and pseudo_targets is not None
            and pseudo_posterior is not None
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
                curly_path_alpha=float(path_alpha),
            )
            if pseudo_lambdas is None or pseudo_rho is None:
                raise ValueError(
                    "Pseudo AL term requires pseudo_lambdas and pseudo_rho in curly_constrained_al mode."
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
        stats["moment_term"] = 0.0
        stats["curly_moment_eta"] = 0.0
        stats["raw_avg_residual_norm"] = 0.0
        stats["normalized_avg_residual_norm"] = 0.0

    stats["pseudo_term"] = float(pseudo_term.detach().item())
    return total, residuals, pseudo_residuals, stats


def _cfm_loss(
    mode: str,
    v_model: VelocityField,
    g_model: PathCorrection | None,
    x0: torch.Tensor,
    x1: torch.Tensor,
    mfm_backend: MetricBackend | None = None,
    curly_path_alpha: float = 1.0,
    curly_sigma: float = 0.0,
    time_generator: torch.Generator | None = None,
    time_sampling_alpha: float = 1.0,
    velocity_input_noise_sigma: float = 0.0,
    noise_generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, float]:
    if time_sampling_alpha <= 0.0:
        raise ValueError(
            f"time_sampling_alpha must be positive, got {time_sampling_alpha}"
        )
    if velocity_input_noise_sigma < 0.0:
        raise ValueError(
            "velocity_input_noise_sigma must be non-negative, "
            f"got {velocity_input_noise_sigma}"
        )
    t_rand = _uniform_time(x0.shape[0], x0.device, x0.dtype, generator=time_generator)
    if time_sampling_alpha != 1.0:
        # If U is uniform, U**(1/alpha) follows Beta(alpha, 1). Alpha > 1
        # allocates more Stage-B regression samples near the endpoint.
        t_rand = t_rand.pow(1.0 / float(time_sampling_alpha))
    if mode in METRIC_MODES:
        if mfm_backend is None:
            raise ValueError("mfm_backend is required for metric modes.")
        t_rand, xt, u_target = mfm_backend.sample_location_and_conditional_flow(
            x0=x0,
            x1=x1,
            t=t_rand,
            create_graph=False,
        )
    elif mode in CURLY_MODES:
        xt, u_target, _ = curly_path_and_velocity(
            t=t_rand,
            x0=x0,
            x1=x1,
            geopath_net=g_model,
            path_alpha=float(curly_path_alpha),
            sigma=float(curly_sigma),
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
    if velocity_input_noise_sigma > 0.0:
        noise = torch.randn(
            xt.shape,
            device=xt.device,
            dtype=xt.dtype,
            generator=noise_generator,
        )
        xt = xt + float(velocity_input_noise_sigma) * noise
    pred = v_model(t_rand, xt)
    loss = torch.mean(torch.sum((pred - u_target) ** 2, dim=1))
    return loss, path_energy_proxy(u_target.detach())


def _stage_b_batch_for_mode(
    *,
    mode: str,
    problem: CouplingProblem,
    batch_size: int,
    coupling: str,
    generator: torch.Generator | None,
    g_model: PathCorrection | None,
    curly_reference_pool: CurlyReferencePool | None,
    curly_stage_b_coupling: str,
    curly_reference_k: int,
    curly_path_alpha: float,
    curly_sigma: float,
    curly_velocity_scale: float,
    curly_learned_coupling_num_times: int,
    curly_learned_coupling_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if mode not in CURLY_MODES or curly_stage_b_coupling == "same":
        return sample_coupled_batch(
            problem,
            batch_size=batch_size,
            coupling=coupling,
            generator=generator,
        )
    if curly_stage_b_coupling != "learned":
        raise ValueError(f"Unsupported Curly Stage-B coupling '{curly_stage_b_coupling}'.")
    if g_model is None:
        raise ValueError("Curly learned Stage-B coupling requires a trained path model.")
    if curly_reference_pool is None:
        raise ValueError("Curly learned Stage-B coupling requires a velocity reference pool.")
    x0, x1, _ = sample_coupled_batch(
        problem,
        batch_size=batch_size,
        coupling="random",
        generator=generator,
    )
    return curly_learned_coupling(
        x0=x0,
        x1=x1,
        geopath_net=g_model,
        reference_x=curly_reference_pool.positions,
        reference_v=curly_reference_pool.velocities,
        k=int(curly_reference_k),
        path_alpha=float(curly_path_alpha),
        sigma=float(curly_sigma),
        velocity_scale=float(curly_velocity_scale),
        num_times=int(curly_learned_coupling_num_times),
        chunk_size=int(curly_learned_coupling_chunk_size),
        generator=generator,
    )


def _init_lambdas(
    times: list[float],
    targets: dict[float, torch.Tensor],
) -> dict[float, torch.Tensor]:
    lambdas: dict[float, torch.Tensor] = {}
    for t in times:
        lambdas[float(t)] = torch.zeros_like(targets[float(t)])
    return lambdas


def _init_block_lambdas(
    times: list[float],
    targets: dict[float, torch.Tensor],
    dim: int,
    moment_feature_blocks: tuple[str, ...],
) -> dict[float, dict[str, torch.Tensor]]:
    lambdas: dict[float, dict[str, torch.Tensor]] = {}
    for t in times:
        target_blocks = split_moment_feature_vector(
            feature=targets[float(t)],
            dim=dim,
            feature_blocks=moment_feature_blocks,
        )
        lambdas[float(t)] = {
            block: torch.zeros_like(target) for block, target in target_blocks.items()
        }
    return lambdas


def _parse_moment_block_schedule(
    raw_schedule: object | None,
    total_steps: int,
    moment_feature_blocks: tuple[str, ...],
    moment_block_normalization: str,
) -> tuple[dict[str, Any], ...]:
    if int(total_steps) <= 0:
        return ()
    active_all = normalize_moment_feature_blocks(moment_feature_blocks)
    if raw_schedule is None:
        return ({"end_step": int(total_steps), "blocks": active_all},)
    if not isinstance(raw_schedule, (list, tuple)):
        raise ValueError(
            "train.moment_block_schedule must be a list of phase mappings when provided."
        )
    if len(raw_schedule) == 0:
        return ({"end_step": int(total_steps), "blocks": active_all},)
    if normalize_moment_block_normalization(moment_block_normalization) == "none":
        raise ValueError(
            "train.moment_block_schedule requires train.moment_block_normalization != 'none' "
            "so moment blocks have independent Lagrange multipliers."
        )

    phases: list[dict[str, Any]] = []
    prev_end = 0
    for idx, raw_phase in enumerate(raw_schedule):
        if not isinstance(raw_phase, dict):
            raise ValueError(
                "Each train.moment_block_schedule phase must be a mapping, "
                f"got {type(raw_phase)} at index {idx}."
            )
        blocks = normalize_moment_feature_blocks(raw_phase.get("blocks", active_all))
        missing = [block for block in blocks if block not in active_all]
        if missing:
            raise ValueError(
                "train.moment_block_schedule phase uses blocks absent from "
                "data.moment_feature_blocks: "
                + ", ".join(missing)
            )
        has_until_step = "until_step" in raw_phase
        has_until_fraction = "until_fraction" in raw_phase
        if has_until_step and has_until_fraction:
            raise ValueError(
                "Each train.moment_block_schedule phase may define only one of "
                "until_step or until_fraction."
            )
        if has_until_step:
            end_step = int(raw_phase["until_step"])
        elif has_until_fraction:
            until_fraction = float(raw_phase["until_fraction"])
            if until_fraction <= 0.0:
                raise ValueError(
                    "train.moment_block_schedule until_fraction must be positive, "
                    f"got {until_fraction}."
                )
            end_step = int(np.ceil(float(total_steps) * until_fraction))
        elif idx == len(raw_schedule) - 1:
            end_step = int(total_steps)
        else:
            raise ValueError(
                "Non-final train.moment_block_schedule phases require until_step "
                "or until_fraction."
            )
        end_step = min(int(end_step), int(total_steps))
        if end_step <= prev_end:
            raise ValueError(
                "train.moment_block_schedule phases must have strictly increasing end steps; "
                f"phase {idx} ends at {end_step} after previous end {prev_end}."
            )
        phases.append({"end_step": end_step, "blocks": blocks})
        prev_end = end_step
        if prev_end >= int(total_steps):
            break
    if prev_end < int(total_steps):
        phases.append({"end_step": int(total_steps), "blocks": active_all})
    return tuple(phases)


def _active_moment_blocks_for_step(
    schedule: tuple[dict[str, Any], ...],
    step: int,
) -> tuple[str, ...]:
    if not schedule:
        raise ValueError("Cannot select active moment blocks from an empty schedule.")
    for phase in schedule:
        if int(step) < int(phase["end_step"]):
            return normalize_moment_feature_blocks(phase["blocks"])
    return normalize_moment_feature_blocks(schedule[-1]["blocks"])


def _summarize_moment_block_schedule(
    schedule: tuple[dict[str, Any], ...],
    total_steps: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for phase in schedule:
        end_step = int(phase["end_step"])
        out.append(
            {
                "end_step": end_step,
                "until_fraction": (
                    None if int(total_steps) <= 0 else float(end_step) / float(total_steps)
                ),
                "blocks": [str(block) for block in normalize_moment_feature_blocks(phase["blocks"])],
            }
        )
    return out


def _targets_for_moment_blocks(
    targets: dict[float, torch.Tensor],
    dim: int,
    source_blocks: tuple[str, ...],
    selected_blocks: tuple[str, ...],
) -> dict[float, torch.Tensor]:
    source = normalize_moment_feature_blocks(source_blocks)
    selected = normalize_moment_feature_blocks(selected_blocks)
    if selected == source:
        return targets
    return {
        float(t): select_moment_feature_blocks(
            feature=target,
            dim=dim,
            source_blocks=source,
            selected_blocks=selected,
        )
        for t, target in targets.items()
    }


def _block_lambdas_for_moment_blocks(
    lambdas: dict[float, dict[str, torch.Tensor]],
    selected_blocks: tuple[str, ...],
) -> dict[float, dict[str, torch.Tensor]]:
    selected = normalize_moment_feature_blocks(selected_blocks)
    return {
        float(t): {block: block_lambdas[block] for block in selected}
        for t, block_lambdas in lambdas.items()
    }


def _merge_block_lambdas(
    base: dict[float, dict[str, torch.Tensor]],
    updated: dict[float, dict[str, torch.Tensor]],
) -> dict[float, dict[str, torch.Tensor]]:
    merged = {float(t): dict(block_lambdas) for t, block_lambdas in base.items()}
    for t, block_lambdas in updated.items():
        if float(t) not in merged:
            raise ValueError(f"Updated Lagrange multipliers contain unexpected time {float(t)}.")
        for block, value in block_lambdas.items():
            if block not in merged[float(t)]:
                raise ValueError(
                    f"Updated Lagrange multipliers contain unexpected block '{block}' "
                    f"at time {float(t)}."
                )
            merged[float(t)][block] = value
    return merged


def _lagrange_multiplier_diagnostics(
    lambdas: dict[float, torch.Tensor] | dict[float, dict[str, torch.Tensor]],
    clip_value: float,
) -> dict[str, float | int]:
    tensors: list[torch.Tensor] = []
    for value in lambdas.values():
        if isinstance(value, dict):
            tensors.extend(tensor.detach().reshape(-1) for tensor in value.values())
        else:
            tensors.append(value.detach().reshape(-1))
    if not tensors:
        return {"l2": 0.0, "linf": 0.0, "clip_fraction": 0.0, "numel": 0}
    flat = torch.cat(tensors)
    abs_flat = torch.abs(flat)
    atol = max(1e-7, abs(float(clip_value)) * 1e-6)
    saturated = abs_flat >= float(clip_value) - atol
    return {
        "l2": float(torch.linalg.vector_norm(flat).item()),
        "linf": float(abs_flat.max().item()),
        "clip_fraction": float(saturated.to(dtype=torch.float32).mean().item()),
        "numel": int(flat.numel()),
    }


def _eval_constraint_norms(
    mode: str,
    problem: CouplingProblem,
    coupling: str,
    batch_size: int,
    times: list[float],
    targets: dict[float, torch.Tensor],
    g_model: PathCorrection | None,
    mfm_alpha: float,
    curly_path_alpha: float,
    generator: torch.Generator,
    moment_feature_blocks: tuple[str, ...] | None = None,
    moment_feature_params: object | None = None,
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
        curly_path_alpha=curly_path_alpha,
        moment_feature_blocks=moment_feature_blocks,
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
    curly_path_alpha: float,
    generator: torch.Generator,
) -> dict[float, float]:
    if isinstance(problem, EmpiricalCouplingProblem) and problem.has_global_ot_support:
        if (
            problem.global_ot_src_idx is None
            or problem.global_ot_tgt_idx is None
            or problem.global_ot_mass is None
        ):
            raise ValueError("Global OT support tensors are missing.")
        src_idx = problem.global_ot_src_idx.to(
            device=problem.x0_pool.device, dtype=torch.long
        )
        tgt_idx = problem.global_ot_tgt_idx.to(
            device=problem.x1_pool.device, dtype=torch.long
        )
        weights = problem.global_ot_mass.to(
            device=problem.x0_pool.device, dtype=problem.x0_pool.dtype
        )
        weights = weights / torch.clamp(
            weights.sum(), min=torch.finfo(weights.dtype).eps
        )
        x0 = problem.x0_pool[src_idx]
        x1 = problem.x1_pool[tgt_idx]
        weighted_residuals: dict[float, torch.Tensor] = {}
        with torch.no_grad():
            for time in times:
                time_value = float(time)
                t_batch = torch.full(
                    (x0.shape[0], 1),
                    time_value,
                    device=x0.device,
                    dtype=x0.dtype,
                )
                xt = _path_samples_for_mode(
                    mode=mode,
                    x0=x0,
                    x1=x1,
                    t_batch=t_batch,
                    g_model=g_model,
                    mfm_alpha=mfm_alpha,
                    curly_path_alpha=curly_path_alpha,
                )
                probabilities = pseudo_posterior(xt)
                if probabilities.ndim != 2:
                    raise ValueError(
                        "pseudo_posterior must return shape (N, K), got "
                        f"{tuple(probabilities.shape)}"
                    )
                target = pseudo_targets[time_value].to(
                    device=probabilities.device, dtype=probabilities.dtype
                )
                local_weights = weights.to(
                    device=probabilities.device, dtype=probabilities.dtype
                )
                weighted_residuals[time_value] = torch.sum(
                    local_weights[:, None] * probabilities, dim=0
                ) - target
        return residual_norms(weighted_residuals)

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
        curly_path_alpha=curly_path_alpha,
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
    curly_reference_pool: CurlyReferencePool | None = None,
    curly_stage_b_coupling: str = "same",
    curly_reference_k: int = 20,
    curly_path_alpha: float = 1.0,
    curly_sigma: float = 0.0,
    curly_velocity_scale: float = 1.0,
    curly_learned_coupling_num_times: int = 1,
    curly_learned_coupling_chunk_size: int = 32,
    time_generator: torch.Generator | None = None,
    velocity_input_noise_sigma: float = 0.0,
    noise_generator: torch.Generator | None = None,
) -> tuple[float, float]:
    if velocity_input_noise_sigma < 0.0:
        raise ValueError(
            "velocity_input_noise_sigma must be non-negative, "
            f"got {velocity_input_noise_sigma}"
        )
    x0, x1, _ = _stage_b_batch_for_mode(
        mode=mode,
        problem=problem,
        batch_size=batch_size,
        coupling=coupling,
        generator=generator,
        g_model=g_model,
        curly_reference_pool=curly_reference_pool,
        curly_stage_b_coupling=curly_stage_b_coupling,
        curly_reference_k=int(curly_reference_k),
        curly_path_alpha=float(curly_path_alpha),
        curly_sigma=float(curly_sigma),
        curly_velocity_scale=float(curly_velocity_scale),
        curly_learned_coupling_num_times=int(curly_learned_coupling_num_times),
        curly_learned_coupling_chunk_size=int(curly_learned_coupling_chunk_size),
    )
    if mode in METRIC_MODES:
        if mfm_backend is None:
            raise ValueError("mfm_backend is required for metric modes.")
        t, xt, u_target = mfm_backend.sample_location_and_conditional_flow(
            x0=x0,
            x1=x1,
            t=_uniform_time(batch_size, x0.device, x0.dtype, generator=time_generator),
            create_graph=False,
        )
    elif mode in CURLY_MODES:
        t = _uniform_time(batch_size, x0.device, x0.dtype, generator=time_generator)
        xt, u_target, _ = curly_path_and_velocity(
            t=t,
            x0=x0,
            x1=x1,
            geopath_net=g_model,
            path_alpha=float(curly_path_alpha),
            sigma=float(curly_sigma),
            create_graph=False,
        )
    else:
        t = _uniform_time(batch_size, x0.device, x0.dtype, generator=time_generator)
        xt, u_target, _ = path_and_velocity(
            mode=mode,
            t=t,
            x0=x0,
            x1=x1,
            g_model=g_model,
            create_graph=False,
        )
    if velocity_input_noise_sigma > 0.0:
        noise = torch.randn(
            xt.shape,
            device=xt.device,
            dtype=xt.dtype,
            generator=noise_generator,
        )
        xt = xt + float(velocity_input_noise_sigma) * noise
    with torch.no_grad():
        pred = v_model(t, xt)
        loss = torch.mean(torch.sum((pred - u_target.detach()) ** 2, dim=1))
    return float(loss.item()), path_energy_proxy(u_target.detach())


def _eval_constrained_regularizer_terms(
    problem: CouplingProblem,
    coupling: str,
    g_model: PathCorrection,
    batch_size: int,
    generator: torch.Generator,
    time_generator: torch.Generator | None = None,
) -> tuple[float, float]:
    x0, x1, _ = sample_coupled_batch(
        problem,
        batch_size=batch_size,
        coupling=coupling,
        generator=generator,
    )
    t = _uniform_time(batch_size, x0.device, x0.dtype, generator=time_generator)
    with torch.enable_grad():
        _, velocity, t_req = path_and_velocity(
            mode="constrained",
            t=t,
            x0=x0,
            x1=x1,
            g_model=g_model,
            create_graph=True,
        )
        base_velocity = x1 - x0
        velocity_deviation = torch.mean(
            torch.sum((velocity - base_velocity) ** 2, dim=1)
        )
        acceleration = vector_time_derivative(
            velocity,
            t_req,
            create_graph=False,
        )
        temporal_smoothness = torch.mean(torch.sum(acceleration**2, dim=1))
    return float(velocity_deviation.detach().item()), float(temporal_smoothness.detach().item())


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


def _lookup_time_tensor(
    values_by_time: dict[float, torch.Tensor],
    t: float,
    tol: float = 1e-8,
) -> torch.Tensor:
    if float(t) in values_by_time:
        return values_by_time[float(t)]
    for key, value in values_by_time.items():
        if abs(float(key) - float(t)) <= tol:
            return value
    raise KeyError(
        f"Missing tensor for time {float(t):.6f}. "
        f"Available times={sorted(float(k) for k in values_by_time.keys())}."
    )


def _maybe_subsample_reference_pool(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    max_samples: int | None,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_samples is None or int(max_samples) <= 0 or positions.shape[0] <= int(max_samples):
        return positions, velocities
    idx = torch.randint(
        low=0,
        high=positions.shape[0],
        size=(int(max_samples),),
        device=positions.device,
        generator=generator,
    )
    return positions[idx], velocities[idx]


def _build_curly_reference_pool(
    *,
    target_samples_by_time: dict[float, torch.Tensor] | None,
    reference_velocity_by_time: dict[float, torch.Tensor] | None,
    policy: str,
    max_samples_per_time: int | None,
    generator: torch.Generator | None,
) -> CurlyReferencePool:
    if target_samples_by_time is None or reference_velocity_by_time is None:
        raise ValueError(
            "Curly modes require single-cell position and RNA-velocity pools. "
            "Provide data.single_cell velocity arrays such as pcs_delta for EB."
        )
    normalized_policy = normalize_reference_pool_policy(policy)
    selected_times = select_reference_times(
        policy=normalized_policy,
        available_times=target_samples_by_time.keys(),
    )
    positions: list[torch.Tensor] = []
    velocities: list[torch.Tensor] = []
    for t in selected_times:
        x_t = _lookup_time_tensor(target_samples_by_time, float(t))
        v_t = _lookup_time_tensor(reference_velocity_by_time, float(t))
        if x_t.shape != v_t.shape:
            raise ValueError(
                f"Curly reference position/velocity shape mismatch at t={float(t):.6f}: "
                f"{tuple(x_t.shape)} vs {tuple(v_t.shape)}."
            )
        x_sel, v_sel = _maybe_subsample_reference_pool(
            positions=x_t,
            velocities=v_t,
            max_samples=max_samples_per_time,
            generator=generator,
        )
        positions.append(x_sel)
        velocities.append(v_sel)
    return CurlyReferencePool(
        positions=torch.cat(positions, dim=0),
        velocities=torch.cat(velocities, dim=0),
        times=tuple(float(t) for t in selected_times),
        policy=normalized_policy,
    )


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


def _full_pool_target_sampler(
    target_samples_by_time: dict[float, torch.Tensor],
) -> Callable[[float, int, torch.Generator | None], torch.Tensor]:
    def _sampler(
        t: float,
        n_samples: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        del generator
        pool = _lookup_target_pool_by_time(target_samples_by_time=target_samples_by_time, t=float(t))
        if int(pool.shape[0]) != int(n_samples):
            raise ValueError(
                "Full-pool target sampler requires an exact pool-size match, "
                f"got n_samples={int(n_samples)} but pool has {int(pool.shape[0])} samples for t={float(t):.2f}."
            )
        return pool

    return _sampler


def _global_ot_support_pairs(problem: EmpiricalCouplingProblem) -> tuple[torch.Tensor, torch.Tensor]:
    if not problem.has_global_ot_support:
        raise ValueError(
            "Full-pool interpolant evaluation requires coupling='ot_global' with cached global OT support."
        )
    if problem.global_ot_src_idx is None or problem.global_ot_tgt_idx is None:
        raise ValueError("Global OT support indices are missing.")
    src_idx = problem.global_ot_src_idx.to(device=problem.x0_pool.device, dtype=torch.long)
    tgt_idx = problem.global_ot_tgt_idx.to(device=problem.x1_pool.device, dtype=torch.long)
    if src_idx.ndim != 1 or tgt_idx.ndim != 1:
        raise ValueError("Global OT support indices must be 1D tensors.")
    if src_idx.shape[0] != tgt_idx.shape[0]:
        raise ValueError(
            f"Global OT support index length mismatch: {tuple(src_idx.shape)} vs {tuple(tgt_idx.shape)}."
        )
    x0_support = problem.x0_pool[src_idx]
    x1_support = problem.x1_pool[tgt_idx]
    if x0_support.shape != x1_support.shape:
        raise ValueError(
            "Global OT support endpoint shapes mismatch after indexing, "
            f"got {tuple(x0_support.shape)} vs {tuple(x1_support.shape)}."
        )
    return x0_support, x1_support


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


def _eval_empirical_rollout_metrics_full_pool(
    problem: CouplingProblem,
    v_model: VelocityField,
    times: list[float],
    n_steps: int,
    target_samples_by_time: dict[float, torch.Tensor],
    holdout_time: float | None = None,
) -> tuple[dict[str, float | dict[str, float]], dict[str, Any]]:
    if not isinstance(problem, EmpiricalCouplingProblem):
        raise ValueError("Full-pool empirical rollout metrics require an EmpiricalCouplingProblem.")
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
    empirical_w2_by_time: dict[str, float] = {}
    empirical_w1_by_time: dict[str, float] = {}
    for t in eval_times:
        t_value = float(t)
        generated = generated_by_time[t_value]
        target = _lookup_target_pool_by_time(target_samples_by_time=target_samples_by_time, t=t_value)
        if generated.shape != target.shape:
            raise ValueError(
                "Full-pool empirical rollout evaluation requires equal-size generated/target pools. "
                f"t={t_value:.2f}, generated={tuple(generated.shape)}, target={tuple(target.shape)}."
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
    reference_velocity_by_time: dict[float, torch.Tensor] | None = None,
    data_family: str = "gaussian",
    evaluate: bool = True,
) -> dict[str, Any]:
    train_experiment_start = time.perf_counter()
    device = torch.device(cfg["device"])
    dtype = torch.float32
    mode = str(cfg["experiment"]["mode"])
    if mode not in {"baseline", "constrained", *METRIC_MODES, *CURLY_MODES}:
        raise ValueError(
            f"Unsupported experiment mode '{mode}'. "
            "Expected one of: baseline, constrained, metric, metric_alpha0, "
            "metric_constrained_al, metric_constrained_soft, curly, curly_constrained_al."
        )

    train_cfg = cfg["train"]
    base_seed = int(cfg["seed"])
    init_seed_raw = train_cfg.get("init_seed", None)
    batch_seed_raw = train_cfg.get("batch_seed", None)
    time_seed_raw = train_cfg.get("time_seed", None)
    init_seed = base_seed if init_seed_raw is None else int(init_seed_raw)
    batch_seed = base_seed if batch_seed_raw is None else int(batch_seed_raw)

    velocity_input_noise_sigma = float(
        train_cfg.get("velocity_input_noise_sigma", 0.0)
    )
    if velocity_input_noise_sigma < 0.0:
        raise ValueError(
            "train.velocity_input_noise_sigma must be non-negative, "
            f"got {velocity_input_noise_sigma}"
        )
    velocity_time_sampling_alpha = float(
        train_cfg.get("velocity_time_sampling_alpha", 1.0)
    )
    if velocity_time_sampling_alpha <= 0.0:
        raise ValueError(
            "train.velocity_time_sampling_alpha must be positive, "
            f"got {velocity_time_sampling_alpha}"
        )
    velocity_input_noise_seed_raw = train_cfg.get("velocity_input_noise_seed", None)
    velocity_input_noise_seed = (
        base_seed + 900_007
        if velocity_input_noise_seed_raw is None
        else int(velocity_input_noise_seed_raw)
    )

    set_seed(init_seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(batch_seed)
    time_generator: torch.Generator | None = None
    if time_seed_raw is not None:
        time_generator = torch.Generator(device=device)
        time_generator.manual_seed(int(time_seed_raw))
    velocity_input_noise_generator = torch.Generator(device=device)
    velocity_input_noise_generator.manual_seed(velocity_input_noise_seed)

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
    if mode in METRIC_MODES and velocity_input_noise_sigma > 0.0 and mfm_sigma > 0.0:
        raise ValueError(
            "Generic train.velocity_input_noise_sigma and native mfm.sigma cannot both "
            "be positive for metric modes; set mfm.sigma=0 to avoid double noise."
        )
    mfm_requested_backend = str(mfm_cfg.get("backend", "auto"))
    mfm_geopath_metric = str(mfm_cfg.get("geopath_metric", "land")).strip().lower()
    if mfm_geopath_metric not in {"land", "rbf"}:
        raise ValueError(
            f"Unsupported mfm.geopath_metric '{mfm_geopath_metric}'. "
            "Expected one of: land, rbf."
        )
    mfm_land_gamma = float(mfm_cfg.get("land_gamma", 0.125))
    mfm_land_rho = float(mfm_cfg.get("land_rho", 1e-3))
    mfm_land_samples = int(mfm_cfg.get("land_metric_samples", 256))
    mfm_reference_pool_policy = str(mfm_cfg.get("reference_pool_policy", "endpoints_only"))
    mfm_moment_eta = float(mfm_cfg.get("moment_eta", 1.0))
    mfm_rbf_n_centers = int(mfm_cfg.get("rbf_n_centers", 100))
    mfm_rbf_kappa = float(mfm_cfg.get("rbf_kappa", 1.0))
    mfm_rbf_epsilon = float(mfm_cfg.get("rbf_epsilon", mfm_land_rho))
    mfm_rbf_alpha_metric = float(mfm_cfg.get("rbf_alpha_metric", 1.0))
    mfm_rbf_metric_epochs = int(mfm_cfg.get("rbf_metric_epochs", 50))
    mfm_rbf_lr = float(mfm_cfg.get("rbf_lr", 1.0e-2))
    mfm_rbf_seed = int(mfm_cfg.get("rbf_seed", init_seed + 991))

    curly_cfg = cfg.get("curly", {})
    curly_path_alpha = float(curly_cfg.get("path_alpha", 1.0))
    curly_sigma = float(curly_cfg.get("sigma", 0.0))
    curly_velocity_scale = float(curly_cfg.get("velocity_scale", 1.0))
    curly_reference_pool_policy = str(curly_cfg.get("reference_pool_policy", "endpoints_only"))
    curly_reference_k = int(curly_cfg.get("reference_k", 20))
    curly_reference_pool_max_raw = curly_cfg.get("reference_pool_max_samples_per_time", None)
    curly_reference_pool_max_samples = (
        None if curly_reference_pool_max_raw is None else int(curly_reference_pool_max_raw)
    )
    curly_cosine_weight = float(curly_cfg.get("cosine_weight", 1.0))
    curly_l2_weight = float(curly_cfg.get("l2_weight", 1.0))
    curly_l2_mu_dot_scale = float(curly_cfg.get("l2_mu_dot_scale", 1.0))
    curly_moment_eta = float(curly_cfg.get("moment_eta", 1.0))
    curly_stage_b_coupling = normalize_stage_b_coupling(
        str(curly_cfg.get("stage_b_coupling", "same"))
    )
    curly_learned_coupling_num_times = int(curly_cfg.get("learned_coupling_num_times", 1))
    curly_learned_coupling_chunk_size = int(curly_cfg.get("learned_coupling_chunk_size", 32))

    g_model: PathCorrection | None = None
    if (
        mode == "constrained"
        or (mode in METRIC_MODES and mfm_alpha != 0.0)
        or mode in CURLY_MODES
    ):
        g_model = PathCorrection(
            state_dim=state_dim,
            hidden_dims=model_cfg["path_hidden_dims"],
            activation=model_cfg["activation"],
            zero_init_output=bool(model_cfg.get("path_zero_init_output", False)),
        ).to(device=device, dtype=dtype)

    times = [float(t) for t in cfg["data"]["constraint_times"]]
    moment_feature_blocks = normalize_moment_feature_blocks(
        cfg.get("data", {}).get("moment_feature_blocks", None)
    )
    moment_feature_params = cfg.get("data", {}).get("moment_feature_params", None)
    moment_block_normalization = normalize_moment_block_normalization(
        train_cfg.get("moment_block_normalization", "none")
    )
    block_moment_constraints = moment_block_normalization != "none"
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
    raw_moment_block_schedule = train_cfg.get("moment_block_schedule", None)
    moment_block_schedule = _parse_moment_block_schedule(
        raw_schedule=raw_moment_block_schedule,
        total_steps=stage_steps["stage_a_steps"],
        moment_feature_blocks=moment_feature_blocks,
        moment_block_normalization=moment_block_normalization,
    )
    moment_target_cache: dict[tuple[str, ...], dict[float, torch.Tensor]] = {}

    def targets_for_active_blocks(active_blocks: tuple[str, ...]) -> dict[float, torch.Tensor]:
        normalized_blocks = normalize_moment_feature_blocks(active_blocks)
        if normalized_blocks not in moment_target_cache:
            moment_target_cache[normalized_blocks] = _targets_for_moment_blocks(
                targets=targets,
                dim=state_dim,
                source_blocks=moment_feature_blocks,
                selected_blocks=normalized_blocks,
            )
        return moment_target_cache[normalized_blocks]

    eval_empirical_w2_full_pool = bool(train_cfg.get("eval_empirical_w2_full_pool", False))
    eval_full_ot_metrics = bool(train_cfg.get("eval_full_ot_metrics", False))
    eval_full_ot_method = str(train_cfg.get("eval_full_ot_method", "pot_emd2")).strip().lower()
    if eval_full_ot_method not in {"exact_lp", "pot_emd2"}:
        raise ValueError(
            f"Unsupported train.eval_full_ot_method '{eval_full_ot_method}'. "
            "Expected one of: exact_lp, pot_emd2."
        )
    stage_b_early_stopping_enabled = bool(
        train_cfg.get("stage_b_early_stopping_enabled", False)
    )
    stage_b_early_stopping_check_every = max(
        1, int(train_cfg.get("stage_b_early_stopping_check_every", 50))
    )
    stage_b_early_stopping_warmup_steps = max(
        0, int(train_cfg.get("stage_b_early_stopping_warmup_steps", 0))
    )
    stage_b_early_stopping_patience = max(
        1, int(train_cfg.get("stage_b_early_stopping_patience", 5))
    )
    stage_b_early_stopping_min_delta = max(
        0.0, float(train_cfg.get("stage_b_early_stopping_min_delta", 0.0))
    )
    stage_b_early_stopping_min_delta_rel = max(
        0.0, float(train_cfg.get("stage_b_early_stopping_min_delta_rel", 0.0))
    )
    stage_b_early_stopping_eval_batch_size = max(
        1,
        int(train_cfg.get("stage_b_early_stopping_eval_batch_size", eval_batch_size)),
    )
    stage_b_early_stopping_eval_batches = max(
        1, int(train_cfg.get("stage_b_early_stopping_eval_batches", 1))
    )
    stage_b_early_stopping_restore_best = bool(
        train_cfg.get("stage_b_early_stopping_restore_best", True)
    )
    stage_b_early_stopping_seed_raw = train_cfg.get("stage_b_early_stopping_seed", None)
    stage_b_early_stopping_seed = (
        batch_seed + 1729
        if stage_b_early_stopping_seed_raw is None
        else int(stage_b_early_stopping_seed_raw)
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
    train_moment_eta = float(train_cfg.get("moment_eta", 1.0))
    if train_moment_eta < 0.0:
        raise ValueError(f"train.moment_eta must be non-negative, got {train_moment_eta}")
    metric_constraint_warmup_steps = int(train_cfg.get("metric_constraint_warmup_steps", 0))
    if metric_constraint_warmup_steps < 0:
        raise ValueError(
            "train.metric_constraint_warmup_steps must be non-negative, "
            f"got {metric_constraint_warmup_steps}"
        )
    if metric_constraint_warmup_steps > stage_steps["stage_a_steps"]:
        raise ValueError(
            "train.metric_constraint_warmup_steps cannot exceed train.stage_a_steps, "
            f"got {metric_constraint_warmup_steps} > {stage_steps['stage_a_steps']}"
        )
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

    if stage_a_only and mode not in {"baseline", "constrained", *METRIC_MODES, *CURLY_MODES}:
        raise ValueError(
            "Stage-A-only profile requires one of: baseline, constrained, metric, metric_alpha0, "
            "metric_constrained_al, metric_constrained_soft, curly, curly_constrained_al."
        )
    if mode in METRIC_MODES and stage_steps["stage_c_steps"] > 0:
        raise ValueError("Metric modes currently require stage_c_steps=0.")
    if mode in CURLY_MODES and stage_steps["stage_c_steps"] > 0:
        raise ValueError("Curly modes currently require stage_c_steps=0.")
    if mfm_moment_eta < 0.0:
        raise ValueError(f"mfm.moment_eta must be non-negative, got {mfm_moment_eta}")
    if curly_moment_eta < 0.0:
        raise ValueError(f"curly.moment_eta must be non-negative, got {curly_moment_eta}")
    if curly_reference_k <= 0:
        raise ValueError(f"curly.reference_k must be positive, got {curly_reference_k}")
    if curly_learned_coupling_num_times <= 0:
        raise ValueError(
            "curly.learned_coupling_num_times must be positive, "
            f"got {curly_learned_coupling_num_times}"
        )
    if curly_learned_coupling_chunk_size <= 0:
        raise ValueError(
            "curly.learned_coupling_chunk_size must be positive, "
            f"got {curly_learned_coupling_chunk_size}"
        )
    if evaluate and data_family in {"bridge_sde", "single_cell"} and target_sampler is None:
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
    rbf_metric: RBFMetric | None = None
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
            if mfm_geopath_metric == "rbf":
                rbf_metric = fit_rbf_metric(
                    samples=metric_reference_pool,
                    n_centers=mfm_rbf_n_centers,
                    kappa=mfm_rbf_kappa,
                    epsilon=mfm_rbf_epsilon,
                    alpha=mfm_rbf_alpha_metric,
                    epochs=mfm_rbf_metric_epochs,
                    lr=mfm_rbf_lr,
                    seed=mfm_rbf_seed,
                )

    curly_reference_pool: CurlyReferencePool | None = None
    if mode in CURLY_MODES:
        curly_reference_pool = _build_curly_reference_pool(
            target_samples_by_time=target_samples_by_time,
            reference_velocity_by_time=reference_velocity_by_time,
            policy=curly_reference_pool_policy,
            max_samples_per_time=curly_reference_pool_max_samples,
            generator=generator,
        )

    history: list[dict[str, float | str | int]] = []
    global_step = 0
    lambdas: dict[float, torch.Tensor] | dict[float, dict[str, torch.Tensor]] = {}
    pseudo_lambdas: dict[float, torch.Tensor] = {}
    constrained_beta_schedule: dict[str, Any] | None = None
    pseudo_constraints_active = bool(
        pseudo_eta > 0.0
        and pseudo_targets is not None
        and pseudo_posterior is not None
        and mode in {"constrained", *METRIC_CONSTRAINED_MODES, *CURLY_CONSTRAINED_MODES}
    )
    pseudo_diagnostics_active = bool(
        pseudo_targets is not None and pseudo_posterior is not None
    )

    if mode in {"constrained", *METRIC_AL_MODES, *CURLY_AL_MODES}:
        if block_moment_constraints:
            lambdas = _init_block_lambdas(
                times=times,
                targets=targets,
                dim=state_dim,
                moment_feature_blocks=moment_feature_blocks,
            )
        else:
            lambdas = _init_lambdas(times=times, targets=targets)
    if pseudo_constraints_active and mode in {"constrained", *METRIC_AL_MODES, *CURLY_AL_MODES}:
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
            moment_feature_blocks=moment_feature_blocks,
            moment_feature_params=moment_feature_params,
        )

    stage_wall_seconds: dict[str, float] = {}
    lambda_linf_max = 0.0
    lambda_clip_fraction_max = 0.0
    lambda_saturated_steps = 0
    lambda_update_steps = 0

    stage_a_start = time.perf_counter()
    if mode == "constrained" and g_model is not None:
        optimizer_g = torch.optim.Adam(g_model.parameters(), lr=float(train_cfg["lr_g"]))
        for step in range(int(train_cfg["stage_a_steps"])):
            active_moment_blocks = _active_moment_blocks_for_step(
                schedule=moment_block_schedule,
                step=step,
            )
            active_targets = targets_for_active_blocks(active_moment_blocks)
            x0, x1, _ = sample_coupled_batch(
                problem,
                batch_size=batch_size,
                coupling=coupling,
                generator=generator,
            )
            optimizer_g.zero_grad(set_to_none=True)
            active_lambdas = (
                _block_lambdas_for_moment_blocks(
                    lambdas=lambdas,  # type: ignore[arg-type]
                    selected_blocks=active_moment_blocks,
                )
                if block_moment_constraints
                else lambdas
            )
            loss_g, residuals, pseudo_residuals, stats = _constrained_objective(
                g_model=g_model,
                x0=x0,
                x1=x1,
                times=times,
                targets=active_targets,
                lambdas=active_lambdas,
                rho=float(train_cfg["rho"]),
                alpha=float(train_cfg["alpha"]),
                beta=float(train_cfg["beta"]),
                moment_eta=float(train_moment_eta),
                beta_schedule=constrained_beta_schedule,
                moment_feature_blocks=active_moment_blocks,
                moment_block_normalization=moment_block_normalization,
                moment_feature_params=moment_feature_params,
                pseudo_targets=pseudo_targets,
                pseudo_lambdas=(pseudo_lambdas if pseudo_lambdas else None),
                pseudo_rho=float(pseudo_rho),
                pseudo_eta=float(pseudo_eta),
                pseudo_posterior=pseudo_posterior,
                time_generator=time_generator,
            )
            loss_g.backward()
            optimizer_g.step()
            if block_moment_constraints:
                updated_active_lambdas = update_lagrange_multiplier_blocks(
                    lambdas=active_lambdas,  # type: ignore[arg-type]
                    residuals=residuals,  # type: ignore[arg-type]
                    rho=float(train_cfg["rho"]),
                    clip_value=float(train_cfg["lambda_clip"]),
                )
                lambdas = _merge_block_lambdas(
                    base=lambdas,  # type: ignore[arg-type]
                    updated=updated_active_lambdas,
                )
            else:
                lambdas = update_lagrange_multipliers(
                    lambdas=lambdas,  # type: ignore[arg-type]
                    residuals=residuals,  # type: ignore[arg-type]
                    rho=float(train_cfg["rho"]),
                    clip_value=float(train_cfg["lambda_clip"]),
                )
            lambda_stats = _lagrange_multiplier_diagnostics(
                lambdas=lambdas,
                clip_value=float(train_cfg["lambda_clip"]),
            )
            lambda_update_steps += 1
            lambda_linf_max = max(lambda_linf_max, float(lambda_stats["linf"]))
            lambda_clip_fraction_max = max(
                lambda_clip_fraction_max,
                float(lambda_stats["clip_fraction"]),
            )
            if float(lambda_stats["clip_fraction"]) > 0.0:
                lambda_saturated_steps += 1
            if pseudo_constraints_active and pseudo_residuals is not None and pseudo_lambdas:
                pseudo_lambdas = update_lagrange_multipliers(
                    lambdas=pseudo_lambdas,
                    residuals=pseudo_residuals,
                    rho=float(pseudo_rho),
                    clip_value=float(pseudo_lambda_clip),
                )
                pseudo_lambda_stats = _lagrange_multiplier_diagnostics(
                    lambdas=pseudo_lambdas,
                    clip_value=float(pseudo_lambda_clip),
                )
                pseudo_avg_residual_norm = float(
                    np.mean(list(residual_norms(pseudo_residuals).values()))
                )
            else:
                pseudo_lambda_stats = None
                pseudo_avg_residual_norm = None
            history.append(
                {
                    "stage": "stage_a",
                    "step": step,
                    "global_step": global_step,
                    "loss": float(loss_g.detach().item()),
                    "avg_residual_norm": float(stats["normalized_avg_residual_norm"]),
                    "raw_avg_residual_norm": float(stats["raw_avg_residual_norm"]),
                    "normalized_avg_residual_norm": float(stats["normalized_avg_residual_norm"]),
                    "pseudo_avg_residual_norm": pseudo_avg_residual_norm,
                    "regularizer": stats["regularizer"],
                    "energy_term": stats["energy_term"],
                    "smoothness_term": stats["smoothness_term"],
                    "weighted_smoothness_term": stats["weighted_smoothness_term"],
                    "lambda_l2": float(lambda_stats["l2"]),
                    "lambda_linf": float(lambda_stats["linf"]),
                    "lambda_clip_fraction": float(lambda_stats["clip_fraction"]),
                    "pseudo_lambda_l2": (
                        None
                        if pseudo_lambda_stats is None
                        else float(pseudo_lambda_stats["l2"])
                    ),
                    "pseudo_lambda_linf": (
                        None
                        if pseudo_lambda_stats is None
                        else float(pseudo_lambda_stats["linf"])
                    ),
                    "pseudo_lambda_clip_fraction": (
                        None
                        if pseudo_lambda_stats is None
                        else float(pseudo_lambda_stats["clip_fraction"])
                    ),
                    "pseudo_term": stats["pseudo_term"],
                    "moment_blocks": ",".join(active_moment_blocks),
                }
            )
            global_step += 1
    elif mode in METRIC_MODES and g_model is not None and mfm_alpha != 0.0:
        if metric_reference_pool is None:
            raise ValueError("metric_reference_pool must be initialized for metric mode.")
        optimizer_g = torch.optim.Adam(g_model.parameters(), lr=float(train_cfg["lr_g"]))
        for step in range(int(train_cfg["stage_a_steps"])):
            active_moment_blocks = moment_feature_blocks
            active_targets = targets
            active_lambdas = lambdas
            if mode in METRIC_CONSTRAINED_MODES:
                active_moment_blocks = _active_moment_blocks_for_step(
                    schedule=moment_block_schedule,
                    step=step,
                )
                active_targets = targets_for_active_blocks(active_moment_blocks)
                active_lambdas = (
                    _block_lambdas_for_moment_blocks(
                        lambdas=lambdas,  # type: ignore[arg-type]
                        selected_blocks=active_moment_blocks,
                    )
                    if block_moment_constraints
                    else lambdas
                )
            x0, x1, _ = sample_coupled_batch(
                problem,
                batch_size=batch_size,
                coupling=coupling,
                generator=generator,
            )
            optimizer_g.zero_grad(set_to_none=True)
            pseudo_residuals: dict[float, torch.Tensor] | None = None
            pseudo_lambda_stats: dict[str, float | int] | None = None
            metric_constraints_active_step = bool(
                mode in METRIC_CONSTRAINED_MODES
                and step >= int(metric_constraint_warmup_steps)
            )
            if metric_constraints_active_step:
                loss_g, residuals, pseudo_residuals, stats = _metric_constrained_geopath_objective(
                    mode=mode,
                    geopath_model=g_model,
                    alpha_mfm=float(mfm_alpha),
                    x0=x0,
                    x1=x1,
                    manifold_samples=metric_reference_pool,
                    times=times,
                    targets=active_targets,
                    lambdas=active_lambdas,
                    rho=float(train_cfg["rho"]),
                    geopath_metric=mfm_geopath_metric,
                    land_gamma=mfm_land_gamma,
                    land_rho=mfm_land_rho,
                    rbf_metric=rbf_metric,
                    moment_eta=float(mfm_moment_eta),
                    moment_feature_blocks=active_moment_blocks,
                    moment_block_normalization=moment_block_normalization,
                    moment_feature_params=moment_feature_params,
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
                    geopath_metric=mfm_geopath_metric,
                    land_gamma=mfm_land_gamma,
                    land_rho=mfm_land_rho,
                    rbf_metric=rbf_metric,
                )
            loss_g.backward()
            optimizer_g.step()
            if mode in METRIC_AL_MODES and metric_constraints_active_step:
                if block_moment_constraints:
                    updated_active_lambdas = update_lagrange_multiplier_blocks(
                        lambdas=active_lambdas,  # type: ignore[arg-type]
                        residuals=residuals,  # type: ignore[arg-type]
                        rho=float(train_cfg["rho"]),
                        clip_value=float(train_cfg["lambda_clip"]),
                    )
                    lambdas = _merge_block_lambdas(
                        base=lambdas,  # type: ignore[arg-type]
                        updated=updated_active_lambdas,
                    )
                else:
                    lambdas = update_lagrange_multipliers(
                        lambdas=lambdas,  # type: ignore[arg-type]
                        residuals=residuals,  # type: ignore[arg-type]
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
                    pseudo_lambda_stats = _lagrange_multiplier_diagnostics(
                        lambdas=pseudo_lambdas,
                        clip_value=float(pseudo_lambda_clip),
                    )
            history_row: dict[str, float | int | str] = {
                "stage": "stage_a",
                "step": step,
                "global_step": global_step,
                "loss": float(loss_g.detach().item()),
                "land_loss": stats["land_loss"],
                "geopath_metric": mfm_geopath_metric,
                "geopath_metric_loss": stats["geopath_metric_loss"],
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
                history_row["metric_constraint_warmup_steps"] = int(
                    metric_constraint_warmup_steps
                )
                history_row["metric_constraints_active"] = int(
                    metric_constraints_active_step
                )
            if mode in METRIC_CONSTRAINED_MODES and metric_constraints_active_step:
                history_row["avg_residual_norm"] = float(stats["normalized_avg_residual_norm"])
                history_row["raw_avg_residual_norm"] = float(stats["raw_avg_residual_norm"])
                history_row["normalized_avg_residual_norm"] = float(
                    stats["normalized_avg_residual_norm"]
                )
                history_row["moment_blocks"] = ",".join(active_moment_blocks)
                if pseudo_residuals is not None:
                    history_row["pseudo_avg_residual_norm"] = float(
                        np.mean(list(residual_norms(pseudo_residuals).values()))
                    )
                if pseudo_lambda_stats is not None:
                    history_row["pseudo_lambda_l2"] = float(
                        pseudo_lambda_stats["l2"]
                    )
                    history_row["pseudo_lambda_linf"] = float(
                        pseudo_lambda_stats["linf"]
                    )
                    history_row["pseudo_lambda_clip_fraction"] = float(
                        pseudo_lambda_stats["clip_fraction"]
                    )
            history.append(history_row)
            global_step += 1
    elif mode in CURLY_MODES and g_model is not None:
        if curly_reference_pool is None:
            raise ValueError("curly_reference_pool must be initialized for Curly modes.")
        optimizer_g = torch.optim.Adam(g_model.parameters(), lr=float(train_cfg["lr_g"]))
        for step in range(int(train_cfg["stage_a_steps"])):
            active_moment_blocks = moment_feature_blocks
            active_targets = targets
            active_lambdas = lambdas
            if mode in CURLY_CONSTRAINED_MODES:
                active_moment_blocks = _active_moment_blocks_for_step(
                    schedule=moment_block_schedule,
                    step=step,
                )
                active_targets = targets_for_active_blocks(active_moment_blocks)
                active_lambdas = (
                    _block_lambdas_for_moment_blocks(
                        lambdas=lambdas,  # type: ignore[arg-type]
                        selected_blocks=active_moment_blocks,
                    )
                    if block_moment_constraints
                    else lambdas
                )
            x0, x1, _ = sample_coupled_batch(
                problem,
                batch_size=batch_size,
                coupling=coupling,
                generator=generator,
            )
            optimizer_g.zero_grad(set_to_none=True)
            loss_g, residuals, pseudo_residuals, stats = _curly_geopath_objective(
                mode=mode,
                geopath_model=g_model,
                x0=x0,
                x1=x1,
                reference_pool=curly_reference_pool,
                times=times,
                targets=active_targets,
                lambdas=active_lambdas,
                rho=float(train_cfg["rho"]),
                path_alpha=float(curly_path_alpha),
                sigma=float(curly_sigma),
                velocity_scale=float(curly_velocity_scale),
                reference_k=int(curly_reference_k),
                cosine_weight=float(curly_cosine_weight),
                l2_weight=float(curly_l2_weight),
                l2_mu_dot_scale=float(curly_l2_mu_dot_scale),
                moment_eta=float(curly_moment_eta),
                moment_feature_blocks=active_moment_blocks,
                moment_block_normalization=moment_block_normalization,
                moment_feature_params=moment_feature_params,
                pseudo_targets=pseudo_targets,
                pseudo_lambdas=(pseudo_lambdas if pseudo_lambdas else None),
                pseudo_rho=float(pseudo_rho),
                pseudo_eta=float(pseudo_eta),
                pseudo_posterior=pseudo_posterior,
                time_generator=time_generator,
            )
            loss_g.backward()
            optimizer_g.step()
            if mode in CURLY_AL_MODES:
                if residuals is None:
                    raise RuntimeError("Curly AL mode did not return moment residuals.")
                if block_moment_constraints:
                    updated_active_lambdas = update_lagrange_multiplier_blocks(
                        lambdas=active_lambdas,  # type: ignore[arg-type]
                        residuals=residuals,  # type: ignore[arg-type]
                        rho=float(train_cfg["rho"]),
                        clip_value=float(train_cfg["lambda_clip"]),
                    )
                    lambdas = _merge_block_lambdas(
                        base=lambdas,  # type: ignore[arg-type]
                        updated=updated_active_lambdas,
                    )
                else:
                    lambdas = update_lagrange_multipliers(
                        lambdas=lambdas,  # type: ignore[arg-type]
                        residuals=residuals,  # type: ignore[arg-type]
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
                "curly_loss": float(stats["curly_loss"]),
                "curly_cosine_loss": float(stats["curly_cosine_loss"]),
                "curly_l2_loss": float(stats["curly_l2_loss"]),
                "curly_reference_speed": float(stats["curly_reference_speed"]),
                "curly_path_speed": float(stats["curly_path_speed"]),
            }
            if "moment_term" in stats:
                history_row["moment_term"] = float(stats["moment_term"])
            if "pseudo_term" in stats:
                history_row["pseudo_term"] = float(stats["pseudo_term"])
            if mode in CURLY_CONSTRAINED_MODES:
                history_row["avg_residual_norm"] = float(stats["normalized_avg_residual_norm"])
                history_row["raw_avg_residual_norm"] = float(stats["raw_avg_residual_norm"])
                history_row["normalized_avg_residual_norm"] = float(
                    stats["normalized_avg_residual_norm"]
                )
                history_row["moment_blocks"] = ",".join(active_moment_blocks)
                if pseudo_residuals is not None:
                    history_row["pseudo_avg_residual_norm"] = float(
                        np.mean(list(residual_norms(pseudo_residuals).values()))
                    )
            history.append(history_row)
            global_step += 1
    stage_wall_seconds["stage_a_wall_sec"] = time.perf_counter() - stage_a_start

    stage_b_start = time.perf_counter()
    optimizer_v = torch.optim.Adam(v_model.parameters(), lr=float(train_cfg["lr_v"]))
    if mode in {"constrained", *METRIC_MODES, *CURLY_MODES} and g_model is not None:
        for param in g_model.parameters():
            param.requires_grad_(False)

    stage_b_executed_steps = 0
    stage_b_early_stop_triggered = False
    stage_b_early_stop_step: int | None = None
    stage_b_early_stop_best_loss: float | None = None
    stage_b_early_stop_last_loss: float | None = None
    stage_b_early_stop_bad_checks = 0
    stage_b_early_stop_checks = 0
    stage_b_early_stop_best_step: int | None = None
    stage_b_early_stop_restored_best = False
    stage_b_early_stop_best_state_dict: dict[str, torch.Tensor] | None = None
    early_stop_generator: torch.Generator | None = None
    early_stop_time_generator: torch.Generator | None = None
    early_stop_noise_generator: torch.Generator | None = None
    if stage_b_early_stopping_enabled:
        early_stop_generator = torch.Generator(device=device)
        early_stop_generator.manual_seed(stage_b_early_stopping_seed)
        early_stop_time_generator = torch.Generator(device=device)
        early_stop_time_generator.manual_seed(stage_b_early_stopping_seed + 1)
        early_stop_noise_generator = torch.Generator(device=device)
        early_stop_noise_generator.manual_seed(stage_b_early_stopping_seed + 2)

    for step in range(int(train_cfg["stage_b_steps"])):
        x0, x1, _ = _stage_b_batch_for_mode(
            mode=mode,
            problem=problem,
            batch_size=batch_size,
            coupling=coupling,
            generator=generator,
            g_model=g_model,
            curly_reference_pool=curly_reference_pool,
            curly_stage_b_coupling=curly_stage_b_coupling,
            curly_reference_k=int(curly_reference_k),
            curly_path_alpha=float(curly_path_alpha),
            curly_sigma=float(curly_sigma),
            curly_velocity_scale=float(curly_velocity_scale),
            curly_learned_coupling_num_times=int(curly_learned_coupling_num_times),
            curly_learned_coupling_chunk_size=int(curly_learned_coupling_chunk_size),
        )
        optimizer_v.zero_grad(set_to_none=True)
        loss_v, energy_proxy = _cfm_loss(
            mode=mode,
            v_model=v_model,
            g_model=g_model,
            x0=x0,
            x1=x1,
            mfm_backend=mfm_backend,
            curly_path_alpha=float(curly_path_alpha),
            curly_sigma=float(curly_sigma),
            time_generator=time_generator,
            time_sampling_alpha=velocity_time_sampling_alpha,
            velocity_input_noise_sigma=velocity_input_noise_sigma,
            noise_generator=velocity_input_noise_generator,
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
        stage_b_executed_steps = step + 1
        if (
            stage_b_early_stopping_enabled
            and stage_b_executed_steps >= stage_b_early_stopping_warmup_steps
            and stage_b_executed_steps % stage_b_early_stopping_check_every == 0
        ):
            if (
                early_stop_generator is None
                or early_stop_time_generator is None
                or early_stop_noise_generator is None
            ):
                raise RuntimeError("Stage-B early stopping generators were not initialized.")
            check_losses: list[float] = []
            check_energies: list[float] = []
            for _ in range(stage_b_early_stopping_eval_batches):
                check_loss, check_energy = _eval_cfm_loss(
                    mode=mode,
                    problem=problem,
                    coupling=coupling,
                    v_model=v_model,
                    g_model=g_model,
                    mfm_backend=mfm_backend,
                    batch_size=stage_b_early_stopping_eval_batch_size,
                    generator=early_stop_generator,
                    curly_reference_pool=curly_reference_pool,
                    curly_stage_b_coupling=curly_stage_b_coupling,
                    curly_reference_k=int(curly_reference_k),
                    curly_path_alpha=float(curly_path_alpha),
                    curly_sigma=float(curly_sigma),
                    curly_velocity_scale=float(curly_velocity_scale),
                    curly_learned_coupling_num_times=int(curly_learned_coupling_num_times),
                    curly_learned_coupling_chunk_size=int(curly_learned_coupling_chunk_size),
                    time_generator=early_stop_time_generator,
                    velocity_input_noise_sigma=velocity_input_noise_sigma,
                    noise_generator=early_stop_noise_generator,
                )
                check_losses.append(float(check_loss))
                check_energies.append(float(check_energy))
            stage_b_early_stop_last_loss = float(np.mean(check_losses))
            early_stop_energy = float(np.mean(check_energies))
            min_improvement = float(stage_b_early_stopping_min_delta)
            if stage_b_early_stop_best_loss is not None:
                min_improvement = max(
                    min_improvement,
                    abs(float(stage_b_early_stop_best_loss))
                    * float(stage_b_early_stopping_min_delta_rel),
                )
            improved = (
                stage_b_early_stop_best_loss is None
                or stage_b_early_stop_last_loss
                < float(stage_b_early_stop_best_loss) - min_improvement
            )
            if improved:
                stage_b_early_stop_best_loss = float(stage_b_early_stop_last_loss)
                stage_b_early_stop_best_step = int(stage_b_executed_steps)
                stage_b_early_stop_bad_checks = 0
                if stage_b_early_stopping_restore_best:
                    stage_b_early_stop_best_state_dict = {
                        key: value.detach().cpu().clone()
                        for key, value in v_model.state_dict().items()
                    }
            else:
                stage_b_early_stop_bad_checks += 1
            stage_b_early_stop_checks += 1
            history[-1].update(
                {
                    "stage_b_early_stop_check": True,
                    "stage_b_early_stop_val_loss": float(stage_b_early_stop_last_loss),
                    "stage_b_early_stop_val_path_energy_proxy": early_stop_energy,
                    "stage_b_early_stop_best_loss": float(stage_b_early_stop_best_loss),
                    "stage_b_early_stop_bad_checks": int(stage_b_early_stop_bad_checks),
                    "stage_b_early_stop_improved": bool(improved),
                }
            )
            if stage_b_early_stop_bad_checks >= stage_b_early_stopping_patience:
                stage_b_early_stop_triggered = True
                stage_b_early_stop_step = int(stage_b_executed_steps)
                break
    if (
        stage_b_early_stop_triggered
        and stage_b_early_stopping_restore_best
        and stage_b_early_stop_best_state_dict is not None
    ):
        v_model.load_state_dict(stage_b_early_stop_best_state_dict)
        stage_b_early_stop_restored_best = True
    stage_wall_seconds["stage_b_wall_sec"] = time.perf_counter() - stage_b_start

    stage_c_start = time.perf_counter()
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
                time_generator=time_generator,
                velocity_input_noise_sigma=velocity_input_noise_sigma,
                noise_generator=velocity_input_noise_generator,
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
                moment_eta=float(train_moment_eta),
                beta_schedule=constrained_beta_schedule,
                moment_feature_blocks=moment_feature_blocks,
                moment_block_normalization=moment_block_normalization,
                moment_feature_params=moment_feature_params,
                pseudo_targets=pseudo_targets,
                pseudo_lambdas=(pseudo_lambdas if pseudo_lambdas else None),
                pseudo_rho=float(pseudo_rho),
                pseudo_eta=float(pseudo_eta),
                pseudo_posterior=pseudo_posterior,
                time_generator=time_generator,
            )
            joint = cfm + float(train_cfg["eta_joint"]) * lg
            joint.backward()
            optimizer_joint.step()
            if block_moment_constraints:
                lambdas = update_lagrange_multiplier_blocks(
                    lambdas=lambdas,  # type: ignore[arg-type]
                    residuals=residuals,  # type: ignore[arg-type]
                    rho=float(train_cfg["rho"]),
                    clip_value=float(train_cfg["lambda_clip"]),
                )
            else:
                lambdas = update_lagrange_multipliers(
                    lambdas=lambdas,  # type: ignore[arg-type]
                    residuals=residuals,  # type: ignore[arg-type]
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
                    "avg_residual_norm": float(stats["normalized_avg_residual_norm"]),
                    "raw_avg_residual_norm": float(stats["raw_avg_residual_norm"]),
                    "normalized_avg_residual_norm": float(stats["normalized_avg_residual_norm"]),
                    "pseudo_avg_residual_norm": pseudo_avg_residual_norm,
                    "pseudo_term": stats["pseudo_term"],
                }
            )
            global_step += 1
    stage_wall_seconds["stage_c_wall_sec"] = time.perf_counter() - stage_c_start

    eval_start = time.perf_counter()
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
        curly_path_alpha=curly_path_alpha,
        generator=generator,
        moment_feature_blocks=moment_feature_blocks,
        moment_feature_params=moment_feature_params,
    )
    eval_pseudo_residuals: dict[float, float] | None = None
    if pseudo_diagnostics_active and pseudo_targets is not None and pseudo_posterior is not None:
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
            curly_path_alpha=curly_path_alpha,
            generator=generator,
        )

    eval_velocity_deviation: float | None = None
    eval_temporal_smoothness: float | None = None
    if mode == "constrained" and g_model is not None:
        eval_velocity_deviation, eval_temporal_smoothness = _eval_constrained_regularizer_terms(
            problem=problem,
            coupling=coupling,
            g_model=g_model,
            batch_size=eval_batch_size,
            generator=generator,
            time_generator=time_generator,
        )

    if isinstance(problem, EmpiricalCouplingProblem) and problem.has_global_ot_support:
        x0_moment_eval, x1_moment_eval = _global_ot_support_pairs(problem)
    else:
        x0_moment_eval, x1_moment_eval, _ = sample_coupled_batch(
            problem,
            batch_size=eval_batch_size,
            coupling=coupling,
            generator=generator,
        )
    interpolant_moment_errors = _eval_interpolant_moment_errors(
        mode=mode,
        x0=x0_moment_eval,
        x1=x1_moment_eval,
        times=times,
        targets=targets,
        g_model=g_model,
        mfm_alpha=float(mfm_alpha),
        curly_path_alpha=float(curly_path_alpha),
        moment_feature_blocks=moment_feature_blocks,
        moment_block_normalization=moment_block_normalization,
        moment_feature_params=moment_feature_params,
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
    rollout_eval_times: list[float] | None = None

    # Published runners train with sealed evaluation pools and evaluate after freezing.
    if evaluate:
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
                curly_reference_pool=curly_reference_pool,
                curly_stage_b_coupling=curly_stage_b_coupling,
                curly_reference_k=int(curly_reference_k),
                curly_path_alpha=float(curly_path_alpha),
                curly_sigma=float(curly_sigma),
                curly_velocity_scale=float(curly_velocity_scale),
                curly_learned_coupling_num_times=int(curly_learned_coupling_num_times),
                curly_learned_coupling_chunk_size=int(curly_learned_coupling_chunk_size),
                time_generator=time_generator,
                velocity_input_noise_sigma=velocity_input_noise_sigma,
                noise_generator=velocity_input_noise_generator,
            )
            if data_family in {"bridge_sde", "single_cell"}:
                holdout_time_raw = cfg.get("experiment", {}).get("holdout_time", None)
                holdout_time = None if holdout_time_raw is None else float(holdout_time_raw)
                if interpolant_eval_times_override is not None:
                    rollout_eval_times = [
                        float(t) for t in interpolant_eval_times_override if 0.0 < float(t) < 1.0
                    ]
                else:
                    rollout_eval_times = [float(t) for t in times if 0.0 < float(t) < 1.0]
                if holdout_time is not None and 0.0 < float(holdout_time) < 1.0:
                    rollout_eval_times.append(float(holdout_time))
                rollout_eval_times = sorted({float(t) for t in rollout_eval_times})
                if not rollout_eval_times:
                    rollout_eval_times = sorted({float(t) for t in times})
                if eval_empirical_w2_full_pool:
                    if target_samples_by_time is None:
                        raise ValueError(
                            "train.eval_empirical_w2_full_pool=true requires target_samples_by_time "
                            "for empirical data families."
                        )
                    empirical_metrics, rollout_artifacts = _eval_empirical_rollout_metrics_full_pool(
                        problem=problem,
                        v_model=v_model,
                        times=rollout_eval_times,
                        n_steps=int(train_cfg["eval_transport_steps"]),
                        target_samples_by_time=target_samples_by_time,
                        holdout_time=holdout_time,
                    )
                else:
                    if target_sampler is None:
                        raise ValueError(f"{data_family} non-Stage-A-only evaluation requires target_sampler.")
                    empirical_metrics, rollout_artifacts = _eval_empirical_rollout_metrics(
                        problem=problem,
                        coupling=coupling,
                        v_model=v_model,
                        times=rollout_eval_times,
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
                        times=rollout_eval_times,
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
            if eval_empirical_w2_full_pool:
                if target_samples_by_time is None:
                    raise ValueError(
                        "train.eval_empirical_w2_full_pool=true requires target_samples_by_time "
                        "for Stage-A-only empirical evaluation."
                    )
                if not isinstance(problem, EmpiricalCouplingProblem):
                    raise ValueError(
                        "train.eval_empirical_w2_full_pool=true for Stage-A-only requires "
                        "an EmpiricalCouplingProblem."
                    )
                x0_eval, x1_eval = _global_ot_support_pairs(problem)
                target_sampler_fn = _full_pool_target_sampler(target_samples_by_time=target_samples_by_time)
            else:
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
                curly_path_alpha=float(curly_path_alpha),
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
                    curly_path_alpha=float(curly_path_alpha),
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
                curly_path_alpha=float(curly_path_alpha),
                generator=generator,
            )
            interpolant_artifacts = {
                "x0": x0_eval.detach().cpu(),
                "x1": x1_eval.detach().cpu(),
                "linear_by_time": _to_cpu_snapshot_dict(linear_by_time),
                "learned_by_time": _to_cpu_snapshot_dict(learned_by_time),
                "target_by_time": _to_cpu_snapshot_dict(target_by_time),
            }

        if not stage_a_only and data_family in {"bridge_sde", "single_cell"}:
            if interpolant_eval_times_override is not None:
                ab_interpolant_times = list(interpolant_eval_times_override)
            elif rollout_eval_times is not None:
                ab_interpolant_times = list(rollout_eval_times)
            else:
                ab_interpolant_times = sorted({float(t) for t in times})
            if eval_empirical_w2_full_pool:
                if target_samples_by_time is None or not isinstance(problem, EmpiricalCouplingProblem):
                    raise ValueError(
                        "A+B full-pool interpolant evaluation requires empirical target pools."
                    )
                x0_ab_interp, x1_ab_interp = _global_ot_support_pairs(problem)
                ab_target_sampler = _full_pool_target_sampler(
                    target_samples_by_time=target_samples_by_time
                )
            else:
                if target_sampler is None:
                    raise ValueError("A+B interpolant evaluation requires a target sampler.")
                x0_ab_interp, x1_ab_interp, _ = sample_coupled_batch(
                    problem,
                    batch_size=int(train_cfg.get("eval_intermediate_ot_samples", 256)),
                    coupling=coupling,
                    generator=generator,
                )
                ab_target_sampler = target_sampler
            interpolant_eval = interpolant_empirical_w2_metrics(
                x0=x0_ab_interp,
                x1=x1_ab_interp,
                times=ab_interpolant_times,
                target_sampler=ab_target_sampler,
                g_model=g_model,
                mode=mode,
                mfm_alpha=float(mfm_alpha),
                curly_path_alpha=float(curly_path_alpha),
                holdout_time=None,
                generator=generator,
            )

    final_lambda_stats = (
        _lagrange_multiplier_diagnostics(
            lambdas=lambdas,
            clip_value=float(train_cfg["lambda_clip"]),
        )
        if lambdas
        else None
    )
    final_pseudo_lambda_stats = (
        _lagrange_multiplier_diagnostics(
            lambdas=pseudo_lambdas,
            clip_value=float(pseudo_lambda_clip),
        )
        if pseudo_lambdas
        else None
    )
    summary: dict[str, Any] = {
        "mode": mode,
        "coupling": coupling,
        "data_family": data_family,
        "stage_steps": stage_steps,
        "stage_a_only": bool(stage_a_only),
        "stage_c_enabled": bool(stage_steps["stage_c_steps"] > 0),
        "velocity_input_noise_sigma": float(velocity_input_noise_sigma),
        "velocity_time_sampling_alpha": float(velocity_time_sampling_alpha),
        "velocity_input_noise_seed": int(velocity_input_noise_seed),
        "eval_empirical_w2_full_pool": bool(eval_empirical_w2_full_pool),
        "eval_full_ot_metrics": bool(eval_full_ot_metrics),
        "eval_full_ot_method": str(eval_full_ot_method),
        "eval_full_ot_num_itermax": eval_full_ot_num_itermax,
        "eval_full_ot_max_variables": eval_full_ot_max_variables,
        "eval_full_ot_support_tol": float(eval_full_ot_support_tol),
        "constraint_times": [float(t) for t in times],
        "moment_feature_blocks": [str(block) for block in moment_feature_blocks],
        "moment_feature_params": moment_feature_params,
        "moment_block_normalization": str(moment_block_normalization),
        "moment_block_schedule_enabled": bool(raw_moment_block_schedule is not None),
        "moment_block_schedule": _summarize_moment_block_schedule(
            schedule=moment_block_schedule,
            total_steps=stage_steps["stage_a_steps"],
        ),
        "path_zero_init_output": bool(model_cfg.get("path_zero_init_output", False)),
        "interpolant_eval_times": (
            None if interpolant_eval_times_override is None else [float(t) for t in interpolant_eval_times_override]
        ),
        "rollout_eval_times": (None if rollout_eval_times is None else [float(t) for t in rollout_eval_times]),
        "cfm_val_loss": cfm_val,
        "path_energy_proxy": eval_path_energy,
        "interpolant_velocity_deviation": eval_velocity_deviation,
        "interpolant_temporal_smoothness": eval_temporal_smoothness,
        "stage_b_executed_steps": int(stage_b_executed_steps),
        "stage_b_early_stopping_enabled": bool(stage_b_early_stopping_enabled),
        "stage_b_early_stopping_check_every": int(stage_b_early_stopping_check_every),
        "stage_b_early_stopping_warmup_steps": int(stage_b_early_stopping_warmup_steps),
        "stage_b_early_stopping_patience": int(stage_b_early_stopping_patience),
        "stage_b_early_stopping_min_delta": float(stage_b_early_stopping_min_delta),
        "stage_b_early_stopping_min_delta_rel": float(stage_b_early_stopping_min_delta_rel),
        "stage_b_early_stopping_eval_batch_size": int(
            stage_b_early_stopping_eval_batch_size
        ),
        "stage_b_early_stopping_eval_batches": int(stage_b_early_stopping_eval_batches),
        "stage_b_early_stopping_restore_best": bool(stage_b_early_stopping_restore_best),
        "stage_b_early_stopping_seed": int(stage_b_early_stopping_seed),
        "stage_b_early_stop_checks": int(stage_b_early_stop_checks),
        "stage_b_early_stop_triggered": bool(stage_b_early_stop_triggered),
        "stage_b_early_stop_step": stage_b_early_stop_step,
        "stage_b_early_stop_best_step": stage_b_early_stop_best_step,
        "stage_b_early_stop_restored_best": bool(stage_b_early_stop_restored_best),
        "stage_b_early_stop_best_loss": stage_b_early_stop_best_loss,
        "stage_b_early_stop_last_loss": stage_b_early_stop_last_loss,
        "stage_b_early_stop_bad_checks": int(stage_b_early_stop_bad_checks),
        "constraint_residual_norms": {f"{k:.2f}": v for k, v in eval_residuals.items()},
        "constraint_residual_avg": float(np.mean(list(eval_residuals.values()))),
        "interpolant_moment_errors_raw": interpolant_moment_errors["raw"],
        "interpolant_moment_errors_normalized": interpolant_moment_errors["normalized"],
        "interpolant_moment_errors_raw_l2_avg_by_block": interpolant_moment_errors[
            "raw_l2_avg_by_block"
        ],
        "interpolant_moment_errors_raw_rmse_avg_by_block": interpolant_moment_errors[
            "raw_rmse_avg_by_block"
        ],
        "interpolant_moment_errors_normalized_l2_avg_by_block": interpolant_moment_errors[
            "normalized_l2_avg_by_block"
        ],
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
        "pseudo_diagnostics_active": bool(pseudo_diagnostics_active),
        "train_moment_eta": float(train_moment_eta),
        "lambda_diagnostics": (
            None
            if final_lambda_stats is None
            else {
                "final_l2": float(final_lambda_stats["l2"]),
                "final_linf": float(final_lambda_stats["linf"]),
                "final_clip_fraction": float(final_lambda_stats["clip_fraction"]),
                "numel": int(final_lambda_stats["numel"]),
                "max_linf": float(lambda_linf_max),
                "max_clip_fraction": float(lambda_clip_fraction_max),
                "saturated_steps": int(lambda_saturated_steps),
                "update_steps": int(lambda_update_steps),
            }
        ),
        "pseudo_lambda_diagnostics": (
            None
            if final_pseudo_lambda_stats is None
            else {
                "final_l2": float(final_pseudo_lambda_stats["l2"]),
                "final_linf": float(final_pseudo_lambda_stats["linf"]),
                "final_clip_fraction": float(final_pseudo_lambda_stats["clip_fraction"]),
                "numel": int(final_pseudo_lambda_stats["numel"]),
            }
        ),
        "metric_constraint_warmup_steps": int(metric_constraint_warmup_steps),
        "pseudo_eta": float(pseudo_eta),
        "pseudo_rho": float(pseudo_rho),
        "pseudo_lambda_clip": float(pseudo_lambda_clip),
        "seed": int(cfg["seed"]),
        "base_seed": int(base_seed),
        "init_seed": int(init_seed),
        "batch_seed": int(batch_seed),
        "time_seed": None if time_seed_raw is None else int(time_seed_raw),
    }
    if "protocol" in cfg.get("experiment", {}):
        summary["protocol"] = str(cfg["experiment"].get("protocol"))
    if "holdout_index" in cfg.get("experiment", {}):
        summary["holdout_index"] = cfg["experiment"].get("holdout_index")
    if "holdout_time" in cfg.get("experiment", {}):
        summary["holdout_time"] = cfg["experiment"].get("holdout_time")
    if isinstance(problem, EmpiricalCouplingProblem):
        summary["x0_pool_size"] = int(problem.x0_pool.shape[0])
        summary["x1_pool_size"] = int(problem.x1_pool.shape[0])
        summary["global_ot_support_size"] = (
            None if problem.global_ot_mass is None else int(problem.global_ot_mass.numel())
        )
        summary["global_ot_total_cost"] = (
            None if problem.global_ot_total_cost is None else float(problem.global_ot_total_cost)
        )
        if eval_empirical_w2_full_pool:
            summary["eval_empirical_w2_full_pool_size"] = int(problem.x0_pool.shape[0])
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
        summary["mfm_geopath_metric"] = str(mfm_geopath_metric)
        summary["mfm_land_gamma"] = float(mfm_land_gamma)
        summary["mfm_land_rho"] = float(mfm_land_rho)
        summary["mfm_reference_pool_policy"] = str(mfm_reference_pool_policy)
        summary["mfm_moment_style"] = _metric_moment_style(mode)
        summary["mfm_moment_eta"] = float(mfm_moment_eta)
        if rbf_metric is not None:
            summary["mfm_rbf_n_centers"] = int(rbf_metric.n_centers)
            summary["mfm_rbf_kappa"] = float(rbf_metric.kappa)
            summary["mfm_rbf_epsilon"] = float(mfm_rbf_epsilon)
            summary["mfm_rbf_alpha_metric"] = float(mfm_rbf_alpha_metric)
            summary["mfm_rbf_metric_epochs"] = int(mfm_rbf_metric_epochs)
            summary["mfm_rbf_lr"] = float(mfm_rbf_lr)
            summary["mfm_rbf_seed"] = int(mfm_rbf_seed)
            summary["mfm_rbf_train_loss"] = float(rbf_metric.train_loss)
    if mode in CURLY_MODES:
        summary["curly_path_alpha"] = float(curly_path_alpha)
        summary["curly_sigma"] = float(curly_sigma)
        summary["curly_velocity_scale"] = float(curly_velocity_scale)
        summary["curly_reference_pool_policy"] = str(
            None if curly_reference_pool is None else curly_reference_pool.policy
        )
        summary["curly_reference_pool_times"] = (
            None
            if curly_reference_pool is None
            else [float(t) for t in curly_reference_pool.times]
        )
        summary["curly_reference_pool_size"] = (
            None if curly_reference_pool is None else int(curly_reference_pool.size)
        )
        summary["curly_reference_pool_max_samples_per_time"] = curly_reference_pool_max_samples
        summary["curly_reference_k"] = int(curly_reference_k)
        summary["curly_cosine_weight"] = float(curly_cosine_weight)
        summary["curly_l2_weight"] = float(curly_l2_weight)
        summary["curly_l2_mu_dot_scale"] = float(curly_l2_mu_dot_scale)
        summary["curly_stage_b_coupling"] = str(curly_stage_b_coupling)
        summary["curly_learned_coupling_num_times"] = int(curly_learned_coupling_num_times)
        summary["curly_learned_coupling_chunk_size"] = int(curly_learned_coupling_chunk_size)
        summary["curly_moment_style"] = _curly_moment_style(mode)
        summary["curly_moment_eta"] = float(curly_moment_eta)
    summary.update(transport)
    if interpolant_eval is not None:
        summary["interpolant_eval"] = interpolant_eval
    stage_wall_seconds["eval_wall_sec"] = time.perf_counter() - eval_start
    stage_wall_seconds["train_experiment_wall_sec"] = (
        time.perf_counter() - train_experiment_start
    )

    checkpoint = {
        "velocity_state_dict": v_model.state_dict(),
        "path_state_dict": None if g_model is None else g_model.state_dict(),
        "pseudo_lagrange_multipliers": {
            float(t): value.detach().cpu()
            for t, value in pseudo_lambdas.items()
        },
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
        "cost_metrics": {
            "stage_wall_seconds": stage_wall_seconds,
        },
    }
