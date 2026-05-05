from __future__ import annotations

import pytest
import torch

from cfm_project.data import (
    EmpiricalCouplingProblem,
    analytic_target_moment_features,
    to_problem_from_config,
)
from cfm_project.models import PathCorrection
from cfm_project.pipeline import run_pipeline
from cfm_project.training import (
    _beta_weights_at_times,
    _build_constrained_beta_schedule,
    _constrained_objective,
    _init_lambdas,
)


def _small_constrained_config(beta_schedule: str) -> dict:
    return {
        "seed": 123,
        "device": "cpu",
        "experiment": {
            "mode": "constrained",
            "run_both_modes": False,
        },
        "data": {
            "dim": 2,
            "mean0": [0.0, 0.0],
            "cov0": [[1.0, 0.2], [0.2, 0.9]],
            "mean1": [1.2, -0.4],
            "cov1": [[0.8, -0.1], [-0.1, 1.1]],
            "kappa": 0.5,
            "constraint_times": [0.25, 0.5, 0.75],
        },
        "model": {
            "activation": "silu",
            "velocity_hidden_dims": [24, 24],
            "path_hidden_dims": [24, 24],
        },
        "train": {
            "stage_a_steps": 3,
            "stage_b_steps": 4,
            "stage_c_steps": 2,
            "batch_size": 16,
            "eval_batch_size": 24,
            "eval_transport_samples": 48,
            "eval_transport_steps": 10,
            "eval_intermediate_empirical_w2": True,
            "eval_intermediate_ot_samples": 24,
            "lr_g": 0.001,
            "lr_v": 0.001,
            "alpha": 1.0,
            "beta": 0.05,
            "beta_schedule": beta_schedule,
            "beta_drift_p": 1.0,
            "beta_drift_eps": 1.0e-6,
            "beta_min_scale": 0.3,
            "beta_max_scale": 3.0,
            "rho": 3.0,
            "eta_joint": 0.05,
            "lambda_clip": 100.0,
        },
        "output": {
            "save_checkpoint": False,
            "save_plots": False,
            "plot_pairs": 8,
        },
    }


def test_constant_schedule_matches_legacy_scalar_beta_objective() -> None:
    problem = to_problem_from_config(
        mean0=[0.0, 0.0],
        cov0=[[1.0, 0.1], [0.1, 0.9]],
        mean1=[1.0, -0.3],
        cov1=[[0.9, -0.05], [-0.05, 1.2]],
        kappa=0.4,
        device=torch.device("cpu"),
    )
    times = [0.25, 0.5, 0.75]
    targets = analytic_target_moment_features(times=times, problem=problem)
    schedule = _build_constrained_beta_schedule(
        problem=problem,
        targets=targets,
        constraint_times=times,
        beta0=0.05,
        beta_schedule="constant",
        drift_p=1.0,
        drift_eps=1.0e-6,
        min_scale=0.3,
        max_scale=3.0,
    )

    model = PathCorrection(state_dim=2, hidden_dims=(16, 16))
    x0 = torch.randn(48, 2)
    x1 = torch.randn(48, 2)
    lambdas = _init_lambdas(times=times, targets=targets)

    torch.manual_seed(777)
    legacy_loss, _, legacy_stats = _constrained_objective(
        g_model=model,
        x0=x0,
        x1=x1,
        times=times,
        targets=targets,
        lambdas=lambdas,
        rho=1.0,
        alpha=1.0,
        beta=0.05,
        beta_schedule=None,
    )
    torch.manual_seed(777)
    scheduled_loss, _, scheduled_stats = _constrained_objective(
        g_model=model,
        x0=x0,
        x1=x1,
        times=times,
        targets=targets,
        lambdas=lambdas,
        rho=1.0,
        alpha=1.0,
        beta=0.05,
        beta_schedule=schedule,
    )

    assert torch.allclose(legacy_loss, scheduled_loss, atol=1e-7, rtol=1e-7)
    assert scheduled_stats["beta_t_mean"] == pytest.approx(0.05, abs=1e-8)
    assert legacy_stats["regularizer"] == pytest.approx(scheduled_stats["regularizer"], abs=1e-8)


def test_piecewise_and_linear_schedule_values_match_expected_synthetic_case() -> None:
    x0_pool = torch.zeros(64, 2)
    x1_pool = torch.zeros(64, 2)
    x1_pool[:, 0] = 2.0
    problem = EmpiricalCouplingProblem(x0_pool=x0_pool, x1_pool=x1_pool, label="synthetic")

    targets = {
        0.25: torch.tensor([0.2, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        0.50: torch.tensor([0.6, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        0.75: torch.tensor([1.6, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
    }
    times = [0.25, 0.5, 0.75]
    piecewise = _build_constrained_beta_schedule(
        problem=problem,
        targets=targets,
        constraint_times=times,
        beta0=0.1,
        beta_schedule="piecewise",
        drift_p=1.0,
        drift_eps=1.0e-12,
        min_scale=1.0e-9,
        max_scale=10.0,
    )
    linear = _build_constrained_beta_schedule(
        problem=problem,
        targets=targets,
        constraint_times=times,
        beta0=0.1,
        beta_schedule="linear",
        drift_p=1.0,
        drift_eps=1.0e-12,
        min_scale=1.0e-9,
        max_scale=10.0,
    )

    assert piecewise["interval_drifts"] == pytest.approx([0.8, 1.6, 4.0, 1.6], rel=1e-6)
    assert piecewise["interval_betas"] == pytest.approx([0.25, 0.125, 0.05, 0.125], rel=1e-6)
    assert linear["anchor_betas"] == pytest.approx([0.25, 0.1875, 0.0875, 0.0875, 0.125], rel=1e-6)

    t_probe = torch.tensor([[0.10], [0.30], [0.60], [0.90]], dtype=torch.float32)
    piecewise_values = _beta_weights_at_times(t=t_probe, beta0=0.1, beta_schedule=piecewise)
    linear_values = _beta_weights_at_times(t=t_probe, beta0=0.1, beta_schedule=linear)
    assert piecewise_values.tolist() == pytest.approx([0.25, 0.125, 0.05, 0.125], rel=1e-6)
    assert linear_values.tolist() == pytest.approx([0.225, 0.1675, 0.0875, 0.11], rel=1e-6)


def test_higher_drift_intervals_receive_lower_beta_within_bounds() -> None:
    x0_pool = torch.zeros(64, 2)
    x1_pool = torch.zeros(64, 2)
    x1_pool[:, 0] = 2.0
    problem = EmpiricalCouplingProblem(x0_pool=x0_pool, x1_pool=x1_pool, label="synthetic")
    targets = {
        0.25: torch.tensor([0.2, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        0.50: torch.tensor([0.6, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        0.75: torch.tensor([1.6, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
    }
    schedule = _build_constrained_beta_schedule(
        problem=problem,
        targets=targets,
        constraint_times=[0.25, 0.5, 0.75],
        beta0=0.1,
        beta_schedule="piecewise",
        drift_p=1.0,
        drift_eps=1.0e-12,
        min_scale=0.5,
        max_scale=2.0,
    )
    drifts = schedule["interval_drifts"]
    betas = schedule["interval_betas"]
    max_drift_idx = int(torch.tensor(drifts).argmax().item())
    min_beta_idx = int(torch.tensor(betas).argmin().item())
    assert max_drift_idx == min_beta_idx
    assert min(betas) >= 0.1 * 0.5 - 1e-8
    assert max(betas) <= 0.1 * 2.0 + 1e-8


@pytest.mark.parametrize("beta_schedule", ["piecewise", "linear"])
def test_constrained_pipeline_smoke_supports_beta_schedule_modes(
    tmp_path,
    beta_schedule: str,
) -> None:
    cfg = _small_constrained_config(beta_schedule=beta_schedule)
    result = run_pipeline(cfg, output_dir=tmp_path / beta_schedule)
    summary = result["summary"]
    assert summary["mode"] == "constrained"
    assert summary["beta_schedule"] == beta_schedule
    assert summary["beta_schedule_base"] == pytest.approx(cfg["train"]["beta"])
    assert len(summary["beta_schedule_anchor_times"]) == 5
    assert len(summary["beta_schedule_interval_values"]) == 4
