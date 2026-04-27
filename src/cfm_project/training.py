from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

from cfm_project.constraints import (
    augmented_lagrangian_terms,
    constraint_residuals,
    residual_norms,
    update_lagrange_multipliers,
)
from cfm_project.data import GaussianOTProblem, sample_coupled_batch
from cfm_project.metrics import (
    intermediate_empirical_w2_metrics,
    intermediate_wasserstein_metrics,
    path_energy_proxy,
    transport_quality_metrics,
)
from cfm_project.models import PathCorrection, VelocityField
from cfm_project.paths import corrected_path, path_and_velocity, vector_time_derivative


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _uniform_time(batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.rand((batch_size, 1), device=device, dtype=dtype)


def _constraint_residuals_for_mode(
    mode: str,
    x0: torch.Tensor,
    x1: torch.Tensor,
    times: list[float],
    targets: dict[float, torch.Tensor],
    g_model: PathCorrection | None,
) -> dict[float, torch.Tensor]:
    def path_fn(t_value: float) -> torch.Tensor:
        t_batch = torch.full((x0.shape[0], 1), t_value, device=x0.device, dtype=x0.dtype)
        if mode == "baseline":
            return (1.0 - t_batch) * x0 + t_batch * x1
        if g_model is None:
            raise ValueError("g_model is required in constrained mode.")
        return corrected_path(t_batch, x0, x1, g_model)

    return constraint_residuals(path_fn=path_fn, times=times, targets=targets)


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
) -> tuple[torch.Tensor, dict[float, torch.Tensor], dict[str, float]]:
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
    smoothness = torch.mean(torch.sum(du_dt**2, dim=1))
    regularizer = alpha * energy + beta * smoothness

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
    stats = {
        "regularizer": float(regularizer.detach().item()),
        "energy_term": float(energy.detach().item()),
        "smoothness_term": float(smoothness.detach().item()),
        "al_term": float(al_term.detach().item()),
    }
    for t, value in per_time.items():
        stats[f"al_t_{t:.2f}"] = float(value)
    return total, residuals, stats


def _cfm_loss(
    mode: str,
    v_model: VelocityField,
    g_model: PathCorrection | None,
    x0: torch.Tensor,
    x1: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    t_rand = _uniform_time(x0.shape[0], x0.device, x0.dtype)
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
    problem: GaussianOTProblem,
    coupling: str,
    batch_size: int,
    times: list[float],
    targets: dict[float, torch.Tensor],
    g_model: PathCorrection | None,
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
    )
    return residual_norms(residuals)


def _eval_cfm_loss(
    mode: str,
    problem: GaussianOTProblem,
    coupling: str,
    v_model: VelocityField,
    g_model: PathCorrection | None,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[float, float]:
    x0, x1, _ = sample_coupled_batch(
        problem,
        batch_size=batch_size,
        coupling=coupling,
        generator=generator,
    )
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


def train_experiment(
    cfg: dict[str, Any],
    problem: GaussianOTProblem,
    targets: dict[float, torch.Tensor],
) -> dict[str, Any]:
    device = torch.device(cfg["device"])
    dtype = torch.float32
    mode = str(cfg["experiment"]["mode"])

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

    g_model: PathCorrection | None = None
    if mode == "constrained":
        g_model = PathCorrection(
            state_dim=state_dim,
            hidden_dims=model_cfg["path_hidden_dims"],
            activation=model_cfg["activation"],
        ).to(device=device, dtype=dtype)

    train_cfg = cfg["train"]
    times = [float(t) for t in cfg["data"]["constraint_times"]]
    coupling = str(cfg["data"].get("coupling", "ot")).lower()
    batch_size = int(train_cfg["batch_size"])
    eval_batch_size = int(train_cfg["eval_batch_size"])

    history: list[dict[str, float | str | int]] = []
    global_step = 0

    if mode == "constrained" and g_model is not None:
        lambdas = _init_lambdas(times=times, targets=targets)
        optimizer_g = torch.optim.Adam(g_model.parameters(), lr=float(train_cfg["lr_g"]))
        for step in range(int(train_cfg["stage_a_steps"])):
            x0, x1, _ = sample_coupled_batch(
                problem,
                batch_size=batch_size,
                coupling=coupling,
                generator=generator,
            )
            optimizer_g.zero_grad(set_to_none=True)
            loss_g, residuals, stats = _constrained_objective(
                g_model=g_model,
                x0=x0,
                x1=x1,
                times=times,
                targets=targets,
                lambdas=lambdas,
                rho=float(train_cfg["rho"]),
                alpha=float(train_cfg["alpha"]),
                beta=float(train_cfg["beta"]),
            )
            loss_g.backward()
            optimizer_g.step()
            lambdas = update_lagrange_multipliers(
                lambdas=lambdas,
                residuals=residuals,
                rho=float(train_cfg["rho"]),
                clip_value=float(train_cfg["lambda_clip"]),
            )
            history.append(
                {
                    "stage": "stage_a",
                    "step": step,
                    "global_step": global_step,
                    "loss": float(loss_g.detach().item()),
                    "avg_residual_norm": float(np.mean(list(residual_norms(residuals).values()))),
                    "regularizer": stats["regularizer"],
                }
            )
            global_step += 1
    else:
        lambdas = {}

    optimizer_v = torch.optim.Adam(v_model.parameters(), lr=float(train_cfg["lr_v"]))
    if mode == "constrained" and g_model is not None:
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
            lg, residuals, _ = _constrained_objective(
                g_model=g_model,
                x0=x0,
                x1=x1,
                times=times,
                targets=targets,
                lambdas=lambdas,
                rho=float(train_cfg["rho"]),
                alpha=float(train_cfg["alpha"]),
                beta=float(train_cfg["beta"]),
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
            history.append(
                {
                    "stage": "stage_c",
                    "step": step,
                    "global_step": global_step,
                    "loss": float(joint.detach().item()),
                    "cfm_loss": float(cfm.detach().item()),
                    "path_energy_proxy": float(energy_proxy),
                    "avg_residual_norm": float(np.mean(list(residual_norms(residuals).values()))),
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
        generator=generator,
    )
    cfm_val, eval_path_energy = _eval_cfm_loss(
        mode=mode,
        problem=problem,
        coupling=coupling,
        v_model=v_model,
        g_model=g_model,
        batch_size=eval_batch_size,
        generator=generator,
    )
    transport = transport_quality_metrics(
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
    empirical_w2 = {}
    if bool(train_cfg.get("eval_intermediate_empirical_w2", True)):
        empirical_w2 = intermediate_empirical_w2_metrics(
            velocity_fn=v_model,
            problem=problem,
            times=times,
            n_samples=int(train_cfg.get("eval_intermediate_ot_samples", 256)),
            n_steps=int(train_cfg["eval_transport_steps"]),
            generator=generator,
        )

    summary = {
        "mode": mode,
        "coupling": coupling,
        "cfm_val_loss": cfm_val,
        "path_energy_proxy": float(eval_path_energy),
        "constraint_residual_norms": {f"{k:.2f}": v for k, v in eval_residuals.items()},
        "constraint_residual_avg": float(np.mean(list(eval_residuals.values()))),
        **transport,
        **intermediate_w2,
        **empirical_w2,
        "seed": int(cfg["seed"]),
    }

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
    }
