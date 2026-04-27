from __future__ import annotations

from typing import Tuple

import torch

from cfm_project.models import PathCorrection


def format_time(
    t: torch.Tensor | float,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool = False,
) -> torch.Tensor:
    if isinstance(t, (float, int)):
        tensor_t = torch.full((batch_size, 1), float(t), device=device, dtype=dtype)
    else:
        tensor_t = t.to(device=device, dtype=dtype)
        if tensor_t.ndim == 0:
            tensor_t = tensor_t.reshape(1, 1).expand(batch_size, 1)
        elif tensor_t.ndim == 1:
            tensor_t = tensor_t.unsqueeze(1)
        if tensor_t.shape != (batch_size, 1):
            raise ValueError(f"Expected t shape ({batch_size}, 1), got {tuple(tensor_t.shape)}")
    if requires_grad:
        tensor_t = tensor_t.detach().clone().requires_grad_(True)
    return tensor_t


def linear_path(t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    return (1.0 - t) * x0 + t * x1


def corrected_path(
    t: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
    g_model: PathCorrection,
) -> torch.Tensor:
    linear = linear_path(t, x0, x1)
    gate = t * (1.0 - t)
    g_val = g_model(t, x0, x1)
    return linear + gate * g_val


def vector_time_derivative(
    y: torch.Tensor,
    t: torch.Tensor,
    create_graph: bool,
) -> torch.Tensor:
    comps: list[torch.Tensor] = []
    for idx in range(y.shape[1]):
        grad = torch.autograd.grad(
            y[:, idx].sum(),
            t,
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=False,
        )[0]
        comps.append(grad)
    return torch.cat(comps, dim=1)


def corrected_velocity(
    t: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
    g_model: PathCorrection,
    create_graph: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    t_req = format_time(
        t=t,
        batch_size=x0.shape[0],
        device=x0.device,
        dtype=x0.dtype,
        requires_grad=True,
    )
    g_val = g_model(t_req, x0, x1)
    dg_dt = vector_time_derivative(g_val, t_req, create_graph=create_graph)
    gate = t_req * (1.0 - t_req)
    base_velocity = x1 - x0
    velocity = base_velocity + (1.0 - 2.0 * t_req) * g_val + gate * dg_dt
    return velocity, g_val, dg_dt, t_req


def path_and_velocity(
    mode: str,
    t: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
    g_model: PathCorrection | None = None,
    create_graph: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t_fmt = format_time(
        t=t,
        batch_size=x0.shape[0],
        device=x0.device,
        dtype=x0.dtype,
        requires_grad=(mode == "constrained"),
    )
    if mode == "baseline":
        xt = linear_path(t_fmt, x0, x1)
        ut = x1 - x0
        return xt, ut, t_fmt

    if g_model is None:
        raise ValueError("g_model must be provided in constrained mode.")

    xt = corrected_path(t_fmt, x0, x1, g_model)
    ut, _, _, t_req = corrected_velocity(
        t=t_fmt,
        x0=x0,
        x1=x1,
        g_model=g_model,
        create_graph=create_graph,
    )
    return xt, ut, t_req

