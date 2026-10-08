from __future__ import annotations

from typing import Sequence

import torch

from cfm_project.data import sample_gaussian


def simulate_bridge_sde_trajectories(
    n_samples: int,
    n_steps: int,
    total_time: float,
    mean0: Sequence[float],
    cov0: Sequence[Sequence[float]],
    vx: float,
    sigma_x: float,
    sigma_y: float,
    bridge_center_x: float,
    bridge_width: float,
    bridge_pull: float,
    bridge_diffusion_drop: float,
    arc_y_amplitude: float = 0.0,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if total_time <= 0.0:
        raise ValueError(f"total_time must be positive, got {total_time}")
    if bridge_width <= 0.0:
        raise ValueError(f"bridge_width must be positive, got {bridge_width}")

    dev = torch.device(device)
    mean0_t = torch.tensor(mean0, device=dev, dtype=dtype)
    cov0_t = torch.tensor(cov0, device=dev, dtype=dtype)
    start = sample_gaussian(mean0_t, cov0_t, n_samples=n_samples, generator=generator)

    trajectories = torch.empty((n_samples, n_steps + 1, 2), device=dev, dtype=dtype)
    trajectories[:, 0, :] = start
    times = torch.linspace(0.0, float(total_time), n_steps + 1, device=dev, dtype=dtype)

    dt = float(total_time) / float(n_steps)
    sqrt_dt = dt**0.5
    center = float(bridge_center_x)
    width_sq_twice = 2.0 * float(bridge_width) * float(bridge_width)
    arc_amplitude = float(arc_y_amplitude)

    for step in range(n_steps):
        x_prev = trajectories[:, step, 0]
        y_prev = trajectories[:, step, 1]
        t_prev = times[step]
        gate = torch.exp(-((x_prev - center) ** 2) / width_sq_twice)

        eps = torch.randn((n_samples, 2), device=dev, dtype=dtype, generator=generator)

        x_next = x_prev + float(vx) * dt + float(sigma_x) * sqrt_dt * eps[:, 0]
        arc_center = arc_amplitude * torch.sin(torch.pi * t_prev / float(total_time))
        y_drift = -float(bridge_pull) * gate * (y_prev - arc_center)
        if arc_amplitude != 0.0:
            arc_velocity = arc_amplitude * torch.pi / float(total_time) * torch.cos(
                torch.pi * t_prev / float(total_time)
            )
            y_drift = y_drift + arc_velocity
        y_diffusion_scale = torch.clamp(
            1.0 - float(bridge_diffusion_drop) * gate,
            min=0.0,
        )
        y_next = y_prev + y_drift * dt + float(sigma_y) * y_diffusion_scale * sqrt_dt * eps[:, 1]

        trajectories[:, step + 1, 0] = x_next
        trajectories[:, step + 1, 1] = y_next

    return times, trajectories


def sample_bridge_sde_at_times(
    sample_times: Sequence[float],
    trajectory_times: torch.Tensor,
    trajectories: torch.Tensor,
) -> dict[float, torch.Tensor]:
    if trajectory_times.ndim != 1:
        raise ValueError(f"trajectory_times must be 1D, got shape {tuple(trajectory_times.shape)}")
    if trajectories.ndim != 3 or trajectories.shape[2] != 2:
        raise ValueError(f"trajectories must have shape (N, T, 2), got {tuple(trajectories.shape)}")
    if trajectories.shape[1] != trajectory_times.shape[0]:
        raise ValueError(
            "trajectory_times length must match trajectory step dimension, got "
            f"{trajectory_times.shape[0]} and {trajectories.shape[1]}"
        )

    t_min = float(trajectory_times[0].item())
    t_max = float(trajectory_times[-1].item())
    out: dict[float, torch.Tensor] = {}
    for t in sorted({float(ti) for ti in sample_times}):
        if t < t_min or t > t_max:
            raise ValueError(f"sample time {t} outside trajectory range [{t_min}, {t_max}]")
        idx = int(torch.argmin(torch.abs(trajectory_times - float(t))).item())
        out[float(t)] = trajectories[:, idx, :].clone()
    return out
