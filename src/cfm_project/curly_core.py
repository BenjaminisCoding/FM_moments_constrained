from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


CURLY_REFERENCE_POOL_POLICIES = {
    "endpoints_only",
    "endpoints_plus_middle",
    "all_marginals",
}
CURLY_STAGE_B_COUPLINGS = {"same", "learned"}


@dataclass(frozen=True)
class CurlyReferencePool:
    positions: torch.Tensor
    velocities: torch.Tensor
    times: tuple[float, ...]
    policy: str

    @property
    def size(self) -> int:
        return int(self.positions.shape[0])


def normalize_reference_pool_policy(policy: str) -> str:
    normalized = str(policy).strip().lower().replace("-", "_").replace("+", "_plus_")
    aliases = {
        "endpoints": "endpoints_only",
        "endpoint_only": "endpoints_only",
        "endpoints_only": "endpoints_only",
        "endpoints_plus_middle": "endpoints_plus_middle",
        "endpoints_plus_midpoint": "endpoints_plus_middle",
        "endpoints_plus_midpoints": "endpoints_plus_middle",
        "all": "all_marginals",
        "all_marginal": "all_marginals",
        "all_marginals": "all_marginals",
    }
    out = aliases.get(normalized)
    if out is None:
        raise ValueError(
            f"Unsupported curly.reference_pool_policy '{policy}'. "
            f"Expected one of: {sorted(CURLY_REFERENCE_POOL_POLICIES)}."
        )
    return out


def normalize_stage_b_coupling(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "same": "same",
        "fixed": "same",
        "configured": "same",
        "curly": "learned",
        "curly_learned": "learned",
        "learned": "learned",
    }
    out = aliases.get(normalized)
    if out is None:
        raise ValueError(
            f"Unsupported curly.stage_b_coupling '{value}'. "
            f"Expected one of: {sorted(CURLY_STAGE_B_COUPLINGS)}."
        )
    return out


def curly_gamma(t: torch.Tensor) -> torch.Tensor:
    return t * (1.0 - t)


def _as_time_column(
    t: torch.Tensor | None,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if t is None:
        out = torch.rand((batch_size, 1), device=device, dtype=dtype, generator=generator)
    else:
        out = t.to(device=device, dtype=dtype)
        if out.ndim == 0:
            out = out.reshape(1, 1).expand(batch_size, 1)
        elif out.ndim == 1:
            out = out.unsqueeze(1)
    if out.shape != (batch_size, 1):
        raise ValueError(f"Expected t with shape ({batch_size}, 1), got {tuple(out.shape)}")
    if requires_grad:
        out = out.detach().clone().requires_grad_(True)
    return out


def _vector_time_derivative(
    y: torch.Tensor,
    t: torch.Tensor,
    create_graph: bool,
) -> torch.Tensor:
    comps: list[torch.Tensor] = []
    for i in range(y.shape[1]):
        grad = torch.autograd.grad(
            y[:, i].sum(),
            t,
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=False,
        )[0]
        comps.append(grad)
    return torch.cat(comps, dim=1)


def curly_mean_path(
    t: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
    geopath_net: torch.nn.Module | None,
    path_alpha: float = 1.0,
) -> torch.Tensor:
    linear = (1.0 - t) * x0 + t * x1
    if geopath_net is None or float(path_alpha) == 0.0:
        return linear
    return linear + float(path_alpha) * curly_gamma(t) * geopath_net(t, x0, x1)


def curly_path_and_velocity(
    t: torch.Tensor | None,
    x0: torch.Tensor,
    x1: torch.Tensor,
    geopath_net: torch.nn.Module | None,
    path_alpha: float,
    sigma: float,
    create_graph: bool,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    needs_time_grad = geopath_net is not None and float(path_alpha) != 0.0
    t_col = _as_time_column(
        t=t,
        batch_size=x0.shape[0],
        device=x0.device,
        dtype=x0.dtype,
        requires_grad=needs_time_grad,
        generator=generator,
    )
    base_velocity = x1 - x0
    mu_t = curly_mean_path(
        t=t_col,
        x0=x0,
        x1=x1,
        geopath_net=geopath_net,
        path_alpha=float(path_alpha),
    )
    if not needs_time_grad:
        mu_dot = base_velocity
    else:
        mu_dot = _vector_time_derivative(mu_t, t_col, create_graph=create_graph)

    if float(sigma) == 0.0:
        return mu_t, mu_dot, t_col
    eps = torch.randn(
        mu_t.shape,
        device=mu_t.device,
        dtype=mu_t.dtype,
        generator=generator,
    )
    noise_scale = torch.sqrt(torch.clamp(curly_gamma(t_col), min=0.0)) * float(sigma)
    return mu_t + noise_scale * eps, mu_dot, t_col


def knn_reference_velocity(
    x: torch.Tensor,
    reference_x: torch.Tensor,
    reference_v: torch.Tensor,
    k: int = 20,
    eps: float = 1e-12,
) -> torch.Tensor:
    if x.ndim != 2 or reference_x.ndim != 2 or reference_v.ndim != 2:
        raise ValueError(
            "x, reference_x, and reference_v must all have shape (N, d); "
            f"got {tuple(x.shape)}, {tuple(reference_x.shape)}, {tuple(reference_v.shape)}."
        )
    if reference_x.shape != reference_v.shape:
        raise ValueError(
            "reference_x and reference_v must have the same shape, got "
            f"{tuple(reference_x.shape)} and {tuple(reference_v.shape)}."
        )
    if x.shape[1] != reference_x.shape[1]:
        raise ValueError(
            f"Dimension mismatch: x has d={x.shape[1]}, reference has d={reference_x.shape[1]}."
        )
    if reference_x.shape[0] <= 0:
        raise ValueError("Curly reference pool is empty.")
    k_eff = min(int(k), int(reference_x.shape[0]))
    if k_eff <= 0:
        raise ValueError(f"curly.reference_k must be positive, got {k}")
    dists = torch.cdist(x, reference_x)
    knn_dists, knn_idx = torch.topk(dists, k=k_eff, dim=1, largest=False)
    bandwidth = knn_dists[:, -1:].clamp_min(float(eps))
    weights = torch.exp(-(knn_dists**2) / (2.0 * bandwidth**2))
    weights = weights / (weights.sum(dim=1, keepdim=True) + float(eps))
    velocity_neighbors = reference_v[knn_idx]
    return (weights.unsqueeze(-1) * velocity_neighbors).sum(dim=1)


def curly_drift_alignment_loss(
    path_velocity: torch.Tensor,
    reference_velocity: torch.Tensor,
    velocity_scale: float,
    cosine_weight: float,
    l2_weight: float,
    l2_mu_dot_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    scaled_reference = float(velocity_scale) * reference_velocity
    cosine_loss = 1.0 - F.cosine_similarity(scaled_reference, path_velocity, dim=1).mean()
    l2_loss = torch.mean((scaled_reference - float(l2_mu_dot_scale) * path_velocity) ** 2)
    loss = float(cosine_weight) * cosine_loss + float(l2_weight) * l2_loss
    stats = {
        "curly_loss": float(loss.detach().item()),
        "curly_cosine_loss": float(cosine_loss.detach().item()),
        "curly_l2_loss": float(l2_loss.detach().item()),
        "curly_reference_speed": float(
            torch.linalg.norm(scaled_reference.detach(), dim=1).mean().item()
        ),
        "curly_path_speed": float(torch.linalg.norm(path_velocity.detach(), dim=1).mean().item()),
    }
    return loss, stats


def _coupling_times(
    num_times: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if int(num_times) <= 1:
        return torch.rand((1,), device=device, dtype=dtype, generator=generator)
    return torch.linspace(0.0, 1.0, int(num_times), device=device, dtype=dtype)


def curly_learned_coupling(
    x0: torch.Tensor,
    x1: torch.Tensor,
    geopath_net: torch.nn.Module,
    reference_x: torch.Tensor,
    reference_v: torch.Tensor,
    *,
    k: int,
    path_alpha: float,
    sigma: float,
    velocity_scale: float,
    num_times: int = 1,
    chunk_size: int = 32,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if x0.shape != x1.shape:
        raise ValueError(f"x0 and x1 must have the same shape, got {tuple(x0.shape)} and {tuple(x1.shape)}")
    if x0.ndim != 2:
        raise ValueError(f"Expected endpoint batches with shape (B, d), got {tuple(x0.shape)}")
    batch_size, dim = x0.shape
    if batch_size <= 0:
        raise ValueError("Cannot compute Curly learned coupling for an empty batch.")
    chunk = max(1, min(int(chunk_size), batch_size))
    costs = torch.zeros((batch_size, batch_size), device=x0.device, dtype=x0.dtype)
    times = _coupling_times(
        num_times=int(num_times),
        device=x0.device,
        dtype=x0.dtype,
        generator=generator,
    )

    for t_scalar in times:
        for start in range(0, batch_size, chunk):
            end = min(batch_size, start + chunk)
            n_target = end - start
            x1_chunk = x1[start:end]
            x1_flat = x1_chunk[:, None, :].expand(n_target, batch_size, dim).reshape(-1, dim)
            x0_flat = x0[None, :, :].expand(n_target, batch_size, dim).reshape(-1, dim)
            t_flat = torch.full(
                (x0_flat.shape[0], 1),
                float(t_scalar.detach().item()),
                device=x0.device,
                dtype=x0.dtype,
            )
            with torch.enable_grad():
                xt, mu_dot, _ = curly_path_and_velocity(
                    t=t_flat,
                    x0=x0_flat,
                    x1=x1_flat,
                    geopath_net=geopath_net,
                    path_alpha=float(path_alpha),
                    sigma=float(sigma),
                    create_graph=False,
                    generator=generator,
                )
            reference_dot = knn_reference_velocity(
                x=xt.detach(),
                reference_x=reference_x,
                reference_v=reference_v,
                k=int(k),
            )
            chunk_cost = 0.5 * torch.sum(
                (mu_dot.detach() - float(velocity_scale) * reference_dot) ** 2,
                dim=1,
            )
            costs[start:end, :] += chunk_cost.reshape(n_target, batch_size)

    target_ind, source_ind = linear_sum_assignment(costs.detach().cpu().numpy())
    target_idx = torch.as_tensor(target_ind, device=x1.device, dtype=torch.long)
    source_idx = torch.as_tensor(source_ind, device=x0.device, dtype=torch.long)
    selected_cost = costs[target_idx, source_idx].sum().detach().item()
    return x0[source_idx], x1[target_idx], float(selected_cost)


def select_reference_times(policy: str, available_times: Iterable[float]) -> tuple[float, ...]:
    normalized = normalize_reference_pool_policy(policy)
    times = sorted({float(t) for t in available_times})
    if not times:
        raise ValueError("Cannot build a Curly reference pool without observed times.")
    if normalized == "all_marginals":
        return tuple(times)
    endpoints = [times[0], times[-1]]
    if normalized == "endpoints_only" or len(times) <= 2:
        return tuple(endpoints)
    middle = min(times, key=lambda value: abs(float(value) - 0.5))
    selected = sorted({*endpoints, float(middle)})
    return tuple(selected)
