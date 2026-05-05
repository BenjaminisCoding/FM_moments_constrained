from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import torch


def _as_time_column(
    t: torch.Tensor | None,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool = False,
) -> torch.Tensor:
    if t is None:
        out = torch.rand((batch_size, 1), device=device, dtype=dtype)
    else:
        out = t.to(device=device, dtype=dtype)
        if out.ndim == 1:
            out = out.unsqueeze(1)
        if out.ndim == 0:
            out = out.reshape(1, 1).expand(batch_size, 1)
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


def mfm_gamma(
    t: torch.Tensor,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> torch.Tensor:
    return (
        1.0
        - ((t - t_min) / (t_max - t_min)) ** 2
        - ((t_max - t) / (t_max - t_min)) ** 2
    )


def mfm_d_gamma(
    t: torch.Tensor,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> torch.Tensor:
    return 2.0 * (-2.0 * t + t_max + t_min) / ((t_max - t_min) ** 2)


def mfm_mean_path(
    t: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
    geopath_net: torch.nn.Module | None,
    alpha: float,
) -> torch.Tensor:
    linear = (1.0 - t) * x0 + t * x1
    if geopath_net is None or float(alpha) == 0.0:
        return linear
    geopath = geopath_net(t, x0, x1)
    return linear + float(alpha) * mfm_gamma(t) * geopath


def mfm_path_and_velocity(
    t: torch.Tensor | None,
    x0: torch.Tensor,
    x1: torch.Tensor,
    geopath_net: torch.nn.Module | None,
    alpha: float,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t_col = _as_time_column(
        t=t,
        batch_size=x0.shape[0],
        device=x0.device,
        dtype=x0.dtype,
        requires_grad=(geopath_net is not None and float(alpha) != 0.0),
    )
    linear = (1.0 - t_col) * x0 + t_col * x1
    base_velocity = x1 - x0
    if geopath_net is None or float(alpha) == 0.0:
        return linear, base_velocity, t_col

    geopath = geopath_net(t_col, x0, x1)
    d_geopath_dt = _vector_time_derivative(geopath, t_col, create_graph=create_graph)
    gamma_t = mfm_gamma(t_col)
    d_gamma_t = mfm_d_gamma(t_col)
    mu_t = linear + float(alpha) * gamma_t * geopath
    ut = base_velocity + float(alpha) * (d_gamma_t * geopath + gamma_t * d_geopath_dt)
    return mu_t, ut, t_col


def land_metric_tensor(
    x: torch.Tensor,
    samples: torch.Tensor,
    gamma: float,
    rho: float,
) -> torch.Tensor:
    if gamma <= 0.0:
        raise ValueError(f"LAND gamma must be positive, got {gamma}")
    if rho < 0.0:
        raise ValueError(f"LAND rho must be non-negative, got {rho}")
    pairwise_sq_diff = (x[:, None, :] - samples[None, :, :]) ** 2
    pairwise_sq_dist = pairwise_sq_diff.sum(dim=-1)
    weights = torch.exp(-pairwise_sq_dist / (2.0 * (gamma**2)))
    differences = samples[None, :, :] - x[:, None, :]
    weighted_sq = torch.einsum("bn,bnd->bd", weights, differences**2)
    metric_diag_inv = 1.0 / (weighted_sq + float(rho))
    return metric_diag_inv


def land_geopath_loss(
    x_t: torch.Tensor,
    u_t: torch.Tensor,
    manifold_samples: torch.Tensor,
    gamma: float,
    rho: float,
) -> torch.Tensor:
    metric_diag_inv = land_metric_tensor(
        x=x_t,
        samples=manifold_samples,
        gamma=gamma,
        rho=rho,
    )
    vel_sq = torch.sum((u_t**2) * metric_diag_inv, dim=1)
    return torch.mean(vel_sq)


def _torchcfm_installed() -> bool:
    try:
        import torchcfm.conditional_flow_matching  # noqa: F401
    except Exception:
        return False
    return True


def resolve_mfm_backend(requested: str) -> str:
    lowered = str(requested).lower().strip()
    if lowered not in {"auto", "native", "torchcfm"}:
        raise ValueError(
            f"Unsupported mfm.backend '{requested}'. Expected one of: auto, native, torchcfm."
        )
    if lowered == "native":
        return "native"
    if lowered == "torchcfm":
        if not _torchcfm_installed():
            raise RuntimeError("mfm.backend=torchcfm requested but torchcfm is not installed.")
        return "torchcfm"
    return "torchcfm" if _torchcfm_installed() else "native"


@dataclass
class MetricBackend:
    name: str
    impl: str
    matcher: Any | None
    sigma: float
    alpha: float
    geopath_net: torch.nn.Module | None

    def sample_location_and_conditional_flow(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor | None = None,
        create_graph: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.name == "native":
            mu_t, u_t, t_col = mfm_path_and_velocity(
                t=t,
                x0=x0,
                x1=x1,
                geopath_net=self.geopath_net,
                alpha=self.alpha,
                create_graph=create_graph,
            )
            if self.sigma > 0.0:
                x_t = mu_t + self.sigma * torch.randn_like(x0)
            else:
                x_t = mu_t
            return t_col, x_t, u_t
        if self.matcher is None:
            raise RuntimeError("TorchCFM backend requested but matcher is not initialized.")
        t_raw = None if t is None else t.reshape(-1)
        t_out, x_t, u_t = self.matcher.sample_location_and_conditional_flow(
            x0=x0,
            x1=x1,
            t_min=0.0,
            t_max=1.0,
            t=t_raw,
        )
        if t_out.ndim == 1:
            t_out = t_out.unsqueeze(1)
        return t_out, x_t, u_t


def _maybe_add_authors_repo_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[2] / "metric-flow-matching"
    if repo_root.exists() and str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _build_torchcfm_matcher(
    geopath_net: torch.nn.Module | None,
    sigma: float,
    alpha: float,
) -> tuple[Any, str]:
    if geopath_net is None and float(alpha) != 0.0:
        raise ValueError("TorchCFM metric matcher requires geopath_net when alpha != 0.")
    _maybe_add_authors_repo_to_path()
    wrapped_geopath = None
    if geopath_net is not None:
        wrapped_geopath = _TorchCFMGeopathWrapper(geopath_net)
    try:
        from mfm.flow_matchers.models.mfm import MetricFlowMatcher as AuthorsMetricFlowMatcher

        return (
            AuthorsMetricFlowMatcher(
                geopath_net=wrapped_geopath,
                sigma=float(sigma),
                alpha=float(alpha),
            ),
            "authors",
        )
    except Exception:
        pass
    return (
        _InternalTorchCFMMetricFlowMatcher(
            geopath_net=wrapped_geopath,
            sigma=float(sigma),
            alpha=float(alpha),
        ),
        "internal",
    )


class _TorchCFMGeopathWrapper(torch.nn.Module):
    def __init__(self, path_model: torch.nn.Module) -> None:
        super().__init__()
        self.path_model = path_model
        self.time_geopath = True

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(1)
        return self.path_model(t, x0, x1)


class _InternalTorchCFMMetricFlowMatcher(torch.nn.Module):
    def __init__(
        self,
        geopath_net: torch.nn.Module | None = None,
        sigma: float = 0.1,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

        self._base = ConditionalFlowMatcher(sigma=float(sigma))
        self.geopath_net = geopath_net
        self.alpha = float(alpha)
        if self.alpha != 0.0 and self.geopath_net is None:
            raise ValueError("geopath_net must be provided when alpha != 0.")

    @staticmethod
    def _pad_t_like_x(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        from torchcfm.conditional_flow_matching import pad_t_like_x

        return pad_t_like_x(t, x)

    @staticmethod
    def _doutput_dt_fun(
        model: torch.nn.Module,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t_raw: torch.Tensor,
    ) -> torch.Tensor:
        from torch.func import jvp

        def f(tt: torch.Tensor) -> torch.Tensor:
            t_col = _InternalTorchCFMMetricFlowMatcher._pad_t_like_x(tt, x0)
            return model(x0, x1, t_col)

        _, dydt = jvp(f, (t_raw,), (torch.ones_like(t_raw),))
        return dydt.squeeze(-1)

    def sample_noise_like(self, x: torch.Tensor) -> torch.Tensor:
        return self._base.sample_noise_like(x)

    def compute_sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        return self._base.compute_sigma_t(t)

    def _compute_mu_t(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        t_min: float,
        t_max: float,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        t_col = self._pad_t_like_x(t, x0)
        linear = (t_max - t_col) / (t_max - t_min) * x0 + (t_col - t_min) / (t_max - t_min) * x1
        if self.alpha == 0.0 or self.geopath_net is None:
            return linear, None, None
        geopath = self.geopath_net(x0, x1, t_col)
        d_geopath_dt = self._doutput_dt_fun(self.geopath_net, x0, x1, t)
        gamma_t = mfm_gamma(t_col, t_min=t_min, t_max=t_max)
        mu_t = linear + self.alpha * gamma_t * geopath
        return mu_t, geopath, d_geopath_dt

    def sample_location_and_conditional_flow(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t_min: float = 0.0,
        t_max: float = 1.0,
        t: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.enable_grad():
            if t is None:
                t = torch.rand(x0.shape[0], requires_grad=True, device=x0.device, dtype=x0.dtype)
            else:
                t = t.to(device=x0.device, dtype=x0.dtype)
                if t.ndim > 1:
                    t = t.reshape(-1)
                t = t.detach().clone().requires_grad_(True)
            t = t * (t_max - t_min) + t_min
            eps = self.sample_noise_like(x0)
            mu_t, geopath, d_geopath_dt = self._compute_mu_t(
                x0=x0,
                x1=x1,
                t=t,
                t_min=t_min,
                t_max=t_max,
            )
            sigma_t = self._pad_t_like_x(self.compute_sigma_t(t), x0)
            x_t = mu_t + sigma_t * eps

            t_col = self._pad_t_like_x(t, x0)
            if self.alpha == 0.0 or geopath is None:
                u_t = (x1 - x0) / (t_max - t_min)
            else:
                d_gamma_t = mfm_d_gamma(t_col, t_min=t_min, t_max=t_max)
                gamma_t = mfm_gamma(t_col, t_min=t_min, t_max=t_max)
                u_t = (x1 - x0) / (t_max - t_min) + self.alpha * (
                    d_gamma_t * geopath + gamma_t * d_geopath_dt
                )
        return t, x_t, u_t


def build_metric_backend(
    requested_backend: str,
    geopath_net: torch.nn.Module | None,
    sigma: float,
    alpha: float,
) -> MetricBackend:
    selected = resolve_mfm_backend(requested_backend)
    if selected == "native":
        return MetricBackend(
            name="native",
            impl="native",
            matcher=None,
            sigma=float(sigma),
            alpha=float(alpha),
            geopath_net=geopath_net,
        )
    try:
        matcher, impl = _build_torchcfm_matcher(
            geopath_net=geopath_net,
            sigma=sigma,
            alpha=alpha,
        )
    except Exception as exc:
        if str(requested_backend).lower().strip() == "auto":
            return MetricBackend(
                name="native",
                impl="native_fallback",
                matcher=None,
                sigma=float(sigma),
                alpha=float(alpha),
                geopath_net=geopath_net,
            )
        raise RuntimeError(f"Failed to initialize torchcfm backend: {exc}") from exc
    return MetricBackend(
        name="torchcfm",
        impl=impl,
        matcher=matcher,
        sigma=float(sigma),
        alpha=float(alpha),
        geopath_net=geopath_net,
    )
