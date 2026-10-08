from __future__ import annotations

import math
from typing import Any, Callable, Mapping

import torch


MOMENT_DEFAULT_BLOCKS = ("mean", "covariance")
MOMENT_BLOCK_ORDER = (
    "mean",
    "covariance",
    "y_variance",
    "y_fourth_central",
    "local_y_variance",
)
MOMENT_BLOCK_NORMALIZATIONS = ("none", "sqrt_dim")


def normalize_moment_feature_blocks(raw_blocks: object | None) -> tuple[str, ...]:
    if raw_blocks is None:
        return MOMENT_DEFAULT_BLOCKS
    if isinstance(raw_blocks, str):
        blocks = [raw_blocks]
    elif isinstance(raw_blocks, (list, tuple)):
        blocks = [str(block) for block in raw_blocks]
    else:
        raise ValueError(
            "moment feature blocks must be a string or list/tuple of strings, "
            f"got {type(raw_blocks)}."
        )
    normalized: list[str] = []
    seen: set[str] = set()
    for block in blocks:
        key = str(block).strip().lower()
        if key == "cov":
            key = "covariance"
        if key in {"y_var", "var_y", "variance_y"}:
            key = "y_variance"
        if key in {"kurtosis", "y_kurtosis", "y_fourth", "fourth_central"}:
            key = "y_fourth_central"
        if key in {"local_y_var", "localized_y_variance", "localised_y_variance"}:
            key = "local_y_variance"
        if key not in MOMENT_BLOCK_ORDER:
            raise ValueError(
                f"Unsupported moment feature block '{block}'. "
                f"Expected one of: {', '.join(MOMENT_BLOCK_ORDER)}."
            )
        if key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    if not normalized:
        raise ValueError("At least one moment feature block must be active.")
    ordered = [block for block in MOMENT_BLOCK_ORDER if block in seen]
    return tuple(ordered)


def normalize_moment_block_normalization(raw: object | None) -> str:
    value = "none" if raw is None else str(raw).strip().lower()
    if value not in MOMENT_BLOCK_NORMALIZATIONS:
        raise ValueError(
            f"Unsupported moment block normalization '{raw}'. "
            f"Expected one of: {', '.join(MOMENT_BLOCK_NORMALIZATIONS)}."
        )
    return value


def covariance_feature(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"Expected x with shape (N, d), got {tuple(x.shape)}")
    if x.shape[0] <= 0:
        raise ValueError("covariance_feature requires at least one sample.")
    mean = x.mean(dim=0)
    centered = x - mean
    return centered.T @ centered / x.shape[0]


def y_fourth_central_feature(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError(
            f"Expected x with shape (N, d>=2) for y fourth central moment, got {tuple(x.shape)}"
        )
    y = x[:, 1]
    centered = y - y.mean()
    return torch.mean(centered**4).reshape(1)


def _as_mapping(raw: object | None) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return raw
    raise ValueError(f"moment_feature_params must be a mapping when provided, got {type(raw)}.")


def _local_y_variance_params(feature_params: object | None) -> tuple[float, float, float]:
    params = _as_mapping(feature_params)
    local_raw = params.get("local_y_variance", params)
    local = _as_mapping(local_raw)
    center_x = float(local.get("center_x", local.get("bridge_center_x", 1.0)))
    width = float(local.get("width", local.get("bridge_width", 0.35)))
    eps = float(local.get("eps", 1e-8))
    if width <= 0.0:
        raise ValueError(f"local_y_variance width must be positive, got {width}.")
    if eps <= 0.0:
        raise ValueError(f"local_y_variance eps must be positive, got {eps}.")
    return center_x, width, eps


def _affine_moment_coordinates(
    mean: torch.Tensor,
    cov: torch.Tensor,
    feature_params: object | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optionally express mean/covariance features in fixed affine coordinates.

    The path samples and geometric objective remain in their original coordinates.
    Only the reported moment feature map changes to ``y = A (x - c)``.  This is
    useful when constraints should be standardized without changing the path
    geometry itself.
    """
    params = _as_mapping(feature_params)
    affine_raw = params.get("affine_moment_coordinates", None)
    if affine_raw is None:
        return mean, cov
    affine = _as_mapping(affine_raw)
    if "matrix" not in affine:
        raise ValueError("affine_moment_coordinates requires a 'matrix'.")
    matrix = torch.as_tensor(affine["matrix"], device=mean.device, dtype=mean.dtype)
    dim = int(mean.numel())
    if tuple(matrix.shape) != (dim, dim):
        raise ValueError(
            "affine_moment_coordinates.matrix must have shape "
            f"({dim}, {dim}), got {tuple(matrix.shape)}."
        )
    center = torch.as_tensor(
        affine.get("center", torch.zeros_like(mean)),
        device=mean.device,
        dtype=mean.dtype,
    ).reshape(-1)
    if int(center.numel()) != dim:
        raise ValueError(
            "affine_moment_coordinates.center must have length "
            f"{dim}, got {int(center.numel())}."
        )
    if not torch.isfinite(matrix).all() or not torch.isfinite(center).all():
        raise ValueError("Affine moment coordinates must be finite.")
    transformed_mean = matrix @ (mean - center)
    transformed_cov = matrix @ cov @ matrix.T
    return transformed_mean, transformed_cov


def _moment_feature_block_scale(feature_params: object | None, block: str) -> float:
    params = _as_mapping(feature_params)
    scales_raw = params.get("moment_block_scales", None)
    if scales_raw is None:
        return 1.0
    scales = _as_mapping(scales_raw)
    value = float(scales.get(block, 1.0))
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"moment_block_scales.{block} must be finite and positive, got {value}."
        )
    return value


def local_y_variance_feature(
    x: torch.Tensor,
    feature_params: object | None = None,
) -> torch.Tensor:
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(
            f"Expected x with shape (N, 2) for local_y_variance, got {tuple(x.shape)}"
        )
    center_x, width, eps = _local_y_variance_params(feature_params)
    center = torch.as_tensor(center_x, device=x.device, dtype=x.dtype)
    width_tensor = torch.as_tensor(width, device=x.device, dtype=x.dtype)
    gate = torch.exp(-0.5 * ((x[:, 0] - center) / width_tensor) ** 2)
    denom = torch.clamp(gate.mean(), min=torch.as_tensor(eps, device=x.device, dtype=x.dtype))
    local_mean_y = torch.mean(gate * x[:, 1]) / denom
    local_var_y = torch.mean(gate * (x[:, 1] - local_mean_y) ** 2) / denom
    return local_var_y.reshape(1)


def moment_feature_blocks_from_mean_cov(
    mean: torch.Tensor,
    cov: torch.Tensor,
    feature_blocks: object | None = None,
    feature_params: object | None = None,
) -> dict[str, torch.Tensor]:
    active_blocks = normalize_moment_feature_blocks(feature_blocks)
    mean, cov = _affine_moment_coordinates(
        mean=mean,
        cov=cov,
        feature_params=feature_params,
    )
    out: dict[str, torch.Tensor] = {}
    if "mean" in active_blocks:
        out["mean"] = mean * _moment_feature_block_scale(feature_params, "mean")
    if "covariance" in active_blocks:
        out["covariance"] = cov.reshape(-1) * _moment_feature_block_scale(
            feature_params, "covariance"
        )
    if "y_variance" in active_blocks:
        if cov.ndim != 2 or cov.shape[0] < 2 or cov.shape[1] < 2:
            raise ValueError(f"Expected covariance with shape (d>=2, d>=2), got {tuple(cov.shape)}")
        out["y_variance"] = cov[1, 1].reshape(1) * _moment_feature_block_scale(
            feature_params, "y_variance"
        )
    if "local_y_variance" in active_blocks:
        raise ValueError(
            "local_y_variance targets require sample-based moment features; "
            "closed-form mean/covariance features are not sufficient."
        )
    if "y_fourth_central" in active_blocks:
        raise ValueError(
            "y_fourth_central targets require sample-based moment features; "
            "closed-form mean/covariance features are not sufficient."
        )
    return out


def moment_feature_blocks(
    x: torch.Tensor,
    feature_blocks: object | None = None,
    feature_params: object | None = None,
) -> dict[str, torch.Tensor]:
    if x.ndim != 2:
        raise ValueError(f"Expected x with shape (N, d), got {tuple(x.shape)}")
    if x.shape[0] <= 0:
        raise ValueError("moment_features requires at least one sample.")
    active_blocks = normalize_moment_feature_blocks(feature_blocks)
    mean = x.mean(dim=0)
    cov = covariance_feature(x)
    mean_cov_blocks = [
        block for block in active_blocks if block in {"mean", "covariance", "y_variance"}
    ]
    out: dict[str, torch.Tensor] = {}
    if mean_cov_blocks:
        out.update(
            moment_feature_blocks_from_mean_cov(
                mean=mean,
                cov=cov,
                feature_blocks=mean_cov_blocks,
                feature_params=feature_params,
            )
        )
    if "local_y_variance" in active_blocks:
        out["local_y_variance"] = local_y_variance_feature(
            x=x,
            feature_params=feature_params,
        )
    if "y_fourth_central" in active_blocks:
        out["y_fourth_central"] = y_fourth_central_feature(x=x)
    return out


def concatenate_moment_feature_blocks(
    blocks: dict[str, torch.Tensor],
    feature_blocks: object | None = None,
) -> torch.Tensor:
    active_blocks = normalize_moment_feature_blocks(feature_blocks)
    pieces = [blocks[block].reshape(-1) for block in active_blocks]
    return torch.cat(pieces, dim=0)


def moment_features(
    x: torch.Tensor,
    feature_blocks: object | None = None,
    feature_params: object | None = None,
) -> torch.Tensor:
    blocks = moment_feature_blocks(
        x=x,
        feature_blocks=feature_blocks,
        feature_params=feature_params,
    )
    return concatenate_moment_feature_blocks(blocks=blocks, feature_blocks=feature_blocks)


def moment_block_sizes(dim: int, feature_blocks: object | None = None) -> dict[str, int]:
    if int(dim) <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    active_blocks = normalize_moment_feature_blocks(feature_blocks)
    sizes: dict[str, int] = {}
    if "mean" in active_blocks:
        sizes["mean"] = int(dim)
    if "covariance" in active_blocks:
        sizes["covariance"] = int(dim) * int(dim)
    if "y_variance" in active_blocks:
        if int(dim) < 2:
            raise ValueError(f"y_variance requires dim >= 2, got {dim}")
        sizes["y_variance"] = 1
    if "y_fourth_central" in active_blocks:
        if int(dim) < 2:
            raise ValueError(f"y_fourth_central requires dim >= 2, got {dim}")
        sizes["y_fourth_central"] = 1
    if "local_y_variance" in active_blocks:
        if int(dim) != 2:
            raise ValueError(f"local_y_variance requires dim == 2, got {dim}")
        sizes["local_y_variance"] = 1
    return sizes


def split_moment_feature_vector(
    feature: torch.Tensor,
    dim: int,
    feature_blocks: object | None = None,
) -> dict[str, torch.Tensor]:
    flat = feature.reshape(-1)
    sizes = moment_block_sizes(dim=dim, feature_blocks=feature_blocks)
    expected = int(sum(sizes.values()))
    if int(flat.numel()) != expected:
        raise ValueError(
            "Moment feature vector size mismatch for configured blocks. "
            f"Expected {expected}, got {int(flat.numel())}."
        )
    out: dict[str, torch.Tensor] = {}
    start = 0
    for block in normalize_moment_feature_blocks(feature_blocks):
        size = int(sizes[block])
        out[block] = flat[start : start + size]
        start += size
    return out


def select_moment_feature_blocks(
    feature: torch.Tensor,
    dim: int,
    source_blocks: object | None,
    selected_blocks: object | None,
) -> torch.Tensor:
    source_active = normalize_moment_feature_blocks(source_blocks)
    selected_active = normalize_moment_feature_blocks(selected_blocks)
    source = split_moment_feature_vector(
        feature=feature,
        dim=dim,
        feature_blocks=source_active,
    )
    missing = [block for block in selected_active if block not in source]
    if missing:
        raise ValueError(
            "Cannot select moment feature blocks absent from source vector: "
            + ", ".join(missing)
        )
    return concatenate_moment_feature_blocks(
        blocks={block: source[block] for block in selected_active},
        feature_blocks=selected_active,
    )


def moment_block_scale(block: str, dim: int, normalization: str) -> float:
    normalized = normalize_moment_block_normalization(normalization)
    if normalized == "none":
        return 1.0
    sizes = moment_block_sizes(dim=dim, feature_blocks=[block])
    return 1.0 / math.sqrt(float(sizes[block]))


def normalize_residual_blocks(
    residuals: dict[float, dict[str, torch.Tensor]],
    dim: int,
    normalization: str,
) -> dict[float, dict[str, torch.Tensor]]:
    normalize_moment_block_normalization(normalization)
    return {
        float(t): {
            block: res * moment_block_scale(block=block, dim=dim, normalization=normalization)
            for block, res in block_residuals.items()
        }
        for t, block_residuals in residuals.items()
    }


def residual_from_samples(
    x: torch.Tensor,
    target_feature: torch.Tensor,
    feature_blocks: object | None = None,
    feature_params: object | None = None,
) -> torch.Tensor:
    return moment_features(
        x,
        feature_blocks=feature_blocks,
        feature_params=feature_params,
    ) - target_feature


def residual_blocks_from_samples(
    x: torch.Tensor,
    target_feature: torch.Tensor,
    feature_blocks: object | None = None,
    feature_params: object | None = None,
) -> dict[str, torch.Tensor]:
    active_blocks = normalize_moment_feature_blocks(feature_blocks)
    sample_blocks = moment_feature_blocks(
        x=x,
        feature_blocks=active_blocks,
        feature_params=feature_params,
    )
    target_blocks = split_moment_feature_vector(
        feature=target_feature,
        dim=int(x.shape[1]),
        feature_blocks=active_blocks,
    )
    return {block: sample_blocks[block] - target_blocks[block] for block in active_blocks}


def constraint_residuals(
    path_fn: Callable[[float], torch.Tensor],
    times: list[float],
    targets: dict[float, torch.Tensor],
    feature_blocks: object | None = None,
    feature_params: object | None = None,
) -> dict[float, torch.Tensor]:
    residuals: dict[float, torch.Tensor] = {}
    for t in times:
        xt = path_fn(float(t))
        residuals[float(t)] = residual_from_samples(
            xt,
            targets[float(t)],
            feature_blocks=feature_blocks,
            feature_params=feature_params,
        )
    return residuals


def constraint_residual_blocks(
    path_fn: Callable[[float], torch.Tensor],
    times: list[float],
    targets: dict[float, torch.Tensor],
    feature_blocks: object | None = None,
    feature_params: object | None = None,
) -> dict[float, dict[str, torch.Tensor]]:
    residuals: dict[float, dict[str, torch.Tensor]] = {}
    for t in times:
        xt = path_fn(float(t))
        residuals[float(t)] = residual_blocks_from_samples(
            xt,
            targets[float(t)],
            feature_blocks=feature_blocks,
            feature_params=feature_params,
        )
    return residuals


def residual_norms(residuals: dict[float, torch.Tensor]) -> dict[float, float]:
    return {float(t): float(torch.linalg.norm(res).item()) for t, res in residuals.items()}


def block_residual_norms(
    residuals: dict[float, dict[str, torch.Tensor]],
) -> dict[str, dict[float, float]]:
    out: dict[str, dict[float, float]] = {}
    for t, block_residuals in residuals.items():
        for block, res in block_residuals.items():
            out.setdefault(block, {})[float(t)] = float(torch.linalg.norm(res).item())
    return out


def augmented_lagrangian_terms(
    residuals: dict[float, torch.Tensor],
    lambdas: dict[float, torch.Tensor],
    rho: float,
) -> tuple[torch.Tensor, dict[float, float]]:
    total = torch.zeros((), device=next(iter(residuals.values())).device)
    per_time: dict[float, float] = {}
    for t, res in residuals.items():
        lam = lambdas[float(t)]
        term = torch.dot(lam, res) + 0.5 * rho * torch.dot(res, res)
        total = total + term
        per_time[float(t)] = float(term.detach().item())
    return total, per_time


def augmented_lagrangian_block_terms(
    residuals: dict[float, dict[str, torch.Tensor]],
    lambdas: dict[float, dict[str, torch.Tensor]],
    rho: float,
) -> tuple[torch.Tensor, dict[float, float], dict[str, dict[float, float]]]:
    first_time = next(iter(residuals.values()))
    first_residual = next(iter(first_time.values()))
    total = torch.zeros((), device=first_residual.device, dtype=first_residual.dtype)
    per_time: dict[float, float] = {}
    per_block: dict[str, dict[float, float]] = {}
    for t, block_residuals in residuals.items():
        time_total = torch.zeros((), device=first_residual.device, dtype=first_residual.dtype)
        for block, res in block_residuals.items():
            lam = lambdas[float(t)][block]
            term = torch.dot(lam, res) + 0.5 * rho * torch.dot(res, res)
            time_total = time_total + term
            per_block.setdefault(block, {})[float(t)] = float(term.detach().item())
        time_total = time_total / float(len(block_residuals))
        total = total + time_total
        per_time[float(t)] = float(time_total.detach().item())
    return total, per_time, per_block


def update_lagrange_multipliers(
    lambdas: dict[float, torch.Tensor],
    residuals: dict[float, torch.Tensor],
    rho: float,
    clip_value: float | None = None,
) -> dict[float, torch.Tensor]:
    updated: dict[float, torch.Tensor] = {}
    for t, lam in lambdas.items():
        new_lam = lam + rho * residuals[float(t)].detach()
        if clip_value is not None:
            new_lam = torch.clamp(new_lam, -clip_value, clip_value)
        updated[float(t)] = new_lam
    return updated


def update_lagrange_multiplier_blocks(
    lambdas: dict[float, dict[str, torch.Tensor]],
    residuals: dict[float, dict[str, torch.Tensor]],
    rho: float,
    clip_value: float | None = None,
) -> dict[float, dict[str, torch.Tensor]]:
    updated: dict[float, dict[str, torch.Tensor]] = {}
    for t, block_lambdas in lambdas.items():
        updated[float(t)] = {}
        for block, lam in block_lambdas.items():
            new_lam = lam + rho * residuals[float(t)][block].detach()
            if clip_value is not None:
                new_lam = torch.clamp(new_lam, -clip_value, clip_value)
            updated[float(t)][block] = new_lam
    return updated
