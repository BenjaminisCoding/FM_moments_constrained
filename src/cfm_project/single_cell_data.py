from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from cfm_project.single_cell_utils import sha256_array, sort_unique_labels

import numpy as np
import torch

from cfm_project.data import (
    EmpiricalCouplingProblem,
    exact_discrete_ot_indices,
    moment_feature_vector_from_samples,
)
from cfm_project.ot_utils import solve_balanced_ot_lp
from cfm_project.pseudo_labels import prepare_pseudo_labels
from cfm_project.classifier_moments import empirical_feature_mean


@dataclass
class SingleCellPreparedData:
    problem: EmpiricalCouplingProblem
    targets: dict[float, torch.Tensor]
    pseudo_targets: dict[float, torch.Tensor] | None
    target_samples_by_time: dict[float, torch.Tensor]
    velocity_samples_by_time: dict[float, torch.Tensor] | None
    target_sampler: Callable[[float, int, torch.Generator | None], torch.Tensor]
    pseudo_posterior: Callable[[torch.Tensor], torch.Tensor] | None
    velocity_source_key: str | None
    velocity_feature_std: list[float] | None
    all_time_indices: list[int]
    all_time_labels: list[str]
    normalized_times_all: list[float]
    constraint_time_indices: list[int]
    constraint_times: list[float]
    eval_times: list[float]
    holdout_index: int | None
    holdout_time: float | None
    protocol: str
    constraint_time_policy: str
    global_ot_cache_path: str | None
    global_ot_cache_hit: bool
    global_ot_support_size: int | None
    global_ot_total_cost: float | None
    global_ot_solve_seconds: float | None
    pseudo_labels_k: int | None
    pseudo_labels_method: str | None
    pseudo_labels_cache_path: str | None
    pseudo_labels_cache_hit: bool
    pseudo_labels_bic_by_k: dict[int, float] | None
    pseudo_labels_stability_by_k: dict[int, float] | None
    pseudo_posterior_temperature: float | None
    pseudo_fit_times: list[float] | None
    pseudo_fit_sample_count: int | None
    pseudo_constraint_class_labels: list[str] | None
    pseudo_constraint_class_mode: str | None
    pseudo_constraint_dim: int | None
    pseudo_constraint_feature_mode: str | None
    moment_target_source: str
    pseudo_target_source: str | None
    pseudo_supervised_kept_class_labels: list[str] | None
    pseudo_supervised_dropped_class_labels: list[str] | None
    pseudo_supervised_train_count: int | None
    pseudo_supervised_val_count: int | None
    pseudo_supervised_val_split_used: bool | None
    pseudo_supervised_val_split_fallback: bool | None
    pseudo_supervised_epochs_trained: int | None
    pseudo_supervised_best_epoch: int | None
    pseudo_supervised_best_val_loss: float | None
    pseudo_supervised_early_stop_triggered: bool | None
    pseudo_gmm_weights: torch.Tensor | None = None
    pseudo_gmm_means: torch.Tensor | None = None
    pseudo_gmm_covariances: torch.Tensor | None = None
    pseudo_target_kind: str | None = None


def _normalize_time(index: int, n_times: int) -> float:
    if n_times <= 1:
        return 0.0
    return float(index) / float(n_times - 1)


def _as_1d_labels(labels: np.ndarray) -> np.ndarray:
    if labels.ndim == 1:
        return labels
    if labels.ndim == 2 and labels.shape[1] == 1:
        return labels.reshape(-1)
    raise ValueError(f"Expected labels as shape (N,) or (N, 1), got {labels.shape}")


def _infer_npz_velocity_key(embed_key: str) -> str | None:
    if embed_key == "pcs":
        return "pcs_delta"
    if embed_key == "phate":
        return "delta_embedding"
    return None


def _load_npz_dataset(
    path: str,
    cfg: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], str | None]:
    data = np.load(path, allow_pickle=True)
    embed_key = str(cfg.get("embed_key_npz", "pcs"))
    label_key = str(cfg.get("label_key_npz", "sample_labels"))
    if embed_key not in data:
        raise KeyError(f"Missing key '{embed_key}' in NPZ dataset.")
    if label_key not in data:
        raise KeyError(f"Missing key '{label_key}' in NPZ dataset.")
    features = np.asarray(data[embed_key])
    labels = _as_1d_labels(np.asarray(data[label_key]))
    velocity_key_raw = cfg.get("velocity_key_npz", None)
    velocity_key = (
        _infer_npz_velocity_key(embed_key)
        if velocity_key_raw is None
        else str(velocity_key_raw).strip()
    )
    extra_arrays: dict[str, np.ndarray] = {}
    velocity_source_key: str | None = None
    if velocity_key:
        if velocity_key in data:
            extra_arrays["__velocity__"] = np.asarray(data[velocity_key])
            velocity_source_key = velocity_key
        elif velocity_key_raw is not None:
            raise KeyError(f"Missing velocity key '{velocity_key}' in NPZ dataset.")
    return features, labels, extra_arrays, velocity_source_key


def _read_h5ad(path: str) -> Any:
    try:
        import anndata as ad
    except ImportError:
        ad = None
    if ad is not None:
        return ad.read_h5ad(path)
    try:
        import scanpy as sc
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Loading .h5ad datasets requires anndata or scanpy. "
            "Install one of these packages to use single-cell h5ad inputs."
        ) from exc
    return sc.read_h5ad(path)


def _load_h5ad_dataset(
    path: str,
    cfg: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], str | None]:
    adata = _read_h5ad(path)
    embed_key = str(cfg.get("embed_key_h5ad", "X_pca"))
    label_key = str(cfg.get("label_key_h5ad", "day"))
    if embed_key not in adata.obsm:
        raise KeyError(f"Missing embedding '{embed_key}' in adata.obsm.")
    if label_key not in adata.obs:
        raise KeyError(f"Missing label column '{label_key}' in adata.obs.")
    features = np.asarray(adata.obsm[embed_key])
    labels = _as_1d_labels(np.asarray(adata.obs[label_key]))
    pseudo_cfg = cfg.get("pseudo_labels", {})
    supervised_label_key = str(pseudo_cfg.get("supervised_label_key", "cell_sets"))
    extra_obs_keys = {label_key, supervised_label_key}
    obs_columns: dict[str, np.ndarray] = {}
    for key in extra_obs_keys:
        if key in adata.obs.columns:
            obs_columns[key] = _as_1d_labels(np.asarray(adata.obs[key]))
    velocity_source_key: str | None = None
    velocity_key_raw = cfg.get("velocity_key_h5ad", None)
    if velocity_key_raw is not None:
        velocity_key = str(velocity_key_raw).strip()
        if velocity_key not in adata.obsm:
            raise KeyError(f"Missing velocity embedding '{velocity_key}' in adata.obsm.")
        obs_columns["__velocity__"] = np.asarray(adata.obsm[velocity_key])
        velocity_source_key = velocity_key
    return features, labels, obs_columns, velocity_source_key


def _load_single_cell_dataset(
    data_cfg: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], str | None]:
    single_cfg = data_cfg.get("single_cell", {})
    path = str(single_cfg.get("path", "")).strip()
    if not path:
        raise ValueError("data.single_cell.path must be provided for data.family=single_cell.")
    if path.endswith(".npz"):
        return _load_npz_dataset(path=path, cfg=single_cfg)
    if path.endswith(".h5ad"):
        return _load_h5ad_dataset(path=path, cfg=single_cfg)
    raise ValueError(
        f"Unsupported single-cell dataset format for path '{path}'. "
        "Expected .npz or .h5ad."
    )


def _parse_include_labels(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, (str, bytes)):
        return [raw.decode("utf-8") if isinstance(raw, bytes) else raw]
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return [raw]


def _parse_string_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (str, bytes)):
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        stripped = text.strip()
        if not stripped or stripped.lower() in {"none", "null", "[]"}:
            return []
        if "," in stripped:
            return [part.strip() for part in stripped.split(",") if part.strip()]
        return [stripped]
    if isinstance(raw, (list, tuple)):
        return [str(value) for value in raw]
    try:
        values = list(raw)
    except TypeError:
        return [str(raw)]
    return [str(value) for value in values]


def _project_pseudo_vector(
    values: torch.Tensor,
    indices: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if values.ndim == 2:
        selected = values.index_select(dim=1, index=indices)
        if mode == "sum":
            return selected.sum(dim=1, keepdim=True)
        return selected
    if values.ndim == 1:
        selected = values.index_select(dim=0, index=indices)
        if mode == "sum":
            return selected.sum().reshape(1)
        return selected
    raise ValueError(
        "Pseudo-label projection expects a vector or matrix, "
        f"got tensor with shape {tuple(values.shape)}."
    )


def _label_filter_mask(labels: np.ndarray, include_labels: list[Any]) -> np.ndarray:
    if not include_labels:
        return np.ones(labels.shape[0], dtype=bool)
    include_str = {str(label) for label in include_labels}
    include_float: list[float] = []
    for label in include_labels:
        try:
            include_float.append(float(label))
        except (TypeError, ValueError):
            pass

    mask = np.zeros(labels.shape[0], dtype=bool)
    for idx, label in enumerate(labels.tolist()):
        if str(label) in include_str:
            mask[idx] = True
            continue
        try:
            label_float = float(label)
        except (TypeError, ValueError):
            continue
        if any(abs(label_float - allowed) <= 1.0e-8 for allowed in include_float):
            mask[idx] = True
    return mask


def _whiten_features_and_velocity(
    features: np.ndarray,
    velocity: np.ndarray | None,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True)
    std = np.maximum(std, float(eps))
    features_out = (features - mean) / std
    velocity_out = None if velocity is None else velocity / std
    return features_out, velocity_out, std.reshape(-1)


def _sample_from_pool(
    pool: torch.Tensor,
    n_samples: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if pool.shape[0] <= 0:
        raise ValueError("Cannot sample from an empty pool.")
    idx = torch.randint(
        low=0,
        high=pool.shape[0],
        size=(n_samples,),
        device=pool.device,
        generator=generator,
    )
    return pool[idx]


def _nearest_time_key(available_times: list[float], t: float, tol: float = 1e-6) -> float:
    best = min(available_times, key=lambda value: abs(float(value) - float(t)))
    if abs(float(best) - float(t)) > tol:
        raise ValueError(
            f"Requested time {float(t):.6f} is not available. "
            f"Nearest available={float(best):.6f}, all={available_times}"
        )
    return float(best)


def _parse_normalized_times(raw: Any, field_name: str) -> list[float]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        values = raw
    else:
        raise ValueError(f"{field_name} must be a list of normalized times in [0, 1], got {type(raw)}.")
    out: list[float] = []
    for value in values:
        parsed = float(value)
        if parsed < 0.0 or parsed > 1.0:
            raise ValueError(f"{field_name} entries must be in [0, 1], got {parsed}.")
        out.append(parsed)
    return out


def _normalized_time_values(
    *,
    raw: Any,
    n_times: int,
) -> list[float]:
    if raw is None:
        return [_normalize_time(idx, n_times) for idx in range(n_times)]
    values = _parse_normalized_times(
        raw,
        field_name="data.single_cell.normalized_time_values",
    )
    if len(values) != int(n_times):
        raise ValueError(
            "data.single_cell.normalized_time_values must have one entry per "
            f"sorted observed label, got {len(values)} entries for {n_times} labels."
        )
    if any(right <= left for left, right in zip(values[:-1], values[1:])):
        raise ValueError(
            "data.single_cell.normalized_time_values must be strictly increasing, "
            f"got {values}."
        )
    if not np.isclose(values[0], 0.0, atol=1.0e-8) or not np.isclose(
        values[-1], 1.0, atol=1.0e-8
    ):
        raise ValueError(
            "data.single_cell.normalized_time_values must start at 0 and end at 1, "
            f"got {values}."
        )
    return values


def _resolve_time_indices_from_normalized(
    *,
    requested_times: list[float],
    normalized_time_by_index: Mapping[int, float],
    field_name: str,
    tol: float = 1.0e-6,
) -> list[int]:
    if not requested_times:
        return []
    available = {int(idx): float(value) for idx, value in normalized_time_by_index.items()}
    resolved: list[int] = []
    for requested in requested_times:
        matches = [
            (idx, value, abs(float(value) - float(requested)))
            for idx, value in available.items()
            if abs(float(value) - float(requested)) <= float(tol)
        ]
        if not matches:
            raise ValueError(
                f"{field_name} requested time {float(requested):.6f} is not observed in dataset times. "
                f"Available normalized times={sorted(available.values())}."
            )
        matches.sort(key=lambda item: (item[2], item[1], item[0]))
        idx = int(matches[0][0])
        if idx not in resolved:
            resolved.append(idx)
    resolved.sort(key=lambda idx: available[idx])
    return resolved


def _resolve_holdout_index(
    protocol: str,
    holdout_index: int | None,
    holdout_indices: list[int],
    n_times: int,
) -> int | None:
    if protocol == "no_leaveout":
        return None
    if n_times < 3:
        raise ValueError(
            f"Strict leaveout requires at least 3 timepoints, got {n_times}."
        )
    if holdout_index is None:
        if holdout_indices:
            holdout_index = int(holdout_indices[0])
        else:
            holdout_index = int((n_times - 1) // 2)
            if holdout_index <= 0 or holdout_index >= n_times - 1:
                holdout_index = 1
    resolved = int(holdout_index)
    if resolved <= 0 or resolved >= n_times - 1:
        raise ValueError(
            f"experiment.holdout_index must be a middle index in [1, {n_times - 2}], got {resolved}."
        )
    return resolved


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ot_cache_root(single_cfg: Mapping[str, Any]) -> Path:
    configured = str(single_cfg.get("global_ot_cache_dir", ".cache/ot_plans")).strip()
    root = Path(configured)
    if not root.is_absolute():
        root = _project_root() / root
    return root.resolve()


def _global_ot_cache_signature(
    *,
    single_cfg: Mapping[str, Any],
    data_cfg: Mapping[str, Any],
    dtype: torch.dtype,
    sorted_labels: list[Any],
    x0_pool: torch.Tensor,
    x1_pool: torch.Tensor,
) -> dict[str, Any]:
    dataset_path = Path(str(single_cfg.get("path", ""))).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Single-cell dataset path does not exist: {dataset_path}")
    stat = dataset_path.stat()
    x0_np = np.asarray(x0_pool.detach().cpu(), dtype=np.float32)
    x1_np = np.asarray(x1_pool.detach().cpu(), dtype=np.float32)
    signature: dict[str, Any] = {
        "schema": "single_cell_global_balanced_ot_plan_v2",
        "data_label": str(data_cfg.get("label", "single_cell")),
        "dataset_path": str(dataset_path),
        "dataset_size_bytes": int(stat.st_size),
        "dataset_mtime_ns": int(stat.st_mtime_ns),
        "whiten": bool(single_cfg.get("whiten", True)),
        "max_dim": int(single_cfg.get("max_dim", x0_np.shape[1])),
        "embed_key_npz": str(single_cfg.get("embed_key_npz", "pcs")),
        "label_key_npz": str(single_cfg.get("label_key_npz", "sample_labels")),
        "embed_key_h5ad": str(single_cfg.get("embed_key_h5ad", "X_pca")),
        "label_key_h5ad": str(single_cfg.get("label_key_h5ad", "day")),
        "endpoint_label_start": str(sorted_labels[0]),
        "endpoint_label_end": str(sorted_labels[-1]),
        "endpoint_count_start": int(x0_np.shape[0]),
        "endpoint_count_end": int(x1_np.shape[0]),
        "feature_dim": int(x0_np.shape[1]),
        "dtype": str(dtype),
        "x0_hash": sha256_array(x0_np),
        "x1_hash": sha256_array(x1_np),
        "solver": (
            "linear_sum_assignment"
            if tuple(x0_np.shape) == tuple(x1_np.shape)
            else "scipy_balanced_lp"
        ),
    }
    return signature


def _global_ot_cache_key(signature: Mapping[str, Any]) -> str:
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_or_build_global_ot_support(
    *,
    single_cfg: Mapping[str, Any],
    data_cfg: Mapping[str, Any],
    dtype: torch.dtype,
    sorted_labels: list[Any],
    x0_pool: torch.Tensor,
    x1_pool: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, str | None, bool, float | None]:
    cache_enabled = bool(single_cfg.get("global_ot_cache_enabled", True))
    force_recompute = bool(single_cfg.get("global_ot_force_recompute", False))
    support_tol = float(single_cfg.get("global_ot_support_tol", 1e-12))
    max_variables_raw = single_cfg.get("global_ot_max_variables", None)
    max_variables = None if max_variables_raw is None else int(max_variables_raw)

    signature = _global_ot_cache_signature(
        single_cfg=single_cfg,
        data_cfg=data_cfg,
        dtype=dtype,
        sorted_labels=sorted_labels,
        x0_pool=x0_pool,
        x1_pool=x1_pool,
    )
    cache_key = _global_ot_cache_key(signature)
    cache_path: Path | None = None
    if cache_enabled:
        cache_root = _ot_cache_root(single_cfg)
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_path = cache_root / f"{cache_key}.pt"

    if cache_enabled and cache_path is not None and cache_path.exists() and not force_recompute:
        payload = torch.load(cache_path, map_location="cpu")
        payload_signature = payload.get("signature")
        if payload_signature != signature:
            raise RuntimeError(
                "Global OT cache signature mismatch for existing cache file. "
                f"Delete {cache_path} or set data.single_cell.global_ot_force_recompute=true."
            )
        src_idx = torch.as_tensor(payload["src_idx"], dtype=torch.long)
        tgt_idx = torch.as_tensor(payload["tgt_idx"], dtype=torch.long)
        mass = torch.as_tensor(payload["mass"], dtype=torch.float64)
        total_cost = float(payload["total_cost"])
        mass_sum = float(mass.sum().item())
        if mass_sum <= 0.0:
            raise RuntimeError(f"Cached global OT mass is invalid in {cache_path}.")
        mass = mass / mass_sum
        solve_seconds = payload.get("solve_seconds", None)
        if solve_seconds is not None:
            solve_seconds = float(solve_seconds)
        return src_idx, tgt_idx, mass, total_cost, str(cache_path), True, solve_seconds

    start = time.perf_counter()
    if tuple(x0_pool.shape) == tuple(x1_pool.shape):
        src_idx, tgt_idx, summed_cost = exact_discrete_ot_indices(
            x0_pool.detach().cpu(),
            x1_pool.detach().cpu(),
        )
        mass = torch.full(
            (int(src_idx.numel()),),
            1.0 / float(src_idx.numel()),
            dtype=torch.float64,
        )
        total_cost = float(summed_cost) / float(src_idx.numel())
    else:
        plan = solve_balanced_ot_lp(
            x=x0_pool.detach().cpu(),
            y=x1_pool.detach().cpu(),
            src_weights=None,
            tgt_weights=None,
            support_tol=support_tol,
            max_variables=max_variables,
        )
        src_idx = torch.as_tensor(plan.src_idx, dtype=torch.long)
        tgt_idx = torch.as_tensor(plan.tgt_idx, dtype=torch.long)
        mass = torch.as_tensor(plan.mass, dtype=torch.float64)
        total_cost = float(plan.total_cost)
    solve_seconds = float(time.perf_counter() - start)
    mass = mass / torch.clamp(mass.sum(), min=torch.finfo(mass.dtype).eps)

    if cache_enabled and cache_path is not None:
        torch.save(
            {
                "signature": signature,
                "src_idx": src_idx,
                "tgt_idx": tgt_idx,
                "mass": mass,
                "total_cost": total_cost,
                "solve_seconds": float(solve_seconds),
            },
            cache_path,
        )
    return (
        src_idx,
        tgt_idx,
        mass,
        total_cost,
        (None if cache_path is None else str(cache_path)),
        False,
        solve_seconds,
    )


def prepare_single_cell_problem_and_targets(
    data_cfg: Mapping[str, Any],
    experiment_cfg: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> SingleCellPreparedData:
    single_cfg = data_cfg.get("single_cell", {})
    moment_feature_blocks = data_cfg.get("moment_feature_blocks", None)
    coupling = str(data_cfg.get("coupling", "ot")).strip().lower()
    features_np, labels_np, obs_columns, velocity_source_key = _load_single_cell_dataset(
        data_cfg=data_cfg
    )
    velocity_np = obs_columns.pop("__velocity__", None)
    if features_np.ndim != 2:
        raise ValueError(f"Expected feature matrix shape (N, d), got {features_np.shape}")
    if features_np.shape[0] != labels_np.shape[0]:
        raise ValueError(
            "Feature/label sample count mismatch: "
            f"{features_np.shape[0]} features vs {labels_np.shape[0]} labels."
        )
    if velocity_np is not None:
        if velocity_np.ndim != 2:
            raise ValueError(f"Expected velocity matrix shape (N, d), got {velocity_np.shape}")
        if velocity_np.shape[0] != features_np.shape[0]:
            raise ValueError(
                "Feature/velocity sample count mismatch: "
                f"{features_np.shape[0]} features vs {velocity_np.shape[0]} velocities."
            )
    include_labels = _parse_include_labels(single_cfg.get("include_labels", None))
    if include_labels:
        include_mask = _label_filter_mask(labels_np, include_labels)
        kept = int(include_mask.sum())
        if kept <= 0:
            raise ValueError(
                "data.single_cell.include_labels did not match any samples. "
                f"Requested={include_labels}, available={sort_unique_labels(labels_np)}."
            )
        features_np = features_np[include_mask]
        labels_np = labels_np[include_mask]
        if velocity_np is not None:
            velocity_np = velocity_np[include_mask]
        obs_columns = {
            key: (value[include_mask] if value.shape[0] == include_mask.shape[0] else value)
            for key, value in obs_columns.items()
        }
        if len(sort_unique_labels(labels_np)) < 2:
            raise ValueError(
                "data.single_cell.include_labels must retain at least two time labels, "
                f"got {sort_unique_labels(labels_np)}."
            )

    max_dim = int(single_cfg.get("max_dim", features_np.shape[1]))
    if max_dim <= 0:
        raise ValueError(f"data.single_cell.max_dim must be positive, got {max_dim}")
    features_np = features_np[:, :max_dim]
    if velocity_np is not None:
        if velocity_np.shape[1] < max_dim:
            raise ValueError(
                f"Velocity dimension {velocity_np.shape[1]} is smaller than max_dim={max_dim}."
            )
        velocity_np = velocity_np[:, :max_dim]
    velocity_feature_std: list[float] | None = None
    if bool(single_cfg.get("whiten", True)):
        features_np, velocity_np, std_np = _whiten_features_and_velocity(
            features=features_np,
            velocity=velocity_np,
        )
        if velocity_np is not None:
            velocity_feature_std = [float(v) for v in std_np.tolist()]

    expected_dim = int(data_cfg.get("dim", features_np.shape[1]))
    if int(features_np.shape[1]) != expected_dim:
        raise ValueError(
            f"Configured data.dim={expected_dim}, but loaded feature dimension is {features_np.shape[1]}. "
            "Align data.dim and single-cell max_dim/embed settings."
        )

    sorted_labels = sort_unique_labels(labels_np)
    if len(sorted_labels) < 2:
        raise ValueError(
            f"Single-cell benchmark requires at least 2 time labels, got {len(sorted_labels)}."
        )
    label_to_index = {label: idx for idx, label in enumerate(sorted_labels)}
    time_indices = np.array([label_to_index[label] for label in labels_np.tolist()], dtype=np.int64)
    n_times = len(sorted_labels)

    protocol = str(experiment_cfg.get("protocol", "strict_leaveout")).strip().lower()
    if protocol not in {"strict_leaveout", "no_leaveout"}:
        raise ValueError(
            f"Unsupported experiment.protocol '{protocol}'. "
            "Expected one of: strict_leaveout, no_leaveout."
        )
    raw_holdout_index = experiment_cfg.get("holdout_index", None)
    holdout_index = None if raw_holdout_index is None else int(raw_holdout_index)
    holdout_indices = [int(value) for value in experiment_cfg.get("holdout_indices", [])]
    holdout_index = _resolve_holdout_index(
        protocol=protocol,
        holdout_index=holdout_index,
        holdout_indices=holdout_indices,
        n_times=n_times,
    )

    constraint_policy = str(
        data_cfg.get(
            "constraint_time_policy",
            single_cfg.get("constraint_time_policy", "observed_nonendpoint_excluding_holdout"),
        )
    ).strip().lower()
    if constraint_policy not in {
        "observed_nonendpoint_excluding_holdout",
        "observed_nonendpoint_all",
    }:
        raise ValueError(
            "Unsupported data.constraint_time_policy "
            f"'{constraint_policy}'. Expected one of: "
            "observed_nonendpoint_excluding_holdout, observed_nonendpoint_all."
        )

    all_time_indices = list(range(n_times))
    intermediate_indices = list(range(1, n_times - 1))
    features = torch.tensor(features_np, dtype=dtype, device=device)
    velocities = None if velocity_np is None else torch.tensor(velocity_np, dtype=dtype, device=device)
    pools_by_index: dict[int, torch.Tensor] = {}
    velocity_pools_by_index: dict[int, torch.Tensor] | None = None
    if velocities is not None:
        velocity_pools_by_index = {}
    for idx in all_time_indices:
        mask = torch.as_tensor(time_indices == idx, device=device)
        pool = features[mask]
        if pool.shape[0] <= 0:
            raise ValueError(f"No samples found for time index {idx}.")
        pools_by_index[idx] = pool
        if velocity_pools_by_index is not None and velocities is not None:
            velocity_pool = velocities[mask]
            if velocity_pool.shape != pool.shape:
                raise ValueError(
                    "Velocity pool shape mismatch at time index "
                    f"{idx}: velocity={tuple(velocity_pool.shape)}, position={tuple(pool.shape)}."
                )
            velocity_pools_by_index[idx] = velocity_pool

    normalized_times_all = _normalized_time_values(
        raw=single_cfg.get("normalized_time_values", None),
        n_times=n_times,
    )
    normalized_time_by_index = {
        idx: float(normalized_times_all[idx]) for idx in all_time_indices
    }
    if constraint_policy == "observed_nonendpoint_excluding_holdout" and holdout_index is not None:
        constraint_time_indices = [idx for idx in intermediate_indices if idx != int(holdout_index)]
    else:
        constraint_time_indices = list(intermediate_indices)
    explicit_constraint_times = _parse_normalized_times(
        single_cfg.get("constraint_times_normalized", None),
        field_name="data.single_cell.constraint_times_normalized",
    )
    if explicit_constraint_times:
        constraint_time_indices = _resolve_time_indices_from_normalized(
            requested_times=explicit_constraint_times,
            normalized_time_by_index=normalized_time_by_index,
            field_name="data.single_cell.constraint_times_normalized",
        )
    if not constraint_time_indices:
        raise ValueError(
            "Resolved constraint times are empty. Adjust holdout index/policy or provide more timestamps."
        )

    eval_times_override = _parse_normalized_times(
        single_cfg.get("eval_times_normalized", None),
        field_name="data.single_cell.eval_times_normalized",
    )
    if eval_times_override:
        eval_time_indices = _resolve_time_indices_from_normalized(
            requested_times=eval_times_override,
            normalized_time_by_index=normalized_time_by_index,
            field_name="data.single_cell.eval_times_normalized",
        )
    else:
        eval_time_indices = sorted(set(int(idx) for idx in constraint_time_indices))
        if holdout_index is not None:
            eval_time_indices = sorted(set(eval_time_indices + [int(holdout_index)]))
    if not eval_time_indices:
        raise ValueError("Resolved eval times are empty after applying single-cell eval time settings.")
    eval_times = [float(normalized_time_by_index[idx]) for idx in eval_time_indices]
    target_samples_by_time = {
        float(normalized_time_by_index[idx]): pools_by_index[idx] for idx in all_time_indices
    }
    velocity_samples_by_time = (
        None
        if velocity_pools_by_index is None
        else {
            float(normalized_time_by_index[idx]): velocity_pools_by_index[idx]
            for idx in all_time_indices
        }
    )

    moment_feature_params = data_cfg.get("moment_feature_params", None)
    targets = {
        float(normalized_time_by_index[idx]): moment_feature_vector_from_samples(
            pools_by_index[idx],
            feature_blocks=moment_feature_blocks,
            feature_params=moment_feature_params,
        )
        for idx in constraint_time_indices
    }
    moment_target_source = "observed_constraint_marginal"
    raw_moment_target_overrides = data_cfg.get("moment_target_overrides", None)
    if raw_moment_target_overrides is not None:
        if not isinstance(raw_moment_target_overrides, Mapping):
            raise ValueError(
                "data.moment_target_overrides must be a mapping from normalized "
                "times to moment-feature target vectors."
            )
        moment_target_source = str(
            data_cfg.get("moment_target_override_source", "")
        ).strip()
        if not moment_target_source:
            raise ValueError(
                "Explicit moment target overrides require the non-empty provenance "
                "field data.moment_target_override_source."
            )
        override_targets: dict[float, torch.Tensor] = {}
        expected_dim = int(next(iter(targets.values())).numel())
        for raw_time, raw_value in raw_moment_target_overrides.items():
            try:
                time_value = float(raw_time)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Moment target override time must be numeric, got {raw_time!r}."
                ) from exc
            tensor = torch.as_tensor(raw_value, device=device, dtype=dtype).reshape(-1)
            if int(tensor.numel()) != expected_dim:
                raise ValueError(
                    "Moment target override dimension must match the configured moment "
                    f"features: {int(tensor.numel())} versus {expected_dim}."
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(
                    f"Moment target override at t={time_value:.6f} must be finite."
                )
            override_targets[time_value] = tensor
        expected_times = set(targets)
        missing_times = [
            time_value
            for time_value in sorted(expected_times)
            if not any(abs(time_value - key) <= 1.0e-8 for key in override_targets)
        ]
        extra_times = [
            key
            for key in sorted(override_targets)
            if not any(abs(key - time_value) <= 1.0e-8 for time_value in expected_times)
        ]
        if missing_times or extra_times:
            raise ValueError(
                "Moment target override times must match the configured constraint "
                f"times; missing={missing_times}, extra={extra_times}."
            )
        targets = {
            time_value: override_targets[
                next(
                    key
                    for key in override_targets
                    if abs(key - time_value) <= 1.0e-8
                )
            ].detach().clone()
            for time_value in sorted(expected_times)
        }
    pseudo_targets: dict[float, torch.Tensor] | None = None
    pseudo_posterior: Callable[[torch.Tensor], torch.Tensor] | None = None
    pseudo_labels_k: int | None = None
    pseudo_labels_cache_path: str | None = None
    pseudo_labels_cache_hit = False
    pseudo_labels_method: str | None = None
    pseudo_labels_bic_by_k: dict[int, float] | None = None
    pseudo_labels_stability_by_k: dict[int, float] | None = None
    pseudo_posterior_temperature: float | None = None
    pseudo_supervised_kept_class_labels: list[str] | None = None
    pseudo_supervised_dropped_class_labels: list[str] | None = None
    pseudo_constraint_class_labels: list[str] | None = None
    pseudo_constraint_class_mode: str | None = None
    pseudo_constraint_dim: int | None = None
    pseudo_constraint_feature_mode: str | None = None
    pseudo_target_source: str | None = None
    pseudo_target_kind: str | None = None
    pseudo_supervised_train_count: int | None = None
    pseudo_supervised_val_count: int | None = None
    pseudo_supervised_val_split_used: bool | None = None
    pseudo_supervised_val_split_fallback: bool | None = None
    pseudo_supervised_epochs_trained: int | None = None
    pseudo_supervised_best_epoch: int | None = None
    pseudo_supervised_best_val_loss: float | None = None
    pseudo_supervised_early_stop_triggered: bool | None = None
    pseudo_gmm_weights: torch.Tensor | None = None
    pseudo_gmm_means: torch.Tensor | None = None
    pseudo_gmm_covariances: torch.Tensor | None = None
    pseudo_cfg = single_cfg.get("pseudo_labels", {})
    pseudo_method = str(pseudo_cfg.get("method", "gmm")).strip().lower()
    if pseudo_method not in {"gmm", "supervised_mlp", "supervised_logreg"}:
        raise ValueError(
            f"Unsupported data.single_cell.pseudo_labels.method '{pseudo_method}'. "
            "Expected one of: gmm, supervised_mlp, supervised_logreg."
        )
    supervised_label_key = str(pseudo_cfg.get("supervised_label_key", "cell_sets"))
    supervised_labels_all: np.ndarray | None = None
    if bool(pseudo_cfg.get("enabled", False)) and pseudo_method in {
        "supervised_mlp",
        "supervised_logreg",
    }:
        if supervised_label_key not in obs_columns:
            raise KeyError(
                "Supervised pseudo-label mode requires an observed label column in the loaded dataset. "
                f"Missing '{supervised_label_key}'."
            )
        supervised_labels_all = _as_1d_labels(np.asarray(obs_columns[supervised_label_key]))
        if supervised_labels_all.shape[0] != features_np.shape[0]:
            raise ValueError(
                "Supervised pseudo labels length mismatch: "
                f"{supervised_labels_all.shape[0]} labels vs {features_np.shape[0]} features."
            )
    pseudo_fit_times: list[float] | None = None
    pseudo_fit_sample_count: int | None = None
    supervised_labels_for_pseudo: np.ndarray | None = None
    pseudo_fit_indices = list(all_time_indices)
    pseudo_fit_override_times = _parse_normalized_times(
        pseudo_cfg.get("fit_times_normalized", None),
        field_name="data.single_cell.pseudo_labels.fit_times_normalized",
    )
    if bool(pseudo_cfg.get("enabled", False)):
        if pseudo_fit_override_times:
            pseudo_fit_indices = _resolve_time_indices_from_normalized(
                requested_times=pseudo_fit_override_times,
                normalized_time_by_index=normalized_time_by_index,
                field_name="data.single_cell.pseudo_labels.fit_times_normalized",
            )
        pseudo_fit_times = [float(normalized_time_by_index[idx]) for idx in pseudo_fit_indices]
        fit_index_array = np.asarray(pseudo_fit_indices, dtype=np.int64)
        fit_mask = np.isin(time_indices, fit_index_array)
        features_for_pseudo = np.asarray(features_np[fit_mask], dtype=np.float64)
        time_indices_for_pseudo = np.asarray(time_indices[fit_mask], dtype=np.int64)
        if supervised_labels_all is not None:
            supervised_labels_for_pseudo = _as_1d_labels(
                np.asarray(supervised_labels_all[fit_mask])
            )
        pseudo_fit_sample_count = int(features_for_pseudo.shape[0])
        if pseudo_fit_sample_count <= 0:
            raise ValueError(
                "Pseudo-label fit subset is empty. "
                "Check data.single_cell.pseudo_labels.fit_times_normalized."
            )
        if pseudo_method == "gmm":
            k_max = int(pseudo_cfg.get("k_max", 10))
            if pseudo_fit_sample_count < k_max:
                raise ValueError(
                    "Pseudo-label fit subset has too few samples: "
                    f"{pseudo_fit_sample_count} < k_max={k_max}. "
                    "Adjust fit_times_normalized or k_max."
                )
        if pseudo_method in {"supervised_mlp", "supervised_logreg"}:
            if supervised_labels_for_pseudo is None:
                raise ValueError(
                    "Supervised pseudo-label mode requires supervised labels for fit subset."
                )
            n_classes = int(np.unique(supervised_labels_for_pseudo).shape[0])
            if n_classes < 2:
                raise ValueError(
                    "Supervised pseudo-label mode requires at least 2 classes in fit subset, "
                    f"got {n_classes}."
                )
    else:
        features_for_pseudo = np.asarray(features_np, dtype=np.float64)
        time_indices_for_pseudo = np.asarray(time_indices, dtype=np.int64)
    pseudo_prepared = prepare_pseudo_labels(
        dataset_path=str(single_cfg.get("path", "")),
        features_np=features_for_pseudo,
        time_indices=time_indices_for_pseudo,
        supervised_labels_np=supervised_labels_for_pseudo,
        single_cfg=single_cfg,
        device=device,
        dtype=dtype,
    )
    if pseudo_prepared is not None:
        full_pseudo_posterior = pseudo_prepared.posterior
        pseudo_posterior = full_pseudo_posterior
        pseudo_labels_k = int(pseudo_prepared.selected_k)
        pseudo_labels_method = str(pseudo_prepared.method)
        pseudo_labels_cache_path = pseudo_prepared.cache_path
        pseudo_labels_cache_hit = bool(pseudo_prepared.cache_hit)
        pseudo_labels_bic_by_k = {
            int(k): float(v) for k, v in pseudo_prepared.bic_by_k.items()
        }
        pseudo_labels_stability_by_k = {
            int(k): float(v) for k, v in pseudo_prepared.stability_by_k.items()
        }
        pseudo_posterior_temperature = float(pseudo_prepared.posterior_temperature)
        pseudo_gmm_weights = pseudo_prepared.gmm_weights
        pseudo_gmm_means = pseudo_prepared.gmm_means
        pseudo_gmm_covariances = pseudo_prepared.gmm_covariances
        pseudo_supervised_kept_class_labels = (
            None
            if pseudo_prepared.supervised_kept_class_labels is None
            else [str(label) for label in pseudo_prepared.supervised_kept_class_labels]
        )
        pseudo_supervised_dropped_class_labels = (
            None
            if pseudo_prepared.supervised_dropped_class_labels is None
            else [str(label) for label in pseudo_prepared.supervised_dropped_class_labels]
        )
        pseudo_supervised_train_count = (
            None
            if pseudo_prepared.supervised_train_count is None
            else int(pseudo_prepared.supervised_train_count)
        )
        pseudo_supervised_val_count = (
            None
            if pseudo_prepared.supervised_val_count is None
            else int(pseudo_prepared.supervised_val_count)
        )
        pseudo_supervised_val_split_used = (
            None
            if pseudo_prepared.supervised_val_split_used is None
            else bool(pseudo_prepared.supervised_val_split_used)
        )
        pseudo_supervised_val_split_fallback = (
            None
            if pseudo_prepared.supervised_val_split_fallback is None
            else bool(pseudo_prepared.supervised_val_split_fallback)
        )
        pseudo_supervised_epochs_trained = (
            None
            if pseudo_prepared.supervised_epochs_trained is None
            else int(pseudo_prepared.supervised_epochs_trained)
        )
        pseudo_supervised_best_epoch = (
            None
            if pseudo_prepared.supervised_best_epoch is None
            else int(pseudo_prepared.supervised_best_epoch)
        )
        pseudo_supervised_best_val_loss = (
            None
            if pseudo_prepared.supervised_best_val_loss is None
            else float(pseudo_prepared.supervised_best_val_loss)
        )
        pseudo_supervised_early_stop_triggered = (
            None
            if pseudo_prepared.supervised_early_stop_triggered is None
            else bool(pseudo_prepared.supervised_early_stop_triggered)
        )
        requested_constraint_classes = _parse_string_list(
            pseudo_cfg.get("constraint_classes", None)
        )
        pseudo_constraint_class_mode = "all"
        pseudo_constraint_class_labels = (
            None
            if pseudo_prepared.class_labels is None
            else [str(label) for label in pseudo_prepared.class_labels]
        )
        pseudo_constraint_dim = int(pseudo_labels_k)
        pseudo_constraint_feature_mode = str(
            pseudo_cfg.get("constraint_feature_mode", "class_probability")
        ).strip().lower()
        if pseudo_constraint_feature_mode not in {
            "class_probability",
            "class_probability_and_state_first_moment",
            "class_probability_and_random_fourier",
        }:
            raise ValueError(
                "data.single_cell.pseudo_labels.constraint_feature_mode must be "
                "'class_probability' or "
                "'class_probability_and_state_first_moment' or "
                "'class_probability_and_random_fourier'."
            )
        constraint_indices: torch.Tensor | None = None
        constraint_mode = str(
            pseudo_cfg.get("constraint_class_mode", "independent")
        ).strip().lower()
        if constraint_mode not in {"independent", "sum"}:
            raise ValueError(
                "data.single_cell.pseudo_labels.constraint_class_mode must be "
                "'independent' or 'sum'."
            )
        if requested_constraint_classes:
            if pseudo_prepared.class_labels is None:
                raise ValueError(
                    "Pseudo-label class filtering requires class label metadata. "
                    "Use supervised pseudo labels or leave constraint_classes unset."
                )
            class_to_index = {
                str(label): i for i, label in enumerate(pseudo_prepared.class_labels)
            }
            missing_classes = [
                label for label in requested_constraint_classes if label not in class_to_index
            ]
            if missing_classes:
                raise ValueError(
                    "Requested pseudo-label constraint class(es) are not available: "
                    f"{missing_classes}. Available classes: {list(class_to_index)}."
                )
            constraint_indices = torch.as_tensor(
                [int(class_to_index[label]) for label in requested_constraint_classes],
                device=device,
                dtype=torch.long,
            )
            pseudo_constraint_class_mode = constraint_mode
            pseudo_constraint_class_labels = (
                ["+".join(requested_constraint_classes)]
                if constraint_mode == "sum"
                else [str(label) for label in requested_constraint_classes]
            )
            pseudo_constraint_dim = (
                1 if constraint_mode == "sum" else len(requested_constraint_classes)
            )

            def projected_pseudo_posterior(
                x: torch.Tensor,
                *,
                _posterior: Callable[[torch.Tensor], torch.Tensor] = full_pseudo_posterior,
                _indices: torch.Tensor = constraint_indices,
                _mode: str = constraint_mode,
            ) -> torch.Tensor:
                return _project_pseudo_vector(_posterior(x), _indices, _mode)

            pseudo_posterior = projected_pseudo_posterior
        if pseudo_constraint_feature_mode == (
            "class_probability_and_state_first_moment"
        ):
            if pseudo_posterior is None or pseudo_constraint_dim is None:
                raise RuntimeError("Classifier posterior is unavailable for feature expansion.")
            probability_posterior = pseudo_posterior
            probability_dim = int(pseudo_constraint_dim)

            def probability_and_state_first_moment(
                x: torch.Tensor,
                *,
                _posterior: Callable[[torch.Tensor], torch.Tensor] = probability_posterior,
            ) -> torch.Tensor:
                probabilities = _posterior(x)
                weighted_state = probabilities[:, :, None] * x[:, None, :]
                return torch.cat(
                    [probabilities, weighted_state.reshape(x.shape[0], -1)], dim=1
                )

            pseudo_posterior = probability_and_state_first_moment
            pseudo_constraint_dim = probability_dim * (int(features.shape[1]) + 1)
        elif pseudo_constraint_feature_mode == "class_probability_and_random_fourier":
            if pseudo_posterior is None or pseudo_constraint_dim is None:
                raise RuntimeError("Classifier posterior is unavailable for RFF expansion.")
            raw_matrix = pseudo_cfg.get("random_fourier_matrix", None)
            raw_phase = pseudo_cfg.get("random_fourier_phase", None)
            if raw_matrix is None or raw_phase is None:
                raise ValueError(
                    "class_probability_and_random_fourier requires "
                    "random_fourier_matrix and random_fourier_phase."
                )
            rff_matrix = torch.as_tensor(raw_matrix, device=device, dtype=dtype)
            rff_phase = torch.as_tensor(raw_phase, device=device, dtype=dtype).reshape(-1)
            if rff_matrix.ndim != 2 or int(rff_matrix.shape[1]) != int(features.shape[1]):
                raise ValueError(
                    "random_fourier_matrix must have shape (M, state_dim), got "
                    f"{tuple(rff_matrix.shape)} for state_dim={int(features.shape[1])}."
                )
            if int(rff_phase.numel()) != int(rff_matrix.shape[0]):
                raise ValueError(
                    "random_fourier_phase length must match the matrix row count."
                )
            if not torch.isfinite(rff_matrix).all() or not torch.isfinite(rff_phase).all():
                raise ValueError("Random Fourier parameters must be finite.")
            rff_scale = float(
                pseudo_cfg.get(
                    "random_fourier_scale",
                    (2.0 / max(int(rff_matrix.shape[0]), 1)) ** 0.5,
                )
            )
            if not np.isfinite(rff_scale) or rff_scale <= 0.0:
                raise ValueError("random_fourier_scale must be finite and positive.")
            probability_posterior = pseudo_posterior
            probability_dim = int(pseudo_constraint_dim)

            def probability_and_random_fourier(
                x: torch.Tensor,
                *,
                _posterior: Callable[[torch.Tensor], torch.Tensor] = probability_posterior,
                _matrix: torch.Tensor = rff_matrix,
                _phase: torch.Tensor = rff_phase,
                _scale: float = rff_scale,
            ) -> torch.Tensor:
                probabilities = _posterior(x)
                rff = float(_scale) * torch.cos(x @ _matrix.T + _phase)
                return torch.cat([probabilities, rff], dim=1)

            pseudo_posterior = probability_and_random_fourier
            pseudo_constraint_dim = probability_dim + int(rff_matrix.shape[0])
        pseudo_targets = {}
        pseudo_target_kind = str(pseudo_cfg.get("target_kind", "auto")).strip().lower()
        if pseudo_target_kind not in {"auto", "label_proportions", "posterior_mean"}:
            raise ValueError(
                "data.single_cell.pseudo_labels.target_kind must be "
                "'auto', 'label_proportions', or 'posterior_mean'."
            )
        if pseudo_target_kind == "auto":
            # Label-frequency targets are supported. Paper Multi preparation selects
            # posterior_mean: label counts and class-weighted soft outputs differ.
            pseudo_target_kind = (
                "label_proportions"
                if pseudo_prepared.method in {"supervised_mlp", "supervised_logreg"}
                else "posterior_mean"
            )
        if pseudo_target_kind == "label_proportions" and pseudo_prepared.class_labels is None:
            raise ValueError("label_proportions targets require a supervised classifier.")
        raw_target_overrides = pseudo_cfg.get("target_overrides", None)
        if (
            pseudo_constraint_feature_mode
            in {
                "class_probability_and_state_first_moment",
                "class_probability_and_random_fourier",
            }
            and raw_target_overrides is None
        ):
            raise ValueError(
                "class_probability_and_state_first_moment constraints require explicit "
                "provenance-tagged target_overrides."
            )
        target_overrides: dict[float, torch.Tensor] | None = None
        if raw_target_overrides is not None:
            pseudo_target_kind = "provided"
            if not isinstance(raw_target_overrides, Mapping):
                raise ValueError(
                    "data.single_cell.pseudo_labels.target_overrides must be a mapping "
                    "from normalized times to target vectors."
                )
            pseudo_target_source = str(
                pseudo_cfg.get("target_override_source", "")
            ).strip()
            if not pseudo_target_source:
                raise ValueError(
                    "Explicit pseudo target overrides require the non-empty provenance "
                    "field data.single_cell.pseudo_labels.target_override_source."
                )
            target_overrides = {}
            for raw_time, raw_value in raw_target_overrides.items():
                try:
                    time_value = float(raw_time)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Pseudo target override time must be numeric, got {raw_time!r}."
                    ) from exc
                tensor = torch.as_tensor(raw_value, device=device, dtype=dtype).reshape(-1)
                if int(tensor.numel()) != int(pseudo_constraint_dim):
                    raise ValueError(
                        "Pseudo target override dimension must match the active classifier "
                        f"constraint: {int(tensor.numel())} versus {int(pseudo_constraint_dim)}."
                    )
                nonnegative_required = (
                    pseudo_constraint_feature_mode == "class_probability"
                )
                if not torch.isfinite(tensor).all() or (
                    nonnegative_required and torch.any(tensor < 0.0)
                ):
                    raise ValueError(
                        f"Pseudo target override at t={time_value:.6f} must be finite"
                        + (" and nonnegative." if nonnegative_required else ".")
                    )
                target_overrides[time_value] = tensor
            expected_times = {
                float(normalized_time_by_index[idx]) for idx in constraint_time_indices
            }
            missing_times = [
                time_value
                for time_value in sorted(expected_times)
                if not any(abs(time_value - key) <= 1.0e-8 for key in target_overrides)
            ]
            extra_times = [
                key
                for key in sorted(target_overrides)
                if not any(abs(key - time_value) <= 1.0e-8 for time_value in expected_times)
            ]
            if missing_times or extra_times:
                raise ValueError(
                    "Pseudo target override times must match the configured constraint times; "
                    f"missing={missing_times}, extra={extra_times}."
                )
        else:
            pseudo_target_source = (
                "observed_constraint_posterior_mean"
                if pseudo_target_kind == "posterior_mean"
                else "observed_constraint_marginal"
            )
        with torch.no_grad():
            for idx in constraint_time_indices:
                t_key = float(normalized_time_by_index[idx])
                if target_overrides is not None:
                    matched_key = next(
                        key for key in target_overrides if abs(key - t_key) <= 1.0e-8
                    )
                    pseudo_targets[t_key] = target_overrides[matched_key].detach().clone()
                    continue
                if pseudo_target_kind == "label_proportions":
                    if supervised_labels_all is None:
                        raise ValueError(
                            "Supervised pseudo-label mode requires labels for all samples."
                        )
                    if pseudo_prepared.class_labels is None:
                        raise ValueError(
                            "Supervised pseudo-label mode requires class label metadata."
                        )
                    labels_at_t = _as_1d_labels(np.asarray(supervised_labels_all[time_indices == idx]))
                    class_to_index = {
                        str(label): i for i, label in enumerate(pseudo_prepared.class_labels)
                    }
                    counts = np.zeros((pseudo_labels_k,), dtype=np.float64)
                    for label in labels_at_t.tolist():
                        key = str(label)
                        if key in class_to_index:
                            counts[class_to_index[key]] += 1.0
                    total = float(counts.sum())
                    if total <= 0.0:
                        raise ValueError(
                            "Supervised pseudo-label target has zero mass at "
                            f"time index {idx} (normalized t={t_key:.6f})."
                        )
                    full_target = torch.as_tensor(
                        counts / total,
                        device=device,
                        dtype=dtype,
                    )
                    if constraint_indices is not None:
                        pseudo_targets[t_key] = _project_pseudo_vector(
                            full_target,
                            constraint_indices,
                            constraint_mode,
                        )
                    else:
                        pseudo_targets[t_key] = full_target
                else:
                    pseudo_targets[t_key] = empirical_feature_mean(
                        pseudo_posterior, pools_by_index[idx]
                    ).to(device=device, dtype=dtype)

    available_times = sorted(target_samples_by_time.keys())

    def target_sampler(
        t: float,
        n_samples: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        matched = _nearest_time_key(available_times, t=float(t))
        pool = target_samples_by_time[matched]
        return _sample_from_pool(pool, n_samples=n_samples, generator=generator)

    x0_pool = pools_by_index[0]
    x1_pool = pools_by_index[n_times - 1]
    global_ot_src_idx: torch.Tensor | None = None
    global_ot_tgt_idx: torch.Tensor | None = None
    global_ot_mass: torch.Tensor | None = None
    global_ot_total_cost: float | None = None
    global_ot_cache_path: str | None = None
    global_ot_cache_hit = False
    global_ot_solve_seconds: float | None = None
    if coupling == "ot_global":
        (
            global_ot_src_idx,
            global_ot_tgt_idx,
            global_ot_mass,
            global_ot_total_cost,
            global_ot_cache_path,
            global_ot_cache_hit,
            global_ot_solve_seconds,
        ) = _load_or_build_global_ot_support(
            single_cfg=single_cfg,
            data_cfg=data_cfg,
            dtype=dtype,
            sorted_labels=sorted_labels,
            x0_pool=x0_pool,
            x1_pool=x1_pool,
        )
        global_ot_src_idx = global_ot_src_idx.to(device=device, dtype=torch.long)
        global_ot_tgt_idx = global_ot_tgt_idx.to(device=device, dtype=torch.long)
        global_ot_mass = global_ot_mass.to(device=device, dtype=dtype)
    elif coupling == "ot_persistent_minibatch":
        # Import lazily to keep the single-cell loader independent of the
        # multi-marginal orchestration module during module initialization.
        from cfm_project.multimarginal import build_persistent_minibatch_ot_proposal

        minibatch_size = int(
            single_cfg.get("persistent_minibatch_ot_batch_size", 512)
        )
        minibatch_passes = int(single_cfg.get("persistent_minibatch_ot_passes", 1))
        minibatch_seed = int(single_cfg.get("persistent_minibatch_ot_seed", 20260802))
        proposal_generator = torch.Generator(device=x0_pool.device).manual_seed(
            minibatch_seed
        )
        start = time.perf_counter()
        proposal = build_persistent_minibatch_ot_proposal(
            source=x0_pool,
            target=x1_pool,
            time=0.5,
            batch_size=minibatch_size,
            passes=minibatch_passes,
            generator=proposal_generator,
        )
        global_ot_src_idx = proposal.source_indices.to(
            device=device, dtype=torch.long
        )
        global_ot_tgt_idx = proposal.target_indices.to(
            device=device, dtype=torch.long
        )
        global_ot_mass = torch.full(
            (int(global_ot_src_idx.numel()),),
            1.0 / float(global_ot_src_idx.numel()),
            device=device,
            dtype=dtype,
        )
        global_ot_total_cost = float(proposal.mean_pair_squared_distance)
        global_ot_solve_seconds = float(time.perf_counter() - start)

    problem = EmpiricalCouplingProblem(
        x0_pool=x0_pool,
        x1_pool=x1_pool,
        label=str(data_cfg.get("label", "single_cell")),
        global_ot_src_idx=global_ot_src_idx,
        global_ot_tgt_idx=global_ot_tgt_idx,
        global_ot_mass=global_ot_mass,
        global_ot_total_cost=global_ot_total_cost,
    )
    holdout_time = (
        None
        if holdout_index is None
        else float(normalized_time_by_index[int(holdout_index)])
    )
    return SingleCellPreparedData(
        problem=problem,
        targets=targets,
        pseudo_targets=pseudo_targets,
        target_samples_by_time=target_samples_by_time,
        velocity_samples_by_time=velocity_samples_by_time,
        target_sampler=target_sampler,
        pseudo_posterior=pseudo_posterior,
        velocity_source_key=velocity_source_key,
        velocity_feature_std=velocity_feature_std,
        all_time_indices=all_time_indices,
        all_time_labels=[str(label) for label in sorted_labels],
        normalized_times_all=[float(value) for value in normalized_times_all],
        constraint_time_indices=[int(idx) for idx in constraint_time_indices],
        constraint_times=[float(normalized_time_by_index[idx]) for idx in constraint_time_indices],
        eval_times=[float(value) for value in eval_times],
        holdout_index=None if holdout_index is None else int(holdout_index),
        holdout_time=holdout_time,
        protocol=protocol,
        constraint_time_policy=constraint_policy,
        global_ot_cache_path=global_ot_cache_path,
        global_ot_cache_hit=bool(global_ot_cache_hit),
        global_ot_support_size=None if global_ot_mass is None else int(global_ot_mass.numel()),
        global_ot_total_cost=global_ot_total_cost,
        global_ot_solve_seconds=global_ot_solve_seconds,
        pseudo_labels_k=pseudo_labels_k,
        pseudo_labels_method=pseudo_labels_method,
        pseudo_labels_cache_path=pseudo_labels_cache_path,
        pseudo_labels_cache_hit=bool(pseudo_labels_cache_hit),
        pseudo_labels_bic_by_k=pseudo_labels_bic_by_k,
        pseudo_labels_stability_by_k=pseudo_labels_stability_by_k,
        pseudo_posterior_temperature=pseudo_posterior_temperature,
        pseudo_fit_times=pseudo_fit_times,
        pseudo_fit_sample_count=pseudo_fit_sample_count,
        pseudo_constraint_class_labels=pseudo_constraint_class_labels,
        pseudo_constraint_class_mode=pseudo_constraint_class_mode,
        pseudo_constraint_dim=pseudo_constraint_dim,
        pseudo_constraint_feature_mode=pseudo_constraint_feature_mode,
        moment_target_source=moment_target_source,
        pseudo_target_source=pseudo_target_source,
        pseudo_target_kind=pseudo_target_kind,
        pseudo_supervised_kept_class_labels=pseudo_supervised_kept_class_labels,
        pseudo_supervised_dropped_class_labels=pseudo_supervised_dropped_class_labels,
        pseudo_supervised_train_count=pseudo_supervised_train_count,
        pseudo_supervised_val_count=pseudo_supervised_val_count,
        pseudo_supervised_val_split_used=pseudo_supervised_val_split_used,
        pseudo_supervised_val_split_fallback=pseudo_supervised_val_split_fallback,
        pseudo_supervised_epochs_trained=pseudo_supervised_epochs_trained,
        pseudo_supervised_best_epoch=pseudo_supervised_best_epoch,
        pseudo_supervised_best_val_loss=pseudo_supervised_best_val_loss,
        pseudo_supervised_early_stop_triggered=pseudo_supervised_early_stop_triggered,
        pseudo_gmm_weights=pseudo_gmm_weights,
        pseudo_gmm_means=pseudo_gmm_means,
        pseudo_gmm_covariances=pseudo_gmm_covariances,
    )
