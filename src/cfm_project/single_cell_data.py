from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

from cfm_project.data import EmpiricalCouplingProblem, moment_feature_vector_from_samples
from cfm_project.ot_utils import solve_balanced_ot_lp
from cfm_project.pseudo_labels import prepare_pseudo_labels


@dataclass
class SingleCellPreparedData:
    problem: EmpiricalCouplingProblem
    targets: dict[float, torch.Tensor]
    pseudo_targets: dict[float, torch.Tensor] | None
    target_samples_by_time: dict[float, torch.Tensor]
    target_sampler: Callable[[float, int, torch.Generator | None], torch.Tensor]
    pseudo_posterior: Callable[[torch.Tensor], torch.Tensor] | None
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
    pseudo_labels_k: int | None
    pseudo_labels_cache_path: str | None
    pseudo_labels_cache_hit: bool
    pseudo_labels_bic_by_k: dict[int, float] | None
    pseudo_labels_stability_by_k: dict[int, float] | None
    pseudo_fit_times: list[float] | None
    pseudo_fit_sample_count: int | None


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


def _sort_unique_labels(labels: np.ndarray) -> list[Any]:
    label_list = labels.tolist()
    unique = list(dict.fromkeys(label_list))
    numeric_values: list[tuple[float, Any]] = []
    numeric_ok = True
    for label in unique:
        try:
            numeric_values.append((float(label), label))
        except (TypeError, ValueError):
            numeric_ok = False
            break
    if numeric_ok:
        numeric_values.sort(key=lambda item: item[0])
        return [item[1] for item in numeric_values]
    return sorted(unique, key=lambda value: str(value))


def _load_npz_dataset(path: str, cfg: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    embed_key = str(cfg.get("embed_key_npz", "pcs"))
    label_key = str(cfg.get("label_key_npz", "sample_labels"))
    if embed_key not in data:
        raise KeyError(f"Missing key '{embed_key}' in NPZ dataset.")
    if label_key not in data:
        raise KeyError(f"Missing key '{label_key}' in NPZ dataset.")
    features = np.asarray(data[embed_key])
    labels = _as_1d_labels(np.asarray(data[label_key]))
    return features, labels


def _load_h5ad_dataset(path: str, cfg: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    try:
        import scanpy as sc
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Loading .h5ad datasets requires scanpy. Install scanpy to use single-cell h5ad inputs."
        ) from exc
    adata = sc.read_h5ad(path)
    embed_key = str(cfg.get("embed_key_h5ad", "X_pca"))
    label_key = str(cfg.get("label_key_h5ad", "day"))
    if embed_key not in adata.obsm:
        raise KeyError(f"Missing embedding '{embed_key}' in adata.obsm.")
    if label_key not in adata.obs:
        raise KeyError(f"Missing label column '{label_key}' in adata.obs.")
    features = np.asarray(adata.obsm[embed_key])
    labels = _as_1d_labels(np.asarray(adata.obs[label_key]))
    return features, labels


def _load_single_cell_dataset(data_cfg: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
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


def _whiten_features(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True)
    std = np.maximum(std, float(eps))
    return (features - mean) / std


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


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    hasher = hashlib.sha256()
    hasher.update(str(contiguous.dtype).encode("utf-8"))
    hasher.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    hasher.update(contiguous.tobytes())
    return hasher.hexdigest()


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
        "schema": "single_cell_global_balanced_ot_plan_v1",
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
        "x0_hash": _sha256_array(x0_np),
        "x1_hash": _sha256_array(x1_np),
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, str | None, bool]:
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
        return src_idx, tgt_idx, mass, total_cost, str(cache_path), True

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
    mass = mass / torch.clamp(mass.sum(), min=torch.finfo(mass.dtype).eps)
    total_cost = float(plan.total_cost)

    if cache_enabled and cache_path is not None:
        torch.save(
            {
                "signature": signature,
                "src_idx": src_idx,
                "tgt_idx": tgt_idx,
                "mass": mass,
                "total_cost": total_cost,
            },
            cache_path,
        )
    return src_idx, tgt_idx, mass, total_cost, (None if cache_path is None else str(cache_path)), False


def prepare_single_cell_problem_and_targets(
    data_cfg: Mapping[str, Any],
    experiment_cfg: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> SingleCellPreparedData:
    single_cfg = data_cfg.get("single_cell", {})
    coupling = str(data_cfg.get("coupling", "ot")).strip().lower()
    features_np, labels_np = _load_single_cell_dataset(data_cfg=data_cfg)
    if features_np.ndim != 2:
        raise ValueError(f"Expected feature matrix shape (N, d), got {features_np.shape}")
    if features_np.shape[0] != labels_np.shape[0]:
        raise ValueError(
            "Feature/label sample count mismatch: "
            f"{features_np.shape[0]} features vs {labels_np.shape[0]} labels."
        )

    max_dim = int(single_cfg.get("max_dim", features_np.shape[1]))
    if max_dim <= 0:
        raise ValueError(f"data.single_cell.max_dim must be positive, got {max_dim}")
    features_np = features_np[:, :max_dim]
    if bool(single_cfg.get("whiten", True)):
        features_np = _whiten_features(features_np)

    expected_dim = int(data_cfg.get("dim", features_np.shape[1]))
    if int(features_np.shape[1]) != expected_dim:
        raise ValueError(
            f"Configured data.dim={expected_dim}, but loaded feature dimension is {features_np.shape[1]}. "
            "Align data.dim and single-cell max_dim/embed settings."
        )

    sorted_labels = _sort_unique_labels(labels_np)
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
    pools_by_index: dict[int, torch.Tensor] = {}
    for idx in all_time_indices:
        mask = torch.as_tensor(time_indices == idx, device=device)
        pool = features[mask]
        if pool.shape[0] <= 0:
            raise ValueError(f"No samples found for time index {idx}.")
        pools_by_index[idx] = pool

    normalized_times_all = [_normalize_time(idx, n_times) for idx in all_time_indices]
    normalized_time_by_index = {idx: _normalize_time(idx, n_times) for idx in all_time_indices}
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

    targets = {
        float(normalized_time_by_index[idx]): moment_feature_vector_from_samples(pools_by_index[idx])
        for idx in constraint_time_indices
    }
    pseudo_targets: dict[float, torch.Tensor] | None = None
    pseudo_posterior: Callable[[torch.Tensor], torch.Tensor] | None = None
    pseudo_labels_k: int | None = None
    pseudo_labels_cache_path: str | None = None
    pseudo_labels_cache_hit = False
    pseudo_labels_bic_by_k: dict[int, float] | None = None
    pseudo_labels_stability_by_k: dict[int, float] | None = None
    pseudo_cfg = single_cfg.get("pseudo_labels", {})
    pseudo_fit_times: list[float] | None = None
    pseudo_fit_sample_count: int | None = None
    pseudo_fit_indices = list(all_time_indices)
    pseudo_fit_override_times = _parse_normalized_times(
        pseudo_cfg.get("fit_times_normalized", None),
        field_name="data.single_cell.pseudo_labels.fit_times_normalized",
    )
    if pseudo_fit_override_times:
        pseudo_fit_indices = _resolve_time_indices_from_normalized(
            requested_times=pseudo_fit_override_times,
            normalized_time_by_index=normalized_time_by_index,
            field_name="data.single_cell.pseudo_labels.fit_times_normalized",
        )
    if bool(pseudo_cfg.get("enabled", False)):
        pseudo_fit_times = [float(normalized_time_by_index[idx]) for idx in pseudo_fit_indices]
        fit_index_array = np.asarray(pseudo_fit_indices, dtype=np.int64)
        fit_mask = np.isin(time_indices, fit_index_array)
        features_for_pseudo = np.asarray(features_np[fit_mask], dtype=np.float64)
        time_indices_for_pseudo = np.asarray(time_indices[fit_mask], dtype=np.int64)
        pseudo_fit_sample_count = int(features_for_pseudo.shape[0])
        if pseudo_fit_sample_count <= 0:
            raise ValueError(
                "Pseudo-label fit subset is empty. "
                "Check data.single_cell.pseudo_labels.fit_times_normalized."
            )
        k_max = int(pseudo_cfg.get("k_max", 10))
        if pseudo_fit_sample_count < k_max:
            raise ValueError(
                "Pseudo-label fit subset has too few samples: "
                f"{pseudo_fit_sample_count} < k_max={k_max}. "
                "Adjust fit_times_normalized or k_max."
            )
    else:
        features_for_pseudo = np.asarray(features_np, dtype=np.float64)
        time_indices_for_pseudo = np.asarray(time_indices, dtype=np.int64)
    pseudo_prepared = prepare_pseudo_labels(
        dataset_path=str(single_cfg.get("path", "")),
        features_np=features_for_pseudo,
        time_indices=time_indices_for_pseudo,
        single_cfg=single_cfg,
        device=device,
        dtype=dtype,
    )
    if pseudo_prepared is not None:
        pseudo_posterior = pseudo_prepared.posterior
        pseudo_labels_k = int(pseudo_prepared.selected_k)
        pseudo_labels_cache_path = pseudo_prepared.cache_path
        pseudo_labels_cache_hit = bool(pseudo_prepared.cache_hit)
        pseudo_labels_bic_by_k = {
            int(k): float(v) for k, v in pseudo_prepared.bic_by_k.items()
        }
        pseudo_labels_stability_by_k = {
            int(k): float(v) for k, v in pseudo_prepared.stability_by_k.items()
        }
        pseudo_targets = {}
        with torch.no_grad():
            for idx in constraint_time_indices:
                t_key = float(normalized_time_by_index[idx])
                pseudo_targets[t_key] = pseudo_posterior(pools_by_index[idx]).mean(dim=0).detach()

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
    if coupling == "ot_global":
        (
            global_ot_src_idx,
            global_ot_tgt_idx,
            global_ot_mass,
            global_ot_total_cost,
            global_ot_cache_path,
            global_ot_cache_hit,
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
        target_sampler=target_sampler,
        pseudo_posterior=pseudo_posterior,
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
        pseudo_labels_k=pseudo_labels_k,
        pseudo_labels_cache_path=pseudo_labels_cache_path,
        pseudo_labels_cache_hit=bool(pseudo_labels_cache_hit),
        pseudo_labels_bic_by_k=pseudo_labels_bic_by_k,
        pseudo_labels_stability_by_k=pseudo_labels_stability_by_k,
        pseudo_fit_times=pseudo_fit_times,
        pseudo_fit_sample_count=pseudo_fit_sample_count,
    )
