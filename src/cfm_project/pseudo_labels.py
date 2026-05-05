from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch


@dataclass
class PseudoLabelPreparedData:
    posterior: Callable[[torch.Tensor], torch.Tensor]
    selected_k: int
    bic_by_k: dict[int, float]
    stability_by_k: dict[int, float]
    cache_path: str | None
    cache_hit: bool


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    hasher = hashlib.sha256()
    hasher.update(str(contiguous.dtype).encode("utf-8"))
    hasher.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    hasher.update(contiguous.tobytes())
    return hasher.hexdigest()


def _pseudo_cache_root(single_cfg: Mapping[str, Any]) -> Path:
    pseudo_cfg = single_cfg.get("pseudo_labels", {})
    configured = str(pseudo_cfg.get("cache_dir", ".cache/pseudo_labels")).strip()
    root = Path(configured)
    if not root.is_absolute():
        root = _project_root() / root
    return root.resolve()


def _pseudo_signature(
    *,
    dataset_path: str,
    features_np: np.ndarray,
    time_indices: np.ndarray,
    single_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    pseudo_cfg = single_cfg.get("pseudo_labels", {})
    dataset = Path(dataset_path).expanduser().resolve()
    if not dataset.exists():
        raise FileNotFoundError(f"Single-cell dataset path does not exist: {dataset}")
    stat = dataset.stat()
    signature: dict[str, Any] = {
        "schema": "single_cell_pseudo_labels_gmm_v1",
        "dataset_path": str(dataset),
        "dataset_size_bytes": int(stat.st_size),
        "dataset_mtime_ns": int(stat.st_mtime_ns),
        "embed_key_npz": str(single_cfg.get("embed_key_npz", "pcs")),
        "label_key_npz": str(single_cfg.get("label_key_npz", "sample_labels")),
        "embed_key_h5ad": str(single_cfg.get("embed_key_h5ad", "X_pca")),
        "label_key_h5ad": str(single_cfg.get("label_key_h5ad", "day")),
        "max_dim": int(single_cfg.get("max_dim", features_np.shape[1])),
        "whiten": bool(single_cfg.get("whiten", True)),
        "features_hash": _sha256_array(features_np),
        "time_indices_hash": _sha256_array(time_indices.astype(np.int64, copy=False)),
        "k_min": int(pseudo_cfg.get("k_min", 2)),
        "k_max": int(pseudo_cfg.get("k_max", 10)),
        "seeds": [int(seed) for seed in pseudo_cfg.get("seeds", [7, 11, 19])],
        "n_init": int(pseudo_cfg.get("n_init", 3)),
        "max_iter": int(pseudo_cfg.get("max_iter", 300)),
        "tol": float(pseudo_cfg.get("tol", 1e-3)),
        "reg_covar": float(pseudo_cfg.get("reg_covar", 1e-6)),
        "stability_threshold": float(pseudo_cfg.get("stability_threshold", 0.7)),
    }
    return signature


def _pseudo_cache_key(signature: Mapping[str, Any]) -> str:
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _torch_posterior(
    *,
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable[[torch.Tensor], torch.Tensor]:
    weights_t = torch.as_tensor(weights, device=device, dtype=dtype)
    means_t = torch.as_tensor(means, device=device, dtype=dtype)
    cov_t = torch.as_tensor(covariances, device=device, dtype=dtype)
    if cov_t.ndim != 3:
        raise ValueError(f"Expected covariances with shape (K, d, d), got {tuple(cov_t.shape)}")
    if means_t.ndim != 2:
        raise ValueError(f"Expected means with shape (K, d), got {tuple(means_t.shape)}")
    if cov_t.shape[0] != means_t.shape[0]:
        raise ValueError(
            "GMM parameters mismatch: "
            f"{cov_t.shape[0]} covariances vs {means_t.shape[0]} means."
        )
    if cov_t.shape[1] != means_t.shape[1] or cov_t.shape[2] != means_t.shape[1]:
        raise ValueError(
            "GMM covariance/mean dimensions mismatch: "
            f"cov={tuple(cov_t.shape)}, means={tuple(means_t.shape)}."
        )
    precision_t = torch.linalg.inv(cov_t)
    sign, logdet = torch.linalg.slogdet(cov_t)
    if torch.any(sign <= 0):
        raise ValueError("GMM covariance matrices must be positive definite.")
    log_weights = torch.log(torch.clamp(weights_t, min=torch.finfo(dtype).eps))
    d = int(means_t.shape[1])
    normal_const = float(d) * math.log(2.0 * math.pi)

    def posterior(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected x with shape (N, d), got {tuple(x.shape)}")
        if x.shape[1] != means_t.shape[1]:
            raise ValueError(
                f"Posterior input dimension mismatch: expected {means_t.shape[1]}, got {x.shape[1]}"
            )
        local_weights = log_weights
        local_means = means_t
        local_precision = precision_t
        local_logdet = logdet
        if x.device != means_t.device or x.dtype != means_t.dtype:
            local_weights = log_weights.to(device=x.device, dtype=x.dtype)
            local_means = means_t.to(device=x.device, dtype=x.dtype)
            local_precision = precision_t.to(device=x.device, dtype=x.dtype)
            local_logdet = logdet.to(device=x.device, dtype=x.dtype)

        diff = x[:, None, :] - local_means[None, :, :]
        mahalanobis = torch.einsum("nkd,kde,nke->nk", diff, local_precision, diff)
        log_probs = local_weights[None, :] - 0.5 * (
            mahalanobis + local_logdet[None, :] + float(normal_const)
        )
        return torch.softmax(log_probs, dim=1)

    return posterior


def prepare_pseudo_labels(
    *,
    dataset_path: str,
    features_np: np.ndarray,
    time_indices: np.ndarray,
    single_cfg: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> PseudoLabelPreparedData | None:
    pseudo_cfg = single_cfg.get("pseudo_labels", {})
    if not bool(pseudo_cfg.get("enabled", False)):
        return None

    try:
        from sklearn.metrics import adjusted_rand_score
        from sklearn.mixture import GaussianMixture
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Pseudo-label constraints require scikit-learn. "
            "Install scikit-learn to enable data.single_cell.pseudo_labels.enabled=true."
        ) from exc

    k_min = int(pseudo_cfg.get("k_min", 2))
    k_max = int(pseudo_cfg.get("k_max", 10))
    if k_min < 2:
        raise ValueError(f"pseudo_labels.k_min must be >=2, got {k_min}")
    if k_max < k_min:
        raise ValueError(
            f"pseudo_labels.k_max must be >= pseudo_labels.k_min, got {k_max} < {k_min}"
        )
    seeds = [int(seed) for seed in pseudo_cfg.get("seeds", [7, 11, 19])]
    if not seeds:
        raise ValueError("pseudo_labels.seeds must contain at least one seed.")
    n_init = int(pseudo_cfg.get("n_init", 3))
    max_iter = int(pseudo_cfg.get("max_iter", 300))
    tol = float(pseudo_cfg.get("tol", 1e-3))
    reg_covar = float(pseudo_cfg.get("reg_covar", 1e-6))
    stability_threshold = float(pseudo_cfg.get("stability_threshold", 0.7))
    cache_enabled = bool(pseudo_cfg.get("cache_enabled", True))
    force_recompute = bool(pseudo_cfg.get("force_recompute", False))

    signature = _pseudo_signature(
        dataset_path=dataset_path,
        features_np=features_np.astype(np.float64, copy=False),
        time_indices=time_indices,
        single_cfg=single_cfg,
    )
    cache_key = _pseudo_cache_key(signature)
    cache_path: Path | None = None
    if cache_enabled:
        cache_root = _pseudo_cache_root(single_cfg)
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_path = cache_root / f"{cache_key}.pt"

    if cache_enabled and cache_path is not None and cache_path.exists() and not force_recompute:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        payload_signature = payload.get("signature")
        if payload_signature != signature:
            raise RuntimeError(
                "Pseudo-label cache signature mismatch for existing cache file. "
                f"Delete {cache_path} or set pseudo_labels.force_recompute=true."
            )
        selected_k = int(payload["selected_k"])
        bic_by_k_raw = payload.get("bic_by_k", {})
        stability_by_k_raw = payload.get("stability_by_k", {})
        posterior = _torch_posterior(
            weights=np.asarray(payload["weights"], dtype=np.float64),
            means=np.asarray(payload["means"], dtype=np.float64),
            covariances=np.asarray(payload["covariances"], dtype=np.float64),
            device=device,
            dtype=dtype,
        )
        return PseudoLabelPreparedData(
            posterior=posterior,
            selected_k=selected_k,
            bic_by_k={int(k): float(v) for k, v in bic_by_k_raw.items()},
            stability_by_k={int(k): float(v) for k, v in stability_by_k_raw.items()},
            cache_path=str(cache_path),
            cache_hit=True,
        )

    features = np.asarray(features_np, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"Expected features shape (N, d), got {features.shape}")
    if features.shape[0] < k_max:
        raise ValueError(
            f"Need at least k_max={k_max} samples to fit GMM, got {features.shape[0]} samples."
        )

    bic_by_k: dict[int, float] = {}
    stability_by_k: dict[int, float] = {}
    best_models_by_k: dict[int, Any] = {}
    for k in range(k_min, k_max + 1):
        labels_per_seed: list[np.ndarray] = []
        model_bics: list[float] = []
        best_model: Any | None = None
        best_bic = float("inf")
        for seed in seeds:
            model = GaussianMixture(
                n_components=int(k),
                covariance_type="full",
                reg_covar=float(reg_covar),
                n_init=int(n_init),
                max_iter=int(max_iter),
                tol=float(tol),
                random_state=int(seed),
            )
            model.fit(features)
            bic = float(model.bic(features))
            labels = model.predict(features)
            model_bics.append(bic)
            labels_per_seed.append(np.asarray(labels, dtype=np.int64))
            if bic < best_bic:
                best_bic = bic
                best_model = model
        if best_model is None:
            raise RuntimeError(f"Failed to fit any GMM model for k={k}")
        if len(labels_per_seed) <= 1:
            stability = 1.0
        else:
            pairwise: list[float] = []
            for idx_a in range(len(labels_per_seed)):
                for idx_b in range(idx_a + 1, len(labels_per_seed)):
                    pairwise.append(
                        float(adjusted_rand_score(labels_per_seed[idx_a], labels_per_seed[idx_b]))
                    )
            stability = float(np.mean(pairwise)) if pairwise else 1.0
        bic_by_k[int(k)] = float(np.median(model_bics))
        stability_by_k[int(k)] = stability
        best_models_by_k[int(k)] = best_model

    candidate_ks = [k for k in range(k_min, k_max + 1) if stability_by_k[k] >= stability_threshold]
    if not candidate_ks:
        candidate_ks = list(range(k_min, k_max + 1))
    selected_k = min(candidate_ks, key=lambda k: (bic_by_k[k], int(k)))
    selected_model = best_models_by_k[int(selected_k)]

    weights = np.asarray(selected_model.weights_, dtype=np.float64)
    means = np.asarray(selected_model.means_, dtype=np.float64)
    covariances = np.asarray(selected_model.covariances_, dtype=np.float64)
    posterior = _torch_posterior(
        weights=weights,
        means=means,
        covariances=covariances,
        device=device,
        dtype=dtype,
    )

    if cache_enabled and cache_path is not None:
        torch.save(
            {
                "signature": signature,
                "selected_k": int(selected_k),
                "weights": weights,
                "means": means,
                "covariances": covariances,
                "bic_by_k": {int(k): float(v) for k, v in bic_by_k.items()},
                "stability_by_k": {int(k): float(v) for k, v in stability_by_k.items()},
            },
            cache_path,
        )

    return PseudoLabelPreparedData(
        posterior=posterior,
        selected_k=int(selected_k),
        bic_by_k={int(k): float(v) for k, v in bic_by_k.items()},
        stability_by_k={int(k): float(v) for k, v in stability_by_k.items()},
        cache_path=None if cache_path is None else str(cache_path),
        cache_hit=False,
    )
