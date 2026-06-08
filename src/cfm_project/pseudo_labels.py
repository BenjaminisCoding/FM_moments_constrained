from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from cfm_project.models import MLP


@dataclass
class PseudoLabelPreparedData:
    posterior: Callable[[torch.Tensor], torch.Tensor]
    selected_k: int
    method: str
    posterior_temperature: float
    class_labels: list[str] | None
    bic_by_k: dict[int, float]
    stability_by_k: dict[int, float]
    cache_path: str | None
    cache_hit: bool
    supervised_kept_class_labels: list[str] | None
    supervised_dropped_class_labels: list[str] | None
    supervised_train_count: int | None
    supervised_val_count: int | None
    supervised_val_split_used: bool | None
    supervised_val_split_fallback: bool | None
    supervised_epochs_trained: int | None
    supervised_best_epoch: int | None
    supervised_best_val_loss: float | None
    supervised_early_stop_triggered: bool | None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    hasher = hashlib.sha256()
    hasher.update(str(contiguous.dtype).encode("utf-8"))
    hasher.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    hasher.update(contiguous.tobytes())
    return hasher.hexdigest()


def _sha256_label_array(labels: np.ndarray) -> str:
    flat = np.asarray(labels).reshape(-1)
    payload = json.dumps(
        [str(value) for value in flat.tolist()],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
    method: str,
    supervised_labels_np: np.ndarray | None,
    single_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    pseudo_cfg = single_cfg.get("pseudo_labels", {})
    dataset = Path(dataset_path).expanduser().resolve()
    if not dataset.exists():
        raise FileNotFoundError(f"Single-cell dataset path does not exist: {dataset}")
    stat = dataset.stat()
    signature: dict[str, Any] = {
        "schema": "single_cell_pseudo_labels_v3",
        "method": str(method),
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
    }
    if method == "gmm":
        signature.update(
            {
                "k_min": int(pseudo_cfg.get("k_min", 2)),
                "k_max": int(pseudo_cfg.get("k_max", 10)),
                "seeds": [int(seed) for seed in pseudo_cfg.get("seeds", [7, 11, 19])],
                "n_init": int(pseudo_cfg.get("n_init", 3)),
                "max_iter": int(pseudo_cfg.get("max_iter", 300)),
                "tol": float(pseudo_cfg.get("tol", 1e-3)),
                "reg_covar": float(pseudo_cfg.get("reg_covar", 1e-6)),
                "stability_threshold": float(pseudo_cfg.get("stability_threshold", 0.7)),
            }
        )
    elif method == "supervised_mlp":
        supervised_cfg = _parse_supervised_mlp_config(pseudo_cfg)
        if supervised_labels_np is None:
            raise ValueError(
                "Supervised pseudo-label mode requires supervised labels for cache signature."
            )
        signature.update(
            {
                "supervised_label_key": str(pseudo_cfg.get("supervised_label_key", "cell_sets")),
                "supervised_labels_hash": _sha256_label_array(
                    np.asarray(supervised_labels_np)
                ),
                "supervised_hidden_dims": [int(value) for value in supervised_cfg["hidden_dims"]],
                "supervised_activation": str(supervised_cfg["activation"]),
                "supervised_epochs": int(supervised_cfg["epochs"]),
                "supervised_batch_size": int(supervised_cfg["batch_size"]),
                "supervised_lr": float(supervised_cfg["lr"]),
                "supervised_weight_decay": float(supervised_cfg["weight_decay"]),
                "supervised_seed": int(supervised_cfg["seed"]),
                "supervised_min_class_count": int(supervised_cfg["min_class_count"]),
                "supervised_val_fraction": float(supervised_cfg["val_fraction"]),
                "supervised_split_seed": int(supervised_cfg["split_seed"]),
                "supervised_early_stopping_patience": int(
                    supervised_cfg["early_stopping_patience"]
                ),
                "supervised_early_stopping_min_epochs": int(
                    supervised_cfg["early_stopping_min_epochs"]
                ),
                "supervised_early_stopping_min_delta": float(
                    supervised_cfg["early_stopping_min_delta"]
                ),
            }
        )
    elif method == "supervised_logreg":
        supervised_cfg = _parse_supervised_logreg_config(pseudo_cfg)
        if supervised_labels_np is None:
            raise ValueError(
                "Supervised pseudo-label mode requires supervised labels for cache signature."
            )
        signature.update(
            {
                "supervised_label_key": str(pseudo_cfg.get("supervised_label_key", "cell_sets")),
                "supervised_labels_hash": _sha256_label_array(
                    np.asarray(supervised_labels_np)
                ),
                "supervised_logreg_c": float(supervised_cfg["c"]),
                "supervised_logreg_penalty": str(supervised_cfg["penalty"]),
                "supervised_logreg_max_iter": int(supervised_cfg["max_iter"]),
                "supervised_logreg_class_weight": supervised_cfg["class_weight_signature"],
                "supervised_logreg_seed": int(supervised_cfg["seed"]),
            }
        )
    else:
        raise ValueError(f"Unsupported pseudo-label method '{method}'.")
    return signature


def _pseudo_cache_key(signature: Mapping[str, Any]) -> str:
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def _parse_supervised_mlp_config(pseudo_cfg: Mapping[str, Any]) -> dict[str, Any]:
    hidden_dims_raw = pseudo_cfg.get("supervised_hidden_dims", [128, 128])
    if not isinstance(hidden_dims_raw, (list, tuple)):
        raise ValueError(
            "pseudo_labels.supervised_hidden_dims must be a list of positive integers."
        )
    hidden_dims = [int(value) for value in hidden_dims_raw]
    if not hidden_dims or any(value <= 0 for value in hidden_dims):
        raise ValueError(
            "pseudo_labels.supervised_hidden_dims must be a non-empty list of positive integers."
        )
    activation = str(pseudo_cfg.get("supervised_activation", "silu"))
    epochs = int(pseudo_cfg.get("supervised_epochs", 300))
    batch_size = int(pseudo_cfg.get("supervised_batch_size", 256))
    lr = float(pseudo_cfg.get("supervised_lr", 1e-3))
    weight_decay = float(pseudo_cfg.get("supervised_weight_decay", 1e-4))
    seed = int(pseudo_cfg.get("supervised_seed", 7))
    min_class_count = int(pseudo_cfg.get("supervised_min_class_count", 1))
    val_fraction = float(pseudo_cfg.get("supervised_val_fraction", 0.0))
    split_seed_raw = pseudo_cfg.get("supervised_split_seed", None)
    split_seed = int(seed if split_seed_raw is None else split_seed_raw)
    early_stopping_patience = int(pseudo_cfg.get("supervised_early_stopping_patience", 20))
    early_stopping_min_epochs = int(pseudo_cfg.get("supervised_early_stopping_min_epochs", 20))
    early_stopping_min_delta = float(pseudo_cfg.get("supervised_early_stopping_min_delta", 0.0))

    if epochs <= 0:
        raise ValueError(f"pseudo_labels.supervised_epochs must be positive, got {epochs}")
    if batch_size <= 0:
        raise ValueError(
            f"pseudo_labels.supervised_batch_size must be positive, got {batch_size}"
        )
    if lr <= 0.0:
        raise ValueError(f"pseudo_labels.supervised_lr must be positive, got {lr}")
    if weight_decay < 0.0:
        raise ValueError(
            "pseudo_labels.supervised_weight_decay must be non-negative, "
            f"got {weight_decay}"
        )
    if min_class_count < 1:
        raise ValueError(
            "pseudo_labels.supervised_min_class_count must be >= 1, "
            f"got {min_class_count}"
        )
    if not math.isfinite(val_fraction) or val_fraction < 0.0 or val_fraction >= 1.0:
        raise ValueError(
            "pseudo_labels.supervised_val_fraction must be in [0, 1), "
            f"got {val_fraction}"
        )
    if early_stopping_patience <= 0:
        raise ValueError(
            "pseudo_labels.supervised_early_stopping_patience must be positive, "
            f"got {early_stopping_patience}"
        )
    if early_stopping_min_epochs <= 0:
        raise ValueError(
            "pseudo_labels.supervised_early_stopping_min_epochs must be positive, "
            f"got {early_stopping_min_epochs}"
        )
    if early_stopping_min_delta < 0.0 or not math.isfinite(early_stopping_min_delta):
        raise ValueError(
            "pseudo_labels.supervised_early_stopping_min_delta must be finite and >= 0, "
            f"got {early_stopping_min_delta}"
        )
    return {
        "hidden_dims": hidden_dims,
        "activation": activation,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "seed": seed,
        "min_class_count": min_class_count,
        "val_fraction": val_fraction,
        "split_seed": split_seed,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_epochs": early_stopping_min_epochs,
        "early_stopping_min_delta": early_stopping_min_delta,
    }


def _parse_supervised_logreg_config(pseudo_cfg: Mapping[str, Any]) -> dict[str, Any]:
    c_value = float(pseudo_cfg.get("supervised_logreg_c", 1.0))
    penalty = str(pseudo_cfg.get("supervised_logreg_penalty", "l2")).strip().lower()
    max_iter = int(pseudo_cfg.get("supervised_logreg_max_iter", 1000))
    class_weight_raw = pseudo_cfg.get("supervised_logreg_class_weight", "balanced")
    seed = int(pseudo_cfg.get("supervised_logreg_seed", 7))

    if not math.isfinite(c_value) or c_value <= 0.0:
        raise ValueError(
            "pseudo_labels.supervised_logreg_c must be a finite positive value, "
            f"got {c_value}."
        )
    if penalty not in {"l2", "none"}:
        raise ValueError(
            "pseudo_labels.supervised_logreg_penalty must be one of {'l2', 'none'}, "
            f"got '{penalty}'."
        )
    if max_iter <= 0:
        raise ValueError(
            "pseudo_labels.supervised_logreg_max_iter must be positive, "
            f"got {max_iter}."
        )

    class_weight: str | dict[int, float] | None
    class_weight_signature: Any
    if isinstance(class_weight_raw, str):
        normalized = class_weight_raw.strip().lower()
        if normalized in {"", "none", "null"}:
            class_weight = None
            class_weight_signature = None
        elif normalized == "balanced":
            class_weight = "balanced"
            class_weight_signature = "balanced"
        else:
            raise ValueError(
                "pseudo_labels.supervised_logreg_class_weight must be 'balanced' or null-like "
                f"for v1, got '{class_weight_raw}'."
            )
    elif class_weight_raw is None:
        class_weight = None
        class_weight_signature = None
    else:
        raise ValueError(
            "pseudo_labels.supervised_logreg_class_weight must be 'balanced' or null-like "
            f"for v1, got value of type {type(class_weight_raw).__name__}."
        )

    return {
        "c": c_value,
        "penalty": penalty,
        "max_iter": max_iter,
        "class_weight": class_weight,
        "class_weight_signature": class_weight_signature,
        "seed": seed,
    }


def _parse_posterior_temperature(pseudo_cfg: Mapping[str, Any]) -> float:
    temperature = float(pseudo_cfg.get("posterior_temperature", 1.0))
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(
            "pseudo_labels.posterior_temperature must be a finite positive value, "
            f"got {temperature}."
        )
    return temperature


def _stratified_train_val_indices(
    *,
    encoded_labels: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if encoded_labels.ndim != 1:
        raise ValueError(
            f"Expected encoded labels shape (N,), got {tuple(encoded_labels.shape)}"
        )
    n_samples = int(encoded_labels.shape[0])
    if n_samples <= 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    if val_fraction <= 0.0:
        train = np.arange(n_samples, dtype=np.int64)
        return train, np.zeros((0,), dtype=np.int64)

    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for cls in np.unique(encoded_labels).tolist():
        cls_idx = np.where(encoded_labels == int(cls))[0]
        if cls_idx.size <= 0:
            continue
        shuffled = np.asarray(cls_idx, dtype=np.int64).copy()
        rng.shuffle(shuffled)
        if shuffled.size <= 1:
            n_val_cls = 0
        else:
            n_val_cls = int(math.floor(float(shuffled.size) * float(val_fraction)))
            if n_val_cls <= 0:
                n_val_cls = 1
            n_val_cls = min(n_val_cls, int(shuffled.size) - 1)
        if n_val_cls > 0:
            val_parts.append(shuffled[:n_val_cls])
            train_parts.append(shuffled[n_val_cls:])
        else:
            train_parts.append(shuffled)
    if train_parts:
        train_idx = np.concatenate(train_parts, axis=0)
    else:
        train_idx = np.zeros((0,), dtype=np.int64)
    if val_parts:
        val_idx = np.concatenate(val_parts, axis=0)
    else:
        val_idx = np.zeros((0,), dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx.astype(np.int64, copy=False), val_idx.astype(np.int64, copy=False)


def _fit_supervised_mlp_posterior(
    *,
    features_np: np.ndarray,
    labels_np: np.ndarray,
    pseudo_cfg: Mapping[str, Any],
    posterior_temperature: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[
    Callable[[torch.Tensor], torch.Tensor],
    int,
    list[str],
    list[str],
    MLP,
    int,
    list[int],
    str,
    int,
    int,
    bool,
    bool,
    int,
    int | None,
    float | None,
    bool,
]:
    features = np.asarray(features_np, dtype=np.float64)
    labels = np.asarray(labels_np)
    if features.ndim != 2:
        raise ValueError(f"Expected features shape (N, d), got {features.shape}")
    if labels.ndim != 1:
        raise ValueError(f"Expected supervised labels shape (N,), got {labels.shape}")
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            "Supervised pseudo-label training requires aligned features/labels. "
            f"Got {features.shape[0]} features vs {labels.shape[0]} labels."
        )
    supervised_cfg = _parse_supervised_mlp_config(pseudo_cfg)
    min_class_count = int(supervised_cfg["min_class_count"])
    val_fraction = float(supervised_cfg["val_fraction"])
    split_seed = int(supervised_cfg["split_seed"])
    early_stopping_patience = int(supervised_cfg["early_stopping_patience"])
    early_stopping_min_epochs = int(supervised_cfg["early_stopping_min_epochs"])
    early_stopping_min_delta = float(supervised_cfg["early_stopping_min_delta"])

    raw_counts: dict[str, int] = {}
    for label in labels.tolist():
        key = str(label)
        raw_counts[key] = raw_counts.get(key, 0) + 1
    sorted_labels = _sort_unique_labels(labels)
    sorted_label_strings = [str(label) for label in sorted_labels]
    kept_class_labels = [
        label for label in sorted_label_strings if int(raw_counts.get(label, 0)) >= min_class_count
    ]
    dropped_class_labels = [
        label for label in sorted_label_strings if int(raw_counts.get(label, 0)) < min_class_count
    ]
    if len(kept_class_labels) < 2:
        raise ValueError(
            "Supervised pseudo-label mode requires at least 2 classes after "
            f"supervised_min_class_count filtering, got {len(kept_class_labels)} "
            f"(threshold={min_class_count})."
        )
    kept_set = set(kept_class_labels)
    keep_mask = np.asarray([str(label) in kept_set for label in labels.tolist()], dtype=bool)
    filtered_features = features[keep_mask]
    filtered_labels = np.asarray([str(label) for label in labels[keep_mask].tolist()], dtype=object)
    if filtered_features.shape[0] <= 0:
        raise ValueError(
            "All supervised labels were filtered out by supervised_min_class_count; "
            f"threshold={min_class_count}."
        )
    class_labels = list(kept_class_labels)
    label_to_index = {label: idx for idx, label in enumerate(class_labels)}
    encoded_labels = np.asarray(
        [label_to_index[str(label)] for label in filtered_labels.tolist()],
        dtype=np.int64,
    )

    hidden_dims = [int(value) for value in supervised_cfg["hidden_dims"]]
    activation = str(supervised_cfg["activation"])
    epochs = int(supervised_cfg["epochs"])
    batch_size = int(supervised_cfg["batch_size"])
    lr = float(supervised_cfg["lr"])
    weight_decay = float(supervised_cfg["weight_decay"])
    seed = int(supervised_cfg["seed"])

    train_idx, val_idx = _stratified_train_val_indices(
        encoded_labels=encoded_labels,
        val_fraction=val_fraction,
        seed=split_seed,
    )
    val_split_requested = val_fraction > 0.0
    val_split_used = bool(val_split_requested and val_idx.size > 0)
    val_split_fallback = bool(val_split_requested and val_idx.size <= 0)
    if train_idx.size <= 0:
        raise ValueError("Supervised pseudo-label split produced empty training subset.")
    if val_split_fallback:
        train_idx = np.arange(encoded_labels.shape[0], dtype=np.int64)

    x_all = torch.as_tensor(filtered_features, device=device, dtype=dtype)
    y_all = torch.as_tensor(encoded_labels, device=device, dtype=torch.long)
    train_idx_t = torch.as_tensor(train_idx, device=device, dtype=torch.long)
    y_train = y_all[train_idx_t]
    x_train = x_all[train_idx_t]
    if val_split_used:
        val_idx_t = torch.as_tensor(val_idx, device=device, dtype=torch.long)
        x_val = x_all[val_idx_t]
        y_val = y_all[val_idx_t]
    else:
        x_val = None
        y_val = None
    num_classes = int(len(class_labels))

    counts = torch.bincount(y_train, minlength=num_classes).to(dtype=dtype)
    class_weights = torch.clamp(counts.sum() / torch.clamp(counts, min=1.0), min=1.0)
    class_weights = class_weights / torch.clamp(
        class_weights.mean(),
        min=torch.finfo(class_weights.dtype).eps,
    )

    model = MLP(
        in_dim=int(x_train.shape[1]),
        hidden_dims=hidden_dims,
        out_dim=num_classes,
        activation=activation,
    ).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    rng = np.random.default_rng(seed)
    n_train = int(x_train.shape[0])
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_val_loss: float | None = None
    best_epoch: int | None = None
    epochs_trained = 0
    epochs_since_improve = 0
    early_stop_triggered = False

    model.train()
    for epoch in range(1, epochs + 1):
        shuffled = rng.permutation(n_train)
        for start in range(0, n_train, batch_size):
            batch = shuffled[start : start + batch_size]
            batch_idx = torch.as_tensor(batch, device=device, dtype=torch.long)
            logits = model(x_train[batch_idx])
            loss = F.cross_entropy(logits, y_train[batch_idx], weight=class_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        epochs_trained = int(epoch)
        if val_split_used and x_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_logits = model(x_val)
                val_loss = float(
                    F.cross_entropy(val_logits, y_val, weight=class_weights).item()
                )
            improved = (
                best_val_loss is None
                or val_loss < (float(best_val_loss) - float(early_stopping_min_delta))
            )
            if improved:
                best_val_loss = float(val_loss)
                best_epoch = int(epoch)
                best_state_dict = {
                    name: tensor.detach().clone()
                    for name, tensor in model.state_dict().items()
                }
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
            if (
                int(epoch) >= int(early_stopping_min_epochs)
                and epochs_since_improve >= int(early_stopping_patience)
            ):
                early_stop_triggered = True
                break
            model.train()

    if val_split_used and best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        if best_epoch is None:
            best_epoch = int(epochs_trained)
    elif val_split_used and best_epoch is None:
        best_epoch = int(epochs_trained)

    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    posterior = _build_supervised_mlp_posterior(
        model=model,
        in_dim=int(x_train.shape[1]),
        posterior_temperature=float(posterior_temperature),
    )
    return (
        posterior,
        num_classes,
        class_labels,
        dropped_class_labels,
        model,
        int(x_all.shape[1]),
        hidden_dims,
        activation,
        int(train_idx.shape[0]),
        int(val_idx.shape[0]) if val_split_used else 0,
        bool(val_split_used),
        bool(val_split_fallback),
        int(epochs_trained),
        best_epoch,
        best_val_loss,
        bool(early_stop_triggered),
    )


def _build_supervised_mlp_posterior(
    *,
    model: MLP,
    in_dim: int,
    posterior_temperature: float,
) -> Callable[[torch.Tensor], torch.Tensor]:
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    def posterior(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected x with shape (N, d), got {tuple(x.shape)}")
        if x.shape[1] != int(in_dim):
            raise ValueError(
                "Posterior input dimension mismatch: "
                f"expected {in_dim}, got {x.shape[1]}"
            )
        if x.device != model_device:
            raise ValueError(
                "Supervised posterior input device mismatch: "
                f"expected {model_device}, got {x.device}"
            )
        x_local = x if x.dtype == model_dtype else x.to(dtype=model_dtype)
        logits = model(x_local)
        return torch.softmax(logits / float(posterior_temperature), dim=1)

    return posterior


def _fit_supervised_logreg_posterior(
    *,
    features_np: np.ndarray,
    labels_np: np.ndarray,
    pseudo_cfg: Mapping[str, Any],
    posterior_temperature: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[
    Callable[[torch.Tensor], torch.Tensor],
    int,
    list[str],
    int,
    np.ndarray,
    np.ndarray,
]:
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Supervised pseudo-label mode with method='supervised_logreg' requires scikit-learn. "
            "Install scikit-learn or switch to method='supervised_mlp'."
        ) from exc

    features = np.asarray(features_np, dtype=np.float64)
    labels = np.asarray(labels_np)
    if features.ndim != 2:
        raise ValueError(f"Expected features shape (N, d), got {features.shape}")
    if labels.ndim != 1:
        raise ValueError(f"Expected supervised labels shape (N,), got {labels.shape}")
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            "Supervised pseudo-label training requires aligned features/labels. "
            f"Got {features.shape[0]} features vs {labels.shape[0]} labels."
        )
    sorted_labels = _sort_unique_labels(labels)
    if len(sorted_labels) < 2:
        raise ValueError(
            "Supervised pseudo-label mode requires at least 2 unique classes, "
            f"got {len(sorted_labels)}."
        )
    class_labels = [str(label) for label in sorted_labels]
    label_to_index = {label: idx for idx, label in enumerate(class_labels)}
    encoded_labels = np.asarray(
        [label_to_index[str(label)] for label in labels.tolist()],
        dtype=np.int64,
    )

    logreg_cfg = _parse_supervised_logreg_config(pseudo_cfg)
    penalty = None if str(logreg_cfg["penalty"]) == "none" else str(logreg_cfg["penalty"])
    model = LogisticRegression(
        penalty=penalty,
        C=float(logreg_cfg["c"]),
        solver="lbfgs",
        class_weight=logreg_cfg["class_weight"],
        max_iter=int(logreg_cfg["max_iter"]),
        random_state=int(logreg_cfg["seed"]),
    )
    model.fit(features, encoded_labels)

    weights_np = np.asarray(model.coef_, dtype=np.float64)
    bias_np = np.asarray(model.intercept_, dtype=np.float64).reshape(-1)
    num_classes = int(len(class_labels))
    in_dim = int(features.shape[1])
    if weights_np.ndim != 2 or weights_np.shape[1] != in_dim:
        raise ValueError(
            "Invalid LogisticRegression coefficient shape: "
            f"expected (?, {in_dim}), got {tuple(weights_np.shape)}."
        )

    # sklearn may return a single-row parameterization for binary case.
    if weights_np.shape[0] == 1 and num_classes == 2:
        zeros_w = np.zeros((1, in_dim), dtype=np.float64)
        zeros_b = np.zeros((1,), dtype=np.float64)
        weights_np = np.concatenate([zeros_w, weights_np], axis=0)
        bias_np = np.concatenate([zeros_b, bias_np], axis=0)

    if weights_np.shape[0] != num_classes or bias_np.shape[0] != num_classes:
        raise ValueError(
            "LogisticRegression class-parameter mismatch: "
            f"expected {num_classes} rows, got coef={weights_np.shape[0]}, "
            f"intercept={bias_np.shape[0]}."
        )

    posterior = _build_supervised_logreg_posterior(
        weights_np=weights_np,
        bias_np=bias_np,
        in_dim=in_dim,
        device=device,
        dtype=dtype,
        posterior_temperature=float(posterior_temperature),
    )
    return posterior, num_classes, class_labels, in_dim, weights_np, bias_np


def _build_supervised_logreg_posterior(
    *,
    weights_np: np.ndarray,
    bias_np: np.ndarray,
    in_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    posterior_temperature: float,
) -> Callable[[torch.Tensor], torch.Tensor]:
    weights_t = torch.as_tensor(weights_np, device=device, dtype=dtype)
    bias_t = torch.as_tensor(bias_np, device=device, dtype=dtype)
    if weights_t.ndim != 2:
        raise ValueError(
            f"Expected logreg weights shape (K, d), got {tuple(weights_t.shape)}"
        )
    if bias_t.ndim != 1:
        raise ValueError(
            f"Expected logreg bias shape (K,), got {tuple(bias_t.shape)}"
        )
    if weights_t.shape[1] != int(in_dim):
        raise ValueError(
            f"Logreg input dimension mismatch: expected {in_dim}, got {weights_t.shape[1]}"
        )
    if bias_t.shape[0] != weights_t.shape[0]:
        raise ValueError(
            "Logreg class dimension mismatch: "
            f"weights rows={weights_t.shape[0]}, bias size={bias_t.shape[0]}"
        )

    def posterior(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected x with shape (N, d), got {tuple(x.shape)}")
        if x.shape[1] != int(in_dim):
            raise ValueError(
                "Posterior input dimension mismatch: "
                f"expected {in_dim}, got {x.shape[1]}"
            )
        local_w = weights_t
        local_b = bias_t
        if x.device != weights_t.device or x.dtype != weights_t.dtype:
            local_w = weights_t.to(device=x.device, dtype=x.dtype)
            local_b = bias_t.to(device=x.device, dtype=x.dtype)
        logits = x @ local_w.transpose(0, 1) + local_b.unsqueeze(0)
        return torch.softmax(logits / float(posterior_temperature), dim=1)

    return posterior


def _torch_posterior(
    *,
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    posterior_temperature: float,
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
        return torch.softmax(log_probs / float(posterior_temperature), dim=1)

    return posterior


def prepare_pseudo_labels(
    *,
    dataset_path: str,
    features_np: np.ndarray,
    time_indices: np.ndarray,
    supervised_labels_np: np.ndarray | None = None,
    single_cfg: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> PseudoLabelPreparedData | None:
    pseudo_cfg = single_cfg.get("pseudo_labels", {})
    if not bool(pseudo_cfg.get("enabled", False)):
        return None
    posterior_temperature = _parse_posterior_temperature(pseudo_cfg)

    method = str(pseudo_cfg.get("method", "gmm")).strip().lower()
    cache_enabled = bool(pseudo_cfg.get("cache_enabled", True))
    force_recompute = bool(pseudo_cfg.get("force_recompute", False))
    if method in {"supervised_mlp", "supervised_logreg"}:
        if supervised_labels_np is None:
            raise ValueError(
                "Supervised pseudo-label mode requires supervised labels. "
                "Pass supervised_labels_np from the dataset label column."
            )
        signature = _pseudo_signature(
            dataset_path=dataset_path,
            features_np=features_np.astype(np.float64, copy=False),
            time_indices=time_indices,
            method=method,
            supervised_labels_np=np.asarray(supervised_labels_np),
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
            payload_method = str(payload.get("method", method))
            if payload_method != method:
                raise RuntimeError(
                    "Pseudo-label cache method mismatch for existing cache file. "
                    f"Expected '{method}', got '{payload_method}'. "
                    f"Delete {cache_path} or set pseudo_labels.force_recompute=true."
                )
            in_dim = int(payload["in_dim"])
            selected_k = int(payload["selected_k"])
            class_labels = [str(label) for label in payload.get("class_labels", [])]
            if len(class_labels) != selected_k:
                raise ValueError(
                    "Cached supervised pseudo-label payload is inconsistent: "
                    f"{len(class_labels)} class labels vs selected_k={selected_k}."
                )
            supervised_kept_class_labels = [
                str(label)
                for label in payload.get("supervised_kept_class_labels", class_labels)
            ]
            supervised_dropped_class_labels = [
                str(label)
                for label in payload.get("supervised_dropped_class_labels", [])
            ]
            supervised_train_count_raw = payload.get("supervised_train_count", None)
            supervised_val_count_raw = payload.get("supervised_val_count", None)
            supervised_val_split_used_raw = payload.get("supervised_val_split_used", None)
            supervised_val_split_fallback_raw = payload.get(
                "supervised_val_split_fallback",
                None,
            )
            supervised_epochs_trained_raw = payload.get("supervised_epochs_trained", None)
            supervised_best_epoch_raw = payload.get("supervised_best_epoch", None)
            supervised_best_val_loss_raw = payload.get("supervised_best_val_loss", None)
            supervised_early_stop_triggered_raw = payload.get(
                "supervised_early_stop_triggered",
                None,
            )
            if method == "supervised_mlp":
                hidden_dims = [int(value) for value in payload["hidden_dims"]]
                activation = str(payload["activation"])
                state_dict = payload["state_dict"]
                model = MLP(
                    in_dim=in_dim,
                    hidden_dims=hidden_dims,
                    out_dim=selected_k,
                    activation=activation,
                ).to(device=device, dtype=dtype)
                model.load_state_dict(state_dict)
                for param in model.parameters():
                    param.requires_grad_(False)
                model.eval()
                posterior = _build_supervised_mlp_posterior(
                    model=model,
                    in_dim=in_dim,
                    posterior_temperature=float(posterior_temperature),
                )
            else:
                posterior = _build_supervised_logreg_posterior(
                    weights_np=np.asarray(payload["weights"], dtype=np.float64),
                    bias_np=np.asarray(payload["bias"], dtype=np.float64),
                    in_dim=in_dim,
                    device=device,
                    dtype=dtype,
                    posterior_temperature=float(posterior_temperature),
                )
            return PseudoLabelPreparedData(
                posterior=posterior,
                selected_k=selected_k,
                method=method,
                posterior_temperature=float(posterior_temperature),
                class_labels=class_labels,
                bic_by_k={},
                stability_by_k={},
                cache_path=str(cache_path),
                cache_hit=True,
                supervised_kept_class_labels=(
                    supervised_kept_class_labels if supervised_kept_class_labels else None
                ),
                supervised_dropped_class_labels=(
                    supervised_dropped_class_labels
                    if supervised_dropped_class_labels
                    else []
                ),
                supervised_train_count=(
                    None
                    if supervised_train_count_raw is None
                    else int(supervised_train_count_raw)
                ),
                supervised_val_count=(
                    None
                    if supervised_val_count_raw is None
                    else int(supervised_val_count_raw)
                ),
                supervised_val_split_used=(
                    None
                    if supervised_val_split_used_raw is None
                    else bool(supervised_val_split_used_raw)
                ),
                supervised_val_split_fallback=(
                    None
                    if supervised_val_split_fallback_raw is None
                    else bool(supervised_val_split_fallback_raw)
                ),
                supervised_epochs_trained=(
                    None
                    if supervised_epochs_trained_raw is None
                    else int(supervised_epochs_trained_raw)
                ),
                supervised_best_epoch=(
                    None if supervised_best_epoch_raw is None else int(supervised_best_epoch_raw)
                ),
                supervised_best_val_loss=(
                    None
                    if supervised_best_val_loss_raw is None
                    else float(supervised_best_val_loss_raw)
                ),
                supervised_early_stop_triggered=(
                    None
                    if supervised_early_stop_triggered_raw is None
                    else bool(supervised_early_stop_triggered_raw)
                ),
            )

        if method == "supervised_mlp":
            (
                posterior,
                selected_k,
                class_labels,
                dropped_class_labels,
                model,
                in_dim,
                hidden_dims,
                activation,
                supervised_train_count,
                supervised_val_count,
                supervised_val_split_used,
                supervised_val_split_fallback,
                supervised_epochs_trained,
                supervised_best_epoch,
                supervised_best_val_loss,
                supervised_early_stop_triggered,
            ) = _fit_supervised_mlp_posterior(
                features_np=features_np,
                labels_np=supervised_labels_np,
                pseudo_cfg=pseudo_cfg,
                posterior_temperature=float(posterior_temperature),
                device=device,
                dtype=dtype,
            )
            if cache_enabled and cache_path is not None:
                state_dict_cpu = {
                    name: tensor.detach().to(device="cpu", dtype=torch.float32).clone()
                    for name, tensor in model.state_dict().items()
                }
                torch.save(
                    {
                        "signature": signature,
                        "method": "supervised_mlp",
                        "in_dim": int(in_dim),
                        "selected_k": int(selected_k),
                        "class_labels": [str(label) for label in class_labels],
                        "supervised_kept_class_labels": [str(label) for label in class_labels],
                        "supervised_dropped_class_labels": [
                            str(label) for label in dropped_class_labels
                        ],
                        "supervised_train_count": int(supervised_train_count),
                        "supervised_val_count": int(supervised_val_count),
                        "supervised_val_split_used": bool(supervised_val_split_used),
                        "supervised_val_split_fallback": bool(supervised_val_split_fallback),
                        "supervised_epochs_trained": int(supervised_epochs_trained),
                        "supervised_best_epoch": (
                            None
                            if supervised_best_epoch is None
                            else int(supervised_best_epoch)
                        ),
                        "supervised_best_val_loss": (
                            None
                            if supervised_best_val_loss is None
                            else float(supervised_best_val_loss)
                        ),
                        "supervised_early_stop_triggered": bool(
                            supervised_early_stop_triggered
                        ),
                        "hidden_dims": [int(value) for value in hidden_dims],
                        "activation": str(activation),
                        "state_dict": state_dict_cpu,
                    },
                    cache_path,
                )
            supervised_kept_class_labels = [str(label) for label in class_labels]
            supervised_dropped_class_labels = [str(label) for label in dropped_class_labels]
        else:
            (
                posterior,
                selected_k,
                class_labels,
                in_dim,
                weights_np,
                bias_np,
            ) = _fit_supervised_logreg_posterior(
                features_np=features_np,
                labels_np=supervised_labels_np,
                pseudo_cfg=pseudo_cfg,
                posterior_temperature=float(posterior_temperature),
                device=device,
                dtype=dtype,
            )
            supervised_kept_class_labels = [str(label) for label in class_labels]
            supervised_dropped_class_labels = []
            supervised_train_count = int(np.asarray(supervised_labels_np).shape[0])
            supervised_val_count = 0
            supervised_val_split_used = False
            supervised_val_split_fallback = False
            supervised_epochs_trained = None
            supervised_best_epoch = None
            supervised_best_val_loss = None
            supervised_early_stop_triggered = False
            if cache_enabled and cache_path is not None:
                torch.save(
                    {
                        "signature": signature,
                        "method": "supervised_logreg",
                        "in_dim": int(in_dim),
                        "selected_k": int(selected_k),
                        "class_labels": [str(label) for label in class_labels],
                        "weights": np.asarray(weights_np, dtype=np.float64),
                        "bias": np.asarray(bias_np, dtype=np.float64),
                        "supervised_kept_class_labels": [str(label) for label in class_labels],
                        "supervised_dropped_class_labels": [],
                        "supervised_train_count": int(supervised_train_count),
                        "supervised_val_count": int(supervised_val_count),
                        "supervised_val_split_used": bool(supervised_val_split_used),
                        "supervised_val_split_fallback": bool(supervised_val_split_fallback),
                        "supervised_epochs_trained": None,
                        "supervised_best_epoch": None,
                        "supervised_best_val_loss": None,
                        "supervised_early_stop_triggered": bool(
                            supervised_early_stop_triggered
                        ),
                    },
                    cache_path,
                )
        return PseudoLabelPreparedData(
            posterior=posterior,
            selected_k=int(selected_k),
            method=method,
            posterior_temperature=float(posterior_temperature),
            class_labels=class_labels,
            bic_by_k={},
            stability_by_k={},
            cache_path=(None if cache_path is None else str(cache_path)),
            cache_hit=False,
            supervised_kept_class_labels=supervised_kept_class_labels,
            supervised_dropped_class_labels=supervised_dropped_class_labels,
            supervised_train_count=supervised_train_count,
            supervised_val_count=supervised_val_count,
            supervised_val_split_used=supervised_val_split_used,
            supervised_val_split_fallback=supervised_val_split_fallback,
            supervised_epochs_trained=supervised_epochs_trained,
            supervised_best_epoch=supervised_best_epoch,
            supervised_best_val_loss=supervised_best_val_loss,
            supervised_early_stop_triggered=supervised_early_stop_triggered,
        )
    if method != "gmm":
        raise ValueError(
            f"Unsupported pseudo_labels.method '{method}'. "
            "Expected one of: gmm, supervised_mlp, supervised_logreg."
        )

    try:
        from sklearn.metrics import adjusted_rand_score
        from sklearn.mixture import GaussianMixture
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Pseudo-label constraints with method='gmm' require scikit-learn. "
            "Install scikit-learn or switch to method='supervised_mlp' or "
            "'supervised_logreg'."
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

    signature = _pseudo_signature(
        dataset_path=dataset_path,
        features_np=features_np.astype(np.float64, copy=False),
        time_indices=time_indices,
        method=method,
        supervised_labels_np=None,
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
            posterior_temperature=float(posterior_temperature),
        )
        return PseudoLabelPreparedData(
            posterior=posterior,
            selected_k=selected_k,
            method="gmm",
            posterior_temperature=float(posterior_temperature),
            class_labels=None,
            bic_by_k={int(k): float(v) for k, v in bic_by_k_raw.items()},
            stability_by_k={int(k): float(v) for k, v in stability_by_k_raw.items()},
            cache_path=str(cache_path),
            cache_hit=True,
            supervised_kept_class_labels=None,
            supervised_dropped_class_labels=None,
            supervised_train_count=None,
            supervised_val_count=None,
            supervised_val_split_used=None,
            supervised_val_split_fallback=None,
            supervised_epochs_trained=None,
            supervised_best_epoch=None,
            supervised_best_val_loss=None,
            supervised_early_stop_triggered=None,
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
        posterior_temperature=float(posterior_temperature),
    )

    if cache_enabled and cache_path is not None:
        torch.save(
            {
                "signature": signature,
                "method": "gmm",
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
        method="gmm",
        posterior_temperature=float(posterior_temperature),
        class_labels=None,
        bic_by_k={int(k): float(v) for k, v in bic_by_k.items()},
        stability_by_k={int(k): float(v) for k, v in stability_by_k.items()},
        cache_path=None if cache_path is None else str(cache_path),
        cache_hit=False,
        supervised_kept_class_labels=None,
        supervised_dropped_class_labels=None,
        supervised_train_count=None,
        supervised_val_count=None,
        supervised_val_split_used=None,
        supervised_val_split_fallback=None,
        supervised_epochs_trained=None,
        supervised_best_epoch=None,
        supervised_best_val_loss=None,
        supervised_early_stop_triggered=None,
    )
