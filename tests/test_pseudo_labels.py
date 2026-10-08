from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from cfm_project.pseudo_labels import prepare_pseudo_labels


def _write_mixture_npz(path: Path, dim: int = 5, n_per_cluster: int = 60) -> tuple[Path, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2026)
    centers = np.array(
        [
            [-2.0, -1.5, 0.5, 0.0, 1.0],
            [0.5, 2.0, -1.0, 1.5, -0.5],
            [2.5, -0.5, 1.5, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    chunks = [
        rng.normal(loc=center, scale=0.20, size=(int(n_per_cluster), int(dim)))
        for center in centers
    ]
    pcs = np.concatenate(chunks, axis=0).astype(np.float32)
    n = pcs.shape[0]
    sample_labels = np.tile(np.array([0, 1, 2, 3, 4], dtype=np.int64), int(np.ceil(n / 5)))[:n]
    np.savez(path, pcs=pcs, sample_labels=sample_labels)
    return path, pcs.astype(np.float64), sample_labels.astype(np.int64)


def _single_cfg(dataset_path: Path, cache_dir: Path, force_recompute: bool = False) -> dict:
    return {
        "path": str(dataset_path),
        "max_dim": 5,
        "whiten": True,
        "embed_key_npz": "pcs",
        "label_key_npz": "sample_labels",
        "pseudo_labels": {
            "enabled": True,
            "k_min": 2,
            "k_max": 4,
            "seeds": [1, 3, 7],
            "n_init": 2,
            "max_iter": 200,
            "tol": 1.0e-3,
            "reg_covar": 1.0e-6,
            "stability_threshold": 0.2,
            "cache_enabled": True,
            "cache_dir": str(cache_dir),
            "force_recompute": bool(force_recompute),
        },
    }


def _single_cfg_supervised(
    dataset_path: Path,
    cache_dir: Path,
    posterior_temperature: float = 1.0,
    force_recompute: bool = False,
    method: str = "supervised_mlp",
    supervised_min_class_count: int = 1,
    supervised_val_fraction: float = 0.0,
    supervised_split_seed: int | None = None,
    supervised_early_stopping_patience: int = 20,
    supervised_early_stopping_min_epochs: int = 20,
    supervised_early_stopping_min_delta: float = 0.0,
    supervised_logreg_c: float = 1.0,
    supervised_logreg_penalty: str = "l2",
    supervised_logreg_max_iter: int = 1000,
    supervised_logreg_class_weight: str | None = "balanced",
    supervised_logreg_seed: int = 7,
) -> dict:
    return {
        "path": str(dataset_path),
        "max_dim": 5,
        "whiten": True,
        "embed_key_npz": "pcs",
        "label_key_npz": "sample_labels",
        "pseudo_labels": {
            "enabled": True,
            "method": str(method),
            "posterior_temperature": float(posterior_temperature),
            "supervised_hidden_dims": [32, 32],
            "supervised_activation": "silu",
            "supervised_epochs": 60,
            "supervised_batch_size": 64,
            "supervised_lr": 1.0e-3,
            "supervised_weight_decay": 1.0e-4,
            "supervised_seed": 7,
            "supervised_min_class_count": int(supervised_min_class_count),
            "supervised_val_fraction": float(supervised_val_fraction),
            "supervised_split_seed": (
                None if supervised_split_seed is None else int(supervised_split_seed)
            ),
            "supervised_early_stopping_patience": int(supervised_early_stopping_patience),
            "supervised_early_stopping_min_epochs": int(supervised_early_stopping_min_epochs),
            "supervised_early_stopping_min_delta": float(supervised_early_stopping_min_delta),
            "supervised_logreg_c": float(supervised_logreg_c),
            "supervised_logreg_penalty": str(supervised_logreg_penalty),
            "supervised_logreg_max_iter": int(supervised_logreg_max_iter),
            "supervised_logreg_class_weight": supervised_logreg_class_weight,
            "supervised_logreg_seed": int(supervised_logreg_seed),
            "cache_enabled": True,
            "cache_dir": str(cache_dir),
            "force_recompute": bool(force_recompute),
        },
    }


def test_prepare_pseudo_labels_simplex_and_cache_reuse(tmp_path: Path) -> None:
    dataset_path, features_np, time_indices = _write_mixture_npz(tmp_path / "mixture.npz")
    single_cfg = _single_cfg(dataset_path=dataset_path, cache_dir=tmp_path / "pseudo_cache")

    first = prepare_pseudo_labels(
        dataset_path=str(dataset_path),
        features_np=features_np,
        time_indices=time_indices,
        single_cfg=single_cfg,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    second = prepare_pseudo_labels(
        dataset_path=str(dataset_path),
        features_np=features_np,
        time_indices=time_indices,
        single_cfg=single_cfg,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert first is not None
    assert second is not None
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.cache_path is not None
    assert second.cache_path == first.cache_path
    assert first.selected_k == second.selected_k
    assert first.selected_k == 3

    x = torch.as_tensor(features_np[:32], dtype=torch.float32)
    probs = first.posterior(x)
    assert probs.shape == (32, first.selected_k)
    row_sums = probs.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5)
    assert torch.all(probs >= 0.0)


def test_prepare_pseudo_labels_k_selection_deterministic_under_fixed_seeds(tmp_path: Path) -> None:
    dataset_path, features_np, time_indices = _write_mixture_npz(tmp_path / "mixture.npz")
    cache_dir = tmp_path / "pseudo_cache"
    single_cfg = _single_cfg(dataset_path=dataset_path, cache_dir=cache_dir)
    single_cfg_force = _single_cfg(
        dataset_path=dataset_path,
        cache_dir=cache_dir,
        force_recompute=True,
    )

    first = prepare_pseudo_labels(
        dataset_path=str(dataset_path),
        features_np=features_np,
        time_indices=time_indices,
        single_cfg=single_cfg,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    second = prepare_pseudo_labels(
        dataset_path=str(dataset_path),
        features_np=features_np,
        time_indices=time_indices,
        single_cfg=single_cfg_force,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert first is not None
    assert second is not None
    assert first.selected_k == second.selected_k
    assert first.bic_by_k == second.bic_by_k
    assert first.stability_by_k == second.stability_by_k


def test_prepare_pseudo_labels_cache_key_changes_with_fit_subset(tmp_path: Path) -> None:
    dataset_path, features_np, time_indices = _write_mixture_npz(tmp_path / "mixture_subset.npz")
    cache_dir = tmp_path / "pseudo_cache_subset"
    single_cfg = _single_cfg(dataset_path=dataset_path, cache_dir=cache_dir)

    full = prepare_pseudo_labels(
        dataset_path=str(dataset_path),
        features_np=features_np,
        time_indices=time_indices,
        single_cfg=single_cfg,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    subset_mask = (time_indices % 2) == 0
    subset = prepare_pseudo_labels(
        dataset_path=str(dataset_path),
        features_np=features_np[subset_mask],
        time_indices=time_indices[subset_mask],
        single_cfg=single_cfg,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert full is not None
    assert subset is not None
    assert full.cache_path is not None
    assert subset.cache_path is not None
    assert full.cache_path != subset.cache_path


def test_prepare_pseudo_labels_supervised_mlp_trains_once_and_reuses_cache(tmp_path: Path) -> None:
    dataset_path, features_np, time_indices = _write_mixture_npz(tmp_path / "mixture_supervised.npz")
    supervised_labels = np.where(
        features_np[:, 0] < -0.5,
        "state_a",
        np.where(features_np[:, 0] > 1.0, "state_c", "state_b"),
    )
    single_cfg = _single_cfg_supervised(
        dataset_path=dataset_path,
        cache_dir=tmp_path / "pseudo_supervised_cache",
    )

    first = prepare_pseudo_labels(
        dataset_path=str(dataset_path),
        features_np=features_np,
        time_indices=time_indices,
        supervised_labels_np=supervised_labels,
        single_cfg=single_cfg,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    second = prepare_pseudo_labels(
        dataset_path=str(dataset_path),
        features_np=features_np,
        time_indices=time_indices,
        supervised_labels_np=supervised_labels,
        single_cfg=single_cfg,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert first is not None
    assert second is not None
    assert first.method == "supervised_mlp"
    assert second.method == "supervised_mlp"
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.cache_path is not None
    assert second.cache_path == first.cache_path
    assert first.class_labels is not None
    assert second.class_labels == first.class_labels
    assert first.selected_k == len(np.unique(supervised_labels))

    x = torch.as_tensor(features_np[:24], dtype=torch.float32).requires_grad_(True)
    probs = second.posterior(x)
    assert probs.shape == (24, int(second.selected_k))
    assert torch.allclose(
        probs.sum(dim=1),
        torch.ones(probs.shape[0], dtype=probs.dtype),
        atol=1e-5,
        rtol=1e-5,
    )
    grad = torch.autograd.grad(probs[:, 0].sum(), x, retain_graph=False, create_graph=False)[0]
    assert grad is not None
    assert torch.isfinite(grad).all()


def test_prepare_pseudo_labels_supervised_logreg_trains_once_and_reuses_cache(tmp_path: Path) -> None:
    dataset_path, features_np, time_indices = _write_mixture_npz(tmp_path / "mixture_supervised_logreg.npz")
    supervised_labels = np.where(
        features_np[:, 0] < -0.5,
        "state_a",
        np.where(features_np[:, 0] > 1.0, "state_c", "state_b"),
    )
    single_cfg = _single_cfg_supervised(
        dataset_path=dataset_path,
        cache_dir=tmp_path / "pseudo_supervised_logreg_cache",
        method="supervised_logreg",
        supervised_logreg_c=1.0,
        supervised_logreg_penalty="l2",
        supervised_logreg_max_iter=800,
        supervised_logreg_class_weight="balanced",
        supervised_logreg_seed=9,
    )

    first = prepare_pseudo_labels(
        dataset_path=str(dataset_path),
        features_np=features_np,
        time_indices=time_indices,
        supervised_labels_np=supervised_labels,
        single_cfg=single_cfg,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    second = prepare_pseudo_labels(
        dataset_path=str(dataset_path),
        features_np=features_np,
        time_indices=time_indices,
        supervised_labels_np=supervised_labels,
        single_cfg=single_cfg,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert first is not None
    assert second is not None
    assert first.method == "supervised_logreg"
    assert second.method == "supervised_logreg"
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.cache_path is not None
    assert second.cache_path == first.cache_path
    assert first.class_labels is not None
    assert second.class_labels == first.class_labels
    assert first.selected_k == len(np.unique(supervised_labels))

    x = torch.as_tensor(features_np[:24], dtype=torch.float32).requires_grad_(True)
    probs = second.posterior(x)
    assert probs.shape == (24, int(second.selected_k))
    assert torch.isfinite(probs).all()
    assert torch.allclose(
        probs.sum(dim=1),
        torch.ones(probs.shape[0], dtype=probs.dtype),
        atol=1e-5,
        rtol=1e-5,
    )
    grad = torch.autograd.grad(probs[:, 0].sum(), x, retain_graph=False, create_graph=False)[0]
    assert grad is not None
    assert torch.isfinite(grad).all()


def test_prepare_pseudo_labels_supervised_logreg_cache_key_changes_with_hparams(tmp_path: Path) -> None:
    dataset_path, features_np, time_indices = _write_mixture_npz(tmp_path / "mixture_supervised_logreg_hparams.npz")
    supervised_labels = np.where(
        features_np[:, 0] < -0.5,
        "state_a",
        np.where(features_np[:, 0] > 1.0, "state_c", "state_b"),
    )
    cache_dir = tmp_path / "pseudo_supervised_logreg_hparam_cache"
    cfg_c1 = _single_cfg_supervised(
        dataset_path=dataset_path,
        cache_dir=cache_dir,
        method="supervised_logreg",
        supervised_logreg_c=1.0,
    )
    cfg_c2 = _single_cfg_supervised(
        dataset_path=dataset_path,
        cache_dir=cache_dir,
        method="supervised_logreg",
        supervised_logreg_c=2.0,
    )

    run_c1 = prepare_pseudo_labels(
        dataset_path=str(dataset_path),
        features_np=features_np,
        time_indices=time_indices,
        supervised_labels_np=supervised_labels,
        single_cfg=cfg_c1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    run_c2 = prepare_pseudo_labels(
        dataset_path=str(dataset_path),
        features_np=features_np,
        time_indices=time_indices,
        supervised_labels_np=supervised_labels,
        single_cfg=cfg_c2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert run_c1 is not None
    assert run_c2 is not None
    assert run_c1.cache_path is not None
    assert run_c2.cache_path is not None
    assert run_c1.cache_path != run_c2.cache_path


def test_prepare_pseudo_labels_supervised_temperature_softens_posterior(tmp_path: Path) -> None:
    dataset_path, features_np, time_indices = _write_mixture_npz(tmp_path / "mixture_supervised_temp.npz")
    supervised_labels = np.where(
        features_np[:, 0] < -0.5,
        "state_a",
        np.where(features_np[:, 0] > 1.0, "state_c", "state_b"),
    )
    cache_dir = tmp_path / "pseudo_supervised_temp_cache"
    cfg_t1 = _single_cfg_supervised(
        dataset_path=dataset_path,
        cache_dir=cache_dir,
        posterior_temperature=1.0,
    )
    cfg_t4 = _single_cfg_supervised(
        dataset_path=dataset_path,
        cache_dir=cache_dir,
        posterior_temperature=4.0,
    )

    t1 = prepare_pseudo_labels(
        dataset_path=str(dataset_path),
        features_np=features_np,
        time_indices=time_indices,
        supervised_labels_np=supervised_labels,
        single_cfg=cfg_t1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    t4 = prepare_pseudo_labels(
        dataset_path=str(dataset_path),
        features_np=features_np,
        time_indices=time_indices,
        supervised_labels_np=supervised_labels,
        single_cfg=cfg_t4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert t1 is not None
    assert t4 is not None
    assert t1.method == "supervised_mlp"
    assert t4.method == "supervised_mlp"
    assert t4.cache_hit is True
    assert t1.cache_path == t4.cache_path
    assert t1.posterior_temperature == 1.0
    assert t4.posterior_temperature == 4.0

    x = torch.as_tensor(features_np[:64], dtype=torch.float32)
    p1 = t1.posterior(x)
    p4 = t4.posterior(x)
    max1 = p1.max(dim=1).values.mean().item()
    max4 = p4.max(dim=1).values.mean().item()
    ent1 = (-(p1 * torch.log(torch.clamp(p1, min=1e-8))).sum(dim=1)).mean().item()
    ent4 = (-(p4 * torch.log(torch.clamp(p4, min=1e-8))).sum(dim=1)).mean().item()
    assert max4 < max1
    assert ent4 > ent1


def test_prepare_pseudo_labels_supervised_logreg_temperature_softens_posterior(tmp_path: Path) -> None:
    dataset_path, features_np, time_indices = _write_mixture_npz(tmp_path / "mixture_supervised_logreg_temp.npz")
    supervised_labels = np.where(
        features_np[:, 0] < -0.5,
        "state_a",
        np.where(features_np[:, 0] > 1.0, "state_c", "state_b"),
    )
    cache_dir = tmp_path / "pseudo_supervised_logreg_temp_cache"
    cfg_t1 = _single_cfg_supervised(
        dataset_path=dataset_path,
        cache_dir=cache_dir,
        posterior_temperature=1.0,
        method="supervised_logreg",
    )
    cfg_t4 = _single_cfg_supervised(
        dataset_path=dataset_path,
        cache_dir=cache_dir,
        posterior_temperature=4.0,
        method="supervised_logreg",
    )
    t1 = prepare_pseudo_labels(
        dataset_path=str(dataset_path),
        features_np=features_np,
        time_indices=time_indices,
        supervised_labels_np=supervised_labels,
        single_cfg=cfg_t1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    t4 = prepare_pseudo_labels(
        dataset_path=str(dataset_path),
        features_np=features_np,
        time_indices=time_indices,
        supervised_labels_np=supervised_labels,
        single_cfg=cfg_t4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert t1 is not None
    assert t4 is not None
    assert t1.method == "supervised_logreg"
    assert t4.method == "supervised_logreg"
    assert t4.cache_hit is True
    assert t1.cache_path == t4.cache_path

    x = torch.as_tensor(features_np[:64], dtype=torch.float32)
    p1 = t1.posterior(x)
    p4 = t4.posterior(x)
    max1 = p1.max(dim=1).values.mean().item()
    max4 = p4.max(dim=1).values.mean().item()
    ent1 = (-(p1 * torch.log(torch.clamp(p1, min=1e-8))).sum(dim=1)).mean().item()
    ent4 = (-(p4 * torch.log(torch.clamp(p4, min=1e-8))).sum(dim=1)).mean().item()
    assert max4 < max1
    assert ent4 > ent1


def test_prepare_pseudo_labels_invalid_temperature_raises(tmp_path: Path) -> None:
    dataset_path, features_np, time_indices = _write_mixture_npz(tmp_path / "mixture_invalid_temp.npz")
    supervised_labels = np.where(
        features_np[:, 0] < -0.5,
        "state_a",
        np.where(features_np[:, 0] > 1.0, "state_c", "state_b"),
    )
    bad_cfg = _single_cfg_supervised(
        dataset_path=dataset_path,
        cache_dir=tmp_path / "pseudo_invalid_temp_cache",
        posterior_temperature=0.0,
    )
    with pytest.raises(ValueError, match="posterior_temperature"):
        prepare_pseudo_labels(
            dataset_path=str(dataset_path),
            features_np=features_np,
            time_indices=time_indices,
            supervised_labels_np=supervised_labels,
            single_cfg=bad_cfg,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_prepare_pseudo_labels_supervised_mlp_drops_rare_class_and_keeps_simplex(
    tmp_path: Path,
) -> None:
    dataset_path, features_np, time_indices = _write_mixture_npz(tmp_path / "mixture_supervised_rare.npz")
    supervised_labels = np.where(
        features_np[:, 0] < -0.5,
        "state_a",
        np.where(features_np[:, 0] > 1.0, "state_c", "state_b"),
    ).astype(object)
    supervised_labels[0] = "state_rare_singleton"
    cfg = _single_cfg_supervised(
        dataset_path=dataset_path,
        cache_dir=tmp_path / "pseudo_supervised_rare_cache",
        supervised_min_class_count=2,
        supervised_val_fraction=0.2,
        supervised_split_seed=17,
    )

    prepared = prepare_pseudo_labels(
        dataset_path=str(dataset_path),
        features_np=features_np,
        time_indices=time_indices,
        supervised_labels_np=supervised_labels,
        single_cfg=cfg,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert prepared is not None
    assert prepared.method == "supervised_mlp"
    assert prepared.class_labels is not None
    assert "state_rare_singleton" not in prepared.class_labels
    assert prepared.supervised_kept_class_labels is not None
    assert "state_rare_singleton" not in prepared.supervised_kept_class_labels
    assert prepared.supervised_dropped_class_labels is not None
    assert "state_rare_singleton" in prepared.supervised_dropped_class_labels
    assert prepared.selected_k == len(prepared.class_labels)

    x = torch.as_tensor(features_np[:24], dtype=torch.float32).requires_grad_(True)
    probs = prepared.posterior(x)
    assert probs.shape[0] == 24
    assert probs.shape[1] == int(prepared.selected_k)
    assert torch.allclose(
        probs.sum(dim=1),
        torch.ones(probs.shape[0], dtype=probs.dtype),
        atol=1e-5,
        rtol=1e-5,
    )
    grad = torch.autograd.grad(probs[:, 0].sum(), x, retain_graph=False, create_graph=False)[0]
    assert grad is not None
    assert torch.isfinite(grad).all()


def test_prepare_pseudo_labels_supervised_mlp_split_and_early_stop_metadata(
    tmp_path: Path,
) -> None:
    dataset_path, features_np, time_indices = _write_mixture_npz(tmp_path / "mixture_supervised_es.npz")
    supervised_labels = np.where(
        features_np[:, 0] < -0.5,
        "state_a",
        np.where(features_np[:, 0] > 1.0, "state_c", "state_b"),
    )
    cfg = _single_cfg_supervised(
        dataset_path=dataset_path,
        cache_dir=tmp_path / "pseudo_supervised_es_cache",
        supervised_val_fraction=0.2,
        supervised_split_seed=13,
        supervised_early_stopping_patience=2,
        supervised_early_stopping_min_epochs=3,
        supervised_early_stopping_min_delta=1.0e6,
    )
    cfg["pseudo_labels"]["supervised_epochs"] = 40

    prepared = prepare_pseudo_labels(
        dataset_path=str(dataset_path),
        features_np=features_np,
        time_indices=time_indices,
        supervised_labels_np=supervised_labels,
        single_cfg=cfg,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert prepared is not None
    assert prepared.method == "supervised_mlp"
    assert prepared.supervised_train_count is not None
    assert prepared.supervised_train_count > 0
    assert prepared.supervised_val_count is not None
    assert prepared.supervised_val_count > 0
    assert prepared.supervised_val_split_used is True
    assert prepared.supervised_val_split_fallback is False
    assert prepared.supervised_epochs_trained is not None
    assert prepared.supervised_epochs_trained < int(cfg["pseudo_labels"]["supervised_epochs"])
    assert prepared.supervised_best_epoch is not None
    assert prepared.supervised_best_epoch >= 1
    assert prepared.supervised_best_val_loss is not None
    assert np.isfinite(prepared.supervised_best_val_loss)
    assert prepared.supervised_early_stop_triggered is True


def test_prepare_pseudo_labels_supervised_mlp_cache_key_changes_with_split_knobs(
    tmp_path: Path,
) -> None:
    dataset_path, features_np, time_indices = _write_mixture_npz(
        tmp_path / "mixture_supervised_split_cache_key.npz"
    )
    supervised_labels = np.where(
        features_np[:, 0] < -0.5,
        "state_a",
        np.where(features_np[:, 0] > 1.0, "state_c", "state_b"),
    )
    cache_dir = tmp_path / "pseudo_supervised_split_cache_key"
    cfg_full = _single_cfg_supervised(
        dataset_path=dataset_path,
        cache_dir=cache_dir,
        supervised_min_class_count=1,
        supervised_val_fraction=0.0,
    )
    cfg_val = _single_cfg_supervised(
        dataset_path=dataset_path,
        cache_dir=cache_dir,
        supervised_min_class_count=1,
        supervised_val_fraction=0.2,
        supervised_split_seed=9,
    )
    cfg_min = _single_cfg_supervised(
        dataset_path=dataset_path,
        cache_dir=cache_dir,
        supervised_min_class_count=2,
        supervised_val_fraction=0.0,
    )

    run_full = prepare_pseudo_labels(
        dataset_path=str(dataset_path),
        features_np=features_np,
        time_indices=time_indices,
        supervised_labels_np=supervised_labels,
        single_cfg=cfg_full,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    run_val = prepare_pseudo_labels(
        dataset_path=str(dataset_path),
        features_np=features_np,
        time_indices=time_indices,
        supervised_labels_np=supervised_labels,
        single_cfg=cfg_val,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    run_min = prepare_pseudo_labels(
        dataset_path=str(dataset_path),
        features_np=features_np,
        time_indices=time_indices,
        supervised_labels_np=supervised_labels,
        single_cfg=cfg_min,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert run_full is not None
    assert run_val is not None
    assert run_min is not None
    assert run_full.cache_path is not None
    assert run_val.cache_path is not None
    assert run_min.cache_path is not None
    assert run_full.cache_path != run_val.cache_path
    assert run_full.cache_path != run_min.cache_path
    assert run_val.cache_path != run_min.cache_path


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"supervised_logreg_c": 0.0}, "supervised_logreg_c"),
        ({"supervised_logreg_penalty": "l1"}, "supervised_logreg_penalty"),
        ({"supervised_logreg_max_iter": 0}, "supervised_logreg_max_iter"),
    ],
)
def test_prepare_pseudo_labels_invalid_supervised_logreg_config_raises(
    tmp_path: Path,
    override: dict[str, object],
    match: str,
) -> None:
    dataset_path, features_np, time_indices = _write_mixture_npz(tmp_path / f"mixture_bad_logreg_{match}.npz")
    supervised_labels = np.where(
        features_np[:, 0] < -0.5,
        "state_a",
        np.where(features_np[:, 0] > 1.0, "state_c", "state_b"),
    )
    cfg = _single_cfg_supervised(
        dataset_path=dataset_path,
        cache_dir=tmp_path / f"pseudo_bad_logreg_cache_{match}",
        method="supervised_logreg",
    )
    cfg["pseudo_labels"].update(override)
    with pytest.raises(ValueError, match=match):
        prepare_pseudo_labels(
            dataset_path=str(dataset_path),
            features_np=features_np,
            time_indices=time_indices,
            supervised_labels_np=supervised_labels,
            single_cfg=cfg,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
