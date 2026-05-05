from __future__ import annotations

from pathlib import Path

import numpy as np
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
