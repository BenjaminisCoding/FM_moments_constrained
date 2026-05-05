from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cfm_project.single_cell_data import prepare_single_cell_problem_and_targets


def _write_synthetic_single_cell_npz(path: Path, dim: int = 5, n_per_time: int = 18) -> Path:
    rng = np.random.default_rng(1234)
    time_labels = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    labels = np.repeat(time_labels, int(n_per_time))
    perm = rng.permutation(labels.shape[0])
    shuffled_labels = labels[perm]
    noise = rng.normal(loc=0.0, scale=0.25, size=(labels.shape[0], int(dim)))
    pcs = noise + shuffled_labels.reshape(-1, 1) * 0.15
    np.savez(path, pcs=pcs.astype(np.float32), sample_labels=shuffled_labels)
    return path


def test_prepare_single_cell_strict_leaveout_excludes_holdout_constraint_time(tmp_path: Path) -> None:
    dataset_path = _write_synthetic_single_cell_npz(tmp_path / "eb_like.npz")
    prepared = prepare_single_cell_problem_and_targets(
        data_cfg={
            "label": "single_cell_eb_5d",
            "dim": 5,
            "constraint_time_policy": "observed_nonendpoint_excluding_holdout",
            "single_cell": {
                "path": str(dataset_path),
                "max_dim": 5,
                "whiten": True,
            },
        },
        experiment_cfg={
            "protocol": "strict_leaveout",
            "holdout_index": 2,
            "holdout_indices": [1, 2, 3],
        },
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert prepared.all_time_indices == [0, 1, 2, 3, 4]
    assert prepared.constraint_time_indices == [1, 3]
    assert prepared.constraint_times == [0.25, 0.75]
    assert prepared.eval_times == [0.25, 0.5, 0.75]
    assert prepared.holdout_index == 2
    assert prepared.holdout_time == 0.5
    assert set(prepared.targets.keys()) == {0.25, 0.75}
    assert prepared.problem.x0_pool.shape[1] == 5
    assert prepared.problem.x1_pool.shape[1] == 5
    sampled = prepared.target_sampler(0.50, 7, generator=torch.Generator().manual_seed(9))
    assert sampled.shape == (7, 5)


def test_prepare_single_cell_constraint_policy_all_keeps_all_intermediates(tmp_path: Path) -> None:
    dataset_path = _write_synthetic_single_cell_npz(tmp_path / "eb_like.npz")
    prepared = prepare_single_cell_problem_and_targets(
        data_cfg={
            "label": "single_cell_eb_5d",
            "dim": 5,
            "constraint_time_policy": "observed_nonendpoint_all",
            "single_cell": {
                "path": str(dataset_path),
                "max_dim": 5,
                "whiten": True,
            },
        },
        experiment_cfg={
            "protocol": "strict_leaveout",
            "holdout_index": 2,
            "holdout_indices": [1, 2, 3],
        },
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert prepared.constraint_time_indices == [1, 2, 3]
    assert prepared.constraint_times == [0.25, 0.5, 0.75]
    assert prepared.eval_times == [0.25, 0.5, 0.75]
    assert set(prepared.targets.keys()) == {0.25, 0.5, 0.75}


def test_prepare_single_cell_explicit_constraint_eval_and_pseudo_fit_time_overrides(tmp_path: Path) -> None:
    dataset_path = _write_synthetic_single_cell_npz(tmp_path / "eb_like_overrides.npz", n_per_time=12)
    prepared = prepare_single_cell_problem_and_targets(
        data_cfg={
            "label": "single_cell_eb_5d",
            "family": "single_cell",
            "dim": 5,
            "constraint_time_policy": "observed_nonendpoint_all",
            "single_cell": {
                "path": str(dataset_path),
                "max_dim": 5,
                "whiten": True,
                "constraint_times_normalized": [0.5],
                "eval_times_normalized": [0.25, 0.5, 0.75],
                "pseudo_labels": {
                    "enabled": True,
                    "fit_times_normalized": [0.5],
                    "k_min": 2,
                    "k_max": 4,
                    "seeds": [3, 5],
                    "n_init": 1,
                    "max_iter": 100,
                    "tol": 1.0e-3,
                    "reg_covar": 1.0e-6,
                    "stability_threshold": 0.0,
                    "cache_enabled": True,
                    "cache_dir": str(tmp_path / "pseudo_cache"),
                    "force_recompute": False,
                },
            },
        },
        experiment_cfg={
            "protocol": "no_leaveout",
        },
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert prepared.constraint_times == [0.5]
    assert prepared.eval_times == [0.25, 0.5, 0.75]
    assert set(prepared.targets.keys()) == {0.5}
    assert prepared.holdout_index is None
    assert prepared.holdout_time is None
    assert prepared.pseudo_fit_times == [0.5]
    assert prepared.pseudo_fit_sample_count == 12
    assert prepared.pseudo_targets is not None
    assert set(prepared.pseudo_targets.keys()) == {0.5}


def test_prepare_single_cell_explicit_times_fail_when_unobserved(tmp_path: Path) -> None:
    dataset_path = _write_synthetic_single_cell_npz(tmp_path / "eb_like_bad_time.npz", n_per_time=10)
    try:
        prepare_single_cell_problem_and_targets(
            data_cfg={
                "label": "single_cell_eb_5d",
                "family": "single_cell",
                "dim": 5,
                "constraint_time_policy": "observed_nonendpoint_all",
                "single_cell": {
                    "path": str(dataset_path),
                    "max_dim": 5,
                    "whiten": True,
                    "constraint_times_normalized": [0.6],
                },
            },
            experiment_cfg={"protocol": "no_leaveout"},
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    except ValueError as exc:
        assert "is not observed in dataset times" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unobserved explicit normalized times.")


def test_prepare_single_cell_ot_global_builds_and_reuses_cached_plan(tmp_path: Path) -> None:
    dataset_path = _write_synthetic_single_cell_npz(tmp_path / "eb_like_ot_global.npz", n_per_time=10)
    cache_dir = tmp_path / "ot_cache"
    data_cfg = {
        "label": "single_cell_eb_5d",
        "family": "single_cell",
        "coupling": "ot_global",
        "dim": 5,
        "constraint_time_policy": "observed_nonendpoint_excluding_holdout",
        "single_cell": {
            "path": str(dataset_path),
            "max_dim": 5,
            "whiten": True,
            "global_ot_cache_enabled": True,
            "global_ot_cache_dir": str(cache_dir),
            "global_ot_force_recompute": False,
            "global_ot_support_tol": 1.0e-12,
        },
    }
    experiment_cfg = {
        "protocol": "strict_leaveout",
        "holdout_index": 2,
        "holdout_indices": [1, 2, 3],
    }

    first = prepare_single_cell_problem_and_targets(
        data_cfg=data_cfg,
        experiment_cfg=experiment_cfg,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    second = prepare_single_cell_problem_and_targets(
        data_cfg=data_cfg,
        experiment_cfg=experiment_cfg,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert first.problem.has_global_ot_support
    assert second.problem.has_global_ot_support
    assert first.global_ot_cache_path is not None
    assert second.global_ot_cache_path == first.global_ot_cache_path
    assert first.global_ot_cache_hit is False
    assert second.global_ot_cache_hit is True
    assert first.global_ot_support_size is not None
    assert first.global_ot_support_size > 0


def test_prepare_single_cell_with_pseudo_labels_builds_targets_and_reuses_cache(tmp_path: Path) -> None:
    dataset_path = _write_synthetic_single_cell_npz(tmp_path / "eb_like_pseudo.npz", n_per_time=14)
    data_cfg = {
        "label": "single_cell_eb_5d",
        "family": "single_cell",
        "coupling": "ot",
        "dim": 5,
        "constraint_time_policy": "observed_nonendpoint_excluding_holdout",
        "single_cell": {
            "path": str(dataset_path),
            "max_dim": 5,
            "whiten": True,
            "pseudo_labels": {
                "enabled": True,
                "k_min": 2,
                "k_max": 4,
                "seeds": [5, 7, 11],
                "n_init": 2,
                "max_iter": 200,
                "tol": 1.0e-3,
                "reg_covar": 1.0e-6,
                "stability_threshold": 0.2,
                "cache_enabled": True,
                "cache_dir": str(tmp_path / "pseudo_cache"),
                "force_recompute": False,
            },
        },
    }
    experiment_cfg = {
        "protocol": "strict_leaveout",
        "holdout_index": 2,
        "holdout_indices": [1, 2, 3],
    }

    first = prepare_single_cell_problem_and_targets(
        data_cfg=data_cfg,
        experiment_cfg=experiment_cfg,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    second = prepare_single_cell_problem_and_targets(
        data_cfg=data_cfg,
        experiment_cfg=experiment_cfg,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert first.pseudo_targets is not None
    assert first.pseudo_posterior is not None
    assert first.pseudo_labels_k is not None
    assert first.pseudo_labels_k >= 2
    assert first.pseudo_labels_cache_path is not None
    assert first.pseudo_labels_cache_hit is False
    assert second.pseudo_labels_cache_hit is True
    assert second.pseudo_labels_cache_path == first.pseudo_labels_cache_path
    assert first.pseudo_fit_times == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert first.pseudo_fit_sample_count == 70
    for t in first.constraint_times:
        key = float(t)
        target = first.pseudo_targets[key]
        assert target.ndim == 1
        assert target.shape[0] == int(first.pseudo_labels_k)
        assert abs(float(target.sum().item()) - 1.0) <= 1e-4
