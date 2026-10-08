from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cfm_project.data import sample_coupled_batch
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


def test_prepare_single_cell_npz_scales_velocity_with_feature_std(tmp_path: Path) -> None:
    labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    pcs = np.array(
        [
            [0.0, 10.0, 2.0],
            [2.0, 14.0, 4.0],
            [4.0, 18.0, 6.0],
            [6.0, 22.0, 8.0],
            [8.0, 26.0, 10.0],
            [10.0, 30.0, 12.0],
        ],
        dtype=np.float32,
    )
    pcs_delta = np.full_like(pcs, fill_value=2.0)
    dataset_path = tmp_path / "eb_velocity_like.npz"
    np.savez(dataset_path, pcs=pcs, pcs_delta=pcs_delta, sample_labels=labels)

    prepared = prepare_single_cell_problem_and_targets(
        data_cfg={
            "label": "single_cell_eb_2d",
            "family": "single_cell",
            "dim": 2,
            "constraint_time_policy": "observed_nonendpoint_all",
            "single_cell": {
                "path": str(dataset_path),
                "max_dim": 2,
                "whiten": True,
                "velocity_key_npz": "pcs_delta",
            },
        },
        experiment_cfg={"protocol": "no_leaveout"},
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert prepared.velocity_source_key == "pcs_delta"
    assert prepared.velocity_samples_by_time is not None
    expected_std = np.std(pcs[:, :2], axis=0)
    assert prepared.velocity_feature_std is not None
    assert np.allclose(np.asarray(prepared.velocity_feature_std), expected_std)
    velocity_t0 = prepared.velocity_samples_by_time[0.0]
    expected_velocity = torch.tensor(
        np.full((2, 2), 2.0, dtype=np.float32) / expected_std.reshape(1, 2),
        dtype=torch.float32,
    )
    assert torch.allclose(velocity_t0.cpu(), expected_velocity)


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


def test_persistent_minibatch_ot_single_cell_coupling_covers_endpoints(
    tmp_path: Path,
) -> None:
    dataset_path = _write_synthetic_single_cell_npz(
        tmp_path / "persistent_minibatch.npz", n_per_time=7
    )
    prepared = prepare_single_cell_problem_and_targets(
        data_cfg={
            "label": "single_cell_persistent_minibatch",
            "family": "single_cell",
            "coupling": "ot_persistent_minibatch",
            "dim": 5,
            "constraint_time_policy": "observed_nonendpoint_all",
            "single_cell": {
                "path": str(dataset_path),
                "max_dim": 5,
                "whiten": True,
                "persistent_minibatch_ot_batch_size": 4,
                "persistent_minibatch_ot_passes": 2,
                "persistent_minibatch_ot_seed": 17,
            },
        },
        experiment_cfg={"protocol": "no_leaveout"},
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    problem = prepared.problem
    assert problem.has_global_ot_support
    assert problem.global_ot_src_idx is not None
    assert problem.global_ot_tgt_idx is not None
    assert problem.global_ot_mass is not None
    assert int(problem.global_ot_src_idx.numel()) == 16
    assert int(torch.unique(problem.global_ot_src_idx).numel()) == 7
    assert int(torch.unique(problem.global_ot_tgt_idx).numel()) == 7
    assert torch.isclose(problem.global_ot_mass.sum(), torch.tensor(1.0))
    x0, x1, _ = sample_coupled_batch(
        problem,
        batch_size=6,
        coupling="ot_persistent_minibatch",
        generator=torch.Generator().manual_seed(23),
    )
    assert tuple(x0.shape) == (6, 5)
    assert tuple(x1.shape) == (6, 5)


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


def test_prepare_single_cell_uses_provenance_tagged_pseudo_target_overrides(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(91)
    n_per_time = 12
    times = np.repeat(np.arange(5, dtype=np.int64), n_per_time)
    pcs = rng.normal(size=(times.shape[0], 4)).astype(np.float32)
    dataset_path = tmp_path / "supervised_target_overrides.npz"
    np.savez(
        dataset_path,
        pcs=pcs,
        sample_labels=times,
    )
    expected = {
        0.25: torch.tensor([0.7, 0.3]),
        0.50: torch.tensor([0.5, 0.5]),
        0.75: torch.tensor([0.2, 0.8]),
    }
    prepared = prepare_single_cell_problem_and_targets(
        data_cfg={
            "label": "pseudo_target_override_test",
            "family": "single_cell",
            "dim": 4,
            "constraint_time_policy": "observed_nonendpoint_all",
            "single_cell": {
                "path": str(dataset_path),
                "max_dim": 4,
                "whiten": False,
                "constraint_times_normalized": [0.25, 0.5, 0.75],
                "pseudo_labels": {
                    "enabled": True,
                    "method": "gmm",
                    "fit_times_normalized": [0.5],
                    "k_min": 2,
                    "k_max": 2,
                    "seeds": [3],
                    "n_init": 1,
                    "max_iter": 100,
                    "stability_threshold": 0.0,
                    "cache_enabled": False,
                    "target_overrides": {
                        "0.25": expected[0.25].tolist(),
                        "0.5": expected[0.5].tolist(),
                        "0.75": expected[0.75].tolist(),
                    },
                    "target_override_source": (
                        "unit-test targets derived without intermediate marginals"
                    ),
                },
            },
        },
        experiment_cfg={"protocol": "no_leaveout"},
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert prepared.pseudo_targets is not None
    assert prepared.pseudo_target_source == (
        "unit-test targets derived without intermediate marginals"
    )
    assert set(prepared.pseudo_targets) == set(expected)
    for time_value, target in expected.items():
        assert torch.allclose(prepared.pseudo_targets[time_value].cpu(), target)


def test_prepare_single_cell_expands_classifier_state_first_moment_features(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(97)
    n_per_time = 12
    times = np.repeat(np.arange(5, dtype=np.int64), n_per_time)
    pcs = rng.normal(size=(times.shape[0], 4)).astype(np.float32)
    dataset_path = tmp_path / "classifier_state_moments.npz"
    np.savez(dataset_path, pcs=pcs, sample_labels=times)
    target = torch.linspace(-0.4, 0.5, 10)
    prepared = prepare_single_cell_problem_and_targets(
        data_cfg={
            "label": "classifier_state_moment_test",
            "family": "single_cell",
            "dim": 4,
            "constraint_time_policy": "observed_nonendpoint_all",
            "single_cell": {
                "path": str(dataset_path),
                "max_dim": 4,
                "whiten": False,
                "constraint_times_normalized": [0.5],
                "pseudo_labels": {
                    "enabled": True,
                    "method": "gmm",
                    "fit_times_normalized": [0.5],
                    "k_min": 2,
                    "k_max": 2,
                    "seeds": [3],
                    "n_init": 1,
                    "max_iter": 100,
                    "stability_threshold": 0.0,
                    "cache_enabled": False,
                    "constraint_feature_mode": (
                        "class_probability_and_state_first_moment"
                    ),
                    "target_overrides": {"0.5": target.tolist()},
                    "target_override_source": "unit-test MaxEnt aggregates",
                },
            },
        },
        experiment_cfg={"protocol": "no_leaveout"},
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert prepared.pseudo_constraint_feature_mode == (
        "class_probability_and_state_first_moment"
    )
    assert prepared.pseudo_constraint_dim == 10
    assert prepared.pseudo_targets is not None
    assert torch.allclose(prepared.pseudo_targets[0.5], target)
    assert prepared.pseudo_posterior is not None
    features = prepared.pseudo_posterior(prepared.problem.x0_pool[:3])
    assert features.shape == (3, 10)


def test_prepare_single_cell_expands_classifier_random_fourier_features(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(101)
    n_per_time = 12
    times = np.repeat(np.arange(5, dtype=np.int64), n_per_time)
    pcs = rng.normal(size=(times.shape[0], 4)).astype(np.float32)
    dataset_path = tmp_path / "classifier_rff.npz"
    np.savez(dataset_path, pcs=pcs, sample_labels=times)
    target = torch.tensor([0.55, 0.45, -0.1, 0.2, -0.3])
    prepared = prepare_single_cell_problem_and_targets(
        data_cfg={
            "label": "classifier_rff_test",
            "family": "single_cell",
            "dim": 4,
            "constraint_time_policy": "observed_nonendpoint_all",
            "single_cell": {
                "path": str(dataset_path),
                "max_dim": 4,
                "whiten": False,
                "constraint_times_normalized": [0.5],
                "pseudo_labels": {
                    "enabled": True,
                    "method": "gmm",
                    "fit_times_normalized": [0.5],
                    "k_min": 2,
                    "k_max": 2,
                    "seeds": [3],
                    "n_init": 1,
                    "max_iter": 100,
                    "stability_threshold": 0.0,
                    "cache_enabled": False,
                    "constraint_feature_mode": (
                        "class_probability_and_random_fourier"
                    ),
                    "random_fourier_matrix": [
                        [0.2, 0.1, -0.3, 0.4],
                        [-0.1, 0.5, 0.2, -0.2],
                        [0.3, -0.4, 0.1, 0.2],
                    ],
                    "random_fourier_phase": [0.1, 0.2, 0.3],
                    "target_overrides": {"0.5": target.tolist()},
                    "target_override_source": "unit-test MaxEnt RFF aggregates",
                },
            },
        },
        experiment_cfg={"protocol": "no_leaveout"},
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert prepared.pseudo_constraint_feature_mode == (
        "class_probability_and_random_fourier"
    )
    assert prepared.pseudo_constraint_dim == 5
    assert prepared.pseudo_targets is not None
    assert torch.allclose(prepared.pseudo_targets[0.5], target)
    assert prepared.pseudo_posterior is not None
    features = prepared.pseudo_posterior(prepared.problem.x0_pool[:3])
    assert features.shape == (3, 5)


def test_prepare_single_cell_uses_provenance_tagged_moment_target_overrides(
    tmp_path: Path,
) -> None:
    dataset_path = _write_synthetic_single_cell_npz(
        tmp_path / "moment_target_overrides.npz",
        n_per_time=10,
    )
    expected = {
        0.25: torch.tensor([-0.4, 0.3, 0.2, 0.1, -0.2]),
        0.50: torch.tensor([-0.1, 0.2, 0.4, -0.3, 0.5]),
        0.75: torch.tensor([0.2, 0.1, -0.2, 0.6, 0.3]),
    }
    prepared = prepare_single_cell_problem_and_targets(
        data_cfg={
            "label": "moment_target_override_test",
            "family": "single_cell",
            "dim": 5,
            "constraint_time_policy": "observed_nonendpoint_all",
            "moment_feature_blocks": ["mean"],
            "moment_target_overrides": {
                str(time_value): target.tolist()
                for time_value, target in expected.items()
            },
            "moment_target_override_source": (
                "unit-test means derived without intermediate marginals"
            ),
            "single_cell": {
                "path": str(dataset_path),
                "max_dim": 5,
                "whiten": False,
                "constraint_times_normalized": [0.25, 0.5, 0.75],
            },
        },
        experiment_cfg={"protocol": "no_leaveout"},
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert prepared.moment_target_source == (
        "unit-test means derived without intermediate marginals"
    )
    assert set(prepared.targets) == set(expected)
    for time_value, target in expected.items():
        assert torch.allclose(prepared.targets[time_value].cpu(), target)


def test_prepare_single_cell_accepts_physical_normalized_time_values(tmp_path: Path) -> None:
    dataset_path = _write_synthetic_single_cell_npz(
        tmp_path / "physical_times.npz",
        n_per_time=10,
    )
    prepared = prepare_single_cell_problem_and_targets(
        data_cfg={
            "label": "physical_time_single_cell",
            "family": "single_cell",
            "dim": 5,
            "constraint_time_policy": "observed_nonendpoint_all",
            "single_cell": {
                "path": str(dataset_path),
                "max_dim": 5,
                "whiten": False,
                "normalized_time_values": [0.0, 0.1, 0.4, 0.8, 1.0],
                "constraint_times_normalized": [0.1],
                "eval_times_normalized": [0.4],
            },
        },
        experiment_cfg={"protocol": "strict_leaveout", "holdout_index": 2},
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert prepared.normalized_times_all == [0.0, 0.1, 0.4, 0.8, 1.0]
    assert prepared.constraint_times == [0.1]
    assert prepared.eval_times == [0.4]
    assert prepared.holdout_time == 0.4


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
