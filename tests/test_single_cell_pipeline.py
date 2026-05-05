from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import pytest

from cfm_project.pipeline import run_pipeline


def _write_synthetic_single_cell_npz(path: Path, dim: int = 5, n_per_time: int = 20) -> Path:
    rng = np.random.default_rng(321)
    labels = np.repeat(np.array([0, 1, 2, 3, 4], dtype=np.int64), int(n_per_time))
    perm = rng.permutation(labels.shape[0])
    shuffled_labels = labels[perm]
    pcs = rng.normal(size=(labels.shape[0], int(dim))).astype(np.float32)
    pcs += shuffled_labels.reshape(-1, 1).astype(np.float32) * 0.12
    np.savez(path, pcs=pcs, sample_labels=shuffled_labels)
    return path


def _single_cell_cfg(dataset_path: Path) -> dict:
    return {
        "seed": 11,
        "device": "cpu",
        "experiment": {
            "mode": "constrained",
            "run_both_modes": False,
            "comparison_methods": [
                "baseline",
                "constrained",
                "metric",
                "metric_alpha0",
                "metric_constrained_al",
                "metric_constrained_soft",
            ],
            "protocol": "strict_leaveout",
            "holdout_index": 2,
            "holdout_indices": [1, 2, 3],
        },
        "data": {
            "label": "single_cell_eb_5d",
            "family": "single_cell",
            "coupling": "ot",
            "dim": 5,
            "constraint_times": [],
            "constraint_time_policy": "observed_nonendpoint_excluding_holdout",
            "single_cell": {
                "path": str(dataset_path),
                "max_dim": 5,
                "whiten": True,
                "embed_key_npz": "pcs",
                "label_key_npz": "sample_labels",
                "embed_key_h5ad": "X_pca",
                "label_key_h5ad": "day",
            },
        },
        "model": {
            "activation": "silu",
            "velocity_hidden_dims": [24, 24],
            "path_hidden_dims": [24, 24],
        },
        "train": {
            "label": "single_cell_ab_only",
            "stage_a_steps": 2,
            "stage_b_steps": 3,
            "stage_c_steps": 0,
            "batch_size": 24,
            "eval_batch_size": 64,
            "eval_transport_samples": 96,
            "eval_transport_steps": 12,
            "eval_intermediate_empirical_w2": True,
            "eval_intermediate_ot_samples": 48,
            "lr_g": 0.001,
            "lr_v": 0.001,
            "alpha": 1.0,
            "beta": 0.05,
            "rho": 1.0,
            "eta_joint": 0.05,
            "lambda_clip": 100.0,
        },
        "mfm": {
            "backend": "auto",
            "sigma": 0.1,
            "alpha": 1.0,
            "land_gamma": 0.125,
            "land_rho": 0.001,
            "land_metric_samples": 32,
            "reference_pool_policy": "endpoints_only",
            "moment_eta": 1.0,
        },
        "output": {
            "save_checkpoint": True,
            "save_plots": True,
            "plot_pairs": 8,
        },
    }


def test_single_cell_strict_leaveout_comparison_writes_all_modes_and_holdout_metrics(
    tmp_path: Path,
) -> None:
    dataset_path = _write_synthetic_single_cell_npz(tmp_path / "eb_like.npz")
    cfg = _single_cell_cfg(dataset_path)

    result = run_pipeline(cfg, output_dir=tmp_path / "single_cell_cmp")
    comparison_path = Path(result["comparison_mfm_path"])
    assert comparison_path.exists()

    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert payload["meta"]["protocol"] == "strict_leaveout"
    assert int(payload["meta"]["holdout_index"]) == 2
    assert float(payload["meta"]["holdout_time"]) == 0.5
    expected_modes = {
        "baseline",
        "constrained",
        "metric",
        "metric_alpha0",
        "metric_constrained_al",
        "metric_constrained_soft",
    }
    assert expected_modes.issubset(payload.keys())
    for mode in expected_modes:
        summary = payload[mode]
        assert summary["data_family"] == "single_cell"
        assert summary["protocol"] == "strict_leaveout"
        assert summary["holdout_index"] == 2
        assert abs(float(summary["holdout_time"]) - 0.5) <= 1e-8
        assert summary["holdout_empirical_w2"] is not None
        assert summary["holdout_empirical_w1"] is not None
        assert set(summary["constraint_residual_norms"].keys()) == {"0.25", "0.75"}

    constrained_dir = Path(result["constrained_dir"])
    assert (constrained_dir / "sample_paths_proj12.png").exists()
    assert (constrained_dir / "rollout_marginal_grid_proj12.png").exists()


def test_single_cell_ab_ot_global_adds_full_ot_rollout_metrics(tmp_path: Path) -> None:
    dataset_path = _write_synthetic_single_cell_npz(tmp_path / "eb_like_ot_global.npz", n_per_time=12)
    cfg = _single_cell_cfg(dataset_path)
    cfg["experiment"]["comparison_methods"] = ["constrained"]
    cfg["data"]["coupling"] = "ot_global"
    cfg["data"]["single_cell"]["global_ot_cache_enabled"] = True
    cfg["data"]["single_cell"]["global_ot_cache_dir"] = str(tmp_path / "ot_cache")
    cfg["train"]["stage_a_steps"] = 1
    cfg["train"]["stage_b_steps"] = 2
    cfg["train"]["eval_intermediate_ot_samples"] = 24
    cfg["train"]["eval_full_ot_metrics"] = True
    cfg["train"]["eval_full_ot_method"] = "exact_lp"
    cfg["train"]["eval_full_ot_max_variables"] = 200000
    cfg["train"]["eval_full_ot_support_tol"] = 1.0e-12

    result = run_pipeline(cfg, output_dir=tmp_path / "single_cell_ab_ot_global")
    payload = json.loads(Path(result["comparison_mfm_path"]).read_text(encoding="utf-8"))
    summary = payload["constrained"]

    assert summary["coupling"] == "ot_global"
    assert summary["global_ot_support_size"] is not None
    assert summary["intermediate_empirical_w2_avg"] is not None
    assert summary["intermediate_full_ot_w2_avg"] is not None
    assert summary["transport_endpoint_full_ot_w2"] is not None
    assert summary["holdout_full_ot_w2"] is not None


def test_single_cell_ab_ot_global_full_ot_pot_backend(tmp_path: Path) -> None:
    pytest.importorskip("ot")
    dataset_path = _write_synthetic_single_cell_npz(tmp_path / "eb_like_ot_global_pot.npz", n_per_time=12)
    cfg = _single_cell_cfg(dataset_path)
    cfg["experiment"]["comparison_methods"] = ["constrained"]
    cfg["data"]["coupling"] = "ot_global"
    cfg["data"]["single_cell"]["global_ot_cache_enabled"] = True
    cfg["data"]["single_cell"]["global_ot_cache_dir"] = str(tmp_path / "ot_cache")
    cfg["train"]["stage_a_steps"] = 1
    cfg["train"]["stage_b_steps"] = 2
    cfg["train"]["eval_intermediate_ot_samples"] = 24
    cfg["train"]["eval_full_ot_metrics"] = True
    cfg["train"]["eval_full_ot_method"] = "pot_emd2"
    cfg["train"]["eval_full_ot_num_itermax"] = 200000

    result = run_pipeline(cfg, output_dir=tmp_path / "single_cell_ab_ot_global_pot")
    payload = json.loads(Path(result["comparison_mfm_path"]).read_text(encoding="utf-8"))
    summary = payload["constrained"]

    assert summary["eval_full_ot_method"] == "pot_emd2"
    assert int(summary["eval_full_ot_num_itermax"]) == 200000
    assert summary["intermediate_full_ot_w2_avg"] is not None
    assert summary["transport_endpoint_full_ot_w2"] is not None
    assert summary["holdout_full_ot_w2"] is not None
