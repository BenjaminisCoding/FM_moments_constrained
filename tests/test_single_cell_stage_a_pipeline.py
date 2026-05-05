from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import pytest

from cfm_project.pipeline import run_pipeline


def _write_synthetic_single_cell_npz(path: Path, dim: int = 5, n_per_time: int = 16) -> Path:
    rng = np.random.default_rng(654)
    labels = np.repeat(np.array([0, 1, 2, 3, 4], dtype=np.int64), int(n_per_time))
    perm = rng.permutation(labels.shape[0])
    shuffled_labels = labels[perm]
    pcs = rng.normal(size=(labels.shape[0], int(dim))).astype(np.float32)
    pcs += shuffled_labels.reshape(-1, 1).astype(np.float32) * 0.1
    np.savez(path, pcs=pcs, sample_labels=shuffled_labels)
    return path


def _cfg_stage_a_single_cell(dataset_path: Path) -> dict:
    return {
        "seed": 7,
        "device": "cpu",
        "experiment": {
            "mode": "constrained",
            "run_both_modes": False,
            "comparison_methods": [
                "constrained",
                "metric",
                "metric_alpha0",
                "metric_constrained_al",
                "metric_constrained_soft",
            ],
            "protocol": "strict_leaveout",
            "holdout_index": 2,
            "holdout_indices": [1, 2, 3],
            "method_overrides": {
                "constrained": {
                    "train": {"alpha": 1.0, "beta": 0.05, "rho": 25.0},
                },
                "metric_alpha0": {
                    "mfm": {"alpha": 0.0},
                },
                "metric_constrained_al": {
                    "mfm": {"alpha": 0.4, "moment_eta": 2.0},
                    "train": {"rho": 15.0},
                },
                "metric_constrained_soft": {
                    "mfm": {"alpha": 0.4, "moment_eta": 2.0},
                },
            },
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
            "velocity_hidden_dims": [20, 20],
            "path_hidden_dims": [20, 20],
        },
        "train": {
            "label": "single_cell_stage_a_only",
            "stage_a_steps": 2,
            "stage_b_steps": 0,
            "stage_c_steps": 0,
            "batch_size": 20,
            "eval_batch_size": 64,
            "eval_transport_samples": 64,
            "eval_transport_steps": 12,
            "eval_intermediate_empirical_w2": False,
            "eval_intermediate_ot_samples": 32,
            "lr_g": 0.001,
            "lr_v": 0.001,
            "alpha": 1.0,
            "beta": 0.05,
            "rho": 5.0,
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
            "save_checkpoint": False,
            "save_plots": True,
            "plot_pairs": 8,
        },
    }


def test_single_cell_stage_a_only_comparison_modes_have_holdout_interpolant_metrics(
    tmp_path: Path,
) -> None:
    dataset_path = _write_synthetic_single_cell_npz(tmp_path / "eb_like_stage_a.npz")
    cfg = _cfg_stage_a_single_cell(dataset_path)
    result = run_pipeline(cfg, output_dir=tmp_path / "single_cell_stage_a")

    comparison_path = Path(result["comparison_mfm_path"])
    assert comparison_path.exists()
    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert payload["meta"]["stage_a_only"] is True
    assert float(payload["meta"]["holdout_time"]) == 0.5

    expected_modes = {
        "constrained",
        "metric",
        "metric_alpha0",
        "metric_constrained_al",
        "metric_constrained_soft",
    }
    assert expected_modes.issubset(payload.keys())
    for mode in expected_modes:
        summary = payload[mode]
        assert summary["stage_a_only"] is True
        assert summary["transport_score"] is None
        interp = summary["interpolant_eval"]
        assert float(interp["linear_holdout_empirical_w2"]) >= 0.0
        assert float(interp["learned_holdout_empirical_w2"]) >= 0.0
        assert "delta_holdout_learned_minus_linear" in interp

    constrained_dir = Path(result["constrained_dir"])
    assert (constrained_dir / "interpolant_trajectories.png").exists()
    assert (constrained_dir / "interpolant_marginal_grid.png").exists()
    assert (constrained_dir / "interpolant_trajectories_proj12.png").exists()
    assert (constrained_dir / "interpolant_marginal_grid_proj12.png").exists()
    assert not (constrained_dir / "rollout_marginal_grid.png").exists()


def test_single_cell_stage_a_only_method_overrides_apply_per_mode(tmp_path: Path) -> None:
    dataset_path = _write_synthetic_single_cell_npz(tmp_path / "eb_like_stage_a.npz")
    cfg = _cfg_stage_a_single_cell(dataset_path)
    result = run_pipeline(cfg, output_dir=tmp_path / "single_cell_stage_a")

    constrained_metrics = json.loads(
        (Path(result["constrained_dir"]) / "metrics.json").read_text(encoding="utf-8")
    )
    metric_alpha0_metrics = json.loads(
        (Path(result["metric_alpha0_dir"]) / "metrics.json").read_text(encoding="utf-8")
    )
    metric_al_metrics = json.loads(
        (Path(result["metric_constrained_al_dir"]) / "metrics.json").read_text(encoding="utf-8")
    )
    metric_soft_metrics = json.loads(
        (Path(result["metric_constrained_soft_dir"]) / "metrics.json").read_text(encoding="utf-8")
    )

    assert float(constrained_metrics["config"]["train"]["rho"]) == 25.0
    assert float(metric_alpha0_metrics["config"]["mfm"]["alpha"]) == 0.0

    assert float(metric_al_metrics["config"]["mfm"]["alpha"]) == 0.4
    assert float(metric_al_metrics["config"]["mfm"]["moment_eta"]) == 2.0
    assert float(metric_al_metrics["config"]["train"]["rho"]) == 15.0
    assert float(metric_al_metrics["summary"]["mfm_alpha"]) == 0.4
    assert float(metric_al_metrics["summary"]["mfm_moment_eta"]) == 2.0

    assert float(metric_soft_metrics["config"]["mfm"]["alpha"]) == 0.4
    assert float(metric_soft_metrics["config"]["mfm"]["moment_eta"]) == 2.0
    assert float(metric_soft_metrics["summary"]["mfm_alpha"]) == 0.4
    assert float(metric_soft_metrics["summary"]["mfm_moment_eta"]) == 2.0


def test_single_cell_stage_a_ot_global_adds_full_ot_interpolant_metrics(tmp_path: Path) -> None:
    dataset_path = _write_synthetic_single_cell_npz(tmp_path / "eb_like_stage_a_ot_global.npz")
    cfg = _cfg_stage_a_single_cell(dataset_path)
    cfg["experiment"]["comparison_methods"] = ["constrained"]
    cfg["data"]["coupling"] = "ot_global"
    cfg["data"]["single_cell"]["global_ot_cache_enabled"] = True
    cfg["data"]["single_cell"]["global_ot_cache_dir"] = str(tmp_path / "ot_cache")
    cfg["train"]["eval_full_ot_metrics"] = True
    cfg["train"]["eval_full_ot_method"] = "exact_lp"
    cfg["train"]["eval_full_ot_max_variables"] = 200000
    cfg["train"]["eval_full_ot_support_tol"] = 1.0e-12

    result = run_pipeline(cfg, output_dir=tmp_path / "single_cell_stage_a_ot_global")
    payload = json.loads(Path(result["comparison_mfm_path"]).read_text(encoding="utf-8"))
    summary = payload["constrained"]
    interp = summary["interpolant_eval"]

    assert summary["coupling"] == "ot_global"
    assert summary["global_ot_support_size"] is not None
    assert float(interp["learned_empirical_w2_avg"]) >= 0.0
    assert float(interp["learned_full_ot_w2_avg"]) >= 0.0
    assert "linear_holdout_full_ot_w2" in interp
    assert "learned_holdout_full_ot_w2" in interp


def test_single_cell_stage_a_ot_global_full_ot_pot_backend(tmp_path: Path) -> None:
    pytest.importorskip("ot")
    dataset_path = _write_synthetic_single_cell_npz(tmp_path / "eb_like_stage_a_ot_global_pot.npz")
    cfg = _cfg_stage_a_single_cell(dataset_path)
    cfg["experiment"]["comparison_methods"] = ["constrained"]
    cfg["data"]["coupling"] = "ot_global"
    cfg["data"]["single_cell"]["global_ot_cache_enabled"] = True
    cfg["data"]["single_cell"]["global_ot_cache_dir"] = str(tmp_path / "ot_cache")
    cfg["train"]["eval_full_ot_metrics"] = True
    cfg["train"]["eval_full_ot_method"] = "pot_emd2"
    cfg["train"]["eval_full_ot_num_itermax"] = 200000

    result = run_pipeline(cfg, output_dir=tmp_path / "single_cell_stage_a_ot_global_pot")
    payload = json.loads(Path(result["comparison_mfm_path"]).read_text(encoding="utf-8"))
    summary = payload["constrained"]
    interp = summary["interpolant_eval"]

    assert summary["eval_full_ot_method"] == "pot_emd2"
    assert int(summary["eval_full_ot_num_itermax"]) == 200000
    assert float(interp["learned_full_ot_w2_avg"]) >= 0.0
    assert "linear_holdout_full_ot_w2" in interp
    assert "learned_holdout_full_ot_w2" in interp


def test_single_cell_stage_a_pseudo_constraints_emit_summary_metrics(tmp_path: Path) -> None:
    dataset_path = _write_synthetic_single_cell_npz(tmp_path / "eb_like_stage_a_pseudo.npz")
    cfg = _cfg_stage_a_single_cell(dataset_path)
    cfg["experiment"]["comparison_methods"] = ["constrained"]
    cfg["data"]["single_cell"]["pseudo_labels"] = {
        "enabled": True,
        "k_min": 2,
        "k_max": 4,
        "seeds": [3, 5, 7],
        "n_init": 2,
        "max_iter": 150,
        "tol": 1.0e-3,
        "reg_covar": 1.0e-6,
        "stability_threshold": 0.1,
        "cache_enabled": True,
        "cache_dir": str(tmp_path / "pseudo_cache"),
        "force_recompute": False,
    }
    cfg["train"]["pseudo_eta"] = 1.0
    cfg["train"]["pseudo_rho"] = 5.0
    cfg["train"]["pseudo_lambda_clip"] = 100.0

    result = run_pipeline(cfg, output_dir=tmp_path / "single_cell_stage_a_pseudo")
    payload = json.loads(Path(result["comparison_mfm_path"]).read_text(encoding="utf-8"))
    summary = payload["constrained"]

    assert summary["pseudo_constraints_active"] is True
    assert summary["pseudo_constraint_residual_norms"] is not None
    assert summary["pseudo_constraint_residual_avg"] is not None
    assert summary["pseudo_labels_k"] is not None
    assert summary["pseudo_labels_cache_path"] is not None
    assert summary["bic_by_k"] is not None
    assert summary["stability_by_k"] is not None
    assert summary["constraint_residual_avg"] is not None


def test_single_cell_stage_a_pseudo_constraints_cover_metric_constrained_modes(tmp_path: Path) -> None:
    dataset_path = _write_synthetic_single_cell_npz(tmp_path / "eb_like_stage_a_pseudo_metric.npz")
    cfg = _cfg_stage_a_single_cell(dataset_path)
    cfg["experiment"]["comparison_methods"] = [
        "constrained",
        "metric_constrained_al",
        "metric_constrained_soft",
    ]
    cfg["data"]["single_cell"]["pseudo_labels"] = {
        "enabled": True,
        "k_min": 2,
        "k_max": 4,
        "seeds": [3, 5],
        "n_init": 1,
        "max_iter": 100,
        "tol": 1.0e-3,
        "reg_covar": 1.0e-6,
        "stability_threshold": 0.0,
        "cache_enabled": True,
        "cache_dir": str(tmp_path / "pseudo_cache_metric"),
        "force_recompute": False,
    }
    cfg["train"]["pseudo_eta"] = 0.5
    cfg["train"]["pseudo_rho"] = 5.0
    cfg["train"]["pseudo_lambda_clip"] = 100.0

    result = run_pipeline(cfg, output_dir=tmp_path / "single_cell_stage_a_pseudo_metric")
    payload = json.loads(Path(result["comparison_mfm_path"]).read_text(encoding="utf-8"))
    for mode in ["constrained", "metric_constrained_al", "metric_constrained_soft"]:
        summary = payload[mode]
        assert summary["pseudo_constraints_active"] is True
        assert summary["pseudo_constraint_residual_avg"] is not None
        assert summary["pseudo_constraint_residual_norms"] is not None


def test_single_cell_stage_a_pseudo_only_t05_constraints_and_eval_all_snapshots(tmp_path: Path) -> None:
    dataset_path = _write_synthetic_single_cell_npz(tmp_path / "eb_like_stage_a_pseudo_t05.npz")
    cfg = _cfg_stage_a_single_cell(dataset_path)
    cfg["experiment"]["protocol"] = "no_leaveout"
    cfg["experiment"]["comparison_methods"] = [
        "constrained",
        "metric_constrained_al",
        "metric_constrained_soft",
    ]
    cfg["data"]["coupling"] = "ot_global"
    cfg["data"]["single_cell"]["global_ot_cache_enabled"] = True
    cfg["data"]["single_cell"]["global_ot_cache_dir"] = str(tmp_path / "ot_cache")
    cfg["data"]["single_cell"]["constraint_times_normalized"] = [0.5]
    cfg["data"]["single_cell"]["eval_times_normalized"] = [0.25, 0.5, 0.75]
    cfg["data"]["single_cell"]["pseudo_labels"] = {
        "enabled": True,
        "fit_times_normalized": [0.5],
        "k_min": 2,
        "k_max": 4,
        "seeds": [3, 5],
        "n_init": 1,
        "max_iter": 120,
        "tol": 1.0e-3,
        "reg_covar": 1.0e-6,
        "stability_threshold": 0.0,
        "cache_enabled": True,
        "cache_dir": str(tmp_path / "pseudo_cache_t05"),
        "force_recompute": False,
    }
    cfg["train"]["pseudo_eta"] = 1.0
    cfg["train"]["pseudo_rho"] = 5.0
    cfg["train"]["pseudo_lambda_clip"] = 100.0
    cfg["train"]["eval_full_ot_metrics"] = True
    cfg["train"]["eval_full_ot_method"] = "exact_lp"
    cfg["train"]["eval_full_ot_max_variables"] = 200000
    cfg["train"]["eval_full_ot_support_tol"] = 1.0e-12

    result = run_pipeline(cfg, output_dir=tmp_path / "single_cell_stage_a_pseudo_t05")
    payload = json.loads(Path(result["comparison_mfm_path"]).read_text(encoding="utf-8"))

    for mode in ["constrained", "metric_constrained_al", "metric_constrained_soft"]:
        summary = payload[mode]
        assert summary["pseudo_constraints_active"] is True
        assert set(summary["constraint_residual_norms"].keys()) == {"0.50"}
        assert set(summary["pseudo_constraint_residual_norms"].keys()) == {"0.50"}
        assert summary["single_cell_constraint_times"] == [0.5]
        assert summary["single_cell_eval_times"] == [0.25, 0.5, 0.75]
        assert summary["single_cell_pseudo_fit_times"] == [0.5]
        assert int(summary["single_cell_pseudo_fit_sample_count"]) > 0
        interp = summary["interpolant_eval"]
        assert set(interp["learned_empirical_w2"].keys()) == {"0.25", "0.50", "0.75"}
        assert set(interp["learned_full_ot_w2"].keys()) == {"0.25", "0.50", "0.75"}
