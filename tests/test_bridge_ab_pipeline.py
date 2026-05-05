from pathlib import Path

import json
import math

from cfm_project.pipeline import run_pipeline


def _bridge_ab_config(cache_dir: Path, run_both_modes: bool = False) -> dict:
    return {
        "seed": 211,
        "device": "cpu",
        "experiment": {
            "mode": "constrained",
            "run_both_modes": run_both_modes,
        },
        "data": {
            "label": "bridge_ot",
            "family": "bridge_sde",
            "coupling": "ot",
            "dim": 2,
            "constraint_times": [0.25, 0.5, 0.75],
            "target_mc_samples": 768,
            "target_cache_enabled": True,
            "target_cache_dir": str(cache_dir),
            "bridge": {
                "n_steps": 80,
                "total_time": 1.5,
                "mean0": [0.0, 0.0],
                "cov0": [[0.35, 0.0], [0.0, 0.60]],
                "vx": 2.0,
                "sigma_x": 0.15,
                "sigma_y": 0.45,
                "bridge_center_x": 1.0,
                "bridge_width": 0.35,
                "bridge_pull": 8.0,
                "bridge_diffusion_drop": 0.8,
            },
        },
        "model": {
            "activation": "silu",
            "velocity_hidden_dims": [16, 16],
            "path_hidden_dims": [16, 16],
        },
        "train": {
            "label": "ab_only",
            "stage_a_steps": 3,
            "stage_b_steps": 4,
            "stage_c_steps": 0,
            "batch_size": 48,
            "eval_batch_size": 96,
            "eval_transport_samples": 128,
            "eval_transport_steps": 20,
            "eval_intermediate_empirical_w2": True,
            "eval_intermediate_ot_samples": 64,
            "lr_g": 0.001,
            "lr_v": 0.001,
            "alpha": 1.0,
            "beta": 0.05,
            "rho": 1.0,
            "eta_joint": 0.05,
            "lambda_clip": 100.0,
        },
        "output": {
            "save_checkpoint": True,
            "save_plots": False,
            "plot_pairs": 12,
        },
    }


def _assert_bridge_rollout_summary(summary: dict) -> None:
    per_time = summary["intermediate_empirical_w2"]
    assert isinstance(per_time, dict)
    assert set(per_time.keys()) == {"0.25", "0.50", "0.75"}
    for value in per_time.values():
        assert math.isfinite(float(value))
    assert math.isfinite(float(summary["intermediate_empirical_w2_avg"]))
    assert math.isfinite(float(summary["transport_endpoint_empirical_w2"]))
    assert float(summary["transport_score"]) == float(summary["transport_endpoint_empirical_w2"])
    assert summary["transport_mean_error_l2"] is None
    assert summary["transport_cov_error_fro"] is None
    assert summary["intermediate_w2_gaussian"] is None
    assert summary["intermediate_w2_gaussian_avg"] is None


def test_bridge_ab_only_constrained_writes_rollout_metrics(tmp_path: Path) -> None:
    cfg = _bridge_ab_config(cache_dir=tmp_path / "cache", run_both_modes=False)
    result = run_pipeline(cfg, output_dir=tmp_path / "single")

    mode_dir = Path(result["mode_dir"])
    assert mode_dir.exists()
    with (mode_dir / "metrics.json").open("r", encoding="utf-8") as f:
        payload = json.load(f)
    stages = {item["stage"] for item in payload["history"]}
    assert stages == {"stage_a", "stage_b"}
    summary = payload["summary"]
    assert summary["stage_a_only"] is False
    assert summary["stage_steps"]["stage_b_steps"] == 4
    assert summary["stage_steps"]["stage_c_steps"] == 0
    assert summary["stage_c_enabled"] is False
    assert summary["cfm_val_loss"] is not None
    _assert_bridge_rollout_summary(summary)


def test_bridge_ab_only_constrained_writes_rollout_plots(tmp_path: Path) -> None:
    cfg = _bridge_ab_config(cache_dir=tmp_path / "cache", run_both_modes=False)
    cfg["output"]["save_plots"] = True
    result = run_pipeline(cfg, output_dir=tmp_path / "single_plots")
    mode_dir = Path(result["mode_dir"])
    assert (mode_dir / "rollout_marginal_grid.png").exists()
    assert (mode_dir / "rollout_empirical_w2.png").exists()


def test_bridge_ab_only_comparison_writes_baseline_and_constrained_rollout_metrics(
    tmp_path: Path,
) -> None:
    cfg = _bridge_ab_config(cache_dir=tmp_path / "cache", run_both_modes=True)
    result = run_pipeline(cfg, output_dir=tmp_path / "comparison")

    comparison = result["comparison"]
    assert set(comparison.keys()) >= {"baseline", "constrained", "meta"}
    _assert_bridge_rollout_summary(comparison["baseline"])
    _assert_bridge_rollout_summary(comparison["constrained"])
