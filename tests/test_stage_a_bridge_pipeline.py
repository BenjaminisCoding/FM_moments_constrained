from pathlib import Path

import json

from cfm_project.pipeline import run_pipeline


def _bridge_stage_a_config(coupling: str, cache_dir: Path) -> dict:
    return {
        "seed": 123,
        "device": "cpu",
        "experiment": {
            "mode": "constrained",
            "run_both_modes": False,
        },
        "data": {
            "label": f"bridge_{coupling}",
            "family": "bridge_sde",
            "coupling": coupling,
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
            "label": "stage_a_only",
            "stage_a_steps": 4,
            "stage_b_steps": 0,
            "stage_c_steps": 0,
            "batch_size": 48,
            "eval_batch_size": 96,
            "eval_transport_samples": 128,
            "eval_transport_steps": 20,
            "eval_intermediate_empirical_w2": False,
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


def test_stage_a_only_bridge_ot_pipeline(tmp_path: Path) -> None:
    cfg = _bridge_stage_a_config("ot", tmp_path / "cache")
    result = run_pipeline(cfg, output_dir=tmp_path / "run_ot")

    mode_dir = Path(result["mode_dir"])
    assert mode_dir.exists()
    with (mode_dir / "metrics.json").open("r", encoding="utf-8") as f:
        payload = json.load(f)
    stages = {item["stage"] for item in payload["history"]}
    assert stages == {"stage_a"}
    summary = result["summary"]
    assert summary["stage_a_only"] is True
    assert summary["stage_steps"]["stage_b_steps"] == 0
    assert summary["stage_steps"]["stage_c_steps"] == 0
    assert summary["stage_c_enabled"] is False
    assert summary["cfm_val_loss"] is None
    assert summary["transport_score"] is None
    assert summary["transport_endpoint_empirical_w2"] is None
    assert "interpolant_eval" in summary
    interp = summary["interpolant_eval"]
    assert isinstance(interp, dict)
    assert "linear_empirical_w2_avg" in interp
    assert "learned_empirical_w2_avg" in interp


def test_stage_a_only_bridge_random_pipeline(tmp_path: Path) -> None:
    cfg = _bridge_stage_a_config("random", tmp_path / "cache")
    result = run_pipeline(cfg, output_dir=tmp_path / "run_random")

    mode_dir = Path(result["mode_dir"])
    assert mode_dir.exists()
    with (mode_dir / "metrics.json").open("r", encoding="utf-8") as f:
        payload = json.load(f)
    stages = {item["stage"] for item in payload["history"]}
    assert stages == {"stage_a"}
    summary = result["summary"]
    assert summary["coupling"] == "random"
    assert summary["stage_a_only"] is True
    assert summary["stage_steps"]["stage_b_steps"] == 0
    assert summary["stage_steps"]["stage_c_steps"] == 0
    assert summary["cfm_val_loss"] is None
    assert summary["transport_endpoint_empirical_w2"] is None
    assert "interpolant_eval" in summary
