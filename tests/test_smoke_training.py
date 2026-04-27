import json
from pathlib import Path

from cfm_project.pipeline import run_pipeline


def _base_config(mode: str) -> dict:
    return {
        "seed": 123,
        "device": "cpu",
        "experiment": {
            "mode": mode,
            "run_both_modes": False,
        },
        "data": {
            "dim": 2,
            "mean0": [0.0, 0.0],
            "cov0": [[1.0, 0.2], [0.2, 0.9]],
            "mean1": [1.5, -0.5],
            "cov1": [[0.8, -0.1], [-0.1, 1.1]],
            "kappa": 0.6,
            "constraint_times": [0.25, 0.5, 0.75],
        },
        "model": {
            "activation": "silu",
            "velocity_hidden_dims": [32, 32],
            "path_hidden_dims": [32, 32],
        },
        "train": {
            "stage_a_steps": 4,
            "stage_b_steps": 6,
            "stage_c_steps": 3,
            "batch_size": 24,
            "eval_batch_size": 48,
            "eval_transport_samples": 64,
            "eval_transport_steps": 12,
            "eval_intermediate_empirical_w2": True,
            "eval_intermediate_ot_samples": 32,
            "lr_g": 0.001,
            "lr_v": 0.001,
            "alpha": 1.0,
            "beta": 0.05,
            "rho": 5.0,
            "eta_joint": 0.05,
            "lambda_clip": 100.0,
        },
        "output": {
            "save_checkpoint": True,
            "save_plots": True,
            "plot_pairs": 12,
        },
    }


def _assert_artifacts(mode_dir: Path) -> None:
    metrics_path = mode_dir / "metrics.json"
    assert metrics_path.exists()
    assert (mode_dir / "checkpoint.pt").exists()
    assert (mode_dir / "training_curve.png").exists()
    assert (mode_dir / "constraint_residuals.png").exists()
    assert (mode_dir / "sample_paths.png").exists()
    with metrics_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    summary = payload["summary"]
    assert "intermediate_w2_gaussian" in summary
    assert "intermediate_w2_gaussian_avg" in summary
    assert "intermediate_empirical_w2" in summary
    assert "intermediate_empirical_w2_avg" in summary


def test_smoke_baseline_pipeline(tmp_path: Path) -> None:
    cfg = _base_config(mode="baseline")
    result = run_pipeline(cfg, output_dir=tmp_path)
    mode_dir = Path(result["mode_dir"])
    _assert_artifacts(mode_dir)


def test_smoke_constrained_pipeline(tmp_path: Path) -> None:
    cfg = _base_config(mode="constrained")
    result = run_pipeline(cfg, output_dir=tmp_path)
    mode_dir = Path(result["mode_dir"])
    _assert_artifacts(mode_dir)
