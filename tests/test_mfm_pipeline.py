from pathlib import Path

import json

import pytest

from cfm_project.pipeline import run_pipeline


def _base_cfg() -> dict:
    return {
        "seed": 321,
        "device": "cpu",
        "experiment": {
            "mode": "metric",
            "run_both_modes": False,
            "comparison_methods": [],
        },
        "data": {
            "dim": 2,
            "mean0": [0.0, 0.0],
            "cov0": [[1.0, 0.2], [0.2, 0.9]],
            "mean1": [1.5, -0.5],
            "cov1": [[0.8, -0.1], [-0.1, 1.1]],
            "kappa": 0.6,
            "constraint_times": [0.25, 0.5, 0.75],
            "coupling": "ot",
        },
        "model": {
            "activation": "silu",
            "velocity_hidden_dims": [24, 24],
            "path_hidden_dims": [24, 24],
        },
        "train": {
            "label": "ab_only",
            "stage_a_steps": 3,
            "stage_b_steps": 4,
            "stage_c_steps": 0,
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
            "land_metric_samples": 64,
            "reference_pool_policy": "endpoints_only",
            "moment_eta": 1.0,
        },
        "output": {
            "save_checkpoint": True,
            "save_plots": False,
            "plot_pairs": 8,
        },
    }


def test_metric_mode_smoke_writes_mfm_summary_fields(tmp_path: Path) -> None:
    cfg = _base_cfg()
    cfg["experiment"]["mode"] = "metric"
    result = run_pipeline(cfg, output_dir=tmp_path / "metric")

    mode_dir = Path(result["mode_dir"])
    with (mode_dir / "metrics.json").open("r", encoding="utf-8") as f:
        payload = json.load(f)
    summary = payload["summary"]
    assert summary["mode"] == "metric"
    assert summary["mfm_backend"] in {"native", "torchcfm"}
    assert "mfm_backend_impl" in summary
    assert "mfm_alpha" in summary
    assert summary["mfm_reference_pool_policy"] == "endpoints_only"
    assert summary["mfm_moment_style"] == "none"
    assert summary["mfm_moment_eta"] == 1.0


@pytest.mark.parametrize(
    ("mode", "expected_style"),
    [("metric_constrained_al", "al"), ("metric_constrained_soft", "soft")],
)
def test_metric_constrained_modes_smoke_report_moment_style(
    mode: str,
    expected_style: str,
    tmp_path: Path,
) -> None:
    cfg = _base_cfg()
    cfg["experiment"]["mode"] = mode
    result = run_pipeline(cfg, output_dir=tmp_path / mode)

    mode_dir = Path(result["mode_dir"])
    with (mode_dir / "metrics.json").open("r", encoding="utf-8") as f:
        payload = json.load(f)
    summary = payload["summary"]
    assert summary["mode"] == mode
    assert summary["mfm_moment_style"] == expected_style
    assert summary["mfm_reference_pool_policy"] == "endpoints_only"
    assert summary["mfm_moment_eta"] == 1.0


def test_comparison_methods_writes_comparison_mfm_and_legacy_comparison(tmp_path: Path) -> None:
    cfg = _base_cfg()
    cfg["experiment"]["comparison_methods"] = [
        "baseline",
        "constrained",
        "metric",
        "metric_alpha0",
        "metric_constrained_al",
        "metric_constrained_soft",
    ]
    cfg["experiment"]["mode"] = "constrained"

    result = run_pipeline(cfg, output_dir=tmp_path / "cmp")
    cmp_path = Path(result["comparison_mfm_path"])
    assert cmp_path.exists()
    with cmp_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    assert set(payload.keys()) >= {
        "meta",
        "baseline",
        "constrained",
        "metric",
        "metric_alpha0",
        "metric_constrained_al",
        "metric_constrained_soft",
    }
    assert payload["meta"]["methods"] == [
        "baseline",
        "constrained",
        "metric",
        "metric_alpha0",
        "metric_constrained_al",
        "metric_constrained_soft",
    ]

    legacy_path = (tmp_path / "cmp" / "comparison.json")
    assert legacy_path.exists()
    with legacy_path.open("r", encoding="utf-8") as f:
        legacy = json.load(f)
    assert set(legacy.keys()) == {"meta", "baseline", "constrained"}
