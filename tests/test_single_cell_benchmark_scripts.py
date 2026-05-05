from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_a_benchmark_leaderboard_prefers_full_ot_with_legacy_fallback() -> None:
    module = _load_script_module(
        "run_single_cell_eb_stage_a_benchmark_test",
        ROOT / "scripts" / "run_single_cell_eb_stage_a_benchmark.py",
    )
    base_metrics = {
        "linear_empirical_w2_avg": None,
        "learned_empirical_w2_avg": None,
        "delta_avg_learned_minus_linear": None,
        "linear_holdout_empirical_w2": None,
        "learned_holdout_empirical_w2": None,
        "delta_holdout_learned_minus_linear": None,
        "linear_full_ot_w2_avg": None,
        "learned_full_ot_w2_avg": None,
        "delta_avg_learned_minus_linear_full_ot": None,
        "linear_holdout_full_ot_w2": None,
        "learned_holdout_full_ot_w2": None,
        "delta_holdout_learned_minus_linear_full_ot": None,
    }

    def mode_metrics(*, robust: float | None, legacy: float | None) -> dict[str, float | None]:
        out = dict(base_metrics)
        out["learned_holdout_full_ot_w2"] = robust
        out["learned_holdout_empirical_w2"] = legacy
        out["linear_holdout_full_ot_w2"] = robust
        out["linear_holdout_empirical_w2"] = legacy
        return out

    # metric has robust key -> should rank first even though constrained has better legacy.
    results = [
        {
            "holdout_index": 1,
            "seed": 3,
                "mode_metrics": {
                    "constrained": mode_metrics(robust=None, legacy=2.40),
                    "metric": mode_metrics(robust=0.55, legacy=1.20),
                    "metric_alpha0": mode_metrics(robust=0.95, legacy=1.10),
                    "metric_constrained_al": mode_metrics(robust=0.70, legacy=1.00),
                    "metric_constrained_soft": mode_metrics(robust=0.80, legacy=1.05),
                },
        },
        {
            "holdout_index": 2,
            "seed": 7,
                "mode_metrics": {
                    "constrained": mode_metrics(robust=None, legacy=2.50),
                    "metric": mode_metrics(robust=0.45, legacy=1.10),
                    "metric_alpha0": mode_metrics(robust=1.00, legacy=1.20),
                    "metric_constrained_al": mode_metrics(robust=0.72, legacy=1.05),
                    "metric_constrained_soft": mode_metrics(robust=0.82, legacy=1.15),
                },
        },
    ]

    aggregate = module._aggregate(results)
    leaderboard = aggregate["leaderboard"]
    assert leaderboard[0]["mode"] == "metric"
    constrained_row = next(row for row in leaderboard if row["mode"] == "constrained")
    assert constrained_row["learned_holdout_full_ot_w2_mean"] is None
    assert constrained_row["learned_holdout_empirical_w2_mean"] is not None


def test_ab_benchmark_aggregate_includes_full_ot_keys() -> None:
    module = _load_script_module(
        "run_single_cell_eb_benchmark_test",
        ROOT / "scripts" / "run_single_cell_eb_benchmark.py",
    )
    modes = [
        "baseline",
        "constrained",
        "metric",
        "metric_alpha0",
        "metric_constrained_al",
        "metric_constrained_soft",
    ]

    def make_summary(v: float) -> dict[str, float]:
        return {
            "intermediate_empirical_w2_avg": v,
            "transport_endpoint_empirical_w2": v + 0.1,
            "holdout_empirical_w2": v + 0.2,
            "holdout_empirical_w1": v + 0.3,
            "intermediate_full_ot_w2_avg": v + 0.4,
            "transport_endpoint_full_ot_w2": v + 0.5,
            "holdout_full_ot_w2": v + 0.6,
        }

    results = [
        {
            "comparison_mfm": {
                mode: make_summary(1.0) for mode in modes
            }
        },
        {
            "comparison_mfm": {
                mode: make_summary(2.0) for mode in modes
            }
        },
    ]
    aggregate = module._aggregate_holdout_results(results)
    baseline = aggregate["baseline"]
    assert abs(float(baseline["intermediate_full_ot_w2_avg_mean"]) - 1.9) <= 1e-8
    assert abs(float(baseline["transport_endpoint_full_ot_w2_mean"]) - 2.0) <= 1e-8
    assert abs(float(baseline["holdout_full_ot_w2_mean"]) - 2.1) <= 1e-8
