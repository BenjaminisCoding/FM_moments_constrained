#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_EXPERIMENT = ROOT / "scripts" / "run_experiment.py"
MODES = [
    "constrained",
    "metric",
    "metric_alpha0",
    "metric_constrained_al",
    "metric_constrained_soft",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage-A-only EB 5D strict-leaveout benchmark across "
            "holdouts and seeds for constrained + metric-family interpolants."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Full local path to EB dataset (.npz or .h5ad).",
    )
    parser.add_argument(
        "--holdouts",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Middle holdout indices to run. Default: 1 2 3",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[3, 7, 11],
        help="Random seeds to run. Default: 3 7 11",
    )
    parser.add_argument(
        "--mfm-backend",
        type=str,
        default="auto",
        choices=["auto", "native", "torchcfm"],
        help="MFM backend selection policy.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help=(
            "Benchmark root folder. Default: "
            "outputs/<date>/single_cell_eb_5d_stage_a_strict_leaveout/<time>"
        ),
    )
    return parser.parse_args()


def _default_run_root() -> Path:
    day = datetime.now().strftime("%Y-%m-%d")
    ts = datetime.now().strftime("%H-%M-%S")
    return (ROOT / "outputs" / day / "single_cell_eb_5d_stage_a_strict_leaveout" / ts).resolve()


def _extract_mode_metrics(summary: dict[str, Any]) -> dict[str, float | None]:
    interp = summary.get("interpolant_eval", {})
    if not isinstance(interp, dict):
        raise RuntimeError("Missing summary.interpolant_eval in Stage-A benchmark output.")

    def _as_float(key: str) -> float | None:
        value = interp.get(key)
        if value is None:
            return None
        return float(value)

    return {
        "linear_empirical_w2_avg": _as_float("linear_empirical_w2_avg"),
        "learned_empirical_w2_avg": _as_float("learned_empirical_w2_avg"),
        "delta_avg_learned_minus_linear": _as_float("delta_avg_learned_minus_linear"),
        "linear_holdout_empirical_w2": _as_float("linear_holdout_empirical_w2"),
        "learned_holdout_empirical_w2": _as_float("learned_holdout_empirical_w2"),
        "delta_holdout_learned_minus_linear": _as_float("delta_holdout_learned_minus_linear"),
        "linear_full_ot_w2_avg": _as_float("linear_full_ot_w2_avg"),
        "learned_full_ot_w2_avg": _as_float("learned_full_ot_w2_avg"),
        "delta_avg_learned_minus_linear_full_ot": _as_float("delta_avg_learned_minus_linear_full_ot"),
        "linear_holdout_full_ot_w2": _as_float("linear_holdout_full_ot_w2"),
        "learned_holdout_full_ot_w2": _as_float("learned_holdout_full_ot_w2"),
        "delta_holdout_learned_minus_linear_full_ot": _as_float("delta_holdout_learned_minus_linear_full_ot"),
    }


def _run_trial(
    *,
    data_path: Path,
    holdout_index: int,
    seed: int,
    mfm_backend: str,
    run_dir: Path,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(RUN_EXPERIMENT),
        "experiment=comparison_mfm_single_cell_stage_a",
        "train=single_cell_stage_a_only",
        "data=single_cell_eb_5d",
        "experiment.label=single_cell_eb_5d_stage_a_strict_leaveout",
        "experiment.protocol=strict_leaveout",
        f"experiment.holdout_index={int(holdout_index)}",
        "data.constraint_time_policy=observed_nonendpoint_excluding_holdout",
        f"data.single_cell.path={str(data_path.resolve())}",
        f"seed={int(seed)}",
        f"mfm.backend={mfm_backend}",
        "output.save_plots=true",
        f"hydra.run.dir={str(run_dir)}",
    ]
    print(f"[run] holdout={holdout_index} seed={seed} cmd={' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)

    comparison_path = run_dir / "comparison_mfm.json"
    if not comparison_path.exists():
        raise RuntimeError(
            f"Missing comparison_mfm.json for holdout={holdout_index} seed={seed}: {comparison_path}"
        )
    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    mode_metrics = {
        mode: _extract_mode_metrics(payload[mode])
        for mode in MODES
    }
    return {
        "holdout_index": int(holdout_index),
        "seed": int(seed),
        "run_dir": str(run_dir),
        "comparison_path": str(comparison_path),
        "mode_metrics": mode_metrics,
    }


def _mean_std(values: list[float | None]) -> tuple[float | None, float | None, int]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None, None, 0
    mean = float(sum(clean) / len(clean))
    std = float(statistics.pstdev(clean)) if len(clean) > 1 else 0.0
    return mean, std, len(clean)


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = [
        "linear_empirical_w2_avg",
        "learned_empirical_w2_avg",
        "delta_avg_learned_minus_linear",
        "linear_holdout_empirical_w2",
        "learned_holdout_empirical_w2",
        "delta_holdout_learned_minus_linear",
        "linear_full_ot_w2_avg",
        "learned_full_ot_w2_avg",
        "delta_avg_learned_minus_linear_full_ot",
        "linear_holdout_full_ot_w2",
        "learned_holdout_full_ot_w2",
        "delta_holdout_learned_minus_linear_full_ot",
    ]

    overall: dict[str, Any] = {}
    by_holdout: dict[str, Any] = {}
    by_seed: dict[str, Any] = {}

    for mode in MODES:
        row: dict[str, Any] = {}
        for key in metric_keys:
            values = [item["mode_metrics"][mode][key] for item in results]
            mean, std, n = _mean_std(values)
            row[f"{key}_mean"] = mean
            row[f"{key}_std"] = std
            row[f"{key}_n"] = n
        overall[mode] = row

    holdouts = sorted({int(item["holdout_index"]) for item in results})
    for holdout in holdouts:
        subset = [item for item in results if int(item["holdout_index"]) == holdout]
        holdout_payload: dict[str, Any] = {}
        for mode in MODES:
            row: dict[str, Any] = {}
            for key in metric_keys:
                values = [item["mode_metrics"][mode][key] for item in subset]
                mean, std, n = _mean_std(values)
                row[f"{key}_mean"] = mean
                row[f"{key}_std"] = std
                row[f"{key}_n"] = n
            holdout_payload[mode] = row
        by_holdout[str(holdout)] = holdout_payload

    seeds = sorted({int(item["seed"]) for item in results})
    for seed in seeds:
        subset = [item for item in results if int(item["seed"]) == seed]
        seed_payload: dict[str, Any] = {}
        for mode in MODES:
            row: dict[str, Any] = {}
            for key in metric_keys:
                values = [item["mode_metrics"][mode][key] for item in subset]
                mean, std, n = _mean_std(values)
                row[f"{key}_mean"] = mean
                row[f"{key}_std"] = std
                row[f"{key}_n"] = n
            seed_payload[mode] = row
        by_seed[str(seed)] = seed_payload

    leaderboard_rows: list[dict[str, Any]] = []
    for mode in MODES:
        row = overall[mode]
        leaderboard_rows.append(
            {
                "mode": mode,
                "learned_holdout_full_ot_w2_mean": row["learned_holdout_full_ot_w2_mean"],
                "linear_holdout_full_ot_w2_mean": row["linear_holdout_full_ot_w2_mean"],
                "delta_holdout_learned_minus_linear_full_ot_mean": row[
                    "delta_holdout_learned_minus_linear_full_ot_mean"
                ],
                "learned_full_ot_w2_avg_mean": row["learned_full_ot_w2_avg_mean"],
                "linear_full_ot_w2_avg_mean": row["linear_full_ot_w2_avg_mean"],
                "delta_avg_learned_minus_linear_full_ot_mean": row[
                    "delta_avg_learned_minus_linear_full_ot_mean"
                ],
                "learned_holdout_empirical_w2_mean": row["learned_holdout_empirical_w2_mean"],
                "linear_holdout_empirical_w2_mean": row["linear_holdout_empirical_w2_mean"],
                "delta_holdout_learned_minus_linear_mean": row[
                    "delta_holdout_learned_minus_linear_mean"
                ],
                "learned_empirical_w2_avg_mean": row["learned_empirical_w2_avg_mean"],
                "linear_empirical_w2_avg_mean": row["linear_empirical_w2_avg_mean"],
                "delta_avg_learned_minus_linear_mean": row["delta_avg_learned_minus_linear_mean"],
            }
        )
    def _leaderboard_sort_value(item: dict[str, Any]) -> float:
        robust = item.get("learned_holdout_full_ot_w2_mean")
        if robust is not None:
            return float(robust)
        legacy = item.get("learned_holdout_empirical_w2_mean")
        if legacy is not None:
            return float(legacy)
        return float("inf")

    leaderboard_rows.sort(key=_leaderboard_sort_value)

    return {
        "overall_by_mode": overall,
        "by_holdout": by_holdout,
        "by_seed": by_seed,
        "leaderboard": leaderboard_rows,
    }


def _write_leaderboard_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    header = [
        "rank",
        "mode",
        "learned_holdout_full_ot_w2_mean",
        "linear_holdout_full_ot_w2_mean",
        "delta_holdout_learned_minus_linear_full_ot_mean",
        "learned_full_ot_w2_avg_mean",
        "linear_full_ot_w2_avg_mean",
        "delta_avg_learned_minus_linear_full_ot_mean",
        "learned_holdout_empirical_w2_mean",
        "linear_holdout_empirical_w2_mean",
        "delta_holdout_learned_minus_linear_mean",
        "learned_empirical_w2_avg_mean",
        "linear_empirical_w2_avg_mean",
        "delta_avg_learned_minus_linear_mean",
    ]
    lines = ["\t".join(header)]
    for idx, row in enumerate(rows, start=1):
        fields = [str(idx), str(row["mode"])]
        for key in header[2:]:
            value = row.get(key)
            fields.append("" if value is None else f"{float(value):.8f}")
        lines.append("\t".join(fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    if not args.data_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {args.data_path}")

    run_root = args.run_root.resolve() if args.run_root is not None else _default_run_root()
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[info] benchmark_root={run_root}")

    holdouts = [int(idx) for idx in args.holdouts]
    seeds = [int(seed) for seed in args.seeds]
    if not holdouts:
        raise ValueError("At least one holdout index must be provided.")
    if not seeds:
        raise ValueError("At least one seed must be provided.")

    results: list[dict[str, Any]] = []
    for holdout_index in holdouts:
        for seed in seeds:
            run_dir = run_root / f"holdout_{holdout_index}" / f"seed_{seed}"
            results.append(
                _run_trial(
                    data_path=args.data_path,
                    holdout_index=holdout_index,
                    seed=seed,
                    mfm_backend=str(args.mfm_backend),
                    run_dir=run_dir,
                )
            )

    aggregate = _aggregate(results)
    payload = {
        "meta": {
            "benchmark_label": "single_cell_eb_5d_stage_a_strict_leaveout",
            "dataset_path": str(args.data_path.resolve()),
            "holdout_indices": holdouts,
            "seeds": seeds,
            "mfm_backend": str(args.mfm_backend),
            "benchmark_root": str(run_root),
            "modes": MODES,
            "train_profile": "single_cell_stage_a_only",
            "experiment_preset": "comparison_mfm_single_cell_stage_a",
        },
        "runs": results,
        "aggregate": aggregate,
    }
    summary_path = run_root / "benchmark_summary_stage_a.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    leaderboard_path = run_root / "leaderboard_stage_a.tsv"
    _write_leaderboard_tsv(leaderboard_path, aggregate["leaderboard"])

    print(f"[done] wrote benchmark summary: {summary_path}")
    print(f"[done] wrote leaderboard: {leaderboard_path}")
    for row in aggregate["leaderboard"]:
        print(
            f"[leaderboard] mode={row['mode']} "
            f"learned_holdout_full_ot_w2_mean={row['learned_holdout_full_ot_w2_mean']} "
            f"learned_holdout_empirical_w2_mean={row['learned_holdout_empirical_w2_mean']} "
            f"delta_holdout_mean={row['delta_holdout_learned_minus_linear_mean']}"
        )


if __name__ == "__main__":
    main()
