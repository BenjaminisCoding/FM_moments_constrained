#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_EXPERIMENT = ROOT / "scripts" / "run_experiment.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run EB 5D single-cell strict-leaveout benchmark across holdouts "
            "with full 6-way comparison."
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for all holdout runs. Default: 42",
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
            "outputs/<date>/single_cell_eb_5d_strict_leaveout/<time>"
        ),
    )
    return parser.parse_args()


def _default_run_root() -> Path:
    day = datetime.now().strftime("%Y-%m-%d")
    ts = datetime.now().strftime("%H-%M-%S")
    return (ROOT / "outputs" / day / "single_cell_eb_5d_strict_leaveout" / ts).resolve()


def _safe_float(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if value is None:
        return None
    return float(value)


def _run_holdout(
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
        "experiment=comparison_mfm_single_cell",
        "train=single_cell_ab_only",
        "data=single_cell_eb_5d",
        "experiment.label=single_cell_eb_5d_strict_leaveout",
        "experiment.protocol=strict_leaveout",
        f"experiment.holdout_index={int(holdout_index)}",
        "data.constraint_time_policy=observed_nonendpoint_excluding_holdout",
        f"data.single_cell.path={str(data_path.resolve())}",
        f"seed={int(seed)}",
        f"mfm.backend={mfm_backend}",
        f"hydra.run.dir={str(run_dir)}",
    ]
    print(f"[run] holdout={holdout_index} cmd={' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)

    comparison_path = run_dir / "comparison_mfm.json"
    if not comparison_path.exists():
        raise RuntimeError(f"Missing comparison_mfm.json for holdout={holdout_index}: {comparison_path}")
    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    return {
        "holdout_index": int(holdout_index),
        "run_dir": str(run_dir),
        "comparison_path": str(comparison_path),
        "comparison_mfm": payload,
    }


def _aggregate_holdout_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    modes = [
        "baseline",
        "constrained",
        "metric",
        "metric_alpha0",
        "metric_constrained_al",
        "metric_constrained_soft",
    ]
    metrics = [
        "intermediate_empirical_w2_avg",
        "transport_endpoint_empirical_w2",
        "holdout_empirical_w2",
        "holdout_empirical_w1",
        "intermediate_full_ot_w2_avg",
        "transport_endpoint_full_ot_w2",
        "holdout_full_ot_w2",
    ]
    aggregate: dict[str, Any] = {}
    for mode in modes:
        per_mode: dict[str, Any] = {}
        for key in metrics:
            values: list[float] = []
            for item in results:
                summary = item["comparison_mfm"][mode]
                value = _safe_float(summary, key)
                if value is not None:
                    values.append(value)
            per_mode[f"{key}_mean"] = None if not values else float(sum(values) / len(values))
        aggregate[mode] = per_mode
    return aggregate


def main() -> None:
    args = _parse_args()
    if not args.data_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {args.data_path}")

    run_root = args.run_root.resolve() if args.run_root is not None else _default_run_root()
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[info] benchmark_root={run_root}")

    holdouts = [int(idx) for idx in args.holdouts]
    if not holdouts:
        raise ValueError("At least one holdout index must be provided.")

    holdout_results: list[dict[str, Any]] = []
    for holdout_index in holdouts:
        holdout_dir = run_root / f"holdout_{holdout_index}"
        holdout_results.append(
            _run_holdout(
                data_path=args.data_path,
                holdout_index=holdout_index,
                seed=int(args.seed),
                mfm_backend=str(args.mfm_backend),
                run_dir=holdout_dir,
            )
        )

    aggregate = _aggregate_holdout_results(holdout_results)
    payload = {
        "meta": {
            "benchmark_label": "single_cell_eb_5d_strict_leaveout",
            "seed": int(args.seed),
            "mfm_backend": str(args.mfm_backend),
            "dataset_path": str(args.data_path.resolve()),
            "holdout_indices": holdouts,
            "benchmark_root": str(run_root),
        },
        "holdouts": [
            {
                "holdout_index": item["holdout_index"],
                "run_dir": item["run_dir"],
                "comparison_path": item["comparison_path"],
            }
            for item in holdout_results
        ],
        "aggregate": aggregate,
    }
    summary_path = run_root / "benchmark_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[done] wrote benchmark summary: {summary_path}")
    for mode, stats in aggregate.items():
        print(
            f"[summary] {mode} "
            f"intermediate_empirical_w2_mean={stats['intermediate_empirical_w2_avg_mean']} "
            f"endpoint_w2_mean={stats['transport_endpoint_empirical_w2_mean']} "
            f"holdout_w2_mean={stats['holdout_empirical_w2_mean']} "
            f"holdout_w1_mean={stats['holdout_empirical_w1_mean']} "
            f"intermediate_full_ot_w2_mean={stats['intermediate_full_ot_w2_avg_mean']} "
            f"endpoint_full_ot_w2_mean={stats['transport_endpoint_full_ot_w2_mean']} "
            f"holdout_full_ot_w2_mean={stats['holdout_full_ot_w2_mean']}"
        )


if __name__ == "__main__":
    main()
