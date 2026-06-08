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
METHODS = [
    "baseline",
    "constrained",
    "metric",
    "metric_constrained_al",
    "metric_constrained_soft",
]
STAGE_A_COLUMNS = ["0.25", "0.50", "0.75"]
AB_COLUMNS = ["0.25", "0.50", "0.75", "1.00"]


TRACKS: dict[str, dict[str, str]] = {
    "stage_a": {
        "experiment_preset": "comparison_mfm_bridge_global_3k_stage_a",
        "train_profile": "bridge_global_ot_3k_stage_a_only",
        "experiment_label": "bridge_global_ot_3k_stage_a",
    },
    "ab": {
        "experiment_preset": "comparison_mfm_bridge_global_3k",
        "train_profile": "bridge_global_ot_3k_ab_only",
        "experiment_label": "bridge_global_ot_3k_ab",
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run bridge 3k global-OT dual-track benchmark (Stage-A and A+B) "
            "across seeds and export paper-ready tables."
        )
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[3, 7, 11, 13, 17],
        help="Random seeds to run. Default: 3 7 11 13 17",
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
            "outputs/<date>/bridge_global_ot_3k_dual/<time>"
        ),
    )
    parser.add_argument(
        "--continue-if-exists",
        action="store_true",
        help="Reuse an existing seed run folder if comparison_mfm.json already exists.",
    )
    return parser.parse_args()


def _default_run_root() -> Path:
    day = datetime.now().strftime("%Y-%m-%d")
    ts = datetime.now().strftime("%H-%M-%S")
    return (ROOT / "outputs" / day / "bridge_global_ot_3k_dual" / ts).resolve()


def _mean_std(values: list[float]) -> tuple[float, float, int]:
    if not values:
        raise ValueError("Cannot compute mean/std for an empty list.")
    mean = float(sum(values) / len(values))
    std = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    return mean, std, len(values)


def _format_pm(mean: float, std: float, precision: int = 4) -> str:
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def _run_track_seed(
    *,
    track: str,
    seed: int,
    mfm_backend: str,
    run_dir: Path,
    continue_if_exists: bool,
) -> dict[str, Any]:
    track_cfg = TRACKS[track]
    run_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = run_dir / "comparison_mfm.json"
    if continue_if_exists and comparison_path.exists():
        return json.loads(comparison_path.read_text(encoding="utf-8"))

    cmd = [
        sys.executable,
        str(RUN_EXPERIMENT),
        f"experiment={track_cfg['experiment_preset']}",
        f"train={track_cfg['train_profile']}",
        "data=bridge_ot_global_3k",
        f"experiment.label={track_cfg['experiment_label']}",
        f"seed={int(seed)}",
        f"mfm.backend={mfm_backend}",
        "output.save_plots=false",
        f"hydra.run.dir={str(run_dir)}",
    ]
    print(f"[run] track={track} seed={seed} cmd={' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not comparison_path.exists():
        raise RuntimeError(f"Missing comparison_mfm.json at {comparison_path}")
    return json.loads(comparison_path.read_text(encoding="utf-8"))


def _extract_stage_a_mode_metrics(summary: dict[str, Any]) -> dict[str, float]:
    interpolant = summary.get("interpolant_eval")
    if not isinstance(interpolant, dict):
        raise RuntimeError("Missing interpolant_eval in Stage-A summary.")
    learned = interpolant.get("learned_empirical_w2")
    if not isinstance(learned, dict):
        raise RuntimeError("Missing interpolant_eval.learned_empirical_w2 in Stage-A summary.")
    out: dict[str, float] = {}
    for key in STAGE_A_COLUMNS:
        if key not in learned:
            raise RuntimeError(f"Missing Stage-A OT key {key} in learned_empirical_w2.")
        out[key] = float(learned[key])
    return out


def _extract_ab_mode_metrics(summary: dict[str, Any]) -> dict[str, float]:
    intermediate = summary.get("intermediate_empirical_w2")
    if not isinstance(intermediate, dict):
        raise RuntimeError("Missing intermediate_empirical_w2 in A+B summary.")
    out: dict[str, float] = {}
    for key in STAGE_A_COLUMNS:
        if key not in intermediate:
            raise RuntimeError(f"Missing A+B OT key {key} in intermediate_empirical_w2.")
        out[key] = float(intermediate[key])
    endpoint = summary.get("transport_endpoint_empirical_w2")
    if endpoint is None:
        raise RuntimeError("Missing transport_endpoint_empirical_w2 in A+B summary.")
    out["1.00"] = float(endpoint)
    return out


def _aggregate_track(
    *,
    track: str,
    per_seed_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    columns = STAGE_A_COLUMNS if track == "stage_a" else AB_COLUMNS
    by_mode: dict[str, Any] = {}
    for mode in METHODS:
        per_metric_values: dict[str, list[float]] = {col: [] for col in columns}
        per_seed: dict[str, dict[str, float]] = {}
        for row in per_seed_payloads:
            seed = str(int(row["seed"]))
            summary = row["comparison"][mode]
            if track == "stage_a":
                metrics = _extract_stage_a_mode_metrics(summary)
            else:
                metrics = _extract_ab_mode_metrics(summary)
            per_seed[seed] = metrics
            for col in columns:
                per_metric_values[col].append(float(metrics[col]))

        agg_row: dict[str, Any] = {
            "per_seed": per_seed,
            "columns": columns,
            "values": {},
        }
        for col in columns:
            mean, std, n = _mean_std(per_metric_values[col])
            agg_row["values"][col] = {
                "mean": mean,
                "std": std,
                "n": n,
                "mean_pm_std": _format_pm(mean, std),
            }
        by_mode[mode] = agg_row
    return {
        "track": track,
        "columns": columns,
        "by_mode": by_mode,
    }


def _write_table_csv(path: Path, aggregate: dict[str, Any]) -> None:
    columns: list[str] = list(aggregate["columns"])
    lines = ["method," + ",".join(f"OT@{col}" for col in columns)]
    for mode in METHODS:
        row = aggregate["by_mode"][mode]
        formatted = [row["values"][col]["mean_pm_std"] for col in columns]
        lines.append(",".join([mode, *formatted]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_table_md(path: Path, aggregate: dict[str, Any]) -> None:
    columns: list[str] = list(aggregate["columns"])
    header = "| Method | " + " | ".join(f"OT@{col}" for col in columns) + " |"
    divider = "|---|" + "|".join(["---"] * len(columns)) + "|"
    rows = [header, divider]
    for mode in METHODS:
        row = aggregate["by_mode"][mode]
        formatted = [row["values"][col]["mean_pm_std"] for col in columns]
        rows.append("| " + " | ".join([mode, *formatted]) + " |")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_table_tex(path: Path, aggregate: dict[str, Any]) -> None:
    columns: list[str] = list(aggregate["columns"])
    spec = "l" + "c" * len(columns)
    header = "Method & " + " & ".join(f"OT@{col}" for col in columns) + " \\\\"
    rows = []
    for mode in METHODS:
        row = aggregate["by_mode"][mode]
        formatted = [row["values"][col]["mean_pm_std"].replace("±", r"$\\pm$") for col in columns]
        rows.append(f"{mode} & " + " & ".join(formatted) + " \\\\")
    tex = [
        r"\begin{tabular}{" + spec + "}",
        r"\toprule",
        header,
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    path.write_text("\n".join(tex), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    seeds = [int(seed) for seed in args.seeds]
    if not seeds:
        raise ValueError("At least one seed must be provided.")

    run_root = args.run_root.resolve() if args.run_root is not None else _default_run_root()
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[info] benchmark_root={run_root}")

    runs: dict[str, list[dict[str, Any]]] = {"stage_a": [], "ab": []}
    for track in ["stage_a", "ab"]:
        for seed in seeds:
            run_dir = run_root / track / f"seed_{seed}"
            comparison = _run_track_seed(
                track=track,
                seed=seed,
                mfm_backend=str(args.mfm_backend),
                run_dir=run_dir,
                continue_if_exists=bool(args.continue_if_exists),
            )
            runs[track].append(
                {
                    "seed": int(seed),
                    "run_dir": str(run_dir),
                    "comparison_path": str(run_dir / "comparison_mfm.json"),
                    "comparison": comparison,
                }
            )

    stage_a_agg = _aggregate_track(track="stage_a", per_seed_payloads=runs["stage_a"])
    ab_agg = _aggregate_track(track="ab", per_seed_payloads=runs["ab"])

    artifacts_dir = run_root / "tables"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    _write_table_csv(artifacts_dir / "stage_a_table.csv", stage_a_agg)
    _write_table_csv(artifacts_dir / "ab_table.csv", ab_agg)
    _write_table_md(artifacts_dir / "stage_a_table.md", stage_a_agg)
    _write_table_md(artifacts_dir / "ab_table.md", ab_agg)
    _write_table_tex(artifacts_dir / "stage_a_table.tex", stage_a_agg)
    _write_table_tex(artifacts_dir / "ab_table.tex", ab_agg)

    payload = {
        "meta": {
            "benchmark_label": "bridge_global_ot_3k_dual",
            "benchmark_root": str(run_root),
            "seeds": seeds,
            "mfm_backend": str(args.mfm_backend),
            "methods": METHODS,
            "stage_a_columns": STAGE_A_COLUMNS,
            "ab_columns": AB_COLUMNS,
            "data_preset": "bridge_ot_global_3k",
            "stage_a_experiment_preset": TRACKS["stage_a"]["experiment_preset"],
            "ab_experiment_preset": TRACKS["ab"]["experiment_preset"],
            "stage_a_train_profile": TRACKS["stage_a"]["train_profile"],
            "ab_train_profile": TRACKS["ab"]["train_profile"],
        },
        "runs": {
            "stage_a": [
                {
                    "seed": row["seed"],
                    "run_dir": row["run_dir"],
                    "comparison_path": row["comparison_path"],
                }
                for row in runs["stage_a"]
            ],
            "ab": [
                {
                    "seed": row["seed"],
                    "run_dir": row["run_dir"],
                    "comparison_path": row["comparison_path"],
                }
                for row in runs["ab"]
            ],
        },
        "aggregate": {
            "stage_a": stage_a_agg,
            "ab": ab_agg,
        },
    }
    summary_path = run_root / "benchmark_summary_bridge_global_ot_3k_dual.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[done] wrote summary: {summary_path}")
    print(f"[done] wrote tables dir: {artifacts_dir}")


if __name__ == "__main__":
    main()
