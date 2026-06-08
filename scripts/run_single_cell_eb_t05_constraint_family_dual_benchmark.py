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
DATA_PRESET = "single_cell_eb_5d_t05_ot_global"
REFERENCE_METHODS = ["baseline", "metric"]
FAMILY_METHODS = ["constrained", "metric_constrained_al", "metric_constrained_soft"]
FAMILIES = ["moments", "classifier_only", "classifier_plus_moments"]
STAGE_A_COLUMNS = ["0.25", "0.50", "0.75"]
AB_COLUMNS = ["0.25", "0.50", "0.75", "1.00"]
CLASSIFIER_PSEUDO_ETA = 320.0
CLASSIFIER_PSEUDO_RHO = 5.0
CLASSIFIER_PSEUDO_LAMBDA_CLIP = 100.0

TRACKS: dict[str, dict[str, str]] = {
    "stage_a": {
        "experiment_preset": "comparison_mfm_single_cell_eb_t05_stage_a",
        "train_profile": "single_cell_eb_t05_stage_a_only",
        "experiment_label": "single_cell_eb_t05_stage_a_constraint_families",
    },
    "ab": {
        "experiment_preset": "comparison_mfm_single_cell_eb_t05",
        "train_profile": "single_cell_eb_t05_ab_only",
        "experiment_label": "single_cell_eb_t05_ab_constraint_families",
    },
}

ROW_LAYOUT: list[dict[str, str | None]] = [
    {
        "row_id": "baseline",
        "display": "baseline",
        "method": "baseline",
        "family": None,
    },
    {
        "row_id": "metric",
        "display": "metric",
        "method": "metric",
        "family": None,
    },
    {
        "row_id": "constrained__moments",
        "display": "constrained (moments)",
        "method": "constrained",
        "family": "moments",
    },
    {
        "row_id": "constrained__classifier_only",
        "display": "constrained (classifier only)",
        "method": "constrained",
        "family": "classifier_only",
    },
    {
        "row_id": "constrained__classifier_plus_moments",
        "display": "constrained (classifier + moments)",
        "method": "constrained",
        "family": "classifier_plus_moments",
    },
    {
        "row_id": "metric_constrained_al__moments",
        "display": "metric_constrained_al (moments)",
        "method": "metric_constrained_al",
        "family": "moments",
    },
    {
        "row_id": "metric_constrained_al__classifier_only",
        "display": "metric_constrained_al (classifier only)",
        "method": "metric_constrained_al",
        "family": "classifier_only",
    },
    {
        "row_id": "metric_constrained_al__classifier_plus_moments",
        "display": "metric_constrained_al (classifier + moments)",
        "method": "metric_constrained_al",
        "family": "classifier_plus_moments",
    },
    {
        "row_id": "metric_constrained_soft__moments",
        "display": "metric_constrained_soft (moments)",
        "method": "metric_constrained_soft",
        "family": "moments",
    },
    {
        "row_id": "metric_constrained_soft__classifier_only",
        "display": "metric_constrained_soft (classifier only)",
        "method": "metric_constrained_soft",
        "family": "classifier_only",
    },
    {
        "row_id": "metric_constrained_soft__classifier_plus_moments",
        "display": "metric_constrained_soft (classifier + moments)",
        "method": "metric_constrained_soft",
        "family": "classifier_plus_moments",
    },
]

GROUPS = [
    {"name": "reference", "methods": REFERENCE_METHODS},
    {"name": "moments", "methods": FAMILY_METHODS},
    {"name": "classifier_only", "methods": FAMILY_METHODS},
    {"name": "classifier_plus_moments", "methods": FAMILY_METHODS},
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run EB no-leaveout p=0.5 constraint-family dual-track benchmark "
            "(Stage-A and A+B), and export paper tables from full-OT W2 metrics."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Full local path to EB dataset (.npz or .h5ad).",
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
            "outputs/<date>/single_cell_eb_t05_constraint_family_dual/<time>"
        ),
    )
    parser.add_argument(
        "--continue-if-exists",
        action="store_true",
        help="Reuse an existing group run folder if comparison_mfm.json already exists.",
    )
    parser.add_argument(
        "--stage-a-steps-override",
        type=int,
        default=None,
        help=(
            "Optional override for train.stage_a_steps in both tracks. "
            "Useful for step-sufficiency checks."
        ),
    )
    parser.add_argument(
        "--stage-b-steps-override",
        type=int,
        default=None,
        help=(
            "Optional override for train.stage_b_steps in A+B track only. "
            "Ignored for stage_a track."
        ),
    )
    return parser.parse_args()


def _default_run_root() -> Path:
    day = datetime.now().strftime("%Y-%m-%d")
    ts = datetime.now().strftime("%H-%M-%S")
    return (ROOT / "outputs" / day / "single_cell_eb_t05_constraint_family_dual" / ts).resolve()


def _mean_std(values: list[float]) -> tuple[float, float, int]:
    if not values:
        raise ValueError("Cannot compute mean/std for an empty list.")
    mean = float(sum(values) / len(values))
    std = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    return mean, std, len(values)


def _format_pm(mean: float, std: float, precision: int = 4) -> str:
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def _methods_override(methods: list[str]) -> str:
    return "[" + ",".join(methods) + "]"


def _classifier_overrides(*, classifier_only: bool) -> list[str]:
    overrides = [
        "data.single_cell.pseudo_labels.enabled=true",
        "data.single_cell.pseudo_labels.method=gmm",
        "data.single_cell.pseudo_labels.fit_times_normalized=[0.5]",
        f"train.pseudo_eta={CLASSIFIER_PSEUDO_ETA:.1f}",
        f"train.pseudo_rho={CLASSIFIER_PSEUDO_RHO:.1f}",
        f"train.pseudo_lambda_clip={CLASSIFIER_PSEUDO_LAMBDA_CLIP:.1f}",
    ]
    if classifier_only:
        overrides.extend(
            [
                "train.moment_eta=0.0",
                "experiment.method_overrides.metric_constrained_al.mfm.moment_eta=0.0",
                "experiment.method_overrides.metric_constrained_soft.mfm.moment_eta=0.0",
            ]
        )
    return overrides


def _group_overrides(group_name: str) -> list[str]:
    if group_name == "reference":
        return [
            "data.single_cell.pseudo_labels.enabled=false",
            "train.pseudo_eta=0.0",
            "train.moment_eta=1.0",
        ]
    if group_name == "moments":
        return [
            "data.single_cell.pseudo_labels.enabled=false",
            "train.pseudo_eta=0.0",
            "train.moment_eta=1.0",
        ]
    if group_name == "classifier_only":
        return _classifier_overrides(classifier_only=True)
    if group_name == "classifier_plus_moments":
        return _classifier_overrides(classifier_only=False)
    raise ValueError(f"Unsupported group '{group_name}'.")


def _run_group(
    *,
    track: str,
    group_name: str,
    methods: list[str],
    seed: int,
    data_path: Path,
    mfm_backend: str,
    run_dir: Path,
    continue_if_exists: bool,
    stage_a_steps_override: int | None,
    stage_b_steps_override: int | None,
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
        f"data={DATA_PRESET}",
        f"experiment.label={track_cfg['experiment_label']}",
        f"experiment.comparison_methods={_methods_override(methods)}",
        f"data.single_cell.path={str(data_path.resolve())}",
        f"seed={int(seed)}",
        f"mfm.backend={mfm_backend}",
        "output.save_plots=false",
        f"hydra.run.dir={str(run_dir)}",
    ]
    if stage_a_steps_override is not None:
        cmd.append(f"train.stage_a_steps={int(stage_a_steps_override)}")
    if track == "stage_a":
        # Stage-A-only track must keep velocity stages disabled.
        cmd.extend(["train.stage_b_steps=0", "train.stage_c_steps=0"])
    else:
        if stage_b_steps_override is not None:
            cmd.append(f"train.stage_b_steps={int(stage_b_steps_override)}")
        cmd.append("train.stage_c_steps=0")
    cmd.extend(_group_overrides(group_name))

    print(
        f"[run] track={track} seed={seed} group={group_name} cmd={' '.join(cmd)}"
    )
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not comparison_path.exists():
        raise RuntimeError(f"Missing comparison_mfm.json at {comparison_path}")
    return json.loads(comparison_path.read_text(encoding="utf-8"))


def _extract_required_metric(metric_map: dict[str, Any], key: str) -> float:
    if key in metric_map:
        return float(metric_map[key])
    alt = f"{float(key):.2f}"
    if alt in metric_map:
        return float(metric_map[alt])
    raise RuntimeError(f"Missing metric key '{key}' in payload keys={list(metric_map.keys())}.")


def _extract_stage_a_metrics(summary: dict[str, Any]) -> dict[str, float]:
    interpolant_eval = summary.get("interpolant_eval")
    if not isinstance(interpolant_eval, dict):
        raise RuntimeError("Missing interpolant_eval in Stage-A summary.")
    full_ot = interpolant_eval.get("learned_full_ot_w2")
    if not isinstance(full_ot, dict):
        raise RuntimeError("Missing interpolant_eval.learned_full_ot_w2 in Stage-A summary.")
    return {key: _extract_required_metric(full_ot, key) for key in STAGE_A_COLUMNS}


def _extract_ab_metrics(summary: dict[str, Any]) -> dict[str, float]:
    intermediate = summary.get("intermediate_full_ot_w2")
    if not isinstance(intermediate, dict):
        raise RuntimeError("Missing intermediate_full_ot_w2 in A+B summary.")
    out: dict[str, float] = {}
    for key in STAGE_A_COLUMNS:
        try:
            out[key] = _extract_required_metric(intermediate, key)
        except RuntimeError:
            out[key] = float("nan")
    endpoint = summary.get("transport_endpoint_full_ot_w2")
    if endpoint is None:
        raise RuntimeError("Missing transport_endpoint_full_ot_w2 in A+B summary.")
    out["1.00"] = float(endpoint)
    return out


def _build_seed_rows(
    *,
    track: str,
    group_payloads: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    per_row: dict[str, dict[str, float]] = {}
    for spec in ROW_LAYOUT:
        row_id = str(spec["row_id"])
        method = str(spec["method"])
        family = spec["family"]
        if family is None:
            source = group_payloads["reference"]
        else:
            source = group_payloads[str(family)]
        if method not in source:
            raise RuntimeError(f"Missing method '{method}' in group payload for row {row_id}.")
        summary = source[method]
        if track == "stage_a":
            per_row[row_id] = _extract_stage_a_metrics(summary)
        else:
            per_row[row_id] = _extract_ab_metrics(summary)
    return per_row


def _aggregate_track(
    *,
    track: str,
    per_seed_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    columns = STAGE_A_COLUMNS if track == "stage_a" else AB_COLUMNS
    row_by_id = {str(spec["row_id"]): spec for spec in ROW_LAYOUT}
    rows_out: list[dict[str, Any]] = []

    for spec in ROW_LAYOUT:
        row_id = str(spec["row_id"])
        display = str(spec["display"])
        per_metric_values: dict[str, list[float]] = {col: [] for col in columns}
        per_seed: dict[str, dict[str, float]] = {}
        for payload in per_seed_rows:
            seed = str(int(payload["seed"]))
            metrics = payload["rows"][row_id]
            per_seed[seed] = metrics
            for col in columns:
                per_metric_values[col].append(float(metrics[col]))

        values: dict[str, dict[str, float | int | str]] = {}
        for col in columns:
            mean, std, n = _mean_std(per_metric_values[col])
            values[col] = {
                "mean": mean,
                "std": std,
                "n": n,
                "mean_pm_std": _format_pm(mean, std),
            }

        rows_out.append(
            {
                "row_id": row_id,
                "display": display,
                "method": str(row_by_id[row_id]["method"]),
                "family": row_by_id[row_id]["family"],
                "columns": columns,
                "per_seed": per_seed,
                "values": values,
            }
        )

    return {
        "track": track,
        "columns": columns,
        "rows": rows_out,
    }


def _write_table_csv(path: Path, aggregate: dict[str, Any]) -> None:
    columns: list[str] = list(aggregate["columns"])
    lines = ["row_id,method_family," + ",".join(f"OT@{col}" for col in columns)]
    for row in aggregate["rows"]:
        formatted = [row["values"][col]["mean_pm_std"] for col in columns]
        lines.append(",".join([row["row_id"], row["display"], *formatted]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_table_md(path: Path, aggregate: dict[str, Any]) -> None:
    columns: list[str] = list(aggregate["columns"])
    header = "| Method/Family | " + " | ".join(f"OT@{col}" for col in columns) + " |"
    divider = "|---|" + "|".join(["---"] * len(columns)) + "|"
    rows = [header, divider]
    for row in aggregate["rows"]:
        formatted = [row["values"][col]["mean_pm_std"] for col in columns]
        rows.append("| " + " | ".join([row["display"], *formatted]) + " |")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_table_tex(path: Path, aggregate: dict[str, Any]) -> None:
    columns: list[str] = list(aggregate["columns"])
    spec = "l" + "c" * len(columns)
    header = "Method/Family & " + " & ".join(f"OT@{col}" for col in columns) + " \\\\"
    rows = []
    for row in aggregate["rows"]:
        formatted = [row["values"][col]["mean_pm_std"].replace("±", r"$\\pm$") for col in columns]
        rows.append(f"{row['display']} & " + " & ".join(formatted) + " \\\\")
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
    if not args.data_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {args.data_path}")

    seeds = [int(seed) for seed in args.seeds]
    if not seeds:
        raise ValueError("At least one seed must be provided.")

    run_root = args.run_root.resolve() if args.run_root is not None else _default_run_root()
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[info] benchmark_root={run_root}")

    runs: dict[str, list[dict[str, Any]]] = {"stage_a": [], "ab": []}
    for track in ["stage_a", "ab"]:
        for seed in seeds:
            track_seed_dir = run_root / track / f"seed_{seed}"
            group_payloads: dict[str, dict[str, Any]] = {}
            group_records: dict[str, dict[str, Any]] = {}
            for group_spec in GROUPS:
                group_name = str(group_spec["name"])
                methods = [str(v) for v in group_spec["methods"]]
                group_dir = track_seed_dir / group_name
                comparison = _run_group(
                    track=track,
                    group_name=group_name,
                    methods=methods,
                    seed=int(seed),
                    data_path=Path(args.data_path),
                    mfm_backend=str(args.mfm_backend),
                    run_dir=group_dir,
                    continue_if_exists=bool(args.continue_if_exists),
                    stage_a_steps_override=(
                        int(args.stage_a_steps_override)
                        if args.stage_a_steps_override is not None
                        else None
                    ),
                    stage_b_steps_override=(
                        int(args.stage_b_steps_override)
                        if args.stage_b_steps_override is not None
                        else None
                    ),
                )
                group_payloads[group_name] = comparison
                group_records[group_name] = {
                    "methods": methods,
                    "run_dir": str(group_dir),
                    "comparison_path": str(group_dir / "comparison_mfm.json"),
                }

            rows = _build_seed_rows(track=track, group_payloads=group_payloads)
            runs[track].append(
                {
                    "seed": int(seed),
                    "rows": rows,
                    "groups": group_records,
                }
            )

    stage_a_agg = _aggregate_track(track="stage_a", per_seed_rows=runs["stage_a"])
    ab_agg = _aggregate_track(track="ab", per_seed_rows=runs["ab"])

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
            "benchmark_label": "single_cell_eb_t05_constraint_family_dual",
            "benchmark_root": str(run_root),
            "data_path": str(args.data_path.resolve()),
            "data_preset": DATA_PRESET,
            "seeds": seeds,
            "mfm_backend": str(args.mfm_backend),
            "stage_a_columns": STAGE_A_COLUMNS,
            "ab_columns": AB_COLUMNS,
            "reference_methods": REFERENCE_METHODS,
            "family_methods": FAMILY_METHODS,
            "families": FAMILIES,
            "classifier_pseudo_eta": CLASSIFIER_PSEUDO_ETA,
            "classifier_pseudo_rho": CLASSIFIER_PSEUDO_RHO,
            "classifier_pseudo_lambda_clip": CLASSIFIER_PSEUDO_LAMBDA_CLIP,
            "stage_a_experiment_preset": TRACKS["stage_a"]["experiment_preset"],
            "ab_experiment_preset": TRACKS["ab"]["experiment_preset"],
            "stage_a_train_profile": TRACKS["stage_a"]["train_profile"],
            "ab_train_profile": TRACKS["ab"]["train_profile"],
            "stage_a_steps_override": (
                int(args.stage_a_steps_override)
                if args.stage_a_steps_override is not None
                else None
            ),
            "stage_b_steps_override": (
                int(args.stage_b_steps_override)
                if args.stage_b_steps_override is not None
                else None
            ),
        },
        "runs": runs,
        "aggregate": {
            "stage_a": stage_a_agg,
            "ab": ab_agg,
        },
    }
    summary_path = run_root / "benchmark_summary_single_cell_eb_t05_constraint_family_dual.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[done] wrote summary: {summary_path}")
    print(f"[done] wrote tables dir: {artifacts_dir}")


if __name__ == "__main__":
    main()
