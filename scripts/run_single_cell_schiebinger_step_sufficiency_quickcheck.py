#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_EXPERIMENT = ROOT / "scripts" / "run_experiment.py"
DATA_PRESET = "single_cell_schiebinger_pilot_20d"
METHODS = ["constrained", "metric_constrained_al", "metric_constrained_soft"]

PRIMARY_GAIN_THRESHOLD_PCT = 1.5
ENDPOINT_GUARDRAIL_WORSE_PCT = 1.5

CLASSIFIER_PSEUDO_ETA = 1.0
CLASSIFIER_PSEUDO_RHO = 5.0
CLASSIFIER_PSEUDO_LAMBDA_CLIP = 100.0


@dataclass(frozen=True)
class StepCandidate:
    candidate_id: str
    stage_a_steps: int
    stage_b_steps: int


@dataclass(frozen=True)
class TrackSpec:
    track: str
    experiment_preset: str
    train_profile: str
    experiment_label: str
    candidates: tuple[StepCandidate, ...]
    default_candidate_id: str


TRACKS: dict[str, TrackSpec] = {
    "stage_a": TrackSpec(
        track="stage_a",
        experiment_preset="comparison_mfm_single_cell_schiebinger_stage_a",
        train_profile="single_cell_stage_a_only",
        experiment_label="single_cell_schiebinger_step_sufficiency_stage_a",
        candidates=(
            StepCandidate(candidate_id="a80_b0", stage_a_steps=80, stage_b_steps=0),
            StepCandidate(candidate_id="a120_b0", stage_a_steps=120, stage_b_steps=0),
            StepCandidate(candidate_id="a160_b0", stage_a_steps=160, stage_b_steps=0),
        ),
        default_candidate_id="a120_b0",
    ),
    "ab": TrackSpec(
        track="ab",
        experiment_preset="comparison_mfm_single_cell_schiebinger",
        train_profile="single_cell_ab_only",
        experiment_label="single_cell_schiebinger_step_sufficiency_ab",
        candidates=(
            StepCandidate(candidate_id="a120_b120", stage_a_steps=120, stage_b_steps=120),
            StepCandidate(candidate_id="a120_b180", stage_a_steps=120, stage_b_steps=180),
            StepCandidate(candidate_id="a160_b240", stage_a_steps=160, stage_b_steps=240),
        ),
        default_candidate_id="a120_b180",
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Schiebinger step-sufficiency quick check over Stage-A and A+B step grids "
            "for constrained-family methods and overfit/underfit supervised pseudo-classifiers."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=ROOT / "datasets" / "schiebinger_serum_d10_d10p5_d11_hvg_pca50.h5ad",
        help="Local path to Schiebinger pilot .h5ad file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3,
        help="Seed for this quick-check benchmark. Default: 3",
    )
    parser.add_argument(
        "--mfm-backend",
        type=str,
        default="auto",
        choices=["auto", "native", "torchcfm"],
        help="MFM backend policy.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help=(
            "Benchmark root folder. Default: "
            "outputs/<date>/single_cell_schiebinger_step_sufficiency_quickcheck/<time>"
        ),
    )
    parser.add_argument(
        "--continue-if-exists",
        action="store_true",
        help="Reuse existing candidate run folders if comparison_mfm.json is already present.",
    )
    return parser.parse_args()


def _default_run_root() -> Path:
    day = datetime.now().strftime("%Y-%m-%d")
    ts = datetime.now().strftime("%H-%M-%S")
    return (
        ROOT
        / "outputs"
        / day
        / "single_cell_schiebinger_step_sufficiency_quickcheck"
        / ts
    ).resolve()


def _methods_override(methods: list[str]) -> str:
    return "[" + ",".join(methods) + "]"


def _common_overrides(data_path: Path) -> list[str]:
    return [
        f"data.single_cell.path={str(data_path.resolve())}",
        "data.coupling=ot_global",
        "data.single_cell.constraint_times_normalized=[0.5]",
        "data.single_cell.eval_times_normalized=[0.5]",
        "data.single_cell.pseudo_labels.enabled=true",
        "data.single_cell.pseudo_labels.method=supervised_mlp",
        "data.single_cell.pseudo_labels.fit_times_normalized=[0.5]",
        f"train.pseudo_eta={CLASSIFIER_PSEUDO_ETA:.1f}",
        f"train.pseudo_rho={CLASSIFIER_PSEUDO_RHO:.1f}",
        f"train.pseudo_lambda_clip={CLASSIFIER_PSEUDO_LAMBDA_CLIP:.1f}",
        "train.moment_eta=0.0",
        "train.rho=0.0",
        "train.eval_full_ot_metrics=true",
        "train.eval_full_ot_method=pot_emd2",
        "train.eval_full_ot_num_itermax=1600000",
        "experiment.method_overrides.constrained.train.rho=0.0",
        "experiment.method_overrides.metric_constrained_al.train.rho=0.0",
        "experiment.method_overrides.metric_constrained_al.mfm.moment_eta=0.0",
        "experiment.method_overrides.metric_constrained_soft.mfm.moment_eta=0.0",
    ]


def _arm_overrides(arm: str) -> list[str]:
    arm_norm = str(arm).strip().lower()
    if arm_norm == "overfit":
        return [
            "data.single_cell.pseudo_labels.supervised_min_class_count=1",
            "data.single_cell.pseudo_labels.supervised_val_fraction=0.0",
            "experiment.method_overrides.constrained.train.alpha=1.0",
            "experiment.method_overrides.constrained.train.beta=0.05",
        ]
    if arm_norm == "underfit":
        return [
            "data.single_cell.pseudo_labels.supervised_min_class_count=2",
            "data.single_cell.pseudo_labels.supervised_val_fraction=0.2",
            "data.single_cell.pseudo_labels.supervised_split_seed=7",
            "data.single_cell.pseudo_labels.supervised_early_stopping_patience=20",
            "data.single_cell.pseudo_labels.supervised_early_stopping_min_epochs=20",
            "data.single_cell.pseudo_labels.supervised_early_stopping_min_delta=0.0",
            "experiment.method_overrides.constrained.train.alpha=1.5",
            "experiment.method_overrides.constrained.train.beta=0.12",
        ]
    raise ValueError(f"Unsupported arm '{arm}'. Expected one of: overfit, underfit.")


def _candidate_overrides(track: str, candidate: StepCandidate) -> list[str]:
    if track == "stage_a":
        return [
            f"train.stage_a_steps={int(candidate.stage_a_steps)}",
            "train.stage_b_steps=0",
            "train.stage_c_steps=0",
        ]
    if track == "ab":
        return [
            f"train.stage_a_steps={int(candidate.stage_a_steps)}",
            f"train.stage_b_steps={int(candidate.stage_b_steps)}",
            "train.stage_c_steps=0",
        ]
    raise ValueError(f"Unsupported track '{track}'.")


def _run_candidate(
    *,
    track_spec: TrackSpec,
    arm: str,
    candidate: StepCandidate,
    seed: int,
    data_path: Path,
    mfm_backend: str,
    run_dir: Path,
    continue_if_exists: bool,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = run_dir / "comparison_mfm.json"
    if continue_if_exists and comparison_path.exists():
        return json.loads(comparison_path.read_text(encoding="utf-8"))

    cmd = [
        sys.executable,
        str(RUN_EXPERIMENT),
        f"experiment={track_spec.experiment_preset}",
        f"train={track_spec.train_profile}",
        f"data={DATA_PRESET}",
        f"experiment.label={track_spec.experiment_label}",
        f"experiment.comparison_methods={_methods_override(METHODS)}",
        f"seed={int(seed)}",
        f"mfm.backend={mfm_backend}",
        "output.save_plots=false",
        f"hydra.run.dir={str(run_dir)}",
    ]
    cmd.extend(_common_overrides(data_path=data_path))
    cmd.extend(_arm_overrides(arm=arm))
    cmd.extend(_candidate_overrides(track=track_spec.track, candidate=candidate))

    print(
        f"[run] track={track_spec.track} arm={arm} candidate={candidate.candidate_id} "
        f"cmd={' '.join(cmd)}"
    )
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not comparison_path.exists():
        raise RuntimeError(f"Missing comparison_mfm.json at {comparison_path}")
    return json.loads(comparison_path.read_text(encoding="utf-8"))


def _extract_stage_a_primary(summary: dict[str, Any]) -> float:
    interpolant = summary.get("interpolant_eval")
    if not isinstance(interpolant, dict):
        raise RuntimeError("Missing interpolant_eval in Stage-A summary.")
    value = interpolant.get("learned_full_ot_w2_avg")
    if value is None:
        raise RuntimeError("Missing interpolant_eval.learned_full_ot_w2_avg in Stage-A summary.")
    return float(value)


def _extract_ab_metrics(summary: dict[str, Any]) -> tuple[float, float]:
    primary = summary.get("intermediate_full_ot_w2_avg")
    endpoint = summary.get("transport_endpoint_full_ot_w2")
    if primary is None:
        raise RuntimeError("Missing intermediate_full_ot_w2_avg in A+B summary.")
    if endpoint is None:
        raise RuntimeError("Missing transport_endpoint_full_ot_w2 in A+B summary.")
    return float(primary), float(endpoint)


def _safe_pct(numer: float, denom: float) -> float | None:
    if abs(float(denom)) <= 1e-12:
        return None
    return float(100.0 * float(numer) / float(denom))


def _compute_decisions(
    *,
    per_config_rows: list[dict[str, Any]],
    default_candidate_by_track: dict[str, str],
    gain_threshold_pct: float,
    endpoint_guardrail_pct: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in per_config_rows:
        key = (str(row["track"]), str(row["arm"]), str(row["method"]))
        grouped.setdefault(key, []).append(row)

    decisions: list[dict[str, Any]] = []
    for (track, arm, method), rows in sorted(grouped.items()):
        default_id = str(default_candidate_by_track[track])
        default_row = next((r for r in rows if str(r["candidate_id"]) == default_id), None)
        if default_row is None:
            raise RuntimeError(
                f"Missing default candidate '{default_id}' for track={track}, arm={arm}, method={method}."
            )
        default_primary = float(default_row["primary_metric"])
        default_endpoint = default_row.get("endpoint_metric")
        default_endpoint_f = None if default_endpoint is None else float(default_endpoint)

        eligible: list[dict[str, Any]] = []
        for row in rows:
            if track == "stage_a":
                row["eligible_endpoint_guardrail"] = True
                row["endpoint_worse_pct_vs_default"] = None
                eligible.append(row)
                continue
            endpoint = row.get("endpoint_metric")
            endpoint_f = None if endpoint is None else float(endpoint)
            if endpoint_f is None or default_endpoint_f is None:
                row["eligible_endpoint_guardrail"] = False
                row["endpoint_worse_pct_vs_default"] = None
                continue
            worse_pct = _safe_pct(endpoint_f - default_endpoint_f, default_endpoint_f)
            row["endpoint_worse_pct_vs_default"] = worse_pct
            ok = worse_pct is not None and float(worse_pct) <= float(endpoint_guardrail_pct)
            row["eligible_endpoint_guardrail"] = bool(ok)
            if ok:
                eligible.append(row)

        best_row = min(eligible, key=lambda r: float(r["primary_metric"])) if eligible else default_row
        best_primary = float(best_row["primary_metric"])
        improvement_pct = _safe_pct(default_primary - best_primary, default_primary)
        if improvement_pct is None:
            improvement_pct = 0.0

        best_is_default = str(best_row["candidate_id"]) == default_id
        adopt = (not best_is_default) and (float(improvement_pct) >= float(gain_threshold_pct))
        recommendation = "adopt larger steps" if adopt else "keep default"

        decisions.append(
            {
                "track": track,
                "arm": arm,
                "method": method,
                "default_candidate_id": default_id,
                "default_primary_metric": default_primary,
                "default_endpoint_metric": default_endpoint_f,
                "best_candidate_id": str(best_row["candidate_id"]),
                "best_primary_metric": best_primary,
                "best_endpoint_metric": best_row.get("endpoint_metric"),
                "primary_improvement_pct_vs_default": float(improvement_pct),
                "endpoint_worse_pct_vs_default_best": best_row.get("endpoint_worse_pct_vs_default"),
                "n_candidates": len(rows),
                "n_eligible": len(eligible),
                "gain_threshold_pct": float(gain_threshold_pct),
                "endpoint_guardrail_worse_pct": (
                    None if track == "stage_a" else float(endpoint_guardrail_pct)
                ),
                "recommendation": recommendation,
            }
        )
    return decisions


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    def _fmt(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.10g}"
        return str(value)

    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(_fmt(row.get(col)) for col in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_md_per_config(path: Path, rows: list[dict[str, Any]]) -> None:
    header = (
        "| Track | Arm | Method | Candidate | Steps (A/B) | Primary | Endpoint | "
        "Delta Primary vs Default (%) | Endpoint Worse vs Default (%) | Endpoint Guardrail |"
    )
    divider = "|---|---|---|---|---:|---:|---:|---:|---:|---|"
    lines = [header, divider]
    for row in rows:
        steps = f"{int(row['stage_a_steps'])}/{int(row['stage_b_steps'])}"
        primary = float(row["primary_metric"])
        endpoint = row.get("endpoint_metric")
        endpoint_s = "" if endpoint is None else f"{float(endpoint):.6f}"
        d_primary = row.get("delta_primary_pct_vs_default")
        d_primary_s = "" if d_primary is None else f"{float(d_primary):.3f}"
        d_endpoint = row.get("endpoint_worse_pct_vs_default")
        d_endpoint_s = "" if d_endpoint is None else f"{float(d_endpoint):.3f}"
        guard = row.get("eligible_endpoint_guardrail")
        guard_s = "" if guard is None else ("yes" if bool(guard) else "no")
        lines.append(
            f"| {row['track']} | {row['arm']} | {row['method']} | {row['candidate_id']} | "
            f"{steps} | {primary:.6f} | {endpoint_s} | {d_primary_s} | {d_endpoint_s} | {guard_s} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_md_decisions(path: Path, rows: list[dict[str, Any]]) -> None:
    header = (
        "| Track | Arm | Method | Default | Best Eligible | Improvement (%) | "
        "Best Endpoint Worse (%) | Recommendation |"
    )
    divider = "|---|---|---|---|---|---:|---:|---|"
    lines = [header, divider]
    for row in rows:
        improvement = float(row["primary_improvement_pct_vs_default"])
        endpoint_worse = row.get("endpoint_worse_pct_vs_default_best")
        endpoint_s = "" if endpoint_worse is None else f"{float(endpoint_worse):.3f}"
        lines.append(
            f"| {row['track']} | {row['arm']} | {row['method']} | "
            f"{row['default_candidate_id']} | {row['best_candidate_id']} | "
            f"{improvement:.3f} | {endpoint_s} | {row['recommendation']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {data_path}")

    run_root = args.run_root.resolve() if args.run_root is not None else _default_run_root()
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[info] benchmark_root={run_root}")

    arm_names = ["overfit", "underfit"]
    run_records: list[dict[str, Any]] = []
    per_config_rows: list[dict[str, Any]] = []

    for track_name in ["stage_a", "ab"]:
        track_spec = TRACKS[track_name]
        for arm in arm_names:
            for candidate in track_spec.candidates:
                run_dir = run_root / track_name / arm / candidate.candidate_id
                comparison = _run_candidate(
                    track_spec=track_spec,
                    arm=arm,
                    candidate=candidate,
                    seed=int(args.seed),
                    data_path=data_path,
                    mfm_backend=str(args.mfm_backend),
                    run_dir=run_dir,
                    continue_if_exists=bool(args.continue_if_exists),
                )

                run_records.append(
                    {
                        "track": track_name,
                        "arm": arm,
                        "candidate_id": candidate.candidate_id,
                        "stage_a_steps": int(candidate.stage_a_steps),
                        "stage_b_steps": int(candidate.stage_b_steps),
                        "run_dir": str(run_dir),
                        "comparison_path": str(run_dir / "comparison_mfm.json"),
                    }
                )

                for method in METHODS:
                    if method not in comparison:
                        raise RuntimeError(
                            f"Method '{method}' missing in comparison payload for {run_dir}"
                        )
                    summary = comparison[method]
                    if track_name == "stage_a":
                        primary = _extract_stage_a_primary(summary=summary)
                        endpoint = None
                    else:
                        primary, endpoint = _extract_ab_metrics(summary=summary)
                    per_config_rows.append(
                        {
                            "track": track_name,
                            "arm": arm,
                            "method": method,
                            "candidate_id": candidate.candidate_id,
                            "stage_a_steps": int(candidate.stage_a_steps),
                            "stage_b_steps": int(candidate.stage_b_steps),
                            "primary_metric": float(primary),
                            "endpoint_metric": endpoint,
                        }
                    )

    default_candidate_by_track = {
        track: spec.default_candidate_id for track, spec in TRACKS.items()
    }
    defaults: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in per_config_rows:
        key = (str(row["track"]), str(row["arm"]), str(row["method"]))
        if str(row["candidate_id"]) == str(default_candidate_by_track[str(row["track"])]):
            defaults[key] = row
    for row in per_config_rows:
        key = (str(row["track"]), str(row["arm"]), str(row["method"]))
        default_row = defaults.get(key)
        if default_row is None:
            raise RuntimeError(f"Missing default row for key={key}")
        d_primary = float(default_row["primary_metric"])
        row["delta_primary_abs_vs_default"] = float(row["primary_metric"]) - d_primary
        row["delta_primary_pct_vs_default"] = _safe_pct(
            float(row["primary_metric"]) - d_primary,
            d_primary,
        )

    decisions = _compute_decisions(
        per_config_rows=per_config_rows,
        default_candidate_by_track=default_candidate_by_track,
        gain_threshold_pct=PRIMARY_GAIN_THRESHOLD_PCT,
        endpoint_guardrail_pct=ENDPOINT_GUARDRAIL_WORSE_PCT,
    )

    artifacts_dir = run_root / "summary"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "benchmark_label": "single_cell_schiebinger_step_sufficiency_quickcheck",
            "benchmark_root": str(run_root),
            "data_path": str(data_path.resolve()),
            "data_preset": DATA_PRESET,
            "seed": int(args.seed),
            "mfm_backend": str(args.mfm_backend),
            "tracks": list(TRACKS.keys()),
            "methods": METHODS,
            "arms": arm_names,
            "stage_a_candidates": [c.__dict__ for c in TRACKS["stage_a"].candidates],
            "ab_candidates": [c.__dict__ for c in TRACKS["ab"].candidates],
            "default_candidate_by_track": default_candidate_by_track,
            "primary_gain_threshold_pct": PRIMARY_GAIN_THRESHOLD_PCT,
            "endpoint_guardrail_worse_pct": ENDPOINT_GUARDRAIL_WORSE_PCT,
        },
        "runs": run_records,
        "per_config_rows": per_config_rows,
        "decisions": decisions,
    }
    summary_json = artifacts_dir / "step_sufficiency_summary.json"
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    per_config_columns = [
        "track",
        "arm",
        "method",
        "candidate_id",
        "stage_a_steps",
        "stage_b_steps",
        "primary_metric",
        "endpoint_metric",
        "delta_primary_abs_vs_default",
        "delta_primary_pct_vs_default",
        "endpoint_worse_pct_vs_default",
        "eligible_endpoint_guardrail",
    ]
    _write_csv(artifacts_dir / "per_config_metrics.csv", per_config_rows, per_config_columns)
    _write_md_per_config(artifacts_dir / "per_config_metrics.md", per_config_rows)

    decision_columns = [
        "track",
        "arm",
        "method",
        "default_candidate_id",
        "best_candidate_id",
        "default_primary_metric",
        "best_primary_metric",
        "primary_improvement_pct_vs_default",
        "default_endpoint_metric",
        "best_endpoint_metric",
        "endpoint_worse_pct_vs_default_best",
        "recommendation",
    ]
    _write_csv(artifacts_dir / "recommendations.csv", decisions, decision_columns)
    _write_md_decisions(artifacts_dir / "recommendations.md", decisions)

    print(f"[done] wrote summary json: {summary_json}")
    print(f"[done] wrote per-config table: {artifacts_dir / 'per_config_metrics.csv'}")
    print(f"[done] wrote recommendations: {artifacts_dir / 'recommendations.csv'}")
    for row in decisions:
        print(
            "[decision] "
            f"track={row['track']} arm={row['arm']} method={row['method']} "
            f"default={row['default_candidate_id']} best={row['best_candidate_id']} "
            f"improve_pct={float(row['primary_improvement_pct_vs_default']):.3f} "
            f"recommendation={row['recommendation']}"
        )


if __name__ == "__main__":
    main()
