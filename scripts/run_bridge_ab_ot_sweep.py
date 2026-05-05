#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import itertools
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_EXPERIMENT = ROOT / "scripts" / "run_experiment.py"

PHASE1_GRID = {
    "rho": [0.5, 1.0],
    "alpha": [1.0, 2.0],
    "lr_g": [3e-4, 1e-3],
    "lr_v": [5e-4, 1e-3],
}
PHASE2_SEEDS = [3, 7, 11]


@dataclass(frozen=True)
class SweepConfig:
    rho: float
    alpha: float
    lr_g: float
    lr_v: float

    @property
    def name(self) -> str:
        def _fmt(value: float) -> str:
            text = f"{value:g}"
            return text.replace(".", "p").replace("-", "m")

        return (
            f"rho{_fmt(self.rho)}"
            f"_alpha{_fmt(self.alpha)}"
            f"_lrg{_fmt(self.lr_g)}"
            f"_lrv{_fmt(self.lr_v)}"
        )

    @property
    def overrides(self) -> list[str]:
        return [
            f"train.rho={self.rho}",
            f"train.alpha={self.alpha}",
            f"train.lr_g={self.lr_g}",
            f"train.lr_v={self.lr_v}",
        ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bridge OT A+B hyperparameter sweep.")
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=None,
        help=(
            "Root folder for sweep outputs. Default: "
            "outputs/bridge_ab_ot_hparam_sweep/<timestamp>"
        ),
    )
    parser.add_argument(
        "--continue-if-exists",
        action="store_true",
        help="Reuse existing run folders if comparison.json exists.",
    )
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _build_sweep_root(sweep_root: Path | None) -> Path:
    if sweep_root is not None:
        out = sweep_root
    else:
        out = ROOT / "outputs" / "bridge_ab_ot_hparam_sweep" / _timestamp()
    out.mkdir(parents=True, exist_ok=True)
    return out.resolve()


def _common_overrides() -> list[str]:
    return [
        "experiment=comparison",
        "train=ab_only",
        "data=bridge_ot",
        "experiment.label=bridge_ab_ot_sweep",
        "train.stage_a_steps=300",
        "train.stage_b_steps=300",
        "train.stage_c_steps=0",
        "train.batch_size=512",
        "train.beta=0.05",
        "train.eval_intermediate_ot_samples=1024",
        "train.eval_transport_samples=4000",
        "output.save_plots=false",
    ]


def _all_phase1_configs() -> list[SweepConfig]:
    configs: list[SweepConfig] = []
    for rho, alpha, lr_g, lr_v in itertools.product(
        PHASE1_GRID["rho"],
        PHASE1_GRID["alpha"],
        PHASE1_GRID["lr_g"],
        PHASE1_GRID["lr_v"],
    ):
        configs.append(SweepConfig(rho=float(rho), alpha=float(alpha), lr_g=float(lr_g), lr_v=float(lr_v)))
    return configs


def _run_one(
    run_dir: Path,
    extra_overrides: list[str],
    continue_if_exists: bool,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    comparison_path = run_dir / "comparison.json"
    if continue_if_exists and comparison_path.exists():
        return json.loads(comparison_path.read_text(encoding="utf-8"))

    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(RUN_EXPERIMENT),
        *_common_overrides(),
        *extra_overrides,
        f"hydra.run.dir={run_dir}",
    ]
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)
    if not comparison_path.exists():
        raise RuntimeError(f"Missing comparison.json for run: {run_dir}")
    return json.loads(comparison_path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float:
    if value is None:
        raise ValueError("Expected numeric value, got None.")
    out = float(value)
    if out != out:
        raise ValueError("Encountered NaN metric value.")
    return out


def _row_from_comparison(
    comparison: dict[str, Any],
    run_dir: Path,
    name: str,
    seed: int,
) -> dict[str, Any]:
    baseline = comparison["baseline"]
    constrained = comparison["constrained"]
    baseline_intermediate = _safe_float(baseline["intermediate_empirical_w2_avg"])
    constrained_intermediate = _safe_float(constrained["intermediate_empirical_w2_avg"])
    baseline_endpoint = _safe_float(baseline["transport_endpoint_empirical_w2"])
    constrained_endpoint = _safe_float(constrained["transport_endpoint_empirical_w2"])
    baseline_constraint = _safe_float(baseline["constraint_residual_avg"])
    constrained_constraint = _safe_float(constrained["constraint_residual_avg"])
    delta_intermediate = constrained_intermediate - baseline_intermediate
    delta_endpoint = constrained_endpoint - baseline_endpoint
    return {
        "name": name,
        "seed": seed,
        "run_dir": str(run_dir),
        "baseline_constraint_residual_avg": baseline_constraint,
        "constrained_constraint_residual_avg": constrained_constraint,
        "baseline_intermediate_empirical_w2_avg": baseline_intermediate,
        "constrained_intermediate_empirical_w2_avg": constrained_intermediate,
        "delta_intermediate": delta_intermediate,
        "baseline_transport_endpoint_empirical_w2": baseline_endpoint,
        "constrained_transport_endpoint_empirical_w2": constrained_endpoint,
        "delta_endpoint": delta_endpoint,
        "gate_pass": bool((delta_intermediate < 0.0) and (delta_endpoint <= 0.03)),
    }


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _rank_phase1(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gate_pass = [row for row in rows if bool(row["gate_pass"])]
    gate_pass_sorted = sorted(
        gate_pass,
        key=lambda r: (float(r["delta_intermediate"]), float(r["delta_endpoint"])),
    )
    if len(gate_pass_sorted) >= 4:
        return gate_pass_sorted[:4]

    selected = list(gate_pass_sorted)
    selected_names = {str(row["name"]) for row in selected}
    remaining = [row for row in rows if str(row["name"]) not in selected_names]
    remaining_sorted = sorted(
        remaining,
        key=lambda r: (float(r["delta_endpoint"]), float(r["delta_intermediate"])),
    )
    needed = 4 - len(selected)
    selected.extend(remaining_sorted[:needed])
    return selected


def _aggregate_phase2(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["name"]), []).append(row)

    out: list[dict[str, Any]] = []
    for name, group in grouped.items():
        n = len(group)
        delta_intermediate_mean = sum(float(r["delta_intermediate"]) for r in group) / n
        delta_endpoint_mean = sum(float(r["delta_endpoint"]) for r in group) / n
        constrained_intermediate_mean = (
            sum(float(r["constrained_intermediate_empirical_w2_avg"]) for r in group) / n
        )
        baseline_intermediate_mean = (
            sum(float(r["baseline_intermediate_empirical_w2_avg"]) for r in group) / n
        )
        constrained_endpoint_mean = (
            sum(float(r["constrained_transport_endpoint_empirical_w2"]) for r in group) / n
        )
        baseline_endpoint_mean = (
            sum(float(r["baseline_transport_endpoint_empirical_w2"]) for r in group) / n
        )
        gate_pass_count = sum(1 for r in group if bool(r["gate_pass"]))
        gate_pass_avg = bool((delta_intermediate_mean < 0.0) and (delta_endpoint_mean <= 0.03))
        out.append(
            {
                "name": name,
                "num_seeds": n,
                "seeds": ",".join(str(int(r["seed"])) for r in sorted(group, key=lambda x: int(x["seed"]))),
                "baseline_intermediate_empirical_w2_avg_mean": baseline_intermediate_mean,
                "constrained_intermediate_empirical_w2_avg_mean": constrained_intermediate_mean,
                "delta_intermediate_mean": delta_intermediate_mean,
                "baseline_transport_endpoint_empirical_w2_mean": baseline_endpoint_mean,
                "constrained_transport_endpoint_empirical_w2_mean": constrained_endpoint_mean,
                "delta_endpoint_mean": delta_endpoint_mean,
                "gate_pass_count": gate_pass_count,
                "gate_pass_avg": gate_pass_avg,
            }
        )
    return sorted(
        out,
        key=lambda r: (
            -int(bool(r["gate_pass_avg"])),
            float(r["delta_intermediate_mean"]),
            float(r["delta_endpoint_mean"]),
        ),
    )


def _extract_config_from_name(name: str) -> SweepConfig:
    # Expected format: rho{r}_alpha{a}_lrg{g}_lrv{v}
    parts = name.split("_")
    raw: dict[str, str] = {}
    for part in parts:
        if part.startswith("rho"):
            raw["rho"] = part[len("rho") :]
        elif part.startswith("alpha"):
            raw["alpha"] = part[len("alpha") :]
        elif part.startswith("lrg"):
            raw["lr_g"] = part[len("lrg") :]
        elif part.startswith("lrv"):
            raw["lr_v"] = part[len("lrv") :]
    if set(raw.keys()) != {"rho", "alpha", "lr_g", "lr_v"}:
        raise ValueError(f"Could not parse config name: {name}")

    def _parse(value: str) -> float:
        return float(value.replace("p", ".").replace("m", "-"))

    return SweepConfig(
        rho=_parse(raw["rho"]),
        alpha=_parse(raw["alpha"]),
        lr_g=_parse(raw["lr_g"]),
        lr_v=_parse(raw["lr_v"]),
    )


def main() -> None:
    args = _parse_args()
    sweep_root = _build_sweep_root(args.sweep_root)
    phase1_root = sweep_root / "phase1"
    phase2_root = sweep_root / "phase2"
    phase1_root.mkdir(parents=True, exist_ok=True)
    phase2_root.mkdir(parents=True, exist_ok=True)

    print(f"[info] sweep_root={sweep_root}")
    all_configs = _all_phase1_configs()
    print(f"[info] phase1_config_count={len(all_configs)}")

    # Sweep smoke test: run first config and validate comparison.json parsing.
    smoke_cfg = all_configs[0]
    smoke_dir = phase1_root / smoke_cfg.name
    smoke_cmp = _run_one(
        run_dir=smoke_dir,
        extra_overrides=[*smoke_cfg.overrides, "seed=7"],
        continue_if_exists=args.continue_if_exists,
    )
    _ = _row_from_comparison(smoke_cmp, run_dir=smoke_dir, name=smoke_cfg.name, seed=7)
    print(f"[smoke] success run={smoke_cfg.name}")

    # Phase 1 full grid.
    phase1_rows: list[dict[str, Any]] = []
    for cfg in all_configs:
        run_dir = phase1_root / cfg.name
        cmp_payload = _run_one(
            run_dir=run_dir,
            extra_overrides=[*cfg.overrides, "seed=7"],
            continue_if_exists=args.continue_if_exists,
        )
        phase1_rows.append(_row_from_comparison(cmp_payload, run_dir=run_dir, name=cfg.name, seed=7))
    phase1_summary_path = sweep_root / "phase1_summary.tsv"
    _write_tsv(phase1_summary_path, phase1_rows)
    print(f"[phase1] summary={phase1_summary_path}")

    top4 = _rank_phase1(phase1_rows)
    selected_names = [str(row["name"]) for row in top4]
    selected_payload = {
        "selection_rule": {
            "gate": "delta_intermediate < 0 and delta_endpoint <= 0.03",
            "rank_gate_pass": "delta_intermediate asc, delta_endpoint asc",
            "rank_fallback": "delta_endpoint asc, delta_intermediate asc",
        },
        "selected_top4": selected_names,
    }
    selected_path = sweep_root / "phase2_selected.json"
    selected_path.write_text(json.dumps(selected_payload, indent=2), encoding="utf-8")
    print(f"[phase2] selected={selected_names}")

    # Phase 2: top 4 x 3 seeds.
    phase2_rows: list[dict[str, Any]] = []
    for name in selected_names:
        cfg = _extract_config_from_name(name)
        for seed in PHASE2_SEEDS:
            run_dir = phase2_root / name / f"seed_{seed}"
            cmp_payload = _run_one(
                run_dir=run_dir,
                extra_overrides=[*cfg.overrides, f"seed={seed}"],
                continue_if_exists=args.continue_if_exists,
            )
            phase2_rows.append(_row_from_comparison(cmp_payload, run_dir=run_dir, name=name, seed=seed))

    phase2_runs_path = sweep_root / "phase2_runs.tsv"
    _write_tsv(phase2_runs_path, phase2_rows)
    phase2_summary = _aggregate_phase2(phase2_rows)
    phase2_summary_path = sweep_root / "phase2_summary.tsv"
    _write_tsv(phase2_summary_path, phase2_summary)

    success_primary = any(bool(row["gate_pass_avg"]) for row in phase2_summary)
    success_stretch = any(
        (float(row["delta_intermediate_mean"]) < 0.0) and (float(row["delta_endpoint_mean"]) <= 0.0)
        for row in phase2_summary
    )
    final_report = {
        "sweep_root": str(sweep_root),
        "phase1_summary": str(phase1_summary_path),
        "phase2_runs": str(phase2_runs_path),
        "phase2_summary": str(phase2_summary_path),
        "selected_top4": selected_names,
        "success_primary": success_primary,
        "success_stretch": success_stretch,
    }
    report_path = sweep_root / "final_report.json"
    report_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    print(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    main()
