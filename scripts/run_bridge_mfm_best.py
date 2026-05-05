#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUN_EXPERIMENT = ROOT / "scripts" / "run_experiment.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bridge OT best-preset comparison with MFM baselines."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Explicit output run folder. Default: "
            "outputs/<date>/bridge_mfm_best_from_sweep_ab_only/<time>"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3,
        help="Random seed. Default: 3 (best seed-level run from prior sweep tables).",
    )
    parser.add_argument(
        "--mfm-backend",
        type=str,
        default="auto",
        choices=["auto", "native", "torchcfm"],
        help="MFM backend selection policy.",
    )
    return parser.parse_args()


def _default_run_dir() -> Path:
    day = datetime.now().strftime("%Y-%m-%d")
    ts = datetime.now().strftime("%H-%M-%S")
    return (ROOT / "outputs" / day / "bridge_mfm_best_from_sweep_ab_only" / ts).resolve()


def _safe_metric(summary: dict, key: str) -> float:
    value = summary.get(key)
    if value is None:
        raise ValueError(f"Missing metric '{key}' in summary.")
    return float(value)


def _print_report(payload: dict) -> None:
    baseline = payload["baseline"]
    baseline_intermediate = _safe_metric(baseline, "intermediate_empirical_w2_avg")
    baseline_endpoint = _safe_metric(baseline, "transport_endpoint_empirical_w2")

    print("[summary] baseline")
    print(
        "  intermediate_empirical_w2_avg="
        f"{baseline_intermediate:.6f} transport_endpoint_empirical_w2={baseline_endpoint:.6f}"
    )

    for mode in [
        "constrained",
        "metric",
        "metric_alpha0",
        "metric_constrained_al",
        "metric_constrained_soft",
    ]:
        summary = payload[mode]
        mode_intermediate = _safe_metric(summary, "intermediate_empirical_w2_avg")
        mode_endpoint = _safe_metric(summary, "transport_endpoint_empirical_w2")
        delta_intermediate = mode_intermediate - baseline_intermediate
        delta_endpoint = mode_endpoint - baseline_endpoint
        line = (
            f"[summary] {mode} "
            f"intermediate={mode_intermediate:.6f} (delta={delta_intermediate:+.6f}) "
            f"endpoint={mode_endpoint:.6f} (delta={delta_endpoint:+.6f})"
        )
        if mode in {"metric", "metric_alpha0", "metric_constrained_al", "metric_constrained_soft"}:
            backend = summary.get("mfm_backend")
            impl = summary.get("mfm_backend_impl")
            moment_style = summary.get("mfm_moment_style")
            line += (
                f" mfm_backend={backend} mfm_backend_impl={impl}"
                f" mfm_moment_style={moment_style}"
            )
        print(line)


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.resolve() if args.run_dir is not None else _default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(RUN_EXPERIMENT),
        "experiment=comparison_mfm",
        "train=ab_only",
        "data=bridge_ot",
        "experiment.label=bridge_mfm_best_from_sweep",
        "train.stage_a_steps=300",
        "train.stage_b_steps=300",
        "train.stage_c_steps=0",
        "train.batch_size=512",
        "train.beta=0.05",
        "train.eval_intermediate_ot_samples=1024",
        "train.eval_transport_samples=4000",
        "train.rho=0.5",
        "train.alpha=1.0",
        "train.lr_g=0.001",
        "train.lr_v=0.001",
        f"seed={int(args.seed)}",
        f"mfm.backend={args.mfm_backend}",
        "mfm.alpha=1.0",
        "mfm.sigma=0.1",
        "mfm.land_gamma=0.125",
        "mfm.land_rho=0.001",
        "mfm.land_metric_samples=512",
        "mfm.reference_pool_policy=endpoints_only",
        "mfm.moment_eta=1.0",
        "output.save_plots=false",
        f"hydra.run.dir={run_dir}",
    ]
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)

    comparison_path = run_dir / "comparison_mfm.json"
    if not comparison_path.exists():
        raise RuntimeError(f"Missing comparison_mfm.json at {comparison_path}")
    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    print(f"[done] run_dir={run_dir}")
    _print_report(payload)


if __name__ == "__main__":
    main()
