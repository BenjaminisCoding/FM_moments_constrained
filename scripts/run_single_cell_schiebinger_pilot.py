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
            "Run Schiebinger pilot ablations: no-constraints, moments-only, and "
            "moments+supervised-classifier constraints."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=ROOT / "datasets" / "schiebinger_serum_d10_d10p5_d11_hvg_pca50.h5ad",
        help="Local Schiebinger pilot .h5ad path.",
    )
    parser.add_argument(
        "--train-profile",
        type=str,
        default="single_cell_ab_only",
        help="Train config profile (for example: single_cell_ab_only, single_cell_stage_a_only).",
    )
    parser.add_argument(
        "--pseudo-eta",
        type=float,
        default=1.0,
        help="Pseudo-label constraint weight for moments+classifier group.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for all groups.",
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
            "Output root. Default: "
            "outputs/<date>/single_cell_schiebinger_pilot/<time>"
        ),
    )
    return parser.parse_args()


def _default_run_root() -> Path:
    day = datetime.now().strftime("%Y-%m-%d")
    ts = datetime.now().strftime("%H-%M-%S")
    return (ROOT / "outputs" / day / "single_cell_schiebinger_pilot" / ts).resolve()


def _run_group(
    *,
    run_dir: Path,
    data_path: Path,
    train_profile: str,
    methods: list[str],
    pseudo_enabled: bool,
    pseudo_eta: float,
    seed: int,
    mfm_backend: str,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    methods_override = "[" + ",".join(methods) + "]"
    cmd = [
        sys.executable,
        str(RUN_EXPERIMENT),
        "experiment=comparison_mfm_single_cell_schiebinger",
        f"train={train_profile}",
        "data=single_cell_schiebinger_pilot_20d",
        f"experiment.comparison_methods={methods_override}",
        f"data.single_cell.path={str(data_path.resolve())}",
        f"data.single_cell.pseudo_labels.enabled={'true' if pseudo_enabled else 'false'}",
        f"train.pseudo_eta={float(pseudo_eta)}",
        f"seed={int(seed)}",
        f"mfm.backend={mfm_backend}",
        "output.save_plots=true",
        f"hydra.run.dir={str(run_dir)}",
    ]
    print(f"[run] group={run_dir.name} cmd={' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)
    summary_path = run_dir / "comparison_mfm.json"
    if not summary_path.exists():
        raise RuntimeError(f"Missing comparison_mfm.json in {run_dir}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "run_dir": str(run_dir),
        "comparison_path": str(summary_path),
        "methods": methods,
        "pseudo_enabled": bool(pseudo_enabled),
        "pseudo_eta": float(pseudo_eta),
        "comparison": payload,
    }


def main() -> None:
    args = _parse_args()
    if not args.data_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {args.data_path}")
    if float(args.pseudo_eta) < 0.0:
        raise ValueError(f"--pseudo-eta must be non-negative, got {args.pseudo_eta}")

    run_root = args.run_root.resolve() if args.run_root is not None else _default_run_root()
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[info] benchmark_root={run_root}")

    groups = {
        "no_constraints": {
            "methods": ["baseline", "metric", "metric_alpha0"],
            "pseudo_enabled": False,
            "pseudo_eta": 0.0,
        },
        "moments_only": {
            "methods": ["constrained", "metric_constrained_al", "metric_constrained_soft"],
            "pseudo_enabled": False,
            "pseudo_eta": 0.0,
        },
        "moments_plus_classifier": {
            "methods": ["constrained", "metric_constrained_al", "metric_constrained_soft"],
            "pseudo_enabled": True,
            "pseudo_eta": float(args.pseudo_eta),
        },
    }

    results: dict[str, Any] = {}
    for group_name, spec in groups.items():
        group_dir = run_root / group_name
        results[group_name] = _run_group(
            run_dir=group_dir,
            data_path=args.data_path,
            train_profile=str(args.train_profile),
            methods=list(spec["methods"]),
            pseudo_enabled=bool(spec["pseudo_enabled"]),
            pseudo_eta=float(spec["pseudo_eta"]),
            seed=int(args.seed),
            mfm_backend=str(args.mfm_backend),
        )

    summary = {
        "meta": {
            "benchmark_label": "single_cell_schiebinger_pilot",
            "dataset_path": str(args.data_path.resolve()),
            "seed": int(args.seed),
            "mfm_backend": str(args.mfm_backend),
            "train_profile": str(args.train_profile),
            "pseudo_eta_classifier_group": float(args.pseudo_eta),
            "benchmark_root": str(run_root),
        },
        "groups": {
            name: {
                "run_dir": payload["run_dir"],
                "comparison_path": payload["comparison_path"],
                "methods": payload["methods"],
                "pseudo_enabled": payload["pseudo_enabled"],
                "pseudo_eta": payload["pseudo_eta"],
            }
            for name, payload in results.items()
        },
    }
    summary_path = run_root / "pilot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[done] wrote pilot summary: {summary_path}")


if __name__ == "__main__":
    main()
