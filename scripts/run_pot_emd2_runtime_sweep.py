#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import ot  # type: ignore
import torch

from cfm_project.single_cell_data import prepare_single_cell_problem_and_targets


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class SweepRecord:
    phase: str
    num_itermax: int
    runtime_sec: float
    w2: float | None
    emd2_value: float | None
    warning: str | None
    result_code: int | None
    iteration_limit_reached: bool
    error: str | None


def _nearest_time_key(available: list[float], target: float, tol: float = 1e-8) -> float:
    key = min(available, key=lambda value: abs(float(value) - float(target)))
    if abs(float(key) - float(target)) > float(tol):
        raise ValueError(
            f"Requested time {target:.6f} is not available. "
            f"Nearest={float(key):.6f}, available={available}"
        )
    return float(key)


def _pairwise_sqeuclidean(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True).T
    cost = x_norm + y_norm - 2.0 * (x @ y.T)
    np.maximum(cost, 0.0, out=cost)
    return cost


def _parse_scales(raw: str) -> list[float]:
    out: list[float] = []
    for token in raw.split(","):
        value = float(token.strip())
        if value <= 0.0:
            raise ValueError(f"Sweep scales must be positive. Got '{value}' from '{raw}'.")
        out.append(value)
    return out


def _run_emd2_once(
    *,
    a: np.ndarray,
    b: np.ndarray,
    cost_matrix: np.ndarray,
    num_itermax: int,
) -> SweepRecord:
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            raw_result = ot.emd2(a, b, cost_matrix, numItermax=int(num_itermax), log=True)
            runtime_sec = float(time.perf_counter() - t0)
            if not isinstance(raw_result, (list, tuple)) or len(raw_result) != 2:
                return SweepRecord(
                    phase="",
                    num_itermax=int(num_itermax),
                    runtime_sec=runtime_sec,
                    w2=None,
                    emd2_value=None,
                    warning="Unexpected return type from ot.emd2(log=True).",
                    result_code=None,
                    iteration_limit_reached=False,
                    error=f"Unexpected return type: {type(raw_result)}",
                )
            emd2_value = float(raw_result[0])
            log = raw_result[1]
            warning_msg: str | None = None
            result_code: int | None = None
            if isinstance(log, dict):
                warning_obj = log.get("warning", None)
                if warning_obj is not None:
                    warning_msg = str(warning_obj)
                rc = log.get("result_code", None)
                result_code = None if rc is None else int(rc)
            warning_messages = [str(entry.message) for entry in caught]
            if warning_msg is None and warning_messages:
                warning_msg = " | ".join(warning_messages)
            limit_reached = False
            if warning_msg is not None and "numItermax reached before optimality" in warning_msg:
                limit_reached = True
            if any("numItermax reached before optimality" in msg for msg in warning_messages):
                limit_reached = True
            return SweepRecord(
                phase="",
                num_itermax=int(num_itermax),
                runtime_sec=runtime_sec,
                w2=float(math.sqrt(max(emd2_value, 0.0))),
                emd2_value=emd2_value,
                warning=warning_msg,
                result_code=result_code,
                iteration_limit_reached=bool(limit_reached),
                error=None,
            )
        except Exception as exc:  # pragma: no cover - defensive
            runtime_sec = float(time.perf_counter() - t0)
            return SweepRecord(
                phase="",
                num_itermax=int(num_itermax),
                runtime_sec=runtime_sec,
                w2=None,
                emd2_value=None,
                warning=None,
                result_code=None,
                iteration_limit_reached=False,
                error=f"{type(exc).__name__}: {exc}",
            )


def _recommend_num_itermax(
    records: list[SweepRecord],
    stability_tol: float,
) -> dict[str, Any]:
    valid = [record for record in records if record.error is None and record.w2 is not None]
    valid = sorted(valid, key=lambda record: int(record.num_itermax))
    if not valid:
        return {
            "recommended_num_itermax": None,
            "selection_reason": "No valid sweep records available.",
            "stability_tol": float(stability_tol),
        }

    for idx in range(len(valid) - 1):
        current = valid[idx]
        nxt = valid[idx + 1]
        w2_delta = abs(float(current.w2) - float(nxt.w2))
        if (
            not current.iteration_limit_reached
            and not nxt.iteration_limit_reached
            and w2_delta < float(stability_tol)
        ):
            return {
                "recommended_num_itermax": int(current.num_itermax),
                "selection_reason": (
                    f"First stable pair without iteration-limit warnings: "
                    f"{current.num_itermax} vs {nxt.num_itermax} (|delta|={w2_delta:.6e})."
                ),
                "stability_tol": float(stability_tol),
            }

    no_warning = [record for record in valid if not record.iteration_limit_reached]
    if no_warning:
        best = no_warning[-1]
        return {
            "recommended_num_itermax": int(best.num_itermax),
            "selection_reason": (
                "No stable adjacent pair met tolerance; selected largest numItermax "
                "without iteration-limit warning."
            ),
            "stability_tol": float(stability_tol),
        }

    best = valid[-1]
    return {
        "recommended_num_itermax": int(best.num_itermax),
        "selection_reason": (
            "All runs reported iteration-limit warnings; selected largest numItermax tested."
        ),
        "stability_tol": float(stability_tol),
    }


def _default_output_root() -> Path:
    day = datetime.now().strftime("%Y-%m-%d")
    ts = datetime.now().strftime("%H-%M-%S")
    return (ROOT / "outputs" / day / "pot_emd2_runtime_sweep" / ts).resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark POT ot.emd2 runtime vs numItermax for Stage-A-style single-cell "
            "plan-conditioned W2 at holdout time."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=ROOT / "TrajectoryNet" / "data" / "eb_velocity_v5.npz",
        help="Path to single-cell EB npz file.",
    )
    parser.add_argument(
        "--holdout-index",
        type=int,
        default=2,
        help="Strict leaveout index. Default 2 corresponds to normalized t=0.5 for 5 timepoints.",
    )
    parser.add_argument(
        "--interp-time",
        type=float,
        default=None,
        help="Interpolant time for metric evaluation. Defaults to resolved holdout time.",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=5,
        help="Embedding dimensionality for EB benchmark setup.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=".cache/ot_plans",
        help="Cache dir for global OT support.",
    )
    parser.add_argument(
        "--initial-num-itermax",
        type=int,
        default=100000,
        help="Initial numItermax for calibration loop.",
    )
    parser.add_argument(
        "--target-min-seconds",
        type=float,
        default=45.0,
        help="Calibration lower wall-clock target (seconds).",
    )
    parser.add_argument(
        "--target-max-seconds",
        type=float,
        default=75.0,
        help="Calibration upper wall-clock target (seconds).",
    )
    parser.add_argument(
        "--max-calibration-rounds",
        type=int,
        default=8,
        help="Max rounds for runtime calibration.",
    )
    parser.add_argument(
        "--sweep-scales",
        type=str,
        default="0.25,0.5,1.0,2.0",
        help="Comma-separated factors around calibrated numItermax.",
    )
    parser.add_argument(
        "--stability-tol",
        type=float,
        default=1.0e-4,
        help="Absolute W2 delta tolerance for recommendation rule.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output folder. Defaults to outputs/<date>/pot_emd2_runtime_sweep/<time>.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    output_root = _default_output_root() if args.output_root is None else args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    # Use float32 for single-cell prep so we can reuse the existing ot_global cache
    # signature used by the benchmark pipeline. We still cast to float64 for POT.
    dtype = torch.float32
    data_cfg = {
        "label": "single_cell_eb_5d",
        "family": "single_cell",
        "dim": int(args.max_dim),
        "coupling": "ot_global",
        "single_cell": {
            "path": str(args.data_path.resolve()),
            "embed_key_npz": "pcs",
            "label_key_npz": "sample_labels",
            "max_dim": int(args.max_dim),
            "whiten": True,
            "global_ot_cache_enabled": True,
            "global_ot_cache_dir": str(args.cache_dir),
            "global_ot_force_recompute": False,
            "global_ot_support_tol": 1.0e-12,
            "global_ot_max_variables": None,
        },
    }
    experiment_cfg = {
        "protocol": "strict_leaveout",
        "holdout_index": int(args.holdout_index),
    }

    prepared = prepare_single_cell_problem_and_targets(
        data_cfg=data_cfg,
        experiment_cfg=experiment_cfg,
        device=device,
        dtype=dtype,
    )
    problem = prepared.problem
    if (
        problem.global_ot_src_idx is None
        or problem.global_ot_tgt_idx is None
        or problem.global_ot_mass is None
    ):
        raise RuntimeError("Global OT support is missing; expected coupling='ot_global'.")

    print(
        "[info] global OT cache "
        f"hit={prepared.global_ot_cache_hit} "
        f"path={prepared.global_ot_cache_path} "
        f"support_size={prepared.global_ot_support_size}"
    )

    holdout_time = (
        float(args.interp_time)
        if args.interp_time is not None
        else (None if prepared.holdout_time is None else float(prepared.holdout_time))
    )
    if holdout_time is None:
        raise ValueError(
            "No holdout time resolved. Provide --interp-time explicitly or use strict leaveout."
        )

    available_times = sorted(float(t) for t in prepared.target_samples_by_time.keys())
    holdout_key = _nearest_time_key(available_times, holdout_time)

    src_idx = problem.global_ot_src_idx.to(device=device, dtype=torch.long)
    tgt_idx = problem.global_ot_tgt_idx.to(device=device, dtype=torch.long)
    mass = problem.global_ot_mass.to(device=device, dtype=dtype)
    mass = mass / torch.clamp(mass.sum(), min=torch.finfo(mass.dtype).eps)

    x0_support = problem.x0_pool[src_idx]
    x1_support = problem.x1_pool[tgt_idx]
    xt = (1.0 - holdout_key) * x0_support + holdout_key * x1_support
    target = prepared.target_samples_by_time[holdout_key]

    xt_np = np.asarray(xt.detach().cpu(), dtype=np.float64)
    target_np = np.asarray(target.detach().cpu(), dtype=np.float64)
    a = np.asarray(mass.detach().cpu(), dtype=np.float64)
    a = a / a.sum()
    b = np.ones(target_np.shape[0], dtype=np.float64) / float(target_np.shape[0])

    cost_t0 = time.perf_counter()
    cost_matrix = _pairwise_sqeuclidean(xt_np, target_np)
    cost_build_sec = float(time.perf_counter() - cost_t0)
    print(
        "[info] prepared benchmark pair "
        f"xt_n={xt_np.shape[0]} target_n={target_np.shape[0]} "
        f"cost_shape={cost_matrix.shape} cost_build_sec={cost_build_sec:.3f}"
    )

    sweep_records: list[SweepRecord] = []
    calibration_records: list[SweepRecord] = []
    candidate = int(args.initial_num_itermax)
    seen_candidates: set[int] = set()
    calibration_success = False
    center_runtime = 0.5 * (float(args.target_min_seconds) + float(args.target_max_seconds))
    for _ in range(int(args.max_calibration_rounds)):
        candidate = max(1, int(candidate))
        if candidate in seen_candidates:
            break
        seen_candidates.add(candidate)
        record = _run_emd2_once(
            a=a,
            b=b,
            cost_matrix=cost_matrix,
            num_itermax=candidate,
        )
        record.phase = "calibration"
        calibration_records.append(record)
        sweep_records.append(record)
        print(
            "[calibration] "
            f"numItermax={record.num_itermax} "
            f"runtime_sec={record.runtime_sec:.3f} "
            f"w2={'NA' if record.w2 is None else f'{record.w2:.8f}'} "
            f"iter_limit={int(record.iteration_limit_reached)} "
            f"error={'none' if record.error is None else record.error}"
        )

        if record.error is not None:
            break
        if float(args.target_min_seconds) <= record.runtime_sec <= float(args.target_max_seconds):
            calibration_success = True
            break
        if record.runtime_sec < float(args.target_min_seconds):
            candidate = candidate * 2
        else:
            candidate = max(1, candidate // 2)

    if calibration_records:
        calibrated_record = min(
            calibration_records,
            key=lambda rec: abs(float(rec.runtime_sec) - center_runtime),
        )
        calibrated_num_itermax = int(calibrated_record.num_itermax)
    else:
        calibrated_num_itermax = int(args.initial_num_itermax)

    scales = _parse_scales(args.sweep_scales)
    final_iter_values = sorted(
        {
            max(1, int(round(float(calibrated_num_itermax) * float(scale))))
            for scale in scales
        }
    )
    final_records: list[SweepRecord] = []
    for value in final_iter_values:
        record = _run_emd2_once(
            a=a,
            b=b,
            cost_matrix=cost_matrix,
            num_itermax=int(value),
        )
        record.phase = "sweep"
        final_records.append(record)
        sweep_records.append(record)
        print(
            "[sweep] "
            f"numItermax={record.num_itermax} "
            f"runtime_sec={record.runtime_sec:.3f} "
            f"w2={'NA' if record.w2 is None else f'{record.w2:.8f}'} "
            f"iter_limit={int(record.iteration_limit_reached)} "
            f"error={'none' if record.error is None else record.error}"
        )

    recommendation = _recommend_num_itermax(final_records, stability_tol=float(args.stability_tol))

    results_payload: dict[str, Any] = {
        "config": {
            "data_path": str(args.data_path.resolve()),
            "holdout_index": int(args.holdout_index),
            "holdout_time": float(holdout_key),
            "max_dim": int(args.max_dim),
            "cache_dir": str(args.cache_dir),
            "initial_num_itermax": int(args.initial_num_itermax),
            "target_min_seconds": float(args.target_min_seconds),
            "target_max_seconds": float(args.target_max_seconds),
            "max_calibration_rounds": int(args.max_calibration_rounds),
            "sweep_scales": scales,
            "stability_tol": float(args.stability_tol),
        },
        "data_stats": {
            "x0_count": int(problem.x0_pool.shape[0]),
            "x1_count": int(problem.x1_pool.shape[0]),
            "holdout_count": int(target.shape[0]),
            "global_ot_support_size": int(mass.shape[0]),
            "global_ot_total_cost": (
                None if problem.global_ot_total_cost is None else float(problem.global_ot_total_cost)
            ),
            "global_ot_cache_path": prepared.global_ot_cache_path,
            "global_ot_cache_hit": bool(prepared.global_ot_cache_hit),
            "cost_matrix_shape": [int(cost_matrix.shape[0]), int(cost_matrix.shape[1])],
            "cost_matrix_build_sec": cost_build_sec,
        },
        "calibration": {
            "target_runtime_met": bool(calibration_success),
            "calibrated_num_itermax": int(calibrated_num_itermax),
            "records": [asdict(record) for record in calibration_records],
        },
        "sweep": {
            "records": [asdict(record) for record in final_records],
        },
        "recommendation": recommendation,
        "all_records": [asdict(record) for record in sweep_records],
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }

    json_path = output_root / "emd2_runtime_sweep.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(results_payload, handle, indent=2)

    tsv_path = output_root / "emd2_runtime_sweep.tsv"
    with open(tsv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "phase",
                "num_itermax",
                "runtime_sec",
                "w2",
                "emd2_value",
                "result_code",
                "iteration_limit_reached",
                "warning",
                "error",
            ]
        )
        for record in sweep_records:
            writer.writerow(
                [
                    record.phase,
                    record.num_itermax,
                    f"{record.runtime_sec:.6f}",
                    "" if record.w2 is None else f"{record.w2:.12f}",
                    "" if record.emd2_value is None else f"{record.emd2_value:.12f}",
                    "" if record.result_code is None else record.result_code,
                    int(record.iteration_limit_reached),
                    "" if record.warning is None else record.warning,
                    "" if record.error is None else record.error,
                ]
            )

    print(f"[done] wrote: {json_path}")
    print(f"[done] wrote: {tsv_path}")
    print(
        "[summary] "
        f"holdout_t={holdout_key:.2f} support={mass.shape[0]} "
        f"target_n={target.shape[0]} calibrated_numItermax={calibrated_num_itermax} "
        f"target_runtime_met={calibration_success}"
    )
    rec_value = recommendation.get("recommended_num_itermax", None)
    print(f"[summary] recommended_numItermax={rec_value}")
    print("[table] final sweep")
    print("numItermax\truntime_sec\tw2\titer_limit\tresult_code")
    for record in sorted(final_records, key=lambda item: int(item.num_itermax)):
        runtime = f"{record.runtime_sec:.6f}"
        w2 = "NA" if record.w2 is None else f"{record.w2:.12f}"
        limit = int(record.iteration_limit_reached)
        rc = "NA" if record.result_code is None else str(record.result_code)
        print(f"{record.num_itermax}\t{runtime}\t{w2}\t{limit}\t{rc}")


if __name__ == "__main__":
    main()
