from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import torch


def _round_optional(value: float | int | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _safe_ratio(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or baseline <= 0.0:
        return None
    return float(value) / float(baseline)


def _current_rss_mb() -> float | None:
    try:
        import psutil  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        process = psutil.Process()
        return float(process.memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return None


def _peak_rss_mb() -> float | None:
    try:
        import resource
    except Exception:
        return None
    try:
        raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    if raw <= 0.0:
        return None
    if sys.platform == "darwin":
        return raw / (1024.0 * 1024.0)
    return raw / 1024.0


def reset_gpu_peak_memory(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    torch.cuda.reset_peak_memory_stats(device)


def memory_snapshot(device: torch.device) -> dict[str, float | None]:
    current_rss = _current_rss_mb()
    peak_rss = _peak_rss_mb()
    snapshot: dict[str, float | None] = {
        "rss_mb": _round_optional(current_rss),
        "peak_rss_mb": _round_optional(peak_rss),
        "gpu_memory_allocated_mb": None,
        "gpu_memory_reserved_mb": None,
        "gpu_peak_memory_allocated_mb": None,
        "gpu_peak_memory_reserved_mb": None,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        snapshot.update(
            {
                "gpu_memory_allocated_mb": _round_optional(
                    torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)
                ),
                "gpu_memory_reserved_mb": _round_optional(
                    torch.cuda.memory_reserved(device) / (1024.0 * 1024.0)
                ),
                "gpu_peak_memory_allocated_mb": _round_optional(
                    torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
                ),
                "gpu_peak_memory_reserved_mb": _round_optional(
                    torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)
                ),
            }
        )
    return snapshot


def model_parameter_counts(
    velocity_model: torch.nn.Module,
    path_model: torch.nn.Module | None,
) -> dict[str, int]:
    velocity_total = sum(int(param.numel()) for param in velocity_model.parameters())
    velocity_trainable = sum(
        int(param.numel()) for param in velocity_model.parameters() if param.requires_grad
    )
    path_total = 0 if path_model is None else sum(int(param.numel()) for param in path_model.parameters())
    path_trainable = (
        0
        if path_model is None
        else sum(int(param.numel()) for param in path_model.parameters() if param.requires_grad)
    )
    return {
        "velocity_total": int(velocity_total),
        "velocity_trainable_final": int(velocity_trainable),
        "path_total": int(path_total),
        "path_trainable_final": int(path_trainable),
        "total": int(velocity_total + path_total),
    }


def _stage_counts(history: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"stage_a": 0, "stage_b": 0, "stage_c": 0}
    for row in history:
        stage = str(row.get("stage", ""))
        if stage in counts:
            counts[stage] += 1
    return counts


def build_run_cost_metrics(
    *,
    cfg: dict[str, Any],
    mode: str,
    history: list[dict[str, Any]],
    training_cost: dict[str, Any],
    data_build_meta: dict[str, Any],
    data_build_wall_sec: float,
    artifact_wall_sec: float,
    total_wall_sec: float,
    memory_start: dict[str, float | None],
    memory_end: dict[str, float | None],
    parameters: dict[str, int],
) -> dict[str, Any]:
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    stage_counts = _stage_counts(history)
    configured_steps = {
        "stage_a": int(train_cfg["stage_a_steps"]),
        "stage_b": int(train_cfg["stage_b_steps"]),
        "stage_c": int(train_cfg["stage_c_steps"]),
    }
    executed_total_steps = int(sum(stage_counts.values()))
    batch_size = int(train_cfg["batch_size"])
    train_samples_seen = int(executed_total_steps * batch_size)
    stage_wall = training_cost.get("stage_wall_seconds", {})
    stage_a_wall = float(stage_wall.get("stage_a_wall_sec", 0.0))
    stage_b_wall = float(stage_wall.get("stage_b_wall_sec", 0.0))
    stage_c_wall = float(stage_wall.get("stage_c_wall_sec", 0.0))
    eval_wall = float(stage_wall.get("eval_wall_sec", 0.0))
    train_loop_wall = stage_a_wall + stage_b_wall + stage_c_wall
    train_experiment_wall = float(
        stage_wall.get("train_experiment_wall_sec", train_loop_wall + eval_wall)
    )
    train_steps_per_sec = _safe_ratio(float(executed_total_steps), train_loop_wall)
    train_samples_per_sec = _safe_ratio(float(train_samples_seen), train_loop_wall)

    start_rss = memory_start.get("rss_mb")
    end_rss = memory_end.get("rss_mb")
    start_peak_rss = memory_start.get("peak_rss_mb")
    end_peak_rss = memory_end.get("peak_rss_mb")
    peak_delta = None
    if start_peak_rss is not None and end_peak_rss is not None:
        peak_delta = max(0.0, float(end_peak_rss) - float(start_peak_rss))
    rss_delta = None
    if start_rss is not None and end_rss is not None:
        rss_delta = float(end_rss) - float(start_rss)

    cache_meta = {
        key: value
        for key, value in data_build_meta.items()
        if "cache" in str(key).lower() or "solve_seconds" in str(key).lower()
    }
    cost = {
        "schema_version": 1,
        "mode": str(mode),
        "seed": int(cfg["seed"]),
        "experiment_label": str(cfg.get("experiment", {}).get("label", "unknown")),
        "train_label": str(train_cfg.get("label", "unknown")),
        "data_label": str(data_cfg.get("label", "unknown")),
        "data_family": str(data_cfg.get("family", "gaussian")),
        "data_dim": int(data_cfg["dim"]),
        "device": str(cfg["device"]),
        "wall_seconds": {
            "data_build": _round_optional(data_build_wall_sec),
            "stage_a": _round_optional(stage_a_wall),
            "stage_b": _round_optional(stage_b_wall),
            "stage_c": _round_optional(stage_c_wall),
            "train_loop": _round_optional(train_loop_wall),
            "evaluation": _round_optional(eval_wall),
            "train_experiment": _round_optional(train_experiment_wall),
            "artifact_writing": _round_optional(artifact_wall_sec),
            "total": _round_optional(total_wall_sec),
        },
        "memory_mb": {
            "start_rss": _round_optional(start_rss),
            "end_rss": _round_optional(end_rss),
            "rss_delta": _round_optional(rss_delta),
            "start_process_peak_rss": _round_optional(start_peak_rss),
            "end_process_peak_rss": _round_optional(end_peak_rss),
            "process_peak_rss_delta": _round_optional(peak_delta),
            "gpu_peak_memory_allocated": memory_end.get("gpu_peak_memory_allocated_mb"),
            "gpu_peak_memory_reserved": memory_end.get("gpu_peak_memory_reserved_mb"),
            "gpu_end_memory_allocated": memory_end.get("gpu_memory_allocated_mb"),
            "gpu_end_memory_reserved": memory_end.get("gpu_memory_reserved_mb"),
        },
        "parameters": parameters,
        "steps": {
            "configured": configured_steps,
            "executed": stage_counts,
            "executed_total": executed_total_steps,
            "batch_size": batch_size,
            "eval_batch_size": int(train_cfg["eval_batch_size"]),
            "eval_transport_samples": int(train_cfg.get("eval_transport_samples", 0)),
            "eval_transport_steps": int(train_cfg.get("eval_transport_steps", 0)),
            "eval_intermediate_ot_samples": int(train_cfg.get("eval_intermediate_ot_samples", 0)),
            "train_samples_seen": train_samples_seen,
        },
        "throughput": {
            "train_steps_per_second": _round_optional(train_steps_per_sec),
            "train_samples_per_second": _round_optional(train_samples_per_sec),
        },
        "cache": cache_meta,
        "notes": [
            "CPU RSS fields are process-level measurements; in multi-method in-process comparisons, peak-RSS deltas can undercount methods that run after the process peak is already established.",
            "GPU peak fields are per-run only when CUDA is used because the pipeline resets CUDA peak memory stats at method start.",
        ],
    }
    return cost


def build_cost_comparison(
    *,
    costs_by_mode: dict[str, dict[str, Any]],
    methods: list[str],
    meta: dict[str, Any],
) -> dict[str, Any]:
    baseline_mode = "baseline" if "baseline" in costs_by_mode else None
    baseline = None if baseline_mode is None else costs_by_mode[baseline_mode]
    baseline_total = None if baseline is None else baseline["wall_seconds"].get("total")
    baseline_train = None if baseline is None else baseline["wall_seconds"].get("train_loop")
    baseline_peak = None if baseline is None else baseline["memory_mb"].get("end_process_peak_rss")
    baseline_gpu_peak = (
        None if baseline is None else baseline["memory_mb"].get("gpu_peak_memory_allocated")
    )

    rows: list[dict[str, Any]] = []
    for mode in methods:
        if mode not in costs_by_mode:
            continue
        cost = costs_by_mode[mode]
        wall = cost["wall_seconds"]
        memory = cost["memory_mb"]
        steps = cost["steps"]
        row = {
            "method": mode,
            "total_wall_sec": wall.get("total"),
            "total_wall_ratio_vs_baseline": _round_optional(
                _safe_ratio(wall.get("total"), baseline_total)
            ),
            "train_loop_wall_sec": wall.get("train_loop"),
            "train_loop_wall_ratio_vs_baseline": _round_optional(
                _safe_ratio(wall.get("train_loop"), baseline_train)
            ),
            "data_build_wall_sec": wall.get("data_build"),
            "stage_a_wall_sec": wall.get("stage_a"),
            "stage_b_wall_sec": wall.get("stage_b"),
            "stage_c_wall_sec": wall.get("stage_c"),
            "eval_wall_sec": wall.get("evaluation"),
            "artifact_wall_sec": wall.get("artifact_writing"),
            "end_process_peak_rss_mb": memory.get("end_process_peak_rss"),
            "process_peak_rss_ratio_vs_baseline": _round_optional(
                _safe_ratio(memory.get("end_process_peak_rss"), baseline_peak)
            ),
            "gpu_peak_memory_allocated_mb": memory.get("gpu_peak_memory_allocated"),
            "gpu_peak_memory_allocated_ratio_vs_baseline": _round_optional(
                _safe_ratio(memory.get("gpu_peak_memory_allocated"), baseline_gpu_peak)
            ),
            "parameter_count_total": cost["parameters"].get("total"),
            "executed_train_steps": steps.get("executed_total"),
            "train_samples_seen": steps.get("train_samples_seen"),
            "train_steps_per_second": cost["throughput"].get("train_steps_per_second"),
            "train_samples_per_second": cost["throughput"].get("train_samples_per_second"),
        }
        rows.append(row)
    return {
        "schema_version": 1,
        "meta": meta,
        "baseline_mode": baseline_mode,
        "rows": rows,
        "by_mode": costs_by_mode,
    }


def _format_cell(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_cost_summary_artifacts(output_dir: Path, cost_summary: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "cost_summary.json"
    csv_path = output_dir / "cost_summary.csv"
    md_path = output_dir / "cost_summary.md"

    import json

    json_path.write_text(json.dumps(cost_summary, indent=2, sort_keys=True), encoding="utf-8")

    columns = [
        "method",
        "total_wall_sec",
        "total_wall_ratio_vs_baseline",
        "train_loop_wall_sec",
        "train_loop_wall_ratio_vs_baseline",
        "data_build_wall_sec",
        "stage_a_wall_sec",
        "stage_b_wall_sec",
        "stage_c_wall_sec",
        "eval_wall_sec",
        "artifact_wall_sec",
        "end_process_peak_rss_mb",
        "process_peak_rss_ratio_vs_baseline",
        "gpu_peak_memory_allocated_mb",
        "gpu_peak_memory_allocated_ratio_vs_baseline",
        "parameter_count_total",
        "executed_train_steps",
        "train_samples_seen",
        "train_steps_per_second",
        "train_samples_per_second",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in cost_summary["rows"]:
            writer.writerow({col: row.get(col) for col in columns})

    md_headers = [
        "Method",
        "Total sec",
        "x baseline",
        "Train sec",
        "Train x",
        "Stage A",
        "Stage B",
        "Stage C",
        "Eval",
        "Peak RSS MB",
        "Params",
        "Steps/s",
    ]
    rows = [
        "| " + " | ".join(md_headers) + " |",
        "|---|" + "|".join(["---"] * (len(md_headers) - 1)) + "|",
    ]
    for row in cost_summary["rows"]:
        rows.append(
            "| "
            + " | ".join(
                [
                    str(row.get("method", "")),
                    _format_cell(row.get("total_wall_sec")),
                    _format_cell(row.get("total_wall_ratio_vs_baseline"), digits=2),
                    _format_cell(row.get("train_loop_wall_sec")),
                    _format_cell(row.get("train_loop_wall_ratio_vs_baseline"), digits=2),
                    _format_cell(row.get("stage_a_wall_sec")),
                    _format_cell(row.get("stage_b_wall_sec")),
                    _format_cell(row.get("stage_c_wall_sec")),
                    _format_cell(row.get("eval_wall_sec")),
                    _format_cell(row.get("end_process_peak_rss_mb"), digits=1),
                    _format_cell(row.get("parameter_count_total"), digits=0),
                    _format_cell(row.get("train_steps_per_second"), digits=2),
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return {
        "cost_summary_path": str(json_path),
        "cost_summary_csv_path": str(csv_path),
        "cost_summary_md_path": str(md_path),
    }
