from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from cfm_project.data import analytic_target_moment_features, to_problem_from_config
from cfm_project.plotting import (
    save_constraint_residual_plot,
    save_path_samples_plot,
    save_training_curve,
)
from cfm_project.training import train_experiment


def config_to_dict(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return copy.deepcopy(cfg)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _run_single_mode(
    cfg: dict[str, Any],
    mode: str,
    output_dir: Path,
) -> dict[str, Any]:
    local_cfg = copy.deepcopy(cfg)
    local_cfg["experiment"]["mode"] = mode

    device = torch.device(local_cfg["device"])
    dtype = torch.float32
    data_cfg = local_cfg["data"]
    coupling = str(data_cfg.get("coupling", "ot"))
    problem = to_problem_from_config(
        mean0=data_cfg["mean0"],
        cov0=data_cfg["cov0"],
        mean1=data_cfg["mean1"],
        cov1=data_cfg["cov1"],
        kappa=float(data_cfg["kappa"]),
        device=device,
        dtype=dtype,
    )
    times = [float(t) for t in data_cfg["constraint_times"]]
    targets = analytic_target_moment_features(times=times, problem=problem)

    result = train_experiment(local_cfg, problem=problem, targets=targets)
    mode_dir = output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "summary": result["summary"],
        "history": result["history"],
        "mode": mode,
        "config": local_cfg,
    }
    _write_json(mode_dir / "metrics.json", metrics_payload)

    if bool(local_cfg["output"]["save_checkpoint"]):
        torch.save(result["checkpoint"], mode_dir / "checkpoint.pt")

    if bool(local_cfg["output"]["save_plots"]):
        save_training_curve(result["history"], mode_dir / "training_curve.png")
        residuals = {
            float(k): float(v)
            for k, v in result["summary"]["constraint_residual_norms"].items()
        }
        save_constraint_residual_plot(residuals, mode_dir / "constraint_residuals.png")
        save_path_samples_plot(
            mode=mode,
            problem=problem,
            path=mode_dir / "sample_paths.png",
            g_model=result["path_model"],
            coupling=coupling,
            n_pairs=int(local_cfg["output"]["plot_pairs"]),
        )

    return {
        "summary": result["summary"],
        "mode_dir": str(mode_dir),
    }


def run_pipeline(cfg: DictConfig | dict[str, Any], output_dir: str | Path | None = None) -> dict[str, Any]:
    cfg_dict = config_to_dict(cfg)
    out_root = Path.cwd() if output_dir is None else Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if bool(cfg_dict["experiment"]["run_both_modes"]):
        baseline_result = _run_single_mode(cfg_dict, mode="baseline", output_dir=out_root)
        constrained_result = _run_single_mode(cfg_dict, mode="constrained", output_dir=out_root)
        stage_a_steps = int(cfg_dict["train"]["stage_a_steps"])
        stage_b_steps = int(cfg_dict["train"]["stage_b_steps"])
        stage_c_steps = int(cfg_dict["train"]["stage_c_steps"])
        data_cfg = cfg_dict["data"]
        comparison = {
            "meta": {
                "experiment_label": str(cfg_dict["experiment"].get("label", "unknown")),
                "train_label": str(cfg_dict["train"].get("label", "unknown")),
                "data_label": str(data_cfg.get("label", "unknown")),
                "coupling": str(data_cfg.get("coupling", "ot")),
                "stage_steps": {
                    "stage_a_steps": stage_a_steps,
                    "stage_b_steps": stage_b_steps,
                    "stage_c_steps": stage_c_steps,
                },
                "stage_c_enabled": bool(stage_c_steps > 0),
            },
            "baseline": baseline_result["summary"],
            "constrained": constrained_result["summary"],
        }
        _write_json(out_root / "comparison.json", comparison)
        return {
            "comparison": comparison,
            "baseline_dir": baseline_result["mode_dir"],
            "constrained_dir": constrained_result["mode_dir"],
        }

    mode = str(cfg_dict["experiment"]["mode"])
    single = _run_single_mode(cfg_dict, mode=mode, output_dir=out_root)
    return {"summary": single["summary"], "mode_dir": single["mode_dir"]}
