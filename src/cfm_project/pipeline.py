from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from cfm_project.bridge_data import prepare_bridge_problem_and_targets
from cfm_project.data import analytic_target_moment_features, to_problem_from_config
from cfm_project.plotting import (
    save_rollout_empirical_w2_bar_plot,
    save_rollout_marginal_comparison_grid,
    save_constraint_residual_plot,
    save_interpolant_marginal_comparison_grid,
    save_interpolant_trajectory_comparison,
    save_interpolant_w2_bar_plot,
    save_path_samples_plot,
    save_training_curve,
)
from cfm_project.single_cell_data import prepare_single_cell_problem_and_targets
from cfm_project.training import is_stage_a_only_profile, train_experiment

METRIC_MODES = {"metric", "metric_alpha0", "metric_constrained_al", "metric_constrained_soft"}


def config_to_dict(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return copy.deepcopy(cfg)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _apply_method_override(cfg: dict[str, Any], mode: str) -> dict[str, Any]:
    local_cfg = copy.deepcopy(cfg)
    experiment_cfg = local_cfg.setdefault("experiment", {})
    experiment_cfg["mode"] = mode
    raw_overrides = cfg.get("experiment", {}).get("method_overrides", {})
    if raw_overrides is None:
        return local_cfg
    if not isinstance(raw_overrides, dict):
        raise ValueError("experiment.method_overrides must be a mapping of mode -> override dict.")
    mode_override = raw_overrides.get(mode, {})
    if mode_override is None:
        return local_cfg
    if not isinstance(mode_override, dict):
        raise ValueError(
            f"experiment.method_overrides.{mode} must be a dict, got {type(mode_override)}."
        )
    merged = _deep_merge_dict(local_cfg, mode_override)
    merged.setdefault("experiment", {})
    merged["experiment"]["mode"] = mode
    return merged


def _build_problem_and_targets(
    local_cfg: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    data_cfg = local_cfg["data"]
    data_family = str(data_cfg.get("family", "gaussian")).lower()
    times = [float(t) for t in data_cfg["constraint_times"]]
    if data_family == "gaussian":
        problem = to_problem_from_config(
            mean0=data_cfg["mean0"],
            cov0=data_cfg["cov0"],
            mean1=data_cfg["mean1"],
            cov1=data_cfg["cov1"],
            kappa=float(data_cfg["kappa"]),
            device=device,
            dtype=dtype,
        )
        targets = analytic_target_moment_features(times=times, problem=problem)
        return {
            "problem": problem,
            "targets": targets,
            "pseudo_targets": None,
            "pseudo_posterior": None,
            "target_sampler": None,
            "target_samples_by_time": None,
            "data_family": data_family,
            "data_build_meta": {},
            "pseudo_summary_meta": {},
        }
    if data_family == "bridge_sde":
        prepared = prepare_bridge_problem_and_targets(
            data_cfg=data_cfg,
            seed=int(local_cfg["seed"]),
            device=device,
            dtype=dtype,
        )
        return {
            "problem": prepared.problem,
            "targets": prepared.targets,
            "pseudo_targets": None,
            "pseudo_posterior": None,
            "target_sampler": prepared.target_sampler,
            "target_samples_by_time": None,
            "data_family": data_family,
            "data_build_meta": {
                "target_cache_path": str(prepared.cache_path),
                "target_cache_hit": bool(prepared.cache_hit),
                "target_cache_enabled": bool(data_cfg.get("target_cache_enabled", True)),
                "target_mc_samples": int(data_cfg.get("target_mc_samples", 200000)),
            },
            "pseudo_summary_meta": {},
        }
    if data_family == "single_cell":
        prepared = prepare_single_cell_problem_and_targets(
            data_cfg=data_cfg,
            experiment_cfg=local_cfg.get("experiment", {}),
            device=device,
            dtype=dtype,
        )
        local_cfg["data"]["dim"] = int(prepared.problem.dim)
        local_cfg["data"]["constraint_times"] = [float(t) for t in prepared.constraint_times]
        local_cfg["data"]["interpolant_eval_times"] = [float(t) for t in prepared.eval_times]
        local_cfg.setdefault("experiment", {})
        local_cfg["experiment"]["holdout_time"] = (
            None if prepared.holdout_time is None else float(prepared.holdout_time)
        )
        local_cfg["experiment"]["holdout_index"] = (
            None if prepared.holdout_index is None else int(prepared.holdout_index)
        )
        return {
            "problem": prepared.problem,
            "targets": prepared.targets,
            "pseudo_targets": prepared.pseudo_targets,
            "pseudo_posterior": prepared.pseudo_posterior,
            "target_sampler": prepared.target_sampler,
            "target_samples_by_time": prepared.target_samples_by_time,
            "data_family": data_family,
            "data_build_meta": {
                "single_cell_protocol": prepared.protocol,
                "single_cell_constraint_time_policy": prepared.constraint_time_policy,
                "single_cell_all_time_indices": [int(idx) for idx in prepared.all_time_indices],
                "single_cell_all_time_labels": [str(label) for label in prepared.all_time_labels],
                "single_cell_all_times_normalized": [float(t) for t in prepared.normalized_times_all],
                "single_cell_constraint_time_indices": [
                    int(idx) for idx in prepared.constraint_time_indices
                ],
                "single_cell_constraint_times": [float(t) for t in prepared.constraint_times],
                "single_cell_eval_times": [float(t) for t in prepared.eval_times],
                "single_cell_holdout_index": (
                    None if prepared.holdout_index is None else int(prepared.holdout_index)
                ),
                "single_cell_holdout_time": (
                    None if prepared.holdout_time is None else float(prepared.holdout_time)
                ),
                "single_cell_global_ot_cache_path": prepared.global_ot_cache_path,
                "single_cell_global_ot_cache_hit": bool(prepared.global_ot_cache_hit),
                "single_cell_global_ot_support_size": prepared.global_ot_support_size,
                "single_cell_global_ot_total_cost": prepared.global_ot_total_cost,
                "pseudo_labels_k": prepared.pseudo_labels_k,
                "pseudo_labels_cache_path": prepared.pseudo_labels_cache_path,
                "pseudo_labels_cache_hit": bool(prepared.pseudo_labels_cache_hit),
                "single_cell_pseudo_fit_times": (
                    None
                    if prepared.pseudo_fit_times is None
                    else [float(t) for t in prepared.pseudo_fit_times]
                ),
                "single_cell_pseudo_fit_sample_count": prepared.pseudo_fit_sample_count,
                "bic_by_k": (
                    None
                    if prepared.pseudo_labels_bic_by_k is None
                    else {str(int(k)): float(v) for k, v in prepared.pseudo_labels_bic_by_k.items()}
                ),
                "stability_by_k": (
                    None
                    if prepared.pseudo_labels_stability_by_k is None
                    else {
                        str(int(k)): float(v)
                        for k, v in prepared.pseudo_labels_stability_by_k.items()
                    }
                ),
            },
            "pseudo_summary_meta": {
                "pseudo_labels_k": prepared.pseudo_labels_k,
                "pseudo_labels_cache_path": prepared.pseudo_labels_cache_path,
                "pseudo_labels_cache_hit": bool(prepared.pseudo_labels_cache_hit),
                "single_cell_constraint_times": [float(t) for t in prepared.constraint_times],
                "single_cell_eval_times": [float(t) for t in prepared.eval_times],
                "single_cell_pseudo_fit_times": (
                    None
                    if prepared.pseudo_fit_times is None
                    else [float(t) for t in prepared.pseudo_fit_times]
                ),
                "single_cell_pseudo_fit_sample_count": prepared.pseudo_fit_sample_count,
                "bic_by_k": (
                    None
                    if prepared.pseudo_labels_bic_by_k is None
                    else {str(int(k)): float(v) for k, v in prepared.pseudo_labels_bic_by_k.items()}
                ),
                "stability_by_k": (
                    None
                    if prepared.pseudo_labels_stability_by_k is None
                    else {
                        str(int(k)): float(v)
                        for k, v in prepared.pseudo_labels_stability_by_k.items()
                    }
                ),
            },
        }
    raise ValueError(
        f"Unsupported data family '{data_family}'. Expected one of: gaussian, bridge_sde, single_cell."
    )


def _run_single_mode(
    cfg: dict[str, Any],
    mode: str,
    output_dir: Path,
) -> dict[str, Any]:
    local_cfg = _apply_method_override(cfg=cfg, mode=mode)

    device = torch.device(local_cfg["device"])
    dtype = torch.float32
    data_cfg = local_cfg["data"]
    coupling = str(data_cfg.get("coupling", "ot"))
    stage_a_only = is_stage_a_only_profile(local_cfg["train"])

    built = _build_problem_and_targets(local_cfg=local_cfg, device=device, dtype=dtype)
    result = train_experiment(
        local_cfg,
        problem=built["problem"],
        targets=built["targets"],
        pseudo_targets=built.get("pseudo_targets"),
        pseudo_posterior=built.get("pseudo_posterior"),
        target_sampler=built["target_sampler"],
        target_samples_by_time=built.get("target_samples_by_time"),
        data_family=built["data_family"],
    )
    if built.get("pseudo_summary_meta"):
        result["summary"].update(built["pseudo_summary_meta"])

    mode_dir = output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "summary": result["summary"],
        "history": result["history"],
        "mode": mode,
        "config": local_cfg,
        "data_build_meta": built["data_build_meta"],
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
        mfm_alpha = 0.0 if mode == "metric_alpha0" else float(local_cfg.get("mfm", {}).get("alpha", 1.0))
        save_path_samples_plot(
            mode=mode,
            problem=built["problem"],
            path=mode_dir / "sample_paths.png",
            g_model=result["path_model"],
            mfm_alpha=mfm_alpha,
            coupling=coupling,
            n_pairs=int(local_cfg["output"]["plot_pairs"]),
        )
        if int(local_cfg["data"]["dim"]) > 2:
            save_path_samples_plot(
                mode=mode,
                problem=built["problem"],
                path=mode_dir / "sample_paths_proj12.png",
                g_model=result["path_model"],
                mfm_alpha=mfm_alpha,
                coupling=coupling,
                n_pairs=int(local_cfg["output"]["plot_pairs"]),
            )
        if (
            stage_a_only
            and mode in {"constrained", *METRIC_MODES}
            and result.get("interpolant_artifacts") is not None
        ):
            artifacts = result["interpolant_artifacts"]
            save_interpolant_trajectory_comparison(
                x0=artifacts["x0"],
                x1=artifacts["x1"],
                path=mode_dir / "interpolant_trajectories.png",
                g_model=result["path_model"],
                mode=mode,
                mfm_alpha=float(mfm_alpha),
            )
            save_interpolant_marginal_comparison_grid(
                linear_samples_by_time=artifacts["linear_by_time"],
                learned_samples_by_time=artifacts["learned_by_time"],
                target_samples_by_time=artifacts["target_by_time"],
                path=mode_dir / "interpolant_marginal_grid.png",
            )
            if int(local_cfg["data"]["dim"]) > 2:
                save_interpolant_trajectory_comparison(
                    x0=artifacts["x0"],
                    x1=artifacts["x1"],
                    path=mode_dir / "interpolant_trajectories_proj12.png",
                    g_model=result["path_model"],
                    mode=mode,
                    mfm_alpha=float(mfm_alpha),
                )
                save_interpolant_marginal_comparison_grid(
                    linear_samples_by_time=artifacts["linear_by_time"],
                    learned_samples_by_time=artifacts["learned_by_time"],
                    target_samples_by_time=artifacts["target_by_time"],
                    path=mode_dir / "interpolant_marginal_grid_proj12.png",
                )
            interpolant_eval = result["summary"].get("interpolant_eval", {})
            linear_w2 = interpolant_eval.get("linear_empirical_w2", {})
            learned_w2 = interpolant_eval.get("learned_empirical_w2", {})
            if linear_w2 and learned_w2:
                save_interpolant_w2_bar_plot(
                    linear_empirical_w2=linear_w2,
                    learned_empirical_w2=learned_w2,
                    path=mode_dir / "interpolant_empirical_w2.png",
                )
        if (not stage_a_only) and result.get("rollout_artifacts") is not None:
            rollout_artifacts = result["rollout_artifacts"]
            save_rollout_marginal_comparison_grid(
                generated_samples_by_time=rollout_artifacts["generated_by_time"],
                target_samples_by_time=rollout_artifacts["target_by_time"],
                path=mode_dir / "rollout_marginal_grid.png",
            )
            if int(local_cfg["data"]["dim"]) > 2:
                save_rollout_marginal_comparison_grid(
                    generated_samples_by_time=rollout_artifacts["generated_by_time"],
                    target_samples_by_time=rollout_artifacts["target_by_time"],
                    path=mode_dir / "rollout_marginal_grid_proj12.png",
                )
            empirical_w2_by_time = rollout_artifacts.get("empirical_w2_by_time", {})
            if empirical_w2_by_time:
                save_rollout_empirical_w2_bar_plot(
                    empirical_w2_by_time=empirical_w2_by_time,
                    path=mode_dir / "rollout_empirical_w2.png",
                )

    return {
        "summary": result["summary"],
        "mode_dir": str(mode_dir),
    }


def _comparison_meta(cfg: dict[str, Any], stage_a_only: bool, methods: list[str] | None = None) -> dict[str, Any]:
    stage_a_steps = int(cfg["train"]["stage_a_steps"])
    stage_b_steps = int(cfg["train"]["stage_b_steps"])
    stage_c_steps = int(cfg["train"]["stage_c_steps"])
    data_cfg = cfg["data"]
    meta: dict[str, Any] = {
        "experiment_label": str(cfg["experiment"].get("label", "unknown")),
        "train_label": str(cfg["train"].get("label", "unknown")),
        "data_label": str(data_cfg.get("label", "unknown")),
        "data_family": str(data_cfg.get("family", "gaussian")),
        "coupling": str(data_cfg.get("coupling", "ot")),
        "stage_steps": {
            "stage_a_steps": stage_a_steps,
            "stage_b_steps": stage_b_steps,
            "stage_c_steps": stage_c_steps,
        },
        "stage_a_only": bool(stage_a_only),
        "stage_c_enabled": bool(stage_c_steps > 0),
    }
    if "protocol" in cfg.get("experiment", {}):
        meta["protocol"] = str(cfg["experiment"].get("protocol"))
    if "holdout_index" in cfg.get("experiment", {}):
        meta["holdout_index"] = cfg["experiment"].get("holdout_index")
    if "holdout_indices" in cfg.get("experiment", {}):
        raw_holdout_indices = cfg["experiment"].get("holdout_indices", [])
        if isinstance(raw_holdout_indices, (list, tuple)):
            meta["holdout_indices"] = [int(v) for v in raw_holdout_indices]
        else:
            meta["holdout_indices"] = []
    if "holdout_time" in cfg.get("experiment", {}):
        meta["holdout_time"] = cfg["experiment"].get("holdout_time")
    if methods is not None:
        meta["methods"] = [str(m) for m in methods]
    return meta


def _normalize_methods(raw_methods: Any) -> list[str]:
    if raw_methods is None:
        return []
    if not isinstance(raw_methods, (list, tuple)):
        raise ValueError("experiment.comparison_methods must be a list of mode names.")
    methods: list[str] = []
    seen: set[str] = set()
    allowed = {"baseline", "constrained", *METRIC_MODES}
    for item in raw_methods:
        mode = str(item).strip()
        if mode not in allowed:
            raise ValueError(
                f"Unsupported comparison method '{mode}'. "
                "Expected one of: baseline, constrained, metric, metric_alpha0, "
                "metric_constrained_al, metric_constrained_soft."
            )
        if mode in seen:
            continue
        methods.append(mode)
        seen.add(mode)
    return methods


def run_pipeline(
    cfg: DictConfig | dict[str, Any],
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    cfg_dict = config_to_dict(cfg)
    out_root = Path.cwd() if output_dir is None else Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    stage_a_only = is_stage_a_only_profile(cfg_dict["train"])
    methods = _normalize_methods(cfg_dict["experiment"].get("comparison_methods"))

    if methods:
        if bool(cfg_dict["experiment"].get("run_both_modes", False)):
            raise ValueError("experiment.comparison_methods cannot be used with run_both_modes=true.")
        if stage_a_only and any(mode not in {"constrained", *METRIC_MODES} for mode in methods):
            raise ValueError(
                "train=stage_a_only only supports constrained and metric-family modes. "
                "Allowed methods: constrained, metric, metric_alpha0, "
                "metric_constrained_al, metric_constrained_soft."
            )

        summaries: dict[str, Any] = {}
        mode_dirs: dict[str, str] = {}
        for mode in methods:
            run = _run_single_mode(cfg_dict, mode=mode, output_dir=out_root)
            summaries[mode] = run["summary"]
            mode_dirs[f"{mode}_dir"] = run["mode_dir"]

        comparison_meta = _comparison_meta(cfg_dict, stage_a_only, methods=methods)
        if "holdout_time" not in comparison_meta:
            for summary in summaries.values():
                if "holdout_time" in summary:
                    comparison_meta["holdout_time"] = summary["holdout_time"]
                    break
        comparison_mfm = {"meta": comparison_meta}
        comparison_mfm.update(summaries)
        _write_json(out_root / "comparison_mfm.json", comparison_mfm)

        if "baseline" in summaries and "constrained" in summaries:
            legacy_meta = _comparison_meta(cfg_dict, stage_a_only)
            if "holdout_time" not in legacy_meta and "holdout_time" in summaries["baseline"]:
                legacy_meta["holdout_time"] = summaries["baseline"]["holdout_time"]
            legacy = {
                "meta": legacy_meta,
                "baseline": summaries["baseline"],
                "constrained": summaries["constrained"],
            }
            _write_json(out_root / "comparison.json", legacy)

        return {
            "comparison_mfm": comparison_mfm,
            "comparison_mfm_path": str(out_root / "comparison_mfm.json"),
            **mode_dirs,
        }

    if bool(cfg_dict["experiment"]["run_both_modes"]):
        if stage_a_only:
            raise ValueError(
                "run_both_modes is incompatible with train=stage_a_only. "
                "Use constrained mode only for Stage-A-only experiments."
            )
        baseline_result = _run_single_mode(cfg_dict, mode="baseline", output_dir=out_root)
        constrained_result = _run_single_mode(cfg_dict, mode="constrained", output_dir=out_root)
        comparison = {
            "meta": _comparison_meta(cfg_dict, stage_a_only),
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
    if stage_a_only and mode not in {"constrained", *METRIC_MODES}:
        raise ValueError(
            "train=stage_a_only requires experiment.mode in "
            "{constrained, metric, metric_alpha0, metric_constrained_al, metric_constrained_soft}."
        )
    single = _run_single_mode(cfg_dict, mode=mode, output_dir=out_root)
    return {"summary": single["summary"], "mode_dir": single["mode_dir"]}
