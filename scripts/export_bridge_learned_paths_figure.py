#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from omegaconf import OmegaConf
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CACHE_DIR = ROOT / ".cache"
MPL_CONFIG = CACHE_DIR / "matplotlib"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cfm_project.bridge_data import prepare_bridge_problem_and_targets
from cfm_project.data import EmpiricalCouplingProblem
from cfm_project.mfm_core import mfm_mean_path
from cfm_project.models import PathCorrection
from cfm_project.paths import corrected_path, linear_path


DEFAULT_STAGE_A_SEED_DIR = (
    ROOT
    / "outputs"
    / "2026-05-07"
    / "bridge_global_ot_3k_dual_single_seed_regular"
    / "stage_a"
    / "seed_3"
)
DEFAULT_METHODS = ["baseline", "constrained", "metric"]
METRIC_MODES = {"metric", "metric_alpha0", "metric_constrained_al", "metric_constrained_soft"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a 3-panel synthetic bridge learned-path comparison figure "
            "(Flow Matching vs Constrained vs Metric FM) from Stage-A checkpoints."
        ),
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="bridge_ot_global_3k",
        help=(
            "Data config name under configs/data/ (for example: bridge_ot_global_3k), "
            "or an explicit path to a YAML file."
        ),
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=3,
        help="Seed used to prepare/load the bridge endpoint pools (default: 3).",
    )
    parser.add_argument(
        "--stage-a-seed-dir",
        type=Path,
        default=DEFAULT_STAGE_A_SEED_DIR,
        help=(
            "Directory containing per-method Stage-A checkpoints "
            "(for example outputs/.../stage_a/seed_3)."
        ),
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="baseline,constrained,metric",
        help="Comma-separated method folders to render (default: baseline,constrained,metric).",
    )
    parser.add_argument(
        "--max-paths",
        type=int,
        default=120,
        help="Number of coupled endpoint pairs/trajectories to plot (default: 120).",
    )
    parser.add_argument(
        "--n-time-points",
        type=int,
        default=45,
        help="Number of time samples per trajectory curve (default: 45).",
    )
    parser.add_argument(
        "--pair-selection",
        type=str,
        default="mass",
        choices=["mass", "uniform"],
        help=(
            "How to choose support pairs from global OT support: "
            "'mass' (weighted by support mass) or 'uniform'."
        ),
    )
    parser.add_argument(
        "--pair-seed",
        type=int,
        default=17,
        help="Random seed used for support-pair subset selection (default: 17).",
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=0.9,
        help="Trajectory line width (default: 0.9).",
    )
    parser.add_argument(
        "--line-alpha",
        type=float,
        default=0.42,
        help="Trajectory line alpha in [0,1] (default: 0.42).",
    )
    parser.add_argument(
        "--endpoint-size",
        type=float,
        default=9.0,
        help="Endpoint marker size in points^2 (default: 9.0).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=240,
        help="DPI for raster export (default: 240).",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="png,pdf",
        help="Comma-separated output formats from {png,pdf} (default: png,pdf).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: "
            "outputs/paper_figures/bridge_learned_paths/<timestamp>/"
        ),
    )
    parser.add_argument(
        "--separate-panels",
        action="store_true",
        help="Save each method as its own figure instead of a single multi-panel figure.",
    )
    parser.add_argument(
        "--hide-legend",
        action="store_true",
        help="Hide endpoint legend.",
    )
    parser.add_argument(
        "--hide-title",
        action="store_true",
        help="Hide figure title.",
    )
    parser.add_argument(
        "--hide-panel-labels",
        action="store_true",
        help="Hide '(a) Method' panel labels.",
    )
    return parser.parse_args()


def _resolve_data_cfg_path(data_config: str) -> Path:
    candidate = Path(data_config)
    if candidate.suffix in {".yaml", ".yml"}:
        path = candidate if candidate.is_absolute() else (ROOT / candidate).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        return path
    path = ROOT / "configs" / "data" / f"{data_config}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Unknown data config '{data_config}'. Expected: {path}")
    return path.resolve()


def _load_data_cfg(path: Path) -> dict[str, Any]:
    loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping config at {path}, got {type(loaded)}.")
    return loaded


def _parse_methods(raw: str) -> list[str]:
    methods = [chunk.strip() for chunk in str(raw).split(",") if chunk.strip()]
    if not methods:
        raise ValueError("At least one method must be provided in --methods.")
    return methods


def _parse_formats(raw: str) -> list[str]:
    allowed = {"png", "pdf"}
    out: list[str] = []
    for chunk in str(raw).split(","):
        fmt = chunk.strip().lower()
        if not fmt:
            continue
        if fmt not in allowed:
            raise ValueError(f"Unsupported format '{fmt}'. Supported: {sorted(allowed)}.")
        out.append(fmt)
    if not out:
        raise ValueError("At least one format must be provided in --formats.")
    return list(dict.fromkeys(out))


def _resolve_out_dir(cli_out_dir: Path | None) -> Path:
    if cli_out_dir is not None:
        return cli_out_dir if cli_out_dir.is_absolute() else (ROOT / cli_out_dir).resolve()
    tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return (ROOT / "outputs" / "paper_figures" / "bridge_learned_paths" / tag).resolve()


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.max_paths) <= 0:
        raise ValueError(f"--max-paths must be > 0, got {args.max_paths}.")
    if int(args.n_time_points) < 2:
        raise ValueError(f"--n-time-points must be >= 2, got {args.n_time_points}.")
    if float(args.line_width) <= 0.0:
        raise ValueError(f"--line-width must be > 0, got {args.line_width}.")
    if not (0.0 <= float(args.line_alpha) <= 1.0):
        raise ValueError(f"--line-alpha must be in [0,1], got {args.line_alpha}.")
    if float(args.endpoint_size) <= 0.0:
        raise ValueError(f"--endpoint-size must be > 0, got {args.endpoint_size}.")


def _sample_coupled_pairs(
    problem: EmpiricalCouplingProblem,
    n_pairs: int,
    selection: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    if not problem.has_global_ot_support:
        raise ValueError(
            "Expected EmpiricalCouplingProblem with global OT support. "
            "Use a data config with coupling=ot_global."
        )
    assert problem.global_ot_src_idx is not None
    assert problem.global_ot_tgt_idx is not None
    assert problem.global_ot_mass is not None

    src_support = problem.global_ot_src_idx.detach().cpu().numpy()
    tgt_support = problem.global_ot_tgt_idx.detach().cpu().numpy()
    mass = problem.global_ot_mass.detach().cpu().numpy().astype(np.float64)
    support_size = int(mass.shape[0])
    if support_size <= 0:
        raise ValueError("Global OT support is empty.")
    n_take = int(min(int(n_pairs), support_size))
    probs = mass / max(float(mass.sum()), np.finfo(np.float64).eps)

    rng = np.random.default_rng(int(seed))
    if selection == "mass":
        chosen = rng.choice(support_size, size=n_take, replace=False, p=probs)
    elif selection == "uniform":
        chosen = rng.choice(support_size, size=n_take, replace=False)
    else:
        raise ValueError(f"Unsupported pair selection mode '{selection}'.")

    src_idx = torch.as_tensor(src_support[chosen], dtype=torch.long, device=problem.x0_pool.device)
    tgt_idx = torch.as_tensor(tgt_support[chosen], dtype=torch.long, device=problem.x1_pool.device)
    x0 = problem.x0_pool[src_idx]
    x1 = problem.x1_pool[tgt_idx]
    return x0, x1, chosen


def _build_path_model_from_checkpoint(
    checkpoint: dict[str, Any],
    state_dim: int,
) -> PathCorrection | None:
    state_dict = checkpoint.get("path_state_dict")
    if state_dict is None:
        return None
    cfg = checkpoint.get("config", {})
    if not isinstance(cfg, dict):
        raise ValueError("Checkpoint config missing or malformed.")
    model_cfg = cfg.get("model", {})
    if not isinstance(model_cfg, dict):
        raise ValueError("Checkpoint model config missing or malformed.")
    hidden_dims = model_cfg.get("path_hidden_dims", [128, 128])
    activation = str(model_cfg.get("activation", "silu"))
    path_model = PathCorrection(
        state_dim=int(state_dim),
        hidden_dims=hidden_dims,
        activation=activation,
    )
    path_model.load_state_dict(state_dict)
    path_model.eval()
    return path_model


@torch.no_grad()
def _trajectory_stack_for_method(
    method_name: str,
    checkpoint: dict[str, Any],
    x0: torch.Tensor,
    x1: torch.Tensor,
    n_time_points: int,
) -> np.ndarray:
    mode = str(checkpoint.get("mode", method_name)).strip().lower()
    path_model = _build_path_model_from_checkpoint(checkpoint=checkpoint, state_dim=int(x0.shape[1]))
    summary = checkpoint.get("summary", {})
    mfm_alpha = 1.0
    if isinstance(summary, dict) and summary.get("mfm_alpha") is not None:
        mfm_alpha = float(summary["mfm_alpha"])

    times = torch.linspace(
        0.0,
        1.0,
        int(n_time_points),
        device=x0.device,
        dtype=x0.dtype,
    )
    chunks: list[torch.Tensor] = []
    for t in times:
        t_batch = torch.full(
            (x0.shape[0], 1),
            float(t.item()),
            device=x0.device,
            dtype=x0.dtype,
        )
        if mode in {"baseline", "metric_alpha0"}:
            xt = linear_path(t_batch, x0, x1)
        elif mode in METRIC_MODES:
            xt = mfm_mean_path(
                t=t_batch,
                x0=x0,
                x1=x1,
                geopath_net=path_model,
                alpha=float(mfm_alpha),
            )
        else:
            if path_model is None:
                raise ValueError(f"Method '{method_name}' requires path_state_dict but none was found.")
            xt = corrected_path(t_batch, x0, x1, path_model)
        chunks.append(xt)
    stack = torch.stack(chunks, dim=1).detach().cpu().numpy()
    return np.asarray(stack, dtype=np.float64)


def _pretty_label(method: str) -> str:
    key = method.strip().lower()
    labels = {
        "baseline": "Flow Matching",
        "constrained": "Constrained FM",
        "metric": "Metric FM",
        "metric_constrained_al": "Metric+Constrained (AL)",
        "metric_constrained_soft": "Metric+Constrained (soft)",
        "metric_alpha0": "Metric FM (alpha=0)",
    }
    return labels.get(key, method)


def _axis_limits(
    stacks: list[np.ndarray],
    x0: np.ndarray,
    x1: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    all_parts = [x0, x1, *[traj.reshape(-1, traj.shape[-1]) for traj in stacks]]
    merged = np.concatenate(all_parts, axis=0)
    x_low, x_high = np.quantile(merged[:, 0], [0.01, 0.99])
    y_low, y_high = np.quantile(merged[:, 1], [0.01, 0.99])
    x_span = max(float(x_high - x_low), 1e-4)
    y_span = max(float(y_high - y_low), 1e-4)
    x_pad = 0.07 * x_span
    y_pad = 0.07 * y_span
    return (float(x_low - x_pad), float(x_high + x_pad)), (float(y_low - y_pad), float(y_high + y_pad))


def _draw_panel(
    *,
    ax: Any,
    method: str,
    traj: np.ndarray,
    x0_np: np.ndarray,
    x1_np: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    line_color: str,
    x0_color: str,
    x1_color: str,
    line_alpha: float,
    line_width: float,
    endpoint_size: float,
    show_legend: bool,
    panel_label: str | None = None,
) -> None:
    for path_id in range(traj.shape[0]):
        ax.plot(
            traj[path_id, :, 0],
            traj[path_id, :, 1],
            color=line_color,
            alpha=float(line_alpha),
            linewidth=float(line_width),
            zorder=1,
        )

    scatter_x0 = ax.scatter(
        x0_np[:, 0],
        x0_np[:, 1],
        s=float(endpoint_size),
        color=x0_color,
        alpha=0.9,
        linewidths=0.0,
        zorder=2,
        label=r"$x_0$",
    )
    scatter_x1 = ax.scatter(
        x1_np[:, 0],
        x1_np[:, 1],
        s=float(endpoint_size),
        color=x1_color,
        alpha=0.95,
        linewidths=0.0,
        zorder=3,
        label=r"$x_1$",
    )

    if panel_label is not None and panel_label.strip():
        ax.text(
            0.5,
            -0.17,
            panel_label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_frame_on(False)

    if show_legend:
        ax.legend(
            handles=[scatter_x0, scatter_x1],
            labels=[r"$x_0$", r"$x_1$"],
            loc="upper right",
            frameon=True,
            framealpha=0.86,
            fontsize=8,
        )


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    methods = _parse_methods(args.methods)
    formats = _parse_formats(args.formats)

    cfg_path = _resolve_data_cfg_path(args.data_config)
    data_cfg = _load_data_cfg(cfg_path)
    if str(data_cfg.get("family", "")) != "bridge_sde":
        raise ValueError(
            f"Expected bridge_sde data config, got family='{data_cfg.get('family')}' from {cfg_path}."
        )
    if str(data_cfg.get("coupling", "")).strip().lower() != "ot_global":
        raise ValueError(
            "This exporter requires a bridge data config with coupling=ot_global "
            "to ensure a fixed shared endpoint pairing."
        )

    prepared = prepare_bridge_problem_and_targets(
        data_cfg=data_cfg,
        seed=int(args.data_seed),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    problem = prepared.problem
    if not isinstance(problem, EmpiricalCouplingProblem):
        raise TypeError(f"Expected EmpiricalCouplingProblem, got {type(problem)}")

    x0, x1, support_selection = _sample_coupled_pairs(
        problem=problem,
        n_pairs=int(args.max_paths),
        selection=str(args.pair_selection),
        seed=int(args.pair_seed),
    )

    stage_a_seed_dir = args.stage_a_seed_dir if args.stage_a_seed_dir.is_absolute() else (ROOT / args.stage_a_seed_dir).resolve()
    checkpoints: dict[str, dict[str, Any]] = {}
    for method in methods:
        ckpt_path = stage_a_seed_dir / method / "checkpoint.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found for method '{method}': {ckpt_path}")
        checkpoints[method] = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    trajectories: dict[str, np.ndarray] = {}
    for method in methods:
        trajectories[method] = _trajectory_stack_for_method(
            method_name=method,
            checkpoint=checkpoints[method],
            x0=x0,
            x1=x1,
            n_time_points=int(args.n_time_points),
        )

    x0_np = x0.detach().cpu().numpy()
    x1_np = x1.detach().cpu().numpy()
    xlim, ylim = _axis_limits(
        stacks=[trajectories[m] for m in methods],
        x0=x0_np,
        x1=x1_np,
    )

    n_panels = len(methods)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(4.2 * n_panels, 3.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_arr = np.asarray([axes]) if n_panels == 1 else np.asarray(axes)

    line_color = "#8d95c9"
    x0_color = "#ffb482"
    x1_color = "#2563eb"

    out_dir = _resolve_out_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_files: dict[str, Any]

    if bool(args.separate_panels):
        # Close unused multi-panel figure created above.
        plt.close(fig)
        output_files = {}
        for method in methods:
            single_fig, single_ax = plt.subplots(
                1,
                1,
                figsize=(4.2, 3.5),
                constrained_layout=True,
            )
            panel_text = None
            if not bool(args.hide_panel_labels):
                panel_text = _pretty_label(method)
            _draw_panel(
                ax=single_ax,
                method=method,
                traj=trajectories[method],
                x0_np=x0_np,
                x1_np=x1_np,
                xlim=xlim,
                ylim=ylim,
                line_color=line_color,
                x0_color=x0_color,
                x1_color=x1_color,
                line_alpha=float(args.line_alpha),
                line_width=float(args.line_width),
                endpoint_size=float(args.endpoint_size),
                show_legend=not bool(args.hide_legend),
                panel_label=panel_text,
            )
            if not bool(args.hide_title):
                single_fig.suptitle(_pretty_label(method), y=1.02, fontsize=12)

            safe_method = method.strip().lower().replace(" ", "_")
            output_files[method] = {}
            base_path = out_dir / f"bridge_learned_paths_{safe_method}"
            for fmt in formats:
                out_path = base_path.with_suffix(f".{fmt}")
                save_kwargs: dict[str, Any] = {"bbox_inches": "tight"}
                if fmt == "png":
                    save_kwargs["dpi"] = int(args.dpi)
                single_fig.savefig(out_path, **save_kwargs)
                output_files[method][fmt] = str(out_path)
            plt.close(single_fig)
    else:
        for idx, method in enumerate(methods):
            panel_text = None
            if not bool(args.hide_panel_labels):
                letter = chr(ord("a") + idx)
                panel_text = f"({letter}) {_pretty_label(method)}"
            _draw_panel(
                ax=axes_arr[idx],
                method=method,
                traj=trajectories[method],
                x0_np=x0_np,
                x1_np=x1_np,
                xlim=xlim,
                ylim=ylim,
                line_color=line_color,
                x0_color=x0_color,
                x1_color=x1_color,
                line_alpha=float(args.line_alpha),
                line_width=float(args.line_width),
                endpoint_size=float(args.endpoint_size),
                show_legend=(idx == n_panels - 1) and (not bool(args.hide_legend)),
                panel_label=panel_text,
            )

        if not bool(args.hide_title):
            fig.suptitle("Synthetic Bridge: Learned Conditional Paths", y=1.02, fontsize=12)

        base_path = out_dir / "bridge_learned_paths_comparison"
        output_files = {}
        for fmt in formats:
            out_path = base_path.with_suffix(f".{fmt}")
            save_kwargs: dict[str, Any] = {"bbox_inches": "tight"}
            if fmt == "png":
                save_kwargs["dpi"] = int(args.dpi)
            fig.savefig(out_path, **save_kwargs)
            output_files[fmt] = str(out_path)
        plt.close(fig)

    meta = {
        "script": "scripts/export_bridge_learned_paths_figure.py",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_config_input": str(args.data_config),
        "data_config_path": str(cfg_path),
        "data_seed": int(args.data_seed),
        "stage_a_seed_dir": str(stage_a_seed_dir),
        "methods": methods,
        "max_paths": int(args.max_paths),
        "n_time_points": int(args.n_time_points),
        "pair_selection": str(args.pair_selection),
        "pair_seed": int(args.pair_seed),
        "selected_support_indices": support_selection.tolist(),
        "cache_path": str(prepared.cache_path),
        "cache_hit": bool(prepared.cache_hit),
        "global_ot_support_size": (
            None if prepared.global_ot_support_size is None else int(prepared.global_ot_support_size)
        ),
        "global_ot_total_cost": (
            None if prepared.global_ot_total_cost is None else float(prepared.global_ot_total_cost)
        ),
        "xlim": [float(xlim[0]), float(xlim[1])],
        "ylim": [float(ylim[0]), float(ylim[1])],
        "line_width": float(args.line_width),
        "line_alpha": float(args.line_alpha),
        "endpoint_size": float(args.endpoint_size),
        "separate_panels": bool(args.separate_panels),
        "hide_legend": bool(args.hide_legend),
        "hide_title": bool(args.hide_title),
        "hide_panel_labels": bool(args.hide_panel_labels),
        "dpi": int(args.dpi),
        "formats": formats,
        "output_files": output_files,
    }
    meta_path = out_dir / "bridge_learned_paths_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[done] wrote figure artifacts to: {out_dir}")
    if bool(args.separate_panels):
        for method in methods:
            for fmt in formats:
                print(f"  - {output_files[method][fmt]}")
    else:
        for fmt in formats:
            print(f"  - {output_files[fmt]}")
    print(f"  - {meta_path}")


if __name__ == "__main__":
    main()
