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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a publication-ready full-width bridge synthetic dataset anatomy figure.",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="bridge_ot",
        help=(
            "Data config name under configs/data/ (for example: bridge_ot), "
            "or an explicit path to a YAML config."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used to build/load bridge target cache.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: "
            "outputs/paper_figures/bridge_dataset_anatomy/<timestamp>/"
        ),
    )
    parser.add_argument(
        "--times",
        type=str,
        default="0,0.25,0.5,0.75,1",
        help="Comma-separated normalized times in [0,1] (default: 0,0.25,0.5,0.75,1).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=90,
        help="Number of histogram bins per axis (default: 90).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=2200,
        help="Maximum number of white overlay points per panel (default: 2200).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="DPI for raster export formats (default: 220).",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=4.0,
        help="Scatter marker size in points^2 (default: 4.0).",
    )
    parser.add_argument(
        "--point-alpha",
        type=float,
        default=0.22,
        help="Scatter marker alpha in [0,1] (default: 0.22).",
    )
    parser.add_argument(
        "--title-y",
        type=float,
        default=1.08,
        help="Vertical location of figure title (default: 1.08).",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="png,pdf",
        help="Comma-separated export formats (supported: png,pdf). Default: png,pdf.",
    )
    parser.add_argument(
        "--show-bridge-center",
        action="store_true",
        help="Draw a vertical guide line at bridge_center_x.",
    )
    parser.add_argument(
        "--hide-physical-times",
        action="store_true",
        help="Hide physical-time annotation in panel titles.",
    )
    return parser.parse_args()


def _parse_times(raw: str) -> list[float]:
    pieces = [chunk.strip() for chunk in str(raw).split(",")]
    times: list[float] = []
    for piece in pieces:
        if not piece:
            continue
        t = float(piece)
        if t < 0.0 or t > 1.0:
            raise ValueError(f"All times must be in [0,1]. Got {t}.")
        times.append(round(t, 6))
    if not times:
        raise ValueError("At least one time must be provided in --times.")
    return sorted(set(times))


def _parse_formats(raw: str) -> list[str]:
    allowed = {"png", "pdf"}
    formats: list[str] = []
    for piece in str(raw).split(","):
        fmt = piece.strip().lower()
        if not fmt:
            continue
        if fmt not in allowed:
            raise ValueError(f"Unsupported format '{fmt}'. Supported: {sorted(allowed)}.")
        formats.append(fmt)
    if not formats:
        raise ValueError("At least one format must be provided in --formats.")
    return list(dict.fromkeys(formats))


def _resolve_data_cfg_path(data_config: str) -> Path:
    candidate = Path(data_config)
    if candidate.suffix in {".yaml", ".yml"}:
        if candidate.is_absolute():
            path = candidate
        else:
            path = (ROOT / candidate).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        return path

    path = ROOT / "configs" / "data" / f"{data_config}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Unknown data config '{data_config}'. Expected file at: {path}"
        )
    return path.resolve()


def _load_data_cfg(path: Path) -> dict[str, Any]:
    loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping config at {path}, got {type(loaded)}.")
    return loaded


def _resolve_out_dir(cli_out_dir: Path | None) -> Path:
    if cli_out_dir is not None:
        if cli_out_dir.is_absolute():
            return cli_out_dir
        return (ROOT / cli_out_dir).resolve()
    run_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return (ROOT / "outputs" / "paper_figures" / "bridge_dataset_anatomy" / run_tag).resolve()


def _match_available_time(available: list[float], requested: float, tol: float = 1e-6) -> float:
    best = min(available, key=lambda x: abs(x - float(requested)))
    if abs(best - float(requested)) > tol:
        raise ValueError(
            f"Requested time {requested:.6f} not available. Available: {available}"
        )
    return float(best)


def _quantile_limits(
    arrays: list[np.ndarray],
    q_low: float = 0.005,
    q_high: float = 0.995,
    pad_fraction: float = 0.04,
) -> tuple[tuple[float, float], tuple[float, float]]:
    stacked = np.concatenate(arrays, axis=0)
    x_vals = stacked[:, 0]
    y_vals = stacked[:, 1]
    x_low, x_high = np.quantile(x_vals, [q_low, q_high])
    y_low, y_high = np.quantile(y_vals, [q_low, q_high])
    x_span = float(max(x_high - x_low, 1e-4))
    y_span = float(max(y_high - y_low, 1e-4))
    x_pad = x_span * float(pad_fraction)
    y_pad = y_span * float(pad_fraction)
    return (float(x_low - x_pad), float(x_high + x_pad)), (float(y_low - y_pad), float(y_high + y_pad))


def _subsample_indices(n_samples: int, max_points: int) -> np.ndarray:
    if max_points <= 0 or n_samples <= max_points:
        return np.arange(n_samples, dtype=int)
    return np.linspace(0, n_samples - 1, num=max_points, dtype=int)


def main() -> None:
    args = _parse_args()
    if args.bins <= 0:
        raise ValueError(f"--bins must be > 0, got {args.bins}.")
    if args.max_points <= 0:
        raise ValueError(f"--max-points must be > 0, got {args.max_points}.")
    if args.point_size <= 0.0:
        raise ValueError(f"--point-size must be > 0, got {args.point_size}.")
    if args.point_alpha < 0.0 or args.point_alpha > 1.0:
        raise ValueError(f"--point-alpha must be in [0,1], got {args.point_alpha}.")

    requested_times = _parse_times(args.times)
    export_formats = _parse_formats(args.formats)
    cfg_path = _resolve_data_cfg_path(args.data_config)
    data_cfg = _load_data_cfg(cfg_path)

    if str(data_cfg.get("family", "")) != "bridge_sde":
        raise ValueError(
            f"Expected bridge_sde data config, got family='{data_cfg.get('family')}' from {cfg_path}."
        )

    # Ensure cache-aware bridge target preparation includes all requested non-endpoint times.
    data_cfg["constraint_times"] = [t for t in requested_times if 0.0 < t < 1.0]
    prepared = prepare_bridge_problem_and_targets(
        data_cfg=data_cfg,
        seed=int(args.seed),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    available_times = sorted(float(t) for t in prepared.target_samples_by_time.keys())
    selected_times = [_match_available_time(available_times, t) for t in requested_times]

    samples_by_time: dict[float, np.ndarray] = {}
    for t in selected_times:
        arr = prepared.target_samples_by_time[t].detach().cpu().numpy()
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"Expected (N, d>=2) samples at time {t}, got shape {arr.shape}.")
        samples_by_time[t] = np.asarray(arr[:, :2], dtype=np.float64)

    xlim, ylim = _quantile_limits([samples_by_time[t] for t in selected_times])
    n_cols = len(selected_times)
    fig_width = max(12.8, 2.35 * n_cols + 1.1)
    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(fig_width, 3.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if n_cols == 1:
        axes_arr = np.array([axes])
    else:
        axes_arr = np.asarray(axes)

    fig.patch.set_facecolor("white")
    bridge_center_x = float(data_cfg.get("bridge", {}).get("bridge_center_x", 0.0))
    total_time = float(data_cfg.get("bridge", {}).get("total_time", 1.0))
    point_color = "#1f77b4"

    for idx, t in enumerate(selected_times):
        ax = axes_arr[idx]
        ax.set_facecolor("white")

        points = samples_by_time[t]
        point_idx = _subsample_indices(points.shape[0], int(args.max_points))
        pts = points[point_idx]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=float(args.point_size),
            color=point_color,
            alpha=float(args.point_alpha),
            linewidths=0.0,
            rasterized=True,
        )

        if args.show_bridge_center:
            ax.axvline(
                bridge_center_x,
                linestyle="--",
                linewidth=0.9,
                color="#7dd3fc",
                alpha=0.95,
            )

        t_phys = float(t) * total_time
        if args.hide_physical_times:
            title = f"t={float(t):.2f}"
        else:
            title = f"t={float(t):.2f}\n" + rf"$t_{{\mathrm{{phys}}}}={t_phys:.2f}$"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.28, linestyle="--", linewidth=0.6, color="#c7c7c7")

    axes_arr[0].set_ylabel("y")

    figure_title = "Nonlinear Drift SDE"
    fig.suptitle(figure_title, y=float(args.title_y), fontsize=12)
    if args.show_bridge_center:
        fig.text(
            0.5,
            -0.02,
            f"Dashed line: bridge center at x={bridge_center_x:.2f}",
            ha="center",
            va="top",
            fontsize=9,
        )

    out_dir = _resolve_out_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_path = out_dir / "bridge_dataset_anatomy_fullwidth"
    output_files: dict[str, str] = {}
    for fmt in export_formats:
        out_path = base_path.with_suffix(f".{fmt}")
        save_kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if fmt in {"png"}:
            save_kwargs["dpi"] = int(args.dpi)
        fig.savefig(out_path, **save_kwargs)
        output_files[fmt] = str(out_path)
    plt.close(fig)

    meta = {
        "script": "scripts/export_bridge_dataset_figure.py",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "figure_title": figure_title,
        "data_config_input": str(args.data_config),
        "data_config_path": str(cfg_path),
        "seed": int(args.seed),
        "times_normalized": [float(t) for t in selected_times],
        "times_physical": [round(float(t) * total_time, 6) for t in selected_times],
        "total_time": total_time,
        "bridge_center_x": bridge_center_x,
        "show_bridge_center": bool(args.show_bridge_center),
        "show_physical_times": bool(not args.hide_physical_times),
        "bins": int(args.bins),
        "max_points": int(args.max_points),
        "point_size": float(args.point_size),
        "point_alpha": float(args.point_alpha),
        "dpi": int(args.dpi),
        "title_y": float(args.title_y),
        "formats": export_formats,
        "xlim": [float(xlim[0]), float(xlim[1])],
        "ylim": [float(ylim[0]), float(ylim[1])],
        "plot_style": "scatter",
        "point_color": point_color,
        "cache_path": str(prepared.cache_path),
        "cache_hit": bool(prepared.cache_hit),
        "output_files": output_files,
    }
    meta_path = out_dir / "bridge_dataset_anatomy_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[done] wrote figure artifacts to: {out_dir}")
    for fmt in export_formats:
        print(f"  - {output_files[fmt]}")
    print(f"  - {meta_path}")
    print(
        "[summary] shared_limits="
        f"x[{xlim[0]:.4f}, {xlim[1]:.4f}] y[{ylim[0]:.4f}, {ylim[1]:.4f}] "
        f"plot_style=scatter cache_hit={prepared.cache_hit}"
    )


if __name__ == "__main__":
    main()
