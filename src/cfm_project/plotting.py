from __future__ import annotations

from pathlib import Path

import matplotlib.animation as mplanimation
import matplotlib.pyplot as plt
import numpy as np
import torch

from cfm_project.data import CouplingProblem, sample_coupled_batch
from cfm_project.mfm_core import mfm_mean_path
from cfm_project.models import PathCorrection
from cfm_project.paths import corrected_path, linear_path


def save_training_curve(
    history: list[dict[str, float | int | str]],
    path: Path,
) -> None:
    if not history:
        return
    plt.figure(figsize=(8, 4))
    for stage in sorted({str(item["stage"]) for item in history}):
        xs = [int(item["global_step"]) for item in history if str(item["stage"]) == stage]
        ys = [float(item["loss"]) for item in history if str(item["stage"]) == stage]
        plt.plot(xs, ys, label=stage)
    plt.xlabel("Global Step")
    plt.ylabel("Loss")
    plt.title("Training Loss by Stage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_constraint_residual_plot(
    residual_norms: dict[float, float],
    path: Path,
) -> None:
    if not residual_norms:
        return
    times = sorted(residual_norms.keys())
    values = [residual_norms[t] for t in times]
    positions = list(range(len(times)))
    plt.figure(figsize=(6, 4))
    plt.bar(positions, values, color="#4C72B0")
    plt.xticks(positions, [f"{t:.2f}" for t in times])
    plt.xlabel("Constrained Time t")
    plt.ylabel("Residual Norm")
    plt.title("Moment Constraint Residuals")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


@torch.no_grad()
def save_path_samples_plot(
    mode: str,
    problem: CouplingProblem,
    path: Path,
    g_model: PathCorrection | None = None,
    mfm_alpha: float = 1.0,
    coupling: str = "ot",
    n_pairs: int = 32,
    n_time_points: int = 25,
    generator: torch.Generator | None = None,
) -> None:
    x0, x1, _ = sample_coupled_batch(
        problem,
        n_pairs,
        coupling=coupling,
        generator=generator,
    )
    times = torch.linspace(0.0, 1.0, n_time_points, device=x0.device, dtype=x0.dtype)
    if x0.ndim != 2 or x0.shape[1] < 2:
        raise ValueError(f"save_path_samples_plot requires samples with shape (N, d>=2), got {x0.shape}")

    plt.figure(figsize=(6, 6))
    for i in range(n_pairs):
        traj = []
        for t in times:
            t_batch = torch.full((1, 1), float(t.item()), device=x0.device, dtype=x0.dtype)
            x0_i = x0[i : i + 1]
            x1_i = x1[i : i + 1]
            if mode == "baseline":
                xt = linear_path(t_batch, x0_i, x1_i)
            elif mode == "metric_alpha0":
                xt = linear_path(t_batch, x0_i, x1_i)
            elif mode in {"metric", "metric_constrained_al", "metric_constrained_soft"}:
                xt = mfm_mean_path(
                    t=t_batch,
                    x0=x0_i,
                    x1=x1_i,
                    geopath_net=g_model,
                    alpha=float(mfm_alpha),
                )
            else:
                if g_model is None:
                    raise ValueError("g_model is required for constrained path plotting.")
                xt = corrected_path(t_batch, x0_i, x1_i, g_model)
            traj.append(xt.squeeze(0).cpu())
        traj_tensor = torch.stack(traj, dim=0).numpy()
        plt.plot(traj_tensor[:, 0], traj_tensor[:, 1], alpha=0.35, linewidth=1.0, color="#1f77b4")
    plt.scatter(x0[:, 0].cpu().numpy(), x0[:, 1].cpu().numpy(), s=12, label="x0", color="#2ca02c")
    plt.scatter(x1[:, 0].cpu().numpy(), x1[:, 1].cpu().numpy(), s=12, label="x1", color="#d62728")
    plt.title(f"Sampled Paths ({mode})")
    plt.xlabel("x[0]")
    plt.ylabel("x[1]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


@torch.no_grad()
def save_interpolant_trajectory_comparison(
    x0: torch.Tensor,
    x1: torch.Tensor,
    path: Path,
    g_model: PathCorrection | None = None,
    mode: str = "constrained",
    mfm_alpha: float = 1.0,
    n_time_points: int = 40,
    max_paths: int = 120,
) -> None:
    if x0.shape != x1.shape:
        raise ValueError(f"x0 and x1 must have same shape, got {x0.shape} and {x1.shape}")
    n_pairs = min(int(max_paths), x0.shape[0])
    if n_pairs <= 0:
        raise ValueError("Need at least one path to plot.")

    x0_plot = x0[:n_pairs]
    x1_plot = x1[:n_pairs]
    times = torch.linspace(0.0, 1.0, int(n_time_points), device=x0.device, dtype=x0.dtype)
    linear_traj: list[torch.Tensor] = []
    learned_traj: list[torch.Tensor] = []
    for t in times:
        t_batch = torch.full((n_pairs, 1), float(t.item()), device=x0.device, dtype=x0.dtype)
        linear_traj.append(linear_path(t_batch, x0_plot, x1_plot))
        mode_name = str(mode).strip().lower()
        if mode_name in {"baseline", "metric_alpha0"}:
            learned_traj.append(linear_traj[-1])
        elif mode_name in {"metric", "metric_constrained_al", "metric_constrained_soft"}:
            learned_traj.append(
                mfm_mean_path(
                    t=t_batch,
                    x0=x0_plot,
                    x1=x1_plot,
                    geopath_net=g_model,
                    alpha=float(mfm_alpha),
                )
            )
        else:
            if g_model is None:
                raise ValueError(f"g_model is required for interpolant trajectory mode '{mode}'.")
            learned_traj.append(corrected_path(t_batch, x0_plot, x1_plot, g_model))
    linear_stack = torch.stack(linear_traj, dim=1).detach().cpu().numpy()
    learned_stack = torch.stack(learned_traj, dim=1).detach().cpu().numpy()
    x0_np = x0_plot.detach().cpu().numpy()
    x1_np = x1_plot.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    panels = [
        ("Linear Interpolant Trajectories", linear_stack, "#1f77b4"),
        ("Learned Interpolant Trajectories", learned_stack, "#d62728"),
    ]
    for ax, (title, traj, color) in zip(axes, panels):
        for i in range(n_pairs):
            ax.plot(traj[i, :, 0], traj[i, :, 1], alpha=0.3, linewidth=0.8, color=color)
        ax.scatter(x0_np[:, 0], x0_np[:, 1], s=10, color="#2ca02c", alpha=0.8, label="x0")
        ax.scatter(x1_np[:, 0], x1_np[:, 1], s=10, color="#9467bd", alpha=0.8, label="x1")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.grid(alpha=0.2, linestyle="--", linewidth=0.6)
    axes[0].set_ylabel("y")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Stage-A Interpolant Particle Trajectories", y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_interpolant_marginal_comparison_grid(
    linear_samples_by_time: dict[float, torch.Tensor | np.ndarray],
    learned_samples_by_time: dict[float, torch.Tensor | np.ndarray],
    target_samples_by_time: dict[float, torch.Tensor | np.ndarray],
    path: Path,
    bins: int = 70,
    max_points: int = 2500,
) -> None:
    if not linear_samples_by_time:
        raise ValueError("linear_samples_by_time must not be empty.")
    times = sorted(linear_samples_by_time.keys())
    row_data = [
        ("Linear", linear_samples_by_time),
        ("Learned", learned_samples_by_time),
        ("True", target_samples_by_time),
    ]

    fig, axes = plt.subplots(3, len(times), figsize=(3.2 * len(times), 8.2), sharex=True, sharey=True)
    if len(times) == 1:
        axes = np.expand_dims(np.asarray(axes), axis=1)  # type: ignore[assignment]
    axes_arr = np.asarray(axes)

    for col, t in enumerate(times):
        for row, (row_label, sample_map) in enumerate(row_data):
            ax = axes_arr[row, col]
            arr = _as_numpy_2d(sample_map[float(t)])
            hist = ax.hist2d(arr[:, 0], arr[:, 1], bins=int(bins), cmap="magma")
            if max_points > 0:
                n_pts = min(int(max_points), arr.shape[0])
                idx = np.linspace(0, arr.shape[0] - 1, num=n_pts, dtype=int)
                pts = arr[idx]
                ax.scatter(pts[:, 0], pts[:, 1], s=1.5, color="white", alpha=0.16, linewidths=0)
            if row == 0:
                ax.set_title(f"t={float(t):.2f}")
            if col == 0:
                ax.set_ylabel(f"{row_label}\ny")
            if row == 2:
                ax.set_xlabel("x")
            if col == len(times) - 1:
                fig.colorbar(hist[3], ax=ax, fraction=0.04, pad=0.02)

    fig.suptitle("Interpolant Marginals: Linear vs Learned vs True", y=1.01)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_interpolant_w2_bar_plot(
    linear_empirical_w2: dict[str, float] | dict[float, float],
    learned_empirical_w2: dict[str, float] | dict[float, float],
    path: Path,
) -> None:
    if not linear_empirical_w2:
        raise ValueError("linear_empirical_w2 must not be empty.")

    linear_map = {round(float(k), 6): float(v) for k, v in linear_empirical_w2.items()}
    learned_map = {round(float(k), 6): float(v) for k, v in learned_empirical_w2.items()}
    times = sorted(linear_map.keys())
    labels = [f"{t:.2f}" for t in times]
    linear_vals = [linear_map[t] for t in times]
    learned_vals = [learned_map.get(t, np.nan) for t in times]
    if not np.isfinite(np.array(learned_vals)).all():
        raise ValueError("learned_empirical_w2 missing one or more time keys present in linear_empirical_w2.")
    x = np.arange(len(times))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, linear_vals, width=width, label="Linear", color="#1f77b4", alpha=0.85)
    ax.bar(x + width / 2, learned_vals, width=width, label="Learned", color="#d62728", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Constrained time t")
    ax.set_ylabel("Empirical W2")
    ax.set_title("Interpolant Empirical W2 by Time")
    ax.legend()
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_rollout_marginal_comparison_grid(
    generated_samples_by_time: dict[float, torch.Tensor | np.ndarray],
    target_samples_by_time: dict[float, torch.Tensor | np.ndarray],
    path: Path,
    bins: int = 70,
    max_points: int = 2500,
) -> None:
    if not generated_samples_by_time:
        raise ValueError("generated_samples_by_time must not be empty.")
    times = sorted(generated_samples_by_time.keys())
    row_data = [
        ("Generated", generated_samples_by_time),
        ("True", target_samples_by_time),
    ]
    fig, axes = plt.subplots(2, len(times), figsize=(3.2 * len(times), 5.8), sharex=True, sharey=True)
    if len(times) == 1:
        axes = np.expand_dims(np.asarray(axes), axis=1)  # type: ignore[assignment]
    axes_arr = np.asarray(axes)

    for col, t in enumerate(times):
        for row, (row_label, sample_map) in enumerate(row_data):
            ax = axes_arr[row, col]
            arr = _as_numpy_2d(sample_map[float(t)])
            hist = ax.hist2d(arr[:, 0], arr[:, 1], bins=int(bins), cmap="magma")
            if max_points > 0:
                n_pts = min(int(max_points), arr.shape[0])
                idx = np.linspace(0, arr.shape[0] - 1, num=n_pts, dtype=int)
                pts = arr[idx]
                ax.scatter(pts[:, 0], pts[:, 1], s=1.5, color="white", alpha=0.16, linewidths=0)
            if row == 0:
                ax.set_title(f"t={float(t):.2f}")
            if col == 0:
                ax.set_ylabel(f"{row_label}\ny")
            if row == 1:
                ax.set_xlabel("x")
            if col == len(times) - 1:
                fig.colorbar(hist[3], ax=ax, fraction=0.04, pad=0.02)

    fig.suptitle("Velocity Rollout Marginals: Generated vs True", y=1.01)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_rollout_empirical_w2_bar_plot(
    empirical_w2_by_time: dict[str, float] | dict[float, float],
    path: Path,
) -> None:
    if not empirical_w2_by_time:
        raise ValueError("empirical_w2_by_time must not be empty.")

    metric_map = {round(float(k), 6): float(v) for k, v in empirical_w2_by_time.items()}
    times = sorted(metric_map.keys())
    labels = [f"{t:.2f}" for t in times]
    values = [metric_map[t] for t in times]
    x = np.arange(len(times))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, values, width=0.55, color="#1f77b4", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Time t")
    ax.set_ylabel("Empirical W2")
    ax.set_title("Velocity Rollout Empirical W2 by Time")
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _as_numpy_2d(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected samples with shape (N, d>=2), got {arr.shape}")
    if arr.shape[1] > 2:
        return arr[:, :2]
    return arr


def _as_numpy_trajectories(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if arr.ndim != 3 or arr.shape[2] < 2:
        raise ValueError(f"Expected trajectories with shape (N, T, d>=2), got {arr.shape}")
    if arr.shape[2] > 2:
        return arr[:, :, :2]
    return arr


def _as_numpy_time(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"Expected time array with shape (T,), got {arr.shape}")
    return arr


def plot_bridge_snapshot_grid(
    samples_by_time: dict[float, torch.Tensor | np.ndarray],
    bins: int = 80,
    max_points: int = 2000,
    figsize: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    if not samples_by_time:
        raise ValueError("samples_by_time must not be empty.")
    times = sorted(samples_by_time.keys())
    n_cols = len(times)
    fig_width = 3.4 * n_cols if figsize is None else figsize[0]
    fig_height = 3.6 if figsize is None else figsize[1]
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_width, fig_height), sharex=True, sharey=True)
    axes_arr = np.array([axes]) if n_cols == 1 else np.asarray(axes)

    for idx, t in enumerate(times):
        ax = axes_arr[idx]
        arr = _as_numpy_2d(samples_by_time[t])
        hist = ax.hist2d(arr[:, 0], arr[:, 1], bins=int(bins), cmap="viridis")
        if max_points > 0:
            point_count = min(int(max_points), arr.shape[0])
            point_idx = np.linspace(0, arr.shape[0] - 1, num=point_count, dtype=int)
            pts = arr[point_idx]
            ax.scatter(pts[:, 0], pts[:, 1], s=2.0, c="white", alpha=0.20, linewidths=0.0)
        ax.set_title(f"t={float(t):.2f}")
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel("x")
        if idx == 0:
            ax.set_ylabel("y")
        if idx == n_cols - 1:
            fig.colorbar(hist[3], ax=ax, fraction=0.046, pad=0.04, label="count")

    fig.suptitle("Bridge SDE Distribution Snapshots", y=1.02)
    fig.tight_layout()
    return fig, axes_arr


def plot_bridge_y_spread(
    trajectory_times: torch.Tensor | np.ndarray,
    trajectories: torch.Tensor | np.ndarray,
    figsize: tuple[float, float] = (7.0, 3.6),
) -> tuple[plt.Figure, plt.Axes]:
    times = _as_numpy_time(trajectory_times)
    traj = _as_numpy_trajectories(trajectories)
    if traj.shape[1] != times.shape[0]:
        raise ValueError(
            "trajectory time dimension mismatch: "
            f"{traj.shape[1]} trajectory steps vs {times.shape[0]} time points"
        )

    y_std = traj[:, :, 1].std(axis=0)
    min_idx = int(np.argmin(y_std))
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(times, y_std, color="#1f77b4", linewidth=2.0, label="std(Y_t)")
    ax.scatter([times[min_idx]], [y_std[min_idx]], color="#d62728", s=35, zorder=3, label="minimum")
    ax.set_xlabel("time t")
    ax.set_ylabel("std(y)")
    ax.set_title("Bridge Tightness Over Time")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def save_bridge_animation(
    trajectory_times: torch.Tensor | np.ndarray,
    trajectories: torch.Tensor | np.ndarray,
    path: Path,
    max_points: int = 1200,
    frame_stride: int = 2,
    interval_ms: int = 80,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    if frame_stride <= 0:
        raise ValueError(f"frame_stride must be positive, got {frame_stride}")
    times = _as_numpy_time(trajectory_times)
    traj = _as_numpy_trajectories(trajectories)
    if traj.shape[1] != times.shape[0]:
        raise ValueError(
            "trajectory time dimension mismatch: "
            f"{traj.shape[1]} trajectory steps vs {times.shape[0]} time points"
        )

    point_count = min(int(max_points), traj.shape[0])
    point_idx = np.linspace(0, traj.shape[0] - 1, num=point_count, dtype=int)
    sampled = traj[point_idx]
    frame_ids = list(range(0, times.shape[0], int(frame_stride)))
    if frame_ids[-1] != times.shape[0] - 1:
        frame_ids.append(times.shape[0] - 1)

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    scat = ax.scatter([], [], s=5.0, c="#1f77b4", alpha=0.35, linewidths=0.0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Bridge SDE Trajectory Animation")
    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(float(np.min(sampled[:, :, 0])), float(np.max(sampled[:, :, 0])))
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(float(np.min(sampled[:, :, 1])), float(np.max(sampled[:, :, 1])))

    def _update(frame_idx: int) -> tuple:
        pts = sampled[:, frame_idx, :]
        scat.set_offsets(pts)
        ax.set_title(f"Bridge SDE Trajectory Animation (t={times[frame_idx]:.2f})")
        return (scat,)

    anim = mplanimation.FuncAnimation(
        fig,
        _update,
        frames=frame_ids,
        interval=int(interval_ms),
        blit=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer="pillow", dpi=120)
    plt.close(fig)
