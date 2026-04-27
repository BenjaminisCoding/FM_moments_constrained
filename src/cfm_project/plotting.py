from __future__ import annotations

from pathlib import Path

import matplotlib.animation as mplanimation
import matplotlib.pyplot as plt
import numpy as np
import torch

from cfm_project.data import GaussianOTProblem, sample_coupled_batch
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
    problem: GaussianOTProblem,
    path: Path,
    g_model: PathCorrection | None = None,
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

    plt.figure(figsize=(6, 6))
    for i in range(n_pairs):
        traj = []
        for t in times:
            t_batch = torch.full((1, 1), float(t.item()), device=x0.device, dtype=x0.dtype)
            x0_i = x0[i : i + 1]
            x1_i = x1[i : i + 1]
            if mode == "baseline":
                xt = linear_path(t_batch, x0_i, x1_i)
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


def _as_numpy_2d(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected samples with shape (N, 2), got {arr.shape}")
    return arr


def _as_numpy_trajectories(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if arr.ndim != 3 or arr.shape[2] != 2:
        raise ValueError(f"Expected trajectories with shape (N, T, 2), got {arr.shape}")
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
