import matplotlib
import pytest
import torch
from pathlib import Path
from matplotlib import pyplot as plt

from cfm_project.bridge_sde import (
    default_bridge_preview_parameters,
    sample_bridge_sde_at_times,
    simulate_bridge_sde_trajectories,
)
from cfm_project.plotting import (
    plot_bridge_snapshot_grid,
    plot_bridge_y_spread,
    save_rollout_empirical_w2_bar_plot,
    save_rollout_marginal_comparison_grid,
    save_interpolant_marginal_comparison_grid,
    save_interpolant_trajectory_comparison,
    save_interpolant_w2_bar_plot,
    save_bridge_animation,
)
from cfm_project.models import PathCorrection
from cfm_project.paths import corrected_path, linear_path

matplotlib.use("Agg")


def _simulate_bridge(
    seed: int,
    n_samples: int = 2048,
    n_steps: int = 120,
) -> tuple[torch.Tensor, torch.Tensor]:
    params = default_bridge_preview_parameters()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return simulate_bridge_sde_trajectories(
        n_samples=n_samples,
        n_steps=n_steps,
        total_time=float(params["total_time"]),
        mean0=params["mean0"],  # type: ignore[arg-type]
        cov0=params["cov0"],  # type: ignore[arg-type]
        vx=float(params["vx"]),
        sigma_x=float(params["sigma_x"]),
        sigma_y=float(params["sigma_y"]),
        bridge_center_x=float(params["bridge_center_x"]),
        bridge_width=float(params["bridge_width"]),
        bridge_pull=float(params["bridge_pull"]),
        bridge_diffusion_drop=float(params["bridge_diffusion_drop"]),
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    )


def test_bridge_sde_reproducibility() -> None:
    times_a, traj_a = _simulate_bridge(seed=11)
    times_b, traj_b = _simulate_bridge(seed=11)

    assert torch.allclose(times_a, times_b)
    assert torch.allclose(traj_a, traj_b)


def test_bridge_sde_shape_and_finite_values() -> None:
    n_samples = 512
    n_steps = 80
    times, trajectories = _simulate_bridge(seed=12, n_samples=n_samples, n_steps=n_steps)

    assert times.shape == (n_steps + 1,)
    assert trajectories.shape == (n_samples, n_steps + 1, 2)
    assert torch.isfinite(trajectories).all()

    snapshots = sample_bridge_sde_at_times([0.0, 0.5, 1.0], times, trajectories)
    assert set(snapshots.keys()) == {0.0, 0.5, 1.0}
    assert snapshots[0.0].shape == (n_samples, 2)


def test_bridge_sde_middle_is_narrower_in_y() -> None:
    times, trajectories = _simulate_bridge(seed=13, n_samples=5000, n_steps=200)
    std_y = torch.std(trajectories[:, :, 1], dim=0)

    idx_0 = int(torch.argmin(torch.abs(times - 0.0)).item())
    idx_mid = int(torch.argmin(torch.abs(times - 0.5)).item())
    idx_1 = int(torch.argmin(torch.abs(times - 1.0)).item())

    assert std_y[idx_mid] < std_y[idx_0]
    assert std_y[idx_mid] < std_y[idx_1]


def test_bridge_plot_functions_write_artifacts(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    times, trajectories = _simulate_bridge(seed=14, n_samples=320, n_steps=50)
    snapshots = sample_bridge_sde_at_times([0.0, 0.25, 0.5, 0.75, 1.0], times, trajectories)

    fig_snap, _ = plot_bridge_snapshot_grid(snapshots, bins=45, max_points=600)
    snap_path = tmp_path / "snapshot_grid.png"
    fig_snap.savefig(snap_path, dpi=120)
    plt.close(fig_snap)

    fig_spread, _ = plot_bridge_y_spread(times, trajectories)
    spread_path = tmp_path / "y_spread.png"
    fig_spread.savefig(spread_path, dpi=120)
    plt.close(fig_spread)

    gif_path = tmp_path / "bridge_animation.gif"
    save_bridge_animation(
        trajectory_times=times,
        trajectories=trajectories,
        path=gif_path,
        max_points=250,
        frame_stride=3,
        interval_ms=60,
    )

    assert snap_path.exists() and snap_path.stat().st_size > 0
    assert spread_path.exists() and spread_path.stat().st_size > 0
    assert gif_path.exists() and gif_path.stat().st_size > 0


def test_interpolant_plot_functions_write_artifacts(tmp_path: Path) -> None:
    torch.manual_seed(21)
    x0 = torch.randn(96, 2)
    x1 = torch.randn(96, 2)
    g_model = PathCorrection(state_dim=2, hidden_dims=(16, 16))

    traj_path = tmp_path / "interpolant_traj.png"
    save_interpolant_trajectory_comparison(
        x0=x0,
        x1=x1,
        path=traj_path,
        g_model=g_model,
        n_time_points=24,
        max_paths=60,
    )

    times = [0.25, 0.5, 0.75]
    linear_by_time: dict[float, torch.Tensor] = {}
    learned_by_time: dict[float, torch.Tensor] = {}
    target_by_time: dict[float, torch.Tensor] = {}
    for t in times:
        t_batch = torch.full((x0.shape[0], 1), t)
        linear_t = linear_path(t_batch, x0, x1)
        learned_t = corrected_path(t_batch, x0, x1, g_model)
        linear_by_time[t] = linear_t
        learned_by_time[t] = learned_t
        target_by_time[t] = linear_t + 0.05 * torch.randn_like(linear_t)

    grid_path = tmp_path / "interpolant_grid.png"
    save_interpolant_marginal_comparison_grid(
        linear_samples_by_time=linear_by_time,
        learned_samples_by_time=learned_by_time,
        target_samples_by_time=target_by_time,
        path=grid_path,
        bins=32,
        max_points=400,
    )

    w2_path = tmp_path / "interpolant_w2.png"
    save_interpolant_w2_bar_plot(
        linear_empirical_w2={"0.25": 0.8, "0.50": 0.9, "0.75": 1.1},
        learned_empirical_w2={"0.25": 0.6, "0.50": 0.7, "0.75": 0.8},
        path=w2_path,
    )

    assert traj_path.exists() and traj_path.stat().st_size > 0
    assert grid_path.exists() and grid_path.stat().st_size > 0
    assert w2_path.exists() and w2_path.stat().st_size > 0


def test_bridge_rollout_plot_functions_write_artifacts(tmp_path: Path) -> None:
    torch.manual_seed(41)
    times = [0.25, 0.5, 0.75, 1.0]
    generated_by_time: dict[float, torch.Tensor] = {}
    target_by_time: dict[float, torch.Tensor] = {}
    for t in times:
        generated = torch.randn(512, 2) + torch.tensor([2.0 * t, 0.0])
        target = generated + 0.1 * torch.randn_like(generated)
        generated_by_time[float(t)] = generated
        target_by_time[float(t)] = target

    grid_path = tmp_path / "rollout_grid.png"
    save_rollout_marginal_comparison_grid(
        generated_samples_by_time=generated_by_time,
        target_samples_by_time=target_by_time,
        path=grid_path,
        bins=32,
        max_points=400,
    )

    w2_path = tmp_path / "rollout_w2.png"
    save_rollout_empirical_w2_bar_plot(
        empirical_w2_by_time={"0.25": 0.4, "0.50": 0.6, "0.75": 0.7, "1.00": 0.9},
        path=w2_path,
    )

    assert grid_path.exists() and grid_path.stat().st_size > 0
    assert w2_path.exists() and w2_path.stat().st_size > 0
