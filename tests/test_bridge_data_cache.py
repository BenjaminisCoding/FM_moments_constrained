from pathlib import Path

import torch

from cfm_project.bridge_data import prepare_bridge_problem_and_targets
from cfm_project.bridge_sde import sample_bridge_sde_at_times, simulate_bridge_sde_trajectories


def _bridge_cfg(cache_dir: Path, total_time: float = 1.5) -> dict:
    return {
        "label": "bridge_test",
        "family": "bridge_sde",
        "coupling": "ot",
        "dim": 2,
        "constraint_times": [0.25, 0.5, 0.75],
        "target_mc_samples": 1024,
        "target_cache_enabled": True,
        "target_cache_dir": str(cache_dir),
        "bridge": {
            "n_steps": 80,
            "total_time": float(total_time),
            "mean0": [0.0, 0.0],
            "cov0": [[0.35, 0.0], [0.0, 0.60]],
            "vx": 2.0,
            "sigma_x": 0.15,
            "sigma_y": 0.45,
            "bridge_center_x": 1.0,
            "bridge_width": 0.35,
            "bridge_pull": 8.0,
            "bridge_diffusion_drop": 0.8,
        },
    }


def test_bridge_target_cache_hit_and_reproducibility(tmp_path: Path) -> None:
    cfg = _bridge_cfg(tmp_path / "bridge_cache")
    device = torch.device("cpu")

    first = prepare_bridge_problem_and_targets(cfg, seed=17, device=device, dtype=torch.float32)
    assert first.cache_hit is False
    assert first.cache_path.exists()

    second = prepare_bridge_problem_and_targets(cfg, seed=17, device=device, dtype=torch.float32)
    assert second.cache_hit is True
    assert second.cache_path == first.cache_path

    for t in [0.25, 0.5, 0.75]:
        assert torch.allclose(first.targets[t], second.targets[t])
        assert torch.allclose(first.target_samples_by_time[t], second.target_samples_by_time[t])


def test_bridge_target_sampler_returns_expected_shape(tmp_path: Path) -> None:
    cfg = _bridge_cfg(tmp_path / "bridge_cache")
    prepared = prepare_bridge_problem_and_targets(cfg, seed=19, device=torch.device("cpu"), dtype=torch.float32)

    generator_a = torch.Generator(device="cpu")
    generator_a.manual_seed(123)
    batch_a = prepared.target_sampler(0.5, 64, generator_a)
    generator_b = torch.Generator(device="cpu")
    generator_b.manual_seed(123)
    batch_b = prepared.target_sampler(0.5, 64, generator_b)

    assert batch_a.shape == (64, 2)
    assert torch.allclose(batch_a, batch_b)


def test_bridge_target_uses_normalized_to_physical_time_mapping(tmp_path: Path) -> None:
    cfg = _bridge_cfg(tmp_path / "bridge_cache_map", total_time=1.5)
    seed = 23
    device = torch.device("cpu")
    prepared = prepare_bridge_problem_and_targets(cfg, seed=seed, device=device, dtype=torch.float32)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    times, trajectories = simulate_bridge_sde_trajectories(
        n_samples=int(cfg["target_mc_samples"]),
        n_steps=int(cfg["bridge"]["n_steps"]),
        total_time=float(cfg["bridge"]["total_time"]),
        mean0=cfg["bridge"]["mean0"],
        cov0=cfg["bridge"]["cov0"],
        vx=float(cfg["bridge"]["vx"]),
        sigma_x=float(cfg["bridge"]["sigma_x"]),
        sigma_y=float(cfg["bridge"]["sigma_y"]),
        bridge_center_x=float(cfg["bridge"]["bridge_center_x"]),
        bridge_width=float(cfg["bridge"]["bridge_width"]),
        bridge_pull=float(cfg["bridge"]["bridge_pull"]),
        bridge_diffusion_drop=float(cfg["bridge"]["bridge_diffusion_drop"]),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    physical_snapshots = sample_bridge_sde_at_times(
        sample_times=[0.0, 0.375, 0.75, 1.125, 1.5],
        trajectory_times=times,
        trajectories=trajectories,
    )

    assert torch.allclose(prepared.target_samples_by_time[0.0], physical_snapshots[0.0])
    assert torch.allclose(prepared.target_samples_by_time[0.25], physical_snapshots[0.375])
    assert torch.allclose(prepared.target_samples_by_time[0.5], physical_snapshots[0.75])
    assert torch.allclose(prepared.target_samples_by_time[0.75], physical_snapshots[1.125])
    assert torch.allclose(prepared.target_samples_by_time[1.0], physical_snapshots[1.5])
    assert torch.allclose(prepared.problem.x1_pool, physical_snapshots[1.5])


def test_bridge_cache_key_changes_when_total_time_changes(tmp_path: Path) -> None:
    cfg_10 = _bridge_cfg(tmp_path / "bridge_cache", total_time=1.0)
    cfg_15 = _bridge_cfg(tmp_path / "bridge_cache", total_time=1.5)
    device = torch.device("cpu")

    prep_10 = prepare_bridge_problem_and_targets(cfg_10, seed=29, device=device, dtype=torch.float32)
    prep_15 = prepare_bridge_problem_and_targets(cfg_15, seed=29, device=device, dtype=torch.float32)

    assert prep_10.cache_path != prep_15.cache_path


def test_bridge_mapping_backward_compatible_for_total_time_one(tmp_path: Path) -> None:
    cfg = _bridge_cfg(tmp_path / "bridge_cache_backcompat", total_time=1.0)
    seed = 31
    device = torch.device("cpu")
    prepared = prepare_bridge_problem_and_targets(cfg, seed=seed, device=device, dtype=torch.float32)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    times, trajectories = simulate_bridge_sde_trajectories(
        n_samples=int(cfg["target_mc_samples"]),
        n_steps=int(cfg["bridge"]["n_steps"]),
        total_time=float(cfg["bridge"]["total_time"]),
        mean0=cfg["bridge"]["mean0"],
        cov0=cfg["bridge"]["cov0"],
        vx=float(cfg["bridge"]["vx"]),
        sigma_x=float(cfg["bridge"]["sigma_x"]),
        sigma_y=float(cfg["bridge"]["sigma_y"]),
        bridge_center_x=float(cfg["bridge"]["bridge_center_x"]),
        bridge_width=float(cfg["bridge"]["bridge_width"]),
        bridge_pull=float(cfg["bridge"]["bridge_pull"]),
        bridge_diffusion_drop=float(cfg["bridge"]["bridge_diffusion_drop"]),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    direct = sample_bridge_sde_at_times(
        sample_times=[0.0, 0.25, 0.5, 0.75, 1.0],
        trajectory_times=times,
        trajectories=trajectories,
    )
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        assert torch.allclose(prepared.target_samples_by_time[t], direct[t])
