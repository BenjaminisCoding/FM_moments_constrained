from pathlib import Path

import torch

from cfm_project.bridge_data import prepare_bridge_problem_and_targets
from cfm_project.bridge_sde import sample_bridge_sde_at_times, simulate_bridge_sde_trajectories


def _bridge_cfg(cache_dir: Path, total_time: float = 1.5, coupling: str = "ot") -> dict:
    return {
        "label": "bridge_test",
        "family": "bridge_sde",
        "coupling": str(coupling),
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


def test_bridge_eval_only_times_are_cached_without_constraints(tmp_path: Path) -> None:
    cfg = _bridge_cfg(tmp_path / "bridge_cache_eval_only")
    cfg["constraint_times"] = [0.5]
    cfg["interpolant_eval_times"] = [0.25, 0.5, 0.75]
    prepared = prepare_bridge_problem_and_targets(cfg, seed=21, device=torch.device("cpu"), dtype=torch.float32)

    assert sorted(prepared.targets.keys()) == [0.5]
    assert sorted(prepared.target_samples_by_time.keys()) == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert prepared.target_sampler(0.25, 16).shape == (16, 2)
    assert prepared.target_sampler(0.75, 16).shape == (16, 2)

    cached = prepare_bridge_problem_and_targets(cfg, seed=21, device=torch.device("cpu"), dtype=torch.float32)
    assert cached.cache_hit is True
    assert cached.cache_path == prepared.cache_path
    assert sorted(cached.targets.keys()) == [0.5]
    assert sorted(cached.target_samples_by_time.keys()) == [0.0, 0.25, 0.5, 0.75, 1.0]


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


def test_bridge_cache_key_changes_when_arc_amplitude_changes(tmp_path: Path) -> None:
    cfg_plain = _bridge_cfg(tmp_path / "bridge_cache_arc")
    cfg_arc = _bridge_cfg(tmp_path / "bridge_cache_arc")
    cfg_arc["bridge"]["arc_y_amplitude"] = 0.8
    device = torch.device("cpu")

    prep_plain = prepare_bridge_problem_and_targets(
        cfg_plain,
        seed=30,
        device=device,
        dtype=torch.float32,
    )
    prep_arc = prepare_bridge_problem_and_targets(
        cfg_arc,
        seed=30,
        device=device,
        dtype=torch.float32,
    )

    assert prep_plain.cache_path != prep_arc.cache_path


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


def test_bridge_ot_global_builds_and_reuses_cached_support(tmp_path: Path) -> None:
    cfg = _bridge_cfg(tmp_path / "bridge_cache_ot_global", coupling="ot_global")
    cfg["target_mc_samples"] = 96
    device = torch.device("cpu")

    first = prepare_bridge_problem_and_targets(cfg, seed=37, device=device, dtype=torch.float32)
    assert first.cache_hit is False
    assert first.problem.has_global_ot_support is True
    assert first.global_ot_support_size == int(cfg["target_mc_samples"])
    assert first.global_ot_total_cost is not None
    assert first.global_ot_solve_seconds is not None
    assert first.global_ot_cache_path is not None
    assert first.global_ot_cache_hit is False
    assert first.problem.global_ot_mass is not None
    assert abs(float(first.problem.global_ot_mass.sum().item()) - 1.0) < 1e-6

    second = prepare_bridge_problem_and_targets(cfg, seed=37, device=device, dtype=torch.float32)
    assert second.cache_hit is True
    assert second.problem.has_global_ot_support is True
    assert second.global_ot_cache_hit is True
    assert second.global_ot_cache_path == str(first.cache_path)
    assert second.problem.global_ot_src_idx is not None
    assert second.problem.global_ot_tgt_idx is not None
    assert first.problem.global_ot_src_idx is not None
    assert first.problem.global_ot_tgt_idx is not None
    assert torch.equal(second.problem.global_ot_src_idx, first.problem.global_ot_src_idx)
    assert torch.equal(second.problem.global_ot_tgt_idx, first.problem.global_ot_tgt_idx)
