from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from cfm_project.bridge_sde import sample_bridge_sde_at_times, simulate_bridge_sde_trajectories
from cfm_project.data import EmpiricalCouplingProblem, moment_feature_vector_from_samples

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class BridgePreparedData:
    problem: EmpiricalCouplingProblem
    targets: dict[float, torch.Tensor]
    target_samples_by_time: dict[float, torch.Tensor]
    target_sampler: Callable[[float, int, torch.Generator | None], torch.Tensor]
    cache_path: Path
    cache_hit: bool


def _normalize_times(times: Sequence[float]) -> list[float]:
    normalized = sorted({round(float(t), 6) for t in times})
    if not normalized:
        raise ValueError("constraint_times must not be empty.")
    for t in normalized:
        if t < 0.0 or t > 1.0:
            raise ValueError(f"constraint time must be in [0, 1], got {t}")
    return normalized


def _normalized_to_physical_time(t_normalized: float, total_time: float) -> float:
    return round(float(t_normalized) * float(total_time), 6)


def _cache_dir_from_cfg(data_cfg: Mapping[str, Any]) -> Path:
    cache_dir_raw = str(data_cfg.get("target_cache_dir", ".cache/bridge_targets"))
    cache_dir = Path(cache_dir_raw)
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir
    return cache_dir


def _bridge_cache_key(
    data_cfg: Mapping[str, Any],
    seed: int,
    constraint_times: Sequence[float],
    target_mc_samples: int,
) -> tuple[str, dict[str, Any]]:
    bridge_cfg = data_cfg.get("bridge", {})
    key_payload = {
        "schema_version": 1,
        "seed": int(seed),
        "constraint_times": [float(t) for t in _normalize_times(constraint_times)],
        "target_mc_samples": int(target_mc_samples),
        "bridge": {
            "n_steps": int(bridge_cfg["n_steps"]),
            "total_time": float(bridge_cfg["total_time"]),
            "mean0": bridge_cfg["mean0"],
            "cov0": bridge_cfg["cov0"],
            "vx": float(bridge_cfg["vx"]),
            "sigma_x": float(bridge_cfg["sigma_x"]),
            "sigma_y": float(bridge_cfg["sigma_y"]),
            "bridge_center_x": float(bridge_cfg["bridge_center_x"]),
            "bridge_width": float(bridge_cfg["bridge_width"]),
            "bridge_pull": float(bridge_cfg["bridge_pull"]),
            "bridge_diffusion_drop": float(bridge_cfg["bridge_diffusion_drop"]),
        },
    }
    serialized = json.dumps(key_payload, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(serialized).hexdigest()[:16]
    return digest, key_payload


def _tensor_dict_from_serialized(
    payload: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[float, torch.Tensor]:
    raw = payload.get("samples_by_time")
    if not isinstance(raw, dict):
        raise ValueError("Cache payload missing 'samples_by_time' dictionary.")
    out: dict[float, torch.Tensor] = {}
    for key, value in raw.items():
        t = round(float(key), 6)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Expected tensor for cache time '{key}', got {type(value)}")
        out[t] = value.to(device=device, dtype=dtype)
    return out


def _serialize_tensor_dict(samples_by_time: Mapping[float, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        f"{float(t):.6f}": tensor.detach().cpu().to(dtype=torch.float32).clone()
        for t, tensor in samples_by_time.items()
    }


def _select_time_key(available: Sequence[float], t: float, tol: float = 1e-4) -> float:
    candidates = [float(v) for v in available]
    best = min(candidates, key=lambda v: abs(v - float(t)))
    if abs(best - float(t)) > tol:
        raise ValueError(f"Requested time {t} not available in target snapshots: {sorted(candidates)}")
    return round(best, 6)


def _sample_from_pool(
    pool: torch.Tensor,
    n_samples: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    idx = torch.randint(
        low=0,
        high=pool.shape[0],
        size=(n_samples,),
        device=pool.device,
        generator=generator,
    )
    return pool[idx]


def prepare_bridge_problem_and_targets(
    data_cfg: Mapping[str, Any],
    seed: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> BridgePreparedData:
    bridge_cfg = data_cfg.get("bridge", {})
    constraint_times = _normalize_times(data_cfg.get("constraint_times", []))
    total_time = float(bridge_cfg["total_time"])
    target_mc_samples = int(data_cfg.get("target_mc_samples", 200000))
    cache_enabled = bool(data_cfg.get("target_cache_enabled", True))
    if target_mc_samples <= 0:
        raise ValueError(f"target_mc_samples must be positive, got {target_mc_samples}")
    if total_time <= 0.0:
        raise ValueError(f"bridge.total_time must be positive, got {total_time}")

    cache_dir = _cache_dir_from_cfg(data_cfg)
    digest, key_payload = _bridge_cache_key(
        data_cfg=data_cfg,
        seed=seed,
        constraint_times=constraint_times,
        target_mc_samples=target_mc_samples,
    )
    cache_path = cache_dir / f"{digest}.pt"
    cache_hit = cache_enabled and cache_path.exists()

    if cache_hit:
        payload = torch.load(cache_path, map_location="cpu")
        samples_by_time = _tensor_dict_from_serialized(payload, device=device, dtype=dtype)
    else:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        sim_times, trajectories = simulate_bridge_sde_trajectories(
            n_samples=target_mc_samples,
            n_steps=int(bridge_cfg["n_steps"]),
            total_time=total_time,
            mean0=bridge_cfg["mean0"],
            cov0=bridge_cfg["cov0"],
            vx=float(bridge_cfg["vx"]),
            sigma_x=float(bridge_cfg["sigma_x"]),
            sigma_y=float(bridge_cfg["sigma_y"]),
            bridge_center_x=float(bridge_cfg["bridge_center_x"]),
            bridge_width=float(bridge_cfg["bridge_width"]),
            bridge_pull=float(bridge_cfg["bridge_pull"]),
            bridge_diffusion_drop=float(bridge_cfg["bridge_diffusion_drop"]),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        normalized_snapshot_times = sorted({0.0, 1.0, *constraint_times})
        physical_snapshot_times = sorted(
            {
                _normalized_to_physical_time(t_normalized=t_norm, total_time=total_time)
                for t_norm in normalized_snapshot_times
            }
        )
        samples_by_physical_time = sample_bridge_sde_at_times(
            sample_times=physical_snapshot_times,
            trajectory_times=sim_times,
            trajectories=trajectories,
        )
        samples_by_time: dict[float, torch.Tensor] = {}
        available_physical_times = sorted(samples_by_physical_time.keys())
        for t_norm in normalized_snapshot_times:
            t_phys = _normalized_to_physical_time(t_normalized=t_norm, total_time=total_time)
            matched_phys = _select_time_key(available_physical_times, t=t_phys, tol=1e-5)
            samples_by_time[round(float(t_norm), 6)] = samples_by_physical_time[matched_phys].to(
                device=device,
                dtype=dtype,
            )
        if cache_enabled:
            cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "key_payload": key_payload,
                    "samples_by_time": _serialize_tensor_dict(samples_by_time),
                },
                cache_path,
            )

    targets = {
        float(t): moment_feature_vector_from_samples(samples_by_time[round(float(t), 6)])
        for t in constraint_times
    }
    available_times = sorted(samples_by_time.keys())

    def target_sampler(
        t: float,
        n_samples: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        time_key = _select_time_key(available_times, t)
        pool = samples_by_time[time_key]
        return _sample_from_pool(pool, n_samples=n_samples, generator=generator)

    problem = EmpiricalCouplingProblem(
        x0_pool=samples_by_time[0.0],
        x1_pool=samples_by_time[1.0],
        label=str(data_cfg.get("label", "bridge_sde")),
    )
    return BridgePreparedData(
        problem=problem,
        targets=targets,
        target_samples_by_time={float(t): samples_by_time[round(float(t), 6)] for t in available_times},
        target_sampler=target_sampler,
        cache_path=cache_path,
        cache_hit=cache_hit,
    )
