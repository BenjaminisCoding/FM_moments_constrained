from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import torch

from cfm_project.bridge_sde import sample_bridge_sde_at_times, simulate_bridge_sde_trajectories
from cfm_project.data import (
    EmpiricalCouplingProblem,
    exact_discrete_ot_indices,
    moment_feature_vector_from_samples,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class BridgePreparedData:
    problem: EmpiricalCouplingProblem
    targets: dict[float, torch.Tensor]
    target_samples_by_time: dict[float, torch.Tensor]
    target_sampler: Callable[[float, int, torch.Generator | None], torch.Tensor]
    cache_path: Path
    cache_hit: bool
    global_ot_cache_path: str | None
    global_ot_cache_hit: bool
    global_ot_support_size: int | None
    global_ot_total_cost: float | None
    global_ot_solve_seconds: float | None


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
    coupling = str(data_cfg.get("coupling", "ot")).strip().lower()
    key_payload = {
        "schema_version": 2,
        "seed": int(seed),
        "coupling": coupling,
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


def _serialize_global_ot_support(
    src_idx: torch.Tensor | None,
    tgt_idx: torch.Tensor | None,
    mass: torch.Tensor | None,
    total_cost: float | None,
    solve_seconds: float | None,
) -> dict[str, Any] | None:
    if src_idx is None or tgt_idx is None or mass is None:
        return None
    return {
        "src_idx": src_idx.detach().cpu().to(dtype=torch.long).clone(),
        "tgt_idx": tgt_idx.detach().cpu().to(dtype=torch.long).clone(),
        "mass": mass.detach().cpu().to(dtype=torch.float32).clone(),
        "total_cost": (None if total_cost is None else float(total_cost)),
        "solve_seconds": (None if solve_seconds is None else float(solve_seconds)),
    }


def _deserialize_global_ot_support(
    payload: Mapping[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float | None, float | None]:
    src_idx = payload.get("src_idx")
    tgt_idx = payload.get("tgt_idx")
    mass = payload.get("mass")
    if not isinstance(src_idx, torch.Tensor) or not isinstance(tgt_idx, torch.Tensor) or not isinstance(mass, torch.Tensor):
        raise ValueError("Cached global OT payload is missing tensor keys: src_idx, tgt_idx, mass.")
    src_idx_tensor = src_idx.to(device=device, dtype=torch.long)
    tgt_idx_tensor = tgt_idx.to(device=device, dtype=torch.long)
    mass_tensor = mass.to(device=device, dtype=dtype)
    if src_idx_tensor.ndim != 1 or tgt_idx_tensor.ndim != 1 or mass_tensor.ndim != 1:
        raise ValueError("Cached global OT support tensors must be 1D.")
    if not (src_idx_tensor.shape[0] == tgt_idx_tensor.shape[0] == mass_tensor.shape[0]):
        raise ValueError(
            "Cached global OT support tensors must have equal length, got "
            f"{tuple(src_idx_tensor.shape)}, {tuple(tgt_idx_tensor.shape)}, {tuple(mass_tensor.shape)}."
        )
    if int(mass_tensor.numel()) <= 0:
        raise ValueError("Cached global OT support is empty.")
    mass_sum = float(mass_tensor.sum().item())
    if mass_sum <= 0.0:
        raise ValueError("Cached global OT support has non-positive total mass.")
    mass_tensor = mass_tensor / torch.clamp(
        mass_tensor.sum(),
        min=torch.finfo(mass_tensor.dtype).eps,
    )
    total_cost_raw = payload.get("total_cost")
    solve_seconds_raw = payload.get("solve_seconds")
    total_cost = None if total_cost_raw is None else float(total_cost_raw)
    solve_seconds = None if solve_seconds_raw is None else float(solve_seconds_raw)
    return src_idx_tensor, tgt_idx_tensor, mass_tensor, total_cost, solve_seconds


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
    coupling = str(data_cfg.get("coupling", "ot")).strip().lower()
    if coupling not in {"ot", "random", "ot_global"}:
        raise ValueError(
            f"Unsupported bridge coupling '{coupling}'. Expected one of: ot, random, ot_global."
        )
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
    global_ot_src_idx: torch.Tensor | None = None
    global_ot_tgt_idx: torch.Tensor | None = None
    global_ot_mass: torch.Tensor | None = None
    global_ot_total_cost: float | None = None
    global_ot_solve_seconds: float | None = None

    if cache_hit:
        payload = torch.load(cache_path, map_location="cpu")
        payload_key = payload.get("key_payload")
        if payload_key != key_payload:
            raise RuntimeError(
                "Bridge target cache key payload mismatch for existing cache file. "
                f"Delete {cache_path} to recompute."
            )
        samples_by_time = _tensor_dict_from_serialized(payload, device=device, dtype=dtype)
        if coupling == "ot_global":
            raw_ot_payload = payload.get("global_ot_support")
            if raw_ot_payload is None or not isinstance(raw_ot_payload, dict):
                raise RuntimeError(
                    "Bridge cache hit requested coupling='ot_global' but cache payload "
                    f"does not contain global OT support: {cache_path}"
                )
            (
                global_ot_src_idx,
                global_ot_tgt_idx,
                global_ot_mass,
                global_ot_total_cost,
                global_ot_solve_seconds,
            ) = _deserialize_global_ot_support(raw_ot_payload, device=device, dtype=dtype)
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
        if coupling == "ot_global":
            x0_pool = samples_by_time[0.0]
            x1_pool = samples_by_time[1.0]
            if x0_pool.shape != x1_pool.shape:
                raise ValueError(
                    "coupling='ot_global' requires equal-size endpoint pools, got "
                    f"{tuple(x0_pool.shape)} and {tuple(x1_pool.shape)}."
                )
            start = time.perf_counter()
            src_idx, tgt_idx, total_cost = exact_discrete_ot_indices(x0=x0_pool, x1=x1_pool)
            global_ot_solve_seconds = float(time.perf_counter() - start)
            n_pairs = int(src_idx.numel())
            if n_pairs <= 0:
                raise RuntimeError("Global OT support is empty for bridge coupling='ot_global'.")
            mass = torch.full(
                (n_pairs,),
                fill_value=1.0 / float(n_pairs),
                device=device,
                dtype=dtype,
            )
            global_ot_src_idx = src_idx.to(device=device, dtype=torch.long)
            global_ot_tgt_idx = tgt_idx.to(device=device, dtype=torch.long)
            global_ot_mass = mass
            global_ot_total_cost = float(total_cost)
        if cache_enabled:
            cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "key_payload": key_payload,
                    "samples_by_time": _serialize_tensor_dict(samples_by_time),
                    "global_ot_support": _serialize_global_ot_support(
                        src_idx=global_ot_src_idx,
                        tgt_idx=global_ot_tgt_idx,
                        mass=global_ot_mass,
                        total_cost=global_ot_total_cost,
                        solve_seconds=global_ot_solve_seconds,
                    ),
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
        global_ot_src_idx=global_ot_src_idx,
        global_ot_tgt_idx=global_ot_tgt_idx,
        global_ot_mass=global_ot_mass,
        global_ot_total_cost=global_ot_total_cost,
    )
    return BridgePreparedData(
        problem=problem,
        targets=targets,
        target_samples_by_time={float(t): samples_by_time[round(float(t), 6)] for t in available_times},
        target_sampler=target_sampler,
        cache_path=cache_path,
        cache_hit=cache_hit,
        global_ot_cache_path=(None if coupling != "ot_global" else str(cache_path)),
        global_ot_cache_hit=(bool(cache_hit) if coupling == "ot_global" else False),
        global_ot_support_size=(None if global_ot_mass is None else int(global_ot_mass.numel())),
        global_ot_total_cost=global_ot_total_cost,
        global_ot_solve_seconds=global_ot_solve_seconds,
    )
