from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from cfm_project.data import (
    EmpiricalCouplingProblem,
    exact_discrete_ot_indices,
)
from cfm_project.ot_utils import solve_balanced_ot_lp, solve_balanced_ot_pot


@dataclass
class MultiMarginalSegment:
    start_time: float
    end_time: float
    problem: EmpiricalCouplingProblem

    @property
    def duration(self) -> float:
        return float(self.end_time - self.start_time)


@dataclass
class PersistentMinibatchOTProposal:
    """Frozen proposal assembled from coverage-balanced minibatch OT pairs."""

    time: float
    samples: torch.Tensor
    source_indices: torch.Tensor
    target_indices: torch.Tensor
    source_coverage: torch.Tensor
    target_coverage: torch.Tensor
    batch_size: int
    passes: int
    mean_pair_squared_distance: float


def empirical_problem_with_exact_ot(
    *,
    source: torch.Tensor,
    target: torch.Tensor,
    label: str,
) -> EmpiricalCouplingProblem:
    if source.shape != target.shape:
        raise ValueError(
            "Exact adjacent-snapshot OT requires equal-size pools, got "
            f"{tuple(source.shape)} and {tuple(target.shape)}."
        )
    src_idx, tgt_idx, total_cost = exact_discrete_ot_indices(source, target)
    mass = torch.full(
        (int(src_idx.numel()),),
        fill_value=1.0 / float(src_idx.numel()),
        device=source.device,
        dtype=source.dtype,
    )
    return EmpiricalCouplingProblem(
        x0_pool=source,
        x1_pool=target,
        label=str(label),
        global_ot_src_idx=src_idx,
        global_ot_tgt_idx=tgt_idx,
        global_ot_mass=mass,
        global_ot_total_cost=float(total_cost),
    )


def _coverage_balanced_indices(
    *,
    pool_size: int,
    total_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    if int(pool_size) <= 0:
        raise ValueError(f"pool_size must be positive, got {pool_size}.")
    if int(total_size) < int(pool_size):
        raise ValueError(
            f"total_size must cover the complete pool, got {total_size} < {pool_size}."
        )
    permutation = torch.randperm(
        int(pool_size), device=device, generator=generator
    )
    extra_count = int(total_size) - int(pool_size)
    if extra_count == 0:
        return permutation
    extra = torch.randint(
        low=0,
        high=int(pool_size),
        size=(extra_count,),
        device=device,
        generator=generator,
    )
    return torch.cat([permutation, extra], dim=0)


def build_persistent_minibatch_ot_proposal(
    *,
    source: torch.Tensor,
    target: torch.Tensor,
    time: float,
    batch_size: int,
    passes: int,
    generator: torch.Generator,
) -> PersistentMinibatchOTProposal:
    """Build one finite proposal while touching every endpoint once per pass.

    Each pass independently permutes both endpoint pools, pads only to the next
    complete minibatch, solves a fresh equal-mass OT assignment inside every
    minibatch, and freezes the resulting interpolated particles.  This is a
    scalable empirical proposal, not a global endpoint OT plan.
    """

    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError(
            "Persistent minibatch OT requires endpoint matrices with the same "
            f"feature dimension, got {tuple(source.shape)} and {tuple(target.shape)}."
        )
    if source.device != target.device:
        raise ValueError("Persistent minibatch OT endpoints must share one device.")
    if int(source.shape[0]) <= 0 or int(target.shape[0]) <= 0:
        raise ValueError("Persistent minibatch OT endpoint pools must be non-empty.")
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if int(passes) <= 0:
        raise ValueError(f"passes must be positive, got {passes}.")
    if float(time) < 0.0 or float(time) > 1.0:
        raise ValueError(f"time must lie in [0,1], got {time}.")

    source_count = int(source.shape[0])
    target_count = int(target.shape[0])
    batches_per_pass = (
        max(source_count, target_count) + int(batch_size) - 1
    ) // int(batch_size)
    pairs_per_pass = batches_per_pass * int(batch_size)
    paired_source_indices: list[torch.Tensor] = []
    paired_target_indices: list[torch.Tensor] = []
    total_squared_cost = 0.0
    for _ in range(int(passes)):
        source_indices = _coverage_balanced_indices(
            pool_size=source_count,
            total_size=pairs_per_pass,
            generator=generator,
            device=source.device,
        )
        target_indices = _coverage_balanced_indices(
            pool_size=target_count,
            total_size=pairs_per_pass,
            generator=generator,
            device=target.device,
        )
        for start in range(0, pairs_per_pass, int(batch_size)):
            local_source_indices = source_indices[start : start + int(batch_size)]
            local_target_indices = target_indices[start : start + int(batch_size)]
            row_idx, col_idx, batch_cost = exact_discrete_ot_indices(
                source[local_source_indices], target[local_target_indices]
            )
            paired_source_indices.append(local_source_indices[row_idx])
            paired_target_indices.append(local_target_indices[col_idx])
            total_squared_cost += float(batch_cost)

    source_indices_all = torch.cat(paired_source_indices, dim=0)
    target_indices_all = torch.cat(paired_target_indices, dim=0)
    paired_source = source[source_indices_all]
    paired_target = target[target_indices_all]
    samples = (1.0 - float(time)) * paired_source + float(time) * paired_target
    source_coverage = torch.bincount(
        source_indices_all, minlength=source_count
    )
    target_coverage = torch.bincount(
        target_indices_all, minlength=target_count
    )
    return PersistentMinibatchOTProposal(
        time=float(time),
        samples=samples,
        source_indices=source_indices_all,
        target_indices=target_indices_all,
        source_coverage=source_coverage,
        target_coverage=target_coverage,
        batch_size=int(batch_size),
        passes=int(passes),
        mean_pair_squared_distance=(
            total_squared_cost / float(source_indices_all.numel())
        ),
    )


def _tensor_digest(tensor: torch.Tensor) -> str:
    array = np.ascontiguousarray(tensor.detach().cpu().numpy())
    hasher = hashlib.sha256()
    hasher.update(str(array.dtype).encode("utf-8"))
    hasher.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    hasher.update(array.tobytes())
    return hasher.hexdigest()


def empirical_problem_with_balanced_ot(
    *,
    source: torch.Tensor,
    target: torch.Tensor,
    label: str,
    source_weights: torch.Tensor | None = None,
    target_weights: torch.Tensor | None = None,
    support_tol: float = 1.0e-12,
    max_variables: int | None = None,
    cache_dir: str | Path | None = None,
    solver: str = "scipy_lp",
    num_itermax: int | None = None,
) -> EmpiricalCouplingProblem:
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError(
            "Balanced adjacent OT requires two point clouds with the same dimension, got "
            f"{tuple(source.shape)} and {tuple(target.shape)}."
        )
    signature = {
        "schema": "multimarginal_balanced_ot_v1",
        "source_digest": _tensor_digest(source),
        "target_digest": _tensor_digest(target),
        "source_shape": [int(value) for value in source.shape],
        "target_shape": [int(value) for value in target.shape],
        "source_weights_digest": (
            None if source_weights is None else _tensor_digest(source_weights)
        ),
        "target_weights_digest": (
            None if target_weights is None else _tensor_digest(target_weights)
        ),
        "support_tol": float(support_tol),
        "max_variables": None if max_variables is None else int(max_variables),
        "solver": str(solver),
        "num_itermax": None if num_itermax is None else int(num_itermax),
    }
    cache_path: Path | None = None
    if cache_dir is not None:
        root = Path(cache_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(
            repr(sorted(signature.items())).encode("utf-8")
        ).hexdigest()
        cache_path = root / f"{key}.pt"

    payload: dict[str, Any] | None = None
    if cache_path is not None and cache_path.exists():
        candidate = torch.load(cache_path, map_location="cpu", weights_only=False)
        if candidate.get("signature") != signature:
            raise RuntimeError(f"Adjacent OT cache signature mismatch at {cache_path}.")
        payload = candidate
    if payload is None:
        normalized_solver = str(solver).strip().lower()
        if normalized_solver == "scipy_lp":
            plan = solve_balanced_ot_lp(
                x=source.detach().cpu(),
                y=target.detach().cpu(),
                src_weights=(
                    None if source_weights is None else source_weights.detach().cpu()
                ),
                tgt_weights=(
                    None if target_weights is None else target_weights.detach().cpu()
                ),
                support_tol=float(support_tol),
                max_variables=max_variables,
            )
        elif normalized_solver == "pot_emd":
            plan = solve_balanced_ot_pot(
                x=source.detach().cpu(),
                y=target.detach().cpu(),
                src_weights=(
                    None if source_weights is None else source_weights.detach().cpu()
                ),
                tgt_weights=(
                    None if target_weights is None else target_weights.detach().cpu()
                ),
                support_tol=float(support_tol),
                num_itermax=num_itermax,
                max_variables=max_variables,
            )
        else:
            raise ValueError(
                f"Unsupported balanced OT solver '{solver}'. Expected scipy_lp or pot_emd."
            )
        payload = {
            "signature": signature,
            "src_idx": torch.as_tensor(plan.src_idx, dtype=torch.long),
            "tgt_idx": torch.as_tensor(plan.tgt_idx, dtype=torch.long),
            "mass": torch.as_tensor(plan.mass, dtype=torch.float64),
            "total_cost": float(plan.total_cost),
        }
        if cache_path is not None:
            torch.save(payload, cache_path)

    mass = torch.as_tensor(payload["mass"], device=source.device, dtype=source.dtype)
    mass = mass / torch.clamp(mass.sum(), min=torch.finfo(mass.dtype).eps)
    return EmpiricalCouplingProblem(
        x0_pool=source,
        x1_pool=target,
        label=str(label),
        global_ot_src_idx=torch.as_tensor(
            payload["src_idx"], device=source.device, dtype=torch.long
        ),
        global_ot_tgt_idx=torch.as_tensor(
            payload["tgt_idx"], device=source.device, dtype=torch.long
        ),
        global_ot_mass=mass,
        global_ot_total_cost=float(payload["total_cost"]),
    )


def build_adjacent_ot_segments(
    *,
    snapshot_times: Sequence[float],
    snapshot_pools: Sequence[torch.Tensor],
    snapshot_weights: Sequence[torch.Tensor | None] | None = None,
    label_prefix: str,
    ot_method: str = "assignment",
    balanced_ot_support_tol: float = 1.0e-12,
    balanced_ot_max_variables: int | None = None,
    balanced_ot_cache_dir: str | Path | None = None,
    balanced_ot_solver: str = "scipy_lp",
    balanced_ot_num_itermax: int | None = None,
) -> list[MultiMarginalSegment]:
    if len(snapshot_times) != len(snapshot_pools):
        raise ValueError("snapshot_times and snapshot_pools must have equal length.")
    if snapshot_weights is None:
        local_snapshot_weights: list[torch.Tensor | None] = [
            None for _ in snapshot_pools
        ]
    else:
        if len(snapshot_weights) != len(snapshot_pools):
            raise ValueError(
                "snapshot_weights and snapshot_pools must have equal length."
            )
        local_snapshot_weights = list(snapshot_weights)
    if len(snapshot_times) < 2:
        raise ValueError("At least two snapshots are required.")
    times = [float(t) for t in snapshot_times]
    if any(right <= left for left, right in zip(times[:-1], times[1:])):
        raise ValueError(f"snapshot_times must be strictly increasing, got {times}.")
    segments: list[MultiMarginalSegment] = []
    normalized_ot_method = str(ot_method).strip().lower()
    if normalized_ot_method not in {"assignment", "balanced_lp"}:
        raise ValueError(
            f"Unsupported adjacent OT method '{ot_method}'. Expected assignment or balanced_lp."
        )
    if normalized_ot_method == "assignment" and any(
        weights is not None for weights in local_snapshot_weights
    ):
        raise ValueError(
            "Nonuniform snapshot_weights require ot_method='balanced_lp'."
        )
    for index, (start, end) in enumerate(zip(times[:-1], times[1:])):
        if normalized_ot_method == "assignment":
            problem = empirical_problem_with_exact_ot(
                source=snapshot_pools[index],
                target=snapshot_pools[index + 1],
                label=f"{label_prefix}_segment_{index}",
            )
        else:
            problem = empirical_problem_with_balanced_ot(
                source=snapshot_pools[index],
                target=snapshot_pools[index + 1],
                label=f"{label_prefix}_segment_{index}",
                source_weights=local_snapshot_weights[index],
                target_weights=local_snapshot_weights[index + 1],
                support_tol=float(balanced_ot_support_tol),
                max_variables=balanced_ot_max_variables,
                cache_dir=balanced_ot_cache_dir,
                solver=balanced_ot_solver,
                num_itermax=balanced_ot_num_itermax,
            )
        segments.append(
            MultiMarginalSegment(
                start_time=float(start),
                end_time=float(end),
                problem=problem,
            )
        )
    return segments


def local_to_global_time_and_velocity(
    *,
    local_time: torch.Tensor,
    local_velocity: torch.Tensor,
    start_time: float,
    end_time: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    duration = float(end_time - start_time)
    if duration <= 0.0:
        raise ValueError(
            f"Segment duration must be positive, got start={start_time}, end={end_time}."
        )
    global_time = float(start_time) + duration * local_time
    global_velocity = local_velocity / duration
    return global_time, global_velocity
