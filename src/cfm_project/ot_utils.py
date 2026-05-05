from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import sparse
from scipy.optimize import linprog


def _to_numpy_points(x: torch.Tensor) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected point cloud with shape (N, d), got {tuple(x.shape)}")
    return np.asarray(x.detach().cpu(), dtype=np.float64)


def _normalize_weights(
    weights: torch.Tensor | np.ndarray | None,
    n: int,
    label: str,
) -> np.ndarray:
    if n <= 0:
        raise ValueError(f"{label} cardinality must be positive, got {n}")
    if weights is None:
        return np.full((n,), 1.0 / float(n), dtype=np.float64)
    arr = np.asarray(weights.detach().cpu() if isinstance(weights, torch.Tensor) else weights, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError(
            f"{label} weights must be shape ({n},), got {arr.shape}"
        )
    if np.any(arr < 0.0):
        raise ValueError(f"{label} weights must be non-negative.")
    total = float(arr.sum())
    if total <= 0.0:
        raise ValueError(f"{label} weights must sum to a positive value.")
    return arr / total


def pairwise_squared_euclidean_cost(x: torch.Tensor, y: torch.Tensor) -> np.ndarray:
    x_np = _to_numpy_points(x)
    y_np = _to_numpy_points(y)
    if x_np.shape[1] != y_np.shape[1]:
        raise ValueError(
            f"Point dimensions mismatch for OT cost: {x_np.shape} vs {y_np.shape}"
        )
    x_norm = np.sum(x_np * x_np, axis=1, keepdims=True)
    y_norm = np.sum(y_np * y_np, axis=1, keepdims=True).T
    cost = x_norm + y_norm - 2.0 * (x_np @ y_np.T)
    np.maximum(cost, 0.0, out=cost)
    return cost


@dataclass
class SparseOTPlan:
    src_idx: np.ndarray
    tgt_idx: np.ndarray
    mass: np.ndarray
    total_cost: float

    @property
    def support_size(self) -> int:
        return int(self.mass.shape[0])


def _build_transport_constraints(n_src: int, n_tgt: int) -> sparse.csr_matrix:
    num_vars = int(n_src) * int(n_tgt)
    var_idx = np.arange(num_vars, dtype=np.int64)

    src_rows = np.repeat(np.arange(n_src, dtype=np.int64), n_tgt)
    tgt_rows = n_src + (var_idx % n_tgt)

    rows = np.concatenate([src_rows, tgt_rows], axis=0)
    cols = np.concatenate([var_idx, var_idx], axis=0)
    data = np.ones(rows.shape[0], dtype=np.float64)
    return sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(n_src + n_tgt, num_vars),
    ).tocsr()


def solve_balanced_ot_lp_from_cost(
    cost_matrix: np.ndarray,
    src_weights: torch.Tensor | np.ndarray | None = None,
    tgt_weights: torch.Tensor | np.ndarray | None = None,
    support_tol: float = 1e-12,
    max_variables: int | None = None,
) -> SparseOTPlan:
    if cost_matrix.ndim != 2:
        raise ValueError(f"Expected cost matrix with shape (n_src, n_tgt), got {cost_matrix.shape}")
    n_src, n_tgt = int(cost_matrix.shape[0]), int(cost_matrix.shape[1])
    if n_src <= 0 or n_tgt <= 0:
        raise ValueError(f"OT requires non-empty supports, got shape {cost_matrix.shape}")
    num_vars = n_src * n_tgt
    if max_variables is not None and num_vars > int(max_variables):
        raise ValueError(
            f"Exact LP OT skipped because n_src*n_tgt={num_vars} exceeds "
            f"max_variables={int(max_variables)}."
        )

    src_mass = _normalize_weights(src_weights, n_src, "source")
    tgt_mass = _normalize_weights(tgt_weights, n_tgt, "target")
    if not np.isclose(src_mass.sum(), tgt_mass.sum(), atol=1e-9):
        raise ValueError(
            "Balanced OT requires equal total mass between source and target. "
            f"Got source sum={src_mass.sum():.12f}, target sum={tgt_mass.sum():.12f}."
        )

    c = np.asarray(cost_matrix, dtype=np.float64).reshape(-1)
    a_eq = _build_transport_constraints(n_src=n_src, n_tgt=n_tgt)
    b_eq = np.concatenate([src_mass, tgt_mass], axis=0)
    result = linprog(
        c=c,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=(0.0, None),
        method="highs",
    )
    if result.status != 0 or result.x is None:
        message = result.message.strip() if isinstance(result.message, str) else str(result.message)
        raise RuntimeError(
            "Exact balanced OT LP solver failed: "
            f"status={result.status}, message={message}"
        )

    plan_dense = np.asarray(result.x, dtype=np.float64).reshape(n_src, n_tgt)
    np.maximum(plan_dense, 0.0, out=plan_dense)
    total_mass = float(plan_dense.sum())
    if total_mass <= 0.0:
        raise RuntimeError("Exact balanced OT LP returned zero mass plan.")
    plan_dense /= total_mass

    support = plan_dense > float(support_tol)
    src_idx, tgt_idx = np.nonzero(support)
    mass = np.asarray(plan_dense[src_idx, tgt_idx], dtype=np.float64)
    mass_sum = float(mass.sum())
    if mass_sum <= 0.0:
        raise RuntimeError(
            "Exact balanced OT LP produced no support above support_tol. "
            f"Try decreasing support_tol (current={support_tol})."
        )
    mass = mass / mass_sum
    total_cost = float(np.sum(plan_dense * np.asarray(cost_matrix, dtype=np.float64)))
    return SparseOTPlan(
        src_idx=np.asarray(src_idx, dtype=np.int64),
        tgt_idx=np.asarray(tgt_idx, dtype=np.int64),
        mass=mass,
        total_cost=total_cost,
    )


def solve_balanced_ot_lp(
    x: torch.Tensor,
    y: torch.Tensor,
    src_weights: torch.Tensor | np.ndarray | None = None,
    tgt_weights: torch.Tensor | np.ndarray | None = None,
    support_tol: float = 1e-12,
    max_variables: int | None = None,
) -> SparseOTPlan:
    cost = pairwise_squared_euclidean_cost(x=x, y=y)
    return solve_balanced_ot_lp_from_cost(
        cost_matrix=cost,
        src_weights=src_weights,
        tgt_weights=tgt_weights,
        support_tol=support_tol,
        max_variables=max_variables,
    )


def balanced_empirical_w2_distance_exact(
    x: torch.Tensor,
    y: torch.Tensor,
    src_weights: torch.Tensor | np.ndarray | None = None,
    tgt_weights: torch.Tensor | np.ndarray | None = None,
    support_tol: float = 1e-12,
    max_variables: int | None = None,
) -> float:
    plan = solve_balanced_ot_lp(
        x=x,
        y=y,
        src_weights=src_weights,
        tgt_weights=tgt_weights,
        support_tol=support_tol,
        max_variables=max_variables,
    )
    return float(np.sqrt(max(plan.total_cost, 0.0)))


def _require_pot() -> object:
    try:
        import ot  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised when POT missing.
        raise ImportError(
            "POT backend requires the 'POT' package. Install with `pip install POT`."
        ) from exc
    return ot


def balanced_empirical_w2_distance_pot(
    x: torch.Tensor,
    y: torch.Tensor,
    src_weights: torch.Tensor | np.ndarray | None = None,
    tgt_weights: torch.Tensor | np.ndarray | None = None,
    num_itermax: int | None = None,
) -> float:
    ot = _require_pot()
    x_np = _to_numpy_points(x)
    y_np = _to_numpy_points(y)
    n_src = int(x_np.shape[0])
    n_tgt = int(y_np.shape[0])
    src_mass = _normalize_weights(src_weights, n_src, "source")
    tgt_mass = _normalize_weights(tgt_weights, n_tgt, "target")
    cost = pairwise_squared_euclidean_cost(x=x, y=y)
    kwargs: dict[str, int] = {}
    if num_itermax is not None:
        itermax_int = int(num_itermax)
        if itermax_int <= 0:
            raise ValueError(f"num_itermax must be positive when provided, got {num_itermax}")
        kwargs["numItermax"] = itermax_int
    emd2_value = float(ot.emd2(src_mass, tgt_mass, cost, **kwargs))
    return float(np.sqrt(max(emd2_value, 0.0)))
