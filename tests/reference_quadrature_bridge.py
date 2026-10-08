"""Finite-quadrature reference solver used to validate the continuous sampler.

This is test support, not the Multi paper training pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Callable
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.special import logsumexp, ndtri
from scipy.stats import qmc
import torch
from cfm_project.generalized_moment_sb import (posterior_numpy, balance_log_kernel, MomentBridgeSolution)

Array = np.ndarray

def gaussian_quadrature_noise(n: int, dimension: int, seed: int) -> Array:
    """Nested antithetic Sobol normals; both halves have exactly zero mean."""
    if n < 4 or n & (n - 1):
        raise ValueError("Gaussian quadrature size must be a power of two >= 4.")
    uniforms = qmc.Sobol(d=dimension, scramble=True, seed=seed).random_base2(
        int(math.log2(n)) - 1
    )
    normals = ndtri(np.clip(uniforms, 1e-12, 1 - 1e-12))
    return np.stack((normals, -normals), axis=1).reshape(n, dimension).astype(np.float32)


@dataclass
class PairQuadrature:
    x0: Array
    x1: Array
    src: Array
    tgt: Array
    sigma: float
    tau: float
    features: Array
    noise: Array
    signs: Array
    feature_path: Path | None = None
    build_seconds: float = 0.0
    timing: dict[str, float] = field(default_factory=dict)
    log_importance: Array | None = None

    @property
    def pair_count(self) -> int:
        return len(self.src)

    @property
    def n_features(self) -> int:
        return self.features.shape[-1]

    @property
    def n_quadrature(self) -> int:
        return self.features.shape[1]

    def log_normalizers_and_means(self, multiplier: Array) -> tuple[Array, Array]:
        log_z = np.empty(self.pair_count, dtype=np.float64)
        means = np.empty((self.pair_count, self.n_features), dtype=np.float64)
        # Limit temporary float64 arrays; a full dense pair cache can be several GB.
        chunk = max(128, min(16384, 2_000_000 // (
            self.n_quadrature * self.n_features
        )))
        for first in range(0, self.pair_count, chunk):
            last = min(first + chunk, self.pair_count)
            f = np.asarray(self.features[first:last], dtype=np.float64)
            # Renormalization also makes optional float16 caches simplex-valued.
            f = f / f.sum(-1, keepdims=True)
            logits = np.einsum("rsk,k->rs", f, multiplier, optimize=False)
            if self.log_importance is not None:
                logits += np.asarray(self.log_importance[first:last], dtype=np.float64)
            norm = logsumexp(logits, axis=1)
            weights = np.exp(logits - norm[:, None])
            log_z[first:last] = norm - math.log(self.n_quadrature)
            means[first:last] = np.einsum("rs,rsk->rk", weights, f, optimize=False)
        return log_z, means

    def conditional_composition(self, multiplier: Array, pair_mass: Array) -> Array:
        _, means = self.log_normalizers_and_means(multiplier)
        return np.asarray(pair_mass, dtype=np.float64).reshape(-1) @ means


@dataclass
class RefinedPairQuadrature:
    """Replace selected integrals at higher resolution; retain every pair."""
    background: PairQuadrature
    refinement: PairQuadrature
    refined_indices: Array

    def __post_init__(self):
        self.refined_indices = np.asarray(self.refined_indices, dtype=np.int64)
        ids = self.refined_indices
        if (ids.ndim != 1 or len(ids) != self.refinement.pair_count
                or len(np.unique(ids)) != len(ids) or (ids < 0).any()
                or (ids >= self.background.pair_count).any()):
            raise ValueError("Refined indices must identify distinct existing pairs.")
        if not (np.array_equal(self.background.src[ids], self.refinement.src)
                and np.array_equal(self.background.tgt[ids], self.refinement.tgt)
                and np.array_equal(self.background.x0, self.refinement.x0)
                and np.array_equal(self.background.x1, self.refinement.x1)
                and self.background.sigma == self.refinement.sigma
                and self.background.tau == self.refinement.tau
                and self.background.n_features == self.refinement.n_features):
            raise ValueError("Refined integrals must describe the same reference and endpoint pairs.")

    def __getattr__(self, name):
        return getattr(self.background, name)

    @property
    def build_seconds(self):
        return self.background.build_seconds + self.refinement.build_seconds

    @property
    def timing(self):
        return {**self.background.timing,
                **{'refinement_'+k: v for k, v in self.refinement.timing.items()}}

    def log_normalizers_and_means(self, multiplier):
        log_z, means = self.background.log_normalizers_and_means(multiplier)
        refined_z, refined_means = self.refinement.log_normalizers_and_means(multiplier)
        log_z[self.refined_indices] = refined_z
        means[self.refined_indices] = refined_means
        return log_z, means

    def conditional_composition(self, multiplier, pair_mass):
        _, means = self.log_normalizers_and_means(multiplier)
        return np.asarray(pair_mass, dtype=np.float64).reshape(-1) @ means


def build_pair_quadrature(
    *,
    x0: Array,
    x1: Array,
    posterior: Callable[[torch.Tensor], torch.Tensor],
    sigma: float,
    n_quadrature: int,
    seed: int,
    tau: float = 0.5,
    src: Array | None = None,
    tgt: Array | None = None,
    cache_path: Path | None = None,
    storage_dtype: str = "float32",
    proposal_multiplier: Array | None = None,
    proposal_steps: int = 64,
    proposal_mode_cache: Path | None = None,
    independent_pair_noise: bool = False,
    proposal_center_fn: Callable[[Array], Array] | None = None,
    proposal_fractions: tuple[float, ...] | None = None,
    quadrature_batch_states: int = 65536,
    importance_storage_dtype: str = "float32",
    progress: Callable[[dict], None] | None = None,
) -> PairQuadrature:
    started = time.perf_counter()
    if sigma <= 0 or not 0 < tau < 1:
        raise ValueError("A nondegenerate Brownian reference requires sigma > 0 and 0 < tau < 1.")
    x0, x1 = np.asarray(x0, dtype=np.float32), np.asarray(x1, dtype=np.float32)
    if src is None and tgt is None:
        src = np.repeat(np.arange(len(x0)), len(x1))
        tgt = np.tile(np.arange(len(x1)), len(x0))
    elif src is None or tgt is None:
        raise ValueError("Supply both endpoint index arrays or neither.")
    src, tgt = np.asarray(src, dtype=np.int64), np.asarray(tgt, dtype=np.int64)
    if len(src) != len(tgt):
        raise ValueError("Endpoint index arrays must have the same size.")
    n_pairs, dimension = len(src), x0.shape[1]
    fractions = (1.,) if proposal_fractions is None else tuple(proposal_fractions)
    components = 1+len(fractions) if proposal_multiplier is not None else 1
    if proposal_multiplier is not None:
        if (not fractions or not all(0 < f <= 1 for f in fractions)
                or n_quadrature % components or n_quadrature//components < 4):
            raise ValueError("Defensive components need valid fractions and equal power-of-two Gaussian grids.")
        component_noise = gaussian_quadrature_noise(n_quadrature // components, dimension, seed)
        noise = np.concatenate([component_noise]*components)
    else:
        noise = gaussian_quadrature_noise(n_quadrature, dimension, seed)
    # Pair-specific reflections remain exact Gaussian symmetries and reduce
    # correlations across conditional integration errors.
    signs = np.empty((0 if independent_pair_noise else n_pairs, dimension), dtype=np.int8)
    sign_rng = np.random.default_rng(seed + 991)
    sign_chunk = max(2, 4_000_000 // dimension)
    # choice creates an int64 index temporary even for an int8 population.
    # Chunking bounds that temporary for large single-cell endpoint pools.
    for first in range(0, len(signs), sign_chunk):
        last = min(first + sign_chunk, n_pairs)
        signs[first:last] = sign_rng.choice(
            np.array([-1, 1], dtype=np.int8), size=(last - first, dimension)
        )
    k = posterior_numpy(posterior, x0[:1]).shape[1]
    shape = (n_pairs, n_quadrature, k)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        features = np.lib.format.open_memmap(
            cache_path, mode="w+", dtype=storage_dtype, shape=shape
        )
    else:
        features = np.empty(shape, dtype=storage_dtype)
    log_importance = None
    if proposal_multiplier is not None:
        if cache_path is not None:
            log_importance = np.lib.format.open_memmap(
                cache_path.with_name(cache_path.stem + "_importance.npy"),
                mode="w+", dtype=importance_storage_dtype, shape=(n_pairs, n_quadrature),
            )
        else:
            log_importance = np.empty((n_pairs, n_quadrature), dtype=np.float32)
    c_sqrt = sigma * math.sqrt(tau * (1 - tau))
    modes = None
    mode_seconds = 0.
    mode_cache_hit = False
    if proposal_multiplier is not None and proposal_mode_cache is not None and proposal_mode_cache.exists():
        modes = np.load(proposal_mode_cache, mmap_mode="r")
        if modes.shape != (n_pairs, dimension):
            raise ValueError("Cached importance modes do not match endpoint-pair dimensions.")
        mode_cache_hit = True
    if proposal_multiplier is not None and modes is None and proposal_center_fn is None:
        mode_start = time.perf_counter()
        last_notice = mode_start
        modes = np.empty((n_pairs, dimension), dtype=np.float32)
        # Mode batches are independent of quadrature size.
        for first in range(0, n_pairs, 8192):
            last = min(first + 8192, n_pairs)
            mean_t = torch.from_numpy(
                (1 - tau) * x0[src[first:last]] + tau * x1[tgt[first:last]]
            )
            lam_t = torch.as_tensor(proposal_multiplier, dtype=torch.float32)
            current = mean_t.clone().requires_grad_(True)
            optimizer = torch.optim.Adam([current], lr=.2 * c_sqrt)
            best = current.detach().clone()
            best_energy = torch.full((len(mean_t),), math.inf)
            for _ in range(proposal_steps):
                energy = .5 * ((current - mean_t) / c_sqrt).square().sum(1) - posterior(current) @ lam_t
                improved = energy.detach() < best_energy
                best[improved] = current.detach()[improved]
                best_energy = torch.minimum(best_energy, energy.detach())
                optimizer.zero_grad(set_to_none=True)
                energy.sum().backward()
                optimizer.step()
            modes[first:last] = best.numpy()
            if progress is not None and time.perf_counter() - last_notice >= 20:
                progress({"event": "importance_proposal_modes", "pairs": last,
                          "total_pairs": n_pairs, "seconds": time.perf_counter() - mode_start})
                last_notice = time.perf_counter()
        mode_seconds = time.perf_counter() - mode_start
        if proposal_mode_cache is not None:
            proposal_mode_cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(proposal_mode_cache, modes)
            proposal_mode_cache.with_suffix(".json").write_text(json.dumps({
                "sigma_used_to_propose_modes": sigma,
                "lambda_used_to_propose_modes": proposal_multiplier.tolist(),
                "steps": proposal_steps, "seconds": mode_seconds,
                "role": "Reusable importance proposal only; exact Gaussian mixture density ratio is always recomputed.",
                "pairs": n_pairs, "dimension": dimension,
            }, indent=2) + "\n")
    chunk = max(1, quadrature_batch_states // n_quadrature)
    pair_noise_rng = np.random.default_rng(seed + 104729)
    last_notice = started
    for first in range(0, n_pairs, chunk):
        last = min(first + chunk, n_pairs)
        mean = (1 - tau) * x0[src[first:last]] + tau * x1[tgt[first:last]]
        if independent_pair_noise:
            # Independent integration errors across pairs avoid reusing the
            # same finite set of Gaussian radii throughout a dense coupling.
            # Antithetic partners preserve exact conditional sample means.
            n_unique = n_quadrature // (2*components)
            half = pair_noise_rng.standard_normal(
                (last-first, n_unique, dimension), dtype=np.float32)
            local_noise = np.stack((half, -half), axis=2).reshape(last-first, 2*n_unique, dimension)
            if proposal_multiplier is not None:
                local_noise = np.concatenate([local_noise]*components, axis=1)
            points = mean[:, None, :] + c_sqrt * local_noise
        else:
            points = mean[:, None, :] + c_sqrt * signs[first:last, None, :] * noise[None, :, :]
        if proposal_multiplier is not None:
            # A Gaussian centered on a tilted-density mode locates rare relevant
            # regions. Half of the proposal remains the original Gaussian; the
            # exact mixture importance ratio corrects both components.
            if proposal_center_fn is None:
                mode = modes[first:last]
            else:
                mode_start = time.perf_counter()
                mode = np.asarray(proposal_center_fn(mean), dtype=np.float32)
                mode_seconds += time.perf_counter() - mode_start
                if mode.shape != mean.shape or not np.isfinite(mode).all():
                    raise ValueError("Importance proposal centers must match the finite pair means.")
            component_nodes = n_quadrature//components
            for component, fraction in enumerate(fractions,1):
                points[:, component*component_nodes:(component+1)*component_nodes] += fraction*(mode-mean)[:,None,:]
            log_g = -.5 * np.sum(((points - mean[:, None]) / c_sqrt) ** 2, axis=2, dtype=np.float64)
            log_h = log_g.copy()
            for fraction in fractions:
                center = mean+fraction*(mode-mean)
                log_shift = -.5 * np.sum(((points-center[:,None])/c_sqrt)**2,axis=2,dtype=np.float64)
                log_h = np.logaddexp(log_h,log_shift)
            log_h -= math.log(components)
            log_importance[first:last] = log_g - log_h
        f = posterior_numpy(posterior, points.reshape(-1, dimension))
        features[first:last] = f.reshape(last - first, n_quadrature, k)
        if progress is not None and time.perf_counter() - last_notice >= 20:
            progress({"event": "quadrature", "pairs": last, "total_pairs": n_pairs,
                      "seconds": time.perf_counter() - started})
            last_notice = time.perf_counter()
    if isinstance(features, np.memmap):
        features.flush()
    if isinstance(log_importance, np.memmap):
        log_importance.flush()
    return PairQuadrature(
        x0=x0, x1=x1, src=src, tgt=tgt, sigma=sigma, tau=tau,
        features=features, noise=noise, signs=signs, feature_path=cache_path,
        build_seconds=time.perf_counter() - started,
        log_importance=log_importance,
        timing={"importance_proposal_mode_seconds": mode_seconds,
                "importance_proposal_mode_cache_hit": float(mode_cache_hit)},
    )


def solve_moment_bridge(
    quadrature: PairQuadrature,
    target: Array,
    *,
    fixed_coupling: Array | None = None,
    initial_multiplier: Array | None = None,
    max_iterations: int = 120,
    moment_tolerance: float = 2e-5,
    sinkhorn_tolerance: float = 2e-8,
    multiplier_bound: float = 500.,
    stop_on_feasibility: bool = False,
    stop_on_multiplier_bound: bool = False,
    dual_coordinate_scale: Array | None = None,
    sinkhorn_relaxation: float = 1.,
    sinkhorn_stabilize_every: int = 0,
    sinkhorn_newton_refine: bool = False,
    sinkhorn_max_iterations: int = 10000,
    progress: Callable[[dict], None] | None = None,
) -> MomentBridgeSolution:
    """Minimize the entropy dual, rebalancing all endpoints when pi is free."""
    started = time.perf_counter()
    target = np.asarray(target, dtype=np.float64)
    if target.shape != (quadrature.n_features,) or not np.isclose(target.sum(), 1):
        raise ValueError("Target must be a unit-sum vector matching the posterior.")
    if (target <= 0).any():
        raise ValueError("A finite softmax entropy dual needs strictly interior targets.")
    coordinate_scale = (np.ones(len(target)-1) if dual_coordinate_scale is None else
                        np.asarray(dual_coordinate_scale,dtype=np.float64).reshape(-1)[:len(target)-1])
    if coordinate_scale.shape != (len(target)-1,) or not np.isfinite(coordinate_scale).all() or (coordinate_scale <= 0).any():
        raise ValueError("Dual coordinate scales must be finite and positive.")
    n0, n1 = len(quadrature.x0), len(quadrature.x1)
    free = fixed_coupling is None
    if free and quadrature.pair_count != n0 * n1:
        raise ValueError("Free coupling requires all endpoint pairs; sparse truncation is not allowed.")
    if free:
        if not (
            np.array_equal(quadrature.src, np.repeat(np.arange(n0), n1))
            and np.array_equal(quadrature.tgt, np.tile(np.arange(n1), n0))
        ):
            raise ValueError("Dense pairs must be in row-major order.")
        # Avoid materializing (n0*n1, dimension) endpoint differences.  The
        # dense coupling is necessary; dense copies of every coordinate are not.
        a64 = np.asarray(quadrature.x0, dtype=np.float64)
        b64 = np.asarray(quadrature.x1, dtype=np.float64)
        cost = -2 * (a64 @ b64.T)
        cost += np.sum(a64 * a64, axis=1)[:, None]
        cost += np.sum(b64 * b64, axis=1)[None, :]
        np.maximum(cost, 0., out=cost)
        log_k = -cost / (2 * quadrature.sigma ** 2)
    else:
        fixed_coupling = np.asarray(fixed_coupling, dtype=np.float64).reshape(-1)
        if len(fixed_coupling) != quadrature.pair_count or (fixed_coupling < 0).any():
            raise ValueError("Fixed coupling must have one nonnegative mass per pair.")
        if not np.isclose(fixed_coupling.sum(), 1):
            raise ValueError("Fixed coupling must sum to one.")
        row = np.bincount(quadrature.src, fixed_coupling, minlength=n0)
        col = np.bincount(quadrature.tgt, fixed_coupling, minlength=n1)
        if max(np.max(np.abs(row - 1 / n0)), np.max(np.abs(col - 1 / n1))) > 1e-7:
            raise ValueError("Fixed coupling must preserve both uniform empirical endpoints.")
    timings = {"quadrature_build_seconds": quadrature.build_seconds,
               "normalizer_moment_seconds": 0.0, "endpoint_balancing_seconds": 0.0,
               **quadrature.timing}
    history: list[dict] = []
    state: dict = {"warm": None, "accepted_iterations": 0}

    class MomentToleranceReached(Exception):
        pass

    class MultiplierBoundReached(Exception):
        pass

    def objective(reduced: Array) -> tuple[float, Array]:
        multiplier = np.r_[reduced/coordinate_scale, 0.]
        tick = time.perf_counter()
        log_z, conditional_mean = quadrature.log_normalizers_and_means(multiplier)
        normalizer_seconds = time.perf_counter() - tick
        timings["normalizer_moment_seconds"] += normalizer_seconds
        if free:
            try:
                balanced = balance_log_kernel(
                    log_k + log_z.reshape(n0, n1), warm_start=state["warm"],
                    tolerance=sinkhorn_tolerance,
                    relaxation=sinkhorn_relaxation, stabilize_every=sinkhorn_stabilize_every,
                    newton_refine=sinkhorn_newton_refine,
                    max_iterations=sinkhorn_max_iterations,
                )
            except RuntimeError as exc:
                if progress is not None:
                    progress(dict(event="endpoint_balance_failure", multiplier=multiplier.tolist(), error=str(exc)))
                raise
            state["warm"] = (balanced.log_u, balanced.log_v)
            mass = balanced.coupling.reshape(-1)
            dual = -balanced.log_u.mean() - balanced.log_v.mean() - multiplier @ target
            endpoint_linf = balanced.marginal_linf
            endpoint_relative = balanced.marginal_relative_linf
            timings["endpoint_balancing_seconds"] += balanced.seconds
            sinkhorn_steps = balanced.iterations
        else:
            mass = fixed_coupling
            dual = mass @ log_z - multiplier @ target
            endpoint_linf = max(float(np.max(np.abs(row - 1 / n0))),
                                float(np.max(np.abs(col - 1 / n1))))
            endpoint_relative = max(endpoint_linf * n0, endpoint_linf * n1)
            sinkhorn_steps = 0
        # GPU reductions may already return float32 probabilities. Accumulate
        # their weighted mean in float64 without expanding the full dense
        # pair-by-feature array to float64.
        composition = (np.einsum("r,rk->k", mass, conditional_mean, dtype=np.float64,
                                  optimize=False)
                       if conditional_mean.dtype != np.float64 else mass @ conditional_mean)
        residual = composition - target
        item = {
            "evaluation": len(history) + 1, "dual": float(dual),
            "residual_l2": float(np.linalg.norm(residual)),
            "residual_linf": float(np.abs(residual).max()),
            "component_residual": residual.tolist(),
            "lambda_l2": float(np.linalg.norm(multiplier)),
            "lambda_span": float(np.ptp(multiplier)),
            "multiplier": multiplier.tolist(),
            "normalizer_seconds": normalizer_seconds,
            "endpoint_balancing_seconds": balanced.seconds if free else 0.,
            "endpoint_newton_iterations": balanced.newton_iterations if free else 0,
            "endpoint_continuation_stages": balanced.continuation_stages if free else 0,
            "endpoint_linf": endpoint_linf, "sinkhorn_iterations": sinkhorn_steps,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(item)
        state.update(multiplier=multiplier, composition=composition, mass=mass,
                     log_z=log_z, endpoint_linf=endpoint_linf,
                     endpoint_relative=endpoint_relative)
        if progress is not None:
            progress({"event": "dual", **item})
        # For the convex entropy dual, this full residual is its stationarity
        # condition (including the simplex coordinate omitted from the gauge).
        # It is more reliable than tiny objective differences with accelerated
        # float32 Gaussian integration.
        if stop_on_feasibility and np.abs(residual).max() <= moment_tolerance:
            raise MomentToleranceReached
        return float(dual), residual[:-1]/coordinate_scale

    initial = np.zeros(len(target) - 1)
    if initial_multiplier is not None:
        initial = np.asarray(initial_multiplier[:-1]) - float(initial_multiplier[-1])
    initial = initial*coordinate_scale
    def accepted_iteration(point):
        state["accepted_iterations"] += 1
        if stop_on_multiplier_bound:
            raw_point = point/coordinate_scale
            if not np.array_equal(state["multiplier"][:-1], raw_point):
                objective(point)
            active = np.abs(raw_point) >= multiplier_bound*(1-1e-10)
            outward = raw_point*(state["composition"]-target)[:-1] < -moment_tolerance*multiplier_bound
            if np.any(active & outward):
                raise MultiplierBoundReached
    try:
        result = minimize(
            objective, initial, jac=True, method="L-BFGS-B",
            bounds=[(-multiplier_bound*s, multiplier_bound*s) for s in coordinate_scale],
            callback=accepted_iteration,
            options={"maxiter": max_iterations, "ftol": 1e-13, "gtol": moment_tolerance / 10,
                     "maxls": 40, "maxcor": 15},
        )
    except MomentToleranceReached:
        result = OptimizeResult(
            x=state["multiplier"][:-1].copy()*coordinate_scale, success=True,
            message="Full generalized-moment stationarity tolerance reached.",
            nit=state["accepted_iterations"], nfev=len(history))
    except MultiplierBoundReached:
        result = OptimizeResult(
            x=state["multiplier"][:-1].copy()*coordinate_scale, success=False,
            message="Numerical multiplier bound reached; refine quadrature/proposal before accepting a teacher.",
            nit=state["accepted_iterations"], nfev=len(history))
    # A rejected line search can leave state at a trial point, not result.x.
    optimum = np.r_[result.x/coordinate_scale, 0.]
    if not np.allclose(state["multiplier"], optimum, rtol=1e-14, atol=1e-12):
        try:
            objective(result.x)
        except MomentToleranceReached:
            pass
    timings["dual_solve_seconds"] = time.perf_counter() - started
    return MomentBridgeSolution(
        coupling_mode="sb_recomputed" if free else "fixed_global_ot",
        multiplier=state["multiplier"].copy(), coupling=state["mass"].copy(),
        src=quadrature.src, tgt=quadrature.tgt, composition=state["composition"].copy(),
        target=target, log_z=state["log_z"].copy(),
        converged=bool(np.abs(state["composition"] - target).max() <= moment_tolerance),
        optimizer_success=bool(result.success), optimizer_message=str(result.message),
        optimizer_iterations=int(result.nit), evaluations=len(history),
        endpoint_linf=state["endpoint_linf"],
        endpoint_relative_linf=state["endpoint_relative"], timings=timings, history=history,
    )
