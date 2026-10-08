"""Endpoint kernel balancing and continuous classifier-moment bridge sampling.

The final Multi solver fits its continuous dual in benchmarks/classifier_dual.py.
Finite-quadrature reference solvers live in tests/reference_quadrature_bridge.py.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Callable

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp
from scipy.sparse.linalg import LinearOperator, cg
import torch


Array = np.ndarray


def posterior_numpy(
    posterior: Callable[[torch.Tensor], torch.Tensor],
    points: Array,
    batch_size: int = 65536,
) -> Array:
    outputs = []
    with torch.no_grad():
        for first in range(0, len(points), batch_size):
            value = posterior(torch.from_numpy(np.asarray(
                points[first:first + batch_size], dtype=np.float32
            )))
            outputs.append(value.detach().cpu().numpy())
    result = np.concatenate(outputs).astype(np.float32, copy=False)
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("Feature posterior must be finite and nonnegative.")
    if not np.allclose(result.sum(1), 1, atol=1e-5):
        raise ValueError("This categorical-moment implementation requires simplex features.")
    return result


@dataclass
class SinkhornResult:
    coupling: Array
    log_u: Array
    log_v: Array
    iterations: int
    marginal_linf: float
    marginal_relative_linf: float
    seconds: float
    newton_iterations: int = 0
    continuation_stages: int = 0


def _newton_endpoint_refinement(log_kernel, a, b, log_u, log_v, tolerance,
                                previous_iterations, started):
    """Newton correction of endpoint log potentials after slow scaling.

    The Hessian is the covariance block [[diag(row), pi], [pi.T, diag(col)]].
    Fixing one column potential removes its global gauge. Matrix products keep
    the full endpoint support and do not form a square (n+m)-by-(n+m) Hessian.
    """
    n, m = log_kernel.shape
    gauge = log_v[-1]
    lu, lv = log_u+gauge, log_v-gauge
    mass = np.exp(log_kernel+lu[:, None]+lv[None, :])
    for step in range(61):
        row, col = mass.sum(1), mass.sum(0)
        relative = max(float(np.max(np.abs(row/a-1))), float(np.max(np.abs(col/b-1))))
        if relative <= tolerance:
            absolute = max(float(np.max(np.abs(row-a))), float(np.max(np.abs(col-b))))
            return SinkhornResult(mass, lu, lv, previous_iterations+step,
                                  absolute, relative, time.perf_counter()-started, step)
        if step == 60:
            break
        gradient = np.r_[row-a, (col-b)[:-1]]
        diagonal = np.r_[row, col[:-1]]
        merit = float(np.sum((row-a)**2/a) + np.sum((col-b)**2/b))
        ridge = 1e-12 * diagonal.mean()
        def hessian_product(v):
            va, vb = v[:n], np.r_[v[n:], 0.]
            return np.r_[row*va + mass@vb, (col*vb + mass.T@va)[:-1]] + ridge*v
        operator = LinearOperator((n+m-1, n+m-1), matvec=hessian_product)
        preconditioner = LinearOperator(operator.shape, matvec=lambda v: v/(diagonal+ridge))
        delta, status = cg(operator, -gradient, M=preconditioner,
                           rtol=.01, atol=0., maxiter=min(2000, n+m))
        if status < 0 or not np.isfinite(delta).all():
            raise RuntimeError("Endpoint Newton correction failed to obtain a finite direction.")
        if status > 0 and m <= 2500:
            # Nearly disconnected kernels have several weak potential gauges.
            # A diagonal preconditioner can leave CG unresolved. Eliminate row
            # potentials and solve the small column Schur system accurately;
            # the original dense endpoint kernel remains unchanged.
            schur = np.diag(col[:-1]) - mass[:, :-1].T @ (mass[:, :-1]/row[:, None])
            rhs = -(col-b)[:-1] + mass[:, :-1].T @ ((row-a)/row)
            schur = .5*(schur+schur.T)
            jitter = 1e-14 * diagonal.mean()
            for _ in range(6):
                try:
                    factor = cho_factor(schur + jitter*np.eye(m-1), check_finite=False)
                    db = np.r_[cho_solve(factor, rhs, check_finite=False), 0.]
                    da = (-(row-a)-mass@db)/row
                    delta = np.r_[da, db[:-1]]
                    break
                except np.linalg.LinAlgError:
                    jitter *= 10
        delta *= min(1., 20/max(float(np.abs(delta).max()), 1e-30))
        da, db = delta[:n], np.r_[delta[n:], 0.]
        scale = 1.
        for _ in range(24):
            candidate = np.exp(log_kernel + (lu+scale*da)[:, None] + (lv+scale*db)[None, :])
            cr, cc = candidate.sum(1), candidate.sum(0)
            candidate_relative = max(float(np.max(np.abs(cr/a-1))), float(np.max(np.abs(cc/b-1))))
            candidate_merit = float(np.sum((cr-a)**2/a) + np.sum((cc-b)**2/b))
            if candidate_merit < merit or candidate_relative <= tolerance:
                lu += scale*da
                lv += scale*db
                mass = candidate
                break
            scale *= .5
        else:
            raise RuntimeError(f"Endpoint Newton correction stalled at relative residual {relative:.3g}.")
    raise RuntimeError(f"Endpoint Newton correction did not converge: {relative:.3g}.")


def balance_log_kernel(
    log_kernel: Array,
    a: Array | None = None,
    b: Array | None = None,
    *,
    warm_start: tuple[Array, Array] | None = None,
    tolerance: float = 2e-8,
    max_iterations: int = 10000,
    relaxation: float = 1.,
    stabilize_every: int = 0,
    newton_refine: bool = False,
    allow_continuation: bool = True,
) -> SinkhornResult:
    """Balance a positive kernel in float64, retaining original-kernel potentials."""
    started = time.perf_counter()
    if not 1 <= relaxation < 2 or stabilize_every < 0:
        raise ValueError("Relaxation must be in [1,2), and stabilization frequency nonnegative.")
    log_kernel = np.asarray(log_kernel, dtype=np.float64)
    n, m = log_kernel.shape
    a = np.full(n, 1 / n) if a is None else np.asarray(a, dtype=np.float64)
    b = np.full(m, 1 / m) if b is None else np.asarray(b, dtype=np.float64)
    if (
        not np.isfinite(log_kernel).all()
        or (a <= 0).any() or (b <= 0).any()
        or not np.isclose(a.sum(), 1) or not np.isclose(b.sum(), 1)
    ):
        raise ValueError("Sinkhorn requires finite log kernels and positive unit masses.")
    row_shift = log_kernel.max(1)
    shifted = log_kernel - row_shift[:, None]
    col_shift = shifted.max(0)
    kernel = np.exp(shifted - col_shift[None, :])
    absorbed_u, absorbed_v = -row_shift.copy(), -col_shift.copy()
    if warm_start is None:
        u, v = np.ones(n), np.ones(m)
    else:
        warm_u, warm_v = np.asarray(warm_start[0]), np.asarray(warm_start[1])
        warm_kernel = log_kernel + warm_u[:, None] + warm_v[None, :]
        if (warm_kernel.max() < 300 and warm_kernel.max(1).min() > -600
                and warm_kernel.max(0).min() > -600):
            # Absorb a nearby solved kernel directly. Clipping large individual
            # potentials would discard useful weak-component gauge information
            # even when their combined log coupling is numerically ordinary.
            kernel = np.exp(warm_kernel)
            absorbed_u, absorbed_v = warm_u.copy(), warm_v.copy()
            u, v = np.ones(n), np.ones(m)
        else:
            lu, lv = warm_u + row_shift, warm_v + col_shift
            gauge = (lu.mean() - lv.mean()) / 2
            u, v = np.exp(np.clip(lu - gauge, -300, 300)), np.exp(np.clip(lv + gauge, -300, 300))
        del warm_kernel
    relative = math.inf
    active_relaxation = relaxation
    for iteration in range(1, max_iterations + 1):
        kv = kernel @ v
        if (kv <= 0).any() or not np.isfinite(kv).all():
            raise RuntimeError("Kernel scaling underflow: use a larger diffusion or log scaling.")
        next_u = a / kv
        omega = active_relaxation if iteration >= 20 and relative < .5 else 1.
        u = next_u if omega == 1 else np.exp((1-omega)*np.log(u) + omega*np.log(next_u))
        ktu = kernel.T @ u
        if (ktu <= 0).any() or not np.isfinite(ktu).all():
            raise RuntimeError("Kernel scaling underflow in column update.")
        next_v = b / ktu
        v = next_v if omega == 1 else np.exp((1-omega)*np.log(v) + omega*np.log(next_v))
        if iteration % 10 == 0 or iteration == max_iterations:
            row = u * (kernel @ v)
            col = v * (kernel.T @ u)
            previous_relative = relative
            relative = max(float(np.max(np.abs(row / a - 1))),
                           float(np.max(np.abs(col / b - 1))))
            if relative <= tolerance:
                break
            if omega > 1 and relative > 1.2 * previous_relative:
                active_relaxation = 1 + .5 * (active_relaxation-1)
            gauge = math.exp(float(np.clip((np.log(u).mean() - np.log(v).mean()) / 2, -100, 100)))
            u /= gauge
            v *= gauge
        if stabilize_every and iteration % stabilize_every == 0:
            # Recompute from the original log kernel so previously underflowed
            # entries can recover as potentials change. This preserves the
            # original kernel and does not truncate endpoint support.
            absorbed_u += np.log(u)
            absorbed_v += np.log(v)
            log_current = log_kernel + absorbed_u[:, None] + absorbed_v[None, :]
            if float(log_current.max()) > 300:
                shift = float(log_current.max())
                absorbed_u -= shift
                log_current -= shift
            kernel = np.exp(log_current)
            u.fill(1.)
            v.fill(1.)
    if relative > tolerance:
        if newton_refine:
            try:
                return _newton_endpoint_refinement(
                    log_kernel, a, b, np.log(u)+absorbed_u, np.log(v)+absorbed_v,
                    tolerance, iteration, started)
            except RuntimeError:
                if not allow_continuation:
                    raise
                # Temperature continuation initializes difficult scaling
                # problems. The final stage is exactly the original kernel;
                # no softened kernel is returned as the endpoint solution.
                fraction = min(1., 20/max(float(np.ptp(log_kernel)), 20.))
                previous_fraction, continuation_warm = None, None
                stage = None
                stages, total_iterations, total_newton = 0, iteration, 0
                while True:
                    if stage is not None:
                        ratio = fraction/previous_fraction
                        continuation_warm = (stage.log_u*ratio, stage.log_v*ratio)
                    stage = balance_log_kernel(
                        fraction*log_kernel, a, b, warm_start=continuation_warm,
                        tolerance=tolerance, max_iterations=max(1000,max_iterations),
                        relaxation=relaxation, stabilize_every=max(100,stabilize_every),
                        newton_refine=True, allow_continuation=False)
                    stages += 1
                    total_iterations += stage.iterations
                    total_newton += stage.newton_iterations
                    if fraction == 1.:
                        stage.seconds = time.perf_counter()-started
                        stage.iterations, stage.newton_iterations = total_iterations, total_newton
                        stage.continuation_stages = stages
                        return stage
                    previous_fraction = fraction
                    fraction = min(1., 1.7*fraction)
        raise RuntimeError(
            f"Endpoint balancing did not converge: relative residual {relative:.3g}, "
            f"{iteration} iterations."
        )
    coupling = u[:, None] * kernel * v[None, :]
    absolute = max(float(np.max(np.abs(coupling.sum(1) - a))),
                   float(np.max(np.abs(coupling.sum(0) - b))))
    return SinkhornResult(
        coupling=coupling, log_u=np.log(u) + absorbed_u,
        log_v=np.log(v) + absorbed_v, iterations=iteration,
        marginal_linf=absolute, marginal_relative_linf=relative,
        seconds=time.perf_counter() - started,
    )


@dataclass
class MomentBridgeSolution:
    coupling_mode: str
    multiplier: Array
    coupling: Array
    src: Array
    tgt: Array
    composition: Array
    target: Array
    log_z: Array
    converged: bool
    optimizer_success: bool
    optimizer_message: str
    optimizer_iterations: int
    evaluations: int
    endpoint_linf: float
    endpoint_relative_linf: float
    timings: dict[str, float]
    history: list[dict]
    moment_estimator: str = 'quadrature'

    @property
    def residual(self) -> Array:
        return self.composition - self.target

    def metadata(self) -> dict:
        result = {
            "coupling_mode": self.coupling_mode,
            "multiplier": self.multiplier.tolist(),
            "multiplier_convention": "exp(+lambda dot f), last component fixed to zero",
            "target": self.target.tolist(), "composition": self.composition.tolist(),
            "moment_estimator": self.moment_estimator,
            "moment_residual": self.residual.tolist(),
            "moment_residual_l2": float(np.linalg.norm(self.residual)),
            "moment_residual_linf": float(np.abs(self.residual).max()),
            "converged": self.converged, "optimizer_success": self.optimizer_success,
            "optimizer_message": self.optimizer_message,
            "optimizer_iterations": self.optimizer_iterations, "evaluations": self.evaluations,
            "endpoint_linf": self.endpoint_linf,
            "endpoint_relative_linf": self.endpoint_relative_linf,
            "timings": self.timings, "history": self.history,
        }
        if self.moment_estimator == 'quadrature':
            result.update(quadrature_residual=self.residual.tolist(),
                          quadrature_residual_l2=float(np.linalg.norm(self.residual)),
                          quadrature_residual_linf=float(np.abs(self.residual).max()))
        return result


def conditional_feature_covariance(features: Array, pair_indices: Array) -> Array:
    """Estimate E_pi Cov(phi | endpoint pair) as a dual preconditioner.

    Repeated independent chains at each pair estimate within-pair variation.
    Singleton groups contribute zero; the estimate is used for preconditioning,
    not as a replacement for the generated-moment gradient.
    """
    features = np.asarray(features, dtype=np.float64)
    pair_indices = np.asarray(pair_indices)
    if features.ndim != 2 or pair_indices.shape != (len(features),) or not len(features):
        raise ValueError('Expected one endpoint-pair index per feature row.')
    _, inverse, counts = np.unique(pair_indices, return_inverse=True, return_counts=True)
    sums = np.stack([np.bincount(inverse, features[:, k], minlength=len(counts))
                     for k in range(features.shape[1])], axis=1)
    centered = features-sums[inverse]/counts[inverse, None]
    weights = counts[inverse]/np.maximum(counts[inverse]-1, 1)/len(features)
    return (centered*weights[:, None]).T @ centered


def sample_continuous_tilt(
    *,
    x0: Array,
    x1: Array,
    solution: MomentBridgeSolution,
    posterior: Callable[[torch.Tensor], torch.Tensor],
    sigma: float,
    n_samples: int,
    seed: int,
    tau: float = 0.5,
    burn_in: int = 256,
    pcn_beta: float = 0.5,
    initial_candidates: int = 32,
    proposal_modes: Array | Callable[[Array], Array] | None = None,
    checkpoints: tuple[int, ...] = (64, 128, 256),
    independence_every: int = 0,
    proposal_fractions: tuple[float, ...] = (1.,),
    chain_device: str | None = None,
) -> tuple[Array, Array, Array, dict]:
    """Fresh continuous tilted Gaussian draws using a Metropolis-corrected pCN chain.

    Endpoints are drawn once and remain fixed through every rejection, as required
    by the disintegration. pCN is reversible for the conditional Gaussian, so the
    MH log ratio contains only lambda dot (f(proposal)-f(current)).
    Optional independent Gaussian-mixture moves use the full proposal-density
    correction and leave the same conditional law invariant. Finite burn-in is
    an approximation and the returned diagnostics expose it.
    """
    started = time.perf_counter()
    if not isinstance(independence_every, int) or independence_every < 0:
        raise ValueError("independence_every must be a nonnegative integer.")
    fractions = np.r_[0., np.asarray(proposal_fractions, dtype=np.float64)]
    if fractions.ndim != 1 or not np.isfinite(fractions).all():
        raise ValueError("Gaussian proposal fractions must be finite scalars.")
    rng = np.random.default_rng(seed)
    pair = rng.choice(len(solution.coupling), size=n_samples, p=solution.coupling / solution.coupling.sum())
    source, target = solution.src[pair], solution.tgt[pair]
    a, b = np.asarray(x0[source], dtype=np.float32), np.asarray(x1[target], dtype=np.float32)
    mean = (1 - tau) * a + tau * b
    sd = sigma * math.sqrt(tau * (1 - tau))
    beta_by_pair = np.full((n_samples, 1), pcn_beta, dtype=np.float32)
    mode_components = None
    if proposal_modes is not None:
        selected_modes = np.asarray(
            proposal_modes(mean) if callable(proposal_modes) else proposal_modes[pair],
            dtype=np.float32,
        )
        multiple_modes = (selected_modes.ndim == 3 and selected_modes.shape[0] == n_samples
                          and selected_modes.shape[2] == mean.shape[1] and selected_modes.shape[1] > 0)
        if (selected_modes.shape != mean.shape and not multiple_modes) or not np.isfinite(selected_modes).all():
            raise ValueError("Sampling proposal centers must match the finite sampled pair means.")
        if multiple_modes:
            mode_components = np.concatenate((np.zeros((n_samples,1,mean.shape[1]),dtype=np.float32),
                                               selected_modes-mean[:,None,:]),axis=1)
            displacement = np.linalg.norm(mode_components[:,1:]/sd,axis=2).min(1)
        else:
            displacement = np.linalg.norm((selected_modes - mean) / sd, axis=1)
        # Pair-dependent but state-independent: Gaussian reversibility and the
        # simple MH ratio remain valid. Large shifts need smaller pCN proposals.
        beta_by_pair[:, 0] = np.minimum(pcn_beta, 2 / np.maximum(displacement, 1.))
    dimension = mean.shape[1]
    displacement_vector = (selected_modes-mean if proposal_modes is not None and mode_components is None
                           else np.zeros_like(mean))
    scaled_displacement = displacement_vector/sd
    squared_displacement = np.einsum('nd,nd->n', scaled_displacement,
                                     scaled_displacement, dtype=np.float64)
    component_count = len(fractions) if mode_components is None else mode_components.shape[1]
    if mode_components is not None:
        scaled_modes = mode_components/sd
        squared_modes = np.einsum('nmd,nmd->nm',scaled_modes,scaled_modes,dtype=np.float64)

    def log_reference_over_proposal(points, first=0, last=None):
        # Leading axes are pair and, optionally, independent proposal candidate.
        local_mean = mean[first:last]
        if mode_components is not None:
            if points.ndim == 3:
                centered=(points-local_mean[:,None,:])/sd
                projection=np.einsum('nqd,nmd->nqm',centered,scaled_modes[first:last],dtype=np.float64)
                shifted_ratio=projection-.5*squared_modes[first:last,None,:]
            else:
                centered=(points-local_mean)/sd
                projection=np.einsum('nd,nmd->nm',centered,scaled_modes[first:last],dtype=np.float64)
                shifted_ratio=projection-.5*squared_modes[first:last]
            return math.log(component_count)-logsumexp(shifted_ratio,axis=-1)
        delta = scaled_displacement[first:last]
        squared_delta = squared_displacement[first:last]
        if points.ndim == 3:
            local_mean, delta, squared_delta = local_mean[:, None], delta[:, None], squared_delta[:, None]
        centered = (points-local_mean)/sd
        projection = np.einsum('...d,...d->...', centered, delta, dtype=np.float64)
        log_ratio_sum = np.full_like(projection, -np.inf)
        for fraction in fractions:
            # Common spherical Gaussian terms cancel analytically. Only the
            # projection on the displacement enters the mixture density ratio.
            shifted_ratio = fraction*projection-.5*fraction**2*squared_delta
            log_ratio_sum = np.logaddexp(log_ratio_sum, shifted_ratio)
        return math.log(len(fractions))-log_ratio_sum

    z = np.empty_like(mean)
    f = np.empty((n_samples, len(solution.multiplier)), dtype=np.float32)
    initial_ess = []
    for start in range(0, n_samples, max(1, 32768 // initial_candidates)):
        end = min(start + max(1, 32768 // initial_candidates), n_samples)
        proposals = mean[start:end, None] + sd * rng.standard_normal(
            (end - start, initial_candidates, dimension), dtype=np.float32
        )
        initial_log_importance = 0.
        if proposal_modes is not None:
            if initial_candidates % component_count:
                raise ValueError("Defensive candidate count must divide into equal mixture components.")
            if mode_components is None:
                component_fraction = np.repeat(fractions, initial_candidates//len(fractions))
                proposals += (displacement_vector[start:end, None]
                              *component_fraction[None, :, None]).astype(np.float32)
            else:
                proposals += np.repeat(mode_components[start:end],initial_candidates//component_count,axis=1)
            initial_log_importance = log_reference_over_proposal(proposals, start, end)
        features = posterior_numpy(posterior, proposals.reshape(-1, dimension)).reshape(
            end - start, initial_candidates, -1
        )
        logits = features @ solution.multiplier + initial_log_importance
        weights = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        idx = (rng.random(end - start)[:, None] > np.cumsum(weights, axis=1)).sum(1)
        idx = np.minimum(idx, initial_candidates - 1)
        z[start:end] = proposals[np.arange(end - start), idx]
        f[start:end] = features[np.arange(end - start), idx]
        initial_ess.extend((1 / np.sum(weights ** 2, axis=1)).tolist())
    log_likelihood = f @ solution.multiplier
    accepted = np.zeros(n_samples, dtype=np.int32)
    independent_accepted = np.zeros(n_samples, dtype=np.int32)
    independent_steps = 0
    summaries = []
    previous_z = z.copy()
    checkpoints = tuple(sorted(set(checkpoints) | {burn_in}))
    if chain_device is not None:
        from .sb_torch_sampler import run_chain
        z,f,accepted,independent_accepted,independent_steps,summaries = run_chain(
            mean=mean,state=z,features=f,multiplier=solution.multiplier,sd=sd,
            beta=beta_by_pair,displacement=displacement_vector,fractions=fractions,
            posterior=posterior,steps=burn_in,checkpoints=checkpoints,
            independence_every=independence_every,seed=seed,device=chain_device,
            mode_components=mode_components)
        for row in summaries:
            residual=np.asarray(row['composition'])-solution.target
            row.update(residual_l2=float(np.linalg.norm(residual)),
                       residual_linf=float(np.max(np.abs(residual))))
    for step in range(1, (burn_in if chain_device is None else 0) + 1):
        independent_move = independence_every > 0 and step % independence_every == 0
        if independent_move:
            component = rng.integers(component_count, size=n_samples)
            shift = (fractions[component, None]*displacement_vector if mode_components is None
                     else mode_components[np.arange(n_samples),component])
            proposal = (mean+shift
                        +sd*rng.standard_normal(z.shape, dtype=np.float32)).astype(np.float32)
            log_proposal_correction = (log_reference_over_proposal(proposal)
                                       -log_reference_over_proposal(z))
            independent_steps += 1
        else:
            proposal = (
                mean + np.sqrt(1 - beta_by_pair ** 2) * (z - mean)
                + beta_by_pair * sd * rng.standard_normal(z.shape, dtype=np.float32)
            )
            log_proposal_correction = 0.
        proposal_f = posterior_numpy(posterior, proposal)
        proposal_log_likelihood = proposal_f @ solution.multiplier
        accept = (np.log(rng.random(n_samples))
                  < proposal_log_likelihood-log_likelihood+log_proposal_correction)
        z[accept] = proposal[accept]
        f[accept] = proposal_f[accept]
        log_likelihood[accept] = proposal_log_likelihood[accept]
        accepted += accept
        if independent_move:
            independent_accepted += accept
        if step in checkpoints:
            comp = f.mean(0, dtype=np.float64)
            summaries.append({
                "step": step, "composition": comp.tolist(),
                "residual_l2": float(np.linalg.norm(comp - solution.target)),
                "residual_linf": float(np.max(np.abs(comp - solution.target))),
                "mean_acceptance": float(accepted.mean() / step),
                "state_change_rms_since_checkpoint": float(np.sqrt(np.mean((z - previous_z) ** 2))),
            })
            previous_z = z.copy()
    comp = f.mean(0, dtype=np.float64)
    standard_error = f.std(0, dtype=np.float64, ddof=1) / math.sqrt(n_samples)
    diagnostics = {
        "sampler": ("pCN and independent Gaussian-mixture Metropolis" if independence_every
                    else "Gaussian-reversible pCN with exact Metropolis acceptance"),
        "chain_device": chain_device or 'numpy_cpu',
        "chain_random_seed": seed+374761393 if chain_device is not None else None,
        "endpoints_held_fixed_during_rejections": True,
        "n_samples": n_samples, "seed": seed, "burn_in": burn_in, "pcn_beta": pcn_beta,
        "pcn_beta_mean": float(beta_by_pair.mean()),
        "pcn_beta_min": float(beta_by_pair.min()),
        "pcn_beta_rule": "min(configured beta, 2 / nearest standardized mode displacement), fixed per endpoint pair",
        "initial_candidates": initial_candidates,
        "independence_every": independence_every,
        "independence_steps": independent_steps,
        "independence_acceptance_mean": (float(independent_accepted.mean()/independent_steps)
                                         if independent_steps else None),
        "proposal_fractions": fractions[1:].tolist() if mode_components is None else None,
        "proposal_components": component_count,
        "initial_proposal": ("defensive Gaussian mixture with exact importance ratio"
                             if proposal_modes is not None else "reference Gaussian"),
        "initial_importance_ess_mean": float(np.mean(initial_ess)),
        "initial_importance_ess_p05": float(np.quantile(initial_ess, .05)),
        "acceptance_mean": float(accepted.mean() / burn_in),
        "acceptance_p05": float(np.quantile(accepted / burn_in, .05)),
        "zero_accepted_fraction": float(np.mean(accepted == 0)),
        "composition": comp.tolist(), "component_standard_errors": standard_error.tolist(),
        "residual": (comp - solution.target).tolist(),
        "residual_l2": float(np.linalg.norm(comp - solution.target)),
        "residual_linf": float(np.abs(comp - solution.target).max()),
        "checkpoints": summaries, "seconds": time.perf_counter() - started,
        "limitation": "Finite MCMC burn-in and numerical Gaussian normalizers; not exact independent draws.",
    }
    return a, z, b, diagnostics


def brownian_subbridge_batch(
    a: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    sigma: float,
    *,
    generator: torch.Generator,
    tau: float = .5,
    time_epsilon: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Analytic probability-flow CFM target, with integrable positive time weights."""
    n = len(a)
    t = torch.rand((n, 1), generator=generator, dtype=a.dtype, device=a.device)
    first_segment = t < tau
    duration = torch.where(first_segment, tau, 1 - tau)
    start = torch.where(first_segment, 0., tau)
    local_t = ((t - start) / duration).clamp(time_epsilon, 1 - time_epsilon)
    t = start + duration * local_t
    left = torch.where(first_segment, a, z)
    right = torch.where(first_segment, z, b)
    mean = (1 - local_t) * left + local_t * right
    variance = sigma ** 2 * duration * local_t * (1 - local_t)
    noise = torch.randn(mean.shape, generator=generator, dtype=a.dtype, device=a.device)
    delta = variance.sqrt() * noise
    state = mean + delta
    velocity = (
        (right - left) / duration
        + (1 - 2 * local_t) / (2 * duration * local_t * (1 - local_t)) * delta
    )
    # Integral of 6 u(1-u) is one. Its positivity preserves the population
    # regression minimizer and removes the infinite variance at pinned times.
    weight = 6 * local_t * (1 - local_t)
    return t, state, velocity, weight


def save_solution(path: Path, solution: MomentBridgeSolution, **metadata: object) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path / "coupling_and_lambda.npz", multiplier=solution.multiplier,
        coupling=solution.coupling, src=solution.src, tgt=solution.tgt,
        target=solution.target, composition=solution.composition, log_z=solution.log_z,
    )
    (path / "solver.json").write_text(
        json.dumps({**solution.metadata(), **metadata}, indent=2) + "\n", encoding="utf-8"
    )
