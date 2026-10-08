"""Exact Gaussian conditional tilt for one mean/covariance Brownian constraint.

Empirical endpoint marginals are retained in a dense entropic coupling. Only
the endpoint discretization and numerical matrix solve are approximations;
conditional normalizers, moments, and midpoint sampling are analytic.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import numpy as np
from scipy.optimize import brentq, least_squares

from cfm_project.generalized_moment_sb import balance_log_kernel


@dataclass
class QuadraticMomentBridge:
    x0: np.ndarray
    x1: np.ndarray
    coupling: np.ndarray
    sigma: float
    tau: float
    target_mean: np.ndarray
    target_covariance: np.ndarray
    conditional_covariance: np.ndarray
    linear_multiplier: np.ndarray
    quadratic_multiplier: np.ndarray
    diagnostics: dict

    def conditional_mean(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        c = self.sigma ** 2 * self.tau * (1 - self.tau)
        m = (1 - self.tau) * x + self.tau * y
        return (m / c + self.linear_multiplier) @ self.conditional_covariance

    def sample(self, n: int, rng: np.random.Generator):
        pair = rng.choice(self.coupling.size, size=n, p=self.coupling.ravel())
        i, j = np.divmod(pair, len(self.x1))
        x, y = self.x0[i], self.x1[j]
        z = self.conditional_mean(x, y) + rng.standard_normal((n, x.shape[1])) @ np.linalg.cholesky(
            self.conditional_covariance).T
        return x.astype(np.float32), z.astype(np.float32), y.astype(np.float32)


def solve_quadratic_moment_bridge(
    x0: np.ndarray, x1: np.ndarray, target_mean: np.ndarray,
    target_covariance: np.ndarray, *, sigma: float, tau: float = .5,
    tolerance: float = 2e-7, max_evaluations: int = 80,
    progress: Callable[[dict], None] | None = None,
) -> QuadraticMomentBridge:
    """Fit the conditional covariance S and the moment-dependent endpoint coupling.

For c=sigma^2*tau*(1-tau), the optimal conditional mean is
target_mean + S/c * (m_ij - E[m]). Row/column-separable terms in the
modified Brownian kernel cancel, leaving log K_ij=x_i^T S y_j/(sigma^2*c).
The covariance equation is S+(S/c) Cov_pi(m) (S/c)=target_covariance.
Cholesky coordinates maintain normalizability throughout the solve.
"""
    started = time.perf_counter()
    x0, x1 = np.asarray(x0, dtype=np.float64), np.asarray(x1, dtype=np.float64)
    mean = np.asarray(target_mean, dtype=np.float64)
    target = np.asarray(target_covariance, dtype=np.float64)
    if sigma <= 0 or not 0 < tau < 1:
        raise ValueError("A Brownian bridge needs positive sigma and interior tau.")
    if x0.ndim != 2 or x1.ndim != 2 or x0.shape[1] != x1.shape[1]:
        raise ValueError("Endpoint pools must have the same state dimension.")
    d = x0.shape[1]
    if mean.shape != (d,) or target.shape != (d, d):
        raise ValueError("Moment dimensions do not match endpoints.")
    if not all(np.isfinite(v).all() for v in (x0, x1, mean, target)):
        raise ValueError("Endpoint data and moments must be finite.")
    if not np.allclose(target, target.T):
        raise ValueError("Target covariance must be symmetric.")
    target_chol = np.linalg.cholesky(target)
    whitening = np.linalg.inv(target_chol)
    c = sigma ** 2 * tau * (1 - tau)
    e0, e1 = x0.mean(0), x1.mean(0)
    z0, z1 = x0 - e0, x1 - e1
    covariance0, covariance1 = z0.T @ z0 / len(z0), z1.T @ z1 / len(z1)
    fixed_covariance = (1 - tau) ** 2 * covariance0 + tau ** 2 * covariance1
    rows, cols = np.tril_indices(d)
    diagonal = rows == cols

    def unpack(parameters):
        chol = np.zeros((d, d))
        chol[rows, cols] = parameters
        chol[rows[diagonal], cols[diagonal]] = np.exp(parameters[diagonal])
        return chol @ chol.T

    def pack(covariance):
        chol = np.linalg.cholesky(covariance)
        parameters = chol[rows, cols]
        parameters[diagonal] = np.log(parameters[diagonal])
        return parameters

    # Riccati solution with the independent-coupling covariance as initialization.
    eig, vec = np.linalg.eigh(fixed_covariance / c ** 2)
    eig = np.maximum(eig, 1e-12)
    root = (vec * np.sqrt(eig)) @ vec.T
    inverse_root = (vec / np.sqrt(eig)) @ vec.T
    inner_eig, inner_vec = np.linalg.eigh(root @ target @ root)
    y = (inner_vec * (.5 * (np.sqrt(1 + 4 * np.maximum(inner_eig, 0)) - 1))) @ inner_vec.T
    initial = inverse_root @ y @ inverse_root
    state = {"warm": None, "history": []}

    def residual(parameters):
        s = unpack(parameters)
        log_kernel = (z0 @ s) @ z1.T / (sigma ** 2 * c)
        balanced = balance_log_kernel(log_kernel, warm_start=state["warm"], tolerance=1e-10)
        state["warm"] = (balanced.log_u, balanced.log_v)
        cross = z0.T @ balanced.coupling @ z1
        marginal_covariance = fixed_covariance + tau * (1 - tau) * (cross + cross.T)
        actual = s + (s / c) @ marginal_covariance @ (s / c)
        error = actual - target
        scaled = whitening @ error @ whitening.T
        row = dict(evaluation=len(state["history"]) + 1,
                   covariance_residual_fro=float(np.linalg.norm(error)),
                   scaled_residual_fro=float(np.linalg.norm(scaled)),
                   sinkhorn_iterations=balanced.iterations, endpoint_linf=balanced.marginal_linf,
                   seconds=time.perf_counter() - started)
        state["history"].append(row)
        state.update(s=s, coupling=balanced.coupling, covariance=actual)
        if progress is not None:
            progress(row)
        return scaled[rows, cols]

    result = least_squares(residual, pack(initial), diff_step=1e-4,
                           xtol=1e-10, ftol=1e-10, gtol=1e-10,
                           max_nfev=max_evaluations)
    residual(result.x)
    covariance_error = float(np.linalg.norm(state["covariance"] - target))
    if covariance_error > tolerance * max(1., float(np.linalg.norm(target))):
        raise RuntimeError(f"Quadratic SB covariance solve failed: {covariance_error:.3g}")
    s = state["s"]
    b = np.linalg.solve(s, mean) - ((1 - tau) * e0 + tau * e1) / c
    h = .5 * (np.eye(d) / c - np.linalg.inv(s))
    diagnostics = dict(
        covariance_residual_fro=covariance_error, covariance=state["covariance"].tolist(),
        endpoint_linf=state["history"][-1]["endpoint_linf"],
        optimizer_success=bool(result.success), optimizer_message=str(result.message),
        evaluations=len(state["history"]), wall_seconds=time.perf_counter() - started,
        conditional_covariance_eigenvalues=np.linalg.eigvalsh(s).tolist(),
        history=state["history"], empirical_endpoint_discretization=True,
        coupling="dense moment-dependent entropic coupling; no pair truncation",
        moments="analytic Gaussian conditional mean and covariance")
    return QuadraticMomentBridge(x0, x1, state["coupling"], sigma, tau, mean, target, s, b, h, diagnostics)


@dataclass
class CoordinateVarianceMomentBridge:
    """Mean-constrained SB with one observed marginal coordinate variance."""

    x0: np.ndarray
    x1: np.ndarray
    coupling: np.ndarray
    sigma: float
    tau: float
    target_mean: np.ndarray
    target_variance: float
    coordinate: int
    conditional_covariance: np.ndarray
    linear_multiplier: np.ndarray
    quadratic_multiplier: np.ndarray
    diagnostics: dict

    conditional_mean = QuadraticMomentBridge.conditional_mean
    sample = QuadraticMomentBridge.sample


def solve_mean_coordinate_variance_bridge(
    x0: np.ndarray, x1: np.ndarray, target_mean: np.ndarray,
    target_variance: float, *, coordinate: int, sigma: float,
    tau: float = .5, tolerance: float = 2e-7,
    progress: Callable[[dict], None] | None = None,
) -> CoordinateVarianceMomentBridge:
    """Solve without observing any other variances or cross-covariances.

    Only the selected coordinate has a quadratic multiplier. Thus the
    conditional covariance is c*I except for S[j,j], where
    c=sigma**2*tau*(1-tau). The full endpoint coupling is rebalanced at
    every scalar root evaluation. The target mean is enforced analytically;
    row/column-separable linear-tilt terms do not affect that coupling.
    """
    started = time.perf_counter()
    x0, x1 = np.asarray(x0, dtype=np.float64), np.asarray(x1, dtype=np.float64)
    mean = np.asarray(target_mean, dtype=np.float64)
    variance = float(target_variance)
    if x0.ndim != 2 or x1.ndim != 2 or x0.shape[1] != x1.shape[1] or not len(x0) or not len(x1):
        raise ValueError("Nonempty endpoint pools must have the same state dimension.")
    d = x0.shape[1]
    if not isinstance(coordinate, (int, np.integer)) or not 0 <= coordinate < d:
        raise ValueError("Variance coordinate must index the state dimension.")
    if mean.shape != (d,) or not all(np.isfinite(v).all() for v in (x0, x1, mean)):
        raise ValueError("Finite endpoints and a matching finite mean are required.")
    if not np.isfinite([variance, sigma, tau]).all() or variance <= 0 or sigma <= 0 or not 0 < tau < 1:
        raise ValueError("Positive finite variance/diffusion and interior tau are required.")
    c = sigma ** 2 * tau * (1 - tau)
    e0, e1 = x0.mean(0), x1.mean(0)
    z0, z1 = x0 - e0, x1 - e1
    a, b = z0[:, coordinate], z1[:, coordinate]
    fixed_variance = (1 - tau) ** 2 * np.mean(a ** 2) + tau ** 2 * np.mean(b ** 2)
    state = {"warm": None, "history": []}

    def residual(log_s):
        s = np.eye(d) * c
        s[coordinate, coordinate] = sy = float(np.exp(log_s))
        log_kernel = (z0 @ s) @ z1.T / (sigma ** 2 * c)
        balanced = balance_log_kernel(log_kernel, warm_start=state["warm"], tolerance=1e-10)
        state["warm"] = (balanced.log_u, balanced.log_v)
        cross = float(a @ balanced.coupling @ b)
        actual = sy + (sy / c) ** 2 * (fixed_variance + 2 * tau * (1 - tau) * cross)
        row = dict(evaluation=len(state["history"]) + 1, variance_residual=actual - variance,
                   attained_variance=actual, conditional_variance=sy,
                   endpoint_linf=balanced.marginal_linf, sinkhorn_iterations=balanced.iterations,
                   seconds=time.perf_counter() - started)
        state["history"].append(row)
        state.update(s=s, coupling=balanced.coupling, attained_variance=actual)
        if progress is not None:
            progress(row)
        return (actual - variance) / variance

    lower, upper = np.log(variance) - 32., np.log(variance)
    # At S_jj -> 0 the variance vanishes; S_jj=target is an upper bound.
    root = brentq(residual, lower, upper, xtol=1e-10, rtol=1e-12, maxiter=80)
    residual(root)
    error = abs(state["attained_variance"] - variance)
    if error > tolerance * max(1., variance):
        raise RuntimeError(f"Coordinate-variance SB solve failed: {error:.3g}")
    s = state["s"]
    linear = np.linalg.solve(s, mean) - ((1 - tau) * e0 + tau * e1) / c
    quadratic = np.zeros((d, d))
    quadratic[coordinate, coordinate] = .5 * (1 / c - 1 / s[coordinate, coordinate])
    diagnostics = dict(variance_residual_abs=error, attained_variance=state["attained_variance"],
                       observed_covariance_coordinates=[[int(coordinate), int(coordinate)]],
                       endpoint_linf=state["history"][-1]["endpoint_linf"],
                       evaluations=len(state["history"]), wall_seconds=time.perf_counter() - started,
                       history=state["history"], coupling="dense moment-dependent entropic coupling",
                       moments="full mean and one coordinate variance; other covariance entries unobserved")
    return CoordinateVarianceMomentBridge(x0, x1, state["coupling"], sigma, tau, mean,
                                          variance, int(coordinate), s, linear, quadratic, diagnostics)
