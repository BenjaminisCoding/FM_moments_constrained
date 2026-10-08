from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.integrate import quad as integrate
from scipy.special import expit
import torch

from cfm_project.generalized_moment_sb import (
    balance_log_kernel, brownian_subbridge_batch,
    sample_continuous_tilt,
    conditional_feature_covariance,
)

from reference_quadrature_bridge import PairQuadrature, RefinedPairQuadrature, build_pair_quadrature, solve_moment_bridge


def posterior(x):
    p = torch.sigmoid(2 * x[:, :1])
    return torch.cat((p, 1 - p), 1)


def test_conditional_covariance_excludes_between_endpoint_variation():
    fixed_features = np.array([[.1,.9],[.1,.9],[.8,.2],[.8,.2]])
    indices = np.array([5,5,19,19])
    np.testing.assert_allclose(conditional_feature_covariance(fixed_features,indices),0.,atol=1e-15)
    assert np.var(fixed_features[:,0]) > .1
    # Each conditional Bernoulli has sample variance 0.5. The between-pair
    # covariance must not enter the Hessian of the fixed-coupling dual.
    mixed = np.array([[0.,1.],[1.,0.],[0.,1.],[1.,0.]])
    np.testing.assert_allclose(conditional_feature_covariance(mixed,indices),
                               [[.5,-.5],[-.5,.5]])


@pytest.mark.parametrize("relaxation,stabilize_every", [(1., 0), (1.7, 25)])
def test_sinkhorn_preserves_unequal_empirical_marginals_and_kernel_gauge(relaxation, stabilize_every):
    rng = np.random.default_rng(7)
    kernel = rng.normal(size=(4, 7)) * 3
    options = dict(relaxation=relaxation, stabilize_every=stabilize_every)
    result = balance_log_kernel(kernel, **options)
    shifted = balance_log_kernel(kernel + np.arange(4)[:, None] - np.arange(7)[None, :], **options)
    np.testing.assert_allclose(result.coupling.sum(1), .25, atol=1e-8)
    np.testing.assert_allclose(result.coupling.sum(0), 1 / 7, atol=1e-8)
    np.testing.assert_allclose(result.coupling, shifted.coupling, atol=1e-8)


def test_stabilization_recovers_required_mass_on_initially_underflowed_edges():
    log_kernel = np.array([[0., -1000., -1000.], [-1000., 0., 0.]])
    result = balance_log_kernel(log_kernel, relaxation=1.7, stabilize_every=50)
    expected = np.array([[1/3, 1/12, 1/12], [0., 1/4, 1/4]])
    np.testing.assert_allclose(result.coupling, expected, atol=1e-8)
    reconstructed = np.exp(log_kernel + result.log_u[:, None] + result.log_v[None, :])
    np.testing.assert_allclose(reconstructed, result.coupling, atol=1e-12)


def test_newton_refines_nearly_disconnected_equal_mass_endpoints():
    log_kernel = np.array([[0., -50.], [-1., 0.]])
    result = balance_log_kernel(log_kernel, max_iterations=100,
                                stabilize_every=25, relaxation=1.7, newton_refine=True)
    offdiag = .5 / (1 + np.exp(25.5))
    expected = np.array([[.5-offdiag, offdiag], [offdiag, .5-offdiag]])
    np.testing.assert_allclose(result.coupling, expected, atol=1e-8)
    assert result.newton_iterations > 0
    assert result.marginal_relative_linf <= 2e-8
    reconstructed = np.exp(log_kernel + result.log_u[:, None] + result.log_v[None, :])
    np.testing.assert_allclose(reconstructed, result.coupling, atol=1e-12)


def test_temperature_continuation_recovers_required_extremely_small_kernel_edges():
    log_kernel = np.array([[0.,-10000.,-10000.], [0.,-10000.,-10000.], [-10000.,0.,0.]])
    result = balance_log_kernel(log_kernel,max_iterations=100,
                                stabilize_every=25,relaxation=1.7,newton_refine=True)
    expected = np.array([[1/6,1/12,1/12],[1/6,1/12,1/12],[0.,1/6,1/6]])
    np.testing.assert_allclose(result.coupling,expected,atol=1e-8)
    assert result.continuation_stages > 0
    assert result.marginal_relative_linf <= 2e-8
    reconstructed = np.exp(log_kernel+result.log_u[:,None]+result.log_v[None,:])
    np.testing.assert_allclose(reconstructed,result.coupling,atol=1e-10)


@pytest.mark.parametrize("independent_pair_noise", [False, True])
@pytest.mark.parametrize("stop_on_feasibility", [False, True])
@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("coordinate_scale", [None,np.array([.2])])
def test_recomputed_bridge_matches_independent_joint_entropy_dual(independent_pair_noise, stop_on_feasibility, adaptive, coordinate_scale):
    # Compare nested moment-dual/Sinkhorn against a single optimization of all
    # endpoint and moment potentials on the complete three-variable law.
    x0 = np.array([[-1.1], [.9]], dtype=np.float32)
    x1 = np.array([[-.7], [.3], [1.2]], dtype=np.float32)
    quad = build_pair_quadrature(
        x0=x0, x1=x1, posterior=posterior, sigma=1.1, n_quadrature=32, seed=5,
        independent_pair_noise=independent_pair_noise,
    )
    f = np.asarray(quad.features, dtype=np.float64)
    if adaptive:
        ids = np.array([1, 4])
        fine = build_pair_quadrature(
            x0=x0, x1=x1, src=quad.src[ids], tgt=quad.tgt[ids], posterior=posterior,
            sigma=1.1, n_quadrature=64, seed=71, independent_pair_noise=True)
        # Independent joint law: repeat coarse atoms to a common denominator.
        f = np.repeat(f, 2, axis=1)
        f[ids] = fine.features
        quad = RefinedPairQuadrature(quad, fine, ids)
    target = np.array([.67, .33])
    result = solve_moment_bridge(quad, target, moment_tolerance=1e-7,
                                 stop_on_feasibility=stop_on_feasibility,
                                 dual_coordinate_scale=coordinate_scale)
    f = f / f.sum(-1, keepdims=True)
    log_reference = (
        -((x0[:, None, 0] - x1[None, :, 0]) ** 2) / (2 * 1.1 ** 2)
    ).astype(np.float64).reshape(-1, 1) - np.log(f.shape[1])

    def joint(v):
        alpha, beta, multiplier = v[:2], np.r_[v[2:4], 0.], v[4]
        mass = np.exp(
            log_reference + alpha[quad.src, None] + beta[quad.tgt, None]
            + multiplier * f[:, :, 0]
        )
        pairs = mass.sum(1)
        value = mass.sum() - alpha.mean() - beta.mean() - multiplier * target[0]
        gradient = np.r_[
            np.bincount(quad.src, pairs, minlength=2) - .5,
            (np.bincount(quad.tgt, pairs, minlength=3) - 1 / 3)[:2],
            np.sum(mass * f[:, :, 0]) - target[0],
        ]
        return value, gradient

    joint_result = minimize(joint, np.zeros(5), jac=True, method="BFGS",
                            options={"gtol": 1e-9, "maxiter": 500})
    alpha, beta = joint_result.x[:2], np.r_[joint_result.x[2:4], 0.]
    mass = np.exp(log_reference + alpha[quad.src, None] + beta[quad.tgt, None]
                  + joint_result.x[4] * f[:, :, 0])
    assert result.converged
    np.testing.assert_allclose(result.coupling, mass.sum(1), atol=1e-6)
    np.testing.assert_allclose(result.multiplier[0], joint_result.x[4], atol=2e-5)
    np.testing.assert_allclose(result.composition, target, atol=1e-7)
    plain = balance_log_kernel(log_reference[:, 0].reshape(2, 3)).coupling
    assert np.linalg.norm(result.coupling.reshape(2, 3) - plain) > 1e-4


def test_multiplier_safeguard_does_not_accept_infeasible_finite_grid():
    quad = PairQuadrature(
        x0=np.zeros((1,1)), x1=np.zeros((1,1)), src=np.array([0]), tgt=np.array([0]),
        sigma=1., tau=.5, features=np.array([[[.4,.6],[.6,.4]]]),
        noise=np.empty((0,1)), signs=np.empty((0,1)))
    result = solve_moment_bridge(quad,np.array([.9,.1]),multiplier_bound=2.,
                                 stop_on_multiplier_bound=True)
    assert not result.converged and not result.optimizer_success
    assert result.composition[0] <= .6
    assert 'bound reached' in result.optimizer_message
    np.testing.assert_allclose(result.coupling, [1.])


def test_fixed_coupling_stays_fixed_and_continuous_sampler_keeps_endpoints():
    x0 = np.array([[-.4], [.5]], dtype=np.float32)
    x1 = np.array([[-.2], [.8]], dtype=np.float32)
    quad = build_pair_quadrature(
        x0=x0, x1=x1, src=np.array([0, 1]), tgt=np.array([0, 1]),
        posterior=posterior, sigma=1., n_quadrature=1024, seed=11,
    )
    mass = np.array([.5, .5])
    result = solve_moment_bridge(quad, np.array([.63, .37]), fixed_coupling=mass)
    np.testing.assert_array_equal(result.coupling, mass)
    assert result.converged
    kwargs = dict(x0=x0, x1=x1, solution=result, posterior=posterior,
                  sigma=1., n_samples=5000, seed=91)
    a, z, b, diag = sample_continuous_tilt(**kwargs, burn_in=64)
    a2, _, b2, _ = sample_continuous_tilt(**kwargs, burn_in=128)
    _, _, _, defensive = sample_continuous_tilt(
        **kwargs, burn_in=128, proposal_modes=np.array([[.1], [.9]], dtype=np.float32)
    )
    np.testing.assert_array_equal(a, a2)
    np.testing.assert_array_equal(b, b2)
    assert diag["residual_linf"] < .02
    assert diag["acceptance_mean"] > .1
    assert defensive["residual_linf"] < .02
    assert np.unique(z).size > 4900  # continuous draws, not quadrature atoms
    mode_array = .5 * (x0 + x1) + .3
    array_draws = sample_continuous_tilt(**kwargs, burn_in=64, proposal_modes=mode_array)
    streamed_draws = sample_continuous_tilt(
        **kwargs, burn_in=64, proposal_modes=lambda means: means + .3)
    for actual, expected in zip(streamed_draws[:3], array_draws[:3]):
        np.testing.assert_array_equal(actual, expected)


def test_subbridge_target_is_probability_flow_not_forward_sde_drift():
    generator = torch.Generator().manual_seed(5)
    a, z, b = (torch.randn(200, 2, generator=generator) for _ in range(3))
    t, state, velocity, weight = brownian_subbridge_batch(
        a, z, b, .7, generator=generator
    )
    first = t < .5
    start, end = torch.where(first, 0., .5), torch.where(first, .5, 1.)
    left, right = torch.where(first, a, z), torch.where(first, z, b)
    expected = .5 * ((right - state) / (end - t) + (state - left) / (t - start))
    torch.testing.assert_close(velocity, expected, atol=2e-4, rtol=2e-5)
    assert torch.all(weight > 0)
    assert not torch.allclose(velocity, (right - state) / (end - t))


@pytest.mark.parametrize("stream_centers", [False, True])
@pytest.mark.parametrize("fractions", [None, (1/3,2/3,1.)])
def test_defensive_importance_quadrature_matches_continuous_integral(stream_centers,fractions):
    quadrature = build_pair_quadrature(
        x0=np.array([[-.3]], dtype=np.float32),
        x1=np.array([[.6]], dtype=np.float32),
        posterior=posterior, sigma=1.2, n_quadrature=16384, seed=19,
        proposal_multiplier=np.array([6., 0.]), proposal_steps=32,
        proposal_center_fn=(lambda means: means + .9) if stream_centers else None,
        proposal_fractions=fractions,
    )
    log_z, mean = quadrature.log_normalizers_and_means(np.array([6., 0.]))
    reference = lambda z: np.exp(-.5 * ((z - .15) / .6) ** 2) / (.6 * np.sqrt(2 * np.pi))
    normalizer = integrate(lambda z: reference(z) * np.exp(6 * expit(2 * z)), -8, 8)[0]
    moment = integrate(lambda z: reference(z) * np.exp(6 * expit(2 * z)) * expit(2 * z), -8, 8)[0] / normalizer
    np.testing.assert_allclose(log_z[0], np.log(normalizer), atol=8e-4)
    np.testing.assert_allclose(mean[0, 0], moment, atol=8e-4)


@pytest.mark.parametrize('chain_device',[None,'mps'])
@pytest.mark.parametrize('multiple_modes',[False,True])
def test_independent_mixture_moves_preserve_reference_gaussian(chain_device,multiple_modes):
    if chain_device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('Requires Metal')
    endpoints = np.zeros((1, 1), dtype=np.float32)
    quadrature = build_pair_quadrature(x0=endpoints, x1=endpoints, posterior=posterior,
                                       sigma=2., n_quadrature=512, seed=13)
    solution = solve_moment_bridge(quadrature, np.array([.5, .5]),
                                    fixed_coupling=np.ones(1))
    a, z, b, audit = sample_continuous_tilt(
        x0=endpoints, x1=endpoints, solution=solution, posterior=posterior,
        sigma=2., n_samples=12000, seed=113, burn_in=96,
        proposal_modes=(lambda means: np.stack((means+5.,means-2.),axis=1)) if multiple_modes else (lambda means: means+5.),
        independence_every=1,initial_candidates=48 if multiple_modes else 32,
        proposal_fractions=(1/3, 2/3, 1.),chain_device=chain_device)
    np.testing.assert_array_equal(a, np.zeros_like(a))
    np.testing.assert_array_equal(b, np.zeros_like(b))
    # A proposal with mean 2.5 must still reproduce N(0,1), not its own law.
    assert abs(z.mean()) < .035
    assert abs(z.var()-1.) < .055
    assert audit['independence_acceptance_mean'] > .1


@pytest.mark.parametrize('chain_device',[None,'mps'])
@pytest.mark.parametrize('multiple_modes',[False,True])
def test_independent_jumps_sample_both_modes_of_nonlinear_tilt(chain_device,multiple_modes):
    if chain_device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('Requires Metal')
    def bimodal_posterior(x):
        p = torch.exp(-.5*(x[:, :1].square()-4.).square())
        return torch.cat((p, 1-p), 1)
    endpoints = np.zeros((1, 1), dtype=np.float32)
    quadrature = build_pair_quadrature(
        x0=endpoints, x1=endpoints, posterior=bimodal_posterior,
        sigma=2., n_quadrature=16384, seed=19)
    feature = lambda x: np.exp(-.5*(x*x-4.)**2)
    density = lambda x: np.exp(-.5*x*x+8*feature(x))
    normalizer = integrate(density, -8., 8., points=[-2., 0., 2.])[0]
    moment = integrate(lambda x: density(x)*feature(x), -8., 8., points=[-2., 0., 2.])[0]/normalizer
    variance = integrate(lambda x: density(x)*x*x, -8., 8., points=[-2., 0., 2.])[0]/normalizer
    solution = solve_moment_bridge(quadrature, np.array([moment, 1-moment]),
                                    fixed_coupling=np.ones(1))
    _, z, _, audit = sample_continuous_tilt(
        x0=endpoints, x1=endpoints, solution=solution, posterior=bimodal_posterior,
        sigma=2., n_samples=12000, seed=127, burn_in=512,
        proposal_modes=(lambda means: np.stack((means+2.,means-2.),axis=1)) if multiple_modes else (lambda means: means+2.),
        initial_candidates=48 if multiple_modes else 32,independence_every=2,chain_device=chain_device)
    assert abs(z.mean()) < .065
    assert abs(z.var()-variance) < .1
    assert abs((z[:, 0] > 0).mean()-.5) < .02
    assert audit['residual_linf'] < .01
