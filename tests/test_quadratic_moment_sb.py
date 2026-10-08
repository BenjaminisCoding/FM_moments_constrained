import numpy as np
from numpy.polynomial.hermite import hermgauss
from cfm_project.generalized_moment_sb import balance_log_kernel
from cfm_project.quadratic_moment_sb import solve_quadratic_moment_bridge


def test_single_endpoint_pair_has_prescribed_gaussian_midpoint():
    bridge = solve_quadratic_moment_bridge(
        np.array([[-1., .4]]), np.array([[2., -.3]]),
        np.array([.8, .6]), np.array([[.2, .04], [.04, .1]]), sigma=.7)
    np.testing.assert_allclose(bridge.conditional_covariance, bridge.target_covariance, atol=1e-7)
    np.testing.assert_allclose(bridge.conditional_mean(bridge.x0, bridge.x1)[0], bridge.target_mean, atol=1e-9)
    _, z, _ = bridge.sample(60000, np.random.default_rng(6))
    np.testing.assert_allclose(z.mean(0), bridge.target_mean, atol=.006)
    np.testing.assert_allclose(np.cov(z.T, bias=True), bridge.target_covariance, atol=.004)


def test_recovers_independent_gauss_hermite_tilt_and_modified_coupling():
    x = np.array([[-1.], [.1], [1.]])
    y = np.array([[-.5], [.3], [1.1], [1.7]])
    sigma, tau, b, h = 1.1, .35, .7, -1.4
    c = sigma ** 2 * tau * (1 - tau)
    nodes, weights = hermgauss(80)
    m = (1 - tau) * x[:, None, 0] + tau * y[None, :, 0]
    z = m[..., None] + np.sqrt(2 * c) * nodes
    tilt = np.exp(b * z + h * z ** 2) * weights / np.sqrt(np.pi)
    normalizer = tilt.sum(-1)
    conditional_mean = (tilt * z).sum(-1) / normalizer
    conditional_second = (tilt * z ** 2).sum(-1) / normalizer
    log_kernel = -(x[:, None, 0] - y[None, :, 0]) ** 2 / (2 * sigma ** 2) + np.log(normalizer)
    expected = balance_log_kernel(log_kernel).coupling
    mean = np.sum(expected * conditional_mean)
    covariance = np.sum(expected * conditional_second) - mean ** 2
    bridge = solve_quadratic_moment_bridge(x, y, np.array([mean]), np.array([[covariance]]), sigma=sigma, tau=tau)
    np.testing.assert_allclose(bridge.coupling, expected, atol=2e-8)
    np.testing.assert_allclose(bridge.linear_multiplier, [b], atol=1e-7)
    np.testing.assert_allclose(bridge.quadratic_multiplier, [[h]], atol=1e-7)
