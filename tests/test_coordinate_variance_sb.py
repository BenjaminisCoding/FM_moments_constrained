import numpy as np
import pytest

from cfm_project.quadratic_moment_sb import (
    solve_mean_coordinate_variance_bridge, solve_quadratic_moment_bridge,
)


def moments(bridge):
    x, y = bridge.x0[:, None, :], bridge.x1[None, :, :]
    conditional = bridge.conditional_mean(x, y)
    mean = np.einsum("ij,ijd->d", bridge.coupling, conditional)
    centered = conditional - mean
    covariance = bridge.conditional_covariance + np.einsum("ij,ijd,ije->de", bridge.coupling, centered, centered)
    return mean, covariance


def test_partial_constraint_and_endpoint_marginals():
    rng = np.random.default_rng(42)
    x, y = rng.normal(size=(7, 2)), rng.normal(size=(9, 2))
    target = np.array([.4, 1.2])
    bridge = solve_mean_coordinate_variance_bridge(x, y, target, .18, coordinate=1, sigma=.7)
    mean, covariance = moments(bridge)
    np.testing.assert_allclose(bridge.coupling.sum(1), 1 / 7, atol=1e-9)
    np.testing.assert_allclose(bridge.coupling.sum(0), 1 / 9, atol=1e-9)
    np.testing.assert_allclose(mean, target, atol=1e-8)
    np.testing.assert_allclose(covariance[1, 1], .18, atol=1e-8)
    assert bridge.conditional_covariance[0, 0] == .7 ** 2 / 4
    assert np.count_nonzero(bridge.quadratic_multiplier) == 1
    # Adding constraints at this solution's attained covariance preserves the optimum.
    full = solve_quadratic_moment_bridge(x, y, target, covariance, sigma=.7)
    np.testing.assert_allclose(bridge.coupling, full.coupling, atol=2e-7)
    np.testing.assert_allclose(bridge.conditional_covariance, full.conditional_covariance, atol=2e-7)


def test_mean_translation_does_not_change_covariance_or_coupling():
    rng = np.random.default_rng(13)
    x, y = rng.normal(size=(6, 2)), rng.normal(size=(8, 2))
    first = solve_mean_coordinate_variance_bridge(x, y, np.zeros(2), .4, coordinate=1, sigma=1.)
    shifted = solve_mean_coordinate_variance_bridge(x, y, np.array([1., 2.]), .4, coordinate=1, sigma=1.)
    np.testing.assert_allclose(first.coupling, shifted.coupling)
    np.testing.assert_allclose(moments(first)[1], moments(shifted)[1])
    np.testing.assert_allclose(moments(shifted)[0], [1., 2.], atol=1e-8)


@pytest.mark.parametrize("variance,coordinate", [(0., 1), (-1., 1), (np.nan, 1), (.2, 2), (.2, -1)])
def test_invalid_observations_rejected(variance, coordinate):
    with pytest.raises(ValueError):
        solve_mean_coordinate_variance_bridge(np.zeros((3, 2)), np.zeros((4, 2)),
                                              np.zeros(2), variance, coordinate=coordinate, sigma=.5)
