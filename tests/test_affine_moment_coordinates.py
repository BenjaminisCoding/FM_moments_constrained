from __future__ import annotations

import pytest
import torch

from cfm_project.constraints import moment_features


def test_affine_moment_coordinates_match_explicit_sample_transform() -> None:
    samples = torch.tensor(
        [[-1.0, 0.5], [0.0, 2.0], [2.0, -0.5], [1.0, 1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    matrix = torch.tensor([[2.0, -0.25], [0.5, 1.5]], dtype=torch.float64)
    center = torch.tensor([0.25, -0.75], dtype=torch.float64)
    params = {
        "affine_moment_coordinates": {
            "matrix": matrix.tolist(),
            "center": center.tolist(),
        }
    }

    actual = moment_features(samples, feature_params=params)
    transformed = (samples - center) @ matrix.T
    expected = moment_features(transformed)

    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    assert samples.grad is not None
    assert torch.isfinite(samples.grad).all()


def test_affine_moment_coordinates_reject_bad_matrix_shape() -> None:
    samples = torch.zeros((4, 2))
    params = {
        "affine_moment_coordinates": {
            "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "center": [0.0, 0.0],
        }
    }
    with pytest.raises(ValueError, match="must have shape"):
        moment_features(samples, feature_params=params)


def test_moment_block_scales_apply_to_features_and_targets() -> None:
    samples = torch.tensor([[0.0, 1.0], [2.0, 3.0], [1.0, -1.0]])
    unscaled = moment_features(samples)
    scaled = moment_features(
        samples,
        feature_params={"moment_block_scales": {"mean": 2.0, "covariance": 0.5}},
    )
    torch.testing.assert_close(scaled[:2], 2.0 * unscaled[:2])
    torch.testing.assert_close(scaled[2:], 0.5 * unscaled[2:])
