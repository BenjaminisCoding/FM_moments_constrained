import torch

from cfm_project.constraints import moment_features, moment_features_2d


def test_moment_feature_mean_and_covariance() -> None:
    x = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    feats = moment_features_2d(x)
    expected = torch.tensor([2.0, 3.0, 1.0, 1.0, 1.0, 1.0])
    assert torch.allclose(feats, expected, atol=1e-6)


def test_generic_moment_features_for_3d_input() -> None:
    x = torch.tensor(
        [
            [1.0, 0.0, 2.0],
            [3.0, 2.0, 4.0],
        ]
    )
    feats = moment_features(x)
    expected_mean = torch.tensor([2.0, 1.0, 3.0])
    centered = x - expected_mean
    expected_cov = centered.T @ centered / x.shape[0]
    expected = torch.cat([expected_mean, expected_cov.reshape(-1)], dim=0)
    assert torch.allclose(feats, expected, atol=1e-6)
