import torch

from cfm_project.constraints import moment_features_2d


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

