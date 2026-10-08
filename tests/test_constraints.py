import torch

from cfm_project.constraints import (
    moment_block_scale,
    moment_block_sizes,
    moment_features,
    residual_blocks_from_samples,
    select_moment_feature_blocks,
    split_moment_feature_vector,
)


def test_moment_feature_mean_and_covariance() -> None:
    x = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    feats = moment_features(x)
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


def test_moment_feature_blocks_select_mean_covariance_and_concatenation() -> None:
    x = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    mean = torch.tensor([2.0, 3.0])
    cov = torch.tensor([[1.0, 1.0], [1.0, 1.0]])

    assert torch.allclose(moment_features(x, feature_blocks=["mean"]), mean)
    assert torch.allclose(
        moment_features(x, feature_blocks=["covariance"]),
        cov.reshape(-1),
    )
    assert torch.allclose(
        moment_features(x, feature_blocks=["mean", "covariance"]),
        moment_features(x),
    )


def test_y_variance_feature_block_is_global_y_axis_variance() -> None:
    x = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [2.0, 4.0],
        ]
    )

    assert torch.allclose(
        moment_features(x, feature_blocks=["y_variance"]),
        torch.tensor([8.0 / 3.0]),
        atol=1e-6,
    )
    assert moment_block_sizes(dim=2, feature_blocks=["y_variance"]) == {"y_variance": 1}
    assert moment_block_scale(block="y_variance", dim=2, normalization="sqrt_dim") == 1.0


def test_y_fourth_central_feature_block_is_centered_fourth_y_moment() -> None:
    x = torch.tensor(
        [
            [0.0, -1.0],
            [1.0, 1.0],
            [2.0, 3.0],
        ]
    )

    expected = torch.tensor([32.0 / 3.0])
    assert torch.allclose(
        moment_features(x, feature_blocks=["y_fourth_central"]),
        expected,
        atol=1e-6,
    )
    assert torch.allclose(
        moment_features(x, feature_blocks=["y_kurtosis"]),
        expected,
        atol=1e-6,
    )
    assert moment_block_sizes(dim=2, feature_blocks=["y_fourth_central"]) == {
        "y_fourth_central": 1
    }
    assert (
        moment_block_scale(block="y_fourth_central", dim=2, normalization="sqrt_dim")
        == 1.0
    )


def test_local_y_variance_feature_block_uses_gaussian_gate() -> None:
    x = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, 3.0],
            [10.0, 100.0],
        ]
    )
    params = {"local_y_variance": {"center_x": 0.0, "width": 0.05}}

    assert torch.allclose(
        moment_features(
            x,
            feature_blocks=["local_y_variance"],
            feature_params=params,
        ),
        torch.tensor([1.0]),
        atol=1e-5,
    )
    assert moment_block_sizes(dim=2, feature_blocks=["local_y_variance"]) == {
        "local_y_variance": 1
    }
    assert moment_block_scale(block="local_y_variance", dim=2, normalization="sqrt_dim") == 1.0


def test_split_moment_feature_vector_respects_selected_blocks() -> None:
    full = torch.tensor([2.0, 3.0, 1.0, 1.0, 1.0, 1.0])
    split = split_moment_feature_vector(
        full,
        dim=2,
        feature_blocks=["mean", "covariance"],
    )
    assert torch.allclose(split["mean"], torch.tensor([2.0, 3.0]))
    assert torch.allclose(split["covariance"], torch.ones(4))

    mean_only = split_moment_feature_vector(
        full[:2],
        dim=2,
        feature_blocks=["mean"],
    )
    assert list(mean_only) == ["mean"]
    assert torch.allclose(mean_only["mean"], torch.tensor([2.0, 3.0]))


def test_select_moment_feature_blocks_extracts_subset_from_full_target() -> None:
    target = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    selected = select_moment_feature_blocks(
        feature=target,
        dim=2,
        source_blocks=["covariance", "y_fourth_central"],
        selected_blocks=["covariance"],
    )

    assert torch.allclose(selected, torch.tensor([1.0, 2.0, 3.0, 4.0]))

    selected_fourth = select_moment_feature_blocks(
        feature=target,
        dim=2,
        source_blocks=["covariance", "y_fourth_central"],
        selected_blocks=["y_fourth_central"],
    )
    assert torch.allclose(selected_fourth, torch.tensor([5.0]))


def test_block_normalization_scale_is_inverse_sqrt_block_dim() -> None:
    assert abs(moment_block_scale(block="mean", dim=2, normalization="sqrt_dim") - 2**-0.5) <= 1e-12
    assert abs(moment_block_scale(block="covariance", dim=2, normalization="sqrt_dim") - 4**-0.5) <= 1e-12
    assert moment_block_scale(block="mean", dim=2, normalization="none") == 1.0


def test_residual_blocks_from_samples_reports_each_block() -> None:
    x = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    target = torch.zeros(6)
    residuals = residual_blocks_from_samples(
        x,
        target,
        feature_blocks=["mean", "covariance"],
    )
    assert torch.allclose(residuals["mean"], torch.tensor([2.0, 3.0]))
    assert torch.allclose(residuals["covariance"], torch.ones(4))


def test_residual_blocks_from_samples_reports_scalar_variance_blocks() -> None:
    x = torch.tensor(
        [
            [0.0, 1.0],
            [0.0, 3.0],
            [10.0, 100.0],
        ]
    )
    params = {"local_y_variance": {"center_x": 0.0, "width": 0.05}}
    target = torch.tensor([0.5, 0.25])

    residuals = residual_blocks_from_samples(
        x,
        target,
        feature_blocks=["y_variance", "local_y_variance"],
        feature_params=params,
    )

    expected_y_var = torch.var(x[:, 1], unbiased=False).reshape(1) - target[:1]
    assert torch.allclose(residuals["y_variance"], expected_y_var, atol=1e-6)
    assert torch.allclose(residuals["local_y_variance"], torch.tensor([0.75]), atol=1e-5)


def test_residual_blocks_from_samples_reports_y_variance_plus_fourth_central() -> None:
    x = torch.tensor(
        [
            [0.0, -1.0],
            [1.0, 1.0],
            [2.0, 3.0],
        ]
    )
    target = torch.tensor([1.0, 10.0])

    residuals = residual_blocks_from_samples(
        x,
        target,
        feature_blocks=["y_variance", "y_fourth_central"],
    )

    assert torch.allclose(residuals["y_variance"], torch.tensor([8.0 / 3.0 - 1.0]))
    assert torch.allclose(
        residuals["y_fourth_central"],
        torch.tensor([32.0 / 3.0 - 10.0]),
        atol=1e-6,
    )


def test_residual_blocks_from_samples_reports_covariance_plus_fourth_central() -> None:
    x = torch.tensor(
        [
            [0.0, -1.0],
            [1.0, 1.0],
            [2.0, 3.0],
        ]
    )
    target = torch.zeros(5)

    residuals = residual_blocks_from_samples(
        x,
        target,
        feature_blocks=["covariance", "y_fourth_central"],
    )

    expected_cov = torch.tensor([[2.0 / 3.0, 4.0 / 3.0], [4.0 / 3.0, 8.0 / 3.0]])
    assert torch.allclose(residuals["covariance"], expected_cov.reshape(-1), atol=1e-6)
    assert torch.allclose(residuals["y_fourth_central"], torch.tensor([32.0 / 3.0]))
