import torch

from cfm_project.constraints import (
    augmented_lagrangian_block_terms,
    augmented_lagrangian_terms,
    update_lagrange_multiplier_blocks,
    update_lagrange_multipliers,
)
from cfm_project.training import _lagrange_multiplier_diagnostics


def test_augmented_lagrangian_and_multiplier_update() -> None:
    rho = 2.0
    residuals = {
        0.25: torch.tensor([1.0, -2.0]),
        0.50: torch.tensor([0.5, 0.5]),
    }
    lambdas = {
        0.25: torch.zeros(2),
        0.50: torch.zeros(2),
    }
    total, _ = augmented_lagrangian_terms(residuals=residuals, lambdas=lambdas, rho=rho)

    expected = 0.5 * rho * (
        torch.dot(residuals[0.25], residuals[0.25]) + torch.dot(residuals[0.50], residuals[0.50])
    )
    assert torch.allclose(total, expected)

    updated = update_lagrange_multipliers(
        lambdas=lambdas,
        residuals=residuals,
        rho=rho,
        clip_value=None,
    )
    assert torch.allclose(updated[0.25], rho * residuals[0.25])
    assert torch.allclose(updated[0.50], rho * residuals[0.50])


def test_block_augmented_lagrangian_averages_active_blocks_and_updates_independently() -> None:
    rho = 2.0
    residuals = {
        0.50: {
            "mean": torch.tensor([1.0, -1.0]),
            "covariance": torch.tensor([0.5, 0.0, 0.0, -0.5]),
        }
    }
    lambdas = {
        0.50: {
            "mean": torch.zeros(2),
            "covariance": torch.zeros(4),
        }
    }

    total, per_time, per_block = augmented_lagrangian_block_terms(
        residuals=residuals,
        lambdas=lambdas,
        rho=rho,
    )
    mean_term = 0.5 * rho * torch.dot(residuals[0.50]["mean"], residuals[0.50]["mean"])
    cov_term = 0.5 * rho * torch.dot(
        residuals[0.50]["covariance"],
        residuals[0.50]["covariance"],
    )
    expected = (mean_term + cov_term) / 2.0
    assert torch.allclose(total, expected)
    assert abs(per_time[0.50] - float(expected.item())) <= 1e-6
    assert abs(per_block["mean"][0.50] - float(mean_term.item())) <= 1e-6
    assert abs(per_block["covariance"][0.50] - float(cov_term.item())) <= 1e-6

    updated = update_lagrange_multiplier_blocks(
        lambdas=lambdas,
        residuals=residuals,
        rho=rho,
        clip_value=None,
    )
    assert torch.allclose(updated[0.50]["mean"], rho * residuals[0.50]["mean"])
    assert torch.allclose(
        updated[0.50]["covariance"],
        rho * residuals[0.50]["covariance"],
    )


def test_lagrange_multiplier_diagnostics_support_flat_and_block_states() -> None:
    flat = {0.50: torch.tensor([3.0, -4.0])}
    flat_stats = _lagrange_multiplier_diagnostics(flat, clip_value=4.0)
    assert flat_stats == {"l2": 5.0, "linf": 4.0, "clip_fraction": 0.5, "numel": 2}

    blocked = {
        0.50: {
            "mean": torch.tensor([1.0, -1.0]),
            "covariance": torch.tensor([2.0, 0.0]),
        }
    }
    block_stats = _lagrange_multiplier_diagnostics(blocked, clip_value=10.0)
    assert abs(float(block_stats["l2"]) - 6.0**0.5) <= 1e-6
    assert block_stats["linf"] == 2.0
    assert block_stats["clip_fraction"] == 0.0
    assert block_stats["numel"] == 4
