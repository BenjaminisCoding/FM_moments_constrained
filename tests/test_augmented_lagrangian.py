import torch

from cfm_project.constraints import augmented_lagrangian_terms, update_lagrange_multipliers


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

