import torch

from cfm_project.data import (
    GaussianOTProblem,
    exact_discrete_ot_pairs,
    random_discrete_pairs,
    sample_coupled_batch,
)


def test_exact_discrete_ot_assignment_properties() -> None:
    torch.manual_seed(4)
    n = 7
    x0 = torch.randn(n, 2)
    x1 = torch.randn(n, 2)

    paired_x0, paired_x1, ot_cost = exact_discrete_ot_pairs(x0, x1)

    assert paired_x0.shape == x0.shape
    assert paired_x1.shape == x1.shape
    assert ot_cost >= 0.0

    random_cost = torch.sum((x0 - x1) ** 2).item()
    paired_cost = torch.sum((paired_x0 - paired_x1) ** 2).item()
    assert paired_cost <= random_cost + 1e-8


def test_random_discrete_pairs_is_a_permutation() -> None:
    torch.manual_seed(5)
    n = 9
    x0 = torch.randn(n, 2)
    ids = torch.arange(n, dtype=torch.float32)
    x1 = torch.stack([ids, ids**2], dim=1)
    generator = torch.Generator(device=x0.device)
    generator.manual_seed(42)

    paired_x0, paired_x1, random_cost = random_discrete_pairs(x0, x1, generator=generator)

    assert paired_x0.shape == x0.shape
    assert paired_x1.shape == x1.shape
    assert random_cost >= 0.0
    assert torch.allclose(torch.sort(paired_x1[:, 0]).values, torch.sort(x1[:, 0]).values)
    assert not torch.allclose(paired_x1, x1)


def test_sample_coupled_batch_rejects_unknown_coupling() -> None:
    problem = GaussianOTProblem(
        mean0=torch.zeros(2),
        cov0=torch.eye(2),
        mean1=torch.ones(2),
        cov1=torch.eye(2),
        kappa=0.5,
    )

    try:
        sample_coupled_batch(problem, batch_size=8, coupling="bad_mode")
    except ValueError as exc:
        assert "Unsupported coupling" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported coupling mode.")
