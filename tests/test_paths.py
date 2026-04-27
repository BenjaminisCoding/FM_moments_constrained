import torch

from cfm_project.models import PathCorrection
from cfm_project.paths import corrected_path, corrected_velocity


def test_corrected_path_hits_endpoints() -> None:
    torch.manual_seed(0)
    g_model = PathCorrection(state_dim=2, hidden_dims=(16, 16))
    x0 = torch.randn(12, 2)
    x1 = torch.randn(12, 2)
    t0 = torch.zeros(12, 1)
    t1 = torch.ones(12, 1)

    x_start = corrected_path(t0, x0, x1, g_model)
    x_end = corrected_path(t1, x0, x1, g_model)

    assert torch.allclose(x_start, x0, atol=1e-6)
    assert torch.allclose(x_end, x1, atol=1e-6)


def test_corrected_velocity_matches_autodiff_derivative() -> None:
    torch.manual_seed(1)
    g_model = PathCorrection(state_dim=2, hidden_dims=(16, 16))
    x0 = torch.randn(10, 2)
    x1 = torch.randn(10, 2)
    t = torch.rand(10, 1, requires_grad=True)

    u_formula, _, _, _ = corrected_velocity(t, x0, x1, g_model, create_graph=True)
    xt = corrected_path(t, x0, x1, g_model)

    u_auto_parts = []
    for d in range(xt.shape[1]):
        grad_d = torch.autograd.grad(
            xt[:, d].sum(),
            t,
            create_graph=True,
            retain_graph=True,
        )[0]
        u_auto_parts.append(grad_d)
    u_auto = torch.cat(u_auto_parts, dim=1)

    assert torch.allclose(u_formula, u_auto, atol=1e-5, rtol=1e-4)

