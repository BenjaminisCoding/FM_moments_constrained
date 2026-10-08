import torch
from torch import nn

from cfm_project.paths import corrected_path
from cfm_project.semrau_benchmark import path_velocity
from cfm_project.semrau_weighted_positive import PositiveMarkerCorrection


def test_positive_path_endpoint_velocities_and_gradients():
    torch.manual_seed(39)
    model = PositiveMarkerCorrection(3, 10, 2, positive_start=1).double()
    x0 = torch.tensor([[-1., 0., 0.], [2., 0., 1.], [0., 2., 0.]], dtype=torch.float64)
    x1 = torch.tensor([[1., 0., 0.], [-2., 2., 0.], [3., 0., 1.]], dtype=torch.float64)
    t = torch.tensor([[.2], [.4], [.7]], dtype=torch.float64)
    x, velocity = path_velocity(model, t, x0, x1)
    eps = 1e-6
    finite = (corrected_path(t+eps, x0, x1, model)-corrected_path(t-eps, x0, x1, model))/(2*eps)
    torch.testing.assert_close(velocity, finite, atol=2e-8, rtol=2e-8)
    assert x[:, 1:].min() >= -1e-12
    loss = velocity.square().mean() + x.square().mean()
    for value, expected in [(0., x0), (1., x1)]:
        endpoint, endpoint_velocity = path_velocity(model, torch.full_like(t, value), x0, x1)
        torch.testing.assert_close(endpoint, expected, rtol=0, atol=0)
        assert torch.isfinite(endpoint_velocity).all()
        loss = loss + endpoint_velocity.square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_large_negative_corrections_cannot_create_negative_markers():
    class Constant(nn.Module):
        def forward(self, t, x0, x1):
            return torch.full_like(x0, -100.)
    model = PositiveMarkerCorrection(3, 4, 1, positive_start=1).double()
    model.raw = Constant()
    t = torch.linspace(0, 1, 101, dtype=torch.float64).reshape(-1, 1)
    x0 = torch.tensor([[0., 0., 1.]], dtype=torch.float64).expand(101, -1)
    x1 = torch.tensor([[1., 0., 0.]], dtype=torch.float64).expand(101, -1)
    x, velocity = path_velocity(model, t, x0, x1)
    assert x[:, 1:].min() >= -1e-12
    assert torch.isfinite(velocity).all()
