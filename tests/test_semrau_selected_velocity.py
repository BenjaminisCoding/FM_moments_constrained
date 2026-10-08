import numpy as np
import torch

from cfm_project.paths import corrected_path
from cfm_project.semrau_benchmark import path_velocity
from cfm_project.semrau_weighted_positive import PositiveMarkerCorrection
from cfm_project.semrau_selected_velocity import rollout, source_particles


def test_frozen_positive_teacher_velocity_matches_finite_difference():
    torch.manual_seed(23)
    path = PositiveMarkerCorrection(12, 16, 1).double().requires_grad_(False)
    a = torch.randn(32, 12, dtype=torch.float64)
    b = torch.randn(32, 12, dtype=torch.float64)
    a[:, 8:] = a[:, 8:].square(); b[:, 8:] = b[:, 8:].square()
    a[:4, 8] = 0; b[:4, 8] = 0
    t = torch.linspace(.03, .97, 32, dtype=torch.float64)[:, None]
    with torch.no_grad():
        x, u = path_velocity(path, t, a, b)
        numerical = (corrected_path(t+1e-6, a, b, path)-corrected_path(t-1e-6, a, b, path))/(2e-6)
    torch.testing.assert_close(u, numerical, atol=2e-7, rtol=2e-7)
    assert torch.isfinite(u).all() and not x.requires_grad and not u.requires_grad


def test_physical_times_use_declared_interval_and_rk4_converges():
    class Exponential(torch.nn.Module):
        def forward(self, t, x):
            return (1+t)*x
    x = torch.tensor([[1., 2.]], dtype=torch.float64)
    coarse = rollout(Exponential(), x, [12, 18, 24, 36], [12, 36], 12)
    fine = rollout(Exponential(), x, [12, 18, 24, 36], [12, 36], 24)
    for hour, value in fine.items():
        t = (hour-12)/24
        expected = x * np.exp(t + t*t/2)
        torch.testing.assert_close(value, expected, atol=2e-6, rtol=2e-6)
    expected = x * np.exp(1.5)
    assert (fine[36]-expected).abs().max() < (coarse[36]-expected).abs().max()/10
    torch.testing.assert_close(fine[12], x, atol=0, rtol=0)


def test_antithetic_source_preserves_mean_and_prefix():
    x = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    small = source_particles(x, .01, 8)
    large = source_particles(x, .01, 32)
    torch.testing.assert_close(small, large[:len(small)], atol=0, rtol=0)
    torch.testing.assert_close(small.mean(0), x.mean(0), atol=1e-6, rtol=0)
