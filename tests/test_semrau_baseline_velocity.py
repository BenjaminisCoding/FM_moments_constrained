from types import SimpleNamespace
import numpy as np
import torch

from cfm_project.semrau_baseline_velocity import LinearTeacher, BridgeTeacher, moment_residual
from cfm_project.semrau_benchmark import bridge_path_velocity, path_velocity


def test_linear_adapter_is_exact_linear_cfm():
    g = torch.Generator().manual_seed(14)
    a = torch.randn(20, 12, generator=g); b = torch.randn(20, 12, generator=g)
    t = torch.rand((20, 1), generator=g)
    actual = path_velocity(LinearTeacher(), t, a, b)
    expected = path_velocity(None, t, a, b)
    for x, y in zip(actual, expected): torch.testing.assert_close(x, y, atol=0, rtol=0)


def test_bridge_analytic_second_moment_matches_brownian_bridge():
    a = np.array([[0., 1.], [1., 2.]])
    b = np.array([[2., 1.], [3., 2.]])
    coupling = np.array([[.3, .2], [.1, .4]])
    tau = 1/6; sigma = .35
    teacher = BridgeTeacher(dict(x0=a, x1=b, coupling=coupling,
        denominator=np.ones(2), sigma=sigma, tau=tau))
    means = (1-tau)*a[:, None, :] + tau*b[None, :, :]
    expected = np.einsum('ij,ijd->d', coupling, means**2) + sigma**2*tau*(1-tau)
    np.testing.assert_allclose(teacher.second_moment().numpy(), expected, atol=1e-14)
    data = SimpleNamespace(matrix=torch.eye(2), target=torch.tensor(expected), scale=torch.ones(2))
    torch.testing.assert_close(moment_residual(teacher, data), torch.zeros(2,dtype=torch.float64),atol=1e-14,rtol=0)


def test_csb_conditional_velocity_is_derivative_at_arbitrary_constraint_time():
    bank = dict(x0=torch.tensor([[0., 1.]],dtype=torch.float64),
        x1=torch.tensor([[2., 3.]],dtype=torch.float64), weights=torch.ones(1,dtype=torch.float64),
        denominator=torch.tensor([1.2,.8],dtype=torch.float64), sigma=.35,tau=1/6)
    t = torch.tensor([[.05],[.1],[.4],[.8]],dtype=torch.float64)
    def draw(times): return bridge_path_velocity(bank,times,4,torch.Generator().manual_seed(83))
    x,u=draw(t); eps=1e-6
    numerical=(draw(t+eps)[0]-draw(t-eps)[0])/(2*eps)
    torch.testing.assert_close(u,numerical,atol=1e-8,rtol=1e-8)
    assert torch.isfinite(x).all()
