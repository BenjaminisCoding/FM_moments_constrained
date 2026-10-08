"""Frozen linear and analytic-CSB teachers for the 0–36h paths."""
import numpy as np
import torch
from torch import nn

from cfm_project.semrau_benchmark import bridge_path_velocity
from cfm_project.semrau_selected_velocity import sample_teacher as neural_teacher
from cfm_project.semrau_stage_a_setups import moment_residual as neural_residual


class LinearTeacher(nn.Module):
    def forward(self, t, x0, x1):
        return torch.zeros_like(x0)


class BridgeTeacher(nn.Module):
    def __init__(self, arrays):
        super().__init__()
        for key in ['x0', 'x1', 'coupling', 'denominator']:
            self.register_buffer(key, torch.as_tensor(arrays[key], dtype=torch.float64))
        self.sigma = float(arrays['sigma'])
        self.tau = float(arrays['tau'])

    def bank(self):
        return dict(x0=self.x0.float(), x1=self.x1.float(),
                    weights=self.coupling.float().ravel(), denominator=self.denominator.float(),
                    sigma=self.sigma, tau=self.tau)

    def second_moment(self):
        mean = ((1-self.tau)*self.x0[:, None, :] + self.tau*self.x1[None, :, :])/self.denominator
        variance = self.sigma**2*self.tau*(1-self.tau)/self.denominator
        return (self.coupling[:, :, None]*(mean.square()+variance)).sum((0, 1))


@torch.no_grad()
def sample_teacher(path, data, n, generator, noise):
    if not isinstance(path, BridgeTeacher):
        return neural_teacher(path, data, n, generator, noise)
    t = torch.rand((n, 1), generator=generator)
    x, u = bridge_path_velocity(path.bank(), t, n, generator)
    return t, x + noise*torch.randn(x.shape, generator=generator), u


@torch.no_grad()
def moment_residual(path, data, normalized=False):
    if not isinstance(path, BridgeTeacher):
        return neural_residual(path, data, normalized)
    residual = data.matrix.double() @ path.second_moment() - data.target.double()
    return residual/data.scale/np.sqrt(len(data.target)) if normalized else residual
