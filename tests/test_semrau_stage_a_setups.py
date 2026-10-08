import json

import numpy as np
import pytest
import torch

from cfm_project.semrau_benchmark import Data, path_velocity
from cfm_project.semrau_stage_a_setups import load_data, moment_residual


class ConstantCorrection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.correction = torch.nn.Parameter(torch.tensor([.3, -.2]))

    def forward(self, t, a, b):
        return self.correction.expand_as(a)


@pytest.mark.parametrize('tau', [1/6, 1/3, 1/2])
def test_actual_constraint_time_and_gradient(tau):
    a = torch.tensor([[1., 2.], [2., 3.]])
    b = torch.tensor([[3., 1.], [4., 2.]])
    w = torch.tensor([.25, .75])
    data = Data(a, b, a, b, w, torch.eye(2), torch.ones(2), torch.ones(2), {}, {'tau': tau})
    model = ConstantCorrection()
    actual = moment_residual(model, data)
    x = (1-tau)*a+tau*b+tau*(1-tau)*model.correction
    expected = (x.square()*w[:, None]).sum(0)-1
    torch.testing.assert_close(actual, expected)
    derivative = torch.autograd.grad(actual.sum(), model.correction)[0]
    torch.testing.assert_close(derivative, 2*tau*(1-tau)*(x*w[:, None]).sum(0))
    for t in [0., 1.]:
        x, velocity = path_velocity(model, torch.full((2, 1), t), a, b)
        torch.testing.assert_close(x, a if t == 0 else b)
        torch.testing.assert_close(velocity, b-a+(1-2*t)*model.correction)


def test_loader_requires_no_evaluation_data(tmp_path):
    np.savez(tmp_path/'train.npz', x0=np.ones((2, 2)), x1=np.ones((2, 2))*2,
             pair_i=[0, 1], pair_j=[1, 0], pair_weights=[.5, .5])
    (tmp_path/'metadata.json').write_text(json.dumps(dict(tau=1/6, endpoints=[0, 36],
        matrix=[[1., 0.]], target=[2.], normalization=[2.])))
    data = load_data(tmp_path)
    assert not data.validation and data.metadata['tau'] == 1/6
