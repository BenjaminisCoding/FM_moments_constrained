import numpy as np
import pytest
import torch

from cfm_project.semrau_weighted_evaluation import exact_wasserstein
from cfm_project.models import PathCorrection
from cfm_project.semrau_benchmark import (
    Data,bridge_path_velocity,features,path_velocity,solve_diagonal_bridge,
)


def data_for_bridge(matrix,target):
    x0=torch.tensor([[-1.,.2],[.4,.8]])
    x1=torch.tensor([[.5,.4],[1.3,1.5]])
    return Data(x0,x1,x0,x1,torch.tensor([.5,.5]),torch.tensor(matrix),torch.tensor(target),
                torch.ones(len(target)),{}, {})


def test_endpoint_preservation_and_jvp_derivative():
    torch.manual_seed(4);model=PathCorrection(2,[8,8]).double()
    x0=torch.randn(7,2,dtype=torch.float64);x1=torch.randn(7,2,dtype=torch.float64)
    for time,expected in [(0.,x0),(1.,x1)]:
        x,u=path_velocity(model,torch.full((7,1),time,dtype=torch.float64),x0,x1)
        torch.testing.assert_close(x,expected,atol=0,rtol=0)
    t=torch.full((7,1),.37,dtype=torch.float64);x,u=path_velocity(model,t,x0,x1)
    xp,_=path_velocity(model,t+1e-5,x0,x1);xm,_=path_velocity(model,t-1e-5,x0,x1)
    torch.testing.assert_close(u,(xp-xm)/2e-5,atol=1e-8,rtol=1e-8)
    u.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_squared_marker_features_preserve_counts_exactly():
    counts=torch.tensor([[0.,1.,4.],[9.,0.,16.]])
    scales=torch.tensor([.4,2.,3.]);state=torch.sqrt(counts)/scales
    torch.testing.assert_close(features(state,torch.diag(scales**2)),counts)


def test_actual_w1_and_w2_definitions():
    score=exact_wasserstein([[0.],[1.]],[[0.],[3.]])
    assert score["W1"]==pytest.approx(1.)
    assert score["W2"]==pytest.approx(np.sqrt(2))
    unequal=exact_wasserstein([[0.],[2.]],[[0.],[1.],[2.]])
    assert unequal["W1"]==pytest.approx(1/3)
    assert unequal["W2"]==pytest.approx(np.sqrt(1/3))


@pytest.mark.parametrize("matrix,target", [([[1.,0.]],[.2]), ([[1.,0.],[0.,1.]],[.2,.4]), ([[1.,-.5]],[.1])])
def test_analytic_csb_constraints_and_endpoint_masses(matrix,target):
    data=data_for_bridge(matrix,target);bridge=solve_diagonal_bridge(data,.5)
    np.testing.assert_allclose(bridge.coupling.sum(0),[.5,.5],atol=1e-8)
    np.testing.assert_allclose(bridge.coupling.sum(1),[.5,.5],atol=1e-8)
    mean=(bridge.x0[:,None]+bridge.x1[None,:])/2/bridge.denominator
    second=mean**2+.5**2*.25/bridge.denominator
    moment=np.einsum('ij,ijd,kd->k',bridge.coupling,second,np.array(matrix))
    np.testing.assert_allclose(moment,target,atol=2e-5)
    assert (bridge.denominator>0).all()


def test_csb_conditional_probability_velocity_is_time_derivative():
    data=data_for_bridge([[1.,0.]],[.2]);bank=solve_diagonal_bridge(data,.5).torch_bank()
    t=torch.full((40,1),.23)
    x,u=bridge_path_velocity(bank,t,40,torch.Generator().manual_seed(3))
    xp,_=bridge_path_velocity(bank,t+1e-3,40,torch.Generator().manual_seed(3))
    xm,_=bridge_path_velocity(bank,t-1e-3,40,torch.Generator().manual_seed(3))
    torch.testing.assert_close(u,(xp-xm)/.002,atol=1e-4,rtol=2e-4)
