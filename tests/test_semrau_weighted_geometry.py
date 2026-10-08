"""Coordinate changes retain the physical LAND energy and its units."""
import torch
from cfm_project.mfm_core import land_metric_tensor
from cfm_project.semrau_weighted_geometry import land_energy


def test_scalar_coordinate_unit_invariance():
    torch.manual_seed(79578)
    x=torch.randn(7,3,dtype=torch.float64);u=torch.randn_like(x);ref=torch.randn(13,3,dtype=torch.float64)
    original=land_energy(x,u,ref,dict(land_gamma=1.5,land_rho=.05))
    scale=2.7
    transformed=land_energy(x,u,ref,dict(land_gamma=1.5*scale,land_rho=.05*scale**2,
        geometry_matrix=(torch.eye(3,dtype=torch.float64)*scale).tolist()))
    torch.testing.assert_close(transformed,original,rtol=1e-12,atol=1e-12)


def test_nondiagonal_coordinate_map_matches_pullback_metric():
    torch.manual_seed(83)
    x=torch.randn(6,3,dtype=torch.float64);u=torch.randn_like(x);ref=torch.randn(11,3,dtype=torch.float64)
    matrix=torch.tensor([[1.,.2,-.1],[0.,1.7,.3],[.4,0.,.8]],dtype=torch.float64)
    diagonal=land_metric_tensor(x@matrix.T,ref@matrix.T,1.5,.05)
    full=torch.einsum('ki,nk,kj->nij',matrix,diagonal,matrix)
    expected=torch.einsum('ni,nij,nj->n',u,full,u)
    actual=land_energy(x,u,ref,dict(land_gamma=1.5,land_rho=.05,geometry_matrix=matrix.tolist()))
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
