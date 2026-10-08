"""Endpoint-preserving scalar timing adjustment used by reciprocal Multi LAND."""
import torch
from . import multi_path as e

def warp(t, shift):
    ratio=shift.exp()
    denom=1-t+t*ratio
    return t*ratio/denom, ratio/denom.square()


def path_velocity(s,t,x0,x1,shift):
    u,du=warp(t,shift)
    x,v=e.fast_path_velocity(s,u,x0,x1)
    return x,du*v


def mean_path(s,t,shift,x0=None,x1=None):
    x0=s["x0"] if x0 is None else x0
    x1=s["x1"] if x1 is None else x1
    u,_=warp(torch.full((len(x0),1),t),shift)
    g=s["model"](u,x0,x1)
    return (1-u)*x0+u*x1+e.mfm_gamma(u)*g


def geometry(s,shift,x0,x1,t,weights=None):
    x,v=path_velocity(s,t,x0,x1,shift)
    metric=e.fast_metric(x,s["references"],s["cfg"]["mfm"]["land_gamma"],.001)
    value=(v.square()*metric).sum(1)
    return value.mean() if weights is None else (value*weights).sum()


def calibration(s):
    shift=torch.tensor(0.,requires_grad=True)
    residual=(s["q"](mean_path(s,s["tau"],shift))*s["mass"][:,None]).sum(0)-s["target"]
    initial_gq=torch.autograd.grad(.5*residual.square().sum(),shift)[0]
    direction=-float(initial_gq.sign())
    shift=torch.tensor(.5*direction,requires_grad=True)
    residual=(s["q"](mean_path(s,s["tau"],shift))*s["mass"][:,None]).sum(0)-s["target"]
    gq=torch.autograd.grad(.5*residual.square().sum(),shift)[0]
    values=[geometry(s,shift,s["x0"],s["x1"],torch.full((len(s["x0"]),1),t),s["mass"]) for t in (.125,.375,.625,.875)]
    gb=torch.autograd.grad(torch.stack(values).mean(),shift)[0]
    scale=.7*abs(float(gb))/max(abs(float(gq)),1e-6)
    return dict(initial_quadratic_gradient=float(initial_gq),reference_shift=.5*direction,
        base_gradient=float(gb),quadratic_gradient=float(gq),strength=scale,
        rule="Gradient balance at a0.5 log-odds shift in aggregate-loss descent direction; base gradient at zero is nearly degenerate")
