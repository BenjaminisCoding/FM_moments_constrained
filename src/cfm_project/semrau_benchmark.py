"""Two-stage Semrau benchmark and analytic diagonal-quadratic CSB baseline.

Marker coordinates retain raw UMI information exactly as scaled square roots.
The independent observation/calibration assumptions live in dataset metadata;
neither this module nor the training runner derives targets from intermediate
single-cell marginals. Test files are loaded only by the evaluation command.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.spatial.distance import cdist
import torch

from cfm_project.generalized_moment_sb import balance_log_kernel
from cfm_project.mfm_core import land_geopath_loss
from cfm_project.models import PathCorrection, VelocityField
from cfm_project.paths import corrected_path


@dataclass
class Data:
    x0: torch.Tensor
    x1: torch.Tensor
    pair0: torch.Tensor
    pair1: torch.Tensor
    weights: torch.Tensor
    matrix: torch.Tensor
    target: torch.Tensor
    scale: torch.Tensor
    validation: dict
    metadata: dict


def features(x,matrix):return x.square()@matrix.T


def path_velocity(model,t,x0,x1):
    if model is None:return (1-t)*x0+t*x1,x1-x0
    return torch.func.jvp(lambda s:corrected_path(s,x0,x1,model),(t,),(torch.ones_like(t),))


def sample_pairs(data,n,generator):
    indices=torch.multinomial(data.weights,n,replacement=True,generator=generator)
    return data.pair0[indices],data.pair1[indices]


def geometric_loss(model,data,method,config,generator,n=None):
    n=n or config["batch"]
    x0,x1=sample_pairs(data,n,generator)
    t=torch.rand((n,1),generator=generator)
    x,u=path_velocity(model,t,x0,x1)
    if method=="lin":return (u-(x1-x0)).square().sum(1).mean()
    return land_geopath_loss(x,u,torch.cat([data.x0,data.x1]),config["land_gamma"],config["land_rho"])


@dataclass
class DiagonalMomentBridge:
    x0: np.ndarray
    x1: np.ndarray
    coupling: np.ndarray
    denominator: np.ndarray
    sigma: float
    tau: float
    diagnostics: dict

    def torch_bank(self):
        return dict(x0=torch.tensor(self.x0,dtype=torch.float32),x1=torch.tensor(self.x1,dtype=torch.float32),
                    weights=torch.tensor(self.coupling.ravel(),dtype=torch.float32),
                    denominator=torch.tensor(self.denominator,dtype=torch.float32),sigma=self.sigma,tau=self.tau)


def solve_diagonal_bridge(data,sigma,tau=.5):
    """Exact conditional Gaussian tilt exp(sum_k lambda_k f_k), diagonal f.

    Conditional Gaussian integration is analytic in any state dimension.
    The endpoint coupling is rebalanced at every dual evaluation. Gaussian
    normalizability is enforced by linear constraints on the multipliers.
    """
    started=time.perf_counter();x0=data.x0.numpy().astype(float);x1=data.x1.numpy().astype(float)
    a=np.full(len(x0),1/len(x0));b=np.full(len(x1),1/len(x1))
    matrix=data.matrix.numpy().astype(float)/data.scale.numpy()[:,None]
    target=data.target.numpy().astype(float)/data.scale.numpy()
    c=sigma*sigma*tau*(1-tau);mean=(1-tau)*x0[:,None,:]+tau*x1[None,:,:]
    reference=-cdist(x0,x1,"sqeuclidean")/(2*sigma*sigma)
    state={};warm=None;calls=0
    def objective(lam):
        nonlocal state,warm,calls
        q=lam@matrix;den=1-2*c*q
        if (den<=0).any():return 1e100,np.zeros_like(lam)
        logz=-.5*np.log(den).sum()+((q/den)*mean**2).sum(2)
        balanced=balance_log_kernel(reference+logz,a,b,warm_start=warm,tolerance=2e-9,
                                    max_iterations=2000,newton_refine=True,stabilize_every=100)
        warm=(balanced.log_u,balanced.log_v)
        cond_mean=mean/den;second=cond_mean**2+c/den
        moment=np.einsum("ij,ijd,kd->k",balanced.coupling,second,matrix)
        residual=moment-target
        value=-a@balanced.log_u-b@balanced.log_v-lam@target
        state=dict(coupling=balanced.coupling,denominator=den,residual=residual,
                   endpoint_relative_error=balanced.marginal_relative_linf)
        calls+=1
        return float(value),residual
    restriction=LinearConstraint(matrix.T,-np.inf,np.full(x0.shape[1],(1-1e-8)/(2*c)))
    fit=minimize(objective,np.zeros(len(target)),jac=True,method="SLSQP",constraints=[restriction],
                 options=dict(maxiter=150,ftol=1e-12))
    objective(fit.x)
    if np.max(abs(state["residual"]))>2e-5 or state["endpoint_relative_error"]>1e-6:
        raise RuntimeError(f"CSB dual failed: {fit.message}; residual={state['residual']}")
    diagnostics=dict(solver="analytic diagonal-quadratic Gaussian tilt with optimized Brownian endpoint coupling",
        sigma=sigma,tau=tau,dual=fit.x.tolist(),normalized_residual=state["residual"].tolist(),
        endpoint_relative_error=state["endpoint_relative_error"],dual_evaluations=calls,
        iterations=int(fit.nit),optimizer_message=str(fit.message),wall_seconds=time.perf_counter()-started,
        conditional_integration="analytic, no quadrature or conditional MCMC",neural_stage_a_updates=0)
    return DiagonalMomentBridge(x0,x1,state["coupling"],state["denominator"],sigma,tau,diagnostics)


def bridge_path_velocity(bank,t,n,generator):
    pair=torch.multinomial(bank["weights"],n,replacement=True,generator=generator)
    i=pair//len(bank["x1"]);j=pair%len(bank["x1"]);x0=bank["x0"][i];x1=bank["x1"][j]
    tau=bank["tau"];sigma2=bank["sigma"]**2;c=sigma2*tau*(1-tau)
    mid=((1-tau)*x0+tau*x1)/bank["denominator"];s=c/bank["denominator"]
    left=t<tau;r=1-t
    mean=torch.where(left,(1-t/tau)*x0+(t/tau)*mid,((1-t)/(1-tau))*mid+((t-tau)/(1-tau))*x1)
    derivative=torch.where(left,(mid-x0)/tau,(x1-mid)/(1-tau))
    var=torch.where(left,sigma2*(t-t*t/tau)+(t/tau)**2*s,
                    sigma2*(r-r*r/(1-tau))+(r/(1-tau))**2*s)
    dvar=torch.where(left,sigma2*(1-2*t/tau)+2*t/(tau*tau)*s,
                     -sigma2*(1-2*r/(1-tau))-2*r/((1-tau)**2)*s)
    eps=torch.randn((n,x0.shape[1]),generator=generator)
    std=torch.sqrt(torch.clamp(var,min=1e-12))
    return mean+std*eps,derivative+.5*dvar/std*eps


def new_path(seed,config,dimension):
    torch.manual_seed(seed)
    return PathCorrection(dimension,[config["path_width"]]*config["path_depth"],activation="silu")


def new_velocity(seed,config,dimension):
    torch.manual_seed(seed+100000)
    model=VelocityField(dimension,[config["velocity_width"]]*config["velocity_depth"],activation="silu")
    activation=config.get("velocity_activation","silu")
    if activation=="tanh":
        for i,layer in enumerate(model.model.net):
            if isinstance(layer,torch.nn.SiLU):model.model.net[i]=torch.nn.Tanh()
    elif activation!="silu":raise ValueError("Unsupported velocity activation")
    return model
