"""One-dimensional nonlinear-moment Brownian SB with audited quadrature.

Optimizes both the endpoint coupling and the common scalar moment multiplier.
Gaussian-Hermite resolution is a numerical approximation, recorded separately
from moment feasibility. No empirical intermediate distribution enters here.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp,roots_hermitenorm

from cfm_project.generalized_moment_sb import balance_log_kernel


@dataclass
class ScalarMomentBridge:
    x0: np.ndarray
    x1: np.ndarray
    coupling: np.ndarray
    sigma: float
    tau: float
    target: float
    multiplier: float
    noise_nodes: np.ndarray
    conditional_cdf: np.ndarray
    diagnostics: dict

    def sample(self,n,rng):
        probability=self.coupling.ravel()
        pair=rng.choice(probability.size,size=n,p=probability/probability.sum())
        i,j=np.divmod(pair,len(self.x1))
        node=(rng.random((n,1))>self.conditional_cdf[pair]).sum(1)
        node=np.minimum(node,len(self.noise_nodes)-1)
        x,y=self.x0[i],self.x1[j]
        z=(1-self.tau)*x+self.tau*y
        z=z+self.sigma*np.sqrt(self.tau*(1-self.tau))*self.noise_nodes[node,None]
        return x.astype(np.float32),z.astype(np.float32),y.astype(np.float32)


def solve_scalar_moment_bridge(x0,x1,feature,target,*,sigma,tau=.5,
                               weights0=None,weights1=None,quadrature=64,max_quadrature=256,
                               moment_tolerance=1e-4,relative_tolerance=1e-3,
                               progress=None):
    """Solve E[f(Z)]=target, retaining every Cartesian endpoint pair."""
    started=time.perf_counter()
    x0,x1=np.asarray(x0,dtype=float),np.asarray(x1,dtype=float)
    if x0.ndim!=2 or x1.ndim!=2 or x0.shape[1]!=1 or x1.shape[1]!=1:
        raise ValueError('Scalar quadrature requires one-dimensional states')
    if sigma<=0 or not 0<tau<1 or not np.isfinite(target):
        raise ValueError('Invalid Brownian reference or target')
    a=np.full(len(x0),1/len(x0)) if weights0 is None else np.asarray(weights0,dtype=float)
    b=np.full(len(x1),1/len(x1)) if weights1 is None else np.asarray(weights1,dtype=float)
    a=a/a.sum();b=b/b.sum()
    mean=((1-tau)*x0[:,0,None]+tau*x1[:,0][None,:]).ravel()
    std=sigma*np.sqrt(tau*(1-tau))
    log_reference=-(x0[:,0,None]-x1[:,0][None,:])**2/(2*sigma**2)
    shape=log_reference.shape
    attempts=[]
    tolerance=max(moment_tolerance,relative_tolerance*abs(target))

    def make_grid(count):
        nodes,weights=roots_hermitenorm(count)
        logs=np.log(weights)-.5*np.log(2*np.pi)
        # Bound transient memory independently of endpoint count.
        values=np.empty((len(mean),count))
        for start in range(0,len(mean),4096):
            z=mean[start:start+4096,None]+std*nodes[None,:]
            values[start:start+4096]=np.asarray(feature(z.ravel()[:,None])).reshape(z.shape)
        if not np.isfinite(values).all():raise ValueError('Non-finite observation response')
        return nodes,logs,values

    def integrals(multiplier,logs,values,probabilities=False):
        lz=np.empty(len(mean));moments=np.empty(len(mean))
        cdf=np.empty_like(values) if probabilities else None
        for start in range(0,len(mean),4096):
            f=values[start:start+4096]
            shifted=logs[None,:]+multiplier*f
            normalizer=logsumexp(shifted,axis=1)
            probability=np.exp(shifted-normalizer[:,None])
            lz[start:start+len(f)]=normalizer
            moments[start:start+len(f)]=(probability*f).sum(1)
            if probabilities:
                cdf[start:start+len(f)]=np.cumsum(probability,axis=1)
        if probabilities:cdf[:,-1]=1.
        return lz.reshape(shape),moments.reshape(shape),cdf

    q=int(quadrature)
    while True:
        nodes,logs,values=make_grid(q)
        warm=None;calls=0
        state={}
        def discrepancy(multiplier):
            nonlocal warm,calls,state
            log_z,conditional,_=integrals(multiplier,logs,values)
            balanced=balance_log_kernel(log_reference+log_z,a,b,warm_start=warm,
                                        tolerance=1e-9,max_iterations=1000,newton_refine=True,
                                        stabilize_every=100)
            warm=(balanced.log_u,balanced.log_v)
            actual=float((balanced.coupling*conditional).sum())
            calls+=1
            state=dict(multiplier=float(multiplier),moment=actual,residual=actual-target,
                       coupling=balanced.coupling,log_z=log_z,endpoint_error=balanced.marginal_relative_linf)
            return actual-target
        initial=discrepancy(0.)
        if abs(initial)<=1e-10:
            fitted=0.
        else:
            sign=-1. if initial>0 else 1.
            bound=sign*.25/max(np.max(np.abs(values)),1e-5)
            for _ in range(40):
                outer=discrepancy(bound)
                if initial*outer<0:break
                bound*=2
            else:
                raise RuntimeError('Could not bracket scalar moment multiplier')
            fitted=brentq(discrepancy,min(0.,bound),max(0.,bound),xtol=1e-10,rtol=1e-10,maxiter=80)
            discrepancy(fitted)
        coupling=state['coupling'];fitting_residual=state['residual']
        # Independent, doubled numerical resolution checks the same frozen
        # multiplier and coupling, before any empirical marginal evaluation.
        _,audit_logs,audit_values=make_grid(2*q)
        audit_log_z,audit_conditional,_=integrals(fitted,audit_logs,audit_values)
        audit_moment=float((coupling*audit_conditional).sum())
        normalizer_error=float((coupling*np.abs(audit_log_z-state['log_z'])).sum())
        audit_balance=balance_log_kernel(log_reference+audit_log_z,a,b,warm_start=warm,
                                        tolerance=1e-9,max_iterations=1000,newton_refine=True,
                                        stabilize_every=100)
        coupling_tv=float(np.abs(audit_balance.coupling-coupling).sum()/2)
        audit=dict(quadrature=q,audit_quadrature=2*q,multiplier=float(fitted),
                   fitting_residual=float(fitting_residual),audit_residual=audit_moment-target,
                   weighted_log_normalizer_error=normalizer_error,coupling_tv=coupling_tv,
                   endpoint_relative_error=state['endpoint_error'],dual_evaluations=calls,
                   tolerance=tolerance)
        audit['accepted']=bool(abs(audit['audit_residual'])<=tolerance and normalizer_error<1e-3 and coupling_tv<.005)
        attempts.append(audit)
        if progress:progress(dict(event='scalar_quadrature_audit',**audit))
        del audit_values
        if audit['accepted']:break
        if q>=max_quadrature:
            raise RuntimeError('Scalar SB quadrature audit failed: '+str(audit))
        q=min(q*2,max_quadrature)
        del values
    _,_,cdf=integrals(fitted,logs,values,probabilities=True)
    diagnostics=dict(wall_seconds=time.perf_counter()-started,attempts=attempts,**attempts[-1],
                     coupling='optimized dense Brownian SB with common moment tilt',
                     sampler='Gaussian-Hermite conditional quadrature atoms; doubled-resolution audit',
                     intermediate_samples_used=False)
    return ScalarMomentBridge(x0,x1,coupling,sigma,tau,float(target),float(fitted),nodes,cdf,diagnostics)
