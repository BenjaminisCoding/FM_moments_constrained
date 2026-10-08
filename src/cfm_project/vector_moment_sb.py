"""Audited one-dimensional Brownian SB for several bounded observation functions."""
from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp,roots_hermitenorm

from cfm_project.generalized_moment_sb import balance_log_kernel
from cfm_project.scalar_moment_sb import ScalarMomentBridge


@dataclass
class VectorMomentBridge(ScalarMomentBridge):
    target: np.ndarray
    multiplier: np.ndarray


def solve_vector_moment_bridge(x0,x1,feature,target,*,sigma,tau=.5,weights0=None,weights1=None,
                               quadrature=64,max_quadrature=512,progress=None):
    started=time.perf_counter()
    x0,x1=np.asarray(x0,dtype=float),np.asarray(x1,dtype=float)
    target=np.asarray(target,dtype=float).reshape(-1);k=len(target)
    if x0.shape[1]!=1 or x1.shape[1]!=1 or k<2 or sigma<=0 or not 0<tau<1:
        raise ValueError('Vector quadrature requires1D states, at least2 features and a valid Brownian reference')
    a=np.full(len(x0),1/len(x0)) if weights0 is None else np.asarray(weights0,dtype=float)
    b=np.full(len(x1),1/len(x1)) if weights1 is None else np.asarray(weights1,dtype=float)
    a=a/a.sum();b=b/b.sum()
    shape=(len(x0),len(x1));n=shape[0]*shape[1]
    mean=((1-tau)*x0[:,0,None]+tau*x1[:,0][None,:]).ravel()
    std=sigma*np.sqrt(tau*(1-tau))
    log_reference=-(x0[:,0,None]-x1[:,0][None,:])**2/(2*sigma**2)
    tolerances=np.maximum(1e-4,1e-3*np.abs(target))
    attempts=[];q=quadrature;initial=np.zeros(k)

    def features_at(nodes,start,stop):
        z=mean[start:stop,None]+std*nodes
        f=np.asarray(feature(z.reshape(-1,1)),dtype=float).reshape(stop-start,len(nodes),k)
        if not np.isfinite(f).all():raise ValueError('Nonfinite observation response')
        return f

    def grid(count,cache=True):
        nodes,w=roots_hermitenorm(count);logs=np.log(w)-.5*np.log(2*np.pi)
        values=None
        if cache and n*count*k*8<1_200_000_000:
            values=np.empty((n,count,k))
            for start in range(0,n,2048):values[start:start+2048]=features_at(nodes,start,min(start+2048,n))
        return nodes,logs,values

    def integrals(multiplier,nodes,logs,values,cdf=False):
        normalizers=np.empty(n);expectations=np.empty((n,k))
        cumulative=np.empty((n,len(nodes))) if cdf else None
        for start in range(0,n,2048):
            stop=min(start+2048,n)
            f=features_at(nodes,start,stop) if values is None else values[start:stop]
            tilted=logs[None,:]+np.einsum('pqk,k->pq',f,multiplier)
            lz=logsumexp(tilted,axis=1)
            probability=np.exp(tilted-lz[:,None])
            normalizers[start:stop]=lz
            expectations[start:stop]=np.einsum('pq,pqk->pk',probability,f)
            if cdf:cumulative[start:stop]=np.cumsum(probability,axis=1)
        if cdf:cumulative[:,-1]=1.
        return normalizers.reshape(shape),expectations.reshape(*shape,k),cumulative

    while True:
        nodes,logs,values=grid(q)
        warm=None;calls=0;state={}
        def objective(multiplier):
            nonlocal warm,calls,state
            log_z,conditional,_=integrals(multiplier,nodes,logs,values)
            balanced=balance_log_kernel(log_reference+log_z,a,b,warm_start=warm,tolerance=1e-9,
                                        max_iterations=1000,stabilize_every=100,newton_refine=True)
            warm=(balanced.log_u,balanced.log_v)
            moment=np.einsum('ij,ijk->k',balanced.coupling,conditional)
            residual=moment-target
            # Entropy conjugate after endpoint scaling, with its exact envelope
            # gradient. Endpoint-potential gauge cancels because both masses=1.
            value=-a@balanced.log_u-b@balanced.log_v-multiplier@target
            calls+=1
            state=dict(coupling=balanced.coupling,log_z=log_z,residual=residual,
                       endpoint_relative_error=balanced.marginal_relative_linf)
            if progress and calls%10==0:
                progress(dict(event='vector_moment_dual',quadrature=q,evaluations=calls,
                              residual_linf=float(abs(residual).max()),multiplier=multiplier.tolist()))
            return float(value),residual
        fitted=minimize(objective,initial,method='L-BFGS-B',jac=True,
                         options=dict(maxiter=240,ftol=1e-14,gtol=2e-8,maxls=40))
        objective(fitted.x)
        coupling=state['coupling']
        audit_nodes,audit_logs,_=grid(2*q,cache=False)
        audit_z,audit_conditional,_=integrals(fitted.x,audit_nodes,audit_logs,None)
        residual=np.einsum('ij,ijk->k',coupling,audit_conditional)-target
        normalizer_error=float((coupling*abs(audit_z-state['log_z'])).sum())
        balanced=balance_log_kernel(log_reference+audit_z,a,b,warm_start=warm,tolerance=1e-9,
                                    max_iterations=1000,stabilize_every=100,newton_refine=True)
        tv=float(abs(balanced.coupling-coupling).sum()/2)
        row=dict(quadrature=q,audit_quadrature=2*q,feature_count=k,multiplier=fitted.x.tolist(),
                 fitting_residual=state['residual'].tolist(),audit_residual=residual.tolist(),
                 weighted_log_normalizer_error=normalizer_error,coupling_tv=tv,
                 endpoint_relative_error=state['endpoint_relative_error'],dual_evaluations=calls,
                 optimizer_message=str(fitted.message),tolerance=tolerances.tolist())
        row['accepted']=bool((abs(residual)<=tolerances).all() and
                              (abs(state['residual'])<=tolerances/2).all() and normalizer_error<1e-3 and tv<.005)
        attempts.append(row)
        if progress:progress(dict(event='vector_quadrature_audit',**row))
        if row['accepted']:break
        if q>=max_quadrature:raise RuntimeError('Vector moment quadrature/dual audit failed: '+str(row))
        q=min(2*q,max_quadrature);initial=fitted.x.copy();del values
    _,_,cdf=integrals(fitted.x,nodes,logs,values,cdf=True)
    diagnostics=dict(wall_seconds=time.perf_counter()-started,attempts=attempts,**row,
                     coupling='optimized dense Brownian SB with vector moment tilt',
                     sampler='Gaussian-Hermite conditional atoms; doubled-resolution audit',
                     intermediate_samples_used=False)
    return VectorMomentBridge(x0,x1,coupling,sigma,tau,target,fitted.x,nodes,cdf,diagnostics)
