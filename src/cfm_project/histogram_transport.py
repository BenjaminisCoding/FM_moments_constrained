"""Exact one-dimensional W2 from a discrete prediction to a log-bin histogram."""
import numpy as np


def histogram_w2(prediction, prediction_weights, probabilities, log_edges, center, scale,
                 physical_diameter=False):
    """Integrate quantile cost exactly under uniform density within log bins.

    Predictions are in endpoint-standardized log diameter. In physical mode,
    exponential first/second moments are integrated analytically in each bin.
    """
    x=np.asarray(prediction,dtype=float).reshape(-1)
    a=np.full(len(x),1/len(x)) if prediction_weights is None else np.asarray(prediction_weights,dtype=float)
    probability=np.asarray(probabilities,dtype=float);edges=np.asarray(log_edges,dtype=float)
    if len(edges)!=len(probability)+1 or np.any(probability<0) or np.any(np.diff(edges)<=0):
        raise ValueError('Invalid histogram')
    if len(a)!=len(x) or np.any(a<0) or not np.isfinite(x).all():
        raise ValueError('Invalid predicted distribution')
    keep=a>0;x=x[keep];a=a[keep];a=a/a.sum()
    order=np.argsort(x);x=x[order];a=a[order]
    keep=probability>0
    lo,hi=edges[:-1][keep],edges[1:][keep]
    b=probability[keep];b=b/b.sum()
    ca,cb=np.cumsum(a),np.cumsum(b);ca[-1]=cb[-1]=1.
    knots=np.unique(np.r_[0.,ca,cb,1.]);left,right=knots[:-1],knots[1:]
    middle=(left+right)/2
    ix,iy=np.searchsorted(ca,middle),np.searchsorted(cb,middle)
    previous=np.r_[0.,cb[:-1]][iy]
    qleft=lo[iy]+(hi[iy]-lo[iy])*(left-previous)/b[iy]
    qright=lo[iy]+(hi[iy]-lo[iy])*(right-previous)/b[iy]
    if physical_diameter:
        predicted=np.exp(center+scale*x[ix]);width=qright-qleft
        def exp_average(delta):
            return np.divide(np.expm1(delta),delta,out=np.ones_like(delta),where=np.abs(delta)>1e-12)
        mean=np.exp(qleft)*exp_average(width)
        second=np.exp(2*qleft)*exp_average(2*width)
    else:
        predicted=x[ix];qleft=(qleft-center)/scale;qright=(qright-center)/scale
        mean=(qleft+qright)/2
        second=(qleft*qleft+qleft*qright+qright*qright)/3
    cost=(right-left)*(predicted*predicted-2*predicted*mean+second)
    return float(np.sqrt(max(float(cost.sum()),0.)))


def histogram_quadrature(probabilities,log_edges,center,scale,order=32):
    """Integrate an observation on a measured histogram, independently of endpoints."""
    p=np.asarray(probabilities,dtype=float);p=p/p.sum();edges=np.asarray(log_edges,dtype=float)
    nodes,weights=np.polynomial.legendre.leggauss(order)
    x=(edges[:-1,None]+edges[1:,None])/2+(edges[1:]-edges[:-1])[:,None]*nodes/2
    mass=p[:,None]*weights/2
    return ((x.reshape(-1,1)-center)/scale).astype(np.float32),mass.reshape(-1)
