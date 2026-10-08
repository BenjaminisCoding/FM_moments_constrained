"""Shared experiment I/O and exact empirical transport evaluation."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def notice(value):
    print(json.dumps(value, allow_nan=False), flush=True)

def w2(x,y,weights_x=None,weights_y=None):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    a=np.full(len(x),1/len(x)) if weights_x is None else np.asarray(weights_x,dtype=float)
    b=np.full(len(y),1/len(y)) if weights_y is None else np.asarray(weights_y,dtype=float)
    a=a/a.sum();b=b/b.sum()
    if x.shape[1]==1:
        ix,iy=np.argsort(x[:,0]),np.argsort(y[:,0])
        sx,sy=x[ix,0],y[iy,0]
        ca,cb=np.cumsum(a[ix]),np.cumsum(b[iy]);ca[-1]=cb[-1]=1.
        knots=np.unique(np.r_[0.,ca,cb,1.]);middle=(knots[1:]+knots[:-1])/2
        qx=sx[np.searchsorted(ca,middle)];qy=sy[np.searchsorted(cb,middle)]
        return float(np.sqrt((np.diff(knots)*(qx-qy)**2).sum()))
    cost=np.maximum((x*x).sum(1)[:,None]+(y*y).sum(1)[None,:]-2*x@y.T,0.)
    if len(x)==len(y) and weights_x is None and weights_y is None:
        i,j=linear_sum_assignment(cost);return float(np.sqrt(cost[i,j].mean()))
    import ot
    value,diagnostics=ot.emd2(a,b,cost,
                  numItermax=1000000,numThreads=1,log=True)
    if diagnostics.get('warning') or not np.isfinite(value):
        raise RuntimeError('Transport evaluation did not converge: '+str(diagnostics.get('warning')))
    return float(np.sqrt(max(float(value),0.)))


@torch.no_grad()
def snapshots(model,x0,times,n_steps=100):
    # Same Euler grid as the repository, stopping after the last requested time.
    indices={int(round(t*n_steps)):float(t) for t in times}
    x=x0.clone();out={};dt=1.0/n_steps
    for step in range(max(indices)+1):
        if step in indices:out[indices[step]]=x.clone()
        if step<max(indices):x=x+dt*model(torch.full((len(x),1),step*dt),x)
    return out
