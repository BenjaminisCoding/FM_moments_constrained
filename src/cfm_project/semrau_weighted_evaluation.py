"""Exact Wasserstein evaluation with lossless merging of identical atoms."""
import numpy as np
import ot
from scipy.spatial.distance import cdist


def merge_atoms(x, weights):
    points, inverse = np.unique(np.asarray(x, dtype=float), axis=0, return_inverse=True)
    mass = np.bincount(inverse, weights=np.asarray(weights, dtype=float), minlength=len(points))
    return points, mass / mass.sum()


def exact_wasserstein(x, y, weights=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    a = np.full(len(x), 1 / len(x)) if weights is None else np.asarray(weights, dtype=float)
    assert np.all(a >= 0) and a.sum() > 0
    x, a = merge_atoms(x, a)
    y, b = merge_atoms(y, np.full(len(y), 1 / len(y)))
    squared = cdist(x, y, 'sqeuclidean')
    result = {}
    for name, cost in [('W1', np.sqrt(np.maximum(squared, 0))), ('W2', squared)]:
        value, solver = ot.emd2(a, b, cost, numItermax=10000000, log=True)
        if solver.get('warning'):
            raise RuntimeError(solver['warning'])
        dual = float(a @ solver['u'] + b @ solver['v'])
        assert abs(float(value) - dual) < 1e-7
        assert float((cost - solver['u'][:, None] - solver['v'][None, :]).min()) > -1e-7
        result[name] = float(np.sqrt(max(value, 0))) if name == 'W2' else float(value)
    return result
