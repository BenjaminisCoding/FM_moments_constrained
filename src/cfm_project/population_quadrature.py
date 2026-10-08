"""Refine endpoint-histogram integration using only saved training quantiles.

Within each interval in the union of endpoint CDF knots, both inverse CDFs
are affine. Changing Gaussian order therefore requires no new observations.
The physical response, endpoint histograms, coupling and LAND references stay
fixed. This module does not read empirical evaluation distributions.
"""
from dataclasses import replace

import numpy as np
import torch

from cfm_project.aggregate_benchmarks import load_training


def refine_population(data, factor=1):
    factor = int(factor)
    if factor < 1:
        raise ValueError('Population quadrature factor must be positive')
    if factor == 1:
        return data
    if data.weights is None or data.metadata.get('endpoint_representation') != 'shared_quantile_quadrature':
        raise ValueError('Population refinement requires endpoint histogram quadrature')
    order = int(data.metadata.get('population_quadrature_nodes', 4))
    with np.load(data.source / 'train.npz', allow_pickle=False) as archive:
        u = archive['quantiles'].astype(np.float64)
        mass = archive['weights'].astype(np.float64)
        endpoints = [archive['x0'].astype(np.float64), archive['x1'].astype(np.float64)]
    if len(u) % order:
        raise ValueError('Incomplete endpoint-CDF quadrature intervals')
    original_nodes, _ = np.polynomial.legendre.leggauss(order)
    nodes, weights = np.polynomial.legendre.leggauss(order * factor)
    outputs = [[], []]
    masses = []
    for first in range(0, len(u), order):
        last = first + order - 1
        radius = (u[last] - u[first]) / (original_nodes[-1] - original_nodes[0])
        center = (u[last] + u[first]) / 2
        query = center + radius * nodes
        for values, destination in zip(endpoints, outputs):
            slope = (values[last] - values[first]) / (u[last] - u[first])
            destination.append(values[first] + (query - u[first])[:, None] * slope)
        masses.append(mass[first:last + 1].sum() * weights / 2)
    x0, x1 = [torch.tensor(np.concatenate(v), dtype=data.x0.dtype) for v in outputs]
    return replace(data, x0=x0, x1=x1,
                   weights=torch.tensor(np.concatenate(masses), dtype=data.x0.dtype))


def load_fit_training(folder, config):
    return refine_population(load_training(folder), config.get('population_quadrature_factor', 1))
