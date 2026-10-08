"""Stage-B helpers for frozen, arbitrary-hour Semrau positive paths.

The teacher reads endpoint training pairs only. Constant Gaussian input noise
uses the unchanged conditional velocity, hence smooths each teacher marginal.
No projection of marker coordinates is performed by training or integration.
"""
import torch

from cfm_project.semrau_benchmark import path_velocity, sample_pairs


@torch.no_grad()
def sample_teacher(path, data, n, generator, noise):
    t = torch.rand((n, 1), generator=generator)
    a, b = sample_pairs(data, n, generator)
    x, u = path_velocity(path, t, a, b)
    x = x + noise * torch.randn(x.shape, generator=generator)
    return t, x, u


def source_particles(x0, noise, draws, seed=79578900):
    if draws < 2 or draws % 2:
        raise ValueError('An even number of antithetic draws is required')
    generator = torch.Generator().manual_seed(seed)
    # Generate a fixed shape per draw so smaller sources are exact prefixes.
    samples = []
    for _ in range(draws // 2):
        e = noise * torch.randn(x0.shape, generator=generator)
        samples.extend([x0 + e, x0 - e])
    return torch.cat(samples)


@torch.no_grad()
def rollout(model, source, hours, endpoints, steps):
    start, end = endpoints
    if steps < 1 or end <= start or any(h < start or h > end for h in hours):
        raise ValueError('Invalid integration interval')
    x = source.clone()
    now = 0.
    snapshots = {}
    for hour in sorted(set(hours)):
        target = (hour - start) / (end - start)
        while now < target - 1e-12:
            dt = min(1 / steps, target - now)
            t = x.new_full((len(x), 1), now)
            k1 = model(t, x)
            k2 = model(t + dt / 2, x + dt * k1 / 2)
            k3 = model(t + dt / 2, x + dt * k2 / 2)
            k4 = model(t + dt, x + dt * k3)
            x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            now += dt
        if not torch.isfinite(x).all():
            raise RuntimeError('Nonfinite velocity rollout')
        snapshots[hour] = x.clone()
    return snapshots
