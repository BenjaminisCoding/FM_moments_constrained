"""Analytic mean/vertical-variance Brownian bridge teacher."""
import hashlib
import numpy as np
import torch
from cfm_project.generalized_moment_sb import brownian_subbridge_batch

class ExactTeacher:
    kind = "sb"

    def __init__(self, bridge):
        self.bridge, self.sigma = bridge, bridge.sigma
        self.cdf = np.cumsum(bridge.coupling.ravel())
        self.cdf[-1] = 1.
        self.x0 = torch.from_numpy(bridge.x0.astype(np.float32))
        self.x1 = torch.from_numpy(bridge.x1.astype(np.float32))
        self.covariance = torch.from_numpy(bridge.conditional_covariance.astype(np.float32))
        self.chol = torch.linalg.cholesky(self.covariance)
        self.linear = torch.from_numpy(bridge.linear_multiplier.astype(np.float32))

    def fingerprint(self):
        h = hashlib.sha256(f"mean_y_variance_sb/{self.sigma}/{self.bridge.tau}".encode())
        for a in [self.bridge.x0, self.bridge.x1, self.bridge.coupling, self.bridge.conditional_covariance,
                  self.bridge.linear_multiplier]:
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()

    def skeleton(self, n, generator):
        ids = np.searchsorted(self.cdf, torch.rand(n, dtype=torch.float64, generator=generator).numpy())
        i, j = np.divmod(ids, len(self.x1))
        a, b = self.x0[i], self.x1[j]
        tau = self.bridge.tau
        c = self.sigma**2*tau*(1-tau)
        mean = (((1-tau)*a+tau*b)/c+self.linear) @ self.covariance
        return a, mean+torch.randn(mean.shape, generator=generator) @ self.chol.T, b

    def batch(self, n, generator, validation=False):
        return brownian_subbridge_batch(*self.skeleton(n, generator), self.sigma,
                                       tau=self.bridge.tau, generator=generator)
