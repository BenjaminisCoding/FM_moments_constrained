"""Nonnegative marker paths expressed through the existing correction gate."""
import torch
from torch import nn

from cfm_project.models import PathCorrection
from cfm_project.semrau_benchmark import new_path as ordinary_path


class PositiveMarkerCorrection(nn.Module):
    def __init__(self, dimension, width, depth, positive_start=8):
        super().__init__()
        self.positive_start = positive_start
        self.raw = PathCorrection(dimension, [width] * depth, activation='silu')

    def forward(self, t, x0, x1):
        raw = self.raw(t, x0, x1)
        linear = ((1 - t) * x0 + t * x1)[:, self.positive_start:]
        # Never evaluate sqrt at zero: its undefined derivative would create
        # NaNs in JVP/backprop even when multiplied by the vanishing gate.
        positive = linear > 0
        root = torch.where(positive, torch.sqrt(torch.where(positive, linear, torch.ones_like(linear))), torch.zeros_like(linear))
        gate = t * (1 - t)
        marker_raw = raw[:, self.positive_start:]
        marker_correction = 2 * root * marker_raw + gate * marker_raw.square()
        # The standard I=L+h*g becomes (sqrt(L)+h*raw)^2 on markers.
        return torch.cat([raw[:, :self.positive_start], marker_correction], dim=1)


def new_path(seed, cfg, dimension):
    if cfg.get('path_basis') != 'positive_square':
        return ordinary_path(seed, cfg, dimension)
    torch.manual_seed(seed)
    return PositiveMarkerCorrection(dimension, cfg['path_width'], cfg['path_depth'], cfg.get('positive_start', 8))
