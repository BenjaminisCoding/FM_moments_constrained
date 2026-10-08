from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    lowered = name.lower()
    if lowered == "relu":
        return nn.ReLU()
    if lowered == "gelu":
        return nn.GELU()
    if lowered == "silu":
        return nn.SiLU()
    if lowered == "selu":
        return nn.SELU()
    raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Iterable[int],
        out_dim: int,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        dims = [in_dim, *hidden_dims, out_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(_activation(activation))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def zero_init_last_linear(module: nn.Module) -> None:
    linear_layers = [layer for layer in module.modules() if isinstance(layer, nn.Linear)]
    if not linear_layers:
        raise ValueError("zero_init_last_linear requires at least one nn.Linear layer.")
    last = linear_layers[-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)


class VelocityField(nn.Module):
    def __init__(
        self,
        state_dim: int = 2,
        hidden_dims: Iterable[int] = (128, 128),
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.model = MLP(
            in_dim=1 + state_dim,
            hidden_dims=hidden_dims,
            out_dim=state_dim,
            activation=activation,
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([t, x], dim=1))


class PathCorrection(nn.Module):
    def __init__(
        self,
        state_dim: int = 2,
        hidden_dims: Iterable[int] = (128, 128),
        activation: str = "silu",
        zero_init_output: bool = False,
    ) -> None:
        super().__init__()
        self.model = MLP(
            in_dim=1 + 2 * state_dim,
            hidden_dims=hidden_dims,
            out_dim=state_dim,
            activation=activation,
        )
        if zero_init_output:
            zero_init_last_linear(self.model)

    def forward(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([t, x0, x1], dim=1))
