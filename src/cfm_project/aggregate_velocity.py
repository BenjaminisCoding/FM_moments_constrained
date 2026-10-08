"""Velocity-only capacity refinements; constrained interpolants are unchanged."""
import torch
from torch import nn

from cfm_project.models import MLP,VelocityField


class FourierResidualVelocity(nn.Module):
    def __init__(self,dimension,hidden,activation,residual_hidden,frequencies):
        super().__init__()
        self.base=VelocityField(dimension,hidden,activation)
        self.base.requires_grad_(False)
        self.register_buffer('frequencies',torch.tensor(frequencies,dtype=torch.float32))
        self.residual=MLP((dimension+1)*(1+2*len(frequencies)),residual_hidden,dimension,activation)
        last=next(m for m in reversed(list(self.residual.modules())) if isinstance(m,nn.Linear))
        nn.init.zeros_(last.weight);nn.init.zeros_(last.bias)

    def forward(self,t,x):
        raw=torch.cat([t,x],dim=1)
        phase=2*torch.pi*raw[:,:,None]*self.frequencies[None,None,:]
        features=torch.cat([raw,phase.sin().flatten(1),phase.cos().flatten(1)],dim=1)
        return self.base(t,x)+self.residual(features)


def make_velocity(dimension,config):
    hidden=config.get('velocity_hidden') or config['hidden']
    kind=config.get('velocity_architecture','mlp')
    if kind=='mlp':return VelocityField(dimension,hidden,config['activation'])
    if kind=='fourier_residual':
        return FourierResidualVelocity(dimension,hidden,config['activation'],
            config['velocity_residual_hidden'],config['velocity_frequencies'])
    raise ValueError('Unknown velocity architecture: '+kind)
