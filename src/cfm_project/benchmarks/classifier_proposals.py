"""Class-guided proposal centers from the already frozen moment classifier."""
import numpy as np
import torch


def proposal_centers(q, multiplier, sigma, tau, means):
    sd=sigma*np.sqrt(tau*(1-tau))
    classes=len(multiplier)
    lam=torch.tensor(multiplier,dtype=torch.float32,device=getattr(q, 'device', 'cpu'))
    result=np.empty((len(means),classes,means.shape[1]),dtype=np.float32)
    for first in range(0,len(means),512):
        last=min(first+512,len(means))
        base=torch.tensor(means[first:last],device=getattr(q, 'device', 'cpu'))
        repeated=base[:,None,:].expand(-1,classes,-1).reshape(-1,means.shape[1])
        current=repeated.clone().requires_grad_(True)
        labels=torch.arange(classes,device=getattr(q, 'device', 'cpu')).repeat(len(base))
        optimizer=torch.optim.Adam([current],lr=.2*sd)
        # Stable log probabilities provide a direction even when a class has
        # almost zero softmax probability at the reference mean.
        for _ in range(32):
            log_probability=q.log_prob(current).gather(1,labels[:,None]).squeeze(1)
            energy=.5*((current-repeated)/sd).square().sum(1)-24*log_probability
            optimizer.zero_grad(set_to_none=True)
            energy.sum().backward()
            optimizer.step()
        optimizer=torch.optim.Adam([current],lr=.2*sd)
        best=current.detach().clone()
        best_energy=torch.full((len(current),),float('inf'),device=getattr(q, 'device', 'cpu'))
        for _ in range(64):
            energy=.5*((current-repeated)/sd).square().sum(1)-q(current)@lam
            improved=energy.detach()<best_energy
            best=torch.where(improved[:,None],current.detach(),best)
            best_energy=torch.minimum(best_energy,energy.detach())
            optimizer.zero_grad(set_to_none=True)
            energy.sum().backward()
            optimizer.step()
        result[first:last]=best.cpu().numpy().reshape(len(base),classes,-1)
    return result


def centers_for_solution(data,q,solution):
    tau=float(data['tau'])
    means=(1-tau)*data['x0'][solution.src]+tau*data['x1'][solution.tgt]
    return proposal_centers(q,solution.multiplier,float(data['sigma']),tau,means)
