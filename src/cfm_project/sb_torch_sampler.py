"""Device-resident Metropolis transitions for a fixed conditional SB target."""
import math
import numpy as np
import torch


def run_chain(*, mean, state, features, multiplier, sd, beta, displacement,
              fractions, posterior, steps, checkpoints, independence_every,
              seed, device, mode_components=None):
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed+374761393)
    convert = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    mean, z, f, lam = map(convert, (mean,state,features,multiplier))
    beta, displacement, fractions = map(convert,(beta,displacement,fractions))
    scaled_delta = displacement/sd
    squared_delta = scaled_delta.square().sum(1)
    if mode_components is not None:
        mode_components=convert(mode_components)
        scaled_modes=mode_components/sd
        squared_modes=scaled_modes.square().sum(2)
    component_count=len(fractions) if mode_components is None else mode_components.shape[1]
    decay = (1-beta.square()).sqrt()
    accepted = torch.zeros(len(z),dtype=torch.int32,device=device)
    independent_accepted = torch.zeros_like(accepted)
    independent_steps = 0
    previous_z = z.clone()
    summaries = []

    def correction(points):
        if mode_components is not None:
            projection=(((points-mean)/sd)[:,None,:]*scaled_modes).sum(2)
            return math.log(component_count)-torch.logsumexp(projection-.5*squared_modes,dim=1)
        projection = (((points-mean)/sd)*scaled_delta).sum(1)
        log_ratios = (projection[:,None]*fractions[None]
                      -.5*squared_delta[:,None]*fractions[None].square())
        return math.log(len(fractions))-torch.logsumexp(log_ratios,dim=1)

    with torch.no_grad():
        for step in range(1,steps+1):
            independent = independence_every > 0 and step % independence_every == 0
            noise = torch.randn(z.shape,generator=generator,dtype=z.dtype,device=device)
            if independent:
                component = torch.randint(component_count,(len(z),),generator=generator,device=device)
                shift=(fractions[component,None]*displacement if mode_components is None else
                       torch.gather(mode_components,1,component[:,None,None].expand(-1,1,z.shape[1])).squeeze(1))
                proposal = mean+shift+sd*noise
                proposal_correction = correction(proposal)-correction(z)
                independent_steps += 1
            else:
                proposal = mean+decay*(z-mean)+beta*sd*noise
                proposal_correction = 0.
            proposal_f = posterior(proposal).to(device=device,dtype=f.dtype)
            # Differences before summation avoid cancellation between large
            # likelihood offsets. The invariant target is unchanged.
            log_ratio = ((proposal_f-f)*lam).sum(1)+proposal_correction
            log_uniform = torch.rand(len(z),generator=generator,device=device).log()
            take = log_uniform < log_ratio
            z = torch.where(take[:,None],proposal,z)
            f = torch.where(take[:,None],proposal_f,f)
            accepted += take.to(accepted.dtype)
            if independent:
                independent_accepted += take.to(accepted.dtype)
            if step in checkpoints:
                comp = f.cpu().numpy().mean(0,dtype=np.float64)
                summaries.append(dict(step=step,composition=comp.tolist(),
                                       mean_acceptance=float(accepted.float().mean().item()/step),
                                       state_change_rms_since_checkpoint=float((z-previous_z).square().mean().sqrt().item())))
                previous_z = z.clone()
    result = tuple(x.cpu().numpy() for x in (z,f,accepted,independent_accepted))
    if not all(np.isfinite(x).all() for x in result[:2]):
        raise RuntimeError('Device Metropolis chain produced nonfinite states or moments.')
    return (*result,independent_steps,summaries)
