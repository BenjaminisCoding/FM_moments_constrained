"""Continuous moment-dual refinement; no Gaussian normalizer is required."""
from dataclasses import replace
import json
import time
import numpy as np


def polish(api, name, data, q, protocol, initial, multimode=False):
    study = api.OUT/name
    sigma, tau = float(data['sigma']), float(data['tau'])
    sampling = dict(burn_in=4096,pcn_beta=.5,initial_candidates=32,
                    checkpoints=(1024,2048,4096),independence_every=2,
                    proposal_fractions=(1/3,2/3,1.),chain_device=api.device)
    if multimode:
        sampling.update(pcn_beta=.3,initial_candidates=4*(len(initial.target)+1),proposal_fractions=(1.,))
    fitting=dict(sampling)
    if multimode:
        fitting.update(burn_in=1024,checkpoints=(256,512,1024))
    def proposals(solution):
        if multimode:
            from .classifier_proposals import centers_for_solution
            return centers_for_solution(data,q,solution)
        return api.proposal_center_function(q,solution.multiplier,sigma,tau)
    fit_seed, fit_size = 88171, 65536
    current = initial
    if multimode and np.max(np.abs(current.multiplier)) >= 3999:
        current=replace(current,multiplier=current.multiplier*(200/np.max(np.abs(current.multiplier))))
    history = []
    started = time.perf_counter()
    for iteration in range(32 if multimode else 12):
        tag='fixed_ot_multimode' if multimode else 'fixed_ot_continuous'
        out = study/'numerical_fit'/f'{tag}_step{iteration:02d}'
        out.mkdir(parents=True,exist_ok=True)
        if (out/'solver.json').exists():
            current = api.load_solution(out)
            hessian = np.load(out/'conditional_covariance.npy')
            fit = json.loads((out/'fit_sampling.json').read_text())
        else:
            _, z, _, fit = api.sample_continuous_tilt(
                x0=data['x0'],x1=data['x1'],solution=current,posterior=q,
                sigma=sigma,tau=tau,n_samples=fit_size,seed=fit_seed,**fitting,
                proposal_modes=proposals(current))
            features = api.posterior_numpy(q,z)
            # Exactly replay the sampler's initial endpoint draw. Common random
            # numbers stabilize updates; independent audit seeds guard against
            # fitting this Monte Carlo realization rather than the target law.
            pair = np.random.default_rng(fit_seed).choice(
                len(current.coupling),size=fit_size,p=current.coupling/current.coupling.sum())
            hessian = api.conditional_feature_covariance(features,pair)
            current = replace(current,composition=np.asarray(fit['composition']),
                              log_z=np.full_like(current.log_z,np.nan),converged=False,
                              optimizer_success=False,optimizer_iterations=iteration+1,
                              evaluations=iteration+1,moment_estimator='continuous_MCMC',
                              optimizer_message='Sampled fixed-coupling moment gradient; awaiting independent acceptance',
                              timings=dict(continuous_dual_seconds=time.perf_counter()-started),history=[])
            api.write_json(out/'fit_sampling.json',fit)
            np.save(out/'conditional_covariance.npy',hessian)
            api.save_solution(out,current,sigma=sigma,tau=tau,quadrature_size=0,
                              normalizers_evaluated=False,fit_seed=fit_seed,fit_samples=fit_size,
                              gradient='E_pi E_lambda[phi | endpoints] - target',
                              preconditioner='Estimated within-endpoint-pair feature covariance')
        residual = current.composition-current.target
        row = dict(iteration=iteration,residual_linf=float(np.max(np.abs(residual))),
                   multiplier=current.multiplier.tolist(),fit_sampling_seconds=fit['seconds'],
                   fit_burn_in=fit['burn_in'])
        print(json.dumps(dict(study=name,event='continuous_dual',multimode=multimode,**row)),flush=True)
        history.append(row)
        drift = np.asarray(fit['checkpoints'][-1]['composition'])-np.asarray(fit['checkpoints'][-2]['composition'])
        fit_stable = np.all(np.abs(drift) <= .003+4*np.sqrt(2)*np.asarray(fit['component_standard_errors']))
        if not fit_stable:
            fitting=dict(sampling)
        if np.max(np.abs(residual)) <= .003 and fit_stable:
            audit_path=out/'continuous_audit.json'
            if audit_path.exists():
                audit=json.loads(audit_path.read_text())
            else:
                _,_,_,audit=api.sample_continuous_tilt(
                    x0=data['x0'],x1=data['x1'],solution=current,posterior=q,
                    sigma=sigma,tau=tau,n_samples=65536,seed=99173,**sampling,
                    proposal_modes=proposals(current))
                api.write_json(audit_path,audit)
            se=np.asarray(audit['component_standard_errors'])
            audit_drift=np.asarray(audit['checkpoints'][-1]['composition'])-np.asarray(audit['checkpoints'][-2]['composition'])
            passed=(np.all(np.abs(audit['residual']) <= .003+4*se)
                    and np.all(np.abs(audit_drift) <= .003+4*np.sqrt(2)*se)
                    and audit['zero_accepted_fraction'] < .005
                    and not (out/'replicate_accuracy_rejected.json').exists())
            if passed:
                current=replace(current,converged=True,optimizer_success=True,history=history,
                                optimizer_message='Continuous moment gradient and independent sampling audit passed')
                api.save_solution(out,current,sigma=sigma,tau=tau,quadrature_size=0,
                                  normalizers_evaluated=False,fit_seed=fit_seed,fit_samples=fit_size,
                                  gradient='E_pi E_lambda[phi | endpoints] - target',
                                  preconditioner='Estimated within-endpoint-pair feature covariance')
                api.write_json(study/'frozen_bridge.json',dict(
                    numerical_fit=str(out.relative_to(api.OUT)),protocol=protocol,coupling_mode='fixed_global_ot',
                    proposal_mode_policy='classifier_multistart' if multimode else 'single_local_mode',
                    numerical_accuracy_rule='Continuous fit residual <=0.003; independent and pooled moment residuals <=0.003+4 SE; checkpoint stability required; no empirical W2 used',
                    sampling_kwargs=sampling,shared_deterministic_solver='One fixed-OT continuous moment-dual fit; five independent sampling and neural-training seeds',
                    frozen_unix_time=time.time()))
                return True
            # An inaccurate short fitting probe must not set final accuracy.
            # Once an audit rejects it, subsequent fitting uses the full chain.
            fitting=dict(sampling)
        reduced_hessian=hessian[:-1,:-1]
        ridge=max(float(np.trace(reduced_hessian))/len(reduced_hessian)*1e-4,1e-8)
        direction=np.linalg.solve(reduced_hessian+ridge*np.eye(len(reduced_hessian)),residual[:-1])
        # The sampled covariance is a preconditioner, not an exact Hessian.
        # Damp and limit relative multiplier changes to control Monte Carlo noise.
        relative_step=np.max(np.abs(direction)/np.maximum(np.abs(current.multiplier[:-1]),5.))
        damping=min(.7,.3/max(relative_step,1e-12))
        next_multiplier=current.multiplier.copy()
        next_multiplier[:-1]-=damping*direction
        if not np.isfinite(next_multiplier).all() or np.max(np.abs(next_multiplier)) > 100000:
            raise RuntimeError('Continuous moment-dual refinement exceeded its numerical safeguard.')
        api.write_json(out/'update.json',dict(direction=direction.tolist(),damping=damping,
                                             next_multiplier=next_multiplier.tolist()))
        current=replace(current,multiplier=next_multiplier,converged=False)
    return False
