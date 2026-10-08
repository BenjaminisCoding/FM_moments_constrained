"""Fixed-endpoint-OT continuous classifier-moment bridge, fitted from zero."""
from types import SimpleNamespace
import numpy as np
from cfm_project.classifier_moments import load_classifier_moment_inputs
from cfm_project.generalized_moment_sb import (
    MomentBridgeSolution, sample_continuous_tilt, posterior_numpy,
    conditional_feature_covariance, save_solution,
)
from .common import read_json, write_json
from .classifier_dual import polish
from .classifier_proposals import centers_for_solution


def load_solution(out):
    meta = read_json(out / 'solver.json')
    with np.load(out / 'coupling_and_lambda.npz') as a:
        return MomentBridgeSolution(
            coupling_mode=meta['coupling_mode'], multiplier=a['multiplier'], coupling=a['coupling'],
            src=a['src'], tgt=a['tgt'], composition=np.asarray(meta['composition']), target=np.asarray(meta['target']),
            log_z=a['log_z'], converged=meta['converged'], optimizer_success=meta['optimizer_success'],
            optimizer_message=meta['optimizer_message'], optimizer_iterations=meta['optimizer_iterations'],
            evaluations=meta['evaluations'], endpoint_linf=meta['endpoint_linf'],
            endpoint_relative_linf=meta['endpoint_relative_linf'], timings=meta['timings'], history=meta['history'],
            moment_estimator=meta.get('moment_estimator', 'quadrature'))


def fit_bank(data_root, out, seed, device='cpu'):
    data, q, protocol = load_classifier_moment_inputs(data_root, device=device)
    q.device = device
    out.mkdir(parents=True, exist_ok=True)
    plan = dict(np.load(data_root / 'endpoint_coupling.npz'))
    errors = [float(np.max(abs(np.bincount(plan[k], plan['mass'], minlength=len(data[x])) - 1/len(data[x]))))
              for k, x in [('src', 'x0'), ('tgt', 'x1')]]
    initial = MomentBridgeSolution(coupling_mode='fixed_global_ot', multiplier=np.zeros_like(data['target']),
        coupling=plan['mass'], src=plan['src'], tgt=plan['tgt'], composition=np.full_like(data['target'], np.nan),
        target=data['target'], log_z=np.full(len(plan['mass']), np.nan), converged=False,
        optimizer_success=False, optimizer_message='Zero initialization', optimizer_iterations=0,
        evaluations=0, endpoint_linf=max(errors), endpoint_relative_linf=max(errors)*len(data['x0']),
        timings={}, history=[], moment_estimator='continuous_MCMC')
    api = SimpleNamespace(OUT=out, device=device, load_solution=load_solution, write_json=write_json,
        sample_continuous_tilt=sample_continuous_tilt, posterior_numpy=posterior_numpy,
        conditional_feature_covariance=conditional_feature_covariance, save_solution=save_solution)
    if not polish(api, 'solver', data, q, protocol, initial, multimode=True):
        raise RuntimeError('Continuous fixed-OT moment or sampling audit failed')
    frozen = read_json(out / 'solver/frozen_bridge.json')
    solution = load_solution(out / frozen['numerical_fit'])
    centers = centers_for_solution(data, q, solution)
    a, z, b, audit = sample_continuous_tilt(x0=data['x0'], x1=data['x1'], solution=solution,
        posterior=q, sigma=float(data['sigma']), tau=float(data['tau']), n_samples=65536,
        seed=100003+seed, proposal_modes=centers, **frozen['sampling_kwargs'])
    se = np.asarray(audit['component_standard_errors'])
    drift = np.asarray(audit['checkpoints'][-1]['composition'])-np.asarray(audit['checkpoints'][-2]['composition'])
    accepted = bool(np.all(np.abs(audit['residual']) <= .003+4*se)
        and np.all(np.abs(drift) <= .003+4*np.sqrt(2)*se) and audit['zero_accepted_fraction'] < .005)
    write_json(out / 'sampling_audit.json', dict(**audit, accepted=accepted))
    if not accepted: raise RuntimeError('Independent CSB bank accuracy audit failed')
    np.savez_compressed(out / 'bank.npz', a=a, z=z, b=b)
    return dict(a=a, z=z, b=b)
