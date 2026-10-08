"""Reciprocal Multi experiments with empirical classifier-output constraints."""
import copy
import numpy as np
import torch
from cfm_project import training as tr
from cfm_project.classifier_moments import load_classifier_moment_inputs
from cfm_project.data import EmpiricalCouplingProblem, sample_coupled_batch
from cfm_project.models import VelocityField, PathCorrection
from cfm_project.generalized_moment_sb import brownian_subbridge_batch
from . import multi_path as p
from . import multi_clock as clock
from .common import write_json, save, snapshots, w2, notice


def initial_state(data_root, method, seed, cfg):
    cfg = copy.deepcopy(cfg)
    data, q, protocol = load_classifier_moment_inputs(data_root)
    plan = dict(np.load(data_root / 'endpoint_coupling.npz'))
    a, b = torch.from_numpy(data['x0']), torch.from_numpy(data['x1'])
    problem = EmpiricalCouplingProblem(a, b, label='multi',
        global_ot_src_idx=torch.from_numpy(plan['src']).long(),
        global_ot_tgt_idx=torch.from_numpy(plan['tgt']).long(),
        global_ot_mass=torch.from_numpy(plan['mass']).float())
    land = method in ['land', 'mfm']
    reverse = method == 'land' and float(data['tau']) == .4
    tr.set_seed(seed)
    # Preserve the declared initialization stream: velocity is initialized first.
    VelocityField(100, cfg['model']['velocity_hidden_dims'], cfg['model']['activation'])
    model = PathCorrection(100, cfg['model']['path_hidden_dims'], cfg['model']['activation'],
                          zero_init_output=cfg['model']['path_zero_init_output'])
    generator = torch.Generator().manual_seed(seed)
    tau = float(data['tau'])
    references = tr._build_metric_reference_pool(problem=problem, target_sampler=None,
        times=[tau], n_samples_per_time=256, generator=generator,
        reference_pool_policy='endpoints_only') if land else None
    if method == 'mfm' or reverse:
        cfg['train']['pseudo_rho'] = cfg['train']['pseudo_eta'] = 0.
    return dict(method='land' if land else 'lin', kind=method, seed=seed, cfg=cfg,
        model=model, generator=generator, problem=problem, q=q,
        target=torch.from_numpy(data['target'].astype(np.float32)), tau=tau, protocol=protocol,
        references=references, x0=a[problem.global_ot_src_idx], x1=b[problem.global_ot_tgt_idx],
        mass=problem.global_ot_mass, optimizer=torch.optim.Adam(model.parameters(), lr=cfg['train']['lr_g']),
        multipliers={tau: torch.zeros(len(data['target']))},
        warmup=2400 if method == 'mfm' or reverse else 120 if land else 0, shift=None,
        reverse_clock=reverse, data_root=data_root)


def fit_clock(s, validation, out, budget):
    cal = clock.calibration(s)
    strength = cal['strength'] * budget['clock_strength']
    shift = torch.tensor(0., requires_grad=True)
    opt = torch.optim.Adam([shift], lr=.01)
    pairs = torch.Generator().manual_seed(s['seed'] + 51001)
    times = torch.Generator().manual_seed(s['seed'] + 51002)
    best = float('inf'); best_shift = None; best_step = 0; reference = None; bad = 0
    history = []
    for step in range(1, budget['clock_steps'] + 1):
        a, b, _ = sample_coupled_batch(s['problem'], batch_size=256, coupling='ot_global', generator=pairs)
        t = torch.rand((256, 1), generator=times)
        base = clock.geometry(s, shift, a, b, t)
        residual = s['q'](clock.mean_path(s, s['tau'], shift, a, b)).mean(0) - s['target']
        loss = base + .5 * strength * residual.square().sum()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad(): shift.clamp_(-2, 2)
        if step % budget['clock_cadence'] and step != budget['clock_steps']:
            continue
        with torch.no_grad():
            score = w2(clock.mean_path(s, s['validation_time'], shift).numpy(), validation.numpy(), s['mass'].numpy())
        if score < best:
            best, best_step, best_shift = score, step, shift.detach().clone()
        if reference is None or score < reference - .001: reference, bad = score, 0
        else: bad += 1
        row = dict(step=step, validation_W2=score, shift=float(shift.detach()), bad_checks=bad)
        history.append(row); notice(dict(stage='clock', **row))
        if step >= budget['clock_min_steps'] and bad >= 4: break
    s['shift'] = best_shift
    write_json(out / 'clock.json', dict(calibration=cal, strength=strength, history=history,
        executed_updates=step, selected_update=best_step, validation_W2=best,
        validation_time=s['validation_time'], checkpoint_policy='minimum validation W2; patience 4, delta .001'))
    return step


def train_path(s, out, budget):
    kind = s['kind']; cap = budget['clock_geometry_steps'] if s['reverse_clock'] else budget['stage_a_steps']
    if kind in ['cfm', 'csb']: cap = 0
    prefix = int(s['cfg']['train']['stage_a_steps']) if not s['reverse_clock'] and s['seed'] in [3, 7] else 0
    rho = s['cfg']['train']['pseudo_rho']; history = []; diags = []
    for current in range(cap):
        lr = .001 if kind == 'lin' or current < 720 else .0003 if current < 1440 else .0001
        for group in s['optimizer'].param_groups: group['lr'] = lr
        s['cfg']['train']['pseudo_rho'] = rho * (.5 if kind == 'lin' and current >= 1440 else 1.)
        row = p.take_step(s, current, fast=current >= prefix)
        if not np.isfinite(row['loss']): raise RuntimeError('Nonfinite Stage-A objective')
        history.append(row)
        if (current + 1) % 300 == 0 or current + 1 == cap:
            diag = p.diagnostics(s, current + 1); diags.append(diag)
            notice(dict(stage='A', step=current + 1, residual=diag['residual_l2']))
    s['model'].eval().requires_grad_(False).zero_grad(set_to_none=True)
    executed = cap
    if s['reverse_clock']:
        validation = torch.from_numpy(np.load(s['data_root'] / 'validation.npy'))
        executed += fit_clock(s, validation, out, budget)
    save(out / 'path.pt', dict(path_state_dict=s['model'].state_dict(), shift=s['shift']))
    write_json(out / 'stage_a.json', dict(executed_updates=executed, geometry_updates=cap,
        history=history, diagnostics=diags, reverse_clock=s['reverse_clock'], reference_pool_policy='endpoints_only'))


def teacher_batch(s, n, generators, diagnostic=False):
    pairs, times, noise = generators
    if s['kind'] == 'csb':
        first, last = (57344, 65536) if diagnostic else (0, 57344)
        idx = torch.randint(first, last, (n,), generator=pairs)
        return brownian_subbridge_batch(*(x[idx] for x in s['bank']), s['protocol']['sigma'],
                                       tau=s['tau'], generator=times)
    a, b, _ = sample_coupled_batch(s['problem'], batch_size=n, coupling='ot_global', generator=pairs)
    t = torch.rand((n, 1), generator=times)
    with torch.no_grad():
        if s['kind'] == 'cfm': x, u = (1-t)*a+t*b, b-a
        elif s['shift'] is not None: x, u = clock.path_velocity(s, t, a, b, s['shift'])
        else: x, u = p.fast_path_velocity(s, t, a, b)
        x = x + .1 * torch.randn(x.shape, generator=noise)
    return t, x.detach(), u.detach(), torch.ones((n, 1))


def regression(model, batch):
    t, x, u, w = batch
    return ((model(t, x)-u).square().sum(1)*w[:, 0]).mean()


def fit(data_root, out, method, seed, cfg, budget, device='cpu'):
    s = initial_state(data_root, method, seed, cfg)
    s['validation_time'] = .4 if s['tau'] == .2 else .2
    train_path(s, out, budget)
    if method == 'csb':
        from .multi_bridge import fit_bank
        bank = fit_bank(data_root, out / 'bridge', seed, device)
        s['bank'] = [torch.from_numpy(bank[k]).float() for k in ['a', 'z', 'b']]
    model = p.new_velocity(s, seed)
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    generators = p.make_rngs(seed)
    diagnostic = teacher_batch(s, 2048, p.make_rngs(20260922, offset=10000), diagnostic=method == 'csb')
    validation = torch.from_numpy(np.load(data_root / 'validation.npy'))
    control = p.initial_control(); rows = []; best_state = None
    for step in range(1, budget['stage_b_steps'] + 1):
        loss = regression(model, teacher_batch(s, 256, generators))
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite CFM loss')
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % budget['validation_cadence'] and step != budget['stage_b_steps']: continue
        generated = snapshots(model, s['problem'].x0_pool, [s['validation_time']])[s['validation_time']]
        score = w2(generated.numpy(), validation.numpy())
        action, improved, significant = p.assess_w2(control, score, step)
        if improved: best_state = copy.deepcopy(model.state_dict())
        with torch.no_grad(): held_loss = float(regression(model, diagnostic))
        row = dict(step=step, validation_W2=score, fixed_regression_loss=held_loss, action=action)
        rows.append(row); notice(dict(stage='B', **row))
        if action == 'reduce_lr':
            for group in opt.param_groups: group['lr'] = [.0003, .0001][control['lr_drops']-1]
        if action == 'stop': break
    model.load_state_dict(best_state)
    save(out / 'velocity.pt', best_state)
    write_json(out / 'result.json', dict(executed_updates=step, selected_update=control['best_step'],
        validation_W2=control['best_w2'], validation_time=s['validation_time'], history=rows,
        stopping_rule=control, path_frozen=True, stage_c_updates=0))


def evaluate(data_root, out, method, seed, cfg):
    s = initial_state(data_root, method, seed, cfg)
    state = torch.load(out / 'path.pt', weights_only=True)
    s['model'].load_state_dict(state['path_state_dict']); s['model'].eval().requires_grad_(False)
    s['shift'] = state['shift']
    model = p.new_velocity(s, seed)
    model.load_state_dict(torch.load(out / 'velocity.pt', weights_only=True)); model.eval()
    times = [.2, .4, 1.]
    generated = snapshots(model, s['problem'].x0_pool, times)
    refined = snapshots(model, s['problem'].x0_pool, times, 200)
    with np.load(data_root / 'evaluation_only.npz') as truth:
        scores = {str(t): w2(x.numpy(), truth[f't_{t:g}']) for t, x in generated.items() if f't_{t:g}' in truth}
        fine = {str(t): w2(x.numpy(), truth[f't_{t:g}']) for t, x in refined.items() if f't_{t:g}' in truth}
    with torch.no_grad():
        residual = s['q'](generated[s['tau']]).mean(0) - s['target']
        direct = None
        if method not in ['cfm', 'csb']:
            x = clock.mean_path(s, s['tau'], s['shift']) if s['shift'] is not None else p.path_sample(s, s['tau'])
            direct = ((s['q'](x)*s['mass'][:, None]).sum(0)-s['target']).tolist()
    result = dict(W2=scores, W2_euler200=fine, rollout_residual=residual.tolist(), direct_residual=direct,
        validation_time=.4 if s['tau'] == .2 else .2,
        disclosure='Other-day W2 selected checkpoints and is validation, not held-out test.')
    write_json(out / 'evaluation.json', result)
    np.savez_compressed(out / 'particles.npz', **{str(t): x.numpy() for t, x in generated.items()})
    return result
