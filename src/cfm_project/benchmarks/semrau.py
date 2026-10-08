"""Fixed 0–36h Semrau Stage A and frozen-path Stage B."""
import copy
import json
import resource
import sys
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from cfm_project.semrau_benchmark import features, new_velocity, solve_diagonal_bridge, sample_pairs, path_velocity
from cfm_project.semrau_benchmark import geometric_loss as ordinary_geometry
from cfm_project.semrau_stage_a_setups import load_data, moment_residual as path_residual
from cfm_project.semrau_weighted_positive import new_path
from cfm_project.semrau_weighted_geometry import land_energy
from cfm_project.semrau_weighted_evaluation import exact_wasserstein
from cfm_project.semrau_selected_velocity import rollout, source_particles
from cfm_project.semrau_baseline_velocity import LinearTeacher, BridgeTeacher, sample_teacher, moment_residual as bridge_residual
from .common import read_json, write_json, digest, save
read = read_json
write = write_json


def norm(grads):
    return torch.sqrt(sum(g.square().sum() for g in grads if g is not None))


def geometric_loss(path, data, method, cfg, generator, n=None):
    if cfg.get('path_basis') != 'positive_square':
        return ordinary_geometry(path, data, method, cfg, generator, n)
    count = n or cfg['batch']
    a, b = sample_pairs(data, count, generator)
    t = torch.rand((count, 1), generator=generator)
    x, u = path_velocity(path, t, a, b)
    return land_energy(x, u, torch.cat([data.x0, data.x1]), cfg).mean()

def train_path(data_root, folder, job):
    cfg = job["config"]
    folder.mkdir(parents=True, exist_ok=True)
    data = load_data(data_root); method = job['method']; started = time.perf_counter()
    if method == 'cfm': result = dict(executed_updates=0, selected_update=0, wall_seconds=0., history=[])
    elif method == 'csb':
        bridge = solve_diagonal_bridge(data, cfg['sigma'], tau=data.metadata['tau'])
        np.savez_compressed(folder/'bridge.npz', x0=bridge.x0, x1=bridge.x1, coupling=bridge.coupling,
                            denominator=bridge.denominator, sigma=bridge.sigma, tau=bridge.tau)
        result = dict(executed_updates=0, selected_update=None, **bridge.diagnostics)
    else:
        path = new_path(job['seed'], cfg, 12); optimizer = torch.optim.Adam(path.parameters(), lr=cfg['lr_a'])
        generator = torch.Generator().manual_seed(job['seed']+2000)
        constrained = method in ['lin', 'land']; warmup = cfg['warmup_land'] if method in ['mfm', 'land'] else 0
        lam = torch.zeros_like(data.target); history = []; start = 1; spent = 0.
        if (folder/'last.pt').exists():
            state = torch.load(folder/'last.pt', weights_only=False); path.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer']); generator.set_state(state['generator'])
            lam = state['lam']; history = state['history']; start = state['step']+1; spent = state['spent']
        for step in range(start, cfg['stage_a_cap']+1):
            rate = cfg['lr_a']*(.15+.85*.5*(1+np.cos(np.pi*(step-1)/cfg['stage_a_cap'])))
            for g in optimizer.param_groups: g['lr'] = rate
            geo = geometric_loss(path, data, method, cfg, generator)
            active = constrained and step > warmup
            residual = moment_residual(path, data, normalized=True) if active else torch.zeros_like(lam)
            loss = geo+(lam*residual).sum()+.5*cfg['rho']*residual.square().sum()
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite Stage-A loss')
            optimizer.zero_grad(); loss.backward(); grad = float(torch.nn.utils.clip_grad_norm_(path.parameters(), 10.)); optimizer.step()
            if active:
                with torch.no_grad(): lam = torch.clamp(lam+cfg['rho']*moment_residual(path, data, True), -cfg['lambda_clip'], cfg['lambda_clip'])
            if step % 300 == 0 or step == cfg['stage_a_cap']:
                fixed = torch.Generator().manual_seed(79578008)
                base = geometric_loss(path, data, method, cfg, fixed, n=512)
                residual = moment_residual(path, data, True)
                constraint = (lam*residual).sum()+.5*cfg['rho']*residual.square().sum() if active else 0.*residual.sum()
                g0 = torch.autograd.grad(base, tuple(path.parameters()), retain_graph=True)
                gc = torch.autograd.grad(constraint, tuple(path.parameters()))
                n0, nc = float(norm(g0)), float(norm(gc))
                dot = sum((a*b).sum() for a,b in zip(g0,gc))
                raw = moment_residual(path, data).detach()
                row = dict(step=step, geometric=float(base.detach()), raw_residual=raw.tolist(),
                    relative_residual=(raw/data.target).tolist(), multiplier=lam.tolist(),
                    multiplier_clip_fraction=float((abs(lam)>=cfg['lambda_clip']).float().mean()),
                    gradient_base_norm=n0, gradient_constraint_norm=nc, gradient_ratio=nc/max(n0,1e-12),
                    gradient_cosine=float(dot)/max(n0*nc,1e-12), training_gradient_norm=grad)
                history.append(row)
                print(json.dumps(dict(job=job['id'], stage='A', **row)), flush=True)
                save(folder/'last.pt', dict(model=path.state_dict(), optimizer=optimizer.state_dict(), generator=generator.get_state(),
                    lam=lam, history=history, step=step, spent=spent+time.perf_counter()-started))
        save(folder/'path.pt', path.state_dict())
        result = dict(executed_updates=cfg['stage_a_cap'], selected_update=cfg['stage_a_cap'],
                      wall_seconds=spent+time.perf_counter()-started, history=history,
                      raw_residual=moment_residual(path,data).detach().tolist(), multiplier=lam.tolist(),
                      checkpoint_policy='final fixed-budget iterate; no early stopping or selection', warmup_updates=warmup)
    result.update(stage_b_updates=0, stage_c_updates=0,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
    write(folder/'stage_a.json', result)


def train_velocity(protocol, job, cfg, data, path, out):
    job_id = job["id"]
    stamp = dict(job=job, config=cfg)
    with np.load(protocol['validation']) as val:
        validation = val[protocol['validation_key']].copy()
    model = new_velocity(job['seed'], cfg, data.x0.shape[1])
    ema = copy.deepcopy(model).requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr_b'])
    generator = torch.Generator().manual_seed(job['seed'] + 4000)
    diagnostic = sample_teacher(path, data, cfg['diagnostic_size'], torch.Generator().manual_seed(79578901), cfg['stage_b_input_noise'])
    source = source_particles(data.x0, cfg['stage_b_input_noise'], cfg['validation_draws'])
    start = 1
    best = float('inf')
    beststep = 0
    beststate = None
    history = []
    spent = 0.
    last = out / 'stage_b_last.pt'
    if last.exists():
        state = torch.load(last, weights_only=False)
        model.load_state_dict(state['model']); ema.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer']); generator.set_state(state['generator'])
        start = state['step'] + 1
        best, beststep, beststate = state['best'], state['beststep'], state['beststate']
        history, spent = state['history'], state['spent']
    started = time.perf_counter()
    interval_loss = 0.
    clipped = 0
    with threadpool_limits(cfg['threads']):
        for step in range(start, cfg['stage_b_cap'] + 1):
            lr = cfg['lr_b'] * (.15 + .85 * .5 * (1 + np.cos(np.pi * (step - 1) / cfg['stage_b_cap'])))
            for group in optimizer.param_groups:
                group['lr'] = lr
            t, x, u = sample_teacher(path, data, cfg['batch'], generator, cfg['stage_b_input_noise'])
            assert not x.requires_grad and not u.requires_grad
            loss = (model(t, x) - u).square().sum(1).mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite CFM loss')
            optimizer.zero_grad(); loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.))
            optimizer.step()
            interval_loss += float(loss.detach()); clipped += int(norm > 10.)
            with torch.no_grad():
                for average, current in zip(ema.parameters(), model.parameters()):
                    average.lerp_(current, 1 - cfg['ema_decay'])
            if step % cfg['validation_cadence'] == 0:
                snapshots = rollout(ema, source, [6, 36], protocol['endpoints'], cfg['ode_steps_validation'])
                score = exact_wasserstein(snapshots[36].numpy(), validation)['W2']
                with torch.no_grad():
                    td, xd, ud = diagnostic
                    regression = float((ema(td, xd) - ud).square().sum(1).mean())
                    residual = (features(snapshots[6], data.matrix).mean(0) / data.target - 1).tolist()
                row = dict(step=step, validation_W2=score, fixed_regression_loss=regression,
                    training_loss=interval_loss / cfg['validation_cadence'],
                    clipped_updates=clipped, rollout_relative_residual=residual, lr=lr)
                history.append(row); interval_loss = 0.; clipped = 0
                if step >= cfg['stage_b_min'] and score < best:
                    best, beststep, beststate = score, step, copy.deepcopy(ema.state_dict())
                save(last, dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                    generator=generator.get_state(), step=step, best=best, beststep=beststep,
                    beststate=beststate, history=history, spent=spent + time.perf_counter() - started))
                print(json.dumps(dict(job=job_id, **row)), flush=True)
    if beststate is None:
        raise RuntimeError('No selected velocity checkpoint')
    assert all(p.grad is None and not p.requires_grad for p in path.parameters())
    save(out / 'velocity.pt', beststate)
    result = dict(**stamp, executed_updates=cfg['stage_b_cap'], selected_update=beststep,
        selected_validation_W2=best, history=history, path_frozen=True,
        wall_seconds=spent + time.perf_counter() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
        velocity_sha256=digest(out / 'velocity.pt'), stage_a_updates_here=0, stage_c_updates=0)
    write_json(out / 'result.json', result)


def moment_residual(path, data, normalized=False):
    if isinstance(path, BridgeTeacher):
        return bridge_residual(path, data, normalized)
    return path_residual(path, data, normalized)


def load_teacher(folder, job):
    data = load_data(job['data'])
    if job['method'] == 'cfm':
        path = LinearTeacher()
    elif job['method'] == 'csb':
        with np.load(folder / 'bridge.npz') as arrays:
            path = BridgeTeacher(dict(arrays))
    else:
        path = new_path(job['seed'], job['config'], data.x0.shape[1])
        path.load_state_dict(torch.load(folder / 'path.pt', weights_only=True))
    path.eval().requires_grad_(False)
    path.zero_grad(set_to_none=True)
    return data, path


def fit(data_root, out, method, seed, cfg, velocity_cfg):
    group = 'entropic' if method in ['land', 'mfm'] else 'exact'
    job = dict(id=f'{method}_seed{seed}', method=method, seed=seed,
               config=cfg, data=str(data_root / group))
    train_path(data_root / group, out, job)
    data, path = load_teacher(out, job)
    protocol = dict(validation=str(data_root / 'endpoint_validation.npz'),
                    validation_key='x_1', endpoints=[0, 36])
    train_velocity(protocol, job, velocity_cfg, data, path, out)


def evaluate(data_root, out, method, seed, cfg, velocity_cfg):
    group = 'entropic' if method in ['land', 'mfm'] else 'exact'
    data, path = load_teacher(out, dict(method=method, seed=seed, config=cfg,
                                      data=data_root / group))
    model = new_velocity(seed, velocity_cfg, data.x0.shape[1])
    model.load_state_dict(torch.load(out / 'velocity.pt', weights_only=True))
    model.eval()
    hours = [0, 6, 12, 24, 36]
    source = source_particles(data.x0, velocity_cfg['stage_b_input_noise'], velocity_cfg['final_draws'])
    generated = rollout(model, source, hours, [0, 36], velocity_cfg['ode_steps_final'])
    refined = rollout(model, source, hours, [0, 36], velocity_cfg['ode_steps_check'])
    more = rollout(model, source_particles(data.x0, velocity_cfg['stage_b_input_noise'],
                   velocity_cfg['sensitivity_draws']), hours, [0, 36], velocity_cfg['ode_steps_final'])
    with np.load(data_root / 'evaluation.npz', allow_pickle=False) as truth:
        scores = {h: exact_wasserstein(generated[h].numpy(), truth[f'x_{h}']) for h in hours}
        fine = {h: exact_wasserstein(refined[h].numpy(), truth[f'x_{h}']) for h in hours}
        sensitivity = {h: exact_wasserstein(more[h].numpy(), truth[f'x_{h}']) for h in hours}
    with torch.no_grad():
        direct = (moment_residual(path, data) / data.target).tolist()
        residual = (features(generated[6], data.matrix).mean(0) / data.target - 1).tolist()
    result = dict(scores=scores, fine_scores=fine, sampling_scores=sensitivity,
                  direct_relative_residual=direct, rollout_relative_residual=residual,
                  interpretation=data.metadata['interpretation'])
    write_json(out / 'evaluation.json', result)
    np.savez_compressed(out / 'particles.npz', **{f'x_{h}': x.numpy() for h, x in generated.items()})
    return result
