"""LAND variants of the paired scarce-marginal experiment."""
import json
from .scarce import RandomStreams, TIMES, euler_velocity_snapshots, diagnostic
from pathlib import Path
import hashlib
import time
import numpy as np
import torch
from omegaconf import OmegaConf
from cfm_project import training as tr
from cfm_project.mfm_core import land_geopath_loss, mfm_mean_path, mfm_path_and_velocity
from cfm_project.models import PathCorrection, VelocityField
from cfm_project.multimarginal import local_to_global_time_and_velocity
from .common import write_json as SAVE, read_json as READ

class ScarceLandStudy:
    def __init__(self, out, parent):
        self.out = Path(out)
        self.parent = parent


    def run_dir(self, method, repeat, n, smoke=False):
        return self.out / ('smoke' if smoke else 'runs') / method / f'repeat_{repeat}/n_{n}'

    def path_model(self, state=None):
        g = PathCorrection(2, [128, 128], 'silu')
        if state is None:
            state = torch.load(self.parent.out / 'inputs/initial_weights.pt', weights_only=True)['path']
        g.load_state_dict(state)
        return g

    def base_teacher(self, method, repeat, n):
        if method == 'gmi_land':
            teacher = self.parent.build_teacher('cfm', repeat, n, self.out)
        else:
            teacher = self.parent.build_teacher('mm_empirical', repeat, n, self.out)
        teacher['method'] = method
        return teacher

    def stage_a_land(self, repeat, n, run, smoke):
        cfg = OmegaConf.to_container(OmegaConf.load(self.out / 'stage_a_reference_config.yaml'), resolve=True)
        steps = 2 if smoke else 600
        cfg['train'].update(stage_a_steps=steps, stage_b_steps=0, stage_c_steps=0)
        OmegaConf.save(OmegaConf.create(cfg), run / 'stage_a_config.yaml')
        problem = self.parent.endpoints()
        x0, x1 = tr._global_ot_support_pairs(problem)
        target = torch.tensor(self.parent.npz(self.parent.case_dir(repeat, n) / 'moments.npz')['target'], dtype=torch.float32)
        refs = torch.load(self.out / 'global_land_reference.pt', weights_only=True)['global_pool']
        calls = []

        def sampler(t, count, generator=None):
            assert float(t) in (0.0, 1.0), f'Forbidden sampler time {t}'
            calls.append([float(t), count])
            pool = problem.x0_pool if t == 0 else problem.x1_pool
            return pool[torch.randint(len(pool), (count,), generator=generator)]
        names = ['_metric_constrained_geopath_objective', 'update_lagrange_multiplier_blocks', '_build_metric_reference_pool', '_eval_empirical_rollout_metrics', 'interpolant_empirical_w2_metrics', 'interpolant_snapshot_sets']
        originals = {name: getattr(tr, name) for name in names}
        state = dict(step=0, diagnostics=[])
        batch_hash, time_hash = (hashlib.sha256(), hashlib.sha256())

        def objective(**kw):
            state['last'] = kw
            batch_hash.update(kw['x0'].numpy().tobytes())
            batch_hash.update(kw['x1'].numpy().tobytes())
            time_hash.update(torch.get_rng_state().numpy().tobytes())
            if state['step'] == 0:
                state['initial_path_hash'] = self.parent.model_hash(kw['geopath_model'])
                assert state['initial_path_hash'] == READ(self.parent.out / 'protocol.json')['initial_path_hash']
            if state['step'] % 50 == 0:
                before = torch.get_rng_state().clone()
                state['diagnostics'].append(diagnostic(kw, x0, x1, state['step'], cfg, 'land'))
                assert torch.equal(before, torch.get_rng_state())
                SAVE(run / 'stage_a_diagnostics.json', state['diagnostics'])
            value = originals['_metric_constrained_geopath_objective'](**kw)
            assert torch.isfinite(value[0])
            state['step'] += 1
            return value

        def update(**kw):
            result = originals['update_lagrange_multiplier_blocks'](**kw)
            state['last']['lambdas'] = result
            return result

        def reference(**kw):
            assert kw['reference_pool_policy'] == 'endpoints_only' and kw['n_samples_per_time'] == 512
            return refs.clone()
        tr._metric_constrained_geopath_objective = objective
        tr.update_lagrange_multiplier_blocks = update
        tr._build_metric_reference_pool = reference
        tr._eval_empirical_rollout_metrics = lambda **kw: ({}, {})
        tr.interpolant_empirical_w2_metrics = lambda **kw: {}
        tr.interpolant_snapshot_sets = lambda **kw: ({}, {}, {})
        start = time.perf_counter()
        try:
            result = tr.train_experiment(cfg, problem=problem, targets={0.5: target}, target_sampler=sampler, target_samples_by_time=None, data_family='bridge_sde')
        finally:
            for name, fn in originals.items():
                setattr(tr, name, fn)
        state['diagnostics'].append(diagnostic(state['last'], x0, x1, steps, cfg, 'land'))
        assert state['step'] == steps and len(result['history']) == steps
        g = result['path_model'].eval().requires_grad_(False)
        with torch.no_grad():
            for t, expected in [(0.0, x0), (1.0, x1)]:
                assert torch.equal(mfm_mean_path(torch.full((len(x0), 1), t), x0, x1, g, alpha=1.0), expected)
        torch.save(g.state_dict(), run / 'path.pt')
        SAVE(run / 'stage_a_history.json', result['history'])
        SAVE(run / 'stage_a_diagnostics.json', state['diagnostics'])
        audit = dict(steps=steps, initial_path_hash=state['initial_path_hash'], batch_hash=batch_hash.hexdigest(), time_hash=time_hash.hexdigest(), endpoint_sampler_calls=calls, reference_pool_hash=self.parent.tensor_hash(refs), reference_pool_policy='endpoints_only', wall_seconds=time.perf_counter() - start, path_hash=self.parent.model_hash(g), initial_velocity_hash=self.parent.model_hash(result['velocity_model']))
        parent = READ(self.parent.run_dir('gmi', repeat, n, smoke) / 'stage_a_audit.json')
        for key in ['initial_path_hash', 'batch_hash', 'time_hash', 'initial_velocity_hash']:
            assert audit[key] == parent[key], (key, audit[key], parent[key])
        SAVE(run / 'stage_a_audit.json', audit)
        return [g]

    def mm_pairs(self, teacher, pair):
        half = len(pair) // 2
        for k, seg in enumerate(teacher['segments']):
            ids = np.searchsorted(seg['cdf'], pair[k * half:(k + 1) * half], side='right')
            yield (seg['a'][seg['src'][ids]], seg['b'][seg['tgt'][ids]])

    def stage_a_mm(self, teacher, run, smoke):
        steps = 2 if smoke else 600
        models = [self.path_model(), self.path_model()]
        initial_hashes = [self.parent.model_hash(g) for g in models]
        assert set(initial_hashes) == {READ(self.parent.out / 'protocol.json')['initial_path_hash']}
        uniforms = torch.load(self.out / 'segment_reference_uniforms.pt', weights_only=True)
        refs, indices = ([], [])
        for k, seg in enumerate(teacher['segments']):
            ids = [torch.floor(uniforms[k, j] * len(seg[key])).long() for j, key in enumerate(['a', 'b'])]
            indices.append(ids)
            refs.append(torch.cat([seg['a'][ids[0]], seg['b'][ids[1]]]))
        torch.save(dict(pools=refs, indices=indices), run / 'land_references.pt')
        optimizer = torch.optim.Adam([p for g in models for p in g.parameters()], lr=0.001)
        stream = RandomStreams(300000)
        diagnostic_draws = RandomStreams(400000).draw(512)
        history, diagnostics = ([], [])

        def losses(draws):
            pair, local, *_ = draws
            result = []
            for k, (a, b) in enumerate(self.mm_pairs(teacher, pair)):
                x, u, _ = mfm_path_and_velocity(t=local[k * len(a):(k + 1) * len(a)], x0=a, x1=b, geopath_net=models[k], alpha=1.0, create_graph=True)
                result.append(land_geopath_loss(x, u, refs[k], gamma=0.125, rho=0.001))
            return result

        def diagnose(step):
            ls = losses(diagnostic_draws)
            value = torch.stack(ls).mean()
            parameters = [p for g in models for p in g.parameters()]
            gradients = torch.autograd.grad(value, parameters)
            row = dict(step=step, base_loss=float(value.detach()), segment_loss=[float(x.detach()) for x in ls], gradient_l2=float(torch.cat([g.flatten() for g in gradients]).norm()), gradient_clipping=False)
            assert all((np.isfinite(x) for x in [row['base_loss'], row['gradient_l2']]))
            diagnostics.append(row)
            SAVE(run / 'stage_a_diagnostics.json', diagnostics)
        start = time.perf_counter()
        for step in range(steps):
            if step % 50 == 0:
                diagnose(step)
            ls = losses(stream.draw(256, track=True))
            total = torch.stack(ls).mean()
            assert torch.isfinite(total)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            history.append(dict(step=step + 1, loss=float(total.detach()), segments=[float(x.detach()) for x in ls]))
        diagnose(steps)
        for k, (g, seg) in enumerate(zip(models, teacher['segments'])):
            g.eval().requires_grad_(False)
            with torch.no_grad():
                a, b = (seg['a'][seg['src']], seg['b'][seg['tgt']])
                for t, expected in [(0.0, a), (1.0, b)]:
                    assert torch.equal(mfm_mean_path(torch.full((len(a), 1), t), a, b, g, alpha=1.0), expected)
            torch.save(g.state_dict(), run / f'path_{k}.pt')
        SAVE(run / 'stage_a_history.json', history)
        SAVE(run / 'stage_a_audit.json', dict(steps=steps, batch_size_per_segment=128, initial_path_hashes=initial_hashes, path_hashes=[self.parent.model_hash(g) for g in models], random_stream_hashes=stream.fingerprints(), reference_pool_hashes=[self.parent.tensor_hash(x) for x in refs], reference_pool_policy='segment_endpoints_only', reference_pool_sizes=[len(x) for x in refs], wall_seconds=time.perf_counter() - start))
        return models

    def conditional_batch(self, teacher, draws):
        pair, local, t, normal, noise = draws
        if teacher['method'] == 'gmi_land':
            ids = np.searchsorted(teacher['cdf'], pair, side='right')
            a = teacher['x0'][teacher['src'][ids]]
            b = teacher['x1'][teacher['tgt'][ids]]
            x, u, _ = mfm_path_and_velocity(t=t, x0=a, x1=b, geopath_net=teacher['models'][0], alpha=1.0, create_graph=False)
        else:
            xs, us = ([], [])
            for k, (a, b) in enumerate(self.mm_pairs(teacher, pair)):
                x, u, _ = mfm_path_and_velocity(t=local[k * len(a):(k + 1) * len(a)], x0=a, x1=b, geopath_net=teacher['models'][k], alpha=1.0, create_graph=False)
                _, u = local_to_global_time_and_velocity(local_time=local[k * len(a):(k + 1) * len(a)], local_velocity=u, start_time=0.5 * k, end_time=0.5 * (k + 1))
                xs.append(x)
                us.append(u)
            x, u = (torch.cat(xs), torch.cat(us))
        return (t, (x + 0.1 * noise).detach(), u.detach(), torch.ones_like(t))

    def train(self, method, repeat, n, smoke=False):
        run = self.run_dir(method, repeat, n, smoke)
        if (run / 'complete.json').exists():
            return
        assert not (run / 'started.json').exists(), f'Partial run exists: {run}'
        run.mkdir(parents=True, exist_ok=True)
        SAVE(run / 'started.json', dict(method=method, repeat=repeat, n=n, smoke=smoke))
        accessed = []
        original_load = np.load

        def guarded(file, *args, **kw):
            path = str(file)
            assert '/evaluation_only/' not in path and '/observation_source/' not in path, path
            if method == 'gmi_land':
                assert not path.endswith('/observations.npz') and (not path.endswith('/gaussian.npz')), path
            accessed.append(path)
            return original_load(file, *args, **kw)
        np.load = guarded
        started = time.perf_counter()
        try:
            teacher = self.base_teacher(method, repeat, n)
            teacher['models'] = self.stage_a_land(repeat, n, run, smoke) if method == 'gmi_land' else self.stage_a_mm(teacher, run, smoke)
            teacher_seconds = time.perf_counter() - started
            hashes_before = [self.parent.model_hash(g) for g in teacher['models']]
            model = VelocityField(2, [128, 128], 'silu')
            model.load_state_dict(torch.load(self.parent.out / 'inputs/initial_weights.pt', weights_only=True)['velocity'])
            initial_hash = self.parent.model_hash(model)
            assert initial_hash == READ(self.parent.out / 'protocol.json')['initial_velocity_hash']
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            stream = RandomStreams()
            validation = self.conditional_batch(teacher, RandomStreams(100000).draw(4096))
            history, losses = ([], [])
            steps = 2 if smoke else 9600
            train_start = time.perf_counter()
            for step in range(1, steps + 1):
                t, x, u, w = self.conditional_batch(teacher, stream.draw(256, track=True))
                loss = ((model(t, x) - u).square().sum(1) * w[:, 0]).mean()
                assert torch.isfinite(loss), (method, repeat, n, step)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
                if step % 600 == 0 or step == steps:
                    with torch.no_grad():
                        vt, vx, vu, vw = validation
                        val = float(((model(vt, vx) - vu).square().sum(1) * vw[:, 0]).mean())
                    assert np.isfinite(val)
                    row = dict(step=step, training_loss=losses[-1], fixed_regression_loss=val, wall_seconds=time.perf_counter() - train_start)
                    history.append(row)
                    SAVE(run / 'stage_b_diagnostics.json', history)
                    if step % 2400 == 0 or step == steps:
                        print(json.dumps(dict(event='velocity', method=method, repeat=repeat, n=n, **row)), flush=True)
            assert hashes_before == [self.parent.model_hash(g) for g in teacher['models']]
            torch.save(dict(state_dict=model.state_dict(), steps=steps), run / 'velocity.pt')
            np.save(run / 'stage_b_losses.npy', np.array(losses))
            record = dict(method=method, repeat=repeat, n=n, stage_b_steps=steps, batch_size=256, stage_a_steps=2 if smoke else 600, teacher_build_seconds=teacher_seconds, stage_b_seconds=time.perf_counter() - train_start, total_seconds=time.perf_counter() - started, initial_velocity_hash=initial_hash, random_stream_hashes=stream.fingerprints(), path_frozen=True, velocity_sha256=self.parent.digest(run / 'velocity.pt'), velocity_state_hash=self.parent.model_hash(model), test_information_used=False, checkpoint_selection='final fixed budget', loaded_numpy_files=sorted(set(accessed)))
            parent = READ(self.parent.run_dir('cfm', smoke=smoke) / 'complete.json')
            assert record['random_stream_hashes'] == parent['random_stream_hashes']
            SAVE(run / 'complete.json', record)
            print(json.dumps(dict(event='complete', **record)), flush=True)
        finally:
            np.load = original_load

    def load_teacher(self, method, repeat, n, run):
        teacher = self.base_teacher(method, repeat, n)
        names = ['path.pt'] if method == 'gmi_land' else ['path_0.pt', 'path_1.pt']
        teacher['models'] = [self.path_model(torch.load(run / name, weights_only=True)).eval().requires_grad_(False) for name in names]
        return teacher

    def direct_samples(self, teacher):
        result = {}
        with torch.no_grad():
            if teacher['method'] == 'gmi_land':
                a = teacher['x0'][teacher['src']]
                b = teacher['x1'][teacher['tgt']]
                for t in TIMES:
                    x = mfm_mean_path(torch.full((len(a), 1), t), a, b, teacher['models'][0], alpha=1.0)
                    result[t] = (x.numpy(), None)
            else:
                for k, seg in enumerate(teacher['segments']):
                    a = seg['a'][seg['src']]
                    b = seg['b'][seg['tgt']]
                    x = mfm_mean_path(torch.full((len(a), 1), 0.5), a, b, teacher['models'][k], alpha=1.0)
                    result[0.25 + 0.5 * k] = (x.numpy(), seg['weights'])
                result[0.5] = (teacher['midpoint'].numpy(), None)
                result[1.0] = (teacher['x1'].numpy(), None)
        return result

    def evaluate(self, method, repeat, n):
        run = self.run_dir(method, repeat, n)
        if (run / 'evaluation.json').exists():
            return
        assert READ(self.out / 'frozen_models.json')[str(run.relative_to(self.out))] == self.parent.digest(run / 'velocity.pt')
        reference = self.parent.npz(self.parent.out / 'evaluation_only/marginals.npz')
        population = np.array(READ(self.parent.out / 'evaluation_only/population_moments.json')['target'])
        teacher = self.load_teacher(method, repeat, n, run)
        model = VelocityField(2, [128, 128], 'silu')
        model.load_state_dict(torch.load(run / 'velocity.pt', weights_only=True)['state_dict'])
        model.eval()
        generated = euler_velocity_snapshots(model, teacher['x0'], TIMES, n_steps=100)
        direct = self.direct_samples(teacher)
        scores = {}
        direct_scores = {}
        for t in TIMES:
            target = teacher['x1'].numpy() if t == 1.0 else reference[f'{t:.2f}']
            scores[f'{t:.2f}'] = self.parent.weighted_w2(generated[t].numpy(), target)
            direct_scores[f'{t:.2f}'] = 0.0 if t == 1.0 else self.parent.weighted_w2(direct[t][0], target, direct[t][1])
        rollout_mom = self.parent.moments(generated[0.5].numpy())
        direct_mom = self.parent.moments(*direct[0.5])
        supplied = self.parent.npz(self.parent.case_dir(repeat, n) / 'moments.npz')['target']
        teacher_mom = direct_mom.copy()
        teacher_mom[2] += 0.01
        record = dict(method=method, repeat=repeat, n=n, w2=scores, direct_w2=direct_scores, w2_outer=0.5 * (scores['0.25'] + scores['0.75']), direct_w2_outer=0.5 * (direct_scores['0.25'] + direct_scores['0.75']), rollout_moments=rollout_mom.tolist(), direct_moments=direct_mom.tolist(), population_errors=self.parent.moment_errors(rollout_mom, population), direct_population_errors=self.parent.moment_errors(direct_mom, population), supplied_target_errors=self.parent.moment_errors(rollout_mom, supplied), direct_supplied_target_errors=self.parent.moment_errors(direct_mom, supplied), target_estimation_errors=self.parent.moment_errors(supplied, population), direct_moments_source='noise-free mean path, exact weighted discrete support', noisy_teacher_moments=teacher_mom.tolist(), noisy_teacher_supplied_errors=self.parent.moment_errors(teacher_mom, supplied), checkpoint_sha256=self.parent.digest(run / 'velocity.pt'))
        np.savez_compressed(run / 'rollout_samples.npz', **{f'{t:.2f}': generated[t].numpy() for t in TIMES})
        SAVE(run / 'evaluation.json', record)
        print(json.dumps(record), flush=True)
