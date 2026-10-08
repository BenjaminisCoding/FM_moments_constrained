"""Paired scarce-marginal experiments; independent inputs and fixed optimizer streams."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch
import ot
from scipy.spatial.distance import cdist
from omegaconf import OmegaConf
from cfm_project import training as tr
from cfm_project.data import EmpiricalCouplingProblem
from cfm_project.models import PathCorrection, VelocityField
from cfm_project.paths import corrected_path, path_and_velocity
from cfm_project.metrics import euler_velocity_snapshots
from cfm_project.multimarginal import build_adjacent_ot_segments
from cfm_project.quadratic_moment_sb import solve_mean_coordinate_variance_bridge
from .synthetic_diagnostics import diagnostic
NS = [4, 16, 64, 256, 1024]
REPEATS = [1, 2, 3, 4, 5]
TIMES = [0.25, 0.5, 0.75, 1.0]
METHODS = ['cfm', 'gmi', 'mm_empirical', 'mm_gaussian', 'csb']

class RandomStreams:

    def __init__(self, offset=0):
        self.generators = [torch.Generator().manual_seed(500003 + offset + i) for i in range(4)]
        self.hashes = [hashlib.sha256() for _ in range(4)]

    def draw(self, n, track=False):
        assert n % 2 == 0
        pair = torch.rand(n, dtype=torch.float64, generator=self.generators[0])
        local = torch.rand((n, 1), generator=self.generators[1]).clamp(0.0001, 1 - 0.0001)
        normal = torch.randn((n, 2), generator=self.generators[2])
        noise = torch.randn((n, 2), generator=self.generators[3])
        if track:
            for h, x in zip(self.hashes, [pair, local, normal, noise]):
                h.update(x.numpy().tobytes())
        t = 0.5 * local
        t[n // 2:] += 0.5
        return (pair.numpy(), local, t, normal, noise)

    def fingerprints(self):
        return [h.hexdigest() for h in self.hashes]

class ScarceStudy:

    def __init__(self, out):
        self.out = Path(out)

    def save_json(self, path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + '.tmp')
        tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
        tmp.replace(path)

    def read_json(self, path):
        return json.loads(path.read_text())

    def digest(self, path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def tensor_hash(self, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

    def model_hash(self, model):
        h = hashlib.sha256()
        for name, value in model.state_dict().items():
            h.update(name.encode())
            h.update(value.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def npz(self, path):
        with np.load(path) as data:
            return {k: data[k].copy() for k in data.files}

    def case_dir(self, repeat, n):
        return self.out / f'inputs/repeat_{repeat}/n_{n}'

    def run_dir(self, method, repeat=1, n=4, smoke=False):
        base = self.out / ('smoke' if smoke else 'runs') / method
        return base if method == 'cfm' else base / f'repeat_{repeat}/n_{n}'

    def endpoints(self):
        data = self.npz(self.out / 'inputs/endpoints.npz')
        return EmpiricalCouplingProblem(torch.from_numpy(data['x0']), torch.from_numpy(data['x1']), label='fixed_arched_endpoints', global_ot_src_idx=torch.from_numpy(data['src_idx']).long(), global_ot_tgt_idx=torch.from_numpy(data['tgt_idx']).long(), global_ot_mass=torch.from_numpy(data['mass']).float())

    def stage_a_gmi(self, repeat, n, run, smoke=False):
        cfg = OmegaConf.to_container(OmegaConf.load(self.out / 'stage_a_reference_config.yaml'), resolve=True)
        steps = 2 if smoke else 600
        cfg['train'].update(stage_a_steps=steps, stage_b_steps=0, stage_c_steps=0)
        OmegaConf.save(OmegaConf.create(cfg), run / 'stage_a_config.yaml')
        problem = self.endpoints()
        x0, x1 = tr._global_ot_support_pairs(problem)
        target = torch.tensor(self.npz(self.case_dir(repeat, n) / 'moments.npz')['target'], dtype=torch.float32)
        calls = []

        def sampler(t, count, generator=None):
            assert float(t) in (0.0, 1.0), f'Forbidden intermediate sampler call {t}'
            calls.append([float(t), count])
            pool = problem.x0_pool if t == 0 else problem.x1_pool
            return pool[torch.randint(len(pool), (count,), generator=generator)]
        names = ['_constrained_objective', 'update_lagrange_multipliers', '_eval_empirical_rollout_metrics', 'interpolant_empirical_w2_metrics', 'interpolant_snapshot_sets']
        original = {name: getattr(tr, name) for name in names}
        state = dict(step=0, diagnostics=[])
        batch_hash, time_hash = (hashlib.sha256(), hashlib.sha256())

        def objective(**kw):
            state['last'] = kw
            batch_hash.update(kw['x0'].numpy().tobytes())
            batch_hash.update(kw['x1'].numpy().tobytes())
            time_hash.update(torch.get_rng_state().numpy().tobytes())
            if state['step'] == 0:
                state['initial_path_hash'] = self.model_hash(kw['g_model'])
                assert state['initial_path_hash'] == self.read_json(self.out / 'protocol.json')['initial_path_hash']
            if state['step'] % 50 == 0:
                before = torch.get_rng_state().clone()
                state['diagnostics'].append(diagnostic(kw, x0, x1, state['step'], cfg, 'lin'))
                assert torch.equal(before, torch.get_rng_state())
            value = original['_constrained_objective'](**kw)
            assert torch.isfinite(value[0])
            state['step'] += 1
            return value

        def update(**kw):
            value = original['update_lagrange_multipliers'](**kw)
            state['last']['lambdas'] = value
            return value
        tr._constrained_objective = objective
        tr.update_lagrange_multipliers = update
        tr._eval_empirical_rollout_metrics = lambda **kw: ({}, {})
        tr.interpolant_empirical_w2_metrics = lambda **kw: {}
        tr.interpolant_snapshot_sets = lambda **kw: ({}, {}, {})
        started = time.perf_counter()
        try:
            result = tr.train_experiment(cfg, problem=problem, targets={0.5: target}, target_sampler=sampler, target_samples_by_time=None, data_family='bridge_sde')
        finally:
            for name, fn in original.items():
                setattr(tr, name, fn)
        state['diagnostics'].append(diagnostic(state['last'], x0, x1, steps, cfg, 'lin'))
        assert state['step'] == steps and len(result['history']) == steps
        g = result['path_model'].eval().requires_grad_(False)
        with torch.no_grad():
            for t, expected in [(0.0, x0), (1.0, x1)]:
                assert torch.equal(corrected_path(torch.full((len(x0), 1), t), x0, x1, g), expected)
        torch.save(g.state_dict(), run / 'path.pt')
        self.save_json(run / 'stage_a_history.json', result['history'])
        self.save_json(run / 'stage_a_diagnostics.json', state['diagnostics'])
        self.save_json(run / 'stage_a_audit.json', dict(steps=steps, initial_path_hash=state['initial_path_hash'], batch_hash=batch_hash.hexdigest(), time_hash=time_hash.hexdigest(), endpoint_sampler_calls=calls, wall_seconds=time.perf_counter() - started, path_hash=self.model_hash(g), initial_velocity_hash=self.model_hash(result['velocity_model'])))
        return g

    def build_teacher(self, method, repeat, n, run, smoke=False):
        problem = self.endpoints()
        teacher = dict(method=method, x0=problem.x0_pool, x1=problem.x1_pool)
        if method in ['cfm', 'gmi']:
            teacher.update(src=problem.global_ot_src_idx, tgt=problem.global_ot_tgt_idx, cdf=np.cumsum(problem.global_ot_mass.numpy().astype(np.float64)))
            teacher['cdf'] /= teacher['cdf'][-1]
            if method == 'gmi':
                teacher['g'] = self.stage_a_gmi(repeat, n, run, smoke)
            return teacher
        dest = self.out / f'teachers/{method}/repeat_{repeat}/n_{n}'
        dest.mkdir(parents=True, exist_ok=True)
        if method.startswith('mm_'):
            data = self.npz(self.case_dir(repeat, n) / ('observations.npz' if method == 'mm_empirical' else 'gaussian.npz'))
            mid = torch.from_numpy(data['samples'])
            if not (dest / 'segments.pt').exists():
                started = time.perf_counter()
                segments = build_adjacent_ot_segments(snapshot_times=[0.0, 0.5, 1.0], snapshot_pools=[problem.x0_pool, mid, problem.x1_pool], label_prefix=method, ot_method='balanced_lp', balanced_ot_solver='pot_emd', balanced_ot_num_itermax=2000000)
                saved = []
                for seg in segments:
                    p = seg.problem
                    saved.append(dict(src=p.global_ot_src_idx, tgt=p.global_ot_tgt_idx, mass=p.global_ot_mass))
                torch.save(saved, dest / 'segments.pt')
                self.save_json(dest / 'build.json', dict(wall_seconds=time.perf_counter() - started, support_sizes=[len(s['mass']) for s in saved], midpoint_count=len(mid)))
            segments = torch.load(dest / 'segments.pt', weights_only=False)
            for seg, a, b in zip(segments, [problem.x0_pool, mid], [mid, problem.x1_pool]):
                mass = seg['mass'].double().numpy()
                mass /= mass.sum()
                seg.update(a=a, b=b, cdf=np.cumsum(mass), weights=mass)
                seg['cdf'][-1] = 1.0
                assert np.max(np.abs(np.bincount(seg['src'], weights=mass, minlength=len(a)) - 1 / len(a))) < 1e-07
                assert np.max(np.abs(np.bincount(seg['tgt'], weights=mass, minlength=len(b)) - 1 / len(b))) < 1e-07
            teacher.update(segments=segments, midpoint=mid)
            return teacher
        assert method == 'csb'
        if not (dest / 'solution.npz').exists():
            target = self.npz(self.case_dir(repeat, n) / 'moments.npz')['target']
            bridge = solve_mean_coordinate_variance_bridge(problem.x0_pool.numpy(), problem.x1_pool.numpy(), target[:2], float(target[2]), coordinate=1, sigma=0.5, progress=lambda row: print(json.dumps(dict(event='csb_solver', repeat=repeat, n=n, **row)), flush=True))
            np.savez_compressed(dest / 'solution.npz', coupling=bridge.coupling, covariance=bridge.conditional_covariance, linear=bridge.linear_multiplier, quadratic=bridge.quadratic_multiplier, target=target)
            self.save_json(dest / 'solver.json', bridge.diagnostics)
        solution = self.npz(dest / 'solution.npz')
        teacher.update(cdf=np.cumsum(solution['coupling'].ravel()), covariance=torch.tensor(solution['covariance'], dtype=torch.float32), linear=torch.tensor(solution['linear'], dtype=torch.float32))
        teacher['cdf'][-1] = 1.0
        teacher['chol'] = torch.linalg.cholesky(teacher['covariance'])
        analytic_mean = ((0.5 * problem.x0_pool.double().mean(0).numpy() + 0.5 * problem.x1_pool.double().mean(0).numpy()) / 0.0625 + solution['linear']) @ solution['covariance']
        teacher['analytic_moments'] = np.r_[analytic_mean, self.read_json(dest / 'solver.json')['attained_variance']]
        return teacher

    def conditional_batch(self, teacher, draws):
        pair, local, t, normal, noise = draws
        method = teacher['method']
        n = len(t)
        half = n // 2
        if method in ['cfm', 'gmi']:
            ids = np.searchsorted(teacher['cdf'], pair, side='right')
            a = teacher['x0'][teacher['src'][ids]]
            b = teacher['x1'][teacher['tgt'][ids]]
            if method == 'cfm':
                x, u = ((1 - t) * a + t * b, b - a)
            else:
                with torch.enable_grad():
                    x, u, _ = path_and_velocity('constrained', t, a, b, teacher['g'], create_graph=False)
            return (t, x.detach(), u.detach(), torch.ones_like(t))
        if method.startswith('mm_'):
            aa, bb = ([], [])
            for k, seg in enumerate(teacher['segments']):
                ids = np.searchsorted(seg['cdf'], pair[k * half:(k + 1) * half], side='right')
                aa.append(seg['a'][seg['src'][ids]])
                bb.append(seg['b'][seg['tgt'][ids]])
            a, b = (torch.cat(aa), torch.cat(bb))
            return (t, (1 - local) * a + local * b, 2 * (b - a), torch.ones_like(t))
        ids = np.searchsorted(teacher['cdf'], pair, side='right')
        i, j = np.divmod(ids, len(teacher['x1']))
        a, b = (teacher['x0'][i], teacher['x1'][j])
        mean = ((0.5 * a + 0.5 * b) / 0.0625 + teacher['linear']) @ teacher['covariance']
        z = mean + normal @ teacher['chol'].T
        return self.brownian_from_draws(a, z, b, t, noise)

    def brownian_from_draws(self, a, z, b, t, noise):
        first = t < 0.5
        local = torch.where(first, 2 * t, 2 * t - 1).clamp(0.0001, 1 - 0.0001)
        t = torch.where(first, 0.5 * local, 0.5 + 0.5 * local)
        left, right = (torch.where(first, a, z), torch.where(first, z, b))
        delta = (0.25 * 0.5 * local * (1 - local)).sqrt() * noise
        state = (1 - local) * left + local * right + delta
        velocity = 2 * (right - left) + (1 - 2 * local) / (local * (1 - local)) * delta
        return (t, state, velocity, 6 * local * (1 - local))

    def train(self, method, repeat, n, smoke=False):
        run = self.run_dir(method, repeat, n, smoke)
        if (run / 'complete.json').exists():
            return
        assert not (run / 'started.json').exists(), f'Partial run exists: {run}'
        run.mkdir(parents=True, exist_ok=True)
        self.save_json(run / 'started.json', dict(method=method, repeat=repeat, n=n, smoke=smoke))
        accessed = []
        original_load = np.load

        def guarded_load(file, *args, **kwargs):
            path = str(file)
            assert '/evaluation_only/' not in path and '/observation_source/' not in path, path
            if method in ['cfm', 'gmi', 'csb']:
                assert not path.endswith('/observations.npz') and (not path.endswith('/gaussian.npz')), path
            accessed.append(path)
            return original_load(file, *args, **kwargs)
        np.load = guarded_load
        try:
            start = time.perf_counter()
            teacher = self.build_teacher(method, repeat, n, run, smoke)
            teacher_seconds = time.perf_counter() - start
            initial = torch.load(self.out / 'inputs/initial_weights.pt', weights_only=False)
            model = VelocityField(2, [128, 128], 'silu')
            model.load_state_dict(initial['velocity'])
            initial_hash = self.model_hash(model)
            assert initial_hash == self.read_json(self.out / 'protocol.json')['initial_velocity_hash']
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            train_stream = RandomStreams()
            validation = self.conditional_batch(teacher, RandomStreams(100000).draw(4096))
            history = []
            losses = []
            steps = 2 if smoke else 9600
            path_before = self.model_hash(teacher['g']) if method == 'gmi' else None
            train_start = time.perf_counter()
            for step in range(1, steps + 1):
                t, x, u, w = self.conditional_batch(teacher, train_stream.draw(256, track=True))
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
                    self.save_json(run / 'stage_b_diagnostics.json', history)
                    if step % 2400 == 0 or step == steps:
                        print(json.dumps(dict(event='velocity', method=method, repeat=repeat, n=n, **row)), flush=True)
            if method == 'gmi':
                assert path_before == self.model_hash(teacher['g'])
            torch.save(dict(state_dict=model.state_dict(), steps=steps), run / 'velocity.pt')
            np.save(run / 'stage_b_losses.npy', np.array(losses))
            record = dict(method=method, repeat=repeat, n=n, stage_b_steps=steps, batch_size=256, stage_a_steps=(2 if smoke else 600) if method == 'gmi' else 0, teacher_build_seconds=teacher_seconds, stage_b_seconds=time.perf_counter() - train_start, total_seconds=time.perf_counter() - start, initial_velocity_hash=initial_hash, random_stream_hashes=train_stream.fingerprints(), path_frozen=True, velocity_sha256=self.digest(run / 'velocity.pt'), velocity_state_hash=self.model_hash(model), test_information_used=False, checkpoint_selection='final fixed budget', loaded_numpy_files=sorted(set(accessed)))
            self.save_json(run / 'complete.json', record)
            print(json.dumps(dict(event='complete', **record)), flush=True)
        finally:
            np.load = original_load

    def weighted_w2(self, x, y, weights=None):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        a = np.full(len(x), 1 / len(x)) if weights is None else np.asarray(weights, dtype=np.float64)
        a = a / a.sum()
        b = np.full(len(y), 1 / len(y))
        cost = cdist(x, y, 'sqeuclidean')
        value, log = ot.emd2(a, b, cost, numItermax=2000000, log=True)
        assert log.get('warning') is None, log
        assert np.isfinite(value) and value >= -1e-12
        return float(np.sqrt(max(value, 0.0)))

    def moments(self, samples, weights=None):
        x = np.asarray(samples, dtype=np.float64)
        w = np.full(len(x), 1 / len(x)) if weights is None else np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
        mean = w @ x
        return np.r_[mean, float(w @ (x[:, 1] - mean[1]) ** 2)]

    def moment_errors(self, actual, target):
        actual = np.asarray(actual)
        target = np.asarray(target)
        delta = actual - target
        return dict(mean=float(np.linalg.norm(delta[:2])), variance_y=abs(float(delta[2])), standard_y=abs(float(np.sqrt(actual[2]) - np.sqrt(target[2]))), joint=float(np.linalg.norm(delta)), vector=delta.tolist())

    def load_teacher(self, method, repeat, n, run):
        if method != 'gmi':
            return self.build_teacher(method, repeat, n, run)
        teacher = self.build_teacher('cfm', repeat, n, run)
        g = PathCorrection(2, [128, 128], 'silu')
        g.load_state_dict(torch.load(run / 'path.pt', weights_only=True))
        teacher.update(method='gmi', g=g.eval().requires_grad_(False))
        return teacher

    def direct_samples(self, teacher):
        method = teacher['method']
        result = {}
        if method in ['cfm', 'gmi']:
            a = teacher['x0'][teacher['src']]
            b = teacher['x1'][teacher['tgt']]
            with torch.no_grad():
                for t in TIMES:
                    x = (1 - t) * a + t * b if method == 'cfm' else corrected_path(torch.full((len(a), 1), t), a, b, teacher['g'])
                    result[t] = (x.numpy(), None)
        elif method.startswith('mm_'):
            for t, seg in zip([0.25, 0.75], teacher['segments']):
                x = 0.5 * (seg['a'][seg['src']] + seg['b'][seg['tgt']])
                result[t] = (x.numpy(), seg['weights'])
            result[0.5] = (teacher['midpoint'].numpy(), None)
            result[1.0] = (teacher['x1'].numpy(), None)
        else:
            pair, _, _, normal, noise = RandomStreams(200000).draw(3000)
            ids = np.searchsorted(teacher['cdf'], pair, side='right')
            i, j = np.divmod(ids, len(teacher['x1']))
            a, b = (teacher['x0'][i], teacher['x1'][j])
            mean = ((0.5 * a + 0.5 * b) / 0.0625 + teacher['linear']) @ teacher['covariance']
            z = mean + normal @ teacher['chol'].T
            for t in [0.25, 0.75]:
                state = self.brownian_from_draws(a, z, b, torch.full((len(a), 1), t), noise)[1]
                result[t] = (state.numpy(), None)
            result[0.5] = (z.numpy(), None)
            result[1.0] = (teacher['x1'].numpy(), None)
        return result

    def evaluate(self, method, repeat, n):
        run = self.run_dir(method, repeat, n)
        if (run / 'evaluation.json').exists():
            return
        frozen = self.read_json(self.out / 'frozen_models.json')
        assert frozen[str(run.relative_to(self.out))] == self.digest(run / 'velocity.pt')
        reference = self.npz(self.out / 'evaluation_only/marginals.npz')
        population = np.array(self.read_json(self.out / 'evaluation_only/population_moments.json')['target'])
        teacher = self.load_teacher(method, repeat, n, run)
        model = VelocityField(2, [128, 128], 'silu')
        model.load_state_dict(torch.load(run / 'velocity.pt', weights_only=False)['state_dict'])
        model.eval()
        generated = euler_velocity_snapshots(model, teacher['x0'], TIMES, n_steps=100)
        direct = self.direct_samples(teacher)
        scores = {}
        direct_scores = {}
        for t in TIMES:
            target = teacher['x1'].numpy() if t == 1.0 else reference[f'{t:.2f}']
            scores[f'{t:.2f}'] = self.weighted_w2(generated[t].numpy(), target)
            direct_scores[f'{t:.2f}'] = 0.0 if t == 1.0 else self.weighted_w2(*[direct[t][0], target, direct[t][1]])
        rollout_mom = self.moments(generated[0.5].numpy())
        direct_mom = teacher.get('analytic_moments', self.moments(*direct[0.5]))
        supplied = None if method == 'cfm' else self.npz(self.case_dir(repeat, n) / 'moments.npz')['target']
        record = dict(method=method, repeat=repeat, n=n, w2=scores, direct_w2=direct_scores, w2_outer=0.5 * (scores['0.25'] + scores['0.75']), direct_w2_outer=0.5 * (direct_scores['0.25'] + direct_scores['0.75']), rollout_moments=rollout_mom.tolist(), direct_moments=direct_mom.tolist(), population_errors=self.moment_errors(rollout_mom, population), direct_population_errors=self.moment_errors(direct_mom, population), supplied_target_errors=None if supplied is None else self.moment_errors(rollout_mom, supplied), direct_supplied_target_errors=None if supplied is None else self.moment_errors(direct_mom, supplied), target_estimation_errors=None if supplied is None else self.moment_errors(supplied, population), direct_moments_source='analytic continuous SB' if method == 'csb' else 'exact weighted discrete support', csb_direct_w2_sampling='fixed 3000 samples' if method == 'csb' else None, checkpoint_sha256=self.digest(run / 'velocity.pt'))
        np.savez_compressed(run / 'rollout_samples.npz', **{f'{t:.2f}': generated[t].numpy() for t in TIMES})
        self.save_json(run / 'evaluation.json', record)
        print(json.dumps(record), flush=True)
