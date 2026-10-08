"""Scientific invariants at the portable runner boundary."""
from pathlib import Path

import numpy as np
import pytest
import torch

from cfm_project.benchmarks import cli, multi, multi_clock, multi_path, semrau, synthetic
from cfm_project.benchmarks.common import read_json
from cfm_project.models import PathCorrection
from cfm_project.paths import path_and_velocity

ROOT = Path(__file__).resolve().parents[1]


def test_prepared_bundle_hashes_and_training_information(monkeypatch):
    assert len(cli.verify_data(ROOT / 'benchmark_data')) >= 32
    original = np.load
    accessed = []
    def train_only(file, *args, **kwargs):
        name = Path(file).name
        assert name not in ['evaluation_only.npz', 'evaluation.npz', 'validation.npy', 'endpoint_validation.npz']
        accessed.append(name)
        return original(file, *args, **kwargs)
    monkeypatch.setattr(np, 'load', train_only)
    synthetic.load_inputs(ROOT/'benchmark_data/synthetic/seed_3')
    semrau.load_data(ROOT/'benchmark_data/semrau/entropic')
    study = 'constraint_day3_eval_day4'
    state = multi.initial_state(ROOT/'benchmark_data/multi'/study, 'land', 3,
        read_json(ROOT/'configs/paper/multi'/study/'land.json'))
    assert state['references'].shape[0] == 512
    assert 'training_inputs.npz' in accessed
    assert 'train.npz' in accessed


@pytest.mark.parametrize('kind', ['lin', 'land'])
def test_extracted_multi_derivative_and_parameter_gradients_match_reference(kind):
    torch.manual_seed(19)
    model = PathCorrection(3, [8], 'silu').double()
    a, b = torch.randn(7, 3, dtype=torch.float64), torch.randn(7, 3, dtype=torch.float64)
    t = torch.linspace(.1, .9, 7, dtype=torch.float64)[:, None]
    state = dict(model=model, method=kind)
    x, u = multi_path.fast_path_velocity(state, t, a, b)
    if kind == 'lin':
        xx, uu, _ = path_and_velocity('constrained', t, a, b, model, create_graph=True)
    else:
        from cfm_project.mfm_core import mfm_path_and_velocity
        xx, uu, _ = mfm_path_and_velocity(t=t, x0=a, x1=b, geopath_net=model, alpha=1., create_graph=True)
    torch.testing.assert_close(x, xx, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(u, uu, rtol=1e-12, atol=1e-12)
    grads = torch.autograd.grad(u.square().sum(), tuple(model.parameters()), retain_graph=True)
    reference = torch.autograd.grad(uu.square().sum(), tuple(model.parameters()))
    for actual, expected in zip(grads, reference):
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


def test_multi_clock_chain_rule_and_endpoints():
    torch.manual_seed(11)
    s = dict(model=PathCorrection(2, [8], 'silu').double(), method='land')
    a, b = torch.randn(5, 2, dtype=torch.float64), torch.randn(5, 2, dtype=torch.float64)
    t = torch.linspace(.1, .9, 5, dtype=torch.float64)[:, None]
    shift = torch.tensor(.7, dtype=torch.float64)
    x, u = multi_clock.path_velocity(s, t, a, b, shift)
    _, derivative = torch.func.jvp(lambda tt: multi_clock.path_velocity(s, tt, a, b, shift)[0],
                                    (t,), (torch.ones_like(t),))
    torch.testing.assert_close(u, derivative, rtol=1e-12, atol=1e-12)
    for at, endpoint in [(0., a), (1., b)]:
        actual = multi_clock.path_velocity(s, torch.full_like(t, at), a, b, shift)[0]
        torch.testing.assert_close(actual, endpoint, atol=0, rtol=0)


def test_w2_checkpoint_control_keeps_raw_minimum_and_uses_declared_patience():
    state = multi_path.initial_control()
    assert multi_path.assess_w2(state, 2., 300)[1]
    # Sub-delta improvement selects a checkpoint but does not reset patience.
    assert multi_path.assess_w2(state, 1.9995, 600)[1:]==(True, False)
    assert state['bad_checks'] == 1 and state['best_step'] == 600
    for step in [900, 1200, 1500, 1800, 2100]:
        action, _, _ = multi_path.assess_w2(state, 2., step)
    assert action == 'stop' and state['best_step'] == 600


def test_synthetic_training_does_not_call_empirical_evaluation(monkeypatch):
    from cfm_project import training as tr
    cfg = read_json(ROOT/'configs/paper/synthetic/lin.json')
    cfg['seed'] = 3
    cfg['train'].update(stage_a_steps=1,stage_b_steps=1,stage_c_steps=0)
    problem, target = synthetic.load_inputs(ROOT/'benchmark_data/synthetic/seed_3')
    def forbidden(*args, **kwargs): raise AssertionError('Empirical evaluation leaked into fit')
    monkeypatch.setattr(tr, '_eval_empirical_rollout_metrics', forbidden)
    monkeypatch.setattr(tr, 'interpolant_empirical_w2_metrics', forbidden)
    result = tr.train_experiment(cfg, problem, {.5: target}, data_family='bridge_sde', evaluate=False)
    assert result['rollout_artifacts'] is None
    assert result['summary']['stage_c_enabled'] is False


def test_semrau_constraint_has_gradients_and_uses_declared_features():
    data = semrau.load_data(ROOT/'benchmark_data/semrau/exact')
    cfg = read_json(ROOT/'configs/paper/semrau/lin.json')
    model = semrau.new_path(3, cfg, 12)
    residual = semrau.moment_residual(model, data, True)
    assert residual.requires_grad and residual.numel() == 2
    residual.square().sum().backward()
    assert sum(float(p.grad.square().sum()) for p in model.parameters()) > 0


def test_changed_training_input_is_rejected(tmp_path):
    from cfm_project.benchmarks.common import write_json
    (tmp_path/'train.npz').write_bytes(b'changed')
    write_json(tmp_path/'manifest.json',dict(files={'train.npz':{'sha256':'wrong'}}))
    with pytest.raises(ValueError, match='input changed'):
        cli.verify_data(tmp_path)
