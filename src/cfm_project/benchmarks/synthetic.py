"""Synthetic diffusion experiment with sealed empirical evaluation pools."""
import copy
import numpy as np
import torch
from cfm_project import training as tr
from cfm_project.data import EmpiricalCouplingProblem, moment_feature_vector_from_samples
from cfm_project.models import VelocityField, PathCorrection
from cfm_project.quadratic_moment_sb import solve_mean_coordinate_variance_bridge
from .common import write_json, save, snapshots, w2, notice
from .synthetic_bridge import ExactTeacher
from .bridge_velocity import fit_velocity, load_velocity


def load_inputs(data_root):
    with np.load(data_root / 'training_inputs.npz', allow_pickle=False) as arrays:
        a = dict(arrays)
    problem = EmpiricalCouplingProblem(torch.from_numpy(a['x0']), torch.from_numpy(a['x1']),
        label='paper_synthetic', global_ot_src_idx=torch.as_tensor(a['src_idx'], dtype=torch.long),
        global_ot_tgt_idx=torch.as_tensor(a['tgt_idx'], dtype=torch.long),
        global_ot_mass=torch.as_tensor(a['mass'], dtype=torch.float32))
    target = torch.from_numpy(np.r_[a['target_mean'], a['target_y_variance']].astype(np.float32))
    return problem, target


def fit(data_root, out, method, seed, cfg):
    problem, target = load_inputs(data_root)
    if method == 'csb':
        bridge = solve_mean_coordinate_variance_bridge(problem.x0_pool.numpy(), problem.x1_pool.numpy(),
            target[:2].numpy(), float(target[2]), coordinate=1, sigma=cfg['sigma'], progress=notice)
        teacher = ExactTeacher(bridge)
        np.savez_compressed(out / 'bridge.npz', x0=bridge.x0, x1=bridge.x1, coupling=bridge.coupling,
            conditional_covariance=bridge.conditional_covariance, linear_multiplier=bridge.linear_multiplier,
            quadratic_multiplier=bridge.quadratic_multiplier, target_mean=bridge.target_mean,
            target_y_variance=bridge.target_variance)
        write_json(out / 'stage_a.json', bridge.diagnostics)
        fit_velocity(cfg, teacher, seed, cfg['learning_rate'], out / 'velocity')
        return
    cfg = copy.deepcopy(cfg)
    cfg['seed'] = seed
    # Intermediate pools never enter training, even through diagnostics.
    blocks = cfg['data']['moment_feature_blocks']
    active_target = target[:2] if blocks == ['mean'] else target[2:] if blocks == ['y_variance'] else target
    cfg['data'].update(moment_feature_params={}, constraint_times=[.5])
    cfg['mfm']['reference_pool_policy'] = 'endpoints_only'
    cfg['train'].update(eval_empirical_w2_full_pool=False, eval_full_ot_metrics=False,
                        eval_intermediate_empirical_w2=False)
    result = tr.train_experiment(cfg, problem=problem, targets={.5: active_target}, data_family='bridge_sde', evaluate=False)
    save(out / 'checkpoint.pt', result['checkpoint'])
    write_json(out / 'history.json', result['history'])
    write_json(out / 'result.json', result['summary'])


def evaluate(data_root, out, method, seed, cfg):
    problem, target = load_inputs(data_root)
    if method == 'csb':
        model = load_velocity(cfg, out / 'velocity/matched.pt')
        path = None
    else:
        state = torch.load(out / 'checkpoint.pt', weights_only=False, map_location='cpu')
        model = VelocityField(2, cfg['model']['velocity_hidden_dims'], cfg['model']['activation'])
        model.load_state_dict(state['velocity_state_dict'])
        path = None
        if state.get('path_state_dict') is not None:
            path = PathCorrection(2, cfg['model']['path_hidden_dims'], cfg['model']['activation'],
                                  zero_init_output=cfg['model'].get('path_zero_init_output', False))
            path.load_state_dict(state['path_state_dict'])
    model.eval()
    generated = snapshots(model, problem.x0_pool, [.25, .5, .75, 1.], 100)
    with np.load(data_root / 'evaluation_only.npz', allow_pickle=False) as truth:
        scores = {str(t): w2(x.numpy(), truth[f'{t:.2f}']) for t, x in generated.items()}
    residual = moment_feature_vector_from_samples(generated[.5], ('mean', 'y_variance')) - target
    direct = None
    if method != 'csb':
        a, b = tr._global_ot_support_pairs(problem)
        with torch.no_grad():
            x = tr._path_samples_for_mode(mode=cfg['experiment']['mode'], x0=a, x1=b,
                t_batch=torch.full((len(a), 1), .5), g_model=path, mfm_alpha=cfg['mfm']['alpha'])
            direct = (moment_feature_vector_from_samples(x, ('mean', 'y_variance')) - target).tolist()
    result = dict(W2=scores, direct_residual=direct, rollout_residual=residual.tolist(), euler_steps=100)
    write_json(out / 'evaluation.json', result)
    np.savez_compressed(out / 'particles.npz', **{str(t): x.numpy() for t, x in generated.items()})
    return result
