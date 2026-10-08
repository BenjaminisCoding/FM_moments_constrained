"""Explicit preparation, fitting, freezing and evaluation of paper experiments."""
import argparse
import importlib.metadata
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
import torch
from threadpoolctl import threadpool_limits
from .common import read_json, write_json, digest, notice

REPOSITORY = Path(__file__).resolve().parents[3]
SEEDS = [3, 7, 11, 13, 17]
METHODS = ['cfm', 'mfm', 'csb', 'lin', 'land']
STUDIES = ['constraint_day3_eval_day4', 'constraint_day4_eval_day3']


def verify_data(root, prefix=''):
    manifest = read_json(root / 'manifest.json')
    selected = {k: v for k, v in manifest['files'].items() if k.startswith(prefix)}
    if not selected: raise ValueError(f'No bundled inputs for {prefix}')
    for name, record in selected.items():
        if digest(root / name) != record['sha256']:
            raise ValueError(f'Benchmark input changed: {name}')
    return {k: v['sha256'] for k, v in selected.items()}


def specification(args):
    base = args.config_root / args.benchmark
    data = args.data_root / args.benchmark
    if args.benchmark == 'synthetic':
        data /= f'seed_{args.seed}'
        cfg = read_json(base / ('csb.json' if args.method == 'csb' else f'{args.method}.json'))
        if args.method != 'csb':
            cfg['data']['moment_feature_blocks'] = {
                'both': ['mean', 'y_variance'], 'mean': ['mean'], 'variance': ['y_variance']
            }[args.moments]
        extra = {}
    elif args.benchmark == 'aerosol':
        method = 'gmi_' + args.method if args.method in ['lin', 'land'] else args.method
        cfg = read_json(base / f'{method}.json'); extra = read_json(base / 'evaluation.json')
    elif args.benchmark == 'semrau':
        cfg = read_json(base / f'{args.method}.json')
        extra = read_json(base / 'velocity.json')
    else:
        data /= args.study
        cfg = read_json(base / args.study / f'{args.method if args.method in ["lin", "land"] else "land"}.json')
        extra = read_json(base / 'budget.json')
    if args.benchmark in ['synthetic', 'multi'] and args.method != 'csb':
        cfg['seed'] = args.seed
    if args.smoke:
        # A labelled wiring check, never reported as a paper experiment.
        if args.method == 'csb':
            raise ValueError('CSB accuracy checks must not be weakened for smoke tests; use the solver unit tests.')
        if args.benchmark == 'synthetic':
            cfg['train'].update(stage_a_steps=1, stage_b_steps=2, stage_c_steps=0, log_every=1)
        elif args.benchmark == 'aerosol':
            cfg.update(warmup_steps=1, outer_steps=1, inner_steps=2, stage_b_steps=2, stage_b_log_every=1)
        elif args.benchmark == 'semrau':
            cfg.update(stage_a_cap=1, warmup_land=0)
            extra.update(stage_b_cap=2, stage_b_min=1, validation_cadence=1,
                         diagnostic_size=32, validation_draws=2, final_draws=2,
                         sensitivity_draws=2, ode_steps_validation=6, ode_steps_final=6, ode_steps_check=12)
        else:
            extra.update(stage_a_steps=1, stage_b_steps=2, validation_cadence=1,
                         clock_geometry_steps=1, clock_steps=1, clock_cadence=1, clock_min_steps=1)
    return data, cfg, extra


def contract(args, data, cfg, extra):
    prefix = str(data.relative_to(args.data_root)) + '/'
    if args.benchmark == 'semrau': prefix = 'semrau/'
    inputs = verify_data(args.data_root, prefix)
    return dict(schema=1, benchmark=args.benchmark, method=args.method, seed=args.seed,
        study=args.study if args.benchmark == 'multi' else None, config=cfg, extra_config=extra,
        smoke=args.smoke, input_sha256=inputs, device=args.device, moments=args.moments,
        seed_roster=SEEDS, reference_pool_policy='endpoints_only',
        protocol='Configuration frozen before fitting; evaluation uses the selected checkpoint.',
        versions={name: importlib.metadata.version(name) for name in ['torch', 'numpy', 'scipy', 'POT']})


def train(args):
    data, cfg, extra = specification(args)
    stamp = contract(args, data, cfg, extra)
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    if (out / 'protocol.json').exists():
        if read_json(out / 'protocol.json') != stamp: raise ValueError('Run directory has a different frozen configuration')
        if (out / 'complete.json').exists(): raise ValueError('Run is complete; use evaluate or a new output directory')
    else: write_json(out / 'protocol.json', stamp)
    start = time.perf_counter()
    if args.benchmark == 'synthetic':
        from .synthetic import fit
        fit(data, out, args.method, args.seed, cfg)
    elif args.benchmark == 'semrau':
        from .semrau import fit
        fit(data, out, args.method, args.seed, cfg, extra)
    elif args.benchmark == 'multi':
        from .multi import fit
        fit(data, out, args.method, args.seed, cfg, extra, args.device)
    else:
        from .aerosol import fit
        write_json(out / 'fit_config.json', cfg)
        fit(SimpleNamespace(dataset=data, out=out, method='gmi_'+args.method if args.method in ['lin', 'land'] else args.method,
                            seed=args.seed, config=out / 'fit_config.json'))
    artifacts = {str(p.relative_to(out)): digest(p) for p in out.rglob('*')
                 if p.is_file() and p.suffix in ['.pt', '.npz', '.json'] and p.name != 'complete.json'}
    write_json(out / 'complete.json', dict(frozen_artifacts=artifacts,
        wall_seconds=time.perf_counter()-start, smoke=args.smoke,
        stage_c_updates=0, evaluation_status='not yet evaluated'))
    notice(dict(status='fit_complete', out=str(out), smoke=args.smoke))


def evaluate(args):
    out = args.out.resolve()
    stamp = read_json(out / 'protocol.json')
    for key in ['benchmark', 'method', 'seed', 'smoke', 'device']:
        setattr(args, key, stamp[key])
    args.moments = stamp.get('moments', 'both')
    if stamp['study']: args.study = stamp['study']
    data, _, _ = specification(args)
    cfg, extra = stamp['config'], stamp['extra_config']
    for name, expected in stamp['input_sha256'].items():
        if digest(args.data_root / name) != expected: raise ValueError(f'Input changed after fitting: {name}')
    completion = read_json(out / 'complete.json')
    for name, expected in completion['frozen_artifacts'].items():
        if digest(out / name) != expected: raise ValueError(f'Fit artifact changed: {name}')
    if args.benchmark == 'synthetic':
        from .synthetic import evaluate as run
        result = run(data, out, args.method, args.seed, cfg)
    elif args.benchmark == 'semrau':
        from .semrau import evaluate as run
        result = run(data, out, args.method, args.seed, cfg, extra)
    elif args.benchmark == 'multi':
        from .multi import evaluate as run
        result = run(data, out, args.method, args.seed, cfg)
    else:
        from .aerosol import evaluate as run
        policy = dict(ode_steps=extra.get('ode_steps', 400), native_samples=extra.get('native_samples', 32768), population_quadrature_factor=extra.get('population_quadrature_factor', 4))
        record = dict(run=str(out), **{p.name: digest(p) for p in out.glob('*') if p.name in ['velocity.pt', 'path.pt', 'bridge.npz', 'result.json']},
                      **{name: digest(data/name) for name in ['train.npz', 'metadata.json', 'evaluation.npz']})
        write_json(out / 'evaluation_freeze.json', dict(evaluation=policy, runs=[record]))
        run(SimpleNamespace(run=out, freeze=out/'evaluation_freeze.json', ode_steps=policy['ode_steps'],
                            native_samples=policy['native_samples'], population_factor=policy['population_quadrature_factor']))
        result = read_json(out / 'metrics.json')
    notice(dict(status='evaluated', out=str(out), smoke=stamp['smoke'], metrics=result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'train', 'evaluate'])
    parser.add_argument('--benchmark', choices=['synthetic', 'aerosol', 'multi', 'semrau'], default='synthetic')
    parser.add_argument('--method', choices=METHODS, default='lin')
    parser.add_argument('--seed', type=int, choices=SEEDS, default=3)
    parser.add_argument('--moments', choices=['both', 'mean', 'variance'], default='both',
                        help='Synthetic GMI-linear complementary-moment ablation')
    parser.add_argument('--study', choices=STUDIES, default=STUDIES[0])
    parser.add_argument('--data-root', type=Path, default=REPOSITORY/'benchmark_data')
    parser.add_argument('--config-root', type=Path, default=REPOSITORY/'configs/paper')
    parser.add_argument('--out', type=Path, default=Path('outputs/paper/run'))
    parser.add_argument('--destination', type=Path, help='Optional copy of the verified prepared data bundle')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu', help='Continuous Multi bridge sampler device; neural fits use CPU')
    parser.add_argument('--smoke', action='store_true', help='Two-update wiring check; explicitly excluded from paper results')
    args = parser.parse_args()
    if args.command == 'train' and args.moments != 'both' and (args.benchmark != 'synthetic' or args.method != 'lin'):
        parser.error('--moments mean/variance requires --benchmark synthetic --method lin')
    args.data_root = args.data_root.resolve(); args.config_root = args.config_root.resolve()
    torch.set_num_threads(1)
    with threadpool_limits(1):
        if args.command == 'prepare':
            hashes = verify_data(args.data_root)
            if args.destination:
                args.destination.mkdir(parents=True, exist_ok=True)
                for name in [*hashes, 'manifest.json']:
                    target = args.destination/name; target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists() and digest(target) != digest(args.data_root/name):
                        raise ValueError(f'Refusing to overwrite different input: {target}')
                    if target.resolve() != (args.data_root/name).resolve(): shutil.copyfile(args.data_root/name, target)
            notice(dict(status='prepared', files=len(hashes), bytes=sum((args.data_root/n).stat().st_size for n in hashes)))
        elif args.command == 'train': train(args)
        else: evaluate(args)


if __name__ == '__main__':
    main()
