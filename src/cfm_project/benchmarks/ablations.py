"""Prepare and run the paper's fixed-budget ablations."""
import argparse
from pathlib import Path
import shutil
from types import SimpleNamespace
import torch
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits
from .cli import REPOSITORY, verify_data
from .common import read_json, write_json, digest, notice


def copy_inputs(source, target):
    for p in source.rglob('*'):
        if not p.is_file(): continue
        dest = target / p.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and digest(dest) != digest(p): raise ValueError(f'Input changed: {dest}')
        if not dest.exists(): shutil.copyfile(p, dest)


def prepare_scarce(root, data, configs):
    from .scarce import ScarceStudy
    from .scarce_land import ScarceLandStudy
    from cfm_project.models import VelocityField, PathCorrection
    base = ScarceStudy(root / 'base'); land = ScarceLandStudy(root / 'land', base)
    copy_inputs(data / 'scarce', base.out)
    for study, method in [(base, 'lin'), (land, 'land')]:
        study.out.mkdir(parents=True, exist_ok=True)
        cfg = read_json(configs / f'scarce/{method}.json')
        OmegaConf.save(OmegaConf.create(cfg), study.out / 'stage_a_reference_config.yaml')
    if not (base.out / 'protocol.json').exists():
        torch.manual_seed(3)
        v = VelocityField(2, [128, 128], 'silu'); g = PathCorrection(2, [128, 128], 'silu')
        torch.save(dict(velocity=v.state_dict(), path=g.state_dict()), base.out/'inputs/initial_weights.pt')
        write_json(base.out/'protocol.json', dict(initial_velocity_hash=base.model_hash(v),
            initial_path_hash=base.model_hash(g), sizes=[4,16,64,256,1024], repeats=[1,2,3,4,5],
            stage_a_steps=600, stage_b_steps=9600, initial_seed=3,
            disclosure='Five nested observation samples; fixed endpoint pools and optimizer streams. GMI/CSB receive mean and unbiased variance; MM receives midpoint samples.'))
    if not (land.out/'global_land_reference.pt').exists():
        problem = base.endpoints(); rng = torch.Generator().manual_seed(3)
        refs = {k: torch.randint(len(pool), (512,), generator=rng) for k, pool in [('x0', problem.x0_pool), ('x1', problem.x1_pool)]}
        refs['global_pool'] = torch.cat([problem.x0_pool[refs['x0']], problem.x1_pool[refs['x1']]])
        torch.save(refs, land.out/'global_land_reference.pt')
        torch.save(torch.rand((2,2,256),dtype=torch.float64,generator=torch.Generator().manual_seed(306)),
                   land.out/'segment_reference_uniforms.pt')
    return base, land


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','train','evaluate'])
    parser.add_argument('--study', choices=['scarce','redundant','noise'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=REPOSITORY/'benchmark_data')
    parser.add_argument('--config-root', type=Path, default=REPOSITORY/'configs/paper')
    parser.add_argument('--method', default='gmi', choices=['cfm','gmi','gmi_land','mm_empirical','mm_gaussian','mm_mfm','csb'])
    parser.add_argument('--repeat', type=int, choices=range(1,6), default=1)
    parser.add_argument('--n', type=int, choices=[4,16,64,256,1024], default=4)
    parser.add_argument('--seed', type=int, choices=[3,7,11], default=3)
    parser.add_argument('--noise-job', default='oracle', help='oracle or r_005/noise_3, ..., r_095/noise_17')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(); root = args.out.resolve()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(1):
        verify_data(args.data_root, {'noise':'aerosol', 'scarce':'scarce', 'redundant':'redundant'}[args.study]+'/')
        if args.study == 'scarce':
            base, land = prepare_scarce(root, args.data_root, args.config_root)
            study = land if args.method in ['gmi_land','mm_mfm'] else base
            if args.command == 'train':
                if study is land and not (base.run_dir('cfm',smoke=args.smoke)/'complete.json').exists():
                    base.train('cfm',1,4,args.smoke)
                study.train(args.method,args.repeat,args.n,args.smoke)
            elif args.command == 'evaluate':
                if args.smoke: raise ValueError('Ablation smoke checks are not scientific evaluations')
                run = study.run_dir(args.method,args.repeat,args.n)
                complete = read_json(run/'complete.json')
                if digest(run/'velocity.pt') != complete['velocity_sha256']: raise ValueError('Checkpoint changed')
                freeze = study.out/'frozen_models.json'
                records = read_json(freeze) if freeze.exists() else {}
                records[str(run.relative_to(study.out))] = digest(run/'velocity.pt')
                write_json(freeze,records); study.evaluate(args.method,args.repeat,args.n)
        elif args.study == 'redundant':
            from . import redundant
            if args.method not in ['cfm','gmi']: raise ValueError('Redundant control has CFM and GMI-linear only')
            copy_inputs(args.data_root/'redundant',root)
            write_json(root/'train_config.json',read_json(args.config_root/'redundant.json'))
            settings = SimpleNamespace(stage_a_steps=2 if args.smoke else 300,stage_b_steps=2 if args.smoke else 2400,
                                       batch_size=256,euler_steps=100)
            mode = 'baseline' if args.method == 'cfm' else 'constrained'
            if args.command == 'train': redundant.train(root,settings,args.seed,mode)
            elif args.command == 'evaluate': redundant.evaluate(root,settings,args.seed,mode)
        else:
            from . import observation_noise as noise
            noise.prepare(root,args.data_root/'aerosol',read_json(args.config_root/'aerosol/gmi_land.json'))
            if args.command == 'train': noise.fit(root,args.noise_job,'smoke' if args.smoke else None)
            elif args.command == 'evaluate': noise.evaluate(root,args.noise_job)
        notice(dict(status=args.command,study=args.study,out=str(root),smoke=args.smoke))


if __name__ == '__main__': main()
