#!/usr/bin/env python3
"""List or execute the frozen main-table and ablation command matrix."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [3, 7, 11, 13, 17]


def jobs(suite, root, seeds):
    commands = []
    if suite in ['main', 'runtime', 'complementary']:
        benchmarks = ['synthetic'] if suite == 'complementary' else ['synthetic', 'aerosol', 'multi', 'semrau']
        for benchmark in benchmarks:
            for study in ['constraint_day3_eval_day4', 'constraint_day4_eval_day3'] if benchmark == 'multi' else ['default']:
                for seed in seeds:
                    for method in ['cfm', 'mfm', 'csb', 'lin', 'land'] if suite != 'complementary' else ['lin']:
                        for moments in ['mean', 'variance', 'both'] if suite == 'complementary' else ['both']:
                            out = root/benchmark/study/method/moments/f'seed_{seed}'
                            flags = ['--benchmark', benchmark, '--method', method, '--seed', str(seed), '--moments', moments, '--out', str(out)]
                            if benchmark == 'multi': flags += ['--study', study]
                            commands.append(([sys.executable, str(ROOT/'scripts/run_paper.py'), 'train', *flags],
                                             [sys.executable, str(ROOT/'scripts/run_paper.py'), 'evaluate', '--out', str(out)], out/'complete.json'))
    elif suite == 'scarce':
        for method, repeat, n in [('cfm',1,4)] + [(m,r,n) for r in range(1,6) for n in [4,16,64,256,1024] for m in ['gmi','csb','mm_empirical','mm_gaussian','gmi_land','mm_mfm']]:
            flags = ['--study', 'scarce', '--out', str(root), '--method', method, '--repeat', str(repeat), '--n', str(n)]
            out = root/('land' if method in ['gmi_land','mm_mfm'] else 'base')/'runs'/method
            if method != 'cfm': out /= f'repeat_{repeat}/n_{n}'
            commands.append(([sys.executable,str(ROOT/'scripts/run_ablations.py'),'train',*flags],
                             [sys.executable,str(ROOT/'scripts/run_ablations.py'),'evaluate',*flags],out/'complete.json'))
    elif suite == 'redundant':
        for seed in [s for s in seeds if s in [3,7,11]]:
            for method in ['cfm','gmi']:
                flags=['--study','redundant','--out',str(root),'--method',method,'--seed',str(seed)]
                out=root/'runs'/('baseline' if method=='cfm' else 'constrained')/f'seed_{seed}'
                commands.append(([sys.executable,str(ROOT/'scripts/run_ablations.py'),'train',*flags],
                                 [sys.executable,str(ROOT/'scripts/run_ablations.py'),'evaluate',*flags],out/'complete.json'))
    else:
        for name in ['oracle']+[f'r_{r:03d}/noise_{s}' for r in [5,10,20,40,60,80,95] for s in SEEDS]:
            flags=['--study','noise','--out',str(root),'--noise-job',name]
            commands.append(([sys.executable,str(ROOT/'scripts/run_ablations.py'),'train',*flags],
                             [sys.executable,str(ROOT/'scripts/run_ablations.py'),'evaluate',*flags],root/'runs'/name/'result.json'))
    return commands


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',choices=['main','runtime','complementary','scarce','redundant','noise'],default='main')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seeds',nargs='+',type=int,choices=SEEDS,default=SEEDS)
    parser.add_argument('--execute',action='store_true',help='Run the listed full-budget experiments sequentially')
    args=parser.parse_args()
    seeds=[3] if args.suite=='runtime' else args.seeds
    matrix=jobs(args.suite,args.out.resolve(),seeds)
    if not args.execute:
        print(json.dumps([dict(train=a,evaluate=b) for a,b,_ in matrix],indent=2));return
    for fit,evaluate,complete in matrix:
        if not complete.exists(): subprocess.run(fit,check=True,cwd=ROOT)
        subprocess.run(evaluate,check=True,cwd=ROOT)


if __name__=='__main__': main()
