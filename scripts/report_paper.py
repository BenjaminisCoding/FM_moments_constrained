#!/usr/bin/env python3
"""Collect frozen main-table W2 scores and timings; exclude smoke checks."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();rows=[];timings=[]
    for path in sorted(args.root.rglob('protocol.json')):
        protocol=json.loads(path.read_text())
        if protocol.get('schema')!=1 or protocol.get('smoke'): continue
        folder=path.parent;benchmark=protocol['benchmark']
        metric=folder/('metrics.json' if benchmark=='aerosol' else 'evaluation.json')
        if not metric.exists() or not (folder/'complete.json').exists(): continue
        value=json.loads(metric.read_text());complete=json.loads((folder/'complete.json').read_text())
        identity={k:protocol[k] for k in ['benchmark','method','seed','study']}
        identity['moments']=protocol.get('moments','both')
        timings.append(dict(**identity,seconds=complete['wall_seconds']))
        scores=({str(r['time']):r['rollout_w2'] for r in value['rows']} if benchmark=='aerosol' else
                {str(h):r['W2'] for h,r in value['scores'].items()} if benchmark=='semrau' else value['W2'])
        rows.extend(dict(**identity,marginal=t,W2=score) for t,score in scores.items())
    groups=defaultdict(list)
    for row in rows:
        groups[(row['benchmark'],row['method'],row['moments'],row['study'],row['marginal'])].append(row)
    summary=[]
    for (benchmark,method,moments,study,t),values in sorted(groups.items(),key=str):
        scores=np.array([r['W2'] for r in values]);ddof=1 if benchmark=='aerosol' else 0
        seeds=sorted(r['seed'] for r in values)
        if len(set(seeds))!=len(seeds): raise ValueError(f'Duplicate run rows for {benchmark}/{method}/{study}/{t}')
        summary.append(dict(benchmark=benchmark,method=method,moments=moments,study=study,marginal=t,
            mean=float(scores.mean()),sd=float(scores.std(ddof=ddof)) if len(scores)>ddof else None,
            ddof=ddof,seeds=seeds,provisional=seeds!=[3,7,11,13,17]))
    # Reciprocal Multi rows are paired within seed before taking mean and SD.
    reciprocal=defaultdict(dict)
    for row in rows:
        if row['benchmark']=='multi' and float(row['marginal']) == (.4 if row['study']=='constraint_day3_eval_day4' else .2):
            reciprocal[(row['method'],row['seed'])][row['study']]=row['W2']
    paired=defaultdict(list)
    for (method,seed),values in reciprocal.items():
        if len(values)==2: paired[method].append(dict(seed=seed,W2=float(np.mean(list(values.values())))))
    output=dict(rows=rows,summary=summary,multi_reciprocal_validation=paired,timings=timings,
                timing_disclosure='Wall-clock execution time on the current machine; excludes raw-data preprocessing.')
    (args.root/'paper_report.json').write_text(json.dumps(output,indent=2)+'\n')
    if rows:
        with (args.root/'paper_scores.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(runs=len(timings),output=str(args.root/'paper_report.json')),indent=2))


if __name__=='__main__': main()
