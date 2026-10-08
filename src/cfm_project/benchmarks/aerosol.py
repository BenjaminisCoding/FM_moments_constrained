#!/usr/bin/env python3
"""Fit, freeze, evaluate, and report new-domain aggregate experiments separately."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"src"))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache/matplotlib"))

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from cfm_project.aggregate_benchmarks import (
    default_config,fit_path,fit_velocity,moment_bridge,
    observation,paired_endpoints,path_velocity,rollout,sha256,write_json,
)
from .common import w2
from cfm_project.models import PathCorrection
from cfm_project.population_quadrature import load_fit_training
from cfm_project.histogram_transport import histogram_w2,histogram_quadrature
from cfm_project.aggregate_velocity import make_velocity
from cfm_project.quadratic_moment_sb import QuadraticMomentBridge
from cfm_project.scalar_moment_sb import ScalarMomentBridge
from cfm_project.vector_moment_sb import VectorMomentBridge


def notice(value):
    print(json.dumps(value,allow_nan=False),flush=True)


def wait_for_fit_owner(out, stamp):
    """Join an identical live fit; never overwrite an incomplete or different one."""
    record=out/'run.json';deadline=time.monotonic()+5.
    while True:
        try:
            owner=json.loads(record.read_text());break
        except json.JSONDecodeError:
            if time.monotonic()>=deadline:raise RuntimeError('Incomplete ownership record')
            time.sleep(.05)
    if not all(owner.get(k)==v for k,v in stamp.items()):
        raise RuntimeError('Existing live run has different method/configuration/data')
    if owner.get('core_sha256')!=sha256(Path(__import__('cfm_project.aggregate_benchmarks', fromlist=['__file__']).__file__)):
        raise RuntimeError('Existing live run uses a different training core')
    pid=owner.get('pid')
    if not isinstance(pid,int) or pid<=0 or pid==os.getpid():
        raise RuntimeError('Invalid fit owner PID')
    notice(dict(event='waiting_for_identical_fit',out=str(out),owner_pid=pid))
    while True:
        result=out/'result.json'
        if result.exists():
            try:previous=json.loads(result.read_text())
            except json.JSONDecodeError:previous=None
            if previous is not None:
                if not all(previous.get(k)==v for k,v in stamp.items()):
                    raise RuntimeError('Completed owner result has different method/configuration/data')
                notice(dict(event='already_complete',out=str(out)));return
        try:os.kill(pid,0)
        except ProcessLookupError:
            # The owner can publish its result between our read and PID check.
            if result.exists():
                previous=json.loads(result.read_text())
                if all(previous.get(k)==v for k,v in stamp.items()):
                    notice(dict(event='already_complete',out=str(out)));return
            raise RuntimeError('Incomplete run has no live owner; inspect before resuming') from None
        time.sleep(1.)


def fit(args):
    cfg=default_config()
    if args.config:cfg.update(json.loads(Path(args.config).read_text()))
    out=Path(args.out).resolve(); out.mkdir(parents=True,exist_ok=True)
    data=load_fit_training(args.dataset,cfg)
    stamp=dict(method=args.method,seed=args.seed,config=cfg,
               dataset=str(data.source.resolve()),data_sha256=sha256(data.source/"train.npz"),
               metadata_sha256=sha256(data.source/"metadata.json"))
    if (out/"result.json").exists():
        try:previous=json.loads((out/"result.json").read_text())
        except json.JSONDecodeError:
            wait_for_fit_owner(out,stamp);return
        if all(previous.get(k)==v for k,v in stamp.items()):
            notice(dict(event="already_complete",out=str(out)));return
        raise RuntimeError("Existing results have a different method/configuration/data")
    owner=dict(**stamp,pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),
               python=sys.executable,torch_version=torch.__version__,
               core_sha256=sha256(Path(__import__('cfm_project.aggregate_benchmarks', fromlist=['__file__']).__file__)))
    try:
        with (out/'run.json').open('x') as stream:
            json.dump(owner,stream,indent=2,allow_nan=False);stream.write('\n')
    except FileExistsError:
        wait_for_fit_owner(out,stamp);return
    torch.set_num_threads(cfg["threads"])
    started=time.perf_counter()
    with threadpool_limits(limits=cfg["threads"]):
        if args.method=="csb":
            bridge=moment_bridge(data,cfg,out,notice)
            path=None
            stage_a=dict(stage_a_seconds=bridge.diagnostics["wall_seconds"],
                         closure_evaluations=0,history=[],native_bridge=bridge.diagnostics)
            coupling=dict(type="moment-dependent dense entropic Brownian coupling",sigma=cfg["sigma"])
        else:
            path,coupling,stage_a=fit_path(data,args.method,args.seed,cfg,out,notice)
            bridge=None
        stage_b=fit_velocity(data,args.method,args.seed,cfg,path,bridge,out,notice)
    result=dict(**stamp,stage_a=stage_a,stage_b=stage_b,coupling=coupling,
                python=sys.executable,torch_version=torch.__version__,
                wall_seconds=time.perf_counter()-started,information=data.metadata,
                finished_utc=datetime.now(timezone.utc).isoformat(),
                velocity_sha256=sha256(out/"velocity.pt"),
                path_sha256=sha256(out/"path.pt") if (out/"path.pt").exists() else None,
                bridge_sha256=sha256(out/"bridge.npz") if (out/"bridge.npz").exists() else None)
    write_json(out/"result.json",result)
    notice(dict(event="fit_complete",out=str(out),wall_seconds=result["wall_seconds"]))


def load_bridge(path):
    with np.load(path) as a:
        if 'kind' in a and str(a['kind'])=='vector_gauss_hermite':
            return VectorMomentBridge(a['x0'],a['x1'],a['coupling'],float(a['sigma']),float(a['tau']),
                                       a['target'],a['multiplier'],a['noise_nodes'],a['conditional_cdf'],{})
        if 'kind' in a and str(a['kind'])=='scalar_gauss_hermite':
            return ScalarMomentBridge(a['x0'],a['x1'],a['coupling'],float(a['sigma']),float(a['tau']),
                                       float(a['target']),float(a['multiplier']),a['noise_nodes'],a['conditional_cdf'],{})
        return QuadraticMomentBridge(a["x0"],a["x1"],a["coupling"],float(a["sigma"]),
                                     float(a["tau"]),a["mean"],a["covariance"],
                                     a["conditional_covariance"],a["linear_multiplier"],
                                     a["quadratic_multiplier"],{})


def native_bridge_samples(bridge,times,n=4096,seed=129431):
    rng=np.random.default_rng(seed)
    x,z,y=bridge.sample(n,rng)
    states=[]
    for t in times:
        if t==0:states.append(bridge.x0);continue
        if t==1:states.append(bridge.x1);continue
        if np.isclose(t,bridge.tau):states.append(z);continue
        if t<bridge.tau:s,r,a,b=0.,bridge.tau,x,z
        else:s,r,a,b=bridge.tau,1.,z,y
        mean=((r-t)*a+(t-s)*b)/(r-s)
        states.append(mean+bridge.sigma*np.sqrt((t-s)*(r-t)/(r-s))*rng.standard_normal(mean.shape))
    return states


def evaluate(args):
    out=Path(args.run).resolve()
    result=json.loads((out/"result.json").read_text())
    fit_data=load_fit_training(result["dataset"],result['config'])
    data=load_fit_training(result['dataset'],dict(population_quadrature_factor=args.population_factor)) if fit_data.weights is not None else fit_data
    if data.metadata["split"]=="test":
        if not args.freeze:raise RuntimeError("Test evaluation requires a freeze file recording this run and its evaluation settings")
        frozen=json.loads(Path(args.freeze).read_text())
        policy=frozen['evaluation']
        if args.ode_steps!=policy['ode_steps'] or args.native_samples!=policy['native_samples']:
            raise RuntimeError('Evaluation settings must match the frozen protocol')
        if args.population_factor!=policy['population_quadrature_factor']:
            raise RuntimeError('Evaluation population quadrature must match the frozen protocol')
        records={r["run"]:r for r in frozen["runs"]}
        if str(out) not in records:raise RuntimeError("This run is not listed in the freeze file")
        for name in ["velocity.pt","path.pt","bridge.npz"]:
            if (out/name).exists() and records[str(out)][name]!=sha256(out/name):
                raise RuntimeError("Checkpoint changed after freezing")
        record=records[str(out)]
        for name in ['train.npz','metadata.json','evaluation.npz']:
            if record[name]!=sha256(data.source/name):
                raise RuntimeError('Dataset changed after freezing: '+name)
        if record['result.json']!=sha256(out/'result.json'):
            raise RuntimeError('Fit record changed after freezing')
    if result["data_sha256"]!=sha256(data.source/"train.npz"):
        raise RuntimeError("Training data changed after fitting")
    if result['information']!=data.metadata:
        raise RuntimeError('Observation metadata changed after fitting')
    # This is the first function permitted to open the evaluation archive.
    with np.load(data.source/"evaluation.npz",allow_pickle=False) as a:
        times=a["times"]; truths=a["samples"]
        truth_weights=a['weights'] if 'weights' in a else None
        histogram_probabilities=a['bin_probabilities'] if 'bin_probabilities' in a else None
        histogram_edges=a['log_bin_edges'] if 'log_bin_edges' in a else None
    cfg=result["config"];torch.set_num_threads(cfg["threads"])
    model=make_velocity(data.x0.shape[1],cfg)
    model.load_state_dict(torch.load(out/"velocity.pt",weights_only=False,map_location="cpu")["state_dict"])
    model.eval()
    path=None
    if (out/"path.pt").exists():
        path=PathCorrection(data.x0.shape[1],cfg["hidden"],cfg["activation"])
        path.load_state_dict(torch.load(out/"path.pt",weights_only=False,map_location="cpu")["state_dict"])
        for p in path.parameters():p.requires_grad_(False)
    with threadpool_limits(limits=cfg["threads"]):
        generated=[x.numpy() for x in rollout(model,data.x0,times,args.ode_steps)]
        x0,x1,_=paired_endpoints(data)
        if result["method"]=="csb":
            teacher=native_bridge_samples(load_bridge(out/"bridge.npz"),times,n=args.native_samples,
                                          seed=129431+result["seed"])
        else:
            teacher=[path_velocity(path,torch.full((len(x0),1),float(t)),x0,x1)[0].detach().numpy() for t in times]
        rows=[]
        source_weights=data.weights.numpy() if data.weights is not None else None
        fit_weights=fit_data.weights.numpy() if fit_data.weights is not None else None
        for index,(t,truth,pred,direct) in enumerate(zip(times,truths,generated,teacher)):
            direct_weights=source_weights if result['method']!='csb' else fit_weights if t in (0.,1.) else None
            def distance_to_truth(values,weights,physical=False):
                if histogram_probabilities is not None:
                    return histogram_w2(values,weights,histogram_probabilities[index],histogram_edges,
                                        float(data.observation_config['physical_center'][0]),
                                        float(data.observation_config['physical_scale'][0]),physical)
                return w2(values,truth,weights,truth_weights)
            rows.append(dict(time=float(t),role="endpoint" if t in (0.,1.) else
                             "aggregate_observed" if np.isclose(t,data.tau) else "unobserved",
                             rollout_w2=distance_to_truth(pred,source_weights),
                             interpolant_w2=distance_to_truth(direct,direct_weights),
                             rollout_to_interpolant_w2=w2(pred,direct,source_weights,direct_weights),
                             rollout_coordinate_w2=[distance_to_truth(pred,source_weights)] if pred.shape[1]==1 else
                                 [w2(pred[:,i:i+1],truth[:,i:i+1],source_weights,truth_weights) for i in range(pred.shape[1])]))
            if data.observation_config['type'] in ['sigmoid_threshold','tabulated_optical']:
                center=float(data.observation_config['physical_center'][0])
                scale=float(data.observation_config['physical_scale'][0])
                def diameter(x):return np.exp(np.asarray(x,dtype=np.float64)*scale+center)
                rows[-1]['rollout_diameter_w2_nm']=distance_to_truth(pred,source_weights,True)
                rows[-1]['interpolant_diameter_w2_nm']=distance_to_truth(direct,direct_weights,True)
        at=int(np.argmin(np.abs(times-data.tau)))
        def residual(x,weights):
            values=observation(torch.tensor(x,dtype=torch.float32),data.observation_config)
            average=values.mean(0) if weights is None else (values*torch.tensor(weights,dtype=torch.float32)[:,None]).sum(0)
            return (average-data.target).tolist()
        truth_midpoint,truth_midpoint_weights=truths[at],truth_weights
        if histogram_probabilities is not None:
            truth_midpoint,truth_midpoint_weights=histogram_quadrature(histogram_probabilities[at],histogram_edges,
                float(data.observation_config['physical_center'][0]),float(data.observation_config['physical_scale'][0]))
        metrics=dict(method=result["method"],seed=result["seed"],dataset=data.metadata["name"],
                     split=data.metadata["split"],rows=rows,
                     mean_unobserved_w2=float(np.mean([r["rollout_w2"] for r in rows if r["role"]=="unobserved"])),
                     mean_unobserved_interpolant_w2=float(np.mean([r["interpolant_w2"] for r in rows if r["role"]=="unobserved"])),
                     midpoint_rollout_residual=residual(generated[at],source_weights),
                     midpoint_interpolant_residual=residual(teacher[at],None if result['method']=='csb' else source_weights),
                     midpoint_truth_residual=residual(truth_midpoint,truth_midpoint_weights),
                     population_quadrature_factor=args.population_factor if data.weights is not None else None,
                     truth_representation='piecewise uniform within measured log-diameter bins; exact quantile-cost integration' if histogram_probabilities is not None else 'empirical equal-mass samples',
                     ode=dict(method="RK4",steps_per_unit=args.ode_steps),
                     native_bridge_samples=args.native_samples if result['method']=='csb' else None,
                     stage_a_seconds=result["stage_a"]["stage_a_seconds"],
                     stage_b_seconds=result["stage_b"]["stage_b_seconds"],
                     evaluation_sha256=sha256(data.source/"evaluation.npz"))
    write_json(out/"metrics.json",metrics)
    np.savez_compressed(out/"evaluated_samples.npz",times=times,rollout=np.stack(generated),
                        **{f"teacher_{i}":v for i,v in enumerate(teacher)})
    notice(dict(event="evaluated",run=str(out),**metrics))
