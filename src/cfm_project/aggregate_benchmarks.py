"""Common, information-separated training for new aggregate-observation datasets.

Uses the paper's endpoint-preserving interpolant and the existing LAND metric.
The nested AL optimizer changes numerical optimization, not the constrained
path-selection objective. Training deliberately has no evaluation-data argument.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import expit
import torch
from torch.func import jvp

from cfm_project.models import PathCorrection, VelocityField
from cfm_project.paths import corrected_path
from cfm_project.mfm_core import land_geopath_loss
from cfm_project.quadratic_moment_sb import solve_quadratic_moment_bridge
from cfm_project.scalar_moment_sb import solve_scalar_moment_bridge
from cfm_project.vector_moment_sb import solve_vector_moment_bridge


METHODS = ("cfm", "mfm", "csb", "gmi_lin", "gmi_land")
SEEDS = (3, 7, 11, 13, 17)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class TrainingData:
    x0: torch.Tensor
    x1: torch.Tensor
    target: torch.Tensor
    tau: float
    metadata: dict
    source: Path
    observation_config: dict
    weights: torch.Tensor | None = None
    land_references: torch.Tensor | None = None


def load_training(folder: str | Path) -> TrainingData:
    """Only explicitly permitted files are opened; never discover other NPZs."""
    folder = Path(folder)
    metadata = json.loads((folder / "metadata.json").read_text())
    with np.load(folder / "train.npz", allow_pickle=False) as archive:
        x0 = torch.tensor(archive["x0"], dtype=torch.float32)
        x1 = torch.tensor(archive["x1"], dtype=torch.float32)
        target = torch.tensor(archive["target"], dtype=torch.float32)
        tau = float(archive["tau"])
        weights = torch.tensor(archive['weights'],dtype=torch.float32) if 'weights' in archive else None
        references = torch.tensor(archive['land_references'],dtype=torch.float32) if 'land_references' in archive else None
    if x0.ndim != 2 or x1.shape != x0.shape or len(x0) < 2:
        raise ValueError("This study requires equal-size endpoint pools in R^d")
    if not all(torch.isfinite(x).all() for x in (x0, x1, target)):
        raise ValueError("Training inputs must be finite")
    if not 0 < tau < 1:
        raise ValueError("Constraint time must be strictly interior")
    if weights is not None:
        if weights.shape!=(len(x0),) or not torch.isfinite(weights).all() or (weights<=0).any():
            raise ValueError('Invalid endpoint quadrature weights')
        if not torch.isclose(weights.sum(),torch.tensor(1.)):
            raise ValueError('Endpoint quadrature must have unit mass')
    return TrainingData(x0, x1, target, tau, metadata, folder,
                        metadata.get("observation_config", {"type": metadata["feature_type"]}),weights,references)


def observation(x: torch.Tensor, config: dict) -> torch.Tensor:
    kind = config["type"]
    if kind == "raw_moments":
        i, j = torch.triu_indices(x.shape[1], x.shape[1], device=x.device)
        return torch.cat((x, x[:, i]*x[:, j]), dim=1)
    if kind == "sigmoid_threshold":
        center = torch.as_tensor(config["physical_center"], dtype=x.dtype, device=x.device)
        scale = torch.as_tensor(config["physical_scale"], dtype=x.dtype, device=x.device)
        log_d = x[:, :1]*scale + center
        threshold = torch.as_tensor(config["threshold_log_d"], dtype=x.dtype, device=x.device)
        width = float(config["width_log_d"])
        return torch.sigmoid((log_d-threshold)/width)
    if kind == 'tabulated_optical':
        log_d=x[:,0]*float(config['physical_scale'][0])+float(config['physical_center'][0])
        grid=torch.as_tensor(config['log_diameter_grid'],dtype=x.dtype,device=x.device)
        values=torch.as_tensor(config['response_um2'],dtype=x.dtype,device=x.device)
        # Piecewise-linear calibrated physical response. The end values define
        # an explicit bounded continuation, shared with the SB solver.
        position=log_d.clamp(grid[0],grid[-1])
        i=torch.searchsorted(grid,position.contiguous(),right=True).clamp(1,len(grid)-1)
        fraction=(position-grid[i-1])/(grid[i]-grid[i-1])
        return ((values[i-1]+fraction*(values[i]-values[i-1]))/config['feature_scale_um2'])[:,None]
    raise ValueError(f"Unsupported observation {kind}")


def paired_endpoints(data):
    if data.weights is not None:
        if data.metadata.get('endpoint_representation')!='shared_quantile_quadrature' or data.x0.shape[1]!=1:
            raise ValueError('Weighted endpoint support must declare one-dimensional common-quantile OT')
        if (torch.diff(data.x0[:,0])<0).any() or (torch.diff(data.x1[:,0])<0).any():
            raise ValueError('Common-quantile endpoints must be sorted')
        return data.x0,data.x1,dict(type='global 1D monotone OT with endpoint-adaptive probability quadrature',
                                   support=len(data.x0),mean_squared_cost=float((data.weights*(data.x1-data.x0).square().sum(1)).sum()))
    cost = torch.cdist(data.x0, data.x1).square().numpy()
    i, j = linear_sum_assignment(cost)
    return data.x0[i], data.x1[j], dict(type="global balanced OT", support=len(i),
                                      mean_squared_cost=float(cost[i,j].mean()))


def path_velocity(model, t, x0, x1):
    if model is None:
        return (1-t)*x0+t*x1, x1-x0
    return jvp(lambda tt: corrected_path(tt, x0, x1, model), (t,), (torch.ones_like(t),))


def parameter_gradient(loss, model, retain_graph=False):
    grads = torch.autograd.grad(loss, tuple(model.parameters()), retain_graph=retain_graph,
                                allow_unused=True)
    return torch.cat([g.reshape(-1) for g in grads if g is not None])


def quadrature_pairs(x0, x1, n_times, pair_weights=None):
    nodes, weights = np.polynomial.legendre.leggauss(n_times)
    t = torch.tensor((nodes+1)/2, dtype=x0.dtype).repeat_interleave(len(x0))[:,None]
    mass=torch.full((len(x0),),1/len(x0),dtype=x0.dtype) if pair_weights is None else pair_weights
    w = torch.tensor(weights/2, dtype=x0.dtype).repeat_interleave(len(x0))*mass.repeat(n_times)
    return t, x0.repeat(n_times,1), x1.repeat(n_times,1), w


def base_and_residual(model, method, data, pairs, cfg):
    x0, x1 = pairs
    t, q0, q1, weights = quadrature_pairs(x0, x1, cfg["quadrature_times"],data.weights)
    xt, velocity = path_velocity(model, t, q0, q1)
    if method in ("mfm", "gmi_land"):
        # Existing first-party LAND definition. Equal-weight quadrature is
        # applied by scaling each velocity by sqrt(qweight * batch_size).
        refs = torch.cat((data.x0, data.x1)) if data.land_references is None else data.land_references
        if len(refs) > cfg["land_max_references"]:
            ids = torch.linspace(0, len(refs)-1, cfg["land_max_references"]).long()
            refs = refs[ids]
        base = land_geopath_loss(xt, velocity * (weights*len(weights)).sqrt()[:,None],
                                 refs, cfg["land_gamma"], cfg["land_rho"])
    else:
        base = cfg["alpha"] * (weights * ((velocity-(q1-q0)).square().sum(1))).sum()
        if cfg["beta"] > 0:
            acceleration = jvp(lambda tt: path_velocity(model, tt, q0, q1)[1],
                               (t,), (torch.ones_like(t),))[1]
            base = base + cfg["beta"]*(weights*acceleration.square().sum(1)).sum()
    middle = corrected_path(torch.full((len(x0),1), data.tau), x0, x1, model)
    values=observation(middle,data.observation_config)
    expectation=values.mean(0) if data.weights is None else (values*data.weights[:,None]).sum(0)
    residual = expectation-data.target
    return base, residual


def default_config():
    return dict(hidden=[64,64], velocity_hidden=None, activation="silu", alpha=1., beta=0.,
                quadrature_times=10, land_gamma=.7, land_rho=.01,
                land_max_references=512, warmup_steps=100, warmup_lr=1e-3,
                outer_steps=35, inner_steps=20, primal_lr=.8, lbfgs_history=20,
                rho=None, rho_factor=1., rho_max=1e5, multiplier_clip=1e5,
                residual_tolerance=.003, stationarity_tolerance=.05,
                stage_b_steps=5000, stage_b_batch=256, stage_b_lr=1e-3,
                stage_b_log_every=250, sigma=.5, threads=1,
                quadrature_sampling_mix=.5,csb_quadrature=64,csb_max_quadrature=256)


def initialize_path(data,method,seed,cfg):
    """Exact training initialization and base-only prefix for gradient calibration."""
    x0,x1,_=paired_endpoints(data)
    torch.manual_seed(seed)
    model = PathCorrection(data.x0.shape[1], cfg["hidden"], cfg["activation"])
    pairs = (x0, x1)
    initial_base, initial_residual = base_and_residual(model, method, data, pairs, cfg)
    g0 = parameter_gradient(initial_base, model, retain_graph=True)
    gc = parameter_gradient(.5*initial_residual.square().sum(), model)
    initial_balance = float(g0.norm()/gc.norm().clamp_min(1e-12))
    warmup = cfg["warmup_steps"] if method in ("mfm", "gmi_land") else 0
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["warmup_lr"])
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        base, _ = base_and_residual(model, method, data, pairs, cfg)
        base.backward(); optimizer.step()
    base, residual = base_and_residual(model, method, data, pairs, cfg)
    g0 = parameter_gradient(base, model, retain_graph=True)
    gc = parameter_gradient(.5*residual.square().sum(), model)
    balance = float(g0.norm()/gc.norm().clamp_min(1e-12))
    return model,dict(seed=seed,initial_gradient_balance=initial_balance,
                       post_warmup_gradient_balance=balance,warmup_steps=warmup)


def fit_path(data, method, seed, cfg, output, notify=lambda x:None):
    x0, x1, coupling = paired_endpoints(data)
    if method == "cfm":
        return None, coupling, dict(stage_a_seconds=0., closure_evaluations=0, history=[])
    started=time.perf_counter()
    model,calibration=initialize_path(data,method,seed,cfg)
    pairs=(x0,x1)
    balance=calibration['post_warmup_gradient_balance']
    warmup=calibration['warmup_steps']
    rho = cfg["rho"]
    if rho is None:
        # In nested AL, a near-stationary primal solve precedes each dual
        # update. Set its penalty from the instantaneous gradient balance.
        rho = max(1e-3, .7*balance)
    rho = min(cfg["rho_max"], float(rho)*cfg["rho_factor"])
    calibration.update(chosen_rho=rho,convention="eta=1; nested primal solves then dual ascent",
                       median_initialization=cfg.get('gradient_calibration'))
    write_json(output/"calibration.json", calibration)
    multiplier = torch.zeros_like(data.target)
    history = []
    evaluations = 0
    constrained = method.startswith("gmi_")
    previous_residual = float("inf")
    stable = 0
    for outer in range(cfg["outer_steps"]):
        # A changed multiplier changes the objective. Reset L-BFGS secant
        # history while warm-starting the same path parameters.
        optimizer = torch.optim.LBFGS(model.parameters(), lr=cfg["primal_lr"],
                                      max_iter=cfg["inner_steps"],
                                      history_size=cfg["lbfgs_history"],
                                      tolerance_grad=1e-7, tolerance_change=1e-10,
                                      line_search_fn="strong_wolfe")
        def closure():
            nonlocal evaluations
            optimizer.zero_grad(set_to_none=True)
            base, r = base_and_residual(model, method, data, pairs, cfg)
            loss = base + (multiplier@r+.5*rho*r.square().sum() if constrained else 0.)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite path objective")
            loss.backward(); evaluations += 1
            return loss
        optimizer.step(closure)
        base, r = base_and_residual(model, method, data, pairs, cfg)
        gb = parameter_gradient(base, model, retain_graph=True)
        if constrained:
            gconstraint = parameter_gradient(multiplier@r+.5*rho*r.square().sum(), model)
        else:
            gconstraint = torch.zeros_like(gb)
        raw = r.detach().numpy().astype(float)
        raw_norm = float(np.linalg.norm(raw))
        row = dict(outer=outer+1, closure_evaluations=evaluations, base=float(base.detach()),
                   residual=raw.tolist(), residual_l2=raw_norm,
                   residual_linf=float(np.max(np.abs(raw))), rho=rho,
                   multiplier_norm=float(multiplier.norm()),
                   multiplier_clip_fraction=float((multiplier.abs()>=cfg["multiplier_clip"]).float().mean()),
                   base_gradient_norm=float(gb.norm()), constraint_gradient_norm=float(gconstraint.norm()),
                   total_gradient_norm=float((gb+gconstraint).norm()),
                   gradient_ratio=float(gconstraint.norm()/gb.norm().clamp_min(1e-12)),
                   gradient_cosine=float((gb@gconstraint)/(gb.norm()*gconstraint.norm()).clamp_min(1e-12)),
                   elapsed_seconds=time.perf_counter()-started)
        history.append(row)
        with (output/"stage_a.jsonl").open("a") as stream:stream.write(json.dumps(row)+"\n")
        if outer % 5 == 0 or outer == cfg["outer_steps"]-1:
            notify(dict(event="stage_a",method=method,seed=seed,**row))
        # KKT gradient is measured before the dual update. Repeated low
        # residuals and a small stationarity error form the stopping criterion.
        adequate = (row["residual_linf"] <= cfg["residual_tolerance"]
                    and row["total_gradient_norm"] <= cfg["stationarity_tolerance"])
        if not constrained:
            adequate = row["total_gradient_norm"] <= cfg["stationarity_tolerance"]
        stable = stable+1 if adequate else 0
        if cfg.get("stage_a_early_stopping", True) and stable >= 4 and outer >= 9:
            break
        if constrained:
            multiplier = (multiplier+rho*r.detach()).clamp(-cfg["multiplier_clip"],cfg["multiplier_clip"])
            if (outer+1) % 5 == 0:
                if raw_norm > .7*previous_residual and raw_norm > cfg["residual_tolerance"]:
                    rho = min(rho*2., cfg["rho_max"])
                previous_residual = raw_norm
    for p in model.parameters():p.requires_grad_(False)
    summary = dict(stage_a_seconds=time.perf_counter()-started, warmup_steps=warmup,
                   outer_steps=len(history), closure_evaluations=evaluations,
                   calibration=calibration, final=history[-1], history=history,
                   optimizer="nested AL; full global-OT support; Gauss-Legendre time quadrature",
                   objective="LAND kinetic action" if method in ("mfm","gmi_land")
                             else "alpha velocity-deviation energy + beta acceleration energy",
                   stable_stopping=bool(cfg.get("stage_a_early_stopping", True) and stable>=4),
                   stopping=("residual/KKT stability" if cfg.get("stage_a_early_stopping", True)
                             else "predeclared fixed outer-step budget; final iterate"))
    torch.save(dict(state_dict=model.state_dict(),config=cfg,method=method,seed=seed,
                    data_sha256=sha256(data.source/"train.npz")),output/"path.pt")
    write_json(output/"stage_a_summary.json",summary)
    return model,coupling,summary


def moment_bridge(data, cfg, output, notify=lambda x:None):
    if data.observation_config["type"] != "raw_moments":
        feature_config=data.observation_config
        if feature_config['type'] not in ['tabulated_optical','sigmoid_threshold']:
            raise ValueError('Non-quadratic CSB requires a supported observation')
        def feature(x):
            physical=x[:,0]*float(feature_config['physical_scale'][0])+float(feature_config['physical_center'][0])
            if feature_config['type']=='sigmoid_threshold':
                threshold=np.atleast_1d(feature_config['threshold_log_d'])
                result=expit((physical[:,None]-threshold[None,:])/float(feature_config['width_log_d']))
                return result[:,0] if len(threshold)==1 else result
            return np.interp(physical,feature_config['log_diameter_grid'],feature_config['response_um2'])/feature_config['feature_scale_um2']
        weights=data.weights.numpy() if data.weights is not None else None
        solver=solve_scalar_moment_bridge if data.target.numel()==1 else solve_vector_moment_bridge
        target=float(data.target[0]) if data.target.numel()==1 else data.target.numpy()
        solved=solver(data.x0.numpy(),data.x1.numpy(),feature,target,
                                          sigma=cfg['sigma'],tau=data.tau,weights0=weights,weights1=weights,
                                          quadrature=cfg['csb_quadrature'],max_quadrature=cfg['csb_max_quadrature'],
                                          progress=notify)
        np.savez_compressed(output/'bridge.npz',kind='scalar_gauss_hermite' if data.target.numel()==1 else 'vector_gauss_hermite',x0=solved.x0,x1=solved.x1,
                            coupling=solved.coupling,sigma=solved.sigma,tau=solved.tau,
                            target=solved.target,multiplier=solved.multiplier,noise_nodes=solved.noise_nodes,
                            conditional_cdf=solved.conditional_cdf)
        write_json(output/'bridge_diagnostics.json',solved.diagnostics)
        return solved
    d = data.x0.shape[1]
    mean = data.target[:d].numpy().astype(float)
    second = np.zeros((d,d));i,j=np.triu_indices(d)
    second[i,j]=data.target[d:].numpy();second[j,i]=second[i,j]
    covariance = second-np.outer(mean,mean)
    solved=solve_quadratic_moment_bridge(data.x0.numpy(),data.x1.numpy(),mean,covariance,
                                        sigma=cfg["sigma"],tau=data.tau,
                                        progress=lambda row:notify(dict(event="csb_solve",**row)))
    np.savez_compressed(output/"bridge.npz",x0=solved.x0,x1=solved.x1,coupling=solved.coupling,
                        conditional_covariance=solved.conditional_covariance,
                        linear_multiplier=solved.linear_multiplier,
                        quadratic_multiplier=solved.quadratic_multiplier,
                        mean=mean,covariance=covariance,sigma=solved.sigma,tau=solved.tau)
    write_json(output/"bridge_diagnostics.json",solved.diagnostics)
    return solved


def bridge_batch(bridge, n, generator, rng):
    x,z,y=bridge.sample(n,rng)
    x,z,y=map(torch.from_numpy,(x,z,y))
    t=torch.rand((n,1),generator=generator).clamp(1e-5,1-1e-5)
    # Float32 RNG can hit the interior anchor exactly (positive discrete
    # probability), where the conditional velocity is singular. Avoid all
    # anchors by 1e-5; the native marginal sampler has no such truncation.
    t=torch.where((t-bridge.tau).abs()<1e-5,torch.full_like(t,bridge.tau+1e-5),t)
    before=t<bridge.tau
    s=torch.where(before,0.,bridge.tau)
    r=torch.where(before,bridge.tau,1.)
    a,b=torch.where(before,x,z),torch.where(before,z,y)
    duration=r-s
    mean=((r-t)*a+(t-s)*b)/duration
    variance=bridge.sigma**2*(t-s)*(r-t)/duration
    noise=torch.randn(mean.shape,generator=generator)
    state=mean+variance.sqrt()*noise
    velocity=(b-a)/duration+(s+r-2*t)*bridge.sigma/(2*(duration*(t-s)*(r-t)).sqrt())*noise
    # Weight only depends on t. It removes anchor velocity singularities and
    # preserves the population conditional regression minimizer.
    weight=(t-s)*(r-t)
    weight=weight / (bridge.tau**3+(1-bridge.tau)**3)*6
    return t,state,velocity,weight


def fit_velocity(data, method, seed, cfg, path, bridge, output, notify=lambda x:None):
    started=time.perf_counter()
    # Velocity initialization is paired across methods, independently of Stage A.
    torch.manual_seed(seed+100003)
    model=VelocityField(data.x0.shape[1],cfg.get("velocity_hidden") or cfg["hidden"],cfg["activation"])
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg["stage_b_lr"])
    generator=torch.Generator().manual_seed(seed+200003)
    rng=np.random.default_rng(seed+200003)
    x0,x1,_=paired_endpoints(data)
    def batch(n,gen,random):
        if bridge is not None:return bridge_batch(bridge,n,gen,random)
        if data.weights is None:
            ids=torch.randint(len(x0),(n,),generator=gen)
            importance=torch.ones((n,1))
        else:
            mix=cfg['quadrature_sampling_mix']
            probability=(1-mix)*data.weights+mix/len(x0)
            ids=torch.multinomial(probability,n,replacement=True,generator=gen)
            importance=(data.weights[ids]/probability[ids])[:,None]
        t=torch.rand((n,1),generator=gen)
        state,velocity=path_velocity(path,t,x0[ids],x1[ids])
        return t,state.detach(),velocity.detach(),importance
    # Disjoint, held-fixed draws of the teacher, with no evaluation marginal.
    diagnostic=batch(2048,torch.Generator().manual_seed(190733),np.random.default_rng(190733))
    trace=[]
    for step in range(cfg["stage_b_steps"]):
        t,state,target,weight=batch(cfg["stage_b_batch"],generator,rng)
        optimizer.zero_grad(set_to_none=True)
        loss=(weight*(model(t,state)-target).square().sum(1,keepdim=True)).mean()
        if not torch.isfinite(loss):raise FloatingPointError("Non-finite velocity loss")
        loss.backward();optimizer.step()
        if step+1==int(.7*cfg["stage_b_steps"]):
            for group in optimizer.param_groups:group["lr"]=cfg["stage_b_lr"]*.3
        if (step+1)%cfg["stage_b_log_every"]==0 or step==0 or step+1==cfg["stage_b_steps"]:
            with torch.no_grad():
                tt,xx,uu,ww=diagnostic
                diagnostic_loss=float((ww*(model(tt,xx)-uu).square().sum(1,keepdim=True)).mean())
            row=dict(step=step+1,training_loss=float(loss.detach()),
                     fixed_teacher_loss=diagnostic_loss,elapsed_seconds=time.perf_counter()-started)
            trace.append(row)
            with (output/"stage_b.jsonl").open("a") as stream:stream.write(json.dumps(row)+"\n")
            if (step+1)%(cfg["stage_b_log_every"]*4)==0:notify(dict(event="stage_b",method=method,seed=seed,**row))
    torch.save(dict(state_dict=model.state_dict(),config=cfg,method=method,seed=seed,
                    dimension=data.x0.shape[1],data_sha256=sha256(data.source/"train.npz"),
                    executed_steps=cfg["stage_b_steps"],final_model=True),output/"velocity.pt")
    summary=dict(stage_b_seconds=time.perf_counter()-started,steps=cfg["stage_b_steps"],
                 final_fixed_teacher_loss=trace[-1]["fixed_teacher_loss"],history=trace,
                 input_noise=0.,stopping="predeclared fixed budget, shared across methods")
    write_json(output/"stage_b_summary.json",summary)
    return summary


@torch.no_grad()
def rollout(model,x0,times,steps_per_unit=200):
    states=[x0.clone()]; x=x0.clone(); current=0.
    for target in times[1:]:
        n=max(1,int(np.ceil((float(target)-current)*steps_per_unit)))
        h=(float(target)-current)/n
        for _ in range(n):
            t=torch.full((len(x),1),current)
            k1=model(t,x);k2=model(t+h/2,x+h*k1/2)
            k3=model(t+h/2,x+h*k2/2);k4=model(t+h,x+h*k3)
            x=x+h*(k1+2*k2+2*k3+k4)/6;current+=h
        states.append(x.clone())
    return states
