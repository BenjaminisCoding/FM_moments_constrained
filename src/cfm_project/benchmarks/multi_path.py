"""Path objectives, derivatives and W2 checkpoint control for Multi."""
import numpy as np
import torch
from cfm_project import training as tr
from cfm_project.data import sample_coupled_batch
from cfm_project.models import VelocityField
from cfm_project.paths import path_and_velocity
from cfm_project.mfm_core import mfm_path_and_velocity, land_metric_tensor, mfm_gamma, mfm_d_gamma
MODES = {"lin": "constrained", "land": "metric_constrained_al"}
MIN_DELTA = .001
MIN_STEPS = 1200
PATIENCE = 6

def path_sample(s, t, a=None, b=None):
    a = s["x0"] if a is None else a
    b = s["x1"] if b is None else b
    return tr._path_samples_for_mode(mode=MODES[s["method"]], x0=a, x1=b,
        t_batch=torch.full((len(a), 1), t), g_model=s["model"], mfm_alpha=1.)


def fast_path_velocity(s, t, a, b):
    # Forward-mode differentiation costs one tangent pass, instead of 100
    # reverse passes. Parameter gradients through this tangent remain enabled.
    g, dg = torch.func.jvp(lambda u: s["model"](u, a, b), (t,), (torch.ones_like(t),))
    if s["method"] == "lin":
        gate, derivative = t * (1-t), 1-2*t
    else:
        gate, derivative = mfm_gamma(t), mfm_d_gamma(t)
    return (1-t)*a+t*b+gate*g, (b-a)+derivative*g+gate*dg


def fast_metric(x, references, gamma, rho):
    # Algebraic weighted squared differences avoid a B x N x D allocation.
    xx, yy = x.square(), references.square()
    dist = (xx.sum(1, keepdim=True) + yy.sum(1)[None, :] - 2*x@references.T).clamp_min(0)
    weights = torch.exp(-dist/(2*gamma**2))
    squared = weights@yy - 2*x*(weights@references) + xx*weights.sum(1, keepdim=True)
    return 1/(squared + rho)


def base_loss(s, a, b, t, fast=True, weights=None):
    if fast:
        position, velocity = fast_path_velocity(s, t, a, b)
    elif s["method"] == "lin":
        position, velocity, _ = path_and_velocity(mode="constrained", t=t, x0=a, x1=b,
            g_model=s["model"], create_graph=True)
    else:
        position, velocity, _ = mfm_path_and_velocity(t=t, x0=a, x1=b,
            geopath_net=s["model"], alpha=1., create_graph=True)
    if s["method"] == "lin":
        loss = s["cfg"]["train"]["alpha"]*(velocity-(b-a)).square().sum(1)
    else:
        fn = fast_metric if fast else land_metric_tensor
        metric = fn(position, s["references"], s["cfg"]["mfm"]["land_gamma"], .001)
        loss = (velocity.square()*metric).sum(1)
    return loss.mean() if weights is None else (loss*weights).sum()


def vector_gradient(loss, model, retain=False):
    params = tuple(model.parameters())
    grads = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=retain)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten() for p,g in zip(params,grads)])


def take_step(s, step, fast=True):
    a,b,_ = sample_coupled_batch(s["problem"], batch_size=256, coupling="ot_global", generator=s["generator"])
    t = torch.rand((len(a),1))
    active = step >= s["warmup"]
    base = base_loss(s,a,b,t,fast=fast)
    residual = s["q"](path_sample(s,s["tau"],a,b)).mean(0)-s["target"] if active else None
    rho = s["cfg"]["train"]["pseudo_rho"]
    loss = base if residual is None else base+(s["multipliers"][s["tau"]].dot(residual)+.5*rho*residual.dot(residual))
    s["optimizer"].zero_grad(set_to_none=True)
    loss.backward(); s["optimizer"].step()
    if active:
        s["multipliers"] = tr.update_lagrange_multipliers(s["multipliers"], {s["tau"]:residual.detach()}, rho=rho, clip_value=100.)
    return dict(step=step+1,loss=float(loss.detach()),base=float(base.detach()),constraint_active=active,
        residual_l2=None if residual is None else float(residual.detach().norm()),
        lambda_linf=float(s["multipliers"][s["tau"]].abs().max()),lr=s["optimizer"].param_groups[0]["lr"])


def diagnostics(s, step):
    losses, gradients = [], []
    a,b,w = s["x0"],s["x1"],s["mass"]
    for t in (.125,.375,.625,.875):
        value = base_loss(s,a,b,torch.full((len(a),1),t),weights=w)
        losses.append(float(value.detach())); gradients.append(vector_gradient(value,s["model"]))
    g0 = torch.stack(gradients).mean(0)
    composition = (s["q"](path_sample(s,s["tau"]))*w[:,None]).sum(0)
    residual = composition-s["target"]
    quadratic = .5*residual.square().sum()
    gq = vector_gradient(quadratic,s["model"],retain=True)
    actual = s["multipliers"][s["tau"]].dot(residual)+s["cfg"]["train"]["pseudo_rho"]*quadratic
    gc = vector_gradient(actual,s["model"])
    n0,nq,nc = [float(g.norm()) for g in (g0,gq,gc)]
    lam = s["multipliers"][s["tau"]]
    return dict(step=step,residual_l2=float(residual.detach().norm()),residual_vector=residual.detach().tolist(),
        base_loss=float(np.mean(losses)),base_gradient_l2=n0,quadratic_gradient_l2=nq,constraint_gradient_l2=nc,
        kappa_balance=n0/max(nq,1e-30),constraint_base_ratio=nc/max(n0,1e-30),
        gradient_cosine=float(g0.dot(gc))/max(n0*nc,1e-30),total_gradient_l2=float((g0+gc).norm()),
        lambda_linf=float(lam.abs().max()),lambda_l2=float(lam.norm()),lambda_clip_fraction=float((lam.abs()>=100).float().mean()))


def new_velocity(s,seed):
    torch.manual_seed(seed)
    cfg=s["cfg"]["model"]
    return VelocityField(100,cfg["velocity_hidden_dims"],cfg["activation"])


def make_rngs(seed,offset=0):
    return [torch.Generator().manual_seed(seed+offset+d) for d in (1200001,1200002,1200003)]


def initial_control():
    return dict(best_w2=None,best_step=None,significant_reference=None,bad_checks=0,
        lr_drops=0,reduced_this_streak=False)


def assess_w2(control,value,step):
    assert np.isfinite(value)
    new_best=control["best_w2"] is None or value<control["best_w2"]
    if new_best:control.update(best_w2=value,best_step=step)
    significant=control["significant_reference"] is None or value<control["significant_reference"]-MIN_DELTA
    if significant:
        control.update(significant_reference=value,bad_checks=0,reduced_this_streak=False)
    else:control["bad_checks"]+=1
    action="continue"
    if step>=MIN_STEPS and control["bad_checks"]>=PATIENCE:
        action="stop"
    elif control["bad_checks"]>=3 and not control["reduced_this_streak"] and control["lr_drops"]<2:
        action="reduce_lr";control["lr_drops"]+=1;control["reduced_this_streak"]=True
    return action,new_best,significant
