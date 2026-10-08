"""Fixed-support loss and constraint-gradient diagnostics."""
import numpy as np
import torch
from cfm_project import training as tr
from cfm_project.constraints import normalize_residual_blocks, augmented_lagrangian_block_terms
from cfm_project.paths import path_and_velocity, vector_time_derivative
from cfm_project.mfm_core import mfm_path_and_velocity, land_geopath_loss
BLOCKS = ('mean', 'y_variance')

def grad(loss, model, retain=False):
    params = tuple(model.parameters())
    values = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=retain)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                      for p, g in zip(params, values)])


def diagnostic(kw, x0, x1, step, cfg, method):
    model = kw["g_model"] if method == "lin" else kw["geopath_model"]
    flags = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(True)
    try:
        bases, gradients = [], []
        for t in (.125, .375, .625, .875):
            times = torch.full((len(x0), 1), t, dtype=x0.dtype)
            if method == "lin":
                _, velocity, t_req = path_and_velocity(mode="constrained", t=times, x0=x0, x1=x1,
                                                        g_model=model, create_graph=True)
                accel = vector_time_derivative(velocity, t_req, create_graph=True)
                base = kw["alpha"] * (velocity - (x1-x0)).square().sum(1).mean() + kw["beta"] * accel.square().sum(1).mean()
            else:
                position, velocity, _ = mfm_path_and_velocity(t=times, x0=x0, x1=x1,
                    geopath_net=model, alpha=kw["alpha_mfm"], create_graph=True)
                base = land_geopath_loss(position, velocity, kw["manifold_samples"],
                    gamma=kw["land_gamma"], rho=kw["land_rho"])
            bases.append(float(base.detach()))
            gradients.append(grad(base, model))
        g0 = torch.stack(gradients).mean(0)
        raw = tr._constraint_residual_blocks_for_mode(mode=cfg["experiment"]["mode"], x0=x0, x1=x1,
            times=[.5], targets=kw["targets"], g_model=model, moment_feature_blocks=BLOCKS,
            moment_feature_params={}, mfm_alpha=cfg["mfm"]["alpha"])
        normalized = normalize_residual_blocks(raw, dim=2, normalization=cfg["train"]["moment_block_normalization"])
        if cfg["train"]["moment_block_normalization"] == "none":
            residual = torch.cat([raw[.5][b] for b in BLOCKS])
            quadratic = .5 * residual.square().sum()
            actual = kw["lambdas"][.5].dot(residual) + kw["rho"] * quadratic
        else:
            zeros = {t: {b: torch.zeros_like(v) for b, v in blocks.items()} for t, blocks in normalized.items()}
            quadratic, _, _ = augmented_lagrangian_block_terms(normalized, zeros, rho=1.)
            actual, _, _ = augmented_lagrangian_block_terms(normalized, kw["lambdas"], rho=kw["rho"])
        gq = grad(quadratic, model, retain=True)
        gc = grad(kw["moment_eta"] * actual, model)
        vector = torch.cat([raw[.5][b] for b in BLOCKS]).detach()
        lambda_values = kw["lambdas"][.5]
        multipliers = torch.cat([v.flatten() for v in lambda_values.values()]) if isinstance(lambda_values, dict) else lambda_values
        n0, nq, nc = [float(g.norm()) for g in (g0, gq, gc)]
        return dict(step=step, support_size=len(x0), base_loss=float(np.mean(bases)),
            raw_residual_vector=vector.tolist(), raw_joint_residual=float(vector.norm()),
            mean_residual_l2=float(vector[:2].norm()), y_variance_residual_abs=abs(float(vector[2])),
            base_gradient_l2=n0, quadratic_gradient_l2=nq, constraint_gradient_l2=nc,
            kappa_balance=n0/max(nq, 1e-30), constraint_base_ratio=nc/max(n0, 1e-30),
            gradient_cosine=float(g0.dot(gc))/max(n0*nc, 1e-30), total_gradient_l2=float((g0+gc).norm()),
            lambda_l2=float(multipliers.norm()), lambda_linf=float(multipliers.abs().max()),
            lambda_clip_fraction=float((multipliers.abs() >= cfg["train"]["lambda_clip"]).float().mean()))
    finally:
        for p, flag in zip(model.parameters(), flags):
            p.requires_grad_(flag)
