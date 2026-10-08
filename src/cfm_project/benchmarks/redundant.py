"""Exactly redundant midpoint-mean control with a zero-initialized path."""
import hashlib
import time
import numpy as np
import torch
from cfm_project.data import EmpiricalCouplingProblem, sample_coupled_batch
from cfm_project.constraints import update_lagrange_multipliers
from cfm_project.models import PathCorrection, VelocityField
from cfm_project.paths import corrected_path, path_and_velocity, vector_time_derivative
from cfm_project.training import _cfm_loss, _constrained_objective, set_seed
from cfm_project.metrics import balanced_empirical_w2_distance, euler_velocity_snapshots
from .common import write_json, digest, read_json
TIMES = [.25, .5, .75, 1.]

def model_digest(model):
    result = hashlib.sha256()
    for key, value in model.state_dict().items():
        result.update(key.encode())
        result.update(value.detach().cpu().numpy().tobytes())
    return result.hexdigest()


def load_inputs(folder):
    with np.load(folder / "training_inputs.npz") as data:
        expected = {"x0", "x1", "src_idx", "tgt_idx", "mass", "target_mean"}
        assert set(data.files) == expected
        values = {k: torch.from_numpy(data[k].copy()) for k in data.files}
    problem = EmpiricalCouplingProblem(
        values["x0"], values["x1"], label="nonarched_redundant_mean",
        global_ot_src_idx=values["src_idx"], global_ot_tgt_idx=values["tgt_idx"],
        global_ot_mass=values["mass"])
    x0 = problem.x0_pool[problem.global_ot_src_idx]
    x1 = problem.x1_pool[problem.global_ot_tgt_idx]
    assert torch.equal(problem.global_ot_mass, problem.global_ot_mass[:1].expand_as(problem.global_ot_mass))
    assert torch.unique(problem.global_ot_src_idx).numel() == len(problem.x0_pool)
    assert torch.unique(problem.global_ot_tgt_idx).numel() == len(problem.x1_pool)
    assert torch.equal(((x0 + x1) * 0.5).mean(0), values["target_mean"])
    return problem, x0, x1, values["target_mean"]


def flat_gradient(loss, model, retain_graph=False):
    parameters = tuple(model.parameters())
    grads = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                      for p, g in zip(parameters, grads)])


def path_diagnostic(g, x0, x1, target, multiplier, config, step):
    midpoint = corrected_path(torch.full((len(x0), 1), 0.5), x0, x1, g)
    residual = midpoint.mean(0) - target
    quadratic = 0.5 * residual.square().sum()
    force = config["moment_eta"]*(multiplier.dot(residual) + config["rho"]*quadratic)
    gq = flat_gradient(quadratic, g, retain_graph=True)
    gc = flat_gradient(force, g)
    bases, energies, accelerations, gradients = [], [], [], []
    for t in (0.125, 0.375, 0.625, 0.875):
        _, velocity, time_tensor = path_and_velocity("constrained", torch.full((len(x0), 1), t),
                                                     x0, x1, g, create_graph=True)
        energy = (velocity - (x1-x0)).square().sum(1).mean()
        acceleration = vector_time_derivative(velocity, time_tensor, create_graph=True).square().sum(1).mean()
        base = config["alpha"]*energy + config["beta"]*acceleration
        gradients.append(flat_gradient(base, g))
        bases.append(float(base.detach()))
        energies.append(float(energy.detach()))
        accelerations.append(float(acceleration.detach()))
    gb = torch.stack(gradients).mean(0)
    with torch.no_grad():
        deformation = (midpoint - (x0+x1)*0.5).square().sum(1).mean()
    nb, nq, nc = [float(v.norm()) for v in (gb, gq, gc)]
    return dict(step=step, residual_vector=residual.detach().tolist(), mean_residual_l2=float(residual.detach().norm()),
        midpoint_deformation_mse=float(deformation), midpoint_deformation_rms=float(deformation.sqrt()),
        base_loss=float(np.mean(bases)), velocity_deviation=float(np.mean(energies)),
        acceleration_energy=float(np.mean(accelerations)), base_gradient_l2=nb, quadratic_gradient_l2=nq,
        constraint_gradient_l2=nc, total_gradient_l2=float((gb+gc).norm()),
        balance_ratio=None if nq < 1e-12 else nb/nq,
        constraint_base_gradient_ratio=None if nb < 1e-12 else nc/nb,
        gradient_cosine=None if nb*nc < 1e-20 else float(gb.dot(gc))/(nb*nc),
        multiplier_l2=float(multiplier.norm()), multiplier_linf=float(multiplier.abs().max()))


def train(out, args, seed, mode):
    folder = out / "runs" / mode / f"seed_{seed}"
    folder.mkdir(parents=True)
    problem, x0, x1, target = load_inputs(out / "data" / f"seed_{seed}")
    config = read_json(out / "train_config.json")
    config.update(stage_a_steps=args.stage_a_steps if mode == "constrained" else 0,
                  stage_b_steps=args.stage_b_steps, batch_size=args.batch_size)
    set_seed(seed)
    velocity = VelocityField(2, [128, 128], "silu")
    initial_velocity_sha256 = model_digest(velocity)
    g = PathCorrection(2, [128, 128], "silu", zero_init_output=True) if mode == "constrained" else None
    write_json(folder / "config.json", dict(seed=seed, mode=mode, train=config,
        path_zero_init_output=True, hidden_dims=[128, 128], activation="silu",
        full_intermediate_samples_available_to_training=False))
    start = time.perf_counter()
    history, diagnostics = [], []
    multiplier = torch.zeros_like(target)
    initial_correction_max = 0.0
    clipped_steps = 0
    if g is not None:
        with torch.no_grad():
            initial_correction_max = float(g(torch.full((len(x0), 1), 0.5), x0, x1).abs().max())
        assert initial_correction_max == 0.0
        optimizer = torch.optim.Adam(g.parameters(), lr=config["lr_g"])
        pairs_rng = torch.Generator().manual_seed(110000+seed)
        times_rng = torch.Generator().manual_seed(120000+seed)
        for step in range(args.stage_a_steps+1):
            if step % 50 == 0 or step == args.stage_a_steps:
                row = path_diagnostic(g, x0, x1, target, multiplier, config, step)
                diagnostics.append(row)
                write_json(folder / "stage_a_diagnostics.json", diagnostics)
                print(f"seed={seed} GMI A={step}: residual={row['mean_residual_l2']:.6g}, deformation RMS={row['midpoint_deformation_rms']:.6g}", flush=True)
            if step == args.stage_a_steps:
                break
            a, b, _ = sample_coupled_batch(problem, args.batch_size, "ot_global", pairs_rng)
            optimizer.zero_grad(set_to_none=True)
            loss, residuals, _, stats = _constrained_objective(
                g, a, b, times=[0.5], targets={0.5: target}, lambdas={0.5: multiplier},
                rho=config["rho"], alpha=config["alpha"], beta=config["beta"],
                moment_eta=config["moment_eta"], moment_feature_blocks=("mean",),
                moment_block_normalization="none", time_generator=times_rng)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            multiplier = update_lagrange_multipliers({0.5: multiplier}, residuals,
                rho=config["rho"], clip_value=config["lambda_clip"])[0.5]
            clipped_steps += int(bool((multiplier.abs() >= config["lambda_clip"]).any()))
            history.append(dict(step=step+1, loss=float(loss.detach()), **stats))
        assert diagnostics[0]["mean_residual_l2"] == 0.0
        assert diagnostics[0]["base_gradient_l2"] == 0.0
        assert diagnostics[0]["quadratic_gradient_l2"] == 0.0
        write_json(folder / "calibration.json", dict(initial=diagnostics[0],
            decision="Gradient balance is undefined (0/0) at the exact reference optimum; a base-only warmup also has zero gradient. Use fixed rho=0.5, eta=1, alpha=1, beta=0.05 without warmup.",
            diagnostic_only=True))
    stage_a_seconds = time.perf_counter()-start
    write_json(folder / "stage_a_training_history.json", history)
    path_hash_before_b = None if g is None else model_digest(g)
    if g is not None:
        g.requires_grad_(False)
    optimizer_v = torch.optim.Adam(velocity.parameters(), lr=config["lr_v"])
    pairs_rng = torch.Generator().manual_seed(210000+seed)
    times_rng = torch.Generator().manual_seed(220000+seed)
    batch_hash = hashlib.sha256()
    regression = []
    start_b = time.perf_counter()
    for step in range(args.stage_b_steps+1):
        if step % 300 == 0 or step == args.stage_b_steps:
            loss, _ = _cfm_loss(mode, velocity, g, x0, x1,
                               time_generator=torch.Generator().manual_seed(230000+seed))
            row = dict(step=step, fixed_regression_loss=float(loss.detach()))
            regression.append(row)
            write_json(folder / "stage_b_diagnostics.json", regression)
            print(f"seed={seed} {mode} B={step}: regression={row['fixed_regression_loss']:.6g}", flush=True)
        if step == args.stage_b_steps:
            break
        a, b, _ = sample_coupled_batch(problem, args.batch_size, "ot_global", pairs_rng)
        batch_hash.update(a.numpy().tobytes()); batch_hash.update(b.numpy().tobytes())
        optimizer_v.zero_grad(set_to_none=True)
        loss, _ = _cfm_loss(mode, velocity, g, a, b, time_generator=times_rng)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer_v.step()
    stage_b_seconds = time.perf_counter()-start_b
    assert path_hash_before_b == (None if g is None else model_digest(g))
    checkpoint = dict(velocity_state_dict=velocity.state_dict(),
        path_state_dict=None if g is None else g.state_dict(), mode=mode, seed=seed,
        stage_a_steps=config["stage_a_steps"], stage_b_steps=args.stage_b_steps, target_mean=target,
        optimizer_velocity=optimizer_v.state_dict(), multiplier=multiplier, config=config)
    torch.save(checkpoint, folder / "checkpoint.pt")
    record = dict(seed=seed, mode=mode, stage_a_steps=config["stage_a_steps"], stage_b_steps=args.stage_b_steps,
        batch_size=args.batch_size, initial_velocity_sha256=initial_velocity_sha256,
        initial_correction_max=initial_correction_max, stage_b_endpoint_batch_sha256=batch_hash.hexdigest(),
        stage_b_time_rng_final_sha256=hashlib.sha256(times_rng.get_state().numpy().tobytes()).hexdigest(),
        path_frozen_during_b=True, stage_a_seconds=stage_a_seconds, stage_b_seconds=stage_b_seconds,
        multiplier_clipped_steps=clipped_steps, checkpoint_selection="Final fixed-budget iterate; no early stopping",
        checkpoint_sha256=digest(folder / "checkpoint.pt"),
        training_inputs_sha256=digest(out / "data" / f"seed_{seed}" / "training_inputs.npz"))
    write_json(folder / "complete.json", record)
    return record


def evaluate(out, args, seed, mode):
    folder = out / "runs" / mode / f"seed_{seed}"
    problem, x0, x1, target = load_inputs(out / "data" / f"seed_{seed}")
    checkpoint = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=False)
    velocity = VelocityField(2, [128, 128], "silu")
    velocity.load_state_dict(checkpoint["velocity_state_dict"]); velocity.eval()
    g = None
    if mode == "constrained":
        g = PathCorrection(2, [128, 128], "silu")
        g.load_state_dict(checkpoint["path_state_dict"]); g.eval()
    with np.load(out / "data" / f"seed_{seed}" / "evaluation_only.npz") as archive:
        references = {t: torch.from_numpy(archive[f"t_{t:.2f}"].copy()) for t in TIMES}
    rollout = euler_velocity_snapshots(velocity, problem.x0_pool, TIMES, args.euler_steps)
    direct, w2_direct, w2_rollout = {}, {}, {}
    time_grid = np.linspace(0, 1, 33)
    deformation = []
    with torch.no_grad():
        for t in sorted(set(TIMES) | set(time_grid)):
            times = torch.full((len(x0), 1), float(t))
            reference = (1-t)*x0+t*x1
            samples = reference if g is None else corrected_path(times, x0, x1, g)
            if t in TIMES:
                direct[t] = samples
            if t in time_grid:
                deformation.append(float((samples-reference).square().sum(1).mean()))
        assert torch.equal(direct[1.0], x1)
        for t in TIMES:
            w2_direct[f"{t:.2f}"] = balanced_empirical_w2_distance(direct[t], references[t], method="pot_emd2", num_itermax=1600000)
            w2_rollout[f"{t:.2f}"] = balanced_empirical_w2_distance(rollout[t], references[t], method="pot_emd2", num_itermax=1600000)
    row = dict(seed=seed, mode=mode, direct_w2=w2_direct, rollout_w2=w2_rollout,
        direct_mean_residual_l2=float((direct[0.5].mean(0)-target).norm()),
        rollout_mean_residual_l2=float((rollout[0.5].mean(0)-target).norm()),
        midpoint_deformation_rms=float(np.sqrt(deformation[16])),
        integrated_deformation_mse=float(np.trapezoid(deformation, time_grid)),
        integrated_deformation_rms=float(np.sqrt(np.trapezoid(deformation, time_grid))),
        deformation_time_grid=time_grid.tolist(), deformation_mse_by_time=deformation,
        rollout_flanking_w2=0.5*(w2_rollout["0.25"]+w2_rollout["0.75"]),
        direct_flanking_w2=0.5*(w2_direct["0.25"]+w2_direct["0.75"]))
    write_json(folder / "evaluation.json", row)
    np.savez_compressed(folder / "evaluation_samples.npz", **{
        f"{kind}_{t:.2f}": samples[t].numpy() for kind, samples in [("direct", direct), ("rollout", rollout)] for t in TIMES})
    print(f"Evaluated seed={seed} {mode}: {w2_rollout}", flush=True)
    return row
