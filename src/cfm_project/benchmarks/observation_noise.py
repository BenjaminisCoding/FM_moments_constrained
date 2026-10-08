"""Controlled optical-observation noise with a fixed seed and training budget."""
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from cfm_project import aggregate_benchmarks as core
from cfm_project.histogram_transport import histogram_w2
from cfm_project.models import PathCorrection, VelocityField
from cfm_project.population_quadrature import load_fit_training, refine_population
TRAINING_SEED = 3
MAGNITUDES = (0., .05, .10, .20, .40, .60, .80, .95)
NOISE_SEEDS = (3, 7, 11, 13, 17)

def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_digest(model):
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def notice(value):
    print(json.dumps(value, allow_nan=False), flush=True)


def verify_hashes(root, manifest):
    for name, expected in manifest.items():
        if digest(Path(root) / name) != expected:
            raise RuntimeError(f"Frozen artifact changed: {name}")


def protocol(output):
    value = read(output / "protocol.json")
    verify_hashes(output, value["input_hashes"])
    verify_hashes(output, value["source_hashes"])
    return value


def oracle_integral(evaluation, config):
    grid = np.asarray(config["log_diameter_grid"])
    response = np.asarray(config["response_um2"]) / config["feature_scale_um2"]
    with np.load(evaluation, allow_pickle=False) as archive:
        index = int(np.flatnonzero(np.isclose(archive["times"], .5))[0])
        probabilities = archive["bin_probabilities"][index].astype(float)
        edges = archive["log_bin_edges"].astype(float)
    probabilities /= probabilities.sum()
    total = 0.
    for mass, left, right in zip(probabilities, edges[:-1], edges[1:]):
        knots = np.r_[left, grid[(grid > left) & (grid < right)], right]
        total += mass * np.trapezoid(np.interp(knots, grid, response), knots) / (right - left)
    return float(total)


@contextmanager
def fitting_input_guard():
    """Fail if an NPZ other than a declared endpoint/scalar training file is opened."""
    original = np.load
    def guarded(file, *args, **kwargs):
        if not isinstance(file, (str, Path)) or Path(file).name != "train.npz":
            raise RuntimeError(f"Fit attempted to open a non-training NumPy archive: {file}")
        return original(file, *args, **kwargs)
    np.load = guarded
    try:
        yield
    finally:
        np.load = original


def fit(output, name, smoke_suffix=None):
    declaration = protocol(output)
    job = next(j for j in declaration["jobs"] if j["name"] == name)
    config = dict(declaration["config"])
    folder = output / "runs" / name
    if smoke_suffix is not None:
        folder = output / "smoke" / smoke_suffix
        config.update(outer_steps=2, inner_steps=2, warmup_steps=3,
                      stage_b_steps=8, stage_b_log_every=4)
    if (folder / "result.json").exists():
        result = read(folder / "result.json")
        assert result["job"] == job and result["config"] == config
        verify_hashes(folder, result["checkpoint_hashes"])
        return result
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "owner.json").open("x") as stream:
        json.dump(dict(pid=os.getpid(), started_utc=now(), job=job), stream)
    started = time.perf_counter()
    with fitting_input_guard(), threadpool_limits(limits=1):
        data = load_fit_training(output / "inputs" / name, config)
        assert data.tau == .5 and data.target.numel() == 1
        assert float(data.target[0]) == float(np.float32(job["target"]))
        torch.manual_seed(TRAINING_SEED)
        initial_path = PathCorrection(1, config["hidden"], config["activation"])
        path_initial_hash = model_digest(initial_path)
        torch.manual_seed(TRAINING_SEED + 100003)
        initial_velocity = VelocityField(1, config["velocity_hidden"], config["activation"])
        velocity_initial_hash = model_digest(initial_velocity)
        path, coupling, stage_a = core.fit_path(data, "gmi_land", TRAINING_SEED, config, folder, notice)
        teacher_hash = model_digest(path)
        stream_hash = hashlib.sha256()
        original_path_velocity = core.path_velocity
        calls = 0
        def traced_path_velocity(model, t, x0, x1):
            nonlocal calls
            for tensor in (t, x0, x1):
                stream_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            calls += 1
            return original_path_velocity(model, t, x0, x1)
        core.path_velocity = traced_path_velocity
        try:
            stage_b = core.fit_velocity(data, "gmi_land", TRAINING_SEED, config, path, None, folder, notice)
        finally:
            core.path_velocity = original_path_velocity
        assert model_digest(path) == teacher_hash
        assert stage_a["outer_steps"] == config["outer_steps"]
        assert stage_b["steps"] == config["stage_b_steps"]
        assert calls == config["stage_b_steps"] + 1
    result = dict(job=job, config=config, protocol_sha256=digest(output / "protocol.json"),
                  data_sha256=digest(output / "inputs" / name / "train.npz"),
                  coupling=coupling, stage_a=stage_a, stage_b=stage_b,
                  randomness=dict(path_initial_hash=path_initial_hash, velocity_initial_hash=velocity_initial_hash,
                                  stage_b_endpoint_time_stream_hash=stream_hash.hexdigest(),
                                  stage_b_stream_calls=calls, deterministic_algorithms=True,
                                  initial_seed=TRAINING_SEED),
                  stage_b_preserves_teacher=True,
                  checkpoint_hashes={p: digest(folder / p) for p in ("path.pt", "velocity.pt")},
                  wall_seconds=time.perf_counter() - started, completed_utc=now())
    write(folder / "result.json", result)
    notice(dict(event="fit_complete", job=name, seconds=result["wall_seconds"]))
    return result


def evaluate(output, name):
    declaration = protocol(output)
    folder = output / "runs" / name
    result = read(folder / "result.json")
    assert result["protocol_sha256"] == digest(output / "protocol.json")
    verify_hashes(folder, result["checkpoint_hashes"])
    frozen = dict(frozen_utc=now(), result_sha256=digest(folder / "result.json"),
                  checkpoint_hashes=result["checkpoint_hashes"], protocol_sha256=result["protocol_sha256"])
    if (folder / "evaluation_freeze.json").exists():
        previous = read(folder / "evaluation_freeze.json")
        for key in ("result_sha256", "checkpoint_hashes", "protocol_sha256"):
            assert previous[key] == frozen[key]
    else:
        write(folder / "evaluation_freeze.json", frozen)
    if (folder / "metrics.json").exists():
        return read(folder / "metrics.json")
    config = result["config"]
    settings = declaration["evaluation"]
    data = load_fit_training(output / "inputs" / name, dict(population_quadrature_factor=settings["population_factor"]))
    fit_data = load_fit_training(output / "inputs" / name, config)
    path = PathCorrection(1, config["hidden"], config["activation"])
    path.load_state_dict(torch.load(folder / "path.pt", map_location="cpu", weights_only=False)["state_dict"])
    path.requires_grad_(False)
    model = VelocityField(1, config["velocity_hidden"], config["activation"])
    model.load_state_dict(torch.load(folder / "velocity.pt", map_location="cpu", weights_only=False)["state_dict"])
    model.eval()
    with np.load(output / "evaluation_only/june1.npz", allow_pickle=False) as archive:
        times = archive["times"].copy()
        probabilities = archive["bin_probabilities"].copy()
        edges = archive["log_bin_edges"].copy()
    center = float(data.observation_config["physical_center"][0])
    scale = float(data.observation_config["physical_scale"][0])
    with threadpool_limits(limits=1), torch.no_grad():
        generated = core.rollout(model, data.x0, times, settings["ode_steps"])
        generated_dense = core.rollout(model, data.x0, times, settings["ode_audit_steps"])
        teacher = [core.path_velocity(path, torch.full((len(data.x0), 1), float(t)), data.x0, data.x1)[0] for t in times]
        weights = data.weights.numpy()
        rows = []
        def distance(x, index, physical=False):
            return histogram_w2(x.numpy(), weights, probabilities[index], edges, center, scale, physical)
        for i, (t, pred, direct, dense) in enumerate(zip(times, generated, teacher, generated_dense)):
            rows.append(dict(time=float(t), rollout_w2=distance(pred, i), interpolant_w2=distance(direct, i),
                             rollout_diameter_w2_nm=distance(pred, i, True),
                             interpolant_diameter_w2_nm=distance(direct, i, True),
                             ode_refinement_w2_change=abs(distance(pred, i) - distance(dense, i))))
        def moment(x, source):
            return float((core.observation(x, source.observation_config) * source.weights[:, None]).sum())
        direct_moment, rollout_moment = moment(teacher[2], data), moment(generated[2], data)
        moments = []
        for factor in (1, 2, 4):
            refined = refine_population(fit_data, factor)
            middle = core.path_velocity(path, torch.full((len(refined.x0), 1), .5), refined.x0, refined.x1)[0]
            moments.append(moment(middle, refined))
    target = result["job"]["target"]
    metrics = dict(job=result["job"], rows=rows,
                   mean_unobserved_w2=float(np.mean([rows[i]["rollout_w2"] for i in (1, 3)])),
                   mean_unobserved_interpolant_w2=float(np.mean([rows[i]["interpolant_w2"] for i in (1, 3)])),
                   oracle=declaration["oracle"], target=target,
                   midpoint_rollout_moment=rollout_moment, midpoint_interpolant_moment=direct_moment,
                   rollout_supplied_residual=rollout_moment - target,
                   interpolant_supplied_residual=direct_moment - target,
                   rollout_oracle_residual=rollout_moment - declaration["oracle"],
                   interpolant_oracle_residual=direct_moment - declaration["oracle"],
                   audit=dict(population_moments_factors_1_2_4=moments,
                              maximum_population_moment_change=max(abs(moments[i] - moments[0]) for i in (1, 2)),
                              max_ode_w2_change=max(r["ode_refinement_w2_change"] for r in rows),
                              stage_a_final_residual=result["stage_a"]["final"]["residual_linf"],
                              stage_a_final_gradient=result["stage_a"]["final"]["total_gradient_norm"]),
                   evaluation_sha256=digest(output / "evaluation_only/june1.npz"),
                   evaluation_freeze_sha256=digest(folder / "evaluation_freeze.json"))
    write(folder / "metrics.json", metrics)
    np.savez_compressed(folder / "evaluated_samples.npz", times=times, weights=weights,
                        rollout=np.stack([x.numpy() for x in generated]),
                        interpolant=np.stack([x.numpy() for x in teacher]))
    notice(dict(event="evaluated", job=name, primary=metrics["mean_unobserved_w2"]))
    return metrics


def prepare(output, data_root, config):
    if (output / 'protocol.json').exists():
        protocol(output)
        return
    metadata = read(data_root / 'metadata.json')
    reference = oracle_integral(data_root / 'evaluation.npz', metadata['observation_config'])
    with np.load(data_root / 'train.npz', allow_pickle=False) as archive:
        arrays = dict(archive)
    config = dict(config, stage_a_early_stopping=False, threads=1)
    draws = {str(s): float(np.random.default_rng(s).uniform(-1., 1.)) for s in NOISE_SEEDS}
    jobs = []
    for magnitude in MAGNITUDES:
        for seed in (None,) if magnitude == 0 else NOISE_SEEDS:
            epsilon = 0. if seed is None else draws[str(seed)]
            target = reference * (1 + magnitude * epsilon)
            name = 'oracle' if magnitude == 0 else f'r_{round(100*magnitude):03d}/noise_{seed}'
            job = dict(name=name, magnitude=magnitude, noise_seed=seed, epsilon=epsilon,
                       relative_error=magnitude*epsilon, target=target, training_seed=TRAINING_SEED)
            dest = output / 'inputs' / name
            dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest / 'train.npz', **(arrays | {'target': np.asarray([target], dtype=arrays['target'].dtype)}))
            write(dest / 'metadata.json', dict(metadata, observation_noise=job,
                information='Only endpoints and scalar oracle target plus controlled noise enter training; endpoint-only LAND.'))
            jobs.append(job)
    (output / 'evaluation_only').mkdir(exist_ok=True)
    shutil.copyfile(data_root / 'evaluation.npz', output / 'evaluation_only/june1.npz')
    files = [p for folder in ['inputs', 'evaluation_only'] for p in (output/folder).rglob('*') if p.is_file()]
    write(output / 'protocol.json', dict(config=config, jobs=jobs, oracle=reference,
        input_hashes={str(p.relative_to(output)): digest(p) for p in files}, source_hashes={},
        noise=dict(magnitudes=MAGNITUDES, seeds=NOISE_SEEDS, draws=draws),
        evaluation=dict(ode_steps=400, ode_audit_steps=800, population_factor=4),
        disclosure='June1 observation-error ablation. Preparation computes one oracle scalar from the midpoint histogram; model training receives the scalar and endpoints.'))
