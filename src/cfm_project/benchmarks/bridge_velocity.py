"""Fixed-budget velocity regression for continuous Brownian bridge teachers."""
import math
import time
import torch
from cfm_project.models import VelocityField
from .common import write_json, notice


def fit_velocity(cfg, teacher, seed, learning_rate, out):
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = VelocityField(cfg['dimension'], cfg['velocity_hidden_dims'], cfg['activation'])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed + 100003)
    diagnostic_generator = torch.Generator().manual_seed(seed + 200003)
    diagnostic = teacher.batch(cfg['regression_validation_samples'], diagnostic_generator, validation=True)
    history = []; started = time.perf_counter()
    for step in range(1, cfg['stage_b_steps'] + 1):
        t, x, u, w = teacher.batch(cfg['batch_size'], generator)
        loss = ((model(t, x) - u).square().sum(1) * w[:, 0]).mean()
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite bridge regression loss')
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step % cfg['diagnostic_every'] == 0 or step == cfg['stage_b_steps']:
            with torch.no_grad():
                td, xd, ud, wd = diagnostic
                errors = (model(td, xd)-ud).square().sum(1)*wd[:, 0]
            row = dict(step=step, training_loss=float(loss.detach()),
                       fixed_regression_loss=float(errors.mean()),
                       fixed_regression_standard_error=float(errors.std()/math.sqrt(len(errors))))
            history.append(row); notice(row)
    torch.save(dict(state_dict=model.state_dict(),step=step,lr=learning_rate),out/'matched.pt')
    write_json(out/'training.json',dict(seed=seed,teacher_fingerprint=teacher.fingerprint(),
        executed_steps=step,selected_step=step,history=history,wall_seconds=time.perf_counter()-started,
        checkpoint_policy='Final fixed-budget iterate; regression remains diagnostic, with no early stopping.',
        sb_loss_weight='6*u*(1-u) within each Brownian subbridge'))


def load_velocity(cfg, path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    model = VelocityField(cfg['dimension'], cfg['velocity_hidden_dims'], cfg['activation'])
    model.load_state_dict(state['state_dict'])
    return model.eval()
