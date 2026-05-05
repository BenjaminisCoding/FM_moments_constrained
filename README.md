# Constrained Flow Matching v1

CPU-first PyTorch + Hydra implementation for moment-constrained path learning in flow matching.

## Quickstart
```bash
python3 -m pip install -e '.[dev]'
pytest -q
```

## Run Experiments
```bash
# Constrained mode (default experiment)
python3 scripts/run_experiment.py

# Baseline mode
python3 scripts/run_experiment.py experiment=baseline

# Run baseline + constrained comparison in one command
python3 scripts/run_experiment.py experiment=comparison

# Run baseline + constrained + MFM + MFM(alpha=0) comparison
python3 scripts/run_experiment.py experiment=comparison_mfm

# Fast smoke budget
python3 scripts/run_experiment.py experiment=comparison train=smoke

# A+B only ablation (disable Stage C joint optimization)
python3 scripts/run_experiment.py experiment=comparison train=ab_only

# Random-coupling ablation: baseline vs constrained A+B (no OT pairing)
python3 scripts/run_experiment.py \
  experiment=comparison \
  train=ab_only \
  data=gaussian_random \
  experiment.label=comparison_random_coupling

# Compare OT vs random coupling with both intermediate metrics enabled
python3 scripts/run_experiment.py \
  experiment=comparison \
  train=ab_only \
  data=gaussian_ot \
  experiment.label=comparison_ot_empirical \
  output.save_plots=false

python3 scripts/run_experiment.py \
  experiment=comparison \
  train=ab_only \
  data=gaussian_random \
  experiment.label=comparison_random_empirical \
  output.save_plots=false

# Friendly bridge best-preset runner (single seed) with MFM comparisons
python3 scripts/run_bridge_mfm_best.py
```

Outputs are saved under `outputs/<date>/<experiment_label>_<train_label>/<time>/`.
Per-run summaries now include intermediate-time distribution metrics:
`intermediate_w2_gaussian` (Gaussian-moment proxy) and
`intermediate_empirical_w2` (sample-based discrete OT), each with per-time and average fields.

When using `experiment=comparison_mfm`, outputs include:
- `comparison_mfm.json` with method keys `baseline`, `constrained`, `metric`, `metric_alpha0`.
- legacy `comparison.json` preserved with only `baseline` and `constrained` for compatibility with existing tooling.

You can control sample-based OT evaluation cost with:
- `train.eval_intermediate_empirical_w2=true|false`
- `train.eval_intermediate_ot_samples=<int>`

## Bridge SDE Preview Notebook
Use the notebook to validate bridge geometry before training:
- `notebooks/bridge_sde_visualization.ipynb`

The notebook uses shared functions from `src/cfm_project/bridge_sde.py` and
`src/cfm_project/plotting.py`, and exports preview artifacts under:
- `outputs/preview_bridge_sde/<YYYY-MM-DD_HH-MM-SS>/`
