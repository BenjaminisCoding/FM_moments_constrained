# EXPERIMENTS.md

## Purpose
This file is the experiment index for the `outputs/` directory.
Each entry links experiment folders to what was tested, so results are traceable without re-reading commands.

## Repository Boundary
`FM/` is environment tooling (Python virtual environment), not project source architecture or experiment output.
Do not log `FM/` internals as experiment artifacts.

## Update Rule
- Whenever a new experiment is run, add an entry in this file in the same session.
- For single runs, record the exact output folder.
- For sweeps, record the sweep root folder and list variant subfolders.

## Experiment Log
### [2026-04-27] Bridge-SDE notebook preview artifacts
- Folder pattern:
  - `outputs/preview_bridge_sde/<YYYY-MM-DD_HH-MM-SS>/`
- Executed folder:
  - `outputs/preview_bridge_sde/2026-04-27_12-13-13`
- What was tested:
  - Pre-training visual validation of bridge geometry using shared simulation/plotting utilities.
  - Generated `snapshot_grid.png`, `y_spread.png`, `bridge_animation.gif`, and `summary.txt`.

### [2026-04-26] OT vs random coupling with empirical intermediate OT metric (A+B only)
- Folders:
  - `outputs/2026-04-26/comparison_ot_empirical_ab_only/19-59-09`
  - `outputs/2026-04-26/comparison_random_empirical_ab_only/19-59-23`
- What was tested:
  - Baseline vs constrained with Stage C disabled, now including both intermediate metrics:
    - Gaussian moment-based `intermediate_w2_gaussian`,
    - sample-based `intermediate_empirical_w2` (discrete OT matching).
  - Direct coupling sensitivity check (`data.coupling=ot` vs `data.coupling=random`) for intermediate distribution fit and endpoint transport metrics.

### [2026-04-26] Metric integration smoke check (intermediate Wasserstein)
- Folder:
  - `outputs/2026-04-26/comparison_metric_check_smoke/19-51-55`
- What was tested:
  - Baseline vs constrained smoke comparison to verify the new intermediate-time Gaussian Wasserstein metrics are produced in summaries.
  - Confirmed presence of `intermediate_w2_gaussian` and `intermediate_w2_gaussian_avg` in `comparison.json`.

### [2026-04-26] Random-coupling baseline vs A+B comparison
- Folder:
  - `outputs/2026-04-26/comparison_random_coupling_ab_only/19-45-50`
- What was tested:
  - Same Gaussian benchmark and A+B-only schedule (`stage_c_steps=0`) but with `data.coupling=random` instead of OT pairing.
  - Direct baseline vs constrained comparison under random pair coupling.

### [2026-04-26] Timestamped Hydra comparison runs
- Folders:
  - `outputs/2026-04-26/18-45-43`
  - `outputs/2026-04-26/18-46-33`
  - `outputs/2026-04-26/18-47-28`
  - `outputs/2026-04-26/18-50-13`
- What was tested:
  - Baseline vs constrained comparisons from CLI/Hydra runs.
  - Includes smoke-scale and default-budget comparisons, with plot/no-plot variants.

### [2026-04-26] Constrained hyperparameter sweep (with Stage C enabled)
- Folder root:
  - `outputs/hparam_sweep`
- Variants:
  - `alpha0_2_rho1_eta0_5`
  - `beta0_2_eta0_5`
  - `default`
  - `eta_0_2`
  - `eta_0_5`
  - `eta_1_0`
  - `rho10_eta0_5`
  - `rho1_eta0_5`
  - `rho1_eta1_0`
  - `stagec20_eta0_5`
  - `stagec_0`
- What was tested:
  - Sensitivity of constrained training to `rho`, `eta_joint`, and Stage C length.
  - Baseline settings kept fixed for direct constrained-side comparison.

### [2026-04-26] Seed robustness checks
- Folder root:
  - `outputs/seed_check`
- Variants:
  - `default`
  - `tuned_rho1_eta0_5`
  - `rho1_eta1`
  - `rho1_eta0_3`
  - `rho1_eta0_5_stagec20`
  - `rho1_stagec0`
  - `rho1_lr3e4_eta0_5`
  - `rho1_lr3e4_eta1`
  - `rho1_stagea400_lr3e4_eta0_5`
- Seed folders:
  - each variant includes `seed_3`, `seed_7`, `seed_11`
- What was tested:
  - Stability of baseline vs constrained metrics across random seeds for selected configurations.

### [2026-04-26] Explicit A+B-only comparison runs
- Folders:
  - `outputs/2026-04-26/comparison_ab_only/19-34-05`
  - `outputs/2026-04-26/comparison_ab_only/19-35-38`
- What was tested:
  - Baseline vs constrained with Stage C disabled (`stage_c_steps=0`).
  - Isolates path-pretrain + velocity-train behavior without joint finetuning.

### [2026-04-26] A+B-only hyperparameter sweep (Stage C disabled)
- Folder root:
  - `outputs/ab_only_sweep`
- Variants:
  - `ab_only_default`
  - `rho_0p5`
  - `rho_1p0`
  - `rho_2p0`
  - `lrg_3e4`
  - `lrg_5e4`
  - `rho1_lrg3e4`
  - `rho1_lrg5e4`
  - `alpha_0p5`
  - `alpha_2p0`
  - `beta_0p01`
  - `beta_0p10`
  - `rho1_lrg3e4_beta0p01`
  - `rho1_lrg3e4_beta0p10`
  - `rho1_lrg3e4_alpha0p5`
  - `rho1_lrg3e4_alpha2p0`
- Additional artifact:
  - `outputs/ab_only_sweep/summary.tsv`
- What was tested:
  - Constraint-vs-transport tradeoff for A+B-only training by varying `rho`, `lr_g`, `alpha`, and `beta`.
