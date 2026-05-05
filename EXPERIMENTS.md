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
### [2026-05-04] Single-cell Stage-A strict leaveout (`ot_global`, POT full-OT) across holdouts `0.25/0.5/0.75` and seeds `3/7/11` — completed
- Folder root:
  - `outputs/2026-05-04/single_cell_eb_stage_a_ot_global_pot_holdout3_seed3/17-43-31`
- What was tested:
  - Stage-A-only strict-leaveout benchmark on EB 5D with:
    - coupling: `data.coupling=ot_global`
    - full-OT metric backend: `train.eval_full_ot_method=pot_emd2`
    - POT iterations: `train.eval_full_ot_num_itermax=1600000`
    - holdouts: indices `1,2,3` (times `0.25,0.50,0.75`)
    - seeds: `3,7,11`
  - Used `comparison_mfm_single_cell_stage_a` with all five methods; extracted constrained-mode holdout metric summary.
- Key artifact:
  - `holdout3_seed3_summary_constrained.json`
- Constrained `learned_holdout_full_ot_w2` summary:
  - Seed-wise (`p0.25`, `p0.50`, `p0.75`, mean):
    - seed `3`: `1.608531`, `1.287885`, `1.269009`, mean `1.388475`
    - seed `7`: `1.600693`, `1.107508`, `1.303015`, mean `1.337072`
    - seed `11`: `2.052194`, `1.364642`, `1.264111`, mean `1.560316`
  - Holdout mean across seeds:
    - `p0.25`: mean `1.753806`, std `0.211016`
    - `p0.50`: mean `1.253345`, std `0.107778`
    - `p0.75`: mean `1.278712`, std `0.017301`
  - Global mean over all 9 values: `1.428621`

### [2026-05-04] Single-cell Stage-A strict leaveout (`ot_global`, full-OT default `pot_emd2`) — completed
- Folder:
  - `outputs/2026-05-04/single_cell_eb_5d_stage_a_strict_leaveout_ot_global_pot_default_single_cell_stage_a_only/17-35-42`
- What was tested:
  - Stage-A-only strict-leaveout run on EB 5D with Kantorovich coupling and full-OT metric backend via POT.
  - Configuration highlights:
    - `experiment=comparison_mfm_single_cell_stage_a`
    - `train=single_cell_stage_a_only`
    - `data=single_cell_eb_5d`
    - `data.single_cell.path=/Users/benpro/Documents/PHD/Projects/Neurips26/code/TrajectoryNet/data/eb_velocity_v5.npz`
    - `data.coupling=ot_global`
    - `experiment.holdout_index=2`
    - `seed=3`
    - `train.eval_full_ot_method=pot_emd2`
    - `train.eval_full_ot_num_itermax=1600000`
    - `output.save_plots=false`
- Outcome snapshot (holdout full-OT \(W_2\), lower is better):
  - constrained: `1.287885`
  - metric_constrained_al: `1.380381`
  - metric_alpha0: `1.460437`
  - metric_constrained_soft: `1.468867`
  - metric: `1.472467`
- Artifacts:
  - `comparison_mfm.json`
  - per-mode folders: `constrained/`, `metric/`, `metric_alpha0/`, `metric_constrained_al/`, `metric_constrained_soft/`

### [2026-05-04] POT `ot.emd2` runtime sweep on EB 5D Stage-A holdout (`t=0.5`) — completed
- Folder:
  - `outputs/2026-05-04/pot_emd2_runtime_sweep/17-05-06`
- What was tested:
  - Runtime sensitivity of POT `ot.emd2` to `numItermax` under plan-conditioned holdout-time W2 evaluation:
    - dataset: `TrajectoryNet/data/eb_velocity_v5.npz`
    - holdout index `2` (normalized `t=0.5`)
    - source measure: weighted global-OT support pushforward at `t=0.5`
    - target measure: empirical holdout pool at label `2` with uniform weights
    - precomputed one cost matrix (`5712 x 3278`) reused across runs
  - Calibration:
    - start `numItermax=100000`
    - target window `[45s, 75s]`
    - reached stable runtime plateau around `~3.2–3.3s` (target window not met)
  - Final sweep (`0.25x, 0.5x, 1x, 2x` around calibrated center):
    - `1600000`, `3200000`, `6400000`, `12800000`
- Key artifacts:
  - `emd2_runtime_sweep.json`
  - `emd2_runtime_sweep.tsv`
- Outcome:
  - First two calibration points hit iteration-limit warnings (`100000`, `200000`).
  - From `400000` onward, no iteration-limit warning and identical W2 (`1.46043712323063`) on this setup.
  - Script recommendation: `numItermax=1600000` (first stable adjacent pair without warnings).

### [2026-05-04] Single-cell Stage-A `ot_global` tuned full-grid benchmark (legacy empirical metrics, mean/std) — completed
- Folder root:
  - `outputs/2026-05-04/single_cell_eb_5d_stage_a_strict_leaveout_ot_global_empirical_tuned/11-22-04`
- What was tested:
  - Full strict-leaveout grid with tuned method overrides under `ot_global`:
    - holdouts `1,2,3`
    - seeds `3,7,11`
    - methods `constrained, metric, metric_alpha0, metric_constrained_al, metric_constrained_soft`
    - dataset `TrajectoryNet/data/eb_velocity_v5.npz`
    - `train.eval_full_ot_metrics=false` (legacy empirical only)
  - Tuned overrides applied:
    - constrained: `train.rho=35.0`, `train.beta=0.08`, `train.alpha=1.0`
    - metric_constrained_al: `mfm.alpha=0.3`, `mfm.moment_eta=3.0`, `train.rho=15.0`
    - metric_constrained_soft: `mfm.alpha=0.3`, `mfm.moment_eta=3.0`
- Key artifacts:
  - `benchmark_summary_stage_a_ot_global_empirical_tuned.json`
  - `leaderboard_stage_a_ot_global_empirical_tuned.tsv`
  - `tuned_vs_untuned_ot_global_diff.json`
  - `tuned_vs_untuned_ot_global_diff.tsv`
- Outcome:
  - `9/9` runs completed and aggregated.
  - Leaderboard preserves the same method ordering as untuned `ot_global`, with improved mean holdout score for `metric_constrained_al` and `constrained`.

### [2026-05-04] Single-cell Stage-A `ot_global` parameter sweep (legacy empirical metrics, holdout-2 seed-3) — completed
- Folder root:
  - `outputs/2026-05-04/single_cell_eb_5d_stage_a_ot_global_param_sweep_empirical_only/11-10-28`
- What was tested:
  - Focused Stage-A hyperparameter sweep under `data.coupling=ot_global` with legacy empirical interpolant metrics only (`train.eval_full_ot_metrics=false`).
  - Fixed scope:
    - holdout `2`, seed `3`, dataset `TrajectoryNet/data/eb_velocity_v5.npz`
    - protocol `strict_leaveout`
    - methods: `constrained`, `metric`, `metric_alpha0`, `metric_constrained_al`, `metric_constrained_soft`
  - Grid size:
    - constrained: `rho x beta = 3 x 3 = 9`
    - metric: `sigma = 3`
    - metric_alpha0: `1`
    - metric_constrained_al: `alpha x moment_eta = 3 x 3 = 9`
    - metric_constrained_soft: `alpha x moment_eta = 3 x 3 = 9`
    - total configs: `31`
- Key artifacts:
  - `sweep_summary.json`
  - `sweep_leaderboard.tsv`
  - `best_by_mode.tsv`
- Outcome:
  - Completed `31/31` configs with `comparison_mfm.json` for each.
  - Effective-parameter checks were recorded to confirm overrides were actually applied.

### [2026-05-04] Single-cell Stage-A `ot_global` parameter sweep (legacy empirical metrics) — interrupted invalid attempt
- Folder root:
  - `outputs/2026-05-04/single_cell_eb_5d_stage_a_ot_global_param_sweep_empirical_only/11-07-52`
- What happened:
  - Initial sweep launcher used `experiment.method_overrides.metric.*` for `metric` mode, which Hydra rejected because `metric` is not a predeclared key in that struct map.
  - Run was interrupted and replaced by corrected relaunch in `11-10-28`.
- Outcome:
  - Partial outputs exist but should be treated as invalid for sweep conclusions.

### [2026-05-04] Single-cell Stage-A coupling-only benchmark (`ot_global`, legacy empirical metrics) — completed
- Folder root:
  - `outputs/2026-05-04/single_cell_eb_5d_stage_a_strict_leaveout_ot_global_empirical_only/10-59-32`
- What was tested:
  - Reproduced the previous Stage-A interpolant pushforward benchmark scope with coupling-only change:
    - holdouts `1,2,3`
    - seeds `3,7,11`
    - methods `constrained, metric, metric_alpha0, metric_constrained_al, metric_constrained_soft`
    - dataset `TrajectoryNet/data/eb_velocity_v5.npz`
    - strict leaveout protocol with `data.constraint_time_policy=observed_nonendpoint_excluding_holdout`
  - Overrides for apples-to-apples metric comparability:
    - `data.coupling=ot_global`
    - `train.eval_full_ot_metrics=false`
    - `mfm.backend=auto`
    - `output.save_plots=false`
  - Baseline reference:
    - `outputs/2026-05-01/single_cell_eb_5d_stage_a_strict_leaveout/18-55-20/benchmark_summary_stage_a.json`
- Key artifacts:
  - `benchmark_summary_stage_a_ot_global_empirical_only.json`
  - `leaderboard_stage_a_ot_global_empirical_only.tsv`
  - `coupling_diff_stage_a_empirical.json`
  - `coupling_diff_stage_a_empirical.tsv`
- Outcome:
  - All `9/9` runs produced `comparison_mfm.json`.
  - Leaderboard order matched baseline (`metric_constrained_al` best by `learned_holdout_empirical_w2_mean`).

### [2026-05-03] Single-cell Stage-A `ot_global` + robust full-OT pilot (strict leaveout) — relaunched then canceled
- Folder:
  - `outputs/2026-05-03/single_cell_eb_5d_stage_a_strict_leaveout_ot_global_fullot/19-47-23/holdout_2/seed_3`
- What was tested:
  - Pilot benchmark run after successful standalone global OT cache precompute.
  - Configuration:
    - `experiment=comparison_mfm_single_cell_stage_a`
    - `train=single_cell_stage_a_only`
    - `data=single_cell_eb_5d`
    - `experiment.protocol=strict_leaveout`
    - `experiment.holdout_index=2`
    - `data.constraint_time_policy=observed_nonendpoint_excluding_holdout`
    - `data.single_cell.path=/Users/benpro/Documents/PHD/Projects/Neurips26/code/TrajectoryNet/data/eb_velocity_v5.npz`
    - `data.coupling=ot_global`
    - `seed=3`
    - `mfm.backend=auto`
    - `output.save_plots=false`
- Current status:
  - Canceled manually on request before completion.
  - No `comparison_mfm.json` generated for this attempt.

### [2026-05-03] Single-cell global OT cache build (standalone precompute for `ot_global`) — completed
- Folder:
  - `outputs/2026-05-03/ot_cache_build_single_cell_eb5d/19-18-17`
- What was tested:
  - Standalone precompute of the balanced global Kantorovich endpoint plan for EB 5D single-cell data so subsequent `data.coupling=ot_global` runs can reuse cache.
  - Configuration mirrors single-cell Stage-A benchmark data settings:
    - dataset: `TrajectoryNet/data/eb_velocity_v5.npz`
    - `coupling=ot_global`
    - cache dir: `.cache/ot_plans`
    - exact LP path (`global_ot_max_variables=null`)
    - preprocessing: `max_dim=5`, `whiten=true`
  - Launcher:
    - `PYTHONPATH=src python -u outputs/2026-05-03/ot_cache_build_single_cell_eb5d/19-18-17/build_ot_cache.py`
- Current status:
  - Completed successfully.
  - Cache artifact:
    - `.cache/ot_plans/f2055c582cae0bf05eb73ed6ed97fc225ff748e4c861e509a6c6b6ca5ab0355d.pt`
  - Reported summary:
    - `global_ot_cache_hit=false` (fresh build),
    - `global_ot_support_size=5712`,
    - `global_ot_total_cost=15.589441731606119`,
    - elapsed solve/prep time `1389.41s` (~23.2 minutes).

### [2026-05-03] Single-cell Stage-A `ot_global` + robust full-OT pilot (strict leaveout) — aborted
- Folder:
  - `outputs/2026-05-03/single_cell_eb_5d_stage_a_strict_leaveout_ot_global_fullot/17-57-09/holdout_2/seed_3`
- What was tested:
  - Pilot for exact full-set robust OT benchmark under:
    - `experiment=comparison_mfm_single_cell_stage_a`
    - `train=single_cell_stage_a_only`
    - `data=single_cell_eb_5d`
    - overrides:
      - `data.coupling=ot_global`
      - `data.single_cell.path=/Users/benpro/Documents/PHD/Projects/Neurips26/code/TrajectoryNet/data/eb_velocity_v5.npz`
      - `experiment.protocol=strict_leaveout`
      - `experiment.holdout_index=2`
      - `seed=3`
      - `mfm.backend=auto`
      - `output.save_plots=false`
- Outcome:
  - Run was manually stopped before completion due very long exact LP compute time in the global Kantorovich OT solve step on full endpoint pools.
  - No `comparison_mfm.json` was produced in this aborted attempt.

### [2026-05-01] Single-cell EB 5D Stage-A-only benchmark (strict leaveout, holdouts 1/2/3, seeds 3/7/11, 5-mode)
- Folder root:
  - `outputs/2026-05-01/single_cell_eb_5d_stage_a_strict_leaveout/18-55-20`
- What was tested:
  - Stage-A-only interpolant benchmark on real EB file:
    - `TrajectoryNet/data/eb_velocity_v5.npz`
  - Methods:
    - `constrained`
    - `metric`
    - `metric_alpha0`
    - `metric_constrained_al`
    - `metric_constrained_soft`
  - Grid:
    - holdouts `1,2,3` (normalized times `0.25,0.50,0.75`)
    - seeds `3,7,11`
    - total runs `9` (`3 holdouts x 3 seeds`)
  - Shared profile:
    - `experiment=comparison_mfm_single_cell_stage_a`
    - `train=single_cell_stage_a_only` (`stage_a_steps=120`, `stage_b_steps=0`, `stage_c_steps=0`)
    - `data=single_cell_eb_5d`
    - `protocol=strict_leaveout`
    - `data.constraint_time_policy=observed_nonendpoint_excluding_holdout`
  - Per-mode tuned overrides (from preset):
    - constrained: `train.alpha=1.0`, `train.beta=0.05`, `train.rho=25.0`
    - metric_alpha0: `mfm.alpha=0.0`
    - metric_constrained_al: `mfm.alpha=0.4`, `mfm.moment_eta=2.0`, `train.rho=15.0`
    - metric_constrained_soft: `mfm.alpha=0.4`, `mfm.moment_eta=2.0`
- Key aggregate artifacts:
  - `outputs/2026-05-01/single_cell_eb_5d_stage_a_strict_leaveout/18-55-20/benchmark_summary_stage_a.json`
  - `outputs/2026-05-01/single_cell_eb_5d_stage_a_strict_leaveout/18-55-20/leaderboard_stage_a.tsv`
- Command:
  - `python scripts/run_single_cell_eb_stage_a_benchmark.py --data-path TrajectoryNet/data/eb_velocity_v5.npz`

### [2026-04-30] Bridge constrained aggressive piecewise scheduler sanity run
- Folder:
  - `outputs/2026-04-30/bridge_beta_sched_piecewise_aggressive_ab_only/15-24-33`
- What was tested:
  - Single constrained scheduler stress test to amplify `beta(t)` variation versus the default schedule settings.
  - Settings changed from the standard seed-3 bridge run:
    - `train.beta_schedule=piecewise`
    - `train.beta_drift_p=3.0`
    - `train.beta_min_scale=0.1`
    - `train.beta_max_scale=5.0`
  - Shared profile:
    - `experiment=comparison`, `train=ab_only`, `data=bridge_ot`, `seed=3`
    - `train.stage_a_steps=300`, `train.stage_b_steps=300`, `train.stage_c_steps=0`
    - `train.batch_size=512`
    - `train.eval_intermediate_ot_samples=1024`
    - `train.eval_transport_samples=4000`

### [2026-04-30] Bridge constrained `beta_schedule` sanity runs (constant vs piecewise vs linear)
- Folders:
  - `outputs/2026-04-30/bridge_beta_sched_constant_ab_only/14-57-44`
  - `outputs/2026-04-30/bridge_beta_sched_piecewise_ab_only/14-59-40`
  - `outputs/2026-04-30/bridge_beta_sched_linear_ab_only/14-59-40`
- What was tested:
  - Direct constrained-mode comparison of new smoothness schedulers with matched budget and seed:
    - `beta_schedule=constant`
    - `beta_schedule=piecewise`
    - `beta_schedule=linear`
  - Shared run settings:
    - `experiment=comparison`, `train=ab_only`, `data=bridge_ot`
    - `seed=3`
    - `train.stage_a_steps=300`, `train.stage_b_steps=300`, `train.stage_c_steps=0`
    - `train.batch_size=512`
    - `train.eval_intermediate_ot_samples=1024`
    - `train.eval_transport_samples=4000`
    - constrained base hyperparameters:
      - `train.rho=0.5`, `train.alpha=1.0`, `train.lr_g=0.001`, `train.lr_v=0.001`, `train.beta=0.05`
  - Purpose:
    - verify the new schedule plumbing on real bridge runs, preserve legacy behavior at `constant`, and inspect endpoint/intermediate sensitivity under `piecewise` and `linear`.

### [2026-04-29] Bridge metric-constrained (AL + soft) balance sweep
- Folder root:
  - `outputs/bridge_mfm_constrained_hparam_sweep/2026-04-29_17-27-15`
- Key artifacts:
  - `phase1_runs.tsv`
  - `phase1_ranked_metric_constrained_al.tsv`
  - `phase1_ranked_metric_constrained_soft.tsv`
  - `final_report.json`
- What was tested:
  - Focused hyperparameter sweep for the two new fair-hybrid methods:
    - `metric_constrained_al`
    - `metric_constrained_soft`
  - Baseline reference run (seed 3) was executed once in:
    - `phase1/baseline`
  - Common training/eval budget:
    - `train.stage_a_steps=300`, `train.stage_b_steps=300`, `train.stage_c_steps=0`
    - `train.batch_size=512`
    - `train.eval_intermediate_ot_samples=1024`
    - `train.eval_transport_samples=4000`
    - `seed=3`
    - fair-information policy:
      - `mfm.reference_pool_policy=endpoints_only`
  - AL grid (16 configs):
    - `mfm.moment_eta ∈ {0.2, 0.5, 1.0, 2.0}`
    - `train.rho ∈ {0.5, 1.0}`
    - `mfm.sigma ∈ {0.0, 0.05}`
    - fixed `mfm.alpha=1.0`
  - Soft grid (16 configs):
    - `mfm.moment_eta ∈ {0.2, 0.5, 1.0, 2.0}`
    - `mfm.alpha ∈ {0.5, 1.0}`
    - `mfm.sigma ∈ {0.0, 0.05}`
    - fixed `train.rho=0.5`
  - Gate:
    - `delta_intermediate < 0`
    - `delta_endpoint <= +0.03`
  - Outcome:
    - `gate_pass_count = 0/16` for AL and `0/16` for soft.

### [2026-04-29] Top-config plot reruns from metric-constrained balance sweep
- Folder root:
  - `outputs/bridge_mfm_constrained_hparam_sweep/2026-04-29_17-27-15/top_plots`
- What was tested:
  - Re-ran baseline and best-ranked config per mode with `output.save_plots=true`.
  - Plot run folders:
    - `top_plots/baseline`
    - `top_plots/metric_constrained_al/best`
    - `top_plots/metric_constrained_soft/best`
  - Best-ranked configs selected by endpoint-first ranking under the non-gate regime:
    - AL: `eta0p5_rho1_s0p05`
    - soft: `eta2_a0p5_s0`

### [2026-04-29] Bridge fair-hybrid MFM 6-way sanity run (default constrained rho diagnostic)
- Folder:
  - `outputs/2026-04-29/bridge_mfm_hybrid_sanity_ab_only/15-44-34`
- What was tested:
  - First 6-way benchmark execution after adding:
    - `metric_constrained_al`
    - `metric_constrained_soft`
  - Methods:
    - `baseline`, `constrained`, `metric`, `metric_alpha0`, `metric_constrained_al`, `metric_constrained_soft`
  - Fairness settings:
    - `mfm.reference_pool_policy=endpoints_only`
    - `mfm.moment_eta=1.0`
  - Note:
    - used default `train.rho=5.0` for constrained (diagnostic only), then reran with best constrained settings for the canonical comparison.

### [2026-04-29] Bridge fair-hybrid MFM 6-way sanity run (best constrained settings)
- Folder:
  - `outputs/2026-04-29/bridge_mfm_hybrid_sanity_ab_only/16-00-17`
- What was tested:
  - Canonical 6-way bridge OT comparison with fairness-constrained MFM and best constrained hyperparameters:
    - methods: `baseline`, `constrained`, `metric`, `metric_alpha0`, `metric_constrained_al`, `metric_constrained_soft`
    - training/eval budget:
      - `train.stage_a_steps=300`, `train.stage_b_steps=300`, `train.stage_c_steps=0`
      - `train.batch_size=512`
      - `train.eval_intermediate_ot_samples=1024`
      - `train.eval_transport_samples=4000`
      - `seed=3`
    - constrained settings:
      - `train.rho=0.5`, `train.alpha=1.0`, `train.lr_g=0.001`, `train.lr_v=0.001`
    - MFM fairness settings:
      - `mfm.reference_pool_policy=endpoints_only`
      - `mfm.moment_eta=1.0`
  - Purpose:
    - verify end-to-end artifact contract (`comparison_mfm.json`, legacy `comparison.json`, plots) and compare endpoint/intermediate tradeoffs under equal intermediate-information access policy.

### [2026-04-29] Bridge MFM (alpha>0) seed-3 hyperparameter sweep
- Folder root:
  - `outputs/bridge_mfm_hparam_sweep/2026-04-29_12-01-21`
- Key artifacts:
  - `phase1_summary.tsv`
  - `final_report.json`
- What was tested:
  - Focused MFM-only tuning on bridge OT with nonzero alpha (`alpha ∈ {0.5, 1.0}`) and no `metric_alpha0` in the objective.
  - Comparison methods per run:
    - `baseline`, `metric`
  - Fixed bridge budget/settings (matching prior high-sample bridge comparisons):
    - `train.stage_a_steps=300`, `train.stage_b_steps=300`, `train.stage_c_steps=0`
    - `train.batch_size=512`
    - `train.eval_intermediate_ot_samples=1024`
    - `train.eval_transport_samples=4000`
    - `seed=3`
  - Swept MFM parameters (16 configs):
    - `mfm.alpha ∈ {0.5, 1.0}`
    - `mfm.sigma ∈ {0.0, 0.05}`
    - `mfm.land_gamma ∈ {0.08, 0.125}`
    - `mfm.land_rho ∈ {0.0005, 0.001}`
  - Ranking gate (same style as prior constrained sweep):
    - `delta_intermediate < 0`
    - `delta_endpoint <= +0.03`
  - Outcome:
    - no config passed the gate (`0/16`);
    - best-ranked config by the sweep rule:
      - `a1_s0p05_g0p125_r0p001`
      - `delta_intermediate=+0.00570`
      - `delta_endpoint=+0.13022`

### [2026-04-29] Bridge MFM top-from-sweep rerun with plots
- Folder:
  - `outputs/2026-04-29/bridge_mfm_top_from_sweep_plots_ab_only/12-23-57`
- What was tested:
  - Re-ran the top-ranked sweep config (`a1_s0p05_g0p125_r0p001`) with plotting enabled for visual inspection.
  - Methods:
    - `baseline`, `metric`
  - MFM settings:
    - `mfm.alpha=1.0`, `mfm.sigma=0.05`, `mfm.land_gamma=0.125`, `mfm.land_rho=0.001`, `mfm.land_metric_samples=512`
  - Plot outputs confirmed:
    - `baseline/sample_paths.png`
    - `metric/sample_paths.png`
    - plus rollout grids and per-time empirical W2 bars for both methods.

### [2026-04-29] Bridge best-preset 4-method MFM rerun with plotting enabled
- Folder:
  - `outputs/2026-04-29/bridge_mfm_best_from_sweep_plots_ab_only/11-17-07`
- What was tested:
  - Re-ran the same best-preset single-seed 4-method comparison (`baseline`, `constrained`, `metric`, `metric_alpha0`) with plotting enabled.
  - Command profile:
    - `experiment=comparison_mfm train=ab_only data=bridge_ot`
    - `experiment.label=bridge_mfm_best_from_sweep_plots`
    - best-preset hyperparameters (`rho=0.5`, `alpha=1.0`, `lr_g=0.001`, `lr_v=0.001`, `seed=3`)
    - high-sample evaluation budget (`batch_size=512`, `eval_intermediate_ot_samples=1024`, `eval_transport_samples=4000`)
    - `output.save_plots=true`
- Output checks:
  - Path visualizations are now present for all methods, including:
    - `metric/sample_paths.png`
    - `metric_alpha0/sample_paths.png`
  - `comparison_mfm.json` and legacy `comparison.json` were both written.

### [2026-04-28] Bridge best-preset 4-method MFM comparison (single-seed)
- Folder:
  - `outputs/2026-04-28/bridge_mfm_best_from_sweep_ab_only/22-42-21`
- What was tested:
  - New method-list comparison preset `experiment=comparison_mfm` with:
    - `baseline`, `constrained`, `metric`, `metric_alpha0`.
  - Bridge OT best-preset settings aligned with prior sweep winner:
    - `train.stage_a_steps=300`, `train.stage_b_steps=300`, `train.stage_c_steps=0`
    - `train.batch_size=512`
    - `train.eval_intermediate_ot_samples=1024`
    - `train.eval_transport_samples=4000`
    - `train.rho=0.5`, `train.alpha=1.0`, `train.lr_g=0.001`, `train.lr_v=0.001`
    - `seed=3`
  - MFM settings:
    - `mfm.backend=auto`
    - `mfm.alpha=1.0`, `mfm.sigma=0.1`
    - `mfm.land_gamma=0.125`, `mfm.land_rho=0.001`, `mfm.land_metric_samples=512`
- Output contract checks:
  - `comparison_mfm.json` contains all four method summaries and method-list metadata.
  - legacy `comparison.json` was also written with unchanged baseline/constrained schema.

### [2026-04-28] Best-parameter rerun from bridge A+B OT sweep
- Folder:
  - `outputs/2026-04-28/bridge_ab_ot_best_from_sweep_ab_only/12-24-22`
- What was tested:
  - Recovered the top phase-2 sweep configuration from
    - `outputs/bridge_ab_ot_hparam_sweep/2026-04-27_19-28-16/phase2_summary.tsv` (top row by the sweep ranking),
    - and matched the best seed-level run from
      `outputs/bridge_ab_ot_hparam_sweep/2026-04-27_19-28-16/phase2_runs.tsv`.
  - Re-executed bridge A+B comparison with those settings:
    - `experiment=comparison train=ab_only data=bridge_ot`
    - `seed=3`
    - `train.rho=0.5`
    - `train.alpha=1.0`
    - `train.lr_g=0.001`
    - `train.lr_v=0.001`
    - sweep-matched training/eval budget:
      - `train.stage_a_steps=300`, `train.stage_b_steps=300`, `train.stage_c_steps=0`
      - `train.batch_size=512`
      - `train.eval_intermediate_ot_samples=1024`
      - `train.eval_transport_samples=4000`
  - Outcome check:
    - constrained `intermediate_empirical_w2_avg=0.1923324056` vs baseline `0.3363516705`
    - constrained endpoint `transport_endpoint_empirical_w2=0.1160300875` vs baseline `0.1320245030`
    - reproduces the best-run behavior from the sweep tables for this config/seed.

### [2026-04-27] Bridge OT A+B high-sample two-phase hyperparameter sweep
- Folder root:
  - `outputs/bridge_ab_ot_hparam_sweep/2026-04-27_19-28-16`
- Key artifacts:
  - `phase1_summary.tsv`
  - `phase2_runs.tsv`
  - `phase2_summary.tsv`
  - `phase2_selected.json`
  - `final_report.json`
- What was tested:
  - `experiment=comparison train=ab_only data=bridge_ot` with increased data usage:
    - `train.batch_size=512`
    - `train.eval_intermediate_ot_samples=1024`
    - `train.eval_transport_samples=4000`
    - `train.stage_a_steps=300`, `train.stage_b_steps=300`, `stage_c_steps=0`
  - Phase 1: full 16-config grid over
    - `rho ∈ {0.5, 1.0}`
    - `alpha ∈ {1.0, 2.0}`
    - `lr_g ∈ {3e-4, 1e-3}`
    - `lr_v ∈ {5e-4, 1e-3}`
    - seed `7`
  - Phase 2: top-4 configs from phase 1 rerun over seeds `{3,7,11}`.
  - Gate used:
    - `delta_intermediate < 0`
    - `delta_endpoint <= +0.03`
  - Final report:
    - `success_primary=true`
    - `success_stretch=false`

### [2026-04-27] Bridge A+B with Stage-A-only hyperparameters (OT coupling)
- Folder:
  - `outputs/2026-04-27/bridge_ab_match_stagea_ab_only/18-55-22`
- What was tested:
  - Ran A+B bridge comparison while matching Stage-A settings to the strong Stage-A-only run:
    - `train.stage_a_steps=300`
    - `train.rho=1.0`
  - Command:
    - `python3 scripts/run_experiment.py experiment=comparison train=ab_only data=bridge_ot experiment.label=bridge_ab_match_stagea train.stage_a_steps=300 train.rho=1.0`
  - Purpose:
    - isolate whether previous A+B degradation was mainly due to different Stage-A hyperparameters rather than Stage-B itself.

### [2026-04-27] Stage-B path-freeze sanity check (debug)
- Folder root:
  - `outputs/2026-04-27/debug_stageb_effect`
- Variants:
  - `a_only` (Stage A only: `stage_a_steps=12`, `stage_b_steps=0`)
  - `ab` (same Stage A settings + Stage B: `stage_b_steps=20`)
- What was tested:
  - Controlled check that Stage B does not update `g_\theta` when Stage-A settings are identical.
  - Result: path-parameter difference between the two checkpoints was exactly zero (`max_abs_diff=0.0`, `l2_diff=0.0`), confirming Stage B leaves interpolant parameters unchanged.

### [2026-04-27] Bridge A+B comparison enablement run (OT coupling, rescaled endpoint)
- Folder:
  - `outputs/2026-04-27/bridge_ab_ot_ab_only/18-35-42`
- What was tested:
  - First official bridge velocity-enabled comparison run after lifting the Stage-A-only bridge restriction.
  - Command: `python3 scripts/run_experiment.py experiment=comparison train=ab_only data=bridge_ot experiment.label=bridge_ab_ot`.
  - Bridge metric contract check in `comparison.json`:
    - constrained-time rollout empirical OT: `intermediate_empirical_w2`, `intermediate_empirical_w2_avg`;
    - endpoint rollout empirical OT: `transport_endpoint_empirical_w2`;
    - bridge transport score semantics: `transport_score == transport_endpoint_empirical_w2`.
  - Verified per-mode rollout diagnostic plots were written:
    - `baseline/rollout_marginal_grid.png`, `baseline/rollout_empirical_w2.png`;
    - `constrained/rollout_marginal_grid.png`, `constrained/rollout_empirical_w2.png`.

### [2026-04-27] Bridge endpoint-rescaled Stage-A-only runs (`t_norm=1.0 -> t_phys=1.5`)
- Folders:
  - `outputs/2026-04-27/bridge_stagea_rescaled_ot_stage_a_only/17-00-24`
  - `outputs/2026-04-27/bridge_stagea_rescaled_random_stage_a_only/17-01-22`
- What was tested:
  - Same Stage-A-only constrained setup with normalized-time semantics and physical horizon rescaling (`bridge.total_time=1.5`).
  - Constraint times remained normalized (`0.25, 0.50, 0.75`) and were mapped internally to physical times (`0.375, 0.75, 1.125`).
  - Compared OT vs random coupling under the rescaled endpoint.

### [2026-04-27] Bridge Stage-A-only interpolant evaluation (OT vs random coupling)
- Folders:
  - `outputs/2026-04-27/bridge_stagea_bridge_ot_stage_a_only/15-15-56`
  - `outputs/2026-04-27/bridge_stagea_bridge_random_stage_a_only/15-17-24`
- What was tested:
  - Constrained mode with `train=stage_a_only` (Stage A enabled, Stage B/C disabled).
  - Bridge data family with cached high-precision target marginals (`target_mc_samples=200000`).
  - Interpolant-only evaluation: linear vs learned empirical OT at `t={0.25,0.50,0.75}`.
  - Same seed and training budget across OT and random coupling for clean comparison.

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

### [2026-04-30] Single-cell synthetic smoke run (strict leaveout, 6-way)
- Folder:
  - `outputs/2026-04-30/single_cell_eb5d_synth_smoke/00-00-01`
- What was tested:
  - End-to-end `family=single_cell` integration on a synthetic EB-like `.npz` dataset (`.cache/synth_eb5d.npz`).
  - Strict leaveout protocol (`holdout_index=2`) with full 6-way comparison:
    - `baseline`, `constrained`, `metric`, `metric_alpha0`, `metric_constrained_al`, `metric_constrained_soft`.
  - Verification targets:
    - `comparison_mfm.json` written with protocol/holdout metadata and holdout metrics,
    - per-mode projected artifacts generated for \(d>2\): `sample_paths_proj12.png`, `rollout_marginal_grid_proj12.png`.
  - Command:
    - `python scripts/run_experiment.py experiment=comparison_mfm_single_cell train=single_cell_ab_only data=single_cell_eb_5d experiment.label=single_cell_eb5d_synth_smoke experiment.protocol=strict_leaveout experiment.holdout_index=2 data.constraint_time_policy=observed_nonendpoint_excluding_holdout data.single_cell.path=/Users/benpro/Documents/PHD/Projects/Neurips26/code/.cache/synth_eb5d.npz train.stage_a_steps=2 train.stage_b_steps=3 train.stage_c_steps=0 train.batch_size=24 train.eval_batch_size=64 train.eval_transport_samples=96 train.eval_intermediate_ot_samples=48 train.eval_transport_steps=12 output.save_plots=true output.plot_pairs=8 seed=11 hydra.run.dir=outputs/2026-04-30/single_cell_eb5d_synth_smoke/00-00-01`

### [2026-04-30] Single-cell EB 5D benchmark run (strict leaveout, holdouts 1/2/3, 6-way)
- Folder root:
  - `outputs/2026-04-30/single_cell_eb_5d_strict_leaveout/22-25-27`
- Holdout folders:
  - `outputs/2026-04-30/single_cell_eb_5d_strict_leaveout/22-25-27/holdout_1`
  - `outputs/2026-04-30/single_cell_eb_5d_strict_leaveout/22-25-27/holdout_2`
  - `outputs/2026-04-30/single_cell_eb_5d_strict_leaveout/22-25-27/holdout_3`
- What was tested:
  - Real EB single-cell dataset benchmark using TrajectoryNet file:
    - `/Users/benpro/Documents/PHD/Projects/Neurips26/code/TrajectoryNet/data/eb_velocity_v5.npz`
  - Strict leaveout protocol over all middle timepoints (`holdout_indices=[1,2,3]`) with full 6-way comparison:
    - `baseline`, `constrained`, `metric`, `metric_alpha0`, `metric_constrained_al`, `metric_constrained_soft`.
  - Wrote per-holdout `comparison_mfm.json` files and top-level aggregate:
    - `outputs/2026-04-30/single_cell_eb_5d_strict_leaveout/22-25-27/benchmark_summary.json`
- Command:
  - `python scripts/run_single_cell_eb_benchmark.py --data-path /Users/benpro/Documents/PHD/Projects/Neurips26/code/TrajectoryNet/data/eb_velocity_v5.npz --holdouts 1 2 3 --seed 42 --mfm-backend auto`

### [2026-04-30] Single-cell constrained-metric coarse sweep (holdout 2, reduced budget)
- Folder root:
  - `outputs/2026-04-30/single_cell_metric_constrained_tuning/22-43-49`
- Variants:
  - `metric_constrained_al_a{0.4,0.7,1.0}_eta{0.25,0.5,1.0,2.0}_rho{1.0,5.0,15.0}`
  - `metric_constrained_soft_a{0.4,0.7,1.0}_eta{0.25,0.5,1.0,2.0}`
- What was tested:
  - Focused tuning of constrained-metric tradeoff parameters:
    - `mfm.alpha`, `mfm.moment_eta`, and `train.rho` (AL only).
  - Strict leaveout protocol on `holdout_index=2` with reduced training budget (`stage_a_steps=80`, `stage_b_steps=120`) to rank candidates quickly.
  - Wrote ranking artifact:
    - `outputs/2026-04-30/single_cell_metric_constrained_tuning/22-43-49/summary_ranked.json`.
- Command family:
  - Repeated `python scripts/run_experiment.py ... experiment.comparison_methods=[metric_constrained_al|metric_constrained_soft] ...` over the parameter grid above.

### [2026-04-30] Single-cell constrained-metric tuned full-budget validation (strict leaveout, holdouts 1/2/3)
- Folder roots:
  - `outputs/2026-04-30/single_cell_metric_constrained_tuned_validation/22-49-11/al_alpha0.4_eta2.0_rho15`
  - `outputs/2026-04-30/single_cell_metric_constrained_tuned_validation/22-49-39/soft_alpha0.4_eta2.0`
- Holdout folders:
  - each root contains `holdout_1`, `holdout_2`, `holdout_3` with per-holdout `comparison_mfm.json` and plots.
- What was tested:
  - Full-budget strict-leaveout validation of top sweep candidates:
    - AL tuned: `mfm.alpha=0.4`, `mfm.moment_eta=2.0`, `train.rho=15.0`,
    - Soft tuned: `mfm.alpha=0.4`, `mfm.moment_eta=2.0`.
  - Produced plot artifacts for each holdout (`sample_paths_proj12.png`, `rollout_marginal_grid_proj12.png`, `rollout_empirical_w2.png`, etc.).
  - Wrote aggregate comparison helper:
    - `outputs/2026-04-30/single_cell_metric_constrained_tuned_validation/summary_compare_default_vs_tuned.json`.

### [2026-04-30] Single-cell constrained-only tuning (strict leaveout, holdout 2)
- Coarse sweep root (reduced budget):
  - `outputs/2026-04-30/single_cell_constrained_tuning/22-53-40`
- Variants:
  - `constrained_a{0.5,1.0,2.0}_b{0.01,0.05}_rho{5.0,15.0}_eta{0.02,0.10}`
- What was tested:
  - Initial constrained-only parameter scan for `train.alpha`, `train.beta`, `train.rho` (and `eta_joint`) on strict leaveout holdout 2 with reduced budget (`stage_a_steps=80`, `stage_b_steps=120`).
  - Used to identify useful parameter region before full-budget rerun.

### [2026-04-30] Single-cell constrained-only full-budget sweep (strict leaveout, holdout 2)
- Sweep root:
  - `outputs/2026-04-30/single_cell_constrained_tuning_full/22-56-45`
- Variants:
  - `constrained_a{0.5,1.0}_b{0.01,0.05}_rho{10.0,15.0,25.0}`
- What was tested:
  - Full-budget constrained-only tuning over the key active tradeoff parameters (`alpha`, `beta`, `rho`) on holdout 2.
  - Wrote ranked summary:
    - `outputs/2026-04-30/single_cell_constrained_tuning_full/22-56-45/summary_ranked.json`.

### [2026-04-30] Single-cell constrained-only tuned validation (strict leaveout, holdouts 1/2/3)
- Folder root:
  - `outputs/2026-04-30/single_cell_constrained_tuned_validation/22-58-39/constrained_a1.0_b0.05_rho25`
- Holdout folders:
  - `outputs/2026-04-30/single_cell_constrained_tuned_validation/22-58-39/constrained_a1.0_b0.05_rho25/holdout_1`
  - `outputs/2026-04-30/single_cell_constrained_tuned_validation/22-58-39/constrained_a1.0_b0.05_rho25/holdout_2`
  - `outputs/2026-04-30/single_cell_constrained_tuned_validation/22-58-39/constrained_a1.0_b0.05_rho25/holdout_3`
- What was tested:
  - Full strict-leaveout validation of best constrained-only setting from holdout-2 sweep.
  - Generated per-holdout plots (`sample_paths_proj12.png`, `rollout_marginal_grid_proj12.png`, `rollout_empirical_w2.png`, etc.).
  - Wrote aggregate before/after helper:
    - `outputs/2026-04-30/single_cell_constrained_tuned_validation/summary_compare_default_vs_tuned.json`.

### [2026-05-04] Single-cell pseudo-constraint Stage-A-only smoke run (strict leaveout)
- Folder:
  - `outputs/2026-05-04/single_cell_pseudo_stagea_smoke/00-00-01`
- What was tested:
  - Pseudo-label-enabled constrained mode with Stage A only (`stage_b_steps=0`, `stage_c_steps=0`).
  - Real EB dataset path:
    - `/Users/benpro/Documents/PHD/Projects/Neurips26/code/TrajectoryNet/data/eb_velocity_v5.npz`
  - Verified pseudo summary keys are present:
    - `pseudo_constraint_residual_norms`,
    - `pseudo_constraint_residual_avg`,
    - `pseudo_labels_k`,
    - `bic_by_k`, `stability_by_k`.
- Command:
  - `python scripts/run_experiment.py experiment=comparison_mfm_single_cell_stage_a train=single_cell_stage_a_only data=single_cell_eb_5d experiment.label=single_cell_pseudo_stagea_smoke experiment.protocol=strict_leaveout experiment.holdout_index=2 'experiment.comparison_methods=[constrained]' data.constraint_time_policy=observed_nonendpoint_excluding_holdout data.single_cell.path=/Users/benpro/Documents/PHD/Projects/Neurips26/code/TrajectoryNet/data/eb_velocity_v5.npz data.single_cell.pseudo_labels.enabled=true train.stage_a_steps=2 train.stage_b_steps=0 train.stage_c_steps=0 train.batch_size=64 train.eval_batch_size=128 train.eval_intermediate_ot_samples=64 train.eval_full_ot_metrics=false train.pseudo_eta=1.0 train.pseudo_rho=5.0 train.pseudo_lambda_clip=100.0 output.save_plots=false hydra.run.dir=outputs/2026-05-04/single_cell_pseudo_stagea_smoke/00-00-01`

### [2026-05-04] Single-cell pseudo-constraint A+B smoke run (strict leaveout)
- Folder:
  - `outputs/2026-05-04/single_cell_pseudo_ab_smoke/00-00-01`
- What was tested:
  - Pseudo-label-enabled constrained mode with A+B schedule (`stage_c_steps=0`) on real EB data.
  - Verified pseudo constraints coexist with rollout/holdout transport metrics and legacy summary schema.
- Command:
  - `python scripts/run_experiment.py experiment=comparison_mfm_single_cell train=single_cell_ab_only data=single_cell_eb_5d experiment.label=single_cell_pseudo_ab_smoke experiment.protocol=strict_leaveout experiment.holdout_index=2 'experiment.comparison_methods=[constrained]' data.constraint_time_policy=observed_nonendpoint_excluding_holdout data.single_cell.path=/Users/benpro/Documents/PHD/Projects/Neurips26/code/TrajectoryNet/data/eb_velocity_v5.npz data.single_cell.pseudo_labels.enabled=true train.stage_a_steps=2 train.stage_b_steps=3 train.stage_c_steps=0 train.batch_size=64 train.eval_batch_size=128 train.eval_transport_samples=256 train.eval_transport_steps=20 train.eval_intermediate_ot_samples=128 train.eval_full_ot_metrics=false train.pseudo_eta=1.0 train.pseudo_rho=5.0 train.pseudo_lambda_clip=100.0 output.save_plots=false hydra.run.dir=outputs/2026-05-04/single_cell_pseudo_ab_smoke/00-00-01`

### [2026-05-04] Stage-A `metric_constrained_al` pseudo sweep (full-OT, holdout 2, 3 seeds)
- Baseline reference (simple constraints, pseudo disabled):
  - `outputs/2026-05-04/single_cell_eb_stage_a_ot_global_pot_holdout3_seed3/17-43-31`
- Sweep root:
  - `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_stage_a_fullot/18-24-04`
- Sweep scope:
  - mode: `metric_constrained_al` only.
  - holdout: `2`; seeds: `3, 7, 11`.
  - full-OT settings forced:
    - `data.coupling=ot_global`,
    - `train.eval_full_ot_metrics=true`,
    - `train.eval_full_ot_method=pot_emd2`,
    - `train.eval_full_ot_num_itermax=1600000`.
  - pseudo enabled:
    - `data.single_cell.pseudo_labels.enabled=true`.
  - pseudo grid:
    - `(pseudo_eta, pseudo_rho, pseudo_lambda_clip) in`
      - `(0.10, 1.0, 100.0)`,
      - `(0.25, 1.0, 100.0)`,
      - `(0.50, 1.0, 100.0)`,
      - `(0.25, 5.0, 100.0)`,
      - `(0.50, 5.0, 100.0)`,
      - `(1.00, 5.0, 100.0)`.
- Artifacts:
  - search summary:
    - `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_stage_a_fullot/18-24-04/phase2_search_summary.json`.
  - per-run outputs under:
    - `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_stage_a_fullot/18-24-04/holdout_2/<config>/seed_<seed>/`.
- Outcome:
  - all 18/18 runs succeeded, all runs had `pseudo_constraints_active=true`, `pseudo_labels_k=9`, and cache hits after warm start (`cache_hit_rate=1.0` in summaries).
  - best holdout-2 aggregate by primary metric:
    - `eta1.00_rho5.0_clip100.0` with mean `learned_full_ot_w2_avg=1.0553`.

### [2026-05-04] Stage-A `metric_constrained_al` pseudo final validation (full-OT, all holdouts)
- Validation root:
  - `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_stage_a_fullot/18-24-04/final_validation`
- Candidates validated:
  - `eta1.00_rho5.0_clip100.0`
  - `eta0.50_rho5.0_clip100.0`
- Validation scope:
  - holdouts: `1, 2, 3`; seeds: `3, 7, 11` (9 runs per config).
- Artifacts:
  - final summary:
    - `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_stage_a_fullot/18-24-04/final_validation_summary.json`.
  - baseline-vs-final delta summary:
    - `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_stage_a_fullot/18-24-04/final_vs_baseline_summary.json`.
- Outcome:
  - winner by primary metric (`learned_full_ot_w2_avg`): `eta1.00_rho5.0_clip100.0`.
  - winner aggregate (`n=9`):
    - primary mean: `1.129247562864637`,
    - secondary mean (`learned_holdout_full_ot_w2`): `1.1573922965458503`.
  - vs simple-constraint baseline (`n=9`, primary `1.1340704282917455`, secondary `1.1598324896315004`):
    - primary delta: `-0.004822865427108525`,
    - secondary delta: `-0.002440193085650133`.

### [2026-05-04] Stage-A `constrained` pseudo evaluation (full-OT, all holdouts, no sweep)
- Run root:
  - `outputs/2026-05-04/single_cell_constrained_pseudo_stage_a_fullot/18-54-15`
- Baseline reference:
  - `outputs/2026-05-04/single_cell_eb_stage_a_ot_global_pot_holdout3_seed3/17-43-31`
- Scope:
  - mode: `constrained` only.
  - holdouts: `1,2,3`; seeds: `3,7,11` (`9` runs).
  - pseudo settings:
    - `train.pseudo_eta=1.0`,
    - `train.pseudo_rho=5.0`,
    - `train.pseudo_lambda_clip=100.0`.
  - full-OT settings:
    - `data.coupling=ot_global`,
    - `train.eval_full_ot_metrics=true`,
    - `train.eval_full_ot_method=pot_emd2`,
    - `train.eval_full_ot_num_itermax=1600000`.
- Artifacts:
  - run summary:
    - `outputs/2026-05-04/single_cell_constrained_pseudo_stage_a_fullot/18-54-15/summary.json`
  - baseline-vs-unsup comparison:
    - `outputs/2026-05-04/single_cell_constrained_pseudo_stage_a_fullot/18-54-15/baseline_vs_unsup_summary.json`
- Outcome (`learned_holdout_full_ot_w2`, lower is better):
  - baseline overall mean: `1.4286208909355773`.
  - constrained+unsup overall mean: `1.4053413782439794`.
  - delta: `-0.023279512691597892`.

### [2026-05-04] Stage-A pseudo extension to remaining methods (`metric`, `metric_alpha0`, `metric_constrained_soft`)
- Run root:
  - `outputs/2026-05-04/single_cell_all_methods_pseudo_stage_a_fullot/19-03-38`
- Scope:
  - methods: `metric`, `metric_alpha0`, `metric_constrained_soft`.
  - holdouts: `1,2,3`; seeds: `3,7,11` (`27` runs total).
  - full-OT settings:
    - `data.coupling=ot_global`,
    - `train.eval_full_ot_metrics=true`,
    - `train.eval_full_ot_method=pot_emd2`,
    - `train.eval_full_ot_num_itermax=1600000`.
  - pseudo settings:
    - `data.single_cell.pseudo_labels.enabled=true`,
    - `train.pseudo_eta=1.0`,
    - `train.pseudo_rho=5.0`,
    - `train.pseudo_lambda_clip=100.0`.
- Artifacts:
  - run summary:
    - `outputs/2026-05-04/single_cell_all_methods_pseudo_stage_a_fullot/19-03-38/summary.json`
  - full baseline-vs-unsup table artifact (all methods):
    - `outputs/2026-05-04/single_cell_all_methods_pseudo_stage_a_fullot/19-03-38/full_table_baseline_vs_unsup.json`
    - `outputs/2026-05-04/single_cell_all_methods_pseudo_stage_a_fullot/19-03-38/full_table_baseline_vs_unsup.csv`
    - `outputs/2026-05-04/single_cell_all_methods_pseudo_stage_a_fullot/19-03-38/full_table_baseline_vs_unsup.md`
- Outcome notes:
  - all `27/27` runs succeeded (`n_failures=0`).
  - `metric_constrained_soft`: pseudo active in all runs (`pseudo_constraints_active=true`).
  - `metric` and `metric_alpha0`: pseudo inactive by design in these families (`pseudo_constraints_active=false`), so `+unsup` rows are effectively unchanged from baseline.

### [2026-05-04] Stage-A `metric_constrained_al` fixed-`K` pseudo sensitivity sweep (full-OT, K x eta)
- Run root:
  - `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_k_sensitivity_stage_a_fullot/19-56-10`
- Baseline references:
  - prior auto-`K` pseudo winner:
    - `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_stage_a_fullot/18-24-04/final_validation_summary.json`
  - simple-constraint baseline:
    - `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_stage_a_fullot/18-24-04/final_vs_baseline_summary.json`
- Search scope:
  - mode: `metric_constrained_al` only.
  - holdout: `2`; seeds: `3,7,11`.
  - fixed `K` grid via `k_min=k_max=K`:
    - `K in {4,6,8,9,10,12}`.
  - pseudo-strength grid:
    - `pseudo_eta in {0.5,1.0}`.
    - `pseudo_rho=5.0`, `pseudo_lambda_clip=100.0`.
  - search volume: `36` runs total.
- Final validation scope:
  - top-2 search configs on holdouts `1,2,3` with seeds `3,7,11`.
  - validation volume: `18` runs total.
- Artifacts:
  - search summary:
    - `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_k_sensitivity_stage_a_fullot/19-56-10/search_summary.json`
  - final validation summary:
    - `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_k_sensitivity_stage_a_fullot/19-56-10/final_validation_summary.json`
  - comparison vs prior auto-`K` winner and simple baseline:
    - `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_k_sensitivity_stage_a_fullot/19-56-10/new_best_vs_previous_and_simple_summary.json`
  - p0.25/p0.50/p0.75 + overall comparison table:
    - `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_k_sensitivity_stage_a_fullot/19-56-10/auto_k9_vs_best_fixed_k_table.md`
- Outcome:
  - all `54/54` runs succeeded (`n_failures=0`).
  - search top-2:
    - `k8_eta1.00_rho5.0_clip100.0`,
    - `k4_eta1.00_rho5.0_clip100.0`.
  - final winner:
    - `k8_eta1.00_rho5.0_clip100.0` with
      - `learned_full_ot_w2_avg` mean `1.1290412290211447`,
      - `learned_holdout_full_ot_w2` mean `1.1571221975204848`.
  - deltas:
    - vs prior auto-`K=9` winner: primary `-0.00020633384349211248`, holdout `-0.0002700990253654556`.
    - vs simple baseline: primary `-0.005029199270600859`, holdout `-0.0027102921110155886`.

### [2026-05-04] Stage-A `metric_constrained_al` fixed-`K=9` high-eta dichotomy sweep (full-OT, up to eta=100)
- Run root:
  - `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_k9_eta_dichotomy_stage_a_fullot/20-43-25`
- Scope:
  - mode: `metric_constrained_al` only.
  - strict leaveout Stage-A full-OT protocol, holdout `2`, seeds `3,7,11`.
  - fixed pseudo-label classes via:
    - `data.single_cell.pseudo_labels.k_min=9`,
    - `data.single_cell.pseudo_labels.k_max=9`.
  - exponential/dichotomy eta ladder:
    - `train.pseudo_eta in {1,2,4,8,16,32,64,100}`.
  - fixed pseudo AL knobs:
    - `train.pseudo_rho=5.0`,
    - `train.pseudo_lambda_clip=100.0`.
  - full-OT settings:
    - `data.coupling=ot_global`,
    - `train.eval_full_ot_metrics=true`,
    - `train.eval_full_ot_method=pot_emd2`,
    - `train.eval_full_ot_num_itermax=1600000`.
- Volume:
  - `24` runs (`8` eta values x `3` seeds), `0` failures.
- Artifacts:
  - sweep summary:
    - `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_k9_eta_dichotomy_stage_a_fullot/20-43-25/search_summary.json`
  - ranked table:
    - `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_k9_eta_dichotomy_stage_a_fullot/20-43-25/search_summary_table.md`
  - eta-ordered table with deltas vs eta=1:
    - `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_k9_eta_dichotomy_stage_a_fullot/20-43-25/eta_ordered_with_deltas.md`
- Outcome (means over seeds, lower is better):
  - `eta=1`: `learned_full_ot_w2_avg=1.055324`, `learned_holdout_full_ot_w2=1.363583`.
  - `eta=100`: `learned_full_ot_w2_avg=0.906111`, `learned_holdout_full_ot_w2=1.200209`.
  - best within this holdout-2 sweep:
    - `eta=100` (primary and secondary).
  - monotonic trend on this grid:
    - larger eta consistently improved both primary and secondary means from `1 -> 100`.

### [2026-05-04] Stage-A `metric_constrained_al` fixed-`K=9` continuation sweep (auto-stop when gain stops)
- Previous anchor:
  - `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_k9_eta_dichotomy_stage_a_fullot/20-43-25` (best prior eta was `100`).
- Continuation run root:
  - `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_k9_eta_dichotomy_stage_a_fullot/21-01-36`
- Scope:
  - mode: `metric_constrained_al` only.
  - strict leaveout Stage-A full-OT protocol.
  - fixed pseudo classes (`k_min=k_max=9`), holdout `2`, seeds `3,7,11`.
  - candidate eta ladder after anchor:
    - `128, 160, 200, 256, 320, 400, ...` (auto-stop enabled).
  - stop criterion:
    - stop when improvement in mean `learned_full_ot_w2_avg` vs previous eta is `<= 5e-4`.
- Volume and status:
  - executed `18` new runs (`6` eta levels x `3` seeds), `0` failures.
- Artifacts:
  - continuation summary (with stop reason and ranked combined table):
    - `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_k9_eta_dichotomy_stage_a_fullot/21-01-36/continuation_summary.json`
  - combined eta table:
    - `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_k9_eta_dichotomy_stage_a_fullot/21-01-36/combined_eta_table.md`
  - combined ranked table:
    - `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_k9_eta_dichotomy_stage_a_fullot/21-01-36/combined_ranked_table.md`
- Outcome:
  - improved from `eta=100` (`0.906111`) through `eta=320` (`0.826541`).
  - first non-improving point occurred at `eta=400` (`0.833223`), triggering stop.
  - best observed eta in the extended range:
    - `eta=320` with
      - `learned_full_ot_w2_avg=0.826541`,
      - `learned_holdout_full_ot_w2=1.085330`.

### [2026-05-04] Stage-A 0.5-only pseudo-constraint protocol (`constrained`, `metric_constrained_al`, `metric_constrained_soft`)
- Run root:
  - `outputs/2026-05-04/single_cell_t05_only_pseudo_stage_a_fullot/21-39-57`
- Scope:
  - `data=single_cell_eb_5d`, `experiment.protocol=no_leaveout`, `data.coupling=ot_global`.
  - Stage-A only with methods:
    - `constrained`,
    - `metric_constrained_al`,
    - `metric_constrained_soft`.
  - seeds: `3,7,11`.
  - pseudo settings:
    - `data.single_cell.pseudo_labels.enabled=true`,
    - `data.single_cell.pseudo_labels.fit_times_normalized=[0.5]` (GMM fit only on \(t=0.5\) marginal),
    - `train.pseudo_eta=320`, `train.pseudo_rho=5.0`, `train.pseudo_lambda_clip=100.0`.
  - time overrides:
    - `data.single_cell.constraint_times_normalized=[0.5]` (moment + pseudo constraints only at \(t=0.5\)),
    - `data.single_cell.eval_times_normalized=[0.25,0.5,0.75]` (interpolant metrics on all three snapshots).
  - full-OT settings:
    - `train.eval_full_ot_metrics=true`,
    - `train.eval_full_ot_method=pot_emd2`,
    - `train.eval_full_ot_num_itermax=1600000`.
- Artifacts:
  - run summary:
    - `outputs/2026-05-04/single_cell_t05_only_pseudo_stage_a_fullot/21-39-57/summary.json`
  - summary table:
    - `outputs/2026-05-04/single_cell_t05_only_pseudo_stage_a_fullot/21-39-57/summary_table.md`
- Outcome (mean ± std over seeds):
  - `constrained`: overall `1.142474 ± 0.114557` (`K=[6]`, pseudo active `3/3`).
  - `metric_constrained_al`: overall `0.878618 ± 0.007908` (`K=[6]`, pseudo active `3/3`).
  - `metric_constrained_soft`: overall `1.222570 ± 0.007806` (`K=[6]`, pseudo active `3/3`).
  - all runs confirmed the intended protocol metadata:
    - `single_cell_constraint_times=[0.5]`,
    - `single_cell_eval_times=[0.25,0.5,0.75]`,
    - `single_cell_pseudo_fit_times=[0.5]`.
