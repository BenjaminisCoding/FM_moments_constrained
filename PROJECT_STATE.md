# PROJECT_STATE.md

## Purpose
Append-only hybrid project log for implementation history, decisions, risks, and next steps.
Use this file to preserve continuity across sessions.

## Repository Boundary
`FM/` is environment tooling (Python virtual environment), not project source architecture.
Do not log `FM/` internals as product architecture work.

## Logging Rules
- Add a new dated entry for each meaningful change session.
- Keep older entries unchanged; append new entries at the top under "Entries".
- For small non-architectural edits, include a short note and state why architecture was unchanged.
- Include clear next actions to maintain momentum.

## Entry Template
```md
## [YYYY-MM-DD] <short title>
- What changed:
- Why (decision/rationale):
- Impact or risk:
- Architecture updates:
- Validation status:
- Next steps:
```

## Entries
## [2026-05-04] Ran 3x3 Stage-A `ot_global` + POT full-OT holdout benchmark and summarized constrained metrics
- What changed:
  - Executed a strict-leaveout Stage-A benchmark grid across:
    - holdouts `1,2,3` (times `0.25,0.50,0.75`)
    - seeds `3,7,11`
  - Configuration:
    - `experiment=comparison_mfm_single_cell_stage_a`
    - `train=single_cell_stage_a_only`
    - `data=single_cell_eb_5d`
    - `data.coupling=ot_global`
    - `train.eval_full_ot_method=pot_emd2`
    - `train.eval_full_ot_num_itermax=1600000`
    - `output.save_plots=false`
  - Wrote run artifacts under:
    - `outputs/2026-05-04/single_cell_eb_stage_a_ot_global_pot_holdout3_seed3/17-43-31`
  - Extracted constrained-mode holdout metric summary:
    - `holdout3_seed3_summary_constrained.json`
- Why (decision/rationale):
  - User requested per-seed holdout values for `p0.25`, `p0.5`, `p0.75` and corresponding means.
- Impact or risk:
  - Produced a direct 9-point estimate set for constrained `learned_holdout_full_ot_w2` under the current default POT full-OT path.
  - Observed higher variability at `p0.25` versus `p0.50`/`p0.75`.
- Architecture updates:
  - None; experiment execution and reporting only.
- Validation status:
  - All 9 runs completed with `comparison_mfm.json` present.
  - Aggregated constrained summary successfully written and parsed.
- Next steps:
  1. If desired, produce the same per-seed/holdout table for all methods (not only constrained).
  2. Optionally add this 3x3 run as a reusable benchmark script preset.

## [2026-05-04] Switched full-OT metric default to POT and ran Stage-A `ot_global` strict-leaveout experiment
- What changed:
  - Switched the default full-OT evaluation backend from `exact_lp` to `pot_emd2`:
    - `src/cfm_project/training.py` default fallback now uses `train.eval_full_ot_method=pot_emd2` when unspecified.
    - train profile defaults updated to `eval_full_ot_method: pot_emd2` in:
      - `configs/train/default.yaml`
      - `configs/train/smoke.yaml`
      - `configs/train/stage_a_only.yaml`
      - `configs/train/ab_only.yaml`
      - `configs/train/single_cell_ab_only.yaml`
      - `configs/train/single_cell_stage_a_only.yaml`
  - Ran a new Stage-A strict-leaveout single-cell experiment with Kantorovich (`ot_global`) coupling on EB 5D:
    - output root: `outputs/2026-05-04/single_cell_eb_5d_stage_a_strict_leaveout_ot_global_pot_default_single_cell_stage_a_only/17-35-42`
    - holdout index `2`, seed `3`, methods: constrained + metric family.
    - full-OT backend in run summary confirms:
      - `eval_full_ot_method=pot_emd2`
      - `eval_full_ot_num_itermax=1600000`
- Why (decision/rationale):
  - User requested POT full-metric implementation to be the default and asked for a Stage-A experiment using Kantorovich coupling with the new metric path.
- Impact or risk:
  - New runs that enable full-OT metrics now default to POT behavior without needing explicit method overrides.
  - Environments missing POT will now fail earlier unless `eval_full_ot_method` is manually set back to `exact_lp`.
- Architecture updates:
  - Updated `ARCHITECTURE.md` to record that the configurable full-OT backend defaults to `pot_emd2`.
- Validation status:
  - Regression sanity:
    - `./FM/bin/pytest tests/test_single_cell_stage_a_pipeline.py::test_single_cell_stage_a_ot_global_full_ot_pot_backend tests/test_single_cell_pipeline.py::test_single_cell_ab_ot_global_full_ot_pot_backend`
    - passed.
  - Experiment execution:
    - completed and wrote `comparison_mfm.json` with `pot_emd2` backend metadata.
- Next steps:
  1. If you want, I can launch the full `holdouts={1,2,3} x seeds={3,7,11}` Stage-A benchmark grid with these new defaults under `ot_global`.
  2. If needed, we can add an automatic fallback (`pot_emd2 -> exact_lp`) when POT is unavailable.

## [2026-05-04] Added POT backend for full-set OT metric computation (`pot_emd2`)
- What changed:
  - Implemented a new balanced empirical \(W_2\) backend using POT in `src/cfm_project/ot_utils.py`:
    - `balanced_empirical_w2_distance_pot(...)` now computes full-set OT distance with `ot.emd2`.
  - Extended `src/cfm_project/metrics.py` dispatcher:
    - `balanced_empirical_w2_distance(..., method=...)` now supports `exact_lp` and `pot_emd2`.
    - `interpolant_full_ot_w2_metrics(...)` now accepts/passes method and optional `num_itermax`.
  - Extended `src/cfm_project/training.py` full-OT evaluation plumbing:
    - `train.eval_full_ot_method` now accepts `pot_emd2` in addition to `exact_lp`.
    - added optional `train.eval_full_ot_num_itermax` and passed it through Stage-A and A+B full-OT metric paths.
    - summary payload now records `eval_full_ot_num_itermax`.
  - Added default config key `eval_full_ot_num_itermax: null` in train profiles.
  - Added tests covering the new backend and full pipeline wiring:
    - `tests/test_metrics.py` POT-vs-LP parity check,
    - `tests/test_single_cell_pipeline.py` A+B `pot_emd2` full-OT path,
    - `tests/test_single_cell_stage_a_pipeline.py` Stage-A `pot_emd2` full-OT path.
- Why (decision/rationale):
  - User requested full metric computation via POT because measured solve times were only seconds on the EB setup.
- Impact or risk:
  - Full-OT metric can now run with POT where available, while existing exact-LP behavior stays default and unchanged.
  - POT is treated as optional dependency; selecting `pot_emd2` without POT installed raises a clear import error.
- Architecture updates:
  - Updated `ARCHITECTURE.md` to document the new full-OT backend option and config knob.
- Validation status:
  - Ran:
    - `./FM/bin/pytest tests/test_metrics.py::test_balanced_empirical_w2_distance_pot_matches_exact_lp tests/test_single_cell_pipeline.py::test_single_cell_ab_ot_global_full_ot_pot_backend tests/test_single_cell_stage_a_pipeline.py::test_single_cell_stage_a_ot_global_full_ot_pot_backend`
    - `./FM/bin/pytest tests/test_metrics.py tests/test_single_cell_pipeline.py::test_single_cell_ab_ot_global_adds_full_ot_rollout_metrics tests/test_single_cell_stage_a_pipeline.py::test_single_cell_stage_a_ot_global_adds_full_ot_interpolant_metrics`
  - Result: all passed.
- Next steps:
  1. If you want POT as default for single-cell full-OT evaluation, we can switch `eval_full_ot_method` defaults in the single-cell train profiles.
  2. Optionally add a strict convergence check from POT `log` output (fail/flag when iteration cap is hit).

## [2026-05-04] Added and ran POT `ot.emd2` runtime sweep utility for Stage-A holdout W2
- What changed:
  - Added `scripts/run_pot_emd2_runtime_sweep.py`, a dedicated benchmark utility that:
    - loads EB single-cell data (`TrajectoryNet/data/eb_velocity_v5.npz`),
    - reuses cached `ot_global` endpoint support/masses,
    - constructs plan-conditioned pushforward samples at holdout `t=0.5`,
    - precomputes one squared-Euclidean cost matrix,
    - calibrates `numItermax` runtime, then runs a `[0.25x, 0.5x, 1x, 2x]` sweep,
    - writes `emd2_runtime_sweep.json` and `.tsv` artifacts.
  - Installed POT in project env (`./FM/bin/pip install POT`).
  - Executed one benchmark run:
    - `outputs/2026-05-04/pot_emd2_runtime_sweep/17-05-06`.
- Why (decision/rationale):
  - User requested direct POT `ot.emd2` runtime characterization as a function of `numItermax` for the Stage-A holdout setting.
- Impact or risk:
  - On this EB 5D setup, `ot.emd2` converged quickly once `numItermax >= 400000`; runtime plateaued around `~3.2–3.3s`, so the intended `~60s` target window was not reachable by `numItermax` scaling alone.
  - Recommendation from sweep logic: `numItermax=1600000` (first stable adjacent pair without iteration-limit warning).
- Architecture updates:
  - Updated `ARCHITECTURE.md` (new script responsibility row + scope note).
- Validation status:
  - Script CLI validated (`--help`).
  - Full benchmark run completed and produced JSON/TSV outputs with calibration, sweep records, and recommendation.
  - POT import and solver execution validated in `FM`.
- Next steps:
  1. If a ~60s benchmark is still desired, increase problem size (for example denser source support or larger target pool) rather than only increasing `numItermax`.
  2. Optionally add `ot.solve` as a second backend in this script for side-by-side runtime/value parity checks.

## [2026-05-04] Ran tuned `ot_global` Stage-A full-grid benchmark and reported mean ± std across seeds/holdouts
- What changed:
  - Executed tuned Stage-A strict-leaveout benchmark grid under `ot_global` with legacy empirical metrics only:
    - run root: `outputs/2026-05-04/single_cell_eb_5d_stage_a_strict_leaveout_ot_global_empirical_tuned/11-22-04`
    - holdouts `1,2,3`, seeds `3,7,11`, five Stage-A methods.
  - Applied tuned overrides from prior holdout-2/seed-3 sweep:
    - constrained: `rho=35`, `beta=0.08`, `alpha=1.0`
    - metric_constrained_al: `alpha=0.3`, `moment_eta=3.0`, `rho=15`
    - metric_constrained_soft: `alpha=0.3`, `moment_eta=3.0`
  - Generated aggregate artifacts with standard deviations:
    - `benchmark_summary_stage_a_ot_global_empirical_tuned.json`
    - `leaderboard_stage_a_ot_global_empirical_tuned.tsv`
  - Generated tuned-vs-untuned `ot_global` comparison artifacts:
    - `tuned_vs_untuned_ot_global_diff.json`
    - `tuned_vs_untuned_ot_global_diff.tsv`
  - Logged run details in `EXPERIMENTS.md`.
- Why (decision/rationale):
  - User requested multi-seed reporting with standard deviations and a full-grid confirmation after observing that single-slice sweep results were not directly comparable to 9-run means.
- Impact or risk:
  - Tuned settings improved mean holdout score for `metric_constrained_al` and constrained mode while preserving overall method ranking.
  - Improvements are modest for some modes; further gains may need broader search axes or per-mode training-budget tuning.
- Architecture updates:
  - No architecture changes; experiment orchestration and reporting only.
- Validation status:
  - `9/9` runs produced `comparison_mfm.json`.
  - Aggregated leaderboard contains mean/std per metric key.
  - Diff report confirms `metric_constrained_al` mean holdout improved vs untuned `ot_global` (`-0.02113` absolute in \(W_2\), lower is better).
- Next steps:
  1. If desired, run a second sweep on additional seeds/holdouts to tune robustly rather than from one slice (`h=2, s=3`).
  2. Compare tuned `ot_global` against the original batch-`ot` benchmark with uncertainty bands in the same table.

## [2026-05-04] Completed Stage-A `ot_global` hyperparameter sweep (legacy empirical metrics, holdout-2 seed-3)
- What changed:
  - Ran a focused Stage-A parameter sweep under coupling `ot_global` with legacy empirical metric mode only:
    - completed root: `outputs/2026-05-04/single_cell_eb_5d_stage_a_ot_global_param_sweep_empirical_only/11-10-28`
    - holdout `2`, seed `3`, strict leaveout.
  - Sweep grid (`31` configs total):
    - constrained: `rho x beta` (`9`)
    - metric: `sigma` (`3`)
    - metric_alpha0: baseline (`1`)
    - metric_constrained_al: `alpha x moment_eta` (`9`)
    - metric_constrained_soft: `alpha x moment_eta` (`9`)
  - Produced artifacts:
    - `sweep_summary.json`
    - `sweep_leaderboard.tsv`
    - `best_by_mode.tsv`
  - Added effective-parameter capture in sweep outputs to verify override application per run.
- Why (decision/rationale):
  - Coupling changed from batch `ot` to global `ot_global`; wanted to test whether best Stage-A method hyperparameters shift under the new coupling while keeping metric computation legacy-empirical for comparability.
- Impact or risk:
  - Best settings shifted for some methods (notably constrained and metric_constrained_al) on this holdout/seed slice.
  - This is a focused tuning pass (single holdout/seed), so multi-seed/holdout confirmation is still needed before promoting defaults.
- Architecture updates:
  - No architecture changes; experiment orchestration only.
- Validation status:
  - `31/31` configs completed with `comparison_mfm.json`.
  - Effective-parameter uniqueness checks confirmed sweep knobs were actually applied (not silently masked).
  - Best-by-mode summary generated and persisted.
- Next steps:
  1. Re-evaluate selected top candidates on full holdouts `1,2,3` and seeds `3,7,11`.
  2. Compare tuned-`ot_global` winners directly against previous untuned coupling-only benchmark outputs.

## [2026-05-04] Interrupted initial Stage-A `ot_global` sweep attempt due Hydra override-struct mismatch
- What changed:
  - Started a first sweep attempt at:
    - `outputs/2026-05-04/single_cell_eb_5d_stage_a_ot_global_param_sweep_empirical_only/11-07-52`
  - Encountered Hydra error for `experiment.method_overrides.metric.*` (non-predeclared key in struct map), then interrupted and relaunched corrected sweep.
- Why (decision/rationale):
  - Needed to stop invalid parameter path and avoid mixing partially invalid outputs with corrected sweep results.
- Impact or risk:
  - Partial outputs exist in `11-07-52` but are invalid for conclusions.
- Architecture updates:
  - No architecture changes; run-control correction only.
- Validation status:
  - Failure cause identified from run logs; corrected launcher completed in follow-up run (`11-10-28`).
- Next steps:
  1. Treat `11-07-52` as invalid and use `11-10-28` as source of truth.

## [2026-05-04] Completed Stage-A coupling-only reproduction (`ot_global` vs prior batch `ot`) with legacy empirical metrics
- What changed:
  - Executed the full Stage-A single-cell benchmark grid under `ot_global` with legacy empirical metric mode only:
    - run root: `outputs/2026-05-04/single_cell_eb_5d_stage_a_strict_leaveout_ot_global_empirical_only/10-59-32`
    - holdouts `1,2,3`, seeds `3,7,11`, five Stage-A methods.
  - Used coupling-only override set:
    - `data.coupling=ot_global`
    - `train.eval_full_ot_metrics=false`
    - `output.save_plots=false`
  - Produced new aggregate + comparison artifacts:
    - `benchmark_summary_stage_a_ot_global_empirical_only.json`
    - `leaderboard_stage_a_ot_global_empirical_only.tsv`
    - `coupling_diff_stage_a_empirical.json`
    - `coupling_diff_stage_a_empirical.tsv`
  - Updated `EXPERIMENTS.md` with this completed benchmark entry.
- Why (decision/rationale):
  - Needed a strict coupling-only comparison against prior batch-OT Stage-A results while keeping the same empirical Hungarian interpolant metric computation.
- Impact or risk:
  - New results are directly comparable to the previous 2026-05-01 Stage-A baseline without metric-family confounding.
  - In this run, ranking remained stable across methods, while absolute values shifted modestly under `ot_global`.
- Architecture updates:
  - No architecture changes; experiment orchestration and artifacts only.
- Validation status:
  - `9/9` runs generated `comparison_mfm.json`.
  - Required legacy interpolant keys were present for all runs/modes.
  - Aggregated leaderboard has 5 valid modes.
  - Diff artifacts include old/new metrics, signed/absolute deltas, and rank deltas.
  - Baseline sanity check preserved expected top method (`metric_constrained_al`).
- Next steps:
  1. Review `coupling_diff_stage_a_empirical.tsv` to decide whether to promote `ot_global` as default Stage-A coupling for future sweeps.
  2. If needed, repeat this coupling-only protocol for A+B rollout metrics (legacy empirical family only).

## [2026-05-03] Canceled relaunched Stage-A `ot_global` pilot on request
- What changed:
  - Stopped the active pilot process for:
    - `outputs/2026-05-03/single_cell_eb_5d_stage_a_strict_leaveout_ot_global_fullot/19-47-23/holdout_2/seed_3`
  - Verified process termination (no remaining matching `run_experiment.py` job).
  - Updated `EXPERIMENTS.md` status for this run to canceled.
- Why (decision/rationale):
  - User requested to stop the run and execute it at another time.
- Impact or risk:
  - No benchmark metrics were produced from this attempt.
  - Precomputed global OT cache remains available for future relaunches.
- Architecture updates:
  - No architecture changes; runtime control only.
- Validation status:
  - Process PID `6897` terminated successfully.
  - `comparison_mfm.json` absent in the canceled run folder.
- Next steps:
  1. Relaunch the same pilot when desired (cache hit expected for endpoint plan).
  2. After pilot completion, run success-gate checks and decide on full-grid rollout.

## [2026-05-03] Relaunched Stage-A `ot_global` pilot benchmark after global-plan cache completion
- What changed:
  - Launched a fresh strict-leaveout pilot run using the precomputed global OT cache:
    - `outputs/2026-05-03/single_cell_eb_5d_stage_a_strict_leaveout_ot_global_fullot/19-47-23/holdout_2/seed_3`
    - holdout `2`, seed `3`, Stage-A-only profile.
  - Kept requested run overrides:
    - `data.coupling=ot_global`
    - EB dataset path `TrajectoryNet/data/eb_velocity_v5.npz`
    - `mfm.backend=auto`
    - `output.save_plots=false`
  - Updated `EXPERIMENTS.md` with the relaunched pilot entry.
- Why (decision/rationale):
  - Prior pilot attempts stalled before a completed benchmark output; after cache precompute, this relaunch should avoid upfront global-plan solve and exercise the intended Stage-A benchmark path.
- Impact or risk:
  - If this completes, we can immediately apply the planned pilot success gate checks and proceed to full-grid orchestration.
  - Runtime risk remains in robust full-OT evaluation stages, even with cached endpoint plan.
- Architecture updates:
  - No architecture changes; experiment execution only.
- Validation status:
  - Process confirmed active and CPU-bound at launch window.
  - Completion artifacts pending at time of entry.
- Next steps:
  1. Wait for `comparison_mfm.json` generation for this run and validate robust + legacy key presence.
  2. If pilot gate passes, launch full holdout/seed grid with identical overrides.
  3. Aggregate and compare against previous Stage-A baseline summary.

## [2026-05-03] Completed standalone single-cell global OT cache precompute (`ot_global`) for EB 5D
- What changed:
  - The dedicated cache-precompute run completed:
    - script: `outputs/2026-05-03/ot_cache_build_single_cell_eb5d/19-18-17/build_ot_cache.py`
    - cache output: `.cache/ot_plans/f2055c582cae0bf05eb73ed6ed97fc225ff748e4c861e509a6c6b6ca5ab0355d.pt`
  - Recorded completion details in `EXPERIMENTS.md`.
- Why (decision/rationale):
  - Needed the expensive exact global endpoint Kantorovich plan to be available as a cache hit before re-running Stage-A `ot_global` benchmarks.
- Impact or risk:
  - Upcoming runs with matching single-cell signature should skip plan solve and begin SGD/evaluation directly.
  - Cache validity depends on unchanged dataset/preprocessing signature.
- Architecture updates:
  - No architecture changes; experiment/runtime state update only.
- Validation status:
  - Builder reported:
    - `global_ot_cache_hit=false` (fresh build),
    - `global_ot_support_size=5712`,
    - `global_ot_total_cost=15.589441731606119`,
    - elapsed `1389.41s` (~23.2 minutes).
  - Process exited cleanly; cache file present on disk.
- Next steps:
  1. Re-run Stage-A pilot with `data.coupling=ot_global` and verify cache-hit metadata in run summary.
  2. If pilot passes, launch full holdout/seed grid and aggregate robust-vs-legacy comparison.

## [2026-05-03] Launched standalone single-cell global OT cache precompute (`ot_global`) for EB 5D
- What changed:
  - Verified no ghost benchmark/training process was active from prior stalled attempts.
  - Started a dedicated standalone cache builder for single-cell global OT coupling:
    - `outputs/2026-05-03/ot_cache_build_single_cell_eb5d/19-18-17/build_ot_cache.py`
    - launcher: `PYTHONPATH=src python -u .../build_ot_cache.py`
  - Logged this run in `EXPERIMENTS.md` with exact folder and intent.
- Why (decision/rationale):
  - Separate cache precompute from benchmark orchestration so we can validate completion of the expensive exact LP endpoint plan once, then reuse it in subsequent Stage-A comparisons.
- Impact or risk:
  - If this completes, future `data.coupling=ot_global` runs should skip global-plan solve and start SGD immediately via cache hit.
  - Exact LP runtime remains the main operational risk on full endpoint cardinalities.
- Architecture updates:
  - No architecture changes; run orchestration only.
- Validation status:
  - Process confirmed active (CPU-bound) after launch.
  - Cache artifact not yet produced at log time (expected until solve completes).
- Next steps:
  1. Wait for `.cache/ot_plans/<key>.pt` write and capture support size/cost from script output.
  2. Re-run Stage-A pilot benchmark with `data.coupling=ot_global` once cache exists.
  3. If runtime remains impractical, evaluate controlled fallback options for benchmark feasibility.

## [2026-05-03] Attempted Stage-A `ot_global` robust full-OT benchmark run (pilot) and hit exact-LP runtime bottleneck
- What changed:
  - Launched the requested pilot run for single-cell Stage-A strict leaveout with exact global Kantorovich coupling and robust full-OT metrics:
    - holdout `2`, seed `3`,
    - `data.coupling=ot_global`,
    - `output.save_plots=false`,
    - run folder:
      - `outputs/2026-05-03/single_cell_eb_5d_stage_a_strict_leaveout_ot_global_fullot/17-57-09/holdout_2/seed_3`.
  - Aborted the process manually after extended runtime because the exact LP global-plan solve had not completed and no benchmark outputs had been produced.
  - Logged the attempt in `EXPERIMENTS.md` with command context and status.
- Why (decision/rationale):
  - This was the pilot gate for the requested full-grid benchmark; exact-LP feasibility on full EB endpoint cardinalities must be validated before scaling to 9 runs.
- Impact or risk:
  - Full-grid Stage-A benchmark has not started yet because the pilot did not pass completion gate.
  - Current exact-LP path on full EB endpoint pools is likely too slow for interactive execution windows without long-running/background execution.
- Architecture updates:
  - No architecture changes; run orchestration only, no code-path modifications.
- Validation status:
  - Pilot run launched with correct overrides and wrote Hydra config scaffolding.
  - Success artifacts (`comparison_mfm.json`, per-mode metrics) were not generated due early stop.
- Next steps:
  1. Re-run the same pilot in a long-running/offline window and wait for completion.
  2. If exact completion is still impractical, decide between:
     - reducing problem size for pilot (for feasibility diagnostics only), or
     - adding a faster robust OT backend for benchmark execution.
  3. Once pilot completes, proceed automatically to full-grid holdout/seed runs and aggregate comparison vs the 2026-05-01 baseline benchmark.

## [2026-05-03] Added single-cell global OT coupling mode and robust full-set OT metrics (Stage-A and A+B)
- What changed:
  - Added exact balanced OT LP utilities in `src/cfm_project/ot_utils.py` (sparse-constraint LP solve, sparse support extraction, and weighted rectangular \(W_2\) cost helper).
  - Extended empirical coupling data model and sampler:
    - `EmpiricalCouplingProblem` now optionally carries global OT sparse support (`src_idx`, `tgt_idx`, `mass`, total cost),
    - `sample_coupled_batch` now supports `coupling='ot_global'` with replacement sampling from cached support.
  - Extended single-cell preparation to support global-plan precompute/load cache:
    - cache path default `.cache/ot_plans`,
    - deterministic signature keying over dataset/preprocessing/endpoint hashes,
    - cache metadata propagated in pipeline outputs.
  - Added robust additive metric family (legacy preserved):
    - Stage-A interpolant robust keys: `linear_full_ot_w2*`, `learned_full_ot_w2*`, holdout and deltas,
    - A+B rollout robust keys: `intermediate_full_ot_w2*`, `transport_endpoint_full_ot_w2`, `holdout_full_ot_w2`.
  - Extended benchmark scripts:
    - stage-A benchmark now aggregates robust + legacy keys and ranks by robust holdout key with legacy fallback if robust is missing,
    - A+B benchmark aggregation now includes robust rollout keys.
  - Added config controls:
    - `data.coupling: ot_global` (single-cell),
    - single-cell OT cache controls (`global_ot_cache_*`, support tolerance),
    - train robust metric toggles (`eval_full_ot_metrics`, method, variable guard, tolerance).
  - Added tests:
    - global OT sampling behavior,
    - weighted/rectangular robust OT metric correctness,
    - single-cell global-plan cache hit path,
    - Stage-A and A+B single-cell pipeline coverage for robust keys with `ot_global`,
    - benchmark aggregation/ranking behavior with robust-key priority and fallback.
- Why (decision/rationale):
  - Needed to move from per-batch Hungarian coupling to optional global Kantorovich coupling for training, and to add a more faithful full-set OT evaluation path while retaining backward comparability with legacy sampled metrics.
- Impact or risk:
  - Training/eval capabilities are broader and better aligned with unequal-cardinality empirical OT settings.
  - Exact LP robust/full-set computations can be heavy on large pools; cache reuse and `*_max_variables` guards are important operationally.
- Architecture updates:
  - Updated `ARCHITECTURE.md` for new `ot_global` coupling path, `ot_utils` module role, robust metric flow, and single-cell cache semantics.
  - Updated `DISCUSSION.md` with method tradeoffs and robust-vs-legacy metric rationale.
  - Updated `FORMALIZATION.tex`/`FORMALIZATION.pdf` with global-plan sampling semantics and full-set balanced OT metric definitions.
- Validation status:
  - `pytest -q` -> `64 passed`.
  - `python -m compileall -q src scripts tests` succeeded.
  - Rebuilt `FORMALIZATION.pdf` successfully from updated TeX.
- Next steps:
  1. Run EB strict-leaveout benchmarks with `data.coupling=ot_global` and robust metrics enabled, then compare robust-vs-legacy rank stability.
  2. Profile LP memory/runtime on full EB endpoint sizes and set practical `eval_full_ot_max_variables` defaults if needed.
  3. If exact LP becomes prohibitive on larger datasets, add an approximate robust-OT backend option (while preserving exact path for reference runs).

## [2026-05-02] Clarified empirical OT metric sampling semantics in formalization
- What changed:
  - Updated `FORMALIZATION.tex` metrics documentation in:
    - sample-based empirical OT variant section,
    - Stage-A-only interpolant evaluation section.
  - Added explicit implementation notes that current empirical OT evaluation uses `N = eval_intermediate_ot_samples` (current single-cell default `256`), computes one sampled evaluation set per run, runs Hungarian matching per evaluated timepoint on that sampled set, and reports `*_avg` fields as averages over timepoints (not over multiple eval batches).
  - Rebuilt `FORMALIZATION.pdf` from the updated TeX source.
- Why (decision/rationale):
  - These sampling/aggregation details materially affect interpretation of empirical OT numbers and needed to be explicit in the formalization metrics section.
- Impact or risk:
  - Documentation-only clarification; no training, metric, or pipeline behavior changed.
- Architecture updates:
  - No architecture changes; system/data-flow behavior unchanged.
- Validation status:
  - Verified new clarification text appears in both targeted sections of `FORMALIZATION.tex`.
  - Rebuilt PDF to ensure the formalization artifact reflects the updated wording.
- Next steps:
  1. Optionally mirror the same “single sampled eval set per run” wording in bridge rollout subsection for symmetry.
  2. If needed, add a short README pointer to these metric-interpretation notes for quick discoverability.

## [2026-05-01] Added single-cell Stage-A-only benchmark pipeline, per-mode overrides, and completed EB strict-leaveout grid
- What changed:
  - Enabled robust Stage-A-only support for metric-family modes across pipeline/eval/plotting:
    - Stage-A guards now allow `constrained`, `metric`, `metric_alpha0`, `metric_constrained_al`, `metric_constrained_soft`.
    - Stage-A interpolant evaluation is now mode-aware:
      - constrained uses `corrected_path`,
      - metric-family uses MFM mean path,
      - `metric_alpha0` uses linear path.
    - Added holdout-aware Stage-A interpolant metrics:
      - `linear_holdout_empirical_w2`,
      - `learned_holdout_empirical_w2`,
      - `delta_holdout_learned_minus_linear`.
  - Added per-mode config override support for comparison runs via `experiment.method_overrides` (deep merge per mode).
  - Added new Stage-A single-cell preset:
    - `configs/experiment/comparison_mfm_single_cell_stage_a.yaml`.
  - Added new benchmark runner:
    - `scripts/run_single_cell_eb_stage_a_benchmark.py`
    - loops holdouts and seeds, writes:
      - per-run `comparison_mfm.json`,
      - aggregate `benchmark_summary_stage_a.json`,
      - `leaderboard_stage_a.tsv`.
  - Added tests:
    - `tests/test_single_cell_stage_a_pipeline.py` (5-mode Stage-A smoke, holdout fields, projected plots, method-overrides correctness),
    - updated `tests/test_metrics.py` with holdout-metric assertions.
- Why (decision/rationale):
  - Needed an execution-ready Stage-A-only embryo benchmark to compare interpolant quality directly under strict leaveout, with fair/consistent method settings and reusable aggregate artifacts.
- Impact or risk:
  - Stage-A single-cell benchmarking is now first-class and reproducible with explicit holdout reconstruction metrics.
  - Added override flexibility reduces script-side complexity and keeps method-tuned comparison runs in one artifact, but introduces one more config surface (`experiment.method_overrides`) that must stay validated.
- Architecture updates:
  - Updated `ARCHITECTURE.md` with Stage-A metric-family path behavior, per-mode override flow, and Stage-A benchmark artifact contract.
  - Updated `DISCUSSION.md` with protocol rationale and empirical interpretation of the new EB Stage-A benchmark.
- Validation status:
  - Tests passed:
    - `pytest -q tests/test_metrics.py tests/test_single_cell_stage_a_pipeline.py tests/test_single_cell_pipeline.py tests/test_mfm_pipeline.py`
    - Result: `16 passed`.
  - Real benchmark run completed:
    - `python scripts/run_single_cell_eb_stage_a_benchmark.py --data-path TrajectoryNet/data/eb_velocity_v5.npz`
    - Outputs:
      - `outputs/2026-05-01/single_cell_eb_5d_stage_a_strict_leaveout/18-55-20/benchmark_summary_stage_a.json`
      - `outputs/2026-05-01/single_cell_eb_5d_stage_a_strict_leaveout/18-55-20/leaderboard_stage_a.tsv`
    - Aggregate learned holdout \(W_2\) mean ranking:
      - `metric_constrained_al` (best), then `metric`, `metric_constrained_soft`, `metric_alpha0`, `constrained`.
- Next steps:
  1. Tune Stage-A constrained-only on EB (reduce heldout overfit) and rerun the same grid.
  2. Promote the Stage-A leaderboard into the reporting notebook/table generator alongside Stage-A+B metrics.
  3. Optionally run the same Stage-A protocol with non-strict constraints (`observed_nonendpoint_all`) as a fairness sensitivity check.

## [2026-04-30] Implemented constrained `beta(t)` smoothness scheduling and formalized it in Stage-A math
- What changed: Extended constrained-mode Stage A/C objective to use time-varying smoothness weighting `beta(t)` instead of fixed scalar `beta`, with scheduler options `train.beta_schedule in {constant, piecewise, linear}`. Added drift-based schedule builder in `training.py` using anchors `{0.0, constraint_times..., 1.0}` and full moment-feature drift (mean + covariance vector) with clipping controls (`train.beta_drift_p`, `train.beta_drift_eps`, `train.beta_min_scale`, `train.beta_max_scale`). Added constrained summary metadata fields for schedule traceability (anchor times, drifts, interval values, anchor values, and scheduler hyperparameters). Added config defaults in all train profiles. Added unit/smoke tests in `tests/test_constrained_beta_schedule.py` for constant-equivalence, synthetic schedule correctness, drift-vs-beta monotonicity, and constrained pipeline support for `piecewise`/`linear`. Updated `FORMALIZATION.tex` with a new subsection “Time-Varying Smoothness Weighting in Stage A/C” and updated Algorithm 1 regularizer equation; rebuilt `FORMALIZATION.pdf`.
- Why (decision/rationale): Needed a constrained-only extension to allow stronger/looser temporal smoothness at different times based on observed moment drift, while preserving strict backward compatibility through `beta_schedule=constant`.
- Impact or risk: Adds a controllable smoothness schedule without changing metric-family behavior. In quick bridge seed-3 sanity runs, `piecewise`/`linear` were near-parity with constant on intermediate marginals and slightly better on endpoint, suggesting low risk and immediate utility for further tuning.
- Architecture updates: Updated `ARCHITECTURE.md` (constrained schedule flow + path-prior equation + new summary fields) and `DISCUSSION.md` (rationale/tradeoffs + first empirical comparison note).
- Validation status:
  - Tests: `pytest -q tests/test_constrained_beta_schedule.py tests/test_smoke_training.py tests/test_mfm_pipeline.py tests/test_mfm_core.py` -> `18 passed`.
  - PDF build: `TEXMFVAR=.cache/texmf-var pdflatex -interaction=nonstopmode FORMALIZATION.tex` completed successfully.
  - Bridge quick sanity runs (seed 3, plots enabled):
    - `outputs/2026-04-30/bridge_beta_sched_constant_ab_only/14-57-44`
    - `outputs/2026-04-30/bridge_beta_sched_piecewise_ab_only/14-59-40`
    - `outputs/2026-04-30/bridge_beta_sched_linear_ab_only/14-59-40`
- Next steps: 1) run a focused constrained sweep over `beta_drift_p`, `beta_min_scale`, and `beta_max_scale` to amplify schedule contrast, 2) repeat best schedule settings on seeds `7` and `11`, 3) compare schedule effects under bridge random coupling where drift heterogeneity may be stronger.

## [2026-04-29] Ran metric-constrained AL/soft balance sweep on bridge OT and generated top-config plots
- What changed: Executed a focused 32-run sweep under `outputs/bridge_mfm_constrained_hparam_sweep/2026-04-29_17-27-15/` to tune the balance between LAND metric objective and intermediate moment constraints for both new modes (`metric_constrained_al`, `metric_constrained_soft`). Added ranked TSV artifacts and `final_report.json`; reran baseline + best AL + best soft configs with plotting enabled under `top_plots/`.
- Why (decision/rationale): Needed a targeted search over the key balance knobs (`mfm.moment_eta`, plus AL `train.rho` and MFM `alpha/sigma`) to test whether endpoint transport can be improved while maintaining good intermediate marginals.
- Impact or risk: In this seed-3 sweep, both AL and soft improved intermediate marginals relative to baseline but none met the endpoint gate (`delta_endpoint <= +0.03`); AL remained substantially stronger than soft on marginal reconstruction, but endpoint degradation persisted.
- Architecture updates: No architecture changes (experiment-only session; code structure unchanged).
- Validation status:
  - Sweep root: `outputs/bridge_mfm_constrained_hparam_sweep/2026-04-29_17-27-15/`
  - Total configs: 32 (`16` AL + `16` soft)
  - Gate pass counts: AL `0/16`, soft `0/16`
  - Best-ranked AL (endpoint-first): `eta0p5_rho1_s0p05`
    - `delta_intermediate=-0.13781`, `delta_endpoint=+0.15921`
  - Best-ranked soft (endpoint-first): `eta2_a0p5_s0`
    - `delta_intermediate=-0.05393`, `delta_endpoint=+0.19633`
  - Top-config plot folders created:
    - `.../top_plots/baseline/`
    - `.../top_plots/metric_constrained_al/best/`
    - `.../top_plots/metric_constrained_soft/best/`
- Next steps: 1) run a second sweep explicitly endpoint-oriented (lower `mfm.alpha`, smaller/no noise `mfm.sigma`, potentially higher `land_rho`), 2) retest best AL settings across seeds `7` and `11`, 3) compare against constrained baseline at matched budgets to decide whether endpoint-focused regularization needs an objective change.

## [2026-04-29] Implemented fair-hybrid metric-constrained MFM modes and formalized math in PDF
- What changed: Added two new metric-family training modes, `metric_constrained_al` and `metric_constrained_soft`, that keep Stage-B metric flow matching unchanged while augmenting Stage-A geopath learning with moment constraints at constrained times. Added MFM config knobs `mfm.reference_pool_policy` (default `endpoints_only`) and `mfm.moment_eta` (default `1.0`). Updated metric reference-pool construction to support `endpoints_only` vs `anchors_all`; default now uses only endpoint marginals (`t=0,1`) for LAND geometry. Extended method-list comparisons and preset methods to 6-way runs (`baseline`, `constrained`, `metric`, `metric_alpha0`, `metric_constrained_al`, `metric_constrained_soft`). Added metric summary fields: `mfm_reference_pool_policy`, `mfm_moment_style`, `mfm_moment_eta`. Extended path plotting support for new modes. Added/updated tests in `tests/test_mfm_core.py` and `tests/test_mfm_pipeline.py` for pool policy, AL/soft objectives, eta-zero collapse, and 6-way comparison contract. Updated `scripts/run_bridge_mfm_best.py` to report new modes.
- Why (decision/rationale): Needed a fair comparison where metric methods preserve MFM geometry and endpoint-based manifold learning, while receiving intermediate information only through the same moment-constraint channel used by constrained training.
- Impact or risk: Enables direct apples-to-apples constrained vs metric-constrained comparisons under matched information access assumptions. On the seed-3 bridge sanity run, `metric_constrained_al` substantially improved intermediate marginals versus vanilla metric/soft, but endpoint degradation remained significant, so endpoint recovery is still the main open issue for metric-family methods.
- Architecture updates: Updated `ARCHITECTURE.md` with new mode taxonomy, hybrid Stage-A objectives, and endpoint-only pool policy. Updated `DISCUSSION.md` with rationale/tradeoff notes and empirical 6-way run interpretation. Updated `FORMALIZATION.tex` with a dedicated section “Constrained-Information Metric Flow Matching (Endpoint LAND + Moment Constraints)” including full equations and algorithm; rebuilt `FORMALIZATION.pdf`.
- Validation status:
  - Tests:
    - `pytest -q tests/test_mfm_core.py tests/test_mfm_pipeline.py` -> `11 passed`
    - `pytest -q tests/test_smoke_training.py tests/test_bridge_ab_pipeline.py tests/test_stage_a_bridge_pipeline.py` -> `7 passed`
  - Benchmark sanity runs logged under:
    - `outputs/2026-04-29/bridge_mfm_hybrid_sanity_ab_only/15-44-34` (default constrained rho diagnostic)
    - `outputs/2026-04-29/bridge_mfm_hybrid_sanity_ab_only/16-00-17` (canonical best constrained settings)
  - PDF build: `TEXMFVAR=.cache/texmf-var pdflatex -interaction=nonstopmode FORMALIZATION.tex` completed successfully and regenerated `FORMALIZATION.pdf`.
- Next steps: 1) run a focused sweep over `mfm.moment_eta` and LAND hyperparameters for `metric_constrained_al`; 2) test endpoint-oriented variants (for example lower `mfm.alpha` and/or lower `mfm.sigma`) while monitoring constrained-time OT gains; 3) repeat best hybrid settings on seeds `7` and `11`, then compare with `mfm.backend=torchcfm` when available.

## [2026-04-29] Ran bridge MFM (alpha>0) sweep and plotted top seed-3 configuration
- What changed: Executed a focused 16-config bridge OT MFM hyperparameter sweep under `outputs/bridge_mfm_hparam_sweep/2026-04-29_12-01-21/` using methods `baseline` vs `metric` (nonzero alpha only), with fixed high-sample budget (`stage_a_steps=300`, `stage_b_steps=300`, `stage_c_steps=0`, `batch_size=512`, `eval_intermediate_ot_samples=1024`, `eval_transport_samples=4000`, `seed=3`). Swept parameters: `mfm.alpha ∈ {0.5,1.0}`, `mfm.sigma ∈ {0.0,0.05}`, `mfm.land_gamma ∈ {0.08,0.125}`, `mfm.land_rho ∈ {0.0005,0.001}`. Then reran the top-ranked config with plots enabled at `outputs/2026-04-29/bridge_mfm_top_from_sweep_plots_ab_only/12-23-57/`.
- Why (decision/rationale): Needed an MFM-focused analogue of the constrained-method tuning process to test whether nonzero-alpha metric flow matching can improve intermediate marginals while preserving endpoint quality on the bridge benchmark.
- Impact or risk: Sweep found no gate-passing configuration under the tested grid (`0/16` for gate `delta_intermediate < 0` and `delta_endpoint <= +0.03`). Best-ranked config (`a1_s0p05_g0p125_r0p001`) was near baseline on intermediate marginals (`delta_intermediate=+0.00570`) but still substantially worse at endpoint (`delta_endpoint=+0.13022`), so MFM remains behind constrained in this setting.
- Architecture updates: None (experiment-only runs; no structural code changes).
- Validation status: Sweep artifacts generated (`phase1_summary.tsv`, `final_report.json`) and plotted rerun produced `baseline/sample_paths.png` and `metric/sample_paths.png` plus rollout diagnostics in the top-run folder.
- Next steps: 1) run a second MFM sweep focused on endpoint recovery (for example lower `mfm.alpha`, larger `land_rho`, and smaller/no-noise `mfm.sigma` variants); 2) add multi-seed confirmation for the best endpoint-oriented configs; 3) once `torchcfm` is available, repeat the best configs with `mfm.backend=torchcfm` to separate algorithmic vs backend effects.

## [2026-04-29] Re-ran bridge MFM best-preset comparison with plots enabled
- What changed: Executed the bridge best-preset 4-method comparison again with `output.save_plots=true` so per-method visual artifacts are generated, including `sample_paths.png` for `metric` and `metric_alpha0`.
- Why (decision/rationale): Needed direct visual inspection of learned interpolant/path geometry for metric-flow-matching runs, analogous to previous constrained-mode path plots.
- Impact or risk: Adds a fully comparable plotted run folder at `outputs/2026-04-29/bridge_mfm_best_from_sweep_plots_ab_only/11-17-07/` without changing code or metric schema.
- Architecture updates: None (experiment-only rerun; architecture unchanged).
- Validation status: Run completed successfully and produced `comparison_mfm.json`, legacy `comparison.json`, and per-method plot artifacts including:
  - `metric/sample_paths.png`
  - `metric_alpha0/sample_paths.png`
  - `baseline/sample_paths.png`
  - `constrained/sample_paths.png`
- Next steps: 1) inspect `metric/sample_paths.png` and `metric_alpha0/sample_paths.png` against constrained to assess path geometry differences; 2) if useful, repeat the plotted run for seeds `7` and `11` for visual robustness checks.

## [2026-04-28] Added hybrid Metric Flow Matching integration and 4-method bridge comparison workflow
- What changed: Implemented `metric` and `metric_alpha0` modes in the main training pipeline; added `src/cfm_project/mfm_core.py` with MFM path/velocity formulas, LAND metric objective utilities, and backend resolver (`mfm.backend = auto|native|torchcfm`) plus explicit summary fields (`mfm_backend`, `mfm_backend_impl`, `mfm_alpha`, `mfm_sigma`, `mfm_land_gamma`, `mfm_land_rho`); extended orchestration to support `experiment.comparison_methods` and write `comparison_mfm.json`; preserved legacy `comparison.json` schema when baseline+constrained are included; added new configs `configs/mfm/default.yaml` and `configs/experiment/comparison_mfm.yaml`; added friendly runner `scripts/run_bridge_mfm_best.py`; updated plotting to support metric-mode path plots; added tests `tests/test_mfm_core.py` and `tests/test_mfm_pipeline.py`; updated README commands.
- Why (decision/rationale): Needed apples-to-apples Metric Flow Matching benchmarking on the bridge example inside the existing evaluation/output contract without forcing heavyweight new dependencies.
- Impact or risk: Enables one-command 4-method comparisons (`baseline`, `constrained`, `metric`, `metric_alpha0`) in the existing repo structure; current environment selected native backend (`mfm_backend=native`) because `torchcfm` is not installed, so strict torchcfm parity remains a follow-up validation item.
- Architecture updates: Updated `ARCHITECTURE.md` to document the new `mfm_core` component, method-list comparison flow, `comparison_mfm.json` artifact, and MFM-specific limitations.
- Validation status:
  - Tests: `pytest -q tests/test_mfm_core.py tests/test_mfm_pipeline.py tests/test_smoke_training.py tests/test_bridge_ab_pipeline.py` -> `11 passed`.
  - Real benchmark run: `python3 scripts/run_bridge_mfm_best.py` produced
    - `outputs/2026-04-28/bridge_mfm_best_from_sweep_ab_only/22-42-21/comparison_mfm.json`
    - `outputs/2026-04-28/bridge_mfm_best_from_sweep_ab_only/22-42-21/comparison.json`
  - Key run result: constrained remained strongest on this preset; metric (LAND) underperformed baseline on both intermediate and endpoint OT.
- Next steps: 1) rerun the new `comparison_mfm` preset for seeds `7` and `11`; 2) once `torchcfm` is available, rerun with `mfm.backend=torchcfm` and compare against native-backend outputs; 3) tune LAND hyperparameters (`land_gamma`, `land_rho`, metric sample budget) and retest bridge OT/random.

## [2026-04-28] Re-ran best bridge A+B OT sweep configuration as a fresh comparison run
- What changed: Recovered the top phase-2 sweep configuration (`rho0p5_alpha1_lrg0p001_lrv0p001`) from `outputs/bridge_ab_ot_hparam_sweep/2026-04-27_19-28-16/phase2_summary.tsv`, selected the best seed-level run (`seed=3`) from `phase2_runs.tsv`, and executed a fresh run `python3 scripts/run_experiment.py experiment=comparison train=ab_only data=bridge_ot experiment.label=bridge_ab_ot_best_from_sweep train.stage_a_steps=300 train.stage_b_steps=300 train.stage_c_steps=0 train.batch_size=512 train.beta=0.05 train.eval_intermediate_ot_samples=1024 train.eval_transport_samples=4000 train.rho=0.5 train.alpha=1.0 train.lr_g=0.001 train.lr_v=0.001 seed=3`.
- Why (decision/rationale): Needed to operationalize the sweep winner into a standalone reproducible run artifact using the same high-sample evaluation budget as the tuning sweep.
- Impact or risk: Produced a clean best-configuration run folder at `outputs/2026-04-28/bridge_ab_ot_best_from_sweep_ab_only/12-24-22/`; results reproduce sweep behavior for this config/seed (`intermediate_empirical_w2_avg`: baseline `0.33635` -> constrained `0.19233`, endpoint OT: baseline `0.13202` -> constrained `0.11603`).
- Architecture updates: None (experiment execution only; no code or structural changes).
- Validation status: Run completed successfully and emitted `comparison.json` plus per-mode artifacts under baseline/constrained subfolders.
- Next steps: 1) run the same best hyperparameters for seeds `7` and `11` in the new output namespace for a local multi-seed confirmation set; 2) rerun best config with `data=bridge_random` to test coupling sensitivity under tuned bridge A+B settings; 3) if stable, promote this config as the default bridge A+B starting point in docs/config presets.

## [2026-04-27] Completed bridge OT A+B two-phase high-sample sweep (16 + 12 runs)
- What changed: Added and used `scripts/run_bridge_ab_ot_sweep.py` to run a reproducible two-phase sweep for bridge OT in A+B mode with higher sample budgets (`batch_size=512`, `eval_intermediate_ot_samples=1024`, `eval_transport_samples=4000`); completed phase 1 (16 configs, seed 7) and phase 2 (top 4 configs x seeds 3/7/11), writing summaries under `outputs/bridge_ab_ot_hparam_sweep/2026-04-27_19-28-16/`.
- Why (decision/rationale): Needed to tune constrained bridge A+B for better intermediate transport while keeping endpoint transport close to baseline under a stricter, more reliable evaluation budget.
- Impact or risk: Primary sweep objective was achieved (`success_primary=true`): at least one config improved intermediate metric with endpoint gap <= +0.03 on multi-seed average; stretch objective was not met (`success_stretch=false`): no config achieved non-positive endpoint delta while improving intermediate metric.
- Architecture updates: No architecture change (experiment-orchestration and analysis only).
- Validation status: Sweep artifacts and summaries produced:
  - `phase1_summary.tsv` (16 configs),
  - `phase2_summary.tsv` (4 configs x 3 seeds aggregated),
  - `final_report.json`.
  Top phase-2 config by mean intermediate gain:
  - `rho0p5_alpha1_lrg0p001_lrv0p001` with `delta_intermediate_mean=-0.12835`, `delta_endpoint_mean=+0.00472`.
- Next steps: 1) run this top config against `data=bridge_random`; 2) perform 3-seed confirmation against the matched Stage-A baseline run at full plotting enabled; 3) consider mild endpoint-focused regularization/tuning if we want to close the remaining endpoint gap.

## [2026-04-27] Bridge A+B rerun with Stage-A-only hyperparameters (OT)
- What changed: Executed a new bridge A+B comparison run with Stage-A settings aligned to the Stage-A-only setup (`stage_a_steps=300`, `rho=1.0`) while keeping Stage B enabled (`stage_b_steps=300`, `stage_c_steps=0`).
- Why (decision/rationale): Needed to verify whether the stronger path curvature/quality gap seen in prior A+B outputs came from Stage B itself or from mismatched Stage-A hyperparameters between runs.
- Impact or risk: This controlled rerun shows much stronger constrained performance than the previous bridge A+B default (`rho=5`, `stage_a_steps=200`), supporting the conclusion that hyperparameter mismatch (not Stage-B updating `g`) was the primary driver of the earlier discrepancy.
- Architecture updates: None (experiment-only rerun; no structural code changes).
- Validation status: Run completed at `outputs/2026-04-27/bridge_ab_match_stagea_ab_only/18-55-22/`; key constrained metrics: `constraint_residual_avg=0.0532`, `intermediate_empirical_w2_avg=0.2860`, `transport_endpoint_empirical_w2=0.2125`; baseline in same run: `0.2043`, `0.3296`, `0.1299` respectively.
- Next steps: 1) run the same matched-hyperparameter A+B setup for `data=bridge_random`; 2) repeat matched OT run across 3 seeds to check stability; 3) if stable, promote this as bridge A+B default starting point.

## [2026-04-27] Enabled bridge Stage-A+B (`ab_only`) with rollout OT metrics and comparison workflow
- What changed: Removed the bridge-only Stage-A restriction in training; enabled bridge runs with velocity learning for non-Stage-A-only profiles (notably `train=ab_only`); added bridge rollout metric contract for velocity-enabled runs: `intermediate_empirical_w2`, `intermediate_empirical_w2_avg`, `transport_endpoint_empirical_w2`, and bridge `transport_score = transport_endpoint_empirical_w2`; kept Gaussian-only transport fields explicitly `null` for bridge runs; added rollout diagnostics plots (`rollout_marginal_grid.png`, `rollout_empirical_w2.png`) for bridge velocity-enabled runs; added notebook `notebooks/bridge_ab_comparison_analysis.ipynb` to load `comparison.json`, print baseline-vs-constrained bridge metrics, and display per-mode rollout plots.
- Why (decision/rationale): Needed to compare baseline vs constrained on bridge data after path pretraining, using metrics that remain meaningful for empirical/non-Gaussian bridge marginals; Stage-A-only workflow remains intact for interpolant-isolation studies.
- Impact or risk: Bridge comparison runs now complete with A+B and emit interpretable rollout metrics/artifacts; empirical OT evaluation is still sample-budget sensitive (via `eval_intermediate_ot_samples`), so larger/complex settings should raise this budget.
- Architecture updates: Updated `ARCHITECTURE.md` and `FORMALIZATION.tex` to separate Stage-A interpolant-only metrics from Stage-A+B bridge velocity-rollout metrics and to document bridge-specific `transport_score` semantics.
- Validation status: `pytest -q` passed (`32 passed`); executed official run `python3 scripts/run_experiment.py experiment=comparison train=ab_only data=bridge_ot experiment.label=bridge_ab_ot` with artifacts at `outputs/2026-04-27/bridge_ab_ot_ab_only/18-35-42/` (including `comparison.json`, per-mode metrics/checkpoints, and rollout plots).
- Next steps: 1) run the same A+B comparison on `data=bridge_random`; 2) run 3-seed repeats for bridge OT A+B; 3) decide whether to introduce a low-weight Stage-C bridge sweep after A+B stability checks.

## [2026-04-27] Rescaled bridge endpoint time (normalized timeline, physical horizon 1.5)
- What changed: Updated bridge target construction to map normalized experiment times to physical SDE time via `t_phys = t_norm * bridge.total_time`; bridge target snapshots now sample at physical `{0, total_time, constraint_times*total_time}` and are re-indexed back to normalized keys (`0.0, 0.25, 0.50, 0.75, 1.0`) for downstream compatibility; set `bridge.total_time=1.5` in `configs/data/bridge_ot.yaml` and `configs/data/bridge_random.yaml`; updated bridge notebooks to display normalized→physical mapping and use `total_time=1.5` preview settings.
- Why (decision/rationale): At `total_time=1.0`, many particles were still concentrated near the bridge at normalized endpoint; extending physical horizon while keeping normalized training times unchanged gives additional post-bridge expansion without changing model interfaces.
- Impact or risk: Improves endpoint spread at normalized `t=1` for bridge experiments; introduces new cache keys for bridge targets by design because `total_time` is part of cache identity; no changes to SDE equations or metric schema.
- Architecture updates: Updated `ARCHITECTURE.md` and `FORMALIZATION.tex` to formalize normalized-time semantics and physical-time mapping for bridge targets.
- Validation status: `pytest -q` passed (`28 passed`), including new mapping/cache/backward-compat tests; stage-a-only smoke runs completed:
  - `outputs/2026-04-27/bridge_stagea_rescaled_ot_stage_a_only/17-00-24`
  - `outputs/2026-04-27/bridge_stagea_rescaled_random_stage_a_only/17-01-22`
  Both produced full Stage-A artifacts and `interpolant_eval`; in the rescaled target cache, target spread increased from normalized `t=0.5` to `t=1.0` (`std_y`: `0.2835 -> 0.3824`).
- Next steps: 1) run multi-seed repeats for rescaled OT/random settings; 2) tune `rho/alpha/beta/lr_g` under the new horizon; 3) then add bridge Stage-B velocity training and compare interpolant-only vs rollout metrics.

## [2026-04-27] Added bridge Stage-A-only interpolant pipeline with cached targets and OT-vs-random runs
- What changed: Implemented bridge data integration (`data=bridge_ot|bridge_random`) backed by SDE simulation and target caching under `.cache/bridge_targets`; added `train=stage_a_only` profile (`stage_b_steps=0`, `stage_c_steps=0`) with constrained-only enforcement; added interpolant-only empirical OT metrics (`interpolant_eval.*`) comparing linear vs learned interpolants against true bridge marginals at constrained times; added Stage-A plots (interpolant trajectories, linear-vs-learned-vs-true marginal grid, interpolant empirical OT bars); added Stage-A analysis notebook `notebooks/bridge_stage_a_interpolant_analysis.ipynb`; added tests for cache behavior, stage-a bridge pipeline smoke, interpolant metric contract, and plotting artifacts.
- Why (decision/rationale): Needed to validate path/interpolant quality first on bridge data before re-introducing velocity learning, and needed run artifacts that clearly separate interpolant improvements from velocity-rollout effects.
- Impact or risk: Bridge experiments are now reproducible and comparable across coupling choices with explicit Stage-A-only metadata; current bridge integration is intentionally limited to Stage-A-only in pipeline (velocity stages disabled for bridge family at this milestone).
- Architecture updates: Updated architecture/formalization docs to distinguish velocity-rollout metrics from Stage-A interpolant-only metrics; added bridge data-cache component and Stage-A-only flow notes.
- Validation status: `pytest -q` passed (`25 passed`); executed requested run matrix:
  - `outputs/2026-04-27/bridge_stagea_bridge_ot_stage_a_only/15-15-56`
  - `outputs/2026-04-27/bridge_stagea_bridge_random_stage_a_only/15-17-24`
  Observed interpolant empirical OT averages:
  - OT: linear `0.3104` -> learned `0.2774` (improved, delta `-0.0331`)
  - random: linear `0.2909` -> learned `0.2986` (worse, delta `+0.0076`)
- Next steps: 1) run 3-seed repeats for both bridge couplings to test stability; 2) tune Stage-A hyperparameters on bridge (especially `rho`, `alpha`, `beta`, `lr_g`) for random-coupling recovery; 3) once interpolant behavior is stable, add bridge Stage-B velocity training and compare interpolant-only vs velocity-rollout metrics.

## [2026-04-27] Added ignore rules for local env and LaTeX artifacts
- What changed: Updated `.gitignore` to ignore `FM/` (local virtual environment) and LaTeX-generated artifacts (`*.aux`, `*.log`, `missfont.log`) to keep future commits focused on source/doc content.
- Why (decision/rationale): Needed to avoid accidentally pushing environment internals and transient build byproducts to GitHub.
- Impact or risk: Cleaner repository history and reduced commit noise; no runtime behavior change.
- Architecture updates: No architecture changes (non-architectural repository hygiene update).
- Validation status: Verified `.gitignore` now contains the new ignore patterns.
- Next steps: If desired, add additional TeX temporary extensions later (for example `.toc`, `.out`, `.fls`) if they begin appearing locally.

## [2026-04-27] Added experiment-settings summary section to formalization
- What changed: Extended `FORMALIZATION.tex` with a new section, ``Example Experiment Settings (Current Repository)'', summarizing current experiment families: (A) primary trained 2D Gaussian-to-2D Gaussian pipeline (coupling and train-profile variants), and (B) Bridge-SDE Gaussian-start preview setting (SDE form, purpose, and output folder pattern).
- Why (decision/rationale): Needed the PDF to explicitly distinguish what is already integrated in the main training/evaluation pipeline from what is currently a preview workflow.
- Impact or risk: Improves interpretability and onboarding; no code or runtime behavior change.
- Architecture updates: No architecture changes (documentation-only clarification of existing scope).
- Validation status: Recompiled `FORMALIZATION.pdf` from the updated source to confirm successful rendering.
- Next steps: When Bridge-SDE is promoted to full training experiments, add its exact training objective/config path in this section.

## [2026-04-27] Added bridge-SDE preview utilities, plotting, and notebook bootstrap
- What changed: Added shared `bridge_sde` simulation utilities (`simulate_bridge_sde_trajectories`, `sample_bridge_sde_at_times`) and bridge-focused plotting helpers (`plot_bridge_snapshot_grid`, `plot_bridge_y_spread`, `save_bridge_animation`); created `notebooks/bridge_sde_visualization.ipynb` as an interactive front-end using only shared project functions; added unit/smoke tests for simulation behavior and plotting artifact generation.
- Why (decision/rationale): Needed a pre-training validation workflow to verify that the intended bottleneck geometry (narrow middle then re-expand) is present before launching bridge benchmark training runs.
- Impact or risk: Improves reliability of benchmark design and visual diagnosis; no training pipeline behavior changed because automatic bridge-preview export remains intentionally postponed.
- Architecture updates: Updated `ARCHITECTURE.md` to include bridge-SDE preview workflow, new component responsibilities, and notebook role.
- Validation status: `pytest -q` passed (`19 passed`); executed a preview export run at `outputs/preview_bridge_sde/2026-04-27_12-13-13/` producing snapshot grid, y-spread curve, animation GIF, and summary text.
- Next steps: If preview shape is accepted, integrate bridge process into experiment data configs and run baseline-vs-A+B comparisons under OT and random coupling.

## [2026-04-27] Added code-equivalent AL training algorithm to formalization
- What changed: Expanded `FORMALIZATION.tex` Section 6.2 (Augmented Lagrangian) with a precise LaTeX pseudocode algorithm (`Algorithm 1`) describing mini-batch Stage A, Stage B, and Stage C updates, including explicit computation of \(c_k(\theta)\), AL loss assembly, Adam parameter steps, and multiplier clipping updates.
- Why (decision/rationale): Needed a more implementation-faithful and auditable description of how the current training loop executes in practice, beyond equation-only notation.
- Impact or risk: Improves reproducibility and onboarding clarity for readers mapping equations to code; no runtime or experimental behavior change.
- Architecture updates: No architecture changes (documentation-only clarification of existing training flow).
- Validation status: Recompiled `FORMALIZATION.pdf` from updated `FORMALIZATION.tex` to confirm LaTeX rendering.
- Next steps: If useful, add direct source-file/line references in an appendix table to provide one-click mapping from algorithm lines to implementation functions.

## [2026-04-26] Clarified scaling guidance for empirical intermediate OT metric
- What changed: Updated `ARCHITECTURE.md` with an explicit operational note that `eval_intermediate_ot_samples` is a CPU-default small budget and should be increased for more complex/non-Gaussian experiments; also noted approximate OT/Sinkhorn as an alternative at scale.
- Why (decision/rationale): Needed to prevent underpowered empirical \(W_2\) evaluation when scaling experiments and make this expectation explicit in architecture docs.
- Impact or risk: Improves interpretation reliability for future larger experiments; no code/runtime behavior change.
- Architecture updates: Clarified evaluation-metric scaling policy and refined the related limitation/mitigation text.
- Validation status: Documentation update only; no tests required.
- Next steps: When launching larger benchmarks, set `train.eval_intermediate_ot_samples` explicitly per experiment and log the chosen value in experiment notes.

## [2026-04-26] Added sample-based intermediate empirical OT metric and ran OT-vs-random comparison
- What changed: Added `intermediate_empirical_w2` and `intermediate_empirical_w2_avg` metrics based on exact discrete OT matching between generated samples and target samples at each constrained time; added config controls (`eval_intermediate_empirical_w2`, `eval_intermediate_ot_samples`); exposed an optional target-sampler hook in metric code for future non-Gaussian references; updated tests/docs; ran two A+B-only comparisons with new metric for `data=gaussian_ot` and `data=gaussian_random`.
- Why (decision/rationale): Needed an intermediate-time distribution metric that remains meaningful beyond Gaussian settings, and wanted direct OT-vs-random coupling comparison against endpoint transport quality.
- Impact or risk: Improves non-Gaussian-readiness of evaluation; empirical OT metric is more distribution-aware but computationally heavier, so it uses a smaller dedicated evaluation sample budget.
- Architecture updates: Evaluation now emits both Gaussian-proxy and sample-based empirical intermediate Wasserstein metrics; `README.md`, `ARCHITECTURE.md`, `FORMALIZATION.tex`, and `DISCUSSION.md` updated accordingly.
- Validation status: `pytest -q` passed (`15 passed`); runs completed at `outputs/2026-04-26/comparison_ot_empirical_ab_only/19-59-09/` and `outputs/2026-04-26/comparison_random_empirical_ab_only/19-59-23/`.
- Next steps: Run 3-seed repeats for OT and random coupling with the new metric to test whether coupling-dependent trends are stable.

## [2026-04-26] Added intermediate-time Gaussian Wasserstein evaluation metric
- What changed: Implemented new evaluation metrics `intermediate_w2_gaussian` (per constrained time) and `intermediate_w2_gaussian_avg` by integrating the learned velocity field up to each \(t_k\) and computing Gaussian \(W_2\) distance to analytic target distributions; updated metric aggregation in training summaries and comparison outputs; added unit tests in `tests/test_metrics.py`; updated docs (`ARCHITECTURE.md`, `FORMALIZATION.tex`, `README.md`, `DISCUSSION.md`) and logged validation smoke run.
- Why (decision/rationale): Needed a distribution-level intermediate-time metric to assess whether learned dynamics recover better full marginals along the path, beyond moment residual norms alone.
- Impact or risk: Improves interpretability of path quality along time; metric is a Gaussian-\(W_2\) proxy (moment-matched approximation) rather than exact empirical OT between full sample sets.
- Architecture updates: Evaluation layer now includes intermediate distribution diagnostics based on Euler trajectory snapshots and closed-form Gaussian Wasserstein computation.
- Validation status: `pytest -q` passed (`12 passed`); smoke comparison run completed at `outputs/2026-04-26/comparison_metric_check_smoke/19-51-55/` with new fields present in `comparison.json`; regenerated `FORMALIZATION.pdf`.
- Next steps: Use this metric in upcoming ablations (OT vs random coupling, A+B vs A+B+C) and check whether lower intermediate \(W_2\) correlates with better final transport.

## [2026-04-26] Added configurable coupling (`ot`/`random`) and ran random-coupling A+B comparison
- What changed: Added configurable data coupling mode with new `data.coupling` support (`ot` and `random`), including `configs/data/gaussian_random.yaml`; updated training/evaluation/plot sampling to use the selected coupling; added coupling metadata in run summaries/comparison meta; updated tests and README command examples; ran `experiment=comparison train=ab_only data=gaussian_random experiment.label=comparison_random_coupling`.
- Why (decision/rationale): Needed to evaluate baseline vs constrained A+B when pair coupling is random instead of exact OT, as requested for a new ablation axis.
- Impact or risk: Enables direct coupling ablations without changing core objectives; random coupling can degrade transport quality even when intermediate constraints improve.
- Architecture updates: Updated `ARCHITECTURE.md` to document coupling as a configurable data component responsibility; added experiment log entry in `EXPERIMENTS.md`; added discussion notes in `DISCUSSION.md`.
- Validation status: `pytest -q` passed (`9 passed`); random-coupling comparison run completed at `outputs/2026-04-26/comparison_random_coupling_ab_only/19-45-50/` with `comparison.json` generated.
- Next steps: Run multi-seed random-coupling checks to determine whether the observed constraint-vs-transport tradeoff is consistent or seed-specific.

## [2026-04-26] Added experiment index governance with `EXPERIMENTS.md`
- What changed: Added root `EXPERIMENTS.md` to map experiment output folders to experiment intent; updated `AGENTS.md` to require logging every new run/sweep in `EXPERIMENTS.md`; updated `ARCHITECTURE.md` file-responsibility table to include the new document.
- Why (decision/rationale): Needed a persistent, human-readable registry of experiments so result folders are understandable without reconstructing CLI history.
- Impact or risk: Improves reproducibility and onboarding for experiment analysis; low risk because this is documentation/process only.
- Architecture updates: Documentation architecture updated to include `EXPERIMENTS.md` as a governance artifact and repository map entry.
- Validation status: Verified docs consistency across `AGENTS.md`, `ARCHITECTURE.md`, `PROJECT_STATE.md`, and `EXPERIMENTS.md`.
- Next steps: Keep adding every new run/sweep to `EXPERIMENTS.md` in the same session it is executed.

## [2026-04-26] Ran A+B-only hyperparameter sweep (no Stage C)
- What changed: Executed a 16-variant sweep under `outputs/ab_only_sweep/` with `stage_c_steps=0`, tuning `rho`, `lr_g`, `alpha`, and `beta`; produced consolidated `outputs/ab_only_sweep/summary.tsv`.
- Why (decision/rationale): Needed to evaluate whether A+B-only training can improve constraint satisfaction while keeping transport quality close to baseline flow matching.
- Impact or risk: Confirmed that removing Stage C improves stability, but current A+B tuning still shows a constraint-vs-transport tradeoff; no variant simultaneously beat baseline on both metrics in this sweep.
- Architecture updates: No code architecture changes; experiment-only analysis.
- Validation status: All 16 variants completed successfully with per-variant `comparison.json` artifacts.
- Next steps: Run a focused second-round A+B sweep around the two best-constraint settings (`rho1_lrg3e4_alpha2p0`, `rho1_lrg3e4_beta0p10`) with multi-seed evaluation and adjusted regularization targets.

## [2026-04-26] Added explicit A+B-only run profile and output labeling
- What changed: Added `configs/train/ab_only.yaml` (`stage_c_steps=0`), added config labels for experiment/train groups, changed Hydra run directory pattern to `outputs/<date>/<experiment_label>_<train_label>/<time>/`, and added `comparison.meta` fields (`stage_c_enabled`, stage step counts, labels).
- Why (decision/rationale): Needed a reliable way to identify A+B-only runs directly from output paths and JSON files without manually inspecting full configs.
- Impact or risk: Improves experiment traceability and ablation clarity; no change to baseline/constrained core objectives.
- Architecture updates: Updated `ARCHITECTURE.md` and `README.md` to reflect output naming and A+B-only ablation workflow; added note in `DISCUSSION.md`.
- Validation status: `pytest -q` passed (`7 passed`); verified run with `python3 scripts/run_experiment.py experiment=comparison train=ab_only output.save_plots=false` produced folder `outputs/2026-04-26/comparison_ab_only/...` and `comparison.meta.stage_c_enabled=false`.
- Next steps: Optionally add a dedicated `experiment=comparison_ab_only` alias for even simpler CLI usage.

## [2026-04-26] Extended LaTeX formalization with metric definitions
- What changed: Added a new section in `FORMALIZATION.tex` that defines how all reported metrics are computed (`constraint_residual_norms`, `constraint_residual_avg`, `cfm_val_loss`, `path_energy_proxy`, and transport metrics), including the Euler transport evaluation procedure.
- Why (decision/rationale): Needed metric definitions in the PDF so training and evaluation outputs are mathematically explicit and auditable.
- Impact or risk: Improves interpretability of experiment outputs; no code behavior changes.
- Architecture updates: No architecture changes; documentation-only update.
- Validation status: Recompiled successfully to `FORMALIZATION.pdf` with `pdflatex`.
- Next steps: Optionally add a parallel metric section for a simpler two-stage/Lipschitz variant if we implement that alternative.

## [2026-04-26] Added full LaTeX formalization of current optimization loop
- What changed: Created `FORMALIZATION.tex` with a complete mathematical write-up of the current implementation: path parameterization, target moments, augmented-Lagrangian objective, CFM objective, and Stage A/B/C training loop.
- Why (decision/rationale): Needed a transparent formal document to understand exactly what the current code is optimizing and why it is more complex than a simple path-first then flow-matching pipeline.
- Impact or risk: Improves interpretability and onboarding for method development; no runtime behavior changed.
- Architecture updates: No architecture/code changes; documentation-only addition.
- Validation status: Checked for consistency with current formulas and training stages in the implementation files; compiled successfully with `pdflatex` (using local `TEXMFVAR`) to produce `FORMALIZATION.pdf`.
- Next steps: If desired, simplify the training strategy in code toward a stricter two-stage variant and compare against the current Stage C joint finetune approach.

## [2026-04-26] Ran default-budget comparison and constrained hyperparameter sweeps
- What changed: Executed `train=default` comparison plus targeted constrained sweeps over `rho`, `eta_joint`, `stage_c_steps`, and `lr_g`; then validated leading candidates over seeds `{3, 7, 11}`.
- Why (decision/rationale): Needed to test whether constrained training improves intermediate-time moment residuals versus baseline and identify practical hyperparameters.
- Impact or risk: Found major improvement opportunities over the current default constrained setup, but stability across seeds remains mixed and constrained mode does not yet consistently beat baseline.
- Architecture updates: No code architecture changes; methodology findings were captured in `DISCUSSION.md`.
- Validation status: Full suite still passes (`pytest -q` -> `7 passed`); long-run metrics saved under `outputs/` including `outputs/hparam_sweep/` and `outputs/seed_check/`.
- Next steps: Promote `rho=1.0` and reduced `lr_g` candidates as new tuning starting points, then run broader seed sweeps and consider Stage C stabilization changes if baseline-beating consistency remains weak.

## [2026-04-26] Implemented constrained flow-matching v1 pipeline
- What changed: Added full `src/`, `configs/`, `scripts/`, and `tests/` implementation for baseline and constrained CFM, including exact discrete OT pairing, analytic Gaussian moment targets, augmented-Lagrangian constrained path training, and stage A/B/C optimization; added `README.md` run instructions and `.gitignore` for generated artifacts.
- Why (decision/rationale): Needed a decision-complete v1 implementation matching the selected method: endpoint-preserving path correction with explicit moment constraints and CPU-first reproducibility.
- Impact or risk: Enables immediate experiments and ablations; main risk is computational overhead from exact OT and higher-order autograd terms in constrained training.
- Architecture updates: Filled `ARCHITECTURE.md` with mathematical formulation, module map, and training flow; added `DISCUSSION.md` for alternatives and tradeoffs; updated `AGENTS.md` to enforce `DISCUSSION.md` maintenance.
- Validation status: `pytest -q` passed (`7 passed`), and `python3 scripts/run_experiment.py experiment=comparison train=smoke output.save_plots=true` executed successfully, producing baseline/constrained artifacts and comparison metrics.
- Next steps: Run longer training budget (`train=default`) for a fair baseline-vs-constrained comparison, then tune AL/regularization hyperparameters to improve constrained residuals relative to baseline.

## [2026-04-26] Documentation governance bootstrap
- What changed: Created `AGENTS.md`, `ARCHITECTURE.md`, and `PROJECT_STATE.md` at repository root to establish documentation governance.
- Why (decision/rationale): Need durable project memory and a consistent process contract before feature implementation begins.
- Impact or risk: Improves onboarding and continuity; low risk, but requires discipline to keep docs current.
- Architecture updates: Initialized `ARCHITECTURE.md` template and marked `FM/` as non-architectural environment tooling.
- Validation status: Documentation files created and reviewed for required sections and policy alignment.
- Next steps: Fill project-specific architecture content after project context is provided; start first implementation entry in this log.

## [2026-04-30] Implemented single-cell benchmark integration (EB 5D strict leaveout default)
- What changed:
  - Added `single_cell` data family with loader module `src/cfm_project/single_cell_data.py` supporting:
    - `.npz` (`pcs`, `sample_labels`) and optional `.h5ad` (`X_pca`, `day`),
    - normalized time indexing from sorted observed labels,
    - endpoint empirical pools (`t0`, `tT`) and per-time target pools/moment targets.
  - Added leaveout controls:
    - `experiment.protocol: strict_leaveout | no_leaveout`,
    - `experiment.holdout_index`, `experiment.holdout_indices`,
    - `data.constraint_time_policy: observed_nonendpoint_excluding_holdout | observed_nonendpoint_all`.
  - Generalized constrained moment features from 2D-only to \(d\)-dimensional mean + full covariance (`src/cfm_project/constraints.py`).
  - Extended empirical rollout evaluation to report:
    - `intermediate_empirical_w1`, `intermediate_empirical_w1_avg`,
    - `transport_endpoint_empirical_w1`,
    - holdout diagnostics `holdout_empirical_w2`, `holdout_empirical_w1` when holdout is active.
  - Enabled dimension-safe plotting for \(d>2\) via projection on first two dims with new artifacts:
    - `sample_paths_proj12.png`,
    - `rollout_marginal_grid_proj12.png`,
    - and stage-a projected variants where applicable.
  - Added single-cell presets:
    - `configs/data/single_cell_eb_5d.yaml`,
    - `configs/experiment/comparison_mfm_single_cell.yaml`,
    - `configs/train/single_cell_ab_only.yaml`.
  - Added benchmark orchestration script:
    - `scripts/run_single_cell_eb_benchmark.py` (loops holdouts and writes `benchmark_summary.json`).
- Why (decision/rationale):
  - Needed first-class single-cell benchmarking in the same in-repo comparison pipeline while preserving fair leaveout defaults and method comparability.
- Impact or risk:
  - Enables 6-way strict leaveout comparisons on empirical timestamped data without changing legacy bridge/gaussian workflows.
  - Exact OT metrics remain cubic in sample count, so larger single-cell runs may require approximate OT or lower eval budgets.
- Architecture updates:
  - Updated `ARCHITECTURE.md` (single-cell flow, strict leaveout policy, projected plotting).
  - Updated `DISCUSSION.md` (strict vs non-strict constraint policy rationale).
  - Updated `EXPERIMENTS.md` (new synthetic smoke run path).
- Validation status:
  - Targeted regression passed:
    - `pytest -q tests/test_constraints.py tests/test_metrics.py tests/test_single_cell_data.py tests/test_single_cell_pipeline.py tests/test_bridge_ab_pipeline.py` (`16 passed`)
    - `pytest -q tests/test_mfm_pipeline.py tests/test_stage_a_bridge_pipeline.py tests/test_smoke_training.py` (`8 passed`)
  - Added and passed new tests:
    - `tests/test_single_cell_data.py`,
    - `tests/test_single_cell_pipeline.py`.
  - Executed synthetic end-to-end single-cell comparison run:
    - `outputs/2026-04-30/single_cell_eb5d_synth_smoke/00-00-01/`.
- Next steps:
  1. Run `scripts/run_single_cell_eb_benchmark.py` on the real EB file path provided by user and collect per-holdout aggregates.
  2. Add CITE/MULTI presets and higher-dimensional profiles (50D/100D) after EB strict-leaveout baseline is stable.
  3. Consider optional Sinkhorn/approximate OT evaluation path for larger single-cell sample budgets.

## [2026-04-30] Executed real EB 5D strict-leaveout benchmark (holdouts 1/2/3, 6-way)
- What changed:
  - Ran the real-data benchmark script on:
    - `/Users/benpro/Documents/PHD/Projects/Neurips26/code/TrajectoryNet/data/eb_velocity_v5.npz`
  - Completed all holdouts (`1,2,3`) with the full 6-way method list:
    - `baseline`, `constrained`, `metric`, `metric_alpha0`, `metric_constrained_al`, `metric_constrained_soft`.
  - Produced per-holdout artifacts and aggregate summary at:
    - `outputs/2026-04-30/single_cell_eb_5d_strict_leaveout/22-25-27/benchmark_summary.json`.
- Why (decision/rationale):
  - Needed the first real benchmark pass (not synthetic smoke) to compare methods under strict leaveout.
- Impact or risk:
  - Delivered the requested benchmark outputs with comparable protocol across methods.
  - Current endpoint-vs-holdout tradeoff remains visible: baseline/alpha0 are stronger on endpoint W2, while metric-constrained variants are stronger on holdout/intermediate W2.
- Architecture updates:
  - No architecture changes (experiment execution only).
  - Experiment index updated in `EXPERIMENTS.md` with exact folder paths.
- Validation status:
  - Benchmark script completed successfully (exit code 0) and wrote:
    - root `benchmark_summary.json`,
    - per-holdout `comparison_mfm.json`,
    - per-mode plots including `sample_paths_proj12.png`, `rollout_marginal_grid_proj12.png`, and `rollout_empirical_w2.png`.
- Next steps:
  1. Run a focused sweep on `metric_constrained_al`/`metric_constrained_soft` (`mfm.moment_eta`, `train.rho`, `mfm.alpha`) to improve endpoint W2 without degrading holdout W2.
  2. Add an aggregate report script that emits a compact leaderboard table across bridge + single-cell benchmarks.

## [2026-04-30] Tuned constrained-metric single-cell settings and validated full-budget candidates
- What changed:
  - Executed a coarse tuning sweep for constrained-metric modes on EB strict leaveout (`holdout_index=2`) over:
    - `mfm.alpha in {0.4, 0.7, 1.0}`,
    - `mfm.moment_eta in {0.25, 0.5, 1.0, 2.0}`,
    - `train.rho in {1.0, 5.0, 15.0}` for AL.
  - Selected top candidates from sweep ranking and ran full-budget strict-leaveout validation across holdouts `1,2,3` with plots for:
    - `metric_constrained_al` tuned (`alpha=0.4`, `moment_eta=2.0`, `rho=15.0`),
    - `metric_constrained_soft` tuned (`alpha=0.4`, `moment_eta=2.0`).
  - Wrote aggregate comparison helper:
    - `outputs/2026-04-30/single_cell_metric_constrained_tuned_validation/summary_compare_default_vs_tuned.json`.
- Why (decision/rationale):
  - User requested direct tuning to improve the constrained metric variants and check whether the endpoint/intermediate tradeoff can be improved.
- Impact or risk:
  - `metric_constrained_al` improved versus its default on both endpoint and holdout/intermediate means:
    - endpoint W2: `1.1082 -> 1.0995`,
    - holdout W2: `1.2965 -> 1.2552`,
    - intermediate W2: `1.2908 -> 1.2396`.
  - `metric_constrained_soft` only improved marginally relative to its default and remains weaker than tuned AL.
  - Baseline endpoint remains best overall on this protocol (`1.0439`), so further tuning is still needed if endpoint parity is the primary objective.
- Architecture updates:
  - No architecture/code changes (experiment execution and analysis only).
  - Experiment paths logged in `EXPERIMENTS.md`.
- Validation status:
  - Sweep completed: `outputs/2026-04-30/single_cell_metric_constrained_tuning/22-43-49/`.
  - Full-budget validations completed:
    - `outputs/2026-04-30/single_cell_metric_constrained_tuned_validation/22-49-11/al_alpha0.4_eta2.0_rho15/`,
    - `outputs/2026-04-30/single_cell_metric_constrained_tuned_validation/22-49-39/soft_alpha0.4_eta2.0/`.
  - Plots generated for all tuned holdout runs.
- Next steps:
  1. Run a second, narrower AL sweep around `alpha in [0.2, 0.5]`, `moment_eta in [1.5, 3.0]`, `rho in [10, 25]` with full-budget holdout-2/3 checks.
  2. Add one tuned full 6-way benchmark root (same seed/protocol) so all methods are compared in a single artifact timestamp.

## [2026-04-30] Tuned constrained-only single-cell mode and validated on all holdouts
- What changed:
  - Executed constrained-only sweeps on EB strict leaveout:
    - reduced-budget exploratory sweep at `outputs/2026-04-30/single_cell_constrained_tuning/22-53-40/`,
    - full-budget sweep at `outputs/2026-04-30/single_cell_constrained_tuning_full/22-56-45/`.
  - Selected best full-budget holdout-2 setting:
    - `train.alpha=1.0`, `train.beta=0.05`, `train.rho=25.0`.
  - Ran full strict-leaveout validation (holdouts `1,2,3`) with plots at:
    - `outputs/2026-04-30/single_cell_constrained_tuned_validation/22-58-39/constrained_a1.0_b0.05_rho25/`.
  - Wrote aggregate comparison helper:
    - `outputs/2026-04-30/single_cell_constrained_tuned_validation/summary_compare_default_vs_tuned.json`.
- Why (decision/rationale):
  - User requested tuning specifically for the non-metric `constrained` method.
- Impact or risk:
  - Constrained mode improved substantially vs its previous default aggregate:
    - endpoint W2: `1.0997 -> 1.0022`,
    - holdout W2: `1.4636 -> 1.3239`,
    - intermediate W2: `1.5215 -> 1.2705`.
  - Tuned constrained now also improves over default baseline aggregate on this benchmark split:
    - baseline endpoint/holdout W2 means: `1.0439 / 1.4505`,
    - tuned constrained endpoint/holdout W2 means: `1.0022 / 1.3239`.
  - Potential risk: best hyperparameters were chosen from holdout-2 ranking before 3-holdout validation; additional seed robustness checks are still needed.
- Architecture updates:
  - No architecture/code changes (experiment execution and tuning only).
  - Experiment paths logged in `EXPERIMENTS.md`.
- Validation status:
  - Constrained-only full-budget sweep completed and ranked (`summary_ranked.json`).
  - Tuned constrained strict-leaveout validation completed for holdouts `1,2,3` with PNG artifacts in each holdout folder.
- Next steps:
  1. Run the tuned constrained config on additional seeds (e.g., `7`, `11`) to test robustness.
  2. Re-run one unified 6-way benchmark with tuned constrained included for direct side-by-side reporting.

## [2026-05-04] Implemented unsupervised soft pseudo-type constraints for single-cell training
- What changed:
  - Added a new pseudo-label module `src/cfm_project/pseudo_labels.py` that:
    - fits GMMs on whitened single-cell embeddings,
    - selects `K` with BIC + ARI stability,
    - caches/reuses fitted model parameters under `.cache/pseudo_labels`,
    - exposes a differentiable torch posterior scorer \(q_k(x)\).
  - Extended `single_cell_data.prepare_single_cell_problem_and_targets` to:
    - optionally build/load pseudo-label artifacts,
    - compute per-constrained-time pseudo targets \(\pi_t^{ps}\),
    - return pseudo metadata (`pseudo_labels_k`, cache info, `bic_by_k`, `stability_by_k`) and scorer.
  - Extended constrained training objectives to add pseudo residual terms (additive with moments):
    - `constrained`: adds pseudo AL term in Stage A and Stage C with independent multipliers (`train.pseudo_*`);
    - `metric_constrained_al`: adds pseudo AL term in Stage A;
    - `metric_constrained_soft`: adds pseudo soft squared-residual term in Stage A.
  - Added new summary outputs:
    - `pseudo_constraint_residual_norms`, `pseudo_constraint_residual_avg`,
    - `pseudo_labels_k`, `pseudo_labels_cache_path`, `pseudo_labels_cache_hit`,
    - `bic_by_k`, `stability_by_k`.
  - Added config interface:
    - `data.single_cell.pseudo_labels.*` in `configs/data/single_cell_eb_5d.yaml`,
    - `train.pseudo_eta`, `train.pseudo_rho`, `train.pseudo_lambda_clip` in train profiles.
  - Added dependency declaration `scikit-learn>=1.5.0,<2.0.0` in `pyproject.toml`.
  - Added/updated tests:
    - `tests/test_pseudo_labels.py`,
    - `tests/test_single_cell_data.py` (pseudo targets + cache reuse),
    - `tests/test_single_cell_stage_a_pipeline.py` (pseudo summary contract).
- Why (decision/rationale):
  - Needed a principled indicator-like constraint channel for unlabeled cell-state data; soft pseudo-types allow differentiable class-membership constraints without requiring curated annotations.
- Impact or risk:
  - Enables new unsupervised class-proportion constraints while preserving backward compatibility (`pseudo_eta=0`, `enabled=false` by default).
  - Risk remains that latent pseudo-types may not align with biological ontologies; diagnostics are surfaced and feature remains optional/additive.
- Architecture updates:
  - Updated `ARCHITECTURE.md` for pseudo-label component responsibilities, interaction flow changes, and pseudo-constraint math.
  - Updated `DISCUSSION.md` with method rationale/tradeoffs and smoke-run interpretation.
  - Updated `FORMALIZATION.tex` with a dedicated mathematical section for GMM posteriors, `K` selection, pseudo residuals, hybrid objectives, and algorithm order.
  - Updated `EXPERIMENTS.md` with new validation run folders and commands.
- Validation status:
  - `pytest -q tests/test_pseudo_labels.py tests/test_single_cell_data.py tests/test_single_cell_stage_a_pipeline.py tests/test_single_cell_pipeline.py tests/test_mfm_pipeline.py` (`19 passed`).
  - Executed pseudo-enabled Stage-A smoke run:
    - `outputs/2026-05-04/single_cell_pseudo_stagea_smoke/00-00-01/`.
  - Executed pseudo-enabled strict-leaveout A+B smoke run:
    - `outputs/2026-05-04/single_cell_pseudo_ab_smoke/00-00-01/`.
- Next steps:
  1. Run multi-seed holdout sweeps comparing `pseudo_eta` values against moment-only baselines.
  2. Add marker-gene/annotation-alignment diagnostics to evaluate pseudo-type biological interpretability.

## [2026-05-04] Executed Stage-A `metric_constrained_al` pseudo full-OT sweep and all-holdout validation
- What changed:
  - Ran the planned Stage-A-only pseudo sweep for `metric_constrained_al` (full-OT evaluation) at:
    - `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_stage_a_fullot/18-24-04/`.
  - Search phase:
    - holdout `2`, seeds `3/7/11`,
    - pseudo grid over `train.pseudo_eta`, `train.pseudo_rho`, `train.pseudo_lambda_clip`,
    - 18/18 runs completed successfully.
  - Final validation phase:
    - top-2 configs validated on holdouts `1/2/3` × seeds `3/7/11`,
    - summary at:
      - `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_stage_a_fullot/18-24-04/final_validation_summary.json`.
- Why (decision/rationale):
  - User requested a direct test of whether unsupervised pseudo constraints can beat simple constraints for `metric_constrained_al` under the strict full-OT Stage-A protocol.
- Impact or risk:
  - On holdout-2-only search aggregates, pseudo variants strongly improved the primary metric but worsened the holdout metric (expected since objective targeted constrained-time full-OT only).
  - On all-holdout validation, the winning pseudo config improved both tracked means versus the simple-constraint baseline:
    - baseline (simple constraints, 9 runs): primary `1.1340704282917455`, secondary `1.1598324896315004`,
    - winner (`eta1.00_rho5.0_clip100.0`, 9 runs): primary `1.129247562864637`, secondary `1.1573922965458503`,
    - deltas: primary `-0.004822865427108525`, secondary `-0.002440193085650133`.
  - Remaining risk: gains are modest in absolute magnitude; additional seeds or repeated runs can shift ranking.
- Architecture updates:
  - No architecture/code changes (experiment execution and result analysis only).
  - `EXPERIMENTS.md` updated with run roots, search/validation scope, and winner metrics.
- Validation status:
  - Search phase summary:
    - `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_stage_a_fullot/18-24-04/phase2_search_summary.json`.
  - Final validation summary:
    - `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_stage_a_fullot/18-24-04/final_validation_summary.json`.
  - Baseline-vs-final delta summary:
    - `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_stage_a_fullot/18-24-04/final_vs_baseline_summary.json`.
  - Sanity checks passed in all runs:
    - `pseudo_constraints_active=true`,
    - `pseudo_labels_k=9`,
    - pseudo-label cache reused (`pseudo_labels_cache_hit=true`).
- Next steps:
  1. Re-run the winner (`eta=1.0, rho=5.0`) on a fresh seed block (e.g., 5 additional seeds) to quantify confidence intervals.
  2. Compare winner vs simple constraints under a jointly weighted objective that explicitly includes both constrained-time and holdout targets.

## [2026-05-04] Executed Stage-A `constrained` pseudo full-OT evaluation on all holdouts
- What changed:
  - Ran `constrained` mode with pseudo constraints enabled (`pseudo_eta=1.0`, `pseudo_rho=5.0`, `pseudo_lambda_clip=100.0`) across holdouts `1/2/3` and seeds `3/7/11`:
    - `outputs/2026-05-04/single_cell_constrained_pseudo_stage_a_fullot/18-54-15/`.
  - Generated comparison artifact vs simple constrained baseline:
    - `outputs/2026-05-04/single_cell_constrained_pseudo_stage_a_fullot/18-54-15/baseline_vs_unsup_summary.json`.
- Why (decision/rationale):
  - User requested the same unsupervised constraint treatment for the other method family, with no sweep unless results were clearly poor.
- Impact or risk:
  - Overall holdout full-OT mean improved vs simple constrained baseline:
    - baseline `1.4286208909355773` -> unsup `1.4053413782439794` (delta `-0.023279512691597892`).
  - Improvements were uneven by holdout:
    - `p0.25`: improved,
    - `p0.50`: slightly worse,
    - `p0.75`: improved.
  - Variance remains high for constrained family (same qualitative risk as baseline constrained runs).
- Architecture updates:
  - No architecture/code changes (experiment execution only).
  - `EXPERIMENTS.md` updated with run root and summary artifact paths.
- Validation status:
  - 9/9 runs completed successfully (`summary.json`), pseudo constraints active and pseudo-label cache reused in all runs.
- Next steps:
  1. If tighter stability is needed for constrained+unsup, test a small pseudo-eta reduction (`0.5`) on holdout 1 where variance remained highest.
  2. Recompute the multi-method leaderboard including both unsup rows and decide whether to adopt one or both unsup variants as defaults.

## [2026-05-04] Executed Stage-A pseudo extension for all remaining benchmark methods and generated full baseline-vs-unsup table
- What changed:
  - Ran additional pseudo-enabled Stage-A strict-leaveout full-OT experiments for:
    - `metric`,
    - `metric_alpha0`,
    - `metric_constrained_soft`.
  - Run root:
    - `outputs/2026-05-04/single_cell_all_methods_pseudo_stage_a_fullot/19-03-38`.
  - Scope:
    - holdouts `1/2/3`, seeds `3/7/11` (`27` runs total).
  - Generated consolidated all-method comparison artifacts:
    - `outputs/2026-05-04/single_cell_all_methods_pseudo_stage_a_fullot/19-03-38/full_table_baseline_vs_unsup.json`
    - `outputs/2026-05-04/single_cell_all_methods_pseudo_stage_a_fullot/19-03-38/full_table_baseline_vs_unsup.csv`
    - `outputs/2026-05-04/single_cell_all_methods_pseudo_stage_a_fullot/19-03-38/full_table_baseline_vs_unsup.md`
- Why (decision/rationale):
  - User requested extending unsupervised pseudo-constraint evaluation to every method in the reported table and returning the full side-by-side metrics table.
- Impact or risk:
  - `metric_constrained_soft+unsup` changed only marginally vs baseline.
  - `metric+unsup` and `metric_alpha0+unsup` are numerically unchanged in this protocol because pseudo constraints are not active in those method families (`pseudo_constraints_active=false` by design).
  - `metric_constrained_al+unsup` and `constrained+unsup` remain the only families with active pseudo updates in this benchmark set.
- Architecture updates:
  - No architecture/code changes (experiment execution and aggregation only).
  - `EXPERIMENTS.md` updated with the new run root and artifacts.
- Validation status:
  - `27/27` runs completed successfully (`n_failures=0`).
  - All expected summary artifacts were emitted and aggregated.
- Next steps:
  1. If we want pseudo effects on unconstrained families, define and implement an explicit pseudo objective for `metric`/`metric_alpha0`; currently flags are no-op there by design.
  2. If desired, run one robustness block with additional seeds for the two genuinely active pseudo families (`metric_constrained_al`, `metric_constrained_soft`).

## [2026-05-04] Executed fixed-`K` sensitivity sweep for `metric_constrained_al+unsup` and selected a new winner
- What changed:
  - Ran the full planned fixed-`K` sweep with small eta retune:
    - run root: `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_k_sensitivity_stage_a_fullot/19-56-10`.
  - Search phase:
    - holdout `2`, seeds `3/7/11`,
    - `K in {4,6,8,9,10,12}` using `k_min=k_max=K`,
    - `pseudo_eta in {0.5,1.0}` with `pseudo_rho=5.0`, `pseudo_lambda_clip=100.0`,
    - `36` runs.
  - Final validation phase:
    - top-2 configs over holdouts `1/2/3` x seeds `3/7/11`,
    - `18` runs.
  - Generated comparison artifacts:
    - `search_summary.json`, `final_validation_summary.json`,
    - `new_best_vs_previous_and_simple_summary.json`,
    - `auto_k9_vs_best_fixed_k_table.md`.
- Why (decision/rationale):
  - User requested testing whether explicitly controlling latent pseudo-class count `K` can further improve `metric_constrained_al+unsup` beyond auto-selected `K=9`.
- Impact or risk:
  - Winner moved from auto-`K=9` to fixed `K=8` (same eta/rho/clip as prior winner).
  - Improvement exists but is small:
    - vs prior auto-`K` winner:
      - primary mean delta: `-0.00020633384349211248`,
      - holdout mean delta: `-0.0002700990253654556`.
    - vs simple baseline:
      - primary mean delta: `-0.005029199270600859`,
      - holdout mean delta: `-0.0027102921110155886`.
  - Risk remains that this small margin may not be stable under larger seed blocks.
- Architecture updates:
  - No architecture/code changes (experiment execution and aggregation only).
  - `EXPERIMENTS.md`, `PROJECT_STATE.md`, and `DISCUSSION.md` updated with the sweep and outcomes.
- Validation status:
  - `54/54` runs completed (`0` failures).
  - Per-run sanity checks satisfied:
    - `pseudo_constraints_active=true`,
    - `pseudo_labels_k == K`,
    - expected cache behavior (`cache_hit_rate` typically `2/3` for first eta at each K and `1.0` for second eta).
- Next steps:
  1. Run an extra robustness block (for example 5 additional seeds) on `k8_eta1.0` vs prior auto-`K=9` to establish confidence intervals.
  2. If robustness is weak, keep auto-`K` as default and treat fixed-`K=8` as an optional tuning profile.

## [2026-05-04] Executed high-eta dichotomy sweep at fixed `K=9` for `metric_constrained_al+unsup`
- What changed:
  - Ran a new Stage-A strict-leaveout full-OT sweep at fixed `K=9` with exponentially increasing pseudo weight:
    - run root: `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_k9_eta_dichotomy_stage_a_fullot/20-43-25`.
    - eta ladder: `1,2,4,8,16,32,64,100`.
    - seeds: `3,7,11`; holdout: `2`.
    - fixed knobs: `pseudo_rho=5.0`, `pseudo_lambda_clip=100.0`.
  - Produced summary artifacts:
    - `search_summary.json`,
    - `search_summary_table.md`,
    - `eta_ordered_with_deltas.md`.
- Why (decision/rationale):
  - User requested testing whether substantially larger pseudo weight (`eta`) can continue improving performance for `metric_constrained_al+unsup` rather than stopping near `eta=1`.
- Impact or risk:
  - On holdout `2`, increasing eta up to `100` gave strong monotonic improvement in both tracked full-OT means:
    - primary: `1.055324 -> 0.906111` (delta `-0.149213`),
    - holdout: `1.363583 -> 1.200209` (delta `-0.163373`).
  - Best config in this sweep is `eta=100`; no instability/failures observed in these `24` runs.
  - Risk: this result is currently validated on holdout `2` only; cross-holdout behavior may differ.
- Architecture updates:
  - No architecture/code changes (experiment execution and aggregation only).
  - `EXPERIMENTS.md`, `PROJECT_STATE.md`, and `DISCUSSION.md` updated with the sweep outcomes.
- Validation status:
  - `24/24` runs succeeded (`0` failures).
  - per-run sanity checks remained valid (`pseudo_constraints_active=true`, `pseudo_labels_k=9`, cache hits true).
- Next steps:
  1. Run a full validation block for top high-eta configs (for example `eta in {64,100}`) across holdouts `1/2/3` x seeds `3/7/11`.
  2. If high eta still dominates across holdouts, decide whether to expand eta search above `100` or retune `pseudo_rho`/`lambda_clip` to avoid hidden saturation effects.

## [2026-05-04] Continued high-eta sweep past 100 and stopped at first non-improving eta
- What changed:
  - Continued fixed-`K=9` high-eta sweep beyond `eta=100` using an auto-stop rule:
    - previous root: `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_k9_eta_dichotomy_stage_a_fullot/20-43-25`,
    - continuation root: `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_k9_eta_dichotomy_stage_a_fullot/21-01-36`.
  - New eta levels executed (holdout `2`, seeds `3/7/11`):
    - `128, 160, 200, 256, 320, 400`.
  - Auto-stop criterion:
    - stop when improvement in mean primary metric vs previous eta `<= 5e-4`.
- Why (decision/rationale):
  - User requested to continue increasing eta until objective no longer improved.
- Impact or risk:
  - Strong gains persisted up to `eta=320`:
    - `eta=100` mean primary `0.906111`,
    - `eta=320` mean primary `0.826541`.
  - At `eta=400`, mean primary worsened to `0.833223`, so progression stopped.
  - Best seen in this continuation:
    - `eta=320` with holdout mean `1.085330`.
  - Risk remains that optimal eta on holdout `2` may not transfer perfectly to holdouts `1/3`.
- Architecture updates:
  - No architecture/code changes (experiment continuation + reporting only).
  - `EXPERIMENTS.md`, `PROJECT_STATE.md`, and `DISCUSSION.md` updated with stop-rule and outcomes.
- Validation status:
  - `18/18` new runs completed successfully (`0` failures).
  - pseudo constraints remained active in all runs; pseudo-label cache hits were true.
- Next steps:
  1. Validate `eta in {256,320,400}` across holdouts `1/2/3` x seeds `3/7/11` to confirm global optimum and robustness.
  2. If `eta=320` remains best, run a local refinement around it (for example `280, 320, 360`) before finalizing default sweep range.

## [2026-05-04] Implemented 0.5-only pseudo-fit/constraint protocol and ran Stage-A 3-method block
- What changed:
  - Added explicit single-cell time-selection controls:
    - `data.single_cell.constraint_times_normalized` to override policy-derived constrained times,
    - `data.single_cell.eval_times_normalized` to decouple Stage-A interpolant evaluation times,
    - `data.single_cell.pseudo_labels.fit_times_normalized` to restrict pseudo-label GMM fit times.
  - Added strict validation for requested normalized times and pseudo-fit subset size.
  - Wired single-cell prep to emit and propagate:
    - resolved constraint times,
    - resolved eval times,
    - pseudo-fit times and pseudo-fit sample count.
  - Updated Stage-A interpolant/full-OT evaluation path to consume decoupled `data.interpolant_eval_times`.
  - Added/updated tests covering:
    - pseudo cache key behavior under fit-subset changes,
    - explicit time overrides (`constraint/eval/pseudo-fit`) in single-cell prep,
    - Stage-A pipeline behavior with constraints-only-at-0.5 and eval-on-0.25/0.5/0.75.
- Why (decision/rationale):
  - User requested a protocol where pseudo labels are learned only from the \(t=0.5\) marginal, constraints are enforced only at \(t=0.5\), and metrics are still observed on all three intermediate snapshots.
- Impact or risk:
  - New functionality is backward compatible by default:
    - no override => prior policy-derived constraints and prior evaluation semantics.
  - New protocol run completed successfully across three seeds and three constrained families:
    - run root: `outputs/2026-05-04/single_cell_t05_only_pseudo_stage_a_fullot/21-39-57`,
    - all summaries showed `single_cell_constraint_times=[0.5]`, `single_cell_eval_times=[0.25,0.5,0.75]`, `single_cell_pseudo_fit_times=[0.5]`.
  - Observed overall full-OT means:
    - `metric_constrained_al` best among tested families (`0.878618 ± 0.007908`).
- Architecture updates:
  - `ARCHITECTURE.md` updated to include explicit single-cell time-selection controls and decoupled Stage-A interpolant evaluation flow.
- Validation status:
  - Targeted tests passed: `16 passed` (`tests/test_pseudo_labels.py`, `tests/test_single_cell_data.py`, `tests/test_single_cell_stage_a_pipeline.py`).
  - Experiment block succeeded: `9/9` method-runs (`3 seeds x 3 methods`, no failures).
- Next steps:
  1. Compare this new protocol directly against previous all-times pseudo-fit baseline on identical methods/seeds.
  2. If needed, run a small eta retune under the new 0.5-only protocol (for example `eta in {100,320,400}`).
