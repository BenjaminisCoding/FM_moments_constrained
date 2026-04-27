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
