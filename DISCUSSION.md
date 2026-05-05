# DISCUSSION.md

## Purpose
Track research alternatives, tradeoffs, and rejected directions separately from implementation history.
`PROJECT_STATE.md` records what was done; this file records why design choices were made.

## Repository Boundary
`FM/` is environment tooling (Python virtual environment), not project source architecture.
Methodological discussions in this file only refer to source code under `src/`, `configs/`, `scripts/`, and `tests/`.

## Current Question
How to learn physically plausible flow-matching paths under intermediate-time moment constraints while preserving endpoints and keeping training stable.

## Candidate Directions Considered
### 1) Penalty-only constrained path learning
- Idea: train correction network with fixed weighted penalty on moment residuals.
- Benefit: simplest implementation.
- Risk: sensitive weighting; weak guarantee of constraint satisfaction.
- Status: not selected for v1; kept as possible ablation.

### 2) Augmented Lagrangian constrained path learning
- Idea: optimize path prior plus Lagrange multiplier and quadratic penalty terms for each constrained time.
- Benefit: stronger and more stable constraint enforcement than fixed penalty.
- Risk: requires tuning `rho` and multiplier clipping.
- Status: selected for v1.

### 3) Fully joint training from iteration 1
- Idea: train path model and velocity model simultaneously from scratch.
- Benefit: single stage.
- Risk: unstable early optimization when constraints are still largely violated.
- Status: not selected for v1.

### 4) Strict two-stage only
- Idea: train path model first, freeze it, then train velocity model.
- Benefit: straightforward.
- Risk: mismatch between path and velocity may remain unresolved.
- Status: superseded by staged schedule with final joint refinement.

## Chosen v1 Direction
- Path family:
  \[
  x_t = (1-t)x_0 + t x_1 + t(1-t) g_\theta(t,x_0,x_1)
  \]
  to preserve endpoint constraints exactly.
- Constraint mechanism: augmented Lagrangian over residuals
  \[
  c_k(\theta) = \mathbb{E}[\phi(x_{t_k})] - m_k
  \]
  with updates
  \[
  \lambda_k \leftarrow \lambda_k + \rho c_k.
  \]
- Regularization: energy + temporal smoothness instead of Lipschitz-in-time only.
- Training schedule: Stage A (`g` pretrain) -> Stage B (`v` train) -> Stage C (joint finetune).
- Benchmark: 2D Gaussian-to-Gaussian with exact-formula intermediate moments and exact discrete OT pairings.

## Why This Direction
- Preserves endpoint validity by construction.
- Balances constraint satisfaction with path plausibility.
- Reduces collapse risk relative to underconstrained single-time/single-feature setups.
- Provides a controlled synthetic benchmark before real-data deployment.

## Open Follow-ups
- Add penalty-only and Lipschitz-only ablations for robustness comparison.
- Evaluate noisy moment targets instead of exact formula targets.
- Extend feature map beyond mean/covariance to nonlinear moments.

## Method Note (2026-05-04 Full-OT metric backend selection: `exact_lp` vs POT `pot_emd2`)
- Decision:
  - keep exact-LP (`scipy.optimize.linprog`) as the default backend;
  - add POT `ot.emd2` as an explicit alternative backend for full-set OT metric computation in Stage-A and A+B single-cell evaluation.
- Why:
  - recent runtime checks showed POT can solve the benchmark-sized metric in seconds, making it practical for repeated metric evaluation.
  - preserving exact-LP as default avoids forcing a new mandatory dependency and keeps historical behavior unchanged.
- Tradeoff:
  - `exact_lp` offers a single in-repo deterministic LP path but can be slower/scaling-limited.
  - `pot_emd2` is typically faster in this setup and exposes iteration control (`numItermax`), but depends on POT availability and convergence settings.
- Configuration contract:
  - `train.eval_full_ot_method in {exact_lp, pot_emd2}`;
  - optional `train.eval_full_ot_num_itermax` used only by the POT backend.
  - default policy (as of 2026-05-04): train profiles now default to `pot_emd2`, while keeping `exact_lp` as a manual fallback path.

## Method Note (2026-05-04 POT `ot.emd2` runtime behavior on Stage-A holdout metric)
- Question tested:
  - whether increasing `numItermax` can be used as a practical knob to target roughly one-minute runtime for plan-conditioned holdout-time W2 evaluation.
- Setup:
  - EB 5D single-cell data, holdout `t=0.5`, source from weighted global-OT support pushforward, target from full holdout pool.
- Observation:
  - `ot.emd2` showed iteration-limit warnings at low `numItermax` (`100k`, `200k`), then converged by `~400k`.
  - After convergence, runtime was essentially flat (`~3.2–3.3s`) despite much larger `numItermax`.
- Implication:
  - For this problem size, `numItermax` is mainly a convergence floor, not a reliable runtime dial once optimality is reached.
  - If longer stress-runtime benchmarks are needed, increase problem size or switch benchmark objective (for example denser supports or repeated solves), rather than only scaling `numItermax`.

## Method Note (2026-05-03 Global Kantorovich coupling + robust full-set OT metrics)
- Training-coupling decision:
  - added `data.coupling=ot_global` for single-cell empirical training.
  - one balanced global Kantorovich plan is solved once (exact LP), cached, and represented as sparse support edges plus masses.
  - SGD then samples endpoint pairs from that sparse support \emph{with replacement}.
- Why this is useful:
  - removes per-step re-solving noise from batch Hungarian pairing;
  - gives a single consistent coupling prior across Stage A/B/C updates;
  - keeps each optimizer step stochastic while respecting global-plan weights.
- Tradeoff:
  - large up-front solve cost and potential memory pressure for large endpoint pools;
  - requires explicit cache controls and solver-size guards.

- Evaluation decision:
  - keep legacy sampled Hungarian metrics (`*_empirical_w2*`) unchanged for historical comparability;
  - add robust full-set balanced OT metrics as additive keys.
- Robust metric semantics:
  - Stage A: generated measure uses weighted global-plan support edges interpolated at time \(t\), compared to full observed target pool at that time.
  - Stage A+B: generated rollout uses full source endpoint pool pushforward, compared to full observed target pool.
- Why both metric families:
  - legacy metrics are cheap and match past runs;
  - robust metrics are more faithful distribution-level checks under unequal set sizes and weighted sources.
- Benchmark policy update:
  - stage-A leaderboard sorting uses robust holdout key first, with fallback to legacy holdout key only when robust key is missing.

## Metric Design Note (2026-04-26)
- Added intermediate-time Gaussian Wasserstein diagnostics (`intermediate_w2_gaussian`) computed from generated trajectory moments versus analytic Gaussian targets.
- Added intermediate-time sample-based empirical OT diagnostics (`intermediate_empirical_w2`) using exact discrete matching between generated and target samples at each constrained time.
- Implementation detail: empirical metric exposes a target-sampler hook so non-Gaussian reference samplers can be plugged in later without changing the OT-metric code path.
- Chosen tradeoff:
  - Gaussian metric is cheap and stable but only moment-aware.
  - Empirical OT metric is distribution-aware but computationally heavier (\(O(N^3)\) matching), so it uses a separate smaller evaluation sample budget.

## Empirical Notes (2026-04-26 Tuning Session)
- Default constrained setting (`rho=5.0`, `eta_joint=0.05`) under `train=default` was unstable in this benchmark and often much worse than baseline on constraint residuals.
- Inspection of constrained histories showed Stage A can reach low residuals, but Stage C often degrades constraints, indicating a Stage C balancing issue.
- Best single-seed constrained residual in the sweep was obtained with `rho=1.0`, `eta_joint=0.5` (seed 7), substantially better than the current constrained default.
- Multi-seed checks showed improved robustness versus the default constrained setup when lowering `rho` and lowering `lr_g`, but baseline-beating behavior was not yet consistent across all tested seeds.
- Practical next tuning defaults:
  - start from `rho=1.0`;
  - reduce `lr_g` to around `3e-4`;
  - keep joint finetune but continue tuning `eta_joint` around `0.3-1.0`;
  - prioritize seed-robust evaluation before locking new defaults.
- Engineering note: an explicit `train=ab_only` profile is now available to disable Stage C and isolate Stage A+B behavior in labeled output folders.

## Empirical Notes (2026-04-26 A+B-only Sweep)
- A 16-variant A+B-only sweep (`stage_c_steps=0`) was run over `rho`, `lr_g`, `alpha`, and `beta`.
- Best constraint scores in this sweep came from:
  - `rho1_lrg3e4_alpha2p0` (constraint delta < 0 versus baseline),
  - `rho1_lrg3e4_beta0p10` (constraint delta < 0 versus baseline).
- Best transport scores stayed closer to baseline with:
  - `rho1_lrg5e4`, `rho_0p5`, and `lrg_5e4`,
  but these did not improve constraints.
- Conclusion: even without Stage C, current objective/tuning still exposes a clear tradeoff between intermediate-time constraint matching and final transport quality.

## Empirical Notes (2026-04-26 Random Coupling Ablation)
- Ran `experiment=comparison`, `train=ab_only`, `data=gaussian_random` to replace OT pairing with random pairing while keeping the same marginal setup and constraints.
- Observed behavior on this run:
  - constrained improved constraint residual average versus baseline (`0.7488` vs `0.9755`);
  - constrained worsened transport score versus baseline (`0.6762` vs `0.4068`);
  - constrained also had higher CFM validation loss than baseline (`3.8522` vs `2.8328`).
- Interpretation: random coupling changes the tradeoff but does not remove it; in this run, constraint fitting improved at the cost of downstream transport quality.

## Empirical Notes (2026-04-26 OT vs Random with Empirical Intermediate OT Metric)
- Added the sample-based intermediate metric and ran:
  - `comparison_ot_empirical_ab_only`,
  - `comparison_random_empirical_ab_only`.
- OT coupling run:
  - baseline had better transport score than constrained (`0.1776` vs `0.3352`);
  - baseline also had better intermediate metrics (`intermediate_w2_gaussian_avg`: `0.2653` vs `0.3536`, `intermediate_empirical_w2_avg`: `0.4203` vs `0.5024`).
- Random coupling run:
  - constrained improved intermediate metrics versus baseline (`intermediate_w2_gaussian_avg`: `0.2707` vs `0.4336`, `intermediate_empirical_w2_avg`: `0.4494` vs `0.6386`);
  - but constrained worsened transport score (`0.6762` vs `0.4068`).
- Interpretation: coupling choice strongly changes where the method wins; with random coupling we can improve intermediate distribution fit while still hurting endpoint transport, whereas OT coupling already provides strong path geometry for baseline.

## Method Note (2026-04-27 Bridge Preview Bootstrap)
- Before integrating the new bridge benchmark into training, we added a pre-training visualization workflow (shared simulation + plotting utilities and a notebook).
- Decision rationale: verify that the intended geometric prior (narrow bottleneck around mid-time with re-expansion by terminal time) is visually and statistically present before spending compute on model training.

## Method Note (2026-04-27 Bridge Stage-A-Only Evaluation Choice)
- For the first bridge benchmark integration pass, we intentionally evaluate only Stage A (`g_\theta`) and skip velocity learning (Stage B/C).
- Rationale:
  - isolates whether the learned interpolant itself improves intermediate marginals;
  - avoids conflating interpolant quality with velocity-network approximation error;
  - gives clearer diagnostics before re-introducing joint objectives.
- Chosen metric framing:
  - compare \emph{linear interpolant} vs \emph{learned interpolant} against true bridge marginals at constrained times using empirical OT;
  - mark velocity-rollout metrics as out-of-scope (`null`) in Stage-A-only summaries to prevent false conclusions.
- Early observation from first runs:
  - OT coupling improved average interpolant empirical OT relative to linear baseline in this setup;
  - random coupling did not improve average interpolant empirical OT under the same budget.

## Method Note (2026-04-27 Bridge Stage-A+B Enablement Decision)
- Decision: enable bridge runs for `train=ab_only` (Stage A + Stage B, Stage C disabled) and keep Stage-A-only path unchanged.
- Rationale:
  - Stage-A-only remains useful to isolate path/interpolant behavior.
  - We also need rollout metrics from learned velocity fields to compare baseline vs constrained on bridge data in comparison mode.
- Chosen bridge rollout metric contract:
  - constrained-time empirical OT: `intermediate_empirical_w2` and `intermediate_empirical_w2_avg`;
  - endpoint empirical OT: `transport_endpoint_empirical_w2`;
  - bridge `transport_score := transport_endpoint_empirical_w2`.
- Why this contract:
  - bridge targets are empirical/non-Gaussian in general, so endpoint mean/cov errors can be misleading;
  - using empirical OT consistently at intermediate times and endpoint gives a family-consistent transport objective proxy.
- Tradeoff:
  - empirical OT is computationally heavier, so `eval_intermediate_ot_samples` still controls precision-vs-cost.
  - Stage C on bridge remains deferred to avoid reintroducing instability before A+B behavior is characterized.

## Empirical Note (2026-04-27 Bridge OT high-sample sweep)
- Ran a two-phase A+B sweep on bridge OT with increased sample budgets (`batch_size=512`, `eval_intermediate_ot_samples=1024`, `eval_transport_samples=4000`).
- Multi-seed outcome: several configs satisfy the current Pareto gate (`delta_intermediate < 0` and `delta_endpoint <= +0.03`), so we can improve intermediate transport while keeping endpoint close to baseline.
- Current best mean intermediate config from phase-2 aggregate:
  - `rho=0.5`, `alpha=1.0`, `lr_g=1e-3`, `lr_v=1e-3`.
- Remaining gap:
  - stretch target (improve intermediate while also having non-positive endpoint delta) was not reached in this sweep.

## Method Note (2026-04-28 MFM integration choice)
- Added a first-pass Metric Flow Matching integration directly in the current training stack (hybrid mode), rather than launching the full upstream Lightning/WandB pipeline.
- Decision rationale:
  - keeps benchmark outputs directly comparable to existing bridge `comparison.json` contracts;
  - avoids introducing heavyweight mandatory dependencies;
  - still supports `mfm.backend=torchcfm` when that environment is available.
- Implemented MFM scope:
  - LAND metric geopath Stage A;
  - metric flow Stage B;
  - `metric_alpha0` ablation (`mfm_alpha=0`) in the same comparison artifact;
  - Stage C disabled for metric modes in this first pass.
- Tradeoff:
  - we gain fast, reproducible in-repo comparison now;
  - we still need future validation against a strict upstream stack for one-to-one reproduction claims.

## Empirical Note (2026-04-28 Bridge OT best-preset 4-method comparison)
- Executed `experiment=comparison_mfm` on the bridge OT best preset (seed 3) with sweep-matched budgets:
  - `stage_a_steps=300`, `stage_b_steps=300`, `stage_c_steps=0`, `batch_size=512`,
  - `eval_intermediate_ot_samples=1024`, `eval_transport_samples=4000`,
  - constrained hyperparameters: `rho=0.5`, `alpha=1.0`, `lr_g=1e-3`, `lr_v=1e-3`.
- Outcome on this run:
  - `constrained` remains best:
    - intermediate `0.1923` vs baseline `0.3364` (delta `-0.1440`);
    - endpoint `0.1160` vs baseline `0.1320` (delta `-0.0160`).
  - `metric` (LAND) underperformed baseline in this setting:
    - intermediate `0.3388` (delta `+0.0025`);
    - endpoint `0.2312` (delta `+0.0991`).
  - `metric_alpha0` stayed close to baseline as expected:
    - intermediate `0.3400` (delta `+0.0036`);
    - endpoint `0.1349` (delta `+0.0029`).
- Interpretation:
  - on this bridge benchmark/preset, constrained-moment path learning is still clearly stronger than current LAND-MFM settings;
  - alpha-0 ablation behaving near baseline validates that the new metric-mode plumbing is internally consistent.

## Empirical Note (2026-04-29 Bridge MFM alpha>0 sweep, seed 3)
- Ran a focused 16-config MFM sweep (nonzero alpha only) on bridge OT with methods `baseline` vs `metric` and high-sample budgets aligned to prior bridge comparisons.
- Swept:
  - `mfm.alpha ∈ {0.5, 1.0}`,
  - `mfm.sigma ∈ {0.0, 0.05}`,
  - `mfm.land_gamma ∈ {0.08, 0.125}`,
  - `mfm.land_rho ∈ {0.0005, 0.001}`.
- Result:
  - no configuration met the constrained-style gate (`delta_intermediate < 0` and `delta_endpoint <= +0.03`) on seed 3.
  - best-ranked config by intermediate-first rule (`a1_s0p05_g0p125_r0p001`) had:
    - `delta_intermediate=+0.00570` (close to baseline but not better),
    - `delta_endpoint=+0.13022` (substantially worse endpoint).
  - best endpoint config still had large endpoint degradation:
    - `a1_s0_g0p125_r0p0005`, `delta_endpoint=+0.10473`, with marginal degradation `delta_intermediate=+0.05563`.
- Interpretation:
  - in this tested MFM-LAND region, we do not yet recover the bridge endpoint while matching/improving intermediate marginals;
  - endpoint degradation appears to remain the dominant failure mode even when intermediate fit is near baseline.

## Method Note (2026-04-29 Fair hybrid MFM with constrained-information access)
- Implemented a new fair-hybrid family to compare against constrained with similar intermediate supervision access:
  - `metric_constrained_al`: Stage A uses `LAND + eta_moment * augmented_lagrangian(moment residuals)`.
  - `metric_constrained_soft`: Stage A uses `LAND + eta_moment * mean ||moment residual||^2`.
- Fairness policy:
  - LAND manifold references use endpoint marginals only (`mfm.reference_pool_policy=endpoints_only`);
  - intermediate-time information enters only through moment residuals at constrained times.
- Why this matters:
  - avoids giving metric methods direct access to intermediate marginals via LAND geometry;
  - keeps the MFM geometric prior while adding the same type of intermediate supervision channel used by constrained.
- Tradeoff:
  - AL variant is closer to constrained mechanics and enforces constraints more aggressively;
  - soft variant is simpler and potentially smoother but usually weaker at strict residual control.

## Empirical Note (2026-04-29 Bridge OT 6-way sanity run with fair-hybrid MFM, seed 3)
- Run profile:
  - `experiment=comparison_mfm`, `train=ab_only`, `data=bridge_ot`,
  - `stage_a_steps=300`, `stage_b_steps=300`, `stage_c_steps=0`,
  - constrained best settings: `rho=0.5`, `lr_g=1e-3`, `lr_v=1e-3`,
  - metric fairness settings: `mfm.reference_pool_policy=endpoints_only`, `mfm.moment_eta=1.0`.
- Folder:
  - `outputs/2026-04-29/bridge_mfm_hybrid_sanity_ab_only/16-00-17`
- Results (intermediate avg / endpoint):
  - baseline: `0.33635 / 0.13202`
  - constrained: `0.19233 / 0.11603`
  - metric: `0.25955 / 0.32475`
  - metric_alpha0: `0.34000 / 0.13494`
  - metric_constrained_al: `0.19589 / 0.26787`
  - metric_constrained_soft: `0.25838 / 0.32415`
- Interpretation:
  - both metric-constrained variants improve intermediate marginals versus baseline;
  - AL is much stronger than soft on intermediate fit and constraint residuals;
  - endpoint transport remains the failure mode for metric-family methods in this setting;
  - constrained still dominates on both intermediate and endpoint in this seed-3 sanity run.

## Empirical Note (2026-04-29 Balance sweep for metric-constrained AL vs soft)
- Ran a 32-config focused sweep (seed 3) over objective-balance parameters:
  - AL mode (`metric_constrained_al`): `mfm.moment_eta`, `train.rho`, `mfm.sigma`.
  - soft mode (`metric_constrained_soft`): `mfm.moment_eta`, `mfm.alpha`, `mfm.sigma`.
  - fixed fairness policy: `mfm.reference_pool_policy=endpoints_only`.
- Main outcome:
  - both AL and soft can improve intermediate marginals relative to baseline;
  - neither mode achieved the endpoint gate (`delta_endpoint <= +0.03`) in this sweep.
- Relative behavior:
  - AL consistently gave stronger intermediate gains than soft.
  - soft had weaker constraint fitting and generally worse endpoint deltas than AL.
- Best endpoint-oriented configs from this sweep:
  - AL: `eta0p5_rho1_s0p05` (`delta_intermediate=-0.1378`, `delta_endpoint=+0.1592`)
  - soft: `eta2_a0p5_s0` (`delta_intermediate=-0.0539`, `delta_endpoint=+0.1963`)
- Interpretation:
  - tuning balance terms alone is not sufficient yet to recover endpoint performance;
  - next sweep should prioritize endpoint-focused MFM settings (lower path curvature/noise and LAND stabilization) while keeping AL constraints active.

## Method Note (2026-04-30 Time-varying constrained smoothness weighting)
- Decision: extended constrained Stage A/C smoothness from scalar `beta` to scheduler-driven `beta(t)` with three options:
  - `constant` (exact backward-compatible behavior),
  - `piecewise` (interval-constant weights),
  - `linear` (linearly interpolated anchor weights).
- Why:
  - a single global smoothness penalty can over-regularize intervals where bridge marginals move quickly and under-regularize flatter intervals;
  - drift-adaptive weights target the user goal of allowing steeper dynamics where intermediate distributions change more.
- Construction tradeoff:
  - weights are driven by full moment drift (mean + covariance feature vector) over anchors `{0, constraint_times..., 1}`;
  - high-drift intervals get smaller smoothness weight, low-drift intervals get larger weight (with clipping bounds for stability).
- Fairness/compatibility:
  - this change is constrained-mode only; metric-family objectives are untouched in this pass;
  - `beta_schedule=constant` preserves prior behavior exactly.

## Empirical Note (2026-04-30 Bridge OT seed-3 sanity for `beta_schedule`)
- Profile:
  - `experiment=comparison`, `train=ab_only`, `data=bridge_ot`, `seed=3`,
  - `stage_a_steps=300`, `stage_b_steps=300`, `stage_c_steps=0`,
  - constrained best base settings: `rho=0.5`, `alpha=1.0`, `lr_g=1e-3`, `lr_v=1e-3`, `beta=0.05`.
- Runs:
  - `constant`: `outputs/2026-04-30/bridge_beta_sched_constant_ab_only/14-57-44`
  - `piecewise`: `outputs/2026-04-30/bridge_beta_sched_piecewise_ab_only/14-59-40`
  - `linear`: `outputs/2026-04-30/bridge_beta_sched_linear_ab_only/14-59-40`
- Constrained results (intermediate avg / endpoint):
  - constant: `0.19233 / 0.11603`
  - piecewise: `0.19253 / 0.11572`
  - linear: `0.19254 / 0.11575`
- Interpretation:
  - on this seed and setting, adaptive schedules are near-parity with constant on intermediate fit and slightly better on endpoint;
  - drift profile was mild (`beta` interval values close to `0.05`), so gains are expected to be small;
  - larger effects likely require regimes with stronger drift heterogeneity or broader scheduler ranges.

## Method Note (2026-04-30 Single-cell strict leaveout benchmark design)
- Decision:
  - introduced `family=single_cell` with strict leaveout as default benchmark protocol.
  - benchmark default is all middle holdouts for EB (`holdout_indices=[1,2,3]`) executed as separate runs and aggregated.
- Fairness rationale:
  - heldout timestamp is excluded from training constraints under default policy (`observed_nonendpoint_excluding_holdout`);
  - this keeps heldout performance as a true interpolation generalization check.
- Optional non-strict branch:
  - added policy `observed_nonendpoint_all` to allow using all intermediate observed moments (except endpoints), including heldout.
  - this is retained for ablations where strict generalization is not the primary target.
- Constraint feature choice:
  - generalized constrained moment map from 2D-only to \(d\)-dimensional mean + full covariance.
  - for EB 5D this remains compact and expressive; for higher dimensions we may revisit diagonal/low-rank alternatives.
- Evaluation tradeoff:
  - kept endpoint/intermediate empirical \(W_2\) contract for comparability with existing runs;
  - added holdout-focused metrics (`holdout_empirical_w2`, `holdout_empirical_w1`) to align with leaveout reporting needs.

## Method Note (2026-05-01 Single-cell Stage-A-only benchmark protocol)
- Decision:
  - added a dedicated Stage-A-only benchmark path for EB 5D strict leaveout:
    - methods: `constrained`, `metric`, `metric_alpha0`, `metric_constrained_al`, `metric_constrained_soft`;
    - holdouts: `0.25`, `0.50`, `0.75` (`indices 1,2,3`);
    - seeds: `3,7,11`.
- Why:
  - isolate interpolant quality from velocity-model approximation (Stage B/C);
  - compare learned interpolants directly against linear interpolation at heldout time.
- Stage-A holdout metrics:
  - `linear_holdout_empirical_w2`: OT error at heldout time using linear interpolant.
  - `learned_holdout_empirical_w2`: OT error at heldout time using learned interpolant (mode-aware path).
  - `delta_holdout_learned_minus_linear`: primary leaveout win/loss indicator (negative is better).
- Design note:
  - kept strict leaveout fairness: heldout time is excluded from constraint targets;
  - MFM-constrained variants still use endpoint-only LAND references and inject intermediate information only through moment constraints.

## Empirical Note (2026-05-01 EB 5D Stage-A-only strict leaveout benchmark)
- Run root:
  - `outputs/2026-05-01/single_cell_eb_5d_stage_a_strict_leaveout/18-55-20`
- Aggregate learned holdout \(W_2\) mean (lower is better):
  - `metric_constrained_al`: `1.2806` (best)
  - `metric`: `1.3421`
  - `metric_constrained_soft`: `1.3438`
  - `metric_alpha0`: `1.3811`
  - `constrained`: `1.6149`
- Aggregate holdout delta vs linear (negative is better):
  - `metric_constrained_al`: `-0.0860`
  - `metric`: `-0.0244`
  - `metric_constrained_soft`: `-0.0228`
  - `metric_alpha0`: `0.0000`
  - `constrained`: `+0.2413`
- Interpretation:
  - in this strict-leaveout Stage-A setting, AL-hybrid metric interpolation is strongest overall on heldout reconstruction;
  - pure constrained overfits/undershoots heldout behavior under the current tuning despite good constrained-time fitting in some slices;
  - alpha-0 behaves as expected linear reference.

## Method Note (2026-05-04 Unsupervised pseudo-type constraints for unlabeled single-cell states)
- Decision:
  - add optional soft pseudo-type constraints built from unsupervised GMM posteriors on whitened PC space;
  - keep pseudo constraints additive with existing moment constraints (do not replace moments);
  - activate pseudo terms in all constrained families:
    - `constrained`,
    - `metric_constrained_al`,
    - `metric_constrained_soft`.
- Why:
  - user goal requires indicator-like constraints without curated cell-type labels;
  - soft posteriors \(q_k(x)\) provide differentiable, boundary-robust surrogates of class indicators.
- \(K\)-selection policy:
  - sweep `K in [k_min, k_max]`;
  - score each K by median BIC across seeds and mean pairwise ARI stability;
  - pick minimum-BIC model among Ks above stability threshold; fallback to global minimum-BIC if no K passes threshold.
- Tradeoffs:
  - improves optimization signal for latent-fate-like constraints, but pseudo-types are latent/statistical and may not map one-to-one to biological ontologies;
  - therefore feature is optional (`enabled=false`, `pseudo_eta=0.0` by default) and diagnostics are surfaced (`pseudo_labels_k`, `bic_by_k`, `stability_by_k`).

## Empirical Note (2026-05-04 Pseudo-constraint smoke validation on EB 5D)
- Stage-A-only smoke run (constrained mode, strict leaveout):
  - `outputs/2026-05-04/single_cell_pseudo_stagea_smoke/00-00-01`
  - pseudo metrics present and finite:
    - `pseudo_constraint_residual_avg=0.3772`,
    - `pseudo_labels_k=9`,
    - cache hit path recorded in summary.
- A+B smoke run (constrained mode, strict leaveout):
  - `outputs/2026-05-04/single_cell_pseudo_ab_smoke/00-00-01`
  - pseudo metrics present and finite during velocity-enabled run:
    - `pseudo_constraint_residual_avg=0.3436`,
    - legacy transport/holdout metrics remain populated and schema-compatible.
- Interpretation:
  - pseudo constraints are integrated end-to-end and coexist with moment constraints without breaking existing summary contracts;
  - next step is full holdout/seed sweeps to assess whether pseudo terms improve holdout reconstruction and/or endpoint transport tradeoffs.

## Empirical Note (2026-05-04 Stage-A `metric_constrained_al` pseudo full-OT sweep and validation)
- Search run root:
  - `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_stage_a_fullot/18-24-04`
- Baseline reference:
  - `outputs/2026-05-04/single_cell_eb_stage_a_ot_global_pot_holdout3_seed3/17-43-31`
- Search finding (holdout 2 only, seeds 3/7/11):
  - all pseudo settings improved constrained-time full-OT average strongly vs baseline aggregate;
  - however holdout full-OT metric worsened in this holdout-2-only tuning phase.
- Final validation (top-2 configs, holdouts 1/2/3, seeds 3/7/11):
  - winner: `eta1.00_rho5.0_clip100.0`.
  - winner aggregate:
    - `learned_full_ot_w2_avg = 1.12925` (vs simple baseline `1.13407`, better by `-0.00482`);
    - `learned_holdout_full_ot_w2 = 1.15739` (vs simple baseline `1.15983`, better by `-0.00244`).
- Interpretation:
  - pseudo AL terms helped `metric_constrained_al` modestly but consistently on all-holdout means under this protocol;
  - gain size is small, so robustness checks on additional seeds are still warranted before treating this as a stable default.

## Empirical Note (2026-05-04 Stage-A `constrained` pseudo full-OT evaluation)
- Run root:
  - `outputs/2026-05-04/single_cell_constrained_pseudo_stage_a_fullot/18-54-15`
- Baseline-vs-unsup summary:
  - `outputs/2026-05-04/single_cell_constrained_pseudo_stage_a_fullot/18-54-15/baseline_vs_unsup_summary.json`
- Result (`learned_holdout_full_ot_w2`, lower is better):
  - baseline constrained overall mean: `1.42862`,
  - constrained+unsup overall mean: `1.40534`,
  - delta: `-0.02328`.
- Nuance:
  - effect is not uniform across holdouts (`p0.25` and `p0.75` improved, `p0.50` slightly degraded);
  - constrained family remains much less stable than metric-constrained AL family under strict leaveout.

## Empirical Note (2026-05-04 Full-table unsup extension across all methods)
- Run root:
  - `outputs/2026-05-04/single_cell_all_methods_pseudo_stage_a_fullot/19-03-38`
- Key observation:
  - extending pseudo-enabled flags to `metric` and `metric_alpha0` does not change outcomes in this benchmark because these families do not include pseudo residual terms in their Stage-A objectives (`pseudo_constraints_active=false` in all runs).
  - `metric_constrained_soft` does activate pseudo terms, but effect size is very small on the holdout full-OT aggregate.
- Methodological implication:
  - for a meaningful pseudo-vs-non-pseudo comparison, the active constraint families are:
    - `metric_constrained_al`,
    - `metric_constrained_soft`,
    - `constrained`.
  - reporting `+unsup` rows for non-active families is still useful as a negative control proving that improvements are not caused by unrelated run noise or evaluation drift.

## Empirical Note (2026-05-04 Fixed-`K` sensitivity for `metric_constrained_al+unsup`)
- Run root:
  - `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_k_sensitivity_stage_a_fullot/19-56-10`
- Setup:
  - strict leaveout Stage-A full-OT protocol,
  - fixed `K` tested by setting `k_min=k_max=K`,
  - search over `K in {4,6,8,9,10,12}` and `pseudo_eta in {0.5,1.0}` (holdout 2, seeds 3/7/11),
  - final validation on top-2 configs across holdouts `1/2/3` and seeds `3/7/11`.
- Findings:
  - top-2 from search were `k8_eta1.0` and `k4_eta1.0`;
  - final winner: `k8_eta1.00_rho5.0_clip100.0`.
  - winner means:
    - `learned_full_ot_w2_avg = 1.129041`,
    - `learned_holdout_full_ot_w2 = 1.157122`.
  - compared to prior auto-`K=9` winner:
    - primary delta `-0.000206`,
    - holdout delta `-0.000270`.
- Interpretation:
  - fixed-`K` tuning can improve over auto-`K`, but margin is very small in this regime;
  - practical takeaway is that `K` acts as a fine-tuning knob rather than a high-leverage control here;
  - we should avoid over-claiming and treat `K=8` as provisional until extra-seed robustness confirms ranking stability.

## Empirical Note (2026-05-04 High-eta dichotomy at fixed `K=9` for `metric_constrained_al+unsup`)
- Run root:
  - `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_k9_eta_dichotomy_stage_a_fullot/20-43-25`
- Setup:
  - strict-leaveout Stage-A full-OT protocol,
  - fixed pseudo classes (`k_min=k_max=9`),
  - eta ladder `1,2,4,8,16,32,64,100`,
  - fixed `pseudo_rho=5.0`, `pseudo_lambda_clip=100.0`,
  - holdout `2`, seeds `3/7/11`.
- Findings:
  - no failures (`24/24` successful runs),
  - both full-OT aggregates improved monotonically as eta increased on this grid,
  - best in-sweep point: `eta=100` with
    - `learned_full_ot_w2_avg=0.906111`,
    - `learned_holdout_full_ot_w2=1.200209`.
- Interpretation:
  - earlier eta ranges (`<=1`) were likely underweighting pseudo constraints for this method and protocol;
  - at least up to `eta=100`, stronger pseudo weighting did not cause visible degradation on holdout `2`;
  - this contradicts the initial small-effect impression and suggests eta is a high-leverage knob once allowed to scale.
- Caveat:
  - this is a single-holdout finding so far; we should not claim global superiority until holdouts `1/3` confirm similar behavior.

## Empirical Note (2026-05-04 Continuation past eta=100 with stop-at-non-improvement rule)
- Continuation root:
  - `outputs/2026-05-04/single_cell_metric_constrained_al_pseudo_k9_eta_dichotomy_stage_a_fullot/21-01-36`
- Protocol:
  - fixed `K=9`, strict leaveout Stage-A full-OT, holdout `2`, seeds `3/7/11`,
  - continued eta ladder beyond 100: `128,160,200,256,320,400`,
  - stop rule: stop when improvement in mean primary metric vs previous eta is `<= 5e-4`.
- Findings:
  - improvement continued through `eta=320`,
  - `eta=400` was the first non-improving point:
    - primary mean worsened from `0.826541` (eta 320) to `0.833223` (eta 400).
  - best observed point in combined ladder (`1 -> 400`):
    - `eta=320`, secondary mean `1.085330`.
- Interpretation:
  - eta is not just a mild tuning knob in this regime; it strongly reshapes the objective tradeoff up to a high range,
  - there appears to be a turning point around `320-400` for this holdout/seed block.
- Caveat:
  - current stop decision is based on holdout `2` only; cross-holdout validation is needed before promoting `eta=320` as the global setting.

## Method Note (2026-05-04 0.5-only pseudo-fit + 0.5-only constraints with decoupled eval times)
- Protocol decision:
  - fit pseudo-label GMM on the single marginal `t=0.5` only;
  - apply both moment constraints and pseudo constraints only at `t=0.5`;
  - still evaluate interpolant metrics on `t in {0.25, 0.5, 0.75}`.
- Why this protocol is useful:
  - isolates whether latent-type structure learned at the mid-snapshot can regularize Stage-A dynamics without directly supervising other constrained times;
  - separates "where constraints are enforced" from "where performance is measured", which is important to test cross-time generalization.
- Required design tradeoff:
  - we introduced explicit time controls (`constraint_times_normalized`, `eval_times_normalized`, `pseudo_labels.fit_times_normalized`) rather than overloading one policy field.
  - this increases config surface slightly, but avoids hidden coupling between optimization targets and evaluation diagnostics.
- Main empirical readout for this protocol:
  - run root: `outputs/2026-05-04/single_cell_t05_only_pseudo_stage_a_fullot/21-39-57`;
  - among tested constrained families (seed-mean overall full-OT):
    - `metric_constrained_al`: `0.878618 +/- 0.007908` (best),
    - `constrained`: `1.142474 +/- 0.114557`,
    - `metric_constrained_soft`: `1.222570 +/- 0.007806`.
- Interpretation:
  - concentrating pseudo supervision at `t=0.5` can still improve all-snapshot evaluation in the AL-based metric-constrained family, suggesting meaningful cross-time regularization rather than pure local overfitting to the constrained snapshot.
- Caveat:
  - these conclusions are for `no_leaveout` protocol and `eta=320`; strict-leaveout validation is still needed before making this the default single-cell protocol.
