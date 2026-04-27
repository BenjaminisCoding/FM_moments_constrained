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
