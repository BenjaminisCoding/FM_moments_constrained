# ARCHITECTURE.md

## Document Status
- Last updated: 2026-05-04
- Owner: project contributors and coding agents
- Update trigger: any meaningful change to structure, file responsibilities, key functions, or flow

## Repository Boundary
`FM/` is environment tooling (Python virtual environment), not project source architecture.
Exclude `FM/` internals from architectural documentation and component mapping.

## 1. Project Purpose and Scope
### Purpose
Implement and study constrained flow matching where the learned path must satisfy intermediate-time moment constraints while keeping endpoint consistency.
The v1 target is a reproducible 2D synthetic benchmark comparing:
- standard linear-path CFM baseline;
- constrained-path CFM with learned correction;
- metric flow matching (MFM) variants on the same bridge benchmark contract.

### In Scope
- PyTorch + Hydra research pipeline running on CPU.
- Configurable batch coupling for training pairs (`ot` exact discrete OT or `random` permutation coupling).
- Optional global balanced OT coupling mode (`ot_global`) for single-cell data with precomputed sparse plan support and cache reuse.
- Optional unsupervised pseudo-type constraints for single-cell data via cached GMM posteriors in whitened PC space.
- Exact-formula moment targets at times `t = {0.25, 0.50, 0.75}`.
- Pre-training bridge-SDE distribution preview utilities and notebook workflow.
- Bridge Stage-A-only interpolant training/evaluation with cached high-precision bridge targets.
- Bridge Stage-A+B (`ab_only`) velocity-rollout evaluation with empirical OT metrics at constrained times and endpoint.
- Stage A/B/C training schedule for constrained mode.
- Constrained Stage A/C smoothness scheduling with `train.beta_schedule in {constant,piecewise,linear}` driven by full moment drift across anchors.
- Stage A/B metric-flow schedule for `metric`, `metric_alpha0`, `metric_constrained_al`, and `metric_constrained_soft` modes (Stage C intentionally disabled for metric-family modes).
- Fair-hybrid MFM objective: endpoint-only LAND manifold references plus optional intermediate-time moment constraints (AL or soft) for metric-constrained variants.
- A+B-only constrained ablation profile (`stage_c_steps = 0`) for isolating pretrain-then-CFM behavior.
- Stage-A-only constrained profile (`stage_b_steps = 0`, `stage_c_steps = 0`) for interpolant-only bridge validation before velocity learning.
- Additive robust full-set balanced OT evaluation for single-cell (Stage-A interpolants and A+B rollouts), alongside legacy sampled Hungarian metrics, with selectable backend (`exact_lp` or POT `pot_emd2`).
- POT `ot.emd2` runtime-sweep utility for holdout-time plan-conditioned W2 benchmarking.
- Baseline and constrained comparison artifacts (metrics JSON, checkpoint, plots).
- Multi-method comparison artifacts (`comparison_mfm.json`) with method list metadata and legacy `comparison.json` preservation.

### Out of Scope
- Real-data pipelines and domain-specific physical simulators.
- High-dimensional scaling and production optimization.
- Noisy or partially observed intermediate moments (planned future work).

## 2. High-Level Component Map
Document the system as major components and their responsibilities.

| Component | Responsibility | Inputs | Outputs | Dependencies |
| --- | --- | --- | --- | --- |
| `data` | Gaussian problem setup, coupling sampling (`ot`/`random`/`ot_global`), analytic target moments | config means/covs/kappa/coupling | paired batches, target feature vectors | `torch`, `scipy` |
| `bridge_data` | Bridge target-cache build/load and empirical bridge marginals/sampler for training and evaluation | bridge SDE config + seed + times | empirical coupling problem + cached target samples/features | `torch`, `bridge_sde` |
| `ot_utils` | Balanced OT utilities for global coupling and full-set metric computation (exact LP + optional POT backend) | point clouds + optional weights | sparse global OT support, full-set balanced OT costs | `numpy`, `scipy.optimize.linprog`, optional `POT` |
| `pseudo_labels` | GMM pseudo-label fit/selection/cache and differentiable posterior scorer for soft class membership | whitened single-cell features + pseudo-label config | frozen posterior scorer \(q_k(x)\), selected \(K\), BIC/stability diagnostics | `numpy`, `torch`, `scikit-learn` |
| `bridge_sde` | Bridge-process trajectory simulation and time-slice sampling | SDE params + seed | trajectory tensor and snapshot samples | `torch` |
| `models` | `v_psi` velocity net and `g_theta` path-correction net | tensors `(t, x)` or `(t, x0, x1)` | predicted velocity or correction | `torch.nn` |
| `mfm_core` | MFM interpolation/velocity core, LAND metric utilities, and backend selection (`auto/native/torchcfm`) | path model + MFM config | metric backend and path/velocity targets | `torch`, optional `torchcfm` |
| `paths` | linear/corrected paths and constrained velocity target computation | paired points + time + `g_theta` | `x_t`, `u_t`, derivatives in `t` | `torch.autograd` |
| `constraints` | moment features, residuals, augmented Lagrangian terms | path samples and target moments | residual vectors, AL loss, multiplier updates | `torch` |
| `training` | stage-wise optimization (A/B/C or A-only), eval summaries | config + problem + targets | trained models, histories, metrics | core modules |
| `pipeline` | orchestration, artifact writing, baseline/constrained comparison | Hydra config | run outputs under `outputs/` | all runtime modules |
| `plotting` | diagnostics plots for loss/residuals/path geometry, bridge previews, and interpolant-only bridge comparisons | histories/residuals/models or trajectory samples | PNG/GIF artifacts | `matplotlib` |

## 3. File and Folder Responsibilities
Use this section to quickly understand what each important path does.

| Path | Role | Key Contents | Notes |
| --- | --- | --- | --- |
| `/` | Project root | Governance docs, configs, scripts, source, tests | Keep intent + method + state discoverable |
| `FM/` | Environment tooling only | Virtual environment binaries and packages | Not part of product architecture |
| `DISCUSSION.md` | Research design log | Alternatives, tradeoffs, rejected options | Complements chronological `PROJECT_STATE.md` |
| `EXPERIMENTS.md` | Experiment index log | Output folders grouped by run/sweep with test intent | Complements chronological `PROJECT_STATE.md` with artifact traceability |
| `src/cfm_project/` | Core implementation | data/models/paths/constraints/training/pipeline | Main research code |
| `src/cfm_project/ot_utils.py` | Balanced OT utilities | sparse LP OT plan/cost helpers plus POT `emd2` full-set metric helper used by robust full-OT evaluation | Shared by data prep and evaluation |
| `notebooks/` | Interactive diagnostics | bridge-SDE preview, Stage-A interpolant analysis, and bridge A+B comparison review | reusable analysis notebooks | Uses shared `src/cfm_project` helpers |
| `configs/` | Hydra configuration | experiment/data/model/train groups (including coupling mode) | Controls reproducible runs |
| `scripts/run_experiment.py` | Entry point | Hydra main invoking pipeline | Supports single-mode, legacy comparison, and method-list comparisons |
| `scripts/run_bridge_mfm_best.py` | Friendly benchmark runner | fixed best bridge OT preset for multi-method MFM comparison | Executes run + prints deltas vs baseline |
| `scripts/run_single_cell_eb_stage_a_benchmark.py` | Stage-A embryo benchmark runner | strict-leaveout holdout/seed grid for interpolant-only evaluation | Writes per-run `comparison_mfm.json`, `benchmark_summary_stage_a.json`, and leaderboard TSV |
| `scripts/run_pot_emd2_runtime_sweep.py` | POT runtime benchmark utility | calibrates/sweeps `numItermax` for plan-conditioned holdout-time W2 on EB 5D | Writes `emd2_runtime_sweep.json` and `.tsv` tables under `outputs/` |
| `outputs/` | Experiment artifacts | date/label/time runs, sweep folders, seed checks | Includes per-mode metrics and comparison summary |
| `tests/` | Verification suite | unit + smoke tests | CPU-first regression safety |

## 4. Key Functions, Modules, and Interaction Flow
Capture the most important execution flow and key functions/modules.

### Key Functions or Modules
| Module or Function | Responsibility | Called By | Calls Into |
| --- | --- | --- | --- |
| `data.analytic_target_moment_features` | Build exact intermediate moment targets | pipeline/training | Gaussian OT map helpers |
| `bridge_data.prepare_bridge_problem_and_targets` | Build/load cached bridge targets and empirical endpoint pools | pipeline | bridge SDE simulator + target cache IO |
| `single_cell_data.prepare_single_cell_problem_and_targets` | Build single-cell endpoint pools/targets, explicit time-selection overrides, and optional global OT cache support | pipeline | data loading + optional `ot_utils.solve_balanced_ot_lp` |
| `pseudo_labels.prepare_pseudo_labels` | Fit/load GMM pseudo-label model with BIC+stability \(K\) selection | single-cell prep | differentiable posterior scorer + cache metadata |
| `bridge_sde.simulate_bridge_sde_trajectories` | Simulate bridge SDE trajectories in 2D | notebook/tests (future data mode) | Gaussian initial sampling + Euler-Maruyama |
| `ot_utils.solve_balanced_ot_lp` | Solve exact balanced rectangular OT LP and extract sparse support | single-cell prep + robust metrics | sparse LP (`scipy.optimize.linprog`) |
| `ot_utils.balanced_empirical_w2_distance_pot` | Compute balanced empirical full-set \(W_2\) with POT `ot.emd2` | robust full-OT metric paths | optional `POT` backend |
| `mfm_core.build_metric_backend` | Resolve and build metric backend (`auto/native/torchcfm`) | MFM config + geopath net | backend used by metric modes |
| `mfm_core.mfm_path_and_velocity` | Compute MFM path mean and conditional flow | `(t, x0, x1, g)` | `\mu_t`, `u_t` for metric training |
| `paths.corrected_path` | Endpoint-preserving corrected interpolation | training/evaluation | `g_theta` model |
| `paths.corrected_velocity` | Compute analytic target velocity for constrained CFM | training | autograd derivative wrt `t` |
| `constraints.augmented_lagrangian_terms` | Compute AL constraint objective | stage A / stage C | residual vectors + multipliers |
| `training.train_experiment` | Execute selected training schedule and evaluations | pipeline | all core modules |
| `pipeline.run_pipeline` | Artifact orchestration and optional mode comparison | script entrypoint | training + plotting + IO |

### Interaction Flow
1. Hydra loads config and dispatches `pipeline.run_pipeline`.
2. Pipeline builds data bundle:
   - Gaussian family: analytic Gaussian targets.
   - Bridge family: cached Monte Carlo bridge targets (`.cache/bridge_targets`) and empirical endpoint pools.
   - Single-cell family: observed timestamp pools, optional global OT plan cache (`.cache/ot_plans`) when `data.coupling=ot_global`, and optional pseudo-label cache (`.cache/pseudo_labels`) when `data.single_cell.pseudo_labels.enabled=true`.
   - Single-cell time-selection controls:
     - `data.single_cell.constraint_times_normalized` can explicitly override policy-derived constrained times;
     - `data.single_cell.eval_times_normalized` can decouple Stage-A interpolant evaluation times from constrained times;
     - `data.single_cell.pseudo_labels.fit_times_normalized` can restrict pseudo-label GMM fitting to selected observed marginals.
   - Bridge time semantics: configured times are normalized in `[0, 1]` and mapped to physical SDE time via `t_phys = t_norm * bridge.total_time`.
3. Trainer runs requested method schedule:
   - constrained: Stage A (`g_theta` with AL and smoothness weighted by `beta(t)`) -> Stage B (`v_psi` CFM) -> optional Stage C (joint, disabled when `stage_c_steps=0`).
   - when single-cell pseudo constraints are enabled (`train.pseudo_eta>0`): constrained mode adds an additive AL term on soft pseudo-type residuals with independent multipliers/hyperparameters.
   - constrained `beta(t)` schedule is built once per run from anchors `{0, constraint_times..., 1}` using full moment drift; `constant` reproduces legacy scalar-`beta` behavior exactly.
  - metric: Stage A (`g_theta` via LAND metric objective) -> Stage B (`v_psi` flow matching with MFM targets), with backend chosen by `mfm.backend`.
  - metric_alpha0: linear-path ablation under metric pipeline (`mfm_alpha=0`), no geopath training.
  - metric_constrained_al: Stage A hybrid objective (`LAND + eta_moment * AL(moment residuals)`), plus optional additive pseudo-type AL term (`pseudo_eta * AL(pseudo residuals)`); Stage B metric flow matching.
  - metric_constrained_soft: Stage A hybrid objective (`LAND + eta_moment * mean ||residual||^2`), plus optional additive pseudo-type soft term (`pseudo_eta * mean ||pseudo residual||^2`); Stage B metric flow matching.
  - MFM manifold reference policy is configurable (`mfm.reference_pool_policy`); default `endpoints_only` uses only `t \in \{0,1\}` samples for LAND geometry.
   - Stage-A-only profile: Stage A only (`stage_b_steps=0`, `stage_c_steps=0`) with interpolant-only evaluation.
   - baseline: Stage B only with linear path CFM.
   - coupling behavior:
     - `ot`: per-step Hungarian on independently sampled endpoint batches;
     - `random`: per-step random pairing;
     - `ot_global` (single-cell): sample endpoint pairs with replacement from cached sparse global balanced OT support.
4. Evaluation computes:
   - always: constraint residual norms;
   - velocity-enabled runs: CFM validation and rollout metrics;
   - Gaussian family rollout metrics: endpoint mean/cov errors and Gaussian/empirical intermediate diagnostics;
   - Bridge family rollout metrics: empirical OT at constrained times and endpoint (`transport_endpoint_empirical_w2`, `transport_score`).
   - Single-cell robust full-set OT metrics (optional, additive):
     - Stage-A: weighted-edge interpolant metrics (`linear_full_ot_w2*`, `learned_full_ot_w2*`);
     - A+B rollout: full-pool transport metrics (`intermediate_full_ot_w2*`, `transport_endpoint_full_ot_w2`, `holdout_full_ot_w2`).
     - Backend selection is controlled by `train.eval_full_ot_method` (`exact_lp` or `pot_emd2`) with optional `train.eval_full_ot_num_itermax` for POT iteration cap.
     - Current default in train profiles is `pot_emd2` (exact LP remains available as an explicit override).
   - Stage-A-only runs: interpolant-only empirical \(W_2\) (linear vs learned vs target marginals).
   - For single-cell Stage-A, interpolant metrics use `data.interpolant_eval_times` when provided (set by single-cell prep from explicit eval-time overrides or defaulting to constrained+holdout semantics).
5. Pipeline saves metrics JSON, checkpoint, and plots, plus comparison summaries:
   - Stage-A-only constrained bridge runs: interpolant trajectory/marginal/W2 diagnostics.
   - Stage-A-only metric-family runs: the same interpolant diagnostics with mode-aware learned paths (`corrected_path` vs MFM mean path vs alpha0-linear).
   - Bridge velocity-enabled runs: rollout generated-vs-true marginal grid plus per-time empirical \(W_2\) bars.
   - Method-list comparisons write `comparison_mfm.json`; when baseline+constrained are included, legacy `comparison.json` is also written with unchanged schema.
   - Optional `experiment.method_overrides` applies per-mode deep config merges before each mode run inside a comparison list.
6. Bridge-SDE preview is currently notebook-driven (shared utilities + explicit export), not auto-generated by pipeline runs.

## 5. Mathematical Formulation
### Path Families
- Linear baseline path:
  \[
  \ell_t = (1-t)x_0 + tx_1.
  \]
- Constrained path with endpoint-preserving correction:
  \[
  x_t = \ell_t + a_t g_\theta(t,x_0,x_1), \quad a_t=t(1-t).
  \]

### Velocity Target for CFM (Constrained Mode)
\[
u_t = \partial_t x_t = (x_1-x_0) + (1-2t)g_\theta + a_t \partial_t g_\theta.
\]

### Moment Constraints
- Feature map in v1+: mean + flattened covariance in \(d\)-dimensions.
- Residual at constrained times \(t_k \in \{0.25, 0.50, 0.75\}\):
  \[
  c_k(\theta) = \mathbb{E}[\phi(x_{t_k})] - m_k.
  \]

### Path Prior
\[
R(\theta)=\alpha\,\mathbb{E}\|u_t-(x_1-x_0)\|^2+\mathbb{E}\!\left[\beta(t)\,\|\partial_t u_t\|^2\right].
\]

### Augmented Lagrangian
\[
\mathcal{L}_g = R(\theta) + \sum_k \lambda_k^\top c_k + \frac{\rho}{2}\sum_k \|c_k\|^2,
\quad
\lambda_k \leftarrow \lambda_k + \rho c_k.
\]

### CFM Objective
\[
\mathcal{L}_{CFM} = \mathbb{E}\|v_\psi(t,x_t)-u_t\|^2.
\]

### Joint Finetune Objective
\[
\mathcal{L}_{joint} = \mathcal{L}_{CFM} + \eta \mathcal{L}_g,\quad \eta \ll 1.
\]

### Metric Flow Path Family (MFM)
\[
\mu_t = (1-t)x_0 + tx_1 + \alpha_{\text{mfm}}\gamma(t)\,g_\theta(t,x_0,x_1),
\]
\[
u_t = \partial_t \mu_t
=
(x_1-x_0)
\alpha_{\text{mfm}}\left[\dot{\gamma}(t)g_\theta+\gamma(t)\partial_t g_\theta\right].
\]
Stage-B metric FM trains on noisy states \(x_t=\mu_t+\sigma_t\varepsilon\), \(\varepsilon\sim\mathcal{N}(0,I)\), with target \(u_t\).

### Hybrid Metric-Constrained Stage A Objectives
- `metric_constrained_soft`:
  \[
  \mathcal{L}_A^{\text{soft}} = \mathcal{L}_{\text{LAND}} + \eta_{\text{moment}}\cdot \frac{1}{|\mathcal{T}_c|}\sum_k \|c_k(\theta)\|_2^2.
  \]
- `metric_constrained_al`:
  \[
  \mathcal{L}_A^{\text{AL}} = \mathcal{L}_{\text{LAND}} + \eta_{\text{moment}}\left(\sum_k \lambda_k^\top c_k(\theta) + \frac{\rho}{2}\sum_k\|c_k(\theta)\|_2^2\right),
  \]
  \[
  \lambda_k \leftarrow \mathrm{clip}\big(\lambda_k + \rho\,c_k(\theta),-\lambda_{\max},\lambda_{\max}\big).
  \]
Default fairness policy uses endpoint-only manifold references for LAND (`mfm.reference_pool_policy=endpoints_only`), so intermediate information enters through \(c_k\) only.

### Single-Cell Soft Pseudo-Type Constraints (Optional)
When `data.single_cell.pseudo_labels.enabled=true`, a frozen GMM posterior scorer \(q_k(x)=P(z=k\mid x)\) is learned once (or loaded from cache) in whitened PC space.
At each constrained time \(t\), target pseudo proportions are:
\[
\pi^{\text{ps}}_{t,k}
=
\frac{1}{N_t}\sum_{x\in\mathcal{D}_t} q_k(x).
\]
Model-implied proportions on generated path samples are:
\[
\hat{\pi}^{\text{ps}}_{t,k}(\theta)
=
\frac{1}{B}\sum_{i=1}^{B} q_k(x_t^{(i)}(\theta)).
\]
Pseudo residual vector:
\[
r_t^{\text{ps}}(\theta)=\hat{\pi}^{\text{ps}}_t(\theta)-\pi_t^{\text{ps}}.
\]
Constrained mode adds:
\[
\eta_{\text{ps}}
\left(
\sum_t \nu_t^\top r_t^{\text{ps}}
\frac{\rho_{\text{ps}}}{2}\sum_t \|r_t^{\text{ps}}\|_2^2
\right),
\quad
\nu_t\leftarrow\mathrm{clip}\!\left(\nu_t+\rho_{\text{ps}}r_t^{\text{ps}},-\lambda^{\text{ps}}_{\max},\lambda^{\text{ps}}_{\max}\right).
\]
`metric_constrained_al` uses the same pseudo AL term; `metric_constrained_soft` uses \(\eta_{\text{ps}}\frac{1}{|\mathcal{T}_c|}\sum_t\|r_t^{\text{ps}}\|_2^2\).

### Evaluation Metric Extension
- Intermediate-time distribution quality includes Gaussian \(W_2\) diagnostics at constrained times \(t_k\), computed between empirical generated samples (integrating \(v_\psi\) up to \(t_k\)) and analytic Gaussian targets \((\mu_{t_k}, \Sigma_{t_k})\).
- Intermediate-time distribution quality also includes sample-based empirical \(W_2\) diagnostics using exact discrete OT matching between generated samples and target samples at each \(t_k\), with a separate small evaluation sample budget for CPU tractability.
- Stage-A-only bridge evaluation adds interpolant-only empirical OT diagnostics:
  - `interpolant_eval.linear_empirical_w2` and `interpolant_eval.learned_empirical_w2` at each constrained time;
  - averages plus `delta_avg_learned_minus_linear`.
- Bridge velocity-enabled evaluation (for example `train=ab_only`) reports rollout empirical OT diagnostics:
  - `intermediate_empirical_w2` and `intermediate_empirical_w2_avg` at constrained normalized times;
  - `transport_endpoint_empirical_w2` at normalized `t=1.0`;
  - `transport_score = transport_endpoint_empirical_w2` for bridge family.
- Metric-mode summaries add backend and hyperparameter traceability fields:
  - `mfm_backend`, `mfm_backend_impl`, `mfm_alpha`, `mfm_sigma`, `mfm_land_gamma`, `mfm_land_rho`.
- Hybrid metric-constrained summaries also report:
  - `mfm_reference_pool_policy`, `mfm_moment_style`, `mfm_moment_eta`.
- Constrained summaries report smoothness-scheduler traceability:
  - `beta_schedule`, anchor times, interval drifts, and resolved interval/anchor beta values.
- Pseudo-label-enabled single-cell summaries additionally report:
  - `pseudo_constraint_residual_norms`, `pseudo_constraint_residual_avg`,
  - `pseudo_labels_k`, `pseudo_labels_cache_path`, `pseudo_labels_cache_hit`,
  - `bic_by_k`, `stability_by_k`.
- For bridge velocity-enabled runs, Gaussian-only transport fields are intentionally `null`:
  - `transport_mean_error_l2`, `transport_cov_error_fro`,
  - `intermediate_w2_gaussian`, `intermediate_w2_gaussian_avg`.
- For bridge data, constrained-time diagnostics at normalized `t_k` are computed against target marginals sampled at physical time `t_k * bridge.total_time` (with endpoint pool at physical `bridge.total_time`).
- For `single_cell` data family (empirical timestamped marginals):
  - endpoint pools are first/last observed times;
  - constrained times are selected from non-endpoint observed timestamps via policy;
  - strict leaveout benchmarks exclude heldout timestamp moments from training constraints by default;
  - summaries include holdout diagnostics (`holdout_empirical_w2`, `holdout_empirical_w1`) when a heldout time is active.
- Robust full-set OT metrics are additive (legacy keys unchanged):
  - Stage-A robust source measure uses global-plan weighted support edges (`ot_global`) interpolated at each time;
  - A+B robust rollout uses full source endpoint pool pushforwards against full target pools.
- Robust leaderboard preference in stage-A benchmark aggregation is robust-key first with per-mode fallback to legacy holdout key when robust key is missing.
- In Stage-A-only runs, velocity-rollout metrics are explicitly out-of-scope and recorded as `null` to avoid misinterpretation.
- Operational note: `eval_intermediate_ot_samples` is intentionally small in default configs; when scaling to more complex/non-Gaussian experiments, increase this budget (or switch to approximate OT/Sinkhorn diagnostics) before drawing conclusions from empirical \(W_2\).

## 5.1 Single-Cell Leaveout Flow
- Data entrypoint:
  - `family=single_cell` loads `.npz` (`pcs`, `sample_labels`) or `.h5ad` (`X_pca`, `day`) from `data.single_cell.path`.
- Time handling:
  - unique observed labels are sorted, reindexed as \(0,\dots,T-1\), and mapped to normalized times \(t=i/(T-1)\).
- Endpoint problem:
  - `EmpiricalCouplingProblem` uses pools from index `0` and `T-1`.
  - coupling options:
    - `ot`: per-batch Hungarian on sampled endpoints,
    - `random`: per-batch random pairing,
    - `ot_global`: precompute/load one global balanced OT plan and sample pairs from sparse support.
- Global OT cache (`ot_global`):
  - cache directory defaults to `.cache/ot_plans`;
  - key includes dataset/preprocessing signature and endpoint hashes;
  - stored payload contains sparse support edges (`src_idx`, `tgt_idx`, `mass`) and total plan cost.
- Constraint targets:
  - per-time moment targets are built from observed pools with policy:
    - `observed_nonendpoint_excluding_holdout` (default strict leaveout),
    - `observed_nonendpoint_all` (optional non-strict setting).
- Optional pseudo-type targets:
  - GMM pseudo-label model is selected with BIC+stability over `K in [k_min, k_max]`,
  - model is cached under `.cache/pseudo_labels` keyed by dataset/preprocessing/config signature,
  - per-constrained-time pseudo targets are mean posterior vectors over observed pools.
- Protocol controls:
  - `experiment.protocol: strict_leaveout | no_leaveout`,
  - `experiment.holdout_index` active per run,
  - `experiment.holdout_indices` used by benchmark orchestration loops.
- Stage-A single-cell benchmark contract:
  - modes: `constrained`, `metric`, `metric_alpha0`, `metric_constrained_al`, `metric_constrained_soft`;
  - holdout-aware interpolant metrics in `summary.interpolant_eval`:
    - `linear_holdout_empirical_w2`,
    - `learned_holdout_empirical_w2`,
    - `delta_holdout_learned_minus_linear`;
  - aggregate artifacts:
    - `benchmark_summary_stage_a.json`,
    - `leaderboard_stage_a.tsv`.
  - leaderboard ranking key:
    - primary: `learned_holdout_full_ot_w2_mean`,
    - fallback: `learned_holdout_empirical_w2_mean` when robust key is absent.
- Plotting in \(d>2\):
  - density/path plots are projected to first two embedding dimensions and written as `*_proj12.png`.

## 6. Current Limitations
List known architectural constraints, debt, or risks.

| Limitation | Impact | Mitigation |
| --- | --- | --- |
| Single-cell strict leaveout currently targets EB 5D first | Cross-dataset/high-dimension generalization still unverified | Extend presets to CITE/MULTI and 50D/100D profiles after EB baseline stabilizes |
| Exact discrete OT is \(O(n^3)\) per batch | Limits large batch experiments | Introduce Sinkhorn approximation for larger-scale runs |
| Exact balanced LP OT for `ot_global` and full-set robust metrics can be memory-heavy (\(n\times m\) variables) | Large endpoint pools may make robust evaluation/precompute slow or infeasible on limited hardware | Use cache reuse, `*_max_variables` guards, and fall back to legacy sampled metrics when needed |
| Intermediate empirical OT metric is \(O(n^3)\) in eval sample count | Can be noisy at very small sample budgets and slow at large budgets | Start with a small dedicated `eval_intermediate_ot_samples` budget, then increase with experiment complexity (or use approximate OT) |
| Bridge integration currently emphasizes Stage-A-only and A+B (`ab_only`) workflows | Stage-C behavior on bridge data remains underexplored | Add dedicated Stage-C bridge sweeps after A+B behavior is stable |
| MFM first pass only implements LAND metric objective | Cannot yet compare LAND vs RBF metric-learning variants in this repo | Add RBF metric-learning option in a follow-up |
| `mfm.backend=auto` may fall back to native implementation when `torchcfm` is unavailable | Results can differ from a strict torchcfm-authors stack | Record `mfm_backend` fields in summaries and rerun with explicit `torchcfm` when environment is ready |
| Pseudo-types are unsupervised latent classes, not curated biological labels | Constraints may optimize mixture structure that is not directly interpretable as canonical cell types | Keep pseudo constraints optional/additive and validate with marker-driven external checks when annotations become available |
| v1 moments are mean/cov only | Constraint class may be too weak for complex trajectories | Add nonlinear feature maps in follow-up |

## 7. Planned Architectural Changes
Track intended architecture-level evolution.

| Planned Change | Motivation | Expected Impact | Target Window |
| --- | --- | --- | --- |
| Add penalty-only baseline variant | Compare AL vs fixed penalties | Stronger ablation evidence | v1.1 |
| Add noisy-moment training mode | Move toward realistic observations | Better applicability to real data | v1.2 |
| Add higher-dimensional benchmarks | Stress-test optimization and OT choices | Improved method confidence | v2 |

## 8. Maintenance Notes
- Keep this document concise and structural; prefer stable explanations over session details.
- Session-by-session implementation history belongs in `PROJECT_STATE.md`.
- Whenever `PROJECT_STATE.md` records a structural change, update this file in the same session.
