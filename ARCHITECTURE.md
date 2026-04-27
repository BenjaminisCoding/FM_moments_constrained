# ARCHITECTURE.md

## Document Status
- Last updated: 2026-04-27
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
- constrained-path CFM with learned correction.

### In Scope
- PyTorch + Hydra research pipeline running on CPU.
- Configurable batch coupling for training pairs (`ot` exact discrete OT or `random` permutation coupling).
- Exact-formula moment targets at times `t = {0.25, 0.50, 0.75}`.
- Pre-training bridge-SDE distribution preview utilities and notebook workflow.
- Stage A/B/C training schedule for constrained mode.
- A+B-only constrained ablation profile (`stage_c_steps = 0`) for isolating pretrain-then-CFM behavior.
- Baseline and constrained comparison artifacts (metrics JSON, checkpoint, plots).

### Out of Scope
- Real-data pipelines and domain-specific physical simulators.
- High-dimensional scaling and production optimization.
- Noisy or partially observed intermediate moments (planned future work).

## 2. High-Level Component Map
Document the system as major components and their responsibilities.

| Component | Responsibility | Inputs | Outputs | Dependencies |
| --- | --- | --- | --- | --- |
| `data` | Gaussian problem setup, coupling sampling (`ot`/`random`), analytic target moments | config means/covs/kappa/coupling | paired batches, target feature vectors | `torch`, `scipy` |
| `bridge_sde` | Bridge-process trajectory simulation and time-slice sampling | SDE params + seed | trajectory tensor and snapshot samples | `torch` |
| `models` | `v_psi` velocity net and `g_theta` path-correction net | tensors `(t, x)` or `(t, x0, x1)` | predicted velocity or correction | `torch.nn` |
| `paths` | linear/corrected paths and constrained velocity target computation | paired points + time + `g_theta` | `x_t`, `u_t`, derivatives in `t` | `torch.autograd` |
| `constraints` | moment features, residuals, augmented Lagrangian terms | path samples and target moments | residual vectors, AL loss, multiplier updates | `torch` |
| `training` | stage-wise optimization (A/B/C), eval summaries | config + problem + targets | trained models, histories, metrics | core modules |
| `pipeline` | orchestration, artifact writing, baseline/constrained comparison | Hydra config | run outputs under `outputs/` | all runtime modules |
| `plotting` | diagnostics plots for loss/residuals/path geometry and bridge SDE previews | histories/residuals/models or trajectory samples | PNG/GIF artifacts | `matplotlib` |

## 3. File and Folder Responsibilities
Use this section to quickly understand what each important path does.

| Path | Role | Key Contents | Notes |
| --- | --- | --- | --- |
| `/` | Project root | Governance docs, configs, scripts, source, tests | Keep intent + method + state discoverable |
| `FM/` | Environment tooling only | Virtual environment binaries and packages | Not part of product architecture |
| `DISCUSSION.md` | Research design log | Alternatives, tradeoffs, rejected options | Complements chronological `PROJECT_STATE.md` |
| `EXPERIMENTS.md` | Experiment index log | Output folders grouped by run/sweep with test intent | Complements chronological `PROJECT_STATE.md` with artifact traceability |
| `src/cfm_project/` | Core implementation | data/models/paths/constraints/training/pipeline | Main research code |
| `notebooks/` | Interactive diagnostics | bridge-SDE preview and visual checks | reusable analysis notebooks | Uses shared `src/cfm_project` helpers |
| `configs/` | Hydra configuration | experiment/data/model/train groups (including coupling mode) | Controls reproducible runs |
| `scripts/run_experiment.py` | Entry point | Hydra main invoking pipeline | Supports baseline/constrained/comparison modes |
| `outputs/` | Experiment artifacts | date/label/time runs, sweep folders, seed checks | Includes per-mode metrics and comparison summary |
| `tests/` | Verification suite | unit + smoke tests | CPU-first regression safety |

## 4. Key Functions, Modules, and Interaction Flow
Capture the most important execution flow and key functions/modules.

### Key Functions or Modules
| Module or Function | Responsibility | Called By | Calls Into |
| --- | --- | --- | --- |
| `data.analytic_target_moment_features` | Build exact intermediate moment targets | pipeline/training | Gaussian OT map helpers |
| `bridge_sde.simulate_bridge_sde_trajectories` | Simulate bridge SDE trajectories in 2D | notebook/tests (future data mode) | Gaussian initial sampling + Euler-Maruyama |
| `paths.corrected_path` | Endpoint-preserving corrected interpolation | training/evaluation | `g_theta` model |
| `paths.corrected_velocity` | Compute analytic target velocity for constrained CFM | training | autograd derivative wrt `t` |
| `constraints.augmented_lagrangian_terms` | Compute AL constraint objective | stage A / stage C | residual vectors + multipliers |
| `training.train_experiment` | Execute selected training schedule and evaluations | pipeline | all core modules |
| `pipeline.run_pipeline` | Artifact orchestration and optional mode comparison | script entrypoint | training + plotting + IO |

### Interaction Flow
1. Hydra loads config and dispatches `pipeline.run_pipeline`.
2. Pipeline builds Gaussian OT problem and exact target moments.
3. Trainer runs baseline or constrained schedule:
   - constrained: Stage A (`g_theta` with AL) -> Stage B (`v_psi` CFM) -> optional Stage C (joint, disabled when `stage_c_steps=0`).
   - baseline: Stage B only with linear path CFM.
4. Evaluation computes constraint residual norms, CFM validation loss, intermediate-time distribution metrics (Gaussian \(W_2\) and sample-based empirical \(W_2\)), and transport-quality metrics.
5. Pipeline saves metrics JSON, checkpoint, and plots.
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
- Feature map in v1: mean + flattened covariance in 2D.
- Residual at constrained times \(t_k \in \{0.25, 0.50, 0.75\}\):
  \[
  c_k(\theta) = \mathbb{E}[\phi(x_{t_k})] - m_k.
  \]

### Path Prior
\[
R(\theta)=\alpha\,\mathbb{E}\|u_t-(x_1-x_0)\|^2+\beta\,\mathbb{E}\|\partial_t u_t\|^2.
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

### Evaluation Metric Extension
- Intermediate-time distribution quality includes Gaussian \(W_2\) diagnostics at constrained times \(t_k\), computed between empirical generated samples (integrating \(v_\psi\) up to \(t_k\)) and analytic Gaussian targets \((\mu_{t_k}, \Sigma_{t_k})\).
- Intermediate-time distribution quality also includes sample-based empirical \(W_2\) diagnostics using exact discrete OT matching between generated samples and target samples at each \(t_k\), with a separate small evaluation sample budget for CPU tractability.
- Operational note: `eval_intermediate_ot_samples` is intentionally small in default configs; when scaling to more complex/non-Gaussian experiments, increase this budget (or switch to approximate OT/Sinkhorn diagnostics) before drawing conclusions from empirical \(W_2\).

## 6. Current Limitations
List known architectural constraints, debt, or risks.

| Limitation | Impact | Mitigation |
| --- | --- | --- |
| 2D Gaussian benchmark only | Unknown scaling behavior in high dimensions | Add dimensionality sweep after v1 stabilization |
| Exact discrete OT is \(O(n^3)\) per batch | Limits large batch experiments | Introduce Sinkhorn approximation for larger-scale runs |
| Intermediate empirical OT metric is \(O(n^3)\) in eval sample count | Can be noisy at very small sample budgets and slow at large budgets | Start with a small dedicated `eval_intermediate_ot_samples` budget, then increase with experiment complexity (or use approximate OT) |
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
