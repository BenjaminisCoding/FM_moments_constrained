# Generalized-Moment Interpolant Flow Matching

Code and prepared inputs for experiments comparing CFM, MFM, constrained
Schrödinger bridges, and generalized-moment flow matching (GMI).

## Installation

Use Python 3.12 or newer. Run these commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reproduction.txt
python -m pip install -e .
```

The requirements file pins the dependencies for experiments, tests and raw-data
preprocessing. Experiments use CPU by default.

## Run one experiment

Verify the bundled inputs, train GMI on the synthetic benchmark, then evaluate
the saved model:

```bash
python scripts/run_paper.py prepare
python scripts/run_paper.py train --benchmark synthetic --method lin --seed 3 \
  --out outputs/example
python scripts/run_paper.py evaluate --out outputs/example
```

- Benchmarks: `synthetic`, `aerosol`, `multi`, `semrau`.
- Methods: `cfm`, `mfm`, `csb` (constrained Schrödinger bridge),
  `lin` (GMI-linear), `land` (GMI-LAND).
- Seeds: `3`, `7`, `11`, `13`, `17`.

For Multi, add `--study constraint_day3_eval_day4` (the default) or
`--study constraint_day4_eval_day3` to the training command.
Evaluation reads the method and configuration from the saved run.
Use a new output directory for each configuration and seed.

## Run the full comparison

Run all main benchmarks and methods with the five fixed seeds, then collect
their scores and timings:

```bash
python scripts/paper_suite.py --suite main --out outputs/paper/main --execute
python scripts/report_paper.py --root outputs/paper/main
```

Omit `--execute` to preview the commands. Other suites are `complementary`,
`redundant`, `scarce`, `noise` and `runtime`; use a separate output directory
for each. The runtime suite uses seed 3.

Each run saves its configuration, checkpoints and metrics. The main comparison
report writes `paper_report.json` and `paper_scores.csv` under the supplied root.

## Checks and configuration

```bash
python -m pytest -q
python scripts/run_paper.py train --benchmark synthetic --method lin --smoke \
  --out outputs/smoke/synthetic
```

The smoke run checks execution with a reduced budget and is excluded from
scientific reports.

Experiment settings are in [`configs/paper/`](configs/paper/).
The [experiment manifest](configs/paper/manifest.json) maps commands to paper
tables and describes budgets and validation protocols.
Prepared inputs are included in [`benchmark_data/`](benchmark_data/);
`prepare` verifies their checksums. See the
[preprocessing guide](configs/preprocessing/usage.json) for raw-data preparation
and custom datasets, or run `python scripts/run_paper.py --help` for CLI options.
