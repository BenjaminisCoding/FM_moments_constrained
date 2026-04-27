# AGENTS.md

## Purpose
This file is the authoritative behavioral policy for contributors and coding agents in this repository.
It defines required documentation maintenance rules that must be followed for every meaningful change.

## Required Living Documents
The following files are mandatory maintenance artifacts:
- `ARCHITECTURE.md`: global code/system map, component roles, and key interaction flow.
- `PROJECT_STATE.md`: append-only dated history of changes, decisions, risks, and next steps.
- `DISCUSSION.md`: alternatives, tradeoffs, rejected directions, and rationale for chosen methodology.
- `EXPERIMENTS.md`: index of experiment output folders and what each run/sweep tested.

## Repository Boundary
`FM/` is environment tooling (Python virtual environment) and is not project source architecture.
Do not treat `FM/` internals as architectural modules of this project.

## Hard-Gate Completion Rule
A meaningful code or design change is not complete unless required documentation updates are made in the same session.

Meaningful change includes:
- New feature behavior.
- Structural refactor.
- Public interface or data flow change.
- Non-trivial bug fix with design implications.
- Change that affects onboarding understanding of the codebase.

## Exception Rule
Trivial edits with no architecture impact may skip `ARCHITECTURE.md` updates.
When relevant, still add a short note in `PROJECT_STATE.md` describing what changed and why architecture remained unchanged.

## Default Update Policy
When in doubt, update both `ARCHITECTURE.md` and `PROJECT_STATE.md`.
Documentation updates are expected per meaningful change, not only by milestone or end-of-day batching.

## Experiment Logging Rule
Every newly executed experiment must be logged in `EXPERIMENTS.md` in the same session.
For single runs, include the exact output folder path.
For sweeps, include the sweep root and the variant subfolders (or explicit selection criteria if variants are many).

## Done Checklist (Must Be True Before Closing a Task)
- Code and behavior changes are implemented as requested.
- Validation status is recorded (tests/checks run, or explicit note if not run).
- `PROJECT_STATE.md` contains a dated entry for the session when relevant.
- `ARCHITECTURE.md` is updated if file roles, key functions, or system flow changed.
- `DISCUSSION.md` is updated if methodological alternatives or tradeoffs changed.
- `EXPERIMENTS.md` is updated for every new run/sweep with folder paths and test intent.
- Next planned action is captured in `PROJECT_STATE.md`.

## Governance Verification Scenarios
- Meaningful feature change: expect a new `PROJECT_STATE.md` entry and `ARCHITECTURE.md` updates when structure or flow changed.
- Small non-architectural refactor: expect a `PROJECT_STATE.md` entry that explicitly states architecture remained unchanged.
- New experiment execution: expect an `EXPERIMENTS.md` entry with output paths and a short description of what was tested.
- Session close check: all items in the Done Checklist can be answered "yes" before marking task complete.
- New collaborator onboarding: `ARCHITECTURE.md`, `PROJECT_STATE.md`, `DISCUSSION.md`, and `EXPERIMENTS.md` together should explain purpose, key files, current direction, rejected alternatives, and empirical progress.
