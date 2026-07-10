# Conductor Eval Agent Guide

## Scope

This repository owns evaluation orchestration, deterministic MIDI checks,
result reporting, and the optional Dash analysis UI. Generation must go through
the public `conductor-core` engine rather than routing providers or persisting
generation artifacts inside Eval.

## Key paths

- `src/conductor_eval/evaluator.py`: evaluation orchestration.
- `src/conductor_eval/checks.py`: deterministic MIDI checks.
- `src/conductor_eval/analysis.py`: optional dashboard and exports.
- `tests/`: evaluator, boundary, and direct-run guard coverage.

## Working rules

- Keep evaluation as a consumer of `LoopGenerationEngine`.
- Do not broaden model matrices, run paid providers, or start broad evaluations without explicit approval.
- Preserve the direct-run confirmation guard for expensive examples.
- Keep Dash, pandas, and Plotly in the dashboard extra.
- Use package-relative or configurable output paths suitable for a standalone checkout.
- Do not commit evaluation outputs, credentials, build artifacts, or planning files.

## Validation

Install a compatible `conductor-core`, then run:

```powershell
python -m ruff format --check .
python -m ruff check .
python -m pytest -q
python -m build
```

Use deterministic tests for ordinary validation. Before a commit, inspect
`git status` and the intended diff and keep generated evaluations uncommitted.
