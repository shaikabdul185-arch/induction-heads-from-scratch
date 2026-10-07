# CLAUDE.md

## Rules

- Describe changes in project terms only; never reference earlier repository names or renames in commits, PR titles, PR descriptions, or docs.

## Project notes

- Package code lives in `induction/`; hyperparameters only in `configs/*.yaml` (mirrored by `induction/config.py`).
- Train: `python -m induction.train --config configs/default.yaml`. Figures: `python -m induction.analyze --run runs/default`.
- Tests: `python -m pytest`.
- Report the transition step that the run actually produces; never tune hyperparameters to move it.
