# Repo Operating Guide

## Purpose

This file is the stable map for how this repo is organized and how ongoing
research should be documented.

## Repo Map

- `train_gpt.py`, `train_gpt_mlx.py`, `data/`, `records/`: core project code and
  official record artifacts.
- `.agents/skills/`: Codex skill definitions and local agent workflow notes.
- `scripts/`: reusable operational helpers that are not tied to a single
  experiment idea.
- `experiments/active/`: current live directions. Keep one idea per folder.
- `experiments/benchmarks/`: replay and measurement harnesses used across
  directions.
- `experiments/archive/`: older experiment trees kept for reference, not current
  planning.

## Read Order

When resuming work, read in this order:

1. `experiments/README.md`
2. `experiments/active/README.md`
3. the relevant active direction `README.md` or `STATUS.md`
4. the relevant benchmark `STATUS.md` if active work is still being scoped

## Documentation Rules

- Keep one active folder per live idea.
- Keep one canonical strategy file per direction only when the direction is
  large enough to need one.
- Keep one `STATUS.md` per active direction when the work spans multiple runs.
  This should answer:
  - what is currently believed
  - what is dead
  - what should be run next
- If a result folder is removed or not preserved in git, summarize the decision
  in `STATUS.md` or the direction notes before deleting it.
- Archive obsolete multi-lane bundles instead of leaving them under
  `experiments/active/`.
- Reusable infra belongs in `scripts/`, not inside an experiment folder.

## Branch Notes

- `develop` is the operational mainline.
- Experimental branches may exist, but the repo layout should still make sense
  when viewed from `develop` alone.
