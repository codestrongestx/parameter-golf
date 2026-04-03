# Repo Operating Guide

## Purpose

This file is the stable map for how this repo is organized and how ongoing
research should be documented.

## Repo Map

- `train_gpt.py`, `train_gpt_mlx.py`, `data/`, `records/`: core project code and
  official record artifacts.
- `.agents/skills/`: Codex skill definitions and local agent workflow notes.
- `scripts/`: reusable operational helpers that are not tied to a single
  experiment campaign.
- `experiments/active/`: campaigns that are still being actively worked.
- `experiments/benchmarks/`: replay and measurement harnesses used across
  campaigns.
- `experiments/archive/`: older experiment trees kept for reference, not current
  planning.

## Read Order

When resuming work, read in this order:

1. `experiments/README.md`
2. the relevant active campaign `STATUS.md`
3. that campaign's `notes/strategy.md`
4. that campaign's `notes/run_observations.md`

## Documentation Rules

- Keep one canonical strategy file per campaign.
- Keep one `STATUS.md` per active campaign. This should answer:
  - what is currently believed
  - what is dead
  - what should be run next
- If a result folder is removed or not preserved in git, summarize the decision
  in `STATUS.md` or the campaign notes before deleting it.
- Reusable infra belongs in `scripts/`, not inside a campaign folder.

## Branch Notes

- `develop` is the operational mainline.
- Experimental branches may exist, but the repo layout should still make sense
  when viewed from `develop` alone.
