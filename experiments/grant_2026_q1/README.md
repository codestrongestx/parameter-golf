# Grant 2026 Q1

This folder is the private planning and execution area for the first `$20-$50`
of grant-funded iteration.

The active plan is:

- use `3090` first to kill weak directions cheaply
- move only promising directions to `1x H100`
- keep `8x H100` out of scope for now
- decide later, from the accumulated docs, observations, and logs, whether any lane deserves `8x H100`

We keep one branch for the whole campaign:

- `grant-2026-q1`

We keep separate lanes inside this folder:

- `official_neural/`: mainline, safest official-track work
- `ngram_rescore/`: highest raw upside, higher review risk
- `tokenizer_wildcard/`: separate moonshot, highest verification burden

Shared notes live under `notes/`.

Per-run curated evidence should go under:

- `official_neural/results/YYYY-MM-DD_<gpu>_<run_id>/`
- `ngram_rescore/results/YYYY-MM-DD_<gpu>_<run_id>/`
- `tokenizer_wildcard/results/YYYY-MM-DD_<gpu>_<run_id>/`

Each preserved run folder should contain:

- `command.txt`
- `metrics.json`
- `summary.md`
- `train.log`
- `eval.log` when evaluation is separate or unusually expensive

Read first:

- `notes/strategy.md`
- `notes/operations.md`
- `notes/run_observations.md`
