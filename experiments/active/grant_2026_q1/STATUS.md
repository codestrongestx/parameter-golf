# Grant 2026 Q1 Status

## Read This First

- canonical strategy: `notes/strategy.md`
- operating notes: `notes/operations.md`
- rolling campaign log: `notes/run_observations.md`

## Current Picture

- `official_neural` remains the mainline lane.
- `ngram_rescore` exists, but should still branch only after the neural base is
  strong enough to justify it.
- `tokenizer_wildcard` is still isolated and unopened for normal mainline work.

## Current Best Takeaways

- The old March 22 official-style neural baseline collapses badly on `1xH100`
  under the 600s budget.
- The newer `PR #1060` direction survives that `1xH100` screen much better.
- The main useful import from the replay work is:
  - coprime-stride loader
  - full Hessian GPTQ within the reserved train budget
  - XSA on all 11 layers
  - `BigramHash(2816x112)`
- `PR #875` should not be treated as a canonical replay target until its
  evaluation path is ported to the standard full-validation scorer.

## Benchmark Snapshot

See `experiments/benchmarks/leaderboard_1xh100_replay/STATUS.md`.

Current key numbers from that benchmark:

- March 22 baseline replay:
  - `903` steps
  - `664.67 ms/step`
  - final sliding exact `2.24658520`
- `PR #1060` replay:
  - `899` steps
  - `652.34 ms/step`
  - final sliding exact `1.77129631`

## Next Recommended Work

1. Import the `PR #1060` ideas into the official-neural lane cleanly.
2. Re-run the strongest official-style neural path after that import.
3. Only then reopen `ngram_rescore`.
