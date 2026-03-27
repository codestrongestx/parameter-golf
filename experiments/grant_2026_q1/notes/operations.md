# Operations

## Budget

- treat `$20-$50` as the budget for the whole first campaign
- use `1x H100` as the normal iteration machine
- use `8x H100` once at the very end, only for final verification

## Run Ladder

1. local sanity work
2. `3090`-class CUDA smoke only if a risky systems/code change needs the cheapest GPU check
3. `1x H100` for real iteration
4. `8x H100` once at the end

The previous `3090` work already proved:

- CUDA path works
- training works
- validation works
- export works
- int8+zlib roundtrip works

So `3090` is optional now, not mandatory.

## One-Run Rule

Every paid run must answer one explicit question.

Good examples:

- does `#728`-style GPTQ beat the current `#549` reproduction?
- does `#609` pruning help after full GPTQ?
- does the rescore lane still win after swapping in a stronger neural base?

Bad examples:

- broad kitchen-sink changes with no isolated question

## Preservation

Do not keep scratch output as the record.

Preserve only curated artifacts under the lane's `results/` folder with:

- `command.txt`
- `metrics.json`
- `summary.md`
- `train.log`
- `eval.log` when needed

Add human conclusions to:

- `notes/run_observations.md`

## Cost Discipline

- stop or delete the GPU as soon as the run is finished
- idle time is wasted budget
- treat evaluation time as a real cost center

Known lesson from previous work:

- sliding-window roundtrip eval can dominate wall-clock time
- do not assume training is the only expensive phase

## Known Engineering Lessons

- older PyTorch builds may not support `enable_gqa` in `scaled_dot_product_attention`
- `torch.compile` can be the wrong choice for cheap smoke validation
- if compile overhead dominates, disable it for smoke runs

## Hard No's

- no pre-eval TTT on unscored tokens
- no post-budget use of training data for calibration
- no selecting between cache and LM after seeing the true next token
