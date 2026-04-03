# Leaderboard 1xH100 Replay Status

## Purpose

Use `1xH100` replays as a relative screen, not as a direct estimator of the
reported `8xH100` leaderboard score.

The question is:

- which directions still look strong when the 600s budget only buys about
  `900` steps on a single H100

## Completed Replays

### 2026-03-29 Official 11L GPTQ-lite Baseline

Source:

- merged record `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`

Result:

- steps: `903`
- `ms/step`: `664.67`
- pre-roundtrip `val_bpb`: `1.3930`
- post-EMA `val_bpb`: `1.5098`
- final roundtrip exact `val_bpb`: `2.25578828`
- final sliding exact `val_bpb`: `2.24658520`

Interpretation:

- this stack does not just lose throughput on `1xH100`; it loses a large amount
  of quality under the reduced step budget

### 2026-03-29 PR #1060 Loader + Full GPTQ + XSA-all

Source:

- `PR #1060`

Result:

- steps: `899`
- `ms/step`: `652.34`
- pre-roundtrip `val_bpb`: `1.3738`
- post-EMA `val_bpb`: `1.5044`
- final roundtrip exact `val_bpb`: `1.79060061`
- final sliding exact `val_bpb`: `1.77129631`

Interpretation:

- this is the strongest credible official-style neural direction screened so far
- the main gain shows up in the quantized/exported model quality, not just in
  float pre-roundtrip loss

## Non-Canonical Direction

### PR #875

Do not use this as the main replay target yet.

Why:

- its bundled judge scores only a small validation slice
- its BPB calculation is not the canonical tokenizer-derived full-validation path

## Decision

The best next import candidate for the mainline official-neural lane is the
`PR #1060` stack, not the older March 22 baseline and not the DeltaNet branch.
