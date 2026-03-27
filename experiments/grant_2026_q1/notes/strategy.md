# Strategy

## State Of Play

- The merged official board is much weaker than the pending frontier.
- The strongest pure-neural path appears to be the `#549 -> #609 -> #728` line.
- The strongest raw-score path appears to be `#728`-grade neural base plus the `#870 / #888` full-rescore n-gram method.
- Tokenizer work is a separate high-upside, high-scrutiny lane.

## Lanes

### Official Neural

Goal:

- maximize the chance of an accepted official-style record attempt

Starting point:

- reproduce `#549`
- then upgrade toward `#728` and `#609`

Default stack direction:

- `11L / 512d / 8H / 4KV`
- `MLP 3x`
- `LeakyReLU(0.5)^2`
- `XSA` on all layers
- `Partial RoPE 16/64`
- `BigramHash` around `3072x112`
- `VE128` late layers
- `SmearGate + U-Net skips`
- `EMA 0.997`
- `Parameter Banking + Parallel Muon`
- self-generated full GPTQ int6 + strong compression
- selective `+/-1` pruning as late ablation

### N-Gram Rescore

Goal:

- best raw score, even if review risk is higher

Starting point:

- only branch here after the official-neural base is strong and stable
- use the best available neural base under the rescore mixer

### Tokenizer Wildcard

Goal:

- isolate tokenizer work so it never contaminates the safer mainline

Rule:

- do not make this the first mainline bet
- only pursue if BPB verification is extremely clean

## Tactical Order

1. reproduce the strongest accepted-style neural stack
2. upgrade it with the best pieces from `#609` and `#728`
3. only then branch into full-rescore n-gram
4. keep tokenizer work fully separate

## What Counts As Progress

Track only:

- post-roundtrip `val_bpb`
- artifact bytes
- unique parameter count
- `ms/step`
