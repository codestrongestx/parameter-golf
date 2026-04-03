# 1xH100 Pivot Run: 10L FP16 Embed + WARMDOWN_ITERS=2500

Single-GPU control run against the `WARMDOWN_ITERS=20000` baseline.

What changed:

- kept the same patched 10-layer public-stack script
- kept `TRAIN_BATCH_TOKENS=65536`
- kept `EVAL_STRIDE=64`
- changed only `WARMDOWN_ITERS` from `20000` back to the record default `2500`

Key metrics:

- `step_stop: 5590`
- `ms/step: 107.35`
- `pre-roundtrip val_bpb: 1.3291`
- `post-roundtrip val_bpb: 1.29406563`
- `artifact bytes: 15051498`
- `total submission bytes: 15107488`
- `roundtrip eval_time: 279739 ms`

Takeaway:

- this lost to the `WARMDOWN_ITERS=20000` run on both quality and compression
- `val_bpb` got worse by `+0.00638341`
- compressed artifact size grew by `1665587` bytes

Read:

- `train.log` for the exact run
- `train_gpt.py` for the exact patched script that ran
- `command.txt` for the launch configuration
