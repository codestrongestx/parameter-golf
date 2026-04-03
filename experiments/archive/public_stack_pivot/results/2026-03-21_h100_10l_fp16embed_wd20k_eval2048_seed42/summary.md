# 1xH100 Pivot Run: 10L FP16 Embed + WARMDOWN_ITERS=20000 + EVAL_SEQ_LEN=2048

Single-GPU directional follow-up to the `WARMDOWN_ITERS=20000` baseline.

What changed:

- kept the same patched 10-layer public-stack script
- kept `TRAIN_BATCH_TOKENS=65536`
- kept `WARMDOWN_ITERS=20000`
- changed only `EVAL_SEQ_LEN` from `1024` to `2048`

Key metrics:

- `step_stop: 5869`
- `ms/step: 102.25`
- `pre-roundtrip val_bpb: 1.3203`
- `post-roundtrip val_bpb: 1.30287290`
- `artifact bytes: 13355586`
- `total submission bytes: 13411576`
- `roundtrip eval_time: 532571 ms`

Takeaway:

- this was a miss versus the `EVAL_SEQ_LEN=1024` baseline
- `val_bpb` got worse by `+0.01519068`
- roundtrip eval got much slower, from `272362 ms` to `532571 ms`

Read:

- `train.log` for the exact run
- `train_gpt.py` for the exact patched script that ran
- `command.txt` for the launch configuration
