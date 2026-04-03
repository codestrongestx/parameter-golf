# 8xH100 Pivot Run: 10L FP16 Embed + WARMDOWN_ITERS=20000

First real `8x H100` verification run for the public-stack pivot branch.

Base:

- merged record `2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit`
- kept the same 10-layer stack and global batch shape
- changed one tuning knob: `WARMDOWN_ITERS=20000`
- used the same patched script that passed the single-H100 sweep because this PyTorch 2.4 image needs the SDPA GQA fallback

Key metrics:

- `step_stop: 10919`
- `ms/step: 54.96`
- `pre-roundtrip val_bpb: 1.2055`
- `post-roundtrip val_bpb: 1.17389939`
- `artifact bytes: 14122782`
- `total submission bytes: 14178772`
- `roundtrip eval_time: 57969 ms`

Comparison to the merged seed-42 baseline:

- merged seed-42 baseline: `1.17423973`
- this run: `1.17389939`
- improvement: `0.00034034`

Takeaway:

- the warmdown-20k tweak survived the jump from `1x H100` screening to a real `8x H100` run
- the run stayed comfortably under the `16MB` cap
- this is strong enough to package as the PR candidate for the pivot branch

Read:

- `train.log` for the exact run
- `train_gpt.py` for the exact patched script that ran
- `command.txt` / `command.sh` for the launch configuration
