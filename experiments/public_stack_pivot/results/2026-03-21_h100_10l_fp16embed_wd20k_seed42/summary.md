# 1xH100 Pivot Run: 10L FP16 Embed + Sliding Eval + WARMDOWN_ITERS=20000

Single-GPU directional run built from the merged record `2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit`.

Main differences from the merged 8xH100 record:

- translated to `1x NVIDIA H100 80GB HBM3`
- reduced to `TRAIN_BATCH_TOKENS=65536` with `grad_accum_steps: 8`
- downloaded `10` train shards for a quicker directional run
- changed only one main tuning knob: `WARMDOWN_ITERS=20000`
- patched the script with the SDPA GQA fallback already validated in `experiments/depth_film_recurrent/`

Key metrics:

- `step_stop: 5974`
- `ms/step: 100.45`
- `pre-roundtrip val_bpb: 1.3199`
- `post-roundtrip val_bpb: 1.28768222`
- `artifact bytes: 13385911`
- `total submission bytes: 13441901`
- `roundtrip eval_time: 272362 ms`

Read:

- `train.log` for the exact run
- `train_gpt.py` for the exact patched script that ran
- `command.txt` for the launch configuration
