# Public Stack Pivot

This experiment family pivots away from the earlier recurrent/shared-core path and instead starts from a strong public baseline already merged in `records/`.

Current goal:

- reproduce strong public stacks on `1x H100`
- make one narrow tuning change at a time
- measure post-roundtrip `val_bpb`, artifact bytes, total submission bytes, and `ms/step`
- only escalate to `8x H100` once the `1x H100` direction looks strong enough

Current result:

- `results/2026-03-21_8xh100_10l_fp16embed_wd20k_seed42/`
  - first real `8x H100` verification run on the pivot branch
  - same 10-layer public stack with `WARMDOWN_ITERS=20000`
  - post-roundtrip `val_bpb: 1.17389939`, slightly better than the merged seed-42 baseline
- `results/2026-03-21_h100_10l_fp16embed_wd20k_seed42/`
  - based on `2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit`
  - single-GPU directional run with `WARMDOWN_ITERS=20000`
- `results/2026-03-21_h100_10l_fp16embed_wd20k_eval2048_seed42/`
  - same `1x H100` run with `EVAL_SEQ_LEN=2048`
  - worse `val_bpb` and much slower roundtrip eval than the `1024` eval baseline
- `results/2026-03-21_h100_10l_fp16embed_wd2500_seed42/`
  - same `1x H100` run with the record-default `WARMDOWN_ITERS=2500`
  - worse `val_bpb` and larger artifact than `WARMDOWN_ITERS=20000`
- `results/2026-03-21_h100_10l_fp16embed_sweep/summary.md`
  - three-run comparison table and current recommendation

Current best result:

- `8x H100`, `WARMDOWN_ITERS=20000`
- post-roundtrip `val_bpb: 1.17389939`
- artifact `14122782` bytes
- total submission size `14178772` bytes
- `step_stop: 10919`
- `ms/step: 54.96`

Best single-H100 tuning result:

- `WARMDOWN_ITERS=20000`
- `EVAL_SEQ_LEN=1024`
- post-roundtrip `val_bpb: 1.28768222`
- artifact `13385911` bytes
