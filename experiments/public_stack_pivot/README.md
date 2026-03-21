# Public Stack Pivot

This experiment family pivots away from the earlier recurrent/shared-core path and instead starts from a strong public baseline already merged in `records/`.

Current goal:

- reproduce strong public stacks on `1x H100`
- make one narrow tuning change at a time
- measure post-roundtrip `val_bpb`, artifact bytes, total submission bytes, and `ms/step`
- only escalate to `8x H100` once the `1x H100` direction looks strong enough

Current result:

- `results/2026-03-21_h100_10l_fp16embed_wd20k_seed42/`
  - based on `2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit`
  - single-GPU directional run with `WARMDOWN_ITERS=20000`
