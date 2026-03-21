# Run Observations

## 2026-03-21

- Goal was a cheapest-possible single-GPU CUDA smoke test for `experiments/depth_film_recurrent/train_gpt.py` before considering any larger hardware.
- Validation succeeded on `1x NVIDIA GeForce RTX 3090` on RunPod. This was enough to prove the basic CUDA path and did not require moving to H100.
- The first CUDA attempt failed immediately on the RunPod PyTorch 2.4 image because `torch.nn.functional.scaled_dot_product_attention` did not accept the `enable_gqa` keyword on that build.
- Fix: add a compatibility fallback that manually expands KV heads when grouped-query attention is needed and the runtime SDPA API does not support `enable_gqa`.
- The second issue was economic rather than correctness-related: `torch.compile` spent far too long in Inductor startup on the cheap GPU, which made the smoke run unnecessarily slow.
- Fix: add `ENABLE_TORCH_COMPILE=0/1` so smoke runs can skip compile overhead while larger runs can still opt back in.
- A compile-enabled retry eventually ran, but it spent roughly 70 seconds just to reach step 1. That confirmed compile was the wrong default for this tier of smoke validation.
- The clean validating run used `ENABLE_TORCH_COMPILE=0`, `ITERATIONS=2`, `TRAIN_BATCH_TOKENS=16384`, `TRAIN_SEQ_LEN=256`, `VAL_BATCH_SIZE=4096`, and `VAL_MAX_TOKENS=1048576`.
- Clean 3090 smoke result:
  - `val_loss: 7.8441`
  - `val_bpb: 4.6998`
  - `peak memory allocated: 419 MiB`
  - `Serialized model int8+zlib: 2523469 bytes`
  - `final_int8_zlib_roundtrip val_loss: 8.1560`
  - `final_int8_zlib_roundtrip val_bpb: 4.8866`
- Conclusion: the idea is validated enough on cheap single-GPU CUDA to justify later `1x H100` work if we want larger settings or faster iteration, but H100 is not required just to prove the path works.

## 2026-03-21 H100 Ablations

- Ran the planned single-GPU ablations on `1x NVIDIA H100 80GB HBM3` with `ENABLE_TORCH_COMPILE=0`, `TRAIN_SEQ_LEN=1024`, `TRAIN_BATCH_TOKENS=65536`, `MAX_WALLCLOCK_SECONDS=600`, and the full cached `sp1024` validation split.
- Shared blocks only (`NUM_LAYERS=12 NUM_UNIQUE_LAYERS=4 DEPTH_CONDITION_DIM=0`) was the cleanest baseline:
  - `post-roundtrip val_bpb: 1.4863`
  - `artifact bytes: 8663549`
  - `unique parameter count: 9972768`
  - `ms/step: 343.88`
- Plain DepthFiLM on the same shared-block stack (`DEPTH_CONDITION_DIM=32`) was not a win at this scale:
  - `post-roundtrip val_bpb: 1.4886`
  - `artifact bytes: 9400467`
  - `unique parameter count: 10243488`
  - `ms/step: 391.06`
- That means the DepthFiLM path added `270720` unique parameters, increased step time by about `13.7%`, and slightly worsened post-roundtrip quality relative to shared blocks only.
- The stronger frontier-style variant kept the same DepthFiLM core but turned on `EVAL_STRIDE=64` and `INT4_BLOCKS=0,1,2,3` with `INT4_STEP=4`.
  - `post-roundtrip val_bpb: 1.4596`
  - `artifact bytes: 7496799`
  - `unique parameter count: 10243488`
  - `ms/step: 391.01`
- Conclusion: the `1x H100` phase did not validate plain DepthFiLM as the next architectural move on top of shared blocks, but it did validate the stronger frontier-style package built on that recurrent core. For a non-record follow-up, the strongest measured option in this experiment family is the frontier-style `12/4/32` run with sliding-window roundtrip eval and reduced-precision shared-block export.
- Operational note: the frontier-style sliding-window roundtrip eval was much more expensive than the full-context roundtrip path. Its final post-roundtrip eval alone took `1227327 ms` (about `20.5 minutes`) after training/export, so future H100 sweeps should treat that metric path as a meaningful cost center.
