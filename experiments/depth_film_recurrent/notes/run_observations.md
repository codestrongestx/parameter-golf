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
