# RTX 3090 CUDA Smoke

This was the first successful cheap-GPU CUDA validation for the DepthFiLM recurrent experiment.

Result:

- `1x NVIDIA GeForce RTX 3090` was enough to validate the CUDA training path.
- The run trained for 2 steps, validated, exported, quantized, and completed the int8+zlib roundtrip validation.
- No H100 was required to prove the basic idea.

Key metrics:

- `val_loss: 7.8441`
- `val_bpb: 4.6998`
- `peak memory allocated: 419 MiB`
- `Serialized model int8+zlib: 2523469 bytes`
- `final_int8_zlib_roundtrip val_loss: 8.1560`
- `final_int8_zlib_roundtrip val_bpb: 4.8866`

Important lessons:

- Older PyTorch builds may not support `scaled_dot_product_attention(..., enable_gqa=...)`; the experiment now has a compatibility fallback.
- `torch.compile` is too expensive for cheap smoke validation on this tier; `ENABLE_TORCH_COMPILE=0` is the correct smoke-run setting.
