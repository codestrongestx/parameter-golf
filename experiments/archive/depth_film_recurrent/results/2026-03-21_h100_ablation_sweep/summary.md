# H100 Ablation Sweep

Single-GPU `1x NVIDIA H100 80GB HBM3` comparison for the planned DepthFiLM recurrent ablations.

| Variant | Post-roundtrip val_bpb | Artifact bytes | Unique params | ms/step |
| --- | ---: | ---: | ---: | ---: |
| Shared blocks only (`12/4/0`) | `1.4863` | `8663549` | `9972768` | `343.88` |
| Shared blocks + DepthFiLM (`12/4/32`) | `1.4886` | `9400467` | `10243488` | `391.06` |
| Frontier DepthFiLM (`12/4/32`, `EVAL_STRIDE=64`, `INT4_BLOCKS=0,1,2,3`) | `1.4596` | `7496799` | `10243488` | `391.01` |

Read the per-run directories for full logs and metrics:

- `results/2026-03-21_h100_shared_blocks_only_u4_d0/`
- `results/2026-03-21_h100_shared_blocks_depthfilm32_u4/`
- `results/2026-03-21_h100_frontier_depthfilm32_u4_sw64_int4/`

Takeaway:

- Plain DepthFiLM was worse than shared blocks only on both quality and efficiency at this setting.
- The stronger frontier-style package was the best measured configuration in this experiment family on `1x H100`.
- Sliding-window roundtrip eval materially improved the frontier variant's post-roundtrip `val_bpb`, but it also made post-export evaluation much slower than the plain full-context runs.
