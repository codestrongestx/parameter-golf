# 1xH100 Pivot Sweep: 10L FP16 Embed Tuning

Three single-H100 directional runs were measured against the merged record `2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit`.

| Variant | Post-roundtrip val_bpb | Artifact bytes | Total submission bytes | ms/step | Eval time |
| --- | ---: | ---: | ---: | ---: | ---: |
| `WARMDOWN_ITERS=20000`, `EVAL_SEQ_LEN=1024` | `1.28768222` | `13385911` | `13441901` | `100.45` | `272362 ms` |
| `WARMDOWN_ITERS=2500`, `EVAL_SEQ_LEN=1024` | `1.29406563` | `15051498` | `15107488` | `107.35` | `279739 ms` |
| `WARMDOWN_ITERS=20000`, `EVAL_SEQ_LEN=2048` | `1.30287290` | `13355586` | `13411576` | `102.25` | `532571 ms` |

Current recommendation:

- keep `WARMDOWN_ITERS=20000`
- keep `EVAL_SEQ_LEN=1024`
- stop single-H100 tuning here for this narrow branch
- use this as the current best baseline for PR packaging or an `8x H100` escalation

Why stop here:

- the warmdown control was clearly worse
- the `2048` eval sweep was clearly worse and nearly doubled roundtrip eval cost
- all three runs stayed under the `16MB` cap, so the current bottleneck is score, not size
