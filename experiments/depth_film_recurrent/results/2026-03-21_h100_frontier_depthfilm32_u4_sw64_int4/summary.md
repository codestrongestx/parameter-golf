# H100 Frontier DepthFiLM + Sliding Eval + Int4 Export

This directory contains a curated single-GPU ablation result for the DepthFiLM recurrent experiment.

Config:

- `1x NVIDIA H100 80GB HBM3`
- `RUN_ID=depthfilm_h100_frontier_u4_depthfilm32_sw64_int4`
- `NUM_LAYERS=12 NUM_UNIQUE_LAYERS=4 DEPTH_CONDITION_DIM=32`
- `MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3`
- `TRAIN_SEQ_LEN=1024 TRAIN_BATCH_TOKENS=65536 ENABLE_TORCH_COMPILE=0`
- `EVAL_STRIDE=64`
- `INT4_BLOCKS=0,1,2,3 INT4_STEP=4`

Plan metrics:

- `post-roundtrip val_bpb: 1.4596`
- `artifact bytes: 7496799`
- `unique parameter count: 10243488`
- `ms/step: 391.01`

Additional metrics:

- `pre-roundtrip val_bpb: 1.4889`
- `pre-roundtrip val_loss: 2.5140`
- `post-roundtrip val_loss: 2.4646`
- `peak memory allocated: 4272 MiB`
- `total submission size int8+zlib: 7565867`
