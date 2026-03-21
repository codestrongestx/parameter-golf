# H100 Shared Blocks Only

This directory contains a curated single-GPU ablation result for the DepthFiLM recurrent experiment.

Config:

- `1x NVIDIA H100 80GB HBM3`
- `RUN_ID=depthfilm_h100_shared_only_u4_d0`
- `NUM_LAYERS=12 NUM_UNIQUE_LAYERS=4 DEPTH_CONDITION_DIM=0`
- `MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3`
- `TRAIN_SEQ_LEN=1024 TRAIN_BATCH_TOKENS=65536 ENABLE_TORCH_COMPILE=0`

Plan metrics:

- `post-roundtrip val_bpb: 1.4863`
- `artifact bytes: 8663549`
- `unique parameter count: 9972768`
- `ms/step: 343.88`

Additional metrics:

- `pre-roundtrip val_bpb: 1.4861`
- `pre-roundtrip val_loss: 2.5093`
- `post-roundtrip val_loss: 2.5096`
- `peak memory allocated: 3886 MiB`
- `total submission size int8+zlib: 8732617`
