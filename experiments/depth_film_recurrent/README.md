# DepthFiLM Recurrent Stack

This experiment implements the core idea from the plan:

- reuse a small set of unique transformer blocks across a deeper virtual stack
- add a tiny learned depth-conditioning path so each virtual pass can specialize
- keep the rest of the training/export flow close to the repo baseline

Files:

- `train_gpt_mlx.py`: local Apple Silicon iteration path
- `train_gpt.py`: CUDA path for 1xH100 and beyond

## Architecture

Default model settings:

- `NUM_LAYERS=12`
- `NUM_UNIQUE_LAYERS=4`
- `DEPTH_CONDITION_DIM=32`
- `DEPTH_CONDITION_SCALE=0.10`
- `MLP_MULT=3`

`NUM_UNIQUE_LAYERS` controls weight sharing. `DEPTH_CONDITION_DIM=0` disables the DepthFiLM path and gives plain shared blocks.

## Planned Ablations

1. No sharing:

```bash
NUM_LAYERS=12 NUM_UNIQUE_LAYERS=12 DEPTH_CONDITION_DIM=0
```

2. Shared blocks only:

```bash
NUM_LAYERS=12 NUM_UNIQUE_LAYERS=4 DEPTH_CONDITION_DIM=0
```

3. Shared blocks + DepthFiLM:

```bash
NUM_LAYERS=12 NUM_UNIQUE_LAYERS=4 DEPTH_CONDITION_DIM=32
```

## Local Mac Smoke Run

Install the repo’s local dependencies, then download the smallest cached dataset slice:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```

Smoke command:

```bash
source .venv/bin/activate
RUN_ID=depthfilm_mlx_smoke \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
ITERATIONS=30 \
TRAIN_SEQ_LEN=256 \
TRAIN_BATCH_TOKENS=4096 \
GRAD_ACCUM_STEPS=2 \
MLX_MAX_MICROBATCH_TOKENS=2048 \
VAL_BATCH_SIZE=4096 \
VAL_MAX_TOKENS=1048576 \
VAL_LOSS_EVERY=0 \
NUM_LAYERS=8 \
NUM_UNIQUE_LAYERS=4 \
DEPTH_CONDITION_DIM=16 \
MODEL_DIM=256 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=3 \
python3 experiments/depth_film_recurrent/train_gpt_mlx.py
```

## 1xH100 Run

This keeps the same architecture family but turns on the CUDA path, optional sliding-window eval, and optional reduced-precision export for the shared blocks.

```bash
RUN_ID=depthfilm_h100 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=12 \
NUM_UNIQUE_LAYERS=4 \
DEPTH_CONDITION_DIM=32 \
DEPTH_CONDITION_SCALE=0.10 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=3 \
TRAIN_SEQ_LEN=1024 \
TRAIN_BATCH_TOKENS=65536 \
VAL_LOSS_EVERY=200 \
EVAL_STRIDE=64 \
INT4_BLOCKS=0,1,2,3 \
INT4_STEP=4 \
TTT_ENABLE=0 \
torchrun --standalone --nproc_per_node=1 experiments/depth_film_recurrent/train_gpt.py
```

Notes:

- `INT4_STEP=4` is the int6-style export knob from the plan. It rounds selected shared-block int8 tensors to multiples of 4 after quantization.
- `EVAL_STRIDE=64` enables sliding-window evaluation at the training sequence length.
- Start with `TRAIN_BATCH_TOKENS=65536` on 1xH100 and scale up after checking memory/step time.
- Leave `VAL_MAX_TOKENS=0` for real runs. Set it only on local smoke tests when you want a bounded validation slice.
