from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HEADER_WORDS = 256
MAGIC = 20240520
VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a smaller validation shard by truncating tokens.")
    parser.add_argument("input_shard", type=Path, help="Source fineweb_val_*.bin shard")
    parser.add_argument("output_shard", type=Path, help="Destination shard path")
    parser.add_argument("--tokens", type=int, required=True, help="Number of tokens to keep")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    header = np.fromfile(args.input_shard, dtype="<i4", count=HEADER_WORDS)
    if header.size != HEADER_WORDS or int(header[0]) != MAGIC or int(header[1]) != VERSION:
        raise SystemExit(f"Unexpected shard header: {args.input_shard}")
    total_tokens = int(header[2])
    keep_tokens = min(args.tokens, total_tokens)
    if keep_tokens <= 0:
        raise SystemExit("--tokens must be positive")

    token_offset = HEADER_WORDS * np.dtype("<i4").itemsize
    tokens = np.fromfile(args.input_shard, dtype="<u2", count=keep_tokens, offset=token_offset)
    if tokens.size != keep_tokens:
        raise SystemExit(f"Short read from {args.input_shard}")

    out_header = header.copy()
    out_header[2] = keep_tokens
    args.output_shard.parent.mkdir(parents=True, exist_ok=True)
    with args.output_shard.open("wb") as handle:
        out_header.astype("<i4", copy=False).tofile(handle)
        tokens.astype("<u2", copy=False).tofile(handle)

    print(args.output_shard)


if __name__ == "__main__":
    main()
