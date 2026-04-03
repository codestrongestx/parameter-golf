#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TRAIN_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iterations>\d+) "
    r"train_loss:(?P<train_loss>[-+]?\d+(?:\.\d+)?) "
    r"train_time:(?P<train_time_ms>\d+)ms "
    r"step_avg:(?P<step_avg_ms>[-+]?\d+(?:\.\d+)?)ms"
)
VAL_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iterations>\d+) "
    r"val_loss:(?P<val_loss>[-+]?\d+(?:\.\d+)?) "
    r"val_bpb:(?P<val_bpb>[-+]?\d+(?:\.\d+)?) "
    r"train_time:(?P<train_time_ms>\d+)ms "
    r"step_avg:(?P<step_avg_ms>[-+]?\d+(?:\.\d+)?)ms"
)
MODEL_RE = re.compile(
    r"model_params:(?P<model_params>\d+) "
    r"virtual_layers:(?P<num_layers>\d+) "
    r"unique_layers:(?P<num_unique_layers>\d+) "
    r"depth_condition_dim:(?P<depth_condition_dim>\d+) "
    r"mlp_mult:(?P<mlp_mult>\d+)"
)
PEAK_MEMORY_RE = re.compile(
    r"peak memory allocated: (?P<allocated>\d+) MiB reserved: (?P<reserved>\d+) MiB"
)
SERIALIZED_MODEL_RE = re.compile(r"Serialized model: (?P<value>\d+) bytes")
CODE_SIZE_RE = re.compile(r"Code size: (?P<value>\d+) bytes")
SUBMISSION_SIZE_RE = re.compile(r"Total submission size: (?P<value>\d+) bytes")
INT8_RE = re.compile(
    r"Serialized model int8\+zlib: (?P<artifact>\d+) bytes "
    r"\(payload:(?P<payload>\d+) raw_torch:(?P<raw_torch>\d+) payload_ratio:(?P<ratio>[-+]?\d+(?:\.\d+)?)x\)"
)
INT8_SUBMISSION_SIZE_RE = re.compile(r"Total submission size int8\+zlib: (?P<value>\d+) bytes")
ROUNDTRIP_RE = re.compile(
    r"final_int8_zlib_roundtrip val_loss:(?P<val_loss>[-+]?\d+(?:\.\d+)?) "
    r"val_bpb:(?P<val_bpb>[-+]?\d+(?:\.\d+)?) "
    r"eval_time:(?P<eval_time_ms>\d+)ms"
)


def read_command_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.isupper():
            env[key] = value
    return env


def parse_log(path: Path) -> dict[str, object]:
    train_steps: list[dict[str, object]] = []
    final_val: dict[str, object] | None = None
    final_roundtrip: dict[str, object] | None = None
    model_info: dict[str, object] | None = None
    peak_memory: dict[str, object] | None = None
    serialized_model_bytes: int | None = None
    code_size_bytes: int | None = None
    submission_size_bytes: int | None = None
    int8_info: dict[str, object] | None = None
    int8_submission_size_bytes: int | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = TRAIN_RE.search(line)
        if match:
            train_steps.append(
                {
                    "step": int(match.group("step")),
                    "iterations": int(match.group("iterations")),
                    "train_loss": float(match.group("train_loss")),
                    "train_time_ms": int(match.group("train_time_ms")),
                    "step_avg_ms": float(match.group("step_avg_ms")),
                }
            )
            continue

        match = VAL_RE.search(line)
        if match:
            final_val = {
                "step": int(match.group("step")),
                "iterations": int(match.group("iterations")),
                "val_loss": float(match.group("val_loss")),
                "val_bpb": float(match.group("val_bpb")),
                "train_time_ms": int(match.group("train_time_ms")),
                "step_avg_ms": float(match.group("step_avg_ms")),
            }
            continue

        match = MODEL_RE.search(line)
        if match:
            model_info = {
                "model_params": int(match.group("model_params")),
                "num_layers": int(match.group("num_layers")),
                "num_unique_layers": int(match.group("num_unique_layers")),
                "depth_condition_dim": int(match.group("depth_condition_dim")),
                "mlp_mult": int(match.group("mlp_mult")),
            }
            continue

        match = PEAK_MEMORY_RE.search(line)
        if match:
            peak_memory = {
                "peak_memory_allocated_mib": int(match.group("allocated")),
                "peak_memory_reserved_mib": int(match.group("reserved")),
            }
            continue

        match = SERIALIZED_MODEL_RE.search(line)
        if match:
            serialized_model_bytes = int(match.group("value"))
            continue

        match = CODE_SIZE_RE.search(line)
        if match:
            code_size_bytes = int(match.group("value"))
            continue

        match = SUBMISSION_SIZE_RE.search(line)
        if match:
            submission_size_bytes = int(match.group("value"))
            continue

        match = INT8_RE.search(line)
        if match:
            int8_info = {
                "serialized_model_int8_zlib_bytes": int(match.group("artifact")),
                "int8_payload_bytes": int(match.group("payload")),
                "int8_raw_torch_bytes": int(match.group("raw_torch")),
                "payload_ratio": float(match.group("ratio")),
            }
            continue

        match = INT8_SUBMISSION_SIZE_RE.search(line)
        if match:
            int8_submission_size_bytes = int(match.group("value"))
            continue

        match = ROUNDTRIP_RE.search(line)
        if match:
            final_roundtrip = {
                "final_int8_zlib_roundtrip_val_loss": float(match.group("val_loss")),
                "final_int8_zlib_roundtrip_val_bpb": float(match.group("val_bpb")),
                "final_int8_zlib_roundtrip_eval_time_ms": int(match.group("eval_time_ms")),
            }
            continue

    metrics: dict[str, object] = {
        "train_steps": train_steps,
    }
    if model_info is not None:
        metrics.update(model_info)
        metrics["unique_parameter_count"] = model_info["model_params"]
    if final_val is not None:
        metrics.update(final_val)
    if peak_memory is not None:
        metrics.update(peak_memory)
    if serialized_model_bytes is not None:
        metrics["serialized_model_bytes"] = serialized_model_bytes
    if code_size_bytes is not None:
        metrics["code_size_bytes"] = code_size_bytes
    if submission_size_bytes is not None:
        metrics["submission_size_bytes"] = submission_size_bytes
    if int8_info is not None:
        metrics.update(int8_info)
        metrics["artifact_bytes"] = int8_info["serialized_model_int8_zlib_bytes"]
    if int8_submission_size_bytes is not None:
        metrics["total_submission_size_int8_zlib_bytes"] = int8_submission_size_bytes
    if final_roundtrip is not None:
        metrics.update(final_roundtrip)
    return metrics


def build_metrics(
    *,
    date: str,
    gpu: str,
    env: dict[str, str],
    parsed: dict[str, object],
) -> dict[str, object]:
    metrics: dict[str, object] = {
        "date": date,
        "gpu": gpu,
        "run_id": env.get("RUN_ID"),
        "compile_enabled": env.get("ENABLE_TORCH_COMPILE", "1") == "1",
        "iterations": int(env["ITERATIONS"]) if "ITERATIONS" in env else parsed.get("iterations"),
        "max_wallclock_seconds": float(env["MAX_WALLCLOCK_SECONDS"]) if "MAX_WALLCLOCK_SECONDS" in env else None,
        "train_seq_len": int(env["TRAIN_SEQ_LEN"]) if "TRAIN_SEQ_LEN" in env else None,
        "train_batch_tokens": int(env["TRAIN_BATCH_TOKENS"]) if "TRAIN_BATCH_TOKENS" in env else None,
        "val_batch_size": int(env["VAL_BATCH_SIZE"]) if "VAL_BATCH_SIZE" in env else None,
        "val_max_tokens": int(env["VAL_MAX_TOKENS"]) if "VAL_MAX_TOKENS" in env else 0,
        "eval_stride": int(env["EVAL_STRIDE"]) if "EVAL_STRIDE" in env else 0,
        "num_layers": int(env["NUM_LAYERS"]) if "NUM_LAYERS" in env else parsed.get("num_layers"),
        "num_unique_layers": int(env["NUM_UNIQUE_LAYERS"]) if "NUM_UNIQUE_LAYERS" in env else parsed.get("num_unique_layers"),
        "depth_condition_dim": int(env["DEPTH_CONDITION_DIM"]) if "DEPTH_CONDITION_DIM" in env else parsed.get("depth_condition_dim"),
        "depth_condition_scale": float(env["DEPTH_CONDITION_SCALE"]) if "DEPTH_CONDITION_SCALE" in env else None,
        "model_dim": int(env["MODEL_DIM"]) if "MODEL_DIM" in env else None,
        "num_heads": int(env["NUM_HEADS"]) if "NUM_HEADS" in env else None,
        "num_kv_heads": int(env["NUM_KV_HEADS"]) if "NUM_KV_HEADS" in env else None,
        "mlp_mult": int(env["MLP_MULT"]) if "MLP_MULT" in env else parsed.get("mlp_mult"),
        "int4_blocks": env.get("INT4_BLOCKS", ""),
        "int4_step": int(env["INT4_STEP"]) if "INT4_STEP" in env else None,
        "warmup_steps": int(env["WARMUP_STEPS"]) if "WARMUP_STEPS" in env else 20,
    }

    train_steps = parsed.get("train_steps", [])
    if train_steps:
        last_train = train_steps[-1]
        metrics["final_train_step"] = last_train["step"]
        metrics["final_train_loss"] = last_train["train_loss"]
        metrics["train_step_avg_ms"] = last_train["step_avg_ms"]
        metrics["train_time_ms"] = last_train["train_time_ms"]
    if "step_avg_ms" in parsed:
        metrics["train_step_avg_ms"] = parsed["step_avg_ms"]
    for key, value in parsed.items():
        if key == "train_steps":
            continue
        metrics[key] = value
    return {key: value for key, value in metrics.items() if value is not None}


def write_summary(path: Path, title: str, metrics: dict[str, object]) -> None:
    lines = [
        f"# {title}",
        "",
        "This directory contains a curated single-GPU ablation result for the DepthFiLM recurrent experiment.",
        "",
        "Config:",
        "",
        f"- `1x {metrics['gpu']}`",
        f"- `RUN_ID={metrics['run_id']}`",
        f"- `NUM_LAYERS={metrics['num_layers']} NUM_UNIQUE_LAYERS={metrics['num_unique_layers']} DEPTH_CONDITION_DIM={metrics['depth_condition_dim']}`",
        f"- `MODEL_DIM={metrics['model_dim']} NUM_HEADS={metrics['num_heads']} NUM_KV_HEADS={metrics['num_kv_heads']} MLP_MULT={metrics['mlp_mult']}`",
        f"- `TRAIN_SEQ_LEN={metrics['train_seq_len']} TRAIN_BATCH_TOKENS={metrics['train_batch_tokens']} ENABLE_TORCH_COMPILE={int(bool(metrics['compile_enabled']))}`",
    ]
    if metrics.get("eval_stride", 0):
        lines.append(f"- `EVAL_STRIDE={metrics['eval_stride']}`")
    if metrics.get("int4_blocks"):
        lines.append(f"- `INT4_BLOCKS={metrics['int4_blocks']} INT4_STEP={metrics['int4_step']}`")

    lines.extend(
        [
            "",
            "Plan metrics:",
            "",
            f"- `post-roundtrip val_bpb: {metrics['final_int8_zlib_roundtrip_val_bpb']:.4f}`",
            f"- `artifact bytes: {metrics['artifact_bytes']}`",
            f"- `unique parameter count: {metrics['unique_parameter_count']}`",
            f"- `ms/step: {metrics['train_step_avg_ms']:.2f}`",
            "",
            "Additional metrics:",
            "",
            f"- `pre-roundtrip val_bpb: {metrics['val_bpb']:.4f}`",
            f"- `pre-roundtrip val_loss: {metrics['val_loss']:.4f}`",
            f"- `post-roundtrip val_loss: {metrics['final_int8_zlib_roundtrip_val_loss']:.4f}`",
            f"- `peak memory allocated: {metrics['peak_memory_allocated_mib']} MiB`",
            f"- `total submission size int8+zlib: {metrics['total_submission_size_int8_zlib_bytes']}`",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--command-file", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    env = read_command_env(args.command_file)
    parsed = parse_log(args.train_log)
    metrics = build_metrics(date=args.date, gpu=args.gpu, env=env, parsed=parsed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_summary(args.output_dir / "summary.md", args.title, metrics)


if __name__ == "__main__":
    main()
