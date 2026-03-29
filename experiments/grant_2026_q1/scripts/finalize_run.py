from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

TRAIN_STEP_RE = re.compile(
    r"^step:(?P<step>\d+)/(?P<iterations>\d+) train_loss:(?P<train_loss>[-+]?\d+(?:\.\d+)?) "
    r"train_time:(?P<train_time_ms>\d+)ms step_avg:(?P<step_avg_ms>[-+]?\d+(?:\.\d+)?)ms$"
)
VAL_STEP_RE = re.compile(
    r"^step:(?P<step>\d+)/(?P<iterations>\d+) val_loss:(?P<val_loss>[-+]?\d+(?:\.\d+)?) "
    r"val_bpb:(?P<val_bpb>[-+]?\d+(?:\.\d+)?) train_time:(?P<train_time_ms>\d+)ms "
    r"step_avg:(?P<step_avg_ms>[-+]?\d+(?:\.\d+)?)ms$"
)
DIAG_RE = re.compile(
    r"^DIAGNOSTIC (?P<label>post_ema|post_average) val_loss:(?P<val_loss>[-+]?\d+(?:\.\d+)?) "
    r"val_bpb:(?P<val_bpb>[-+]?\d+(?:\.\d+)?) eval_time:(?P<eval_time_ms>\d+)ms$"
)
GENERIC_FINAL_RE = re.compile(
    r"^(?P<label>[A-Za-z0-9_]+) val_loss:(?P<val_loss>[-+]?\d+(?:\.\d+)?) "
    r"val_bpb:(?P<val_bpb>[-+]?\d+(?:\.\d+)?)"
    r"(?: stride:(?P<stride>\d+))?(?: eval_time:(?P<eval_time_ms>\d+)ms)?$"
)
DATASET_RE = re.compile(
    r"^train_loader:dataset:(?P<dataset>\S+)(?: train_shards:(?P<train_shards>\d+))?$"
)
VAL_LOADER_RE = re.compile(r"^val_loader:shards pattern=(?P<pattern>\S+) tokens:(?P<tokens>\d+)$")
VAL_MAX_SEQS_RE = re.compile(r"^val_loader:max_seqs:(?P<max_seqs>\d+)$")
MODEL_PARAMS_RE = re.compile(r"^model_params:(?P<model_params>\d+)$")
WORLD_RE = re.compile(r"^world_size:(?P<world_size>\d+) grad_accum_steps:(?P<grad_accum_steps>\d+)$")
ATTN_RE = re.compile(
    r"^attention_mode:(?P<attention_mode>\S+) num_heads:(?P<num_heads>\d+) "
    r"num_kv_heads:(?P<num_kv_heads>\d+)$"
)
TRAIN_CFG_RE = re.compile(
    r"^train_batch_tokens:(?P<train_batch_tokens>\d+) train_seq_len:(?P<train_seq_len>\d+) "
    r"iterations:(?P<iterations>\d+) warmup_steps:(?P<warmup_steps>\d+) "
    r"max_wallclock_seconds:(?P<max_wallclock_seconds>[-+]?\d+(?:\.\d+)?)$"
)
SEED_RE = re.compile(r"^seed:(?P<seed>\d+)$")
STOP_RE = re.compile(
    r"^stopping_early: (?P<stop_reason>\S+) train_time:(?P<train_time_ms>\d+)ms "
    r"step:(?P<step>\d+)/(?P<iterations>\d+)$"
)
PEAK_MEM_RE = re.compile(
    r"^peak memory allocated: (?P<allocated>\d+) MiB reserved: (?P<reserved>\d+) MiB$"
)
SERIALIZED_MODEL_RE = re.compile(r"^Serialized model: (?P<bytes>\d+) bytes$")
SERIALIZED_MODEL_RESEARCH_EXPORT_RE = re.compile(r"^Serialized model research_export: (?P<bytes>\d+) bytes$")
CODE_SIZE_RE = re.compile(r"^Code size: (?P<bytes>\d+) bytes$")
TOTAL_SIZE_RE = re.compile(r"^Total submission size: (?P<bytes>\d+) bytes$")
TOTAL_SIZE_RESEARCH_EXPORT_RE = re.compile(r"^Total submission size research_export: (?P<bytes>\d+) bytes$")
INT8_ZLIB_RE = re.compile(
    r"^Serialized model int8\+zlib: (?P<quant_bytes>\d+) bytes "
    r"\(payload:(?P<payload_bytes>\d+) raw_torch:(?P<raw_bytes>\d+) "
    r"payload_ratio:(?P<payload_ratio>[-+]?\d+(?:\.\d+)?)x\)$"
)
INT8_ZLIB_TOTAL_RE = re.compile(r"^Total submission size int8\+zlib: (?P<bytes>\d+) bytes$")
INT6_LZMA_RE = re.compile(r"^Serialized model int6\+lzma: (?P<bytes>\d+) bytes$")
INT6_LZMA_TOTAL_RE = re.compile(r"^Total submission size int6\+lzma: (?P<bytes>\d+) bytes$")
PHASE_TIMINGS_RE = re.compile(r"^phase_timings:(?P<payload>\{.*\})$")

PRIMARY_PRIORITY = [
    "final_ngram_exact",
    "final_ngram",
    "final_research_export_exact",
    "final_research_export_roundtrip",
    "legal_ttt_exact",
    "final_int6_sliding_window_s64_exact",
    "final_int6_sliding_window_exact",
    "final_int8_zlib_roundtrip_exact",
    "final_int6_roundtrip_exact",
    "final_int6_sliding_window_s64",
    "final_int6_sliding_window",
    "final_int8_zlib_roundtrip",
    "final_int6_roundtrip",
]

SHARED_COMPARISON_PRIORITY = [
    "final_research_export_sliding_exact",
    "final_int6_sliding_window_s64_exact",
    "final_int6_sliding_window_exact",
    "final_research_export_exact",
    "final_int8_zlib_roundtrip_exact",
    "final_int6_roundtrip_exact",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse a Grant 2026 Q1 run log into curated metrics and summary.")
    parser.add_argument("run_dir", type=Path, help="Path to the curated run folder")
    parser.add_argument("--train-log", default="train.log", help="Train log filename inside run_dir")
    parser.add_argument("--eval-log", default="eval.log", help="Eval log filename inside run_dir")
    parser.add_argument(
        "--status",
        default="completed",
        choices=("completed", "aborted", "failed", "planned"),
        help="Final run status to record in metrics.json and summary.md",
    )
    parser.add_argument("--conclusion", default="", help="Short human conclusion for summary.md")
    parser.add_argument("--next-change", default="", help="What changed next for run_observations.md")
    parser.add_argument(
        "--append-observation",
        action="store_true",
        help="Append a block to notes/run_observations.md",
    )
    return parser.parse_args()


def pick_metric(
    final_metrics: dict[str, dict[str, object]],
    priority: list[str],
) -> tuple[str, dict[str, object]] | None:
    for label in priority:
        metric = final_metrics.get(label)
        if metric is not None:
            return label, metric
    return None


def parse_log(path: Path) -> dict[str, object]:
    out: dict[str, object] = {}
    if not path.exists():
        return out
    final_metrics: dict[str, dict[str, object]] = {}
    last_train: dict[str, object] | None = None
    last_val: dict[str, object] | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        train_match = TRAIN_STEP_RE.match(line)
        if train_match:
            last_train = {
                "step": int(train_match.group("step")),
                "iterations": int(train_match.group("iterations")),
                "train_loss": float(train_match.group("train_loss")),
                "train_time_ms": int(train_match.group("train_time_ms")),
                "step_avg_ms": float(train_match.group("step_avg_ms")),
            }
            continue

        val_match = VAL_STEP_RE.match(line)
        if val_match:
            last_val = {
                "step": int(val_match.group("step")),
                "iterations": int(val_match.group("iterations")),
                "val_loss": float(val_match.group("val_loss")),
                "val_bpb": float(val_match.group("val_bpb")),
                "train_time_ms": int(val_match.group("train_time_ms")),
                "step_avg_ms": float(val_match.group("step_avg_ms")),
            }
            continue

        diag_match = DIAG_RE.match(line)
        if diag_match:
            out["post_ema_val_loss"] = float(diag_match.group("val_loss"))
            out["post_ema_val_bpb"] = float(diag_match.group("val_bpb"))
            out["post_ema_eval_time_ms"] = int(diag_match.group("eval_time_ms"))
            out["post_ema_label"] = diag_match.group("label")
            continue

        final_match = GENERIC_FINAL_RE.match(line)
        if final_match:
            label = final_match.group("label")
            final_metrics[label] = {
                "val_loss": float(final_match.group("val_loss")),
                "val_bpb": float(final_match.group("val_bpb")),
            }
            if final_match.group("eval_time_ms"):
                final_metrics[label]["eval_time_ms"] = int(final_match.group("eval_time_ms"))
            if final_match.group("stride"):
                final_metrics[label]["stride"] = int(final_match.group("stride"))
            continue

        dataset_match = DATASET_RE.match(line)
        if dataset_match:
            out["dataset"] = dataset_match.group("dataset")
            if dataset_match.group("train_shards") is not None:
                out["train_shards"] = int(dataset_match.group("train_shards"))
            continue

        val_loader_match = VAL_LOADER_RE.match(line)
        if val_loader_match:
            out.update(
                val_pattern=val_loader_match.group("pattern"),
                val_tokens=int(val_loader_match.group("tokens")),
            )
            continue

        val_max_seqs_match = VAL_MAX_SEQS_RE.match(line)
        if val_max_seqs_match:
            out["val_max_seqs"] = int(val_max_seqs_match.group("max_seqs"))
            continue

        model_params_match = MODEL_PARAMS_RE.match(line)
        if model_params_match:
            out["unique_parameter_count"] = int(model_params_match.group("model_params"))
            continue

        world_match = WORLD_RE.match(line)
        if world_match:
            out.update(
                world_size=int(world_match.group("world_size")),
                grad_accum_steps=int(world_match.group("grad_accum_steps")),
            )
            continue

        attn_match = ATTN_RE.match(line)
        if attn_match:
            out.update(
                attention_mode=attn_match.group("attention_mode"),
                num_heads=int(attn_match.group("num_heads")),
                num_kv_heads=int(attn_match.group("num_kv_heads")),
            )
            continue

        train_cfg_match = TRAIN_CFG_RE.match(line)
        if train_cfg_match:
            out.update(
                train_batch_tokens=int(train_cfg_match.group("train_batch_tokens")),
                train_seq_len=int(train_cfg_match.group("train_seq_len")),
                iterations_target=int(train_cfg_match.group("iterations")),
                warmup_steps=int(train_cfg_match.group("warmup_steps")),
                max_wallclock_seconds=float(train_cfg_match.group("max_wallclock_seconds")),
            )
            continue

        seed_match = SEED_RE.match(line)
        if seed_match:
            out["seed"] = int(seed_match.group("seed"))
            continue

        stop_match = STOP_RE.match(line)
        if stop_match:
            out.update(
                stop_reason=stop_match.group("stop_reason"),
                stopped_at_step=int(stop_match.group("step")),
                stopped_train_time_ms=int(stop_match.group("train_time_ms")),
            )
            continue

        peak_mem_match = PEAK_MEM_RE.match(line)
        if peak_mem_match:
            out.update(
                peak_memory_allocated_mib=int(peak_mem_match.group("allocated")),
                peak_memory_reserved_mib=int(peak_mem_match.group("reserved")),
            )
            continue

        serialized_model_match = SERIALIZED_MODEL_RE.match(line)
        if serialized_model_match:
            artifact_bytes = int(serialized_model_match.group("bytes"))
            out["serialized_model_bytes"] = artifact_bytes
            out["artifact_bytes"] = artifact_bytes
            continue

        serialized_model_research_export_match = SERIALIZED_MODEL_RESEARCH_EXPORT_RE.match(line)
        if serialized_model_research_export_match:
            artifact_bytes = int(serialized_model_research_export_match.group("bytes"))
            out["serialized_model_bytes"] = artifact_bytes
            out["artifact_bytes"] = artifact_bytes
            continue

        code_size_match = CODE_SIZE_RE.match(line)
        if code_size_match:
            out["code_bytes"] = int(code_size_match.group("bytes"))
            continue

        total_size_match = TOTAL_SIZE_RE.match(line)
        if total_size_match:
            out["total_submission_size_bytes"] = int(total_size_match.group("bytes"))
            continue

        total_size_research_export_match = TOTAL_SIZE_RESEARCH_EXPORT_RE.match(line)
        if total_size_research_export_match:
            out["total_submission_size_bytes"] = int(total_size_research_export_match.group("bytes"))
            continue

        int8_match = INT8_ZLIB_RE.match(line)
        if int8_match:
            artifact_bytes = int(int8_match.group("quant_bytes"))
            out.update(
                artifact_bytes=artifact_bytes,
                serialized_model_int8_zlib_bytes=artifact_bytes,
                int8_payload_bytes=int(int8_match.group("payload_bytes")),
                int8_raw_torch_bytes=int(int8_match.group("raw_bytes")),
                int8_payload_ratio=float(int8_match.group("payload_ratio")),
            )
            continue

        int8_total_match = INT8_ZLIB_TOTAL_RE.match(line)
        if int8_total_match:
            out["total_submission_size_int8_zlib_bytes"] = int(int8_total_match.group("bytes"))
            continue

        int6_match = INT6_LZMA_RE.match(line)
        if int6_match:
            artifact_bytes = int(int6_match.group("bytes"))
            out["artifact_bytes"] = artifact_bytes
            out["serialized_model_int6_lzma_bytes"] = artifact_bytes
            continue

        int6_total_match = INT6_LZMA_TOTAL_RE.match(line)
        if int6_total_match:
            out["total_submission_size_int6_lzma_bytes"] = int(int6_total_match.group("bytes"))
            continue

        phase_timings_match = PHASE_TIMINGS_RE.match(line)
        if phase_timings_match:
            out["phase_timings"] = json.loads(phase_timings_match.group("payload"))
            continue

    if last_train:
        out.update(
            last_train_step=last_train["step"],
            last_train_loss=last_train["train_loss"],
            last_train_time_ms=last_train["train_time_ms"],
            ms_per_step=last_train["step_avg_ms"],
        )
    if last_val:
        out.update(
            pre_roundtrip_step=last_val["step"],
            pre_roundtrip_val_loss=last_val["val_loss"],
            pre_roundtrip_val_bpb=last_val["val_bpb"],
            pre_roundtrip_train_time_ms=last_val["train_time_ms"],
            ms_per_step=last_val["step_avg_ms"],
        )
    if final_metrics:
        out["final_metrics"] = final_metrics
        primary_metric = pick_metric(final_metrics, PRIMARY_PRIORITY)
        if primary_metric is not None:
            label, metric = primary_metric
            out["post_roundtrip_metric_label"] = label
            out["post_roundtrip_val_loss"] = metric["val_loss"]
            out["post_roundtrip_val_bpb"] = metric["val_bpb"]
            out["lane_primary_metric_label"] = label
            out["lane_primary_val_loss"] = metric["val_loss"]
            out["lane_primary_val_bpb"] = metric["val_bpb"]
            if "eval_time_ms" in metric:
                out["post_roundtrip_eval_time_ms"] = metric["eval_time_ms"]
                out["lane_primary_eval_time_ms"] = metric["eval_time_ms"]

        shared_metric = pick_metric(final_metrics, SHARED_COMPARISON_PRIORITY)
        if shared_metric is not None:
            label, metric = shared_metric
            out["shared_comparison_metric_label"] = label
            out["shared_comparison_val_loss"] = metric["val_loss"]
            out["shared_comparison_val_bpb"] = metric["val_bpb"]
            if "eval_time_ms" in metric:
                out["shared_comparison_eval_time_ms"] = metric["eval_time_ms"]
    return out


def write_metrics(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_summary(meta: dict[str, object]) -> str:
    question = str(meta.get("question", "")).strip() or "None."
    notes = str(meta.get("notes", "")).strip() or "None."
    conclusion = str(meta.get("conclusion", "")).strip() or "Pending human conclusion."
    next_change = str(meta.get("next_change", "")).strip() or "Pending."
    pre_bpb = str(meta.get("pre_roundtrip_val_bpb", "None."))
    primary_label = str(meta.get("lane_primary_metric_label", meta.get("post_roundtrip_metric_label", "None.")))
    primary_bpb = str(meta.get("lane_primary_val_bpb", meta.get("post_roundtrip_val_bpb", "None.")))
    shared_label = str(meta.get("shared_comparison_metric_label", "None."))
    shared_bpb = str(meta.get("shared_comparison_val_bpb", "None."))
    artifact = str(
        meta.get("serialized_model_int8_zlib_bytes")
        or meta.get("serialized_model_int6_lzma_bytes")
        or meta.get("serialized_model_bytes")
        or "None."
    )
    params = str(meta.get("unique_parameter_count", "None."))
    step_ms = str(meta.get("ms_per_step", "None."))
    val_tokens = str(meta.get("val_tokens", "None."))
    lane_label = str(meta.get("lane_label", "Run"))
    run_name = str(meta.get("run_name", "unknown"))
    command = str(meta.get("command", "")).rstrip()
    return (
        f"# {lane_label}: {run_name}\n\n"
        "## Question\n\n"
        f"{question}\n\n"
        "## Result\n\n"
        f"- status: {meta.get('status', 'unknown')}\n"
        f"- shared comparison val_bpb: {shared_bpb} ({shared_label})\n"
        f"- lane primary val_bpb: {primary_bpb} ({primary_label})\n"
        f"- pre-roundtrip val_bpb: {pre_bpb}\n"
        f"- artifact bytes: {artifact}\n"
        f"- unique parameter count: {params}\n"
        f"- ms/step: {step_ms}\n"
        f"- val tokens: {val_tokens}\n\n"
        "## Conclusion\n\n"
        f"{conclusion}\n\n"
        "## Config Difference\n\n"
        f"{meta.get('config_diff', '')}\n\n"
        "## Promotion Rule\n\n"
        f"{meta.get('promotion_rule', '')}\n\n"
        "## Next Change\n\n"
        f"{next_change}\n\n"
        "## Command\n\n"
        "```bash\n"
        f"{command}\n"
        "```\n\n"
        "## Notes\n\n"
        f"{notes}\n"
    )


def append_observation(run_dir: Path, meta: dict[str, object]) -> None:
    notes_path = run_dir.parents[2] / "notes" / "run_observations.md"
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    date_heading = run_dir.name.split("_", 1)[0]
    block = (
        f"\n## {date_heading}\n\n"
        f"- question being asked: {meta.get('question', '')}\n"
        f"- exact config difference: {meta.get('config_diff', '')}\n"
        f"- gpu used: {meta.get('gpu', 'Pending')}\n"
        f"- key metrics: shared_comparison_val_bpb={meta.get('shared_comparison_val_bpb', 'Pending')} "
        f"({meta.get('shared_comparison_metric_label', 'none')}), "
        f"lane_primary_val_bpb={meta.get('lane_primary_val_bpb', meta.get('post_roundtrip_val_bpb', 'Pending'))} "
        f"({meta.get('lane_primary_metric_label', meta.get('post_roundtrip_metric_label', 'none'))}), "
        f"artifact_bytes={meta.get('serialized_model_int8_zlib_bytes') or meta.get('serialized_model_int6_lzma_bytes') or meta.get('serialized_model_bytes', 'Pending')}, "
        f"unique_parameter_count={meta.get('unique_parameter_count', 'Pending')}, "
        f"ms_per_step={meta.get('ms_per_step', 'Pending')}, "
        f"val_tokens={meta.get('val_tokens', 'Pending')}\n"
        f"- what changed next: {meta.get('next_change', 'Pending')}\n"
        f"- conclusion: {meta.get('conclusion', 'Pending')}\n"
        f"- run folder: `{run_dir.relative_to(run_dir.parents[2])}`\n"
    )
    with notes_path.open("a", encoding="utf-8") as handle:
        handle.write(block)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    metrics_path = run_dir / "metrics.json"
    summary_path = run_dir / "summary.md"
    train_log_path = run_dir / args.train_log
    eval_log_path = run_dir / args.eval_log

    if not metrics_path.exists():
        raise SystemExit(f"Missing metrics.json in run dir: {run_dir}")
    if not train_log_path.exists():
        raise SystemExit(f"Missing train log: {train_log_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    parsed = parse_log(train_log_path)
    if eval_log_path.exists() and eval_log_path.read_text(encoding="utf-8").strip():
        parsed_eval = parse_log(eval_log_path)
        if "final_metrics" in parsed and "final_metrics" in parsed_eval:
            parsed["final_metrics"].update(parsed_eval["final_metrics"])
            parsed_eval = {k: v for k, v in parsed_eval.items() if k != "final_metrics"}
        parsed.update(parsed_eval)

    if parsed:
        metrics.update(parsed)
    metrics["status"] = args.status
    metrics["finalized_at_utc"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    if args.conclusion:
        metrics["conclusion"] = args.conclusion.strip()
    if args.next_change:
        metrics["next_change"] = args.next_change.strip()

    write_metrics(metrics_path, metrics)
    summary_path.write_text(build_summary(metrics), encoding="utf-8")

    if args.append_observation:
        append_observation(run_dir, metrics)
    print(run_dir)


if __name__ == "__main__":
    main()
