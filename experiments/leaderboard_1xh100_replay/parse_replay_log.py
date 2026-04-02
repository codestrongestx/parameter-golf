from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

STEP_RE = re.compile(
    r"^step:(?P<step>\d+)/(?P<iterations>\d+) "
    r"(?:(?:train_loss:(?P<train_loss>[-+]?\d+(?:\.\d+)?) )?"
    r"(?:val_loss:(?P<val_loss>[-+]?\d+(?:\.\d+)?) val_bpb:(?P<val_bpb>[-+]?\d+(?:\.\d+)?) )?)?"
    r"train_time:(?P<train_time_ms>\d+)ms step_avg:(?P<step_avg_ms>[-+]?\d+(?:\.\d+)?)ms$"
)
DIAG_RE = re.compile(
    r"^DIAGNOSTIC (?P<label>[A-Za-z0-9_]+) val_loss:(?P<val_loss>[-+]?\d+(?:\.\d+)?) "
    r"val_bpb:(?P<val_bpb>[-+]?\d+(?:\.\d+)?) eval_time:(?P<eval_time_ms>\d+)ms$"
)
FINAL_RE = re.compile(
    r"^(?P<label>[A-Za-z0-9_]+) val_loss:(?P<val_loss>[-+]?\d+(?:\.\d+)?) "
    r"val_bpb:(?P<val_bpb>[-+]?\d+(?:\.\d+)?)"
    r"(?: stride:(?P<stride>\d+))?(?: eval_time:(?P<eval_time_ms>\d+)ms)?$"
)
SERIALIZED_RE = re.compile(r"^Serialized model(?: [A-Za-z0-9_+]+)?: (?P<bytes>\d+) bytes$")
TOTAL_SIZE_RE = re.compile(r"^Total submission size(?: [A-Za-z0-9_+]+)?: (?P<bytes>\d+) bytes$")
STOP_RE = re.compile(
    r"^stopping_early: (?P<reason>\S+) train_time:(?P<train_time_ms>\d+)ms step:(?P<step>\d+)/(?P<iterations>\d+)$"
)
PHASE_RE = re.compile(r"^phase_timings:(?P<payload>\{.*\})$")


def parse_log(path: Path) -> dict[str, object]:
    metrics: dict[str, object] = {"train_log": str(path)}
    exact_metrics: dict[str, dict[str, object]] = {}
    last_step: dict[str, object] | None = None
    last_val_step: dict[str, object] | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        if match := STEP_RE.match(line):
            last_step = {
                "step": int(match.group("step")),
                "iterations": int(match.group("iterations")),
                "train_time_ms": int(match.group("train_time_ms")),
                "step_avg_ms": float(match.group("step_avg_ms")),
            }
            if match.group("val_bpb") is not None:
                last_val_step = {
                    "step": int(match.group("step")),
                    "iterations": int(match.group("iterations")),
                    "val_loss": float(match.group("val_loss")),
                    "val_bpb": float(match.group("val_bpb")),
                    "train_time_ms": int(match.group("train_time_ms")),
                    "step_avg_ms": float(match.group("step_avg_ms")),
                }
            continue

        if match := DIAG_RE.match(line):
            metrics["diagnostic_label"] = match.group("label")
            metrics["post_ema_val_loss"] = float(match.group("val_loss"))
            metrics["post_ema_val_bpb"] = float(match.group("val_bpb"))
            metrics["post_ema_eval_time_ms"] = int(match.group("eval_time_ms"))
            continue

        if match := FINAL_RE.match(line):
            label = match.group("label")
            if label.endswith("_exact"):
                exact_metrics[label] = {
                    "val_loss": float(match.group("val_loss")),
                    "val_bpb": float(match.group("val_bpb")),
                }
                if match.group("stride"):
                    exact_metrics[label]["stride"] = int(match.group("stride"))
                if match.group("eval_time_ms"):
                    exact_metrics[label]["eval_time_ms"] = int(match.group("eval_time_ms"))
            continue

        if match := SERIALIZED_RE.match(line):
            metrics["artifact_bytes"] = int(match.group("bytes"))
            continue

        if match := TOTAL_SIZE_RE.match(line):
            metrics["total_submission_size_bytes"] = int(match.group("bytes"))
            continue

        if match := STOP_RE.match(line):
            metrics["stop_reason"] = match.group("reason")
            metrics["stopped_at_step"] = int(match.group("step"))
            metrics["stopped_train_time_ms"] = int(match.group("train_time_ms"))
            continue

        if match := PHASE_RE.match(line):
            metrics["phase_timings"] = json.loads(match.group("payload"))
            continue

    if last_step is not None:
        metrics["last_train_step"] = last_step["step"]
        metrics["iterations_target"] = last_step["iterations"]
        metrics["ms_per_step"] = last_step["step_avg_ms"]
    if last_val_step is not None:
        metrics["pre_roundtrip_step"] = last_val_step["step"]
        metrics["pre_roundtrip_val_loss"] = last_val_step["val_loss"]
        metrics["pre_roundtrip_val_bpb"] = last_val_step["val_bpb"]
    if exact_metrics:
        metrics["exact_metrics"] = exact_metrics
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a 1xH100 leaderboard replay train log.")
    parser.add_argument("train_log", type=Path)
    parser.add_argument("--output", type=Path, help="Optional path to write metrics JSON")
    args = parser.parse_args()

    metrics = parse_log(args.train_log)
    payload = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
