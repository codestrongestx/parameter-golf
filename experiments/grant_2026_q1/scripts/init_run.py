from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

LANE_LABELS = {
    "official_neural": "Official Neural",
    "ngram_rescore": "N-Gram Rescore",
    "tokenizer_wildcard": "Tokenizer Wildcard",
}

DEFAULT_PROMOTION_RULES = {
    "official_neural": (
        "Promote to 1x H100 only if the 3090 run completes cleanly, the roundtrip "
        "metric is strong enough to justify better measurement, and evaluation cost stays acceptable."
    ),
    "ngram_rescore": (
        "Run only after the official_neural base is strong and stable, and only for one isolated cache-vs-base question."
    ),
    "tokenizer_wildcard": (
        "Keep isolated from official_neural. Continue only if BPB verification remains extremely clean."
    ),
}

SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a curated Grant 2026 Q1 run folder before a paid run.")
    parser.add_argument("lane", choices=sorted(LANE_LABELS))
    parser.add_argument("--gpu", required=True, help="GPU label, for example 3090 or 1xH100")
    parser.add_argument("--run-id", required=True, help="Short path-safe run identifier")
    parser.add_argument("--question", required=True, help="One explicit question this run will answer")
    parser.add_argument("--command", required=True, help="Exact command to preserve in command.txt")
    parser.add_argument("--config-diff", default="TODO", help="Exact config difference or hypothesis under test")
    parser.add_argument("--hypothesis", default="TODO", help="Expected outcome if the lane is worth continuing")
    parser.add_argument(
        "--promotion-rule",
        help="Rule for moving this lane to the next rung of the ladder",
    )
    parser.add_argument("--notes", default="", help="Optional freeform planning notes for the run summary")
    parser.add_argument(
        "--date",
        help="UTC date for the run folder name, default: today in UTC",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite generated files if the target folder already exists",
    )
    return parser.parse_args()


def require_safe_name(label: str, value: str) -> None:
    if not SAFE_NAME_RE.fullmatch(value):
        raise SystemExit(f"{label} must match {SAFE_NAME_RE.pattern}; got {value!r}")


def write_text(path: Path, content: str, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"Refusing to overwrite existing file without --force: {path}")
    path.write_text(content, encoding="utf-8")


def build_summary(meta: dict[str, object]) -> str:
    notes = str(meta.get("notes", "")).strip() or "None yet."
    command = str(meta["command"]).strip()
    lane_label = str(meta["lane_label"])
    run_name = str(meta["run_name"])
    return (
        f"# {lane_label}: {run_name}\n\n"
        "## Question\n\n"
        f"{meta['question']}\n\n"
        "## Hypothesis\n\n"
        f"{meta['hypothesis']}\n\n"
        "## Config Difference\n\n"
        f"{meta['config_diff']}\n\n"
        "## Promotion Rule\n\n"
        f"{meta['promotion_rule']}\n\n"
        "## Command\n\n"
        "```bash\n"
        f"{command}\n"
        "```\n\n"
        "## Result\n\n"
        "Pending run.\n\n"
        "## Notes\n\n"
        f"{notes}\n"
    )


def main() -> None:
    args = parse_args()
    require_safe_name("gpu", args.gpu)
    require_safe_name("run_id", args.run_id)
    if not args.question.strip():
        raise SystemExit("--question must not be empty")
    if not args.command.strip():
        raise SystemExit("--command must not be empty")

    root = Path(__file__).resolve().parents[1]
    run_date = args.date or datetime.now(timezone.utc).date().isoformat()
    run_name = f"{run_date}_{args.gpu}_{args.run_id}"
    run_dir = root / args.lane / "results" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    command_txt = run_dir / "command.txt"
    metrics_json = run_dir / "metrics.json"
    summary_md = run_dir / "summary.md"
    train_log = run_dir / "train.log"
    eval_log = run_dir / "eval.log"

    promotion_rule = args.promotion_rule or DEFAULT_PROMOTION_RULES[args.lane]
    meta: dict[str, object] = {
        "status": "planned",
        "lane": args.lane,
        "lane_label": LANE_LABELS[args.lane],
        "gpu": args.gpu,
        "run_id": args.run_id,
        "run_name": run_name,
        "question": args.question.strip(),
        "command": args.command.strip(),
        "config_diff": args.config_diff.strip(),
        "hypothesis": args.hypothesis.strip(),
        "promotion_rule": promotion_rule.strip(),
        "notes": args.notes.strip(),
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "paths": {
            "run_dir": str(run_dir.relative_to(root)),
            "command_txt": str(command_txt.relative_to(root)),
            "metrics_json": str(metrics_json.relative_to(root)),
            "summary_md": str(summary_md.relative_to(root)),
            "train_log": str(train_log.relative_to(root)),
            "eval_log": str(eval_log.relative_to(root)),
        },
        "lane_primary_metric_label": None,
        "lane_primary_val_bpb": None,
        "post_roundtrip_val_bpb": None,
        "shared_comparison_metric_label": None,
        "shared_comparison_val_bpb": None,
        "artifact_bytes": None,
        "unique_parameter_count": None,
        "ms_per_step": None,
    }

    write_text(command_txt, f"{args.command.strip()}\n", args.force)
    write_text(metrics_json, json.dumps(meta, indent=2, sort_keys=True) + "\n", args.force)
    write_text(summary_md, build_summary(meta), args.force)
    write_text(train_log, "", args.force)
    write_text(eval_log, "", args.force)
    print(run_dir)


if __name__ == "__main__":
    main()
