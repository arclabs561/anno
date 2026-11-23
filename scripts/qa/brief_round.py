"""Prepare a coordinator's challenge from an existing QA run."""
import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from round_trace import initialize
from run_panel import render_markdown

GUIDE = Path(__file__).resolve().parents[2] / "docs" / "ADVERSARIAL_QA.md"


def evidence(directory):
    panel = directory / "panel-summary.json"
    schedule = directory / "schedule.json"
    if panel.exists() and schedule.exists():
        raise ValueError("give the directory of a single panel or matrix slot")
    if panel.is_file():
        return render_markdown(json.loads(panel.read_text())["results"]).rstrip()
    if not schedule.is_file():
        raise ValueError("expected panel-summary.json or schedule.json in the run directory")
    plan = json.loads(schedule.read_text())
    lines = [f"Matrix slot {plan['slot']}, requested {plan['requested_strategy']}, seed {plan['seed']}."]
    result = directory / "result.json"
    if not result.is_file():
        return lines[0] + " The slot did not produce a validated result. Investigate execution first."
    run = json.loads(result.read_text())
    if not run["strategy_matches"]:
        return lines[0] + " The observed strategy differed. Exclude this slot from policy comparisons."
    if run["plan_only"]:
        return lines[0] + " Selection only. No backend was evaluated."
    for line in (directory / "decisions.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("record_type") == "outcome":
            lines.append(f"- {row['dataset']} / {row['backend']}: ok={row['ok']}, primary F1={row.get('primary_f1')}")
    if len(lines) == 1:
        raise ValueError("completed measurement has no outcome rows")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", type=Path)
    parser.add_argument("claim")
    parser.add_argument("--trace", type=Path, help="create a local round trace at a new SQLite path")
    args = parser.parse_args()
    try:
        summary = evidence(args.receipts)
        handoff = (f"# Coordinator handoff\n\nClaim to investigate: {args.claim}\n\n"
                   f"Starting evidence from {args.receipts}\n\n{summary}\n\n{GUIDE.read_text()}\n")
        if args.trace:
            trace_id = initialize(args.trace, handoff)
            print(f"QA trace {trace_id} at {args.trace}", file=sys.stderr)
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Cannot prepare QA round: {exc}\n")
    print(handoff, end="")


if __name__ == "__main__":
    main()
