"""Prepare a coordinator's challenge from an existing QA run."""
import argparse
import json
import math
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

from round_trace import initialize
from domain_coverage import render_coverage, render_domain_scores
from run_panel import render_markdown

GUIDE = Path(__file__).resolve().parents[2] / "docs" / "ADVERSARIAL_QA.md"


def measured_counts(values, *keys):
    """Legacy or malformed observations cannot support a structural diagnosis."""
    if not isinstance(values, dict):
        return None
    result = [values.get(key) for key in keys]
    if not all(type(v) in (int, float) and 0 <= v <= 2**53 and int(v) == v for v in result):
        return None
    return tuple(map(int, result))


def outcome_evidence(path):
    lines = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record_type") != "outcome":
                continue
            dataset, backend = row["dataset"], row["backend"]
            if not all(isinstance(s, str) and re.fullmatch(r"[A-Za-z0-9_.+-]{1,128}", s)
                       for s in (dataset, backend)):
                raise ValueError("expected registry identifiers for the dataset and backend")
            score = row.get("primary_f1")
            if row.get("hard_junk") or row.get("cost_units") == 0 or score is None:
                lines.append(f"- {dataset} / {backend}: no usable score. Check execution first.")
                continue
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("invalid primary F1")
            if type(row.get("ok")) is not bool:
                raise ValueError("invalid outcome status")
            lines.append(f"- {dataset} / {backend}: ok={row['ok']}, primary F1={score}")
            counts = row.get("structural_counts")
            coref = measured_counts(counts, "gold_coref_links", "reachable_gold_links",
                                    "matched_coref_links", "predicted_coref_links")
            if coref:
                gold, reachable, matched, predicted = coref
                if reachable > gold or matched > min(predicted, gold):
                    raise ValueError("inconsistent coreference link counts")
                lines.append(f"  Resolver inputs cover both ends of {reachable}/{gold} gold links. "
                             f"The resolver output contains {matched} correct and {predicted-matched} extra links.")
                if gold - reachable > reachable:
                    lines.append("  Freeze a mention-proposal comparison on the same documents and resolver. "
                                 "Check extra links on a second, authorized source and no-referent controls.")
                elif reachable > matched:
                    lines.append(f"  At least {reachable-matched} reachable links were missed. "
                                 "Hold mention proposals fixed and vary only the resolver.")
            elif (links := measured_counts(counts, "gold_coref_links", "matched_coref_links",
                                           "predicted_coref_links")) is not None:
                gold, matched, predicted = links
                if matched > min(predicted, gold):
                    raise ValueError("inconsistent coreference link counts")
                lines.append(f"  Output links: {matched} correct, {predicted-matched} extra, "
                             f"{gold-matched} missed. Resolver input coverage was unobserved.")
            temporal = measured_counts(counts, "unmatched_predictions_overlapping_missed_gold",
                                       "matched_spans", "predicted_spans", "gold_empty_sentences",
                                       "gold_empty_sentences_with_predictions")
            if temporal:
                near, matched, predicted, empty, alarms = temporal
                if matched > predicted or near > predicted - matched or alarms > empty:
                    raise ValueError("inconsistent temporal span counts")
                if near:
                    lines.append(f"  {near}/{predicted-matched} unmatched predictions overlap missed gold. "
                                 "Compare native and reference decoding on identical frozen model outputs.")
                if empty:
                    lines.append(f"  {alarms}/{empty} gold-empty sentences triggered. Preserve that control.")
                else:
                    lines.append("  No gold-empty sentences were scored. Reserve a labeled negative panel.")
    if not lines:
        raise ValueError("completed measurement has no outcome rows")
    return "\n".join(lines)


def evidence(directory):
    if directory.is_file():
        if directory.suffix == ".jsonl":
            return outcome_evidence(directory)
        receipt = json.loads(directory.read_text(encoding="utf-8"))
        if isinstance(receipt, dict) and receipt.get("kind") == "anno.qa.domain-spans":
            return render_domain_scores(receipt)
        return render_coverage(receipt)
    panel = directory / "panel-summary.json"
    schedule = directory / "schedule.json"
    if panel.exists() and schedule.exists():
        raise ValueError("give the directory of a single panel or matrix slot")
    if panel.is_file():
        return render_markdown(json.loads(panel.read_text())["results"]).rstrip()
    if not schedule.is_file():
        raise ValueError("expected panel-summary.json or schedule.json in the run directory, or a cross-domain receipt file")
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
    return lines[0] + "\n" + outcome_evidence(directory / "decisions.jsonl")


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
