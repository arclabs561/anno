"""Check coverage from the current CI run's matrix decision/outcome JSONL."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


RANDOM_TASKS = ("ner", "re", "intra-coref", "temporal")


def outcomes(lines):
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
            if row.get("record_type") != "outcome":
                continue
            strategy = row["strategy"]
            task = row["slice"].split(".", 1)[0]
            if strategy not in {"random", "worst-first"}:
                raise ValueError(f"unexpected sampling strategy {strategy!r}")
            if strategy == "random" and task not in RANDOM_TASKS:
                raise ValueError(f"unexpected random task {task!r}")
            hard_failure = row["hard_junk"]
            examples = row["cost_units"]
            if type(hard_failure) is not bool or type(examples) is not int or examples < 0:
                raise ValueError("expected boolean hard_junk and nonnegative integer cost_units")
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            raise ValueError(f"invalid matrix log at line {number}: {exc}") from exc

        yield task if strategy == "random" else strategy, row


def check_smoke(lines):
    rows = list(outcomes(lines))
    if len(rows) == 1:
        lane, row = rows[0]
        if (
            lane == "ner"
            and row.get("backend") == "heuristic"
            and row.get("dataset") == "WikiGold"
            and row["hard_junk"] is False
            and row["cost_units"] == 1
            and row.get("primary_f1") == 1.0
        ):
            print("Offline sampler scored all labeled spans correctly on the fixture.")
            return True
    print("Expected one complete heuristic NER outcome with F1=1 on the fixture.", file=sys.stderr)
    return False


def check(lines, report=False):
    attempts = Counter()
    completions = Counter()
    failures = Counter()
    for lane, row in outcomes(lines):
        attempts[lane] += 1
        if row["hard_junk"]:
            failures[lane] += 1
        elif row["cost_units"] > 0:
            completions[lane] += 1

    print("| Sample lane | Completed | Hard failures | Empty evaluations |")
    print("|---|---:|---:|---:|")
    for lane in (*RANDOM_TASKS, "worst-first"):
        empty = attempts[lane] - completions[lane] - failures[lane]
        label = lane if attempts[lane] else f"{lane} (no outcome)"
        print(f"| {label} | {completions[lane]} | {failures[lane]} | {empty} |")

    missing = [task for task in RANDOM_TASKS if not completions[task]]
    if missing:
        print(f"No completed evaluations with examples for {', '.join(missing)}", file=sys.stderr)
    # Worst-first deliberately targets poor/failing cells. A recorded failure
    # demonstrates an attempt, but a selection decision alone does not.
    if not attempts["worst-first"]:
        print("No worst-first evaluation outcomes", file=sys.stderr)
    return report or (not missing and bool(attempts["worst-first"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="require the exact offline fixture result")
    mode.add_argument("--report", action="store_true", help="report missing external coverage without failing")
    args = parser.parse_args()
    try:
        with args.log.open(encoding="utf-8") as stream:
            passed = check_smoke(stream) if args.smoke else check(stream, args.report)
            sys.exit(0 if passed else 1)
    except (OSError, ValueError) as error:
        sys.exit(str(error))
