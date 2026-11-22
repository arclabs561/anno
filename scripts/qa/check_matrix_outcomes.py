"""Check coverage from the current CI run's matrix decision/outcome JSONL."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


RANDOM_TASKS = ("ner", "re", "intra-coref", "temporal")


def check(lines):
    attempts = Counter()
    completions = Counter()
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
            lane = task if strategy == "random" else strategy
            hard_failure = row["hard_junk"]
            examples = row["cost_units"]
            if type(hard_failure) is not bool or type(examples) is not int or examples < 0:
                raise ValueError("expected boolean hard_junk and nonnegative integer cost_units")
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            raise ValueError(f"invalid matrix log at line {number}: {exc}") from exc

        attempts[lane] += 1
        if not hard_failure and examples > 0:
            completions[lane] += 1

    for lane in (*RANDOM_TASKS, "worst-first"):
        print(f"{lane}: {completions[lane]} completed / {attempts[lane]} observed")

    missing = [task for task in RANDOM_TASKS if not completions[task]]
    if missing:
        print(f"No completed evaluations with examples for {', '.join(missing)}", file=sys.stderr)
    # Worst-first deliberately targets poor/failing cells. A recorded failure
    # demonstrates an attempt, but a selection decision alone does not.
    if not attempts["worst-first"]:
        print("No worst-first evaluation outcomes", file=sys.stderr)
    return not missing and bool(attempts["worst-first"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    try:
        with args.log.open(encoding="utf-8") as stream:
            sys.exit(0 if check(stream) else 1)
    except (OSError, ValueError) as error:
        sys.exit(str(error))
