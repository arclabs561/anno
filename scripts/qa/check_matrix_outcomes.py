"""Check coverage from the current CI run's matrix decision/outcome JSONL."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys


RANDOM_TASKS = ("ner", "re", "intra-coref", "temporal")


def outcomes(lines, decisions=None):
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
            if row.get("record_type") != "outcome":
                if (decisions is not None and not decisions
                        and row.get("strategy") == "worst-first" and row.get("chosen")):
                    decisions.append(row)
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


def check_wikigold(lines, bootstrap=False):
    rows = list(outcomes(lines))
    if len(rows) == 1:
        lane, row = rows[0]
        if (
            lane == "ner"
            and row.get("backend") == "heuristic"
            and row.get("dataset") == "WikiGold"
            and row["hard_junk"] is False
            and row["cost_units"] == (5 if bootstrap else 1)
            and type(row.get("primary_f1")) in (float, int)
            and math.isfinite(row["primary_f1"])
            and (bootstrap or row["primary_f1"] == 1.0)
        ):
            if bootstrap:
                print(f"Cold-start WikiGold baseline: heuristic, 5 examples, F1 {row['primary_f1']:.4f}.")
            else:
                print("Offline sampler scored all labeled spans correctly on the fixture.")
            return True
    print("Unexpected WikiGold baseline result." if bootstrap else "Unexpected offline fixture result.", file=sys.stderr)
    return False


def check(lines, report=False):
    attempts = Counter()
    completions = Counter()
    failures = Counter()
    failure_kinds = Counter()
    decisions = []
    observed = {}
    for lane, row in outcomes(lines, decisions):
        attempts[lane] += 1
        if lane == "worst-first":
            observed[(row.get("run_id"), row.get("backend"), row.get("dataset"))] = row
        if row["hard_junk"]:
            failures[lane] += 1
            failure_kinds[row.get("fail_kind") or "unknown"] += 1
        elif row["cost_units"] > 0:
            completions[lane] += 1

    print("| Sample lane | Completed | Hard failures | Empty evaluations |")
    print("|---|---:|---:|---:|")
    for lane in (*RANDOM_TASKS, "worst-first"):
        empty = attempts[lane] - completions[lane] - failures[lane]
        label = lane if attempts[lane] else f"{lane} (no outcome)"
        print(f"| {label} | {completions[lane]} | {failures[lane]} | {empty} |")

    if failure_kinds:
        print("Hard failures by recorded kind: " + ", ".join(
            f"{kind} {count}" for kind, count in sorted(failure_kinds.items())))

    if decisions:
        first = decisions[0]
        backend = first["chosen"]
        candidates = first.get("top_candidates", {}).get("rows", [])
        selected = next((r for r in candidates if r["arm"] == backend), {})
        prior_calls = selected.get("calls", "unknown")
        reason = "exploration" if first.get("explore_first") else "ranked"
        retry = (first.get("worst_first_round") or {}).get("retry")
        if retry:
            reason = {"outstanding_failure": "unresolved failure",
                      "repeated_infrastructure": "probe of repeated infrastructure failure",
                      "recurrent_after_recovery": "repeated failures after recovery"}.get(retry.get("reason"), "ranked")
            print(f"Retry evidence: {retry['observation_id']} "
                  f"(risk weight {retry['risk']:.3f} over {retry['observations']} observations).")
        for dataset in first.get("datasets", []):
            row = observed.get((first.get("run_id"), backend, dataset))
            result = "no outcome"
            if row is not None:
                result = "hard failure" if row["hard_junk"] else f"{row['cost_units']} examples, F1 {row.get('primary_f1')}"
            print(f"First worst-first selection: {first['slice']} / {dataset} / {backend} ({reason}, prior calls {prior_calls}) — {result}.")
    else:
        print("No worst-first selection recorded.")

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
    mode.add_argument("--bootstrap", action="store_true", help="require five real WikiGold examples")
    mode.add_argument("--smoke", action="store_true", help="require the exact offline fixture result")
    mode.add_argument("--report", action="store_true", help="report missing external coverage without failing")
    args = parser.parse_args()
    try:
        with args.log.open(encoding="utf-8") as stream:
            passed = (check_wikigold(stream, args.bootstrap) if args.smoke or args.bootstrap
                      else check(stream, args.report))
            sys.exit(0 if passed else 1)
    except (OSError, ValueError) as error:
        sys.exit(str(error))
