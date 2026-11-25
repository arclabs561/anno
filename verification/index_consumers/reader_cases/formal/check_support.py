# SPDX-License-Identifier: MIT OR Apache-2.0
"""Compare both readers to Z3 for every two-issuer, two-edition correction in the model."""
import argparse
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from check import build_rust_reader, call_json, read_suite

sys.path.insert(0, str(HERE.parents[1] / "congress"))
from check_catalog_history import build_cli

PROPERTIES = {
    "independent-issuer-corrections-commute": "unsat",
    "dropping-A-preserves-permitted-B-support": "unsat",
    "hidden-A-cannot-alter-displayed-targets": "unsat",
    "corrected-conflict-is-reachable": "sat",
    "unqualified-withdrawal-loses-B": "sat",
    "old-policy-leaks-historical-credit": "sat",
    "latest-content-rewrites-permitted-history": "sat",
    "first-eligible-choice-resolves-a-real-conflict": "sat",
}
ISSUERS = ("press", "archive")
STATES = list(product(("Absent", "X", "Y"), repeat=2))
PERMISSIONS = list(product((True, False), repeat=2))
TARGETS = {"#b00": [], "#b01": ["X"], "#b10": ["Y"], "#b11": ["X", "Y"]}


def oracle():
    model = (HERE / "support.smt2").read_text(encoding="utf-8")
    inputs = list(product(STATES, PERMISSIONS))
    queries = []
    for (a, b), (show_a, show_b) in inputs:
        shown = f"(shown (supports {a} {b}) {str(show_a).lower()} {str(show_b).lower()})"
        queries.extend((f"(simplify {shown})", f"(simplify (single-slot {shown}))"))
    run = subprocess.run(
        ["z3", "-in"], input=model + "\n".join(queries), text=True,
        capture_output=True, check=True, timeout=30,
    )
    lines = run.stdout.splitlines()
    named = [value for pair in PROPERTIES.items() for value in pair]
    if run.stderr or lines[:len(named)] != named:
        raise ValueError(f"Unexpected named Z3 results:\n{run.stdout}{run.stderr}")
    values = iter(lines[len(named):])
    if len(lines) != len(named) + 2 * len(inputs):
        raise ValueError("Incomplete Z3 truth table")
    return {key: (TARGETS[next(values)], next(values)) for key in inputs}


def local_id(issuer):
    return {"issuer": issuer, "id": "7"}


def patch(revision, before, after):
    step = {"revision": revision, "put": [], "withdraw": []}
    for issuer, previous, current in zip(ISSUERS, before, after):
        if previous == current:
            continue
        address = local_id(issuer)
        if current == "Absent":
            step["withdraw"].append(address)
        else:
            step["put"].append({
                **address, "kind": "credit", "role": "editor", "target": current, "source": "page",
            })
    return step


def generate(table):
    cases = []
    expected = {}
    policies = [
        {"revision": policy, "hidden": [local_id(i) for i, visible in zip(ISSUERS, mask) if not visible]}
        for policy, mask in enumerate(PERMISSIONS, 1)
    ]
    for old, new in product(STATES, repeat=2):
        case_id = "-".join((*old, *new))
        answers = {}
        questions = []
        for edition, state in enumerate((old, new), 1):
            for policy, mask in enumerate(PERMISSIONS, 1):
                for issuer, audience in product((None, *ISSUERS), ("public", "staff")):
                    eligible = tuple(
                        (visible or audience == "staff") and issuer in (None, who)
                        for who, visible in zip(ISSUERS, mask)
                    )
                    key = f"{edition}-{policy}-{issuer or 'all'}-{audience}"
                    answers[key] = table[state, eligible]
                    query = {
                        "id": key, "at": edition, "policy": policy, "audience": audience,
                        "kind": "credit", "role": "editor",
                    }
                    if issuer:
                        query["issuer"] = issuer
                    questions.append(query)
        cases.append({
            "id": case_id, "work": "one-slot-review",
            "sources": [{"id": "page", "edition": "original", "text": "Editorial record."}],
            "steps": [patch(1, ("Absent", "Absent"), old), patch(2, old, new)],
            "policies": policies, "queries": questions,
        })
        expected[case_id] = answers
    return {"version": 1, "cases": cases}, expected


def check_answers(reader, data, expected):
    if data["version"] != 1 or data["answers"].keys() != expected.keys():
        raise ValueError(f"{reader}: missing or unexpected cases")
    for case, answers in expected.items():
        actual = data["answers"][case]
        if actual.keys() != answers.keys():
            raise ValueError(f"{reader}/{case}: missing or unexpected questions")
        for query, (targets, decision) in answers.items():
            observed = actual[query]
            # This form accepts a single explicit target. It can still display
            # the reader's full list when there is a conflict to review.
            selected = observed[0] if len(observed) == 1 else "Absent"
            if observed != targets or selected != decision:
                raise ValueError(
                    f"{reader}/{case}/{query}: got {(observed, selected)}; "
                    f"Z3 expected {(targets, decision)}"
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anno", type=Path, help="built anno CLI")
    parser.add_argument("--rust-reader", type=Path, help="built reader_answer example")
    args = parser.parse_args()
    suite, expected = generate(oracle())
    answers = read_suite(suite, args.anno or build_cli())
    check_answers("SQLite", answers, expected)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "cases.json"
        path.write_text(json.dumps(suite), encoding="utf-8")
        answers = call_json([args.rust_reader or build_rust_reader(), path])
    check_answers("Rust", answers, expected)
    print("Z3: 3 unsatisfiable violations, 1 reachable conflict and 4 satisfiable mutant witnesses.")
    count = sum(map(len, expected.values()))
    print(f"{len(expected)} two-edition histories, {count} solver-checked queries per reader.")


if __name__ == "__main__":
    main()
