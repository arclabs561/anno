# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate short, independently scored histories of two publishers crediting one work."""
import argparse
from itertools import product
import json
from pathlib import Path
from tempfile import TemporaryDirectory


EVENTS = ("grant-a", "drop-a", "grant-b", "drop-b", "hide-a", "show-a")


def claim(issuer):
    return {"id": issuer, "kind": "credit", "issuer": issuer, "source": issuer,
            "role": "editing", "target": "editor"}


def history(events):
    # The answer key keeps only two facts (each issuer's support) per content edition.
    # A separate switch controls whether the public can see A's evidence right now.
    versions = [(True, False)]
    steps = [{"revision": 1, "put": [claim("a")]}]
    hidden = False
    policies = [{"revision": 1, "hidden": []}]
    questions, answers = [], {}
    for n, event in enumerate(("start", *events)):
        if event in ("hide-a", "show-a"):
            hidden = event == "hide-a"
            policies.append({"revision": len(policies) + 1, "hidden": ["a"] if hidden else []})
        elif event != "start":
            verb, issuer = event.split("-")
            before = versions[-1]
            active = verb == "grant"
            updated = (active, before[1]) if issuer == "a" else (before[0], active)
            if updated != before:
                versions.append(updated)
                operation = {"put": [claim(issuer)]} if active else {"withdraw": [issuer]}
                steps.append({"revision": len(versions), **operation})
        # Read every historical content state with the policy in force at this prefix.
        for at, (a, b) in enumerate(versions, 1):
            for audience in ("public", "staff"):
                support = []
                if a and (audience == "staff" or not hidden):
                    support.append("a")
                if b:
                    support.append("b")
                for kind in ("credit", "evidence"):
                    key = f"{n}-at{at}-{audience}-{kind}"
                    q = {"id": key, "at": at, "policy": len(policies), "audience": audience,
                         "kind": kind, "role": "editing"}
                    if kind == "evidence":
                        q["target"] = "editor"
                    questions.append(q)
                    answers[key] = support if kind == "evidence" else (["editor"] if support else [])
    case = {"id": ".".join(events), "work": "one-work",
            "sources": [{"id": i, "edition": "1", "text": "Credit."} for i in ("a", "b")],
            "steps": steps, "policies": policies, "queries": questions}
    return case, answers


def generate(sequences):
    cases, answers = [], {}
    for events in sequences:
        case, key = history(events)
        cases.append(case)
        answers[case["id"]] = key
    return {"version": 1, "cases": cases}, {"version": 1, "answers": answers}


def check_histories(anno, rust_reader):
    from check import call_json, compare, read_suite

    cases, key = generate(product(EVENTS, repeat=3))
    compare("SQLite bounded histories", read_suite(cases, anno), key)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "cases.json"
        path.write_text(json.dumps(cases), encoding="utf-8")
        compare("Rust bounded histories", call_json([rust_reader, str(path)]), key)
    count = sum(map(len, key["answers"].values()))
    print(f"{len(cases['cases'])} three-event histories, {count} content/policy queries per reader")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emit an original proposed event history or its separate answer key.")
    parser.add_argument("events", nargs="+", choices=EVENTS)
    parser.add_argument("--key", action="store_true", help="emit the independently computed expected answers")
    args = parser.parse_args()
    cases, key = generate([args.events])
    print(json.dumps(key if args.key else cases, indent=2))
