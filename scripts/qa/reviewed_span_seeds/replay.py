"""Admit reviewed, exposed organization-span fixtures after verified native replay."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_span_assertions import jsonl_rows, load_cases, run_batch, score


def digest(data):
    return sha256(data).hexdigest()


def verified(path, expected):
    data = path.read_bytes()
    if digest(data) != expected:
        raise ValueError(f"Evidence digest mismatch for {path.name}")
    return data


def reviewed_source(entry):
    """Only the maintainer registry supplies these fields."""
    return (entry.get("scope"), entry.get("origin"), entry.get("license")) == (
        "exposed-regression", "project-authored-synthetic", "MIT OR Apache-2.0"
    )


def check_round(fixture, bundle):
    # Only checked-in maintainer decisions can admit a source. An incoming
    # trace, judge or downloaded fixture cannot declare itself redistributable.
    data = fixture.read_bytes()
    fixture_id = digest(data)
    registry = json.loads(Path(__file__).with_name("registry.json").read_bytes())
    admitted = registry.get(fixture_id, {})
    if not reviewed_source(admitted):
        raise ValueError("Fixture is outside the reviewed synthetic regression corpus")
    trace = json.loads(verified(bundle / "trace.json", admitted["trace_sha256"]))
    ticket = json.loads(verified(bundle / "dispatch-receipt.json", admitted["dispatch_sha256"]))
    spans = {s["role"]: s for s in trace["spans"]}
    oracle = spans["oracle"]
    if (oracle["id"] != ticket["span_id"] or ticket["fork_turns"] != "none"
            or ticket["tool"] != oracle["tool"] or oracle["kind"] != "agent" or oracle["status"] != "ok"
            or oracle["prompt_sha256"] != ticket["prompt_sha256"]
            or oracle["result_sha256"] != ticket["response_sha256"]):
        raise ValueError("Oracle dispatch was not isolated and bound to its response")
    verified(bundle / "oracle-prompt.txt", ticket["prompt_sha256"])
    answer = json.loads(verified(bundle / "oracle-response.json", ticket["response_sha256"]))
    cases = load_cases(json.loads(data))
    contract = {
        k: [[a["start"], a["end"], a["text"]] for a in row["assertions"]]
        for k, row in cases.items()
    }
    if answer != contract or any(
        a["kind"] != "exact_span" or a["types"] != ["ORG"] for r in cases.values() for a in r["assertions"]
    ):
        raise ValueError("Fixture assertions disagree with the blind oracle's ORG spans")

    results = {}
    for role, name in [("baseline", "base.jsonl"), ("comparison", "candidate.jsonl")]:
        command = spans[role]
        if (command["prompt_sha256"] != fixture_id or command["kind"] != "command"
                or command["tool"] != "anno.qa" or oracle["ended_at"] > command["started_at"]):
            raise ValueError("The command does not measure this fixture after oracle freeze")
        predictions = jsonl_rows(verified(bundle / name, command["result_sha256"]).decode("utf-8"))
        results[role] = score(cases, predictions)
    return cases, results, fixture_id, admitted["trace_sha256"], ticket["response_sha256"]


def failure_set(report, case):
    return {(f["start"], f["end"]) for f in report["findings"] if f["id"] == case and not f["passed"]}


def replay(fixture, bundle, binary):
    cases, previous, fixture_id, trace_id, oracle_id = check_round(fixture, bundle)
    observed = score(cases, run_batch(cases, str(binary), "heuristic"))
    entries = []
    for key, row in cases.items():
        before = failure_set(previous["baseline"], key)
        after = failure_set(previous["comparison"], key)
        now = failure_set(observed, key)
        entries.append({
            "seed_sha256": digest((fixture_id + "/" + key).encode()),
            "oracle_assertions": len(row["assertions"]),
            "historical_fixed_assertions": len(before - after),
            "historical_lost_assertions": len(after - before),
            "reproduced_fixed_assertions": len((before - after) & now),
            "current_failed_assertions": len(now),
        })
    return {"schema_version": 1, "domain": "heuristic-org-exact-spans",
            "scope": "exposed-regression", "origin": "project-authored-synthetic",
            "fixture_sha256": fixture_id, "trace_sha256": trace_id,
            "oracle_sha256": oracle_id, "binary_sha256": digest(binary.read_bytes()), "seeds": entries}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("trace_directory", type=Path)
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(replay(args.fixture, args.trace_directory, args.binary), indent=2))
    except (OSError, KeyError, ValueError) as err:
        parser.exit(2, f"Seed admission failed: {err}\n")
