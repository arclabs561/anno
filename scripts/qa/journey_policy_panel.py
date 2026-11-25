# SPDX-License-Identifier: MIT OR Apache-2.0
"""Replay three offline developer journeys with a sealed fault schedule and equal probe slots."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time

import crossdoc_cases
import journey_pilot as pilot


def cli(directory, binary, *args):
    return subprocess.run([binary, *args], cwd=directory, capture_output=True, timeout=20,
                          env={**os.environ, **pilot.OFFLINE})


def json_output(result):
    result.check_returncode()
    return json.loads(result.stdout)


def write(directory, name, value):
    (directory / name).write_text(json.dumps(value), encoding="utf-8")


def span(directory, binary, mutate):
    (directory / "mentions.txt").write_text("Éva met Zoë", encoding="utf-8")
    ann = directory / "mentions.ann"
    dest = directory / "reviewed.jsonl"
    args = ("dataset", "import", "--format", "brat", "--include-text", "--input", ann.name,
            "--output", dest.name)
    ann.write_text("T1\tPER 0 3;8 11\tÉva Zoë\n", encoding="utf-8")
    cli(directory, binary, *args).check_returncode()
    row, = [json.loads(line) for line in dest.read_bytes().splitlines()]
    if mutate:
        row["segments"].pop()
    if row.get("segments") != [{"start": 0, "end": 3}, {"start": 8, "end": 11}]:
        return "brat-output-segments"
    ann.write_text("T1\tPER 0 3\tÉva\nT2\tPER\n", encoding="utf-8")
    dest.write_bytes(b"reviewed corpus\n")
    bad = cli(directory, binary, *args)
    if bad.returncode == 0 or dest.read_bytes() != b"reviewed corpus\n":
        return "brat-incomplete-input-overwrites-review"
    if b"line 2" not in bad.stderr:
        raise ValueError("BRAT control failed for an unexpected reason")


def catalog(directory, binary, mutate):
    case = json.loads((pilot.ROOT / pilot.FIXTURE).read_bytes())
    reports = {4: directory / "report.json"}
    crossdoc_cases.run_case(case, binary, reports=reports)
    query, = (q for q in json.loads((pilot.ROOT / pilot.QUERIES).read_bytes())["queries"]
              if q["after"] == 4 and q.get("catalog_record") == ["stage", "22"])
    rows = json_output(cli(directory, binary, "query", reports[4].name, "--format", "json",
                           "--catalog-record", *query["catalog_record"]))
    if mutate:
        rows[0]["mentions"].pop()
    actual = sorted(posting for row in rows for posting in row["mentions"])
    if actual != sorted(query["postings"]):
        return "catalog-current-edition-postings"
    write(directory, "doc.json", crossdoc_cases.validate_case(case)[0])
    dest = directory / "published.json"
    dest.write_bytes(b"previous report\n")
    bad = cli(directory, binary, "cross-doc", "--output", dest.name, ".", "--import", "doc.json")
    if bad.returncode == 0 or dest.read_bytes() != b"previous report\n":
        return "catalog-incomplete-source-selection"
    if b"cannot be used with" not in bad.stderr:
        raise ValueError("mixed-source control failed for an unexpected reason")


def stale(directory, binary, mutate):
    def corpus(*args):
        return cli(directory, binary, "corpus", "--store", "journal.sqlite", *args)

    def deliver(name):
        write(directory, "operation", {"kind": "delivery", "request": {
            "issuer": "archive", "delivery_id": name, "source_revision": 1,
            "ownership": {"mode": "independent"}}})
        write(directory, "doc", {"id": name, "text": "Reed", "signals": [],
                                 "tracks": {}, "identities": {}})
        return json_output(corpus("apply", "operation", "--document", "doc"))["receipt"]["document_id"]

    first = deliver("first")
    head = json_output(corpus("read-many", first))["head"]
    saved = corpus("read-many", "--at-head", head, first)
    if json_output(saved)["head"] != head:
        raise ValueError("pinned positive control returned another head")
    deliver("second")
    rejected = corpus("read-many", "--at-head", head, first)
    result = saved if mutate else rejected
    if result.returncode == 0 or result.stdout:
        return "journal-stale-pinned-response"
    if b"head" not in result.stderr.lower():
        raise ValueError("stale control failed for an unexpected reason")


PROBES = dict(zip(pilot.JOURNEYS, (span, catalog, stale), strict=True))
MUTATION_SIGNATURES = {"span-roundtrip": "brat-output-segments",
                      "catalog-update": "catalog-current-edition-postings",
                      "stale-revision": "journal-stale-pinned-response"}


def observe(binary, name, mutate):
    def one():
        with tempfile.TemporaryDirectory(prefix="anno-panel-") as temp:
            return PROBES[name](Path(temp), binary, mutate)
    started = time.monotonic()
    signature = one()
    if signature and one() != signature:
        raise ValueError(f"{name} did not reproduce")
    return {"failure": signature, "elapsed_ms": round((time.monotonic() - started) * 1000)}


def mutation_census(binary, faults):
    census = {}
    for name, signature in MUTATION_SIGNATURES.items():
        active = name in faults
        arm = observe(binary, name, active)
        expected = signature if active else None
        if arm["failure"] != expected:
            raise ValueError(f"{name} expected {expected!r}, observed {arm['failure']!r}")
        census[name] = arm
    return census


def simulate(selector, rounds, seed):
    state = {"rounds": 0, "history": []}
    trace = []
    for index, reveal in enumerate(rounds):
        picks = pilot.picks(state, seed + index, selector)
        # Selection is complete before the experiment harness discloses this round.
        census = reveal()
        observations = {}
        for policy, name in picks.items():
            signature = census[name]["failure"]
            value = hashlib.sha256(signature.encode()).hexdigest() if signature else "pass"
            observations[policy] = [name, value]
        state["history"] = (state["history"] + [observations])[-pilot.WINDOW:]
        state["rounds"] += 1
        trace.append({"picks": picks, "census": census})
    return trace


def summarize(trace):
    counts = Counter((name, arm["failure"]) for row in trace for name, arm in row["census"].items()
                     if arm["failure"])
    result = {}
    for policy in pilot.POLICIES:
        choices = [row["picks"][policy] for row in trace]
        attempts = [row["census"][name] for row, name in zip(trace, choices)]
        hits = [(i + 1, name, measurement["failure"])
                for i, (name, measurement) in enumerate(zip(choices, attempts))
                if measurement["failure"]]
        result[policy] = {"slots": len(trace), "hits": len(hits), "first_hit": hits[0][0] if hits else None,
                          "distinct_failures": len({(name, cause) for _, name, cause in hits}),
                          "arms_seen": len(set(choices)), "probe_ms": sum(a["elapsed_ms"] for a in attempts)}
    size = len(pilot.JOURNEYS)
    result["uniform_expectation"] = {"hits": sum(counts.values()) / size,
        "distinct_failures": math.fsum(1 - (1 - 1 / size) ** n for n in counts.values())}
    return result


def sealed_schedule(path, digest):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("schedule differs from its precommitted digest")
    rounds = json.loads(raw)
    if not isinstance(rounds, list) or len(rounds) != 12:
        raise ValueError("expected twelve rounds")
    for row in rounds:
        if (not isinstance(row, list) or not all(name in PROBES for name in row)
                or len(set(row)) != len(row)):
            raise ValueError("invalid arm list")
    return rounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anno", required=True, type=Path)
    parser.add_argument("--historical", required=True, type=Path)
    parser.add_argument("--schedule", required=True, action="append", nargs=2, metavar=("PATH", "SHA256"))
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    current, historical = args.anno.resolve(strict=True), args.historical.resolve(strict=True)
    selector, selector_hash = pilot.built_selector(None)
    report = {"schema": "anno.qa.journey-policy-panel.v1", "selector_sha256": selector_hash,
              "seed": args.seed, "panels": {},
              "producers": {k: pilot.native.digest(v) for k, v in (("current", current), ("historical", historical))}}

    def baseline(binary):
        return {name: observe(binary, name, False) for name in PROBES}

    # Positive and mutation controls are measured on a fresh temporary corpus every time.
    history = simulate(selector, [lambda: baseline(historical), lambda: baseline(current)], args.seed)
    historical_causes = {name: arm["failure"] for name, arm in history[0]["census"].items()}
    if historical_causes != {"span-roundtrip": "brat-incomplete-input-overwrites-review",
                             "catalog-update": "catalog-incomplete-source-selection",
                             "stale-revision": None}:
        raise ValueError("historical controls did not reproduce the two known regressions")
    if any(a["failure"] for a in history[-1]["census"].values()):
        raise ValueError("fixed controls failed")
    mutation_census(current, PROBES)

    def record(key, trace):
        report["panels"][key] = {"rounds": trace, "scores": summarize(trace)}
    record("historical", history)
    for path, digest in args.schedule:
        if digest in report["panels"]:
            raise ValueError("duplicate schedule")
        schedule = sealed_schedule(Path(path), digest)
        trace = simulate(selector, (lambda faults=faults: mutation_census(current, faults) for faults in schedule), args.seed)
        record(digest, trace)
    pilot.write_json(args.out, report)


if __name__ == "__main__":
    main()
