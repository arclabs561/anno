"""Replay one reviewed typed-span case with the offline stacked extractor."""
import hashlib
import json
from pathlib import Path

import check_span_assertions as judge
import domain_coverage

SCOPE = "synthetic_contract"
MODEL = "stacked"


def _case(root, case):
    path = root / "scripts/qa/fixtures/typed-span-assertions.json"
    cases = judge.load_cases(json.loads(path.read_bytes()))
    if case not in cases:
        raise ValueError(f"unknown typed-span case {case!r}")
    return cases[case]


def describe(root, case):
    row = _case(root, case)
    sources = []
    for actual in (Path(__file__), Path(judge.__file__), Path(domain_coverage.__file__)):
        contents = actual.read_bytes()
        if contents != (root / "scripts/qa" / actual.name).read_bytes():
            raise ValueError("scorer differs from the executable adapter's checkout")
        sources.append(hashlib.sha256(contents).hexdigest())
    oracle = json.dumps(
        [MODEL, sources, row["assertions"]],
        sort_keys=True, separators=(",", ":"),
    ).encode()
    return {
        "source_sha256": judge.source_sha256(row["text"]),
        "oracle_sha256": hashlib.sha256(oracle).hexdigest(),
        "family": row.get("family", case),
    }


def replay(root, case):
    row = _case(root, case)
    binary, digest = judge.binary_and_digest()
    cases = {case: row}
    report = judge.score(cases, judge.run_batch(cases, str(binary), MODEL, require_native_metadata=True))
    if judge.binary_and_digest() != (binary, digest):
        raise ValueError("extractor changed during replay")
    failed = [str(i) for i, finding in enumerate(report["findings"]) if not finding["passed"]]
    return {
        "outcome": "fail" if failed else "pass",
        "signature": "failed_rules:" + ",".join(failed) if failed else "all_assertions_passed",
        "producer_sha256": digest,
    }
