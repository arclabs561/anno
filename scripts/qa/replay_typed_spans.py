"""Replay one reviewed typed-span case with the offline stacked extractor."""
import hashlib
import json
import os
from pathlib import Path
import shutil

import check_span_assertions as judge

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
    local_scorer = Path(judge.__file__).read_bytes()
    if local_scorer != (root / "scripts/qa/check_span_assertions.py").read_bytes():
        raise ValueError("scorer differs from the executable adapter's checkout")
    oracle = json.dumps(
        [MODEL, hashlib.sha256(local_scorer).hexdigest(), row["assertions"]],
        sort_keys=True, separators=(",", ":"),
    ).encode()
    return {
        "source_sha256": judge.source_sha256(row["text"]),
        "oracle_sha256": hashlib.sha256(oracle).hexdigest(),
        "family": row.get("family", case),
    }


def binary_and_digest():
    resolved = shutil.which(os.environ.get("ANNO_BIN", "anno"))
    if resolved is None:
        raise FileNotFoundError("offline replay needs anno on PATH or ANNO_BIN")
    binary = Path(resolved).resolve()
    with binary.open("rb") as handle:
        return binary, hashlib.file_digest(handle, "sha256").hexdigest()


def replay(root, case):
    row = _case(root, case)
    binary, digest = binary_and_digest()
    cases = {case: row}
    report = judge.score(cases, judge.run_batch(cases, str(binary), MODEL))
    if binary_and_digest() != (binary, digest):
        raise ValueError("extractor changed during replay")
    failed = [str(i) for i, finding in enumerate(report["findings"]) if not finding["passed"]]
    return {
        "outcome": "fail" if failed else "pass",
        "signature": "failed_rules:" + ",".join(failed) if failed else "all_assertions_passed",
        "producer_sha256": digest,
    }
