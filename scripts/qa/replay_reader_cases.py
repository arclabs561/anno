"""Replay a reviewed reader history against separate answers with Rust and SQLite."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile

SCOPE = "reviewed_reader_answers_v1"
PASS_SIGNATURE = ""
FAIL_SIGNATURE = r"answers:[0-9a-f]{64}"
REVIEWED = frozenset({"issuer-local-credit-citations", "two-support-withdrawals"})
DIRECTORY = Path("verification/index_consumers/reader_cases")


def family_for_case(root, case):
    if case not in REVIEWED:
        raise ValueError("unknown reviewed reader history")
    return case


def unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def fingerprint(value):
    return hashlib.sha256(encode(value)).hexdigest()


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_rows(rows, questions):
    if not isinstance(rows, dict) or set(rows) != set(questions):
        raise ValueError("reader did not return the exact question set")
    for question, values in rows.items():
        row_type = dict if questions[question] == "cited_credit" else str
        if not isinstance(values, list) or any(not isinstance(row, row_type) for row in values):
            raise ValueError("invalid reader answer rows")


def reviewed(root, case):
    family_for_case(root, case)
    directory = root / DIRECTORY / "fixtures"

    def load(name):
        path = directory / name
        if path.is_symlink():
            raise ValueError("expected a regular reviewed fixture")
        data = json.loads(path.read_bytes(), object_pairs_hook=unique_fields)
        if type(data["version"]) is not int or data["version"] != 1:
            raise ValueError("unsupported fixture version")
        return data

    history, = [item for item in load("cases.json")["cases"] if item["id"] == case]
    queries = history["queries"]
    questions = {item["id"]: item["kind"] for item in queries}
    if not questions or len(questions) != len(queries):
        raise ValueError("expected nonempty, distinct reader questions")
    key = load("answer_key.json")["answers"][case]
    check_rows(key, questions)
    return {"version": 1, "cases": [history]}, questions, key


def describe(root, case):
    suite, _, key = reviewed(root, case)
    return {"family": case, "source_sha256": fingerprint(suite),
            "oracle_sha256": fingerprint([case, key, file_hash(Path(__file__))])}


def executables(root):
    debug = root / os.environ.get("CARGO_TARGET_DIR", "target") / "debug"
    suffix = ".exe" if os.name == "nt" else ""
    anno = os.environ.get("ANNO_BIN", debug / ("anno" + suffix))
    reader = os.environ.get("ANNO_READER_BIN", debug / "examples" / ("reader_answer" + suffix))
    return {name: Path(path).resolve(strict=True) for name, path in {
        "anno": anno, "rust": reader, "python": sys.executable, "sqlite_reader": root / DIRECTORY / "check.py",
    }.items()}


def producer_hash(paths):
    return fingerprint({**{name: file_hash(path) for name, path in paths.items()},
                        "sqlite_version": sqlite3.sqlite_version})


def answers(command, root, case, questions):
    result = subprocess.run(command, cwd=root, capture_output=True, check=True, timeout=120)
    data = json.loads(result.stdout, object_pairs_hook=unique_fields)
    if (not isinstance(data, dict) or set(data) != {"version", "answers"}
            or type(data["version"]) is not int or data["version"] != 1
            or not isinstance(data["answers"], dict) or set(data["answers"]) != {case}):
        raise ValueError("invalid reader answer envelope")
    rows = data["answers"][case]
    check_rows(rows, questions)
    return rows


def replay(root, case):
    suite, questions, key = reviewed(root, case)
    paths = executables(root)
    producer = producer_hash(paths)
    failed = []
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "case.json"
        path.write_bytes(encode(suite))
        commands = {
            "rust": [paths["rust"], path],
            "sqlite": [paths["python"], "-I", paths["sqlite_reader"], "--emit",
                       "--anno", paths["anno"], "--cases", path],
        }
        for reader, command in commands.items():
            rows = answers(command, root, case, questions)
            if path.read_bytes() != encode(suite):
                raise ValueError("input changed during replay")
            failed.extend((reader, q) for q in sorted(questions) if encode(rows[q]) != encode(key[q]))
        if producer_hash(paths) != producer:
            raise ValueError("reader changed during replay")
    return {"outcome": "fail" if failed else "pass", "producer_sha256": producer,
            "signature": "answers:" + fingerprint(failed) if failed else PASS_SIGNATURE}
