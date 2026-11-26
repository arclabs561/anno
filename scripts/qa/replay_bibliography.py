# SPDX-License-Identifier: MIT OR Apache-2.0
"""Replay the authored attribution correction with the selected local anno binary."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

SCOPE = "bibliography_attribution_history_v1"
CASE = "attribution-correction"
PASS_SIGNATURE = ""
FAIL_SIGNATURE = r"answer:[0-9a-f]{64}"
CREDITS = Path("verification/index_consumers/credits")
BIBLIOGRAPHY = CREDITS.parent / "reader_cases/bibliography"
FIXTURES = [CREDITS / "bibliography" / name for name in ("tide.xml", "salt.xml", "catalog.json")]
CHECKER = CREDITS / "check_attribution_history.py"
READERS = (CREDITS / "attribution_history.py", CREDITS / "check_cases.py",
           Path("verification/memberships/replay.py"), BIBLIOGRAPHY / "reader.py",
           BIBLIOGRAPHY / "sources/capture.py")


def family_for_case(root, case):
    if case != CASE:
        raise ValueError("unknown bibliography correction case")
    return case


def hash_files(paths):
    digests = {}
    for name, path in paths.items():
        with path.open("rb") as stream:
            digests[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()


def describe(root, case):
    return {"family": family_for_case(root, case),
            "source_sha256": hash_files({path.name: root / path for path in FIXTURES}),
            "oracle_sha256": hash_files({"checker": root / CHECKER, "adapter": Path(__file__)})}


def verdict(result):
    # Only a completed answer comparison may register a contract violation.
    lines = result.stdout.splitlines()
    if result.returncode == 0 and lines[-1:] == ["attribution-history ok"]:
        return "pass", PASS_SIGNATURE
    if result.returncode == 1 and lines:
        mismatch = re.fullmatch(r"attribution-history mismatch: (.+)", lines[-1])
        if mismatch:
            return "fail", "answer:" + hashlib.sha256(mismatch[1].encode()).hexdigest()
    raise ValueError("bibliography checker did not return a completed verdict")


def replay(root, case):
    family_for_case(root, case)
    suffix = ".exe" if os.name == "nt" else ""
    default = root / os.environ.get("CARGO_TARGET_DIR", "target") / "debug" / ("anno" + suffix)
    binary = Path(os.environ.get("ANNO_BIN", default)).resolve(strict=True)
    paths = {path.as_posix(): root / path for path in READERS}
    paths.update(anno=binary, python=Path(sys.executable))
    producer = hash_files(paths)
    result = subprocess.run([sys.executable, "-E", "-s", root / CHECKER, "--anno", binary],
                            cwd=root, capture_output=True, text=True, timeout=120)
    outcome, signature = verdict(result)
    if hash_files(paths) != producer:
        raise ValueError("bibliography reader changed during replay")
    return {"outcome": outcome, "signature": signature, "producer_sha256": producer}
