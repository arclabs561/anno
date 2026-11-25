# SPDX-License-Identifier: MIT OR Apache-2.0
"""Shared wire checks for the dated bibliography evaluation."""
from hashlib import sha256
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
CUTS = ("first", "pre", "post")
ARMS = ("dated", "mutable")
DELIVERIES = ("online", "bulk")
COMPONENTS = ("edition", "byline", "printed_byline")
CODE = ("protocol.txt", "common.py", "predict.py", "score.py", "../reader.py")
SHA256 = re.compile(r"[0-9a-f]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def fingerprint(values):
    return {"count": len(values), "sha256": sha256(packed(values)).hexdigest()}


def doi(value):
    require(isinstance(value, str) and value, "missing DOI")
    return ("10.7554/elife." + value if value.isdigit() else value).casefold()


def component(value, name):
    if value is None:
        return
    if name == "edition":
        require(type(value) is int and value > 0, "invalid edition")
    else:
        require(isinstance(value, dict) and set(value) == {"count", "sha256"}
                and type(value["count"]) is int and value["count"] > 0
                and SHA256.fullmatch(value["sha256"]), "invalid " + name)


def file_map():
    return {name: file_hash(ROOT / name) for name in CODE}


def load_rows(document):
    return document["rows"] if isinstance(document, dict) else document
