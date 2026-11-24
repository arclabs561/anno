"""Replay exact PER, ORG, LOC and MISC spans on one checksum-pinned WikiGold sentence."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

import check_span_assertions as judge
import export_muxer_summary

WIKIGOLD = export_muxer_summary.WIKIGOLD

SCOPE = "pinned_source_contract"
LABELS = {"PER", "ORG", "LOC", "MISC"}


def sentence(case):
    if not re.fullmatch(r"sentence-(0|[1-9][0-9]{0,3})", case):
        raise ValueError("expected sentence-<zero-based WikiGold ordinal>")
    cache = os.environ.get("ANNO_CACHE_DIR")
    if not cache:
        raise FileNotFoundError("set ANNO_CACHE_DIR to the existing pinned dataset cache")
    path = Path(cache) / "datasets/WikiGold.cache"
    with path.open("rb") as file:
        data = file.read(1_000_001)
    if hashlib.sha256(data).hexdigest() != WIKIGOLD:
        raise ValueError("cached WikiGold differs from the pinned corpus")
    # WikiGold has blank-line-separated sentences and standalone document markers.
    blocks = [s for s in data.decode("utf-8").strip().split("\n\n")
              if s.strip() and s.strip() != "-DOCSTART- O"]
    index = int(case.removeprefix("sentence-"))
    if index >= len(blocks):
        raise ValueError("WikiGold sentence ordinal exceeds the corpus")
    return annotated_sentence(blocks[index])


def annotated_sentence(block):
    words = []
    gold = []
    pos = 0
    previous = None
    for line in block.splitlines():
        token, tag = line.split()
        end = pos + len(token)
        if tag == "O":
            previous = None
        else:
            prefix, label = tag.split("-", 1)
            if prefix not in ("I", "B") or label not in LABELS:
                raise ValueError(f"unexpected WikiGold label {tag!r}")
            # WikiGold also uses I- for an entity start, as in anno-eval's loader.
            if prefix == "I" and label == previous:
                gold[-1] = (gold[-1][0], end, label)
            else:
                gold.append((pos, end, label))
            previous = label
        words.append(token)
        pos = end + 1
    return " ".join(words), gold


def describe(root, case):
    text, gold = sentence(case)
    sources = []
    for file in (Path(__file__), Path(judge.__file__), Path(export_muxer_summary.__file__)):
        data = file.read_bytes()
        if data != (root / "scripts/qa" / file.name).read_bytes():
            raise ValueError(f"{file.name} differs from the replay checkout")
        sources.append(hashlib.sha256(data).hexdigest())
    oracle = json.dumps([WIKIGOLD, case, gold, sources], separators=(",", ":")).encode()
    return {"source_sha256": judge.source_sha256(text),
            "oracle_sha256": hashlib.sha256(oracle).hexdigest(),
            "family": "wikigold-stacked-exact-spans"}


def binary_digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def replay(root, case):
    text, gold = sentence(case)
    found = shutil.which(os.environ.get("ANNO_BIN", "anno"))
    if not found:
        raise FileNotFoundError("offline WikiGold replay needs anno on PATH or ANNO_BIN")
    binary = Path(found).resolve()
    producer = binary_digest(binary)
    row, = judge.run_batch({case: {"text": text}}, str(binary), "stacked")
    if binary_digest(binary) != producer:
        raise ValueError("extractor changed during replay")
    if (row.get("id") != case or row.get("model") != "stacked"
            or row.get("source_sha256") != judge.source_sha256(text)):
        raise ValueError("batch receipt differs from the requested WikiGold case or model")
    predicted = set()
    for entity in judge.objects(row.get("entities"), case):
        judge.check_range(entity, text, case)
        label = entity.get("type")
        if not judge.valid_label(label):
            raise ValueError("invalid prediction type")
        if label in LABELS:
            predicted.add((entity["start"], entity["end"], label))
    missing = sorted(set(gold) - predicted)
    extra = sorted(predicted - set(gold))
    return {"outcome": "fail" if missing or extra else "pass",
            "signature": json.dumps({"missing": missing, "extra": extra}, separators=(",", ":")),
            "producer_sha256": producer}
