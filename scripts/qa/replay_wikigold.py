"""Replay exact PER, ORG, LOC and MISC spans on one checksum-pinned WikiGold sentence."""
import hashlib
import json
import os
from pathlib import Path
import re

import check_span_assertions as judge
import export_muxer_summary
import domain_coverage

WIKIGOLD = export_muxer_summary.WIKIGOLD

SCOPE = "pinned_source_contract_digest_v2"
PASS_SIGNATURE = ""
FAIL_SIGNATURE = r"sha256:[0-9a-f]{64}"
FAMILY = "wikigold-stacked-exact-spans"
LABELS = {"PER", "ORG", "LOC", "MISC"}


def sentence_index(case):
    if not isinstance(case, str) or not re.fullmatch(r"sentence-(0|[1-9][0-9]{0,3})", case):
        raise ValueError("expected sentence-<zero-based WikiGold ordinal>")
    return int(case.removeprefix("sentence-"))


def family_for_case(root, case):
    sentence_index(case)
    return FAMILY


def sentence(case):
    index = sentence_index(case)
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
    for file in (Path(__file__), Path(judge.__file__), Path(export_muxer_summary.__file__), Path(domain_coverage.__file__)):
        data = file.read_bytes()
        if data != (root / "scripts/qa" / file.name).read_bytes():
            raise ValueError(f"{file.name} differs from the replay checkout")
        sources.append(hashlib.sha256(data).hexdigest())
    oracle = json.dumps([WIKIGOLD, case, gold, sources], separators=(",", ":")).encode()
    return {"source_sha256": judge.source_sha256(text),
            "oracle_sha256": hashlib.sha256(oracle).hexdigest(),
            "family": FAMILY}


def replay(root, case):
    text, gold = sentence(case)
    binary, producer = judge.binary_and_digest()
    row, = judge.run_batch({case: {"text": text}}, str(binary), "stacked", require_native_metadata=True)
    if judge.binary_and_digest() != (binary, producer):
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
    if not missing and not extra:
        return {"outcome": "pass", "signature": PASS_SIGNATURE, "producer_sha256": producer}
    # Receipts and the shared ledger carry an equality witness for the replay.
    # Keep the sentence-specific gold and prediction differences local.
    differences = json.dumps([SCOPE, WIKIGOLD, case, missing, extra], separators=(",", ":"))
    digest = hashlib.sha256(differences.encode()).hexdigest()
    return {"outcome": "fail", "signature": f"sha256:{digest}", "producer_sha256": producer}
