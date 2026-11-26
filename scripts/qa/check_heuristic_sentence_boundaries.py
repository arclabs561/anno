"""Compare native heuristic versions before/after deterministic RED-FM punctuation edits."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

from ner_language_length import HASHES, source_rows, spans

TERMINALS = "。؟।！？॥።"


def edit_punctuation(row, gold):
    text = row["text"]
    positions = set()
    for _, end, _ in gold:
        match = re.match(r"[.!?]\s+(.)", text[end:])
        if match and match[1].isupper() and not any(a <= end < b for a, b, _ in gold):
            positions.add(end)
    edited = list(text)
    for offset in sorted(positions):
        seed = hashlib.sha256(f"{row['docid']}{offset}".encode()).hexdigest()
        edited[offset] = TERMINALS[int(seed, 16) % len(TERMINALS)]
    return "".join(edited), len(positions)


def predict(binary, text):
    cmd = [binary, "extract", "--quiet", "--model", "heuristic", "--format", "json", "--text", text]
    output = json.loads(subprocess.check_output(cmd, timeout=60))
    if output["provenance"]["runtime_model"] != "heuristic":
        raise ValueError("unexpected runtime")
    found = set()
    for entity in output["entities"]:
        start, end = entity["start"], entity["end"]
        if not 0 <= start < end <= len(text) or text[start:end] != entity["text"]:
            raise ValueError(f"unaligned span at [{start}, {end})")
        if entity["type"] in ("PER", "ORG", "LOC"):
            found.add((start, end, entity["type"]))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    args = parser.parse_args()
    results = {}
    binaries = [p.resolve(strict=True) for p in (args.baseline, args.candidate)]
    for language in ("de", "en", "es", "fr", "it"):
        key = f"test.{language}"
        counts = Counter()
        for row in source_rows(args.data_dir, key, False):
            if row["lan"] != language:
                raise ValueError(f"unexpected language in {key}")
            gold = spans(row["text"], row["entities"], "boundaries")
            changed, edits = edit_punctuation(row, gold)
            if not edits:
                continue
            old, new = (predict(binary, row["text"]) for binary in binaries)
            base, candidate = (predict(binary, changed) for binary in binaries)
            counts.update(documents=1, edits=edits, gold=len(gold), original_set_changes=len(old ^ new),
                          baseline_tp=len(base & gold), candidate_tp=len(candidate & gold),
                          baseline_unmatched=len(base - gold), candidate_unmatched=len(candidate - gold),
                          recovered=len((candidate - base) & gold), lost=len((base - candidate) & gold))
        results[language] = dict(source_sha256=HASHES[key], **counts)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
