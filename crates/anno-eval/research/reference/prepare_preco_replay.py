"""Prepare a deterministic, disjoint calibration/audit sample from PreCo dev."""

import argparse
import hashlib
import json
from collections import defaultdict
from itertools import accumulate
from pathlib import Path


def conll(record):
    sentences = record["sentences"]
    offsets = list(accumulate(map(len, sentences), initial=0))
    starts, ends, singletons = defaultdict(list), defaultdict(list), defaultdict(list)
    seen = set()
    for entity, mentions in enumerate(record["mention_clusters"]):
        for sentence, start, end in mentions:
            if not (0 <= sentence < len(sentences) and 0 <= start < end <= len(sentences[sentence])):
                raise ValueError(f"{record['id']}: invalid mention coordinates")
            lo, hi = offsets[sentence] + start, offsets[sentence] + end
            if (lo, hi) in seen:
                raise ValueError(f"{record['id']}: duplicate occurrence")
            seen.add((lo, hi))
            if lo + 1 == hi:
                singletons[lo].append(f"({entity})")
            else:
                starts[lo].append((hi, entity))
                ends[hi - 1].append((lo, entity))

    name = hashlib.sha256(record["id"].encode()).hexdigest()
    rows = [f"#begin document ({name}); part 0"]
    for pos, token in enumerate(token for sentence in sentences for token in sentence):
        if any(character in token for character in "\t\r\n"):
            raise ValueError(f"{record['id']}: invalid CoNLL token")
        fields = ([f"{entity})" for _, entity in sorted(ends[pos], reverse=True)]
                  + singletons[pos]
                  + [f"({entity}" for _, entity in sorted(starts[pos], reverse=True)])
        rows.append(f"{name}\t0\t{pos}\t{token}\t{'|'.join(fields)}")
    rows.append("#end document")
    return name, "\n".join(rows) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--calibration", type=int, default=40)
    parser.add_argument("--audit", type=int, default=80)
    args = parser.parse_args()
    with args.source.open(encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]
    records.sort(key=lambda r: hashlib.sha256(r["id"].encode()).digest())
    if min(args.calibration, args.audit) < 1 or args.calibration + args.audit > len(records):
        parser.error("invalid calibration and audit sizes")
    if len({r["id"] for r in records}) != len(records):
        parser.error("duplicate document IDs")
    if args.output.exists():
        parser.error("use a fresh output directory")
    for split, selected in (
        ("calibration", records[:args.calibration]),
        ("audit", records[args.calibration:args.calibration + args.audit]),
    ):
        destination = args.output / split
        destination.mkdir(parents=True)
        for record in selected:
            name, contents = conll(record)
            (destination / f"{name}.conll").write_text(contents, encoding="utf-8")


if __name__ == "__main__":
    main()
