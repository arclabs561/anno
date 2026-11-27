# SPDX-License-Identifier: MIT OR Apache-2.0
"""Prepare blinded referring-expression reviews and score the resulting private labels."""
import argparse
from collections import Counter
import hashlib
import json
import os
from statistics import median
from pathlib import Path

from backend_residuals.private_ledger import external_path
from coref_gap import decode_clusters, index_partition
from fcoref_emitted_boundaries import MODEL_FILES, check_file, panel
from fcoref_endpoint_stages import REFERENCE, marked, span_ranks, strictly_contains
from gap_receipts import digest

PACKET_SHA = "2ce4977caf0e5b9378ffa2bb78c1cea9c04fe0f2dfe0d27d8e2494fdc5ba8fa3"
VERDICTS = ("same", "different", "uncertain")


def select_cases(rows, records, tokenizer):
    """Return (GAP row, A/B field, enclosing character span) in frozen panel order."""
    reference = json.loads(REFERENCE.read_text())
    partitions = []
    cases = []
    for row, record in zip(rows, records, strict=True):
        if record["id"] != row["ID"]:
            raise ValueError("Mismatched trace identity")
        clusters = decode_clusters(record["stock"], row["ID"], row["Text"], "fcoref")
        partitions.append((row["ID"], sorted(sorted(c) for c in clusters)))
        indices = index_partition(clusters)
        pronoun = marked(row, "Pronoun")
        if pronoun not in indices:
            continue
        k, _ = record["trace"]
        ranks = span_ranks(row["Text"], tokenizer.encode(row["Text"]), record["trace"])
        for field in ("A", "B"):
            name = marked(row, field)
            if row[field + "-coref"] != "TRUE" or name not in ranks or ranks[name] <= k:
                continue
            rivals = [s for s in clusters[indices[pronoun]] if strictly_contains(s, name)]
            if rivals:
                # Compare a single occurrence per marked pair even in nested clusters.
                cases.append((row, field, min(rivals, key=lambda s: (s[1]-s[0], *s))))
    if digest(partitions) != reference["stock_partitions_sha256"] or len(cases) != 43:
        raise ValueError("Expected the 43 enclosing rivals from the frozen stock partitions")
    return cases


# Project-authored checks on the distinction between a mention and an embedded name.
CONTROLS = [
    ("The editor Ada Wynn sent a correction before noon.", "Ada Wynn", "The editor Ada Wynn", "same"),
    ("Lena Vos, the new chair, called the meeting.", "Lena Vos", "Lena Vos, the new chair", "same"),
    ("His colleague Dr. Ivo Marsh was present.", "Ivo Marsh", "Dr. Ivo Marsh", "same"),
    ("Nika's sister Jori brought lunch.", "Jori", "Nika's sister Jori", "same"),
    ("Ada Wynn's bicycle was locked outside.", "Ada Wynn", "Ada Wynn's bicycle", "different"),
    ("The brother of Lena Vos gave evidence.", "Lena Vos", "The brother of Lena Vos", "different"),
    ("Mira and Ivo Marsh left together.", "Ivo Marsh", "Mira and Ivo Marsh", "different"),
    ("They restored the fountain named after Jori.", "Jori", "the fountain named after Jori", "different"),
]


def packet(cases):
    entries = []
    def add(key, text, name, phrase, control=None):
        code = hashlib.sha256(("anno-gap-identity-v1/" + key).encode()).hexdigest()[:16]
        entries.append((dict(code=code, context=text, name=list(name), phrase=list(phrase)), control))
    for row, field, rival in cases:
        add(row["ID"] + "/" + field, row["Text"], marked(row, field), rival)
    for i, (text, name, phrase, label) in enumerate(CONTROLS):
        start, left = text.index(name), text.index(phrase)
        add("authored/" + str(i), text, (start, start+len(name)), (left, left+len(phrase)), label)
    entries.sort(key=lambda entry: entry[0]["code"])
    if len({entry["code"] for entry, _ in entries}) != len(entries):
        raise ValueError("Duplicate review code")
    return entries


def read_labels(path, expected):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    labels = {r["code"]: r["label"] for r in records}
    if len(records) != len(labels) or labels.keys() != expected or any(v not in VERDICTS for v in labels.values()):
        raise ValueError("Require one valid verdict for each review item")
    return labels


def compare(entries, first, second):
    totals = Counter()
    controls = {entry["code"]: label for entry, label in entries if label is not None}
    edges = Counter()
    overhang = []
    for entry, label in entries:
        if label is None:
            start, end = entry["phrase"]
            a, b = entry["name"]
            if not strictly_contains((start,end), (a,b)):
                raise ValueError("Expected strict enclosure")
            left, right = a-start, end-b
            edges["both" if left and right else "left" if left else "right"] += 1
            overhang.append(left + right)
            a, b = first[entry["code"]], second[entry["code"]]
            totals[(a,b)] += 1
    n = sum(totals.values())
    return {
        "dataset_pairs": n,
        "exact_highlight_matches": 0,  # All scored cases strictly enclose the reference.
        "highlight_extra_characters": dict(min=min(overhang), median=median(overhang), max=max(overhang)),
        "highlight_extended_on": {side: edges[side] for side in ("left","right","both")},
        "reviewer_cross_tab": {a: {b: totals[a,b] for b in VERDICTS} for a in VERDICTS},
        "controls": len(controls),
        "control_correct": [sum(labels[key] == truth for key, truth in controls.items())
                            for labels in (first, second)],
        "review_fingerprints": [digest(sorted(labels.items())) for labels in (first, second)],
    }


def load_packet(path):
    data = [json.loads(line) for line in external_path(path).read_text().splitlines()]
    if digest(data) != PACKET_SHA:
        raise ValueError("Review packet differs from the frozen study")
    controls = {entry["code"]: truth for entry, truth in packet([])}
    return [(entry, controls.get(entry["code"])) for entry in data]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    prepare = subs.add_parser("prepare", help="Write the private packet from the frozen native replay")
    for name in ("development", "test", "validation", "model", "binary", "output"):
        prepare.add_argument(name, type=Path)
    review = subs.add_parser("score", help="Compare two independently completed private review files")
    for name in ("packet", "first", "second"):
        review.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.action == "score":
        entries = load_packet(args.packet)
        keys = {entry["code"] for entry, _ in entries}
        if args.first.samefile(args.second):
            parser.error("Supply separate reviewer files")
        result = compare(entries, *(read_labels(external_path(p), keys) for p in (args.first, args.second)))
        print(json.dumps({"packet_sha256": PACKET_SHA, **result}, indent=2))
        return

    from tokenizers import Tokenizer
    from run_fcoref_endpoint_stages import observations
    path = external_path(args.output)
    rows = panel(args.development, args.test, args.validation)
    for name, expected in MODEL_FILES.items():
        check_file(args.model / name, expected)
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    cases = select_cases(rows, observations(rows, args.binary, args.model), tokenizer)
    data = [entry for entry, _ in packet(cases)]
    if digest(data) != PACKET_SHA:
        raise ValueError("Native replay differs from the frozen review packet")
    with os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "w", encoding="utf-8") as output:
        for entry in data:
            print(json.dumps(entry), file=output)


if __name__ == "__main__":
    main()
