# SPDX-License-Identifier: MIT OR Apache-2.0
"""Evaluate one contained PER highlight on the 43 reviewed FCoref expressions."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess

from backend_residuals.private_ledger import external_path
from check_span_assertions import check_range, valid_label
from fcoref_identity_review import PACKET_SHA, load_packet, read_labels
from gap_receipts import digest
from score_compact_display import display_views

RUNTIME = "stacked(regex+heuristic)"


def predict(binary, entry):
    text = entry["context"]
    proc = subprocess.run([binary, "extract", "--model", "stacked", "--format", "json"],
                          input=text, capture_output=True, text=True, check=True,
                          env=dict(os.environ, ANNO_NO_DOWNLOADS="1", HF_HUB_OFFLINE="1"))
    result = json.loads(proc.stdout)
    provenance = result["provenance"]
    source_digest = hashlib.sha256(text.encode()).hexdigest()
    if (provenance["model"] != "stacked" or provenance["runtime_model"] != RUNTIME
            or result["source_map"]["analyzed_sha256"] != source_digest
            or result["source_map"]["source_sha256"] != source_digest):
        raise ValueError("Unexpected backend or source edition")
    people = set()
    for entity in result["entities"]:
        check_range(entity, text, entry["code"])
        if not valid_label(entity["type"]):
            raise ValueError("Unexpected entity label")
        if entity["type"] == "PER":
            people.add((entity["start"], entity["end"]))
    return people


def outcome(phrase, name, people, judgment):
    highlight = display_views(phrase, people)["unique_contained"]
    if not highlight:
        return "abstained"
    if judgment == "different":
        return "unsafe_emission"
    if judgment != "same":
        return "unassessed_emission"
    return "exact_highlight" if name in highlight else "other_span"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("packet", "first_review", "second_review", "binary"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    entries = load_packet(args.packet)
    codes = {entry["code"] for entry, _ in entries}
    if args.first_review.samefile(args.second_review):
        parser.error("Supply separate reviewer files")
    reviews = [read_labels(external_path(p), codes) for p in (args.first_review, args.second_review)]
    counts = defaultdict(lambda: defaultdict(Counter))
    names = defaultdict(Counter)
    for entry, control in entries:
        code = entry["code"]
        people = predict(args.binary.resolve(), entry)
        if control is None:
            cohort = "reviewed_GAP"
            a, b = (labels[code] for labels in reviews)
            judgment = a if a == b else "uncertain"
        else:
            cohort, judgment = "authored_controls", control
        phrase, name = tuple(entry["phrase"]), tuple(entry["name"])
        counts[cohort][judgment][outcome(phrase, name, people, judgment)] += 1
        if name in people:
            names[cohort][judgment] += 1
    with args.binary.open("rb") as binary:
        binary_digest = hashlib.file_digest(binary, "sha256").hexdigest()
    print(json.dumps(dict(
        packet_sha256=PACKET_SHA, runtime=RUNTIME, binary_sha256=binary_digest,
        reviewer_fingerprints=[digest(sorted(labels.items())) for labels in reviews],
        standalone_exact_name_spans=names, unique_contained=counts,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
