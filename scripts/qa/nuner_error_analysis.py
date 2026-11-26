"""Summarize unmatched NuNER spans from the threshold-0.5 paired ledger."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def mismatch(span, alternatives):
    start, end, label = span
    overlaps = [
        (left, right, other)
        for left, right, other in alternatives
        if start < right and left < end
    ]
    if any((start, end) == (left, right) for left, right, _ in overlaps):
        return "type_only"
    if any(label == other for _, _, other in overlaps):
        return "boundary_only"
    if overlaps:
        return "boundary_and_type"
    return "no_overlap"


def diagnose(cases):
    groups = defaultdict(Counter)
    for case in cases:
        gold = set(map(tuple, case["gold"]))
        predictions = {key: set(map(tuple, case[key])) for key in ("fp32", "u8")}
        counts = groups[case["group"]]
        for key, other in [("fp32", "u8"), ("u8", "fp32")]:
            pred = predictions[key]
            for direction, errors, alternatives, other_errors in [
                ("fp", pred - gold, gold, predictions[other] - gold),
                ("fn", gold - pred, pred, gold - predictions[other]),
            ]:
                for error in errors:
                    shared = "both" if error in other_errors else "only"
                    counts[(key, direction, shared, mismatch(error, alternatives))] += 1
    return groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    ledger = json.loads(args.receipt.read_text())
    groups = diagnose(ledger["cases"])
    print("group,model,direction,arms,category,count")
    for group, counts in groups.items():
        for model in ("fp32", "u8"):
            reference = ledger["models"][model]["groups"][group]["thresholds"]["0.5"]
            for direction in ("fp", "fn"):
                total = sum(
                    v
                    for (m, d, _, _), v in counts.items()
                    if (m, d) == (model, direction)
                )
                if total != reference[direction]:
                    raise ValueError(
                        f"{group} {model} {direction}: expected {reference[direction]}, got {total}"
                    )
        for key, count in sorted(counts.items()):
            print(",".join((group, *key, str(count))))
