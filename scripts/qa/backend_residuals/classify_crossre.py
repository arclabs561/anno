# SPDX-License-Identifier: MIT OR Apache-2.0
"""Reconcile emitted CrossRE triplets with the published native scorer before classifying residuals."""

import argparse
import re
import json
from collections import Counter
from pathlib import Path

from private_ledger import external_path, save_new


def edge(raw):
    head, tail, predicate = raw[:3]
    return (tuple(head), tuple(tail), predicate.lower())


def overlap(left, right):
    return max(left[0], right[0]) < min(left[1], right[1])


def outcome(item, other):
    h, t, p = item
    if item in other:
        return "duplicate_beyond_opposite_multiplicity"
    if (t, h, p) in other:
        return "same_predicate_reversed_exact_endpoints"
    if any((a, b) == (h, t) for a, b, _ in other):
        return "forward_exact_pair_other_predicate"
    if any((a, b) == (t, h) for a, b, _ in other):
        return "reversed_exact_pair_other_predicate"
    if any(q == p and overlap(a, h) and overlap(b, t) for a, b, q in other):
        return "same_predicate_both_forward_spans_overlap"
    if any(overlap(a, h) and overlap(b, t) for a, b, _ in other):
        return "other_predicate_both_forward_spans_overlap"
    endpoints = {span for a, b, _ in other for span in (a, b)}
    if h in endpoints and t in endpoints:
        return "both_exact_spans_occur_elsewhere"
    if h in endpoints or t in endpoints:
        return "one_exact_span_occurs_elsewhere"
    return (
        "neither_exact_span_occurs_elsewhere"
        if other
        else "opposite_relation_set_empty"
    )


def classify(residual, other, counts):
    errors = []
    for item, multiplicity in sorted(residual.items()):
        reason = outcome(item, other)
        counts[reason] += multiplicity
        errors.append([*item, multiplicity, reason])
    return errors


def load_result(path):
    (r,) = json.loads(path.read_text())["results"]
    if not r["success"]:
        raise ValueError(f"benchmark reported failure in {path}")
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--baseline", type=Path, default=Path(__file__).resolve().parents[3] /
                        "docs/eval/crossre-full-prompt-comparison.json",
                        help="published eight-arm native evaluation")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.trace_dir = external_path(args.trace_dir)
    args.output = external_path(args.output)
    baseline = json.loads(args.baseline.read_bytes())
    arms = {}
    for domain in ("ai", "literature"):
        reference_rows = None
        for backend in ("tplinker", "gliner_relex"):
            for policy in ("fixed", "oracle"):
                stem = f"{domain}-{backend}-{policy}"
                replay = load_result(args.trace_dir / (stem + ".json"))
                original, = (arm for arm in baseline["domains"][domain]["arms"]
                             if arm["backend"] == backend and arm["prompt"] == policy)
                sample = replay["provenance"]["evaluated_sample"]
                if sample != original["sample"]:
                    raise ValueError(f"{stem}: changed evaluated sample or producer")
                source = external_path(args.trace_dir / (stem + ".rows"))
                rows = [json.loads(line) for line in source.read_bytes().splitlines()]
                if len(rows) != sample["unit_count"]:
                    raise ValueError(f"{stem}: incomplete diagnostic rows")
                missed, extras, totals, documents = Counter(), Counter(), Counter(), []
                first_arm = reference_rows is None
                if first_arm:
                    reference_rows = []
                elif len(rows) != len(reference_rows):
                    raise ValueError(f"{stem}: different source count across arms")
                for index, row in enumerate(rows):
                    if (row["index"] != index or row["sample_sha256"] != sample["sample_sha256"]
                            or not re.fullmatch(r"[0-9a-f]{64}", row["text_sha256"])):
                        raise ValueError(f"{stem}: wrong row order or source identity")
                    gold = Counter(map(edge, row["gold"]))
                    # Prompt and producer can change predictions. The paired comparison
                    # still requires the same ordered sources and directed gold multiset.
                    identity = (row["text_sha256"], gold)
                    if first_arm:
                        reference_rows.append(identity)
                    elif reference_rows[index] != identity:
                        raise ValueError(f"{stem}: source or gold relations differ across arms at row {index}")
                    pred = Counter(map(edge, row["predicted"]))
                    matched = (gold & pred).total()
                    totals.update(
                        gold=gold.total(), predicted=pred.total(), matched=matched
                    )
                    documents.append(
                        {
                            "text_sha256": row["text_sha256"],
                            "matched": matched,
                            "missed": classify(gold - pred, pred, missed),
                            "extra": classify(pred - gold, gold, extras),
                        }
                    )
                for key, metric in (
                    ("gold", "num_gold_relations"),
                    ("predicted", "num_predicted_relations"),
                    ("matched", "strict_relation_matches"),
                ):
                    if not totals[key] == original["metrics"][metric] == replay["metrics"][metric]:
                        raise ValueError(f"{stem}: {metric} differs from the published arm")
                if (missed.total() != totals["gold"] - totals["matched"]
                        or extras.total() != totals["predicted"] - totals["matched"]):
                    raise ValueError(f"{stem}: residual inventory disagrees with strict scoring")
                arms[stem] = {
                    "totals": totals,
                    "missed": missed,
                    "extra": extras,
                    "documents": documents,
                    "sample": sample,
                }
                print(stem, totals, missed.total(), extras.total())
    save_new(args.output, arms)


if __name__ == "__main__":
    main()
