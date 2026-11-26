"""Reconcile emitted CrossRE triplets with the published native scorer before classifying residuals."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


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
    assert r["success"], path
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("original_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    arms = {}
    for domain in ("ai", "literature"):
        for backend in ("tplinker", "gliner_relex"):
            for policy in ("fixed", "oracle"):
                stem = f"{domain}-{backend}-{policy}"
                replay = load_result(args.trace_dir / (stem + ".json"))
                original = load_result(
                    args.original_dir
                    / (stem.replace("gliner_relex", "relex") + ".json")
                )
                sample = replay["provenance"]["evaluated_sample"]
                assert sample == original["provenance"]["evaluated_sample"], stem
                rows = [
                    json.loads(line)
                    for line in (args.trace_dir / (stem + ".rows"))
                    .read_text()
                    .splitlines()
                ]
                assert len(rows) == sample["unit_count"]
                missed, extras, totals, documents = Counter(), Counter(), Counter(), []
                for row in rows:
                    gold = Counter(map(edge, row["gold"]))
                    pred = Counter(map(edge, row["predicted"]))
                    matched = (gold & pred).total()
                    totals.update(
                        gold=gold.total(), predicted=pred.total(), matched=matched
                    )
                    documents.append(
                        {
                            "text_sha256": hashlib.sha256(
                                row["text"].encode()
                            ).hexdigest(),
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
                    assert (
                        totals[key]
                        == original["metrics"][metric]
                        == replay["metrics"][metric]
                    ), (stem, key, totals, original["metrics"][metric])
                assert missed.total() == totals["gold"] - totals["matched"]
                assert extras.total() == totals["predicted"] - totals["matched"]
                arms[stem] = {
                    "totals": totals,
                    "missed": missed,
                    "extra": extras,
                    "documents": documents,
                    "sample": sample,
                }
                print(stem, totals, missed.total(), extras.total())
    args.output.write_text(json.dumps(arms, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
