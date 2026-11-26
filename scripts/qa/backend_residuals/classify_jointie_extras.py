"""Describe every extra decoded JointIE edge against that sentence's typed gold edges."""

import argparse
import json
from collections import Counter
from pathlib import Path


def reason(h, p, t, gold):
    if (t, p, h) in gold:
        return "same_typed_endpoints_and_predicate_reversed"
    if any((h, t) == (a, b) for a, _, b in gold):
        return "forward_typed_pair_other_predicate"
    if any((h[1:], p, t[1:]) == (a[1:], q, b[1:]) for a, q, b in gold):
        return "forward_spans_and_predicate_endpoint_type_mismatch"
    if any((h[1:], t[1:]) == (a[1:], b[1:]) for a, _, b in gold):
        return "forward_exact_spans_other_predicate_and_endpoint_type_mismatch"
    if any((h[1:], t[1:]) == (b[1:], a[1:]) for a, _, b in gold):
        return "reverse_exact_spans_with_other_label_or_type"
    spans = {s[1:] for a, _, b in gold for s in (a, b)}
    if not gold:
        return "no_gold_relations"
    if h[1:] in spans and t[1:] in spans:
        return "both_exact_spans_elsewhere"
    if h[1:] in spans or t[1:] in spans:
        return "one_exact_span_elsewhere"
    return "neither_exact_span_elsewhere"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    results = {}
    for arm, expected in [
        ("w16", (1, 27)),
        ("w384", (0, 2)),
        ("training_routes", (0, 4)),
    ]:
        src = json.loads((args.receipts / f"jointie_ai64_{arm}.json").read_text())
        counts, rows, tp = Counter(), [], 0
        for d in src["documents"]:
            gold = {
                (tuple(e["head"]), e["predicate"], tuple(e["tail"]))
                for e in d["edge_trace"]
            }
            for h, p, t in d["decoded_edge_keys"]:
                h, t = tuple(h), tuple(t)
                if (h, p, t) in gold:
                    tp += 1
                    continue
                r = reason(h, p, t, gold)
                counts[r] += 1
                rows.append([d["source_sha256"], h, p, t, r])
        assert (tp, tp + len(rows)) == expected
        results[arm] = {
            "matched": tp,
            "extra": len(rows),
            "counts": counts,
            "errors": rows,
        }
        print(arm, dict(counts))
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
